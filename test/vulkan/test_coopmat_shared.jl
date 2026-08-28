using Test, Lava, KernelAbstractions
using Lava: AcceleratedMatrix, MatrixA, MatrixB, Accumulator
using Mantle: coopmat_gemm_available

# Can a cooperative matrix be loaded out of `@localmem`?
#
# It could not, until `loadw`/`storew`: `coopmat_base_pointer!` hardcoded
# `SC.PhysicalStorageBuffer`, so `OpCooperativeMatrixLoadKHR` could only address
# global memory. That is the one thing standing between `coopmat_gemm_kernel!`
# and a shared-memory staged GEMM, which is what every high-performance
# implementation does and what ours does not — see `perf-plan.md`.
#
# Smallest possible statement of it: stage a tile into shared, load a matrix
# from there, and compare against the same tile loaded from global. Anything
# larger would confound a codegen fault with a tiling bug.

const TILE = 16

@kernel cpu=false function stage_and_roundtrip!(out, @Const(inp))
    # One subgroup, one tile. `Val`-free literal sizes on purpose: a `@localmem`
    # extent that comes from a local variable miscompiles silently and the
    # kernel writes nothing at all.
    tile = @localmem Float16 (TILE * TILE,)
    i = @index(Local, Linear)
    @inbounds for k in 0:7                      # 32 lanes stage 256 elements
        j = i + k * 32
        tile[j] = inp[j]
    end
    @synchronize
    m = AcceleratedMatrix{Float16,TILE,TILE,MatrixA}(tile, 1, TILE)
    copyto!(pointer(out), 1, TILE, m)
end

# Same, but reading the tile from `OFF` rather than from the start — the case
# that decides which SPIR-V construct the pointer comes from.
@kernel cpu=false function shared_at_offset!(out, @Const(inp), ::Val{OFF}) where {OFF}
    tile = @localmem Float16 (512,)
    i = @index(Local, Linear)
    @inbounds for k in 0:15
        tile[i + k * 32] = inp[i + k * 32]
    end
    @synchronize
    m = AcceleratedMatrix{Float16,TILE,TILE,MatrixA}(tile, OFF, TILE)
    copyto!(pointer(out), 1, TILE, m)
end

@kernel cpu=false function global_roundtrip!(out, @Const(inp))
    m = AcceleratedMatrix{Float16,TILE,TILE,MatrixA}(pointer(inp), 1, TILE)
    copyto!(pointer(out), 1, TILE, m)
end

# A x B accumulated, with A staged through shared and B read from global: the
# shape the GEMM will actually use, where a matrix loaded from `Workgroup`
# memory has to be a valid operand of `OpCooperativeMatrixMulAddKHR` and not
# merely storable.
@kernel cpu=false function shared_a_gemm!(out, @Const(a), @Const(b))
    tile = @localmem Float16 (TILE * TILE,)
    i = @index(Local, Linear)
    @inbounds for k in 0:7
        j = i + k * 32
        tile[j] = a[j]
    end
    @synchronize
    A = AcceleratedMatrix{Float16,TILE,TILE,MatrixA}(tile, 1, TILE)
    B = AcceleratedMatrix{Float16,TILE,TILE,MatrixB}(pointer(b), 1, TILE)
    C = zero(AcceleratedMatrix{Float32,TILE,TILE,Accumulator})
    C = muladd(A, B, C)
    copyto!(pointer(out), 1, TILE, C)
end

@testset "cooperative matrix from Workgroup memory" begin
    # Guarded on the shape query, NOT on `coopmat_gemm_available()`: this test
    # launches its own 32-wide kernel and never uses the block GEMM's `lane ÷ 32`
    # subgroup indexing, so it is meaningful (and passes) on a wave64 device where
    # the GEMM path is correctly disabled.
    if !Mantle.coopmat_shape(Mantle.vk_context(), Float16, TILE, TILE, TILE)
        @info "skipping: device reports no $(TILE)^3 Float16 cooperative-matrix shape"
    else
        backend = LavaBackend()
        h = Float16.(reshape(1:(TILE * TILE), TILE, TILE) ./ 16)

        @testset "load from shared == load from global" begin
            inp = LavaArray(vec(h))
            fromshared = LavaArray(zeros(Float16, TILE * TILE))
            fromglobal = LavaArray(zeros(Float16, TILE * TILE))
            stage_and_roundtrip!(backend, 32)(fromshared, inp; ndrange = 32)
            global_roundtrip!(backend, 32)(fromglobal, inp; ndrange = 32)
            KernelAbstractions.synchronize(backend)
            # Against the global path rather than against `h` directly: that
            # isolates the storage class, which is the only thing new here.
            @test Array(fromshared) == Array(fromglobal)
            @test Array(fromshared) == vec(h)          # …and both are right
        end

        @testset "at an offset into the tile" begin
            # The offset is what the whole lowering turns on. Passed as a GEP it
            # vanished at 1 (leaving a pointer to the array) and survived at 17
            # as a *constant expression* rather than an instruction, so no test
            # on the LLVM node kind separated the two — and the wrong branch
            # emitted an `OpBitcast` between Workgroup pointers, which is
            # illegal in Vulkan's Logical addressing model, validates clean, and
            # **segfaults NVIDIA's shader compiler**. The index is an argument
            # now; these three offsets are why.
            for off in (1, 17, 33)
                inp = LavaArray(Float16.(collect(1:512)))
                out = LavaArray(zeros(Float16, TILE * TILE))
                shared_at_offset!(backend, 32, 32)(out, inp, Val(off))
                KernelAbstractions.synchronize(backend)
                @test Array(out) == Float16.(collect(off:(off + TILE * TILE - 1)))
            end
        end

        @testset "a shared-loaded matrix multiplies" begin
            b = Float16.(reshape(1:(TILE * TILE), TILE, TILE) ./ 32)
            A = LavaArray(vec(h)); B = LavaArray(vec(b))
            C = LavaArray(zeros(Float32, TILE * TILE))
            shared_a_gemm!(backend, 32)(C, A, B; ndrange = 32)
            KernelAbstractions.synchronize(backend)
            want = Float32.(h) * Float32.(b)           # column-major throughout
            @test reshape(Array(C), TILE, TILE) ≈ want rtol = 1e-2
        end
    end
end

# ── A `vec2`-typed staging buffer ────────────────────────────────────────────
#
# `mul_mm.comp` stages into `FLOAT_TYPEV2 buf_a[]` and counts `SHMEM_STRIDE` in
# `vec2`, so every shared access is 32 bits wide. Lava's staged GEMM stages into
# `@localmem Float16`, which makes them 16 — and widening only the *global* load
# without widening the shared array loses to bank conflicts at every width
# (scalar 1.04-1.09x, 2-wide 0.80-0.92x, 4-wide 0.69-0.81x; see `array/gemm.jl`).
#
# Two things had to exist for the reference's form to be reachable, and both are
# checked here because each was missing:
#
#   * a **vector-typed `@localmem`**. `Op.OpCompositeInsert` was used by the
#     emitter and never declared, so building a vector value element by element
#     died with `UndefVarError` instead of compiling.
#   * a **cooperative-matrix load whose pointer addresses that vector**.
#     `OpCooperativeMatrixLoadKHR` permits a vector pointee whose component type
#     matches the matrix and counts `Stride` in those vectors; the emitter built
#     the access chain to the *scalar* component type unconditionally. `loadw2`
#     is the variant that does not.
#
# This is the capability, not the integration: `gemm.jl` still stages scalar.
const CMV2 = NTuple{2,VecElement{Float16}}

@kernel cpu=false unsafe_indices=true function coopmat_vec2_tile!(out, @Const(A))
    sh = @localmem CMV2 (8 * 16,)      # a 16x16 fp16 tile = 8 vec2 per column
    t = @index(Local, Linear) - 1
    @inbounds begin
        if t < 128
            i2 = t % 8; j = t ÷ 8
            # one 32-bit store per lane where the scalar form needs two 16-bit
            sh[1 + i2 + j * 8] = (VecElement(A[1 + 2i2 + j * 16]),
                                  VecElement(A[1 + 2i2 + 1 + j * 16]))
        end
        @synchronize
        if t < 32
            # stride counted in vec2, hence 8 rather than 16
            a = AcceleratedMatrix{Float16,16,16,MatrixA}(sh, 1, 8)
            copyto!(pointer(out), 1, 16, a)
        end
    end
end

@testset "cooperative matrix from a vec2-typed @localmem" begin
    backend = LavaBackend()
    if !Mantle.coopmat_gemm_available()
        @info "skipping: no cooperative-matrix support on this device"
    else
        h = Float16.(reshape(1:256, 16, 16))
        A = LavaArray(vec(h))
        out = LavaArray(fill(Float16(-1), 256))
        coopmat_vec2_tile!(backend, 128)(out, A; ndrange = 128)
        KernelAbstractions.synchronize(backend)
        @test reshape(Array(out), 16, 16) == h
    end
end

# The same, 4-wide. It is a separate test because it exercises a DIFFERENT bug:
# `wg_compute_type_size`/`wg_compute_type_alignment` had no `VectorType` branch
# at all and fell through to a constant 4. That is right for `<2 x half>` by
# coincidence, so the vec2 test above passed while a 4-wide `@localmem` got
# `ArrayStride 4` for an 8-byte element and `spirv-val` refused the module
# ("array with stride 4 not satisfying alignment to 8"). A `<2 x float>` staging
# buffer would have hit it too and nothing in the tree had one.
const CMV4 = NTuple{4,VecElement{Float16}}

@kernel cpu=false unsafe_indices=true function coopmat_vec4_tile!(out, @Const(A))
    sh = @localmem CMV4 (4 * 16,)      # a 16x16 fp16 tile = 4 vec4 per column
    t = @index(Local, Linear) - 1
    @inbounds begin
        if t < 64
            i4 = t % 4; j = t ÷ 4
            sh[1 + i4 + j * 4] = (VecElement(A[1 + 4i4 + j * 16]),
                                  VecElement(A[2 + 4i4 + j * 16]),
                                  VecElement(A[3 + 4i4 + j * 16]),
                                  VecElement(A[4 + 4i4 + j * 16]))
        end
        @synchronize
        if t < 32
            # stride counted in vec4, hence 4 rather than 16
            a = AcceleratedMatrix{Float16,16,16,MatrixA}(sh, 1, 4)
            copyto!(pointer(out), 1, 16, a)
        end
    end
end

@testset "cooperative matrix from a vec4-typed @localmem" begin
    backend = LavaBackend()
    if !Mantle.coopmat_gemm_available()
        @info "skipping: no cooperative-matrix support on this device"
    else
        h = Float16.(reshape(1:256, 16, 16))
        A = LavaArray(vec(h))
        out = LavaArray(fill(Float16(-1), 256))
        coopmat_vec4_tile!(backend, 128)(out, A; ndrange = 128)
        KernelAbstractions.synchronize(backend)
        @test reshape(Array(out), 16, 16) == h
    end
end

@testset "workgroup layout sizes and alignments for vectors" begin
    # The rule the fix encodes, checked without a GPU: packed size, and Vulkan's
    # 2x/4x alignment. A vector was previously reported as 4 bytes whatever it
    # was, which is why only the 2-wide fp16 case worked.
    LLVM.@dispose ctx = LLVM.Context() begin
        h = LLVM.HalfType(); f = LLVM.FloatType()
        for (ty, sz, al) in ((LLVM.VectorType(h, 2),  4,  4),
                             (LLVM.VectorType(h, 4),  8,  8),
                             (LLVM.VectorType(f, 2),  8,  8),
                             (LLVM.VectorType(f, 4), 16, 16))
            @test Lava.wg_compute_type_size(ty) == sz
            @test Lava.wg_compute_type_alignment(ty) == al
        end
    end
end

# The layout operand on a *store* into `@localmem`, which the load has taken
# since the staged GEMM needed it and the store did not.
#
# Stated as a transpose, because that is the whole observable difference: load a
# tile column-major, store it back row-major, and what lands in shared is the
# transpose of what came in. A store that ignored the operand would return the
# tile unchanged and pass any test that only checked "it wrote something".
@kernel cpu=false function coopmat_store_rowmajor!(out, @Const(inp))
    tile = @localmem Float16 (16 * 16,)
    i = @index(Local, Linear)
    @inbounds for k in 0:7
        j = i + k * 32
        tile[j] = inp[j]
    end
    @synchronize
    m = AcceleratedMatrix{Float16,16,16,MatrixA}(tile, 1, 16)
    @synchronize                      # nothing may overwrite the tile mid-load
    copyto!(tile, 1, 16, m, Val(true))
    @synchronize
    @inbounds for k in 0:7
        j = i + k * 32
        out[j] = tile[j]
    end
end

@testset "row-major cooperative-matrix store into @localmem" begin
    backend = LavaBackend()
    if !Mantle.coopmat_gemm_available()
        @info "skipping: no cooperative-matrix support on this device"
    else
        h = Float16.(reshape(1:256, 16, 16))
        A = LavaArray(vec(h))
        out = LavaArray(fill(Float16(-1), 256))
        coopmat_store_rowmajor!(backend, 32)(out, A; ndrange = 32)
        KernelAbstractions.synchronize(backend)
        @test reshape(Array(out), 16, 16) == permutedims(h)
    end
end

