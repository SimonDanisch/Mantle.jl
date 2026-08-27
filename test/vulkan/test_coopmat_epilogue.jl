"""
The two cooperative-matrix operations a fused GEMM epilogue needs.

Our GEMM accumulates in fp32 and the model's destination is fp16, and there was
no way across in registers: the kernel stored fp32 to a scratch and
`mm_epilogue_kernel!` read all of it back, added the bias and wrote fp16. That
second pass is **23% of matmul time** — 0.78 ms against the GEMM's 2.65 over one
call of each of the encoder's six shapes — and `splitk == 1` for every one of
them, so it is not even doing a reduction. It is a copy with a bias in it.

Two things remove it, both checked here:

  * **`convert`** between cooperative matrix types (`OpFConvert`), so an fp32
    accumulator can be stored into an fp16 destination directly.
  * **a stride-0 load**, which broadcasts a length-`M` vector across every
    column of a tile — so the accumulator can be *initialised* with the bias
    instead of zero, and the bias add costs nothing at all.

Neither is exotic, but neither was available, and the stride-0 behaviour is not
something the Vulkan spec spells out — it is checked rather than assumed.
"""

using Test, Lava, KernelAbstractions
using Lava: AcceleratedMatrix, Accumulator, MatrixA, MatrixB
const KA = KernelAbstractions

"Broadcast `bias` across a tile, reading it in its OWN element type."
@kernel cpu=false function bias_broadcast_tile!(out, @Const(bias))
    lane = @index(Global, Linear) - 1
    if lane < 32
        m = AcceleratedMatrix{eltype(bias),16,16,Accumulator}(pointer(bias), 1, 0)
        copyto!(pointer(out), 1, 16,
                convert(AcceleratedMatrix{eltype(out),16,16,Accumulator}, m))
    end
end

@kernel cpu=false function convert_store_tile!(out, @Const(A), @Const(B), @Const(bias))
    lane = @index(Global, Linear) - 1
    if lane < 32
        c = AcceleratedMatrix{Float32,16,16,Accumulator}(pointer(bias), 1, 0)
        a = AcceleratedMatrix{Float16,16,16,MatrixA}(pointer(A), 1, 16)
        b = AcceleratedMatrix{Float16,16,16,MatrixB}(pointer(B), 1, 16)
        c = muladd(a, b, c)
        copyto!(pointer(out), 1, 16, convert(AcceleratedMatrix{Float16,16,16,Accumulator}, c))
    end
end

"""
The same buffer as an A fragment both ways, multiplied by the identity so the
stored result is the fragment itself.

`MemoryLayout` was hardcoded to column-major, which forecloses the reference's
scheme: `mul_mm.comp` stages A and B identically and reads A `RowMajor` and B
`ColumnMajor`, because A is `(M, K)`, B is `(K, N)` and they share the k axis.
With one layout, one of the two has to be transposed while staging.
"""
@kernel cpu=false function layout_probe!(outc, outr, @Const(A), @Const(Id))
    lane = @index(Global, Linear) - 1
    if lane < 32
        b = AcceleratedMatrix{Float16,16,16,MatrixB}(pointer(Id), 1, 16)
        z() = zero(AcceleratedMatrix{Float32,16,16,Accumulator})
        ac = AcceleratedMatrix{Float16,16,16,MatrixA}(pointer(A), 1, 16)
        ar = AcceleratedMatrix{Float16,16,16,MatrixA}(pointer(A), 1, 16, Val(true))
        copyto!(pointer(outc), 1, 16, muladd(ac, b, z()))
        copyto!(pointer(outr), 1, 16, muladd(ar, b, z()))
    end
end

@kernel cpu=false function convert_roundtrip!(out, @Const(A))
    lane = @index(Global, Linear) - 1
    if lane < 32
        a = AcceleratedMatrix{Float32,16,16,Accumulator}(pointer(A), 1, 16)
        h = convert(AcceleratedMatrix{Float16,16,16,Accumulator}, a)
        copyto!(pointer(out), 1, 16, convert(AcceleratedMatrix{Float32,16,16,Accumulator}, h))
    end
end

@testset "cooperative-matrix epilogue" begin
    backend = LavaBackend()

    @testset "stride 0 broadcasts a vector across the tile" begin
        # **Both element types.** Under autocast the model's biases are fp16, and
        # a version of this that only covered fp32 shipped a kernel reading fp16
        # bytes as fp32 — the unit tests passed and SAM 2's masks went to
        # IoU 0.0. A bias is not always fp32; test what the model has.
        for T in (Float32, Float16)
            hb = T.(1:16)
            bias = KA.allocate(backend, T, 16); copyto!(bias, hb)
            out = KA.allocate(backend, Float32, 16, 16); fill!(out, -1.0f0)
            bias_broadcast_tile!(backend, 32)(out, bias; ndrange = 32)
            KA.synchronize(backend)
            g = Array(out)
            @test all(j -> g[:, j] == Float32.(hb), 1:16)
        end
    end

    @testset "both memory layouts are available" begin
        hA = Float16.(reshape(1:256, 16, 16))
        hI = zeros(Float16, 16, 16)
        for i in 1:16
            hI[i, i] = one(Float16)
        end
        A = KA.allocate(backend, Float16, 16, 16); copyto!(A, hA)
        Id = KA.allocate(backend, Float16, 16, 16); copyto!(Id, hI)
        oc = KA.allocate(backend, Float32, 16, 16); fill!(oc, 0.0f0)
        orow = KA.allocate(backend, Float32, 16, 16); fill!(orow, 0.0f0)
        layout_probe!(backend, 32)(oc, orow, A, Id; ndrange = 32)
        KA.synchronize(backend)
        @test Array(oc) == Float32.(hA)             # column-major: as stored
        @test Array(orow) == Float32.(hA)'          # row-major: transposed
    end

    @testset "convert changes only the component type" begin
        # Values exactly representable in fp16, so the round trip is exact and a
        # failure means the conversion moved data, not that it rounded.
        hA = Float32.(reshape(collect(-128:127), 16, 16)) ./ 4
        A = KA.allocate(backend, Float32, 16, 16); copyto!(A, hA)
        out = KA.allocate(backend, Float32, 16, 16); fill!(out, -1.0f0)
        convert_roundtrip!(backend, 32)(out, A; ndrange = 32)
        KA.synchronize(backend)
        @test Array(out) == hA
    end

    @testset "bias + mma + fp16 store, in registers" begin
        hA = rand(Float16, 16, 16) .- Float16(0.5)
        hB = rand(Float16, 16, 16) .- Float16(0.5)
        hbias = Float32.(1:16) ./ 8
        A = KA.allocate(backend, Float16, 16, 16); copyto!(A, hA)
        B = KA.allocate(backend, Float16, 16, 16); copyto!(B, hB)
        bias = KA.allocate(backend, Float32, 16); copyto!(bias, hbias)
        out = KA.allocate(backend, Float16, 16, 16); fill!(out, zero(Float16))
        convert_store_tile!(backend, 32)(out, A, B, bias; ndrange = 32)
        KA.synchronize(backend)
        got = Float32.(Array(out))
        ref = Float32.(hA) * Float32.(hB) .+ hbias
        # fp16 destination, so the tolerance is the format's, not the kernel's.
        @test maximum(abs.(got .- ref)) / maximum(abs.(ref)) < 1.0f-3
    end
end
