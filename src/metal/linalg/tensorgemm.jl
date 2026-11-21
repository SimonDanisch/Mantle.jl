# Metal 4 tensor-ops GEMM for Mantle's graphs: a dense product with a fused row bias and
# elementwise epilogue, and a strided-batched product with per-plane transposes, a scale
# and a per-column divisor. Both accumulate in a Float32 threadgroup tile.
#
# Metal.jl's own tensor GEMM (`Metal.gemm_tensor_kernel!`) does not cover these: it
# accumulates into the destination over K slices, has no bias, epilogue or batch, and is
# reached through `mul!`, which launches a kernel rather than describing one for a graph
# to record. Its simdgroup GEMM does cover the plain product, and is the fallback here.

## element types

"""
    tensoreltypes(TA, TB, TC) -> Bool

Whether the tensor kernels here compute `C = A*B` for these element types. Both accumulate
in a Float32 threadgroup tile, so `matmul2d` itself runs `(TA, TB) -> Float32` and `TC`
is only what the store converts to.
"""
tensoreltypes(::Type{T}, ::Type{T}, ::Type{T}) where {T<:Union{Float16,Float32,BFloat16}} =
    tensoroperand(T)
# A half product landing in single. Both tensor bodies accumulate in a Float32
# threadgroup tile and convert only at the store, so the destination's type is the
# store's business and the product is the same one. Without this, fp16 operands
# with an fp32 destination fell to the SIMD kernel: 0.6 TFLOP/s on an M5 at
# 17408x512x5120, where this path runs the same product at 11.
tensoreltypes(::Type{T}, ::Type{T}, ::Type{Float32}) where {T<:Union{Float16,BFloat16}} =
    tensoroperand(T)
tensoreltypes(::Type, ::Type, ::Type) = false

# The "MatMul2d data type supported" table of the MSL specification lists BFloat16
# operands from macOS 26.1.
tensoroperand(::Type) = true
tensoroperand(::Type{BFloat16}) = macos_version() >= v"26.1"

"""
    tensorcapable(dev::MTLDevice) -> Bool

Whether `dev` runs Metal 4 tensor operations: this macOS offers Metal 4 and the device is
in the Metal 4 family. Gating a dispatch on it also keeps the `__tensorops_*` symbols out
of pipelines for devices whose validator rejects them.
"""
tensorcapable(dev::MTL.MTLDevice) =
    Metal.metal_support() >= v"4" && MTL.supports_family(dev, MTL.MTLGPUFamilyMetal4)

"""
    simdeltypes(TA, TB, TC) -> Bool

Whether Metal.jl's simdgroup GEMM kernel covers these element types. It stages both
operands in threadgroup memory as Float32 and accumulates in Float32, so only its loads
and its store are typed.
"""
simdeltypes(::Type{<:Union{Float16,Float32,BFloat16}},
            ::Type{<:Union{Float16,Float32,BFloat16}},
            ::Type{<:Union{Float16,Float32,BFloat16}}) = true
simdeltypes(::Type, ::Type, ::Type) = false

## dense product

@inline addbias(value::Float32, ::Nothing, row) = value
@inline addbias(value::Float32, bias, row) = value + Float32(@inbounds bias[row])

"""
    gemm_tensor_kernel!(C, A, B, bias, epilogue, M, N, K, Val(TM), Val(TN), Val(NSIMD))

`C = epilogue.(A*B .+ bias)`, one threadgroup of `NSIMD` simdgroups per `TM x TN` tile of
`C`. `bias` is `nothing` or one value per row, `epilogue` a function of a Float32.

One threadgroup computes one `C[TM, TN]` output tile, in ONE `matmul2d` over the whole
contraction.

**The accumulator is a THREADGROUP tile in Float32, not the destination in device memory.**
`matmul2d` takes its destination in whichever address space the tensor names, and pointing
it at `C` made this a read-modify-write of the output tile through device memory — at the
destination's OWN precision, so an fp16 `C` rounded the running sum once per K slice.
Accumulating in threadgroup memory costs `TM*TN*4` bytes and is worth, measured on an M5 at
fp16 `576x4096x2304`, 4.12 -> 8.29 TFLOP/s AND a relative error against MPSGraph of
4.8e-3 -> 8.0e-4. The bias and the epilogue apply on the way out, where the value is already
in registers, rather than as a second device round-trip over `C`.

**And the contraction is ONE call, not a loop over K slices.** Apple's own schedule for the
whole contraction beats any slicing this side can write: measured over the 13 distinct GEMM
shapes of SAM 2.1's encoder, 263.1 -> 186.2 ms, and it wins on all eight out-of-sample
shapes tried as well. It is also EXACT against MPSGraph — bit-identical on every shape here,
where the sliced form differed by 8e-4 — which says the two are running the same thing.
Two constraints go away with the loop: `K` needs no tile (so no `K % 8`, verified
bit-identical at K = 76 and 100), and there is no trip count to keep dynamic.
"""
function gemm_tensor_kernel!(C::MtlDeviceArray, A::MtlDeviceArray, B::MtlDeviceArray,
                             bias, epilogue, M::UInt32, N::UInt32, K::UInt32,
                             ::Val{TM}, ::Val{TN}, ::Val{NSIMD}) where {TM, TN, NSIMD}
    tgid  = threadgroup_position_in_grid_3d()
    m_off = (unsafe_trunc(Int32, tgid.x) - Int32(1)) * Int32(TM)
    n_off = (unsafe_trunc(Int32, tgid.y) - Int32(1)) * Int32(TN)

    tA = MtlInlineTensor(A, (M, K))
    tB = MtlInlineTensor(B, (K, N))

    acc = MtlThreadGroupArray(Float32, (TM, TN))
    mC = view(MtlInlineTensor(acc), (Int32(1), Int32(1)), (Int32(TM), Int32(TN)))

    # `k = -1`: the contraction length comes from the operand views at run time.
    product = TensorOpsMatmul2D{matmul2d_descriptor(TM, TN, -1;
                                                    mode = matmul2d_multiply),
                                Int32(NSIMD)}()
    Kc = unsafe_trunc(Int32, K)
    product(view(tA, (m_off + Int32(1), Int32(1)), (Int32(TM), Kc)),
            view(tB, (Int32(1), n_off + Int32(1)), (Kc, Int32(TN))), mC)
    # matmul2d is collective over the threadgroup, so the accumulator is only complete for
    # every thread after a barrier — and it is a THREADGROUP barrier, matching where the
    # tile lives.
    threadgroup_barrier(MemoryFlagThreadGroup)
    tid = Int(thread_index_in_threadgroup()) - 1
    stride = NSIMD * 32
    for linear in tid:stride:(TM * TN - 1)
        mi = linear % TM
        ni = linear ÷ TM
        row = Int(m_off) + mi + 1
        col = Int(n_off) + ni + 1
        @inbounds value = acc[mi + 1, ni + 1]
        value = epilogue(addbias(value, bias, row))
        @inbounds C[row, col] = value
    end
    return
end

## strided-batched product

"""
    gemm_tensor_batched_kernel!(C, A, B, coldiv, alpha, M, N, K, Val(TM), Val(TN), Val(TK),
                                Val(NSIMD), Val(TA), Val(TB), Val(HASSCALE), Val(HASDIV))

Strided-batched tensor GEMM, `C = alpha * op(A) * op(B)`, per plane, with `op` a transpose
where `TA`/`TB`. The matrix plane is contiguous and every trailing dimension is flattened
into the grid's z axis. Keeping the transpose in the tensor descriptor avoids
materialising attention's `(E,L,H,B) -> (L,E,H,B)` query permutation.
"""
function gemm_tensor_batched_kernel!(C::MtlDeviceArray, A::MtlDeviceArray,
                                     B::MtlDeviceArray, coldiv, alpha::Float32,
                                     M::UInt32, N::UInt32, K::UInt32,
                                     ::Val{TM}, ::Val{TN}, ::Val{TK},
                                     ::Val{NSIMD}, ::Val{TA}, ::Val{TB},
                                     ::Val{HASSCALE}, ::Val{HASDIV}) where
                                    {TM, TN, TK, NSIMD, TA, TB, HASSCALE, HASDIV}
    tgid = threadgroup_position_in_grid_3d()
    m_off = (unsafe_trunc(Int32, tgid.x) - Int32(1)) * Int32(TM)
    n_off = (unsafe_trunc(Int32, tgid.y) - Int32(1)) * Int32(TN)
    batch = unsafe_trunc(Int32, tgid.z) - Int32(1)

    arows = TA ? K : M
    acols = TA ? M : K
    brows = TB ? N : K
    bcols = TB ? K : N
    A2 = MtlDeviceArray((Int(arows), Int(acols)),
                        pointer(A, Int(batch) * Int(M) * Int(K) + 1))
    B2 = MtlDeviceArray((Int(brows), Int(bcols)),
                        pointer(B, Int(batch) * Int(K) * Int(N) + 1))
    C2 = MtlDeviceArray((Int(M), Int(N)),
                        pointer(C, Int(batch) * Int(M) * Int(N) + 1))

    tA = MtlInlineTensor(A2)
    tB = MtlInlineTensor(B2)
    # Threadgroup Float32 accumulator, for the reason spelled out at `gemm_tensor_kernel!`:
    # a device-memory destination turns each of the `K/TK` slices into a read-modify-write
    # of the output tile at the destination's own precision. `alpha` folds into the
    # write-out below, so a scaled product costs no second pass over `C` either.
    acc = MtlThreadGroupArray(Float32, (TM, TN))
    mC = view(MtlInlineTensor(acc), (Int32(1), Int32(1)), (Int32(TM), Int32(TN)))

    initialize = TensorOpsMatmul2D{
        matmul2d_descriptor(TM, TN, TK; transpose_left = TA,
                            transpose_right = TB, mode = matmul2d_multiply),
        Int32(NSIMD)}()
    accumulate = TensorOpsMatmul2D{
        matmul2d_descriptor(TM, TN, TK; transpose_left = TA,
                            transpose_right = TB,
                            mode = matmul2d_multiply_accumulate),
        Int32(NSIMD)}()
    # Dynamic trip count: a compile-time constant one crashes Apple's back-end (see
    # `matmul2d_descriptor`).
    nslices = unsafe_trunc(Int32, K ÷ UInt32(TK))

    aextent = TA ? (Int32(TK), Int32(TM)) : (Int32(TM), Int32(TK))
    bextent = TB ? (Int32(TN), Int32(TK)) : (Int32(TK), Int32(TN))

    aorigin = TA ? (Int32(1), m_off + Int32(1)) :
                   (m_off + Int32(1), Int32(1))
    borigin = TB ? (n_off + Int32(1), Int32(1)) :
                   (Int32(1), n_off + Int32(1))
    mA = view(tA, aorigin, aextent)
    mB = view(tB, borigin, bextent)
    initialize(mA, mB, mC)
    for s in Int32(1):(nslices - Int32(1))
        k_off = s * Int32(TK)
        aorigin = TA ? (k_off + Int32(1), m_off + Int32(1)) :
                       (m_off + Int32(1), k_off + Int32(1))
        borigin = TB ? (n_off + Int32(1), k_off + Int32(1)) :
                       (k_off + Int32(1), n_off + Int32(1))
        mA = view(tA, aorigin, aextent)
        mB = view(tB, borigin, bextent)
        accumulate(mA, mB, mC)
    end

    threadgroup_barrier(MemoryFlagThreadGroup)
    tid = Int(thread_index_in_threadgroup()) - 1
    stride = NSIMD * 32
    for linear in tid:stride:(TM * TN - 1)
        mi = linear % TM
        ni = linear ÷ TM
        row = Int(m_off) + mi + 1
        col = Int(n_off) + ni + 1
        @inbounds value = acc[mi + 1, ni + 1]
        HASSCALE && (value *= alpha)
        # One divisor per output COLUMN, laid out `(N, batch...)` — attention's unnormalized
        # softmax leaves its row sums for the apply product to divide by, and doing it here
        # saves a second pass over an output this kernel already holds in registers.
        HASDIV && (@inbounds value /= Float32(coldiv[Int(batch) * Int(N) + col]))
        @inbounds C2[row, col] = value
    end
    return
end

## tile selection

# `matmul2d` tiles every axis in multiples of eight.
const TENSOR_MULT = 8
# What a Float32 accumulator may take of the 32 KiB of threadgroup memory a threadgroup gets.
const TENSOR_TGMEM = 32768
# A dimension that a power-of-two tile of at least this size divides takes that tile; any
# other takes the largest multiple of eight that divides it (`tilemultiple`).
const TENSOR_POW2MIN = 16

"""The first of `candidates` that divides `d`, or 0 for none."""
function tiledivisor(d::Integer, candidates)
    for t in candidates
        d % t == 0 && return t
    end
    return 0
end

"""
    tilemultiple(d, below = typemax(Int); largest = 128) -> Int

The largest multiple of eight that divides `d`, is at most `largest` and is strictly below
`below`, or 0.

A dimension NO tile of sixteen or more divides takes this tile instead. `matmul2d` accepts
any multiple of eight, verified bit-exact against the 8-row tile for every one of
8/16/24/32/36/40/48/56/64/72/96/128 on an M5. The power-of-two lists are still what a
dimension divisible by 16 uses, because they are the ones measured against this workload's
`addmm` shapes and a LARGER tile is not automatically better there (M=1152 at fp16
`1152x4096x1152`: the 64-tile is 6.59 TFLOP/s, the 128-tile 3.41). This is for a dimension
like attention's head width E = 72, where the power-of-two rule lands on 8 and the other
operand is then read M/8 times over. Covering E in ONE tile reads it once: measured on
SAM 2.1's 4096-token global block, fp16, `72x4096x4096` batched over 8 heads,
7.72 -> 2.89 ms (2.51 -> 6.70 TFLOP/s).
"""
function tilemultiple(d::Integer, below::Integer = typemax(Int); largest::Int = 128)
    t = min(Int(d), largest, Int(below) - TENSOR_MULT)
    t -= t % TENSOR_MULT
    while t >= TENSOR_MULT
        d % t == 0 && return t
        t -= TENSOR_MULT
    end
    return 0
end

"""
    tilefit(M, N, tm, tn) -> (tm, tn)

Shrink the output tile until its Float32 accumulator fits threadgroup memory. Each step
strictly shrinks one side, so this ends at a fit or at "no usable tile" (a zero), which is
better than launching a kernel the driver rejects.
"""
function tilefit(M::Integer, N::Integer, tm::Int, tn::Int)
    while tm != 0 && tn != 0 && tm * tn * sizeof(Float32) > TENSOR_TGMEM
        if tn >= tm
            tn = tilemultiple(N, tn)
        else
            tm = tilemultiple(M, tm)
        end
    end
    return (tm, tn)
end

"""
    gemm_tensor_tiles(M, N; mtiles, ntiles, nsimd) -> (tm, tn, nsimd)

The output tile and simdgroup count of the dense product; zeros mean no usable tile. `K`
takes no tile — the contraction is one `matmul2d` call whatever its length.

`64 x 32` with ONE simdgroup, flat. Measured on an M5, fp16, over the 13 distinct GEMM
shapes of SAM 2.1's encoder: 186.2 ms against 181.4 for picking each shape's best tile by
hand, so a per-shape rule there is worth 2.6% and not worth having. (The same sweep over the
SLICED kernel this replaced came to 263.1 ms at ITS best fixed tile.) `64 x 64` costs 4%,
`64 x 16` 13%, and two simdgroups 4%.
"""
function gemm_tensor_tiles(M::Integer, N::Integer; mtiles = (64, 32, 16, 8),
                           ntiles = (32, 16, 8), nsimd::Int = 1)
    tm = tiledivisor(M, mtiles)
    tn = tiledivisor(N, ntiles)
    tm < TENSOR_POW2MIN && (tm = max(tm, tilemultiple(M)))
    tn < TENSOR_POW2MIN && (tn = max(tn, tilemultiple(N)))
    tm, tn = tilefit(M, N, tm, tn)
    (tm == 0 || tn == 0) && return (0, 0, 0)
    return (tm, tn, nsimd)
end

"""
    gemm_tensor_batched_tiles(M, N, K; ...) -> (tm, tn, tk, nsimd)

The tiles and simdgroup count of the strided-batched product, which keeps its K loop;
zeros mean no usable tile.

The output tile depends on the contraction DEPTH: one `matmul2d` over the whole contraction
is 2.2 TFLOP/s there against the sliced form's 5.2 — the reverse of the dense case — and
attention's two products, which are what this path runs, both read an operand transposed.
A contraction of at least `kdeep` takes `deeptiles` and `nsimd_deep` simdgroups, a
shallower one `shallowtiles` and `nsimd_shallow`.
"""
function gemm_tensor_batched_tiles(M::Integer, N::Integer, K::Integer; kdeep::Int = 1024,
                                   deeptiles = (64, 32, 16, 8),
                                   shallowtiles = (32, 16, 8), ktiles = (32, 16, 8),
                                   nsimd_deep::Int = 2, nsimd_shallow::Int = 1)
    deep = K >= kdeep
    mn = deep ? deeptiles : shallowtiles
    tm = tiledivisor(M, mn)
    tn = tiledivisor(N, mn)
    tm < TENSOR_POW2MIN && (tm = max(tm, tilemultiple(M)))
    tn < TENSOR_POW2MIN && (tn = max(tn, tilemultiple(N)))
    # The contraction takes the same fallback, and for the same reason: attention's scores
    # contract over E = 72, where the 8-slice loop runs nine times to the 72-slice's one.
    # Measured on the 4096-token block, `4096x4096x72` over 8 heads: 12.17 -> 10.38 ms.
    tk = tiledivisor(K, ktiles)
    tk < TENSOR_POW2MIN && (tk = max(tk, tilemultiple(K)))
    tm, tn = tilefit(M, N, tm, tn)
    (tm == 0 || tn == 0 || tk == 0) && return (0, 0, 0, 0)
    return (tm, tn, tk, deep ? nsimd_deep : nsimd_shallow)
end

## dispatch descriptions

"""
    gemm_kernel_config(dev, C, A, B; bias = nothing, epilogue = identity, simd = ...)
        -> (; kernel, args, ndrange, group, fused) or nothing

The fastest recordable kernel on `dev` for `C = epilogue.(A*B .+ bias)`, for a graph to
dispatch: `ndrange` and `group` count threads, and `fused` says whether the row `bias` and
the elementwise `epilogue` were folded in, so the caller declares the rest. `nothing` for
element types no kernel here covers.

The tensor-ops kernel where `dev` has tensor operations and a tile fits; otherwise
Metal.jl's simdgroup GEMM, which fuses neither. `simd` is that kernel's tile: a `wm x wn`
grid of simdgroups each computing a `tm x tn` grid of 8x8 blocks, over `kb` 8-deep slices
of K. The defaults are Metal.jl's measured `GEMM_SIMD_*` values; its tile staging needs
`64·tm·wm·kb` and `64·tn·wn·kb` to be multiples of the `32·wm·wn` threads.
"""
function gemm_kernel_config(dev::MTL.MTLDevice, C, A, B; bias = nothing,
                            epilogue = identity,
                            simd = (; wm = 2, wn = 2, tm = 4, tn = 4, kb = 1))
    simdeltypes(eltype(A), eltype(B), eltype(C)) || return nothing
    M, N, K = size(C, 1), size(C, 2), size(A, 2)
    if tensorcapable(dev) && tensoreltypes(eltype(A), eltype(B), eltype(C))
        tm, tn, nsimd = gemm_tensor_tiles(M, N)
        if tm != 0 && tn != 0
            threads = nsimd * 32
            fusebias = bias !== nothing && length(bias) == M
            args = (C, A, B, fusebias ? bias : nothing, epilogue,
                    UInt32(M), UInt32(N), UInt32(K),
                    Val(Int32(tm)), Val(Int32(tn)), Val(Int32(nsimd)))
            return (; kernel = gemm_tensor_kernel!, args,
                    ndrange = ((M ÷ tm) * threads, N ÷ tn), group = (threads, 1),
                    fused = (; bias = fusebias, epilogue = epilogue !== identity))
        end
    end

    (; wm, wn, tm, tn, kb) = simd
    bm, bn = 8 * tm * wm, 8 * tn * wn
    threads = wm * wn * 32
    # An assembled `<64 x bfloat>` fragment stores zeros (Metal.jl's `simd_direct_store`),
    # so a BFloat16 destination goes through the kernel's Float32 scratch, which writes `C`
    # elementwise, like a tile on the edge does.
    direct = eltype(C) !== BFloat16
    edge = !(M % bm == 0 && N % bn == 0) || !direct
    args = (C, A, B, 1.0f0, 0.0f0, M, N, K,
            Val('N'), Val('N'), Val(wm), Val(wn), Val(tm), Val(tn), Val(kb),
            Val(edge), Val(true), Val(direct))
    return (; kernel = Metal.gemm_simd_kernel!, args,
            ndrange = (cld(M, bm) * threads, cld(N, bn)), group = (threads, 1),
            fused = (; bias = false, epilogue = false))
end

"""
    batched_gemm_kernel_config(dev, C, A, B; transpose_a = false, transpose_b = false,
                               alpha = 1, coldiv = nothing)
        -> (; kernel, args, ndrange, group) or nothing

The tensor-ops strided-batched GEMM, `C = alpha * op(A) * op(B)` per plane, for a graph to
dispatch. The first two axes hold each matrix and all remaining axes form a contiguous
batch. `coldiv` fuses a per-output-column divisor, laid out `(N, batch...)`, into the
store. `nothing` when `dev` has no tensor operations or no tile fits.
"""
function batched_gemm_kernel_config(dev::MTL.MTLDevice, C, A, B; transpose_a = false,
                                    transpose_b = false, alpha = 1, coldiv = nothing)
    tensorcapable(dev) || return nothing
    tensoreltypes(eltype(A), eltype(B), eltype(C)) || return nothing
    ndims(A) >= 3 && ndims(B) >= 3 && ndims(C) >= 3 || return nothing

    M = transpose_a ? size(A, 2) : size(A, 1)
    K = transpose_a ? size(A, 1) : size(A, 2)
    Kb = transpose_b ? size(B, 2) : size(B, 1)
    N = transpose_b ? size(B, 1) : size(B, 2)
    size(C, 1) == M && size(C, 2) == N && Kb == K || return nothing
    nbatch = prod(size(C)[3:end])
    prod(size(A)[3:end]) == nbatch || return nothing
    prod(size(B)[3:end]) == nbatch || return nothing
    tm, tn, tk, nsimd = gemm_tensor_batched_tiles(M, N, K)
    tm != 0 && tn != 0 && tk != 0 || return nothing

    threads = nsimd * 32
    hasdiv = coldiv !== nothing
    hasdiv && (length(coldiv) == N * nbatch || return nothing)
    args = (C, A, B, hasdiv ? coldiv : nothing, Float32(alpha),
            UInt32(M), UInt32(N), UInt32(K),
            Val(Int32(tm)), Val(Int32(tn)), Val(Int32(tk)), Val(Int32(nsimd)),
            Val(Bool(transpose_a)), Val(Bool(transpose_b)), Val(!isone(alpha)), Val(hasdiv))
    return (; kernel = gemm_tensor_batched_kernel!, args,
            ndrange = ((M ÷ tm) * threads, N ÷ tn, nbatch),
            group = (threads, 1, 1))
end
