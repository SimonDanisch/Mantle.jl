# A GEMM with no shared memory: workgroup-scope cooperative matrices and
# tensor-addressed loads, after llama.cpp's `mul_mm_cm2.comp`. Not routed to: it
# loses to `gemm.jl`'s staged kernel by 22-30% at its best tiling. The cause is
# register allocation: every route to more work per lane (bigger tile, deeper
# `BK`, more invocations) makes the driver allocate two to four times the
# registers the matrices need, up to 255, where the staged kernel sits at 128; the
# tensor-addressed load itself costs that even into subgroup-scope tiles
# (`gemm_cm2_sg!`). Clamped loads are the fast path on the measured driver: the
# reference's unclamped layout ran 2.8x slower without more register pressure.
#
# No transposing views: a column-major Julia `(r, c)` array is the tensor `(c, r)`,
# so `C = A*B` is `C̃ = B̃·Ã` and B supplies the A-operand, A the B-operand. Clamped
# loads pad with zeros, so extents need not divide the tile.

"""
    gemm_cm2_fits(dev, BM, BN, BK, NT) -> Bool

Whether this device can run a `(BM, BN, BK, NT)` tiling: `NT` must be a workgroup
size it reports workgroup-scope matrices for, and each extent a multiple of that
size's granularity on its axis. The kernel computes the transposed product, so
`BN` is checked as M, `BM` as N, `BK` as K.
"""
function gemm_cm2_fits(dev, BM::Int, BN::Int, BK::Int, NT::Int)
    NT <= dev.workgrouplimit || return false
    g = wggranularity(dev, NT)
    g === nothing && return false
    Mg, Ng, Kg = g
    BN % Mg == 0 && BM % Ng == 0 && BK % Kg == 0
end

"""
    gemm_cm2_tiling(dev) -> (BM, BN, BK, NT) | nothing

The tiling to run, or `nothing` where the device has no workgroup-scope matrices.
Tries the measured winner `64x64/32 @256`, then the smallest legal tile at each
reported workgroup size, largest first, since fixed literals may be illegal on
another device.
"""
function gemm_cm2_tiling(dev::DeviceCaps)
    isempty(dev.wggran) && return nothing
    # The measured winner, if this device admits it.
    gemm_cm2_fits(dev, 64, 64, 32, 256) && return (64, 64, 32, 256)
    for (nt, mg, ng, kg) in Iterators.reverse(dev.wggran)
        # The smallest legal tile at this size: small tiles stay inside the
        # register file.
        BM, BN, BK = max(64, ng), max(64, mg), max(32, kg)
        gemm_cm2_fits(dev, BM, BN, BK, nt) && return (BM, BN, BK, nt)
    end
    nothing
end

"""
    gemm_cm2!(C, A, B, M, N, K, ::Val{BM}, ::Val{BN}, ::Val{BK})

`C = A*B` for column-major fp16 `A (M,K)` and `B (K,N)`, `C (M,N)` fp16 or fp32.
One workgroup owns a `BM x BN` tile of `C` and walks `K` in steps of `BK`, each
step two tensor loads and one `coopmat_muladd` over matrices spanning all
invocations, with no shared memory or barrier.
"""
function gemm_cm2!(
        C, A, B, M::Int32, N::Int32, K::Int32,
        ::Val{BM}, ::Val{BN}, ::Val{BK}, ::Val{UNROLL} = Val(false),
        ::Val{CM} = Val(TENSOR_CLAMP_CONSTANT)) where {BM,BN,BK,UNROLL,CM}
    WM = WorkgroupMatrix
    grp = KI.get_group_id()
    m0 = Int32((grp.x - 1) * BM)
    n0 = Int32((grp.y - 1) * BN)

    # A 2-D layout over a column-major Julia `(r, c)` array: the tensor `(c, r)`
    # with strides `(r, 1)`. The clamp mode is the type parameter itself: `Val(x)`
    # of a local is not a compile-time constant and fails with `InvalidIRError`.
    # Dimension and stride are built once outside the loop, as the reference does,
    # and only the slice moves with `k`; building them per load cost 177 against
    # 128 registers at `128x128/32`, one resident workgroup instead of two.
    base(r, c) = tensor_setstride(
        tensor_setdim(tensor_layout(Val(2), Val(CM)), (c, r)), (r, Int32(1)))
    @inline slice(l, o0, o1, n0, n1) = tensor_slice(l, (o0, o1), (Int32(n0), Int32(n1)))

    baseA = base(K, N)
    baseB = base(M, K)

    # Under a constant clamp, out-of-range elements come from the clamp value, so
    # the destination is never read and a zeroed one only occupies registers.
    # Under `TENSOR_CLAMP_UNDEFINED` the destination survives out of range and
    # must be zero.
    @inline dest(::Type{MT}) where {MT} =
        CM === TENSOR_CLAMP_CONSTANT ? coopmat_undef(MT) : coopmat_zero(MT)
    @inline loadA(k) = tensor_load(dest(WM{Float16,BN,BK,MatrixA}),
                                   UInt64(pointer(B)), slice(baseA, n0, k, BN, BK))
    @inline loadB(k) = tensor_load(dest(WM{Float16,BK,BM,MatrixB}),
                                   UInt64(pointer(A)), slice(baseB, k, m0, BK, BM))

    acc = coopmat_zero(WM{Float32,BN,BM,Accumulator})
    k0 = Int32(0)
    # Four k-steps per iteration, each pair of loads consumed by its own product
    # as in the reference. Issuing all eight loads first keeps eight fragments
    # live and hits the 255-register ceiling; the scheduler overlaps loads anyway.
    if UNROLL
        while k0 + Int32(4 * BK) <= K
            Base.Cartesian.@nexprs 4 j -> begin
                kk_j = k0 + Int32((j - 1) * BK)
                acc = coopmat_muladd(loadA(kk_j), loadB(kk_j), acc)
            end
            k0 += Int32(4 * BK)
        end
    end
    while k0 < K
        acc = coopmat_muladd(loadA(k0), loadB(k0), acc)
        k0 += Int32(BK)
    end
    # A tensor store writes the matrix's own bytes, so an fp16 destination needs
    # the accumulator converted; `eltype` is static, so one store survives.
    olay = slice(base(M, N), n0, m0, BN, BM)
    if eltype(C) === Float16
        tensor_store(coopmat_convert(WM{Float16,BN,BM,Accumulator}, acc),
                     UInt64(pointer(C)), olay)
    else
        tensor_store(acc, UInt64(pointer(C)), olay)
    end
    return nothing
end

"""
    gemm_cm2_sg!(C, A, B, M, N, K, ::Val{NW})

Tensor-addressed loads into subgroup-scope `16x16` tiles: the staged kernel's
`4x4` register block per subgroup (four A and four B fragments per k-step feed
sixteen products) without its shared memory, so each of the `NW` subgroups
fetches its own A-block. Always clamped, the fast path (see the top of the file).
"""
function gemm_cm2_sg!(
        C, A, B, M::Int32, N::Int32, K::Int32,
        ::Val{NW}) where {NW}
    T = GEMM_TILE                       # 16
    RB = 4                              # register block, a literal for `@nexprs`
    AM = AcceleratedMatrix
    tid = KI.get_local_id().x - 1
    w = Int32(tid ÷ 32)                 # this invocation's subgroup
    grp = KI.get_group_id()
    # The workgroup covers `RB*T` of M and `NW*RB*T` of N; subgroup `w` takes the
    # w-th slice of N.
    m0 = Int32((grp.x - 1) * (RB * T))
    n0 = Int32((grp.y - 1) * (NW * RB * T)) + w * Int32(RB * T)

    lay(r, c, o0, o1, n0, n1) =
        tensor_slice(
            tensor_setstride(
                tensor_setdim(tensor_layout(Val(2), Val(TENSOR_CLAMP_CONSTANT)),
                              (c, r)),
                (r, Int32(1))),
            (o0, o1), (Int32(n0), Int32(n1)))

    # `C̃ = B̃·Ã` as above: B supplies the A-operand, A the B-operand.
    Base.Cartesian.@nexprs 4 i -> Base.Cartesian.@nexprs 4 j ->
        acc_i_j = coopmat_zero(AM{Float32,T,T,Accumulator})

    k0 = Int32(0)
    while k0 < K
        Base.Cartesian.@nexprs 4 i -> a_i =
            tensor_load(coopmat_zero(AM{Float16,T,T,MatrixA}), UInt64(pointer(B)),
                        lay(K, N, n0 + Int32((i - 1) * T), k0, T, T))
        Base.Cartesian.@nexprs 4 j -> b_j =
            tensor_load(coopmat_zero(AM{Float16,T,T,MatrixB}), UInt64(pointer(A)),
                        lay(M, K, k0, m0 + Int32((j - 1) * T), T, T))
        Base.Cartesian.@nexprs 4 i -> Base.Cartesian.@nexprs 4 j ->
            (acc_i_j = coopmat_muladd(a_i, b_j, acc_i_j))
        k0 += Int32(T)
    end

    Base.Cartesian.@nexprs 4 i -> Base.Cartesian.@nexprs 4 j -> begin
        olay = lay(M, N, n0 + Int32((i - 1) * T), m0 + Int32((j - 1) * T), T, T)
        if eltype(C) === Float16
            tensor_store(coopmat_convert(AM{Float16,T,T,Accumulator}, acc_i_j),
                         UInt64(pointer(C)), olay)
        else
            tensor_store(acc_i_j, UInt64(pointer(C)), olay)
        end
    end
    return nothing
end

"""
    coopmat_gemm_cm2_sg!(C, A, B, M, N, K; nw = 2) -> C

Launch [`gemm_cm2_sg!`](@ref). `nw` subgroups per workgroup, so the workgroup is
`nw * 32` invocations and covers `64 x (nw * 64)` of `C`.
"""
function coopmat_gemm_cm2_sg!(C, A, B, M::Int, N::Int, K::Int; nw::Int = 2)
    backend = KernelAbstractions.get_backend(C)
    dev = caps(backend)
    # Subgroup-scope tiles: needs tensor addressing and the subgroup shape table,
    # not a workgroup-scope shape.
    KI.supports_tensor_addressing(backend) || return nothing
    KI.supports(backend, MatrixShape(Float16, Float32, GEMM_TILE, GEMM_TILE, GEMM_TILE,
                                     SubgroupScope())) || return nothing
    nw * 32 <= dev.workgrouplimit ||
        throw(ArgumentError("nw=$nw wants $(nw * 32) invocations, past this " *
                            "device's limit of $(dev.workgrouplimit)"))
    span = 4 * GEMM_TILE
    KI.Kernel(backend, gemm_cm2_sg!)(C, A, B, Int32(M), Int32(N), Int32(K), Val(nw);
                                     numgroups = (cld(M, span), cld(N, nw * span)),
                                     workgroupsize = nw * 32)
    C
end

"""
    coopmat_gemm_cm2!(C, A, B, M, N, K; tiling = nothing, unroll = true, clamp = nothing) -> C

Launch [`gemm_cm2!`](@ref) for dense column-major operands. Returns `nothing`
instead of `C` when this device cannot run it, so a caller can fall back. No
batching, split-K, bias or epilogue while it is unrouted.
"""
function coopmat_gemm_cm2!(C, A, B, M::Int, N::Int, K::Int; tiling = nothing,
                           unroll::Bool = true, clamp::Union{Nothing,Bool} = nothing)
    # From the array: an unpinned backend resolves its queue through the global
    # context and would put the work on the wrong GPU.
    backend = KernelAbstractions.get_backend(C)
    dev = caps(backend)
    KI.supports_tensor_addressing(backend) || return nothing
    tiling = something(tiling, gemm_cm2_tiling(dev))
    tiling === nothing && return nothing
    BM, BN, BK, NT = tiling
    # A caller's tiling must still fit; refused here rather than in
    # `vkCreateComputePipelines`, which names neither tiling nor rule.
    gemm_cm2_fits(dev, BM, BN, BK, NT) ||
        throw(ArgumentError("tiling BM=$BM BN=$BN BK=$BK NT=$NT is not a legal " *
                            "workgroup-scope shape on this device; it reports " *
                            "(invocations, M, N, K) granularities $(dev.wggran)"))
    # Clamp by default only where a tile can hang off an edge, as the reference
    # does; `clamp = true` forces it.
    cl = something(clamp, !(M % BM == 0 && N % BN == 0 && K % BK == 0))
    cmode = cl ? TENSOR_CLAMP_CONSTANT : TENSOR_CLAMP_UNDEFINED
    KI.Kernel(backend, gemm_cm2!)(C, A, B, Int32(M), Int32(N), Int32(K),
                                  Val(BM), Val(BN), Val(BK), Val(unroll), Val(cmode);
                                  numgroups = (cld(M, BM), cld(N, BN)),
                                  workgroupsize = NT)
    C
end
