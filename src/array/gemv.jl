# Portable GEMV, one kernel per contiguous axis. K-contiguous (derived from
# llama.cpp's mul_mat_vec.comp / mul_mat_vec_base.glsl, MIT): a vector slice is
# reused across NROWS columns, reduced within subgroups, then across them in
# shared memory. N-contiguous: K is split across row groups and combined by a
# shared-memory tree; bias and epilogue fuse into the store.

struct KContiguousGEMV{NROWS,BLOCK,SUB} <: Function end

"""
    gemv_kcontig_kernel(NROWS, BLOCK, SUB) -> kernel

`C[n] = sum_k A[k] * B[k, n]`, one workgroup per `NROWS` output columns. Lanes
split contiguous K reads and share each vector load across the NROWS columns.
`SUB` is the device's subgroup width; a partial last subgroup counts in
`cld(BLOCK, SUB)`.
"""
function gemv_kcontig_kernel(NROWS::Int, BLOCK::Int, SUB::Int)
    0 < NROWS <= BLOCK && SUB > 0 || throw(ArgumentError(
        "gemv: require 0 < NROWS <= BLOCK and SUB > 0"))
    return KContiguousGEMV{NROWS,BLOCK,SUB}()
end

@generated function (kernel::KContiguousGEMV{NROWS,BLOCK,SUB})(
        C,A,B,K::Int,N::Int) where {NROWS,BLOCK,SUB}
    nsub = cld(BLOCK,SUB)
    shid = Symbol(:mantle_gemvk_,NROWS,:_,BLOCK,:_,SUB)
    return quote
        # one partial per (subgroup, output column)
        parts = KI.localmemory(Float32, Val(($(NROWS * nsub),)), Val($(QuoteNode(shid))))

        tid = KI.get_local_id().x - 1
        grp = KI.get_group_id().x - 1
        n0 = grp * $NROWS
        # The device's subgroup width, not 32: `sub_group_reduce_add` reduces
        # over it, and any other partition makes several lanes of one subgroup
        # each write the whole sum as its leader.
        lane = tid % $SUB
        sub = tid ÷ $SUB

        Base.Cartesian.@nexprs $NROWS r -> acc_r = 0.0f0

        # Walk K in steps of 4*BLOCK; the A slice is read once per step and
        # reused for all NROWS columns.
        k = tid * 4
        @inbounds while k < K
            a_1 = A[k + 1]
            a_2 = k + 1 < K ? A[k + 2] : 0.0f0
            a_3 = k + 2 < K ? A[k + 3] : 0.0f0
            a_4 = k + 3 < K ? A[k + 4] : 0.0f0
            Base.Cartesian.@nexprs $NROWS r -> begin
                n_r = n0 + r
                if n_r <= N
                    s_r = a_1 * B[k + 1, n_r]
                    if k + 1 < K
                        s_r += a_2 * B[k + 2, n_r]
                    end
                    if k + 2 < K
                        s_r += a_3 * B[k + 3, n_r]
                    end
                    if k + 3 < K
                        s_r += a_4 * B[k + 4, n_r]
                    end
                    acc_r += s_r
                end
            end
            k += $(4 * BLOCK)
        end

        # subgroup first, then across subgroups through shared memory
        Base.Cartesian.@nexprs $NROWS r -> red_r = sub_group_reduce_add(acc_r)
        @inbounds if lane == 0
            Base.Cartesian.@nexprs $NROWS r -> begin
                parts[sub * $NROWS + r] = red_r
            end
        end
        KI.barrier()
        @inbounds if tid < $NROWS
            n = n0 + tid + 1
            if n <= N
                t = 0.0f0
                for s in 0:$(nsub - 1)
                    t += parts[s * $NROWS + tid + 1]
                end
                C[n] = t
            end
        end
        return nothing
    end
end

struct NContiguousGEMV{TM,BLOCK,UNROLL} <: Function end

"""
    gemv_ncontig_kernel(TM, BLOCK, UNROLL) -> kernel

C[m] = epilogue(bias[m] + sum_k W[m,k]*x[k]), with W contiguous along m.
TM rows share BLOCK/TM reduction groups, a power of two. Positive UNROLL uses
independent scalar accumulators; a uniform shared tree combines the groups.
"""
function gemv_ncontig_kernel(TM::Int, BLOCK::Int, UNROLL::Int)
    TM > 0 && BLOCK > 0 && UNROLL > 0 || throw(ArgumentError(
        "gemv: TM, BLOCK and UNROLL must be positive"))
    BLOCK % TM == 0 || throw(ArgumentError("gemv: BLOCK $BLOCK not a multiple of TM $TM"))
    ispow2(BLOCK÷TM) || throw(ArgumentError(
        "gemv: BLOCK/TM = $(BLOCK÷TM) must be a power of two for the reduction tree"))
    return NContiguousGEMV{TM,BLOCK,UNROLL}()
end

@generated function (kernel::NContiguousGEMV{TM,BLOCK,UNROLL})(
        C,W,x,bias,epi,M::Int,K::Int) where {TM,BLOCK,UNROLL}
    ng = BLOCK÷TM
    shid = Symbol(:mantle_gemvn_,TM,:_,BLOCK,:_,UNROLL)
    # Literal accumulator names avoid captured tuple state in the reduction loop.
    accsum = Expr(:call, :+, (Symbol("acc_", u) for u in 1:UNROLL)...)
    # Unroll the tree so every barrier remains outside the lane guards.
    tree = Expr(:block)
    let s = ng ÷ 2
        while s > 0
            push!(tree.args, quote
                if kg < $s
                    @inbounds parts[kg * $TM + tm + 1] +=
                        parts[(kg + $s) * $TM + tm + 1]
                end
                KI.barrier()
            end)
            s ÷= 2
        end
    end
    return quote
        parts = KI.localmemory(Float32, Val(($BLOCK,)), Val($(QuoteNode(shid))))

        tid = KI.get_local_id().x - 1
        grp = KI.get_group_id().x - 1
        tm = tid % $TM
        kg = tid ÷ $TM
        m = grp * $TM + tm + 1

        Base.Cartesian.@nexprs $UNROLL u -> acc_u = 0.0f0
        if m <= M
            k = kg + 1
            @inbounds while k + $(ng * (UNROLL - 1)) <= K
                Base.Cartesian.@nexprs $UNROLL u -> begin
                    kk_u = k + $ng * (u - 1)
                    acc_u += W[m + M * (kk_u - 1)] * x[kk_u]
                end
                k += $(ng * UNROLL)
            end
            @inbounds while k <= K
                acc_1 += W[m + M * (k - 1)] * x[k]
                k += $ng
            end
        end
        @inbounds parts[kg * $TM + tm + 1] = $accsum
        KI.barrier()
        $tree
        @inbounds if kg == 0 && m <= M
            t = parts[tm + 1]
            bias === nothing || (t += Float32(bias[m]))
            C[m] = epi(t)
        end
        return nothing
    end
end

"""
    gemv_ncontig_config(M, K, limit) -> (TM, BLOCK, UNROLL)

TM = 32, block <= 1024 and unroll = 4 (the measured configuration), rounded down
to a power-of-two reduction that fits `limit`.
"""
function gemv_ncontig_config(M::Int, K::Int, limit::Int)
    limit > 0 || throw(ArgumentError("gemv: workgroup limit must be positive"))
    tm = min(32,prevpow(2,limit))
    block = tm*prevpow(2,min(limit,1024)÷tm)
    return (tm, block, 4)
end

"""
    gemv_config(K, N, limit) -> (NROWS, BLOCK)

Rows and threads per workgroup, balancing vector reuse against occupancy; rows
never exceed the block. Starting points, not tuned per device.
"""
function gemv_config(K::Int, N::Int, limit::Int)
    limit > 0 || throw(ArgumentError("gemv: workgroup limit must be positive"))
    block = min(limit, K >= 4096 ? 256 : 128)
    nrows = min(N >= 4096 ? 8 : 4,block)
    return (nrows, block)
end

"""
    gemv!(C, A, B; nrows = nothing, block = nothing) -> C

`C = A * B` for `A` a length-`K` vector and `B` a `K x N` matrix, `C` length `N`.
`nrows`/`block` override the selection for tuning.
"""
function gemv!(C::AbstractGPUArray{Float32}, A::AbstractGPUArray{Float32}, B::AbstractGPUArray{Float32,2};
               nrows::Union{Nothing,Int} = nothing, block::Union{Nothing,Int} = nothing)
    K, N = size(B, 1), size(B, 2)
    length(A) == K || throw(DimensionMismatch(
        "gemv!: A has $(length(A)) elements, expected K = $K"))
    length(C) == N || throw(DimensionMismatch(
        "gemv!: C has $(length(C)) elements, expected N = $N"))
    N == 0 && return C
    backend = get_backend(B)
    limit = workgrouplimit(B)
    pn, pb = gemv_config(K, N, limit)
    nr = nrows === nothing ? pn : nrows
    bl = block === nothing ? pb : block
    bl <= limit || throw(ArgumentError("gemv: block $bl exceeds this device's limit of $limit"))
    kern = gemv_kcontig_kernel(nr, bl, subgroupwidth(B))
    kilaunch!(backend, kern, C, A, B, K, N;
                      ndrange = cld(N, nr) * bl, workgroupsize = bl)
    return C
end

"""
    gemv!(C, x, Wt::Transpose; bias, epilogue, tm, block, unroll) -> C

`C[m] = epilogue(bias[m] + sum_k W[m, k] * x[k])`, for `Wt = transpose(W)` with
`W` stored `(M, K)`; the `Transpose` selects this layout without copying. `C` and
`x` may have any shape of the right length.
"""
function gemv!(C, x, Wt::Transpose{Float32,<:AbstractGPUArray{Float32,2}}; kw...)
    W = parent(Wt)
    runlaunches!(get_backend(W),
                 gemv_ncontig_launches(C, x, W, size(W)..., workgrouplimit(W); kw...))
    return C
end

"""
    gemv_ncontig_launches(C, x, W, M, K, limit; bias, epilogue, tm, block, unroll)

The launch for eager submission or graph declaration, after validating lengths.
`W` is the dense (M, K) storage without the `Transpose`; explicit extents and
`limit` allow resources that are not placed yet.
"""
function gemv_ncontig_launches(C, x, W, M::Int, K::Int, limit::Int;
                               bias = nothing, epilogue = identity,
                               tm::Union{Nothing,Int} = nothing,
                               block::Union{Nothing,Int} = nothing,
                               unroll::Union{Nothing,Int} = nothing)
    length(x) == K || throw(DimensionMismatch(
        "gemv: x has $(length(x)) elements, expected K = $K"))
    length(C) == M || throw(DimensionMismatch(
        "gemv: C has $(length(C)) elements, expected M = $M"))
    length(W) == M*K || throw(DimensionMismatch(
        "gemv: W has $(length(W)) elements, expected M*K = $(M*K)"))
    bias === nothing || length(bias) == M || throw(DimensionMismatch(
        "gemv: bias has $(length(bias)) elements, expected M = $M"))
    M == 0 && return ()
    ptm, pbl, pun = gemv_ncontig_config(M, K, limit)
    t = tm === nothing ? ptm : tm
    bl = block === nothing ? pbl : block
    bl <= limit || throw(ArgumentError("gemv: block $bl exceeds this device's limit of $limit"))
    un = unroll === nothing ? pun : unroll
    return (ArrayLaunch(gemv_ncontig_kernel(t, bl, un),
                        (C, W, x, bias, epilogue, M, K), cld(M, t) * bl, bl),)
end

"""
    gemv_dispatch!(g, C, x, W, M, K; name, …) -> C

Declare the (M,K)-layout product with bias and epilogue fused into its store.
"""
function gemv_dispatch!(g, C, x, W, M::Int, K::Int;
                        name::AbstractString = "gemv", kw...)
    dispatchlaunches!(g, gemv_ncontig_launches(C, x, W, M, K,
                          caps(g.dev).workgrouplimit; kw...); name)
    return C
end

"""
    gemv(A, B) -> AbstractGPUArray

Allocating [`gemv!`](@ref).
"""
gemv(A::AbstractGPUArray{Float32}, B::AbstractGPUArray{Float32,2}) =
    gemv!(similar(A, Float32, size(B, 2)), A, B)

gemv(x::AbstractGPUArray{Float32}, Wt::Transpose{Float32,<:AbstractGPUArray{Float32,2}}; kw...) =
    gemv!(similar(x, Float32, size(Wt, 1)), x, Wt; kw...)
