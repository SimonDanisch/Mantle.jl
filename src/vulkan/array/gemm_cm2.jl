# A GEMM with no shared memory: workgroup-scope cooperative matrices and
# tensor-addressed loads — `mul_mm_cm2.comp`'s structure.
#
# `gemm.jl`'s staged kernel is what ships. This is the coopmat2 form, and it is
# **not routed to**: it loses by 22-30% at its best tiling on every shape
# (`tools/gemm_cm2_ab.jl`, 2026-08-11), and — the part that took the work —
# WHY is measured rather than argued (`tools/gemm_cm2_why.jl`).
#
# The first version differed from `mul_mm_cm2.comp` in three ways. Ablating them
# one at a time, with the driver's own register count beside each
# (2304x4096x576; staged = 37.3 TFLOP/s at 128 registers):
#
#     variant                       TFLOP/s   regs
#     64x64/32  clamp,roll             25.0     60   <- best coopmat2 config
#     64x64/32  clamp,unroll           25.4     62
#     64x64/32  UNCLAMPED               8.9     74   <- the reference's own fast path
#     128x128/32                       14.8    196
#     256x128/32                       20.7    255
#     64x64/64 @128, no unroll         13.8    144
#     128x128/64 @256 (m_warptile)     14.8    255
#     128x256/64 @256 (l_warptile)     12.6    255
#     subgroup 4x4 block, 2 subgroups  18.7    230   <- `gemm_cm2_sg!`
#     subgroup 4x4 block, 4 subgroups  18.7    230
#
# **One of the three was a real porting error.** The first unroll issued all four
# A-fragments and all four B-fragments before any product, holding eight live at
# once; the reference consumes each pair inside the unrolled body. Fixing it was
# worth +65% at `128x128/64` and +118% on the K-heavy shape — and still lost.
#
# **One was not an error at all, and inverts the reference.** `mul_mm_cm2` keeps
# a second, UNCLAMPED layout and a whole separate pipeline to use it whenever the
# tile is in bounds. On this driver that is **2.8x SLOWER** — 25.4 against 8.9 —
# and it is not register pressure (74 against 62). Clamped loads are the fast
# path here. Do not "optimise" this kernel by removing the clamp.
#
# **The third is the answer, and it is about the driver, not the loop.** Every
# route to more work per lane — a bigger tile, a deeper `BK`, either invocation
# count — makes the allocator want two to four times the registers the matrices
# themselves need, and it saturates at 255. `128x128/32` needs ~90 by hand and
# gets 196. `64x64/64` without any unrolling gets 144. The staged kernel sits at
# exactly 128, which is the count the driver targets for two resident workgroups,
# because its accumulators are per-subgroup `16x16` tiles it can place one at a
# time. A workgroup-scope matrix is placed as a unit, so the one configuration
# that stays lean (62 registers) is also the one with the least reuse — and 25-30
# against 36-40 TFLOP/s is what that costs.
#
# **And the obvious escape was tried and is worse.** `gemm_cm2_sg!` below is the
# other combination — tensor-addressed loads into SUBGROUP-scope `16x16` tiles,
# keeping the staged kernel's 4x4 register block and dropping only the shared
# memory. Same sixteen accumulators the staged kernel holds in 128 registers; it
# gets **230, and 18.7 TFLOP/s**. So the cost is not the scope of the matrix and
# not the blocking: it is the tensor-addressed load itself, which is worth 1.5x
# to 4x the registers of the equivalent staged code in every structure tried.
#
# Four kernel structures, one answer. Shared-memory staging is not overhead this
# device wants removed — it is how the addressing state stays out of the register
# file. That is a fact about this driver rather than about the algorithm, so it is
# worth re-testing on a newer one; the harness is `tools/gemm_cm2_why.jl` and it
# prints the register count next to every number.
#
# **No transposing views, and that is not a simplification of the reference.**
# `mul_mm_cm2` loads B through one because ggml's arrays are row-major. Ours are
# column-major, so a Julia `(r, c)` array IS the tensor `(c, r)`, and the natural
# product is the transposed one:
#
#     C = A*B  with Julia (M,K) x (K,N)   is   C̃ = B̃·Ã  in tensor terms
#
# B therefore supplies the A-operand and A the B-operand, every load is plain,
# and so is the store. The mapping is the one `mwe_tensor_gemm_nonsquare.jl`
# pinned; this is the first kernel to lean on it for both operands at once.
#
# What it buys beyond speed: the clamping layout pads every load with zeros,
# which contribute nothing to a sum, so this kernel takes extents that divide
# NOTHING — where the staged kernel needs `M`, `N` and `K` on the tile and hands
# everything else to a slower path. Verified at `100 x 130 x 70`.

"""
    gemm_cm2_fits(dev, BM, BN, BK, NT) -> Bool

Whether a `(BM, BN, BK, NT)` tiling is one this DEVICE can run: `NT` has to be a
workgroup size it reports workgroup-scope matrices for, and every extent a
multiple of that size's granularity.

Every extent appears in some product and is checked against that product's axis —
`BN` as the M of the accumulator and of the A-operand (the kernel computes the
transposed product, so `BN` leads), `BM` as their N, `BK` as the K of both.
Without this a mistyped tiling reaches `vkCreateComputePipelines` and fails there,
which names neither the tiling nor the rule it broke.
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

**Derived from the device's own table, not hardcoded.** `64x64/32 @256` is what
wins here (see the ablation at the top of this file), but the numbers that make
it legal are the granularities the device reports for that workgroup size — on
another card the same literals may be illegal, and a tiling nothing can run is
the one thing a chooser must not return. So it asks `gemm_cm2_fits` for the
measured preference first and walks the reported workgroup sizes if that does not
apply, largest first.
"""
function gemm_cm2_tiling(dev = caps())
    isempty(dev.wggran) && return nothing
    # The measured winner, if this device admits it.
    gemm_cm2_fits(dev, 64, 64, 32, 256) && return (64, 64, 32, 256)
    for (nt, mg, ng, kg) in Iterators.reverse(dev.wggran)
        # The smallest legal square tile at this size, which is the shape the
        # ablation found stays inside the register file.
        BM, BN, BK = max(64, ng), max(64, mg), max(32, kg)
        gemm_cm2_fits(dev, BM, BN, BK, nt) && return (BM, BN, BK, nt)
    end
    nothing
end

"""
    gemm_cm2!(C, A, B, M, N, K, ::Val{BM}, ::Val{BN}, ::Val{BK})

`C = A*B` for column-major `A (M,K)`, `B (K,N)`, `C (M,N)`, with `C` either
`Float16` or `Float32` and both operands `Float16`.

One workgroup owns a `BM x BN` tile of `C` and walks `K` in steps of `BK`. There
is no shared memory, no staging loop and no barrier: each step is two tensor
loads and one `coopmat_muladd` over matrices that span all `NT` invocations.
"""
@kernel cpu=false unsafe_indices=true function gemm_cm2!(
        C, @Const(A), @Const(B), M::Int32, N::Int32, K::Int32,
        ::Val{BM}, ::Val{BN}, ::Val{BK}, ::Val{UNROLL} = Val(false),
        ::Val{CM} = Val(TENSOR_CLAMP_CONSTANT)) where {BM,BN,BK,UNROLL,CM}
    WM = WorkgroupMatrix
    grp = @index(Group, NTuple)
    m0 = Int32((grp[1] - 1) * BM)
    n0 = Int32((grp[2] - 1) * BN)

    # A 2-D layout over a column-major Julia `(r, c)` array, whose tensor is
    # `(c, r)` with strides `(r, 1)`.
    #
    # The clamp mode is the type parameter itself, NOT a `Bool` turned into one
    # here: `Val(x)` for an `x` computed from a local is not a compile-time
    # constant, and the kernel fails to compile with an `InvalidIRError` three
    # layers down. Same family as the `@localmem` whose size comes from a local.
    #
    # The reference keeps two layouts for exactly this and picks the unclamped
    # one whenever the whole tile is in bounds — a bounds check on every element
    # of every load is not free, and it is one of the three things this port did
    # differently from `mul_mm_cm2.comp`.
    # Dimension and stride are loop-INVARIANT; only the slice moves with `k`.
    # Building the whole layout inside each load — which is what this did — puts
    # `tensor_layout`/`setdim`/`setstride` in the loop body, and with `UNROLL`
    # that is EIGHT full constructions per iteration against the reference's two
    # hoisted layouts and eight slices. `mul_mm_cm2.comp` creates `tensorLayoutA`
    # and `tensorLayoutB` before the loop and only calls `sliceTensorLayoutNV`
    # inside, and matching that is worth 177 -> 128 registers at `128x128/32`,
    # which is the difference between one resident workgroup and two.
    base(r, c) = tensor_setstride(
        tensor_setdim(tensor_layout(Val(2), Val(CM)), (c, r)), (r, Int32(1)))
    @inline slice(l, o0, o1, n0, n1) = tensor_slice(l, (o0, o1), (Int32(n0), Int32(n1)))

    baseA = base(K, N)
    baseB = base(M, K)

    # `coopmat_undef`, not `coopmat_zero`: under a CONSTANT clamp mode the
    # out-of-range elements come from the layout's clamp value, so the
    # destination is never read and a zeroed fragment is a value the allocator
    # holds until the load overwrites it. GLSL declares `coopmat mat_a;` and
    # assigns nothing, which is what this matches. With `TENSOR_CLAMP_UNDEFINED`
    # the destination DOES survive out of range, so the zero has to stay.
    @inline dest(::Type{MT}) where {MT} =
        CM === TENSOR_CLAMP_CONSTANT ? coopmat_undef(MT) : coopmat_zero(MT)
    @inline loadA(k) = tensor_load(dest(WM{Float16,BN,BK,MatrixA}),
                                   UInt64(pointer(B)), slice(baseA, n0, k, BN, BK))
    @inline loadB(k) = tensor_load(dest(WM{Float16,BK,BM,MatrixB}),
                                   UInt64(pointer(A)), slice(baseB, k, m0, BK, BM))

    acc = coopmat_zero(WM{Float32,BN,BM,Accumulator})
    k0 = Int32(0)
    # Four k-steps per iteration, each pair consumed by its own product —
    # `[[unroll]] for (j) { load; load; mulAdd; }`, which is the reference's
    # shape.
    #
    # **Not all four loads first.** That was the first version, and it holds
    # eight fragments live at once: at `128x128/64` the driver came back with
    # **255 registers**, its ceiling, and the kernel ran at 8.9 TFLOP/s against
    # the staged kernel's 37.8 at 128 registers. Batching the loads looks like
    # it hides latency and instead prices the tile out of the register file —
    # the scheduler can overlap the loads by itself, but it cannot un-extend a
    # live range the source insists on.
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
    # The destination decides the component type — a tensor store writes the
    # matrix's own bytes, so an fp32 accumulator into an fp16 slot is corruption
    # and not a conversion. `eltype` is a compile-time property, so only one
    # store survives. (This is the bug that took SAM 2's encoder to NaN when the
    # flash kernel assumed its destination; it is not repeated here.)
    olay = slice(base(M, N), n0, m0, BN, BM)
    if eltype(C) === Float16
        tensor_store(coopmat_convert(WM{Float16,BN,BM,Accumulator}, acc),
                     UInt64(pointer(C)), olay)
    else
        tensor_store(acc, UInt64(pointer(C)), olay)
    end
end

"""
    gemm_cm2_sg!(C, A, B, M, N, K, ::Val{NW})

The other combination: tensor-addressed loads into **subgroup-scope** tiles,
keeping the staged kernel's register blocking and dropping only the shared
memory.

This is the one hypothesis the ablation above leaves open. What it measured is
that a *workgroup-scope* matrix is placed as a unit and the allocator wants two
to four times the registers its elements need, saturating at 255 — while the
staged kernel holds sixteen `16x16` subgroup accumulators in exactly 128. So the
question is whether the tensor loads are worth anything when the tiles they feed
are the ones the driver is demonstrably good at placing.

Each subgroup owns a `4x4` block of `16x16` accumulators — 128 fp32 a lane, the
same as the staged kernel — and loads four A-fragments and four B-fragments per
k-step to feed sixteen products. That ratio is the whole point of register
blocking, and it is unchanged; what changes is that the fragments come from
global memory through a tensor layout instead of from `@localmem` after a staging
pass and two barriers.

**What it gives up** is the sharing that shared memory buys: with `NW` subgroups
each reading its own fragments, an A-block is fetched `NW` times instead of once.
The bet is that L2 covers it, which is exactly the bet a measurement settles.

Clamped, unconditionally, because that is the fast path on this driver — see
the ablation above, where the unclamped layout cost 2.8x.
"""
@kernel cpu=false unsafe_indices=true function gemm_cm2_sg!(
        C, @Const(A), @Const(B), M::Int32, N::Int32, K::Int32,
        ::Val{NW}) where {NW}
    T = GEMM_TILE                       # 16
    RB = 4                              # register block, a literal for `@nexprs`
    AM = AcceleratedMatrix
    tid = @index(Local, Linear) - 1
    w = Int32(tid ÷ 32)                 # this invocation's subgroup
    grp = @index(Group, NTuple)
    # The workgroup covers `RB*T` of M and `NW*RB*T` of N; subgroup `w` takes the
    # w-th slice of the N axis.
    m0 = Int32((grp[1] - 1) * (RB * T))
    n0 = Int32((grp[2] - 1) * (NW * RB * T)) + w * Int32(RB * T)

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
end

"""
    coopmat_gemm_cm2_sg!(C, A, B, M, N, K; nw = 2) -> C

Launch [`gemm_cm2_sg!`](@ref). `nw` subgroups per workgroup, so the workgroup is
`nw * 32` invocations and covers `64 x (nw * 64)` of `C`.
"""
function coopmat_gemm_cm2_sg!(C, A, B, M::Int, N::Int, K::Int; nw::Int = 2)
    backend = KernelAbstractions.get_backend(C)
    dev = caps(backend)
    # `wggran` is non-empty exactly when this device has coopmat2 — which is what
    # supplies the tensor addressing this kernel needs, even though its matrices
    # are subgroup-scope.
    isempty(dev.wggran) && return nothing
    # The tiles are `GEMM_TILE` squares at subgroup scope, so the device's own
    # KHR shape list is the authority rather than the workgroup-scope table.
    coopmat_shape(vk_context(), Float16, GEMM_TILE, GEMM_TILE, GEMM_TILE) ||
        return nothing
    nw * 32 <= dev.workgrouplimit ||
        throw(ArgumentError("nw=$nw wants $(nw * 32) invocations, past this " *
                            "device's limit of $(dev.workgrouplimit)"))
    span = 4 * GEMM_TILE
    gemm_cm2_sg!(backend, nw * 32)(C, A, B, Int32(M), Int32(N), Int32(K), Val(nw);
                                   ndrange = (nw * 32 * cld(M, span),
                                              cld(N, nw * span)))
    C
end

"""
    coopmat_gemm_cm2!(C, A, B, M, N, K) -> C

Launch [`gemm_cm2!`](@ref) for dense column-major operands. Returns `nothing`
instead of `C` when this device cannot run it, so a caller can fall back.

Deliberately narrow while it is unrouted: no batching, no split-K, no bias and no
epilogue. Each of those is a real feature of `coopmat_gemm!` and each would be
added only after this kernel earns its place on the plain shape.
"""
function coopmat_gemm_cm2!(C, A, B, M::Int, N::Int, K::Int; tiling = nothing,
                           unroll::Bool = true, clamp::Union{Nothing,Bool} = nothing)
    # `KA.get_backend(C)`, not `LavaBackend()`: an unpinned backend resolves its
    # queue through the global context, so on a second device the work lands on
    # the wrong GPU. Same reason `coopmat_gemm!` derives it from the array.
    backend = KernelAbstractions.get_backend(C)
    dev = caps(backend)
    tiling = something(tiling, gemm_cm2_tiling(dev))
    tiling === nothing && return nothing
    BM, BN, BK, NT = tiling
    # A tiling named by the caller is still the device's to allow. Refusing here
    # says which rule broke; letting it through says `vkCreateComputePipelines`
    # failed, three layers from the call that chose it.
    gemm_cm2_fits(dev, BM, BN, BK, NT) ||
        throw(ArgumentError("tiling BM=$BM BN=$BN BK=$BK NT=$NT is not a legal " *
                            "workgroup-scope shape on this device; it reports " *
                            "(invocations, M, N, K) granularities $(dev.wggran)"))
    # Clamping is only needed where a tile can hang off an edge. Deciding it from
    # the shape rather than always paying for it is what the reference's "fast
    # path" is; `clamp = true` forces it, which is what the A/B measures against.
    cl = something(clamp, !(M % BM == 0 && N % BN == 0 && K % BK == 0))
    cmode = cl ? TENSOR_CLAMP_CONSTANT : TENSOR_CLAMP_UNDEFINED
    gemm_cm2!(backend, NT)(C, A, B, Int32(M), Int32(N), Int32(K),
                           Val(BM), Val(BN), Val(BK), Val(unroll), Val(cmode);
                           ndrange = (NT * cld(M, BM), cld(N, BN)))
    C
end
