# GEMV, and it is not the Vulkan backend's.
#
# This file lived in `src/vulkan/array/` and named nothing Vulkan has: the
# kernels are `KernelAbstractions`, the tilings are arithmetic, and the two
# device facts it needs — the workgroup limit for `gemv_config`, and which
# backend to launch on — come from `workgrouplimit` and
# `KernelAbstractions.get_backend`. Both take the array, not a `VkContext`.
#
# So it moved to core, where a second backend inherits it rather than
# reimplementing it. That is step 5 of the split, for the part of step 5 that
# does not need a second backend to exist yet.
#
# The dispatch widened with it, from `::LavaArray` to `::AbstractGPUArray`, and
# that is safe **here** in a way it is not everywhere: `gemv`/`gemv!` are
# Mantle's own functions, so widening them claims nothing from anybody.
# `gemm.jl`'s entry point is `LinearAlgebra.mul!`, which is Base's — widening
# THAT would have Mantle answering for every GPU array package in the session,
# which is why its kernels move and its `mul!` does not.

# Matrix-vector multiply. Two kernels, because there are two layouts.
#
# ## The layout decides the kernel, and it is not a detail
#
# A GEMV reads the matrix exactly once, so the only thing that matters is that
# the read coalesces and that there are enough threads in flight to cover the
# latency. Which decomposition achieves that is the *opposite* in the two
# layouts, so this file has two kernels and `gemv!` dispatches on the type:
#
#   * **`gemv_kcontig_kernel`** — matrix `(K, N)`, contiguous along the
#     reduction axis. Ported from llama.cpp's Vulkan backend
#     (`dev/llama.cpp/ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vec.comp`,
#     MIT). Threads split `K`, the read coalesces along `K`, and the partial sums
#     come back through `sub_group_reduce_add` plus a shared-memory tail.
#
#   * **`gemv_ncontig_kernel`** — matrix `(M, K)`, contiguous along the *output*
#     axis, reached as `gemv!(C, x, transpose(W))`. Here consecutive **outputs**
#     are consecutive addresses, so the coalesced decomposition is one thread per
#     output row and the reduction over `K` happens *inside* a thread. A
#     workgroup still splits `K` across `BLOCK / TM` groups, purely to have
#     enough threads resident — with one thread per row, `M = 1280` is 1280
#     threads, which is 5 workgroups on a 48-core device.
#
# **DNNKernels' graphs produce the second layout, not the first.** `hoistpermutes`
# materialises every weight as torch `(K, N)` — Julia `(N, K)` — because that is
# what `coopmat_gemm!` wants. So the ported llama.cpp kernel, on its own, does not
# apply to the model it was ported for: pointing it at these weights would need a
# 634 MB transpose per model. That is worth stating plainly, because "we ported
# the SOTA kernel" and "the SOTA kernel fits our data" are different claims and
# only the first was true here.
#
# ## What this kernel does NOT fix
#
# Whisper's decoder step went from 13.58 ms to 13.50 ms when these kernels
# replaced `mul!` in it — no change — even though an interleaved same-session A/B
# has them 1.9-3.0x faster on the shapes that step runs. The reason is that the
# step is not GPU-bound:
#
#     a (64, 64) GEMV — 16 KiB — costs   0.0553 ms
#     a (1280, 1280) GEMV — 6.4 MiB —    0.0555 ms
#
# Flat, four hundredfold apart in size. Timing the dispatch loop without waiting
# for the device puts ~89% of that on the host, and a broadcast and a `mul!`
# measure the same, so it is not this code path: **Lava costs ~63 us of host time
# per dispatch**, and the decoder's 96 ops are a 6.1 ms floor under an 11.2 ms
# step. Kernel tuning cannot reach it; `capture` / `replay!` can, which is
# what it already did for SAM 2's decode.
#
# Recorded here because these kernels look like the fix for the decoder and are
# not — they are the fix for the *arithmetic*, which was the smaller half.
#
# ## Why this is not `mul!`
#
# `coopmat_gemm!` needs `M >= 16` — a cooperative matrix *is* a 16x16 tile — so
# at `M = 1` fifteen sixteenths of every tile is padding. Measured on Whisper's
# decoder shapes, fp32, against the 307 GB/s DRAM roofline this card sustains:
#
#     shape                      mul! ms   GB/s    % of roof
#     (1,1280)@(1280,1280)        0.6146   10.7      3.5%
#     (1,1280)@(1280,5120)        0.5263   49.8     16.2%
#     (1,5120)@(5120,1280)        0.2473  106.0     34.5%
#     (1,1280)@(1280,51866)       2.5908  102.5     33.4%
#
# One Whisper decoder token is ~20.4 ms of that against a 2.07 ms bandwidth
# floor. llama.cpp ships `mul_mat_vec.comp` separately from `mul_mm.comp` for
# exactly this reason: batch-1 decode is a different kernel, not a tuning of the
# same one.
#
# ## The structure that was ported
#
# **`NROWS` output columns per workgroup.** This is the whole trick and it is not
# obvious: a GEMV reads `K*N` matrix elements and only `K` vector elements, so it
# is hopelessly bandwidth-bound *unless* the vector read is amortised. Handling
# several output columns per workgroup reads the vector slice once into registers
# and reuses it `NROWS` times. llama.cpp's `NUM_ROWS`.
#
# **Four elements of K per thread per step** (`K_PER_ITER = 4` there for the
# non-quantised types). Wide enough for the loads to coalesce, narrow enough that
# `NROWS` accumulators still fit in registers.
#
# **Reduce with `sub_group_reduce_add`, then across subgroups through shared memory** —
# `reduce_result` in `mul_mat_vec_base.glsl`. The comment there is worth keeping:
# the subgroup path wins *particularly when the workgroup has more than one
# subgroup*, because it collapses a log2(BLOCK) shared-memory tree into one
# instruction plus a log2(nsubgroups) tail.
#
# Not ported: the quantised paths (`dequantize4`, `QUANT_K`/`QUANT_R`) — we have
# no quantised weights — and the 4/2/1 manual unroll ladder, which exists to
# handle `ncols` not dividing the block. Ours takes a bounds check on the tail;
# whether the ladder pays here is a measurement nobody has made.
#
# Constants are NOT llama.cpp's. Per `feedback-port-sota-kernels` the structure
# ports and every number is re-measured — see `gemv_config`.
#
# ## Why this is `@eval`-generated
#
# The first version wrote the accumulators as an `NTuple` rebound inside the
# `K` loop. That does not compile: the tuple is a mutable local captured by the
# `ntuple` closure, so it boxes, and the failure arrives as thirty lines of
# `unsupported dynamic function invocation` naming `ntuple`, `to_indices` and
# `_typeof_captured_variable` — nowhere near the cause. The same trap the
# mixed-radix FFT hit.
#
# With `NROWS`/`BLOCK` fixed at generation time the accumulators are plain
# scalars from `Base.Cartesian.@nexprs`, every index is a literal, and there is
# no closure anywhere.

"Generated K-contiguous GEMV kernels, keyed by (NROWS, BLOCK). Compiled once each."
const GEMV_KERNELS = Dict{Tuple{Int,Int},Any}()

"""
    gemv_kcontig_kernel(NROWS, BLOCK) -> kernel

`C[n] = sum_k A[k] * B[k, n]`, one workgroup per `NROWS` output columns.

Column-major is what makes this coalesce: `B[:, n]` is contiguous in `k`, so
lanes reading `k, k+1, ...` for one `n` touch one cache line. llama.cpp's matrix
is row-major and its "rows" are our columns; the memory pattern is the same.

The name is deterministic, not a `gensym` — `frozen_key` hashes
`string(nameof(F))`, so a per-session counter would miss the frozen cache on
every load.
"""
function gemv_kcontig_kernel(NROWS::Int, BLOCK::Int)
    get!(GEMV_KERNELS, (NROWS, BLOCK)) do
        nsub = BLOCK ÷ 32
        kname = Symbol("gemvk_", NROWS, "_", BLOCK)
        @eval begin
            @kernel cpu=false unsafe_indices=true function $kname(
                    C, @Const(A), @Const(B), K::Int, N::Int)
                # one partial per (subgroup, output column)
                parts = @localmem Float32 ($(NROWS * nsub),)

                tid = @index(Local, Linear) - 1
                grp = @index(Group, Linear) - 1
                n0 = grp * $NROWS
                lane = tid % 32
                sub = tid ÷ 32

                Base.Cartesian.@nexprs $NROWS r -> acc_r = 0.0f0

                # Walk K in steps of 4*BLOCK. The A slice is read ONCE per step
                # and reused for all NROWS columns — the whole point.
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
                @synchronize
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
            end
            $kname
        end
    end
end

"Generated N-contiguous GEMV kernels, keyed by (TM, BLOCK, UNROLL)."
const GEMVN_KERNELS = Dict{Tuple{Int,Int,Int},Any}()

"""
    gemv_ncontig_kernel(TM, BLOCK, UNROLL) -> kernel

`C[m] = epilogue(bias[m] + sum_k W[m, k] * x[k])` for `W` stored `(M, K)`, i.e.
contiguous along `m`.

**One thread per output row.** `W[m, k]` for consecutive `m` and fixed `k` are
consecutive addresses, so a warp covering `TM = 32` rows reads one full 128-byte
transaction per step. That is the whole reason this kernel is shaped the way it
is, and it is why the reduction is *not* a subgroup reduction: the values a
subgroup holds belong to 32 different outputs, and adding them would be adding
unrelated numbers.

**`BLOCK / TM` threads per row, to have any threads at all.** One thread per
output on a `(1280, 1280)` weight is 1280 threads — 5 workgroups on a device with
48 shader cores, so the kernel would be latency-bound at a few percent of
bandwidth no matter how well it coalesced. That is exactly the failure `mul!`
shows on this shape (3.5% of roofline). Splitting `K` across `BLOCK / TM` groups
and reducing through shared memory brings `M = 1280` to 40 workgroups.

`UNROLL` accumulators, because the `K` loop is otherwise a single dependency
chain of `K / (BLOCK / TM)` fused multiply-adds — 160 of them at the shapes here,
each waiting on the last.

**The cross-group reduction is a tree, and that is what makes a large `BLOCK`
usable.** Summing the `BLOCK / TM` partials serially in the `kg == 0` threads
costs `BLOCK / TM` dependent adds performed by `TM` threads while the other
`BLOCK - TM` sit idle — so raising `BLOCK` to get occupancy paid for itself with
a longer tail, and `(5120, 1280)` measured *fastest* at `BLOCK = 128` for exactly
that reason. The tree is `log2(BLOCK / TM)` steps with every thread working, so
the tail stops growing with `BLOCK`.

Bias and epilogue are folded into the store for the same reason the GEMM folds
them: the alternative is a second kernel that reads and writes `M` elements to
add one number to each.
"""
function gemv_ncontig_kernel(TM::Int, BLOCK::Int, UNROLL::Int)
    get!(GEMVN_KERNELS, (TM, BLOCK, UNROLL)) do
        BLOCK % TM == 0 || throw(ArgumentError("gemv: BLOCK $BLOCK not a multiple of TM $TM"))
        ng = BLOCK ÷ TM                      # K-groups per workgroup
        ispow2(ng) || throw(ArgumentError(
            "gemv: BLOCK/TM = $ng must be a power of two for the reduction tree"))
        kname = Symbol("gemvn_", TM, "_", BLOCK, "_", UNROLL)
        # Built rather than written: `@nexprs` gives the accumulators, and their
        # final sum has to name all `UNROLL` of them in one expression.
        accsum = Expr(:call, :+, (Symbol("acc_", u) for u in 1:UNROLL)...)
        # …and the tree is unrolled here rather than written as a loop, so that
        # every `@synchronize` sits at the kernel's top level. A `@synchronize`
        # inside a loop is uniform here and would probably be fine; "probably
        # fine" is not what barriers are for.
        tree = Expr(:block)
        let s = ng ÷ 2
            while s > 0
                push!(tree.args, quote
                    if kg < $s
                        @inbounds parts[kg * $TM + tm + 1] +=
                            parts[(kg + $s) * $TM + tm + 1]
                    end
                    @synchronize
                end)
                s ÷= 2
            end
        end
        @eval begin
            @kernel cpu=false unsafe_indices=true function $kname(
                    C, @Const(W), @Const(x), @Const(bias), epi, M::Int, K::Int)
                parts = @localmem Float32 ($BLOCK,)

                tid = @index(Local, Linear) - 1
                grp = @index(Group, Linear) - 1
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
                @synchronize
                $tree
                @inbounds if kg == 0 && m <= M
                    t = parts[tm + 1]
                    bias === nothing || (t += Float32(bias[m]))
                    C[m] = epi(t)
                end
            end
            $kname
        end
    end
end

"""
    gemv_ncontig_config(M, K, limit) -> (TM, BLOCK, UNROLL)

`TM = 32` is the warp width and is not a free parameter: it is what makes one
warp read one contiguous 128-byte line.

**`BLOCK` and `UNROLL` barely matter, measured.** Swept over `TM ∈ {16, 32}`,
`BLOCK ∈ {128 … 1024}`, `UNROLL ∈ {2, 4, 8}` on the four shapes Whisper's decoder
runs, `(32, 256, 4)` is within 8% of the best configuration on every one of them:

    shape            best        (32,256,4)   spread over 24 configs
    (1280, 1280)     0.0538 ms   0.0586 ms    1.17x
    (5120, 1280)     0.0548 ms   0.0549 ms    1.04x
    (1280, 5120)     0.0545 ms   0.0582 ms    2.18x
    (51866, 1280)    0.8007 ms   0.8108 ms    1.64x

An earlier sweep said `BLOCK = 512` was **2.7x** better than `BLOCK = 256`. It
was not: this card idles at 450 MHz of 3105, a 50 us kernel never boosts it, and
the ranking was the clock ramp. Every row above is measured after spinning the
device for 0.25 s, and the same shape measured 0.030, 0.048 and 0.122 ms in one
session before that was understood.

The other thing that table shows: three shapes differing by 4x in bytes all land
on ~0.055 ms. That is not bandwidth, it is **Lava's ~63 us host cost per
dispatch**, which a 16 KiB GEMV pays in full — see the header.
"""
function gemv_ncontig_config(M::Int, K::Int, limit::Int)
    tm = 32
    block = min(limit, 256)
    block = max(block - block % tm, tm)
    return (tm, block, 4)
end

"""
    gemv_config(K, N, limit) -> (NROWS, BLOCK)

Rows per workgroup and threads per workgroup.

`NROWS` trades the vector re-read against register pressure and against having
enough workgroups to fill the device: at `N = 1280`, `NROWS = 32` leaves 40
workgroups for 48 shader cores, which starves it. `BLOCK` wants enough threads to
cover `K` in a few steps without the tail check dominating.

**Provisional.** These are starting points for the sweep in `test_gemv.jl`, not
measured optima, and llama.cpp's own values are per-architecture and per-
quantisation and do not transfer. Until that sweep runs, treat this function as
the thing under test.
"""
function gemv_config(K::Int, N::Int, limit::Int)
    nrows = N >= 4096 ? 8 : 4
    block = min(limit, K >= 4096 ? 256 : 128)
    return (nrows, block)
end

"""
    gemv!(C, A, B; nrows = nothing, block = nothing) -> C

`C = A * B` for `A` a length-`K` vector and `B` a `K x N` matrix, `C` length `N`.

The batch-1 path. `mul!` is correct at these shapes and roughly ten times slower
(see the table at the top of this file), so this is a separate entry point rather
than a branch inside `mul!` — a caller that knows it is decoding asks for it.

`nrows`/`block` override the plan, for the sweep.
"""
function gemv!(C::AbstractGPUArray{Float32}, A::AbstractGPUArray{Float32}, B::AbstractGPUArray{Float32,2};
               nrows::Union{Nothing,Int} = nothing, block::Union{Nothing,Int} = nothing)
    K, N = size(B, 1), size(B, 2)
    length(A) == K || throw(DimensionMismatch(
        "gemv!: A has $(length(A)) elements, expected K = $K"))
    length(C) == N || throw(DimensionMismatch(
        "gemv!: C has $(length(C)) elements, expected N = $N"))
    backend = get_backend(B)
    pn, pb = gemv_config(K, N, workgrouplimit(B))
    nr = nrows === nothing ? pn : nrows
    bl = block === nothing ? pb : block
    kern = gemv_kcontig_kernel(nr, bl)
    k = Base.invokelatest(kern, backend)     # `@eval`ed: world age, both halves
    Base.invokelatest(k, C, A, B, K, N;
                      ndrange = cld(N, nr) * bl, workgroupsize = bl)
    return C
end

"""
    gemv!(C, x, Wt::Transpose; bias, epilogue, tm, block, unroll) -> C

`C[m] = epilogue(bias[m] + sum_k W[m, k] * x[k])`, for `Wt = transpose(W)` with
`W` stored `(M, K)`.

Same function, other layout — the `Transpose` in the signature is not a
formality, it is *the* dispatch. `W` here is contiguous along `m`, and a kernel
written for a `(K, N)` matrix reading that one would stride the reduction axis by
`M` and lose every coalesced read. See the header for the two decompositions.

`C` and `x` may be any shape whose linear order is right — `(M, 1)` and `(K, 1)`
are what a graph hands over, and reshaping them to vectors first would allocate.
"""
function gemv!(C, x, Wt::Transpose{Float32,<:AbstractGPUArray{Float32,2}};
               bias = nothing, epilogue = identity,
               tm::Union{Nothing,Int} = nothing, block::Union{Nothing,Int} = nothing,
               unroll::Union{Nothing,Int} = nothing)
    W = parent(Wt)
    M, K = size(W)
    length(x) == K || throw(DimensionMismatch(
        "gemv!: x has $(length(x)) elements, expected K = $K"))
    length(C) == M || throw(DimensionMismatch(
        "gemv!: C has $(length(C)) elements, expected M = $M"))
    bias === nothing || length(bias) == M || throw(DimensionMismatch(
        "gemv!: bias has $(length(bias)) elements, expected M = $M"))
    backend = get_backend(W)
    ptm, pbl, pun = gemv_ncontig_config(M, K, workgrouplimit(W))
    t = tm === nothing ? ptm : tm
    bl = block === nothing ? pbl : block
    un = unroll === nothing ? pun : unroll
    kern = gemv_ncontig_kernel(t, bl, un)
    k = Base.invokelatest(kern, backend)     # `@eval`ed: world age, both halves
    Base.invokelatest(k, C, W, x, bias, epilogue, M, K;
                      ndrange = cld(M, t) * bl, workgroupsize = bl)
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

"""
    GEMV_PREGENERATED_LIMITS

Workgroup-size limits to pre-generate GEMV kernels for, at Lava load.

Both GEMV families are `@eval`-generated (see the two "Why this is
`@eval`-generated" notes above), and **an `@eval` on the runtime path cannot be
precompiled**: a downstream package whose `@compile_workload` reaches one dies
with

    Evaluation into the closed module `Lava` breaks incremental compilation

so the workload silently skips and every first call in a fresh process pays the
compile the Runner packages exist to remove. That is not hypothetical and it is
not new — `FFT_PREGENERATED` exists for exactly this, found by `KokoroRunner`.
The GEMV port reintroduced it, and `SAM2Runner` is where it showed up: its
workload skipped with that message, so nothing SAM 2 runs was frozen.

Generating at load moves the `@eval` to ordinary top-level code. The set is small
and derived — the loop below asks `gemv_config` and `gemv_ncontig_config`
themselves rather than restating their answers, so a change to either is followed
automatically. `(1, 4096)` are the two sides of every branch those functions take.

The limit is a *device* property and there is no device at load time, so the
plausible values are enumerated instead. A device outside this list still works;
it merely pays the old cost, which is the same failure mode as before and not a
worse one.
"""
const GEMV_PREGENERATED_LIMITS = (64, 128, 256, 512, 1024)

for L in GEMV_PREGENERATED_LIMITS
    for K in (1, 4096), N in (1, 4096)
        gemv_kcontig_kernel(gemv_config(K, N, L)...)
    end
    gemv_ncontig_kernel(gemv_ncontig_config(1, 1, L)...)
end
