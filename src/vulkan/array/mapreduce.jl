# GPU mapreducedim! for LavaArray
#
# Fast path: Float32 sum via a single-dispatch Vulkan-native reduce using
# OpGroupNonUniformFAdd (subgroup reduce) + OpAtomicFAddEXT to a BAR-mapped
# scalar. One dispatch, one fence wait, no partial-array temp, no CPU
# readback ping-pong — roughly 4-5× faster than the AK path on big arrays.
#
# Slow path: AcceleratedKernels.jl for everything else (partial-dim reductions,
# non-sum ops, non-Float32). KA-based tree reduction that iterates to 1 via
# repeated partial-reduce kernels, each forcing a sync.

import GPUArrays
import AcceleratedKernels as AK
using KernelAbstractions
const _KA_reduce = KernelAbstractions
using Atomix

# ── Vulkan-native single-dispatch sum ──

@kernel cpu=false function _vk_reduce_fadd_kernel!(out, @Const(src), total_threads::Int32)
    gi = _KA_reduce.@index(Global, Linear)
    n = length(src)
    # Strided: each thread walks the array in steps of `total_threads`. Cuts
    # atomic pressure by `n / total_threads` vs "1 thread per element" — at
    # n=10M with ~16k threads, ~160 elements/thread → ~256 atomics/step
    # instead of 156k.
    acc = 0f0
    i = gi
    @inbounds while i <= n
        acc += src[i]
        i += total_threads
    end
    # Subgroup (wave) reduce — single hardware instruction.
    s = subgroup_add(acc)
    # One atomic per wave, from the first active lane.
    if subgroup_elect()
        Atomix.@atomic out[1] += s
    end
    nothing
end

"""
    reduce_scratch(ctx) -> LavaArray{Float32,1}

The one-cell scratch buffer `vk_reduce_sum` writes its result into, on `ctx`.

Singleton because allocating a `LavaArray` per call was most of the per-call
overhead at small sizes. One cell of BAR-mapped memory; its `mapped_ptr` stays
valid for the lifetime of the context, which is now literally true — it is a
field of the context, so it dies with it and no reset callback has to remember.

It was an `IdDict` keyed by context, and the allocation went to `vk_context()`
rather than to `ctx`, so with two devices live the entry stored under the SECOND
context held a buffer belonging to the first. Keyed right, allocated wrong,
which reads as correct until there are two devices.
"""
@inline function reduce_scratch(ctx::VkContext)
    buf = ctx.caches.reduce_scratch
    buf === nothing || return buf::LavaArray{Float32,1}
    buf = LavaArray{Float32}(undef, (1,); unified=true, bq=ctx.default_bq)
    ctx.caches.reduce_scratch = buf
    return buf
end

"""
    vk_reduce_sum(A::LavaArray{Float32}) -> Float32

Vulkan-native single-pass reduction sum for Float32. Launches one compute
dispatch that does:
  per-thread load → subgroup_add → atomic_fadd into a BAR-mapped scalar.

ONE fence wait, result read directly from mapped memory. Replaces the AK
tree-reduce path's multiple CPU readbacks.
"""
function vk_reduce_sum(A::LavaArray{Float32})
    n = length(A)
    ctx = A.buf[].ctx
    out = reduce_scratch(ctx)
    # Zero the mapped cell directly — skips a fill! dispatch.
    out_ptr = Base.unsafe_convert(Ptr{Float32}, out.buf[].mapped_ptr)
    unsafe_store!(out_ptr, 0f0)
    # Fixed thread count — tuned for RX 7900 XTX-class GPUs. Too few threads
    # underutilizes SMs; too many causes atomic contention on out[1]. With
    # ~16k threads, n=10M → ~625 elems/thread → ~256 atomics → fast.
    # Tuning (RX 7900 XTX, n=10M benchmarked): wgsize=64 keeps one subgroup
    # per workgroup → each workgroup's atomic fires once.  nblocks=2048 saturates
    # the 96 CUs with enough overlap to hide memory latency. At this point we hit
    # ~85% of peak memory bandwidth for Float32 sum; smaller arrays are limited
    # by the fixed dispatch+fence overhead (~45 μs).
    wgsize = 64
    nblocks = min(cld(n, wgsize), 2048)
    total_threads = Int32(nblocks * wgsize)
    ndr = Int(total_threads)
    _vk_reduce_fadd_kernel!(KA.get_backend(A), wgsize)(out, A, total_threads; ndrange=ndr)
    bq = ctx.default_bq
    vk_flush!(bq)
    # out is mapped — read directly.
    return unsafe_load(out_ptr)
end

function GPUArrays.mapreducedim!(f::F, op::OP, R::LavaArray{T}, A::AbstractArray;
                                  init=nothing) where {F, OP, T}
    mapreducedim_ak!(f, op, R, A; init)
    return R
end

function GPUArrays.mapreducedim!(f::F, op::OP, R::LavaArray{T},
                                  A::Base.Broadcast.Broadcasted;
                                  init=nothing) where {F, OP, T}
    # Materialize broadcasted to LavaArray first — AK expects AbstractArray
    A_mat = Base.materialize(A)
    mapreducedim_ak!(f, op, R, A_mat; init)
    return R
end

# Transposed destinations.
#
# `Base.mapreducedim!` sends any GPU-array destination here, and `transpose(v)`
# of a LavaArray is one — but it is not itself a `LavaArray`, so it fell past
# both methods above into GPUArrays' generic `error("Not implemented")`. That
# single gap was all 133 errors the GPUArrays conformance suite reported: it
# reduces into `transpose(zeros(ET, ...))` and `adjoint(...)` for every eltype.
#
# The obvious repair — reduce into `parent(R)` — is wrong twice over. The parent
# of a transposed vector has the reduction's axes swapped, and `Adjoint`
# conjugates on access, so R's contents are `conj.(parent)` rather than the
# parent itself. Measured on a 2x2 `Complex{Int64}` sum: unwrapping gives
# `[3+3im, 7+7im]` where the answer is `[4-4im, 6-6im]`. A real-valued smoke
# test cannot see either error.
#
# So go through a dense temporary in R's own logical shape and let the wrapper
# do what it exists to do: `tmp .= R` reads through the transpose, `R .= tmp`
# writes back through it and conjugates when R is an `Adjoint`. No branch on
# which wrapper it is, and no assumption about the parent's strides — which is
# what makes the matrix case (`transpose` of a 2x3, reduced into as a 3x2) come
# out right as well.
const LavaTransposed{T} = Union{Transpose{T, <:LavaArray{T}}, Adjoint{T, <:LavaArray{T}}}

# `A` must be spelled exactly as GPUArrays spells it. `Base.AbstractArrayOrBroadcasted`
# looks like the same type but unions in `Base.AbstractBroadcasted`, the abstract
# supertype, where GPUArrays unions in the concrete `Broadcast.Broadcasted`. That
# makes this method narrower in `R` and *wider* in `A` than the fallback it is
# meant to beat, so neither is more specific and every call is an ambiguity error
# rather than a dispatch to this method.
function GPUArrays.mapreducedim!(f::F, op::OP, R::LavaTransposed{T},
                                  A::Union{AbstractArray, Base.Broadcast.Broadcasted};
                                  init=nothing) where {F, OP, T}
    tmp = similar(parent(R), T, size(R))
    # Seed with R's current contents: with `init === nothing` the reduction
    # accumulates into the destination rather than overwriting it.
    tmp .= R
    GPUArrays.mapreducedim!(f, op, tmp, A; init)
    R .= tmp
    return R
end

function mapreducedim_ak!(f::F, op::OP, R::LavaArray{T}, A;
                            init=nothing) where {F, OP, T}
    n = length(A)
    n == 0 && return R

    # What `init` means, which is not what it looks like. A supplied `init` says R
    # is uninitialized scratch — `_mapreduce` allocates R with `similar` and then
    # always passes one, which is why `sum`/`prod` work — so R's contents are
    # garbage and must be overwritten. `init === nothing` is the opposite: it is
    # `Base.mapreducedim!`'s own call, and there R's current contents ARE the
    # accumulator seed. Reducing `[1 2; 3 4]` into `[10, 20]` gives `[13, 27]`.
    #
    # Both were treated as "overwrite with the neutral element", so every
    # `mapreducedim!`/`reducedim!`/`sum!` into a non-empty destination silently
    # dropped what was already there. The GPUArrays conformance suite cannot catch
    # it: it only ever seeds with the neutral element itself (`zeros` for `+`,
    # `ones` for `*`), and `op(neutral, x) == x` hides the difference.
    #
    # Snapshotting here keeps every branch below to the single job of overwriting R.
    prior = init === nothing ? copy(R) : nothing
    init_val = init === nothing ? GPUArrays.neutral_element(op, T) : convert(T, init)

    if length(R) == 1
        # Full reduction (dims=:)
        #
        # Fast path: f=identity, op=+, eltype Float32, input is a plain
        # LavaArray{Float32} — one-dispatch Vulkan-native reduce with
        # OpGroupNonUniformFAdd + atomic_fadd. Handles ~85% of peak memory
        # bandwidth; AK's tree-reduce is ~2× slower because of per-level
        # scalar readbacks. Any non-matching case falls through to AK.
        # `Base.add_sum` is `Base.sum`'s default op (=== +, but different fn object).
        result = if T === Float32 && (op === (+) || op === Base.add_sum) && f === identity && A isa LavaArray{Float32}
            vk_reduce_sum(A::LavaArray{Float32}) + init_val
        else
            # Fallback: AK.mapreduce to scalar
            AK.mapreduce(f, op, A, KA.get_backend(A);
                         init=init_val, neutral=init_val,
                         block_size=64, switch_below=0)
        end
        # Write scalar result into R
        R_host = T[convert(T, result)]
        copyto!(R, 1, R_host, 1, 1)
    else
        # Partial reduction (dims=N) — determine which dim is being reduced.
        # AK.mapreduce needs a dense array, so wrappers (views, PermutedDimsArray,
        # ...) get materialised — on the *device*, via broadcast. This used to be
        # `convert(LavaArray, collect(A))`, i.e. a blocking download to the host
        # followed by a re-upload; `sum(view(x, ...); dims=)` alone was 21% of a
        # DNNKernels inference step.
        A_arr = A isa LavaArray ? A : densify(A)
        rdim = find_reduced_dim(size(A_arr), size(R))
        if rdim !== nothing
            # AK.mapreduce with dims= expects temp to have same ndims as input
            # with the reduced dim collapsed to 1. If R has fewer dims (implicit
            # singleton), reshape it.
            R_temp = if ndims(R) < ndims(A_arr)
                expected_sz = ntuple(i -> i == rdim ? 1 : size(A_arr, i), ndims(A_arr))
                reshape(R, expected_sz)
            else
                R
            end
            AK.mapreduce(f, op, A_arr, KA.get_backend(R_temp);
                         init=init_val, neutral=init_val,
                         dims=rdim, temp=R_temp, block_size=64)
        elseif size(A_arr) == size(R)
            # No dimension reduced: R and A have same shape (e.g., dims=[]).
            #
            # Seeding via `fill!` rather than broadcasting `init_val` in
            # directly: for `findmax`-style reductions the neutral element is a
            # `Tuple{Float32,Int64}`, and broadcast treats a tuple as a container
            # to iterate, not a scalar — `op.(init_val, ...)` fails
            # `check_broadcast_shape` against a 2D destination. `fill!` takes it
            # as the value it is.
            fill!(R, init_val)
            R .= op.(R, f.(A_arr))
        else
            # Multiple dims reduced or ndims mismatch — reduce sequentially
            # along each reduced dimension
            multi_dim_reduce!(f, op, R, A_arr, init_val)
        end
    end
    # Fold R's pre-existing contents back in. Every branch above wrote the
    # reduction seeded with the neutral element, so this is what turns it into an
    # accumulation.
    prior === nothing || (R .= op.(prior, R))
    return R
end

"""Reduce along multiple dimensions by iterating over each reduced dim."""
function multi_dim_reduce!(f::F, op::OP, R::LavaArray{T}, A::LavaArray,
                            init_val) where {F, OP, T}
    # Find all dimensions that need reducing (size went to 1 or was dropped)
    szA = size(A)
    szR = size(R)
    ndA = ndims(A)
    ndR = ndims(R)

    # Pad R's size to match A's dimensionality
    szR_padded = ntuple(i -> i <= ndR ? szR[i] : 1, ndA)

    # Reduce one dimension at a time
    current = A
    first_reduction = true
    for d in ndA:-1:1
        if szA[d] != szR_padded[d] && szR_padded[d] == 1
            # Apply f only on the first reduction step
            map_fn = first_reduction ? f : identity
            first_reduction = false

            # Compute target size for this reduction step
            new_sz = ntuple(ndA) do i
                i == d ? 1 : size(current, i)
            end
            if new_sz == size(R)
                # Last reduction — write directly into R
                AK.mapreduce(map_fn, op, current, KA.get_backend(R);
                             init=init_val, neutral=init_val, dims=d, temp=R, block_size=64)
            else
                temp = LavaArray{T}(undef, new_sz)
                fill!(temp, init_val)
                AK.mapreduce(map_fn, op, current, KA.get_backend(temp);
                             init=init_val, neutral=init_val, dims=d, temp=temp, block_size=64)
                current = temp
            end
        end
    end
    return R
end

"""Find which single dimension was reduced (size went to 1).
Handles implicit singleton dimensions (e.g., (2,2) → (2,) means dim 2 reduced)."""
function find_reduced_dim(szA, szR)
    ndA = length(szA)
    ndR = length(szR)

    # Pad R's size with trailing 1s to match A's dimensionality
    szR_padded = ntuple(i -> i <= ndR ? szR[i] : 1, ndA)

    rdim = nothing
    for i in 1:ndA
        if szA[i] != szR_padded[i]
            if szR_padded[i] == 1 && rdim === nothing
                rdim = i
            else
                return nothing  # Multiple dims or unexpected shape
            end
        end
    end
    return rdim
end

# ── Sort ──
# There is deliberately no `AK.merge_sort_by_key!` override here.
#
# One used to exist, implementing sort-by-key as sortperm + permute, because
# AK's block-level merge kernel reads shared-memory positions it never wrote
# when `len < 2 * block_size`, and Vulkan leaves workgroup memory undefined.
# That override was circular: AK implements `sortperm` *via* `merge_sort_by_key!`
# (see AcceleratedKernels/src/sort/merge_sortperm.jl), so the two called each
# other forever — a StackOverflowError, or 34 GB of pool growth and
# ERROR_OUT_OF_DEVICE_MEMORY when the recursion allocated temporaries first.
#
# The real defect was the uninitialized shared memory, and that is now fixed at
# the source: the Workgroup Block variable is emitted with an OpConstantNull
# initializer (see `emit_workgroup_block!` in compiler/compilation.jl), so every
# kernel starts with zeroed shared memory and AK's own implementation is correct.
