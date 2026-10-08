# Full Float32 sum for LavaArray
#
# A single-dispatch Vulkan-native reduce using OpGroupNonUniformFAdd (subgroup
# reduce) + OpAtomicFAddEXT to a BAR-mapped scalar. One dispatch, one fence wait,
# no partial-array temp, no CPU readback ping-pong — roughly 4-5× faster than the
# AcceleratedKernels path on big arrays.
#
# Every other reduction is GPUArrays' (12 and later), on top of
# AcceleratedKernels: GPUArrays has no extension point for a vendor reduction, so
# this is a narrower `Base.sum` method rather than a hook.

using KernelAbstractions
using Atomix

# ── Vulkan-native single-dispatch sum ──

function _vk_reduce_fadd_kernel!(out, src, total_threads::Int32)
    gi = KI.get_global_id().x
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

A field and not an `IdDict` keyed by context: keyed right and allocated from
`vk_context()` rather than from `ctx`, the entry stored under the second context
holds a buffer belonging to the first, which reads as correct until there are
two devices.
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
    KI.Kernel(KA.get_backend(A), _vk_reduce_fadd_kernel!)(
        out, A, total_threads; ndrange=ndr, workgroupsize=wgsize)
    bq = ctx.default_bq
    flush!(bq)
    # out is mapped — read directly.
    return unsafe_load(out_ptr)
end

"""
    sum(A::LavaArray{Float32})

The full sum, through [`vk_reduce_sum`](@ref). A call with `dims` or `init` is a
keyword call, which this method does not take, so it stays GPUArrays'.
"""
Base.sum(A::LavaArray{Float32}) = vk_reduce_sum(A)

# ── Sort ──
# There is deliberately no `AK.merge_sort_by_key!` override here.
#
# Implementing sort-by-key as sortperm + permute is circular: AK implements
# `sortperm` *via* `merge_sort_by_key!` (see
# AcceleratedKernels/src/sort/merge_sortperm.jl), so the two call each other
# forever — a StackOverflowError, or 34 GB of pool growth and
# ERROR_OUT_OF_DEVICE_MEMORY when the recursion allocates temporaries first.
#
# What such an override would be working around is AK's block-level merge kernel
# reading shared-memory positions it never wrote when `len < 2 * block_size`,
# with Vulkan leaving workgroup memory undefined. That is fixed at
# the source: the Workgroup Block variable is emitted with an OpConstantNull
# initializer (see `emit_workgroup_block!` in compiler/compilation.jl), so every
# kernel starts with zeroed shared memory and AK's own implementation is correct.
