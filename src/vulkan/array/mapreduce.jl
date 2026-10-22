# Full Float32 `sum` for LavaArray in one dispatch: subgroup add, then one
# float atomic add per subgroup into a host-mapped scalar. Every other reduction
# is GPUArrays' on AcceleratedKernels; GPUArrays has no hook for a vendor
# reduction, so this is a narrower `Base.sum` method.

using KernelAbstractions
using Atomix

function _vk_reduce_fadd_kernel!(out, src, total_threads::Int32)
    gi = KI.get_global_id().x
    n = length(src)
    # Grid-stride loop, so atomics scale with the thread count, not with `n`.
    # -0.0 is the identity of float addition; starting from +0.0 turns a sum of
    # negative zeros into +0.0 where Base gives -0.0.
    acc = -0f0
    i = gi
    @inbounds while i <= n
        acc += src[i]
        i += total_threads
    end
    s = subgroup_add(acc)
    # One atomic per subgroup. Monotonic suffices: the submission fence makes
    # the result visible to the host read of the mapped scalar.
    if subgroup_elect()
        Atomix.@atomic :monotonic out[1] += s
    end
    nothing
end

"""
    reduce_scratch(ctx) -> LavaArray{Float32,1}

The host-mapped one-cell buffer `vk_reduce_sum` writes into, allocated once
per context on `ctx`'s queue and stored in `ctx.caches`, so its `mapped_ptr`
lives as long as `ctx`. Allocating per call dominated small sums.
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

Sum `A` with one dispatch and one flush, reading the result from the mapped
`reduce_scratch(ctx)` cell.
"""
function vk_reduce_sum(A::LavaArray{Float32})
    n = length(A)
    # Base's empty sum is +0.0, not the kernel's -0.0 seed.
    n == 0 && return 0f0
    ctx = A.buf[].ctx
    out = reduce_scratch(ctx)
    # Seed the mapped cell from the host instead of a fill! dispatch.
    out_ptr = Base.unsafe_convert(Ptr{Float32}, out.buf[].mapped_ptr)
    unsafe_store!(out_ptr, -0f0)
    # Tuned on RX 7900 XTX (96 CUs): at most 2048 workgroups of 64 fill the GPU
    # while bounding contention on out[1]; reaches about 85% of peak bandwidth.
    wgsize = 64
    nblocks = min(cld(n, wgsize), 2048)
    total_threads = Int32(nblocks * wgsize)
    ndr = Int(total_threads)
    KI.Kernel(KA.get_backend(A), _vk_reduce_fadd_kernel!)(
        out, A, total_threads; ndrange=ndr, workgroupsize=wgsize)
    bq = ctx.default_bq
    flush!(bq)
    return unsafe_load(out_ptr)
end

"""
    sum(A::LavaArray{Float32})

The full sum via [`vk_reduce_sum`](@ref). Calls with `dims` or `init` are
keyword calls and stay with GPUArrays.
"""
Base.sum(A::LavaArray{Float32}) = vk_reduce_sum(A)

# No `AK.merge_sort_by_key!` override: sortperm + permute would recurse, since
# AK implements `sortperm` via `merge_sort_by_key!`. AK's merge kernel reads
# unwritten shared memory when `len < 2 * block_size`; Lava zero-initializes
# workgroup memory (`emit_combined_workgroup_block!`), which makes that correct.
