# Runtime diagnostics over the default `LavaDevice`.

GPUArraysCore.allowscalar(false)


"""
    gpu_memory_usage() -> NamedTuple

Live bytes and buffers, objects awaiting a deferred destroy, unified arena
blocks and cache sizes of the default context. All zeros when no device exists.
"""
function gpu_memory_usage()
    ctx = VK_CONTEXT_REF[]
    bq_deferred = 0
    if ctx !== nothing
        bq = ctx.default_bq
        bq_deferred = length(bq.pending) + length(bq.retiring)
    end
    n_pipelines = ctx === nothing ? 0 : length(ctx.caches.pipelines)
    n_kernels   = ctx === nothing ? 0 : length(ctx.caches.linked)
    (live_bytes = ctx === nothing ? 0 : gpu_live_bytes(ctx),
     live_buffers = ctx === nothing ? 0 : live_buffer_count(ctx),
     retired = bq_deferred,
     unified_blocks = ctx === nothing ? 0 : length(unifiedblocks(ctx)),
     pipelines_cached = n_pipelines,
     kernels_cached = n_kernels)
end

