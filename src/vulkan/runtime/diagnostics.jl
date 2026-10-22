# Runtime diagnostics over the default `VkContext`.

GPUArraysCore.allowscalar(false)

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

"""
    dump_state(; io::IO=stdout)

Print the default context's device, memory, cache, queue and dispatch counters.
"""
function dump_state(; io::IO=stdout)
    ctx = VK_CONTEXT_REF[]
    println(io, "=== Lava.jl State ===")
    println(io, "Device: ", ctx === nothing ? "not initialized" : ctx.device_name)
    println(io, "Device lost: ", device_lost())
    mem = gpu_memory_usage()
    live_mb = mem.live_bytes ÷ (1024 * 1024)
    println(io, "GPU memory: $(live_mb) MiB in $(mem.live_buffers) buffers ($(mem.retired) waiting to be destroyed)")
    println(io, "Pipelines cached: $(mem.pipelines_cached) (max $(MAX_PIPELINE_CACHE_SIZE[]))")
    println(io, "Kernels cached: $(mem.kernels_cached)")
    if ctx !== nothing
        println(io, "Unified blocks: $(mem.unified_blocks)")
    end
    if ctx !== nothing
        bq = ctx.default_bq
        println(io, "Outstanding: $(length(bq.outstanding)) submission(s)")
        println(io, "Recordings pooled: $(length(bq.free))")
    end
    println(io, "Flushes: $(ctx === nothing ? 0 : ctx.diag.flush_counter[])")
    println(io, "Total dispatches: $(ctx === nothing ? 0 : ctx.diag.total_dispatches[])")
    ctx === nothing || println(io, "Dispatch logging: $(ctx.diag.dispatch_logging)")
    dlog = ctx === nothing ? String[] : ctx.diag.dispatch_log
    if !isempty(dlog)
        println(io, "Last dispatch: ", last(dlog))
    end
    return nothing
end
