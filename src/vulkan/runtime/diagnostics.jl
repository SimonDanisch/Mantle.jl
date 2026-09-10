# Runtime diagnostics: what is on the device, and what it is holding.
#
# These were the tail of `Lava.jl` and moved here with everything else on
# 2026-08-27. Every one of them reads a `VkContext` — live buffers, deferred
# frees, argument slabs, the pipeline and kernel caches — so none of them could
# stay with a compiler that no longer names one.
#
# `allowscalar(false)` came too: it is a statement about the ARRAY type, and the
# array type is here.

GPUArraysCore.allowscalar(false)

# ---- Default settings ----
# Disable scalar indexing by default (GPU arrays should not be accessed element-by-element)
GPUArraysCore.allowscalar(false)


"""
    gpu_memory_usage() -> NamedTuple

Return current GPU memory usage statistics.
"""
function gpu_memory_usage()
    ctx = VK_CONTEXT_REF[]
    bq_deferred = 0
    if ctx !== nothing
        bq = ctx.default_bq
        bq_deferred = length(bq.pending) + length(bq.retiring)
    end
    # Both counts come off the context now, so both are guarded by the same
    # `ctx !== nothing` as the queue fields above — a caller with no device gets
    # zeros rather than an error.
    n_pipelines = ctx === nothing ? 0 : length(ctx.caches.pipelines)
    n_kernels   = ctx === nothing ? 0 : length(ctx.caches.linked)
    # `ARG_SLABS` was here, the count of the queue's argument slab ring. There is
    # no ring: a recording's arguments are blocks of the unified arena, which the
    # pool reports like any other memory it holds.
    (live_bytes = ctx === nothing ? 0 : gpu_live_bytes(ctx),
     live_buffers = ctx === nothing ? 0 : live_buffer_count(ctx),
     retired = bq_deferred,
     unified_blocks = ctx === nothing ? 0 : length(unifiedblocks(ctx)),
     pipelines_cached = n_pipelines,
     kernels_cached = n_kernels)
end

"""
    dump_state(; io::IO=stdout)

Print a comprehensive summary of Lava.jl runtime state for debugging.
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
