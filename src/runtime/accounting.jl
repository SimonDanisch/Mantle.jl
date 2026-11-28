# What a device holds from its driver and what it may still use, answered alike for
# every device. A backend keeps the ledger and reports its heaps; core does the rest.

"""
    BufferLedger()

The buffers a device holds from its driver outside its pool blocks, each made for one
resource: how many there are, and their bytes. A backend calls [`allocated!`](@ref)
when it allocates such a buffer and [`freed!`](@ref) when it frees it. The counters
are atomic, so a release that runs in a finalizer can record itself.
"""
struct BufferLedger
    count::Threads.Atomic{Int}
    bytes::Threads.Atomic{Int}
end
BufferLedger() = BufferLedger(Threads.Atomic{Int}(0), Threads.Atomic{Int}(0))

"""
    allocated!(ledger, bytes) -> ledger
    freed!(ledger, bytes) -> ledger

Record a buffer of `bytes` the device allocated outside its pool, or its release.
"""
function allocated!(l::BufferLedger, bytes::Integer)
    Threads.atomic_add!(l.count, 1)
    Threads.atomic_add!(l.bytes, Int(bytes))
    return l
end
function freed!(l::BufferLedger, bytes::Integer)
    Threads.atomic_sub!(l.count, 1)
    Threads.atomic_sub!(l.bytes, Int(bytes))
    return l
end

"""
    ledger(device) -> BufferLedger

The [`BufferLedger`](@ref) of `device`'s buffers outside its pool.
"""
function ledger end

"""
    live_buffer_count(device = Device()) -> Int

How many buffers `device` holds from its driver outside its pool blocks, each made for
one resource. Pooled regions cost nothing beyond their block, which
[`gpu_live_bytes`](@ref) counts. A leak check compares it before and after work that
must give everything back, with a full collection and [`drain!`](@ref) before each read.
"""
live_buffer_count(dev::Device = Device()) = ledger(dev).count[]

"""
    gpu_live_bytes(device = Device()) -> Int

Bytes `device` holds from its driver: its pool blocks and the buffers
[`live_buffer_count`](@ref) counts.
"""
gpu_live_bytes(dev::Device = Device()) = reserved(pool(dev)) + ledger(dev).bytes[]

"""
    gpu_memory_usage(device = Device()) -> NamedTuple

`(; live_bytes, live_buffers, pool_bytes, pool_blocks)`: [`gpu_live_bytes`](@ref),
[`live_buffer_count`](@ref), and the bytes and number of `device`'s pool blocks.
"""
function gpu_memory_usage(dev::Device = Device())
    p = pool(dev)
    return (live_bytes = gpu_live_bytes(dev), live_buffers = live_buffer_count(dev),
            pool_bytes = reserved(p), pool_blocks = nblocks(p))
end

"""The number of blocks `pool` holds, over all arenas."""
nblocks(p::Pool) = lock(() -> sum(length, values(p.blocks); init = 0), p.lock)

"""
    probe_device_memory_budget(device) -> Vector{NamedTuple}

`(; heap, device_local, size, budget, usage)` in bytes for each memory heap of `device`,
`heap` 0-based: its size, how much of it the driver lets this process use, and how much
the process uses. `budget` and `usage` are 0 where the driver does not report them.
"""
function probe_device_memory_budget end

"""
    save_pipeline_cache!(device)

Write the pipelines `device`'s driver has compiled to disk, so the next process links
them instead of compiling. Mantle does it at exit; a process that is killed instead
calls it before. A device whose driver keeps no such cache has nothing to write.
"""
function save_pipeline_cache! end

"""
    trim_gpu_pool!(device = Device(); invalidate_recordings = false) -> (blocks, bytes)

Return every empty pool block of `device` to the driver now, after a flush, a full
collection and a drain: until then a buffer held by a submission, or retired but not
yet freed, keeps its block. For "this batch is done, give the memory back" and before
reading a memory figure; the automatic [`autotrim!`](@ref) runs only on allocation.

`invalidate_recordings = true` also waits for the device and discards the recorded
commands of live arena plans, which otherwise keep their blocks; the plans and their
resources stay valid and record again on their next `run!`. Returns `(0, 0)` while a
recording pins the pool, on a retired device, or while a collection is running.
"""
function trim_gpu_pool!(dev::Device; invalidate_recordings::Bool = false)
    retired(dev) && return (0, 0)
    p = pool(dev)
    if invalidate_recordings
        waitidle(dev)
        invalidaterecorded!(p)
    end
    quiesce_pool!(p, dev) || return (0, 0)
    invalidate_recordings && reclaim!(p, dev; wait = true)
    return trimcounted!(p, dev)
end
trim_gpu_pool!(; invalidate_recordings::Bool = false) = trim_gpu_pool!(Device(); invalidate_recordings)

"""
Discard the recorded commands of every recorded plan placed in `pool`'s arenas; each
records again on its next `run!`.
"""
function invalidaterecorded!(p::Pool)
    seen = IdDict{Any,Nothing}()
    for arena in values(p.arenas), plan in tenants!(arena)
        plan isa Plan && recorded(plan) || continue
        haskey(seen, plan) && continue
        seen[plan] = nothing
        invalidate!(plan)
    end
    return p
end

"""
    trimcounted!(pool, dev) -> (blocks, bytes)

[`trim!`](@ref) `pool`, and how many blocks and bytes it gave back to the driver.
"""
function trimcounted!(p::Pool, dev)
    before, n = reserved(p), nblocks(p)
    trim!(p, dev)
    return (n - nblocks(p), before - reserved(p))
end
