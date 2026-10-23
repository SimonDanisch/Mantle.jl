# Vulkan buffers with device addresses: dedicated and pooled allocation, freeing,
# and host transfers.

"""
    scan_unified_for_bda!(buf) -> Int

Count the words of the unified arena (every recording's arguments and indirect
counts) equal to `buf.address`, log each to `diag.freed_bda_scan_log` and, with
`freed_bda_scan_poisons`, zero it so the GPU faults on null instead of reading
freed memory. Debug only: one read per word, about 512 K per 4 MiB block.
"""
function scan_unified_for_bda! end

# `VkManagedBuffer.state`, changed only by CAS: `vk_free!` moves ALIVE →
# DEFERRED, `destroy_buffer!` moves ALIVE or DEFERRED → DEAD once. A second free
# from any thread, finalizers included, is a no-op.
const BUF_STATE_ALIVE    = UInt8(0)
const BUF_STATE_DEFERRED = UInt8(1)
const BUF_STATE_DEAD     = UInt8(2)

# Address a freed buffer is given. No valid BDA is 0, and offsets from null stay
# in unmapped low memory, so a shader using a freed address faults. The host-side
# gate is `buf.state`.
const BDA_POISON = UInt64(0)

# Everything here is per `LavaDevice`, so `reset_device!`'s new context starts fresh.

"""Sum of device-local heap sizes in bytes for `ctx`'s physical device."""
function probe_device_local_heap(ctx::LavaDevice)
    mem_props = ctx.memory_properties
    total = 0
    # `memory_heaps` has VK_MAX_MEMORY_HEAPS (16) slots; `memory_heap_count` are valid.
    for i in 0:(Int(mem_props.memory_heap_count) - 1)
        heap = mem_props.memory_heaps[i + 1]
        if (UInt32(heap.flags) & UInt32(VK.MEMORY_HEAP_DEVICE_LOCAL_BIT)) != 0
            total += Int(heap.size)
        end
    end
    return total
end

"""
Per-heap `(heap, device_local, size, budget, usage)` in bytes, `heap` 0-based.
`budget` and `usage` come from `VK_EXT_memory_budget` and are 0 without it.
"""
function probe_device_memory_budget(ctx::LavaDevice)
    mem_props = ctx.memory_properties
    n = Int(mem_props.memory_heap_count)      # of the 16 slots in `memory_heaps`
    sizes = ntuple(i -> Int(mem_props.memory_heaps[i].size), n)
    flags = ntuple(i -> UInt32(mem_props.memory_heaps[i].flags), n)
    device_local = ntuple(i -> (flags[i] & UInt32(VK.MEMORY_HEAP_DEVICE_LOCAL_BIT)) != 0, n)
    if !ctx.memory_budget_available
        return [(heap=i-1, device_local=device_local[i], size=sizes[i],
                 budget=0, usage=0) for i in 1:n]
    end
    # VK.jl allocates and fills the requested `next` chain types itself.
    props2 = VK.get_physical_device_memory_properties_2(
        ctx.physical_device, VK.PhysicalDeviceMemoryBudgetPropertiesEXT)
    bp = props2.next::VK.PhysicalDeviceMemoryBudgetPropertiesEXT
    return [(heap=i-1, device_local=device_local[i], size=sizes[i],
             budget=Int(bp.heap_budget[i]), usage=Int(bp.heap_usage[i])) for i in 1:n]
end

"""
    overbudget(ctx, memtype, bytes) -> Bool

Whether `bytes` more in memory type `memtype` (0-based) would exceed the
`VK_EXT_memory_budget` budget of its heap; `false` without the extension.
Callers treat it as `ERROR_OUT_OF_DEVICE_MEMORY` even where the driver would
allocate: RADV then spills into GTT, and past about half the RAM amdgpu's
`ttm_global_swapout` (kernel 7.2.6) corrupts a list and locks up the machine.
The second method applies the rule to one heap of a
[`probe_device_memory_budget`](@ref) snapshot.
"""
function overbudget(ctx::LavaDevice, memtype::Integer, bytes::Integer)
    ctx.memory_budget_available || return false
    heap = Int(ctx.memory_properties.memory_types[memtype + 1].heap_index)
    return overbudget(probe_device_memory_budget(ctx)[heap + 1], bytes)
end
overbudget(heap::NamedTuple, bytes::Integer) = heap.usage + bytes > heap.budget

"""
    trim_gpu_pool!(ctx = vk_context(); invalidate_recordings = false) -> (blocks, bytes)

Return every empty pool block to the driver now, after an unconditional
`quiesce_before_reclaim!` (flush, full GC, drain): until then a buffer held by a
submission or retired but not destroyed keeps its block live. For "this batch is
done, give the memory back" and before reading a VRAM figure; the automatic
`autotrim!` is rate-limited, runs only on allocation and only once
`Mantle.reclaimable` sees something to return.

`invalidate_recordings = true` also waits for all queues and discards the
recorded commands of live arena plans, which otherwise keep their blocks; the
plans and their resources stay valid and record again on the next `run!`.
Returns `(0, 0)` while a recording is open, the device is lost or a collection
is already running.
"""
function trim_gpu_pool!(ctx::LavaDevice = vk_context(); invalidate_recordings::Bool=false)
    bq = ctx.default_bq
    if invalidate_recordings
        # Never sweep an open recording's positional hold frame.
        isempty(bq.holds) || return (0, 0)
        all(ch -> ch.thread==Threads.threadid() && isempty(ch.holds),ctx.extra_queues) || return (0, 0)
        device_lost(ctx) && return (0, 0)
        flush!(bq)
        foreach(flush!,ctx.extra_queues)
        waitidle(ctx)
        seen=IdDict{Any,Nothing}()
        for arena in values(pool(ctx).arenas), plan in tenants!(arena)
            plan isa Plan && recorded(plan) || continue
            haskey(seen,plan) && continue
            seen[plan]=nothing
            invalidate!(plan)
        end
    end
    quiesce_before_reclaim!(bq) || return (0, 0)
    if invalidate_recordings
        foreach(drain!,ctx.extra_queues)
        reclaim!(pool(ctx),ctx;wait=true)
    end
    return reclaim_empty_pool_blocks!(bq)
end

"""
    maybe_collect(ctx::LavaDevice; blocking::Bool=false)

Run core's `autotrim!`, then enter core's collection controller with the
device-local heap pressure. Called by `vk_alloc` and `pool_alloc` before
allocating; a no-op with `eager_gc` off. `blocking=true` lowers the pressure
threshold and raises the rate budget, for a caller about to stall anyway.
"""
function maybe_collect(ctx::LavaDevice; blocking::Bool=false)
    p = mempolicy(ctx)
    p.eager_gc || return
    stats = p.stats
    current_time = time()

    # Before the pressure gate: a heap ratio is the wrong signal for dead pool
    # blocks, e.g. on an iGPU.
    autotrim!(pool(ctx), ctx)

    # Re-probe the device-local heap size every 10 s.
    if current_time - (@atomic stats.last_updated) > 10.0
        max_size = probe_device_local_heap(ctx)
        @atomic stats.size = max_size
        @atomic stats.last_updated = current_time
    end

    size = (@atomic stats.size)
    size > 0 || return  # haven't probed yet

    autocollect!(() -> drain!(ctx.default_bq), pool(ctx);
                 pressure = gpu_live_bytes(ctx) / size, blocking) || return

    # Return blocks the collection emptied, without a second collection.
    trimcollected!(pool(ctx), ctx)

    return
end

"""
The Vulkan result and the failing call of a refused allocation, returned by
`try_vk_alloc` so the final `LavaError` can name them.
"""
struct AllocFailure
    code::VK.Result
    op::Symbol          # :Buffer, :budget, :DeviceMemory, :bind_buffer_memory, :map_memory
    nbytes::Int
    mem_type_idx::Int   # -1 if the failure happened before memory-type selection
end

function format_oom_error(ctx::LavaDevice, fail::AllocFailure)
    io = IOBuffer()
    println(io, "Out of GPU memory.")
    req_mb = fail.nbytes ÷ (1024 * 1024)
    live_mb = gpu_live_bytes(ctx) ÷ (1024 * 1024)
    if fail.op === :budget
        println(io, "  $(fail.nbytes) bytes ($(req_mb) MiB) would exceed the driver's budget for the heap (VK_EXT_memory_budget); not asked for.")
    else
        println(io, "  Vulkan returned $(fail.code) from $(fail.op) for $(fail.nbytes) bytes ($(req_mb) MiB).")
    end
    println(io, "  Lava tracked state: $(live_mb) MiB live across $(live_buffer_count(ctx)) buffers.")
    if fail.mem_type_idx >= 0
        mem_props = ctx.memory_properties
        mt = mem_props.memory_types[fail.mem_type_idx + 1]
        heap_idx = Int(mt.heap_index)
        println(io, "  Memory type idx=$(fail.mem_type_idx) → heap idx=$(heap_idx), property_flags=0x$(string(UInt32(mt.property_flags), base=16))")
    end
    snapshot = probe_device_memory_budget(ctx)
    if ctx.memory_budget_available
        println(io, "  Driver heap budgets (VK_EXT_memory_budget):")
        for h in snapshot
            sz_mb = h.size ÷ (1024 * 1024)
            bud_mb = h.budget ÷ (1024 * 1024)
            use_mb = h.usage ÷ (1024 * 1024)
            tag = h.device_local ? "DEVICE_LOCAL" : "HOST"
            println(io, "    heap $(h.heap) [$tag] size=$(sz_mb) MiB budget=$(bud_mb) MiB used=$(use_mb) MiB")
        end
    else
        println(io, "  (VK_EXT_memory_budget unavailable — only static heap sizes known)")
        for h in snapshot
            sz_mb = h.size ÷ (1024 * 1024)
            tag = h.device_local ? "DEVICE_LOCAL" : "HOST"
            println(io, "    heap $(h.heap) [$tag] size=$(sz_mb) MiB")
        end
    end
    return String(take!(io))
end

"""
    vk_alloc(bq::SubmitChannel{<:VulkanQueue}, nbytes; extra_usage=UInt32(0), unified=false) -> VkManagedBuffer

Allocate a dedicated buffer with a device address. Only `bq`'s finished work is
drained first. On failure, quiesce, return empty pool blocks and retry once,
then throw a `LavaError` with the heap figures.

* `extra_usage`: additional `VkBufferUsageFlags` bits (INDIRECT_BUFFER,
  INDEX_BUFFER, AS_BUILD_INPUT, ...).
* `unified=true`: device-local host-visible coherent memory if any (else
  host-visible coherent), mapped; `mapped_ptr` is non-null.

Acceleration-structure scratch alignment is the caller's (`bda_alignment_for`).
"""
function vk_alloc(bq::SubmitChannel{<:VulkanQueue}, nbytes::Integer;
                  extra_usage::UInt32=UInt32(0), unified::Bool=false)
    # On a lost device a Vulkan call may succeed against torn-down state and
    # return garbage addresses.
    refuselost(deviceof(bq), "vk_alloc")
    drain!(bq)
    maybe_collect(deviceof(bq))
    result = try_vk_alloc(bq, nbytes; extra_usage, unified)
    result isa VkManagedBuffer && return result
    # Return empty pool blocks before the retry; without this the pool ratchets
    # up across renders (3.7 GiB of empty blocks pinning the heap). Refused while
    # a recording pins the pool: the allocation then fails with the OOM below
    # instead of freeing a block a replay would later fault on.
    if quiesce_before_reclaim!(bq)
        n_blocks, bytes_freed = reclaim_empty_pool_blocks!(bq)
        if n_blocks > 0
            @info "Lava: reclaimed empty pool blocks on OOM retry" blocks=n_blocks MiB=(bytes_freed >> 20)
        end
    end
    result = try_vk_alloc(bq, nbytes; extra_usage, unified)
    if result isa VkManagedBuffer
        @info "Lava: GPU allocation succeeded after GC retry" bytes=nbytes
        return result
    end
    fail = result::AllocFailure
    throw(LavaError("memory allocation",
        format_oom_error(deviceof(bq), fail),
        "Free unused LavaArrays, reduce problem size, or check for memory leaks with gpu_memory_usage()."))
end

"""
    quiesce_before_reclaim!(bq) -> Bool

Wait for everything submitted on `bq`, run a full GC, drain deferred frees and
reclaim with waiting (core's `quiesce_pool!`). Callers must run it before
returning pool blocks: reclaiming earlier can destroy a VkBuffer that queued
work still reads (later use-after-free, signal desync or a driver segfault).
Runs only on allocation failure or an explicit trim. `false`, doing nothing, on
a lost device, while a recording pins the pool, or while a collection runs.
"""
function quiesce_before_reclaim!(bq::SubmitChannel{<:VulkanQueue})
    dev = deviceof(bq)
    device_lost(dev) && return false
    return quiesce_pool!(() -> flush!(bq), () -> drain!(bq), pool(dev), dev)
end

"""Attempt GPU buffer allocation, returning an `AllocFailure` on OOM."""
function try_vk_alloc(bq::SubmitChannel{<:VulkanQueue}, nbytes::Integer;
                      extra_usage::UInt32=UInt32(0), unified::Bool=false)
    ctx = deviceof(bq)
    dev = ctx.device
    nbytes = max(nbytes, 16)

    usage = VK.BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK.BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
            VK.BUFFER_USAGE_TRANSFER_SRC_BIT |
            VK.BUFFER_USAGE_TRANSFER_DST_BIT
    if extra_usage != UInt32(0)
        usage |= VK.BufferUsageFlag(extra_usage)
    end

    local buf, memory, mem_type_idx
    mapped_ptr = Ptr{UInt8}(0)
    # The call in flight, for `AllocFailure`. `mem_type_idx_local` stays defined
    # when a throw precedes the assignment of `mem_type_idx`.
    op::Symbol = :Buffer
    mem_type_idx_local::Int = -1
    try
        buf = VK.Buffer(dev, nbytes, usage, VK.SHARING_MODE_EXCLUSIVE, UInt32[])
        mem_reqs = VK.get_buffer_memory_requirements(dev, buf)

        if unified
            preferred = VK.MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
                        VK.MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                        VK.MEMORY_PROPERTY_HOST_COHERENT_BIT
            mem_type_idx = find_memory_type_optional(ctx, mem_reqs.memory_type_bits, preferred)
            if mem_type_idx === nothing
                mem_type_idx = find_memory_type(ctx, mem_reqs.memory_type_bits,
                    VK.MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                    VK.MEMORY_PROPERTY_HOST_COHERENT_BIT)
            end
        else
            mem_type_idx = find_memory_type(ctx, mem_reqs.memory_type_bits,
                VK.MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
        end
        mem_type_idx_local = Int(mem_type_idx)
        # Over budget is out of memory (`overbudget`), with the same retry.
        if overbudget(ctx, mem_type_idx, mem_reqs.size)
            clear_validation_messages!(ctx)      # creating `buf` may already have complained
            return AllocFailure(VK.ERROR_OUT_OF_DEVICE_MEMORY, :budget, Int(nbytes), mem_type_idx_local)
        end

        alloc_flags = VK.MemoryAllocateFlagsInfo(UInt32(0);
            flags=VK.MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT)
        op = :DeviceMemory
        memory = VK.DeviceMemory(dev, mem_reqs.size, mem_type_idx; next=alloc_flags)
        op = :bind_buffer_memory
        unwrap(VK.bind_buffer_memory(dev, buf, memory, 0))

        if unified
            op = :map_memory
            mapped_ptr = Ptr{UInt8}(unwrap(VK.map_memory(dev, memory, 0, nbytes)))
        end
    catch e
        # Only out-of-memory becomes an `AllocFailure`; any other error
        # propagates, or `vk_alloc` would retry and report a false OOM.
        if e isa VK.VulkanError &&
           (e.code == VK.ERROR_OUT_OF_DEVICE_MEMORY ||
            e.code == VK.ERROR_OUT_OF_HOST_MEMORY)
            clear_validation_messages!(deviceof(bq))
            return AllocFailure(e.code, op, Int(nbytes), mem_type_idx_local)
        end
        # Mark the device lost, so later calls refuse with a clear error.
        if e isa VK.VulkanError && e.code == VK.ERROR_DEVICE_LOST
            mark_device_lost!(ctx)
        end
        rethrow()
    end

    addr_info = VK.BufferDeviceAddressInfo(buf)
    address = VK.get_buffer_device_address(dev, addr_info)

    result = VkManagedBuffer(buf, memory, address, mapped_ptr, Int(nbytes), nothing, Stamp{UInt64}(), BUF_STATE_ALIVE, ctx)
    let p = mempolicy(ctx)
        @lock p.livelock push!(p.live_buffers, result)
        Threads.atomic_add!(p.live_bytes, nbytes)
    end
    return result
end

"""
    vk_free!(buf::VkManagedBuffer)

Request destruction of the buffer; destroys nothing itself. Core's `retire!`
records it and `reclaim!` calls `rawfree` once the last submission naming the
buffer has passed and no unsubmitted recording holds it. Callable from any
thread, finalizers included: it reads no stamp or timeline.
"""
function vk_free!(buf::VkManagedBuffer)
    _, ok = @atomicreplace buf.state BUF_STATE_ALIVE => BUF_STATE_DEFERRED
    ok || return

    c = buf.ctx
    if !(c isa LavaDevice)
        # No context: not pooled and named by no queue, so destroy inline.
        destroy_buffer!(buf)
        return
    end
    # Parked on the default channel; `reclaim!` reads the stamp on the owning
    # thread and forwards the buffer to the channel that last named it.
    retire!(c.default_bq, buf)
    return
end

"""Destroy a buffer's Vulkan resources, or return its pool region: the `rawfree`
core calls once nothing can read the bytes, and the inline path for a buffer
without a context. The state CAS makes it run exactly once under racing callers."""
function destroy_buffer!(buf::VkManagedBuffer)
    # DEFERRED is the `vk_free!` path; ALIVE a destroy without a prior free.
    _, ok = @atomicreplace buf.state BUF_STATE_DEFERRED => BUF_STATE_DEAD
    if !ok
        _, ok = @atomicreplace buf.state BUF_STATE_ALIVE => BUF_STATE_DEAD
    end
    ok || return  # was already DEAD; idempotent

    # Logged here, not in `vk_free!`: the stamp may only be read on the owning thread.
    let c = buf.ctx
        if c isa LavaDevice && c.diag.free_debug
            st = buf.stamp
            push!(c.diag.free_log,
                  (addr=buf.address, size=buf.size, pool=buf.region !== nothing,
                   lw=(st.channel === nothing ? nothing : (st.channel, st.token))))
        end
        # The address still in an argument block is a pending use-after-free.
        if c isa LavaDevice && c.diag.freed_bda_scan
            hits = scan_unified_for_bda!(buf)
            if hits > 0 && c.diag.destroy_freed_bdas_throws
                throw(LavaError("destroy_buffer!",
                    "destroying buffer at 0x$(string(buf.address, base=16, pad=16)) but its BDA still appears $(hits)x in live arg slabs",
                    "an unheld reference is leaking — see ctx.diag.freed_bda_scan_log"))
            end
        end
    end

    # Suballocated: return the region and leave the block's VkBuffer alone, it is
    # shared by every region of the block. `release!`, not `retire!`: core only
    # calls `rawfree` after the last submission naming the buffer has passed.
    # `release!` takes the block's lock.
    let r = buf.region
        if r !== nothing
            buf.region = nothing
            buf.address = BDA_POISON
            buf.mapped_ptr = Ptr{UInt8}(0)
            buf.size = 0
            buf.stamp.channel = nothing
            release!(r::Region)
            return
        end
    end

    ctx = buf.ctx::LavaDevice
    let p = mempolicy(ctx)
        @lock p.livelock delete!(p.live_buffers, buf)
    end
    # A lost device, which includes every device at exit
    # (`mark_all_devices_lost!`), gets no Vulkan calls: only poison the handle.
    if device_lost(ctx)
        buf.mapped_ptr = Ptr{UInt8}(0)
        buf.address = BDA_POISON
        Threads.atomic_sub!(mempolicy(ctx).live_bytes, buf.size)
        buf.size = 0
        return
    end
    # No vkUnmapMemory: vkFreeMemory unmaps implicitly, and the explicit unmap
    # was a crash site on RADV (a SIGSEGV, which no try/catch catches).
    buf.mapped_ptr = Ptr{UInt8}(0)
    try
        buf.buffer.destructor()
        buf.memory.destructor()
    catch ex
        # Never throw from a destructor; log the actual error.
        safe_fin_log("Lava destroy_buffer!: Vulkan destructor failed: " *
                     sprint(showerror, ex) * "\n")
    end
    buf.address = BDA_POISON
    Threads.atomic_sub!(mempolicy(ctx).live_bytes, buf.size)
    buf.size = 0
end

"""The blocks of the `Unified` arena, which hold every recording's arguments and
indirect counts: where a stale device address can sit when the GPU runs."""
unifiedblocks(ctx::LavaDevice) = get(() -> Block[], pool(ctx).blocks, Unified())

function scan_unified_for_bda!(buf::VkManagedBuffer)
    target = buf.address
    target == UInt64(0) && return 0
    hits = 0
    ctx = buf.ctx
    for (i, blk) in enumerate(unifiedblocks(ctx))
        mb = (blk.memory::BufferBlock).ref[]::VkManagedBuffer
        mb.mapped_ptr == Ptr{UInt8}(0) && continue
        n = mb.size ÷ 8
        p = Ptr{UInt64}(mb.mapped_ptr)
        for k in 0:(n-1)
            v = unsafe_load(p, k+1)
            if v == target
                push!(ctx.diag.freed_bda_scan_log,
                      (slab=:unified, idx=i, offset=k*8, freed_bda=target,
                       buf_size=buf.size))
                # `freed_bda_scan_poisons = false` makes the scan observe only.
                ctx.diag.freed_bda_scan_poisons && unsafe_store!(p, UInt64(0), k+1)
                hits += 1
            end
        end
    end
    return hits
end

# Sub-allocation uses `Mantle.Pool`, the pool graph arenas use too: a second
# allocator over the same `VkDevice` could not reuse its bytes. `carve!` splits
# exactly, and each allocation gets a fresh `VkManagedBuffer` (a suballocation
# costs about 300 ns). Destruction is core's `reclaim!` calling `rawfree`.

"""Size of a new pool block: 64 MiB, `Mantle.blocksize`'s value."""
const POOL_BLOCK_SIZE = 64 * 1024 * 1024

"""
The smallest allocation. Vulkan rejects a zero-length buffer, and 16 bytes keeps
every region's start usable for any scalar type.
"""
const POOL_MIN_SIZE = 16

"""
Alignment of every sub-allocation: `Mantle.acquire!`'s default, covering every
`minStorageBufferOffsetAlignment` seen. Acceleration-structure scratch needs
more and takes the dedicated path (`bda_alignment_for`).
"""
const POOL_ALIGN = 256

MemoryPolicy() = MemoryPolicy(false, true, MemoryStats(),
                              Threads.Atomic{Int}(0), Set{VkManagedBuffer}(), ReentrantLock())

"""
    mempolicy(ctx) -> MemoryPolicy

This device's allocation policy and dedicated-buffer accounting; the memory
itself comes from the device's [`pool`](@ref).
"""
@inline mempolicy(ctx::LavaDevice) = ctx.caches.pool

"""
    pool_offset(buf) -> Int

Byte offset of `buf` in its pool block, 0 when it owns its memory. Derived from
`region`, so it always matches what the allocator recorded.
"""
@inline pool_offset(buf::VkManagedBuffer) =
    buf.region === nothing ? 0 : offset(buf.region::Region)

"""Every block this device has, across the arenas they were cut for."""
poolblocks(ctx::LavaDevice) =
    collect(Iterators.flatten(values(pool(ctx).blocks)))

"""
    gpu_live_bytes(ctx = vk_context()) -> Int
    live_buffer_count(ctx = vk_context()) -> Int

Bytes this device holds from the driver (pool blocks plus dedicated buffers),
and the number of dedicated buffers. Pooled regions are not counted separately:
they cost nothing beyond their block.
"""
gpu_live_bytes(ctx::LavaDevice = vk_context()) =
    reserved(pool(ctx)) + mempolicy(ctx).live_bytes[]
live_buffer_count(ctx::LavaDevice = vk_context()) =
    (p = mempolicy(ctx); @lock p.livelock length(p.live_buffers))

"""
    destroy_pool!(ctx)

Destroy all of this device's blocks, live regions included, when `reset_device!`
retires the context. Unlike [`trim_gpu_pool!`](@ref) it ignores the `live`
ledger: whatever holds those regions points at a dead device either way.
"""
function destroy_pool!(ctx::LavaDevice)
    p = pool(ctx)
    for (_, blks) in p.blocks, blk in blks
        rawfree(ctx, blk.memory)
    end
    empty!(p.blocks)
    empty!(p.arenas)
    # Dropped: they waited on this device. `takeall!` because an `Inbox` has no `empty!`.
    takeall!(p.pending)
    takeall!(p.pendingplans)
    empty!(p.retiring)
    empty!(p.retiring_at)
    return nothing
end

"""
    pool_alloc(bq::SubmitChannel{<:VulkanQueue}, nbytes; extra_usage=UInt32(0)) -> VkManagedBuffer

Allocate GPU memory on `bq` from this device's `Mantle.Pool`, or a dedicated
buffer when the pool is disabled. `extra_usage` is the region's constraint: it
is carved from a block whose usage bits permit it (`compatible`), and a new
block with those bits is grown when none does.
"""
function pool_alloc(bq::SubmitChannel{<:VulkanQueue}, nbytes::Integer; extra_usage::UInt32=UInt32(0))
    ctx = deviceof(bq)
    # Even reusing a free span: the blocks of a lost device give garbage addresses.
    refuselost(ctx, "pool_alloc")
    p = mempolicy(ctx)
    nbytes = max(Int(nbytes), POOL_MIN_SIZE)
    drain!(bq)
    # Same hook as `vk_alloc`; without it the pool grows until full and then
    # pays a 2-5 ms `GC.gc` inside one allocation.
    maybe_collect(ctx)

    p.disabled && return vk_alloc(bq, nbytes; extra_usage)

    sp = pool(ctx)
    # Past the soft cap, collect (spaced by `autocollect!`) before growing: the
    # bytes are then more likely dead than in use. The `drain!` returns what the
    # collection retired to the channel, or the pool cannot reuse it.
    let cap = sp.policy.soft_cap
        cap > 0 && reserved(sp) >= cap && autocollect!(() -> drain!(bq), sp)
    end

    region = acquire_or_reclaim!(bq, sp, ctx, nbytes, extra_usage)
    blk = region.block.memory::BufferBlock
    return VkManagedBuffer(
        blk.buffer, blk.memory,
        blk.address + UInt64(offset(region)),
        Ptr{UInt8}(0),
        length(region),
        region, Stamp{UInt64}(), BUF_STATE_ALIVE, ctx)
end

"""
Acquire a region through core's retry policy (`onpressure` quiesces with
`quiesce_before_reclaim!` and trims); a final driver out-of-memory becomes a
`LavaError` naming the pool's reserved bytes, other errors propagate. `dev` is
untyped because `LavaDevice` is declared in `graph.jl`, included after this file.
Recording scratch (`scratch!`) calls `acquire!` directly with `maywait = false`.
"""
function acquire_or_reclaim!(bq::SubmitChannel{<:VulkanQueue}, sp::Pool, dev,
                             nbytes::Int, extra_usage::UInt32)
    try
        return acquire!(sp, dev, Buffers(), nothing, nbytes;
                        align = POOL_ALIGN, blocksize = POOL_BLOCK_SIZE,
                        constraint = extra_usage,
                        onpressure = () -> makeroom!(() -> quiesce_before_reclaim!(bq), sp, dev))
    catch err
        (err isa VK.VulkanError && isoutofmemory(err)) || rethrow()
        throw(LavaError("pool_alloc",
            "Cannot allocate $(nbytes) bytes: the device is out of memory with " *
            "$(humanbytes(reserved(sp))) already reserved by this pool after " *
            "the core reclamation and retry policy.",
            "Free unused LavaArrays, reduce problem size, or check for leaks with " *
            "gpu_memory_usage()."))
    end
end

"""The driver's two out-of-memory codes; an over-budget `device_memory` throws the device one."""
isoutofmemory(err::VK.VulkanError) =
    err.code == VK.ERROR_OUT_OF_DEVICE_MEMORY || err.code == VK.ERROR_OUT_OF_HOST_MEMORY

"""
    reclaim_empty_pool_blocks!(bq::SubmitChannel{<:VulkanQueue}) -> (n_blocks::Int, bytes::Int)

Return every block with an empty `live` ledger to the driver (`Mantle.trim!`)
and report how many and how many bytes. Callers must run
`quiesce_before_reclaim!` first. Not finalizer-safe.
"""
function reclaim_empty_pool_blocks!(bq::SubmitChannel{<:VulkanQueue})
    ctx = deviceof(bq)
    sp = pool(ctx)
    before = reserved(sp)
    nbefore = length(poolblocks(ctx))
    trim!(sp, ctx)
    return (nbefore - length(poolblocks(ctx)), before - reserved(sp))
end

"""Largest download read straight out of mapped BAR memory; see `copy_buffer!`."""
const BAR_DIRECT_READ_BYTES = 4096

# Staged uploads larger than this go in chunks, so they need no second
# allocation of their full size in host memory. Mapped uploads copy directly.
const MAX_UPLOAD_STAGING_BYTES = 64 * 1024 * 1024

"""copy_buffer!(direction, managed, host_ptr, nbytes; offset=0)

Copy `nbytes` between host and `managed` at byte `offset`, `direction` `:upload`
or `:download`. Mapped uploads and small mapped downloads memcpy after prior use
completes; other transfers stage through pooled Readback regions, downloads on
the buffer's last live channel. The caller keeps `host_ptr` valid during the call."""
function copy_buffer!(direction::Symbol, managed::VkManagedBuffer,
                     host_ptr::Ptr{UInt8}, nbytes::Integer; offset::Integer=0)
    nbytes == 0 && return
    checkspan(managed.size,offset,nbytes)
    @assert direction === :upload || direction === :download  "direction must be :upload or :download"
    buf_offset = pool_offset(managed) + Int(offset)

    # Mapped uploads and small downloads memcpy directly. Large BAR reads use
    # host-cached Readback memory to avoid uncached host reads.
    if managed.mapped_ptr != Ptr{UInt8}(0) &&
       (direction === :upload || nbytes <= BAR_DIRECT_READ_BYTES)
        # The stamp is set at submit and owners submit when closed, so this
        # waits for every prior use.
        waitfor!(stampof(managed))
        if direction === :upload
            unsafe_copyto!(managed.mapped_ptr + offset, host_ptr, nbytes)
        else
            unsafe_copyto!(host_ptr, managed.mapped_ptr + offset, nbytes)
        end
        return
    end

    # Core stamps order transfers against other channels. Downloads use the
    # last live producer channel to avoid a second timeline wait.
    bq = if direction === :upload
        (managed.ctx::LavaDevice).default_bq
    else
        wbq = stampof(managed).channel
        (wbq !== nothing && !queue_released(wbq::SubmitChannel{<:VulkanQueue})) ?
            (wbq::SubmitChannel{<:VulkanQueue}) : (managed.ctx::LavaDevice).default_bq
    end
    # `Mantle.offset`, qualified: this function's `offset` keyword shadows it.
    if direction === :upload
        if nbytes > MAX_UPLOAD_STAGING_BYTES
            copied = 0
            while copied < nbytes
                count = min(Int(nbytes) - copied, MAX_UPLOAD_STAGING_BYTES)
                token = oneshot!(bq; tag = :upload) do e
                    r = scratch!(e.owner, count, Readback())
                    mb = (memoryof(r)::BufferBlock).ref[]::VkManagedBuffer
                    unsafe_copyto!(mb.mapped_ptr + Mantle.offset(r), host_ptr + copied, count)
                    cmd_copy_buffer!(e, mb, managed, count;
                        src_off = pool_offset(mb) + Mantle.offset(r), dst_off = buf_offset + copied)
                end
                # Sweep, not only wait: that returns this chunk's staging region
                # before the next chunk acquires one.
                waitfor!(bq, token)
                sweep!(bq)
                copied += count
            end
            return
        end
        # The upload one-shot owns scratch until its copy completes.
        oneshot!(bq; tag = :upload) do e
            # Staged in host memory and moved by the copy engine; see `scratch!`.
            r = scratch!(e.owner, nbytes, Readback())
            mb = (memoryof(r)::BufferBlock).ref[]::VkManagedBuffer
            unsafe_copyto!(mb.mapped_ptr + Mantle.offset(r), host_ptr, nbytes)
            cmd_copy_buffer!(e, mb, managed, nbytes;
                             src_off = pool_offset(mb) + Mantle.offset(r), dst_off = buf_offset)
        end
    else
        # Owned here until the host has read it; one-shot scratch could be
        # reclaimed before that read.
        dev = managed.ctx::LavaDevice
        r = acquire!(pool(dev), dev, Readback(), nothing, max(Int(nbytes), 16);
                     align = ARG_ALIGN, blocksize = READBACK_BLOCK_SIZE)
        mb = (memoryof(r)::BufferBlock).ref[]::VkManagedBuffer
        tok = oneshot!(bq; tag = :download) do e
            cmd_copy_buffer!(e, managed, mb, nbytes;
                             src_off = buf_offset, dst_off = pool_offset(mb) + Mantle.offset(r))
        end
        waitfor!(bq, tok)
        unsafe_copyto!(host_ptr, mb.mapped_ptr + Mantle.offset(r), nbytes)
        release!(r)
    end
    return
end

"""Upload host bytes to `dst` at byte `offset`."""
function upload!(dst::VkManagedBuffer, host_data::Vector{UInt8}; offset::Int=0)
    upload_typed!(dst, host_data; offset)
end

const HostArrayView = SubArray{T,N,P} where {T,N,P<:Array}
const HostReinterpretArray{T} = Base.ReinterpretArray{T,N,S,A} where {N,S,A<:Union{Array,HostArrayView}}


"""Upload isbits host values. Contiguous host arrays and views transfer directly;
other layouts are collected first, in logical order."""
upload_typed!(dst::VkManagedBuffer, data::AbstractArray; offset::Int=0) =
    upload_typed!(dst,data,1,length(data);offset)

function upload_typed!(dst::VkManagedBuffer, data::AbstractArray{T},
                       first::Integer, n::Integer; offset::Int=0) where {T}
    isbitstype(T) || throw(ArgumentError("upload_typed! requires an isbits element type, got $T"))
    n == 0 && return nothing
    first = copybounds(data,first,n)
    n = Int(n)
    data isa Array || dense_host_storage(data) ||
        return upload_typed!(dst,collect(data),first,n;offset)
    GC.@preserve data copy_buffer!(:upload,dst,Ptr{UInt8}(pointer(data,first)),n*sizeof(T);offset)
end

"""Fill `data`, which must be contiguous, from `src` at byte `offset`."""
function download_typed!(data::AbstractVector{T}, src::VkManagedBuffer; offset::Int=0) where T
    GC.@preserve data copy_buffer!(:download, src, Ptr{UInt8}(pointer(data)),
                                    length(data) * sizeof(T); offset)
end

"""
    bda_alignment_for(ctx::LavaDevice, scratch::Bool) -> UInt64

Device-address alignment a buffer needs beyond Vulkan's own:
`minAccelerationStructureScratchOffsetAlignment` (`ctx.as_scratch_align`) for
acceleration-structure build scratch, otherwise 1.
"""
@inline function bda_alignment_for(ctx::LavaDevice, scratch::Bool)
    return scratch ? ctx.as_scratch_align : UInt64(1)
end

function find_memory_type_optional(ctx::LavaDevice, type_bits::UInt32, required_flags)
    mem_props = ctx.memory_properties
    for i in 0:(length(mem_props.memory_types) - 1)
        if (type_bits & (UInt32(1) << i)) != 0
            mt = mem_props.memory_types[i + 1]
            if (mt.property_flags & required_flags) == required_flags
                return UInt32(i)
            end
        end
    end
    return nothing
end

function find_memory_type(ctx::LavaDevice, type_bits::UInt32, required_flags;
                          preferred_flags=required_flags)
    i = find_memory_type_optional(ctx, type_bits, required_flags | preferred_flags)
    if i === nothing && preferred_flags != required_flags
        i = find_memory_type_optional(ctx, type_bits, required_flags)
    end
    i === nothing && throw(LavaError(
        "memory allocation",
        "No suitable memory type found for flags $required_flags",
        "Check GPU memory capabilities"))
    return i
end
