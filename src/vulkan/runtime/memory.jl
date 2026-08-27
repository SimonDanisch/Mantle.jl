# Vulkan memory management for Lava.jl
#
# Device-local buffers with BDA (Buffer Device Address) for kernel arguments.
# Staging buffer for CPU↔GPU transfers.


# Debug instrumentation for iter6 cross-scene cascade investigation.
# All off by default — opt-in via the *_ENABLED Refs.

# Per-finalizer destruction trace

# Per-allocation trace

# When enabled, vk_free! scans every live BatchQueue's arg/indirect slabs for
# any UInt64 == buf.address.  Hits are logged + zeroed so the GPU faults on a
# clean null reference instead of corrupting random memory.  Optionally
# throws a LavaError instead of just logging (`ctx.diag.destroy_freed_bdas_throws`).

# Pre-submit unknown-BDA scan: scans the active arg slab region for any
# BDA-shaped UInt64 that isn't in the pool's `live_buffers` or any pool block / slab.
# Optionally throws a LavaError instead of just warning.

# When true, pack_arg!(::VkManagedBuffer, ...) asserts the buffer's state ==
# ALIVE before packing its BDA.  Catches use-after-free where a stale buffer
# reference makes it through to a kernel arg.

# When set to a non-zero target BDA, every submit! scans the active arg slab
# for the target value and logs (submit_idx, offsets) to `diag.slab_dump_log`.  Used
# to track when a known-stale address enters/leaves the arg slab.

"""
    scan_arg_slabs_for_bda!(buf) -> Int

Scan every live BatchQueue's `arg_slabs` (and `indirect_slabs`) for any
UInt64 word matching `buf.address`.  For each hit, append a record to
`diag.freed_bda_scan_log` and overwrite the slot with 0 so the GPU faults
cleanly on a null reference instead of touching the freed memory.
Returns the number of hits found.

Defined AFTER VkManagedBuffer + BatchQueue (forward-call from vk_free!).
Cost is ~`(slab_size_bytes / 8)` UInt64 reads per live slab — for 4 MiB
slabs that's ~512 K reads, fast enough for debug.
"""
function scan_arg_slabs_for_bda! end

# Buffer lifecycle states (atomic CAS transitions).  Every VkManagedBuffer
# starts ALIVE.  `unsafe_free!` transitions ALIVE → DEFERRED (queued on a
# bq's deferred_frees list) or ALIVE → DEAD (destroyed immediately).
# `destroy_buffer!` transitions {ALIVE, DEFERRED} → DEAD.  Any call on a
# DEAD buffer is a no-op.  Using an `@atomic` field with CAS guarantees
# idempotent double-free protection across main + finalizer threads.
const BUF_STATE_ALIVE    = UInt8(0)
const BUF_STATE_DEFERRED = UInt8(1)
const BUF_STATE_DEAD     = UInt8(2)


# Strong references to keep Vulkan handles alive until explicit free

# Poison value for freed buffer addresses — enables use-after-free detection
# on the GPU side (a shader dereffing a freed BDA traps with this value).
# The CPU-side use-after-free gate is `buf.state`.
#
# Poison = 0: no valid GPU BDA is ever 0, so this is unambiguous.  Pointer
# arithmetic on null (ptr + N*sizeof(T)) keeps the result in the unmapped
# low-VA region, which faults clearly.  Detection is also trivial — scan
# arg buffer slots for `== 0`.
const BDA_POISON = UInt64(0)

# No reset callback: every counter above is a `DevicePool` field, and the pool
# is a `VkContext` field, so all of it dies with the context `vk_reset_device!`
# retires. Staging, indirect and arg slabs are per-`BatchQueue` and go the same
# way. `reset_memory_stats!` is the one piece left, and `vk_reset_device!` calls
# it directly.

# ── GPU memory pressure tracking (ported from AMDGPU.jl) ──
# Julia's GC doesn't know about GPU memory. LavaArray wrappers are ~50 bytes on
# the CPU heap, but back 100+ MB of VRAM each. Without pressure signals, GC
# never fires and dead GPU buffers accumulate until OOM.
#
# Strategy mirrors `AMDGPU.jl/src/memory.jl::maybe_collect`:
#   * Pressure-based trigger (`live / heap_size`), not raw byte counter, so the
#     same logic works for 8 GiB iGPUs and 24 GiB dGPUs without tuning.
#   * GC-rate budget: collector skips itself when it has already spent more than
#     ~5% of wall time in GC (`last_gc_time / dt`).  Budget doubles on high
#     pressure, on blocking calls, and after a productive collection.
#   * Only `GC.gc(false)` (incremental) is ever called automatically — full GC
#     is too expensive to fire from the alloc hot path; if we're truly OOM the
#     caller's retry loop in `vk_alloc` runs the full GC explicitly.
#   * EWMA on `last_gc_time` so a single slow GC doesn't permanently inhibit
#     future GCs.
#
# `pool.live_bytes` is incremented by `try_vk_alloc` (real Vulkan allocations,
# including pool blocks) and decremented by `destroy_buffer!`.  Sub-pool chunks
# don't move this counter — the pool block they live in already accounts for
# the VRAM.

mutable struct MemoryStats
    # Estimated maximum bytes available to us on the device-local heap.
    # Probed lazily from `ctx.memory_properties` and refreshed every 10s.
    @atomic size::Int
    @atomic last_updated::Float64

    # Last `maybe_collect` run + the rolling cost of that GC.
    @atomic last_time::Float64
    @atomic last_gc_time::Float64
    # Bytes freed by the most recent `maybe_collect`-triggered GC.
    @atomic last_freed::Int
end

MemoryStats() = MemoryStats(0, 0.0, 0.0, 0.0, 0)

const MEMORY_STATS = MemoryStats()

const EAGER_GC = Ref{Bool}(true)

"""
    eager_gc!(flag::Bool)

Enable/disable the pressure-driven `maybe_collect`.  Useful when benchmarking,
to take the allocator's GC hooks out of the measurement.
"""
eager_gc!(flag::Bool) = (EAGER_GC[] = flag)

function reset_memory_stats!()
    @atomic MEMORY_STATS.size = 0
    @atomic MEMORY_STATS.last_updated = 0.0
    @atomic MEMORY_STATS.last_time = 0.0
    @atomic MEMORY_STATS.last_gc_time = 0.0
    @atomic MEMORY_STATS.last_freed = 0
    return
end

"""Sum of device-local heap sizes in bytes for `ctx`'s physical device."""
function probe_device_local_heap(ctx::VkContext)
    mem_props = ctx.memory_properties
    total = 0
    for i in 0:(length(mem_props.memory_heaps) - 1)
        heap = mem_props.memory_heaps[i + 1]
        if (UInt32(heap.flags) & UInt32(VK.MEMORY_HEAP_DEVICE_LOCAL_BIT)) != 0
            total += Int(heap.size)
        end
    end
    return total
end

"""
Per-heap snapshot of (device_local, size, budget, usage) in bytes.  `budget`
and `usage` are the driver's view via VK_EXT_memory_budget; both are 0 when the
extension is unavailable.  Used to attach real driver-side memory pressure to
OOM error messages.
"""
function probe_device_memory_budget(ctx::VkContext)
    mem_props = ctx.memory_properties
    n = Int(length(mem_props.memory_heaps))
    sizes = ntuple(i -> Int(mem_props.memory_heaps[i].size), n)
    flags = ntuple(i -> UInt32(mem_props.memory_heaps[i].flags), n)
    device_local = ntuple(i -> (flags[i] & UInt32(VK.MEMORY_HEAP_DEVICE_LOCAL_BIT)) != 0, n)
    if !ctx.memory_budget_available
        return [(heap=i-1, device_local=device_local[i], size=sizes[i],
                 budget=0, usage=0) for i in 1:n]
    end
    # VK.jl's get_physical_device_memory_properties_2 takes the desired
    # chain types as varargs and allocates+populates them itself.
    props2 = VK.get_physical_device_memory_properties_2(
        ctx.physical_device, VK.PhysicalDeviceMemoryBudgetPropertiesEXT)
    bp = props2.next::VK.PhysicalDeviceMemoryBudgetPropertiesEXT
    return [(heap=i-1, device_local=device_local[i], size=sizes[i],
             budget=Int(bp.heap_budget[i]), usage=Int(bp.heap_usage[i])) for i in 1:n]
end

"""
    maybe_collect(ctx::VkContext; blocking::Bool=false)

Trigger an incremental GC if GPU pressure is high.  Ported from
`AMDGPU.jl/src/memory.jl::maybe_collect`.

Called from `vk_alloc` and `pool_alloc` before allocating.  `blocking=true`
lowers the pressure threshold and inflates the rate budget — use it when the
caller is about to do a heavy synchronous operation anyway.
"""
# Absolute-capacity pool trim.
#
# `maybe_collect`'s pressure gate is a *ratio* against the device heap, which is
# the wrong signal for holding on to dead pool blocks on an iGPU: 3 GB of empty
# blocks is only ~20 % of a large shared heap, so the gate never trips — but that
# 3 GB is system RAM the rest of the machine still needs, and the pool is only
# handed back on an OOM retry. Long multi-scene runs therefore accumulate
# gigabytes of blocks that nothing will ever reclaim.
#
# So trim on absolute dead capacity as well, rate-limited so a render loop can't
# pay for it repeatedly. Blocks only become empty once the GC has run their
# sub-allocations' finalizers, hence the collection before the scan.

# A sub-allocation is returned to its block by a **finalizer**, so a block only
# becomes empty once those have run — which is what the paragraph above says, and
# what `GC.gc(false)` does not do. Julia's incremental collection sweeps the
# young generation and leaves older objects, and a pool chunk that has survived a
# few frames is exactly an older object. So the scan below almost always found
# nothing and the trim almost never fired: the pool ratcheted up to its
# transient high-water mark and stayed there for the life of the process.
#
# Measured on SAM 2.1: encode then 20 decodes reached 2 096 MiB live with two
# empty 64 MiB blocks that no automatic path would return. A full collection
# followed by `reclaim_empty_pool_blocks!` handed back 128 MiB.
#
# So escalate: try the cheap collection first, and if it finds no empty block,
# pay for a full one — but on its own, longer, timer. A full GC is the expensive
# part (tens of ms on a heap this size) and a render loop must not pay it every
# five seconds; unbounded pool growth is the worse of the two, but not by so much
# that it justifies a hitch per frame.

"""
    reclaimable(ctx) -> Bool

Whether the *automatic* trim could return anything without flushing.

Two things make a block empty. It may already be, or the release may be sitting
on a `deferred_frees` list: a sub-allocation's finalizer moves the buffer
ALIVE → DEFERRED, and the `live_count` it holds is not given back until
`drain_deferred_frees!` runs inside `quiesce_before_reclaim!` — after the gate.
So `any(b -> b.live_count == 0, blocks)` on its own is a precondition the trim
establishes, and gating on it alone means declining to look.

**This deliberately does not see the third case**, which is a buffer pinned by a
batch that is still recording: `vk_free!` takes the `pins > 0` branch, sets
`free_requested` and returns, and only the flush inside
`quiesce_before_reclaim!` releases it. Detecting that would make the automatic
path flush whenever a batch is open, which for a render loop is a stall every
`trim_min_interval`. [`trim_gpu_pool!`](@ref) is the caller that wants it and
pays for it explicitly.
"""
function reclaimable(ctx::VkContext)
    p = pool(ctx)
    any(b -> b.live_count == 0, p.blocks) && return true
    bq = ctx.default_bq
    return lock(bq.deferred_frees_lock) do
        !isempty(bq.deferred_frees) || !isempty(bq.deferred_as_frees)
    end
end

function maybe_trim_pool!(ctx::VkContext)
    p = pool(ctx)
    p.live_bytes[] < p.trim_threshold && return
    now = time()
    now - p.last_trim < p.trim_min_interval && return
    p.last_trim = now

    GC.gc(false)
    if !reclaimable(ctx)
        # Nothing reclaimable *yet*; the finalizers may simply not have run.
        now - p.last_full_gc < p.trim_full_gc_interval && return
        p.last_full_gc = now
        GC.gc(true)
        reclaimable(ctx) || return
    end
    bq = ctx.default_bq
    quiesce_before_reclaim!(bq)
    n_blocks, bytes_freed = reclaim_empty_pool_blocks!(bq)
    n_blocks > 0 && @debug "Lava: trimmed empty pool blocks" blocks=n_blocks MiB=(bytes_freed >> 20)
    return
end

"""
    trim_gpu_pool!() -> (blocks, bytes)

Hand every empty pool block back to the driver, now.

The automatic path (`pool(ctx).trim_threshold`) is rate-limited and only runs
while something is allocating, so it is the wrong tool for "I have finished a
batch of work and want the memory back" — and for measuring, where dead pool
capacity otherwise counts as live and makes a VRAM figure depend on GC timing
rather than on demand.

Runs a full collection and then `quiesce_before_reclaim!` — a flush, a wait and a
drain — **unconditionally**, because that is the request. It used to return early
unless some block already read `live_count == 0`, and that gate is a precondition
the flush and the drain establish: a buffer pinned by a recording batch, or one
already moved onto a `deferred_frees` list, holds its count until then, and
between them that is everything a graph evaluator allocates.

The measurement, on a plain KA workload of 60 dispatches never synchronised:
15 blocks, 964 MiB live, **0** blocks empty, so the old code returned `(0, 0)`
and kept all of it. Flushing first made 15 of 15 empty and handed back 960 MiB,
leaving 4 MiB. TRELLIS.2's 30-block torso was the same fault at scale — 190
blocks and 12 410 MiB resident, of which 12 750 MiB was reclaimable — and it is
why a second model in one session hit the allocator's failure path with the
memory for it free.

The automatic path keeps the cheap gate; see [`reclaimable`](@ref). This one is
the explicit "I have finished and want the memory back", so it pays the stall.
"""
function trim_gpu_pool!(ctx::VkContext = vk_context())
    GC.gc(true)
    isempty(pool(ctx).blocks) && return (0, 0)
    bq = ctx.default_bq
    quiesce_before_reclaim!(bq)
    return reclaim_empty_pool_blocks!(bq)
end

function maybe_collect(ctx::VkContext; blocking::Bool=false)
    EAGER_GC[] || return
    stats = MEMORY_STATS
    current_time = time()

    # Runs before the ratio gate below: dead pool capacity has to be returned on
    # its own terms, not only when the heap ratio says we are in trouble.
    maybe_trim_pool!(ctx)

    # Refresh device heap estimate every 10s.  The heap size itself doesn't
    # change, but on iGPUs with shared memory another process could shift what
    # we can actually use; a periodic re-probe keeps us honest if we later
    # adopt VK_EXT_memory_budget for real free-memory tracking.
    if current_time - (@atomic stats.last_updated) > 10.0
        max_size = probe_device_local_heap(ctx)
        @atomic stats.size = max_size
        @atomic stats.last_updated = current_time
    end

    size = (@atomic stats.size)
    size > 0 || return  # haven't probed yet

    live = pool(ctx).live_bytes[]
    pressure = live / size
    min_pressure = blocking ? 0.5 : 0.75
    pressure < min_pressure && return

    # GC rate budget: skip if we've already burned >5% wall time on GC.
    last_time = @atomic stats.last_time
    last_gc_time = @atomic stats.last_gc_time
    dt = current_time - last_time
    gc_rate = dt > 0 ? last_gc_time / dt : 0.0
    max_gc_rate = 0.05
    (@atomic stats.last_freed) > 0.1 * size && (max_gc_rate *= 2)
    blocking && (max_gc_rate *= 2)
    pressure > 0.9 && (max_gc_rate *= 2)
    pressure > 0.95 && (max_gc_rate *= 2)
    gc_rate > max_gc_rate && return

    @atomic stats.last_time = current_time

    pre_gc_live = live
    gc_time = Base.@elapsed GC.gc(false)
    post_gc_live = pool(ctx).live_bytes[]

    # The GC just returned sub-allocations to their pool blocks, but a block is
    # only handed back to the driver on an OOM retry.  `pool.live_bytes` tracks
    # pool *capacity*, so without this the pressure signal never falls: we keep
    # collecting, relieve nothing, and the first thing to notice the pool is
    # holding gigabytes of dead blocks is an allocation failure — or, on an iGPU
    # sharing system RAM, a driver timeout, because the pressure is on memory
    # the rest of the system also needs.
    #
    # Reclaim here, where a collection is already being paid for.  The scan for
    # empty blocks is cheap; only pay `quiesce_before_reclaim!` (which waits for
    # in-flight batches) when a block would actually be returned.
    if any(b -> b.live_count == 0, pool(ctx).blocks)
        bq = ctx.default_bq
        quiesce_before_reclaim!(bq)
        n_blocks, bytes_freed = reclaim_empty_pool_blocks!(bq)
        n_blocks > 0 && @debug "Lava: reclaimed empty pool blocks after GC" blocks=n_blocks MiB=(bytes_freed >> 20)
        post_gc_live = pool(ctx).live_bytes[]
    end

    @atomic stats.last_freed = pre_gc_live - post_gc_live
    @atomic stats.last_gc_time = 0.75 * last_gc_time + 0.25 * gc_time
    return
end

"""
Captures the exact Vulkan error that caused an allocation to fail.  Returned
from `try_vk_alloc` instead of swallowing the result code, so callers can
include the real `VkResult` and the failing op in the LavaError they throw.
"""
struct AllocFailure
    code::VK.Result
    op::Symbol          # :Buffer, :DeviceMemory, :bind_buffer_memory, :map_memory
    nbytes::Int
    mem_type_idx::Int   # -1 if the failure happened before memory-type selection
end

function format_oom_error(ctx::VkContext, fail::AllocFailure)
    io = IOBuffer()
    println(io, "Out of GPU memory.")
    req_mb = fail.nbytes ÷ (1024 * 1024)
    live_mb = pool(ctx).live_bytes[] ÷ (1024 * 1024)
    println(io, "  Vulkan returned $(fail.code) from $(fail.op) for $(fail.nbytes) bytes ($(req_mb) MiB).")
    println(io, "  Lava tracked state: $(live_mb) MiB live across $(length(pool(ctx).live_buffers)) buffers.")
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
    vk_alloc(bq::BatchQueue, nbytes; extra_usage=UInt32(0), unified=false) -> VkManagedBuffer

Allocate a GPU buffer with BDA support.  Takes the queue the allocation is
recorded against — `sweep_retired_batches!` and `drain_deferred_frees!` run
only on that queue's timeline, ready for the multi-queue refactor.

* `extra_usage` — additional `VkBufferUsageFlags` bits (e.g. INDIRECT_BUFFER,
  INDEX_BUFFER, AS_BUILD_INPUT).
* `unified=true` — pick BAR memory (device-local + host-visible + coherent,
  falling back to host-visible only) and map it.  The returned
  `VkManagedBuffer.mapped_ptr` is non-null.  Use for arg/indirect slabs or
  any buffer you want to write from the CPU without staging.

AS-scratch alignment is the caller's responsibility — see
`bda_alignment_for(ctx, scratch::Bool)`, used in `LavaArray(...; scratch=true)`.
"""
function vk_alloc(bq::BatchQueue, nbytes::Integer;
                  extra_usage::UInt32=UInt32(0), unified::Bool=false)
    # Refuse allocation on a lost device — Vulkan calls would either error or
    # (worse) succeed against a torn-down driver state and produce garbage BDAs.
    device_lost(bq.ctx::VkContext) && throw(LavaError(
        "vk_alloc",
        "Vulkan device is lost — cannot allocate new buffers",
        "Call vk_reset_device!() to reinitialize, or restart Julia."))
    if pool(bq.ctx::VkContext).track_allocs
        record_alloc_site!(bq.ctx::VkContext, Int(nbytes))
    end
    # Phase 7 P2: reclaim retired in-flight batches on THIS queue before
    # allocating.  Multi-queue: each caller only drains its own queue's
    # timeline — no implicit reach for `ctx.default_bq` here.
    sweep_retired_batches!(bq)
    drain_deferred_frees!(bq)
    drain_deferred_as_frees!(bq)
    maybe_collect(bq.ctx::VkContext)
    result = try_vk_alloc(bq, nbytes; extra_usage, unified)
    result isa VkManagedBuffer && return result
    quiesce_before_reclaim!(bq)
    # Reclaim any pool blocks that are now fully empty after the GC drained
    # their chunks back to the free list.  Without this the pool ratchets
    # up across renders — see Crown 1400×1000 hw_accel=true repro where
    # render 1 left 3.7 GiB of empty 64 MiB blocks pinning the heap.
    n_blocks, bytes_freed = reclaim_empty_pool_blocks!(bq)
    if n_blocks > 0
        @info "Lava: reclaimed empty pool blocks on OOM retry" blocks=n_blocks MiB=(bytes_freed >> 20)
    end
    result = try_vk_alloc(bq, nbytes; extra_usage, unified)
    if result isa VkManagedBuffer
        @info "Lava: GPU allocation succeeded after GC retry" bytes=nbytes
        return result
    end
    fail = result::AllocFailure
    throw(LavaError("memory allocation",
        format_oom_error(bq.ctx::VkContext, fail),
        "Free unused LavaArrays, reduce problem size, or check for memory leaks with gpu_memory_usage()."))
end

"""
    quiesce_before_reclaim!(bq)

Submit and wait for everything on `bq`, then collect and drain deferred frees.

The order is the point. `drain_deferred_frees!` releases a chunk once its
**`last_write`** semaphore has signaled, which says nothing about dispatches
that only *read* it, nor about commands already recorded into the batch still
being built — and a graph evaluator does almost nothing else: every weight and
every activation is read by the next layer without being written. Draining (or
reclaiming a block those chunks belong to) while such a batch is open destroys a
VkBuffer out from under queued work, and the damage surfaces later and
elsewhere: `sync_access!: buffer is not ALIVE` on a subsequent submit, a
batch-signal desync, or a segfault inside the driver's `vkCmdPipelineBarrier`.

Flushing first makes the invariant unconditional — after it there is no
recorded-but-unsubmitted work and everything submitted has completed — and it
only runs once an allocation has already failed, so the stall is free in the
steady state. `pool.reclaiming` guards the re-entry through `flush!`'s own
allocations.
"""

function quiesce_before_reclaim!(bq::BatchQueue)
    p = pool(bq.ctx::VkContext)
    if !p.reclaiming[] && !device_lost(bq.ctx::VkContext)
        p.reclaiming[] = true
        try
            flush!(bq, bq.device)
        finally
            p.reclaiming[] = false
        end
    end
    GC.gc(true)
    drain_deferred_frees!(bq)
    drain_deferred_as_frees!(bq)
    return nothing
end

"""Attempt GPU buffer allocation, returning an `AllocFailure` on OOM."""
function try_vk_alloc(bq::BatchQueue, nbytes::Integer;
                      extra_usage::UInt32=UInt32(0), unified::Bool=false)
    ctx = bq.ctx::VkContext
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
    # Track which Vulkan call is in-flight so the catch site can tag the
    # failure precisely.  `mem_type_idx_local` mirrors `mem_type_idx` but
    # stays defined even if the DeviceMemory call throws before assignment.
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
        # Only swallow honest OOM — every other VulkanError must propagate so
        # the caller sees the real cause (DEVICE_LOST, INVALID_OPAQUE_CAPTURE_ADDRESS,
        # MEMORY_MAP_FAILED, …).  Otherwise vk_alloc's GC-retry would loop and
        # eventually throw a misleading "Out of GPU memory".
        if e isa VK.VulkanError &&
           (e.code == VK.ERROR_OUT_OF_DEVICE_MEMORY ||
            e.code == VK.ERROR_OUT_OF_HOST_MEMORY)
            # DRAIN, then empty. The validation callback writes into a ring
            # (`ctx.validation`, per device since 49f3f17) and only
            # `drain_validation_messages!` moves entries out of it into
            # `.messages`. Emptying the drained list alone leaves this failure's
            # own messages sitting in the ring, where the next
            # `check_validation_errors!` picks them up and blames its own caller.
            #
            # Observed exactly that way: test_source_mapping.jl:699 asks for 40 GB
            # deliberately, and the error surfaced 40 lines later at :739 as a
            # `LavaError during vk_flush!` on a FOUR-ELEMENT upload. An oversized
            # allocation is the intended, handled outcome here, so its messages
            # belong to it.
            let c = bq.ctx::VkContext
                drain_validation_messages!(c)
                empty!(c.validation.messages)
            end
            return AllocFailure(e.code, op, Int(nbytes), mem_type_idx_local)
        end
        # DEVICE_LOST during alloc is a hard fault — mark + propagate so the
        # subsequent dispatcher gate fires cleanly.
        if e isa VK.VulkanError && e.code == VK.ERROR_DEVICE_LOST
            mark_device_lost!(ctx)
        end
        rethrow()
    end

    addr_info = VK.BufferDeviceAddressInfo(buf)
    address = VK.get_buffer_device_address(dev, addr_info)

    result = VkManagedBuffer(buf, memory, address, mapped_ptr, Int(nbytes), 0, nothing, nothing, BUF_STATE_ALIVE, 0, false, ctx)
    let p = pool(ctx)
        push!(p.live_buffers, result)
        Threads.atomic_add!(p.live_bytes, nbytes)
    end
    if (bq.ctx::VkContext).diag.alloc_debug
        push!((bq.ctx::VkContext).diag.alloc_log,
              (kind=:direct, addr=address, size=Int(nbytes), pool=false,
               mtype=Int(mem_type_idx), unified=unified, usage=UInt32(usage)))
    end
    return result
end

# ── Pin accounting ──
# A CommandBatch that has `pin!`ed an array holds a claim on the backing buffer
# until the batch can no longer submit (it completed, or its submit failed).
# `vk_free!` honours that claim instead of racing it, which makes "pinned by a
# live batch" and "freed" mutually exclusive by construction rather than by
# timing.
function pin_buffer!(buf::VkManagedBuffer)
    @atomic buf.pins += 1
    return nothing
end

function unpin_buffer!(buf::VkManagedBuffer)
    remaining = @atomic buf.pins -= 1
    @assert remaining >= 0 "unpin_buffer!: pins went negative ($remaining) — pin!/release_pinned_refs! are unbalanced"
    # Last batch let go, and a free was owed while we held it: pay it now, at a
    # point where no batch can reference the buffer any more.
    if remaining == 0
        _, owed = @atomicreplace buf.free_requested true => false
        owed && vk_free!(buf)
    end
    return nothing
end

"""
    vk_free!(buf::VkManagedBuffer)

Free a managed buffer's Vulkan resources.

If a command batch is currently recording or in-flight, the destruction is
deferred until after all batches complete. This prevents DEVICE_LOST from GC
finalizers freeing GPU memory that the in-flight command buffer still
references via BDA addresses.
"""
function vk_free!(buf::VkManagedBuffer)
    # A live batch has this buffer pinned, so it can still be submitted with the
    # buffer's BDA baked into an arg slab.  Do NOT touch `state` here: marking a
    # pinned buffer DEFERRED is exactly what made `sync_access!` assert
    # "buffer is not ALIVE (state=1)" at submit.  Record the debt; the last
    # `unpin_buffer!` pays it.
    #
    # The timeline check further down cannot cover this case: it keys off
    # `last_write`, which `sync_access!` only writes at submit, so a buffer
    # pinned into a still-open batch reads as never-written and looks idle.
    if (@atomic :acquire buf.pins) > 0
        @atomic :release buf.free_requested = true
        # The last unpin may have landed between those two lines and seen
        # `free_requested` still false, in which case nobody owes the free.
        # Claim it back and fall through; otherwise the unpin path owns it.
        if (@atomic :acquire buf.pins) > 0
            return
        end
        _, owed = @atomicreplace buf.free_requested true => false
        owed || return
    end

    # Atomic CAS ALIVE → DEFERRED (optimistic — we haven't yet decided we'll
    # defer; we just need to claim the buffer so no other thread races us).
    # If someone else already transitioned this buffer out of ALIVE, we bail
    # out silently — the work has already been done (or is being done).
    _, ok = @atomicreplace buf.state BUF_STATE_ALIVE => BUF_STATE_DEFERRED
    ok || return  # already DEFERRED or DEAD — nothing to do

    delete!(pool(buf.ctx::VkContext).live_buffers, buf)

    if (buf.ctx::VkContext).diag.free_debug
        bqd = buf.ctx.default_bq
        active_dbg = (bqd.active_batch !== nothing) && bqd.active_batch.recording
        push!(buf.ctx.diag.free_log,
              (addr=buf.address, size=buf.size,
               pool=buf.pool_block !== nothing,
               lw=@atomic(:acquire, buf.last_write),
               active=active_dbg))
    end

    # Defer destruction if the GPU still has work in flight that references
    # this buffer. The per-queue timeline semaphore tells us precisely:
    # buf.last_write = (bq, val) means the last dispatch recorded against
    # this buffer will signal `bq.timeline_sem` to `val` when done. If the
    # current counter is already >= val, the GPU is done and free is safe.
    # Otherwise the buffer stays in DEFERRED state on that BQ's deferred-free
    # list; it will be destroyed next time the BQ sweeps completed batches.
    # **A never-submitted buffer is not an idle buffer.** Everything below keys
    # off `last_write`, which `sync_access!` only writes at submit — so a buffer
    # the *currently recording* batch references, but which has never been
    # through a submit, reads as `nothing` here and falls through to immediate
    # destruction while an open command buffer still names it.
    #
    # The `pins > 0` path above covers the case where something took an explicit
    # reference. This covers the rest, and the rest is what a GC landing in the
    # middle of a recording finds: collect during `decode` and the queue wedges
    # with four batches holding timeline values nothing signals. Confining
    # collections to safe points made that 60/60 clean where it otherwise hung
    # within 15, which is what pointed here.
    #
    # Deferring costs one entry on a list that `drain_deferred_frees!` empties at
    # the next flush or submit; destroying early costs the device.
    #
    # **Not certainly the last of it.** After this landed the hang was seen once
    # more, under `with_dispatch_timing`, against roughly 90 clean trials across
    # every reproduction that used to fail in ten or fewer (60 probe-decodes with
    # the collector live, 8 encode+decodes with the trim forced, 6 and 10 timing
    # runs). So the dominant path is closed and the residual rate is low, but
    # either a second window exists or something rarer shares this one. If it
    # recurs, the next thing to check is whether a buffer can be reached by an
    # open batch through something `pins` does not count either.
    let bqa = (buf.ctx::VkContext).default_bq
        if (@atomic :acquire buf.last_write) === nothing &&
           bqa.active_batch !== nothing && bqa.active_batch.recording
            lock(bqa.deferred_frees_lock) do
                push!(bqa.deferred_frees, buf)
            end
            return    # stays DEFERRED; the drain transitions it to DEAD
        end
    end

    lw = @atomic :acquire buf.last_write
    if lw !== nothing
        bq = lw[1]::BatchQueue
        val = lw[2]::UInt64
        if !device_lost(bq.ctx::VkContext) && !queue_released(bq)
            # query_timeline rethrows on healthy-device failure.  We are
            # inside a finalizer-reachable path: a throw here is logged by
            # Julia's finalizer machinery rather than propagating.
            current = query_timeline(bq)
            # Defer destruction if EITHER:
            #   (a) GPU still has in-flight work that references this buffer
            #       (current < val — last submitted dispatch still pending), OR
            #   (b) the BQ has an active recording batch — even if all submits
            #       have completed, the recording batch may have captured this
            #       buffer's BDA via a runtime-pinned reference that pin_leaves!
            #       didn't catch (e.g. a closure capture in a kernel that takes
            #       the buffer indirectly). Destroying mid-recording, then
            #       letting the recording submit, races the freed BDA against
            #       the dispatch and trips a RADV GPUVM PERMISSION_FAULT.
            #       The window from finalizer-pause to next batch retirement is
            #       short, so deferring is cheap; reclaiming the buffer happens
            #       via `drain_deferred_frees!` at the next flush/submit boundary.
            active = (bq.active_batch !== nothing) && bq.active_batch.recording
            if current < val || active
                # Finalizer-thread push into the deferred list — SpinLock so
                # the main thread's drain doesn't race.
                lock(bq.deferred_frees_lock) do
                    push!(bq.deferred_frees, buf)
                end
                return     # state stays DEFERRED; drain_deferred_frees! will transition to DEAD
            end
        end
    end

    # Pre-destroy safety scan: if `buf.address` still appears in any live arg
    # slab, that's a use-after-free waiting to happen.  Log it and zero the
    # slot so the GPU faults on a null reference (BDA_POISON) instead of
    # corrupting whatever memory is mapped at the old address.
    if (buf.ctx::VkContext).diag.freed_bda_scan
        hits = scan_arg_slabs_for_bda!(buf)
        if hits > 0 && (buf.ctx::VkContext).diag.destroy_freed_bdas_throws
            throw(LavaError("vk_free!",
                "destroying buffer at 0x$(string(buf.address, base=16, pad=16)) but its BDA still appears $(hits)× in live arg slabs",
                "an unpinned reference is leaking — see ctx.diag.freed_bda_scan_log"))
        end
    end

    # One more reason to defer, independent of what the GPU is doing: THREAD.
    # `destroy_buffer!` on a pooled chunk calls `return_to_pool!`, which does a
    # plain `push!` onto the pool's free list — a Vector that `pool_alloc` pops
    # from on whichever thread is allocating. `vk_free!` runs from finalizers, so
    # that is a genuine data race, and Julia 1.12 catches it:
    #
    #   error in running finalizer: ConcurrencyViolationError("Vector has invalid
    #   state. Don't modify internal fields incorrectly, or resize without
    #   correct locks")  _growend! → push! → return_to_pool!
    #
    # after which the process segfaults. The in-flight branch above already hands
    # buffers to the owning thread under `deferred_frees_lock`; it just never
    # covered this case, and it cannot — a buffer with `last_write === nothing`
    # (allocated, never written, dropped) does not enter that branch at all.
    # Route every off-thread free the same way and `return_to_pool!` stays
    # single-threaded. State is already DEFERRED here, which is exactly what
    # `drain_deferred_frees!` expects.
    # `ctx` is typed `Any` and can be unset on a buffer that never belonged to a
    # context; such a buffer is not pooled either, so destroying it inline is
    # safe and the `isa` guard keeps this from throwing inside a finalizer.
    let c = buf.ctx
        if c isa VkContext
            bq = c.default_bq
            if Threads.threadid() != bq.owning_thread
                lock(bq.deferred_frees_lock) do
                    push!(bq.deferred_frees, buf)
                end
                return
            end
        end
    end

    destroy_buffer!(buf)
end

"""Actually destroy a buffer's Vulkan resources. Called from `vk_free!` or
`drain_deferred_frees!`.  Atomic CAS ({ALIVE, DEFERRED} → DEAD) ensures the
Vulkan destructor fires exactly once even under racing callers."""
function destroy_buffer!(buf::VkManagedBuffer)
    # Transition {ALIVE, DEFERRED} → DEAD.  First try DEFERRED (the common
    # path — vk_free! or drain always goes through DEFERRED now); fall back
    # to ALIVE for the rare direct-destroy path (e.g. dropping a buffer we
    # never pushed into deferred_frees).
    _, ok = @atomicreplace buf.state BUF_STATE_DEFERRED => BUF_STATE_DEAD
    if !ok
        _, ok = @atomicreplace buf.state BUF_STATE_ALIVE => BUF_STATE_DEAD
    end
    ok || return  # was already DEAD; idempotent

    # Pooled chunk: return to pool, don't destroy the shared VkBuffer
    if buf.pool_block !== nothing
        return_to_pool!(buf)
        return
    end

    # Check if the Vulkan device/context is still valid.
    # During Julia shutdown, finalizers fire after the device may be destroyed.
    # Finalizers cannot do context switches, so we only check simple flags here.
    ctx = buf.ctx::VkContext
    if device_lost(ctx)
        # Device is gone — just poison the handle, don't call Vulkan APIs
        buf.mapped_ptr = Ptr{UInt8}(0)
        buf.address = BDA_POISON
        Threads.atomic_sub!(pool(ctx).live_bytes, buf.size)
        buf.size = 0
        return
    end
    # PORTABILITY EXPERIMENT (not a proposed fix yet): skip the explicit unmap.
    #
    # vkFreeMemory implicitly unmaps, so this call is redundant before
    # memory.destructor() below. It is also the observed crash site on RADV, and
    # the try/catch above it could never have helped: an invalid unmap
    # (VUID-vkUnmapMemory-memory-00689 requires the memory be currently mapped)
    # is undefined behaviour, and a SIGSEGV is not a catchable Julia exception.
    buf.mapped_ptr = Ptr{UInt8}(0)
    try
        buf.buffer.destructor()
        buf.memory.destructor()
    catch ex
        # A destructor may not throw, but it can name the fault: this printed
        # fixed text, so a driver error and a bug in this file read identically.
        safe_fin_log("Lava destroy_buffer!: Vulkan destructor failed: " *
                     sprint(showerror, ex) * "\n")
    end
    buf.address = BDA_POISON
    Threads.atomic_sub!(pool(ctx).live_bytes, buf.size)
    buf.size = 0
end

# Scanner method — reachable now that VkManagedBuffer + BatchQueue are defined.
function scan_arg_slabs_for_bda!(buf::VkManagedBuffer)
    target = buf.address
    target == UInt64(0) && return 0
    hits = 0
    ctx = buf.ctx
    bq = ctx.default_bq
    for (kind, slabs) in ((:arg, bq.arg_slabs), (:indirect, bq.indirect_slabs))
        for (i, slab) in enumerate(slabs)
            # A slab whose own DataRef has been released is not something to
            # read: `getindex` throws "Attempt to use a freed reference", and
            # this runs from `vk_free!`, which finalizers reach — so the throw
            # is swallowed by Julia's finalizer machinery and the scan silently
            # stops partway. `freed` is the predicate for it; there is nothing
            # to scan in a slab whose storage is gone anyway.
            slab.buf.freed && continue
            mb = slab.buf[]::VkManagedBuffer
            mb.mapped_ptr == Ptr{UInt8}(0) && continue
            n = mb.size ÷ 8
            p = Ptr{UInt64}(mb.mapped_ptr)
            for k in 0:(n-1)
                v = unsafe_load(p, k+1)
                if v == target
                    push!(buf.ctx.diag.freed_bda_scan_log,
                          (slab=kind, idx=i, offset=k*8, freed_bda=target,
                           buf_size=buf.size))
                    # Poisoning is the point of finding it — a null BDA faults
                    # cleanly where a recycled one corrupts. Skippable so the
                    # scan can be a pure observer, which is how you tell whether
                    # the poisoning is what stopped a fault or merely the
                    # slowdown that came with it.
                    ctx.diag.freed_bda_scan_poisons && unsafe_store!(p, UInt64(0), k+1)
                    hits += 1
                end
            end
        end
    end
    return hits
end

"""
    scan_slabs_for_unknown_bdas() -> Vector{NamedTuple}

Walk every UInt64 in every live arg/indirect slab.  Report any value that
LOOKS like a BDA (in the upper half of the address space, i.e. high bit
of bit 63 set OR top 16 bits = 0xffff) but is NOT the address of any
buffer in the pool's `live_buffers` and is NOT 0.  Useful for catching stale BDAs
that pin_leaves! / pack_args_direct! missed.

Call this RIGHT BEFORE submit to catch problems before they reach the GPU.
"""
function scan_slabs_for_unknown_bdas(bq)
    bq === nothing && return NamedTuple[]
    live = Set{UInt64}()
    for buf in pool(bq.ctx::VkContext).live_buffers
        push!(live, buf.address)
    end
    pool_ranges = Tuple{UInt64,UInt64}[]
    for blk in pool(bq.ctx::VkContext).blocks
        push!(pool_ranges, (blk.base_address, blk.base_address + UInt64(blk.capacity)))
    end
    # Slabs themselves are valid arenas — the arg slab packs nested structs
    # by writing pointers to within the slab itself (a "byval-inline" arg's
    # outer arg pointer is `slab_base + inline_offset`).  Whitelist any value
    # that lands inside a known slab's address range.
    slab_ranges = Tuple{UInt64,UInt64}[]
    for slabs in (bq.arg_slabs, bq.indirect_slabs)
        for slab in slabs
            mb = slab.buf[]::VkManagedBuffer
            push!(slab_ranges, (mb.address, mb.address + UInt64(mb.size)))
        end
    end
    @inline function in_known_range(addr)
        for (lo, hi) in pool_ranges
            lo <= addr < hi && return true
        end
        for (lo, hi) in slab_ranges
            lo <= addr < hi && return true
        end
        return false
    end
    results = NamedTuple[]
    for (kind, slabs) in ((:arg, bq.arg_slabs), (:indirect, bq.indirect_slabs))
        for (i, slab) in enumerate(slabs)
            mb = slab.buf[]::VkManagedBuffer
            mb.mapped_ptr == Ptr{UInt8}(0) && continue
            n_bytes = if kind == :arg && i == bq.arg_slab_idx
                bq.arg_slab_offset
            else
                Int(mb.size)
            end
            n = n_bytes ÷ 8
            p = Ptr{UInt64}(mb.mapped_ptr)
            for k in 0:(n-1)
                v = unsafe_load(p, k+1)
                # Look only for "0xffff8…" sign-extended BDA-shaped values
                # whose 48-bit form is in the high half (bit 47 set).
                v < UInt64(0xffff800000000000) && continue
                v == typemax(UInt64) && continue  # 0xff..ff often appears in scratch
                v in live && continue
                in_known_range(v) && continue
                push!(results, (slab=kind, idx=i, offset=k*8, val=v))
            end
        end
    end
    return results
end

"""
    drain_deferred_frees!(bq::BatchQueue)

Destroy any buffers in `bq.deferred_frees` whose `last_write` timeline value
has been reached. Safe to call at any time; it only destroys buffers the
GPU is definitely done with. Called at natural sync points: after a flush
completes, and from `sweep_retired!`.
"""
function drain_deferred_frees!(bq::BatchQueue)
    isempty(bq.deferred_frees) && return
    ctx = bq.ctx::VkContext
    # Device-lost shortcut: empty under the lock so no finalizer-thread push
    # survives past our empty!().
    if device_lost(ctx)
        lock(bq.deferred_frees_lock) do
            empty!(bq.deferred_frees)
        end
        return
    end
    current = query_timeline(bq)
    # Hold the lock for the full sweep.  Finalizer-thread pushes are rare
    # (GC pauses only) and the drain body is short; SpinLock contention is
    # negligible.  destroy_buffer! may run Vulkan destructors inside the
    # critical section — that's fine under SpinLock (no yield points).
    lock(bq.deferred_frees_lock) do
        i = 1
        while i <= length(bq.deferred_frees)
            buf = bq.deferred_frees[i]::VkManagedBuffer
            lw = @atomic :acquire buf.last_write
            if lw === nothing || (lw[1]::BatchQueue === bq && lw[2]::UInt64 <= current)
                destroy_buffer!(buf)
                deleteat!(bq.deferred_frees, i)
            else
                i += 1
            end
        end
    end
    return nothing
end

"""Process deferred buffer frees after GPU is idle. Called from vk_flush!()."""
# ── Memory Pool: sub-allocate from large VkBuffer blocks ──
# Eliminates per-array VkBuffer create/destroy overhead (~30μs each).
# All sub-allocations share the parent block's VkBuffer handle.
# Free = return to free list (zero Vulkan API calls).

const POOL_BLOCK_SIZE = 64 * 1024 * 1024  # 64 MiB per block
const POOL_LARGE_THRESHOLD = POOL_BLOCK_SIZE  # Allocs above this bypass the pool
const POOL_MIN_SIZE = 16  # Minimum allocation size (Vulkan requires non-zero)
"""
Subclasses per octave above [`POOL_SUBDIV_MIN`]: a size class is
`2^p * (1 + s/8)` rather than just `2^p`, so rounding wastes at most 1/8 of a
request instead of at most half of it.

Measured on SAM 2's encoder, which is the workload that made this worth doing.
One encode asks the pool for 1 649 MiB across 787 allocations; with plain
powers of two it was handed **2 781 MiB — 59.3% efficient**, and the missing
1 132 MiB is most of the gap between this allocator's footprint and PyTorch's.
Eight subclasses put that at ~94% for 5x more free lists, which are empty
vectors and cost nothing.

Below `POOL_SUBDIV_MIN` the subdivision is skipped: an octave there is a few
kilobytes, the absolute waste is irrelevant, and the step would fall under the
16-byte alignment every chunk relies on (`block.bump` advances by the class
size, so the class size *is* the alignment guarantee).
"""
const POOL_SUBDIV = 8
const POOL_SUBDIV_MIN = 4096          # 2^12; step at that octave is 512 B
const POOL_SUBDIV_MINEXP = 12
const POOL_POW2_CLASSES = 9           # 16 B => 1 … 4096 B => 9
# Up to 2^27 with 8 subclasses each, plus the plain power-of-two head.
const POOL_NUM_SIZE_CLASSES = POOL_POW2_CLASSES + 16 * POOL_SUBDIV

# Debug-only: force every LavaArray onto its own VkBuffer (one vkGetBufferDeviceAddress
# per array). GPU-AV's BDA OOB validation tracks ranges per VkBuffer, so with the pool
# on it cannot see sub-pool overruns; with this flag on, each LavaArray's bounds are
# checked individually. Slow — leave off in production.

# Defaults live here, next to the pool they configure, rather than in eleven
# module-level `Ref`s. `2 GiB` soft cap, trim above 1 GiB and no more than every
# 5 s, a full GC no more than every 30 s.
DevicePool() = DevicePool(PoolBlock[],
                          [VkManagedBuffer[] for _ in 1:POOL_NUM_SIZE_CLASSES],
                          false, false, 2 * 1024^3, 1024 * 1024 * 1024,
                          5.0, 30.0, 0.02, 0.5, false,
                          0.0, 0.0, 0.0, 0.0, 0.0,
                          Threads.Atomic{Int}(0), Set{VkManagedBuffer}(),
                          Threads.Atomic{Int}(0), Threads.Atomic{Int}(0),
                          Threads.Atomic{Int}(0), Threads.Atomic{Int}(0),
                          Threads.Atomic{Bool}(false))

"""
    pool(ctx) -> DevicePool

This device's memory, created on first allocation.

Never call this from a finalizer: it can insert. The free path reaches its pool
through `buf.pool_block.pool` instead, which cannot allocate and cannot miss.
"""
@inline pool(ctx::VkContext) = ctx.caches.pool

"""
    destroy_pool!(ctx)

Destroy this device's pool blocks. Called by `vk_reset_device!` on the context it
is retiring — which is where the old context is actually in scope.

It used to be a `RESET_CALLBACKS` entry walking a global `POOLS` dict, because
the pool did not belong to anything. Destructor failures are expected and
logged, not thrown: by the time this runs the device is marked lost or already
torn down, so the driver may have released the handles itself.
"""
function destroy_pool!(ctx::VkContext)
    for block in ctx.caches.pool.blocks
        try
            block.buffer.destructor()
            block.memory.destructor()
        catch ex
            # A destructor may not throw, but it can name the fault: this printed
            # fixed text, so a driver error and a bug in this file read alike.
            safe_fin_log("Lava pool reset: destructor failed (ok during vk_reset_device!): " *
                         sprint(showerror, ex) * "\n")
        end
    end
    empty!(ctx.caches.pool.blocks)
    foreach(empty!, ctx.caches.pool.free_lists)
    return nothing
end

"""
    size_class(nbytes) -> (idx, bytes)

The free list a request of `nbytes` belongs to, and the size actually handed
out. See [`POOL_SUBDIV`] for why this is not simply the next power of two.

The two results must stay consistent in both directions: `pool_alloc` looks the
class up from the *request*, `return_to_pool!` looks it up again from the size
that was handed out, and a chunk that came back to the wrong list would be
handed to a caller who asked for more than it holds. So
`size_class(size_class(n).bytes) == size_class(n)` for every `n`, which
`test_pool_sizeclass.jl` checks exhaustively over the octave boundaries.
"""
@inline function size_class(nbytes::Int)
    n = max(nbytes, POOL_MIN_SIZE)
    if n <= POOL_SUBDIV_MIN
        r = nextpow(2, n)
        return (trailing_zeros(r) - 3, r)   # 16=2^4 → 1, 32=2^5 → 2, …
    end
    p = 8 * sizeof(Int) - 1 - leading_zeros(n)   # floor(log2 n); ≥ 12 here
    base = 1 << p
    step = base >> 3                            # 2^p / POOL_SUBDIV
    sub = cld(n - base, step)                   # 1…8 (n > base in this branch)
    if sub == POOL_SUBDIV                       # the top subclass IS the next octave
        p += 1; sub = 0; base = 1 << p
        r = base
    else
        r = base + sub * step
    end
    idx = POOL_POW2_CLASSES + (p - POOL_SUBDIV_MINEXP) * POOL_SUBDIV + sub + 1
    return (idx, r)
end

"""Size class index for a given byte size."""
@inline size_class_idx(nbytes::Int) = size_class(nbytes)[1]

"""Rounded-up allocation size for a given byte count."""
@inline size_class_bytes(nbytes::Int) = size_class(nbytes)[2]

"""
    pool(ctx).accounting :: Bool

Record what the pool is asked for against what it hands out. Off by default;
the counters below are only meaningful while it is on.

Power-of-two size classes waste up to 2x per chunk, and the pool is the reason
SAM 2 holds far more VRAM than its live tensors. Whether that rounding is the
cause or a red herring is a measurement, not a guess — hence these.
"""

"""Requested / handed-out bytes since `reset_pool_accounting!`, and the ratio."""
function pool_accounting(ctx::VkContext = vk_context())
    p = pool(ctx)
    req = p.requested[]; rnd = p.rounded[]
    return (; nalloc = p.nalloc[], requested = req, rounded = rnd,
            efficiency = rnd == 0 ? 1.0 : req / rnd)
end

function reset_pool_accounting!(ctx::VkContext = vk_context())
    p = pool(ctx)
    p.requested[] = 0; p.rounded[] = 0; p.nalloc[] = 0
    return
end

"""
    gpu_live_bytes(ctx = vk_context()) -> Int
    live_buffer_count(ctx = vk_context()) -> Int

What this device currently holds: bytes the driver has handed Lava, and how many
buffers they are spread over. Both read `ctx`'s own pool — they were module-level
and so answered for every device at once.
"""
gpu_live_bytes(ctx::VkContext = vk_context()) = pool(ctx).live_bytes[]
live_buffer_count(ctx::VkContext = vk_context()) = length(pool(ctx).live_buffers)

# A block-count watermark used to live here, reclaiming on the allocation path
# once the pool passed 48 blocks. It is gone: [`maybe_trim_pool!`] does the same
# job — return dead capacity to the driver — and does it better, because it is
# limited by elapsed time rather than by allocation count and it only pays
# `quiesce_before_reclaim!` when a block is actually empty. The watermark
# version quiesced whether or not anything came back, which cost **2.83x** on
# SAM 2's encoder (1 046 ms against 370) before a back-off was bolted on.
#
# Two mechanisms, two jobs, and they are not interchangeable:
#
#   * `soft_cap` stops the pool GROWING. On the allocation path, cheap,
#     no queue drain — the memory comes back as free-list chunks the caller
#     takes immediately.
#   * `trim_threshold` RELEASES capacity that is already dead, back to
#     the driver. Periodic, expensive, and the only thing that helps when the
#     pressure is on memory the rest of the machine needs.

"""
    pool(ctx).soft_cap

Pool footprint in bytes past which `pool_alloc` collects *before* committing
another block. `0` disables it.

This is the cheap half of keeping the pool small, and it is deliberately not
the same mechanism as [`maybe_trim_pool!`]:

  * here we run a **GC and reuse** — no queue drain, no Vulkan call, and the
    memory comes back as free-list chunks the caller immediately takes, so this
    can afford to run on the allocation path;
  * there we **destroy blocks** and hand the VkDeviceMemory back, which needs
    `quiesce_before_reclaim!` and therefore stalls the GPU — so it is limited by
    elapsed time and skipped entirely unless a block is actually empty.

Preventing growth and releasing dead capacity are different problems: this one
keeps a steady workload's footprint flat, that one is what stops a finished
workload from holding memory the rest of the machine needs.

Without a trigger here the only backstop is `maybe_collect`, whose pressure
threshold is a fraction of the *device heap* — 0.75 x 20 GiB on this card. That
is a fine OOM guard and a terrible footprint policy: SAM 2's encoder ran to a
stable **16 136 MiB across 200 blocks**, none of it needed, simply because
nothing asked the GC a question until 15 GiB. The same loop with this cap holds
its working set instead. A GPU shared with an editor and a REPL is the normal
case here, not a dedicated one.

2 GiB is where SAM 2's encoder stops caring, measured (blocks / VRAM / p50 / min):

    3.00 GiB   48   4615 MB   340.2   332.2
    2.00 GiB   32   3541 MB   339.8   328.8     <- same speed, 1 GiB less
    1.50 GiB   30   3407 MB   374.0   347.9     <- 10% slower for 134 MB
    1.25 GiB   30   3407 MB   375.4   356.3

The graph's own live set is 26 blocks, so a cap below ~30 leaves nothing to
collect and every allocation past it pays for a collection and grows anyway.
"""

# Rate limits. The incremental collection is cheap enough to run between graph
# steps; the full one is not, and only it sweeps the old generation that
# long-lived activations reach.

"""Collections run by the soft cap, and the seconds they cost."""
pool_gc_stats(ctx::VkContext = vk_context()) =
    (; count = pool(ctx).gc_count[], seconds = pool(ctx).gc_seconds)

# `ctx` was missing from this signature while the counter was global, so the body
# referenced an undefined name and the function could only ever have thrown.
function reset_pool_gc_stats!(ctx::VkContext = vk_context())
    p = pool(ctx)
    p.gc_count[] = 0; p.gc_seconds = 0.0
    return
end

"""
    collect_for_pool!(bq) -> Bool

Try to turn dead LavaArrays back into free-list chunks. Returns whether a
collection actually ran, so the caller knows whether retrying is worthwhile.

`drain_deferred_frees!` after each collection is what makes this work at all: a
buffer freed while the GPU still referenced it went to the deferred list rather
than the pool, and until it is drained the memory is dead to everyone.
"""
function collect_for_pool!(bq::BatchQueue)
    p = pool(bq.ctx::VkContext)
    now = time()
    now - p.gc_last < p.gc_mingap && return false
    t0 = time_ns()
    GC.gc(false)
    drain_deferred_frees!(bq)
    if now - p.gc_full_last >= p.gc_full_mingap
        GC.gc(true)
        drain_deferred_frees!(bq)
        p.gc_full_last = now
    end
    p.gc_last = time()
    p.gc_seconds += (time_ns() - t0) / 1e9
    Threads.atomic_add!(p.gc_count, 1)
    return true
end

"""Allocate a new pool block (one large VkBuffer)."""
function alloc_pool_block(bq::BatchQueue)
    buf_result = try_vk_alloc(bq, POOL_BLOCK_SIZE)
    if buf_result isa AllocFailure
        quiesce_before_reclaim!(bq)
        # Same fallback as vk_alloc: reclaim any pool blocks the GC just
        # emptied before deciding the OOM is real.
        n_blocks, bytes_freed = reclaim_empty_pool_blocks!(bq)
        if n_blocks > 0
            @info "Lava: reclaimed empty pool blocks on pool-block OOM retry" blocks=n_blocks MiB=(bytes_freed >> 20)
        end
        buf_result = try_vk_alloc(bq, POOL_BLOCK_SIZE)
        if buf_result isa AllocFailure
            throw(LavaError("pool block allocation",
                "Cannot allocate $(POOL_BLOCK_SIZE ÷ 1024 ÷ 1024) MiB pool block.\n" *
                format_oom_error(bq.ctx::VkContext, buf_result),
                "Free unused LavaArrays, reduce problem size, or check for memory leaks with gpu_memory_usage()."))
        end
    end
    # Extract Vulkan handles from the VkManagedBuffer, then remove it from `live_buffers`
    # (the pool block manages its own lifetime, not the per-chunk tracking)
    p = pool(bq.ctx::VkContext)
    block = PoolBlock(buf_result.buffer, buf_result.memory, buf_result.address,
                      POOL_BLOCK_SIZE, 0, 0, p)
    delete!(p.live_buffers, buf_result)
    # Don't subtract from `live_bytes` — the block IS live memory. Individual
    # chunks don't add to it since the block already accounts for them.
    push!(p.blocks, block)
    if (bq.ctx::VkContext).diag.alloc_debug
        push!((bq.ctx::VkContext).diag.alloc_log, (kind=:pool_block, addr=buf_result.address,
                                size=POOL_BLOCK_SIZE, pool=true))
    end
    return block
end

# Diagnostic: track allocation call sites during recording.
# Set `pool(ctx).track_allocs = true` to record stack traces of every allocation while the
# active batch is recording. Used to find per-frame allocations leaking into the
# render loop. Reads are merged into `ctx.diag.alloc_trace`; query via `dump_alloc_trace()`.
# site => (count, bytes). Bytes as well as counts because the two rank call
# sites completely differently — a hot site allocating 4 KiB matters far less
# than one full-size tensor per layer.

function record_alloc_site!(ctx::VkContext, nbytes::Int)
    bt = stacktrace(backtrace())
    # Capture full stack as a single key (truncated to first 6 user frames)
    user_frames = String[]
    for f in bt
        s = string(f.file)
        # Skip Lava infra and Base
        if occursin("memory.jl", s) || occursin("lavaarray.jl", s) ||
           occursin("launch.jl", s) || occursin("ka_backend.jl", s) ||
           occursin("KernelAbstractions", s) || occursin("Base.jl", s) ||
           occursin("/Base/", s) || occursin("Adapt/src", s) ||
           occursin("gpuarrays", s) || occursin("dict.jl", s) ||
           occursin("./", s) && length(s) < 30
            continue
        end
        push!(user_frames, "$(basename(s)):$(f.line) ($(f.func))")
        length(user_frames) >= 4 && break
    end
    site = Symbol(isempty(user_frames) ? "unknown" : join(user_frames, " <- "))
    d = ctx.diag
    lock(d.alloc_trace_lock) do
        c, b = get(d.alloc_trace, site, (0, 0))
        d.alloc_trace[site] = (c + 1, b + nbytes)
    end
end

function dump_alloc_trace(ctx::VkContext = vk_context())
    d = ctx.diag
    lock(d.alloc_trace_lock) do
        sorted = sort(collect(d.alloc_trace), by = x -> x[2][2], rev = true)
        tot = sum(x -> x[2][2], sorted; init = 0)
        for (site, (count, bytes)) in sorted
            println("  ", lpad(string(round(bytes / 1e6, digits = 1)), 8), " MB  ",
                    lpad(string(count), 4), " ×  $site")
        end
        println("  ", lpad(string(round(tot / 1e6, digits = 1)), 8), " MB  total")
    end
end

function clear_alloc_trace!(ctx::VkContext = vk_context())
    d = ctx.diag
    lock(d.alloc_trace_lock) do
        empty!(d.alloc_trace)
    end
end

"""
    pool_alloc(bq::BatchQueue, nbytes; extra_usage=UInt32(0)) -> VkManagedBuffer

Allocate GPU memory on `bq`.  Small device-local allocations come from the
sub-allocation pool (zero Vulkan API calls); large allocations or those
with non-default usage flags bypass the pool via `vk_alloc`.
"""
function pool_alloc(bq::BatchQueue, nbytes::Integer; extra_usage::UInt32=UInt32(0))
    ctx = bq.ctx::VkContext
    # Even pure free-list reuse must refuse a dead device — the pool blocks
    # belong to the old (broken) ctx and would hand back garbage BDAs.
    device_lost(ctx) && throw(LavaError(
        "pool_alloc",
        "Vulkan device is lost — cannot allocate new buffers",
        "Call vk_reset_device!() to reinitialize, or restart Julia."))
    nbytes = max(Int(nbytes), POOL_MIN_SIZE)
    p = pool(ctx)
    if p.track_allocs
        record_alloc_site!(bq.ctx::VkContext, nbytes)
    end
    sweep_retired_batches!(bq)
    drain_deferred_frees!(bq)
    # Pressure-driven GC, identical hook to `vk_alloc`.  Without this the pool
    # fast path silently grows VRAM until the bump pointers exhaust every block,
    # then forces a `GC.gc` from inside the alloc — exactly the 2-5 ms spike the
    # AK benchmarks regressed on.
    maybe_collect(ctx)

    if p.disabled || nbytes > POOL_LARGE_THRESHOLD || extra_usage != UInt32(0)
        return vk_alloc(bq, nbytes; extra_usage)
    end

    # No clamp: every class up to `POOL_LARGE_THRESHOLD` has a list, and larger
    # requests took the `vk_alloc` branch above. Clamping would put a chunk in a
    # list whose other members are a different size, and the next caller would be
    # handed a buffer smaller than it asked for — a BoundsError is the better
    # failure.
    idx, alloc_size = size_class(nbytes)
    if p.accounting
        Threads.atomic_add!(p.requested, nbytes)
        Threads.atomic_add!(p.rounded, alloc_size)
        Threads.atomic_add!(p.nalloc, 1)
    end

    # Try the free list first. If empty, run GC (which drains LavaArray
    # finalizers → `return_to_pool!`) and re-check, so we reuse whatever
    # just got freed before burning a new 64-MiB pool block. Without this
    # retry the pool grows monotonically on heavy sim workloads — GC fires
    # sporadically, finalizers enqueue async, and each allocation that
    # races past GC cuts a new block even when thousands of matching
    # buffers are about to return to the free list.
    @inline function try_reuse_or_bump()
        fl = pool(ctx).free_lists[idx]
        if !isempty(fl)
            buf = pop!(fl)
            block = buf.pool_block::PoolBlock
            block.live_count += 1
            buf.address = block.base_address + UInt64(buf.pool_offset)
            buf.size = alloc_size
            buf.ctx = ctx
            @atomic :release buf.state = BUF_STATE_ALIVE
            return buf
        end
        for block in pool(ctx).blocks
            if block.bump + alloc_size <= block.capacity
                byte_offset = block.bump
                block.bump += alloc_size
                block.live_count += 1
                return VkManagedBuffer(
                    block.buffer, block.memory,
                    block.base_address + UInt64(byte_offset),
                    Ptr{UInt8}(0),
                    alloc_size,
                    byte_offset, block, nothing, BUF_STATE_ALIVE, 0, false, ctx)
            end
        end
        return nothing
    end

    buf = try_reuse_or_bump()
    buf === nothing || return buf
    # Free list empty + every block full → cut a new 64-MiB block, unless the
    # pool is already past its soft cap, in which case ask the GC first: past
    # that point the memory this request needs is far more likely to be dead and
    # uncollected than genuinely in use.
    #
    # An unconditional `GC.gc(false)` + `GC.gc(true)` chain used to live here and
    # was removed for showing up as 2-5 ms spikes in tight loops (AK benchmarks).
    # Both rate limits and the cap are there so this is not that: under the cap
    # this path is exactly as it was, and above it a collection runs at most
    # every `POOL_GC_MINGAP` seconds.
    if p.soft_cap > 0 && length(p.blocks) * POOL_BLOCK_SIZE >= p.soft_cap &&
       collect_for_pool!(bq)
        buf = try_reuse_or_bump()
        buf === nothing || return buf
    end
    block = alloc_pool_block(bq)
    byte_offset = block.bump
    block.bump += alloc_size
    block.live_count += 1
    return VkManagedBuffer(
        block.buffer, block.memory,
        block.base_address + UInt64(byte_offset),
        Ptr{UInt8}(0),
        alloc_size,
        byte_offset, block, nothing, BUF_STATE_ALIVE, 0, false, ctx)
end

"""
    reclaim_empty_pool_blocks!(bq::BatchQueue) -> (n_blocks::Int, bytes::Int)

Free every pool block whose `live_count` is 0: drop the block's chunks from
this device's free lists, destroy its `VkBuffer` + `VkDeviceMemory`, and remove
it from this device's block list.  Returns `(n_blocks_reclaimed, bytes_reclaimed)`.

Called from `vk_alloc` / `alloc_pool_block` only on the OOM retry path, so
steady-state allocations don't pay the scan cost.  Not finalizer-safe —
runs only on the main allocator path.

Callers must have run `quiesce_before_reclaim!` first — see there for why.
"""
function reclaim_empty_pool_blocks!(bq::BatchQueue)
    p = pool(bq.ctx::VkContext)
    isempty(p.blocks) && return (0, 0)
    ctx = bq.ctx::VkContext
    empty_blocks = Set{PoolBlock}()
    kept = PoolBlock[]
    bytes_reclaimed = 0
    for block in p.blocks
        if block.live_count == 0
            push!(empty_blocks, block)
            bytes_reclaimed += block.capacity
        else
            push!(kept, block)
        end
    end
    isempty(empty_blocks) && return (0, 0)
    # Drop free-list chunks that belong to any reclaimed block.  Each chunk
    # is a VkManagedBuffer whose `pool_block` field identifies its host.
    for fl in p.free_lists
        filter!(buf -> begin
            pb = buf.pool_block
            pb === nothing || !(pb in empty_blocks)
        end, fl)
    end
    # Swap kept list into the pool so `pool_alloc` no longer sees the
    # reclaimed blocks (must precede destructor calls — a concurrent bump
    # against a freed handle would corrupt the driver).
    empty!(p.blocks)
    append!(p.blocks, kept)
    if !device_lost(ctx)
        for block in empty_blocks
            try
                block.buffer.destructor()
                block.memory.destructor()
            catch ex
                # Match destroy_buffer!: don't propagate from destructors,
                # but log loudly AND name the fault.
                safe_fin_log("Lava reclaim_empty_pool_blocks!: Vulkan destructor failed: " *
                             sprint(showerror, ex) * "\n")
            end
        end
    end
    # Pool blocks ARE counted in `live_bytes` (see alloc_pool_block —
    # we intentionally don't subtract per-chunk because the block is the
    # real live memory).  Subtract the reclaimed capacity here.
    Threads.atomic_sub!(pool(bq.ctx::VkContext).live_bytes, bytes_reclaimed)
    return (length(empty_blocks), bytes_reclaimed)
end

"""Return a pooled chunk to the free list. Keeps VkManagedBuffer object for reuse."""
function return_to_pool!(buf::VkManagedBuffer)
    block = buf.pool_block
    block === nothing && return
    alloc_size = buf.size
    alloc_size == 0 && return
    # `size_class` is idempotent on its own output, so this lands in exactly the
    # list `pool_alloc` took the chunk from — see there about not clamping.
    idx = size_class_idx(alloc_size)
    block.live_count -= 1
    # Poison address to detect use-after-free, but keep pool_block + pool_offset
    # so pool_alloc can restore the address from block.base_address + pool_offset
    buf.address = BDA_POISON
    buf.mapped_ptr = Ptr{UInt8}(0)
    buf.size = 0
    @atomic :release buf.last_write = nothing  # clear scheduling state so a re-use starts fresh
    # Keep buf.pool_block and buf.pool_offset intact for reuse
    # `block.pool`, not `pool(buf.ctx)`: this runs from a finalizer, where a
    # `get!` could allocate and a missing key would throw. The block has carried
    # its pool since it was created, so this cannot fail.
    push!((block.pool::DevicePool).free_lists[idx], buf)
end

# ── Staging buffer for CPU↔GPU transfers ──

"""
    get_staging(bq::BatchQueue, nbytes::Integer)
        -> (buf::VK.Buffer, memory::VK.DeviceMemory, mapped_ptr::Ptr, size::Int)

Return `bq`'s staging buffer, growing it to at least `nbytes` if needed.
Backed by a `VkManagedBuffer` whose lifetime follows the normal
timeline-gated free path when re-allocated.
"""
function get_staging(bq::BatchQueue, nbytes::Integer)
    existing = bq.staging
    if existing !== nothing && (existing::VkManagedBuffer).size >= nbytes
        buf = existing::VkManagedBuffer
        return (buf.buffer, buf.memory, buf.mapped_ptr, buf.size)
    end

    # Release the old staging buffer through vk_free! — that routes through
    # the per-BQ deferred queue so in-flight transfers complete first.
    if existing !== nothing
        vk_free!(existing::VkManagedBuffer)
        bq.staging = nothing
    end

    managed = host_buffer(bq, nbytes)
    bq.staging = managed
    return (managed.buffer, managed.memory, managed.mapped_ptr, managed.size)
end

"""
    host_buffer(bq, nbytes) -> VkManagedBuffer

A host-visible, permanently mapped buffer the GPU can copy out of.

The GPU cannot read a Julia `Vector`, so every host-to-device transfer starts
with a memcpy into one of these. Who owns it, how it is sliced and when a slice
may be reused are all questions for the caller; this only allocates one.
"""
function host_buffer(bq::BatchQueue, nbytes::Integer)
    ctx = bq.ctx::VkContext
    dev = bq.device
    alloc_size = max(65536, nextpow(2, nbytes))
    vkbuf = VK.Buffer(
        dev, alloc_size,
        VK.BUFFER_USAGE_TRANSFER_SRC_BIT |
        VK.BUFFER_USAGE_TRANSFER_DST_BIT,
        VK.SHARING_MODE_EXCLUSIVE,
        UInt32[],
    )
    mem_reqs = VK.get_buffer_memory_requirements(dev, vkbuf)
    # Prefer HOST_CACHED: downloads memcpy FROM this buffer, and CPU reads of
    # write-combined (uncached) host-visible memory run ~70 MB/s vs GB/s cached.
    mem_type_idx = something(
        find_memory_type_optional(ctx, mem_reqs.memory_type_bits,
                                  VK.MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                  VK.MEMORY_PROPERTY_HOST_COHERENT_BIT |
                                  VK.MEMORY_PROPERTY_HOST_CACHED_BIT),
        find_memory_type(ctx, mem_reqs.memory_type_bits,
                         VK.MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                         VK.MEMORY_PROPERTY_HOST_COHERENT_BIT))
    memory = VK.DeviceMemory(dev, mem_reqs.size, mem_type_idx)
    throw_if_error(ctx, "vkBindBufferMemory",
        VK.bind_buffer_memory(dev, vkbuf, memory, 0))
    mapped_ptr = Ptr{UInt8}(throw_if_error(ctx, "vkMapMemory",
        VK.map_memory(dev, memory, 0, alloc_size)))

    managed = VkManagedBuffer(vkbuf, memory, UInt64(0),   # no BDA needed for staging
                              mapped_ptr, Int(alloc_size),
                              0, nothing, nothing, BUF_STATE_ALIVE, 0, false, ctx)
    let p = pool(ctx)
        push!(p.live_buffers, managed)
        Threads.atomic_add!(p.live_bytes, managed.size)
    end
    managed
end

"""
    copy_buffer!(direction, managed, host_ptr, nbytes; offset=0)

Unified CPU↔GPU buffer copy — single entry point for every host-visible
transfer path.  `direction` is `:upload` (host → GPU) or `:download` (GPU → host).

Fast path (BAR memory): direct memcpy via `managed.mapped_ptr` after
`wait_for_write(managed)` drains any in-flight writer.  No staging, no batch.

Slow path (device-local): allocate staging, record `cmd_copy_buffer!` into
the active batch, flush, and memcpy between staging and host.  For downloads
we route the copy through the queue that last wrote `managed` so the copy
piggy-backs on producer dispatches (one fence wait instead of two).  For
uploads we use the buffer's ctx default queue.

The caller is responsible for keeping `host_ptr`'s backing array alive
(typically via `GC.@preserve`).
"""
function copy_buffer!(direction::Symbol, managed::VkManagedBuffer,
                     host_ptr::Ptr{UInt8}, nbytes::Integer; offset::Integer=0)
    nbytes == 0 && return
    @assert direction === :upload || direction === :download  "direction must be :upload or :download"
    buf_offset = managed.pool_offset + Int(offset)

    # BAR fast-path: host-visible mapped memory — direct memcpy, no staging.
    if managed.mapped_ptr != Ptr{UInt8}(0)
        # wait_for_write reads buf.last_write, which is only populated by
        # sync_access! at submit time — so a dispatch that is recorded but
        # not yet submitted is invisible to it. Flush the active batch of
        # the buffer's ctx first so any pending writer actually lands before
        # we memcpy. (wait_for_write still matters for cross-queue writers
        # and already-in-flight batches.)
        bq = (managed.ctx::VkContext).default_bq
        if bq.active_batch !== nothing
            flush!(bq, bq.device)
        end
        wait_for_write(managed)
        if direction === :upload
            unsafe_copyto!(managed.mapped_ptr + offset, host_ptr, nbytes)
        else
            unsafe_copyto!(host_ptr, managed.mapped_ptr + offset, nbytes)
        end
        return
    end

    # Device-local: route through staging + batched copy.  Pinning `managed`
    # triggers sync_access!'s cross-queue semaphore wait whenever the buffer
    # was last written on a different BatchQueue.
    bq = if direction === :upload
        (managed.ctx::VkContext).default_bq
    else
        lw = @atomic :acquire managed.last_write
        lw !== nothing ? (lw[1]::BatchQueue) : (managed.ctx::VkContext).default_bq
    end
    staging_buf, _, mapped_ptr, _ = get_staging(bq, nbytes)
    if direction === :upload
        unsafe_copyto!(Ptr{UInt8}(mapped_ptr), host_ptr, nbytes)
        cmd_copy_buffer!(bq, staging_buf, managed, nbytes; dst_off=buf_offset)
        flush!(bq, bq.device)
    else
        cmd_copy_buffer!(bq, managed, staging_buf, nbytes; src_off=buf_offset)
        flush!(bq, bq.device)
        unsafe_copyto!(host_ptr, Ptr{UInt8}(mapped_ptr), nbytes)
    end
    return
end

"""Upload host data to a device-local buffer (BAR direct or via staging)."""
function upload!(dst::VkManagedBuffer, host_data::Vector{UInt8}; offset::Int=0)
    GC.@preserve host_data copy_buffer!(:upload, dst, pointer(host_data), length(host_data); offset)
end

"""Download data from a device-local buffer to a host byte vector."""
function download!(host_data::Vector{UInt8}, src::VkManagedBuffer; offset::Int=0)
    GC.@preserve host_data copy_buffer!(:download, src, pointer(host_data), length(host_data); offset)
end

"""Upload a typed array to a device-local buffer."""
function upload_typed!(dst::VkManagedBuffer, data::AbstractVector{T}; offset::Int=0) where T
    bytes = Vector{UInt8}(reinterpret(UInt8, vec(collect(data))))
    upload!(dst, bytes; offset)
end

"""Download data from a device-local buffer into a typed array."""
function download_typed!(data::AbstractVector{T}, src::VkManagedBuffer; offset::Int=0) where T
    GC.@preserve data copy_buffer!(:download, src, Ptr{UInt8}(pointer(data)),
                                    length(data) * sizeof(T); offset)
end

"""
    bda_alignment_for(ctx::VkContext, scratch::Bool) -> UInt64

Required BDA alignment.  AS-build scratch buffers must be aligned to
`minAccelerationStructureScratchOffsetAlignment` (cached on ctx).
Everything else defaults to 1 (Vulkan already guarantees the per-usage
minimum alignment via `vkGetBufferDeviceAddress`).
"""
@inline function bda_alignment_for(ctx::VkContext, scratch::Bool)
    return scratch ? ctx.as_scratch_align : UInt64(1)
end

# VkMappedBuffer / VkIndirectBuffer / alloc_indirect_slab / vk_alloc_mapped /
# vk_alloc_unified: deleted.  Every GPU buffer allocation goes through
# `vk_alloc(bq, nbytes; extra_usage, unified)` now.  Slab pools (arg buffer
# + indirect dispatch) live on top of `LavaArray`s declared in
# `array/lavaarray.jl` and allocated via `runtime/launch.jl`.

# Indirect buffer slab size — 256 KB of UInt32 storage, enough for ~1000
# 12-byte indirect dispatches.
const INDIRECT_SLAB_SIZE = 256 * 1024

function find_memory_type_optional(ctx::VkContext, type_bits::UInt32, required_flags)
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

function find_memory_type(ctx::VkContext, type_bits::UInt32, required_flags)
    mem_props = ctx.memory_properties

    for i in 0:(length(mem_props.memory_types) - 1)
        if (type_bits & (UInt32(1) << i)) != 0
            mt = mem_props.memory_types[i + 1]
            if (mt.property_flags & required_flags) == required_flags
                return UInt32(i)
            end
        end
    end
    throw(LavaError(
        "memory allocation",
        "No suitable memory type found for flags $required_flags",
        "Check GPU memory capabilities"))
end
