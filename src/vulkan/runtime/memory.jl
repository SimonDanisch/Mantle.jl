# Vulkan memory management for Lava.jl
#
# Device-local buffers with BDA (Buffer Device Address) for kernel arguments.
# Staging buffer for CPU↔GPU transfers.


# Debug instrumentation for iter6 cross-scene cascade investigation.
# All off by default — opt-in via the *_ENABLED Refs.

# Per-finalizer destruction trace

# Per-allocation trace

# When enabled, vk_free! scans every live VulkanBatchQueue's arg/indirect slabs for
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
    scan_unified_for_bda!(buf) -> Int

Scan every block of the unified arena — the argument blocks and indirect commands
a recording reads — for any UInt64 word matching `buf.address`. Named for what
it scans: the arg slabs it was written against are gone, and the unified
arena's blocks are where those bytes live now.  For each hit,
append a record to `diag.freed_bda_scan_log` and overwrite the slot with 0 so the
GPU faults cleanly on a null reference instead of touching the freed memory.
Returns the number of hits found.

Defined AFTER VkManagedBuffer + VulkanBatchQueue (forward-call from vk_free!).
Cost is ~`(block_bytes / 8)` UInt64 reads per block — for 4 MiB blocks that is
~512 K reads, fast enough for debug.
"""
function scan_unified_for_bda! end

# Buffer lifecycle states (atomic CAS transitions).  Every VkManagedBuffer
# starts ALIVE.  `unsafe_free!` transitions ALIVE → DEFERRED (queued on a
# core's retired list) or ALIVE → DEAD (destroyed immediately).
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

# No reset callback: every counter above is a `MemoryPolicy` field, and the pool
# is a `VkContext` field, so all of it dies with the context `reset_device!`
# retires. Staging, indirect and arg slabs are per-`VulkanBatchQueue` and go the same
# way. `reset_memory_stats!` is the one piece left, and `reset_device!` calls
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
# What this device holds is `gpu_live_bytes(ctx)`, and every gate here reads
# that rather than `mempolicy(ctx).live_bytes[]`. The two used to be the same
# number; they are not any more. `live_bytes` counts only buffers that own their
# memory — staging, mapped, unusual usage flags — while the suballocated bytes
# are `Mantle.reserved(spans(ctx))`, and the total is the sum.
#
# Reading the counter alone is a live bug, not a nuance: it makes every threshold
# here compare device pressure against a few megabytes of staging and conclude
# there is nothing to do. `maybe_trim_pool!` did exactly that and declined to
# trim a pool holding 1.3 GB.

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

Two things make a block empty. It may already be, or the release may be waiting
on core's retired list: a sub-allocation's finalizer moves the buffer
ALIVE → DEFERRED and retires it, and the `live` entry it holds is not given back
until `reclaim!` runs inside `quiesce_before_reclaim!` — after the gate. So
`any(b -> isempty(b.live), blocks)` on its own is a precondition the trim
establishes, and gating on it alone means declining to look.

**This deliberately does not see the third case**, which is a buffer a recording
that has not been submitted still holds: `reclaim!` leaves it pending until the
hold is dropped, and only the flush inside `quiesce_before_reclaim!` gets there.
Detecting that would make the automatic path flush whenever a recording is open,
which for a render loop is a stall every `trim_min_interval`.
[`trim_gpu_pool!`](@ref) is the caller that wants it and pays for it explicitly.
"""
function reclaimable(ctx::VkContext)
    p = mempolicy(ctx)
    any(b -> isempty(b.live), poolblocks(ctx)) && return true
    bq = ctx.default_bq
    return lock(() -> !isempty(bq.pending), bq.pendinglock) || !isempty(bq.retiring)
end

function maybe_trim_pool!(ctx::VkContext)
    p = mempolicy(ctx)
    gpu_live_bytes(ctx) < p.trim_threshold && return
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
    quiesce_before_reclaim!(bq) || return
    n_blocks, bytes_freed = reclaim_empty_pool_blocks!(bq)
    n_blocks > 0 && @debug "Lava: trimmed empty pool blocks" blocks=n_blocks MiB=(bytes_freed >> 20)
    return
end

"""
    trim_gpu_pool!() -> (blocks, bytes)

Hand every empty pool block back to the driver, now.

The automatic path (`mempolicy(ctx).trim_threshold`) is rate-limited and only runs
while something is allocating, so it is the wrong tool for "I have finished a
batch of work and want the memory back" — and for measuring, where dead pool
capacity otherwise counts as live and makes a VRAM figure depend on GC timing
rather than on demand.

Runs a full collection and then `quiesce_before_reclaim!` — a flush, a wait and a
drain — **unconditionally**, because that is the request. It used to return early
unless some block already had nothing live in it, and that gate is a precondition
the flush and the drain establish: a buffer pinned by a recording batch, or one
already retired but not yet destroyed, holds its entry until then, and
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
    isempty(poolblocks(ctx)) && return (0, 0)
    bq = ctx.default_bq
    quiesce_before_reclaim!(bq) || return (0, 0)
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

    live = gpu_live_bytes(ctx)
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
    post_gc_live = gpu_live_bytes(ctx)

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
    if any(b -> isempty(b.live), poolblocks(ctx))
        bq = ctx.default_bq
        if quiesce_before_reclaim!(bq)
            n_blocks, bytes_freed = reclaim_empty_pool_blocks!(bq)
            n_blocks > 0 && @debug "Lava: reclaimed empty pool blocks after GC" blocks=n_blocks MiB=(bytes_freed >> 20)
            post_gc_live = gpu_live_bytes(ctx)
        end
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
    live_mb = gpu_live_bytes(ctx) ÷ (1024 * 1024)
    println(io, "  Vulkan returned $(fail.code) from $(fail.op) for $(fail.nbytes) bytes ($(req_mb) MiB).")
    println(io, "  Lava tracked state: $(live_mb) MiB live across $(length(mempolicy(ctx).live_buffers)) buffers.")
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
    vk_alloc(bq::VulkanBatchQueue, nbytes; extra_usage=UInt32(0), unified=false) -> VkManagedBuffer

Allocate a GPU buffer with BDA support.  Takes the queue the allocation is
recorded against — `drain!` runs only on that queue's timeline, ready for the
multi-queue refactor.

* `extra_usage` — additional `VkBufferUsageFlags` bits (e.g. INDIRECT_BUFFER,
  INDEX_BUFFER, AS_BUILD_INPUT).
* `unified=true` — pick BAR memory (device-local + host-visible + coherent,
  falling back to host-visible only) and map it.  The returned
  `VkManagedBuffer.mapped_ptr` is non-null.  Use for arg/indirect slabs or
  any buffer you want to write from the CPU without staging.

AS-scratch alignment is the caller's responsibility — see
`bda_alignment_for(ctx, scratch::Bool)`, used in `LavaArray(...; scratch=true)`.
"""
function vk_alloc(bq::VulkanBatchQueue, nbytes::Integer;
                  extra_usage::UInt32=UInt32(0), unified::Bool=false)
    # Refuse allocation on a lost device — Vulkan calls would either error or
    # (worse) succeed against a torn-down driver state and produce garbage BDAs.
    device_lost(ctxof(bq)) && throw(LavaError(
        "vk_alloc",
        "Vulkan device is lost — cannot allocate new buffers",
        "Call reset_device!() to reinitialize, or restart Julia."))
    if mempolicy(ctxof(bq)).track_allocs
        record_alloc_site!(ctxof(bq), Int(nbytes))
    end
    # Give back what THIS queue has finished before allocating. Multi-queue:
    # each caller only drains its own queue's timeline — no implicit reach for
    # `ctx.default_bq` here.
    drain!(bq)
    maybe_collect(ctxof(bq))
    result = try_vk_alloc(bq, nbytes; extra_usage, unified)
    result isa VkManagedBuffer && return result
    # Reclaim any pool blocks that are now fully empty after the GC drained
    # their chunks back to the free list.  Without this the pool ratchets
    # up across renders — see Crown 1400×1000 hw_accel=true repro where
    # render 1 left 3.7 GiB of empty 64 MiB blocks pinning the heap.
    #
    # Skipped entirely while capturing: the drain that makes reclaiming safe
    # cannot happen there, so the allocation is left to fail with the ordinary
    # out-of-memory error below rather than free a block a recording still
    # names. An OOM inside `bake!` is a diagnosable error; a reclaimed block
    # under a replay is a GPU fault three frames later.
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
        format_oom_error(ctxof(bq), fail),
        "Free unused LavaArrays, reduce problem size, or check for memory leaks with gpu_memory_usage()."))
end

"""
    quiesce_before_reclaim!(bq)

Submit and wait for everything on `bq`, then collect and drain deferred frees.

The order is the point. `reclaim!` releases a chunk once its
**`last_write`** semaphore has signaled, which says nothing about dispatches
that only *read* it, nor about commands already recorded into the batch still
being built — and a graph evaluator does almost nothing else: every weight and
every activation is read by the next layer without being written. Draining (or
reclaiming a block those chunks belong to) while such a batch is open destroys a
VkBuffer out from under queued work, and the damage surfaces later and
elsewhere: a use-after-free the buffer's state flags on a later submit, a
batch-signal desync, or a segfault inside the driver's `vkCmdPipelineBarrier`.

Flushing first makes the invariant unconditional — after it there is no
recorded-but-unsubmitted work and everything submitted has completed — and it
only runs once an allocation has already failed, so the stall is free in the
steady state. `pool.reclaiming` guards the re-entry through `flush!`'s own
allocations.
"""

function quiesce_before_reclaim!(bq::VulkanBatchQueue)
    # Refuse while any plan on this device holds a recording, and say so, because
    # the caller must not go on to reclaim either.
    #
    # A drain is what makes reclaiming safe, and a recording cannot be drained
    # into safety: it is submitted again next run, so a block its command buffer
    # names is live for as long as the plan is. Waiting does not change that.
    #
    # It used to ask `bq.capturing !== nothing` — whether a capture was OPEN,
    # which is true only during `bake!` and false for every recording that had
    # already been taken. `movable` asks the pool about its tenants instead,
    # which is the property, and it is the same one arena growth already refuses
    # on.
    #
    # Here and not at the two trim entry points, which is where this check went
    # first. Same effect, worse test: an entry point has three gates in front of
    # it (`trim_threshold`, `trim_min_interval`, `reclaimable`) and a test-sized
    # workload trips none of them, so a regression test written against the entry
    # point passes whether the guard is there or not. Two were, and both were
    # worthless. Here there is one thing to ask and one answer.
    movable(pool(lavadevice(ctxof(bq)))) || return false
    p = mempolicy(ctxof(bq))
    if !p.reclaiming[] && !device_lost(ctxof(bq))
        p.reclaiming[] = true
        try
            flush!(bq)
        finally
            p.reclaiming[] = false
        end
    end
    GC.gc(true)
    drain!(bq)
    return true
end

"""Attempt GPU buffer allocation, returning an `AllocFailure` on OOM."""
function try_vk_alloc(bq::VulkanBatchQueue, nbytes::Integer;
                      extra_usage::UInt32=UInt32(0), unified::Bool=false)
    ctx = ctxof(bq)
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
            let c = ctxof(bq)
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

    result = VkManagedBuffer(buf, memory, address, mapped_ptr, Int(nbytes), nothing, Stamp{UInt64}(), BUF_STATE_ALIVE, ctx)
    let p = mempolicy(ctx)
        push!(p.live_buffers, result)
        Threads.atomic_add!(p.live_bytes, nbytes)
    end
    if (ctxof(bq)).diag.alloc_debug
        push!((ctxof(bq)).diag.alloc_log,
              (kind=:direct, addr=address, size=Int(nbytes), pool=false,
               mtype=Int(mem_type_idx), unified=unified, usage=UInt32(usage)))
    end
    return result
end

"""
    vk_free!(buf::VkManagedBuffer)

Say that this buffer's Vulkan resources are no longer wanted.

It does not destroy anything. A destroy requested while work naming the buffer
may still be running is core's to schedule — `retire!` records it, `reclaim!`
runs `rawfree` once the submission that named it has passed, and a buffer still
held by a recording that has not been submitted waits for that too. That is the
whole of what the old `pins` / `free_requested` / `deferred_frees` machinery
did, and none of it was a driver's decision.

Callable from any thread, which is what the finalizer path needs: it flips one
atomic and appends under a lock, and reads no timeline at all.
"""
function vk_free!(buf::VkManagedBuffer)
    # Atomic CAS ALIVE → DEFERRED. If someone else already transitioned this
    # buffer out of ALIVE, bail out silently — the work is already done (or is
    # being done).
    _, ok = @atomicreplace buf.state BUF_STATE_ALIVE => BUF_STATE_DEFERRED
    ok || return

    c = buf.ctx
    if !(c isa VkContext)
        # A buffer that never belonged to a context is not pooled and no queue
        # can be naming it; destroying it inline is safe and the `isa` keeps
        # this from throwing inside a finalizer.
        destroy_buffer!(buf)
        return
    end
    delete!(mempolicy(c).live_buffers, buf)
    # The channel is only where the request is parked: `reclaim!` reads the
    # stamp on the owning thread and hands the buffer to whichever channel last
    # named it, which is the fact this thread must not read.
    retire!(c.default_bq, buf)
    return
end

"""Actually destroy a buffer's Vulkan resources — the `rawfree` core calls once
nothing can be reading the bytes, and the direct path for a buffer no context
owns. Atomic CAS ({ALIVE, DEFERRED} → DEAD) ensures the Vulkan destructor fires
exactly once even under racing callers."""
function destroy_buffer!(buf::VkManagedBuffer)
    # Transition {ALIVE, DEFERRED} → DEAD.  First try DEFERRED (the common
    # path — vk_free! or drain always goes through DEFERRED now); fall back
    # to ALIVE for the rare direct-destroy path (e.g. dropping a buffer we
    # never retired).
    _, ok = @atomicreplace buf.state BUF_STATE_DEFERRED => BUF_STATE_DEAD
    if !ok
        _, ok = @atomicreplace buf.state BUF_STATE_ALIVE => BUF_STATE_DEAD
    end
    ok || return  # was already DEAD; idempotent

    # The free log records destructions, here rather than in `vk_free!`: this
    # is where a buffer actually goes, whichever thread asked, and the stamp it
    # is logged with is read on the owning thread — the only one that may.
    let c = buf.ctx
        if c isa VkContext && c.diag.free_debug
            st = buf.stamp
            push!(c.diag.free_log,
                  (addr=buf.address, size=buf.size, pool=buf.region !== nothing,
                   lw=(st.channel === nothing ? nothing : (st.channel, st.token))))
        end
        # Pre-destroy safety scan: if `buf.address` still appears in any live arg
        # slab, that is a use-after-free waiting to happen. Log it and zero the
        # slot so the GPU faults on a null reference (BDA_POISON) instead of
        # corrupting whatever memory is mapped at the old address.
        if c isa VkContext && c.diag.freed_bda_scan
            hits = scan_unified_for_bda!(buf)
            if hits > 0 && c.diag.destroy_freed_bdas_throws
                throw(LavaError("destroy_buffer!",
                    "destroying buffer at 0x$(string(buf.address, base=16, pad=16)) but its BDA still appears $(hits)x in live arg slabs",
                    "an unheld reference is leaking — see ctx.diag.freed_bda_scan_log"))
            end
        end
    end

    # Suballocated: give the span back, and never touch the block's VkBuffer —
    # it belongs to `Mantle.Block` and is shared with every other chunk in it.
    #
    # `release!` outright, not `retire!`. Retiring is for a caller who cannot
    # know what the device is still doing with the bytes; everything above this
    # point in `vk_free!` exists to establish exactly that, from the buffer's own
    # `last_write` and its queue's timeline — which is finer than the device-wide
    # fence `reclaim!` would take, and already paid for. Retiring here would hold
    # the span for one more submission boundary on top of a wait that has
    # already happened.
    #
    # The thread hazard that used to make this the wrong place is closed at the
    # allocator now: `release!` takes the block's lock, so a free-list corruption
    # like the `ConcurrencyViolationError` described in `vk_free!` cannot recur
    # through this path. The off-thread deferral above stays regardless — it is
    # about the DEVICE still reading the bytes, not about the list.
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

    # Check if the Vulkan device/context is still valid.
    # During Julia shutdown, finalizers fire after the device may be destroyed.
    # Finalizers cannot do context switches, so we only check simple flags here.
    ctx = buf.ctx::VkContext
    if device_lost(ctx)
        # Device is gone — just poison the handle, don't call Vulkan APIs
        buf.mapped_ptr = Ptr{UInt8}(0)
        buf.address = BDA_POISON
        Threads.atomic_sub!(mempolicy(ctx).live_bytes, buf.size)
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
    Threads.atomic_sub!(mempolicy(ctx).live_bytes, buf.size)
    buf.size = 0
end

"""Every mapped block of the unified arena — the memory a recording reads its
arguments and its workgroup counts out of, and therefore the only place a stale
device address can be sitting when the device runs.

It was two slab lists on the queue. The blocks are what backs them now, so the
scans below read the same bytes through the owner that has them."""
unifiedblocks(ctx::VkContext) = get(() -> Block[], spans(ctx).blocks, Unified())

# Scanner method — reachable now that VkManagedBuffer + VulkanBatchQueue are defined.
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
    return hits
end

"""
    scan_slabs_for_unknown_bdas() -> Vector{NamedTuple}

Walk every UInt64 of the unified arena — the argument blocks and indirect
commands a recording reads. Report any value that LOOKS like a BDA (in the upper
half of the address space, i.e. high bit of bit 63 set OR top 16 bits = 0xffff)
but is NOT the address of any buffer in the pool's `live_buffers` and is NOT 0.
Useful for catching stale BDAs that pin_leaves! / pack_args_direct! missed.

Call this RIGHT BEFORE submit to catch problems before they reach the GPU.
"""
function scan_slabs_for_unknown_bdas(bq)
    bq === nothing && return NamedTuple[]
    ctx = ctxof(bq)
    live = Set{UInt64}()
    for buf in mempolicy(ctx).live_buffers
        push!(live, buf.address)
    end
    # A pool block is a valid arena, and so is the unified one: packing a nested
    # struct writes a pointer to WITHIN the argument block itself (a
    # "byval-inline" argument's outer pointer is `base + inline_offset`), so any
    # value landing inside a block is expected rather than stale.
    ranges = Tuple{UInt64,UInt64}[]
    for blk in poolblocks(ctx)
        bb = blk.memory
        bb isa BufferBlock || continue
        push!(ranges, (bb.address, bb.address + UInt64(bb.bytes)))
    end
    @inline function in_known_range(addr)
        for (lo, hi) in ranges
            lo <= addr < hi && return true
        end
        return false
    end
    results = NamedTuple[]
    for (i, blk) in enumerate(unifiedblocks(ctx))
        mb = (blk.memory::BufferBlock).ref[]::VkManagedBuffer
        mb.mapped_ptr == Ptr{UInt8}(0) && continue
        n = Int(mb.size) ÷ 8
        p = Ptr{UInt64}(mb.mapped_ptr)
        for k in 0:(n-1)
            v = unsafe_load(p, k+1)
            # Look only for "0xffff8…" sign-extended BDA-shaped values
            # whose 48-bit form is in the high half (bit 47 set).
            v < UInt64(0xffff800000000000) && continue
            v == typemax(UInt64) && continue  # 0xff..ff often appears in scratch
            v in live && continue
            in_known_range(v) && continue
            push!(results, (slab=:unified, idx=i, offset=k*8, val=v))
        end
    end
    return results
end

# `lastwritepassed`, `drain_deferred_frees!` and the two lists they swept are
# gone with phase 2.2. "Has the device finished the last submission that named
# this buffer" is one question about one stamp, and core asks it once, for
# buffers and acceleration structures alike — `reclaim!` in `graph/lifetime.jl`.
# What this file still answers is the destructor itself: `rawfree`.

# ── Sub-allocation: one pool, and it is Mantle's ──
#
# What was here: 64 MiB blocks carved by a bump pointer, 153 size classes, a free
# list per class holding recycled `VkManagedBuffer` objects, and a block-reclaim
# scan. About seven hundred lines, and a complete second allocator sitting beside
# `Mantle.Pool` with no knowledge of it.
#
# Two allocators over one `VkDevice` is not a tidiness problem. Neither could
# reuse the other's bytes, so an idle 64 MiB arena could not serve an array
# allocation that was about to grow the pool, and the footprint either reported
# was about its own bookkeeping rather than about the device. `Mantle.Pool`'s own
# header says this outright: a backend that routes `rawalloc` through its own
# pool defeats the point.
#
# So the size classes are gone and this file allocates the same way `Place` does.
# What each piece became:
#
#     pool_alloc                  ->  acquire!
#     return_to_pool!             ->  release!
#     alloc_pool_block            ->  Pool's own growth, in acquire!
#     reclaim_empty_pool_blocks!  ->  trim!
#     PoolBlock                   ->  Mantle.Block
#     (pool_offset, pool_block)   ->  Mantle.Region
#     size_class / POOL_SUBDIV    ->  nothing; carving is exact
#
# Two things are lost with the size classes and both were measured, so they are
# stated rather than discovered later:
#
#   * **Rounding waste is gone**, which was the size classes' whole cost. Eight
#     subclasses per octave bounded it at 1/8 of a request — SAM 2's encoder went
#     from 59.3% efficient to ~94% when they were introduced. `carve!` splits at
#     exactly the requested length, so it is 100%, and `pool_accounting` is
#     deleted because it existed to measure a number that is now always 1.0.
#   * **Object recycling is gone.** A free list held the `VkManagedBuffer` itself,
#     so reuse cost a `pop!` and four field writes. A fresh one is now built per
#     allocation: one mutable struct and one finalizer, against a suballocator
#     call that costs roughly 300 ns where the size-class `pop!` cost about 50.
#     That is the price of one allocator instead of two.
#
# The LIFETIME layer above this is untouched, and deliberately. `vk_free!` still
# decides when a buffer's bytes are safe to reuse from that buffer's own
# `last_write` and its queue's deferred list, which is finer-grained than the
# device-wide fence `Mantle.retire!`/`reclaim!` use. By the time `destroy_buffer!`
# reaches a pooled chunk the wait is already done and the thread is the owning
# one, so it can `release!` outright rather than retiring for another submission
# boundary. `retire!` remains what the graph arenas use, where no per-resource
# `last_write` exists to be more precise with.

"""
How large a block the pool cuts when it has to grow.

64 MiB, unchanged: it is the size the old allocator used, the size `Place`
already asks for, and `Mantle.blocksize` reports.
"""
const POOL_BLOCK_SIZE = 64 * 1024 * 1024

"""
The smallest allocation. Vulkan rejects a zero-length buffer, and 16 bytes keeps
every chunk's start usable for any scalar type without a second alignment rule.
"""
const POOL_MIN_SIZE = 16

"""
Alignment every sub-allocation gets.

256 is `Mantle.acquire!`'s own default and covers every
`minStorageBufferOffsetAlignment` this runs on. A buffer needing more — an
acceleration-structure scratch region — asks through `bda_alignment_for` and
takes the dedicated path, where the alignment is applied to a whole allocation.
"""
const POOL_ALIGN = 256

# Defaults live here, next to the policy they configure, rather than in eleven
# module-level `Ref`s. `2 GiB` soft cap, trim above 1 GiB and no more than every
# 5 s, a full GC no more than every 30 s.
MemoryPolicy() = MemoryPolicy(false, 2 * 1024^3, 1024 * 1024 * 1024,
                              5.0, 30.0, 0.02, 0.5, false,
                              0.0, 0.0, 0.0, 0.0, 0.0,
                              Threads.Atomic{Int}(0), Set{VkManagedBuffer}(),
                              Threads.Atomic{Int}(0),
                              Threads.Atomic{Bool}(false))

"""
    lavadevice(ctx) -> LavaDevice

This context's device handle, and so its `Mantle.Pool`.

Taken from `ctx`, never from `vk_context()`. The pool has to be the one that owns
this context's blocks: `MemoryPolicy`'s docstring records what happened the last
time an allocation could reach another device's memory, and reading the global
here would be exactly that path reopened.
"""
lavadevice(ctx::VkContext) = get!(DEVICES, ctx) do
    LavaDevice(ctx, ctx.default_bq)
end

"""The pool this context suballocates from."""
@inline spans(ctx::VkContext) = pool(lavadevice(ctx))

"""
    mempolicy(ctx) -> MemoryPolicy

When this device grows, trims and collects — not where its memory comes from,
which is [`spans`](@ref).

Never call this from a finalizer expecting to allocate: it is a field read, but
the policy it returns is not the free path. A buffer's own `region` is what
`release!` needs, and that has been on the buffer since it was carved.
"""
@inline mempolicy(ctx::VkContext) = ctx.caches.pool

"""
    pool_offset(buf) -> Int

Where this buffer starts inside the block it was carved from, or 0 when it owns
its memory outright.

A field before, kept in step with `pool_block` by hand. Derived now, so the two
cannot disagree — every copy, barrier and image transfer that adds it to a view's
own offset reads the same number the allocator recorded.
"""
@inline pool_offset(buf::VkManagedBuffer) =
    buf.region === nothing ? 0 : offset(buf.region::Region)

"""Every block this device has, across the arenas they were cut for."""
poolblocks(ctx::VkContext) =
    collect(Iterators.flatten(values(spans(ctx).blocks)))

"""
    gpu_live_bytes(ctx = vk_context()) -> Int
    live_buffer_count(ctx = vk_context()) -> Int

What this device currently holds: bytes the driver has handed over, and how many
dedicated buffers they are spread across.

`reserved` is the pool's own total, which is the honest figure — a suballocated
chunk costs nothing beyond the block it sits in, and counting chunks as well
would double every byte. `live_bytes` carries only the buffers that own their
memory: staging, mapped, and anything whose usage flags the pool cannot host.
"""
gpu_live_bytes(ctx::VkContext = vk_context()) =
    reserved(spans(ctx)) + mempolicy(ctx).live_bytes[]
live_buffer_count(ctx::VkContext = vk_context()) = length(mempolicy(ctx).live_buffers)

"""
    destroy_pool!(ctx)

Destroy this device's blocks. Called by `reset_device!` on the context it is
retiring — which is where the old context is actually in scope.

Unconditional, unlike [`trim_gpu_pool!`](@ref): the device is going away, so a
block with live regions still has to be released, and whatever holds those
regions is about to be pointing at a dead device either way. That is the one
place the `live` ledger is deliberately ignored.
"""
function destroy_pool!(ctx::VkContext)
    p = spans(ctx)
    dev = lavadevice(ctx)
    for (_, blks) in p.blocks, blk in blks
        rawfree(dev, blk.memory)
    end
    empty!(p.blocks)
    empty!(p.arenas)
    empty!(p.pending)
    empty!(p.retiring)
    delete!(DEVICES, ctx)
    return nothing
end

"""Collections run by the soft cap, and the seconds they cost."""
pool_gc_stats(ctx::VkContext = vk_context()) =
    (; count = mempolicy(ctx).gc_count[], seconds = mempolicy(ctx).gc_seconds)

function reset_pool_gc_stats!(ctx::VkContext = vk_context())
    p = mempolicy(ctx)
    p.gc_count[] = 0; p.gc_seconds = 0.0
    return
end

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
#     no queue drain — the memory comes back as free spans the caller takes
#     immediately.
#   * `trim_threshold` RELEASES capacity that is already dead, back to
#     the driver. Periodic, expensive, and the only thing that helps when the
#     pressure is on memory the rest of the machine needs.

"""
    mempolicy(ctx).soft_cap

Pool footprint in bytes past which an allocation collects *before* committing
another block. `0` disables it.

This is the cheap half of keeping the pool small, and it is deliberately not
the same mechanism as [`maybe_trim_pool!`]:

  * here we run a **GC and reuse** — no queue drain, no Vulkan call, and the
    memory comes back as free spans the caller immediately takes, so this
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

"""
    collect_for_pool!(bq) -> Bool

Try to turn dead LavaArrays back into free spans. Returns whether a collection
actually ran, so the caller knows whether retrying is worthwhile.

`drain!` after each collection is what makes this work at all: a
buffer freed while the GPU still referenced it went to the deferred list rather
than back to the pool, and until it is drained the memory is dead to everyone.
"""
function collect_for_pool!(bq::VulkanBatchQueue)
    p = mempolicy(ctxof(bq))
    now = time()
    now - p.gc_last < p.gc_mingap && return false
    t0 = time_ns()
    GC.gc(false)
    drain!(bq)
    if now - p.gc_full_last >= p.gc_full_mingap
        GC.gc(true)
        drain!(bq)
        p.gc_full_last = now
    end
    p.gc_last = time()
    p.gc_seconds += (time_ns() - t0) / 1e9
    Threads.atomic_add!(p.gc_count, 1)
    return true
end

# Diagnostic: track allocation call sites during recording.
# Set `mempolicy(ctx).track_allocs = true` to record stack traces of every allocation while
# the active batch is recording. Used to find per-frame allocations leaking into the
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
    pool_alloc(bq::VulkanBatchQueue, nbytes; extra_usage=UInt32(0)) -> VkManagedBuffer

Allocate GPU memory on `bq` out of this device's `Mantle.Pool`.

`extra_usage` is passed through as the region's CONSTRAINT rather than used to
bypass the pool, which is the one behavioural change of the merge. The old
allocator gave every non-default usage its own `VkBuffer`, because a size class
carries no notion of what a block may be used for; `compatible` does, so an
index buffer can now sit in a block whose usage bits already permit it and only
falls out to a dedicated allocation when none does.
"""
function pool_alloc(bq::VulkanBatchQueue, nbytes::Integer; extra_usage::UInt32=UInt32(0))
    ctx = ctxof(bq)
    # Even pure free-span reuse must refuse a dead device — the blocks belong to
    # the old (broken) ctx and would hand back garbage BDAs.
    device_lost(ctx) && throw(LavaError(
        "pool_alloc",
        "Vulkan device is lost — cannot allocate new buffers",
        "Call reset_device!() to reinitialize, or restart Julia."))
    p = mempolicy(ctx)
    nbytes = max(Int(nbytes), POOL_MIN_SIZE)
    p.track_allocs && record_alloc_site!(ctx, nbytes)
    drain!(bq)
    # Pressure-driven GC, identical hook to `vk_alloc`. Without this the pool
    # fast path silently grows VRAM until every block is full, then forces a
    # `GC.gc` from inside the alloc — exactly the 2-5 ms spike the AK benchmarks
    # regressed on.
    maybe_collect(ctx)

    p.disabled && return vk_alloc(bq, nbytes; extra_usage)

    dev = lavadevice(ctx)
    sp = pool(dev)
    # Past the soft cap, ask the GC before committing another block: at that
    # point the memory this request needs is far more likely to be dead and
    # uncollected than genuinely in use. Under the cap this is not paid at all,
    # and above it a collection runs at most every `gc_mingap` seconds.
    if p.soft_cap > 0 && reserved(sp) >= p.soft_cap
        collect_for_pool!(bq)
    end

    region = acquire_or_reclaim!(bq, sp, dev, nbytes, extra_usage)
    blk = region.block.memory::BufferBlock
    return VkManagedBuffer(
        blk.buffer, blk.memory,
        blk.address + UInt64(offset(region)),
        Ptr{UInt8}(0),
        length(region),
        region, Stamp{UInt64}(), BUF_STATE_ALIVE, ctx)
end

"""
Acquire a region, and treat an out-of-memory from the driver as a request to
collect rather than as the answer.

`Mantle.acquire!` already reclaims its own retired regions twice before it grows,
so what is left for this to handle is the case where growth itself fails: the
device is genuinely full, and the memory the caller needs is held by
`LavaArray`s that are dead but uncollected, or by blocks that are empty but not
yet handed back.

`VulkanError` by name, not a bare `catch`. An OOM is the one failure that is
worth retrying and it is the only one retried here; anything else — a lost
device, a bad usage flag — propagates with its own message intact.
"""
# `dev` is untyped for the same reason `ctxof(bq)` is: `LavaDevice` is declared in
# `graph.jl`, which this file is included before. A signature annotation is
# evaluated at load; the body is not.
function acquire_or_reclaim!(bq::VulkanBatchQueue, sp::Pool, dev,
                             nbytes::Int, extra_usage::UInt32)
    try
        return acquire!(sp, dev, Buffers(), nothing, nbytes;
                        align = POOL_ALIGN, blocksize = POOL_BLOCK_SIZE,
                        constraint = extra_usage)
    catch err
        err isa VK.VulkanError || rethrow()
        oom = err.code == VK.ERROR_OUT_OF_DEVICE_MEMORY ||
              err.code == VK.ERROR_OUT_OF_HOST_MEMORY
        oom || rethrow()
    end
    # One escalation, in the order that costs least first: collect, drain, and
    # only then stall the queue to get pinned and in-flight buffers back.
    GC.gc(true)
    drain!(bq)
    # Not while capturing; see `quiesce_before_reclaim!`. The acquire below
    # still runs and still throws its own out-of-memory error if it cannot.
    if quiesce_before_reclaim!(bq)
        nblocks, freed = reclaim_empty_pool_blocks!(bq)
        nblocks > 0 &&
            @info "Lava: reclaimed empty pool blocks on OOM retry" blocks=nblocks MiB=(freed >> 20)
    end
    try
        return acquire!(sp, dev, Buffers(), nothing, nbytes;
                        align = POOL_ALIGN, blocksize = POOL_BLOCK_SIZE,
                        constraint = extra_usage)
    catch err
        err isa VK.VulkanError || rethrow()
        throw(LavaError("pool_alloc",
            "Cannot allocate $(nbytes) bytes: the device is out of memory with " *
            "$(humanbytes(reserved(sp))) already reserved by this pool, and a full " *
            "collection plus a queue drain freed nothing that helps.",
            "Free unused LavaArrays, reduce problem size, or check for leaks with " *
            "gpu_memory_usage()."))
    end
end

"""
    reclaim_empty_pool_blocks!(bq::VulkanBatchQueue) -> (n_blocks::Int, bytes::Int)

Hand every block with nothing live in it back to the driver, and say how much
that was.

`Mantle.trim!` does the deciding — a block is returnable when its `live` ledger
is empty — so what is left here is measuring, which the caller reports.

Callers must have run `quiesce_before_reclaim!` first; see there for why. Not
finalizer-safe, and never was.
"""
function reclaim_empty_pool_blocks!(bq::VulkanBatchQueue)
    ctx = ctxof(bq)
    sp = spans(ctx)
    before = reserved(sp)
    nbefore = length(poolblocks(ctx))
    trim!(sp, lavadevice(ctx))
    return (nbefore - length(poolblocks(ctx)), before - reserved(sp))
end

# ── Staging for CPU↔GPU transfers ──
#
# No staging buffer on the queue. There was one, reused across transfers, and
# it was safe only because the open batch ordered every copy through it; with
# each transfer its own submission, reusing it would race the previous transfer
# still in flight. An upload's staging bytes are a `Unified` scratch region
# owned by the one-shot that copies out of them (`scratch!`), released by the
# sweep once the transfer has passed; a download's are a `Readback` region —
# host-cached, not BAR — the caller acquires, waits for, reads and releases.
# `host_buffer`, the allocation the old staging buffer was made of, went with
# the staging buffer: `rawalloc(dev, ::Readback, …)` is that allocation, pooled.

"""
    copy_buffer!(direction, managed, host_ptr, nbytes; offset=0)

Unified CPU↔GPU buffer copy — single entry point for every host-visible
transfer path.  `direction` is `:upload` (host → GPU) or `:download` (GPU → host).

Fast path (BAR memory): direct memcpy via `managed.mapped_ptr` after
`waitfor!(stampof(managed))` drains any in-flight writer.  No staging, no batch.

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
    buf_offset = pool_offset(managed) + Int(offset)

    # BAR fast-path: host-visible mapped memory — direct memcpy, no staging.
    if managed.mapped_ptr != Ptr{UInt8}(0)
        # `last_write` is stamped at submit, and every writer is submitted the
        # moment it is closed, so this sees every one of them.
        waitfor!(stampof(managed))
        if direction === :upload
            unsafe_copyto!(managed.mapped_ptr + offset, host_ptr, nbytes)
        else
            unsafe_copyto!(host_ptr, managed.mapped_ptr + offset, nbytes)
        end
        return
    end

    # Device-local: through a `Unified` region and one copy, in a one-shot of
    # its own. Holding `managed` is what orders the copy behind whatever other
    # channel wrote it — core derives the wait from the stamp at submit.
    #
    # A download is recorded on the channel that last named the buffer, so the
    # copy piggy-backs on the producer's timeline and the host waits once
    # instead of twice. `stampof` is where that fact is now; it was
    # `last_write_bq`, read directly.
    bq = if direction === :upload
        (managed.ctx::VkContext).default_bq
    else
        wbq = stampof(managed).channel
        (wbq !== nothing && !queue_released(wbq::VulkanBatchQueue)) ?
            (wbq::VulkanBatchQueue) : (managed.ctx::VkContext).default_bq
    end
    # `Mantle.offset`, qualified: this function's `offset` keyword shadows it.
    if direction === :upload
        # The staging bytes belong to the one-shot: written before it is
        # submitted, given back by the sweep once the copy has passed. Nothing
        # waits — the next reader of `managed` on this queue is ordered behind
        # the copy, and a host reader waits on `last_write`.
        oneshot!(bq; tag = :upload) do e
            r = scratch!(e.owner, nbytes)
            mb = (memoryof(r)::BufferBlock).ref[]::VkManagedBuffer
            unsafe_copyto!(mb.mapped_ptr + Mantle.offset(r), host_ptr, nbytes)
            cmd_copy_buffer!(e, mb, managed, nbytes;
                             src_off = pool_offset(mb) + Mantle.offset(r), dst_off = buf_offset)
        end
    else
        # The caller's region: acquired here, read after the copy has passed,
        # and released here — not scratch of the one-shot, which the sweep
        # could hand on the moment the wait returned. From the `Readback` kind,
        # which is host-CACHED memory: this staged through `Unified` for a
        # while, and reading BAR memory back is a 29 MB/s memcpy — 330 ms for
        # one 1368x1026 RGBA{Float32} frame, measured on both drivers here.
        dev = lavadevice(managed.ctx::VkContext)
        r = acquire!(pool(dev), dev, Readback(), nothing, max(Int(nbytes), 16);
                     align = ARG_ALIGN, blocksize = READBACK_BLOCK_SIZE)
        mb = (memoryof(r)::BufferBlock).ref[]::VkManagedBuffer
        tok = oneshot!(bq; tag = :download) do e
            cmd_copy_buffer!(e, managed, mb, nbytes;
                             src_off = buf_offset, dst_off = pool_offset(mb) + Mantle.offset(r))
        end
        # On `bq`'s timeline — the queue that last wrote `managed`, which need
        # not be the device's primary one.
        waitfor!(bq, tok)
        unsafe_copyto!(host_ptr, mb.mapped_ptr + Mantle.offset(r), nbytes)
        release!(r)
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
# `vk_alloc(bq, nbytes; extra_usage, unified)` now.
#
# The two slab pools that stood beside them — the arg buffer ring and the
# indirect dispatch ring, and `INDIRECT_SLAB_SIZE` with them — are gone as well.
# What a recording reads is a `Region` of the `Unified` arena, owned by whoever
# owns the recording: see `scratch!` in `runtime/launch.jl`.

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
