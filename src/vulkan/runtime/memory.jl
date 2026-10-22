# Vulkan memory management for Lava.jl
#
# Device-local buffers with BDA (Buffer Device Address) for kernel arguments.
# Staging buffer for CPU↔GPU transfers.


# Debug instrumentation for iter6 cross-scene cascade investigation.
# All off by default — opt-in via the *_ENABLED Refs.

# Per-finalizer destruction trace

# Per-allocation trace

# When enabled, vk_free! scans every live channel's arg/indirect slabs for
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
a recording reads — for any UInt64 word matching `buf.address`. For each hit,
append a record to `diag.freed_bda_scan_log` and overwrite the slot with 0 so the
GPU faults cleanly on a null reference instead of touching the freed memory.
Returns the number of hits found.

Defined AFTER VkManagedBuffer + the submit channel (forward-call from vk_free!).
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
# retires. Staging, indirect and arg slabs are per-channel and go the same
# way. `reset_memory_stats!` is the one piece left, and `reset_device!` calls
# it directly.

# Native heap facts and usage feed the shared PoolPolicy controller. Usage is
# pooled capacity plus dedicated buffers; the dedicated-byte counter alone
# misses most device memory. No second collection budget lives in this backend.

"""
    eager_gc!(ctx, flag::Bool)

Enable/disable the pressure-driven `maybe_collect` for THIS context. Useful when
benchmarking, to take the allocator's GC hooks out of the measurement.
"""
eager_gc!(ctx::VkContext, flag::Bool) = (mempolicy(ctx).eager_gc = flag)
eager_gc!(flag::Bool) = eager_gc!(vk_context(), flag)

function reset_memory_stats!(ctx::VkContext = vk_context())
    st = mempolicy(ctx).stats
    @atomic st.size = 0
    @atomic st.last_updated = 0.0
    return
end

"""Sum of device-local heap sizes in bytes for `ctx`'s physical device."""
function probe_device_local_heap(ctx::VkContext)
    mem_props = ctx.memory_properties
    total = 0
    # `memory_heaps` is the fixed array of VK_MAX_MEMORY_HEAPS (16); the device
    # fills `memory_heap_count` of them.
    for i in 0:(Int(mem_props.memory_heap_count) - 1)
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
    n = Int(mem_props.memory_heap_count)      # of the 16 slots in `memory_heaps`
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
    overbudget(ctx, memtype, bytes) -> Bool

Whether `bytes` more in memory type `memtype` (0-based) would take this process
past the driver's budget for that type's heap (`VK_EXT_memory_budget`).

The budget, not the heap size, because a driver does not have to fail an
allocation past it. RADV does not: it moves buffers out of VRAM into system
memory (GTT), and once GTT passes about half the RAM, the kernel's
`ttm_global_swapout` path runs, which in amdgpu on kernel 7.2.6 corrupts a list
under a spinlock and locks up the whole machine (two desktop freezes on
2026-10-06, a 7900 XTX filled by one Julia process). An allocation past the
budget is out of memory, as `docs/api.md` says for every backend; callers treat
it as they treat the driver's own `ERROR_OUT_OF_DEVICE_MEMORY` (reclaim, retry,
then throw with the heap figures).

`false` without the extension: then the driver's own error is all there is.
The second method is the rule alone, on one heap of a
[`probe_device_memory_budget`](@ref) snapshot.
"""
function overbudget(ctx::VkContext, memtype::Integer, bytes::Integer)
    ctx.memory_budget_available || return false
    heap = Int(ctx.memory_properties.memory_types[memtype + 1].heap_index)
    return overbudget(probe_device_memory_budget(ctx)[heap + 1], bytes)
end
overbudget(heap::NamedTuple, bytes::Integer) = heap.usage + bytes > heap.budget

"""
    maybe_collect(ctx::VkContext; blocking::Bool=false)

Query native heap pressure and enter core's shared collection controller.

Called from `vk_alloc` and `pool_alloc` before allocating.  `blocking=true`
lowers the pressure threshold and inflates the rate budget — use it when the
caller is about to do a heavy synchronous operation anyway.
"""
# Absolute-capacity pool trim: core's `autotrim!`, the pool's own policy on
# every backend (`PoolPolicy` in `memory/pool.jl`, which has the measurements).
# `maybe_collect`'s pressure gate is a *ratio* against the device heap, the wrong
# signal for holding on to dead blocks on an iGPU, so the trim runs before it on
# its own terms.

"""
    trim_gpu_pool!(; invalidate_recordings=false) -> (blocks, bytes)

Hand every empty pool block back to the driver, now.

With `invalidate_recordings=true`, wait for submitted work and discard reusable
commands held by live arena plans. Plans and their resources remain alive; their
next `run!` records commands again. Use between frames when changing large input
buffers would otherwise leave dead blocks pinned by unrelated recorded plans.
An open recording still refuses cleanup. The default retains all recordings.

The automatic path (`autotrim!`, by `pool(dev).policy`) is rate-limited and only runs
while something is allocating, so it is the wrong tool for "I have finished a
batch of work and want the memory back" — and for measuring, where dead pool
capacity otherwise counts as live and makes a VRAM figure depend on GC timing
rather than on demand.

Runs a full collection and then `quiesce_before_reclaim!` — a flush, a wait and a
drain — **unconditionally**, and NOT gated on some block already having nothing
live in it: that is a precondition the flush and the drain establish. A buffer
pinned by a recording batch, or retired but not yet destroyed, holds its entry
until then, and between them that is everything a graph evaluator allocates.

Gating on it instead leaves the memory resident. On 60 unsynchronised dispatches
that is 15 blocks and 964 MiB with 0 blocks empty; flushing first empties 15 of
15 and hands back 960 MiB.

The automatic path keeps the cheap gate; see `Mantle.reclaimable`. This one is
the explicit "I have finished and want the memory back", so it pays the stall.
"""
function trim_gpu_pool!(ctx::VkContext = vk_context(); invalidate_recordings::Bool=false)
    bq = ctx.default_bq
    if invalidate_recordings
        # A completed, reusable recording still pins the pool. An explicit
        # between-frame cleanup may discard those commands, keeping each Plan
        # and its live resources; run! records it again on its next use.
        # Never sweep an open recording's positional hold frame.
        isempty(bq.holds) || return (0, 0)
        device_lost(ctx) && return (0, 0)
        flush!(bq)
        waitidle(lavadevice(ctx))
        seen=IdDict{Any,Nothing}()
        for arena in values(spans(ctx).arenas), weak in tenants!(arena)
            plan=weak.value
            plan isa Plan && recorded(plan) || continue
            haskey(seen,plan) && continue
            seen[plan]=nothing
            invalidate!(plan)
        end
    end
    quiesce_before_reclaim!(bq) || return (0, 0)
    return reclaim_empty_pool_blocks!(bq)
end

function maybe_collect(ctx::VkContext; blocking::Bool=false)
    p = mempolicy(ctx)
    p.eager_gc || return
    stats = p.stats
    current_time = time()

    # Runs before the ratio gate below: dead pool capacity has to be returned on
    # its own terms, not only when the heap ratio says we are in trouble.
    autotrim!(spans(ctx), lavadevice(ctx))

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

    autocollect!(() -> drain!(ctx.default_bq), spans(ctx);
                 pressure = gpu_live_bytes(ctx) / size, blocking) || return

    # Capacity counts against heap pressure too. Core returns reclaimable
    # blocks without paying for another collection through explicit quiescence.
    trimcollected!(spans(ctx), lavadevice(ctx))

    return
end

"""
Captures the exact Vulkan error that caused an allocation to fail.  Returned
from `try_vk_alloc` instead of swallowing the result code, so callers can
include the real `VkResult` and the failing op in the LavaError they throw.
"""
struct AllocFailure
    code::VK.Result
    op::Symbol          # :Buffer, :budget, :DeviceMemory, :bind_buffer_memory, :map_memory
    nbytes::Int
    mem_type_idx::Int   # -1 if the failure happened before memory-type selection
end

function format_oom_error(ctx::VkContext, fail::AllocFailure)
    io = IOBuffer()
    println(io, "Out of GPU memory.")
    req_mb = fail.nbytes ÷ (1024 * 1024)
    live_mb = gpu_live_bytes(ctx) ÷ (1024 * 1024)
    if fail.op === :budget
        println(io, "  $(fail.nbytes) bytes ($(req_mb) MiB) would exceed the driver's budget for the heap (VK_EXT_memory_budget); not asked for.")
    else
        println(io, "  Vulkan returned $(fail.code) from $(fail.op) for $(fail.nbytes) bytes ($(req_mb) MiB).")
    end
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
    vk_alloc(bq::SubmitChannel{<:VulkanQueue}, nbytes; extra_usage=UInt32(0), unified=false) -> VkManagedBuffer

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
function vk_alloc(bq::SubmitChannel{<:VulkanQueue}, nbytes::Integer;
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
steady state. The core collection claim guards re-entry through `flush!`'s allocations.
"""

function quiesce_before_reclaim!(bq::SubmitChannel{<:VulkanQueue})
    ctx = ctxof(bq)
    device_lost(ctx) && return false
    dev = lavadevice(ctx)
    return quiesce_pool!(() -> flush!(bq), () -> drain!(bq), pool(dev), dev)
end

"""
    discardvalidation!(ctx)

Forget the validation messages a refused allocation produced: an expected and
handled refusal (`try_vk_alloc`) owns them.

DRAIN, then empty. The validation callback writes into a ring
(`ctx.validation`, per device since 49f3f17) and only
`drain_validation_messages!` moves entries out of it into
`.messages`. Emptying the drained list alone leaves this failure's
own messages sitting in the ring, where the next
`check_validation_errors!` picks them up and blames its own caller.

Observed exactly that way: a test (then in test_source_mapping.jl) asked for
40 GB deliberately, and the error surfaced 40 lines later as a
`LavaError during flush!` on a FOUR-ELEMENT upload. An oversized
allocation is the intended, handled outcome here, so its messages
belong to it.
"""
function discardvalidation!(ctx::VkContext)
    drain_validation_messages!(ctx)
    empty!(ctx.validation.messages)
    return
end

"""Attempt GPU buffer allocation, returning an `AllocFailure` on OOM."""
function try_vk_alloc(bq::SubmitChannel{<:VulkanQueue}, nbytes::Integer;
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
        # Past the budget is out of memory even where the driver would say yes
        # (`overbudget`); returning the failure sends it through the same
        # reclaim-and-retry as the driver's own refusal.
        if overbudget(ctx, mem_type_idx, mem_reqs.size)
            discardvalidation!(ctx)      # creating `buf` may already have complained
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
        # Only swallow honest OOM — every other VulkanError must propagate so
        # the caller sees the real cause (DEVICE_LOST, INVALID_OPAQUE_CAPTURE_ADDRESS,
        # MEMORY_MAP_FAILED, …).  Otherwise vk_alloc's GC-retry would loop and
        # eventually throw a misleading "Out of GPU memory".
        if e isa VK.VulkanError &&
           (e.code == VK.ERROR_OUT_OF_DEVICE_MEMORY ||
            e.code == VK.ERROR_OUT_OF_HOST_MEMORY)
            discardvalidation!(ctxof(bq))
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
whole of it, and none of it is a driver's decision.

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
    # The thread hazard is closed at the allocator: `release!` takes the
    # block's lock, so a free-list corruption
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

The blocks are what backs the argument and indirect bytes, so the scans below
read them through the owner that has them."""
unifiedblocks(ctx::VkContext) = get(() -> Block[], spans(ctx).blocks, Unified())

# Scanner method — reachable now that VkManagedBuffer + the channel are defined.
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

# "Has the device finished the last submission that named this buffer" is one
# question about one stamp, and core asks it once, for buffers and acceleration
# structures alike — `reclaim!` in `graph/lifetime.jl`. What this file answers is
# the destructor itself: `rawfree`.

# ── Sub-allocation: one pool, and it is Mantle's ──
#
# Array allocations go through `Mantle.Pool`, the same way `Place` does, and this
# file keeps no allocator of its own. Two allocators over one `VkDevice` cannot
# reuse each other's bytes, so an idle 64 MiB arena could not serve an array
# allocation about to grow the pool, and the footprint either one reported would
# be about its own bookkeeping rather than about the device. `Mantle.Pool`'s own
# header says this outright: a backend that routes `rawalloc` through its own
# pool defeats the point.
#
# Carving is exact (`carve!` splits at the requested length), so there is no
# rounding waste to account for, and a `VkManagedBuffer` is built per allocation
# rather than recycled from a free list: one mutable struct and one finalizer,
# against a suballocator call of roughly 300 ns. That is the price of one
# allocator instead of two.
#
# The LIFETIME layer above this is finer-grained than the pool's, and stays that
# way. `vk_free!` decides when a buffer's bytes are safe to reuse from that
# buffer's own `last_write` and its queue's deferred list, not from the
# device-wide fence `Mantle.retire!`/`reclaim!` use. By the time
# `destroy_buffer!` reaches a pooled chunk the wait is already done and the
# thread is the owning one, so it can `release!` outright rather than retiring
# for another submission boundary. `retire!` is what the graph arenas use, where
# no per-resource `last_write` exists to be more precise with.

"""
How large a block the pool cuts when it has to grow.

64 MiB: the size `Place` asks for and `Mantle.blocksize` reports.
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

# Defaults live here, next to the policy they configure, rather than in
# module-level `Ref`s. The trim and the soft cap are the pool's own policy now
# (`PoolPolicy`, core's), which every backend's pool applies.
MemoryPolicy() = MemoryPolicy(false, false, true, MemoryStats(),
                              Threads.Atomic{Int}(0), Set{VkManagedBuffer}())

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
    # Taken and dropped: what finalizers handed over was waiting for THIS device
    # to finish with it, and the device is gone. `takeall!` because both are the
    # lock-free `Inbox`, which has no `empty!`.
    takeall!(p.pending)
    takeall!(p.pendingplans)
    empty!(p.retiring)
    empty!(p.retiring_at)
    delete!(DEVICES, ctx)
    return nothing
end

"""Pool collection cycles and their cost: the pool's
policy's counters (`PoolPolicy`)."""
pool_gc_stats(ctx::VkContext = vk_context()) =
    (; count = spans(ctx).policy.gc_count, seconds = spans(ctx).policy.gc_seconds)

function reset_pool_gc_stats!(ctx::VkContext = vk_context())
    p = spans(ctx).policy
    p.gc_count = 0; p.gc_seconds = 0.0
    return
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
    pool_alloc(bq::SubmitChannel{<:VulkanQueue}, nbytes; extra_usage=UInt32(0)) -> VkManagedBuffer

Allocate GPU memory on `bq` out of this device's `Mantle.Pool`.

`extra_usage` is passed through as the region's CONSTRAINT rather than used to
bypass the pool: `compatible` knows what a block may be used for, so an index
buffer sits in a block whose usage bits already permit it and falls out to a
dedicated allocation only when none does.
"""
function pool_alloc(bq::SubmitChannel{<:VulkanQueue}, nbytes::Integer; extra_usage::UInt32=UInt32(0))
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
    # and above it the pool's policy spaces the collections (`autocollect!`).
    # `drain!` after each is what makes it work at all: a buffer freed while the
    # GPU still referenced it went to the channel's retired list rather than back
    # to the pool, and until it is drained the memory is dead to everyone.
    let cap = sp.policy.soft_cap
        cap > 0 && reserved(sp) >= cap && autocollect!(() -> drain!(bq), sp)
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
Acquire through core's bounded reclamation/retry policy, then add Vulkan's
diagnostic to a final native OOM. Other errors propagate unchanged. An open
recording's scratch allocation passes `maywait = false` directly to core.
"""
# `dev` is untyped for the same reason `ctxof(bq)` is: `LavaDevice` is declared in
# `graph.jl`, which this file is included before. A signature annotation is
# evaluated at load; the body is not.
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

"""The driver's two out-of-memory codes; `overbudget` throws the device one too."""
isoutofmemory(err::VK.VulkanError) =
    err.code == VK.ERROR_OUT_OF_DEVICE_MEMORY || err.code == VK.ERROR_OUT_OF_HOST_MEMORY

"""
    reclaim_empty_pool_blocks!(bq::SubmitChannel{<:VulkanQueue}) -> (n_blocks::Int, bytes::Int)

Hand every block with nothing live in it back to the driver, and say how much
that was.

`Mantle.trim!` does the deciding — a block is returnable when its `live` ledger
is empty — so what is left here is measuring, which the caller reports.

Callers must have run `quiesce_before_reclaim!` first; see there for why. Not
finalizer-safe, and never was.
"""
function reclaim_empty_pool_blocks!(bq::SubmitChannel{<:VulkanQueue})
    ctx = ctxof(bq)
    sp = spans(ctx)
    before = reserved(sp)
    nbefore = length(poolblocks(ctx))
    trim!(sp, lavadevice(ctx))
    return (nbefore - length(poolblocks(ctx)), before - reserved(sp))
end

# Mapped transfers use host pointers. Other copies use pooled Readback memory:
# upload scratch belongs to its one-shot; download regions belong to the caller
# until completion, host read and release. Transfer storage is not shared in flight.

"""Largest download read straight out of mapped BAR memory; see `copy_buffer!`."""
const BAR_DIRECT_READ_BYTES = 4096

"""copy_buffer!(direction, managed, host_ptr, nbytes; offset=0)

Copy host/device bytes, with direction :upload or :download. Mapped uploads and
small mapped downloads memcpy after prior use completes. Other transfers copy
through pooled Readback regions; downloads use the buffer's last live channel.
The caller must preserve host_ptr storage through this call."""
function copy_buffer!(direction::Symbol, managed::VkManagedBuffer,
                     host_ptr::Ptr{UInt8}, nbytes::Integer; offset::Integer=0)
    nbytes == 0 && return
    @assert direction === :upload || direction === :download  "direction must be :upload or :download"
    # A guard against a host transfer inside a launch capture was here, deleted
    # 2026-09-15: it could not be replayed, because the host moves the bytes and
    # no command in the buffer does. A declared graph states a host write as an
    # update pass (`Update`, `hostwritten!`), which IS in the plan, so there is no
    # longer a way to ask for the unreplayable thing.
    buf_offset = pool_offset(managed) + Int(offset)

    # BAR fast-path: host-visible mapped memory — direct memcpy, no staging.
    # Not for a large DOWNLOAD: BAR memory is write-combined, so the host reads
    # it uncached — 71 ms for a 25 MB 1080p RGB Float32 frame on Strix Halo,
    # against 1.2 ms through HIP. A copy into the host-cached `Readback` kind
    # below costs one submission, which only a few-word read cannot repay.
    if managed.mapped_ptr != Ptr{UInt8}(0) &&
       (direction === :upload || nbytes <= BAR_DIRECT_READ_BYTES)
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
    # instead of twice. `stampof` is which channel that was.
    bq = if direction === :upload
        (managed.ctx::VkContext).default_bq
    else
        wbq = stampof(managed).channel
        (wbq !== nothing && !queue_released(wbq::SubmitChannel{<:VulkanQueue})) ?
            (wbq::SubmitChannel{<:VulkanQueue}) : (managed.ctx::VkContext).default_bq
    end
    # `Mantle.offset`, qualified: this function's `offset` keyword shadows it.
    if direction === :upload
        # The staging bytes belong to the one-shot: written before it is
        # submitted, given back by the sweep once the copy has passed. Nothing
        # waits — the next reader of `managed` on this queue is ordered behind
        # the copy, and a host reader waits on `last_write`.
        oneshot!(bq; tag = :upload) do e
            # Staged in host memory and moved by the copy engine; see `scratch!`.
            r = scratch!(e.owner, nbytes, Readback())
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
    upload_typed!(dst, host_data; offset)
end

"""Download data from a device-local buffer to a host byte vector."""
function download!(host_data::Vector{UInt8}, src::VkManagedBuffer; offset::Int=0)
    GC.@preserve host_data copy_buffer!(:download, src, pointer(host_data), length(host_data); offset)
end

"""Upload plain-bit host values. Dense Arrays transfer directly; other layouts
materialize once in logical order. The native transfer owns any GPU copy lifetime."""
function upload_typed!(dst::VkManagedBuffer, data::AbstractArray{T}; offset::Int=0) where T
    isbitstype(T) || throw(ArgumentError("upload_typed! requires an isbits element type, got $T"))
    src = data isa Array ? data : collect(data)
    GC.@preserve src copy_buffer!(:upload, dst, Ptr{UInt8}(pointer(src)),
                                  length(src) * sizeof(T); offset)
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

function find_memory_type(ctx::VkContext, type_bits::UInt32, required_flags;
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
