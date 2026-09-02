# The submission queue.
#
# 41 fields, of which TEN name a driver object and thirty-one do not. The thirty-one
# are an argument-slab ring allocator, a deferred-free list, barrier elision
# state, capture/replay watermarks and submission thresholds — machinery every
# backend needs and none of it Vulkan's. It was `VulkanBatchQueue`, so a Metal
# backend would have written all of it again.
#
# The ten are type parameters, and the field NAMES are unchanged, so the 115
# places that say `bq.device` or `bq.timeline_sem` still read the same. The
# Vulkan backend keeps `const VulkanBatchQueue{C} = BatchQueue{VK.Device, …, C}`
# so even its own spelling survives.

"""
    BatchQueue

An independent command submission channel owning a Vulkan queue, command pool,
and batch state. Multiple `BatchQueue`s can record and submit independently
(e.g., primary queue for graphics/present, compute queue for async RT).

Create with `BatchQueue(device, queue, queue_family_index)`.
"""
mutable struct BatchQueue{D,Q,P,B,CB,F,S,C}
    device::D
    queue::Q
    family_index::UInt32
    cmd_pool::P
    active_batch::Union{Nothing,B}
    in_flight::Vector{B}
    free_batches::Vector{B}
    free_cmd_bufs::Vector{CB}
    # Dedicated AS-build command buffer + fence — allocated from this BQ's
    # own cmd_pool and submitted on this BQ's queue.  Keeping them on the
    # BQ (not the VkContext) means AS build, submit and queue are locked
    # together by construction.
    as_cmd_buf::CB
    as_fence::F

    # ── Explicit-queue refactor additions ────────────────────────────────
    # One timeline semaphore per queue.  Each submit signals next_timeline+1.
    timeline_sem::S
    next_timeline::UInt64
    # Buffers queued for destruction once their last_write timeline value
    # is reached. Drained by drain_deferred_frees! at natural sync points.
    # Loose type (VkManagedBuffer is declared later in memory.jl).
    #
    # Cross-thread: finalizer threads push into this list via `vk_free!`;
    # the main thread iterates + drains via `drain_deferred_frees!`.  The
    # `deferred_frees_lock` below guards both operations on `deferred_frees`
    # AND `deferred_as_frees`.  SpinLock because contention is near-zero
    # (finalizer pushes at GC pauses, drain happens at sync points).
    deferred_frees::Vector{Any}
    # LavaBLAS / LavaTLAS queued for destruction once their `last_use`
    # timeline value is reached. Drained by `drain_deferred_as_frees!`.
    # Loose type — Lava AS types are declared later in raytracing/acceleration.jl.
    deferred_as_frees::Vector{Any}
    # Guards `deferred_frees` AND `deferred_as_frees`.  Acquired on every
    # push from finalizer threads and on every drain from the main thread.
    deferred_frees_lock::Base.Threads.SpinLock
    # Per-BQ argument-buffer slab pool.  Each submit bump-allocates from
    # the current slab; `reset_arg_buffer_pool!(bq)` (called from
    # reclaim_batch! once in_flight is empty) rewinds the bump pointer.
    # Element type is `LavaArray{UInt8,1}` (unified/BAR memory); kept loose
    # because LavaArray is declared later in array/lavaarray.jl.
    arg_slabs::Vector{Any}
    arg_slab_idx::Int
    arg_slab_offset::Int
    arg_alloc_count::Int
    # Timeline value the GPU must reach before the pool may be rewound: the
    # newest batch that allocated from it. 0 = nothing outstanding. Rewinding
    # earlier hands the next caller bytes an in-flight shader is still reading,
    # since a dispatch's arg address is baked into its command buffer as a push
    # constant (see `arg_pool_in_use!`).
    arg_pool_frontier::UInt64
    # Per-BQ indirect-dispatch buffer slab pool.  Element type is
    # `LavaArray{UInt32,1}` (unified + INDIRECT_BUFFER_BIT).  Reset by
    # `reset_indirect_buffer_pool!(bq)`.
    indirect_slabs::Vector{Any}
    indirect_slab_idx::Int
    indirect_slab_offset::Int
    # Per-BQ staging buffer for CPU↔GPU transfers. A single VkManagedBuffer
    # that grows as needed via get_staging!. Reused across transfers.
    # Loose type — VkManagedBuffer is declared later in memory.jl.
    staging::Union{Nothing, Any}
    # Back-reference to owning VkContext.
    #
    # `::C`, a TYPE PARAMETER, not `::Any`. `VkContext` is declared ~280 lines
    # below this struct, so the field cannot name it directly — that ordering is
    # the only reason it was ever untyped. A parameter closes the cycle without
    # needing the name: `VkContext` holds a `BatchQueue{VkContext}`, exactly the
    # shape `struct Node; next::Vector{Node}; end` already uses.
    #
    # Untyped, `bq.ctx.caches.<anything>` inferred as `Any`, which made the
    # launch-plan lookup a dynamic dispatch and its loop a dynamic ITERATION:
    # **464 bytes of allocation on every dispatch**, on a warm cache that builds
    # nothing. The workaround was `bq.ctx::VkContext` written at eight separate
    # call sites, and the ninth (the plan lookup) simply forgot it. A parameter
    # makes it structural — there is no site left that can forget.
    #
    # The previous comment claimed this could be `nothing` "during the brief
    # window of default_bq construction". It cannot: `VkContext`'s inner
    # constructor is two-phase via `new()` precisely so a live `ctx` exists
    # before `BatchQueue(...)` is called, and every call site passes one.
    ctx::C
    # Single-writer invariant: only this thread may record into or submit
    # from this BatchQueue.  Captured at construction from `Threads.threadid()`.
    # Every dispatch-recording / sweep / slab-alloc entry point asserts that
    # it is running on this thread — an accidental cross-thread call trips
    # the assert immediately instead of silently corrupting state.
    owning_thread::Int

    # ── Recording policy. These were six module-level `Ref`s, which made them
    # process-wide settings for something that is per queue: two BatchQueues on
    # one device already disagree about how much work to batch before submitting,
    # and a second device made it worse. They are still mutable defaults — that
    # is what they are for — but they are now this queue's.
    #
    # `auto_submit_threshold` at 64 rather than 0 is the +44% measured in
    # `perf-plan.md`: at 0, recording and execution never overlapped.
    auto_submit_threshold::Int
    cb_split_threshold::Int
    flush_timeout_ns::UInt64
    barrier_mode::Symbol
    barrier_elision::Bool
    # One-shot, consumed by exactly the next dispatch on THIS queue.
    next_skip_barrier::Bool
    # How many nested "do not end this command buffer here" scopes are open.
    # Nonzero means a construct is recording that spans several commands and
    # cannot survive a split or a submission in the middle — a conditional
    # rendering scope is the one that exists. See `unsplittable`.
    scope_depth::Int
    # Set by the KA launch path for the dispatch it is about to record: "this
    # dispatch enumerated its buffers, so the elision tracker saw everything it
    # touches". Same one-shot shape as `next_skip_barrier`, and it was a global
    # for the same reason — the hand-off is launch → `record_dispatch!` and both
    # already have the queue.
    ranges_declared::Bool
    # The elision tracker itself. `touched_ranges` accumulates what recent
    # dispatches in the current batch wrote; `dispatch_ranges` is scratch for the
    # dispatch being recorded. Reused, never reallocated — and per queue, because
    # two queues recording concurrently into their own command buffers were
    # sharing one tracker, so a range written on one could elide a barrier on the
    # other.
    touched_ranges::Vector{UInt64}
    dispatch_ranges::Vector{UInt64}
    # Non-`nothing` inside `concurrent_indirect_group`: dispatches append here
    # instead of recording, and the group's flush fuses them.
    deferred_indirect::Union{Nothing,Vector{Any}}
    # The `CapturedSequence` being recorded on THIS queue, or `nothing`. It was a
    # module-level `Ref`, so every site that used it had to re-check `cap.bq ===
    # bq` to find out whether the capture was even this queue's — and
    # `cb_begin_flags` had no queue to check with, so a capture running on one
    # queue silently made every OTHER queue's command buffers reusable.
    # `Any` because `CapturedSequence` is declared in `command.jl`; use sites
    # assert it, the same shape as `ctx`.
    capturing::Any
    # Everything handed to the device from this queue and not yet known finished,
    # oldest first. See `graph/submission.jl`.
    #
    # This was `replay_watermark::UInt64` — "highest timeline value signalled by a
    # replay" — which existed because a replay puts no `CommandBatch` in
    # `in_flight`, so `flush!` scanning `in_flight` alone returned before the GPU
    # had run any of it. That is one record of outstanding work per SUBMISSION
    # PATH, and there were two paths, so there were two records and every consumer
    # had to remember both.
    #
    # One list, and a token per entry that the backend understands. `flush!` waits
    # for `newest`; `nextslot!` asks whether the token that last used its slot has
    # `passed`; a replay is a submission like any other. Nothing folds over two
    # lists looking for a maximum any more, and a third submission path would be
    # recorded here without touching a single consumer.
    outstanding::Vector{Outstanding}
    # What the last dispatch on this queue was, for the dispatch log and for
    # DEVICE_LOST diagnostics. Process-wide, these attributed one queue's crash
    # to another queue's kernel.
    last_dispatch_info::String
    prev_dispatch_info::String
    # Which hardware queue of `family_index` this one drives, so
    # `release_batch_queue!` can hand the slot back. -1 for the primary queue and
    # for any queue that had to share it because the family ran out.
    queue_index::Int
end

# ── The queue's lifecycle verbs ──────────────────────────────────────────────
#
# `BatchQueue` above is one shared struct; these are the four things a caller
# does to one, and they are declared here for the same reason the struct is
# shared: getting a queue, opening a batch on it, submitting it and giving it
# back are what every backend does, not what Vulkan does. They lived in
# `src/vulkan/runtime/` and were reached as `Mantle.allocate_batch_queue!` by
# RayMakie, which meant a name that only resolved when a Vulkan driver was
# present — the package did not load on a Mac at all.
#
# Implemented per backend. Vulkan hands out real `VkQueue`s from the family and
# falls back to sharing the primary queue when the device runs out; Metal has
# one `MTLCommandQueue` per device and distinguishes batches rather than queues.

"""
    allocate_batch_queue!(device) -> BatchQueue

Get a queue that records and submits independently of every other one.

Callers that need their own submission order — a graphics queue that must not
interleave with compute, a present queue — take one of these rather than
sharing the device's primary queue. Give it back with
[`release_batch_queue!`](@ref).
"""
function allocate_batch_queue! end

"""
    supports_batch_queue(backend) -> Bool

Can this backend hand out a [`BatchQueue`](@ref)?

A SEPARATE question from [`supports_graphics`](@ref), and the two now differ.
`BatchQueue` is a command-pool, fence and timeline-semaphore arrangement — a
Vulkan shape — and a backend can rasterise perfectly well without one: Metal
compiles vertex and fragment programs and records draws on its own command
queue, but has no command pool to lend and no fence to hand back.

So a caller that needs to RECORD INTO a batch queue (RayMakie's overlay
compositing is the one in tree) has to ask this, not `supports_graphics`.
Answering only the second question sends it down a path whose first call is
`allocate_batch_queue!`, which on Metal deliberately throws.

`false` by default: a backend that has one says so.
"""
supports_batch_queue(backend) = false

"""
    batchqueue(device) -> BatchQueue

The queue a `custom!` pass on `device` records into.

The portable spelling of what was `Mantle.vk_context().default_bq`, which is how
Hikari's hardware ray-tracing pass reached its queue: a global lookup, in the
backend, from a package that is not supposed to know which backend it has. A
`custom!` body already holds the graph, and the graph holds the device, so the
queue was one hop away the whole time — and after the runtime moved into an
extension the global was not reachable at all.

No default. A backend with no queue to record into should say so through
[`supports_batch_queue`](@ref) and be asked that question first; a fallback here
would answer "no queue" as some other value and fail further away.
"""
function batchqueue end


"""
    release_batch_queue!(bq)

Return `bq` to the device. Drains it first. Releasing twice is a no-op, and a
backend must refuse to release the device's primary queue.
"""
function release_batch_queue! end

"""
    ensure_active_batch!(bq) -> batch

Open a batch on `bq` if none is recording, and return it.

A backend must refuse to open one on a lost device: without that gate, work
keeps being recorded into batches that will never run, and the resource leak
plus the flood of follow-on failures hides whatever killed the device.
"""
function ensure_active_batch! end

"""
    flush!(bq, device)

Submit everything `bq` has recorded, and wait until the device has finished it.

The docstring said "does not wait — use `waitidle` for that", and the only
implementation has always waited: it submits and then blocks on the timeline
until the newest submitted value is signalled. Believing the docstring is how
`waitidle(::LavaDevice)` came to be `vkDeviceWaitIdle` alone, which waits for
submitted work and therefore not for the batch the caller was still holding.

To submit without waiting, `submit!(bq)`.
"""
function flush! end

# The queue already knows its device, so the one-argument form is the portable
# spelling and needs no backend method. Callers used to reach for
# `Mantle.vk_device()` to fill the second argument, which is exactly the kind of
# driver-shaped hole this file exists to close.
flush!(bq::BatchQueue) = flush!(bq, bq.device)

# The submission list this queue keeps — see `graph/submission.jl`.
outstanding(bq::BatchQueue) = bq.outstanding

"""
    waitidle(device)

Hand over everything this device's queue is still holding, then block until it
has finished all of it.

"Everything SUBMITTED to it" is what this said, and it is the weaker contract
that made the Vulkan method wrong: a headless plan submits when something asks
it to, so a `run!` sits in an open batch, and a wait that skips it returns
before the device has been told the work exists. The caller cannot tell from the
outside — the readback that usually follows flushes on its own — so what broke
was the caller who wanted the handover itself, to keep one submission from
growing past the driver's timeout.

The blunt instrument, for teardown and for reading back a resource whose
producer was submitted on a queue the reader does not track. To wait for ONE
plan's last run, [`waitfor!`](@ref).
"""
function waitidle end
