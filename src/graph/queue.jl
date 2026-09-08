# The submission queue.
#
# It was 41 fields, of which TEN name a driver object and thirty-one did not. The
# thirty-one were an argument-slab ring allocator, a deferred-free list, barrier
# elision state, capture/replay watermarks and submission thresholds — machinery
# every backend needs and none of it Vulkan's. It was `VulkanBatchQueue`, so a
# Metal backend would have written all of it again. Sixteen of them are gone
# outright rather than shared: they existed because something recorded GPU work
# without a plan, and a plan says what they were guessing. The last of those —
# the open command buffer itself, with its pools of batches and segments, its
# split and submit thresholds, its dedicated acceleration-structure buffer and
# fence, and its staging buffer — went with it: nothing is open on a queue
# between calls, and what a caller closed is submitted when it is closed.
#
# The driver objects are type parameters, and the field NAMES are unchanged, so
# the places that say `bq.device` or `bq.timeline_sem` still read the same. The
# Vulkan backend keeps `const VulkanBatchQueue{C} = BatchQueue{VK.Device, …, C}`
# so even its own spelling survives.

"""
    BatchQueue

An independent command submission channel owning a Vulkan queue, a command pool
and what it has in flight. Multiple `BatchQueue`s can record and submit
independently (e.g., primary queue for graphics/present, compute queue for
async RT).

Create with `BatchQueue(device, queue, queue_family_index)`.
"""
# `T` is the backend's submission-token type (Vulkan: a UInt64 timeline value)
# and `B` its submission, what the sweep hands to `recycle!` once the token has
# passed. Parameters rather than an abstract `Outstanding` eltype because the
# latter boxes every token a sweep reads — 80 bytes of every submission.
mutable struct BatchQueue{D,Q,P,B,O,S,C,T}
    device::D
    queue::Q
    family_index::UInt32
    cmd_pool::P
    # Pooled, so a submission and the one-shot an ad hoc launch takes allocate
    # nothing beyond the scratch region the launch acquires.
    free_submissions::Vector{B}
    free_oneshots::Vector{O}
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
    # Back-reference to owning VkContext.
    #
    # `::C`, a TYPE PARAMETER, not `::Any`. `VkContext` is declared after this
    # struct, so the field cannot name it directly — that ordering is the only
    # reason it was ever untyped. A parameter closes the cycle without needing
    # the name: `VkContext` holds a `BatchQueue{VkContext}`, exactly the shape
    # `struct Node; next::Vector{Node}; end` already uses.
    #
    # Untyped, `bq.ctx.caches.<anything>` inferred as `Any`, which made the
    # launch-plan lookup a dynamic dispatch and its loop a dynamic ITERATION:
    # **464 bytes of allocation on every dispatch**, on a warm cache that builds
    # nothing. The workaround was `bq.ctx::VkContext` written at eight separate
    # call sites, and the ninth (the plan lookup) simply forgot it. A parameter
    # makes it structural — there is no site left that can forget.
    ctx::C
    # Single-writer invariant: only this thread may record into or submit
    # from this BatchQueue.  Captured at construction from `Threads.threadid()`.
    # Every submit / sweep / scratch-alloc entry point asserts that it is
    # running on this thread — an accidental cross-thread call trips the assert
    # immediately instead of silently corrupting state.
    owning_thread::Int
    # How long `flush!` waits before it decides a dispatch is not completing.
    # The submit and split thresholds stood beside it and are gone with the
    # open command buffer they paced: what is handed to the driver is what a
    # caller closed, whole, and it goes at once.
    flush_timeout_ns::UInt64
    # Everything handed to the device from this queue and not yet known finished,
    # oldest first: the token, and the submission it covers — the one-shots it
    # owns and the recordings it pinned — which the sweep gives back in order.
    # See `graph/submission.jl`: `flush!` waits for `newest`; `waitfor!(plan)`
    # asks whether the token covering its last run has `passed`; a recording is
    # a submission like any other. Nothing else sits on the queue between calls:
    # there is no open command buffer, no list of closed-but-unsubmitted work
    # and no threshold deciding the fate of either; `submit!` hands over what it
    # is given, the moment it is called.
    outstanding::Vector{Outstanding{T,B}}
    # What the last dispatch on this queue was, for the dispatch log and for
    # DEVICE_LOST diagnostics. Process-wide, these attributed one queue's crash
    # to another queue's kernel.
    last_dispatch_info::String
    prev_dispatch_info::String
    # Which hardware queue of `family_index` this one drives, so
    # `release_batch_queue!` can hand the slot back. -1 for the primary queue and
    # for any queue that had to share it because the family ran out.
    queue_index::Int
    # Per-queue scratch a backend's fast paths refill rather than allocate:
    # Vulkan hangs a `QueueSlots` here (semaphore / wait-info / counter cells
    # for the wait and query calls a sweep makes). `Any` because the type is
    # the backend's and declared later; it is a heap object, so reading the
    # field hands out a reference, not a box.
    slots::Any
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

The queue `device` records and submits on.

The portable spelling of what was `Mantle.vk_context().default_bq`: a global
lookup, in the backend, from a package that is not supposed to know which backend
it has. A caller holds a device, and a device holds its queue.

No default. A backend with no queue should say so through
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
    submit!(bq, closed...) -> token

Hand closed command buffers to the device, in one submission, in the order
given, and answer with the token that covers them.

The one way GPU work reaches the driver, and it goes the moment it is called:
there is nothing on a queue between calls but what is in flight. A backend's
queue implements it — the Vulkan one takes a plan's `Recording` and the
`OneShot`s every other path closes inside one call, plus the semaphores and
fence a present needs. It replaced the `submit!(device)` hook, "send whatever
the backend has recorded but not yet submitted", which had nothing left to
send once nothing was ever left recorded.
"""
function submit! end

"""
    flush!(bq, device)

Wait until the device has finished everything submitted on `bq`.

It used to submit first, because a queue held an open batch that nothing had
handed over; there is nothing to hand over now, so this is `waitfor!` on the
newest submission and nothing else.
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

Block until this device has finished everything it has been given.

The blunt instrument, for teardown and for reading back a resource whose
producer was submitted on a queue the reader does not track. To wait for ONE
plan's last run, [`waitfor!`](@ref). Everything a caller records is submitted
the moment it is closed, so there is nothing to hand over first.
"""
function waitidle end
