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
# What is left of it, after phases 2.2 and 2.3, is `SubmitChannel` in
# `graph/lifetime.jl`: the submission list, the hold lists, the retired list and
# the recording pool, with the whole driver arrangement behind one opaque field.
# This file is now only the VERBS — get a channel, submit on it, flush it, give
# it back — which are what every backend answers and none of them Vulkan's.

# ── The queue's lifecycle verbs ──────────────────────────────────────────────
#
# Four things a caller does to a channel, declared here for the reason
# `SubmitChannel` itself is shared: getting a channel, submitting on it,
# flushing it and giving it back are what every backend does, not what Vulkan
# does. They lived in
# `src/vulkan/runtime/` and were reached as `Mantle.allocate_batch_queue!` by
# RayMakie, which meant a name that only resolved when a Vulkan driver was
# present — the package did not load on a Mac at all.
#
# Implemented per backend. Vulkan hands out real `VkQueue`s from the family and
# falls back to sharing the primary queue when the device runs out; Metal has
# one `MTLCommandQueue` per device and distinguishes batches rather than queues.

"""
    allocate_batch_queue!(device) -> SubmitChannel

Get a queue that records and submits independently of every other one.

Callers that need their own submission order — a graphics queue that must not
interleave with compute, a present queue — take one of these rather than
sharing the device's primary queue. Give it back with
[`release_batch_queue!`](@ref).
"""
function allocate_batch_queue! end

"""
    supports_batch_queue(backend) -> Bool

Can this backend hand out a [`SubmitChannel`](@ref)?

A SEPARATE question from [`supports_graphics`](@ref), and the two now differ.
A channel is a command-pool, fence and timeline-semaphore arrangement on
Vulkan, and a backend can rasterise perfectly well without one: Metal
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
    batchqueue(device) -> SubmitChannel

The queue `device` records and submits on.

The portable spelling of what was `Mantle.vk_context().default_bq`: a global
lookup, in the backend, from a package that is not supposed to know which backend
it has. A caller holds a device, and a device holds its channel.

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

# The channel already knows its device (`deviceof`), so the one-argument form is
# the portable spelling and needs no backend method. Callers used to reach for
# `Mantle.vk_device()` to fill the second argument, which is exactly the kind of
# driver-shaped hole this file exists to close.
flush!(ch::SubmitChannel) = flush!(ch, deviceof(ch))

"""
    waitidle(device)

Block until this device has finished everything it has been given.

The blunt instrument, for teardown and for reading back a resource whose
producer was submitted on a queue the reader does not track. To wait for ONE
plan's last run, [`waitfor!`](@ref). Everything a caller records is submitted
the moment it is closed, so there is nothing to hand over first.
"""
function waitidle end
