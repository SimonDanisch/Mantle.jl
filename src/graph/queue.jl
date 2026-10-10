# The submission queue's lifecycle verbs: get a channel, submit on it, flush it,
# give it back. The state they act on is `SubmitChannel` in `graph/lifetime.jl`
# — the submission list, the hold lists, the retired list and the recording pool
# — with the driver arrangement behind one opaque field.
#
# Declared here because every backend answers them. Vulkan hands out real
# `VkQueue`s from the family and falls back to sharing the primary queue when the
# device runs out; Metal hands out another `MTLCommandQueue` with Metal.jl's batch
# over it, a `Metal.BatchedCommandQueue`, the same type as the device's own.

"""
    allocate_batch_queue!(device) -> channel

Get a channel that submits independently of every other one: on Vulkan a
[`SubmitChannel`](@ref) over another `VkQueue`, on Metal another command queue.

Callers that need their own submission order — a graphics queue that must not
interleave with compute, a present queue, a second stream of launches — take one
of these rather than sharing the device's primary queue. `backend(channel)` is
the KernelAbstractions backend that launches on it; a buffer used on two
channels is ordered between them (`test_crossqueue_sync.jl`). Give it back with
[`release_batch_queue!`](@ref).
"""
function allocate_batch_queue! end

"""
    supports_batch_queue(backend) -> Bool

Can this backend hand out a second submission channel
([`allocate_batch_queue!`](@ref))?

A SEPARATE question from [`supports_graphics`](@ref): a channel is a stream of
submissions, and rasterising needs none of its own. Vulkan and Metal both answer
`true`; a backend that cannot refuses `allocate_batch_queue!` with an
`ArgumentError` rather than handing back a stub.

`false` by default: a backend that has one says so.
"""
supports_batch_queue(backend) = false
# Asked of a device too: this backend answers for `VulkanAPI`, and a caller
# holding the device would otherwise read the untyped `false`. See
# `supports_graphics` in `graphics/commands.jl` for the class.
supports_batch_queue(dev::Device) = supports_batch_queue(backend(dev))

"""
    batchqueue(device) -> channel

The queue `device` records and submits on: a [`SubmitChannel`](@ref) on Vulkan,
Metal.jl's batch over the device's command queue on Metal.

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

Nothing is submitted first: a queue holds no open batch, so this is `waitfor!`
on the newest submission and nothing else.
"""
function flush! end

# The channel already knows its device (`deviceof`), so the one-argument form is
# the portable spelling and needs no backend method: a caller reaching for
# `Mantle.vk_device()` to fill the second argument is the driver-shaped hole this
# file exists to close.
flush!(ch::SubmitChannel) = flush!(ch, deviceof(ch))

# And the device's own queue, for a caller that holds a device rather than a
# channel. This is what `vk_flush!(ctx)` was: a backend spelling of "flush this
# device's default queue", reached through `vk_context()` at 122 test sites.
# `batchqueue` and `flush!` are both already vocabulary, so this needs nothing
# from a backend. NOT `waitidle`, which also does a full device wait — that is a
# stronger promise and a different verb.
flush!(d::Device) = flush!(batchqueue(d))

"""
    waitidle(device)

Block until this device has finished everything it has been given.

The blunt instrument, for teardown and for reading back a resource whose
producer was submitted on a queue the reader does not track. To wait for ONE
plan's last run, [`waitfor!`](@ref). Everything a caller records is submitted
the moment it is closed, so there is nothing to hand over first.
"""
function waitidle end
