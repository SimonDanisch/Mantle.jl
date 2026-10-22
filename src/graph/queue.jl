# Submission channel lifecycle. State lives in `SubmitChannel` (graph/lifetime.jl);
# backends provide the native queues and submission primitives.

"""
    allocate_batch_queue!(device) -> channel

Allocate a channel with its own submission state; the native queue may be shared
when Vulkan queue slots run out. Cross-channel resource uses are ordered. Give it
back with `release_batch_queue!`.
"""
function allocate_batch_queue! end

"""
    supports_batch_queue(backend) -> Bool

Whether `allocate_batch_queue!` is supported (independent of graphics support);
otherwise allocation throws `ArgumentError`.
"""
supports_batch_queue(backend) = false
# Backends answer for their backend type; a device forwards so it does not hit
# the untyped `false`.
supports_batch_queue(dev::Device) = supports_batch_queue(backend(dev))

"""
    batchqueue(device) -> channel

The device's default submission channel. No fallback: check `supports_batch_queue`
first.
"""
function batchqueue end


"""
    release_batch_queue!(bq)

Drain and release a channel from `allocate_batch_queue!`. Releasing twice is a
no-op; the device's primary channel must be refused.
"""
function release_batch_queue! end

"""
    submit!(bq, closed...) -> token

Submit closed command buffers immediately, in order, and return their token.
Nothing is batched for a later flush; lifetime registration is `handover!`.
"""
function submit! end

"""
    flush!(bq, device)

Wait for everything already submitted on `bq`; submits nothing.
"""
function flush! end

flush!(ch::SubmitChannel) = flush!(ch, deviceof(ch))

# Only the default channel; `waitidle` covers all device work.
flush!(d::Device) = flush!(batchqueue(d))

"""
    waitidle(device)

Wait for all device work on every channel. Use `waitfor!(plan)` to wait for one
plan's last run.
"""
function waitidle end
