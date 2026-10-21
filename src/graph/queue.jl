# Legacy channel lifecycle; shared state lives in SubmitChannel (graph/lifetime.jl).
# Backends provide native queues and submission primitives.

"""
    allocate_batch_queue!(device) -> channel

Allocate a channel with its own submission state. Native queues may be shared when
Vulkan queue slots run out. backend(channel) launches on it; cross-channel resource
uses are ordered. Return it with release_batch_queue!.
"""
function allocate_batch_queue! end

"""
    supports_batch_queue(backend) -> Bool

Whether allocate_batch_queue! is supported; independent of graphics support.
Unsupported backends refuse allocation with ArgumentError.
"""
supports_batch_queue(backend) = false
# Asked of a device too: this backend answers for `VulkanAPI`, and a caller
# holding the device would otherwise read the untyped `false`. See
# `supports_graphics` in `graphics/commands.jl` for the class.
supports_batch_queue(dev::Device) = supports_batch_queue(backend(dev))

"""
    batchqueue(device) -> channel

Return the device's default submission channel. No ambient device lookup or fallback;
query supports_batch_queue before asking an unsupported backend.
"""
function batchqueue end


"""
    release_batch_queue!(bq)

Drain and release a non-primary channel. Repeated release is a no-op; releasing the
device's primary channel must be refused.
"""
function release_batch_queue! end

"""
    submit!(bq, closed...) -> token

Submit closed command buffers immediately in the given order and return their token.
No open batch is retained for a later flush. Lifetime registration is separate: handover!.
"""
function submit! end

"""
    flush!(bq, device)

Wait for everything already submitted on bq; do not submit additional work.
"""
function flush! end

# Resolve the owning device from the channel.
flush!(ch::SubmitChannel) = flush!(ch, deviceof(ch))

# Flush the default channel; waitidle additionally covers all other device work.
flush!(d::Device) = flush!(batchqueue(d))

"""
    waitidle(device)

Wait for all device work, including channels not tracked by a particular reader.
Use waitfor!(plan) when only one plan's last run needs to complete.
"""
function waitidle end
