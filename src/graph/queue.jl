# Submission channel lifecycle. State lives in `SubmitChannel` (graph/lifetime.jl);
# backends provide the native queues and submission primitives.

"""
    allocate_batch_queue!(device) -> channel

Allocate a channel with its own submission state; the native queue may be shared
when Vulkan queue slots run out. Cross-channel resource uses are ordered. Give it
back with `release_batch_queue!`. The device keeps the channels it handed out in
its `channels` field until then.
"""
function allocate_batch_queue!(d::Device)
    supports_batch_queue(d) || throw(ArgumentError(
        "allocate_batch_queue!: $(typeof(d)) has no channels besides its own"))
    refuseretired(d)
    ch = newchannel(d)
    push!(d.channels, ch)
    return ch
end

"""
    newchannel(device) -> channel
    closechannel!(device, channel)

The backend's halves of [`allocate_batch_queue!`](@ref) and
[`release_batch_queue!`](@ref): a new channel, and what giving one back releases
once its work is done (a hardware queue slot, a submission count).
"""
function newchannel end
function closechannel! end

"""
    ownerof(channel) -> device or nothing

The live device whose own channel `channel` is, or that handed it out and has not
had it back; `nothing` for a released channel.
"""
function ownerof(ch)
    for slot in values(DEVICES), r in slot.live
        d = r.value
        # A device without channels owns none.
        (d isa Device && supports_batch_queue(d)) || continue
        (batchqueue(d) === ch || any(c -> c === ch, d.channels)) && return d
    end
    return nothing
end

"""
    released(channel) -> Bool

Whether a channel from `allocate_batch_queue!` was given back. A released channel
must not be used or asked: a buffer's stamp may still name it after its
timeline is gone. It has no work in flight, so nothing naming it needs a wait.
"""
released(ch) = ownerof(ch) === nothing

"""
    supports_batch_queue(backend) -> Bool

Whether `allocate_batch_queue!` is supported (independent of graphics support);
otherwise allocation throws `ArgumentError`.
"""
supports_batch_queue(backend) = false

"""
    batchqueue(device) -> channel

The device's default submission channel. No fallback: check `supports_batch_queue`
first.
"""
function batchqueue end


"""
    release_batch_queue!(bq)

Drain and release a channel from `allocate_batch_queue!`, from its owning thread
once nothing will record on it again. Releasing twice is a no-op; the device's own
channel is refused. On a retired device nothing is waited for.
"""
function release_batch_queue!(ch)
    d = ownerof(ch)
    d === nothing && return nothing
    ch === batchqueue(d) && throw(ArgumentError(
        "release_batch_queue!: this is the device's own channel, which is not released; " *
        "only a channel from `allocate_batch_queue!` is"))
    i = findfirst(c -> c === ch, d.channels)
    retired(d) || flush!(ch)
    deleteat!(d.channels, i)
    closechannel!(d, ch)
    return nothing
end

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
