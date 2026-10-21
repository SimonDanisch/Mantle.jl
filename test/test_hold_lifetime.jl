# Recording holds, completion and deferred destruction, driven by a host timeline.

using Test
import Mantle

"""
The timeline a fake channel submits on: `next` is the last token handed out,
`completed` the last one the "device" has finished. `freed` is every object
`rawfree` was asked to destroy, in order.
"""
mutable struct HoldTimeline
    next::UInt64
    completed::UInt64
    freed::Vector{Any}
end
HoldTimeline() = HoldTimeline(UInt64(0), UInt64(0), Any[])

"""The device a fake channel belongs to: what `rawfree` is asked of."""
struct HoldDevice <: Mantle.Device
    timeline::HoldTimeline
end

"""An object with bytes of its own on the device: it carries a stamp."""
struct HeldBuffer
    stamp::Mantle.Stamp{UInt64}
end
HeldBuffer() = HeldBuffer(Mantle.Stamp{UInt64}())

"""A channel whose recordings are plain counters and whose tokens are `UInt64`s."""
const HoldChannel = Mantle.SubmitChannel{HoldTimeline,Int,UInt64,Mantle.Submission{Int}}

# The four primitives and two answers a backend supplies; everything the tests
# below exercise is core's.
Mantle.deviceof(ch::HoldChannel) = HoldDevice(ch.channel)
Mantle.passed(ch::HoldChannel, tok) = tok <= ch.channel.completed
Mantle.makerecording(::HoldChannel) = 0
Mantle.resetrecording!(::HoldChannel, r) = r
Mantle.recorder(ch::HoldChannel, r) = ch
Mantle.stampof(b::HeldBuffer) = b.stamp
Mantle.rawfree(d::HoldDevice, obj::HeldBuffer) = (push!(d.timeline.freed, obj); nothing)

# What a backend's `submit!` does with what it holds: hand the recording over and
# stamp everything the recording names with the token that covers it. The Vulkan
# backend stamps its one-shot's `sync` list; the hold frame is that list here.
function Mantle.submit!(ch::HoldChannel, r::Int)
    tl = ch.channel
    tl.next += UInt64(1)
    Mantle.stamp!(ch, tl.next, last(ch.holds))
    return tl.next
end

# How many recordings that can still submit hold this object.
holders(b::HeldBuffer) = @atomic b.stamp.holders

# Was `b` destroyed, and only once?
destroyed(tl::HoldTimeline, b) = count(x -> x === b, tl.freed) == 1

@testset "hold! — what a submission holds, and when it lets go" begin

@testset "no backend-side pinning mechanism" begin
    # Not "renamed": these name a decision the backend must not make.
    @test !isdefined(Mantle, :pin!)
    @test !isdefined(Mantle, :sync_access!)
    @test !isdefined(Mantle, :pin_leaves!)
    @test !isdefined(Mantle, :collectsync!)

    # What replaced it: core keeps the lists, the backend keeps the facts.
    @test fieldnames(Mantle.Stamp) == (:channel, :token, :holders)
    for f in (:outstanding, :free, :holds, :spare, :pending, :retiring)
        @test hasfield(Mantle.SubmitChannel, f)
    end

    # Nothing without device-visible bytes of its own is tracked, and asking is
    # not an error: that is what makes `hold!` usable for a pipeline.
    @test Mantle.stampof("not a buffer") === nothing
end

@testset "a held buffer survives a destroy request until its submission passes" begin
    tl = HoldTimeline()
    ch = HoldChannel(tl)
    a = HeldBuffer()
    @test holders(a) == 0

    # A recording, written but NOT submitted — the window the timeline cannot
    # see.
    r = Mantle.acquire!(ch)
    Mantle.hold!(ch, a)
    @test holders(a) == 1

    # The teardown that corrupts the recording if the hold is not honoured.
    Mantle.retire!(ch, a)

    # The request is recorded, not performed: nothing has been destroyed, and a
    # drain that asks now declines because something can still submit.
    Mantle.drain!(ch)
    @test isempty(tl.freed)
    @test holders(a) == 1

    # Submitted and still running: still kept.
    tok = Mantle.submit!(ch, r)
    Mantle.handover!(ch, tok, r)
    Mantle.drain!(ch)
    @test isempty(tl.freed)
    @test holders(a) == 1

    # The device passes it: the hold goes with the submission, and the destroy
    # that was owed runs.
    tl.completed = tok
    Mantle.drain!(ch)
    @test holders(a) == 0
    @test destroyed(tl, a)
end

@testset "an unheld buffer is destroyed at the first drain" begin
    tl = HoldTimeline()
    ch = HoldChannel(tl)
    b = HeldBuffer()
    Mantle.retire!(ch, b)
    # Nothing named it, so nothing defers it beyond the drain that runs the
    # requests: no submission, no wait.
    Mantle.drain!(ch)
    @test destroyed(tl, b)
end

@testset "a buffer is held as often as it is held, and let go together" begin
    tl = HoldTimeline()
    ch = HoldChannel(tl)
    c = HeldBuffer()

    Mantle.oneshot!(ch) do e
        for _ in 1:5
            Mantle.hold!(e, c)
        end
        # The hold list takes each one — they are references, balanced by
        # `unhold!`.
        @test holders(c) == 5
    end
    @test holders(c) == 5                    # submitted, not yet passed

    tl.completed = tl.next
    Mantle.drain!(ch)
    @test holders(c) == 0
end

@testset "a recording that is never submitted gives its holds back" begin
    # The path a recording takes when building it threw: it never reached the
    # device, so nothing had to outlive it, and a hold left behind would keep its
    # object "held by something that can still submit" for ever — a destroy that
    # never runs.
    tl = HoldTimeline()
    ch = HoldChannel(tl)
    d = HeldBuffer()
    @test_throws ErrorException Mantle.oneshot!(ch) do e
        Mantle.hold!(e, d)
        error("the recording failed to build")
    end
    @test holders(d) == 0
    @test isempty(ch.outstanding)

    Mantle.retire!(ch, d)
    Mantle.drain!(ch)
    @test destroyed(tl, d)
end

@testset "all ray-tracing stages retain captured arrays" begin
    arrays = [Float32[i] for i in 1:4]
    capture(a) = () -> a[1]
    pipe = Mantle.RayTracingPipeline(; raygen=capture(arrays[1]),
        closest_hit=(capture(arrays[2]),capture(arrays[3])),
        miss=capture(arrays[4]), any_hit=capture(arrays[1]))
    tl = HoldTimeline()
    ch = HoldChannel(tl)
    r = Mantle.acquire!(ch)
    Mantle.holdleaves!(ch,pipe)
    expected = arrays[[1,2,3,4,1]]
    @test length(last(ch.holds)) == length(expected)
    @test all(a === b for (a,b) in zip(last(ch.holds),expected))
    tok = Mantle.submit!(ch,r)
    Mantle.handover!(ch,tok,r)
    @test all(a === b for (a,b) in zip(only(ch.outstanding).payload.holds,expected))
    tl.completed = tok
    Mantle.drain!(ch)
    @test isempty(ch.outstanding)
    @test all(isempty,ch.spare)
end

end  # @testset
