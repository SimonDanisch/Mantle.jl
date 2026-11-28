# Core's submission channel, driven by a driver that is a few lines of Julia.
#
# `SubmitChannel` is core's (`graph/lifetime.jl`): what a submission holds, when a
# recording may be reused, when a requested destroy may run, and which thread
# may record. A backend supplies primitives — make a recording, submit it, say
# whether a token has passed — and nothing else. Every rule below was broken at
# least once, and each was first pinned on the Vulkan backend's queue, which
# only a Vulkan device could run. They are rules about core's bookkeeping, so they
# are asserted here against core with the driver replaced by counters: no device,
# and the same on every machine.
#
#   * A channel's sweep asks THAT channel's timeline (was `vulkan/test_sweep_per_queue.jl`).
#     A sweep that asked `passed(device, token)` read the DEFAULT queue's counter,
#     which runs far ahead of a queue carrying three submissions a frame
#     (RayMakie's graphics queue, which blits and draws overlays). The second
#     queue's submissions were swept the moment they were made, their one-shots
#     went back to the pool while the GPU was still executing them, the next
#     one-shot popped the SAME command buffer and began it again mid-flight, and
#     the overlay came out blank (`test_overlay_compositing.jl`: 0 green pixels)
#     or the device was lost. The device below answers "passed" for everything,
#     which is exactly the state the wrong comparison misreads.
#   * A build that throws leaves nothing in limbo (was the last testset of
#     `vulkan/test_crossqueue_sync.jl`): the recording goes back to the pool
#     unsubmitted, the claims it took are dropped, and the channel keeps working.
#     On Vulkan the throw had once happened after `vkEndCommandBuffer` and left a
#     closed command buffer half inside a failed submission, which surfaced as a
#     segfault in `vkCmdPipelineBarrier` three frames later.
#   * A channel belongs to the thread that made it (was
#     `vulkan/test_phase4_singlethread.jl`): recording into one channel from two
#     threads interleaves two command streams into one buffer, which is not a race
#     a driver reports — it is a corrupted recording — so the other thread is
#     refused with `WrongThread`, a type of its own that callers act on.
#   * A destroy requested while a submission still names the resource waits for
#     that submission (the core half of `vulkan/test_phase3_lifecycle.jl`): it was
#     two lists, a lock and a stamp on the backend's queue, and is `retire!` plus
#     `reclaim!` on core's channel.

using Test, Mantle
const M = Mantle

"""The driver half: tokens handed out, tokens finished, and what was destroyed."""
mutable struct ToyQueue
    next::UInt64
    done::UInt64
    freed::Vector{Any}
end

"""A recording this driver never executes; counts how often it was reused."""
mutable struct ToyRecording
    resets::Int
end

"""The device a channel belongs to, as `rawfree` and a device-level `passed` see it."""
struct ToyDevice
    q::ToyQueue
end

"""A resource with a lifetime stamp, as a backend's buffer carries one."""
struct ToyBuffer
    stamp::M.Stamp{UInt64}
end
ToyBuffer() = ToyBuffer(M.Stamp{UInt64}())

const ToyChannel = M.SubmitChannel{ToyQueue,ToyRecording,UInt64,M.Submission{ToyRecording}}
toychannel() = ToyChannel(ToyQueue(0, 0, Any[]))

M.makerecording(::ToyChannel) = ToyRecording(0)
M.resetrecording!(::ToyChannel, r::ToyRecording) = (r.resets += 1; r)
M.recorder(::ToyChannel, r::ToyRecording) = r
# A submission stamps what it names with its token, as a backend's `submit!`
# does for the resources its command buffer touches: here, what the open
# recording holds.
function M.submit!(ch::ToyChannel, ::ToyRecording)
    ch.channel.next += 1
    M.stamp!(ch, ch.channel.next, last(ch.holds))
    return ch.channel.next
end
M.passed(ch::ToyChannel, t) = t <= ch.channel.done
M.deviceof(ch::ToyChannel) = ToyDevice(ch.channel)
# The device is far ahead of this channel: every token it is asked about has
# passed. Only a sweep that asks the wrong timeline ever hears this.
M.passed(::ToyDevice, t) = true
M.rawfree(d::ToyDevice, b::ToyBuffer) = (push!(d.q.freed, b); nothing)
M.stampof(b::ToyBuffer) = b.stamp

@testset "a channel's sweep asks that channel's timeline" begin
    ch = toychannel()
    tok = M.oneshot!(ch) do e end
    r = only(o.payload.recording for o in M.outstanding(ch))
    # The misreading's premise: asked of the device, the token has passed; asked
    # of the channel that submitted it, it has not.
    @test M.passed(M.deviceof(ch), tok)
    @test !M.passed(ch, tok)

    # THE assertion: the sweep (`drain!` is what every path on a channel calls
    # before it opens or submits) gives nothing back, because on ITS timeline
    # nothing has passed. Before the fix the submission was swept, the recording
    # went to the pool, and the free list held a command buffer still in flight.
    M.drain!(ch)
    @test length(M.outstanding(ch)) == 1
    @test !any(x -> x === r, ch.free)

    # Let it through.
    ch.channel.done = tok
    M.drain!(ch)
    @test isempty(M.outstanding(ch))
    @test any(x -> x === r, ch.free)
end

@testset "a build that throws leaves nothing in limbo" begin
    ch = toychannel()
    buf = ToyBuffer()
    # One recording through the whole cycle, so the pool holds one.
    tok = M.oneshot!(ch) do e end
    ch.channel.done = tok
    M.drain!(ch)
    before = length(ch.free)
    @test before == 1
    r = only(ch.free)

    # The failure available between opening a recording and submitting it: the
    # body throws. `oneshot!` gives the recording back unsubmitted and drops the
    # claims it took, so nothing is left ended-but-unsubmitted and nothing stays
    # held by a submission that never happened.
    @test_throws ErrorException M.oneshot!(ch) do e
        M.hold!(ch, buf)
        error("build failure")
    end
    @test (@atomic buf.stamp.holders) == 0
    @test buf.stamp.channel === nothing          # nothing was submitted
    @test isempty(M.outstanding(ch))
    @test length(ch.free) == before              # the recording went back…
    @test only(ch.free) === r
    @test isempty(ch.holds)                      # …and no hold frame is left open

    # The channel still works afterwards, reusing that recording.
    tok2 = M.oneshot!(ch) do e
        M.hold!(ch, buf)
    end
    @test tok2 == tok + 1
    @test buf.stamp.channel === ch && buf.stamp.token == tok2
    @test r.resets == 2
end

@testset "a channel belongs to the thread that made it" begin
    ch = toychannel()
    @test ch.thread == Threads.threadid()
    if Threads.nthreads() > 1
        # `:static`, so every thread runs one iteration and at least one of them
        # is not the owner; `@spawn` may land on the owner's thread and prove
        # nothing. The refusal is captured as a value and asserted below.
        seen = Vector{Any}(nothing, Threads.nthreads())
        Threads.@threads :static for k in 1:Threads.nthreads()
            if Threads.threadid() == ch.thread
                seen[k] = :owner
            else
                seen[k] = try
                    M.oneshot!(ch) do e end
                    :not_refused
                catch err
                    err
                end
            end
        end
        others = filter(x -> x !== :owner, seen)
        @test !isempty(others)
        for err in others
            # A `WrongThread` and not an `AssertionError`: it names both threads,
            # and the editor discovers which thread owns the process device by
            # catching exactly this type.
            @test err isa M.WrongThread
            @test err.owner == ch.thread
            @test err.caller != ch.thread
        end
        # Refused before anything was opened.
        @test isempty(M.outstanding(ch))
        @test isempty(ch.holds)
    else
        @info "single-threaded session; the cross-thread refusal needs `julia -t 2` or more"
    end
end

@testset "a destroy waits for the submission that names the resource" begin
    ch = toychannel()
    buf = ToyBuffer()
    tok = M.oneshot!(ch) do e
        M.hold!(ch, buf)
    end
    @test buf.stamp.channel === ch && buf.stamp.token == tok
    @test (@atomic buf.stamp.holders) == 1      # held by the submission in flight

    # Requested now, from wherever: recorded, not run.
    M.retire!(ch, buf)
    M.drain!(ch)
    @test isempty(ch.channel.freed)

    # The submission passes: its hold goes, and the destroy runs — once.
    ch.channel.done = tok
    M.drain!(ch)
    @test (@atomic buf.stamp.holders) == 0
    @test length(ch.channel.freed) == 1 && only(ch.channel.freed) === buf
    M.drain!(ch)
    @test length(ch.channel.freed) == 1

    # A resource no submission ever named is destroyed at the next reclaim.
    fresh = ToyBuffer()
    M.retire!(ch, fresh)
    M.drain!(ch)
    @test length(ch.channel.freed) == 2 && ch.channel.freed[2] === fresh
end
