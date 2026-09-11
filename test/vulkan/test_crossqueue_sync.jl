"""
A buffer written on one channel and then used on another has to wait for the
first, and the wait carries a sync2-typed stage flag.
`Vulkan.PIPELINE_STAGE_2_ALL_COMMANDS_BIT` is — despite the `_2_` in its name —
a *sync1* `PipelineStageFlag`, so pushing it threw a `MethodError` on every
cross-queue wait there has ever been.

It did not present as a MethodError. The push happened *after*
`vkEndCommandBuffer`, so the throw left a closed command buffer stuck half
inside the submission that failed. This is how the underlying bug actually
surfaced: a segfault inside `vkCmdPipelineBarrier` while SAM 2's weights
uploaded in the video editor, three frames removed from the real bug, and only
once enough dispatches had queued up to put a submit in the middle of what the
caller thought was still being recorded.

Who DECIDES the wait has moved: core derives it from the stamps it writes
(`crosswaits!` in `graph/lifetime.jl`), and this backend lowers a (channel,
token) pair to a timeline semaphore and a value. What is still checked here is
the same two halves: that a buffer crossing channels is ordered at all, and
that a call that throws between opening a recording and submitting it leaves
nothing in limbo — the recording goes back to the pool with its claims dropped,
and the channel keeps working.
"""

using Test, Lava, KernelAbstractions
const KA = KernelAbstractions

@kernel function xqfill!(a, v)
    i = @index(Global)
    @inbounds a[i] = v
end

@testset "cross-queue sync" begin
    @testset "the stage flag is sync2-typed" begin
        # The bug in one line: right value, wrong wrapper type.
        # `Mantle.VK`. This file said `import Lava: Vulkan`, which resolved
        # while Lava depended on Vulkan.jl and afterwards bound the name to
        # nothing — Julia reports it as "defined but not assigned a value", so
        # the two assertions below errored rather than failing.
        @test Mantle.STAGE2_ALL_COMMANDS isa Mantle.VK.PipelineStageFlag2
        @test UInt64(Mantle.STAGE2_ALL_COMMANDS.val) ==
              UInt64(Mantle.VK.PIPELINE_STAGE_2_ALL_COMMANDS_BIT.val)
    end

    @testset "a buffer crossing queues submits" begin
        ctx = Mantle.vk_context()
        b1 = LavaBackend()                             # ctx.default_bq
        bq2 = Mantle.allocate_batch_queue!(ctx)
        b2 = LavaBackend(bq2)

        n = 256
        a = KA.allocate(b1, Float32, n)
        xqfill!(b1, 64)(a, 1.0f0; ndrange = n)
        KA.synchronize(b1)
        @test all(Array(a) .== 1.0f0)

        # Same buffer, other channel: core's `crosswaits!` must derive a wait on
        # bq1's timeline from the stamp. This is the wait that used to throw.
        xqfill!(b2, 64)(a, 2.0f0; ndrange = n)
        KA.synchronize(b2)
        @test all(Array(a) .== 2.0f0)

        # And back again, so the wait is exercised in the other direction.
        xqfill!(b1, 64)(a, 3.0f0; ndrange = n)
        KA.synchronize(b1)
        @test all(Array(a) .== 3.0f0)
    end

    @testset "a build that throws leaves nothing in limbo" begin
        bq = Mantle.batchqueue(Mantle.Device())
        b = LavaBackend()
        a = KA.allocate(b, Float32, 64)
        xqfill!(b, 64)(a, 1.0f0; ndrange = 64)
        KA.synchronize(b)
        Mantle.drain!(bq)
        before = length(bq.free)

        # The failure now available between opening a recording and submitting
        # it: the body throws. `oneshot!` gives the recording back unsubmitted
        # and drops the claims it took, so nothing is left ended-but-unsubmitted
        # and nothing stays held by a submission that never happened.
        @test_throws ErrorException Mantle.oneshot!(bq) do e
            Mantle.hold!(e, a)
            error("build failure")
        end
        @test (@atomic Mantle.stampof(a).holders) == 0
        @test length(bq.free) == before           # the recording went back
        @test all(o -> !o.open && isempty(o.sync), bq.free)

        # The channel must still work afterwards.
        xqfill!(b, 64)(a, 4.0f0; ndrange = 64)
        KA.synchronize(b)
        @test all(Array(a) .== 4.0f0)
    end
end
