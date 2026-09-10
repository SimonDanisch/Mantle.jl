"""
A queue's sweep asks THAT queue's timeline, not the device's.

Every `VulkanBatchQueue` has its own timeline semaphore, and a token is a value
on it. Core `sweep!` used to ask `passed(dev, token)`, which the Vulkan device
answers with its DEFAULT queue's counter — so a submission on a second queue
(RayMakie's graphics queue, which blits and draws overlays) was judged by the
wrong timeline. The default queue is always far ahead of a queue that carries
three submissions per frame, so the second queue's submissions were swept the
moment they were made: their one-shots went back to the pool while the GPU was
still executing them, the next `oneshot` on that queue popped the SAME command
buffer and began it again mid-flight, and the overlay came out blank
(`test_overlay_compositing.jl`: 0 green pixels) or the device was lost. The
old per-queue `sweep_retired!` had compared against `query_timeline(bq)`; the
one record of what is in flight (backend-independence step 7) moved the sweep
into core and inherited the device-shaped question.

Deterministic rather than raced: the second queue's submission WAITS on a
timeline the HOST signals, so it cannot complete until this file lets it;
meanwhile the default queue runs past the second queue's token value, which is
exactly the state the wrong comparison misreads.

The gate is a semaphore of this test's own, and the gated submission is made
LAST, because a second channel does not imply a second hardware queue: where
the family exposes one queue, `allocate_batch_queue!` shares the primary
`VkQueue` (`queue_index` -1), and a submission there waiting on work not yet
submitted blocks everything behind it on that queue. Gating on the default
queue's own future counter deadlocked such a device instead of failing on it —
and the property under test is not affected, because every channel has its own
timeline semaphore whether or not it has its own queue.
"""

using Test, Mantle, Lava, KernelAbstractions, Vulkan
const KA = KernelAbstractions

@kernel function spq_fill!(a, v)
    i = @index(Global)
    @inbounds a[i] = v
end

@testset "a queue's sweep asks that queue's timeline" begin
    ctx = MVE.vk_context()
    bq1 = ctx.default_bq
    b1 = LavaBackend()
    bq2 = Mantle.allocate_batch_queue!(ctx)
    dev = MVE.vkdevice(bq1)
    gatesem = Vulkan.unwrap(Vulkan.create_semaphore(dev, Vulkan.SemaphoreCreateInfo(;
        next = Vulkan.SemaphoreTypeCreateInfo(Vulkan.SEMAPHORE_TYPE_TIMELINE, UInt64(0)))))
    try
        # Run the default queue ahead FIRST, so its counter is past any small
        # token value on the second channel's fresh timeline.
        c = KA.allocate(b1, Float32, 64)
        for _ in 1:3
            spq_fill!(b1, 64)(c, 1f0; ndrange = 64)
        end
        KA.synchronize(b1)

        # Then a submission on the second channel that the GPU cannot finish:
        # it waits on `gatesem`, which only this file signals.
        o = MVE.oneshot(bq2) do e end
        tok2 = Mantle.submit!(bq2, o;
            waits = ((gatesem, UInt64(1), MVE.STAGE2_ALL_COMMANDS),))
        Mantle.handover!(bq2, tok2, o; tag = :gated)
        @test length(Mantle.outstanding(bq2)) == 1
        @test !MVE.passed(bq2, tok2)
        @test MVE.query_timeline(bq1) >= tok2      # the misreading's premise

        # THE assertion: the sweep (`drain!` is what every path on this queue
        # calls before it opens or submits) gives nothing back, because on ITS
        # timeline nothing has passed. Before the fix the submission was swept,
        # the one-shot went to the pool, and core's free list held a command
        # buffer still in flight.
        Mantle.drain!(bq2)
        @test length(Mantle.outstanding(bq2)) == 1
        @test !any(x -> x === o, bq2.free)

        # Let it through.
        Vulkan.unwrap(Vulkan.signal_semaphore(dev,
            Vulkan.SemaphoreSignalInfo(gatesem, UInt64(1))))
        Mantle.flush!(bq2)
        @test isempty(Mantle.outstanding(bq2))
        @test any(x -> x === o, bq2.free)
    finally
        Mantle.release_batch_queue!(bq2)
    end
end
