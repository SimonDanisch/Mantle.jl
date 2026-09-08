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
timeline value the default queue has not reached, so it cannot complete until
the host lets it; meanwhile the default queue runs past the second queue's
token value, which is exactly the state the wrong comparison misreads.
"""

using Test, Mantle, Lava, KernelAbstractions
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
    try
        # A submission on the second queue that the GPU cannot finish yet: it
        # waits for the default queue's timeline to reach `gate`, five
        # submissions away.
        gate = bq1.next_timeline + 5
        o = MVE.oneshot(bq2) do e end
        tok2 = Mantle.submit!(bq2, o;
            waits = ((bq1.timeline_sem, UInt64(gate), MVE.STAGE2_ALL_COMMANDS),), tag = :gated)
        @test length(Mantle.outstanding(bq2)) == 1
        @test !MVE.passed(bq2, tok2)

        # Three submissions on the default queue: its counter runs past
        # `tok2` (a small value on the OTHER timeline) and stays short of `gate`.
        c = KA.allocate(b1, Float32, 64)
        for _ in 1:3
            spq_fill!(b1, 64)(c, 1f0; ndrange = 64)
        end
        KA.synchronize(b1)
        @test MVE.query_timeline(bq1) >= tok2      # the misreading's premise
        @test MVE.query_timeline(bq1) < gate       # and the gate still holds

        # THE assertion: the sweep (`drain!` is what every path on this queue
        # calls before it opens or submits) gives nothing back, because on ITS
        # timeline nothing has passed. Before the fix the submission was swept,
        # the one-shot went to the pool, and `free_oneshots` held a command
        # buffer still in flight.
        MVE.drain!(bq2)
        @test length(Mantle.outstanding(bq2)) == 1
        @test !any(x -> x === o, bq2.free_oneshots)

        # Let it through: two more default submissions reach the gate.
        for _ in 1:2
            spq_fill!(b1, 64)(c, 2f0; ndrange = 64)
        end
        Mantle.flush!(bq2)
        @test isempty(Mantle.outstanding(bq2))
        @test any(x -> x === o, bq2.free_oneshots)
    finally
        Mantle.release_batch_queue!(bq2)
    end
end
