# Who owns a queue handed out by `allocate_batch_queue!`, and what happens when
# nobody gives it back.
#
# A `ManagedBuffer` records `last_write = (bq, value)`, and its finalizer asks
# that queue's timeline semaphore whether the GPU has passed `value`. The queue
# used to be reachable only from the buffers naming it, so when a caller (a
# RayMakie Screen) and its buffers became garbage together, the SEMAPHORE's own
# finalizer could run first — and `vkGetSemaphoreCounterValue` on a destroyed
# semaphore segfaults inside the driver rather than returning an error. That is
# the crash RayMakie's runtests.jl documents as the reason
# test_caching_gc_correctness.jl is excluded.
#
# The invariant that closes it: the context holds every handed-out queue, so a
# queue outlives every buffer allocated on it. `release_batch_queue!` is the only
# way out, and it drains first.

using Test, Lava, Mantle
@testset "batch queue lifetime" begin
    ctx = MVE.vk_context()

    @testset "the context owns what it hands out" begin
        held = length(ctx.extra_queues)
        bq = Mantle.allocate_batch_queue!(MVE.vk_context())
        # Reachable from the context, not just from the caller's binding. This
        # single assertion is the memory-safety property: a GC that collects the
        # caller cannot take the semaphore with it.
        @test length(ctx.extra_queues) == held + 1
        @test any(q -> q === bq, ctx.extra_queues)

        Mantle.release_batch_queue!(bq)
        @test length(ctx.extra_queues) == held
        @test !any(q -> q === bq, ctx.extra_queues)
    end

    @testset "a released hardware slot is reused" begin
        bq = Mantle.allocate_batch_queue!(MVE.vk_context())
        idx = MVE.driver(bq).queue_index
        Mantle.release_batch_queue!(bq)

        if idx >= 0
            @test idx in ctx.free_queue_indices
            again = Mantle.allocate_batch_queue!(MVE.vk_context())
            @test MVE.driver(again).queue_index == idx
            @test !(idx in ctx.free_queue_indices)
            Mantle.release_batch_queue!(again)
        else
            # The family ran out of queues, so this one shares the primary and
            # has no slot to give back. Still has to release cleanly.
            @test isempty(ctx.free_queue_indices) || !(idx in ctx.free_queue_indices)
        end
    end

    @testset "releasing twice is a no-op, releasing the primary is an error" begin
        bq = Mantle.allocate_batch_queue!(MVE.vk_context())
        Mantle.release_batch_queue!(bq)
        held = length(ctx.extra_queues)
        freed = copy(ctx.free_queue_indices)
        @test Mantle.release_batch_queue!(bq) === nothing
        @test length(ctx.extra_queues) == held
        @test ctx.free_queue_indices == freed

        @test_throws Lava.LavaError Mantle.release_batch_queue!(ctx.default_bq)
    end

    @testset "a buffer written on a dropped queue survives GC" begin
        # The crash, as directly as it can be staged: allocate on a handed-out
        # queue, drop every reference to that queue, then force collection. The
        # buffer's finalizer queries a timeline semaphore that only the
        # context's reference is keeping alive.
        let bq = Mantle.allocate_batch_queue!(MVE.vk_context())
            a = MVE.LavaArray{Float32, 1}(undef, (4096,))
            Mantle.upload!(a, ones(Float32, 4096))
            Mantle.flush!(bq)
        end
        GC.gc(true)
        GC.gc(true)
        # Reaching here at all is the assertion — the failure mode is SIGSEGV,
        # not a thrown exception. The counter check confirms the context still
        # holds the queue rather than having quietly dropped it.
        @test length(ctx.extra_queues) >= 1

        for q in copy(ctx.extra_queues)
            Mantle.release_batch_queue!(q)
        end
        @test isempty(ctx.extra_queues)
    end
end
