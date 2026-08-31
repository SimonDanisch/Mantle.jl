"""
A buffer freed while a batch is recording must be deferred, not destroyed.

This is the intermittent `vkWaitSemaphores` hang — six occurrences over two
months, recorded as "not reproducible" — and it was one missing case in
`vk_free!`. Destruction is deferred while a batch is open, but that check sat
inside `if last_write !== nothing`, and `last_write` is only written by
`sync_access!` **at submit**. So a buffer the *currently recording* batch
references, which has never been through a submit, read as `nothing` and fell
through to immediate destruction with an open command buffer still naming it.

Asserted directly rather than by trying to provoke the hang. Reproducing it takes
a Julia GC landing inside a recording — 60 SAM 2 decodes with the collector live
did it within 15, and with collections confined to safe points never did — which
is a fine way to *find* a bug and a terrible way to guard one. The state machine
is deterministic: `ALIVE -> DEFERRED` and onto `bq.deferred_frees`, or
`ALIVE -> DEAD`. Check which.
"""

using Test, Lava, KernelAbstractions
const KA = KernelAbstractions

@testset "free during recording is deferred" begin
    backend = LavaBackend()
    bq = MVE.vk_context().default_bq

    # Quiesce, so nothing left over decides the outcome.
    KA.synchronize(backend)
    MVE.drain_deferred_frees!(bq)

    @testset "never submitted, batch open" begin
        # Open a batch and leave it recording.
        Mantle.ensure_active_batch!(bq)
        @test bq.active_batch !== nothing
        @test bq.active_batch.recording

        a = KA.allocate(backend, Float32, 64)
        buf = a.buf[]
        # The case the bug turned on: allocated, never dispatched against, so
        # `sync_access!` has never run and there is no timeline value to test.
        @test (@atomic :acquire buf.last_write) === nothing
        @test (@atomic :acquire buf.state) == MVE.BUF_STATE_ALIVE

        before = length(bq.deferred_frees)
        Mantle.unsafe_free!(a)

        # Deferred, and on the list — not destroyed under the open batch.
        @test (@atomic :acquire buf.state) == MVE.BUF_STATE_DEFERRED
        @test length(bq.deferred_frees) == before + 1
        @test any(x -> x === buf, bq.deferred_frees)

        # And the deferral is honoured, not leaked: the drain at the next
        # submit boundary is what finally destroys it.
        KA.synchronize(backend)
        MVE.drain_deferred_frees!(bq)
        @test (@atomic :acquire buf.state) == MVE.BUF_STATE_DEAD
    end

    @testset "no batch recording: freed immediately" begin
        # The other side, so the guard cannot be satisfied by deferring
        # everything forever — which would leak instead of hanging.
        KA.synchronize(backend)
        MVE.drain_deferred_frees!(bq)
        b = KA.allocate(backend, Float32, 64)
        buf = b.buf[]
        KA.synchronize(backend)          # closes the batch opened by `allocate`
        @test bq.active_batch === nothing || !bq.active_batch.recording
        Mantle.unsafe_free!(b)
        @test (@atomic :acquire buf.state) == MVE.BUF_STATE_DEAD
    end
end
