"""
A buffer named by a submission still in flight must be honoured, not raced.

This is the intermittent `vkWaitSemaphores` hang — six occurrences over two
months, recorded as "not reproducible" — and it was one missing case in
`vk_free!`: a buffer the open batch had recorded against but never submitted
read as idle and was destroyed under the batch. There is no open batch now;
every launch HOLDS what it names into its own closed command buffer and submits
it at once, and the hold is what keeps the buffer: a free that arrives while
the submission is in flight is RECORDED (core's `retire!`), the buffer stays
alive, and the destroy runs at the first drain after the submission has passed.
A buffer nothing in flight names is destroyed at the next drain. (A raw
`vk_free!` of a held buffer, the path a teardown that owns the buffer takes, is
`test_held_buffer_lifetime.jl`.)

Asserted directly rather than by trying to provoke the hang. Reproducing it takes
a Julia GC landing mid-flight — 60 SAM 2 decodes with the collector live
did it within 15, and with collections confined to safe points never did — which
is a fine way to *find* a bug and a terrible way to guard one. The state machine
is deterministic: held and recorded, then `ALIVE -> DEAD` once the hold drops;
or `ALIVE -> DEAD` at the next drain. Check which.
"""

using Test, Lava, KernelAbstractions
const KA = KernelAbstractions

# Keeps the GPU busy for tens of milliseconds, so the launch's submission is
# still in flight while the host issues the free right behind it — that
# in-flight-ness is what holds the hazard window open. Same pattern as
# `heavy_fill_kernel!` in test_argument_memory_isolation.jl.
@kernel function slow_touch_kernel!(arr)
    i = @index(Global)
    acc = Float32(i)
    for _ in 1:20_000
        acc = muladd(acc, 1.0000001f0, 0.5f0)
    end
    @inbounds arr[i] += (acc > 0f0 ? 0.0f0 : 1.0f0)
end

@testset "a free while the buffer is in flight is honoured, not raced" begin
    backend = LavaBackend()
    bq = MVE.vk_context().default_bq

    # Quiesce, so nothing left over decides the outcome.
    KA.synchronize(backend)
    Mantle.drain!(bq)

    @testset "named by a submission still in flight: recorded, run when it passes" begin
        a = KA.allocate(backend, Float32, 64)
        buf = a.buf[]
        holders() = @atomic Mantle.stampof(buf).holders
        @test (@atomic :acquire buf.state) == MVE.BUF_STATE_ALIVE
        @test holders() == 0

        # Slow launch: its submission is still outstanding when the free below
        # lands. The launch held the array into its one-shot, so the buffer is
        # claimed by a recording that has been submitted and not yet passed.
        slow_touch_kernel!(backend, 64)(a; ndrange=64)
        @test holders() == 1

        Mantle.unsafe_free!(a)

        # The request is recorded and nothing is destroyed: a drain that asks
        # now finds the submission still in flight.
        @test (@atomic :acquire buf.state) != MVE.BUF_STATE_DEAD
        Mantle.drain!(bq)
        @test (@atomic :acquire buf.state) != MVE.BUF_STATE_DEAD
        @test holders() == 1

        # And nothing leaks: once the submission has passed, the sweep drops the
        # hold and the destroy that was owed runs.
        KA.synchronize(backend)
        Mantle.drain!(bq)
        @test holders() == 0
        @test (@atomic :acquire buf.state) == MVE.BUF_STATE_DEAD
    end

    @testset "nothing in flight names it: freed immediately" begin
        # The other side, so the guard cannot be satisfied by deferring
        # everything forever — which would leak instead of hanging.
        KA.synchronize(backend)
        Mantle.drain!(bq)
        b = KA.allocate(backend, Float32, 64)
        buf = b.buf[]
        KA.synchronize(backend)          # nothing left in flight naming it
        @test isempty(bq.outstanding) || all(o -> o.token <= MVE.query_timeline(bq), bq.outstanding)
        Mantle.unsafe_free!(b)
        Mantle.drain!(bq)
        @test (@atomic :acquire buf.state) == MVE.BUF_STATE_DEAD
    end
end
