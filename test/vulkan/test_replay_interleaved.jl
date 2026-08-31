"""
`replay!` interleaved with ordinary recording on the same queue.

The intended use of `capture`/`replay!` is a drained queue and nothing else in
flight, and that is what `test`s elsewhere cover. The *real* use is not that: an
editor replays a captured decode on every click while ordinary recording — a
preview render, a thumbnail, the next frame's encode — goes on around it. That
mixture had never been exercised and it did not work.

`ensure_active_batch!` hands a new batch `bq.next_timeline + 1` as its signal
value, reserving it; `submit!` later asserts the reservation still holds. A
`replay!` bumps the same counter, so any batch that was open at the time has a
stale reservation and the next `submit!` dies with

    AssertionError: batch signal desync: 859 vs 860

which names neither the replay nor the batch. `replay!` now closes an open batch
before touching the counter — the replay is ordered after that work anyway.

The shape below is the minimum that reproduces it: record *without* draining,
then replay, then record again. Draining between the two hides the bug entirely,
which is why every earlier test passed.
"""

using Test, Lava, KernelAbstractions
const KA = KernelAbstractions

@testset "replay interleaved with recording" begin
    back = LavaBackend()
    # Touch the backend first: the `VkContext` is created lazily, so reading
    # `VK_CONTEXT_REF` before any allocation gets `nothing`.
    a = MVE.LavaArray(Float32.(collect(1:1024)))
    out = KA.allocate(back, Float32, 1024)
    fill!(out, 0f0)
    KA.synchronize(back)
    bq = MVE.VK_CONTEXT_REF[].default_bq

    seq = MVE.capture(bq) do
        out .= a .* 2f0
    end
    KA.synchronize(back)
    want = 2 .* Float32.(collect(1:1024))
    @test Array(out) == want

    @testset "a replay after an UNDRAINED recording does not desync" begin
        # No `synchronize` between the two: this is the case that failed.
        for _ in 1:4
            out .= a .* 3f0          # records, leaves a batch open
            MVE.replay!(seq)        # bumps the timeline the open batch reserved
        end
        KA.synchronize(back)
        # The replay ran last, so the captured `*2` is what survives.
        @test Array(out) == want
    end

    @testset "recording still works after a replay" begin
        # The desync surfaced on the *next* submit!, not on the replay, so the
        # assertion is that ordinary work keeps running afterwards.
        MVE.replay!(seq)
        out .= a .* 5f0
        KA.synchronize(back)
        @test Array(out) == 5 .* Float32.(collect(1:1024))
    end

    @testset "many replays in a row" begin
        for _ in 1:16; MVE.replay!(seq); end
        KA.synchronize(back)
        @test Array(out) == want
    end

    a = out = seq = nothing
    GC.gc()
end
