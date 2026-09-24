"""
Forgetting `free!` is not a leak.

A `Buffer`, a `GPURef` and a `Plan` each carry a finalizer, so a caller that
drops one gets the memory back without saying anything. That is the guarantee
this file pins, and it is a guarantee about the API's shape rather than about
its performance: a runtime where a missing verb silently costs 274 MiB is one
where every caller is one omission away from a leak, and several here were.

**The finalizer does not free.** It appends — `retire!` for a region,
`retireplan!` for a plan — under a lock, touching no free list and no driver,
because it runs on whatever thread the GC picks. The release itself is
`reclaim!` on the owning thread, which `run!` already calls every submission.
That split is why this can be automatic at all; `trim!` records what the other
arrangement cost, a `ConcurrencyViolationError` and then a SIGSEGV.

`free!` stays, and stays useful: it retires at a point the caller chose instead
of whenever a GC happens. What it is not any more is mandatory.
"""

using Test, Mantle, KernelAbstractions
import KernelInterface as KI
const M = Mantle

function drop_addone!(a, n::Int32)
    i = Int32(KI.get_global_id().x)
    i <= n && (@inbounds a[i] += 1.0f0)
    return nothing
end

const DROP_N = 1024

"""Collect until the ledger stops falling, then report it.

One `GC.gc(true)` is not enough to trust a baseline: an object dropped earlier
in the session may still be waiting for a collection, and its finalizer only
retires — the region comes back at the `reclaim!` after that. A single
before/after pair therefore measures this file plus whatever the previous one
left, which is how the first version of this test read `before = 17` and
`after = 7` and called the guarantee broken."""
function settled(dev)
    onloan() = sum(b -> length(b.live),
                   Iterators.flatten(values(Mantle.pool(dev).blocks)); init = 0)
    # All eight rounds and the MINIMUM, not "stop when two agree": a finalizer
    # runs when the collector gets to it, so the count can sit still for a round
    # and fall on the next. Stopping at the first agreement read 2 where a third
    # collection would have read 0, which looks exactly like a leak.
    best = typemax(Int)
    for _ in 1:8
        GC.gc(true)
        Mantle.reclaim!(Mantle.pool(dev), dev; wait = true)
        best = min(best, onloan())
    end
    best
end

# A WARM-UP before the baseline, in both testsets below. The first upload on a
# device allocates staging that is then cached and reused, so a baseline taken
# on a cold device measures that one-time cost as if it were this file's leak —
# `before = 0` and `after = 3`, which is what the first version of this test
# reported. The guarantee here is about the region a resource holds, not about
# whether the device keeps caches.
"""Allocate a buffer and a ref, use both, and return — **from a function**.

The return is the point. A `let` block inside an `@testset` does not drop its
locals for the collector: they stay rooted in the frame until the whole block
ends, so a `GC.gc(true)` in the middle of a testset collects nothing and the
guarantee under test reads as broken. Measured while writing this file: alive 4,
after the `let` 4, after a settle 4. The same code in a callee that returns goes
back to 1."""
function makeanddrop(dev)
    b = M.Buffer(dev, Float32, (DROP_N,))
    r = M.GPURef(dev, 1.0f0)
    fill!(b, 1.0f0); r[] = 2.0f0
    # Both are used, so neither is dead before the count, and the count is taken
    # HERE — inside the callee, where they are unambiguously alive.
    s = sum(Array(M.storage(b)))
    n = sum(x -> length(x.live), Iterators.flatten(values(M.pool(dev).blocks)); init = 0)
    return s == 0 ? -1 : n
end

@testset "a dropped Buffer and GPURef come back with no free!" begin
    dev = M.Device(TESTBACKEND)
    onloan() = sum(b -> length(b.live),
                   Iterators.flatten(values(M.pool(dev).blocks)); init = 0)

    # FLAT ACROSS ROUNDS is the property, with no absolute baseline anywhere.
    # A device keeps caches — the first upload and the first readback allocate
    # staging that is then reused — so "back to the number before you started"
    # is not the guarantee and pinning it made this test fail for a reason that
    # had nothing to do with finalizers. A resource that did not come back shows
    # up as a count that climbs every round, and that is what is checked.
    counts = Int[]
    for _ in 1:4
        makeanddrop(dev)
        push!(counts, settled(dev))
    end
    @test allequal(counts[2:end])          # the pool does not grow, round over round
    before = counts[end]
    # NOT `alive > settled`: the ledger counts ENTRIES, and a region that is
    # returned and then re-acquired reuses its slot, so a live resource need not
    # raise the count at all once the pool is warm. That a resource takes memory
    # in the first place is asserted below, from a cold slot, where it does.

    # The explicit verb still works and is idempotent. A double `free!` must not
    # push the same region twice: two owners for one region is worse than never
    # getting it back. `b` stays rooted in this frame, which is why the count is
    # taken after the explicit call rather than after a collection.
    b = M.Buffer(dev, Float32, (DROP_N,))
    @test onloan() > before
    M.free!(b); M.free!(b)
    M.reclaim!(M.pool(dev), dev; wait = true)
    @test onloan() == before
    @test M.free!(b) === nothing       # and the finalizer will be a fourth
end

@testset "a dropped Plan gives back its arenas and its recording" begin
    # The whole cycle with nothing freed: a persistent buffer, a transient the
    # plan places, the plan's argument memory and its recording. A plan's
    # teardown is the part that cannot run on a GC thread — it unlistens from
    # the pool's move listeners and destroys a command buffer — so this is the
    # case that says the deferred hand-off works, not just the region one.
    dev = M.Device(TESTBACKEND)
    onloan() = sum(b -> length(b.live),
                   Iterators.flatten(values(M.pool(dev).blocks)); init = 0)
    function cycle()
        b = M.Buffer(dev, zeros(Float32, DROP_N))
        g = M.Graph(dev)
        t = M.Transient.Buffer(g, Float32, (4 * DROP_N,))
        M.dispatch!(g, drop_addone!, (t, Int32(4 * DROP_N)), 4 * DROP_N;
                    group = 64, name = "transient")
        M.dispatch!(g, drop_addone!, (b, Int32(DROP_N)), DROP_N;
                    group = 64, name = "persistent")
        pl = M.Plan(g); M.record!(pl); M.run!(pl); M.waitidle(dev)
        return nothing
    end
    cycle()                      # the same one-time staging, before the baseline
    before = settled(dev)
    for _ in 1:3
        cycle()
        GC.gc(true); M.reclaim!(M.pool(dev), dev; wait = true)
        # Every round, not just at the end: a leak of one region per cycle is
        # the shape this is most likely to regress into, and a single
        # before/after comparison would not see it until it was three times
        # bigger than it needed to be to fail.
        @test onloan() == before
    end
end

@testset "free! on a plan is idempotent" begin
    dev = M.Device(TESTBACKEND)
    b = M.Buffer(dev, zeros(Float32, DROP_N))
    g = M.Graph(dev)
    M.dispatch!(g, drop_addone!, (b, Int32(DROP_N)), DROP_N; group = 64, name = "p")
    pl = M.Plan(g); M.record!(pl); M.run!(pl); M.waitidle(dev)
    M.free!(pl)
    # The second call must not retire the arenas again — and the finalizer that
    # follows when `pl` is collected is a third caller of the same thing.
    @test M.free!(pl) === pl
    M.free!(b)
end
