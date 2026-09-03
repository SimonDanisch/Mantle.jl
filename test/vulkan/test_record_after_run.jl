"""
Recording a plan that has already run.

`record!` writes one command buffer per argument slot, because a recording has
the slot's base offset folded into every address it holds and so belongs to the
slot it was written for. `run!` looks its recording up as
`pl.recordings[pl.args.slot]`, so the two have to agree on what that index means.

They did not. `bake!` collected the recordings with `map(1:ARG_SLOTS)` and let
position stand for slot, which is only true while the ring is still at 0 — that
is, only for a plan that has never run. A plan that HAS run is somewhere else in
the ring, so recording 1 named slot 2, and every run afterwards read the
arguments of a neighbouring slot.

What that looks like from outside is the reason this test exists: not a crash
and not a uniformly wrong answer, but an answer that is right for some run
counts and wrong for others — right exactly when the number of runs brings the
two indices back into phase. Measured on Hikari's `setup` plan at 1200x900: bit
identical at 1, 3 and 6 samples, and wrong at 2, 4, 5, 7 and 8. A test that
recorded a fresh plan and ran it three times would have passed.

So: run first, record second, and sweep enough run counts to cross the ring
several times at every phase. `pre = 0` is the case where `run!` records for
itself; every other value is a plan already somewhere in the ring.
"""

using Test, Mantle, Lava, KernelAbstractions
const KA = KernelAbstractions

@kernel function ring_add!(out, k)
    i = @index(Global)
    @inbounds out[i] += k
end

"""`out += kref[]` on every run, which is what makes a lost slot visible."""
function _barplan(dev, out, kref, n)
    g = Mantle.Graph(dev)
    Mantle.compute!(g, "add") do p
        Mantle.use(p, out; read = true, write = true)
        Mantle.dispatch!(p, ring_add!, (out, kref), n)
    end
    Mantle.Plan(g)
end

"""Run `pre` times, record, then run `post` more; return the accumulator."""
function _runrecordrun(dev, be, n, pre::Int, post::Int)
    out = Mantle.Buffer(dev, zeros(Int32, n))
    kref = Ref(Int32(0))
    pl = Base.invokelatest(_barplan, dev, out, kref, n)
    for i in 1:pre
        kref[] = Int32(i)
        Mantle.run!(pl)
    end
    Mantle.record!(pl)
    for i in 1:post
        kref[] = Int32(pre + i)
        Mantle.run!(pl)
    end
    KA.synchronize(be)
    v = Array(Mantle.storage(out))
    Mantle.free!(pl)
    v
end

@testset "record! after run! keeps the ring in phase" begin
    dev = Mantle.Device(Mantle.VulkanAPI())
    be = Mantle.defaultbackend()
    n = 32

    # Every offset into the ring as the starting phase, and enough runs after
    # recording to wrap it more than twice from each.
    for pre in 0:(2 * Mantle.ARG_SLOTS)
        for post in 1:(3 * Mantle.ARG_SLOTS)
            want = fill(Int32(sum(1:(pre + post))), n)
            @test _runrecordrun(dev, be, n, pre, post) == want
        end
    end
end
