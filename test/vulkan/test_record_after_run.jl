"""
A recording survives any number of runs, and `record!` after them is a no-op.

`record!` writes ONE command buffer, once, and a plan accumulates the same way
across every run count. That is what this sweeps: `pre` runs, a second
`record!` (idempotent — the same recording object), `post` more runs, checked
against the arithmetic. `run!` never records: a plan that was never recorded
is refused, which the last testset pins.

It exists because of what the argument RING did to this. There was a recording
per slot, with the slot's base offset folded into every address it held, and
`run!` looked one up as `pl.recordings[pl.args.slot]` — so the two had to agree
on what the index meant, and they did not. `bake!` collected them with
`map(1:ARG_SLOTS)`, letting position stand for slot, which is only true while the
ring is still at 0 — that is, only for a plan that has never run. A plan that HAS
run is somewhere else in the ring, so recording 1 named slot 2 and every run
afterwards read a neighbouring slot's arguments.

What that looked like from outside is why the sweep is worth keeping: not a crash
and not a uniformly wrong answer, but an answer right for some run counts and
wrong for others — right exactly when the number of runs brought the two indices
back into phase. Measured on Hikari's `setup` plan at 1200x900: bit identical at
1, 3 and 6 samples, and wrong at 2, 4, 5, 7 and 8. A test that recorded a fresh
plan and ran it three times would have passed.

`pre = 0` is the case where the second `record!` follows the first directly.
"""

using Test, Mantle, Lava, KernelAbstractions
const KA = KernelAbstractions

@kernel function ring_add!(out, kref)
    i = @index(Global)
    @inbounds out[i] += kref[1]
end

"""`out += kref[]` on every run, which is what makes a wrong value visible."""
function _barplan(dev, out, kref, n)
    g = Mantle.Graph(dev)
    Mantle.compute!(g, "add") do p
        Mantle.use(p, out; read = true, write = true)
        Mantle.use(p, kref; read = true)
        Mantle.dispatch!(p, ring_add!, (out, kref), n)
    end
    Mantle.record!(Mantle.Plan(g))
end

"""Run `pre` times, record again (a no-op), then run `post` more; return the
accumulator."""
function _runrecordrun(dev, be, n, pre::Int, post::Int)
    out = Mantle.Buffer(dev, zeros(Int32, n))
    kref = Mantle.GPURef(dev, Int32(0))
    pl = Base.invokelatest(_barplan, dev, out, kref, n)
    for i in 1:pre
        kref[] = Int32(i)
        Mantle.run!(pl)
    end
    rec = pl.recording
    Mantle.record!(pl) === pl || error("record! must return the plan")
    pl.recording === rec || error("record! after $pre runs recorded again")
    for i in 1:post
        kref[] = Int32(pre + i)
        Mantle.run!(pl)
    end
    KA.synchronize(be)
    v = Array(Mantle.storage(out))
    Mantle.free!(pl)
    v
end

@testset "record! after run! accumulates the same as before it" begin
    dev = Mantle.Device(Mantle.VulkanAPI())
    be = Mantle.defaultbackend()
    n = 32

    # Six starting points and nine follow-ups from each — the sweep the ring
    # needed, kept because the arithmetic it checks is the point.
    for pre in 0:6
        for post in 1:9
            want = fill(Int32(sum(1:(pre + post))), n)
            @test _runrecordrun(dev, be, n, pre, post) == want
        end
    end
end

@testset "run! never records: an unrecorded plan is refused" begin
    dev = Mantle.Device(Mantle.VulkanAPI())
    n = 32
    out = Mantle.Buffer(dev, zeros(Int32, n))
    kref = Mantle.GPURef(dev, Int32(1))
    g = Mantle.Graph(dev)
    Mantle.compute!(g, "add") do p
        Mantle.use(p, out; read = true, write = true)
        Mantle.use(p, kref; read = true)
        Mantle.dispatch!(p, ring_add!, (out, kref), n)
    end
    pl = Base.invokelatest(Mantle.Plan, g)
    @test !Mantle.recorded(pl)
    @test_throws ArgumentError Mantle.run!(pl)
    @test !Mantle.recorded(pl)                 # the refusal recorded nothing either
    Mantle.record!(pl)
    Mantle.run!(pl)
    Mantle.waitfor!(pl)
    @test Array(Mantle.storage(out)) == fill(Int32(1), n)
    Mantle.free!(pl)
end
