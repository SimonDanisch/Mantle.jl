"""
A recording survives any number of runs, and `record!` after them is a no-op.

`record!` writes ONE command buffer, once, and a plan accumulates the same way
across every run count. That is what this sweeps: `pre` runs, a second
`record!` (idempotent — the same recording object), `post` more runs, checked
against the arithmetic. `run!` never records: a plan that was never recorded
is refused, which the last testset pins.

The sweep exists because a recording tied to an argument SLOT fails as a
function of the run count. With a recording per slot, the slot's base offset
folded into every address it holds, and the recordings collected by position,
position stands for slot only while nothing has run: a plan that HAS run is
elsewhere in the ring, so recording 1 names slot 2 and every run afterwards
reads a neighbouring slot's arguments.

From outside that is neither a crash nor a uniformly wrong answer, but an answer
right for some run counts and wrong for others — right exactly when the number
of runs brings the two indices back into phase. Measured on Hikari's `setup`
plan at 1200x900: bit identical at 1, 3 and 6 samples, and wrong at 2, 4, 5, 7
and 8. A test that records a fresh plan and runs it three times passes.

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
    Mantle.dispatch!(g, ring_add!, (out, kref), n; name = "add")
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
    Mantle.dispatch!(g, ring_add!, (out, kref), n; name = "add")
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
