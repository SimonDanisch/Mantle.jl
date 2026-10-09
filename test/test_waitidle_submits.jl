"""
`waitidle(device)` waits for the work the caller is still holding, and
`waitfor!(plan)` waits for one plan's last run.

Both are "block until the GPU has done what I asked for", and both were
reachable only by going around Mantle — `KA.synchronize(backend)`, which flushes
the batch queue behind the layer that owns submission. Hikari's bounce loop used
it to read a device-written ray count between chunks, so the one stall in the
render was invisible to Mantle and could never be replaced by a device-side
count.

The trap the first testset pins: a `waitidle` that is only the driver's "wait
for the device to go idle" waits for everything SUBMITTED, and a headless plan
that submits when something asks it to leaves a run in an open batch neither
submitted nor waited for.

**Asserting on the buffer contents does not catch a missed wait, and the first
version of this file did exactly that and passed with the fix reverted.** A
readback waits on its own, so the numbers come out right either way. The
question that separates them is asked of the TIMELINE, before anything reads
a buffer: has the value this run signals been reached?

That the token is the queue's newest the moment `run!` returns, and that neither
wait submits anything of its own, is asserted from the queue's submission count
in `test/vulkan/test_closed_command_buffers.jl`: no portable count exists.
"""

using Test, Mantle, KernelAbstractions
include(joinpath(@__DIR__, "testbackend.jl"))
const KA = KernelAbstractions

@kernel function waitidle_bump!(out, kref)
    i = @index(Global)
    @inbounds out[i] += kref[1]
end

"""A plan that adds `kref`'s current value to `out`, and submits only when asked."""
function _waitplan(dev, out, kref, n)
    g = Mantle.Graph(dev)
    Mantle.dispatch!(g, waitidle_bump!, (out, kref), n; name = "bump")
    # `budget = Inf`: recorded whole from the first run, which is what these tests
    # are about. How a plan is cut by its budget is `test_partitioned_recording.jl`.
    Mantle.record!(Mantle.Plan(g; budget = Inf))
end

"""What the plan's last run signals — 0 before it has run."""
lasttoken(pl) = Mantle.argtoken(pl.args)

@testset "waitidle hands the open batch over first" begin
    dev = Mantle.Device(TESTBACKEND)
    n = 64
    out = Mantle.Buffer(dev, zeros(Int32, n))
    kref = Mantle.GPURef(dev, Int32(7))
    pl = Base.invokelatest(_waitplan, dev, out, kref, n)

    Mantle.run!(pl)
    tok = lasttoken(pl)
    @test tok != 0

    Mantle.waitidle(dev)

    # THE assertion, and it is asked of the timeline rather than of a buffer.
    @test Mantle.passed(dev, tok)

    @test Array(Mantle.storage(out)) == fill(Int32(7), n)
    Mantle.free!(pl)
end

# `waitfor!(plan)` is the precise form: it waits for that plan's last run. Same
# assertion, for the same reason — a readback would hide a `waitfor!` that did
# nothing at all.
@testset "waitfor! waits for the plan's last run" begin
    dev = Mantle.Device(TESTBACKEND)
    n = 64
    out = Mantle.Buffer(dev, zeros(Int32, n))
    kref = Mantle.GPURef(dev, Int32(0))
    pl = Base.invokelatest(_waitplan, dev, out, kref, n)

    # Before the first run there is no token, and that is an answer rather than
    # an error.
    @test Mantle.waitfor!(pl) === nothing

    total = Int32(0)
    for k in Int32[3, 5, 11, 13, 17]
        kref[] = k
        Mantle.run!(pl)
        tok = lasttoken(pl)
        @test tok != 0
        Mantle.waitfor!(pl)
        @test Mantle.passed(dev, tok)       # the device is past it, before any readback
        total += k
        @test Array(Mantle.storage(out)) == fill(total, n)
    end

    # Once the device has caught up it is an answer, not a wait or an error.
    @test Mantle.waitfor!(pl) === nothing

    Mantle.free!(pl)
end
