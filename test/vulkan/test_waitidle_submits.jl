"""
`waitidle(device)` waits for the work the caller is still holding, and
`waitfor!(plan)` waits for one plan's last run.

Both are "block until the GPU has done what I asked for", and both were
reachable only by going around Mantle — `KA.synchronize(backend)`, which flushes
the batch queue behind the layer that owns submission. Hikari's bounce loop used
it to read a device-written ray count between chunks, so the one stall in the
render was invisible to Mantle and could never be replaced by a device-side
count.

The trap the first testset used to pin: `waitidle(::LavaDevice)` was
`vkDeviceWaitIdle` alone, which waits for everything SUBMITTED — and a headless
plan used to submit when something asked it to, so a run sitting in an open
batch was neither submitted nor waited for. There is no open batch now: a run
is submitted the moment `run!` is called, and the token it answers with is
out before `run!` returns. So what is pinned is that — the token is the
queue's newest the moment it exists — and that `waitidle` then waits for it
without submitting anything more.

**Asserting on the buffer contents does not catch a missed wait, and the first
version of this file did exactly that and passed with the fix reverted.** A
readback waits on its own, so the numbers come out right either way. The
question that separates them is asked of the TIMELINE, before anything reads
a buffer: has the value this run signals been reached?
"""

using Test, Mantle, Lava, KernelAbstractions
const KA = KernelAbstractions

@kernel function waitidle_bump!(out, kref)
    i = @index(Global)
    @inbounds out[i] += kref[1]
end

"""A plan that adds `kref`'s current value to `out`, and submits only when asked."""
function _waitplan(dev, out, kref, n)
    g = Mantle.Graph(dev)
    Mantle.compute!(g, "bump") do p
        Mantle.use(p, out; read = true, write = true)
        Mantle.use(p, kref; read = true)
        Mantle.dispatch!(p, waitidle_bump!, (out, kref), n)
    end
    Mantle.record!(Mantle.Plan(g))
end

"""What the plan's last run signals — 0 before it has run."""
lasttoken(pl) = Mantle.argtoken(pl.args)

@testset "waitidle hands the open batch over first" begin
    dev = Mantle.Device(Mantle.VulkanAPI())
    n = 64
    out = Mantle.Buffer(dev, zeros(Int32, n))
    kref = Mantle.GPURef(dev, Int32(7))
    pl = Base.invokelatest(_waitplan, dev, out, kref, n)
    bq = Mantle.batchqueue(dev)

    flushes = MVE.ctxof(bq).diag.flush_counter[]
    Mantle.run!(pl)
    tok = lasttoken(pl)
    @test tok != 0
    # Submitted the moment it was run: the token is the queue's newest, and
    # the run was one submission.
    @test tok == MVE.driver(bq).next_timeline
    @test MVE.ctxof(bq).diag.flush_counter[] == flushes + 1

    Mantle.waitidle(dev)

    # THE assertion, and it is asked of the timeline rather than of a buffer.
    @test Mantle.passed(dev, tok)
    # …and nothing was submitted to get there: there was nothing left to hand over.
    @test MVE.ctxof(bq).diag.flush_counter[] == flushes + 1

    @test Array(Mantle.storage(out)) == fill(Int32(7), n)
    Mantle.free!(pl)
end

# `waitfor!(plan)` is the precise form: it waits for that plan's last run. Same
# assertion, for the same reason — a readback would hide a `waitfor!` that did
# nothing at all.
@testset "waitfor! waits for the plan's last run" begin
    dev = Mantle.Device(Mantle.VulkanAPI())
    n = 64
    out = Mantle.Buffer(dev, zeros(Int32, n))
    kref = Mantle.GPURef(dev, Int32(0))
    pl = Base.invokelatest(_waitplan, dev, out, kref, n)
    bq = Mantle.batchqueue(dev)

    # Before the first run there is no token, and that is an answer rather than
    # an error.
    @test Mantle.waitfor!(pl) === nothing

    total = Int32(0)
    for k in Int32[3, 5, 11, 13, 17]
        kref[] = k
        Mantle.run!(pl)
        tok = lasttoken(pl)
        @test tok == MVE.driver(bq).next_timeline       # out the moment the run returned
        Mantle.waitfor!(pl)
        @test Mantle.passed(dev, tok)       # and the device's, after it
        total += k
        @test Array(Mantle.storage(out)) == fill(total, n)
    end

    # Once the device has caught up it is a no-op, not another submission.
    flushes = MVE.ctxof(Mantle.batchqueue(dev)).diag.flush_counter[]
    Mantle.waitfor!(pl)
    @test MVE.ctxof(Mantle.batchqueue(dev)).diag.flush_counter[] == flushes

    Mantle.free!(pl)
end
