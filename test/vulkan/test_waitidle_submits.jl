"""
`waitidle(device)` waits for the work the caller is still holding, and
`waitfor!(plan)` waits for one plan's last run.

Both are "block until the GPU has done what I asked for", and both were
reachable only by going around Mantle — `KA.synchronize(backend)`, which flushes
the batch queue behind the layer that owns submission. Hikari's bounce loop used
it to read a device-written ray count between chunks, so the one stall in the
render was invisible to Mantle and could never be replaced by a device-side
count.

The trap the first testset pins: `waitidle(::LavaDevice)` was `vkDeviceWaitIdle`
alone, which waits for everything SUBMITTED — and a headless plan submits when
something asks it to, so a run sitting in an open batch is neither submitted nor
waited for. The call returned immediately with the dispatch not yet handed to
the driver.

**Asserting on the buffer contents does not catch that, and the first version of
this file did exactly that and passed with the fix reverted.** A readback
flushes on its own, so the numbers come out right either way; and
`outstanding(bq)` is empty for work that was never submitted, so "nothing is
outstanding" is trivially true in the broken case. The question that separates
them is asked of the TIMELINE, before anything reads a buffer: has the value
this run signals been reached?
"""

using Test, Mantle, Lava, KernelAbstractions
const KA = KernelAbstractions

@kernel function waitidle_bump!(out, k)
    i = @index(Global)
    @inbounds out[i] += k
end

"""A plan that adds `kref[]` to `out`, and submits only when asked."""
function _waitplan(dev, out, kref, n)
    g = Mantle.Graph(dev)
    Mantle.compute!(g, "bump") do p
        Mantle.use(p, out; read = true, write = true)
        Mantle.dispatch!(p, waitidle_bump!, (out, kref), n)
    end
    Mantle.Plan(g)
end

"""What the plan's last run signals — `nothing` before it has run."""
lasttoken(pl) = pl.args === nothing ? nothing : pl.args.slot_token[pl.args.slot]

@testset "waitidle hands the open batch over first" begin
    dev = Mantle.Device(Mantle.VulkanAPI())
    n = 64
    out = Mantle.Buffer(dev, zeros(Int32, n))
    kref = Ref(Int32(7))
    pl = Base.invokelatest(_waitplan, dev, out, kref, n)
    bq = Mantle.batchqueue(dev)

    Mantle.run!(pl)
    tok = lasttoken(pl)
    @test tok !== nothing
    # Recorded and NOT yet submitted: this is the state the old `waitidle`
    # returned from without doing anything.
    @test !Mantle.passed(dev, tok)
    flushes = bq.ctx.diag.flush_counter[]

    Mantle.waitidle(dev)

    # THE assertion, and it is asked of the timeline rather than of a buffer.
    @test Mantle.passed(dev, tok)
    # …and the reason it holds is that the batch was submitted, not that
    # something else got there first.
    @test bq.ctx.diag.flush_counter[] > flushes

    @test Array(Mantle.storage(out)) == fill(Int32(7), n)
    Mantle.free!(pl)
end

# `waitfor!(plan)` is the precise form: it waits for that plan's last run,
# submitting it if the host is still holding it. Same assertion, for the same
# reason — a readback would hide a `waitfor!` that did nothing at all.
@testset "waitfor! waits for the plan's last run" begin
    dev = Mantle.Device(Mantle.VulkanAPI())
    n = 64
    out = Mantle.Buffer(dev, zeros(Int32, n))
    kref = Ref(Int32(0))
    pl = Base.invokelatest(_waitplan, dev, out, kref, n)

    # More runs than there are argument slots, so a slot is reused while an
    # earlier run may still be in flight — the case where waiting for the wrong
    # run leaves the host one step behind.
    # Before the first run there is no slot yet — `slot` is 0 and indexing the
    # token vector with it is a `BoundsError`, not an answer.
    @test Mantle.waitfor!(pl) === nothing

    total = Int32(0)
    for k in Int32[3, 5, 11, 13, 17]
        kref[] = k
        Mantle.run!(pl)
        tok = lasttoken(pl)
        @test !Mantle.passed(dev, tok)      # still the host's, before the wait
        Mantle.waitfor!(pl)
        @test Mantle.passed(dev, tok)       # and the device's, after it
        total += k
        @test Array(Mantle.storage(out)) == fill(total, n)
    end

    # Once the device has caught up it is a no-op, not another submission.
    flushes = Mantle.batchqueue(dev).ctx.diag.flush_counter[]
    Mantle.waitfor!(pl)
    @test Mantle.batchqueue(dev).ctx.diag.flush_counter[] == flushes

    Mantle.free!(pl)
end
