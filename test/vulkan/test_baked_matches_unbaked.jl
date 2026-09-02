"""
Baking changes WHEN commands are built. It does not change what a run means.

Three properties, and two of them were false.

**`bake!` records without executing.** `capture` used to fall through into
`submit!`, so a plan ran as a side effect of being captured. It was never a
Vulkan constraint — `vkEndCommandBuffer` and `vkQueueSubmit2` are separate calls,
and Mantle already begins capture buffers with `SIMULTANEOUS_USE` so they can be
replayed — it was `submit!` doing two jobs, with the collection step capture
needed buried between them.

**A baked run reads the current arguments.** `dispatch!` takes a `Ref` to mean
"read this fresh every run", and `argvalue` honours that — but only on the
unbaked path, which re-records and therefore re-packs. A baked plan replayed the
bytes packed at `bake!` unless the caller remembered `rebind!`. So a renderer
advancing a sample index rendered sample 0 forever, and nothing said so.

**And it reads them without racing the device.** `run!` repacking is not enough
on its own: a baked plan used to capture ONE recording and pin the argument slot
it named for life, so the repack wrote bytes a replay still in flight was
reading. The numbers below caught it — 36 where the unbaked run gave 21, which is
`3 + 3*11`, three runs having read the last value written. There is a recording
per slot now and `run!` rotates through them, so the ring covers a baked plan
exactly as it covers an unbaked one.

The assertion that catches all three is the same one: the same graph, run baked
and unbaked with the same sequence of `Ref` values, must produce the same
numbers. Enough steps to wrap the ring several times, so a slot is reused while
earlier runs are still in flight — with three or fewer the ring never wraps and
the race cannot show.

`custom!` is deliberately not exercised here: a `custom!` body packs its own
arguments as it runs, so `rebindable(plan)` is false and `rebind!` refuses rather
than going quietly stale. Nothing in `src/` uses it.
"""

using Test, Mantle, Lava, KernelAbstractions
const KA = KernelAbstractions

@kernel function baked_addk!(out, k)
    i = @index(Global)
    @inbounds out[i] += k
end

"""The graph under test: adds `kref[]` to `out` on every run."""
function _refplan(dev, out, kref, n)
    g = Mantle.Graph(dev)
    Mantle.compute!(g, "add") do p
        Mantle.use(p, out; read = true, write = true)
        Mantle.dispatch!(p, baked_addk!, (out, kref), n)
    end
    Mantle.Plan(g)
end

# Graph builders go through `invokelatest`, as they do everywhere in this suite.
# On the kernel NAME, which is not a free choice: see the note in
# `test_capture_does_not_execute.jl`. No leading underscore.
_plan(f, args...) = Base.invokelatest(f, args...)

"""Two passes over one buffer, so a submission count is not right by accident."""
function _twopassplan(dev, out, kref, n)
    g = Mantle.Graph(dev)
    for name in ("a", "b")
        Mantle.compute!(g, name) do p
            Mantle.use(p, out; read = true, write = true)
            Mantle.dispatch!(p, baked_addk!, (out, kref), n)
        end
    end
    Mantle.Plan(g)
end

"""The same graph with a constant where the `Ref` was: nothing can move."""
function _constplan(dev, out, n)
    g = Mantle.Graph(dev)
    Mantle.compute!(g, "add") do p
        Mantle.use(p, out; read = true, write = true)
        Mantle.dispatch!(p, baked_addk!, (out, Int32(5)), n)
    end
    Mantle.Plan(g)
end

@testset "a baked run means the same as an unbaked one" begin
    dev = Mantle.Device(Mantle.VulkanAPI())
    be = Mantle.defaultbackend()
    n = 32
    # Distinct, none a multiple of another, and more than `ARG_SLOTS` of them.
    steps = Int32[3, 7, 11, 13, 17, 19, 23, 29]
    @test length(steps) > Mantle.ARG_SLOTS

    # ── unbaked: the reference ────────────────────────────────────────────────
    out_u = Mantle.Buffer(dev, zeros(Int32, n))
    kref_u = Ref(Int32(0))
    pl_u = _plan(_refplan, dev, out_u, kref_u, n)
    for k in steps
        kref_u[] = k
        Mantle.run!(pl_u)
    end
    KA.synchronize(be)
    reference = Array(Mantle.storage(out_u))
    @test reference == fill(sum(steps), n)      # the Refs were honoured at all

    # ── baked: same graph, same sequence ──────────────────────────────────────
    out_b = Mantle.Buffer(dev, zeros(Int32, n))
    kref_b = Ref(Int32(0))
    pl_b = _plan(_refplan, dev, out_b, kref_b, n)

    kref_b[] = steps[1]
    Mantle.bake!(pl_b)
    KA.synchronize(be)
    # `bake!` recorded and did not run. Under the old capture this read `steps[1]`.
    @test Array(Mantle.storage(out_b)) == zeros(Int32, n)
    @test Mantle.rebindable(pl_b)
    # One recording per argument slot: that is what lets `run!` rotate.
    @test length(pl_b.baked) == Mantle.ARG_SLOTS
    # And the host-side update plan is the one dispatch, because it holds a `Ref`.
    @test length(pl_b.writes) == 1

    for k in steps
        kref_b[] = k
        Mantle.run!(pl_b)
    end
    KA.synchronize(be)

    # THE assertion. Frozen arguments give `length(steps)*steps[1]`; a slot
    # rewritten under an in-flight replay gives something in between.
    @test Array(Mantle.storage(out_b)) == reference

    Mantle.free!(pl_u)
    Mantle.free!(pl_b)
end

# A replay does not decide when to submit; the ring does.
#
# `replay!` used to reach the queue twice per run: force-close whatever batch was
# open, submit it, then submit the recording behind a hand-rolled wait on the
# newest outstanding timeline value — two `vkQueueSubmit2` calls and a semaphore
# wait, to say "these run in order on one queue", which submission order on one
# queue already says. Then it was one per run. Now it is none: the recording is
# appended to the batch the host is already building, and the batch goes when
# something needs it to — the argument ring wrapping, a flush, a readback.
#
# So `ARG_SLOTS` runs share a submission and the bound below is in terms of the
# ring rather than the run count. Asserting "one per run" would pass for the
# design that submits per replay, which is the one this replaced.
#
# Two passes, so the count is not accidentally right because there is only one
# thing to submit.
@testset "a baked run does not submit per replay" begin
    dev = Mantle.Device(Mantle.VulkanAPI())
    be = Mantle.defaultbackend()
    n = 32
    out = Mantle.Buffer(dev, zeros(Int32, n))
    kref = Ref(Int32(1))
    pl = _plan(_twopassplan, dev, out, kref, n)
    Mantle.bake!(pl)
    KA.synchronize(be)

    bq = Mantle.batchqueue(dev)
    runs = 10
    before = bq.ctx.diag.flush_counter[]
    for i in 1:runs
        kref[] = Int32(i)
        Mantle.run!(pl)
    end
    submits = bq.ctx.diag.flush_counter[] - before
    KA.synchronize(be)

    # The ring is what forces a submission, so `runs` of them cost about
    # `runs / ARG_SLOTS`. Bounded rather than pinned exactly: a flush from
    # anywhere else in the process may add one, and the claim being made is that
    # a replay does not submit on its own.
    @test submits <= cld(runs, Mantle.ARG_SLOTS) + 1
    @test submits < runs
    @test Array(Mantle.storage(out)) == fill(Int32(2 * sum(1:runs)), n)
    # And nothing is left claiming to be in flight once the device has caught up.
    Mantle.sweep!(bq, dev)
    @test isempty(Mantle.outstanding(bq))

    Mantle.free!(pl)
end

# A plan with no `Ref` anywhere does no host work between `run!` and the queue.
#
# `bake!` writes the arguments once and `rebind!` has nothing to write, so the
# update plan is empty. This is the property that makes the classification worth
# having: without it every baked run repacks every dispatch, which is the host-side
# preparation baking exists to remove.
@testset "a plan with no Ref has an empty update plan" begin
    dev = Mantle.Device(Mantle.VulkanAPI())
    be = Mantle.defaultbackend()
    n = 32
    out = Mantle.Buffer(dev, zeros(Int32, n))
    pl = _plan(_constplan, dev, out, n)
    Mantle.bake!(pl)
    KA.synchronize(be)

    @test isempty(pl.writes)
    for _ in 1:4
        Mantle.run!(pl)
    end
    KA.synchronize(be)
    @test Array(Mantle.storage(out)) == fill(Int32(20), n)

    Mantle.free!(pl)
end
