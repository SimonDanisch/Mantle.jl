"""
The memory pool must not reclaim blocks while a plan holds a recording.

`quiesce_before_reclaim!` drains the queue before handing pool blocks back,
because reclaiming a block that queued work still names destroys a `VkBuffer`
out from under it. A recording cannot be drained into safety: it is submitted
again on the next `run!`, so a block its command buffer names is live for as long
as the plan is. Waiting does not change that.

So the trim refuses outright. Not "flush harder": do not trim.

**The question it asks changed, and the old one was wrong.** It used to read
`bq.capturing !== nothing` — whether a capture was OPEN — which is true only
while `bake!` is running and false for every recording that had already been
taken. So the guard covered the allocation `bake!` itself made and nothing
afterwards. It asks the pool about its tenants now (`movable`), which is a
property of the recordings that exist rather than of what the queue is doing, and
it is the same question arena growth already refuses on.

How the original surfaced: `bake!` on Hikari's 72-pass chunk plan allocated the
capture's own argument slabs, `vk_alloc` ran the heap heuristic on the way, and
`maybe_trim_pool!` asked to flush — so recording a large plan threw
`LavaError: cannot flush while capturing` out of the allocator.

The heuristic is why this test sets the policy and manufactures garbage rather
than just building a big plan. `maybe_trim_pool!` returns immediately unless the
pool is over `trim_threshold`, `trim_min_interval` has elapsed, AND something is
actually reclaimable — and a test-sized graph is none of the three. Two earlier
versions of this file passed with the fix REVERTED, which makes them worth
nothing: the first hit the threshold gate, the second the `reclaimable` gate.
"""

using Test, Mantle, Lava, KernelAbstractions
const KA = KernelAbstractions

@kernel function trimchain_bump!(dst, @Const(src))
    i = @index(Global)
    @inbounds dst[i] = src[i] + 1f0
end

"""
A chain of `npass` passes, each reading the previous transient and adding one.

Its own function, and reached through `invokelatest`, as every graph builder in
this suite is.
"""
function _chainplan(dev, out, n, npass)
    g = Mantle.Graph(dev)
    seed = Mantle.Buffer(dev, zeros(Float32, n))
    t = [Mantle.Transient.Buffer(g, Float32, n) for _ in 1:(npass + 1)]
    Mantle.compute!(g, "seed") do p
        s = Mantle.use(p, seed; read = true); d = Mantle.use(p, t[1]; write = true)
        Mantle.dispatch!(p, trimchain_bump!, (d, s), n)
    end
    for k in 1:npass
        Mantle.compute!(g, "s$k") do p
            s = Mantle.use(p, t[k]; read = true); d = Mantle.use(p, t[k + 1]; write = true)
            Mantle.dispatch!(p, trimchain_bump!, (d, s), n)
        end
    end
    Mantle.compute!(g, "out") do p
        s = Mantle.use(p, t[end]; read = true); d = Mantle.use(p, out; write = true)
        Mantle.dispatch!(p, trimchain_bump!, (d, s), n)
    end
    # `seed` is only reachable from the closure above, and the plan has to keep
    # it alive for as long as the recording names it.
    (Mantle.Plan(g), seed)
end

@testset "no pool trim while a plan is recorded" begin
    dev = Mantle.Device(Mantle.VulkanAPI())
    bq = Mantle.batchqueue(dev)
    n = 256

    # The invariant, asked directly: with a recorded plan alive the quiesce
    # refuses and says so, and — this is the part that matters — it does not
    # throw. Asked here rather than through `maybe_trim_pool!` because that has
    # three gates in front of it and a test-sized workload trips none of them:
    # two earlier versions of this file drove the entry point instead and passed
    # with the fix reverted, which is worth nothing.
    out = Mantle.Buffer(dev, zeros(Float32, n))
    pl, _seed = Base.invokelatest(_chainplan, dev, out, n, 4)
    Mantle.record!(pl)
    @test !Mantle.remappable(pl)
    @test !Mantle.movable(Mantle.pool(dev))
    @test Mantle.quiesce_before_reclaim!(bq) === false

    # The converse — that it says `true` again once nothing is recorded — is NOT
    # asserted, and the reason is the guard's actual breadth: ONE recorded plan
    # alive anywhere in the process is enough, and a suite that has built several
    # and not freed them all makes the answer `false` for reasons that have
    # nothing to do with this plan. Asserting it here would pass or fail on file
    # order. What is local, and is what a regression would break, is that
    # recording a plan makes the pool unmovable and the trim refuse.
    Mantle.free!(pl)
    @test Mantle.remappable(pl)
end

@testset "record! a plan that allocates while it is being written" begin
    dev = Mantle.Device(Mantle.VulkanAPI())
    be = Mantle.defaultbackend()
    n = 4096
    npass = 64

    out = Mantle.Buffer(dev, zeros(Float32, n))
    pl, seed = Base.invokelatest(_chainplan, dev, out, n, npass)

    # Enough passes that recording outgrows one unified block and allocates from
    # inside itself, which is the path Hikari's 72-pass chunk plan took.
    Mantle.record!(pl)
    @test Mantle.recorded(pl)

    Mantle.run!(pl)
    KA.synchronize(be)
    @test Array(Mantle.storage(out)) == fill(Float32(npass + 2), n)

    fill!(Mantle.storage(out), 0f0)
    KA.synchronize(be)
    Mantle.run!(pl)
    KA.synchronize(be)
    @test Array(Mantle.storage(out)) == fill(Float32(npass + 2), n)

    Mantle.free!(pl)
end
