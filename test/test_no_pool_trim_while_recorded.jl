"""
The memory pool must not reclaim blocks while a plan holds a recording.

What is asserted, on every backend: recording a plan makes the pool report that
its tenants cannot move (`movable`), and a full trim with a recorded plan alive —
`makeroom!`, which collects, waits for the device and hands back every block
nothing uses — neither throws nor takes anything the recording names: the plan
runs afterwards and computes what it did before.

The history, which is the Vulkan backend's. `quiesce_before_reclaim!` drains the
queue before handing pool blocks back, because reclaiming a block that queued
work still names destroys a `VkBuffer` out from under it. A recording cannot be
drained into safety: it is submitted again on the next `run!`, so a block its
command buffer names is live for as long as the plan is. Waiting does not change
that.

So the trim refuses outright. Not "flush harder": do not trim.

**The question has to be about the recording, not about a capture in
progress.** `bq.capturing !== nothing` — whether a capture is OPEN — is true
only while `bake!` is running and false for every recording that had already been
taken. So the guard covered the allocation `bake!` itself made and nothing
afterwards. It asks the pool about its tenants now (`movable`), which is a
property of the recordings that exist rather than of what the queue is doing, and
it is the same question arena growth already refuses on.

How the original surfaced: `bake!` on Hikari's 72-pass chunk plan allocated the
capture's own argument slabs, `vk_alloc` ran the heap heuristic on the way, and
`maybe_trim_pool!` asked to flush — so recording a large plan threw
`LavaError: cannot flush while capturing` out of the allocator.

The heuristic is why this test does not go through it. `maybe_trim_pool!`
returns immediately unless the pool is over `trim_threshold`, `trim_min_interval`
has elapsed, AND something is actually reclaimable — and a test-sized graph is
none of the three. Two earlier versions of this file passed with the fix
REVERTED, which makes them worth nothing: the first hit the threshold gate, the
second the `reclaimable` gate. The trim below is asked for directly.
"""

using Test, Mantle, KernelAbstractions
include(joinpath(@__DIR__, "testbackend.jl"))
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
function chainplan(dev, out, n, npass)
    g = Mantle.Graph(dev)
    seed = Mantle.Buffer(dev, zeros(Float32, n))
    t = [Mantle.Transient.Buffer(g, Float32, n) for _ in 1:(npass + 1)]
    Mantle.dispatch!(g, trimchain_bump!, (t[1], seed), n; name = "seed")
    for k in 1:npass
        Mantle.dispatch!(g, trimchain_bump!, (t[k + 1], t[k]), n; name = "s$k")
    end
    Mantle.dispatch!(g, trimchain_bump!, (out, t[end]), n; name = "out")
    # `seed` is only reachable from the closure above, and the plan has to keep
    # it alive for as long as the recording names it.
    (Mantle.Plan(g), seed)
end

@testset "no pool trim while a plan is recorded" begin
    dev = Mantle.Device(TESTBACKEND)
    sp = Mantle.pool(dev)
    n = 256
    npass = 4

    out = Mantle.Buffer(dev, zeros(Float32, n))
    pl, _seed = Base.invokelatest(chainplan, dev, out, n, npass)
    Mantle.record!(pl)
    # Recording is what pins the plan's memory: its commands hold the addresses
    # its regions have today, and nothing rewrites a command buffer.
    @test !Mantle.remappable(pl)
    @test !Mantle.movable(sp)

    # THE assertion: a trim with the recording alive. It was asked here rather
    # than through `maybe_trim_pool!` because that has three gates in front of it
    # and a test-sized workload trips none of them: two earlier versions of this
    # file drove the entry point instead and passed with the fix reverted, which
    # is worth nothing. Throwing here is the original failure; a block the
    # recording names going back to the device is the one the guard exists for,
    # and it shows as a wrong result or a lost device in the run below.
    Mantle.makeroom!(sp, dev)
    Mantle.run!(pl)
    Mantle.waitfor!(pl)
    @test Array(Mantle.storage(out)) == fill(Float32(npass + 2), n)

    # The converse — that the pool is movable again once nothing is recorded — is
    # NOT asserted, and the reason is the property's actual breadth: ONE recorded
    # plan alive anywhere in the process is enough, and a suite that has built
    # several and not freed them all makes the answer `false` for reasons that
    # have nothing to do with this plan. Asserting it here would pass or fail on
    # file order. What is local, and is what a regression would break, is that
    # recording a plan pins it and a trim leaves it working.
    Mantle.free!(pl)
    @test Mantle.remappable(pl)
end

@testset "record! a plan that allocates while it is being written" begin
    dev = Mantle.Device(TESTBACKEND)
    be = TESTBACKEND
    n = 4096
    npass = 64

    out = Mantle.Buffer(dev, zeros(Float32, n))
    pl, seed = Base.invokelatest(chainplan, dev, out, n, npass)

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
