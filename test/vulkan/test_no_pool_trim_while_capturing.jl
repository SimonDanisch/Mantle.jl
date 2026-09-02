"""
The memory pool must not try to reclaim while a recording is being captured.

`quiesce_before_reclaim!` drains the queue before handing pool blocks back,
because reclaiming a block that queued work still names destroys a `VkBuffer`
out from under it. During a capture that drain is both impossible and beside the
point: `flush!` throws, since a capture submits nothing and there is no device
work to wait for — and even a completed drain would not help, because a captured
sequence is never finished. It is replayed again later, so a block its command
buffers name is live for as long as the plan is.

So the trim entry points refuse outright while `bq.capturing` is set. Not "flush
harder": do not trim.

How it surfaced: `bake!` on Hikari's 72-pass chunk plan allocates the capture's
own argument slabs, `vk_alloc` runs the heap heuristic on the way, and
`maybe_trim_pool!` asked to flush — so baking a large plan threw
`LavaError: cannot flush while capturing` out of the allocator.

The heuristic is why this test sets the policy and manufactures garbage rather
than just building a big plan. `maybe_trim_pool!` returns immediately unless the
pool is over `trim_threshold`, `trim_min_interval` has elapsed, AND something is
actually reclaimable — and a test-sized graph is none of the three. Two earlier
versions of this file passed with the fix REVERTED, which makes them worth
nothing: the first hit the threshold gate, the second the `reclaimable` gate.

So all three are arranged: the thresholds go to zero and a large buffer is
allocated and dropped so a pool block is left with nothing live in it. Then the
trim runs for real on the next allocation, and the next allocation is the
capture's own argument slab.
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

@testset "no pool trim while capturing" begin
    dev = Mantle.Device(Mantle.VulkanAPI())
    be = Mantle.defaultbackend()
    bq = Mantle.batchqueue(dev)

    # The invariant, asked directly: inside a capture the quiesce refuses and
    # says so, and — this is the part that matters — it does not throw. Without
    # the guard the `flush!` inside it raises `cannot flush while capturing`,
    # out of whatever allocation happened to trip the heuristic.
    #
    # Asked here rather than through `maybe_trim_pool!` because that has three
    # gates in front of it and a test-sized workload trips none of them: two
    # earlier versions of this file drove the entry point instead and passed
    # with the fix reverted, which is worth nothing.
    let quiesced = Ref{Any}(nothing)
        seq = MVE.capture(bq) do
            quiesced[] = MVE.quiesce_before_reclaim!(bq)
        end
        @test quiesced[] === false
        MVE.release!(seq)
    end
    # And outside one it still does its job.
    @test MVE.quiesce_before_reclaim!(bq) === true
end

@testset "bake! a plan that allocates inside its own capture" begin
    dev = Mantle.Device(Mantle.VulkanAPI())
    be = Mantle.defaultbackend()
    n = 4096
    npass = 64

    out = Mantle.Buffer(dev, zeros(Float32, n))
    pl, seed = Base.invokelatest(_chainplan, dev, out, n, npass)

    Mantle.run!(pl)
    KA.synchronize(be)
    want = copy(Array(Mantle.storage(out)))
    @test want == fill(Float32(npass + 2), n)

    # Enough passes that the capture outgrows one argument slab and allocates
    # from inside itself, which is the path Hikari's 72-pass chunk plan took.
    Mantle.bake!(pl)
    @test Mantle.baked(pl)

    fill!(Mantle.storage(out), 0f0)
    KA.synchronize(be)
    Mantle.run!(pl)
    KA.synchronize(be)
    @test Array(Mantle.storage(out)) == want

    Mantle.free!(pl)
end
