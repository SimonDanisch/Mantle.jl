# The compile output, pinned.
#
# This exists for ONE job: the five backend-independent phases (Dag, Schedule,
# Liveness, Place, Aliasing) currently live in `ext/MantleLavaExt.jl` and are
# about to be lifted into core so a second backend does not have to copy them.
# A refactor that changes what they compute must fail here, loudly, rather than
# show up later as a wrong offset on one machine.
#
# It pins the COMPILE output — scheduled order, peak bytes, liveness intervals,
# which passes carry a barrier — and deliberately not pixels or frame times.
# Those are the backend's business; these five phases are not.
#
# Needs Lava today. That is the point of the lift: once the phases are in core
# and a Host backend exists, `buildprobe` takes a host device and this whole file
# runs with no GPU present.

using Test
import Mantle
const M = Mantle

using KernelAbstractions: @kernel, @index, @Const

@kernel function goldenscale!(dst, @Const(src), a::Float32)
    i = @index(Global)
    @inbounds dst[i] = src[i] * a
end

"""
    buildprobe(dev, n) -> Graph

A chain of three passes plus two independent branches off the source.

The chain forces an order; the branches do not, so `Schedule` has a real choice
rather than one legal answer. Both branches only READ `src`, so no data
dependency links them to anything — and they still end up with barriers, because
their transients are placed on memory the chain has finished with. That is the
aliasing hazard, and having it in the probe is the reason this file is worth
more than a smoke test.
"""
function buildprobe(dev, n::Integer)
    g = M.Graph(dev)
    src = M.Buffer(dev, fill(1.0f0, n))
    t = [M.Transient.Buffer(g, Float32, n) for _ in 1:5]
    chain = (src, t[1], t[2], t[3])
    for k in 1:3
        M.compute!(g, "chain$k") do p
            M.dispatch!(p, goldenscale!, (M.use(p, chain[k + 1]; write = true),
                                          M.use(p, chain[k]; read = true), 2.0f0), n)
        end
    end
    for (k, dsti) in enumerate((4, 5))
        M.compute!(g, "branch$k") do p
            M.dispatch!(p, goldenscale!, (M.use(p, t[dsti]; write = true),
                                          M.use(p, src; read = true), Float32(k)), n)
        end
    end
    return g
end

"Everything the five phases decided, as plain data — comparable across a refactor."
compileresult(g, plan) = (
    peak       = M.peakbytes(plan),
    scheduled  = [pp.pass.name for pp in plan.passes],
    barriered  = [pp.barrier !== nothing for pp in plan.passes],
    intervals  = sort([(t.first, t.last) for t in values(g.transient_by_id)]),
)

@testset "compile output is pinned" begin
    dev = M.Device(Lava)
    n = 1 << 16
    g = buildprobe(dev, n)
    r = compileresult(g, M.Plan(g))

    # 5 transients x 65536 Float32 = 1_310_720 B if nothing aliased. The placer
    # gets it to exactly two buffers, which is what `bench/chain.jl` predicts in
    # prose ("about two buffers, not five") and what this asserts in numbers.
    @test r.peak == 524_288

    # The chain is forced; the branches are free but must not overtake it.
    @test r.scheduled == ["chain1", "chain2", "chain3", "branch1", "branch2"]

    # Every pass, including the two that share no data with anything — those are
    # ALIASING barriers. If a refactor drops them the peak stays right and the
    # picture goes wrong intermittently, which is the worst failure available.
    @test all(r.barriered)

    @test r.intervals == [(1, 2), (2, 3), (3, 3), (4, 4), (5, 5)]
end
