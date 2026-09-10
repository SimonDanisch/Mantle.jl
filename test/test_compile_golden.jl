# The compile output, pinned.
#
# This exists for ONE job: the five backend-independent phases (Dag, Schedule,
# Liveness, Place, Aliasing) live in `src/vulkan/graph.jl` and are
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
# The backend this run is for. `runtests.jl` includes this file once per
# available backend (`Mantle.eachbackend()`); a bare `include` from the REPL
# gets the default one. Nothing below names a backend, which is the point:
# these testsets check PORTABLE behaviour and used to check it on Vulkan only.
const TESTBACKEND = isdefined(Main, :MANTLE_TEST_BACKEND) ?
    Main.MANTLE_TEST_BACKEND : M.defaultbackend()


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
    dev = M.Device(TESTBACKEND)
    n = 1 << 16
    g = buildprobe(dev, n)
    r = compileresult(g, M.Plan(g))

    # 5 transients x 65536 Float32 = 1_310_720 B if nothing aliased. The placer
    # gets it to exactly two buffers, which is what `bench/chain.jl` predicts in
    # prose ("about two buffers, not five") and what this asserts in numbers.
    @test r.peak == 524_288

    # The chain is forced; the branches are free but must not overtake it. The
    # `updates` pass is first in every plan: it is where the host's stores to
    # declared `Buffer`s and `GPURef`s land (`Plan.hostwritten`), and it is
    # scheduled ahead of everything that could read them.
    @test r.scheduled == ["updates", "chain1", "chain2", "chain3", "branch1", "branch2"]

    # Every pass, including the two that share no data with anything — those are
    # ALIASING barriers. If a refactor drops them the peak stays right and the
    # picture goes wrong intermittently, which is the worst failure available.
    # Backend-aware, and only since the answer is measured (2.6). A backend
    # whose passes each commit their own command buffer on one queue is ordered
    # by commit order and derives nothing — it says so through
    # `needs_transition`, and `sync/backend.jl` asks exactly that of it. Reading
    # the answer here rather than assuming one is the difference between a
    # portable assertion and one backend's.
    #
    # Asked with the usage a storage WRITE declares, on both sides: two
    # dispatches writing the same bytes race even though the declared access did
    # not change, so a backend that derives anything derives this one. It used
    # to be asked with `Any`, which is not a `Usage` at all — the generic body
    # calls `unordered` on it and a backend that does not short-circuit gets a
    # `MethodError` instead of an answer.
    wr = M.Storage{M.BufferKind, M.WriteOnly}
    derives = M.needs_transition(M.syncbackend(dev), M.BufferKind(), wr, wr)
    @test derives ? all(r.barriered) : !any(r.barriered)

    # Pass indices are 1-based over the schedule above, so the update pass at
    # position 1 shifts every transient's interval by one.
    @test r.intervals == [(2, 3), (3, 4), (4, 4), (5, 5), (6, 6)]
end

# ── KNOWN GAP: nothing above guards `Dag` ─────────────────────────────────────
#
# MEASURED 2026-08-07, by stubbing the phase out entirely:
#
#     Mantle.run!(::Mantle.Dag, c) =
#         (Mantle.analysis(c).deps = [Int[] for _ in Mantle.passes(c)]; c)
#
# With the DAG finding NO dependencies at all, every assertion above still
# passes — identical peak, order, barriers and intervals. So this file currently
# proves nothing about `Dag`, and a refactor could break it silently.
#
# Why: `Dag` only ever makes pass j depend on an EARLIER-declared pass i
# (`for j in eachindex(ps), i in 1:(j-1)`), so declaration order is by
# construction a legal topological order, and `Overlap` prefers declaration
# order. An empty DAG therefore schedules identically. Liveness intervals come
# from `touch!` during graph construction, not from deps, so the peak does not
# move either.
#
# A first attempt at a discriminating probe declared the chain BACKWARDS and
# asserted the scheduler would fix it. It does not, and should not: the
# declaration is the program, as in RPS. That test was wrong and is not here.
#
# What discriminates is `Compact`, which actively reorders to save memory and so
# needs the edges to know what it may not move past. That is the testset below.

"""
    buildtemptation(dev, big, small) -> Graph

A graph `Compact` wants to schedule in an order the DAG forbids.

`fill` writes a large transient: pure allocation, no free, the worst memory score
available. `drain` reads it and writes a tiny one: it FREES the large buffer, so
`Compact`'s memory term wants it first — and it is declared second, so even the
declaration-order tiebreak does not rescue the right answer.

Only the dependency stops it. That is the point: `buildprobe` cannot see a broken
`Dag` (measured — see the note above), and neither can `buildprobe` under
`Compact`: its two branches genuinely have no dependencies, so a correct DAG and
an empty one agree, and both produce
`["branch1", "branch2", "chain1", "chain2", "chain3"]`. Here they disagree.
"""
function buildtemptation(dev, big::Integer, small::Integer)
    g = M.Graph(dev)
    src = M.Buffer(dev, fill(1.0f0, big))
    t = M.Transient.Buffer(g, Float32, big)
    u = M.Transient.Buffer(g, Float32, small)
    M.compute!(g, "fill") do p
        M.dispatch!(p, goldenscale!, (M.use(p, t; write = true),
                                      M.use(p, src; read = true), 2.0f0), big)
    end
    M.compute!(g, "drain") do p
        M.dispatch!(p, goldenscale!, (M.use(p, u; write = true),
                                      M.use(p, t; read = true), 3.0f0), small)
    end
    return g
end

"""
The backend's compilation context, so a phase can be run and read on its own.

`Plan` exposes only what a frame needs. Asserting a phase's OUTPUT means reaching
the context it writes into, which is what `compile!(ctx, prefix)` was built for.
"""
compilectx(g; kw...) = Mantle.Compile(g; kw...)

@testset "Dag: the edges themselves" begin
    dev = M.Device(TESTBACKEND)
    c = M.compile!(compilectx(buildtemptation(dev, 1 << 18, 1 << 4)), (M.Dag(),))
    # "drain" reads what "fill" wrote. One edge, and it points backwards.
    @test M.analysis(c).deps == [Int[], [1]]

    c2 = M.compile!(compilectx(buildprobe(dev, 1 << 12)), (M.Dag(),))
    # chain2 <- chain1, chain3 <- chain2. The two branches only READ `src`, and
    # read-after-read is deliberately not an edge, so they depend on nothing.
    @test M.analysis(c2).deps == [Int[], [1], [2], Int[], Int[]]
end

# Asserted directly, and not through the schedule, for a measured reason: THREE
# attempts to observe `Dag` through `Schedule` all came back identical with the
# phase stubbed out to find nothing.
#
#   buildprobe under Overlap  — same order, peak, barriers, intervals
#   buildprobe under Compact  — same order (["branch1","branch2","chain1",…])
#   buildtemptation, Compact  — same order (["fill","drain"])
#
# The last is the interesting one. It was built specifically so `Compact` would
# want to hoist the pass that frees a megabyte, and it does not: reading a
# transient nothing has written yet counts that transient as an ALLOCATION in the
# memory term, so the illegal order already scores worst. The scoring disfavours
# illegal schedules on its own, which makes the edges nearly unobservable
# downstream on graphs this size — and is why guarding `Dag` by pinning a
# schedule is guarding nothing.

@testset "Schedule: the order itself, and that the policy decides it" begin
    dev = M.Device(TESTBACKEND)
    prefix = (M.Dag(), M.Schedule())
    order(g; kw...) = M.analysis(M.compile!(compilectx(g; kw...), prefix)).order

    # Overlap prefers declaration order and there is nothing forcing it off.
    @test order(buildprobe(dev, 1 << 12)) == [1, 2, 3, 4, 5]

    # Compact prefers whatever frees more than it allocates, and reaches a
    # DIFFERENT legal order on the same graph. Pinned exactly, because "they
    # differ" alone would survive a scoring change that made them differ wrongly.
    @test order(buildprobe(dev, 1 << 12); policy = M.Compact()) == [4, 5, 1, 2, 3]

    # …and no policy may put `drain` before the `fill` it reads.
    @test order(buildtemptation(dev, 1 << 18, 1 << 4); policy = M.Compact()) == [1, 2]
end

@testset "alias = false gives every transient the whole timeline" begin
    dev = M.Device(TESTBACKEND)
    g = buildprobe(dev, 1 << 16)
    plan = M.Plan(g; alias = false)
    # Nothing may share bytes, so the peak IS the naive sum — 5 x 65536 Float32.
    # This is what guards `Liveness`: the intervals are the only thing `alias`
    # changes, and if they stopped being derived from use this would still come
    # back 524_288 like the aliased case.
    @test M.peakbytes(plan) == 1_310_720
end
