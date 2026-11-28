# Fuzzing of the barrier derivation over random DAGs: each result is checked
# against a CPU replay, and with synchronization validation on, the validation
# messages are what detect a missing barrier. Pass i writes resource i and reads
# only resources written earlier, so the result is deterministic.

import Mantle
using Lava, KernelAbstractions, Random
const M = Mantle

@kernel function mix!(dst, @Const(a), @Const(b), k::Float32)
    i = @index(Global)
    @inbounds dst[i] = a[i] * 0.5f0 + b[i] * 0.25f0 + k
end

@kernel function keep!(dst, @Const(src))
    i = @index(Global)
    @inbounds dst[i] = src[i]
end

"""
A random DAG of `passes` compute passes.

`seed` is persistent and reset before each run: a transient that nothing writes
would be an uninitialised read whose contents differ between runs, which looks
exactly like a barrier bug and is not one.

`out` is persistent for the same class of reason. Reading `bufs[end]` back looks
safe under declaration order, where it is written last and so nothing can alias
it, but a scheduler is free to move that pass anywhere its dependencies allow and
a later transient then legitimately takes over its bytes. Only a resource the
placer never touches is comparable after the run.
"""
function random_graph(dev, rng, n, passes; policy = M.Overlap())
    g = M.Graph(dev)
    seed = M.Buffer(dev, fill(1.0f0, n))
    out = M.Buffer(dev, zeros(Float32, n))
    bufs = [M.Transient.Buffer(g, Float32, n) for _ in 1:passes]
    sources = Any[seed]
    for i in 1:passes
        a = sources[rand(rng, 1:length(sources))]
        b = sources[rand(rng, 1:length(sources))]
        dst = bufs[i]
        M.dispatch!(g, mix!, (dst, a, b, Float32(i)), n; name = "p$i")
        push!(sources, dst)
    end
    M.dispatch!(g, keep!, (out,
                               bufs[end]), n; name = "out")
    (; g, seed, out, bufs, plan = M.record!(M.Plan(g; policy)))
end

"""
What the DAG computes, replayed on the CPU.

Every buffer starts uniform and `mix!` is elementwise, so each pass produces one
Float32 and the whole graph collapses to twelve scalars. That makes the check
absolute rather than differential: two schedules agreeing proves only that they
agree.
"""
function expected(rng, passes)
    vals = Float32[1.0f0]
    for i in 1:passes
        a = vals[rand(rng, 1:length(vals))]
        b = vals[rand(rng, 1:length(vals))]
        push!(vals, a * 0.5f0 + b * 0.25f0 + Float32(i))
    end
    vals[end]
end

emitted(plan) = count(pp -> !isempty(pp.pre), plan.passes)

"""
Run one random graph and check it against the CPU replay. Returns the number of
passes with barriers and whether the result was right.
"""
function check(dev, seed; n = 4096, passes = 12, policy = M.Overlap())
    s = random_graph(dev, MersenneTwister(seed), n, passes; policy)
    want = expected(MersenneTwister(seed), passes)
    M.update!(s.seed, fill(1.0f0, n))
    M.run!(s.plan)
    M.waitfor!(s.plan)
    (seed = seed,
     emitted = emitted(s.plan),
     possible = length(s.plan.passes) - 1,
     correct = all(==(want), Array(M.storage(s.out))))
end

"""Sweep seeds. With validation on, `messages` is the number that matters."""
function fuzz(seeds = 1:25; n = 4096, passes = 12, validate = false, policy = M.Overlap())
    dev = M.Device(M.VulkanAPI())
    validate && Mantle.clear_validation_messages!()
    results = [check(dev, s; n, passes, policy) for s in seeds]
    msgs = validate ? Mantle.get_validation_messages() : []
    (cases = length(results),
     wrong = count(r -> !r.correct, results),
     emitted_total = sum(r -> r.emitted, results),
     possible_total = sum(r -> r.possible, results),
     messages = length(msgs),
     detail = unique(msgs))
end
