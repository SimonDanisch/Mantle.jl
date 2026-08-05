# Differential and validation fuzzing of the barrier derivation.
#
# Elision is where races come from, and a race that happens to produce the right
# answer is not evidence of anything. So this checks two things per case:
#
#   1. synchronization validation reports nothing, which is what actually
#      detects a missing barrier
#   2. the results match the same graph run with the backend's unconditional
#      per-dispatch barrier
#
# Graphs are random DAGs by construction: pass i writes resource i and reads only
# resources written earlier, so nothing is ever read before it is written and the
# comparison is deterministic.

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
function random_graph(dev, rng, n, passes; policy = M.Overlap(), coalesce = true)
    g = M.Graph(dev)
    seed = M.Buffer(dev, fill(1.0f0, n))
    out = M.Buffer(dev, zeros(Float32, n))
    bufs = [M.Transient.Buffer(g, Float32, n) for _ in 1:passes]
    sources = Any[seed]
    for i in 1:passes
        a = sources[rand(rng, 1:length(sources))]
        b = sources[rand(rng, 1:length(sources))]
        dst = bufs[i]
        M.compute!(g, "p$i") do p
            ra = M.use(p, a; read = true)
            rb = M.use(p, b; read = true)
            w = M.use(p, dst; write = true)
            M.dispatch!(p, mix!, (w, ra, rb, Float32(i)), n)
        end
        push!(sources, dst)
    end
    M.compute!(g, "out") do p
        M.dispatch!(p, keep!, (M.use(p, out; write = true),
                               M.use(p, bufs[end]; read = true)), n)
    end
    (; g, seed, out, bufs, plan = M.Plan(g; policy, coalesce))
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
Run one random graph both ways and compare, and check both against the CPU
replay. Returns the barrier counts, whether the two agreed, and whether either
was right.
"""
function check(dev, seed; n = 4096, passes = 12, policy = M.Overlap())
    rng = MersenneTwister(seed)
    s = random_graph(dev, rng, n, passes; policy)
    want = expected(MersenneTwister(seed), passes)

    function once(mode)
        M.update!(s.seed, fill(1.0f0, n))
        M.run!(s.plan; barriers = mode)
        Lava.flush!(dev.bq, dev.ctx.device)
        Array(M.storage(s.out))
    end
    reference = once(:backend)
    got = once(:derived)

    # What coalescing removed, measured rather than guessed at. `passes - 1` was
    # the old proxy for "every barrier there could be", and it undercounts: a plan
    # is replayed, so the first pass needs one against the previous frame too.
    # Comparing against the same graph compiled with `coalesce = false` asks the
    # question the number is for, and does not move when a mask gets wider.
    (seed = seed,
     emitted = emitted(s.plan),
     uncoalesced = emitted(random_graph(dev, MersenneTwister(seed), n, passes;
                                        policy, coalesce = false).plan),
     possible = length(s.plan.passes) - 1,
     agree = got == reference,
     correct = all(==(want), got) && all(==(want), reference))
end

"""Sweep seeds. With validation on, `messages` is the number that matters."""
function fuzz(seeds = 1:25; n = 4096, passes = 12, validate = false, policy = M.Overlap())
    dev = M.Device(Lava)
    validate && Lava.clear_validation_messages!()
    results = [check(dev, s; n, passes, policy) for s in seeds]
    msgs = validate ? Lava.get_validation_messages() : []
    (cases = length(results),
     wrong = count(r -> !r.correct, results),
     disagreements = count(r -> !r.agree, results),
     emitted_total = sum(r -> r.emitted, results),
     uncoalesced_total = sum(r -> r.uncoalesced, results),
     possible_total = sum(r -> r.possible, results),
     messages = length(msgs),
     detail = unique(msgs))
end
