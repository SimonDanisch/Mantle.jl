# Stress the barrier derivation against the per-resource hazard set.
#
# The question is not "did the answer come out right" — a race that happens to
# produce the right bytes proves nothing, and on a GPU that serialises anyway it
# will produce them for hours. The question is whether the *emitted set* is the
# set a scoped barrier implementation must emit: one entry per genuine state
# change on each resource's own usage sequence.
#
# The oracle is the same graph compiled with `coalesce = false`, i.e. the
# per-resource walk with nothing removed on top. That walk is confirmed by hand
# against a written-out derivation in test_window.jl ("the emitted set is the
# per-resource hazard set, exactly"); here it is the reference and *coalescing*
# is what is under test. Two directions, because they are different bugs:
#
#   missing  — a hazard on some resource that no barrier on that resource
#              enforces. Correct only while barriers are global.
#   spurious — an entry the hazard walk does not call for: serialisation
#              nobody asked for.
#
# Graphs are generated two ways. `usage_graph` puts random read/write/readwrite
# usages on persistent buffers, so WAR, WAW, RAW and read-after-read all appear
# without any liveness constraint. `chain_graph` follows the write-then-read
# discipline over transients so the placer aliases them and the handover barrier
# is exercised.

import Mantle
using Lava, KernelAbstractions, Random
const M = Mantle

@kernel function touch1!(a)
    i = @index(Global)
    @inbounds a[i] = a[i] * 1.0001f0 + 1f0
end
@kernel function touch2!(a, b)
    i = @index(Global)
    @inbounds a[i] = a[i] * 1.0001f0 + b[i]
end
@kernel function touch3!(a, b, c)
    i = @index(Global)
    @inbounds a[i] = a[i] * 1.0001f0 + b[i] + c[i]
end
@kernel function touch4!(a, b, c, d)
    i = @index(Global)
    @inbounds a[i] = a[i] * 1.0001f0 + b[i] + c[i] + d[i]
end
const TOUCH = (touch1!, touch2!, touch3!, touch4!)

"""
Random usages over persistent buffers.

No transient means no liveness rule to satisfy, so a pass may read, write or do
both to any buffer in any order — which is what makes every hazard kind appear.
Never run: the point is the emitted set, and an uninitialised read would only
make the bytes undefined, not the barriers.
"""
function usage_graph(dev, rng, bufs, npasses; n = 256, coalesce = true,
                     policy = M.Overlap(), slicing = 0.0)
    g = M.Graph(dev)
    # Range bounds drawn from a small fixed set, so the generator produces
    # identical, disjoint *and* partially overlapping slices of one buffer rather
    # than ranges that never interact.
    cuts = [1, n ÷ 4, n ÷ 3, n ÷ 2, (3n) ÷ 4, n]
    for i in 1:npasses
        k = rand(rng, 1:min(4, length(bufs)))
        picks = shuffle(rng, collect(eachindex(bufs)))[1:k]
        flags = [(rand(rng, Bool), rand(rng, Bool)) for _ in picks]
        # A usage has to be at least one of the two.
        flags = [(r || w) ? (r, w) : (true, false) for (r, w) in flags]
        ranges = map(_ -> if rand(rng) < slicing
                         a, b = rand(rng, cuts), rand(rng, cuts)
                         a > b && ((a, b) = (b, a))
                         a:max(b, a)
                     else
                         nothing
                     end, picks)
        M.compute!(g, "p$i") do p
            args = ntuple(length(picks)) do j
                M.use(p, bufs[picks[j]]; read = flags[j][1], write = flags[j][2],
                      range = ranges[j])
            end
            M.dispatch!(p, TOUCH[length(picks)], args, n)
        end
    end
    (; g, plan = M.Plan(g; coalesce, policy))
end

"""
Write-then-read over transients, so the placer aliases them and the handover
barrier — the one no per-resource sequence can see — is in the emitted set.
"""
function chain_graph(dev, rng, npasses; n = 256, alias = true, coalesce = true,
                     policy = M.Overlap())
    g = M.Graph(dev)
    seed = M.Buffer(dev, fill(1f0, n))
    bufs = [M.Transient.Buffer(g, Float32, n) for _ in 1:npasses]
    sources = Any[seed]
    for i in 1:npasses
        a = sources[rand(rng, 1:length(sources))]
        b = sources[rand(rng, 1:length(sources))]
        M.compute!(g, "p$i") do p
            M.dispatch!(p, touch3!, (M.use(p, bufs[i]; write = true),
                                     M.use(p, a; read = true),
                                     M.use(p, b; read = true)), n)
        end
        push!(sources, bufs[i])
    end
    out = M.Buffer(dev, zeros(Float32, n))
    M.compute!(g, "out") do p
        M.dispatch!(p, touch2!, (M.use(p, out; write = true),
                                 M.use(p, bufs[end]; read = true)), n)
    end
    (; g, plan = M.Plan(g; alias, coalesce, policy))
end

"""Every emitted transition, keyed so two compiles are comparable."""
hazards(plan) = Set((pp.pass.name, t.resource, t.from, t.to, Set(t.waits))
                    for pp in plan.passes for t in pp.pre)

"""
A resource id as `(parent, span)`, so the emitted set and the oracle are keyed by
what the id *means* rather than by the number the compiler happened to hand out.
Without this the comparison would only work because both sides call the same
interning table, which is the circularity the oracle exists to avoid.
"""
function normalid(g, id::Int)
    ext = Base.get_extension(Mantle, :MantleLavaExt)
    r = get(g.ids.by_id, id, nothing)
    # `get`, not `resourceid` — an oracle that registers an id while reading the
    # graph would be changing what the next compile sees in order to check it.
    r isa ext.BufferRange ? (get(g.ids, r.parent, 0), r.range) : (id, nothing)
end

normalized(g, plan) = Set((pp.pass.name, normalid(g, t.resource), t.from, t.to, Set(t.waits))
                          for pp in plan.passes for t in pp.pre)

"""
The local barrier set a scoped implementation has to emit, from the declared
usages alone.

Built from the rules rather than from a compile: `needs_transition` decides
whether a state change is a hazard, `writes` decides whether the wait set is the
one previous state or every reader outstanding since the last write. Seeded from
each resource's *last* use in the schedule, because a plan is replayed and frame
N+1's first use follows frame N's last.

This is the oracle. It must not be `coalesce = false` — that is the compiler's
own walk, and comparing a thing to itself proves nothing. Their agreement over a
corpus is what says the walk is right; equality with what is actually *emitted*
is what says coalescing has not removed a barrier that only globality justified.
"""
function local_hazards(g, plan)
    be = M.Vulkan()
    ext = Base.get_extension(Mantle, :MantleLavaExt)

    # The partition, derived here rather than read off the compiler: every range
    # declared anywhere cuts its buffer, and a usage stands for the pieces its
    # own range covers. A usage naming no range covers all of them. Keys are
    # `(parent, span)` throughout, so nothing depends on how ids were handed out.
    ranges = Dict{Int,Vector{UnitRange{Int}}}()
    for pp in plan.passes, (id, _) in pp.pass.usages
        (pid, span) = normalid(g, id)
        span === nothing && continue
        push!(get!(ranges, pid, UnitRange{Int}[]), span)
    end
    spansof = Dict{Int,Vector{UnitRange{Int}}}()
    for (pid, rs) in ranges
        n = length(g.ids.by_id[pid])
        cuts = sort!(unique!(vcat([1, n + 1], first.(rs), last.(rs) .+ 1)))
        spansof[pid] = filter(!isempty, [cuts[k]:(cuts[k + 1] - 1) for k in 1:(length(cuts) - 1)])
    end
    function pieces(id)
        (pid, span) = normalid(g, id)
        haskey(spansof, pid) || return [(pid, span)]
        [(pid, s) for s in spansof[pid] if span === nothing ||
                                           (first(s) >= first(span) && last(s) <= last(span))]
    end

    final = Dict{Any,Type}()
    for pp in plan.passes, (id, U) in pp.pass.usages, key in pieces(id)
        final[key] = U
    end
    cur = Dict{Any,Any}(); pend = Dict{Any,Vector{Type}}(); req = Set()
    for pp in plan.passes, (id, U) in pp.pass.usages, key in pieces(id)
        if !haskey(cur, key)
            cur[key] = get(final, key, nothing); pend[key] = Type[]
        end
        from = cur[key]
        if from !== nothing && M.needs_transition(be, ext.resourcekind(g.ids.by_id[id]), from, U)
            push!(req, (pp.pass.name, key, from, U,
                        Set(M.writes(U) ? unique(vcat(from, pend[key])) : Type[from])))
            cur[key] = U
        end
        M.writes(U) ? (pend[key] = Type[]) : push!(pend[key], U)
    end
    req
end

"""Which usage kind each side of a dropped hazard was, for the breakdown."""
kindof(T) = string(nameof(T isa UnionAll ? T.body.name.wrapper : typeof(T).name.wrapper))

"""
One case both ways. Two graphs, not two plans off one graph: recompiling a graph
rebinds images that are already bound (VUID-vkBindImageMemory-image-07460), so
the second plan is not a clean comparison.
"""
function case(dev, seed; kind = :usage, npasses = 12, nbufs = 5, n = 256,
              alias = true, policy = M.Overlap())
    mk(co) = if kind === :usage
        rng = MersenneTwister(seed)
        bufs = [M.Buffer(dev, fill(Float32(i), n)) for i in 1:nbufs]
        usage_graph(dev, rng, bufs, npasses; n, coalesce = co, policy)
    else
        chain_graph(dev, MersenneTwister(seed), npasses; n, alias, coalesce = co, policy)
    end
    required = hazards(mk(false).plan)
    got = hazards(mk(true).plan)
    (seed = seed,
     required = length(required),
     emitted = length(got),
     missing = setdiff(required, got),
     spurious = setdiff(got, required))
end

"""
Sweep. Reports how much of the hazard set survives coalescing, and what kind of
hazard the dropped ones are.
"""
function stress(seeds = 1:200; kind = :usage, npasses = 12, nbufs = 5,
                alias = true, policy = M.Overlap(), n = 256)
    dev = M.Device(Lava)
    rs = [case(dev, s; kind, npasses, nbufs, n, alias, policy) for s in seeds]
    miss = sum(r -> length(r.missing), rs)
    spur = sum(r -> length(r.spurious), rs)
    req = sum(r -> r.required, rs)
    bykind = Dict{String,Int}()
    for r in rs, m in r.missing
        k = kindof(m[3]) * "->" * kindof(m[4])
        bykind[k] = get(bykind, k, 0) + 1
    end
    (cases = length(rs),
     required = req,
     emitted = sum(r -> r.emitted, rs),
     missing = miss,
     spurious = spur,
     missing_frac = req == 0 ? 0.0 : round(miss / req, digits = 3),
     cases_with_missing = count(r -> !isempty(r.missing), rs),
     cases_with_spurious = count(r -> !isempty(r.spurious), rs),
     by_transition = sort(collect(bykind), by = x -> -x[2]))
end
