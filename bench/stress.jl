# Random graphs and a rule-derived oracle for the barrier derivation: the
# emitted set must be the per-resource hazard set, with no entry missing (a race)
# and none extra (serialisation nobody asked for). `test_window.jl` runs it.
#
# `usage_graph` puts random read/write/readwrite usages on persistent buffers,
# so WAR, WAW, RAW and read-after-read all appear without any liveness constraint.

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

What the generator varies is which buffers a pass takes, in which order, and
which slice of each. The read/write pattern is `touchN!`'s — it writes its first
argument and reads the rest — so one buffer landing in position 1 of one pass and
position 3 of another is what produces a write-after-read, a read-after-write and
a write-after-write between them.
"""
function usage_graph(dev, rng, bufs, npasses; n = 256, policy = M.Overlap(), slicing = 0.0)
    g = M.Graph(dev)
    # Range bounds drawn from a small fixed set, so the generator produces
    # identical, disjoint *and* partially overlapping slices of one buffer rather
    # than ranges that never interact.
    cuts = [1, n ÷ 4, n ÷ 3, n ÷ 2, (3n) ÷ 4, n]
    for i in 1:npasses
        k = rand(rng, 1:min(4, length(bufs)))
        picks = shuffle(rng, collect(eachindex(bufs)))[1:k]
        ranges = map(_ -> if rand(rng) < slicing
                         a, b = rand(rng, cuts), rand(rng, cuts)
                         a > b && ((a, b) = (b, a))
                         a:max(b, a)
                     else
                         nothing
                     end, picks)
        args = ntuple(length(picks)) do j
            b = bufs[picks[j]]
            ranges[j] === nothing ? b : M.slice(g, b, ranges[j])
        end
        M.dispatch!(g, TOUCH[length(picks)], args, n; name = "p$i")
    end
    (; g, plan = M.Plan(g; policy))
end

"""
A resource id as `(parent, span)`, so the emitted set and the oracle are keyed by
what the id *means* rather than by the number the compiler happened to hand out.
Without this the comparison would only work because both sides call the same
interning table, which is the circularity the oracle exists to avoid.
"""
function normalid(g, id::Int)
    ext = Mantle
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

Derived independently of the compiler's own walk, so agreement over a corpus
says the walk is right.
"""
function local_hazards(g, plan)
    be = M.VulkanAPI()
    ext = Mantle

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
