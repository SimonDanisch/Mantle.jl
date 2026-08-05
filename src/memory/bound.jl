"""
    maxload(problem) -> Int

The largest total size live at any one instant: the max-weight clique of the
interval graph, by a sweep over `2n` events.

This is the lower bound on any placement, and it is the number `graph-api-design.md`
never computes. Without it "we fragmented 3%" is not a statement anyone can act on.

MiniMalloc does not compute or report a bound of any kind, so this is ours.
Buchsbaum et al. bound the gap to optimal in terms of `hmax / maxload`, which is
why [`fragmentation`](@ref) reports both.
"""
function maxload(p::Problem)
    events = Tuple{Int,Int}[]
    for item in p.items, (span, height) in segments(item)
        isempty(span) && continue
        push!(events, (span.lower, height), (span.upper, -height))
    end
    isempty(events) && return 0
    sort!(events)
    best = cur = 0
    for (_, delta) in events
        cur += delta
        best = max(best, cur)
    end
    best
end

"""The largest single item, which governs how far optimal can sit above `maxload`."""
hmax(p::Problem) = isempty(p.items) ? 0 : maximum(i -> i.size, p.items)

"""
    fragmentation(problem, placement) -> (ratio, slack)

`ratio` is achieved height over the lower bound; 1.0 is provably optimal.
`slack` is `hmax / maxload`, the quantity that governs how large the gap between
the true optimum and the bound can be, so a bad `ratio` with a large `slack` is
not necessarily a bad placement.
"""
function fragmentation(p::Problem, pl::Placement)
    L = maxload(p)
    L == 0 ? (1.0, 0.0) : (pl.height / L, hmax(p) / L)
end
