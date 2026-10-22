"""
    maxload(problem) -> Int

The largest total size live at any instant (the max-weight clique of the
interval graph, by a sweep over `2n` events): a lower bound on any placement's
height.
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

"""Bytes in B, KB, MB or GB, so a test-sized overflow does not print as "0.0 MB"."""
function humanbytes(x::Integer)
    x < 2^10 && return "$x B"
    x < 2^20 && return "$(round(x / 2^10; digits = 2)) KB"
    x < 2^30 && return "$(round(x / 2^20; digits = 2)) MB"
    "$(round(x / 2^30; digits = 2)) GB"
end

"""
    checkcapacity(problem, placement, what = "arena") -> placement

Throw unless the placement fits the problem's capacity. Separate from
[`place`](@ref), which never checks: a device arena must refuse an overflow,
while a benchmark scores placements that exceed MiniMalloc's capacities. The
error gives the request, the capacity, the lower bound and the largest items.
"""
function checkcapacity(p::Problem, pl::Placement, what::AbstractString = "arena")
    pl.height <= p.capacity && return pl
    biggest = sort(p.items; by = i -> -i.size)
    listed = join(("    $(i.id)  $(humanbytes(i.size))"
                   for i in Iterators.take(biggest, 5)), "\n")
    # The lower bound against the capacity, not the achieved height: a bound
    # above capacity means no placement fits.
    bound = maxload(p)
    verdict = bound > p.capacity ?
        "no placement fits — the items themselves are too large" :
        "a better placement would fit; this one wastes " *
        "$(round(100 * (pl.height - bound) / bound; digits = 1))%"
    throw(ArgumentError("""
        $what does not fit: needs $(humanbytes(pl.height)), \
        $(humanbytes(p.capacity)) available.
        Lower bound for these items is $(humanbytes(bound)), so $verdict.
        $(length(p.items)) items totalling $(humanbytes(sum(i -> i.size, p.items))). Largest items:
        $listed"""))
end

"""
    fragmentation(problem, placement) -> (ratio, slack)

`ratio` is height over [`maxload`](@ref); 1.0 is optimal. `slack` is
`hmax / maxload`, which bounds how far the optimum can sit above `maxload`
(Buchsbaum et al.), so a high `ratio` with a high `slack` need not be bad.
"""
function fragmentation(p::Problem, pl::Placement)
    L = maxload(p)
    L == 0 ? (1.0, 0.0) : (pl.height / L, hmax(p) / L)
end
