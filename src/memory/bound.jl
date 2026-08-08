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

"""Bytes at a scale a reader can act on. Not always MB: an arena that overflows
at a few hundred bytes — which is every test of this — reads as "needs 0.0 MB,
0.0 MB available", and a message whose numbers are all zero is worse than none."""
function humanbytes(x::Integer)
    x < 2^10 && return "$x B"
    x < 2^20 && return "$(round(x / 2^10; digits = 2)) KB"
    x < 2^30 && return "$(round(x / 2^20; digits = 2)) MB"
    "$(round(x / 2^30; digits = 2)) GB"
end

"""
    checkcapacity(problem, placement, what) -> placement

Throw unless the placement fits the problem's bound.

Separate from [`place`](@ref), and opt-in, because the two callers want opposite
things from the same overflow. A backend placing an arena on a real device has to
refuse: memory it cannot have is not a placement, and finding out at allocation
time attributes the failure to whatever allocated next. A benchmark placing
MiniMalloc's instances has to *continue*: those capacities are the target a
solver is scored against, and a greedy rule that lands 1.34x above the lower
bound exceeds them routinely — that ratio is the measurement, not an error.

So `place` never throws and this does, and the caller says which it is.

The message carries the three numbers needed to act on it: what was asked, what
was available, and the largest items — because "over budget by 40 MB" is
answerable by dropping one buffer and unanswerable without knowing which.
"""
function checkcapacity(p::Problem, pl::Placement, what::AbstractString = "arena")
    pl.height <= p.capacity && return pl
    biggest = sort(p.items; by = i -> -i.size)
    listed = join(("    $(i.id)  $(humanbytes(i.size))"
                   for i in Iterators.take(biggest, 5)), "\n")
    # Whether a better placer would help is `maxload` against the *capacity*, not
    # against the height we achieved: a lower bound that already exceeds the
    # budget means no placement of these items fits, however good. Comparing
    # height to maxload instead answers "was this placement tight", which is a
    # different question and the wrong one to put in front of someone whose
    # compile just failed.
    bound = maxload(p)
    verdict = bound > p.capacity ?
        "no placement fits — the items themselves are too large" :
        "a better placement would fit; this one wastes " *
        "$(round(100 * (pl.height - bound) / bound; digits = 1))%"
    throw(ArgumentError("""
        $what does not fit: needs $(humanbytes(pl.height)), \
        $(humanbytes(p.capacity)) available.
        Lower bound for these items is $(humanbytes(bound)), so $verdict.
        Largest items:
        $listed"""))
end

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
