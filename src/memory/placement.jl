alignup(x::Integer, a::Integer) = ((x + a - 1) ÷ a) * a

"""
Two items conflict if they are ever live at the same instant. With gaps this is
per segment: an item that is absent over a sub-interval does not conflict there.
"""
function conflicts(a::Item, b::Item)
    overlaps(a.live, b.live) || return false
    (isempty(a.gaps) && isempty(b.gaps)) && return true
    for (sa, _) in segments(a), (sb, _) in segments(b)
        overlaps(sa, sb) && return true
    end
    false
end

"""
    LowestFit()

Place items largest first, each at the lowest offset no conflicting item already
occupies.

Measured at 1.3365x the lower bound across MiniMalloc's eleven benchmarks, against
1.3549x for RPS's best fit, so the simpler rule is also the better one here. That
result is why `2026-08-03-rps-design-lessons.md` change 9 is marked measured-wrong.
"""
struct LowestFit end

"""
    BestFit()

RPS's rule (`rps_memory_schedule.hpp:359-460`): among the gaps a candidate fits
in, take the tightest. Kept so the comparison stays reproducible, not because it
is recommended.
"""
struct BestFit end

priority(i::Item) = (-i.size, -length(i.live), i.id)

"""
    place(problem, strategy = LowestFit()) -> Placement

Assign every item an offset such that no two conflicting items overlap in bytes.
Pinned items keep their offset and are placed first, which is also RPS's order
(`rps_memory_schedule.hpp:227`).
"""
function place(p::Problem, strategy = LowestFit())
    order = vcat(sort(findall(pinned, p.items); by = i -> priority(p.items[i])),
                 sort(findall(!pinned, p.items); by = i -> priority(p.items[i])))

    offsets = Dict{String,Int}()
    settled = Int[]
    height = 0

    for idx in order
        item = p.items[idx]
        blocked = Tuple{Int,Int}[]
        for j in settled
            other = p.items[j]
            conflicts(item, other) &&
                push!(blocked, (offsets[other.id], offsets[other.id] + other.size))
        end
        sort!(blocked)

        off = pinned(item) ? item.offset : choose(strategy, blocked, item)
        for (lo, hi) in blocked
            off < hi && lo < off + item.size &&
                error("item $(item.id) at $off collides with [$lo, $hi)")
        end

        offsets[item.id] = off
        push!(settled, idx)
        height = max(height, off + item.size)
    end

    Placement(offsets, height)
end

"""Lowest offset that clears every blocked range below it."""
function choose(::LowestFit, blocked, item::Item)
    off = 0
    for (lo, hi) in blocked
        off + item.size <= lo && break
        off = max(off, alignup(hi, item.alignment))
    end
    off
end

"""Tightest gap that fits, falling back to the top. Smaller waste is better."""
function choose(::BestFit, blocked, item::Item)
    best, waste = 0, typemax(Int)
    cursor = 0
    for (lo, hi) in blocked
        if cursor + item.size <= lo && (lo - cursor) - item.size < waste
            best, waste = cursor, (lo - cursor) - item.size
        end
        cursor = max(cursor, alignup(hi, item.alignment))
    end
    waste == typemax(Int) ? cursor : best
end
