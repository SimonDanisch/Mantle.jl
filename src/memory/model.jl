"""
    OffsetWindow(lower, upper)

A half-open sub-range of an item's own extent, relative to its offset.
"""
struct OffsetWindow
    lower::Int
    upper::Int
end

Base.length(w::OffsetWindow) = w.upper - w.lower

"""
    Gap(during, window)

A sub-interval of an item's lifespan over which it needs less than its full
size: nothing when `window === nothing`, else only that sub-range of its extent.
For needs that shrink (RT build scratch, a DNN workspace, a mip chain).
"""
struct Gap
    during::Span
    window::Union{OffsetWindow,Nothing}
end

"""
    Item(id, live, size; alignment = 1, gaps = Gap[], offset = nothing)

One resource to place: liveness interval `live`, `size` bytes, and an offset to
choose. `alignment` is per item because Vulkan reports it per resource; `offset`
pins an item allocated elsewhere.
"""
struct Item
    id::String
    live::Span
    size::Int
    alignment::Int
    gaps::Vector{Gap}
    offset::Union{Int,Nothing}
end

Item(id, live, size; alignment = 1, gaps = Gap[], offset = nothing) =
    Item(String(id), live, Int(size), Int(alignment), gaps, offset)

pinned(i::Item) = i.offset !== nothing

"""The (span, height) pairs an item actually occupies, gaps applied."""
function segments(i::Item)
    isempty(i.gaps) && return [(i.live, i.size)]
    out = Tuple{Span,Int}[]
    t = i.live.lower
    for g in sort(i.gaps; by = g -> g.during.lower)
        t < g.during.lower && push!(out, (Span(t, g.during.lower), i.size))
        g.window === nothing || push!(out, (g.during, length(g.window)))
        t = g.during.upper
    end
    t < i.live.upper && push!(out, (Span(t, i.live.upper), i.size))
    out
end

struct Problem
    items::Vector{Item}
    capacity::Int
end

struct Placement
    offsets::Dict{String,Int}
    height::Int
end
