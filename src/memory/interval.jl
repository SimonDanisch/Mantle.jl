"""
    Span(lower, upper)

A half-open interval `[lower, upper)`, checked at construction. MiniMalloc is
half-open while RPS and DNNKernels' planner are closed ([`from_closed`](@ref));
mixing them silently aliases two live resources.
"""
struct Span
    lower::Int
    upper::Int
    function Span(lower::Integer, upper::Integer)
        lower <= upper || throw(ArgumentError("Span($lower, $upper) is inverted"))
        new(Int(lower), Int(upper))
    end
end

Base.length(s::Span) = s.upper - s.lower
Base.isempty(s::Span) = s.upper == s.lower
Base.show(io::IO, s::Span) = print(io, "[", s.lower, ", ", s.upper, ")")

"""Do two half-open spans share any point?"""
overlaps(a::Span, b::Span) = a.lower < b.upper && b.lower < a.upper

"""Closed-interval input, converted once at the boundary."""
from_closed(lower::Integer, upper::Integer) = Span(lower, upper + 1)
