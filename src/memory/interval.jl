"""
    Span(lower, upper)

A half-open interval `[lower, upper)`.

Three projects we read use two conventions: MiniMalloc is half-open
(`minimalloc.h:47-51`), while RPS (`rps_memory_schedule.hpp:418`) and our own
`DNNKernels/src/plan.jl:256` are closed. Getting this wrong aliases two live
resources onto the same bytes, which is silent, so the convention is carried by
the type and asserted at construction rather than left to a comment.
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
