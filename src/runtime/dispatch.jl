# Dispatch records and the ndrange vocabulary.
#
# This lives in core because it is the same on every backend. `Dispatch` and
# `dispatch!` were byte-identical copies in `MantleLavaExt` and `MantleHostExt`;
# a third backend would have made a third copy, and each copy is a place for the
# usage registration below to go missing on one backend only.

"""
    Dispatch(kernel, args, ndrange, group)

One kernel launch inside a pass. Backends read the fields and record; nothing
here is backend-specific, so the record is shared.
"""
struct Dispatch
    kernel::Any
    args::Tuple
    ndrange::Any
    group::Any
end

"""
    DeviceRange(count; max = nothing)

An ndrange whose size the device computes: `count` is a device-resident ELEMENT
count, not a workgroup count, and not a `VkDispatchIndirectCommand`.

This is the whole point of putting it here rather than exposing the backend's
indirect-dispatch buffer. Every wavefront stage in a path tracer dispatches over
"however many rays survived", which only the GPU knows, and the shape the
graphics APIs offer is a buffer of workgroup triples — already divided by the
workgroup size the caller has not chosen yet. Callers therefore end up writing a
tiny "prepare indirect" kernel per stage to compute `cld(count, groupsize)`, and
then a barrier between that kernel and the dispatch that reads it. Both are
derivable, and both are exactly the kind of hand-placed ordering that has
produced real races here.

So the contract is: hand over the count, in device memory. The backend converts
it to whatever its API wants — on Vulkan a small kernel filling a
`VkDispatchIndirectCommand`, on Metal the threadgroup counts for
`dispatchThreadgroups(indirectBuffer:)` — and the graph orders whatever wrote
`count` before this dispatch, because `count` is registered as an `Indirect`
read.

`max` is an optional upper bound on the count, for backends that must compile
against a static ndrange; `nothing` means the backend picks its own ceiling.

    n = GPURef(g, Int32)                       # written by an earlier pass
    dispatch!(p, compact!, (queue, n), DeviceRange(n))

# The tail

**The kernel must bound itself.** A dispatch covers whole workgroups, so a count
of 777 with a group of 64 launches 832 invocations, and the 55 past the end are
real: the kernel was compiled against `max` precisely so its bounds check would
not clamp to a size known at compile time, which means nothing else clamps
either. Take the count as an argument and return early:

    @kernel function compact!(dst, @Const(src), n)
        i = @index(Global)
        @inbounds if i <= n[1]
            ...
        end
    end

This is not a backend quirk to be papered over. A count that lives on the device
cannot round to a workgroup boundary on the host, and padding the dispatch down
would drop elements. Every API that dispatches indirectly has this tail; what
Mantle removes is the prepare kernel and the barrier, not the guard.
"""
struct DeviceRange{C}
    count::C
    max::Union{Nothing,Int}
end
DeviceRange(count; max = nothing) = DeviceRange(count, max)

"""The resource a `DeviceRange` reads its count from, or `nothing` for a host-side
ndrange. Backends use this to decide which recording path a dispatch takes."""
countresource(r::DeviceRange) = r.count
countresource(::Any) = nothing

"""
    refuserefs(args)

Throw if a `Ref` is anywhere in a dispatch's, draw's or trace's arguments —
nested in a tuple or a struct as much as at the top, because a `Ref` inside a
camera struct is the case a renderer actually had.

A `Ref` used to mean "read this again every run", and honouring that cost the
argument ring: 602 host stores per run on Hikari's fused sample to move 268
bytes, into memory an in-flight submission could still be reading. Nothing
rewrites argument memory after `record!` now, so there is nothing left for a
`Ref` to mean: a value that changes between runs is a [`GPURef`](@ref), stored
with `ref[] = x`, and a value that does not is passed as itself. Refusing it at
the declaration is what keeps one from silently freezing at whatever it held
when the plan recorded.

Walks the argument tree the way [`holdleaves!`](@ref) does — per
concrete type, unrolled at compile time — through tuples, NamedTuples and
immutable structs, which is the value tree a packer flattens. It stops at a
resource, an array, and any other MUTABLE struct: those are handles that own
their insides (an acceleration structure holds the driver's objects, which
reference each other in cycles), and the one mutable cell this refuses is the
`Ref` itself.
"""
refuserefs(::Base.RefValue) = throw(ArgumentError(
    "a Ref is not a dispatch argument. It would be read once, at record!, and never " *
    "again, which is not what a Ref is for: a value that changes between runs is a " *
    "GPURef (`r = GPURef(dev, x)`, then `r[] = x′` before a run), and a value that " *
    "does not is passed as itself."))
refuserefs(::Resource) = nothing
refuserefs(::AbstractArray) = nothing
@generated function refuserefs(x::T) where {T}
    isbitstype(T) && return :(nothing)
    (T <: Tuple || T <: NamedTuple) && return Expr(
        :block, Expr[:(refuserefs(x[$i])) for i in 1:fieldcount(T)]..., :(nothing))
    (T <: Type || T <: Symbol || T <: AbstractString || T <: Module ||
     T === Nothing || T <: Ptr || ismutabletype(T)) && return :(nothing)
    n = fieldcount(T)
    n == 0 && return :(nothing)
    Expr(:block, Expr[:(refuserefs(getfield(x, $i))) for i in 1:n]..., :(nothing))
end

"""
    passof(handle) -> pass
    graphof(handle) -> graph

A backend's pass handle, opened up so core can push dispatches into it. Both
backends name these fields, so the defaults hold and a backend only overrides if
it names them otherwise.
"""
passof(h) = h.pass
graphof(h) = h.graph

"""
    resourceid(graph, r) -> Int

The id `graph` knows `r` by, assigning one if it has not seen it. Every graph
carries an [`IdTable`](@ref); a backend whose graph does not spell it `ids`
overrides this.
"""
# `resourceid(::Graph, r)` is in `graph/build.jl`: it names the concrete
# `Graph`, which is defined after this file.

"""
    touch!(graph, x) -> x

Register `x` with `graph`. The default assigns an id and no more, which is all a
persistent resource needs. A backend specialises this for its transients, whose
liveness interval has to widen to include the pass doing the touching.
"""
touch!(g, x) = (resourceid(g, x); x)

"""
    dispatch!(pass, kernel, args, ndrange; group = nothing)

Launch a kernel as part of a pass, so what it reads and writes is what orders it
against everything else.

`group` is the workgroup size, and it is worth giving one whenever `ndrange` is
two-dimensional. The default partitions an ndrange along its first axis, which
for anything image-shaped means a workgroup is one long row: a pass that reads a
row-major g-buffer and writes a column-major target then has one of the two
uncoalesced, and it costs about nine times the square arrangement — 1.06 ms
against 0.12 ms at 1280x800 on the machine this was measured on.

    dispatch!(p, shade!, args, (w, h); group = (16, 16))

`nothing` means the backend picks, which is right for a one-dimensional ndrange
and is what everything that does not say otherwise gets.

`ndrange` may also be a [`DeviceRange`](@ref), for a count that only exists on
the device. The dispatch is then ordered after whatever wrote that count, and
the caller never sees a workgroup division or a barrier.
"""
function dispatch!(p, kernel, args, ndrange; group = nothing)
    refuserefs(args)
    n = countresource(ndrange)
    n === nothing || indirectcount!(p, n)
    push!(dispatches(passof(p)), Dispatch(kernel, args, ndrange, group))
end

"""
    indirectcount!(pass, count)

Record that `pass` reads `count` to size a dispatch. Separate from `use` because
`Indirect` is its own usage: the stage and access it implies are not a storage
read, and a backend that got it wrong would emit a barrier for the wrong stage.
"""
function indirectcount!(p, n)
    push!(passof(p).usages, resourceid(graphof(p), n) => Indirect)
    touch!(graphof(p), n)
    return n
end
