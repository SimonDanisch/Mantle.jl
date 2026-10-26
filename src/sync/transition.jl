# Hazard tracking: the transitions a sequence of usages on one resource needs, in
# usage vocabulary only. Backends lower them to stages, access masks and layouts.

"""
    Transition(resource, from, waits, to)

`from` is the state the resource leaves (a backend's old layout). `waits` is every
access the new one must be ordered after: all outstanding reads, for a write. A
backend ORs the stage and access masks of `waits` and takes the layout from `from`.
"""
struct Transition
    resource::Int
    from::Type
    waits::Vector{Type}
    to::Type
end

"""
    barrierhazards(transitions) -> Vector{Tuple{Vector{Type},Type}}

The distinct `(waits, to)` hazards a pass has to order, with the resource dropped.
Cache maintenance is per cache, not per address, so a buffer barrier does what a
global barrier with the same masks does, and a recording that names no buffer
stays valid when buffers move in the pool. Tuples are kept distinct, not unioned,
so each barrier invalidates only the caches its masks name. Image barriers keep
naming their image because they carry a layout transition.
"""
function barrierhazards(ts::Vector{Transition})
    out = Tuple{Vector{Type},Type}[]
    for t in ts
        # Wait order does not change a hazard. Comparing the small sets avoids
        # formatting and sorting type names for every transition in a model.
        any(h -> h[2] === t.to && issetequal(h[1], t.waits), out) ||
            push!(out, (t.waits, t.to))
    end
    out
end

"""
Per-resource tracking state. `current` is the usage the resource is in (a
backend's old layout); a read that needs a transition, such as a copy after a
sample, changes it. `pending` is every read since the last write, all of which a
write must wait on.
"""
mutable struct ResourceState{K<:ResourceKind}
    current::Union{Nothing,Type}
    pending::Vector{Type}
end

ResourceState(k::ResourceKind, initial::Type) =
    ResourceState{typeof(k)}(initial, Type[])
ResourceState(k::ResourceKind) = ResourceState{typeof(k)}(nothing, Type[])

"""
    needs_transition(backend, kind, before, after) -> Bool

The portable rule: read after read and two `Unordered` accesses need nothing;
anything else involving a write does, identical storage writes included, since
they race. Backends override it where their API partitions states differently
(copy, clear, depth-stencil).
"""
function needs_transition(::Backend, ::ResourceKind, before::Type, after::Type)
    both_unordered = unordered(before) && unordered(after)
    b, a = inner(before), inner(after)
    if !writes(b) && !writes(a)
        return false
    elseif both_unordered
        return false
    else
        return b !== a || writes(b)
    end
end

"""
    transition!(out, backend, resource, state, next)

Append the transitions `next` requires and advance `state`. Read after read
appends nothing but keeps the reader, so a later write waits on all of them.
"""
function transition!(out, be::Backend, resource::Integer,
                     state::ResourceState{K}, next::Type) where {K}
    kind = K()
    from = state.current
    if from !== nothing && needs_transition(be, kind, from, next)
        waits = writes(next) ? unique(vcat(from, state.pending)) : Type[from]
        push!(out, Transition(Int(resource), from, waits, next))
        state.current = next
    elseif from === nothing
        state.current = next
    end
    writes(next) ? empty!(state.pending) : push!(state.pending, next)
    out
end

"""
    transitions(backend, kind, usages; initial)

Walk one resource's whole usage sequence. `kind` is given because a usage does
not determine it: a copy changes an image's layout but not a buffer's.
"""
function transitions(be::Backend, kind::ResourceKind, usages::AbstractVector{<:Type};
                     initial::Union{Nothing,Type} = nothing)
    state = initial === nothing ? ResourceState(kind) : ResourceState(kind, initial)
    out = Transition[]
    for u in usages
        transition!(out, be, 1, state, u)
    end
    out
end
