# Hazard tracking: which transitions a sequence of usages on one resource needs.
#
# The output is in terms of the usage vocabulary. Nothing here knows what a stage
# or a layout is; the backend turns a transition into whatever it needs, which for
# WebGPU is nothing at all.

"""
    Transition(resource, from, waits, to)

`from` is the state the resource is leaving, which is what a backend turns into
an old layout. `waits` is everything the new access must be ordered against,
which for a write after several concurrent reads is all of them. The two differ
whenever more than one read is outstanding, and a backend ORs the stages and
access masks of `waits` while taking the layout from `from` alone.
"""
struct Transition
    resource::Int
    from::Type
    waits::Vector{Type}
    to::Type
end

"""
    barrierhazards(transitions) -> Vector{Tuple{Vector{Type},Type}}

The distinct `(waits, to)` hazards a pass has to order, with the RESOURCE
dropped.

A resource decides that a transition exists — which is the whole of the
dependency analysis, and it stays per resource and exact. It does not belong in
the barrier that results. A `VkBufferMemoryBarrier2` names a buffer and a byte
range, but the range is not something hardware can act on: cache invalidation is
per cache, not per address, so the driver flushes exactly what a global barrier
would. What the barrier really carries is the stage and access masks, and those
say which caches to make available and visible.

So the resource is used and then forgotten, which is what lets a recording
outlive the placement it was recorded against: nothing in the command stream
names a buffer, so a buffer moving in the pool cannot invalidate it. The code
already reached this conclusion for renameable resources — "a handle cannot be
baked for something that moves" — and baking makes every resource one of those.

DISTINCT tuples, never unioned into one. Union would make shader writes visible
to indirect reads and vice versa, invalidating the command processor's fetch
path and the vector cache when each needed only one; an acceleration-structure
read would drag in caches it never touches. The precision that matters lives in
the masks, and it is kept here exactly.

Images are not here. `VkImageMemoryBarrier2` carries a layout transition, which
is per-image state with no global equivalent, so image barriers keep naming
their image — and an image that moves is what `refit!` already detects.
"""
function barrierhazards(ts::Vector{Transition})
    out = Tuple{Vector{Type},Type}[]
    for t in ts
        key = (sort(t.waits; by = string), t.to)
        any(==(key), out) || push!(out, key)
    end
    out
end

"""
Two things have to be tracked separately, and conflating them emits transitions
from the wrong state.

`current` is the usage the resource is *in*: what a backend turns into an old
layout. Reads change it, because sampling an image and copying from it want
different layouts, so a copy after a sample transitions from the sample and not
from whatever wrote it.

`pending` is every read since the last write, which is what a write must wait
on. A read is not in `current` for that purpose because several reads can be
outstanding at once.
"""
mutable struct ResourceState{K<:ResourceKind}
    current::Union{Nothing,Type}
    pending::Vector{Type}
end

ResourceState(k::ResourceKind, initial::Type) =
    ResourceState{typeof(k)}(initial, Type[])
ResourceState(k::ResourceKind) = ResourceState{typeof(k)}(nothing, Type[])

"""
    needs_transition(backend, before, after) -> Bool

The portable rule. Read after read needs nothing. Anything involving a write
does, and identical storage writes still do, because two dispatches writing the
same bytes race even though the declared access did not change.

Backends override this: whether copy, clear and depth-stencil accesses need a
transition depends on how the API partitions its states, and RPS delegates
exactly those to its runtime device (`rps_access_dag_build.hpp:477-487`).
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

Append the transitions `next` requires and advance `state`.

Read after read appends nothing but keeps the reader, so a later write depends on
all of them. Note this accumulates where RPS *merges* consecutive reads into one
combined access up front (`rps_access_dag_build.hpp:491-492`), which lets it pick
one layout covering both and emit nothing between them. Merging needs to see both
reads before recording either, so it belongs to the DAG phase, not here.
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

Walk one resource's whole usage sequence. The unit the barrier matrix tests.

`kind` is given rather than derived, because a usage does not determine it: copy
source and destination apply to buffers and images alike, and the answer differs
(an image copy changes layout, a buffer copy does not).
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
