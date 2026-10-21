"""
Incremental TLAS with core-owned geometry registration and batch mutations.
Implementations provide `blas_list`, `instances`, `root_aabb`,
`dirty` and `transforms_dirty`; low-level, already-built HWTLAS types do not
participate in this interface. Native construction, array upload and transform
submission are backend methods.
"""
abstract type BatchedTLAS{Tri} <: HWTLAS{Tri} end

"""Portable batch state; `blas` is a native object or its native table index."""
mutable struct InstanceBatch{Tri,Geometry,Records}
    blas::Geometry
    records::Records
    n::Int
    handle::Raycore.TLASHandle
    triangles::Vector{Tri}
    sbt_offset::UInt32
    hidden::Bool
end
Base.length(b::InstanceBatch) = b.n

"""
    InstanceBatches{B}()

The batches of instances a top-level structure holds, and the handle each was
registered under.

`B` selects a batch representation. This container owns order and handles;
`InstanceBatch` owns portable batch state and its native references.
"""
mutable struct InstanceBatches{B}
    batches::Vector{B}
    index::Dict{Raycore.TLASHandle,Int}
    # Monotone, and never reused: a handle whose batch was deleted must not come
    # back naming a different one, or a caller holding it silently updates
    # someone else's transforms.
    next::UInt32
end

InstanceBatches{B}() where {B} =
    InstanceBatches{B}(B[], Dict{Raycore.TLASHandle,Int}(), UInt32(0))

"""
    register!(batches, mk) -> handle

Take the next handle, build the batch with it, and record where it landed.

`mk(handle)` builds the batch, because a batch usually wants to carry its own
handle and handing it one is cheaper than appending and then patching.
"""
function register!(bs::InstanceBatches{B}, mk) where {B}
    h = Raycore.TLASHandle(bs.next)
    bs.next += UInt32(1)
    push!(bs.batches, mk(h)::B)
    bs.index[h] = lastindex(bs.batches)
    return h
end

"""
    batchof(batches, handle) -> batch or nothing

The batch `handle` names. `nothing` when it names none, which a caller holding
a stale handle should be able to ask without an exception.
"""
function batchof(bs::InstanceBatches, h::Raycore.TLASHandle)
    i = get(bs.index, h, nothing)
    return i === nothing ? nothing : bs.batches[i]
end

"""
    delete!(batches, handle) -> Bool

Drop the batch `handle` names, and move every index after it down by one.

The reindex is the half that was written twice and is easy to get wrong: the
map holds positions, not identities, so removing from the middle invalidates
every entry behind it. Returns `false` for a handle that names nothing.
"""
function Base.delete!(bs::InstanceBatches, h::Raycore.TLASHandle)
    i = get(bs.index, h, nothing)
    i === nothing && return false
    deleteat!(bs.batches, i)
    delete!(bs.index, h)
    for (k, j) in bs.index
        j > i && (bs.index[k] = j - 1)
    end
    return true
end

Base.length(bs::InstanceBatches)  = length(bs.batches)
Base.isempty(bs::InstanceBatches) = isempty(bs.batches)
Base.eltype(::InstanceBatches{B}) where {B} = B
Base.iterate(bs::InstanceBatches, s...) = iterate(bs.batches, s...)
Base.getindex(bs::InstanceBatches, i::Integer) = bs.batches[i]

"""The instances across every batch, which is what a build consumes."""
ninstances(bs::InstanceBatches) = sum(length, bs.batches; init = 0)

"""
    recordsof(batches, handle) -> device array of `Raycore.InstanceRecord`

The record array of the batch `handle` names: what `Raycore.instance_buffer`
answers on every backend. An `ArgumentError` for a handle that names none,
because a kernel about to write records into nothing is a caller's mistake.
"""
function recordsof(bs::InstanceBatches, h::Raycore.TLASHandle)
    b = batchof(bs, h)
    b === nothing && throw(ArgumentError("instance_buffer: no batch for handle $h"))
    return b.records
end

"""
    sethidden!(batches, handle, hidden) -> Bool or nothing

Mark the batch `handle` names hidden or shown. `nothing` when it names none,
else whether that changed anything, which is when the structure needs a build.
A hidden batch is built with mask 0, which no ray's mask matches
(`Raycore.set_visible!`); its records keep their own masks for when it is shown.
"""
function sethidden!(bs::InstanceBatches, h::Raycore.TLASHandle, hidden::Bool)
    b = batchof(bs, h)
    b === nothing && return nothing
    b.hidden == hidden && return false
    b.hidden = hidden
    return true
end

"""
    trianglemetadata(Tri, batches, [Val(false)])

Concatenate each batch's shading triangles once, in registration order, and
give each instance its batch's zero-based triangle offset. Triangle-free
batches receive `NOTRIANGLES` in the hit table, including those followed by a
nonempty batch. Hidden batches keep their positions.

`Val(true)` also requests the raw batch/instance offsets used by HardwareAccel;
these bookkeeping arrays are omitted when only the hit table is needed.
"""
function trianglemetadata(::Type{Tri}, batches::InstanceBatches,
                          ::Val{Bookkeeping} = Val(false)) where {Tri,Bookkeeping}
    triangles = Vector{Tri}(undef, sum(b -> length(b.triangles), batches; init=0))
    hits = Vector{UInt32}(undef, ninstances(batches))
    raw = Bookkeeping ? similar(hits) : nothing
    starts = Bookkeeping ? Vector{UInt32}(undef, length(batches)) : nothing
    offset, first = 0, 1
    for (i, batch) in enumerate(batches)
        ntri = length(batch.triangles)
        copyto!(triangles, offset + 1, batch.triangles, 1, ntri)
        hit = ntri == 0 ? NOTRIANGLES : UInt32(offset)
        fill!(view(hits, first:first + batch.n - 1), hit)
        if Bookkeeping
            starts[i] = UInt32(offset)
            fill!(view(raw, first:first + batch.n - 1), UInt32(offset))
        end
        offset += ntri
        first += batch.n
    end
    return (; triangles, hits, raw, starts)
end
