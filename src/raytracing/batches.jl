"""Portable batch state: the native bottom-level structure `blas` its instances name."""
mutable struct InstanceBatch{Tri,Geometry,Records}
    blas::Geometry
    records::Records
    n::Int
    handle::Raycore.TLASHandle
    triangles::Vector{Tri}
    hidden::Bool
    # Instances a transparent-hits query records rather than stops at
    # (`Raycore.transparent_hits`).
    opaque::Bool
end
Base.length(b::InstanceBatch) = b.n

"""
    InstanceBatches{B}()

The batches of type `B` a top-level structure holds, in registration order, and
the handle each was registered under.
"""
mutable struct InstanceBatches{B}
    batches::Vector{B}
    index::Dict{Raycore.TLASHandle,Int}
    # Monotone, never reused: a deleted batch's handle must not name another batch.
    next::UInt32
end

InstanceBatches{B}() where {B} =
    InstanceBatches{B}(B[], Dict{Raycore.TLASHandle,Int}(), UInt32(0))

"""
    register!(batches, mk) -> handle

Take the next handle, build the batch with `mk(handle)` so it can carry its
handle, and record where it landed.
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

The batch `handle` names, or `nothing`, so a stale handle is not an exception.
"""
function batchof(bs::InstanceBatches, h::Raycore.TLASHandle)
    i = get(bs.index, h, nothing)
    return i === nothing ? nothing : bs.batches[i]
end

"""
    delete!(batches, handle) -> Bool

Drop the batch `handle` names and shift every later position down by one, since
`index` maps handles to positions. `false` for a handle that names nothing.
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

The record array of the batch `handle` names, which `Raycore.instance_buffer`
answers. Throws `ArgumentError` for a handle that names none.
"""
function recordsof(bs::InstanceBatches, h::Raycore.TLASHandle)
    b = batchof(bs, h)
    b === nothing && throw(ArgumentError("instance_buffer: no batch for handle $h"))
    return b.records
end

"""
    setbatch!(batches, handle, field, value) -> Bool or nothing

Set the batch `handle` names `hidden` or `opaque` (`field`). `nothing` when it names
none, else whether anything changed (the structure then needs a build). A hidden
batch is built with mask 0, which no ray matches; its records keep their own masks.
"""
function setbatch!(bs::InstanceBatches, h::Raycore.TLASHandle, field::Symbol, value::Bool)
    b = batchof(bs, h)
    b === nothing && return nothing
    getfield(b, field) == value && return false
    setfield!(b, field, value)
    return true
end

"""
    trianglemetadata(Tri, batches) -> (; triangles, hits)

Concatenate each batch's shading triangles in registration order and give each
instance its batch's zero-based triangle offset (`hits`), or `NOTRIANGLES` for a
triangle-free batch. Hidden batches keep their positions.
"""
function trianglemetadata(::Type{Tri}, batches::InstanceBatches) where {Tri}
    triangles = Vector{Tri}(undef, sum(b -> length(b.triangles), batches; init=0))
    hits = Vector{UInt32}(undef, ninstances(batches))
    offset, first = 0, 1
    for batch in batches
        ntri = length(batch.triangles)
        copyto!(triangles, offset + 1, batch.triangles, 1, ntri)
        fill!(view(hits, first:first + batch.n - 1), ntri == 0 ? NOTRIANGLES : UInt32(offset))
        offset += ntri
        first += batch.n
    end
    return (; triangles, hits)
end
