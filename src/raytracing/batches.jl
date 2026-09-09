# The instances a top-level acceleration structure holds.
#
# This was written twice, once per backend, as `_register_batch!` plus three
# fields — `instance_batches`, `handle_to_batch_idx`, `next_handle_id`. The two
# had different signatures, so a scan for shared names called them private
# helpers rather than duplication; reading both showed the same five steps:
# allocate a handle, append a batch, record its index, look one up, drop one and
# reindex what shifted.
#
# None of that is a driver's. What a batch CONTAINS is — Vulkan's holds a
# device-resident instance buffer and a BLAS, Metal's a transform list and an
# index into the BLAS list — so the container is parameterised on it and knows
# nothing else about it.

"""
    InstanceBatches{B}()

The batches of instances a top-level structure holds, and the handle each was
registered under.

`B` is the backend's batch type. This owns the ORDER and the handles; the
backend owns what is in a batch.
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
