# Portable instance records: one `Raycore.InstanceRecord` (transform, custom index,
# mask) per instance, a device array per batch. Each backend converts them to its
# driver's records at build and refit, adding the geometry reference.

"""
    instancerecords(transforms, ids, mask) -> Vector{Raycore.InstanceRecord}

The records `push!` uploads for a batch of `transforms` (`Mat4f`s), with custom
indices `ids` (`nothing` for all zero) and one `mask`. Throws `ArgumentError`
when `ids` does not have one entry per transform, before any geometry is built.
"""
function instancerecords(transforms::AbstractVector, ids::Union{Nothing,AbstractVector{<:Integer}},
                         mask::UInt8)
    ids === nothing || length(ids) == length(transforms) || throw(ArgumentError(
        "instance_ids length $(length(ids)) != transforms length $(length(transforms))"))
    return [Raycore.InstanceRecord(Raycore.mat4_to_mat3x4(m), ids === nothing ? UInt32(0) : UInt32(ids[i]), mask)
            for (i, m) in enumerate(transforms)]
end

"""
    update_instance_records_kernel!(records, transforms)

Write `transforms[i]` into `records[i]`, keeping each record's own custom index
and mask so per-instance ids survive a transform update. A KernelInterface
kernel, one work item per transform; `records` may be longer than `transforms`.
"""
function update_instance_records_kernel!(records, transforms)
    i = KI.get_global_id().x
    i <= length(transforms) || return nothing   # the launch is whole workgroups
    @inbounds old = records[i]
    @inbounds records[i] = Raycore.InstanceRecord(transforms[i], old.id, old.mask)
    return nothing
end

"""
    instancerecord(::Type{R}, transform, id, mask, geometry, opaque) -> R

The driver's instance record `R` of one instance: its `transform` (a `Mat3x4f`), its 24-bit
custom index `id`, its 8-bit `mask`, its `geometry` ([`geometryvalue`](@ref)), opaque or not
whatever that geometry was built as. Each backend packs it for its record type.
"""
function instancerecord end

"""
    instances_kernel!(native, shading, records, offset, n, geometry, keep, opaque)

One batch's instances into the two arrays a build reads: `shading[offset + i] =
records[i]`, which hits read transforms from, and `native[offset + i]`, the
driver record from [`instancerecord`](@ref), with the custom index cut to the 24 bits every
driver keeps and the mask ANDed with `keep`: 0 for a hidden batch, else `0xff`.
"""
function instances_kernel!(native, shading, records, offset::Int32, n::Int32,
                           geometry, keep::UInt32, opaque::Bool)
    i = KI.get_global_id(Int32).x
    i <= n || return nothing   # the launch is whole workgroups
    @inbounds r = records[i]
    j = offset + i
    @inbounds shading[j] = r
    @inbounds native[j] = instancerecord(eltype(native), r.transform, r.id & 0x00ffffff, r.mask & keep,
                                         geometry, opaque)
    return nothing
end
