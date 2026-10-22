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
    return [Raycore.InstanceRecord(mat4_to_vk_transform(m), ids === nothing ? UInt32(0) : UInt32(ids[i]), mask)
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
    write_grain_instances_kernel(positions, quats, radius, phys, rend)

One work item per grain `i`: an instance record into `phys[i]` (mask `0x02`) and
one into `rend[i]` (mask `0x04`), each the instance buffer of its own batch (physics
boxes, render triangles). Both get custom index `i - 1` and the transform of unit
quaternion `quats[i]` (x, y, z, w), uniform scale `radius` and translation
`positions[i]`. A KernelInterface kernel, `ndrange = length(positions)`.
"""
function write_grain_instances_kernel(positions, quats, radius::Float32, phys, rend)
    i = KI.get_global_id().x
    i <= length(positions) || return nothing   # the launch is whole workgroups
    @inbounds p = positions[i]
    @inbounds q = quats[i]
    T = Mat3x4f(build_4x3(quat_to_rot3x3(q), radius, p))
    id = (i - 1) % UInt32
    @inbounds phys[i] = Raycore.InstanceRecord(T, id, UInt32(0x02))
    @inbounds rend[i] = Raycore.InstanceRecord(T, id, UInt32(0x04))
    return nothing
end
