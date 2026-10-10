# The instances of a hardware top-level structure, as callers and kernels see them.
#
# One `Raycore.InstanceRecord` per instance — transform, custom index, mask — in
# a device array per batch. A backend turns those into its driver's records
# (`VkAccelerationStructureInstanceKHR`, `MTLAccelerationStructureInstanceDescriptor`)
# when it builds or refits, adding what only it knows: the geometry the batch
# instances, by address or by index. So everything that WRITES the portable
# records is here, once, and every backend reads them the same way.

"""
    instancerecords(transforms, ids, mask) -> Vector{Raycore.InstanceRecord}

The records `push!` uploads for a batch of `transforms` (`Mat4f`s), with the
custom indices `ids` (`nothing` for all zero) and one `mask` for all.

Throws before anything is built when `ids` does not have one entry per
transform, so a refused push leaves no geometry behind that no instance names.
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

Write `transforms[i]` into `records[i]`, keeping each record's custom index and
mask. A plain KernelInterface kernel, one work item per transform; `records` may
be longer than `transforms`.

The id and mask are READ BACK from the record rather than passed in, and that is
the fix for a silent wrong answer this had on Vulkan: they were scalar arguments
taken from the batch, so one transform update flattened every instance in it to
one custom index and one mask, and the per-instance ids of a
`push!(tlas, mesh, transforms; instance_ids)` could not survive a refit even in
principle. Each work item reads and writes only element `i`.
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

One work item per grain `i`: an instance record for its physics geometry into
`phys[i]` (mask `0x02`) and one for its rendering geometry into `rend[i]` (mask
`0x04`). Both have the same transform — the rotation of the unit quaternion
`quats[i]` (x, y, z, w), uniform scale `radius`, translation `positions[i]` —
and custom index `i - 1`.

Each array is the instance buffer of its own batch, pushed with the geometry its
records instance: the box structure physics queries trace, and the triangles a
renderer does. A ray with mask `0x02` then sees only the first and one with
`0x04` only the second.

A plain KernelInterface kernel: launch it with `KI.Kernel(backend,
write_grain_instances_kernel)(…; ndrange = length(positions))`.
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
