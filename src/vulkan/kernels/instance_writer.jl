# GPU compute kernels and primitives for writing HWTLAS instance records from
# grain state.
#
# write_grain_instances_kernel: one thread per grain, writes 2 instance records
#   (physics @ index 2i-1 with mask=0x02, rendering @ index 2i with mask=0x04).
#   Both records reference different BLASes via the supplied device addresses.
#
# All math is GPU-side; positions/quats live in LavaArrays and never touch CPU.
#
# `quat_to_rot3x3`, `build_4x3`, and `build_4x3_pervec` are kept as inline
# primitives because RayMakie's meshscatter recipe broadcasts over them to
# build per-instance Mat3x4f transforms — the higher-level wrappers / kernel
# variants for that have moved to RayMakie.

"""
    write_grain_instances_kernel(positions, quats, radius,
                                  aabb_blas_addr, tri_blas_addr,
                                  instances)

One thread per grain. Reads `positions[i]` and `quats[i]`, writes two
`VulkanInstanceRecord`s into `instances`:

  - `instances[2i - 1]` references the AABB BLAS, mask = `0x02` (physics).
  - `instances[2i]`     references the triangle BLAS, mask = `0x04` (rendering).

Both records share the same transform: rotation from `quats[i]`, uniform
scale `radius`, translation `positions[i]`. Custom index = `i - 1`.
"""
@kernel cpu=false function write_grain_instances_kernel(
        @Const(positions),
        @Const(quats),
        radius::Float32,
        aabb_blas_addr::UInt64,
        tri_blas_addr::UInt64,
        instances)
    i = @index(Global)
    @inbounds p = positions[i]
    @inbounds q = quats[i]
    rot9 = quat_to_rot3x3(q)
    T = build_4x3(rot9, radius, p)
    cidx = UInt32(i - 1)
    # Pack custom_index_and_mask: low 24 bits = cidx, high 8 bits = mask.
    cim_phys = (cidx & 0x00FFFFFF) | (UInt32(0x02) << 24)
    cim_rend = (cidx & 0x00FFFFFF) | (UInt32(0x04) << 24)
    sof = UInt32(0)  # sbt_offset = 0, flags = 0
    @inbounds instances[2i - 1] = VulkanInstanceRecord(T, cim_phys, sof, aabb_blas_addr)
    @inbounds instances[2i]     = VulkanInstanceRecord(T, cim_rend, sof, tri_blas_addr)
end
