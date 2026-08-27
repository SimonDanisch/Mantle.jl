# GPU compute kernels and primitives for writing TLAS instance records from
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
    quat_to_rot3x3(q::Vec4f) -> NTuple{9, Float32}

Convert a unit quaternion (x, y, z, w) to a row-major 3x3 rotation matrix
returned as a 9-tuple (m00, m01, m02, m10, m11, m12, m20, m21, m22).
"""
@inline function quat_to_rot3x3(q::Vec4f)
    x, y, z, w = q[1], q[2], q[3], q[4]
    xx = x * x; yy = y * y; zz = z * z
    xy = x * y; xz = x * z; yz = y * z
    wx = w * x; wy = w * y; wz = w * z
    return (
        1f0 - 2f0 * (yy + zz),  2f0 * (xy - wz),       2f0 * (xz + wy),
        2f0 * (xy + wz),         1f0 - 2f0 * (xx + zz), 2f0 * (yz - wx),
        2f0 * (xz - wy),         2f0 * (yz + wx),       1f0 - 2f0 * (xx + yy),
    )
end

"""
    build_4x3(rot9::NTuple{9, Float32}, scale::Float32, p::Point3f) -> NTuple{12, Float32}

Combine a 3x3 rotation, uniform scale, and translation into a row-major 3x4
transform suitable for VkAccelerationStructureInstanceKHR.
"""
@inline function build_4x3(rot9::NTuple{9, Float32}, scale::Float32, p::Point3f)
    (rot9[1] * scale, rot9[2] * scale, rot9[3] * scale, p[1],
     rot9[4] * scale, rot9[5] * scale, rot9[6] * scale, p[2],
     rot9[7] * scale, rot9[8] * scale, rot9[9] * scale, p[3])
end

"""
    build_4x3_pervec(rot9::NTuple{9, Float32}, sc::Vec3f, p::Point3f) -> NTuple{12, Float32}

Variant of `build_4x3` that applies a per-axis (Vec3f) scale instead of a
uniform Float32 scale.  Each row of the rotation matrix gets multiplied by
the corresponding axis scale; translation is unchanged.
"""
@inline function build_4x3_pervec(rot9::NTuple{9, Float32}, sc::Vec3f, p::Point3f)
    sx, sy, sz = sc[1], sc[2], sc[3]
    (rot9[1] * sx, rot9[2] * sy, rot9[3] * sz, p[1],
     rot9[4] * sx, rot9[5] * sy, rot9[6] * sz, p[2],
     rot9[7] * sx, rot9[8] * sy, rot9[9] * sz, p[3])
end

"""
    write_grain_instances_kernel(positions, quats, radius,
                                  aabb_blas_addr, tri_blas_addr,
                                  instances)

One thread per grain. Reads `positions[i]` and `quats[i]`, writes two
`LavaInstanceRecord`s into `instances`:

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
    @inbounds instances[2i - 1] = LavaInstanceRecord(T, cim_phys, sof, aabb_blas_addr)
    @inbounds instances[2i]     = LavaInstanceRecord(T, cim_rend, sof, tri_blas_addr)
end
