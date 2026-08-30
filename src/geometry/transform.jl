# The 3×4 affine transform every backend's instance descriptor carries.
#
# This was in `vulkan/raytracing/instance_record.jl`, beside the record whose
# ABI it serves. The record stays there — `VulkanInstanceRecord` is a byte-exact
# mirror of `VkAccelerationStructureInstanceKHR` and Metal's
# `MTLAccelerationStructureInstanceDescriptor` lays its fields out differently,
# so that struct is genuinely the Vulkan backend's.
#
# The matrix is not. A 3×4 row-major float affine is what Metal's instance
# descriptor holds too, and what every kernel reading a transform already wants,
# so keeping the alias behind the Vulkan backend would mean a second backend
# either re-declaring it or depending on the first one for a `StaticArrays`
# typedef.

using StaticArrays: SMatrix

"""
    Mat3x4f

A 3×4 row-major affine transform, stored as `SMatrix{4, 3, Float32, 12}`.

The type parameters look transposed because `SMatrix` storage is column-major
and a column-major 4×3 has byte-for-byte the same layout as a row-major 3×4 —
which is `VkTransformMatrixKHR` exactly, so the Vulkan backend hands these bytes
to the driver with no reinterpret and no per-store conversion.

**That is Vulkan's layout, not a universal one.** Metal's
`MTLPackedFloat4x3` is also 48 bytes and also twelve floats, and it is the
TRANSPOSE: four columns of three, where this is three rows of four. The identity
alone tells them apart — `[1,0,0,0, 0,1,0,0, 0,0,1,0]` here against
`[1,0,0, 0,1,0, 0,0,1, 0,0,0]` there. Reinterpreting one as the other produces a
sheared transform and no error, so the Metal backend converts; see
`metal/raytracing.jl`.

What is shared is the CONCEPT and the size: an affine 3×4 of `Float32`, which is
what every instance descriptor carries. The element order is the backend's.

Pure linear indexing (`T[1]..T[12]`) returns the twelve floats in row-major
order, which is what the GJK, EPA and narrow-phase kernels read.
"""
const Mat3x4f = SMatrix{4, 3, Float32, 12}

"""
    identity_transform() -> Mat3x4f

Row-major 3×4 identity transform. Useful default for instance records that
want pure translation specified later.
"""
identity_transform() = Mat3x4f(1.0f0, 0.0f0, 0.0f0, 0.0f0,
                               0.0f0, 1.0f0, 0.0f0, 0.0f0,
                               0.0f0, 0.0f0, 1.0f0, 0.0f0)

# The layout claim above is load-bearing for every backend that hands these
# bytes to a driver, so it is asserted rather than trusted.
@assert sizeof(Mat3x4f) == 48   "Mat3x4f must be 48 bytes (12 × Float32)"
@assert isbitstype(Mat3x4f)     "Mat3x4f must be plain bits for kernel use"


# ── Building transforms ───────────────────────────────────────────────────────
#
# From `src/vulkan/kernels/instance_writer.jl`, which packs these into instance
# records. The packing is an ABI and stayed there; composing a rotation and a
# scale into a 3x4 is arithmetic and is here.
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


# Moved here from the Vulkan backend: converting a 4×4 into the row-major 3×4 an
# instance descriptor wants is arithmetic, not driver code, and Metal needs the
# identical result — `MTLAccelerationStructureInstanceDescriptor` opens with the
# same 48 bytes `VkAccelerationStructureInstanceKHR` does. The old name is kept
# so no call site changes; `vk` in it is now historical rather than descriptive.
"""Convert a 4×4 matrix to a `Mat3x4f` (Vulkan row-major 3×4 layout).
Works with SMatrix{4,4}, Mat4f, or any indexable 4×4 matrix.

The returned `Mat3x4f` (= `SMatrix{4, 3, Float32, 12}`) is byte-identical to
`VkTransformMatrixKHR.matrix` (`float[12]`), so passing it as a kernel argument
or storing it into `VulkanInstanceRecord.transform` is a memcpy, not a layout
transform."""
function mat4_to_vk_transform(m)::Mat3x4f
    # Vulkan row-major 3×4:
    # Row 0: m[1,1], m[1,2], m[1,3], m[1,4]
    # Row 1: m[2,1], m[2,2], m[2,3], m[2,4]
    # Row 2: m[3,1], m[3,2], m[3,3], m[3,4]
    # SMatrix{4,3} ctor reads column-major, but our rows ARE the SMatrix's
    # columns (the byte-equivalence trick), so we just hand the rows over in
    # order.
    return Mat3x4f(
        Float32(m[1,1]), Float32(m[1,2]), Float32(m[1,3]), Float32(m[1,4]),
        Float32(m[2,1]), Float32(m[2,2]), Float32(m[2,3]), Float32(m[2,4]),
        Float32(m[3,1]), Float32(m[3,2]), Float32(m[3,3]), Float32(m[3,4]),
    )
end
