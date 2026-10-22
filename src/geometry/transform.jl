# Portable affine layout; native instance descriptors convert at their ABI boundary.
using StaticArrays: SMatrix

"""Mat3x4f

Row-major Float32 3×4 affine, stored as `SMatrix{4,3,Float32,12}` so linear
indices run along rows. Byte-identical to `VkTransformMatrixKHR`; Metal repacks
it into its column-major descriptor layout."""
const Mat3x4f = SMatrix{4, 3, Float32, 12}

"""identity_transform() -> Mat3x4f

Row-major 3×4 identity affine."""
identity_transform() = Mat3x4f(1.0f0, 0.0f0, 0.0f0, 0.0f0,
                               0.0f0, 1.0f0, 0.0f0, 0.0f0,
                               0.0f0, 0.0f0, 1.0f0, 0.0f0)

# Backends hand these bytes to drivers, so the layout is asserted.
@assert sizeof(Mat3x4f) == 48   "Mat3x4f must be 48 bytes (12 × Float32)"
@assert isbitstype(Mat3x4f)     "Mat3x4f must be plain bits for kernel use"


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

"""build_4x3_pervec(rot9, sc::Vec3f, p::Point3f) -> NTuple{12,Float32}

Scale rotation row i by sc[i], then append translation in core affine layout."""
@inline function build_4x3_pervec(rot9::NTuple{9, Float32}, sc::Vec3f, p::Point3f)
    sx, sy, sz = sc[1], sc[2], sc[3]
    (rot9[1] * sx, rot9[2] * sy, rot9[3] * sz, p[1],
     rot9[4] * sx, rot9[5] * sy, rot9[6] * sz, p[2],
     rot9[7] * sx, rot9[8] * sy, rot9[9] * sz, p[3])
end
