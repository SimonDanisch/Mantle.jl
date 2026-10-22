# Convex shapes for the narrow phase. `support` works in instance-local space
# (centred at the origin, unit extents); per-instance scale and rotation live in
# the HWTLAS instance transform, which GJK/EPA apply around each `support` call.

using GeometryBasics: Vec3f

"""
    ConvexShape

Abstract supertype for convex collision shapes. Subtypes implement
`support(shape, dir::Vec3f) -> Vec3f` in instance-local space.
"""
abstract type ConvexShape end

"""
    UnitCube

A cube centred at the origin with half-extents 1, corners at (±1, ±1, ±1). Its
size comes from the HWTLAS instance transform's scale.
"""
struct UnitCube <: ConvexShape end

"""
    support(shape::ConvexShape, dir::Vec3f) -> Vec3f

Return the vertex of `shape` farthest along `dir` in instance-local space. For
`UnitCube`, the corner with the signs of `dir`; a zero component picks `+1`,
which is valid since either corner of the tie is a support point.
"""
@inline function support(::UnitCube, dir::Vec3f)
    sx = dir[1] >= 0f0 ? 1f0 : -1f0
    sy = dir[2] >= 0f0 ? 1f0 : -1f0
    sz = dir[3] >= 0f0 ? 1f0 : -1f0
    return Vec3f(sx, sy, sz)
end
