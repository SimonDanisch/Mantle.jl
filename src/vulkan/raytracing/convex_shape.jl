# Convex-shape abstract type for the narrow-phase collision pipeline (Phase P4).
#
# `support(shape::ConvexShape, dir::Vec3f) -> Vec3f` is the GJK building block:
# given a direction `dir`, returns the vertex of `shape` farthest along `dir`.
# Shapes live in instance-local space (centered at origin, "unit" extents);
# per-instance scale + rotation live in the TLAS instance transform.  The
# narrow-phase kernel transforms `dir` into local space, calls support, then
# transforms the result back.  Keeping `support` shape-local makes it pure
# geometry: easy to test on the CPU, easy to verify, GPU-callable without
# any transform-state plumbing.
#
# Future concrete shapes plug in via Julia multiple dispatch:
#   - UnitCube (this file): cube at origin, half-extents 1.
#   - Tetrahedron, ConvexHull{N}, Sphere, ... (later phases).

using GeometryBasics: Vec3f

"""
    ConvexShape

Abstract supertype for convex collision shapes.  Concrete subtypes implement
`support(shape, dir::Vec3f) -> Vec3f` returning the support point of the shape
along direction `dir`.  All shapes are in instance-local space (centered at
origin, with whatever "unit" extents the type defines).
"""
abstract type ConvexShape end

"""
    UnitCube

A cube centered at the origin with half-extents 1 along each axis (so corners
at (+-1, +-1, +-1)).  Per-instance size lives in the TLAS instance transform's
scale -- a grain with radius 0.005 has its UnitCube transformed by a uniform
scale of 0.005, so the world-space cube spans (+-0.005)^3.
"""
struct UnitCube <: ConvexShape end

"""
    support(shape::ConvexShape, dir::Vec3f) -> Vec3f

Return the vertex of `shape` farthest along `dir` in instance-local space.
This is the GJK building block: given any direction, it returns the support
point of the convex Minkowski difference's contributor from this shape.

For UnitCube, the support point is the corner with components in the same sign
as `dir` (so the corner farthest along `dir`).  Components of `dir` that are
exactly zero pick `+1` by convention -- since the resulting tie is symmetric,
either corner is a valid support point and GJK is robust to the choice.
"""
@inline function support(::UnitCube, dir::Vec3f)
    sx = dir[1] >= 0f0 ? 1f0 : -1f0
    sy = dir[2] >= 0f0 ? 1f0 : -1f0
    sz = dir[3] >= 0f0 ? 1f0 : -1f0
    return Vec3f(sx, sy, sz)
end
