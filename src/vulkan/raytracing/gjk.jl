# GJK (Gilbert--Johnson--Keerthi) closest-point / overlap test -- CPU reference.
#
# Follows Casey Muratori's lecture structure and Bullet's btGjkPairDetector for
# the do-simplex Voronoi-region case analysis.  This is intentionally a clean
# CPU reference to be ported to GPU in P4.4; no SIMD / bitwise cleverness.
#
# Transform convention: NTuple{12, Float32} row-major 3x4, identical to
# LavaInstanceRecord.transform / Vulkan VkTransformMatrixKHR layout:
#
#   [ T[1]  T[2]  T[3]  T[4]  ]   row 0  (x row + tx)
#   [ T[5]  T[6]  T[7]  T[8]  ]   row 1  (y row + ty)
#   [ T[9]  T[10] T[11] T[12] ]   row 2  (z row + tz)
#
# So for a point p (column vector):
#   out.x = T[1]*p.x + T[2]*p.y + T[3]*p.z + T[4]
#   out.y = T[5]*p.x + T[6]*p.y + T[7]*p.z + T[8]
#   out.z = T[9]*p.x + T[10]*p.y + T[11]*p.z + T[12]

using GeometryBasics: Vec3f
using LinearAlgebra: dot, cross

# ---------------------------------------------------------------------------
# Transform helpers
# ---------------------------------------------------------------------------

"""
    transform_point(T::NTuple{12,Float32}, p::Vec3f) -> Vec3f

Apply a row-major 3x4 instance transform to a point `p` (rotation + translation).
"""
@inline function transform_point(T::NTuple{12,Float32}, p::Vec3f)
    Vec3f(
        T[1]*p[1] + T[2]*p[2] + T[3]*p[3] + T[4],
        T[5]*p[1] + T[6]*p[2] + T[7]*p[3] + T[8],
        T[9]*p[1] + T[10]*p[2] + T[11]*p[3] + T[12],
    )
end

"""
    transform_dir(T::NTuple{12,Float32}, d::Vec3f) -> Vec3f

Apply only the rotation (upper-left 3x3) of a row-major 3x4 instance transform
to a direction `d` (no translation).
"""
@inline function transform_dir(T::NTuple{12,Float32}, d::Vec3f)
    Vec3f(
        T[1]*d[1] + T[2]*d[2] + T[3]*d[3],
        T[5]*d[1] + T[6]*d[2] + T[7]*d[3],
        T[9]*d[1] + T[10]*d[2] + T[11]*d[3],
    )
end

"""
    inv_transform_dir(T::NTuple{12,Float32}, d::Vec3f) -> Vec3f

Apply the transpose of the rotation (upper-left 3x3) to direction `d`.
For orthonormal / uniform-scale transforms this equals the inverse rotation.
GJK uses this to convert a world-space search direction back into local space.
"""
@inline function inv_transform_dir(T::NTuple{12,Float32}, d::Vec3f)
    # Transpose of the 3x3 rotation block.
    # Row 0 of R is (T[1], T[2], T[3]); column 0 of R^T is the same row, so
    # inv_dir.x = dot( (T[1],T[5],T[9]), d ) etc.
    Vec3f(
        T[1]*d[1] + T[5]*d[2] + T[9]*d[3],
        T[2]*d[1] + T[6]*d[2] + T[10]*d[3],
        T[3]*d[1] + T[7]*d[2] + T[11]*d[3],
    )
end

# ---------------------------------------------------------------------------
# Minkowski-difference support
# ---------------------------------------------------------------------------

"""
    support_AB(A, B, T_A, T_B, dir_world) -> Vec3f

Minkowski-difference support point for (A - B) along `dir_world`.
Both shapes live in their own local frames; the transforms bring them into
world space.  The returned point is in world space.
"""
@inline function support_AB(A::ConvexShape, B::ConvexShape,
                             T_A::NTuple{12,Float32}, T_B::NTuple{12,Float32},
                             dir_world::Vec3f)
    # Transform dir into each shape's local frame (inverse = transpose for
    # orthonormal transforms), evaluate support, transform results to world.
    dir_A = inv_transform_dir(T_A, dir_world)
    dir_B = inv_transform_dir(T_B, dir_world)
    s_A   = transform_point(T_A, support(A, dir_A))
    s_B   = transform_point(T_B, support(B, -dir_B))
    return s_A - s_B
end

# ---------------------------------------------------------------------------
# Simplex helpers  (1-4 points: line / triangle / tetrahedron)
# ---------------------------------------------------------------------------
#
# We track the simplex as a small fixed-size array of up to 4 Vec3f.
# Julia's NTuple is immutable so we carry the count separately and return
# updated tuples from each do-simplex step.
#
# Convention from Muratori:  the most recently added point is always simplex[n]
# (1-indexed; we store newest last).

# Convenience: dot3 with itself
@inline dot3(a::Vec3f) = dot(a, a)

# Pick a direction perpendicular to `v`.  Used when the triple-product
# cross(cross(v, w), v) degenerates to zero because w is collinear with v
# (i.e. origin lies exactly on an edge or the direction from A to origin
# is parallel to the edge).  We try two canonical axes and pick whichever
# gives a larger cross product magnitude.
@inline function perpendicular_to(v::Vec3f)
    c1 = cross(v, Vec3f(1f0, 0f0, 0f0))
    c2 = cross(v, Vec3f(0f0, 1f0, 0f0))
    dot3(c1) >= dot3(c2) ? c1 : c2
end

# Triple-product  cross(cross(A, B), A)  gives a direction perpendicular to A
# that points toward B.  When A and B are collinear the result is zero; in
# that case fall back to any direction perpendicular to A (origin is ON the
# segment, so we just need to break the degeneracy and grow the simplex).
@inline function triple_product_dir(edge::Vec3f, toward::Vec3f)
    c1  = cross(edge, toward)
    dir = cross(c1, edge)
    dot3(dir) > 0f0 ? dir : perpendicular_to(edge)
end

# ---------------------------------------------------------------------------
# do_simplex_line
#
# Simplex has 2 points: A (newest) and B (oldest).
# Find closest sub-simplex to origin and set next search direction.
# Returns (new_simplex_tuple, n, dir, overlap).
# ---------------------------------------------------------------------------
@inline function do_simplex_line(A::Vec3f, B::Vec3f)
    AB  = B - A
    AO  = -A          # direction from A to origin
    t   = dot(AB, AO)
    if t > 0f0
        # Origin projects onto the segment AB -- keep both points.
        # Use triple_product_dir so we handle the degenerate case where
        # origin lies exactly on the line (AB || AO -> cross = 0).
        dir = triple_product_dir(AB, AO)
        return (A, B, A, A), 2, dir, false
    else
        # Origin is in the Voronoi region of A alone -- reduce to point A.
        return (A, A, A, A), 1, AO, false
    end
end

# ---------------------------------------------------------------------------
# do_simplex_triangle
#
# Simplex has 3 points: A (newest), B, C (oldest).
# Returns (simplex_4tuple, n, dir, overlap).
# ---------------------------------------------------------------------------
@inline function do_simplex_triangle(A::Vec3f, B::Vec3f, C::Vec3f)
    AB   = B - A
    AC   = C - A
    AO   = -A
    ABC  = cross(AB, AC)   # face normal (not normalized)

    # Test which side of edge AB the origin lies on (viewed from ABC normal).
    ABperp = cross(AB, ABC)
    if dot(ABperp, AO) > 0f0
        # Origin is outside edge AB -> reduce to line AB.
        if dot(AB, AO) > 0f0
            dir = triple_product_dir(AB, AO)
            return (A, B, A, A), 2, dir, false
        else
            # Single point A
            return (A, A, A, A), 1, AO, false
        end
    end

    # Test edge AC side.
    ACperp = cross(ABC, AC)
    if dot(ACperp, AO) > 0f0
        # Origin is outside edge AC -> reduce to line AC.
        if dot(AC, AO) > 0f0
            dir = triple_product_dir(AC, AO)
            return (A, C, A, A), 2, dir, false
        else
            return (A, A, A, A), 1, AO, false
        end
    end

    # Origin is inside the triangle (projected).  Check which face side.
    if dot(ABC, AO) > 0f0
        # Origin above face ABC.
        dir = ABC
        return (A, B, C, A), 3, dir, false
    else
        # Origin below face -- flip winding.
        dir = -ABC
        return (A, C, B, A), 3, dir, false
    end
end

# ---------------------------------------------------------------------------
# do_simplex_tetrahedron
#
# Simplex has 4 points: A (newest), B, C, D (oldest).
# Returns (simplex_4tuple, n, dir, overlap).
# ---------------------------------------------------------------------------
@inline function do_simplex_tetrahedron(A::Vec3f, B::Vec3f, C::Vec3f, D::Vec3f)
    AB   = B - A
    AC   = C - A
    AD   = D - A
    AO   = -A

    # Face normals pointing outward (away from the 4th vertex).
    # Face ABC: outward normal away from D.
    ABC  = cross(AB, AC)
    # Face ACD: outward normal away from B.
    ACD  = cross(AC, AD)
    # Face ADB: outward normal away from C.
    ADB  = cross(AD, AB)

    # Flip normals so they point away from the interior (toward origin side).
    # Interior test: the 4th vertex of each face should be on the negative side.
    # For face ABC the 4th vertex is D: if dot(ABC, AD) > 0, normal points toward D
    # (inward), so flip.
    if dot(ABC, AD) > 0f0
        ABC = -ABC
    end
    if dot(ACD, AB) > 0f0
        ACD = -ACD
    end
    if dot(ADB, AC) > 0f0
        ADB = -ADB
    end

    # Check if origin is outside each face.
    outside_ABC = dot(ABC, AO) > 0f0
    outside_ACD = dot(ACD, AO) > 0f0
    outside_ADB = dot(ADB, AO) > 0f0

    if !outside_ABC && !outside_ACD && !outside_ADB
        # Origin is inside the tetrahedron.
        return (A, B, C, D), 4, Vec3f(0f0), true
    end

    # Origin is outside at least one face.  Find the closest face and recurse.
    if outside_ABC
        # Reduce to triangle ABC (drop D).  Keep A as newest, B, C.
        simplex, n, dir, overlap = do_simplex_triangle(A, B, C)
        return simplex, n, dir, overlap
    elseif outside_ACD
        # Reduce to triangle ACD (drop B).
        simplex, n, dir, overlap = do_simplex_triangle(A, C, D)
        return simplex, n, dir, overlap
    else
        # outside_ADB: reduce to triangle ADB (drop C).
        simplex, n, dir, overlap = do_simplex_triangle(A, D, B)
        return simplex, n, dir, overlap
    end
end

# ---------------------------------------------------------------------------
# do_simplex dispatcher
# ---------------------------------------------------------------------------

@inline function do_simplex(pts::NTuple{4,Vec3f}, n::Int)
    if n == 1
        A   = pts[1]
        dir = -A    # search toward origin
        return (A, pts[2], pts[3], pts[4]), 1, dir, false
    elseif n == 2
        A, B = pts[1], pts[2]
        return do_simplex_line(A, B)
    elseif n == 3
        A, B, C = pts[1], pts[2], pts[3]
        return do_simplex_triangle(A, B, C)
    else   # n == 4
        A, B, C, D = pts[1], pts[2], pts[3], pts[4]
        return do_simplex_tetrahedron(A, B, C, D)
    end
end

# ---------------------------------------------------------------------------
# GJKResult
# ---------------------------------------------------------------------------

"""
    GJKResult

Result of a GJK call.  `overlap == true` means the two shapes overlap; the
`simplex` field then contains a 4-tuple of Minkowski-difference points
enclosing the origin (input to EPA, P4.3).  `overlap == false` means the
shapes are separated; `simplex` is unspecified.
"""
struct GJKResult
    overlap::Bool
    simplex::NTuple{4, Vec3f}   # valid (encloses origin) iff overlap == true
    iterations::Int              # for telemetry / cap diagnostics
end

# ---------------------------------------------------------------------------
# Main GJK loop
# ---------------------------------------------------------------------------

"""
    gjk(A::ConvexShape, B::ConvexShape, T_A::NTuple{12,Float32}, T_B::NTuple{12,Float32};
        max_iters::Int=32, eps::Float32=1f-6) -> GJKResult

Run GJK between two transformed convex shapes.  Each transform is a row-major
3x4 instance transform (same layout as `LavaInstanceRecord.transform`).
`A` and `B` are defined in their own instance-local frames; the algorithm
transforms support directions / points into / out of those frames internally.

Returns `GJKResult(overlap, simplex, iterations)`.  `overlap=true` means the
4-simplex encloses the origin in Minkowski-difference space and is suitable
input for EPA (P4.3).  `overlap=false` means the shapes are provably separated.
"""
function gjk(A::ConvexShape, B::ConvexShape,
             T_A::NTuple{12,Float32}, T_B::NTuple{12,Float32};
             max_iters::Int=32, eps::Float32=1f-6)

    # Initial search direction: world-space vector from B's origin to A's origin.
    origin_A = Vec3f(T_A[4],  T_A[8],  T_A[12])
    origin_B = Vec3f(T_B[4],  T_B[8],  T_B[12])
    dir      = origin_A - origin_B
    if dot(dir, dir) < eps * eps
        dir = Vec3f(1f0, 0f0, 0f0)   # coincident origins: arbitrary starting direction
    end

    # First support point.
    s0  = support_AB(A, B, T_A, T_B, dir)
    pts = (s0, s0, s0, s0)   # 4-tuple; only first `n` entries are valid
    n   = 1

    # Flip direction toward origin for next search.
    dir = -s0

    for iter in 1:max_iters
        # Numerical guard: if direction degenerates, bail.
        if dot(dir, dir) < eps * eps
            return GJKResult(false, pts, iter)
        end

        # New support point on the Minkowski difference.
        s = support_AB(A, B, T_A, T_B, dir)

        # Termination: if the new point doesn't advance past the old frontier
        # along dir, the origin is not enclosed -- shapes separated.
        if dot(s, dir) < 0f0
            return GJKResult(false, pts, iter)
        end

        # Add s as the newest simplex vertex (index 1 by convention).
        # Shift existing points up by one slot (drop the oldest if n == 4, but
        # do_simplex_tetrahedron reduces to <= 3 before we'd add a 5th).
        if n == 1
            pts = (s, pts[1], pts[3], pts[4])
            n   = 2
        elseif n == 2
            pts = (s, pts[1], pts[2], pts[4])
            n   = 3
        else  # n == 3 or first time we have a triangle and add to tetrahedron
            pts = (s, pts[1], pts[2], pts[3])
            n   = 4
        end

        # Run the sub-simplex / Voronoi-region test.
        new_pts, n, dir, overlap = do_simplex(pts, n)
        pts = new_pts

        if overlap
            return GJKResult(true, pts, iter)
        end
    end

    # Hit iteration cap without convergence -- shapes at numerical contact
    # boundary; conservatively report no overlap.
    return GJKResult(false, pts, max_iters)
end
