# GJK overlap test shared by host references and GPU kernels.
# Transforms use the core row-major affine layout; support directions use its
# transposed linear block. Simplex region tests follow Muratori/Bullet.
using GeometryBasics: Vec3f
using LinearAlgebra: dot, cross

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

"""inv_transform_dir(T::NTuple{12,Float32}, d::Vec3f) -> Vec3f

Apply the transpose of T's linear block to a world-space support direction.
Despite the name, this is not a general inverse; it equals inverse rotation
only when the block is orthonormal."""
@inline function inv_transform_dir(T::NTuple{12,Float32}, d::Vec3f)
    Vec3f(
        T[1]*d[1] + T[5]*d[2] + T[9]*d[3],
        T[2]*d[1] + T[6]*d[2] + T[10]*d[3],
        T[3]*d[1] + T[7]*d[2] + T[11]*d[3],
    )
end

"""
    support_AB(A, B, T_A, T_B, dir_world) -> Vec3f

Minkowski-difference support point for (A - B) along `dir_world`, in world space.
Both shapes live in their own local frames; the transforms bring them to world.
"""
@inline support_AB(A::ConvexShape, B::ConvexShape,
                   T_A::NTuple{12,Float32}, T_B::NTuple{12,Float32}, dir_world::Vec3f) =
    support_pair_world(A, B, T_A, T_B, dir_world)[3]

"World-space support pair and its Minkowski difference, shared by GJK and EPA."
@inline function support_pair_world(A::ConvexShape, B::ConvexShape,
                             T_A::NTuple{12,Float32}, T_B::NTuple{12,Float32},
                             dir_world::Vec3f)
    dir_A = inv_transform_dir(T_A, dir_world)
    dir_B = inv_transform_dir(T_B, dir_world)
    s_A   = transform_point(T_A, support(A, dir_A))
    s_B   = transform_point(T_B, support(B, -dir_B))
    return s_A, s_B, s_A - s_B
end

# A simplex is a fixed four-point tuple plus an active count, newest at index 1.
@inline dot3(a::Vec3f) = dot(a, a)

# Break collinear simplex degeneracy using the better of two canonical axes.
@inline function perpendicular_to(v::Vec3f)
    c1 = cross(v, Vec3f(1f0, 0f0, 0f0))
    c2 = cross(v, Vec3f(0f0, 1f0, 0f0))
    dot3(c1) >= dot3(c2) ? c1 : c2
end

# cross(cross(A, B), A) is perpendicular to A and points toward B. Collinear A
# and B give zero (the origin is on the segment), so fall back to any
# perpendicular to grow the simplex.
@inline function triple_product_dir(edge::Vec3f, toward::Vec3f)
    c1  = cross(edge, toward)
    dir = cross(c1, edge)
    dot3(dir) > 0f0 ? dir : perpendicular_to(edge)
end

# Reduce a two-point simplex to the closest region; A is newest.
@inline function do_simplex_line(A::Vec3f, B::Vec3f)
    AB  = B - A
    AO  = -A          # direction from A to origin
    t   = dot(AB, AO)
    if t > 0f0
        # Origin projects onto segment AB: keep both points. `triple_product_dir`
        # handles the origin lying on the line.
        dir = triple_product_dir(AB, AO)
        return (A, B, A, A), 2, dir, false
    else
        # Origin in A's Voronoi region: reduce to point A.
        return (A, A, A, A), 1, AO, false
    end
end

# Three points, A newest, C oldest. Returns (simplex_4tuple, n, dir, overlap).
@inline function do_simplex_triangle(A::Vec3f, B::Vec3f, C::Vec3f)
    AB   = B - A
    AC   = C - A
    AO   = -A
    ABC  = cross(AB, AC)   # face normal (not normalized)

    # Test which side of edge AB the origin lies on (viewed from ABC normal).
    ABperp = cross(AB, ABC)
    if dot(ABperp, AO) > 0f0
        # Origin is outside edge AB -> reduce to line AB.
        return do_simplex_line(A, B)
    end

    # Test edge AC side.
    ACperp = cross(ABC, AC)
    if dot(ACperp, AO) > 0f0
        # Origin is outside edge AC -> reduce to line AC.
        return do_simplex_line(A, C)
    end

    # Origin is inside the triangle (projected).  Check which face side.
    if dot(ABC, AO) > 0f0
        # Origin above face ABC.
        dir = ABC
        return (A, B, C, A), 3, dir, false
    else
        # Origin below face: flip winding.
        dir = -ABC
        return (A, C, B, A), 3, dir, false
    end
end

# Four points, A newest, D oldest. Returns (simplex_4tuple, n, dir, overlap).
@inline function do_simplex_tetrahedron(A::Vec3f, B::Vec3f, C::Vec3f, D::Vec3f)
    AB   = B - A
    AC   = C - A
    AD   = D - A
    AO   = -A

    # Face normals, flipped below to point away from each face's fourth vertex.
    ABC  = cross(AB, AC)
    ACD  = cross(AC, AD)
    ADB  = cross(AD, AB)

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

    # Reduce to the first face, in this order, that the origin is outside of.
    if outside_ABC
        return do_simplex_triangle(A, B, C)
    elseif outside_ACD
        return do_simplex_triangle(A, C, D)
    else
        return do_simplex_triangle(A, D, B)
    end
end

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

"""GJKResult

Overlap flag, Minkowski simplex and iteration count. On overlap the simplex
feeds EPA; false also covers degeneracy and iteration exhaustion."""
struct GJKResult
    overlap::Bool
    simplex::NTuple{4, Vec3f}   # valid (encloses origin) iff overlap == true
    iterations::Int              # for telemetry / cap diagnostics
end

"""gjk(A, B, T_A, T_B; max_iters=32, eps=1f-6) -> GJKResult

Test transformed convex shapes using row-major Float32 affine tuples.
An enclosing tetrahedron reports overlap; separation, degenerate search
direction or iteration exhaustion reports false. No distance certificate is returned."""
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

        s = support_AB(A, B, T_A, T_B, dir)

        # The new point does not pass the origin along `dir`: separated.
        if dot(s, dir) < 0f0
            return GJKResult(false, pts, iter)
        end

        # Push s as the newest vertex; `do_simplex` leaves n <= 3 unless it
        # reports overlap, so nothing is dropped.
        if n == 1
            pts = (s, pts[1], pts[3], pts[4])
            n   = 2
        elseif n == 2
            pts = (s, pts[1], pts[2], pts[4])
            n   = 3
        else  # n == 3
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

    # Iteration cap without convergence (numerical contact boundary): report no
    # overlap.
    return GJKResult(false, pts, max_iters)
end
