# EPA penetration recovery shared by host references and GPU kernels.
# Expand a bounded polytope in A-B space; interpolate shape A's support points
# at the closest face. Geometry/support conventions come from gjk.jl.
using GeometryBasics: Vec3f
using LinearAlgebra: dot, cross, norm
using StaticArrays: MVector

# Fixed polytope capacity; exhausting any bound returns converged=false.
const EPA_MAX_VERTS = 32
const EPA_MAX_FACES = 64
const EPA_MAX_EDGES = 64

"""EPAResult

Closest-face normal of A-B, penetration depth, world-space contact on A,
iteration count and convergence flag. Separating A uses -depth*normal.
Iteration/storage exhaustion or an invalid polytope sets converged=false;
failure paths may return a zero contact rather than an interpolated point."""
struct EPAResult
    normal::Vec3f
    depth::Float32
    contact::Vec3f
    iterations::Int
    converged::Bool
end

# Four seed support pairs: (Minkowski vertices, shape A's world-space points).
# Degenerate directions fall back to canonical axes; there is no success flag.
@inline function seed_polytope(A::ConvexShape, B::ConvexShape,
                               T_A::NTuple{12,Float32}, T_B::NTuple{12,Float32},
                               eps::Float32)
    # Direction 1: world-space vector from B toward A (same heuristic as gjk()).
    origin_A = Vec3f(T_A[4],  T_A[8],  T_A[12])
    origin_B = Vec3f(T_B[4],  T_B[8],  T_B[12])
    d1       = origin_A - origin_B
    if dot(d1, d1) < eps * eps
        d1 = Vec3f(1f0, 0f0, 0f0)
    end
    sA1, _, m1 = support_pair_world(A, B, T_A, T_B, d1)

    # Direction 2: from m1 toward origin.
    d2 = -m1
    if dot(d2, d2) < eps * eps
        # m1 is essentially at origin; pick an arbitrary perpendicular axis.
        d2 = Vec3f(0f0, 1f0, 0f0)
    end
    sA2, _, m2 = support_pair_world(A, B, T_A, T_B, d2)

    # Direction 3: perpendicular to (m2 - m1), toward the origin.
    edge12 = m2 - m1
    AO     = -m1   # toward origin from m1
    d3     = triple_product_dir(edge12, AO)
    if dot(d3, d3) < eps * eps
        d3 = perpendicular_to(edge12)
    end
    sA3, _, m3 = support_pair_world(A, B, T_A, T_B, d3)

    # Direction 4: face-normal of triangle (m1, m2, m3) on the side that
    # contains the origin.
    e12 = m2 - m1
    e13 = m3 - m1
    n4  = cross(e12, e13)
    # Flip toward the origin, which is on the +n4 side iff dot(n4, m1) < 0.
    if dot(n4, m1) > 0f0
        n4 = -n4
    end
    if dot(n4, n4) < eps * eps
        # Triangle is degenerate; pick any axis.
        n4 = Vec3f(0f0, 0f0, 1f0)
    end
    sA4, _, m4 = support_pair_world(A, B, T_A, T_B, n4)

    return (m1, m2, m3, m4), (sA1, sA2, sA3, sA4)
end

# Face plane: outward unit normal + signed distance from origin to plane.
# Returns (normal, distance).  If the triangle is degenerate (zero-area),
# returns (Vec3f(0), Inf32) which the closest-face search will skip.
@inline function compute_face_plane(v1::Vec3f, v2::Vec3f, v3::Vec3f, interior::Vec3f)
    e1   = v2 - v1
    e2   = v3 - v1
    nraw = cross(e1, e2)
    nlen = sqrt(dot(nraw, nraw))
    if nlen <= 0f0
        return Vec3f(0f0), Inf32
    end
    n = nraw / nlen
    # Flip so n points away from `interior` (i.e. outward from the polytope).
    if dot(n, v1 - interior) < 0f0
        n = -n
    end
    # Outward plane equation dot(n,x)=dot(n,v1); the origin projection is d*n.
    d = dot(n, v1)
    return n, d
end

# Closest-face search: pick the live face with smallest face.distance.
# Returns the index, or -1 if no live faces (which would be a polytope bug).
@inline function find_closest_face(face_alive::MVector, face_dist::MVector, n_faces::Int)
    best_idx = -1
    best_d   = Inf32
    @inbounds for i in 1:n_faces
        if face_alive[i] && face_dist[i] < best_d
            best_d   = face_dist[i]
            best_idx = i
        end
    end
    return best_idx
end

# Barycentric interpolation (Real-Time Collision Detection §3.4).
# Degenerate triangles use weights (1,0,0).
@inline function barycentric_in_triangle(p::Vec3f, v1::Vec3f, v2::Vec3f, v3::Vec3f)
    e1   = v2 - v1
    e2   = v3 - v1
    e0   = p  - v1
    d00  = dot(e1, e1)
    d01  = dot(e1, e2)
    d11  = dot(e2, e2)
    d20  = dot(e0, e1)
    d21  = dot(e0, e2)
    denom = d00 * d11 - d01 * d01
    if abs(denom) < 1f-20
        return (1f0, 0f0, 0f0)
    end
    b2 = (d11 * d20 - d01 * d21) / denom
    b3 = (d00 * d21 - d01 * d20) / denom
    b1 = 1f0 - b2 - b3
    return (b1, b2, b3)
end

# Interpolate shape A's support points at the closest face's origin projection.
@inline function face_result(verts_m, verts_sA, v, normal, distance, iter, converged)
    b1, b2, b3 = barycentric_in_triangle(normal * distance,
        verts_m[v[1]], verts_m[v[2]], verts_m[v[3]])
    contact = b1 * verts_sA[v[1]] + b2 * verts_sA[v[2]] + b3 * verts_sA[v[3]]
    return EPAResult(normal, distance, contact, iter, converged)
end

# The fixed-capacity MVectors are device-local scratch; on the CPU they may
# heap-allocate.

"""epa(A, B, T_A, T_B, gjk_simplex; max_iters=24, eps=1f-4) -> EPAResult

Recover closest-face penetration data after GJK reports overlap. `gjk_simplex`
is unused: EPA rebuilds support pairs to track shape A's contact points. See
EPAResult for orientation and failure data."""
function epa(A::ConvexShape, B::ConvexShape,
             T_A::NTuple{12,Float32}, T_B::NTuple{12,Float32},
             gjk_simplex::NTuple{4,Vec3f};
             max_iters::Int=24, eps::Float32=1f-4)

    # The supplied simplex lacks support-pair bookkeeping; rebuild it below.
    verts_m  = MVector{EPA_MAX_VERTS, Vec3f}(undef)   # Minkowski-difference vertices
    verts_sA = MVector{EPA_MAX_VERTS, Vec3f}(undef)   # per-vertex world-space s_A
    face_v   = MVector{EPA_MAX_FACES, NTuple{3,Int32}}(undef)
    face_n   = MVector{EPA_MAX_FACES, Vec3f}(undef)
    face_d   = MVector{EPA_MAX_FACES, Float32}(undef)
    face_alive = MVector{EPA_MAX_FACES, Bool}(undef)
    edges    = MVector{EPA_MAX_EDGES, NTuple{2,Int32}}(undef)

    seed_m, seed_sA = seed_polytope(A, B, T_A, T_B, eps)
    verts_m[1]  = seed_m[1];  verts_sA[1] = seed_sA[1]
    verts_m[2]  = seed_m[2];  verts_sA[2] = seed_sA[2]
    verts_m[3]  = seed_m[3];  verts_sA[3] = seed_sA[3]
    verts_m[4]  = seed_m[4];  verts_sA[4] = seed_sA[4]
    n_verts = 4

    # The centroid orients outward normals; the origin is interior too but may
    # lie exactly on a face.
    interior = (verts_m[1] + verts_m[2] + verts_m[3] + verts_m[4]) * 0.25f0

    # The tetrahedron's faces; `compute_face_plane` orients each normal away
    # from `interior`.
    face_v[1] = (Int32(1), Int32(2), Int32(3))
    face_v[2] = (Int32(1), Int32(3), Int32(4))
    face_v[3] = (Int32(1), Int32(4), Int32(2))
    face_v[4] = (Int32(2), Int32(4), Int32(3))
    n_faces = 4
    @inbounds for i in 1:4
        v = face_v[i]
        n_, d_ = compute_face_plane(verts_m[v[1]], verts_m[v[2]], verts_m[v[3]], interior)
        face_n[i]     = n_
        face_d[i]     = d_
        face_alive[i] = true
    end

    last_idx = -1
    for iter in 1:max_iters
        idx = find_closest_face(face_alive, face_d, n_faces)
        if idx <= 0
            # No live faces: the polytope is corrupt; fall back to the last face.
            n_  = (last_idx > 0) ? face_n[last_idx] : Vec3f(1f0, 0f0, 0f0)
            d_  = (last_idx > 0) ? face_d[last_idx] : 0f0
            return EPAResult(n_, d_, Vec3f(0f0), iter, false)
        end
        last_idx = idx

        # Find a new support point in the direction of this face's outward normal.
        n_face = face_n[idx]
        d_face = face_d[idx]
        sA_new, _, m_new = support_pair_world(A, B, T_A, T_B, n_face)
        d_new  = dot(m_new, n_face)

        # Converged: the new support point is not farther out than the face plane.
        if d_new - d_face < eps
            return face_result(verts_m, verts_sA, face_v[idx], n_face, d_face, iter, true)
        end

        # Expand: remove every face visible from m_new, collect the silhouette
        # edges of the removed region, and fan a new face from m_new over each.
        n_edges = 0
        @inbounds for fi in 1:n_faces
            if !face_alive[fi]
                continue
            end
            # Visible: m_new is on the outward side of the face plane.
            v_idx_fi = face_v[fi]
            v1_fi    = verts_m[v_idx_fi[1]]
            if dot(face_n[fi], m_new - v1_fi) > 0f0
                # Mark dead and add its edges. An edge shared by two visible
                # faces appears again reversed, and the pair cancels.
                face_alive[fi] = false
                for (a, b) in ((v_idx_fi[1], v_idx_fi[2]),
                               (v_idx_fi[2], v_idx_fi[3]),
                               (v_idx_fi[3], v_idx_fi[1]))
                    found = 0
                    for ei in 1:n_edges
                        e = edges[ei]
                        if e[1] == b && e[2] == a
                            found = ei
                            break
                        end
                    end
                    if found > 0
                        # Swap-remove (compact the buffer).
                        edges[found] = edges[n_edges]
                        n_edges -= 1
                    else
                        if n_edges >= EPA_MAX_EDGES
                            # Silhouette buffer full: closest face, converged=false.
                            # No allocation, so the kernel compiles to SPIR-V.
                            return EPAResult(face_n[idx], face_d[idx], Vec3f(0f0), iter, false)
                        end
                        n_edges += 1
                        edges[n_edges] = (Int32(a), Int32(b))
                    end
                end
            end
        end

        if n_edges == 0
            # No face visible: m_new is inside the polytope through rounding;
            # treat as converged.
            return face_result(verts_m, verts_sA, face_v[idx], n_face, d_face, iter, true)
        end

        if n_verts >= EPA_MAX_VERTS
            # Vertex storage full: closest face, converged=false.
            return EPAResult(face_n[idx], face_d[idx], Vec3f(0f0), iter, false)
        end
        n_verts += 1
        verts_m[n_verts]  = m_new
        verts_sA[n_verts] = sA_new
        new_v_idx = Int32(n_verts)

        # One face per silhouette edge, fanning from m_new; `compute_face_plane`
        # orients it.
        @inbounds for ei in 1:n_edges
            if n_faces >= EPA_MAX_FACES
                # Face storage full: closest face, converged=false.
                return EPAResult(face_n[idx], face_d[idx], Vec3f(0f0), iter, false)
            end
            # Slot reuse: scan for a dead face slot first; only grow if none.
            slot = 0
            for k in 1:n_faces
                if !face_alive[k]
                    slot = k
                    break
                end
            end
            if slot == 0
                n_faces += 1
                slot = n_faces
            end
            e = edges[ei]
            face_v[slot] = (e[1], e[2], new_v_idx)
            v1_ = verts_m[e[1]]
            v2_ = verts_m[e[2]]
            v3_ = m_new
            n_, d_ = compute_face_plane(v1_, v2_, v3_, interior)
            face_n[slot]     = n_
            face_d[slot]     = d_
            face_alive[slot] = true
        end
    end

    # Iteration cap: closest face at the current state, converged=false.
    idx = find_closest_face(face_alive, face_d, n_faces)
    if idx <= 0
        return EPAResult(Vec3f(1f0, 0f0, 0f0), 0f0, Vec3f(0f0), max_iters, false)
    end
    return face_result(verts_m, verts_sA, face_v[idx], face_n[idx], face_d[idx], max_iters, false)
end
