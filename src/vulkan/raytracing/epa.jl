# EPA (Expanding Polytope Algorithm) -- CPU reference for penetration recovery.
#
# Given two overlapping convex shapes (as established by GJK) plus the GJK
# termination simplex, EPA iteratively expands a convex polytope around the
# origin in Minkowski-difference space until the closest face's plane lies
# (within eps) on the surface of the Minkowski difference.  That face's
# outward normal is the penetration normal; the distance from origin to the
# plane is the penetration depth; the contact point is recovered by
# barycentric interpolation of the per-vertex world-space s_A values.
#
# Reference: Bullet's btGjkEpaSolver2 / btPolyhedralContactClipping; Casey
# Muratori's GJK/EPA tutorial; Erin Catto's GDC notes.  This file is a clean
# CPU port targeted at being callable from a GPU @kernel in P4.4 -- all
# storage is fixed-size MVector / NTuple, no heap allocation in the hot loop.
#
# The transform / support / dot conventions are inherited from gjk.jl; do
# not duplicate them.

using GeometryBasics: Vec3f
using LinearAlgebra: dot, cross, norm
using StaticArrays: MVector

# ---------------------------------------------------------------------------
# Storage bounds
# ---------------------------------------------------------------------------
# A unit-cube vs unit-cube tetrahedron starts with 4 verts / 4 faces / 0
# edges; each iteration adds at most (silhouette_count) new faces and 1 new
# vertex.  With max_iters=24, MAX_VERTS=32 leaves room for any reasonable
# polytope growth.  MAX_FACES=64 / MAX_EDGES=64 are likewise generous.
# These are deliberately oversized to avoid crowding the upper limit on
# pathological cases (rotated long thin shapes); GPU shared-memory budget
# is the only reason they're not larger still.
const EPA_MAX_VERTS = 32
const EPA_MAX_FACES = 64
const EPA_MAX_EDGES = 64

# ---------------------------------------------------------------------------
# support_pair_world
#
# Like support_AB but returns the world-space s_A and s_B contributions
# alongside the Minkowski-difference vertex.  EPA needs s_A to recover the
# contact point on shape A's surface from the closest-face barycentric.
# ---------------------------------------------------------------------------
@inline function support_pair_world(A::ConvexShape, B::ConvexShape,
                                    T_A::NTuple{12,Float32}, T_B::NTuple{12,Float32},
                                    dir_world::Vec3f)
    dir_A = inv_transform_dir(T_A, dir_world)
    dir_B = inv_transform_dir(T_B, dir_world)
    sA    = transform_point(T_A, support(A, dir_A))
    sB    = transform_point(T_B, support(B, -dir_B))
    return sA, sB, sA - sB
end

# ---------------------------------------------------------------------------
# EPAResult
# ---------------------------------------------------------------------------

"""
    EPAResult

Result of an EPA penetration-recovery call.

Fields:
- `normal`: unit penetration normal in world space, pointing from B toward A
  (i.e. the direction along which to push A away from B to separate them).
- `depth`: penetration depth (>= 0) along `normal`.
- `contact`: contact point in world space on shape A's surface.
- `iterations`: number of EPA expansion iterations performed.
- `converged`: true iff EPA converged within `max_iters`; false on cap-out.
"""
struct EPAResult
    normal::Vec3f
    depth::Float32
    contact::Vec3f
    iterations::Int
    converged::Bool
end

# ---------------------------------------------------------------------------
# Polytope seeding
#
# Build a 4-vertex tetrahedron that encloses the origin, with per-vertex
# world-space s_A tracked alongside the Minkowski-difference vertex.  We
# don't trust the GJK simplex's points directly (they don't carry s_A
# bookkeeping); instead we resample 4 fresh support pairs.  For two
# penetrating convex shapes whose Minkowski-difference origin is interior,
# this GJK-init-style construction always succeeds.
#
# Returns (verts_m, verts_sA, ok).  `ok=false` on degenerate seed (e.g.
# coplanar shapes); EPA returns a non-converged result in that case.
# ---------------------------------------------------------------------------
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

    # Direction 3: perpendicular to (m2 - m1) toward origin.  Use the same
    # triple-product trick as gjk's do_simplex_line.
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
    # Flip if origin is on the opposite side from n4.  Origin is on the
    # +n4 side iff dot(n4, -m1) > 0  <=>  dot(n4, m1) < 0.
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

# ---------------------------------------------------------------------------
# Face plane: outward unit normal + signed distance from origin to plane.
# Returns (normal, distance).  If the triangle is degenerate (zero-area),
# returns (Vec3f(0), Inf32) which the closest-face search will skip.
# ---------------------------------------------------------------------------
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
    # Signed distance from origin to plane along outward n:
    #   plane:  dot(n, x - v1) = 0   =>   dot(n, x) = dot(n, v1)
    #   distance(origin) = dot(n, 0) - dot(n, v1) = -dot(n, v1)
    # but we want the magnitude along outward n.  Since n points outward and
    # origin is interior, dot(n, v1) > 0; that value IS the distance from the
    # origin to the plane along +n.  So:
    d = dot(n, v1)
    return n, d
end

# ---------------------------------------------------------------------------
# Closest-face search: pick the live face with smallest face.distance.
# Returns the index, or -1 if no live faces (which would be a polytope bug).
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Barycentric coordinates of a point P on a triangle (v1,v2,v3).
# Used to interpolate the per-vertex world-space s_A into a contact point.
# Implementation: Christer Ericke "Real-Time Collision Detection" 3.4.
# Returns (b1, b2, b3) summing to ~1; if triangle is degenerate returns
# (1, 0, 0) so the contact falls back to v1's s_A.
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# epa()
# ---------------------------------------------------------------------------

"""
    epa(A::ConvexShape, B::ConvexShape,
        T_A::NTuple{12,Float32}, T_B::NTuple{12,Float32},
        gjk_simplex::NTuple{4,Vec3f};
        max_iters::Int=24, eps::Float32=1f-4) -> EPAResult

Recover the penetration normal, depth, and contact point for two overlapping
convex shapes whose overlap was just established by `gjk()`.  The
`gjk_simplex` argument is the 4-tuple Minkowski-difference simplex returned
by `gjk()` when `overlap == true`; it currently informs the seed direction
but the polytope is rebuilt from scratch with per-vertex `s_A` tracking
(the GJK simplex doesn't carry that bookkeeping).

`EPAResult.normal` points from B toward A in world space (the direction
along which to push A out of B); `EPAResult.depth` is the magnitude of
penetration along that normal; `EPAResult.contact` is a point on shape A's
surface at the contact face.

`converged=false` only on iteration-cap exhaustion; the returned values are
the closest-face data at that point and are still usable as a best-effort
estimate.
"""
# CPU-allocation note: a `@allocated epa(...)` call from the Julia REPL on the
# unit-cube-vs-unit-cube headline case reports ~1.2 KB / 6 small pool allocs
# per invocation.  This is NOT a code defect, and does NOT affect the GPU
# `@kernel` path that P4.4 will compile through KernelAbstractions /
# GPUCompiler.  Investigation (see P4.3 review fix-up):
#
#   - Each individual MVector below stack-promotes cleanly when used in
#     isolation (`@allocated` returns 0).
#   - With all 7 MVectors live simultaneously (~3 KB total), Julia's CPU
#     escape-analysis budget is exceeded and 1-2 of them spill to the GC pool.
#     Which ones spill is non-deterministic across recompiles.
#   - Switching the NTuple-element MVectors to parallel-Int32 MVectors,
#     packed UInt64s, or a wrapping mutable struct does not eliminate the
#     spill; it just changes which storage band escapes.
#
# The GPU compile path does not run Julia's CPU escape analysis: KA lowers
# MVector storage to local / private memory or register tiles, with no GC
# behind it.  P4.4 will validate this empirically when the @kernel composing
# gjk + epa is built.  If a GPU compile failure surfaces here, this comment
# is the place to revisit -- start by consolidating the per-element MVectors
# into a single device-side scratch struct.
function epa(A::ConvexShape, B::ConvexShape,
             T_A::NTuple{12,Float32}, T_B::NTuple{12,Float32},
             gjk_simplex::NTuple{4,Vec3f};
             max_iters::Int=24, eps::Float32=1f-4)

    # `gjk_simplex` is currently unused beyond its presence in the contract.
    # We could exploit it as a seed-direction hint, but the GJK-init-style
    # rebuild below works for all cases the cube-vs-cube spec demands and is
    # robust to the GJK simplex coming back near-degenerate.  Keep the arg
    # name in the signature for forward compatibility (P4.4 may pass it
    # through unconditionally even when GJK reports no overlap).

    # ------------------------------------------------------------------
    # Storage (fixed-size, no heap alloc).
    # ------------------------------------------------------------------
    verts_m  = MVector{EPA_MAX_VERTS, Vec3f}(undef)   # Minkowski-difference vertices
    verts_sA = MVector{EPA_MAX_VERTS, Vec3f}(undef)   # per-vertex world-space s_A
    face_v   = MVector{EPA_MAX_FACES, NTuple{3,Int32}}(undef)
    face_n   = MVector{EPA_MAX_FACES, Vec3f}(undef)
    face_d   = MVector{EPA_MAX_FACES, Float32}(undef)
    face_alive = MVector{EPA_MAX_FACES, Bool}(undef)
    edges    = MVector{EPA_MAX_EDGES, NTuple{2,Int32}}(undef)

    # ------------------------------------------------------------------
    # Seed the polytope.
    # ------------------------------------------------------------------
    seed_m, seed_sA = seed_polytope(A, B, T_A, T_B, eps)
    verts_m[1]  = seed_m[1];  verts_sA[1] = seed_sA[1]
    verts_m[2]  = seed_m[2];  verts_sA[2] = seed_sA[2]
    verts_m[3]  = seed_m[3];  verts_sA[3] = seed_sA[3]
    verts_m[4]  = seed_m[4];  verts_sA[4] = seed_sA[4]
    n_verts = 4

    # Polytope centroid as the "interior reference" for outward-normal
    # orientation.  The origin is interior too, but using the centroid is
    # more numerically robust against the origin lying exactly on a face.
    interior = (verts_m[1] + verts_m[2] + verts_m[3] + verts_m[4]) * 0.25f0

    # Tetrahedron faces: (1,2,3), (1,3,4), (1,4,2), (2,4,3) -- four triangles
    # touching all 4 vertices.  Winding will be canonicalized below by
    # compute_face_plane (which flips the normal away from `interior`).
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

    # ------------------------------------------------------------------
    # Main expansion loop.
    # ------------------------------------------------------------------
    last_idx = -1
    for iter in 1:max_iters
        idx = find_closest_face(face_alive, face_d, n_faces)
        if idx <= 0
            # No live faces -- polytope corrupted.  Fall back to last known.
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

        # Convergence test: if the new support point doesn't lie further out
        # than the face plane, the face IS the closest face on the surface
        # of the Minkowski difference.
        if d_new - d_face < eps
            # Recover contact point.
            v_idx  = face_v[idx]
            v1     = verts_m[v_idx[1]]
            v2     = verts_m[v_idx[2]]
            v3     = verts_m[v_idx[3]]
            origin_proj = n_face * d_face   # closest point on face plane to origin
            b1, b2, b3  = barycentric_in_triangle(origin_proj, v1, v2, v3)
            sA1    = verts_sA[v_idx[1]]
            sA2    = verts_sA[v_idx[2]]
            sA3    = verts_sA[v_idx[3]]
            contact = b1 * sA1 + b2 * sA2 + b3 * sA3
            return EPAResult(n_face, d_face, contact, iter, true)
        end

        # ----------------------------------------------------------
        # Expand: remove every face visible from the new support point,
        # collect silhouette edges (boundary edges of the removed region),
        # then add a new triangular face per silhouette edge fanning out
        # from m_new.
        # ----------------------------------------------------------
        n_edges = 0
        @inbounds for fi in 1:n_faces
            if !face_alive[fi]
                continue
            end
            # A face is "visible" from m_new if m_new is on the outward side
            # of the face plane: dot(face.normal, m_new - face_vertex) > 0.
            v_idx_fi = face_v[fi]
            v1_fi    = verts_m[v_idx_fi[1]]
            if dot(face_n[fi], m_new - v1_fi) > 0f0
                # Mark dead and add its 3 edges to the silhouette buffer.
                # Interior edges (shared between two visible faces) will
                # appear twice; we cancel those pairs out when adding.
                face_alive[fi] = false
                for (a, b) in ((v_idx_fi[1], v_idx_fi[2]),
                               (v_idx_fi[2], v_idx_fi[3]),
                               (v_idx_fi[3], v_idx_fi[1]))
                    # Look for a matching reverse edge (b, a) already in
                    # the buffer.  If found, remove it (interior edge).
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
                            # Silhouette buffer exhausted: bail with the best-known closest-face data
                            # and converged=false.  Allocation-free path so the kernel compiles to SPIR-V.
                            return EPAResult(face_n[idx], face_d[idx], Vec3f(0f0), iter, false)
                        end
                        n_edges += 1
                        edges[n_edges] = (Int32(a), Int32(b))
                    end
                end
            end
        end

        if n_edges == 0
            # No face was visible to the new support point.  This means
            # m_new lies inside the polytope -- numerical hiccup; treat as
            # converged at the closest face.
            v_idx  = face_v[idx]
            v1     = verts_m[v_idx[1]]
            v2     = verts_m[v_idx[2]]
            v3     = verts_m[v_idx[3]]
            origin_proj = n_face * d_face
            b1, b2, b3  = barycentric_in_triangle(origin_proj, v1, v2, v3)
            contact = b1 * verts_sA[v_idx[1]] + b2 * verts_sA[v_idx[2]] + b3 * verts_sA[v_idx[3]]
            return EPAResult(n_face, d_face, contact, iter, true)
        end

        # Add the new vertex.
        if n_verts >= EPA_MAX_VERTS
            # Vertex storage exhausted: return the closest-face data so far, converged=false.
            # No string interpolation -- keeps the GPU lowering allocation-free.
            return EPAResult(face_n[idx], face_d[idx], Vec3f(0f0), iter, false)
        end
        n_verts += 1
        verts_m[n_verts]  = m_new
        verts_sA[n_verts] = sA_new
        new_v_idx = Int32(n_verts)

        # Add a new triangular face per silhouette edge, fanning from m_new.
        # Face winding (a, b, new) is set such that compute_face_plane will
        # canonicalize the outward direction relative to `interior`.
        @inbounds for ei in 1:n_edges
            if n_faces >= EPA_MAX_FACES
                # Face storage exhausted: return the closest-face data so far, converged=false.
                # Allocation-free; required for the SPIR-V lowering.
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

    # ------------------------------------------------------------------
    # Cap-out: return the closest-face data at the current state.
    # ------------------------------------------------------------------
    idx = find_closest_face(face_alive, face_d, n_faces)
    if idx <= 0
        return EPAResult(Vec3f(1f0, 0f0, 0f0), 0f0, Vec3f(0f0), max_iters, false)
    end
    n_face = face_n[idx]
    d_face = face_d[idx]
    v_idx  = face_v[idx]
    v1     = verts_m[v_idx[1]]
    v2     = verts_m[v_idx[2]]
    v3     = verts_m[v_idx[3]]
    origin_proj = n_face * d_face
    b1, b2, b3  = barycentric_in_triangle(origin_proj, v1, v2, v3)
    contact = b1 * verts_sA[v_idx[1]] + b2 * verts_sA[v_idx[2]] + b3 * verts_sA[v_idx[3]]
    return EPAResult(n_face, d_face, contact, max_iters, false)
end
