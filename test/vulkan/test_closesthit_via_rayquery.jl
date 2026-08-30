# test_closesthit_via_rayquery.jl
#
# Step 1 of the inline-ray-query migration: prove that a kernel which calls
# `Raycore.closest_hit(accel, ray)` (the same polymorphic API the SW path
# uses) can compile + run with a HW accel that lowers to inline ray queries.
#
# Today: Raycore.closest_hit(::StaticTLAS, ray) walks the SW BVH inline.
# Goal:  Raycore.closest_hit(::RayQueryAccel, ray) does the same lookup via
#        VK_KHR_ray_query intrinsics (lava_ray_query_init/proceed/get_*).
# This file builds the second method on a minimal new accel struct and
# verifies a single-triangle scene — the precursor to migrating
# `AdaptedAccel` itself.

using Test, Lava, Raycore
using GeometryBasics: Point3f, Vec3f, Point2f, GLTriangleFace
import GeometryBasics
import LinearAlgebra: I, cross, dot, inv
using StaticArrays: SVector, SMatrix

const Mat4f = SMatrix{4, 4, Float32, 16}

# ----------------------------------------------------------------------------
# RayQueryAccel: kernel-side adapted form for HW ray-query traversal.
#
# Mirrors `PrecomputedHitsAccel`'s field shape (`triangles`, `offsets`,
# `empty`) but does the actual trace inline via `lava_ray_query_*` instead
# of reading pre-computed results.
# ----------------------------------------------------------------------------
struct RayQueryAccel{T, O, Tri} <: Raycore.AbstractAdaptedAccel
    triangles::T   # flat array of all BLAS triangles, concatenated
    offsets::O     # per-instance offset into `triangles`, indexed by gl_InstanceID+1
    empty::Tri     # sentinel returned on miss
end

# Adapt.jl integration: walk into the array fields so the kernel sees
# LavaDeviceArray instead of LavaArray.  `empty` is bitstype, copies as-is.
import Adapt
Adapt.adapt_structure(to, a::RayQueryAccel) = RayQueryAccel(
    Adapt.adapt(to, a.triangles),
    Adapt.adapt(to, a.offsets),
    a.empty,
)

# Internal helper: run an initialized ray query to completion and pack the
# result into the same tuple shape `Raycore.closest_hit` / `any_hit` use.
@inline function _rq_collect(accel::RayQueryAccel)
    while Lava.lava_ray_query_proceed()
        # Opaque triangles auto-commit; nothing to do here.
    end
    kind = Lava.lava_ray_query_get_type(true)  # committed
    if kind != UInt32(1)  # not RayQueryCommittedIntersectionTriangleKHR
        return (false, accel.empty, 0f0, SVector{3,Float32}(1f0, 0f0, 0f0), UInt32(0))
    end
    t = Lava.lava_ray_query_get_t(true)
    inst_id = Lava.lava_ray_query_get_instance_id(true)
    inst_custom_idx = Lava.lava_ray_query_get_instance_custom_index(true)
    prim_idx = Lava.lava_ray_query_get_primitive_index(true)
    bx, by = Lava.lava_ray_query_get_barycentrics(true)

    @inbounds tri_idx = Int(accel.offsets[inst_id + UInt32(1)]) + Int(prim_idx) + 1
    @inbounds tri = accel.triangles[tri_idx]

    bary = SVector{3,Float32}(1f0 - bx - by, bx, by)
    return (true, tri, t, bary, inst_custom_idx)
end

# Closest-hit via inline ray query.  Matches the SW return tuple:
#   (hit::Bool, primitive, t_hit::Float32, bary::SVector{3,Float32}, inst_custom_idx::UInt32)
@inline function Raycore.closest_hit(accel::RayQueryAccel, ray::Raycore.AbstractRay)
    o = ray.o; d = ray.d
    Lava.lava_ray_query_init(UInt32(0), UInt32(0xFF),  # flags=0 (Opaque default)
        Float32(o[1]), Float32(o[2]), Float32(o[3]), Float32(ray.t_min),
        Float32(d[1]), Float32(d[2]), Float32(d[3]), Float32(ray.t_max))
    return _rq_collect(accel)
end

# Any-hit via inline ray query.  Same return shape as closest_hit; uses the
# TerminateOnFirstHitKHR flag so the traversal exits at the first committed
# triangle hit (same semantics as Raycore.any_hit on the SW BVH).
@inline function Raycore.any_hit(accel::RayQueryAccel, ray::Raycore.AbstractRay)
    o = ray.o; d = ray.d
    # SPIR-V RayFlagsTerminateOnFirstHitKHR = 4
    Lava.lava_ray_query_init(UInt32(4), UInt32(0xFF),
        Float32(o[1]), Float32(o[2]), Float32(o[3]), Float32(ray.t_min),
        Float32(d[1]), Float32(d[2]), Float32(d[3]), Float32(ray.t_max))
    return _rq_collect(accel)
end

@testset "Raycore.closest_hit via inline ray query (Step 1)" begin
    ctx = Mantle.vk_context()
    if !ctx.ray_query_available
        @warn "Skipping: VK_KHR_ray_query not available"
        @test_skip true
        return
    end

    backend = Mantle.LavaBackend()
    bq = backend.bq

    # Single-triangle scene at z=5.  Rays from z=0 firing +z hit at t=5.
    tri_v0 = Point3f(-1f0, -1f0, 5f0)
    tri_v1 = Point3f( 1f0, -1f0, 5f0)
    tri_v2 = Point3f( 0f0,  1f0, 5f0)
    verts = [tri_v0, tri_v1, tri_v2]
    faces = [GLTriangleFace(1, 2, 3)]
    mesh = GeometryBasics.normal_mesh(GeometryBasics.Mesh(verts, faces))

    hwtlas = Mantle.VulkanTLAS(backend)
    push!(hwtlas, mesh, Mat4f(I))
    Raycore.sync!(hwtlas)
    @assert hwtlas.tri_gpu !== nothing && hwtlas.off_gpu !== nothing

    # Wrap the synced GPU buffers in our adapted-form struct.
    Tri = eltype(eltype(hwtlas.blas_triangles))
    empty_tri = Raycore.empty_triangle(Tri)
    accel_cpu = RayQueryAccel(hwtlas.tri_gpu, hwtlas.off_gpu, empty_tri)

    # CPU Möller-Trumbore reference
    function ray_tri_t(o, d, v0, v1, v2)
        e1 = v1 - v0
        e2 = v2 - v0
        h = cross(d, e2)
        a = dot(e1, h)
        abs(a) < 1f-7 && return -1f0
        f = inv(a)
        s = o - v0
        u = f * dot(s, h)
        (u < 0f0 || u > 1f0) && return -1f0
        q = cross(s, e1)
        v = f * dot(d, q)
        (v < 0f0 || u + v > 1f0) && return -1f0
        t = f * dot(e2, q)
        return t < 0f0 ? -1f0 : t
    end

    n = 64
    origins = Vector{Point3f}(undef, n)
    for idx in 0:(n-1)
        ix = idx % 8
        iy = div(idx, 8)
        x = (ix - 3.5f0) * 0.25f0
        y = (iy - 3.5f0) * 0.25f0
        origins[idx + 1] = Point3f(x, y, 0f0)
    end
    ref = Float32[
        let t = ray_tri_t(origins[i], Vec3f(0,0,1), tri_v0, tri_v1, tri_v2)
            t < 0f0 ? -1f0 : t
        end
        for i in 1:n
    ]

    # Kernel: one thread per ray, calls polymorphic closest_hit.
    function closesthit_kernel(t_out::Lava.LavaDeviceArray{Float32, 1},
                                prim_out::Lava.LavaDeviceArray{UInt32, 1},
                                origins::Lava.LavaDeviceArray{Point3f, 1},
                                accel)
        i = Int(Lava.lava_global_invocation_id_x()) + 1
        @inbounds o = origins[i]
        ray = Raycore.Ray(o=o, d=Vec3f(0f0, 0f0, 1f0), t_min=0f0, t_max=1f4)
        hit, _prim, t, _bary, _icx = Raycore.closest_hit(accel, ray)
        @inbounds t_out[i] = hit ? t : -1f0
        @inbounds prim_out[i] = hit ? UInt32(1) : UInt32(0)
        return nothing
    end

    t_out = Mantle.LavaArray(fill(-2f0, n))
    prim_out = Mantle.LavaArray(fill(UInt32(99), n))
    origins_g = Mantle.LavaArray(origins)

    Mantle.lava_launch!(bq, closesthit_kernel, t_out, prim_out, origins_g, accel_cpu;
                      ndrange=n, workgroup_size=(64, 1, 1), tlas=hwtlas)
    Mantle.vk_flush!(bq)

    gpu_t = Array(t_out)
    gpu_prim = Array(prim_out)

    n_hits = count(>=(0f0), ref)
    n_misses = count(<(0f0), ref)
    @test n_hits > 0
    @test n_misses > 0

    correct_t = 0
    correct_hit_flag = 0
    for i in 1:n
        if ref[i] < 0f0
            gpu_t[i] < 0f0 && (correct_t += 1)
            gpu_prim[i] == UInt32(0) && (correct_hit_flag += 1)
        else
            isapprox(gpu_t[i], ref[i]; atol=1f-3) && (correct_t += 1)
            gpu_prim[i] == UInt32(1) && (correct_hit_flag += 1)
        end
    end
    @test correct_t == n
    @test correct_hit_flag == n
    println("closest_hit via ray query: $n_hits hits, $n_misses misses, $correct_t/$n correct t, $correct_hit_flag/$n correct hit flag")
end

# ============================================================================
# any_hit: occlusion test — does the ray hit anything within [t_min, t_max]?
# Two-triangle scene: a near triangle at z=2 and a far one at z=5.
# Rays pointing +z that hit the near triangle should report any-hit at t=2,
# closest_hit also at t=2 (the near one).  Rays missing both report miss.
# any_hit must also work when the ray's t_max excludes the only hit.
# ============================================================================
@testset "Raycore.any_hit via inline ray query — occlusion" begin
    ctx = Mantle.vk_context()
    if !ctx.ray_query_available
        @warn "Skipping: VK_KHR_ray_query not available"
        @test_skip true
        return
    end

    backend = Mantle.LavaBackend()
    bq = backend.bq

    near_v = [Point3f(-1, -1, 2), Point3f(1, -1, 2), Point3f(0, 1, 2)]
    far_v  = [Point3f(-1, -1, 5), Point3f(1, -1, 5), Point3f(0, 1, 5)]
    faces  = [GLTriangleFace(1, 2, 3)]
    near_mesh = GeometryBasics.normal_mesh(GeometryBasics.Mesh(near_v, faces))
    far_mesh  = GeometryBasics.normal_mesh(GeometryBasics.Mesh(far_v,  faces))

    hwtlas = Mantle.VulkanTLAS(backend)
    push!(hwtlas, near_mesh, Mat4f(I))
    push!(hwtlas, far_mesh,  Mat4f(I))
    Raycore.sync!(hwtlas)
    Tri = eltype(eltype(hwtlas.blas_triangles))
    accel_cpu = RayQueryAccel(hwtlas.tri_gpu, hwtlas.off_gpu, Raycore.empty_triangle(Tri))

    # 8x8 grid; tmin=0, tmax=10 covers both triangles.
    n = 64
    origins = [Point3f((i%8 - 3.5f0) * 0.25f0, (i÷8 - 3.5f0) * 0.25f0, 0f0) for i in 0:n-1]

    function any_hit_kernel(any_t_out::Lava.LavaDeviceArray{Float32, 1},
                             closest_t_out::Lava.LavaDeviceArray{Float32, 1},
                             origins::Lava.LavaDeviceArray{Point3f, 1},
                             tmax::Float32, accel)
        i = Int(Lava.lava_global_invocation_id_x()) + 1
        @inbounds o = origins[i]
        ray = Raycore.Ray(o=o, d=Vec3f(0f0, 0f0, 1f0), t_min=0f0, t_max=tmax)
        any_hit_b, _ah_prim, any_t, _ah_b, _ah_i = Raycore.any_hit(accel, ray)
        cls_hit, _cl_prim, cls_t, _cl_b, _cl_i = Raycore.closest_hit(accel, ray)
        @inbounds any_t_out[i] = any_hit_b ? any_t : -1f0
        @inbounds closest_t_out[i] = cls_hit ? cls_t : -1f0
        return nothing
    end

    # Case 1: tmax=10 — both triangles in range.  any_hit returns *some* hit
    # (impl-defined which; not necessarily nearest).  closest_hit returns t=2.
    any_t_g = Mantle.LavaArray(fill(-2f0, n))
    cls_t_g = Mantle.LavaArray(fill(-2f0, n))
    origins_g = Mantle.LavaArray(origins)
    Mantle.lava_launch!(bq, any_hit_kernel, any_t_g, cls_t_g, origins_g, 10f0, accel_cpu;
                      ndrange=n, workgroup_size=(64, 1, 1), tlas=hwtlas)
    Mantle.vk_flush!(bq)
    any_full   = Array(any_t_g)
    cls_full   = Array(cls_t_g)

    # Case 2: tmax=3 — only near triangle in range.  any_hit must report t=2,
    # closest_hit must also report t=2.  Rays that miss the near tri report -1.
    any_t_g2 = Mantle.LavaArray(fill(-2f0, n))
    cls_t_g2 = Mantle.LavaArray(fill(-2f0, n))
    Mantle.lava_launch!(bq, any_hit_kernel, any_t_g2, cls_t_g2, origins_g, 3f0, accel_cpu;
                      ndrange=n, workgroup_size=(64, 1, 1), tlas=hwtlas)
    Mantle.vk_flush!(bq)
    any_near = Array(any_t_g2)
    cls_near = Array(cls_t_g2)

    # CPU reference: a ray hits the (axis-aligned) near tri iff it hits the
    # far tri (same xy footprint).  Build a hit mask.
    function in_tri(o)
        # Triangle (-1,-1) (1,-1) (0,1) — barycentric check on xy.
        x, y = o[1], o[2]
        v0 = (-1f0, -1f0); v1 = (1f0, -1f0); v2 = (0f0, 1f0)
        denom = (v1[2] - v2[2])*(v0[1] - v2[1]) + (v2[1] - v1[1])*(v0[2] - v2[2])
        a = ((v1[2] - v2[2])*(x - v2[1]) + (v2[1] - v1[1])*(y - v2[2])) / denom
        b = ((v2[2] - v0[2])*(x - v2[1]) + (v0[1] - v2[1])*(y - v2[2])) / denom
        c = 1f0 - a - b
        return (a >= 0) && (b >= 0) && (c >= 0)
    end
    expected_hit = [in_tri(o) for o in origins]
    n_hit = count(expected_hit)
    @test n_hit > 0
    @test n_hit < n

    # Tmax=10: every hit ray gets a hit from any_hit; closest_hit returns t=2.
    n_correct_any_full     = count(i -> expected_hit[i] ? any_full[i] >= 0f0 : any_full[i] < 0f0,    1:n)
    n_correct_cls_full     = count(i -> expected_hit[i] ? isapprox(cls_full[i], 2f0; atol=1f-3) : cls_full[i] < 0f0, 1:n)
    @test n_correct_any_full == n
    @test n_correct_cls_full == n

    # Tmax=3: hit rays still hit the near triangle at t=2; misses stay -1.
    n_correct_any_near     = count(i -> expected_hit[i] ? isapprox(any_near[i], 2f0; atol=1f-3) : any_near[i] < 0f0, 1:n)
    n_correct_cls_near     = count(i -> expected_hit[i] ? isapprox(cls_near[i], 2f0; atol=1f-3) : cls_near[i] < 0f0, 1:n)
    @test n_correct_any_near == n
    @test n_correct_cls_near == n

    println("any_hit via ray query: $n_hit/$n rays hit; tmax=10 → any+closest both correct; tmax=3 → near-tri-only behaviour confirmed")
end

# ============================================================================
# Polymorphism: same kernel, swap accel between SW BVH and HW ray query.
# This is the design contract we want to lock down — any kernel that uses
# `Raycore.closest_hit(accel, ray)` should produce identical results
# regardless of accel backend.  When this passes, Hikari can swap
# `vp_trace_rays!(::AdaptedAccel, …)` for the unified SW kernel.
# ============================================================================
@testset "closest_hit polymorphism — SW BVH vs HW ray query" begin
    ctx = Mantle.vk_context()
    if !ctx.ray_query_available
        @warn "Skipping: VK_KHR_ray_query not available"
        @test_skip true
        return
    end

    backend = Mantle.LavaBackend()
    bq = backend.bq

    # Build a small two-triangle scene (one mesh with two faces).
    verts = [
        Point3f(-1, -1, 5), Point3f(1, -1, 5), Point3f(0, 1, 5),
        Point3f(-1, -1, 8), Point3f(1, -1, 8), Point3f(0, 1, 8),
    ]
    faces = [GLTriangleFace(1, 2, 3), GLTriangleFace(4, 5, 6)]
    mesh = GeometryBasics.normal_mesh(GeometryBasics.Mesh(verts, faces))

    # HW path
    hwtlas = Mantle.VulkanTLAS(backend)
    push!(hwtlas, mesh, Mat4f(I))
    Raycore.sync!(hwtlas)
    Tri_hw = eltype(eltype(hwtlas.blas_triangles))
    accel_hw = RayQueryAccel(hwtlas.tri_gpu, hwtlas.off_gpu, Raycore.empty_triangle(Tri_hw))

    # SW path: build a Raycore HWTLAS over the same mesh on the same backend.
    sw_tlas = Raycore.TLAS(backend)
    push!(sw_tlas, mesh, Mat4f(I))
    Raycore.sync!(sw_tlas)
    accel_sw = Adapt.adapt(backend, sw_tlas)

    # Common kernel — accel is whatever the caller passes.
    function poly_kernel(t_out::Lava.LavaDeviceArray{Float32, 1},
                         hit_out::Lava.LavaDeviceArray{UInt32, 1},
                         origins::Lava.LavaDeviceArray{Point3f, 1},
                         accel)
        i = Int(Lava.lava_global_invocation_id_x()) + 1
        @inbounds o = origins[i]
        ray = Raycore.Ray(o=o, d=Vec3f(0f0, 0f0, 1f0), t_min=0f0, t_max=1f4)
        hit, _prim, t, _bary, _icx = Raycore.closest_hit(accel, ray)
        @inbounds t_out[i] = hit ? t : -1f0
        @inbounds hit_out[i] = hit ? UInt32(1) : UInt32(0)
        return nothing
    end

    n = 64
    origins = [Point3f((i%8 - 3.5f0) * 0.25f0, (i÷8 - 3.5f0) * 0.25f0, 0f0) for i in 0:n-1]
    origins_g = Mantle.LavaArray(origins)

    # Run on HW
    t_hw = Mantle.LavaArray(fill(-2f0, n))
    h_hw = Mantle.LavaArray(fill(UInt32(99), n))
    Mantle.lava_launch!(bq, poly_kernel, t_hw, h_hw, origins_g, accel_hw;
                      ndrange=n, workgroup_size=(64, 1, 1), tlas=hwtlas)
    Mantle.vk_flush!(bq)
    t_hw_arr = Array(t_hw); h_hw_arr = Array(h_hw)

    # Run on SW (no tlas kwarg; SW kernel doesn't need a HWTLAS descriptor).
    t_sw = Mantle.LavaArray(fill(-2f0, n))
    h_sw = Mantle.LavaArray(fill(UInt32(99), n))
    Mantle.lava_launch!(bq, poly_kernel, t_sw, h_sw, origins_g, accel_sw;
                      ndrange=n, workgroup_size=(64, 1, 1))
    Mantle.vk_flush!(bq)
    t_sw_arr = Array(t_sw); h_sw_arr = Array(h_sw)

    # Same kernel, same scene — results must match.
    n_hit_match = count(i -> h_hw_arr[i] == h_sw_arr[i], 1:n)
    n_t_match   = count(i -> isapprox(t_hw_arr[i], t_sw_arr[i]; atol=1f-3), 1:n)
    @test n_hit_match == n
    @test n_t_match == n
    println("polymorphism: HW and SW paths agree on $n_hit_match/$n hit flags, $n_t_match/$n t values")
end
