# test_hwadapted_via_rayquery.jl
#
# Step 2: verify the refactored `AdaptedAccel` (now carrying triangles/
# offsets/empty alongside the hwtlas reference) works as a polymorphic accel
# argument with `Raycore.closest_hit` / `Raycore.any_hit` lowered via
# inline ray queries.
#
# This is the production type — once this passes, Hikari's volpath HW
# trace_rays! 4-step dance can be deleted in favour of the unified
# kernel that calls `closest_hit(accel, ray)` polymorphically.

using Test, Lava, Raycore, Adapt
using GeometryBasics: Point3f, Vec3f, GLTriangleFace
import GeometryBasics
import LinearAlgebra: I, cross, dot, inv
using StaticArrays: SVector, SMatrix

const Mat4f = SMatrix{4, 4, Float32, 16}

@testset "AdaptedAccel: closest_hit + any_hit via inline ray query" begin
    ctx = Mantle.vk_context()
    if !ctx.ray_query_available
        @warn "Skipping: VK_KHR_ray_query not available"
        @test_skip true
        return
    end

    backend = Mantle.LavaBackend()
    bq = backend.dispatch_bq

    # Two triangles at z=2 and z=5 (same xy footprint).  Same scene as the
    # any_hit test in test_closesthit_via_rayquery.jl, but exercising the
    # production AdaptedAccel directly rather than the surrogate struct.
    near_v = [Point3f(-1, -1, 2), Point3f(1, -1, 2), Point3f(0, 1, 2)]
    far_v  = [Point3f(-1, -1, 5), Point3f(1, -1, 5), Point3f(0, 1, 5)]
    faces  = [GLTriangleFace(1, 2, 3)]
    near_mesh = GeometryBasics.normal_mesh(GeometryBasics.Mesh(near_v, faces))
    far_mesh  = GeometryBasics.normal_mesh(GeometryBasics.Mesh(far_v,  faces))

    hwtlas = Mantle.VulkanTLAS(backend)
    push!(hwtlas, near_mesh, Mat4f(I))
    push!(hwtlas, far_mesh,  Mat4f(I))
    Raycore.sync!(hwtlas)

    # Adapt(backend, hwtlas) returns the CPU-form AdaptedAccel — keeps the
    # live VulkanTLAS reference so Hikari's CPU dispatch code can find descriptor /
    # sync state.  Stripping happens at kernel-arg time via LavaAdaptor.
    accel = Adapt.adapt(backend, hwtlas)
    @test accel isa Mantle.AdaptedAccel
    @test accel.hwtlas === hwtlas
    @test accel.triangles === hwtlas.tri_gpu
    @test accel.offsets   === hwtlas.off_gpu

    # 8x8 grid of rays from z=0, +z direction.
    n = 64
    origins = [Point3f((i%8 - 3.5f0) * 0.25f0, (i÷8 - 3.5f0) * 0.25f0, 0f0) for i in 0:n-1]

    # CPU reference: which rays land inside the (axis-aligned) triangle.
    function in_tri(o)
        x, y = o[1], o[2]
        v0 = (-1f0, -1f0); v1 = (1f0, -1f0); v2 = (0f0, 1f0)
        denom = (v1[2]-v2[2])*(v0[1]-v2[1]) + (v2[1]-v1[1])*(v0[2]-v2[2])
        a = ((v1[2]-v2[2])*(x-v2[1]) + (v2[1]-v1[1])*(y-v2[2])) / denom
        b = ((v2[2]-v0[2])*(x-v2[1]) + (v0[1]-v2[1])*(y-v2[2])) / denom
        c = 1f0 - a - b
        return (a >= 0) && (b >= 0) && (c >= 0)
    end
    expected_hit = [in_tri(o) for o in origins]
    n_hit = count(expected_hit)
    @test n_hit > 0
    @test n_hit < n

    # Polymorphic kernel — accel can be AdaptedAccel (HW) or StaticTLAS (SW).
    function unified_kernel(t_out::Lava.LavaDeviceArray{Float32, 1},
                            any_t_out::Lava.LavaDeviceArray{Float32, 1},
                            origins::Lava.LavaDeviceArray{Point3f, 1},
                            tmax::Float32, accel)
        i = Int(Lava.lava_global_invocation_id_x()) + 1
        @inbounds o = origins[i]
        ray = Raycore.Ray(o=o, d=Vec3f(0f0, 0f0, 1f0), t_min=0f0, t_max=tmax)
        cls_hit, _cp, cls_t, _cb, _ci = Raycore.closest_hit(accel, ray)
        any_b,   _ap, any_t, _ab, _ai = Raycore.any_hit(accel,    ray)
        @inbounds t_out[i]     = cls_hit ? cls_t : -1f0
        @inbounds any_t_out[i] = any_b   ? any_t : -1f0
        return nothing
    end

    origins_g = Mantle.LavaArray(origins)

    # Case 1 — tmax=10: both triangles in range.
    #   closest_hit must return t=2 for hit rays.
    #   any_hit  is allowed to return any of t∈{2,5} for hit rays
    #            (impl-defined commit order).
    cls_g = Mantle.LavaArray(fill(-2f0, n))
    any_g = Mantle.LavaArray(fill(-2f0, n))
    Mantle.lava_launch!(bq, unified_kernel, cls_g, any_g, origins_g, 10f0, accel;
                      ndrange=n, workgroup_size=(64, 1, 1), tlas=hwtlas)
    Mantle.vk_flush!(bq)
    cls_full = Array(cls_g); any_full = Array(any_g)

    @test all(i -> expected_hit[i] ?
                       isapprox(cls_full[i], 2f0; atol=1f-3) :
                       cls_full[i] < 0f0,
              1:n)
    @test all(i -> expected_hit[i] ?
                       (any_full[i] >= 0f0) :  # any of {2, 5} is fine
                       any_full[i] < 0f0,
              1:n)

    # Case 2 — tmax=3: only near triangle in range.
    cls_g2 = Mantle.LavaArray(fill(-2f0, n))
    any_g2 = Mantle.LavaArray(fill(-2f0, n))
    Mantle.lava_launch!(bq, unified_kernel, cls_g2, any_g2, origins_g, 3f0, accel;
                      ndrange=n, workgroup_size=(64, 1, 1), tlas=hwtlas)
    Mantle.vk_flush!(bq)
    cls_near = Array(cls_g2); any_near = Array(any_g2)

    @test all(i -> expected_hit[i] ?
                       isapprox(cls_near[i], 2f0; atol=1f-3) :
                       cls_near[i] < 0f0,
              1:n)
    @test all(i -> expected_hit[i] ?
                       isapprox(any_near[i], 2f0; atol=1f-3) :
                       any_near[i] < 0f0,
              1:n)

    println("AdaptedAccel: $n_hit hits — closest_hit & any_hit verified at tmax=10 and tmax=3")
end

# Polymorphism repeated against the production AdaptedAccel:
# same kernel, swap accel between SW StaticTLAS and HW AdaptedAccel.
@testset "AdaptedAccel polymorphism — SW BVH vs HW ray query" begin
    ctx = Mantle.vk_context()
    if !ctx.ray_query_available
        @warn "Skipping: VK_KHR_ray_query not available"
        @test_skip true
        return
    end

    backend = Mantle.LavaBackend()
    bq = backend.dispatch_bq

    verts = [
        Point3f(-1, -1, 5), Point3f(1, -1, 5), Point3f(0, 1, 5),
        Point3f(-1, -1, 8), Point3f(1, -1, 8), Point3f(0, 1, 8),
    ]
    faces = [GLTriangleFace(1, 2, 3), GLTriangleFace(4, 5, 6)]
    mesh = GeometryBasics.normal_mesh(GeometryBasics.Mesh(verts, faces))

    hwtlas = Mantle.VulkanTLAS(backend)
    push!(hwtlas, mesh, Mat4f(I))
    Raycore.sync!(hwtlas)
    accel_hw = Adapt.adapt(backend, hwtlas)

    sw_tlas = Raycore.TLAS(backend)
    push!(sw_tlas, mesh, Mat4f(I))
    Raycore.sync!(sw_tlas)
    accel_sw = Adapt.adapt(backend, sw_tlas)

    function poly_kernel(t_out::Lava.LavaDeviceArray{Float32, 1},
                         hit_out::Lava.LavaDeviceArray{UInt32, 1},
                         origins::Lava.LavaDeviceArray{Point3f, 1},
                         accel)
        i = Int(Lava.lava_global_invocation_id_x()) + 1
        @inbounds o = origins[i]
        ray = Raycore.Ray(o=o, d=Vec3f(0f0, 0f0, 1f0), t_min=0f0, t_max=1f4)
        hit, _p, t, _b, _i = Raycore.closest_hit(accel, ray)
        @inbounds t_out[i]   = hit ? t : -1f0
        @inbounds hit_out[i] = hit ? UInt32(1) : UInt32(0)
        return nothing
    end

    n = 64
    origins = [Point3f((i%8 - 3.5f0) * 0.25f0, (i÷8 - 3.5f0) * 0.25f0, 0f0) for i in 0:n-1]
    origins_g = Mantle.LavaArray(origins)

    t_hw = Mantle.LavaArray(fill(-2f0, n))
    h_hw = Mantle.LavaArray(fill(UInt32(99), n))
    Mantle.lava_launch!(bq, poly_kernel, t_hw, h_hw, origins_g, accel_hw;
                      ndrange=n, workgroup_size=(64, 1, 1), tlas=hwtlas)
    Mantle.vk_flush!(bq)

    t_sw = Mantle.LavaArray(fill(-2f0, n))
    h_sw = Mantle.LavaArray(fill(UInt32(99), n))
    Mantle.lava_launch!(bq, poly_kernel, t_sw, h_sw, origins_g, accel_sw;
                      ndrange=n, workgroup_size=(64, 1, 1))
    Mantle.vk_flush!(bq)

    t_hw_a = Array(t_hw); h_hw_a = Array(h_hw)
    t_sw_a = Array(t_sw); h_sw_a = Array(h_sw)

    @test count(i -> h_hw_a[i] == h_sw_a[i], 1:n) == n
    @test count(i -> isapprox(t_hw_a[i], t_sw_a[i]; atol=1f-3), 1:n) == n
    println("AdaptedAccel polymorphism: HW and SW agree on all $n rays")
end
