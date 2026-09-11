# test_rayquery_vs_cpu.jl
#
# Tier 3 GPU integration test: fire lava_ray_query_* intrinsics from a compute
# kernel against a single-triangle HWTLAS, compare per-ray t results to a CPU
# Moller-Trumbore reference.
#
# Triangle: (-1,-1,5), (1,-1,5), (0,1,5) -- in the +z half-space at z=5.
# Rays: 8x8 grid from z=0 firing along +z. Rays inside the triangle hit at t=5.
# Rays outside miss (output -1f0).

using Test, Lava, Raycore
using GeometryBasics: Point3f, Vec3f, GLTriangleFace
import GeometryBasics
import LinearAlgebra: I, cross, dot, inv
using StaticArrays: SMatrix

const Mat4f = SMatrix{4, 4, Float32, 16}

@testset "Ray Query - GPU MWE: rayQuery vs CPU Moller-Trumbore" begin

    ctx = Mantle.vk_context()
    if !ctx.ray_query_available
        @warn "Skipping rayQuery test: VK_KHR_ray_query not available on this device"
        @test_skip true
        return
    end

    backend = Mantle.LavaBackend()
    bq = backend.dispatch_bq

    # ----------------------------------------------------------------
    # Build a single-triangle HWTLAS. The triangle is at z=5 so rays fired
    # along +z from z=0 hit at t=5 when they land inside the triangle.
    # ----------------------------------------------------------------
    tri_v0 = Point3f(-1f0, -1f0, 5f0)
    tri_v1 = Point3f( 1f0, -1f0, 5f0)
    tri_v2 = Point3f( 0f0,  1f0, 5f0)

    verts = [tri_v0, tri_v1, tri_v2]
    faces = [GLTriangleFace(1, 2, 3)]
    mesh = GeometryBasics.normal_mesh(GeometryBasics.Mesh(verts, faces))

    tlas = Mantle.VulkanTLAS(backend)
    push!(tlas, mesh, Mat4f(I))
    Raycore.sync!(tlas)

    # ----------------------------------------------------------------
    # CPU Moller-Trumbore reference.  Returns t >= 0 on hit, -1 on miss.
    # ----------------------------------------------------------------
    function ray_tri_t(o::Point3f, d::Vec3f, v0::Point3f, v1::Point3f, v2::Point3f)
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

    # 8x8 grid of ray origins on the xy-plane at z=0, spacing 0.25 apart.
    # Grid center near origin so some rays hit, some miss.
    n = 64
    origins = Vector{Point3f}(undef, n)
    for idx in 0:(n-1)
        ix = idx % 8
        iy = div(idx, 8)
        x = (ix - 3.5f0) * 0.25f0
        y = (iy - 3.5f0) * 0.25f0
        origins[idx + 1] = Point3f(x, y, 0f0)
    end

    ray_dir = Vec3f(0f0, 0f0, 1f0)
    ref = Float32[
        let t = ray_tri_t(origins[i], ray_dir, tri_v0, tri_v1, tri_v2)
            t < 0f0 ? -1f0 : t
        end
        for i in 1:n
    ]

    # ----------------------------------------------------------------
    # GPU kernel: one thread per ray. Writes hit t or -1 to output buffer.
    # Uses lava_ray_query_* intrinsics.
    # ----------------------------------------------------------------
    function rq_kernel(out::LavaDeviceArray{Float32, 1},
                       origins::LavaDeviceArray{Point3f, 1},
                       gid::UInt32)
        i = Int(gid)
        o = origins[i]
        ray = Ray(o=o, d=Vec3f(0f0, 0f0, 1f0), t_min=0f0, t_max=1f4)
        Lava.lava_ray_query_init(ray)
        while Lava.lava_ray_query_proceed()
            Lava.lava_ray_query_confirm()
        end
        kind = Lava.lava_ray_query_get_type(true)   # committed intersection type
        thit = (kind == UInt32(1)) ? Lava.lava_ray_query_get_t(true) : -1f0
        out[i] = thit
        return nothing
    end

    out_g = LavaArray(fill(-1f0, n))
    origins_g = LavaArray(origins)

    # Launch: one workgroup of 64 threads, each thread handles one ray.
    # The global invocation id (1-based) is passed explicitly as UInt32 via
    # lava_global_invocation_id_x() from within a workgroup kernel. But since
    # lava_launch! doesn't wrap in KA, we pass the thread index as a 1-D
    # ndrange and each invocation loads its own gid from lava_global_invocation_id_x.
    #
    # Rewrite as a zero-arg-index kernel: each thread gets its own ID from the
    # GPU built-in and does NOT receive an explicit gid argument.
    function rq_kernel_noarg(out::LavaDeviceArray{Float32, 1},
                             origins::LavaDeviceArray{Point3f, 1})
        i = Int(Lava.lava_global_invocation_id_x()) + 1   # 1-based
        o = origins[i]
        ray = Ray(o=o, d=Vec3f(0f0, 0f0, 1f0), t_min=0f0, t_max=1f4)
        Lava.lava_ray_query_init(ray)
        while Lava.lava_ray_query_proceed()
            Lava.lava_ray_query_confirm()
        end
        kind = Lava.lava_ray_query_get_type(true)
        thit = (kind == UInt32(1)) ? Lava.lava_ray_query_get_t(true) : -1f0
        out[i] = thit
        return nothing
    end

    Mantle.lava_launch!(bq, rq_kernel_noarg, out_g, origins_g;
                      ndrange=n, workgroup_size=(64, 1, 1), tlas=tlas)
    Mantle.vk_flush!(bq)

    gpu = Array(out_g)

    # ----------------------------------------------------------------
    # Compare GPU results to CPU reference.
    # ----------------------------------------------------------------
    n_hits = count(>=(0f0), ref)
    n_misses = count(<(0f0), ref)
    @test n_hits > 0
    @test n_misses > 0

    for i in 1:n
        if ref[i] < 0f0
            @test gpu[i] < 0f0
        else
            @test isapprox(gpu[i], ref[i]; atol=1f-3)
        end
    end

    println("rayQuery GPU test: $n_hits hits, $n_misses misses out of $n rays. " *
            "Max hit error: $(maximum(abs(gpu[i] - ref[i]) for i in 1:n if ref[i] >= 0f0; init=0f0))")

end
