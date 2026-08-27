# test_aabb_blas_overlap.jl
#
# Tier 3 GPU integration test: fire zero-length rayQuery against a procedural-AABB
# BLAS and count how many AABBs contain each query point.  Compare to a CPU
# brute-force reference.
#
# Setup:
#   1000 random AABBs of side 1.0 in [-10, 10]^3.
#   64 random query points.
#
# GPU kernel (Option A): for each AABB candidate, reads the primitive index and
# checks point-in-AABB against the original AABB buffer.  This guards against
# drivers that report bounding-box candidates whose AABB does not strictly
# contain the query point (degenerate zero-length ray edge cases).

using Test, Lava, Raycore, Random
using GeometryBasics: Point3f, Vec3f
import LinearAlgebra: I
using StaticArrays: SMatrix

const Mat4f = SMatrix{4, 4, Float32, 16}

@testset "AABB BLAS - rayQuery overlap count vs CPU brute-force" begin

    ctx = Mantle.vk_context()
    if !ctx.ray_query_available
        @warn "Skipping AABB rayQuery test: VK_KHR_ray_query not available on this device"
        @test_skip true
        return
    end

    backend = Mantle.LavaBackend()
    bq = backend.bq

    # -------------------------------------------------------------------------
    # Build 1000 random AABBs of side ~1.0 in [-10, 10]^3.
    # -------------------------------------------------------------------------
    Random.seed!(42)
    n_aabbs = 1000
    aabbs = Mantle.AABB[]
    for _ in 1:n_aabbs
        c = Point3f(rand(Float32) * 20f0 - 10f0,
                    rand(Float32) * 20f0 - 10f0,
                    rand(Float32) * 20f0 - 10f0)
        h = 0.5f0
        push!(aabbs, Mantle.AABB(c .- h, c .+ h))
    end

    # -------------------------------------------------------------------------
    # Build AABB BLAS and wrap in HWTLAS.
    # -------------------------------------------------------------------------
    blas = Mantle.as_build() do ctx_build
        Mantle.build_blas_aabb(ctx_build, aabbs)
    end

    @test blas.address != UInt64(0)

    tlas = Mantle.HWTLAS(backend)
    push!(tlas, blas, Mat4f(I); instance_id=UInt32(0))
    Raycore.sync!(tlas)

    # -------------------------------------------------------------------------
    # Flatten AABB data into six Float32 arrays for GPU access via primitive index.
    # -------------------------------------------------------------------------
    aabb_min_x = Float32[a.min[1] for a in aabbs]
    aabb_min_y = Float32[a.min[2] for a in aabbs]
    aabb_min_z = Float32[a.min[3] for a in aabbs]
    aabb_max_x = Float32[a.max[1] for a in aabbs]
    aabb_max_y = Float32[a.max[2] for a in aabbs]
    aabb_max_z = Float32[a.max[3] for a in aabbs]

    gmn_x = LavaArray(aabb_min_x)
    gmn_y = LavaArray(aabb_min_y)
    gmn_z = LavaArray(aabb_min_z)
    gmx_x = LavaArray(aabb_max_x)
    gmx_y = LavaArray(aabb_max_y)
    gmx_z = LavaArray(aabb_max_z)

    # -------------------------------------------------------------------------
    # Generate 64 random query points.
    # -------------------------------------------------------------------------
    n_q = 64
    qx_cpu = Float32[rand(Float32) * 20f0 - 10f0 for _ in 1:n_q]
    qy_cpu = Float32[rand(Float32) * 20f0 - 10f0 for _ in 1:n_q]
    qz_cpu = Float32[rand(Float32) * 20f0 - 10f0 for _ in 1:n_q]

    gqx = LavaArray(qx_cpu)
    gqy = LavaArray(qy_cpu)
    gqz = LavaArray(qz_cpu)

    # -------------------------------------------------------------------------
    # CPU reference: count AABBs containing each query point.
    # -------------------------------------------------------------------------
    in_aabb(px, py, pz, a::Mantle.AABB) =
        px >= a.min[1] && px <= a.max[1] &&
        py >= a.min[2] && py <= a.max[2] &&
        pz >= a.min[3] && pz <= a.max[3]

    ref = Int[count(a -> in_aabb(qx_cpu[i], qy_cpu[i], qz_cpu[i], a), aabbs) for i in 1:n_q]

    # -------------------------------------------------------------------------
    # GPU kernel: zero-length ray at each query point, enumerate AABB candidates,
    # filter to true point-in-AABB using the primitive index.
    # -------------------------------------------------------------------------
    out_g = LavaArray(zeros(UInt32, n_q))

    function count_overlaps_kernel(
            out::LavaDeviceArray{UInt32, 1},
            qx::LavaDeviceArray{Float32, 1},
            qy::LavaDeviceArray{Float32, 1},
            qz::LavaDeviceArray{Float32, 1},
            mn_x::LavaDeviceArray{Float32, 1},
            mn_y::LavaDeviceArray{Float32, 1},
            mn_z::LavaDeviceArray{Float32, 1},
            mx_x::LavaDeviceArray{Float32, 1},
            mx_y::LavaDeviceArray{Float32, 1},
            mx_z::LavaDeviceArray{Float32, 1})
        i = Int(Lava.lava_global_invocation_id_x()) + 1  # 1-based
        ox = qx[i]; oy = qy[i]; oz = qz[i]
        ray = Ray(o=Point3f(ox, oy, oz), d=Vec3f(1f0, 0f0, 0f0), t_min=0f0, t_max=0f0)
        Lava.lava_ray_query_init(ray)
        c = UInt32(0)
        while Lava.lava_ray_query_proceed()
            kind = Lava.lava_ray_query_get_type(false)  # candidate type
            if kind == UInt32(1)  # AABB candidate
                # Primitive index is 0-based; convert to 1-based for Julia arrays.
                prim = Int(Lava.lava_ray_query_get_primitive_index(false)) + 1
                lo_x = mn_x[prim]; lo_y = mn_y[prim]; lo_z = mn_z[prim]
                hi_x = mx_x[prim]; hi_y = mx_y[prim]; hi_z = mx_z[prim]
                inside = ox >= lo_x && ox <= hi_x &&
                         oy >= lo_y && oy <= hi_y &&
                         oz >= lo_z && oz <= hi_z
                if inside
                    c += UInt32(1)
                end
            end
        end
        @inbounds out[i] = c
        return nothing
    end

    Mantle.lava_launch!(bq, count_overlaps_kernel,
                      out_g, gqx, gqy, gqz,
                      gmn_x, gmn_y, gmn_z,
                      gmx_x, gmx_y, gmx_z;
                      ndrange=n_q, workgroup_size=(64, 1, 1), tlas=tlas)
    Mantle.vk_flush!(bq)

    gpu = Array(out_g)

    # -------------------------------------------------------------------------
    # Compare GPU vs CPU.
    # -------------------------------------------------------------------------
    n_nonzero = count(>(0), ref)
    @test n_nonzero > 0   # sanity: at least some points are inside some AABB

    for i in 1:n_q
        @test Int(gpu[i]) == ref[i]
    end

    println("AABB rayQuery GPU test: $n_nonzero/$n_q query points hit at least one AABB. " *
            "Max GPU count: $(maximum(gpu)), Max CPU count: $(maximum(ref))")
end
