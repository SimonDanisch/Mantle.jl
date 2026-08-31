# test_instance_masks.jl
#
# Tier 3 GPU integration test: per-instance cullMask filtering on VulkanTLAS.
#
# Two triangle BLASes at different z-depths, each tagged with a distinct mask.
# Three rayQuery calls use mask=0x01, mask=0x02, mask=0xFF and must hit the
# correct instance in each case.

using Test, Lava, Raycore
using GeometryBasics: Point3f, Vec3f, GLTriangleFace
import GeometryBasics
import LinearAlgebra: I
using StaticArrays: SMatrix

const Mat4f_IM = SMatrix{4, 4, Float32, 16}

@testset "Instance masks: rayQuery filters by cullMask" begin

    ctx = MVE.vk_context()
    if !ctx.ray_query_available
        @warn "Skipping instance-mask test: VK_KHR_ray_query not available on this device"
        @test_skip true
        return
    end

    backend = MVE.LavaBackend()
    bq = backend.bq

    # Build two single-triangle meshes at different z-planes.
    function tri_mesh(z::Float32)
        verts = [Point3f(-1f0, -1f0, z), Point3f(1f0, -1f0, z), Point3f(0f0, 1f0, z)]
        faces = [GLTriangleFace(1, 2, 3)]
        GeometryBasics.normal_mesh(GeometryBasics.Mesh(verts, faces))
    end

    tlas = MVE.VulkanTLAS(backend)
    # Instance 1: mask 0x01, z=5
    push!(tlas, tri_mesh(5f0),  Mat4f_IM(I); instance_id=UInt32(1), instance_mask=UInt8(0x01))
    # Instance 2: mask 0x02, z=10
    push!(tlas, tri_mesh(10f0), Mat4f_IM(I); instance_id=UInt32(2), instance_mask=UInt8(0x02))
    Raycore.sync!(tlas)

    # 3-element output buffer: one entry per ray/mask combination.
    out_g = LavaArray(fill(-1f0, 3))

    # Kernel: thread i (0-based from GPU builtin) fires along +z with a distinct mask.
    #   i=0 -> mask 0x01 -> should hit z=5 instance (t~5)
    #   i=1 -> mask 0x02 -> should hit z=10 instance (t~10)
    #   i=2 -> mask 0xFF -> should hit the closer z=5 instance (t~5)
    function mask_kernel(out::LavaDeviceArray{Float32, 1})
        i = Int(Lava.lava_global_invocation_id_x())   # 0-based
        m = i == 0 ? UInt32(0x01) : (i == 1 ? UInt32(0x02) : UInt32(0xFF))
        ray = Ray(o=Point3f(0f0, 0f0, 0f0), d=Vec3f(0f0, 0f0, 1f0), t_min=0f0, t_max=1f3)
        Lava.lava_ray_query_init(ray; mask=m)
        while Lava.lava_ray_query_proceed()
            Lava.lava_ray_query_confirm()
        end
        kind = Lava.lava_ray_query_get_type(true)
        thit = (kind == UInt32(1)) ? Lava.lava_ray_query_get_t(true) : -1f0
        out[i + 1] = thit   # 1-based store
        return nothing
    end

    MVE.lava_launch!(bq, mask_kernel, out_g;
                      ndrange=3, workgroup_size=(64, 1, 1), tlas=tlas)
    MVE.vk_flush!(bq)

    r = Array(out_g)

    @test isapprox(r[1], 5f0;  atol=1f-3)   # mask=0x01 hits z=5 instance
    @test isapprox(r[2], 10f0; atol=1f-3)   # mask=0x02 hits z=10 instance
    @test isapprox(r[3], 5f0;  atol=1f-3)   # mask=0xFF hits closer z=5 instance

    println("Instance mask test: r[1]=$(r[1]) (exp 5), r[2]=$(r[2]) (exp 10), r[3]=$(r[3]) (exp 5)")
end
