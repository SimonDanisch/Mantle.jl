# test_trace_cull_mask.jl
#
# Tier 3 GPU integration test: verify that the cull_mask kwarg on
# trace_closest_hits! is threaded through hw_raygen to lava_rt_trace_ray.
#
# Setup:
#   Two single-triangle meshes stacked in z:
#     Instance A at z=5,  instance_mask=0x01
#     Instance B at z=10, instance_mask=0x02
#
#   One ray fired along +z from the origin, aimed through both triangles.
#
#   Three traces:
#     cull_mask=0xFF -> hits the closer instance A (z=5)
#     cull_mask=0x01 -> hits instance A (z=5)
#     cull_mask=0x02 -> hits instance B (z=10), skipping A

using Test, Lava, Raycore
using GeometryBasics: Point3f, Vec3f, GLTriangleFace
import GeometryBasics
import LinearAlgebra: I
using StaticArrays: SMatrix

const Mat4f_CM = SMatrix{4, 4, Float32, 16}

@testset "trace_closest_hits! cull_mask kwarg filters instances" begin

    ctx = Mantle.vk_context()
    if ctx.rt_pipeline_properties === nothing
        @warn "Skipping cull_mask test: VK_KHR_ray_tracing_pipeline not available"
        @test_skip true
        return
    end

    backend = Mantle.LavaBackend()

    # Single triangle at z=depth_z, spanning [-1,1] in xy.
    function tri_mesh(depth_z::Float32)
        verts = [Point3f(-1f0, -1f0, depth_z),
                 Point3f( 1f0, -1f0, depth_z),
                 Point3f( 0f0,  1f0, depth_z)]
        faces = [GLTriangleFace(1, 2, 3)]
        GeometryBasics.normal_mesh(GeometryBasics.Mesh(verts, faces))
    end

    tlas = Mantle.VulkanTLAS(backend)
    # Instance A: mask 0x01 at z=5
    push!(tlas, tri_mesh(5f0),  Mat4f_CM(I);
          instance_id=UInt32(1), instance_mask=UInt8(0x01))
    # Instance B: mask 0x02 at z=10
    push!(tlas, tri_mesh(10f0), Mat4f_CM(I);
          instance_id=UInt32(2), instance_mask=UInt8(0x02))
    Raycore.sync!(tlas)

    accel = tlas.hw_accel
    accel === nothing && error("VulkanTLAS did not build a HardwareAccel")

    # One ray along +z from origin, aimed through the centroid of both triangles.
    ray = Raycore.RTRay(0f0, 0f0, 0f0,   # origin
                        1f-4,             # tmin
                        0f0, 0f0, 1f0,   # direction
                        1f4)              # tmax
    gpu_rays = Mantle.LavaArray([ray])
    gpu_hits = Mantle.LavaArray([Raycore.RTHitResult(0, 0f0, 0, 0, 0f0, 0f0, 0, 0)])

    function do_trace(mask::UInt32)
        # Re-use the same buffer; each call overwrites it.
        gpu_hits_local = Mantle.LavaArray([Raycore.RTHitResult(0, 0f0, 0, 0, 0f0, 0f0, 0, 0)])
        Mantle.trace_closest_hits!(gpu_hits_local, gpu_rays, accel, 1; cull_mask=mask)
        Mantle.vk_flush!(backend.bq)
        return Array(gpu_hits_local)[1]
    end

    h_ff = do_trace(UInt32(0xFF))
    h_01 = do_trace(UInt32(0x01))
    h_02 = do_trace(UInt32(0x02))

    # cull_mask=0xFF: closest hit is instance A at z=5
    @test h_ff.hit != UInt32(0)
    @test isapprox(h_ff.t, 5f0; atol=1f-2)

    # cull_mask=0x01: only instance A visible, hit at z=5
    @test h_01.hit != UInt32(0)
    @test isapprox(h_01.t, 5f0; atol=1f-2)

    # cull_mask=0x02: only instance B visible, hit at z=10
    @test h_02.hit != UInt32(0)
    @test isapprox(h_02.t, 10f0; atol=1f-2)

    println("cull_mask test: 0xFF -> t=$(h_ff.t), 0x01 -> t=$(h_01.t), 0x02 -> t=$(h_02.t)")
end
