using Test, Lava, Raycore
using Mantle: build_accel!
using .MVE: VulkanInstanceRecord, build_blas_aabb, AS_INPUT_USAGE
using GeometryBasics: Point3f
using GeometryBasics: Mat4f
using LinearAlgebra: I

@testset "push!(hwtlas, blas, instance_buf) -- registration" begin
    aabb = Mantle.AABB(Point3f(-1f0, -1f0, -1f0), Point3f(1f0, 1f0, 1f0))
    blas = build_accel!(MVE.vk_context().default_bq) do ctx; build_blas_aabb(ctx, [aabb]); end

    n = 100
    instance_buf = MVE.LavaArray{VulkanInstanceRecord}(undef, n; extra_usage=AS_INPUT_USAGE)

        # `undef` is recycled memory: when the bytes happen to hold a stale-but-valid
        # BLAS address the AS build "works", and zeros or garbage fault it (GPU-AV:
        # "accelerationStructureReference is an invalid address"). Write real
        # records — identity transform at this test's BLAS.
        for _buf in (instance_buf,)
            copyto!(_buf, [VulkanInstanceRecord(MVE.mat4_to_vk_transform(Mat4f(I)),
                                                blas.address)
                           for _ in 1:length(_buf)])
        end

    backend = MVE.LavaBackend()
    tlas = MVE.VulkanTLAS(backend)

    handle = push!(tlas, blas, instance_buf; n=n, instance_mask=UInt8(0x02))
    @test handle isa Raycore.TLASHandle
    @test length(tlas.instances) == 1
    @test tlas.instances[1].n == n
    @test tlas.instances[1].instance_mask == UInt8(0x02)
    @test tlas.instances[1].blas === blas
    @test tlas.instances[1].instance_buf === instance_buf
    @test tlas.dirty == true
end

