using Test, Lava, Raycore
using Mantle: build_accel!
using Mantle: VulkanInstanceRecord, build_blas_aabb, AS_INPUT_USAGE, write_grain_instances_kernel
using GeometryBasics: Point3f, Vec3f, Vec4f
using GeometryBasics: Mat4f
using LinearAlgebra: I

@testset "Raycore.sync!(VulkanTLAS) refit cycles" begin
    aabb = Mantle.AABB(Point3f(-1f0,-1f0,-1f0), Point3f(1f0,1f0,1f0))
    blas = build_accel!(Mantle.batchqueue(Mantle.Device())) do ctx; build_blas_aabb(ctx, [aabb]); end

    n = 4
    radius = 1f0
    quats = Mantle.LavaArray([Vec4f(0,0,0,1) for _ in 1:n])
    positions = Mantle.LavaArray([Point3f(Float32(3*(i-1)),0,0) for i in 1:n])
    instance_buf = Mantle.LavaArray{VulkanInstanceRecord}(undef, 2 * n; extra_usage=AS_INPUT_USAGE)

        # `undef` is recycled memory: when the bytes happen to hold a stale-but-valid
        # BLAS address the AS build "works", and zeros or garbage fault it (GPU-AV:
        # "accelerationStructureReference is an invalid address"). Write real
        # records — identity transform at this test's BLAS.
        for _buf in (instance_buf,)
            copyto!(_buf, [VulkanInstanceRecord(Mantle.mat4_to_vk_transform(Mat4f(I)),
                                                blas.address)
                           for _ in 1:length(_buf)])
        end
    backend = Mantle.defaultbackend()
    bq = Mantle.batchqueue(Mantle.Device())

    write_grain_instances_kernel(backend)(
        positions, quats, radius, blas.address, blas.address, instance_buf;
        ndrange = n)
    Mantle.flush!(bq)

    tlas = Mantle.VulkanTLAS(backend)
    push!(tlas, blas, instance_buf; n=2*n, instance_mask=UInt8(0x02))
    Raycore.sync!(tlas)
    pinned_hw_tlas = tlas.hw_tlas

    # Refit cycle: write new positions, mark transforms dirty, sync! (refit path).
    new_positions = Mantle.LavaArray([Point3f(Float32(3*(i-1) + 100f0),0,0) for i in 1:n])
    write_grain_instances_kernel(backend)(
        new_positions, quats, radius, blas.address, blas.address, instance_buf;
        ndrange = n)
    Mantle.flush!(bq)

    tlas.transforms_dirty = true
    Raycore.sync!(tlas)
    @test tlas.dirty == false
    @test tlas.transforms_dirty == false
    # Refit reuses the same hw_tlas object (no rebuild allocation).
    @test tlas.hw_tlas === pinned_hw_tlas
end

