using Test, Lava, Raycore
using Mantle: build_accel!
using .MVE: VulkanInstanceRecord, build_blas_aabb, AS_INPUT_USAGE, write_grain_instances_kernel
using GeometryBasics: Point3f, Vec3f, Vec4f

@testset "Raycore.sync!(VulkanTLAS) refit cycles" begin
    aabb = Mantle.AABB(Point3f(-1f0,-1f0,-1f0), Point3f(1f0,1f0,1f0))
    blas = build_accel!() do ctx; build_blas_aabb(ctx, [aabb]); end

    n = 4
    radius = 1f0
    quats = MVE.LavaArray([Vec4f(0,0,0,1) for _ in 1:n])
    positions = MVE.LavaArray([Point3f(Float32(3*(i-1)),0,0) for i in 1:n])
    instance_buf = MVE.LavaArray{VulkanInstanceRecord}(undef, 2 * n; extra_usage=AS_INPUT_USAGE)
    backend = MVE.LavaBackend()
    bq = MVE.vk_context().default_bq

    write_grain_instances_kernel(backend)(
        positions, quats, radius, blas.address, blas.address, instance_buf;
        ndrange = n)
    MVE.vk_flush!(bq)

    tlas = MVE.VulkanTLAS(backend)
    push!(tlas, blas, instance_buf; n=2*n, instance_mask=UInt8(0x02))
    Raycore.sync!(tlas)
    pinned_hw_tlas = tlas.hw_tlas

    # Refit cycle: write new positions, mark transforms dirty, sync! (refit path).
    new_positions = MVE.LavaArray([Point3f(Float32(3*(i-1) + 100f0),0,0) for i in 1:n])
    write_grain_instances_kernel(backend)(
        new_positions, quats, radius, blas.address, blas.address, instance_buf;
        ndrange = n)
    MVE.vk_flush!(bq)

    tlas.transforms_dirty = true
    Raycore.sync!(tlas)
    @test tlas.dirty == false
    @test tlas.transforms_dirty == false
    # Refit reuses the same hw_tlas object (no rebuild allocation).
    @test tlas.hw_tlas === pinned_hw_tlas
end

