using Test, Lava, Raycore
using Mantle: build_accel!
using Mantle: VulkanInstanceRecord, build_blas_aabb, AS_INPUT_USAGE, write_grain_instances_kernel
using GeometryBasics: Point3f, Vec3f, Vec4f, Mat4f
using LinearAlgebra: I

@testset "VulkanTLAS sync! with instance batch" begin
    aabb = Mantle.AABB(Point3f(-1f0,-1f0,-1f0), Point3f(1f0,1f0,1f0))
    blas = build_accel!(Mantle.vk_context().default_bq) do ctx; build_blas_aabb(ctx, [aabb]); end

    n = 8
    radius = 1f0
    quats = Mantle.LavaArray([Vec4f(0,0,0,1) for _ in 1:n])
    positions = Mantle.LavaArray([Point3f(Float32(3*(i-1)),0,0) for i in 1:n])
    instance_buf = Mantle.LavaArray{VulkanInstanceRecord}(undef, 2 * n; extra_usage=AS_INPUT_USAGE)

    backend = Mantle.LavaBackend()
    bq = Mantle.vk_context().default_bq

    # Use the KA backend pattern for @kernel-defined kernels.
    write_grain_instances_kernel(backend)(
        positions, quats, radius, blas.address, blas.address, instance_buf;
        ndrange = n)
    Mantle.vk_flush!(bq)

    tlas = Mantle.VulkanTLAS(backend)
    push!(tlas, blas, instance_buf; n=2*n, instance_mask=UInt8(0x02))
    Raycore.sync!(tlas)

    @test tlas.dirty == false
    @test tlas.hw_tlas !== nothing
    @test tlas.hw_tlas.allow_update == true
    @test tlas.hw_tlas.update_scratch_size > 0
end

@testset "VulkanTLAS sync! supports multiple batches (P3-fu4)" begin
    # P3-fu4 lifted the single-batch guard.  Multiple batches concatenate into
    # a combined instance buffer and the HWTLAS spans them all.
    aabb = Mantle.AABB(Point3f(-1f0, -1f0, -1f0), Point3f(1f0, 1f0, 1f0))
    blas = build_accel!(Mantle.vk_context().default_bq) do ctx; build_blas_aabb(ctx, [aabb]); end

    n_a = 4; n_b = 6
    instance_buf1 = Mantle.LavaArray{VulkanInstanceRecord}(undef, n_a; extra_usage=AS_INPUT_USAGE)
    instance_buf2 = Mantle.LavaArray{VulkanInstanceRecord}(undef, n_b; extra_usage=AS_INPUT_USAGE)
    # `undef` is recycled memory: when the bytes happen to hold a stale-but-valid
    # BLAS address the build "works", and when they hold zeros or garbage it
    # faults (GPU-AV: "accelerationStructureReference is an invalid address").
    # Write real records — an identity transform at this test's BLAS.
    for buf in (instance_buf1, instance_buf2)
        copyto!(buf, [VulkanInstanceRecord(Mantle.mat4_to_vk_transform(Mat4f(I)),
                                           blas.address)
                      for _ in 1:length(buf)])
    end

    backend = Mantle.LavaBackend()
    tlas = Mantle.VulkanTLAS(backend)
    h1 = push!(tlas, blas, instance_buf1; n=n_a, instance_mask=UInt8(0x02))
    h2 = push!(tlas, blas, instance_buf2; n=n_b, instance_mask=UInt8(0x04))

    Raycore.sync!(tlas)
    @test tlas.dirty == false
    @test tlas.hw_tlas !== nothing
    @test tlas.hw_tlas.allow_update == true

    # The combined instance buffer holds n_a + n_b records.
    @test tlas.combined_instance_buf !== nothing
    @test length(tlas.combined_instance_buf) >= n_a + n_b

    # Refit on the multi-batch VulkanTLAS works too.
    tlas.transforms_dirty = true
    Raycore.sync!(tlas)
    @test tlas.dirty == false   # refit does not toggle dirty
end

