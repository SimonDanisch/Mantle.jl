using Test, Lava, Raycore
using Mantle: VulkanInstanceRecord, build_blas_aabb, build_accel!, AS_INPUT_USAGE
using GeometryBasics: Point3f

# P3-fu2: Base.delete!(::VulkanTLAS, ::TLASHandle) for batch handles.

@testset "delete!(hwtlas, batch_handle) removes the batch" begin
    aabb = Mantle.AABB(Point3f(-1f0, -1f0, -1f0), Point3f(1f0, 1f0, 1f0))
    blas = build_accel!() do ctx; build_blas_aabb(ctx, [aabb]); end

    n = 4
    instance_buf = Mantle.LavaArray{VulkanInstanceRecord}(undef, n; extra_usage=AS_INPUT_USAGE)
    backend = Mantle.LavaBackend()
    tlas = Mantle.VulkanTLAS(backend)
    handle = push!(tlas, blas, instance_buf; n=n, instance_mask=UInt8(0x04))
    @test length(tlas.instance_batches) == 1

    deleted = delete!(tlas, handle)
    @test deleted == true
    @test length(tlas.instance_batches) == 0
    @test tlas.dirty == true

    # Deleting again returns false (handle is gone from handle_to_batch_idx).
    @test delete!(tlas, handle) == false
end

@testset "delete! returns false for unknown handle" begin
    backend = Mantle.LavaBackend()
    tlas = Mantle.VulkanTLAS(backend)
    fake_handle = Raycore.TLASHandle(UInt32(99))
    @test delete!(tlas, fake_handle) == false
end

@testset "delete!(hwtlas, batch_handle) leaves siblings alone" begin
    aabb = Mantle.AABB(Point3f(-1f0, -1f0, -1f0), Point3f(1f0, 1f0, 1f0))
    blas = build_accel!() do ctx; build_blas_aabb(ctx, [aabb]); end

    n = 4
    backend = Mantle.LavaBackend()
    tlas = Mantle.VulkanTLAS(backend)
    buf_a = Mantle.LavaArray{VulkanInstanceRecord}(undef, n; extra_usage=AS_INPUT_USAGE)
    buf_b = Mantle.LavaArray{VulkanInstanceRecord}(undef, n; extra_usage=AS_INPUT_USAGE)
    handle_a = push!(tlas, blas, buf_a; n=n, instance_mask=UInt8(0x02))
    handle_b = push!(tlas, blas, buf_b; n=n, instance_mask=UInt8(0x04))
    @test length(tlas.instance_batches) == 2

    @test delete!(tlas, handle_a) == true
    @test length(tlas.instance_batches) == 1
    @test tlas.instance_batches[1].handle === handle_b
end

