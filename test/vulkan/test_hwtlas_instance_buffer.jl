using Test, Lava, Raycore
using Mantle: VulkanInstanceRecord, build_blas_aabb, build_accel!, AS_INPUT_USAGE
using GeometryBasics: Point3f

@testset "instance_buffer returns the buffer behind a batch handle" begin
    aabb = Mantle.AABB(Point3f(-1f0, -1f0, -1f0), Point3f(1f0, 1f0, 1f0))
    blas = build_accel!() do ctx; build_blas_aabb(ctx, [aabb]); end

    n = 8
    instance_buf = Mantle.LavaArray{VulkanInstanceRecord}(undef, n; extra_usage=AS_INPUT_USAGE)
    backend = Mantle.LavaBackend()
    tlas = Mantle.VulkanTLAS(backend)

    handle = push!(tlas, blas, instance_buf; n=n, instance_mask=UInt8(0x04))

    # Accessor returns the SAME LavaArray (identity, not a copy).
    buf_returned = Raycore.instance_buffer(tlas, handle)
    @test buf_returned === instance_buf
end

@testset "instance_buffer errors on invalid handle" begin
    backend = Mantle.LavaBackend()
    tlas = Mantle.VulkanTLAS(backend)
    fake_handle = Raycore.TLASHandle(UInt32(99))
    @test_throws ErrorException Raycore.instance_buffer(tlas, fake_handle)
end

