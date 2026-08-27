using Test, Lava, Raycore
using Lava: LavaInstanceRecord, build_blas_aabb, as_build, AS_INPUT_USAGE
using GeometryBasics: Point3f

@testset "push!(hwtlas, blas, instance_buf) -- registration" begin
    aabb = Mantle.AABB(Point3f(-1f0, -1f0, -1f0), Point3f(1f0, 1f0, 1f0))
    blas = as_build() do ctx; build_blas_aabb(ctx, [aabb]); end

    n = 100
    instance_buf = Mantle.LavaArray{LavaInstanceRecord}(undef, n; extra_usage=AS_INPUT_USAGE)

    backend = Mantle.LavaBackend()
    tlas = Mantle.HWTLAS(backend)

    handle = push!(tlas, blas, instance_buf; n=n, instance_mask=UInt8(0x02))
    @test handle isa Raycore.TLASHandle
    @test length(tlas.instance_batches) == 1
    @test tlas.instance_batches[1].n == n
    @test tlas.instance_batches[1].instance_mask == UInt8(0x02)
    @test tlas.instance_batches[1].blas === blas
    @test tlas.instance_batches[1].instance_buf === instance_buf
    @test tlas.dirty == true
end
