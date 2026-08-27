using Test, Lava, Raycore
using Lava: LavaInstanceRecord, build_blas_aabb, as_build, AS_INPUT_USAGE
using GeometryBasics: Point3f

# Triangle type used by the default HWTLAS.
const Tri = Raycore.Triangle{UInt32}

@testset "push!(hwtlas, blas, instance_buf) triangles kwarg -- batch path populates tri_gpu/off_gpu" begin
    aabb = Mantle.AABB(Point3f(-1f0,-1f0,-1f0), Point3f(1f0,1f0,1f0))
    blas = as_build() do ctx; build_blas_aabb(ctx, [aabb]); end

    n = 8
    instance_buf = Mantle.LavaArray{LavaInstanceRecord}(undef, n; extra_usage=AS_INPUT_USAGE)
    backend = Mantle.LavaBackend()
    tlas = Mantle.HWTLAS(backend)

    # 12 dummy triangles -- same count as a cube BLAS.
    # empty_triangle gives a zero-filled sentinel triangle of the correct type.
    n_tris_per_blas = 12
    dummy_triangles = [Raycore.empty_triangle(Tri) for _ in 1:n_tris_per_blas]

    handle = push!(tlas, blas, instance_buf;
                   n=n, instance_mask=UInt8(0x04),
                   triangles=dummy_triangles)
    @test handle isa Raycore.TLASHandle
    @test tlas.instance_batches[1].triangles === dummy_triangles

    Raycore.sync!(tlas)

    # tri_gpu must hold all 12 triangles.
    @test tlas.tri_gpu !== nothing
    @test length(tlas.tri_gpu) == n_tris_per_blas

    # off_gpu must have one entry per instance, all zero (all instances share triangle 0).
    @test tlas.off_gpu !== nothing
    @test length(tlas.off_gpu) == n
    @test all(==(UInt32(0)), Array(tlas.off_gpu))
end

@testset "push!(hwtlas, blas, instance_buf) default triangles kwarg -- off_gpu sized, tri_gpu empty" begin
    aabb = Mantle.AABB(Point3f(-1f0,-1f0,-1f0), Point3f(1f0,1f0,1f0))
    blas = as_build() do ctx; build_blas_aabb(ctx, [aabb]); end

    n = 4
    instance_buf = Mantle.LavaArray{LavaInstanceRecord}(undef, n; extra_usage=AS_INPUT_USAGE)
    backend = Mantle.LavaBackend()
    tlas = Mantle.HWTLAS(backend)

    # No triangles supplied -- rayQuery-only backward-compat path.
    push!(tlas, blas, instance_buf; n=n, instance_mask=UInt8(0x02))
    @test isempty(tlas.instance_batches[1].triangles)

    Raycore.sync!(tlas)

    @test tlas.tri_gpu !== nothing
    @test length(tlas.tri_gpu) == 0   # empty -- no triangles were passed

    @test tlas.off_gpu !== nothing
    @test length(tlas.off_gpu) == n   # still N entries, all zero
    @test all(==(UInt32(0)), Array(tlas.off_gpu))
end
