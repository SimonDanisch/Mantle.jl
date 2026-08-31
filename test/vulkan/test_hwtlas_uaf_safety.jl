using Test, GeometryBasics, StaticArrays, LinearAlgebra
using Raycore, Lava, Adapt

@testset "MVE.VulkanTLAS — UAF safety without CPU fence" begin
    backend = MVE.LavaBackend()
    hwtlas = MVE.VulkanTLAS(backend)

    mesh1 = GeometryBasics.normal_mesh(Sphere(Point3f(0,0,0), 1f0))
    h1 = push!(hwtlas, mesh1, SMatrix{4,4,Float32}(I); instance_id=UInt32(1))
    Raycore.sync!(hwtlas)

    rays = MVE.LavaArray([Raycore.RTRay(0,0,5, 0, 0,0,-1, 1f3)])
    hits1 = MVE.LavaArray(fill(Raycore.RTHitResult(0,0,0,0,0,0,0,0), 1))
    Mantle.trace_closest_hits!(hits1, rays, hwtlas.hw_accel, 1)
    # DO NOT wait; leave the dispatch in flight.

    # Mutate + sync — drops old tri_gpu/off_gpu via unsafe_free!. If the
    # closure-leaves pin is broken, the still-in-flight dispatch UAFs.
    mesh2 = GeometryBasics.normal_mesh(Sphere(Point3f(0,0,2), 1f0))
    Raycore.delete!(hwtlas, h1)
    h2 = push!(hwtlas, mesh2, SMatrix{4,4,Float32}(I); instance_id=UInt32(2))
    Raycore.sync!(hwtlas)

    # Second dispatch reads new geometry.
    hits2 = MVE.LavaArray(fill(Raycore.RTHitResult(0,0,0,0,0,0,0,0), 1))
    Mantle.trace_closest_hits!(hits2, rays, hwtlas.hw_accel, 1)

    Raycore.wait_for_gpu!(hwtlas)

    h1a = Array(hits1)[1]
    h2a = Array(hits2)[1]
    @test h1a.hit != 0              # first mesh at z=0 -> hits at t ≈ 4
    @test isapprox(Float32(h1a.t), 4f0; atol=0.05f0)
    @test h2a.hit != 0              # second mesh at z=2 -> hits at t ≈ 2
    @test isapprox(Float32(h2a.t), 2f0; atol=0.05f0)
end
