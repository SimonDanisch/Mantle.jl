using Test, GeometryBasics, StaticArrays, LinearAlgebra
using Raycore, Mantle, Adapt
include(joinpath(@__DIR__, "testbackend.jl"))
isdefined(@__MODULE__, :tlasprobe!) || include(joinpath(@__DIR__, "hwtlas_helpers.jl"))

if Mantle.supports_hwtlas(TESTBACKEND)

@testset "HWTLAS — UAF safety without CPU fence" begin
    hwtlas = Mantle.HWTLAS{Raycore.Triangle{UInt32}}(TESTBACKEND)

    mesh1 = GeometryBasics.normal_mesh(Sphere(Point3f(0,0,0), 1f0))
    h1 = push!(hwtlas, mesh1, SMatrix{4,4,Float32}(I); instance_id=UInt32(1))
    Raycore.sync!(hwtlas)

    origin, down = [Point3f(0, 0, 5)], [Vec3f(0, 0, -1)]
    probe1 = TLASProbe(TESTBACKEND, origin, down)
    tlastrace!(probe1, hwtlas)
    # DO NOT wait; leave the trace in flight.

    # Mutate + sync — the rebuild drops the old triangle/offset arrays and the
    # old structures. If what the in-flight trace reads is not held until it
    # completes, it reads freed memory.
    mesh2 = GeometryBasics.normal_mesh(Sphere(Point3f(0,0,2), 1f0))
    Raycore.delete!(hwtlas, h1)
    h2 = push!(hwtlas, mesh2, SMatrix{4,4,Float32}(I); instance_id=UInt32(2))
    Raycore.sync!(hwtlas)

    # Second trace reads new geometry.
    probe2 = TLASProbe(TESTBACKEND, origin, down)
    tlastrace!(probe2, hwtlas)

    r1 = tlasresult(probe1)
    r2 = tlasresult(probe2)
    @test r1.hit[1]                 # first mesh at z=0 -> hits at t ≈ 4
    @test isapprox(r1.t[1], 4f0; atol=0.05f0)
    @test r1.id[1] == UInt32(1)
    @test r2.hit[1]                 # second mesh at z=2 -> hits at t ≈ 2
    @test isapprox(r2.t[1], 2f0; atol=0.05f0)
    @test r2.id[1] == UInt32(2)
end

else
    @info "no hardware acceleration structures on this device; skipping" file = basename(@__FILE__)
end
