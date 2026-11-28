# A BLAS built from device arrays reads them in place, through views that start one
# element in: a build or refit that dropped the views' offset reads the sentinel vertex
# and moves the triangle away from the probe ray.
using Test, Mantle, Raycore, GeometryBasics, GPUArrays
include(joinpath(@__DIR__, "testbackend.jl"))
isdefined(@__MODULE__, :tlasprobe!) || include(joinpath(@__DIR__, "hwtlas_helpers.jl"))

if Mantle.supports_hwtlas(TESTBACKEND)
@testset "a BLAS built from offset device views refits in place" begin
    vertices = [(0f0, 0f0, 0f0), (1f0, 0f0, 0f0), (0f0, 1f0, 0f0)]
    moved = [(x, y, 2f0) for (x, y, _) in vertices]
    vb = Mantle.buildinput(TESTBACKEND, vcat([(9f0, 9f0, 9f0)], vertices, [(8f0, 8f0, 8f0)]))
    ib = Mantle.buildinput(TESTBACKEND, UInt32[99, 0, 1, 2, 99])
    vs = GPUArrays.derive(NTuple{3,Float32}, vb, (3,), 1)
    ids = GPUArrays.derive(UInt32, ib, (3,), 1)
    blas = Mantle.build_accel!(Mantle.batchqueue(TESTBACKEND)) do build
        Mantle.build_blas(build, vs, ids; allow_update = true)
    end
    @test Mantle.refitinput(blas) === vs
    tlas = Mantle.HWTLAS{Raycore.Triangle{UInt32}}(TESTBACKEND)
    push!(tlas, blas)
    Raycore.sync!(tlas)
    probe = TLASProbe(TESTBACKEND, [Point3f(0.25f0, 0.25f0, -1f0)], [Vec3f(0, 0, 1)])
    @test only(tlastrace(probe, tlas).t) ≈ 1f0 atol = 1f-3
    @test Raycore.refit!(blas, moved) === blas
    Raycore.refit!(tlas)
    @test only(tlastrace(probe, tlas).t) ≈ 3f0 atol = 1f-3
    @test Array(vb) == vcat([(9f0, 9f0, 9f0)], moved, [(8f0, 8f0, 8f0)])
    @test_throws ArgumentError Raycore.refit!(blas, vcat(moved, moved))
    Mantle.waitidle(TESTBACKEND)
    @test isempty(Mantle.validationmessages!(TESTBACKEND))
end
end
