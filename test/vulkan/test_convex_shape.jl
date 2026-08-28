using Test, Lava, Mantle
using Mantle: UnitCube, support
using GeometryBasics: Vec3f

@testset "UnitCube support function -- 6 axis-aligned directions" begin
    cube = UnitCube()
    @test support(cube, Vec3f( 1,  0,  0)) == Vec3f( 1,  1,  1)
    @test support(cube, Vec3f(-1,  0,  0)) == Vec3f(-1,  1,  1)
    @test support(cube, Vec3f( 0,  1,  0)) == Vec3f( 1,  1,  1)
    @test support(cube, Vec3f( 0, -1,  0)) == Vec3f( 1, -1,  1)
    @test support(cube, Vec3f( 0,  0,  1)) == Vec3f( 1,  1,  1)
    @test support(cube, Vec3f( 0,  0, -1)) == Vec3f( 1,  1, -1)
end

@testset "UnitCube support function -- diagonal directions" begin
    cube = UnitCube()
    # +X+Y+Z corner direction: should pick the (1, 1, 1) corner.
    @test support(cube, Vec3f(1, 1, 1)) == Vec3f(1, 1, 1)
    # All-negative direction: opposite corner.
    @test support(cube, Vec3f(-1, -1, -1)) == Vec3f(-1, -1, -1)
    # Mixed: -X +Y -Z corner.
    @test support(cube, Vec3f(-2, 3, -7)) == Vec3f(-1, 1, -1)
end

@testset "UnitCube support function -- zero direction component" begin
    # When a component of dir is exactly zero, support picks +1 by convention
    # (the dir[i] >= 0 branch).  Either +1 or -1 along that axis is a valid
    # support point at exact tie; we just need the function to be deterministic.
    cube = UnitCube()
    @test support(cube, Vec3f(1, 0, 0))  == Vec3f(1, 1, 1)
    @test support(cube, Vec3f(0, 0, 0))  == Vec3f(1, 1, 1)   # all zero -> all +1
    @test support(cube, Vec3f(-1, 0, 0)) == Vec3f(-1, 1, 1)
end

@testset "UnitCube support is in Vec3f and unit-magnitude per axis" begin
    cube = UnitCube()
    for dir in (Vec3f(1, 1, 1), Vec3f(0.1f0, -0.7f0, 5f0), Vec3f(1f-9, 0, 1))
        s = support(cube, dir)
        @test s isa Vec3f
        @test abs(s[1]) == 1f0
        @test abs(s[2]) == 1f0
        @test abs(s[3]) == 1f0
    end
end

@testset "support is dispatch-driven on shape type" begin
    # `support` must be a generic function dispatching on the shape's type.
    # Verify by checking ConvexShape is the abstract supertype and UnitCube <: ConvexShape.
    @test UnitCube <: Mantle.ConvexShape
    # A separate concrete subtype with a different support function should not
    # collide with UnitCube via inheritance.
    # (Smoke check the dispatch table.)
    @test length(methods(support, (UnitCube, Vec3f))) >= 1
end

