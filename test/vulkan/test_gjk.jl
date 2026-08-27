using Test, Lava, Mantle
using Lava: UnitCube, gjk, GJKResult, transform_point, support_AB
using GeometryBasics: Vec3f

# Shared narrow-phase helpers: translation_transform, rotation_z_transform,
# ID, tx.  Guarded so re-includes in the same Main are no-ops.
isdefined(@__MODULE__, :translation_transform) ||
    include(joinpath(@__DIR__, "narrow_phase_helpers.jl"))

# ---------------------------------------------------------------------------
# Test utilities
# ---------------------------------------------------------------------------

# Quick sanity check: the GJKResult simplex should be a 4-tuple of Vec3f.
function check_result_type(r::GJKResult)
    @test r isa GJKResult
    @test r.simplex isa NTuple{4, Vec3f}
    @test r.iterations >= 1
end

# ---------------------------------------------------------------------------
# Category 1: Clearly overlapping cubes
# ---------------------------------------------------------------------------

@testset "GJK -- overlapping cubes" begin
    cube = UnitCube()

    @testset "both at origin (completely inside each other)" begin
        r = gjk(cube, cube, ID, ID)
        check_result_type(r)
        @test r.overlap == true
    end

    @testset "offset 1 along X (overlapping by 1 unit)" begin
        # Cube A at (0,0,0): extents [-1,+1] on all axes.
        # Cube B at (1,0,0): extents [0,+2] on X.  Overlap by 1 unit.
        T_B = translation_transform(1, 0, 0)
        r   = gjk(cube, cube, ID, T_B)
        check_result_type(r)
        @test r.overlap == true
    end

    @testset "offset 1.9 along X (near edge, still overlapping)" begin
        T_B = translation_transform(1.9f0, 0, 0)
        r   = gjk(cube, cube, ID, T_B)
        check_result_type(r)
        @test r.overlap == true
    end

    @testset "offset 1 along Y" begin
        T_B = translation_transform(0, 1, 0)
        r   = gjk(cube, cube, ID, T_B)
        check_result_type(r)
        @test r.overlap == true
    end

    @testset "offset diagonal (0.5, 0.5, 0.5) -- deep overlap" begin
        T_B = translation_transform(0.5f0, 0.5f0, 0.5f0)
        r   = gjk(cube, cube, ID, T_B)
        check_result_type(r)
        @test r.overlap == true
    end
end

# ---------------------------------------------------------------------------
# Category 2: Clearly separated cubes
# ---------------------------------------------------------------------------

@testset "GJK -- separated cubes" begin
    cube = UnitCube()

    @testset "offset 3 along X (gap of 1 unit)" begin
        # Cube A extents [-1,+1] on X, Cube B extents [2,4] on X -- 1-unit gap.
        T_B = translation_transform(3, 0, 0)
        r   = gjk(cube, cube, ID, T_B)
        check_result_type(r)
        @test r.overlap == false
    end

    @testset "offset 5 along X" begin
        T_B = translation_transform(5, 0, 0)
        r   = gjk(cube, cube, ID, T_B)
        check_result_type(r)
        @test r.overlap == false
    end

    @testset "offset 3 along Y" begin
        T_B = translation_transform(0, 3, 0)
        r   = gjk(cube, cube, ID, T_B)
        check_result_type(r)
        @test r.overlap == false
    end

    @testset "offset 3 along Z" begin
        T_B = translation_transform(0, 0, 3)
        r   = gjk(cube, cube, ID, T_B)
        check_result_type(r)
        @test r.overlap == false
    end

    @testset "offset large diagonal (100, 100, 100)" begin
        T_B = translation_transform(100, 100, 100)
        r   = gjk(cube, cube, ID, T_B)
        check_result_type(r)
        @test r.overlap == false
        @test r.iterations <= 32  # must terminate within cap
    end
end

# ---------------------------------------------------------------------------
# Category 3: Edge cases
# ---------------------------------------------------------------------------

@testset "GJK -- edge cases" begin
    cube = UnitCube()

    @testset "just touching: offset exactly 2 (deterministic return)" begin
        # Both cubes touch at X = +1 / X = -1 (face-to-face contact).
        # GJK at the contact boundary -- either answer is acceptable but the
        # call must not hang or error.
        T_B = translation_transform(2, 0, 0)
        r   = gjk(cube, cube, ID, T_B)
        check_result_type(r)
        @test r.overlap isa Bool  # just must complete deterministically
    end

    @testset "rotated 45 deg around Z, positions overlapping" begin
        # Cube A at origin (identity), Cube B rotated 45 deg + translated
        # (1.0, 0, 0).  With a 45-degree rotation, the B cube's half-extent
        # along X is sqrt(2) ~ 1.414 -- the world-space AABB of B extends
        # roughly from -0.414 to +2.414 on X when centered at 1.  Overlap
        # with A's [-1,+1] is assured.
        T_B = rotation_z_transform(Float32(pi/4), 1f0, 0f0, 0f0)
        r   = gjk(cube, cube, ID, T_B)
        check_result_type(r)
        @test r.overlap == true
    end

    @testset "rotated 45 deg, clearly separated (offset 5)" begin
        T_B = rotation_z_transform(Float32(pi/4), 5f0, 0f0, 0f0)
        r   = gjk(cube, cube, ID, T_B)
        check_result_type(r)
        @test r.overlap == false
    end

    @testset "both cubes have non-trivial transforms (translate + rotate)" begin
        # A at (-3, 0, 0) identity, B at (3, 0, 0) rotated 30 deg.
        T_A = translation_transform(-3, 0, 0)
        T_B = rotation_z_transform(Float32(pi/6), 3f0, 0f0, 0f0)
        r   = gjk(cube, cube, T_A, T_B)
        check_result_type(r)
        @test r.overlap == false
    end

    @testset "iteration cap: max_iters=1 forces cap exit (no crash)" begin
        T_B = translation_transform(0.5f0, 0f0, 0f0)
        r   = gjk(cube, cube, ID, T_B; max_iters=1)
        @test r isa GJKResult   # must return, not error
        @test r.iterations >= 1
    end

    @testset "GJKResult type fields" begin
        T_B = translation_transform(3, 0, 0)
        r   = gjk(cube, cube, ID, T_B)
        @test fieldnames(GJKResult) == (:overlap, :simplex, :iterations)
        @test r.simplex isa NTuple{4, Vec3f}
    end
end

# ---------------------------------------------------------------------------
# Category 4: Internal helper smoke tests
# ---------------------------------------------------------------------------

@testset "GJK internals -- transform helpers" begin
    using Lava: transform_point, transform_dir, inv_transform_dir

    T = translation_transform(1, 2, 3)

    @testset "transform_point applies rotation + translation" begin
        p = Vec3f(0, 0, 0)
        @test Mantle.transform_point(T, p) ≈ Vec3f(1, 2, 3)
        p2 = Vec3f(1, 0, 0)
        @test Mantle.transform_point(T, p2) ≈ Vec3f(2, 2, 3)
    end

    @testset "transform_dir ignores translation" begin
        d = Vec3f(1, 0, 0)
        @test Mantle.transform_dir(T, d) ≈ Vec3f(1, 0, 0)
    end

    @testset "inv_transform_dir is inverse of transform_dir for orthonormal R" begin
        angle = Float32(pi/5)
        T_rot = rotation_z_transform(angle)
        d_world = Vec3f(1, 1, 0)
        d_local = Mantle.inv_transform_dir(T_rot, d_world)
        # Rotating d_local back to world should recover d_world.
        d_back  = Mantle.transform_dir(T_rot, d_local)
        @test d_back ≈ d_world atol=1f-5
    end
end

@testset "GJK internals -- support_AB" begin
    cube = UnitCube()
    # Two unit cubes at origin: support_AB along +X should give (1,1,1) - (-1,1,1) = (2,0,0).
    dir = Vec3f(1, 0, 0)
    s   = Mantle.support_AB(cube, cube, ID, ID, dir)
    @test s ≈ Vec3f(2, 0, 0)

    # Along -X: gives (-1,1,1) - (1,1,1) = (-2,0,0).
    s2 = Mantle.support_AB(cube, cube, ID, ID, Vec3f(-1, 0, 0))
    @test s2 ≈ Vec3f(-2, 0, 0)
end
