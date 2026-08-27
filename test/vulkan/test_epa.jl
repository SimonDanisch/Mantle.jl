using Test, Lava, Mantle
using Lava: UnitCube, gjk, GJKResult, epa, EPAResult
using GeometryBasics: Vec3f
using LinearAlgebra: norm

# Shared narrow-phase helpers — see test/narrow_phase_helpers.jl.
isdefined(@__MODULE__, :translation_transform) ||
    include(joinpath(@__DIR__, "narrow_phase_helpers.jl"))

const ID = translation_transform(0, 0, 0)

# ---------------------------------------------------------------------------
# EPAResult shape
# ---------------------------------------------------------------------------

@testset "EPAResult type fields" begin
    @test fieldnames(EPAResult) == (:normal, :depth, :contact, :iterations, :converged)
    # GPU portability: EPAResult must be isbits so it can be returned from a
    # @kernel and stored in a device-side struct without Box/heap pointers.
    @test isbitstype(EPAResult)
    cube = UnitCube()
    T_B  = translation_transform(1.9, 0, 0)
    g    = gjk(cube, cube, ID, T_B)
    r    = epa(cube, cube, ID, T_B, g.simplex)
    @test r isa EPAResult
    @test r.normal isa Vec3f
    @test r.contact isa Vec3f
    @test r.depth isa Float32
    @test r.iterations isa Int
    @test r.converged isa Bool
end

# ---------------------------------------------------------------------------
# Two unit cubes overlapping 0.1 along +X (the spec's headline case).
# ---------------------------------------------------------------------------

@testset "EPA -- two unit cubes overlapping 0.1 along +X" begin
    cube = UnitCube()
    T_B  = translation_transform(1.9, 0, 0)
    g    = gjk(cube, cube, ID, T_B)
    @test g.overlap == true

    r = epa(cube, cube, ID, T_B, g.simplex)
    @test r.converged == true
    @test isapprox(r.normal[1], 1f0; atol=1f-3)
    @test isapprox(r.normal[2], 0f0; atol=1f-3)
    @test isapprox(r.normal[3], 0f0; atol=1f-3)
    @test isapprox(norm(r.normal), 1f0; atol=1f-4)
    @test isapprox(r.depth, 0.1f0; atol=1f-3)
    # Contact point should lie on cube A's contact face (x = +1).
    @test isapprox(r.contact[1], 1f0; atol=1f-2)
    # ...and on the overlapping y/z slab of A (|y| <= 1, |z| <= 1).
    @test abs(r.contact[2]) <= 1f0 + 1f-3
    @test abs(r.contact[3]) <= 1f0 + 1f-3
end

# ---------------------------------------------------------------------------
# Y-axis overlap.
# ---------------------------------------------------------------------------

@testset "EPA -- offset 1.7 along +Y, depth 0.3" begin
    cube = UnitCube()
    T_B  = translation_transform(0, 1.7, 0)
    g    = gjk(cube, cube, ID, T_B)
    @test g.overlap == true

    r = epa(cube, cube, ID, T_B, g.simplex)
    @test r.converged == true
    @test isapprox(r.normal[2], 1f0; atol=1f-3)
    @test isapprox(r.normal[1], 0f0; atol=1f-3)
    @test isapprox(r.normal[3], 0f0; atol=1f-3)
    @test isapprox(r.depth, 0.3f0; atol=1f-3)
    # Contact is on A's y = +1 face.
    @test isapprox(r.contact[2], 1f0; atol=1f-2)
end

# ---------------------------------------------------------------------------
# Z-axis overlap (sanity for the third axis).
# ---------------------------------------------------------------------------

@testset "EPA -- offset 1.5 along +Z, depth 0.5" begin
    cube = UnitCube()
    T_B  = translation_transform(0, 0, 1.5)
    g    = gjk(cube, cube, ID, T_B)
    @test g.overlap == true

    r = epa(cube, cube, ID, T_B, g.simplex)
    @test r.converged == true
    @test isapprox(abs(r.normal[3]), 1f0; atol=1f-3)
    @test isapprox(r.normal[1], 0f0; atol=1f-3)
    @test isapprox(r.normal[2], 0f0; atol=1f-3)
    @test isapprox(r.depth, 0.5f0; atol=1f-3)
end

# ---------------------------------------------------------------------------
# Diagonal overlap -- shortest separation is along one of two equally-close
# axes, so EPA picks one but both depth and the in-plane projection are
# constrained.
# ---------------------------------------------------------------------------

@testset "EPA -- diagonal overlap (1.5, 1.5, 0): depth 0.5 along an axis" begin
    cube = UnitCube()
    T_B  = translation_transform(1.5, 1.5, 0)
    g    = gjk(cube, cube, ID, T_B)
    @test g.overlap == true

    r = epa(cube, cube, ID, T_B, g.simplex)
    @test r.converged == true
    @test isapprox(norm(r.normal), 1f0; atol=1f-4)
    @test isapprox(r.depth, 0.5f0; atol=1f-3)
    # Penetration normal must lie in the X-Y plane (no Z component).
    @test isapprox(r.normal[3], 0f0; atol=1f-3)
    # Either +X or +Y axis (one component magnitude 1, the other 0).
    nx = abs(r.normal[1])
    ny = abs(r.normal[2])
    @test isapprox(nx, 1f0; atol=1f-3) || isapprox(ny, 1f0; atol=1f-3)
end

# ---------------------------------------------------------------------------
# Rotated cube -- B is 45 deg around Z, translated 1 along +X.
# After rotation, B's world-space +X half-extent is sqrt(2); centered at (1,0,0)
# it spans x in [1-sqrt(2), 1+sqrt(2)].  The shortest penetration along +X
# from A's [-1,+1] is 1 - (1 - sqrt(2)) = sqrt(2).
# ---------------------------------------------------------------------------

@testset "EPA -- rotated 45 deg around Z + translate 1 along +X" begin
    cube  = UnitCube()
    T_B   = rotation_z_transform(Float32(pi/4), 1f0, 0f0, 0f0)
    g     = gjk(cube, cube, ID, T_B)
    @test g.overlap == true

    r = epa(cube, cube, ID, T_B, g.simplex)
    @test r.converged == true
    @test isapprox(norm(r.normal), 1f0; atol=1f-4)
    # Penetration along +X (no Y component, since the configuration is
    # symmetric about the X axis after the 45 deg rotation).
    @test isapprox(r.normal[1], 1f0; atol=1f-3)
    @test isapprox(r.normal[2], 0f0; atol=1f-3)
    @test isapprox(r.normal[3], 0f0; atol=1f-3)
    @test isapprox(r.depth, Float32(sqrt(2)); atol=1f-3)
end

# ---------------------------------------------------------------------------
# Deep overlap -- both cubes coincident.  Penetration depth = 2 (full
# face-to-face thickness); normal is along one of the 6 axes.
# ---------------------------------------------------------------------------

@testset "EPA -- coincident cubes (deepest possible overlap)" begin
    cube = UnitCube()
    g    = gjk(cube, cube, ID, ID)
    @test g.overlap == true

    r = epa(cube, cube, ID, ID, g.simplex)
    @test r.converged == true
    @test isapprox(norm(r.normal), 1f0; atol=1f-4)
    @test isapprox(r.depth, 2f0; atol=1f-3)
    # Normal is exactly one axis: largest component magnitude is 1, others 0.
    largest = max(abs(r.normal[1]), abs(r.normal[2]), abs(r.normal[3]))
    @test isapprox(largest, 1f0; atol=1f-3)
end

# ---------------------------------------------------------------------------
# Iteration cap -- forcing max_iters=1 must return a non-converged result
# without crashing.  EPA is intentionally lenient on cap-out because P4.4
# wants to keep going with whatever the best estimate is.
# ---------------------------------------------------------------------------

@testset "EPA -- iteration cap returns gracefully" begin
    cube = UnitCube()
    T_B  = translation_transform(1.9, 0, 0)
    g    = gjk(cube, cube, ID, T_B)
    r    = epa(cube, cube, ID, T_B, g.simplex; max_iters=1)
    @test r isa EPAResult
    @test r.iterations >= 1
    # Cap-out must report converged=false; depth must be a non-negative best
    # estimate, not a sentinel.
    @test r.converged == false
    @test r.depth >= 0f0
    # Cap-out flag is consistent: if not converged, normal should still be
    # finite (we want a usable best-estimate, not garbage).
    @test all(isfinite, r.normal)
    @test isfinite(r.depth)
end

# ---------------------------------------------------------------------------
# Both cubes have non-trivial transforms.  This catches transform-handling
# bugs in support_pair_world.
# ---------------------------------------------------------------------------

@testset "EPA -- both cubes translated, overlapping along +X" begin
    cube = UnitCube()
    T_A  = translation_transform(-0.5, 0, 0)
    T_B  = translation_transform(1.4, 0, 0)
    # A: x in [-1.5, +0.5];  B: x in [0.4, 2.4].  Overlap along X is
    # [0.4, 0.5] -- depth 0.1.
    g    = gjk(cube, cube, T_A, T_B)
    @test g.overlap == true

    r = epa(cube, cube, T_A, T_B, g.simplex)
    @test r.converged == true
    @test isapprox(r.normal[1], 1f0; atol=1f-3)
    @test isapprox(r.depth, 0.1f0; atol=1f-3)
    # Contact on A's +X face (x = -0.5 + 1 = +0.5 in world).
    @test isapprox(r.contact[1], 0.5f0; atol=1f-2)
end
