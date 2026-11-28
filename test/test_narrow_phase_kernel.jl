using Test, Mantle
using Mantle: UnitCube, EPAResult, narrow_phase_kernel, NO_CONTACT, gjk, epa
using GeometryBasics: Vec3f
include(joinpath(@__DIR__, "testbackend.jl"))

# Shared narrow-phase helpers — see test/narrow_phase_helpers.jl.  Provides
# `tx` (alias for `translation_transform`) and the other transform builders.
isdefined(@__MODULE__, :tx) ||
    include(joinpath(@__DIR__, "narrow_phase_helpers.jl"))

# What `results` holds before the kernel runs: every field off what the kernel
# writes, so a slot it skipped cannot pass for one it wrote.
narrow_sentinel() = EPAResult(Vec3f(99, 99, 99), 99f0, Vec3f(88, 88, 88), 77, false)

# Through a graph on the device, the one way this API runs a kernel. This ran on
# `KA.CPU` until the kernel became a plain `KernelInterface` function, which
# that backend cannot compile; the host reference the kernel is held to is
# `gjk`/`epa` called directly, in the last testset below.
function run_narrow_phase(transforms, pairs, shape)
    dev = Mantle.Device(TESTBACKEND)
    t = Mantle.Buffer(dev, transforms)
    p = Mantle.Buffer(dev, pairs)
    r = Mantle.Buffer(dev, fill(narrow_sentinel(), length(pairs)))
    g = Mantle.Graph(dev)
    Mantle.dispatch!(g, narrow_phase_kernel, (t, p, shape, r), length(pairs))
    Mantle.runonce!(g)
    results = Array(Mantle.storage(r))
    foreach(Mantle.free!, (t, p, r))
    return results
end

# Local helper, avoids depending on a particular norm overload.
norm_squared(v::Vec3f) = v[1]*v[1] + v[2]*v[2] + v[3]*v[3]

@testset "narrow_phase_kernel" begin

    @testset "EPAResult is bitstype (precondition for GPU dispatch)" begin
        @test isbitstype(EPAResult)
    end

    @testset "NO_CONTACT sentinel has depth == 0f0" begin
        @test NO_CONTACT.depth == 0f0
        @test NO_CONTACT.normal == Vec3f(0f0, 0f0, 0f0)
        @test NO_CONTACT.contact == Vec3f(0f0, 0f0, 0f0)
    end

    @testset "single overlapping pair (0.1 along +X)" begin
        transforms = [tx(0,0,0), tx(1.9, 0, 0)]
        pairs      = [(Int32(1), Int32(2))]

        results = run_narrow_phase(transforms, pairs, UnitCube())

        r = results[1]
        @test r.converged
        @test r.depth ≈ 0.1f0 atol=1f-3
        @test r.normal[1] ≈ 1f0 atol=1f-3
        @test abs(r.normal[2]) < 1f-3
        @test abs(r.normal[3]) < 1f-3
        # Contact lies on the +X face of cube A near x=1.
        @test r.contact[1] ≈ 1f0 atol=1f-2
    end

    @testset "single separated pair → NO_CONTACT sentinel" begin
        transforms = [tx(0,0,0), tx(5, 0, 0)]
        pairs      = [(Int32(1), Int32(2))]

        results = run_narrow_phase(transforms, pairs, UnitCube())

        r = results[1]
        @test r.depth == 0f0
        @test r.normal == Vec3f(0f0, 0f0, 0f0)
        @test r.contact == Vec3f(0f0, 0f0, 0f0)
    end

    @testset "batch of 6 pairs from a 4-cube square (all overlap)" begin
        # Four unit cubes (extents +/-1) at the corners of a 1.9-side square in the
        # XY plane.  Adjacent cubes overlap by 0.1 along one axis; diagonal cubes
        # overlap at a 0.1x0.1 corner -- ALL six C(4,2) pairs overlap.
        # Verified by calling gjk() on each pair directly (depth ≈ 0.1f0 in
        # every case).
        transforms = [tx(0,0,0), tx(1.9,0,0), tx(0,1.9,0), tx(1.9,1.9,0)]
        pairs = [
            (Int32(1), Int32(2)),  # adjacent X
            (Int32(1), Int32(3)),  # adjacent Y
            (Int32(2), Int32(4)),  # adjacent Y
            (Int32(3), Int32(4)),  # adjacent X
            (Int32(1), Int32(4)),  # diagonal
            (Int32(2), Int32(3)),  # mirror diagonal
        ]

        results = run_narrow_phase(transforms, pairs, UnitCube())

        # Every pair overlaps; depth on each is ~0.1.
        for (k, r) in enumerate(results)
            @test r.depth > 0f0
            @test r.depth ≈ 0.1f0 atol=2f-2
            # Normal is a unit vector along an axis (one of +/-X, +/-Y).
            @test abs(norm_squared(r.normal) - 1f0) < 1f-3
        end
    end

    @testset "all-separated batch writes NO_CONTACT in every slot" begin
        n = 8
        # Cubes spaced 5 apart along X — way more than 2 (the cube full extent).
        transforms = [tx(5*(i-1), 0, 0) for i in 1:n]
        pairs = [(Int32(i), Int32(i+1)) for i in 1:(n-1)]

        results = run_narrow_phase(transforms, pairs, UnitCube())

        @test all(r -> r.depth == 0f0,                   results)
        @test all(r -> r.normal  == Vec3f(0f0,0f0,0f0),  results)
        @test all(r -> r.contact == Vec3f(0f0,0f0,0f0),  results)
    end

    @testset "mixed batch (some overlap, some separated)" begin
        # 4 transforms: pair-1 overlaps, pair-2 doesn't, pair-3 overlaps.
        transforms = [tx(0,0,0), tx(1.9, 0, 0), tx(10, 0, 0), tx(11.9, 0, 0)]
        pairs = [
            (Int32(1), Int32(2)),  # overlap depth ~0.1
            (Int32(2), Int32(3)),  # separated (1.9 -> 10)
            (Int32(3), Int32(4)),  # overlap depth ~0.1
        ]

        results = run_narrow_phase(transforms, pairs, UnitCube())

        @test results[1].depth ≈ 0.1f0 atol=2f-2
        @test results[2].depth == 0f0
        @test results[3].depth ≈ 0.1f0 atol=2f-2
    end

    @testset "kernel matches direct gjk+epa composition (sanity)" begin
        # Same inputs through the kernel and through a direct call should give
        # bitwise-identical EPAResult on overlapping pairs and NO_CONTACT for
        # separated pairs.  This guards against the kernel accidentally
        # mutating arguments / using the wrong simplex / etc.
        transforms = [tx(0,0,0), tx(1.95, 0.5, 0), tx(8, 0, 0)]
        pairs      = [(Int32(1), Int32(2)), (Int32(1), Int32(3))]

        results = run_narrow_phase(transforms, pairs, UnitCube())

        # Pair 1: overlap -> compare against direct epa.
        T1, T2 = transforms[1], transforms[2]
        g1 = gjk(UnitCube(), UnitCube(), T1, T2)
        @test g1.overlap
        ref1 = epa(UnitCube(), UnitCube(), T1, T2, g1.simplex)
        @test results[1].depth   ≈ ref1.depth   atol=1f-5
        @test results[1].normal  ≈ ref1.normal  atol=1f-5
        @test results[1].contact ≈ ref1.contact atol=1f-5

        # Pair 2: separated -> sentinel.
        @test results[2].depth == 0f0
    end
end
