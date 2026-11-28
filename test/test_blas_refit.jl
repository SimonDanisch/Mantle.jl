# BLAS refit: `Raycore.refit!(blas, vertices)` moves a bottom-level structure's
# vertices in place.
#
# A triangle in the XY plane, rays from z = -1 along +z, so the hit distance IS
# the triangle's z plus one, which makes "did the acceleration structure
# actually move" a number rather than a judgement call.
#
# That matters here specifically: rewriting the vertex BUFFER is easy to verify
# and proves nothing, because the structure holds its own copy of the geometry.
# If the refit issued a malformed or ineffective build, a buffer check would
# still pass and rays would keep hitting the stale plane.
#
# The top-level structure stores the bounds of what it instances, so it is
# refit too (`Raycore.refit!(tlas)`), as the contract says a caller must.

using Test, Mantle, Raycore, Adapt
using GeometryBasics: Point3f, Vec3f
include(joinpath(@__DIR__, "testbackend.jl"))
isdefined(@__MODULE__, :tlasprobe!) || include(joinpath(@__DIR__, "hwtlas_helpers.jl"))

if Mantle.supports_hwtlas(TESTBACKEND)

const INDICES = UInt32[0, 1, 2]
at_z(z) = [(0.0f0, 0.0f0, z), (1.0f0, 0.0f0, z), (0.0f0, 1.0f0, z)]
buildblas(verts; allow_update) = Mantle.build_accel!(Mantle.batchqueue(Mantle.Device(TESTBACKEND))) do ctx
    Mantle.build_blas(ctx, verts, INDICES; allow_update)
end

@testset "BLAS refit moves the acceleration structure" begin
    blas = buildblas(at_z(0.0f0); allow_update = true)
    tlas = Mantle.HWTLAS{Raycore.Triangle{UInt32}}(TESTBACKEND)
    push!(tlas, blas)
    Raycore.sync!(tlas)

    # Centre ray: (0.25, 0.25) is comfortably inside the triangle for any edge
    # rule, so it is the sample to trust.
    probe = TLASProbe(TESTBACKEND, [Point3f(0.25f0, 0.25f0, -1f0)], [Vec3f(0, 0, 1)])
    t_before = only(tlastrace(probe, tlas).t)
    @test t_before ≈ 1.0f0 atol = 1.0f-3   # plane at z=0, origin at z=-1

    # Move the plane to z=5 and refit in place. Same topology.
    @test Raycore.refit!(blas, at_z(5.0f0)) === blas
    Raycore.refit!(tlas)

    t_after = only(tlastrace(probe, tlas).t)
    @test t_after ≈ 6.0f0 atol = 1.0f-3    # plane at z=5, origin at z=-1
    @test t_after != t_before
end

@testset "refit rejects what an in-place update cannot do" begin
    verts = at_z(0.0f0)
    # A BLAS built without `allow_update` cannot be refit, and saying so beats a
    # driver-level fault or a silently ignored build.
    static_blas = buildblas(verts; allow_update = false)
    @test_throws ArgumentError Raycore.refit!(static_blas, verts)

    # Topology is fixed at build: a refit keeps the existing tree.
    dyn_blas = buildblas(verts; allow_update = true)
    @test_throws ArgumentError Raycore.refit!(dyn_blas, vcat(verts, verts))
end

else
    @info "no hardware acceleration structures on this backend; skipping"
end
