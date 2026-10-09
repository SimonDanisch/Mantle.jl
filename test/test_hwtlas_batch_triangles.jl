# The per-triangle data a hit reads: `sync!` concatenates every batch's triangles
# into one array and gives each INSTANCE an offset into it, so the triangle a ray
# lands on is `triangles[offset(instance) + primitive]`, moved by the instance's
# transform. All instances of a batch share one offset; a later batch's start
# past every earlier batch's triangles, including a batch that has none.
#
# Asserted through what `closest_hit` hands back — the hit triangle's world-space
# centroid against the triangle a host reference says the ray crossed — rather
# than through the offset array itself: a wrong offset gives a ray another
# batch's triangle, moved to this instance's place, and that is what a shader
# would read.

using Test, Raycore, Mantle, Adapt
using GeometryBasics
using GeometryBasics: Point3f, Vec3f, GLTriangleFace
include(joinpath(@__DIR__, "testbackend.jl"))
isdefined(@__MODULE__, :tlasprobe!) || include(joinpath(@__DIR__, "hwtlas_helpers.jl"))

const Tri = Raycore.Triangle{UInt32}

# One triangle, and a quad of two, at z = 0. Different shapes, so a hit that read
# the other batch's triangle has a different centroid.
const ONE_TRI = [(Point3f(0, 0, 0), Point3f(1, 0, 0), Point3f(0, 1, 0))]
const TWO_TRIS = [(Point3f(0, 0, 0), Point3f(1, 0, 0), Point3f(1, 1, 0)),
                  (Point3f(0, 0, 0), Point3f(1, 1, 0), Point3f(0, 1, 0))]

meshof(tris) = GeometryBasics.normal_mesh(GeometryBasics.Mesh(
    [v for t in tris for v in t],
    [GLTriangleFace(3k - 2, 3k - 1, 3k) for k in eachindex(tris)]))

# Rays straight down from z = 1 through a point well inside each triangle of
# each instance, and the world-space triangles they should find.
inside(tri) = Point3f(tricentroid(tri)[1], tricentroid(tri)[2], 1f0)

"""Trace one ray per (instance, triangle) and compare every hit with the host
reference: hit or miss, distance, and WHICH triangle, by its centroid."""
function check_batches(tlas, batches)
    world = [movedby(tri, off) for (tris, offsets) in batches for off in offsets for tri in tris]
    origins = [inside(w) for w in world]
    dirs = fill(Vec3f(0, 0, -1), length(origins))
    r = tlastrace(TESTBACKEND, tlas, origins, dirs)
    ref = [nearesthit(o, d, world) for (o, d) in zip(origins, dirs)]
    @test all(first, ref)                         # the reference itself hits everything
    @test r.hit == first.(ref)
    @test all(isapprox.(r.t, [x[2] for x in ref]; atol = 1f-3))
    @test all(isapprox.(r.centroid, [tricentroid(world[x[3]]) for x in ref]; atol = 1f-4))
    return r
end

if Mantle.supports_hwtlas(TESTBACKEND)

@testset "a batch's instances share its triangles, and the next batch starts past them" begin
    offs_a = [(Float32(2i), 0f0, 0f0) for i in 1:8]
    offs_b = [(Float32(2i), 5f0, 0f0) for i in 1:4]
    tlas = Mantle.HWTLAS{Tri}(TESTBACKEND)
    handle = push!(tlas, meshof(ONE_TRI), [tlastranslation(o...) for o in offs_a])
    @test handle isa Raycore.TLASHandle
    push!(tlas, meshof(TWO_TRIS), [tlastranslation(o...) for o in offs_b])
    Raycore.sync!(tlas)
    @test Raycore.n_instances(tlas) == length(offs_a) + length(offs_b)
    check_batches(tlas, [(ONE_TRI, offs_a), (TWO_TRIS, offs_b)])
end

@testset "a batch with no triangles syncs, and the batches after it still find theirs" begin
    # A procedural BLAS has no triangles: its batch adds nothing to the triangle
    # array, so the offsets behind it must not move. This is also the push a
    # caller with no per-triangle data makes, and the structure still has to build
    # and adapt — a trace binds the triangle array whatever the scene holds.
    aabb = Mantle.AABB(Point3f(-1f0, -1f0, -1f0), Point3f(1f0, 1f0, 1f0))
    blas = Mantle.build_accel!(Mantle.batchqueue(Mantle.Device(TESTBACKEND))) do ctx
        Mantle.build_blas_aabb(ctx, [aabb])
    end
    offs_b = [(Float32(2i), 5f0, 0f0) for i in 1:4]
    tlas = Mantle.HWTLAS{Tri}(TESTBACKEND)
    # Far from every ray below: nothing here answers a box.
    @test push!(tlas, blas, tlastranslation(-20f0, 0f0, 0f0)) isa Raycore.TLASHandle
    push!(tlas, meshof(TWO_TRIS), [tlastranslation(o...) for o in offs_b])
    Raycore.sync!(tlas)
    @test Raycore.n_instances(tlas) == 1 + length(offs_b)
    @test Adapt.adapt(TESTBACKEND, tlas) isa Mantle.AdaptedAccel
    check_batches(tlas, [(TWO_TRIS, offs_b)])
end

else
    @testset "build_blas_aabb errors loudly without hardware acceleration structures" begin
        # Saying so beats a driver-level fault. This used to be checked by
        # switching a Vulkan context's ray-query flag off by hand.
        @test_throws ErrorException Mantle.build_accel!(Mantle.batchqueue(Mantle.Device(TESTBACKEND))) do ctx
            Mantle.build_blas_aabb(ctx, [Mantle.AABB(Point3f(0, 0, 0), Point3f(1, 1, 1))])
        end
    end
end
