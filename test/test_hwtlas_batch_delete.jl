# P3-fu2: `delete!(hwtlas, handle)` for batch handles.
#
# What a caller sees of a delete is what a trace sees after the next `sync!`: the
# deleted batch is gone, every other one is still where it was, and the handles
# of the others still name THEM. That last part is the reindex core's
# `InstanceBatches` does on delete — its map holds positions, so removing a batch
# from the middle moves every entry behind it, and a handle that kept its old
# position would update or delete someone else's instances.

using Test, Raycore, Mantle, Adapt
using GeometryBasics
using GeometryBasics: Point3f, Vec3f, GLTriangleFace
include(joinpath(@__DIR__, "testbackend.jl"))
isdefined(@__MODULE__, :tlasprobe!) || include(joinpath(@__DIR__, "hwtlas_helpers.jl"))

const Tri = Raycore.Triangle{UInt32}

"""One triangle at z = 0: (0,0,0), (1,0,0), (0,1,0)."""
unittriangle() = GeometryBasics.normal_mesh(GeometryBasics.Mesh(
    [Point3f(0, 0, 0), Point3f(1, 0, 0), Point3f(0, 1, 0)], [GLTriangleFace(1, 2, 3)]))

"""One ray straight down onto each instance placed at `offsets[k]`, well inside it,
from one unit above: a hit is at `t = 1`."""
downonto(offsets) = TLASProbe(TESTBACKEND,
    [Point3f(o[1] + 0.25f0, o[2] + 0.25f0, o[3] + 1f0) for o in offsets],
    fill(Vec3f(0, 0, -1), length(offsets)))

if Mantle.supports_hwtlas(TESTBACKEND)

@testset "delete!(hwtlas, batch_handle) removes the batch" begin
    n = 4
    offsets = [(Float32(2i), 0f0, 0f0) for i in 1:n]
    tlas = Mantle.HWTLAS{Tri}(TESTBACKEND)
    handle = push!(tlas, unittriangle(), [tlastranslation(o...) for o in offsets];
                   instance_ids = fill(UInt32(4), n), instance_mask = UInt8(0x04))
    Raycore.sync!(tlas)
    @test Raycore.n_instances(tlas) == n
    probe = downonto(offsets)
    r = tlastrace(probe, tlas)
    @test all(r.hit)
    @test all(==(UInt32(4)), r.id)
    @test all(t -> isapprox(t, 1f0; atol = 1f-3), r.t)

    @test delete!(tlas, handle) == true
    Raycore.sync!(tlas)
    @test Raycore.n_instances(tlas) == 0
    # Gone from the traversal, not only from the count. The structure is now
    # EMPTY, and an empty one is still built and still traced: every ray misses.
    @test !any(tlastrace(probe, tlas).hit)

    # Deleting again returns false: the handle names no batch any more.
    @test delete!(tlas, handle) == false
end

@testset "delete! returns false for unknown handle" begin
    tlas = Mantle.HWTLAS{Tri}(TESTBACKEND)
    fake_handle = Raycore.TLASHandle(UInt32(99))
    @test delete!(tlas, fake_handle) == false
end

@testset "delete!(hwtlas, batch_handle) leaves siblings alone" begin
    n = 4
    at_a = [(Float32(2i), 0f0, 0f0) for i in 1:n]
    at_b = [(Float32(2i), 3f0, 0f0) for i in 1:n]
    tlas = Mantle.HWTLAS{Tri}(TESTBACKEND)
    handle_a = push!(tlas, unittriangle(), [tlastranslation(o...) for o in at_a];
                     instance_ids = fill(UInt32(2), n), instance_mask = UInt8(0x02))
    handle_b = push!(tlas, unittriangle(), [tlastranslation(o...) for o in at_b];
                     instance_ids = fill(UInt32(4), n), instance_mask = UInt8(0x04))
    Raycore.sync!(tlas)
    @test Raycore.n_instances(tlas) == 2n
    probe = downonto(vcat(at_a, at_b))
    r = tlastrace(probe, tlas)
    @test all(r.hit)
    @test r.id == vcat(fill(UInt32(2), n), fill(UInt32(4), n))

    @test delete!(tlas, handle_a) == true
    Raycore.sync!(tlas)
    @test Raycore.n_instances(tlas) == n
    r = tlastrace(probe, tlas)
    @test !any(r.hit[1:n])                       # a is gone
    @test all(r.hit[n+1:end])                    # b is untouched
    @test all(==(UInt32(4)), r.id[n+1:end])

    # b moved from the second slot to the first. Its handle must still name it:
    # moving b through `handle_b` moves b, and deleting it deletes b.
    moved = [(o[1], o[2] + 10f0, o[3]) for o in at_b]
    Raycore.update_transforms!(tlas, handle_b, [tlastranslation(o...) for o in moved])
    Raycore.sync!(tlas)
    @test !any(tlastrace(probe, tlas).hit)       # neither old place is hit
    r = tlastrace(downonto(moved), tlas)
    @test all(r.hit)
    @test all(==(UInt32(4)), r.id)
    @test delete!(tlas, handle_b) == true
    Raycore.sync!(tlas)
    @test Raycore.n_instances(tlas) == 0
end

else
    @info "no hardware acceleration structures on this device; skipping" file = basename(@__FILE__)
end
