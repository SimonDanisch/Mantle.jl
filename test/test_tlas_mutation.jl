# Core's batch mutation of the hardware TLAS (`raytracing/mutation.jl`), on every backend:
# registration, validation before any geometry is built, visibility, queued transform updates
# and delete, checked on the batches themselves rather than on a trace.
using Test, Mantle, Raycore, GeometryBasics, LinearAlgebra
include(joinpath(@__DIR__, "testbackend.jl"))

const TestTri = Raycore.Triangle{UInt32}

if Mantle.supports_hwtlas(TESTBACKEND)
@testset "incremental TLAS mutations" begin
    t = Mantle.HWTLAS{TestTri}(TESTBACKEND)
    mesh = GeometryBasics.Mesh(Point3f[(0,0,0), (1,0,0), (0,1,0)],
                               GLTriangleFace[(1,2,3)])
    transforms = fill(Mat4f(I), 2)
    records = Mantle.instancerecords(transforms, UInt32[11,12], UInt8(0x42))
    @test_throws ArgumentError push!(t, mesh, records; n=-1)
    @test_throws ArgumentError push!(t, mesh, records; n=3)
    @test_throws ArgumentError push!(t, mesh, transforms; instance_ids=UInt32[1])
    @test isempty(t.instances) && isempty(t.blas_list)

    first = push!(t, mesh, records; n=1)
    @test t.dirty
    @test Raycore.n_instances(t) == 1
    @test Raycore.n_geometries(t) == 1
    @test Array(Raycore.instance_buffer(t, first)) == records
    blas = only(t.blas_list)
    second = push!(t, blas, transforms; instance_ids=UInt32[21,22])
    @test Raycore.n_geometries(t) == 1
    @test Raycore.n_instances(t) == 3
    t.dirty = false
    @test Raycore.set_visible!(t, first, false)
    @test t.dirty && Mantle.batchof(t.instances, first).hidden
    t.dirty = false
    @test Raycore.set_visible!(t, first, false)
    @test !t.dirty
    @test_throws ArgumentError Raycore.update_transforms!(t, second, transforms[1:1])
    @test_throws ErrorException Raycore.update_instance_transforms!(t, transforms, 2, 1)
    @test isempty(t.pending) && !t.transforms_dirty
    moved = Mat4f(1,0,0,0, 0,1,0,0, 0,0,1,0, 2,3,4,1)
    Raycore.update_instance_transforms!(t, fill(moved,2), 2, 2)
    @test length(t.pending) == 1 && t.transforms_dirty
    # Queued, written at the sync, keeping each record's own custom index.
    Raycore.sync!(t)
    @test isempty(t.pending)
    updated = Array(Raycore.instance_buffer(t, second))
    @test getproperty.(updated,:id) == UInt32[21,22]
    @test all(r -> r.transform == Raycore.mat4_to_mat3x4(moved), updated)
    @test delete!(t, first)
    @test !delete!(t, first)
    @test_throws ArgumentError Raycore.instance_buffer(t, first)
    @test_throws ArgumentError Raycore.update_transform!(t, first, moved)
    @test !Raycore.set_visible!(t, first, true)
    third = push!(t, blas)
    @test third != first && third != second
    @test Raycore.n_instances(t) == 3
    Raycore.sync!(t)
    @test isempty(Mantle.validationmessages!(TESTBACKEND))
end

@testset "queued device transforms allocate nothing" begin
    mesh = Mesh(Point3f[(0,0,0), (1,0,0), (0,1,0)], GLTriangleFace[(1,2,3)])
    t = Mantle.HWTLAS{TestTri}(TESTBACKEND)
    h = push!(t, mesh, fill(Mat4f(I), 2);
              instance_ids=UInt32[17,29], instance_mask=UInt8(0x42))
    transforms = Mantle.devicearray(TESTBACKEND, fill(Mantle.identity_transform(), 2))
    Raycore.update_transforms!(t, h, transforms)
    Raycore.update_transforms!(t, h, transforms)
    @test (@allocated Raycore.update_transforms!(t, h, transforms)) == 0
    Raycore.sync!(t)
    records = Array(Raycore.instance_buffer(t, h))
    @test getproperty.(records, :id) == UInt32[17,29]
    @test getproperty.(records, :mask) == UInt32[0x42,0x42]
    @test isempty(Mantle.validationmessages!(TESTBACKEND))
end
end
