using Test, Mantle, Raycore, GeometryBasics, LinearAlgebra

mutable struct TestBLAS
    vertices::Vector{NTuple{3,Float32}}
end
const TestTri = Raycore.Triangle{UInt32}
const TestBatch{G} = Mantle.InstanceBatch{TestTri,G,Vector{Raycore.InstanceRecord}}
mutable struct TestBatchedTLAS{G} <: Mantle.BatchedTLAS{TestTri}
    blas_list::Vector{TestBLAS}
    instances::Mantle.InstanceBatches{TestBatch{G}}
    root_aabb::Raycore.Bounds3
    dirty::Bool
    transforms_dirty::Bool
    builds::Int
    updates::Int
end
TestBatchedTLAS{G}() where {G} = TestBatchedTLAS{G}(TestBLAS[],
    Mantle.InstanceBatches{TestBatch{G}}(), Raycore.Bounds3(), false, false, 0, 0)
Mantle.geometryref(::TestBatchedTLAS{Int}, i::Int) = i
function Mantle.buildgeometry!(t::TestBatchedTLAS, vertices)
    t.builds += 1
    return TestBLAS(vertices)
end
Mantle.instancearray(::TestBatchedTLAS, records::AbstractVector) = collect(records)
function Mantle.updatebatch!(t::TestBatchedTLAS, batch, transforms)
    t.updates += 1
    for i in eachindex(transforms)
        old = batch.records[i]
        batch.records[i] = Raycore.InstanceRecord(transforms[i], old.id, old.mask)
    end
    return nothing
end

@testset "shared incremental TLAS mutations ($G)" for G in (TestBLAS, Int)
    t = TestBatchedTLAS{G}()
    mesh = GeometryBasics.Mesh(Point3f[(0,0,0), (1,0,0), (0,1,0)],
                               GLTriangleFace[(1,2,3)])
    transforms = fill(Mat4f(I), 2)
    records = Mantle.instancerecords(transforms, UInt32[11,12], UInt8(0x42))
    @test_throws ArgumentError push!(t, mesh, records; n=-1)
    @test_throws ArgumentError push!(t, mesh, records; n=3)
    @test_throws ArgumentError push!(t, mesh, transforms; instance_ids=UInt32[1])
    @test t.builds == 0 && isempty(t.instances) && isempty(t.blas_list)

    first = push!(t, mesh, records; n=1, sbt_offset=UInt32(7))
    @test t.builds == 1 && t.dirty
    @test Raycore.n_instances(t) == 1
    @test Raycore.n_geometries(t) == 1
    @test Raycore.instance_buffer(t, first) == records
    @test Mantle.batchof(t.instances, first).sbt_offset == 7
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
    @test t.updates == 0 && !t.transforms_dirty
    moved = Mat4f(1,0,0,0, 0,1,0,0, 0,0,1,0, 2,3,4,1)
    Raycore.update_instance_transforms!(t, fill(moved,2), 2, 2)
    @test t.updates == 1 && t.transforms_dirty
    updated = Raycore.instance_buffer(t, second)
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
end
