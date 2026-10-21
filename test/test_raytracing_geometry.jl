using Test, Mantle, Raycore
using GeometryBasics: Mesh, Point2f, Point3f, Vec3f, GLTriangleFace, per_face

@testset "portable BLAS geometry preparation" begin
    points = Point3f[(0, 0, 0), (1, 0, 0), (0, 1, 0),
                     (2, 0, 0), (3, 0, 0), (2, 1, 0)]
    faces = GLTriangleFace[(1, 2, 3), (1, 1, 2), (4, 5, 6)]
    normals = fill(Vec3f(0, 0, 1), length(points))
    uvs = Point2f[(0, 0), (1, 0), (0, 1), (0.25, 0.25), (0.75, 0.25), (0.25, 0.75)]
    mesh = Mesh(points, faces; normal=normals, uv=uvs)
    tris = Mantle.meshtriangles(Raycore.Triangle{UInt32}, mesh)
    @test length(tris) == 2
    # Removing a degenerate face must not renumber the remaining metadata.
    @test getproperty.(tris, :metadata) == UInt32[1, 3]
    @test Tuple(tris[1].vertices) == Tuple(points[1:3])
    @test Tuple(tris[2].vertices) == Tuple(points[4:6])
    @test all(n -> n == Vec3f(0, 0, 1), tris[2].normals)
    @test Tuple(tris[2].uv) == Tuple(uvs[4:6])

    meta = [(material=UInt32(i), part=UInt16(9)) for i in (7, 8, 11)]
    attributed = Mesh(points, faces; normal=normals, uv=uvs, face_meta=per_face(meta, faces))
    typed = Mantle.meshtriangles(Raycore.Triangle{eltype(meta)}, attributed)
    @test getproperty.(typed, :metadata) == meta[[1, 3]]
    @test Tuple(typed[2].uv) == Tuple(uvs[4:6])

    vertices = Mantle.trianglevertices(typed)
    @test eltype(vertices) === NTuple{3,Float32}
    @test vertices == Tuple.(points)
    # Metal consumes these exact bytes as Float32; Vulkan consumes the tuples.
    flat = reinterpret(Float32, vertices)
    @test collect(flat) == Float32[0, 0, 0, 1, 0, 0, 0, 1, 0,
                                 2, 0, 0, 3, 0, 0, 2, 1, 0]
    @test sizeof(vertices) == 72
    GC.@preserve vertices flat begin
        @test Ptr{Float32}(pointer(vertices)) == pointer(flat)
    end

    no_uv = Mesh(points, faces; normal=normals)
    defaults = Mantle.meshtriangles(Raycore.Triangle{UInt32}, no_uv)
    @test Tuple(defaults[1].uv) == (Point2f(0, 0), Point2f(1, 0), Point2f(1, 1))
    degenerate = Mesh(points, GLTriangleFace[(1, 1, 2)]; normal=normals)
    @test_throws ErrorException("Geometry has no valid triangles") Mantle.meshtriangles(Raycore.Triangle{UInt32}, degenerate)
end

mutable struct MetadataBatch
    triangles::Vector{Int}
    n::Int
    handle::Raycore.TLASHandle
    hidden::Bool
end
Base.length(b::MetadataBatch) = b.n

@testset "portable TLAS triangle tables" begin
    bs = Mantle.InstanceBatches{MetadataBatch}()
    empty = Mantle.trianglemetadata(Int, bs, Val(true))
    @test isempty(empty.triangles) && isempty(empty.hits)
    @test isempty(empty.raw) && isempty(empty.starts)
    add(tris, n) = Mantle.register!(bs, h -> MetadataBatch(tris, n, h, false))
    leading = add(Int[], 2)
    first = add([11, 12], 3)
    middle = add(Int[], 1)
    add([21], 0) # Geometry without live instances still occupies metadata space.
    last = add([31, 32, 33], 2)
    @test Mantle.sethidden!(bs, first, true)
    result = Mantle.trianglemetadata(Int, bs, Val(true))
    @test result.triangles == [11, 12, 21, 31, 32, 33]
    @test result.hits == UInt32[Mantle.NOTRIANGLES, Mantle.NOTRIANGLES,
                                0, 0, 0, Mantle.NOTRIANGLES, 3, 3]
    @test result.raw == UInt32[0, 0, 0, 0, 0, 2, 3, 3]
    @test result.starts == UInt32[0, 0, 2, 2, 3]
    hitonly = Mantle.trianglemetadata(Int, bs)
    @test hitonly.triangles == result.triangles && hitonly.hits == result.hits
    @test hitonly.raw === nothing && hitonly.starts === nothing
    @test Mantle.delete!(bs, first)
    changed = Mantle.trianglemetadata(Int, bs, Val(true))
    @test changed.triangles == [21, 31, 32, 33]
    @test changed.hits == UInt32[Mantle.NOTRIANGLES, Mantle.NOTRIANGLES,
                                 Mantle.NOTRIANGLES, 1, 1]
    @test changed.starts == UInt32[0, 0, 0, 1]
    @test changed.raw == UInt32[0, 0, 0, 1, 1]
    @test Mantle.batchof(bs, last).handle == last
end

import Adapt
struct ProjectionAdaptor
    visited::Vector{Symbol}
end
function Adapt.adapt_storage(to::ProjectionAdaptor, values::Vector{Symbol})
    name = only(values)
    push!(to.visited,name)
    return (:device,name)
end

@testset "acceleration projection preserves scene policy and conversion order" begin
    owner = Ref{Any}(); owner[] = owner
    empty = (:empty,9)
    accel = Mantle.AdaptedAccel(owner,[:triangles],[:offsets],empty,
                               [:scene],[:procedural],[:instances])
    for convert_scene in (false,true)
        to = ProjectionAdaptor(Symbol[])
        conversion = convert_scene ? Base.Fix1(Adapt.adapt,to) : identity
        projected = Mantle.adaptaccel(to,accel,conversion)
        @test projected.hwtlas === nothing
        @test projected.empty === empty
        for field in (:triangles,:offsets,:procedural,:instances)
            @test getproperty(projected,field) == (:device,field)
        end
        expected_scene = convert_scene ? (:device,:scene) : accel.scene
        @test projected.scene === expected_scene
        @test to.visited == (convert_scene ? [:triangles,:offsets,:scene,:procedural,:instances] :
                                                            [:triangles,:offsets,:procedural,:instances])
    end
end
