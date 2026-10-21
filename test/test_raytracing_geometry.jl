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
