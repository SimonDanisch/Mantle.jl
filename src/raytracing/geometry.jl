"""Prepare nondegenerate shading triangles, preserving face metadata and order."""
function meshtriangles(::Type{Tri}, mesh::GeometryBasics.Mesh) where {Tri}
    nmesh = GeometryBasics.expand_faceviews(mesh)
    fs = GeometryBasics.decompose(GeometryBasics.TriangleFace{UInt32}, nmesh)
    verts = GeometryBasics.decompose(GeometryBasics.Point3f, nmesh)
    norms = Raycore.Normal3f.(Raycore.decompose_normals(nmesh))
    uvs_raw = GeometryBasics.decompose_uv(nmesh)
    uvs = isnothing(uvs_raw) ? GeometryBasics.Point2f[] : GeometryBasics.Point2f.(uvs_raw)
    indices = collect(reinterpret(UInt32, fs))
    has_meta = hasproperty(nmesh, :face_meta)
    triangles = Tri[]
    for i in eachindex(fs)
        Raycore.is_degenerate_face(verts, indices, i) && continue
        meta = has_meta ? nmesh.face_meta[indices[3*(i-1)+1]] : UInt32(i)
        push!(triangles, Raycore.build_triangle(verts, norms, uvs, indices, i, meta))
    end
    isempty(triangles) && error("Geometry has no valid triangles")
    return triangles
end

"""Expand shading triangles into consecutive Float32 vertices for a native BLAS."""
function trianglevertices(triangles)
    vertices = Vector{NTuple{3,Float32}}(undef, 3 * length(triangles))
    @inbounds for i in eachindex(triangles), j in 1:3
        v = triangles[i].vertices[j]
        vertices[3*(i-1)+j] = (Float32(v[1]), Float32(v[2]), Float32(v[3]))
    end
    return vertices
end
