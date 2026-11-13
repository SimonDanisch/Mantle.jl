"""Expand shading triangles into consecutive Float32 vertices for a native BLAS."""
function trianglevertices(triangles)
    vertices = Vector{NTuple{3,Float32}}(undef, 3 * length(triangles))
    @inbounds for i in eachindex(triangles), j in 1:3
        v = triangles[i].vertices[j]
        vertices[3*(i-1)+j] = (Float32(v[1]), Float32(v[2]), Float32(v[3]))
    end
    return vertices
end
