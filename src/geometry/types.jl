# Geometry-type vocabulary for acceleration-structure builds.
#
# Replaces the older `geometry_type::Symbol` keyword on pack_geometry!,
# query_as_build_sizes, and build_as_on_gpu with proper Julia types.
# Methods dispatch on the concrete subtype.

abstract type GeometryType end

"""
    TrianglesGeometry(vertex_format, vertex_addr, vertex_stride, max_vertex,
                      index_type, index_addr, transform_addr=UInt64(0))

Triangle geometry for a BLAS. Maps to `VK_GEOMETRY_TYPE_TRIANGLES_KHR`.
"""
struct TrianglesGeometry <: GeometryType
    vertex_format::UInt32
    vertex_addr::UInt64
    vertex_stride::UInt64
    max_vertex::UInt32
    index_type::UInt32
    index_addr::UInt64
    transform_addr::UInt64
end
TrianglesGeometry(; vertex_format, vertex_addr, vertex_stride, max_vertex,
                    index_type, index_addr, transform_addr=UInt64(0)) =
    TrianglesGeometry(vertex_format, vertex_addr, vertex_stride, max_vertex,
                      index_type, index_addr, transform_addr)

"""
    AABBsGeometry(aabb_addr, aabb_stride=UInt64(24))

Procedural-AABB geometry for a BLAS. Maps to `VK_GEOMETRY_TYPE_AABBS_KHR`.
The buffer at `aabb_addr` contains `n` `VkAabbPositionsKHR` records of
`aabb_stride` bytes each (24 by default: 6 x Float32).
"""
struct AABBsGeometry <: GeometryType
    aabb_addr::UInt64
    aabb_stride::UInt64
end
AABBsGeometry(; aabb_addr, aabb_stride=UInt64(24)) =
    AABBsGeometry(aabb_addr, aabb_stride)

"""
    AABB(min::Point3f, max::Point3f)

A 3D axis-aligned bounding box, stored as two corner points. Used as the
input element type for `build_blas_aabb`.
"""
struct AABB
    min::Point3f
    max::Point3f
end
