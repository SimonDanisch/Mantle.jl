# The geometry vocabulary acceleration-structure builds are described in. Core
# types, so this needs no device and no driver.
#
# What used to be here as well and is not: the byte layout `pack_geometry!` writes
# for Vulkan's `VkAccelerationStructureGeometryKHR`, `query_as_build_sizes`, and a
# `build_blas_aabb` smoke test. The first two are one backend's driver structs;
# a wrong layout is a BLAS that does not build or traces the wrong geometry, which
# the traced HWTLAS tests and the procedural trace tests see. The AABB build is
# `test_hwtlas_batch_triangles.jl`, on every backend, with its refusal on a device
# that has no hardware acceleration structures.

using Test, Mantle
using GeometryBasics: Point3f, Vec3f

@testset "GeometryType - types and constructors" begin
    tri = TrianglesGeometry(; vertex_format=UInt32(103),
                              vertex_addr=UInt64(0x1000),
                              vertex_stride=UInt64(12),
                              max_vertex=UInt32(2),
                              index_type=UInt32(1),
                              index_addr=UInt64(0x2000))
    @test tri isa GeometryType
    @test tri isa TrianglesGeometry
    @test tri.vertex_addr == UInt64(0x1000)
    @test tri.transform_addr == UInt64(0)          # the keyword default

    ab = AABBsGeometry(; aabb_addr=UInt64(0x3000), aabb_stride=UInt64(24))
    @test ab isa GeometryType
    @test ab isa AABBsGeometry
    @test ab.aabb_stride == UInt64(24)
    @test AABBsGeometry(; aabb_addr=UInt64(0x3000)).aabb_stride == UInt64(24)
end

@testset "AABB struct" begin
    a = AABB(Point3f(0, 0, 0), Point3f(1, 2, 3))
    @test a.min == Point3f(0, 0, 0)
    @test a.max == Point3f(1, 2, 3)
end
