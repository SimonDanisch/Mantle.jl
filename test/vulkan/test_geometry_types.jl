using Test, Lava, Mantle
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

    ab = AABBsGeometry(; aabb_addr=UInt64(0x3000), aabb_stride=UInt64(24))
    @test ab isa GeometryType
    @test ab isa AABBsGeometry
    @test ab.aabb_stride == UInt64(24)
end

@testset "AABB struct" begin
    a = AABB(Point3f(0, 0, 0), Point3f(1, 2, 3))
    @test a.min == Point3f(0, 0, 0)
    @test a.max == Point3f(1, 2, 3)
end

@testset "pack_geometry! :: TrianglesGeometry byte layout" begin
    buf = zeros(UInt8, 128)
    geom = TrianglesGeometry(; vertex_format=UInt32(103),    # VK_FORMAT_R32G32B32_SFLOAT
                              vertex_addr=UInt64(0xCAFEBABE),
                              vertex_stride=UInt64(12),
                              max_vertex=UInt32(2),
                              index_type=UInt32(1),           # VK_INDEX_TYPE_UINT32
                              index_addr=UInt64(0xDEADBEEF),
                              transform_addr=UInt64(0))
    MVE.pack_geometry!(buf, 0, geom; geo_flags=UInt32(1))   # OPAQUE_BIT

    # Outer VkAccelerationStructureGeometryKHR header
    @test reinterpret(Int32,  buf[1:4])[1]   == MVE.VK_STYPE_GEO     # sType @ 0
    @test reinterpret(UInt32, buf[17:20])[1] == UInt32(0)             # geometryType @ 16 = TRIANGLES

    # Triangles sub-struct (VkAccelerationStructureGeometryTrianglesDataKHR) at offset 24
    @test reinterpret(Int32,  buf[25:28])[1] == MVE.VK_STYPE_TRI     # sType @ q+0
    @test reinterpret(UInt32, buf[41:44])[1] == UInt32(103)           # vertexFormat @ q+16
    @test reinterpret(UInt64, buf[49:56])[1] == UInt64(0xCAFEBABE)    # vertex data addr @ q+24
    @test reinterpret(UInt64, buf[57:64])[1] == UInt64(12)            # vertexStride @ q+32
    @test reinterpret(UInt32, buf[65:68])[1] == UInt32(2)             # maxVertex @ q+40
    @test reinterpret(UInt32, buf[69:72])[1] == UInt32(1)             # indexType @ q+44
    @test reinterpret(UInt64, buf[73:80])[1] == UInt64(0xDEADBEEF)    # index data addr @ q+48
    @test reinterpret(UInt64, buf[81:88])[1] == UInt64(0)             # transform addr @ q+56
    @test reinterpret(UInt32, buf[89:92])[1] == UInt32(1)             # geo_flags @ p+88 (OPAQUE)
end

@testset "pack_geometry! :: AABBsGeometry byte layout" begin
    buf = zeros(UInt8, 128)
    geom = AABBsGeometry(; aabb_addr=UInt64(0xCAFEBABEDEADBEEF),
                         aabb_stride=UInt64(24))
    MVE.pack_geometry!(buf, 0, geom; geo_flags=UInt32(1))

    @test reinterpret(Int32,  buf[1:4])[1]   == MVE.VK_STYPE_GEO    # sType @ 0
    @test reinterpret(UInt32, buf[17:20])[1] == UInt32(1)             # geometryType = AABBS @ 16
    @test reinterpret(Int32,  buf[25:28])[1] == MVE.VK_STYPE_AABB   # sType @ q+0
    @test reinterpret(UInt64, buf[41:48])[1] == UInt64(0xCAFEBABEDEADBEEF)  # aabb_addr @ q+16
    @test reinterpret(UInt64, buf[49:56])[1] == UInt64(24)            # aabb_stride @ q+24
    @test reinterpret(UInt32, buf[89:92])[1] == UInt32(1)             # geo_flags @ p+88
end

@testset "AABB BLAS - query_as_build_sizes returns nonzero" begin
    dev = MVE.vk_context().device
    geom = AABBsGeometry(; aabb_addr=UInt64(0), aabb_stride=UInt64(24))
    sizes = MVE.query_as_build_sizes(dev, geom;
        as_type = UInt32(1),
        build_flags = UInt32(0),
        geo_flags = UInt32(1),
        max_primitive_count = UInt32(1000))
    @test sizes.acceleration_structure_size > 0
    @test sizes.build_scratch_size > 0
end

@testset "AABB BLAS - build_blas_aabb smoke" begin
    aabbs = [AABB(Point3f(0, 0, 0), Point3f(1, 1, 1)),
             AABB(Point3f(2, 2, 2), Point3f(3, 3, 3))]
    blas = Mantle.build_accel!(MVE.vk_context().default_bq) do ctx
        MVE.build_blas_aabb(ctx, aabbs)
    end
    @test blas isa MVE.LavaBLAS
    @test blas.address > UInt64(0)
end

@testset "AABB BLAS - build_blas_aabb errors loudly without ray_query" begin
    ctx = MVE.vk_context()
    saved = ctx.ray_query_available
    ctx.ray_query_available = false
    try
        @test_throws ErrorException Mantle.build_accel!(MVE.vk_context().default_bq) do bc
            MVE.build_blas_aabb(bc, [AABB(Point3f(0, 0, 0), Point3f(1, 1, 1))])
        end
    finally
        ctx.ray_query_available = saved
    end
end
