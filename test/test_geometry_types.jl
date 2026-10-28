# The box acceleration-structure builds read. Core's type, so this needs no device and no
# driver; the builds and traces over it are `test_hwtlas_batch_triangles.jl` and
# `test_aabb_blas_device.jl`, on every backend.

using Test, Mantle
using GeometryBasics: Point3f, Vec3f

@testset "AABB struct" begin
    a = AABB(Point3f(0, 0, 0), Point3f(1, 2, 3))
    @test a.min == Point3f(0, 0, 0)
    @test a.max == Point3f(1, 2, 3)
    # Every backend reads boxes in place, 24 bytes apart: `VkAabbPositionsKHR`,
    # `MTLAxisAlignedBoundingBox`.
    @test sizeof(AABB) == 24 && isbitstype(AABB)
end
