# Instance-record writer kernels for HW HWTLAS.
#
# The `write_meshscatter_instances_kernel` testset that used to live here was
# removed: commit 9e0ec1d ("instance transform cleanup", 2026-05-06) deleted that
# kernel and its `_pervec` variant on purpose, replacing them with
# `Raycore.update_transforms!` / `_apply_pending_update!`. The test kept importing
# the deleted symbol and failed 8 assertions for three months without anyone
# noticing, because this file is not registered in runtests.jl.
#
# `write_grain_instances_kernel` is still the live API and is what remains here.

using Test, Lava, Mantle
using Mantle: VulkanInstanceRecord, write_grain_instances_kernel,
              build_blas_aabb, build_accel!, AS_INPUT_USAGE
using GeometryBasics: Point3f, Vec3f, Vec4f

@testset "write_grain_instances_kernel -- 4 grains, identity rotations" begin
    # Build two trivial BLASes so we have non-zero device addresses.
    aabb = Mantle.AABB(Point3f(-1f0,-1f0,-1f0), Point3f(1f0,1f0,1f0))
    aabb_blas = build_accel!() do ctx; build_blas_aabb(ctx, [aabb]); end
    tri_blas  = build_accel!() do ctx; build_blas_aabb(ctx, [aabb]); end

    n = 4
    positions_cpu = [Point3f(Float32(i), 0f0, 0f0) for i in 1:n]
    quats_cpu     = [Vec4f(0f0, 0f0, 0f0, 1f0) for _ in 1:n]   # identity quaternions
    positions_gpu = Mantle.LavaArray(positions_cpu)
    quats_gpu     = Mantle.LavaArray(quats_cpu)
    instances_gpu = Mantle.LavaArray{VulkanInstanceRecord}(undef, 2 * n;
                                                        extra_usage=AS_INPUT_USAGE)

    radius = 0.5f0

    # Launch via KA backend pattern: kernel(LavaBackend())(args...; ndrange=n)
    k = write_grain_instances_kernel(Mantle.LavaBackend())
    k(positions_gpu, quats_gpu, radius,
      aabb_blas.address, tri_blas.address,
      instances_gpu;
      ndrange = n)

    bq = Mantle.vk_context().default_bq
    Mantle.vk_flush!(bq)

    instances_cpu = Array(instances_gpu)

    # Per grain i: 2 records.
    for i in 1:n
        rec_phys = instances_cpu[2i - 1]
        rec_rend = instances_cpu[2i]
        # Both share transform: identity-rot x radius scale x translate by i.
        expected_t = (radius, 0f0, 0f0, Float32(i),
                       0f0, radius, 0f0, 0f0,
                       0f0, 0f0, radius, 0f0)
        # `Tuple(...)`: `transform` is a `Mat3x4f` — `SMatrix{4,3,Float32,12}` —
        # and `expected_t` is the twelve floats in the order Vulkan's
        # `VkTransformMatrixKHR` wants them, which is the SMatrix's own storage
        # order. Comparing the two directly is a type mismatch, not a value one:
        # the numbers here were always right. The assertion predates `transform`
        # becoming a matrix, and did not fail in the meantime because this file
        # imported `VulkanInstanceRecord` from Lava and so errored at its first
        # line from the runtime move until now.
        @test Tuple(rec_phys.transform) == expected_t
        @test Tuple(rec_rend.transform) == expected_t
        # custom_index = i-1 in low 24 bits.
        @test (rec_phys.custom_index_and_mask & 0x00FFFFFF) == UInt32(i - 1)
        @test (rec_rend.custom_index_and_mask & 0x00FFFFFF) == UInt32(i - 1)
        # Mask bits.
        @test (rec_phys.custom_index_and_mask >> 24) == UInt32(0x02)
        @test (rec_rend.custom_index_and_mask >> 24) == UInt32(0x04)
        # BLAS addresses.
        @test rec_phys.blas_address == aabb_blas.address
        @test rec_rend.blas_address == tri_blas.address
    end
end

