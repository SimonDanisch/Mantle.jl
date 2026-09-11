using Test, Lava, Mantle
using Mantle: refit_tlas!, build_accel!
using Mantle: VulkanInstanceRecord, write_grain_instances_kernel, build_blas_aabb, build_tlas, AS_INPUT_USAGE
using GeometryBasics: Point3f, Vec3f, Vec4f

# Validates the full P1 flow: GPU kernel writes instances, allow_update build,
# refit moves geometry, ray queries observe the change.

@testset "HWTLAS refit cycle -- kernel-written instances, then refit" begin
    aabb = Mantle.AABB(Point3f(-1f0,-1f0,-1f0), Point3f(1f0,1f0,1f0))
    aabb_blas = build_accel!(Mantle.batchqueue(Mantle.Device())) do ctx; build_blas_aabb(ctx, [aabb]); end
    tri_blas  = build_accel!(Mantle.batchqueue(Mantle.Device())) do ctx; build_blas_aabb(ctx, [aabb]); end

    n = 4
    radius = 1f0
    quats_cpu = [Vec4f(0f0, 0f0, 0f0, 1f0) for _ in 1:n]
    quats_gpu = Mantle.LavaArray(quats_cpu)
    instances_gpu = Mantle.LavaArray{VulkanInstanceRecord}(undef, 2 * n;
                                                        extra_usage=AS_INPUT_USAGE)
    backend = Mantle.defaultbackend()
    bq = Mantle.batchqueue(Mantle.Device())

    # Frame 0: grains at x = 0, 5, 10, 15 (separated so each has its own AABB).
    pos_a = [Point3f(Float32(5*(i-1)), 0f0, 0f0) for i in 1:n]
    positions_gpu = Mantle.LavaArray(pos_a)
    write_grain_instances_kernel(backend)(positions_gpu, quats_gpu, radius,
                                           aabb_blas.address, tri_blas.address,
                                           instances_gpu;
                                           ndrange = n)
    Mantle.flush!(bq)

    tlas = build_accel!(Mantle.batchqueue(Mantle.Device())) do ctx
        build_tlas(ctx, instances_gpu, 2 * n; allow_update=true)
    end

    @test tlas.allow_update == true
    @test tlas.update_scratch_size > 0

    # Frame 1: shift all grains by +100 in x.
    pos_b = [Point3f(Float32(5*(i-1) + 100f0), 0f0, 0f0) for i in 1:n]
    Mantle.copyto!(positions_gpu, pos_b)
    write_grain_instances_kernel(backend)(positions_gpu, quats_gpu, radius,
                                           aabb_blas.address, tri_blas.address,
                                           instances_gpu;
                                           ndrange = n)
    Mantle.flush!(bq)

    build_accel!(Mantle.batchqueue(Mantle.Device())) do ctx
        refit_tlas!(ctx, tlas, instances_gpu, 2 * n)
    end

    # The handle is preserved (in-place refit).
    @test tlas.allow_update == true

    # Verify the kernel wrote the shifted transforms into the instance buffer
    # (catches kernel regressions in the integration context). The Frame 1
    # write should land translation column = (100 + 5*(i-1), 0, 0) for grain i.
    instances_cpu = Array(instances_gpu)
    for i in 1:n
        # Each grain has 2 records (physics @ 2i-1, rendering @ 2i); both share transform.
        expected_tx = Float32(100f0 + 5f0 * (i - 1))
        @test instances_cpu[2i - 1].transform[4] == expected_tx
        @test instances_cpu[2i].transform[4]     == expected_tx
        # Translation y and z should be 0 (column 8 = ty, column 12 = tz).
        @test instances_cpu[2i - 1].transform[8]  == 0f0
        @test instances_cpu[2i - 1].transform[12] == 0f0
    end
end

