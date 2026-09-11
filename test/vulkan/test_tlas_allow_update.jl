using Test, Lava, Mantle
using Mantle: build_accel!
using Mantle: VulkanInstanceRecord, build_tlas, build_blas_aabb, AS_INPUT_USAGE
using GeometryBasics: Point3f

@testset "build_tlas(LavaArray{VulkanInstanceRecord}, n; allow_update=true)" begin
    # Build a 1-AABB BLAS (unit cube) once.
    aabb = Mantle.AABB(GeometryBasics.Point3f(-1f0, -1f0, -1f0), GeometryBasics.Point3f(1f0, 1f0, 1f0))
    blas = build_accel!(Mantle.batchqueue(Mantle.Device())) do ctx
        build_blas_aabb(ctx, [aabb])
    end

    # Two instances at different positions, both referencing the same BLAS.
    # Translation only (identity rotation) for simplicity.
    function translation_transform(x, y, z)
        (1f0, 0f0, 0f0, x,
         0f0, 1f0, 0f0, y,
         0f0, 0f0, 1f0, z)
    end
    inst_a = VulkanInstanceRecord(translation_transform(0f0, 0f0, 0f0), blas.address;
                                 custom_index=UInt32(0), mask=UInt8(0xff))
    inst_b = VulkanInstanceRecord(translation_transform(5f0, 0f0, 0f0), blas.address;
                                 custom_index=UInt32(1), mask=UInt8(0xff))

    # Instance buffer must have AS_INPUT_USAGE so the driver can read it during build.
    instance_buf = LavaArray([inst_a, inst_b]; extra_usage=AS_INPUT_USAGE)

    # Build with allow_update=true.
    tlas = build_accel!(Mantle.batchqueue(Mantle.Device())) do ctx
        build_tlas(ctx, instance_buf, 2; allow_update=true)
    end

    @test tlas.allow_update == true
    @test tlas.update_scratch_size > 0
    @test tlas.instance_buf === instance_buf
end

