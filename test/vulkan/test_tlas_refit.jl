using Test, Lava, Mantle
using Mantle: refit_tlas!, build_accel!
using .MVE: VulkanInstanceRecord, build_tlas, build_blas_aabb, AS_INPUT_USAGE
using GeometryBasics: Point3f

function translation_transform(x, y, z)
    (1f0, 0f0, 0f0, x,
     0f0, 1f0, 0f0, y,
     0f0, 0f0, 1f0, z)
end

@testset "refit_tlas! moves geometry without rebuilding" begin
    aabb = Mantle.AABB(Point3f(-1f0, -1f0, -1f0), Point3f(1f0, 1f0, 1f0))
    blas = build_accel!(MVE.vk_context().default_bq) do ctx
        build_blas_aabb(ctx, [aabb])
    end

    # Initial layout: instance 0 at (0,0,0), instance 1 at (5,0,0).
    inst_a0 = VulkanInstanceRecord(translation_transform(0f0, 0f0, 0f0), blas.address;
                                  custom_index=UInt32(0), mask=UInt8(0xff))
    inst_b0 = VulkanInstanceRecord(translation_transform(5f0, 0f0, 0f0), blas.address;
                                  custom_index=UInt32(1), mask=UInt8(0xff))
    instance_buf = MVE.LavaArray([inst_a0, inst_b0]; extra_usage=AS_INPUT_USAGE)

    tlas = build_accel!(MVE.vk_context().default_bq) do ctx
        build_tlas(ctx, instance_buf, 2; allow_update=true)
    end

    # Refit: move instance 1 from x=5 to x=20.
    inst_a1 = inst_a0
    inst_b1 = VulkanInstanceRecord(translation_transform(20f0, 0f0, 0f0), blas.address;
                                  custom_index=UInt32(1), mask=UInt8(0xff))
    Mantle.copyto!(instance_buf, [inst_a1, inst_b1])

    build_accel!(MVE.vk_context().default_bq) do ctx
        refit_tlas!(ctx, tlas, instance_buf, 2)
    end

    # Handle preserved across refit (in-place).
    @test tlas.allow_update == true
end

@testset "refit_tlas! errors on non-allow_update HWTLAS" begin
    aabb = Mantle.AABB(Point3f(-1f0, -1f0, -1f0), Point3f(1f0, 1f0, 1f0))
    blas = build_accel!(MVE.vk_context().default_bq) do ctx
        build_blas_aabb(ctx, [aabb])
    end
    inst = VulkanInstanceRecord(Mantle.identity_transform(), blas.address)
    instance_buf = MVE.LavaArray([inst]; extra_usage=AS_INPUT_USAGE)

    tlas = build_accel!(MVE.vk_context().default_bq) do ctx
        build_tlas(ctx, instance_buf, 1; allow_update=false)
    end

    @test_throws ErrorException build_accel!(MVE.vk_context().default_bq) do ctx
        refit_tlas!(ctx, tlas, instance_buf, 1)
    end
end

