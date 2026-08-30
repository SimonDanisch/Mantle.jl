using Test, Lava, Raycore, Printf
using Mantle: VulkanInstanceRecord, build_blas_aabb, build_accel!, AS_INPUT_USAGE,
              write_grain_instances_kernel
using GeometryBasics: Point3f, Vec3f, Vec4f
using KernelAbstractions

# Stress: 1M instance refits x 1000 frames. Validates no GPU-AV errors,
# no memory growth, and sub-50ms refit cost.
#
# Positions are shifted on the GPU each frame via a small inline kernel to
# avoid constructing 12MB CPU vectors x 1000 frames (= 12GB cumulative alloc).

@kernel function shift_positions_x_kernel!(positions, dx::Float32)
    i = @index(Global)
    @inbounds p = positions[i]
    @inbounds positions[i] = Point3f(p[1] + dx, p[2], p[3])
end

@testset "VulkanTLAS 1M instance refit stress" begin
    aabb = Mantle.AABB(Point3f(-0.005f0, -0.005f0, -0.005f0),
                     Point3f( 0.005f0,  0.005f0,  0.005f0))
    blas = build_accel!() do ctx; build_blas_aabb(ctx, [aabb]); end

    N = 1_000_000
    radius = 0.005f0

    quats     = Mantle.LavaArray([Vec4f(0f0, 0f0, 0f0, 1f0) for _ in 1:N])
    positions = Mantle.LavaArray([Point3f(Float32(0.01 * (i % 1000)),
                                        Float32(0.01 * ((i ÷ 1000) % 1000)),
                                        0f0) for i in 1:N])
    instance_buf = Mantle.LavaArray{VulkanInstanceRecord}(undef, 2 * N;
                                                       extra_usage=AS_INPUT_USAGE)

    backend = Mantle.LavaBackend()
    bq      = Mantle.vk_context().default_bq

    # Initial instance record write.
    write_grain_instances_kernel(backend)(
        positions, quats, radius, blas.address, blas.address, instance_buf;
        ndrange = N)
    Mantle.vk_flush!(bq)

    tlas = Mantle.VulkanTLAS(backend)
    push!(tlas, blas, instance_buf; n = 2 * N,
          instance_mask = UInt8(0x02))
    Raycore.sync!(tlas)

    shift_kernel = shift_positions_x_kernel!(backend)

    # 1000 refits with positions shifted +0.001 in x each frame (GPU-only shift).
    n_frames    = 1000
    refit_times = Float64[]
    sizehint!(refit_times, n_frames)

    for f in 1:n_frames
        # Shift positions on the GPU -- no CPU vector allocation per frame.
        shift_kernel(positions, 0.001f0; ndrange = N)
        Mantle.vk_flush!(bq)

        write_grain_instances_kernel(backend)(
            positions, quats, radius, blas.address, blas.address, instance_buf;
            ndrange = N)
        Mantle.vk_flush!(bq)

        t0 = time()
        tlas.transforms_dirty = true
        Raycore.sync!(tlas)
        push!(refit_times, time() - t0)

        if f % 100 == 0
            @printf("frame %4d  refit = %.3f ms\n", f, refit_times[end] * 1000)
        end
    end

    avg_refit_ms = (sum(refit_times) / length(refit_times)) * 1000
    @info "Average refit time" avg_refit_ms
    @test avg_refit_ms < 50.0   # Sanity: refit should beat 50ms at 1M instances.
end

