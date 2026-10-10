# Stress: 1M grains, two instances each (2M instances in two batches), refit
# 1000 times from records a kernel rewrites every frame. The claim is the cost:
# a refit of 2M instances, records to driver descriptors included, under 50 ms
# on average. And that the 1000th refit still moved the instances.
#
# Positions are shifted on the GPU each frame via a small inline kernel to
# avoid constructing 12MB CPU vectors x 1000 frames (= 12GB cumulative alloc).
#
# This file was not registered in runtests.jl while it lived in test/vulkan, so
# it had not run since it was written.

using Test, Mantle, Raycore, Printf
import KernelInterface as KI
using KernelAbstractions
using GeometryBasics: Point3f, Vec3f, Vec4f
include(joinpath(@__DIR__, "testbackend.jl"))
isdefined(@__MODULE__, :tlasprobe!) || include(joinpath(@__DIR__, "hwtlas_helpers.jl"))

@kernel function shift_positions_x_kernel!(positions, dx::Float32)
    i = @index(Global)
    @inbounds p = positions[i]
    @inbounds positions[i] = Point3f(p[1] + dx, p[2], p[3])
end

if Mantle.supports_hwtlas(TESTBACKEND)

@testset "1M-grain instance refit stress" begin
    # A triangle of the grain's radius around its centre, in its plane z = 0.
    blas = Mantle.build_accel!(Mantle.batchqueue(Mantle.Device(TESTBACKEND))) do ctx
        Mantle.build_blas(ctx, [(-1f0, -1f0, 0f0), (1f0, -1f0, 0f0), (0f0, 1f0, 0f0)], UInt32[0, 1, 2])
    end

    N = 1_000_000
    radius = 0.005f0
    grid(i) = Point3f(Float32(0.01 * (i % 1000)), Float32(0.01 * ((i ÷ 1000) % 1000)), 0f0)

    quats     = Mantle.devicearray(TESTBACKEND, [Vec4f(0f0, 0f0, 0f0, 1f0) for _ in 1:N])
    positions = Mantle.devicearray(TESTBACKEND, [grid(i) for i in 1:N])
    phys = Mantle.devicearray(TESTBACKEND, Raycore.InstanceRecord, N)
    rend = Mantle.devicearray(TESTBACKEND, Raycore.InstanceRecord, N)
    writegrains! = KI.Kernel(TESTBACKEND, Mantle.write_grain_instances_kernel)

    # Initial instance record write.
    writegrains!(positions, quats, radius, phys, rend; ndrange = N)
    tlas = Mantle.HWTLAS{Raycore.Triangle{UInt32}}(TESTBACKEND)
    push!(tlas, blas, phys)
    push!(tlas, blas, rend)
    Raycore.sync!(tlas)
    @test Raycore.n_instances(tlas) == 2N

    shift_kernel = shift_positions_x_kernel!(TESTBACKEND)

    # 1000 refits with positions shifted +0.001 in x each frame (GPU-only shift).
    n_frames    = 1000
    dx          = 0.001f0
    refit_times = Float64[]
    sizehint!(refit_times, n_frames)

    for f in 1:n_frames
        # Shift positions on the GPU -- no CPU vector allocation per frame.
        shift_kernel(positions, dx; ndrange = N)
        writegrains!(positions, quats, radius, phys, rend; ndrange = N)
        # The writes are queued work, not refit work: wait for them first.
        KernelAbstractions.synchronize(TESTBACKEND)

        t0 = time()
        Raycore.refit!(tlas)
        push!(refit_times, time() - t0)

        if f % 100 == 0
            @printf("frame %4d  refit = %.3f ms\n", f, refit_times[end] * 1000)
        end
    end

    avg_refit_ms = (sum(refit_times) / length(refit_times)) * 1000
    @info "Average refit time" avg_refit_ms
    @test avg_refit_ms < 50.0   # Sanity: refit should beat 50ms at 2M instances.

    # The last refit moved the grains: a ray onto where a few of them are now
    # hits them (physics records, mask 0x02), and one onto where they started
    # does not. The shift is 1.0 in x, so grains that started below x = 1 left
    # their places empty: 1 and 50 on the first row, 1000 at the start of the
    # second.
    shift = Vec3f(n_frames * dx, 0f0, 0f0)
    ids = [1, 50, 1000]
    now = TLASProbe(TESTBACKEND, [grid(i) + shift + Vec3f(0, 0, 1) for i in ids], fill(Vec3f(0, 0, -1), 3))
    r = tlastrace(now, tlas; mask = 0x02)
    @test all(r.hit)
    @test r.id == UInt32.(ids .- 1)
    then = TLASProbe(TESTBACKEND, [grid(i) + Vec3f(0, 0, 1) for i in ids], fill(Vec3f(0, 0, -1), 3))
    @test !any(tlastrace(then, tlas; mask = 0x02).hit)
end

else
    @info "no hardware acceleration structures on this backend; skipping"
end
