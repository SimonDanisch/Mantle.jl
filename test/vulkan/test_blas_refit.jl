# BLAS refit via MODE_UPDATE_KHR.
#
# Reuses the handwritten RT shaders from test_handwritten_rt.jl: a triangle in
# the XY plane, rays starting at z=-1 pointing at +z, closest-hit writes RayTmax
# into the payload. So the hit distance IS the triangle's z plus one, which
# makes "did the acceleration structure actually move" a number rather than a
# judgement call.
#
# That matters here specifically: rewriting the vertex BUFFER is easy to verify
# and proves nothing, because the AS holds its own copy of the geometry. If
# `refit_blas!` issued a malformed or ineffective build, the buffer check would
# still pass and rays would keep hitting the old plane.
using Lava, Mantle
using Vulkan
using Test

# The hand-written pipeline's dispatch, as the unmodelled path spells it today:
# one one-shot holding the trace, the acceleration structures it reads pinned
# by core's `pintrace!`, submitted when it closes. `rt_dispatch!` was the name
# of the open-batch version.
function rt_dispatch!(bq, pipeline, tlas, push_bda, W, H)
    MVE.oneshot!(bq; tag = :trace) do e
        Mantle.pintrace!(e, tlas)
        MVE.emit_trace!(e, pipeline, tlas, push_bda, W, H, 1)
    end
    return nothing
end

# The shaders come from that file. Guarded so this stays runnable on its own
# while `runtests.jl`, which already includes it, does not run its testsets twice.
isdefined(@__MODULE__, :build_raygen_shader) ||
    include(joinpath(@__DIR__, "test_handwritten_rt.jl"))

@testset "BLAS refit moves the acceleration structure" begin
    ctx = MVE.vk_context()
    if ctx.rt_pipeline_properties === nothing
        @warn "Skipping BLAS refit test: no ray tracing support"
    else
        W, H = 8, 8
        indices = UInt32[0, 1, 2]
        at_z(z) = [(0.0f0, 0.0f0, z), (1.0f0, 0.0f0, z), (0.0f0, 1.0f0, z)]

        blas, tlas = Mantle.build_accel!(MVE.vk_context().default_bq) do c
            b = MVE.build_blas(c, at_z(0.0f0), indices; allow_update = true)
            (b, MVE.build_tlas(c, [b]))
        end
        @test blas.allow_update
        @test blas.update_scratch_size > 0

        pipeline = MVE.create_rt_pipeline(ctx, build_raygen_shader(),
                                           build_miss_shader(), build_closesthit_shader();
                                           push_constant_size = 8)
        output_buf = MVE.vk_alloc(ctx.default_bq, W * H * sizeof(Float32))

        # Centre ray: (0.25, 0.25) is comfortably inside the triangle for any
        # edge rule, so it is the sample to trust.
        function trace_center()
            rt_dispatch!(ctx.default_bq, pipeline, tlas, output_buf.address, W, H)
            bytes = Vector{UInt8}(undef, W * H * sizeof(Float32))
            MVE.download!(bytes, output_buf)
            vals = reinterpret(Float32, bytes)
            hits = filter(>(0.0f0), collect(vals))
            return isempty(hits) ? -1.0f0 : minimum(hits)
        end

        t_before = trace_center()
        @test t_before ≈ 1.0f0 atol = 1.0f-3   # plane at z=0, origin at z=-1

        # Move the plane to z=5 and refit in place. Same topology, same buffer.
        Mantle.build_accel!(MVE.vk_context().default_bq) do c
            MVE.refit_blas!(c, blas, at_z(5.0f0))
        end
        MVE.VK.device_wait_idle(ctx.device)
        # The HWTLAS references the BLAS by address, which the in-place refit
        # preserves, so it does not need rebuilding for the geometry to move.

        t_after = trace_center()
        @test t_after ≈ 6.0f0 atol = 1.0f-3    # plane at z=5, origin at z=-1
        @test t_after != t_before
    end
end

@testset "refit rejects what MODE_UPDATE_KHR cannot do" begin
    indices = UInt32[0, 1, 2]
    verts = [(0.0f0, 0.0f0, 0.0f0), (1.0f0, 0.0f0, 0.0f0), (0.0f0, 1.0f0, 0.0f0)]

    # A BLAS built without ALLOW_UPDATE cannot be refit, and saying so beats a
    # driver-level fault or a silently ignored build.
    static_blas = Mantle.build_accel!(MVE.vk_context().default_bq) do c; MVE.build_blas(c, verts, indices); end
    @test !static_blas.allow_update
    @test static_blas.update_scratch_size == 0
    @test_throws ErrorException Mantle.build_accel!(MVE.vk_context().default_bq) do c
        MVE.refit_blas!(c, static_blas, verts)
    end

    # Topology is fixed at build: MODE_UPDATE_KHR refits the existing tree.
    dyn_blas = Mantle.build_accel!(MVE.vk_context().default_bq) do c
        MVE.build_blas(c, verts, indices; allow_update = true)
    end
    @test_throws ErrorException Mantle.build_accel!(MVE.vk_context().default_bq) do c
        MVE.refit_blas!(c, dyn_blas, vcat(verts, verts))
    end
end
