# GPU-AV regression check: a Hikari hardware ray-tracing render under the Vulkan
# validation layer with GPU-assisted validation, and the layer's fault readback.
#
# Vulkan-only tooling with no Metal counterpart, so it is a script and not part of
# Mantle's suite (decided 2026-10-10; DECISIONS.md, item 5). What it checks:
#
#   1. A minimal Hikari HW RT render produces NO validation messages under GPU-AV.
#      This pins the unaligned-BDA-store fix in Lava's SPIR-V emitter
#      (`psb_needs_decomposition`): the emitter returned false for
#      `access_align <= 4`, AND the byte-offset check used `if offset > 0`, which
#      dropped the negative offsets SROA's end-relative pointer arithmetic emits.
#      Both let `store i32 align 4` land on a 2-aligned BDA, which AMD tolerates and
#      other vendors may not — and which corrupted GPU memory enough to trigger
#      RADV GPUVM faults after many Hikari renders. Root cause in
#      `docs/specs/2026-04-25-unaligned-bda-investigation.md`.
#   2. GPU-AV's fault readback does not deadlock: `verify_gpu_av` drives a known
#      out-of-bounds store twice and must get a report back each time.
#
# How to run, on a machine with the Khronos validation layer (VK_LAYER_KHRONOS_validation)
# and Hikari in the environment:
#
#     julia --project=<env> <Mantle>/tools/gpuav_clean.jl
#
# GPU-AV slows compute dispatches by ~100x, so the render takes minutes. Both
# checks rebuild the device with GPU-AV; run it in a process of its own.
using Test
using Lava, Mantle
begin
    using Hikari, Raycore, Adapt
    using GeometryBasics
    using GeometryBasics: normal_mesh, Tessellation, Sphere, Point3f, Point2f, Vec3f
    using LinearAlgebra: I

    @testset "GPU-AV clean — minimal Hikari HW RT render" begin
        backend = Mantle.defaultbackend()
        ctx = Mantle.vk_context()

        # Tiny scene + tiny film keeps the GPU-AV-instrumented run finite.
        scene = Hikari.Scene(; backend=backend, hw_accel=true)
        sphere = normal_mesh(Tessellation(Sphere(Point3f(0, 0, 0.35), 0.35f0), 8))
        push!(scene, sphere, Hikari.Diffuse(Kd=Hikari.RGBSpectrum(0.6f0, 0.6f0, 0.6f0)))
        push!(scene, Hikari.PointLight(Point3f(0, -2, 3), Hikari.RGBSpectrum(30f0)))
        Hikari.sync!(scene)

        film = Hikari.Film(Point2f(8, 8))
        gpu_film = Adapt.adapt(backend, film)
        camera = Hikari.PerspectiveCamera(Point3f(0, -3, 1.5), Point3f(0, 0, 0.35),
                                          film; fov=50f0)

        Mantle.clear_validation_messages!()
        vp = Hikari.VolPath(samples=1, max_depth=1, hw_accel=true)
        vp(scene, gpu_film, camera)
        close(vp)

        # The async debug callback writes into this context's ring; pull
        # anything not yet surfaced by the render's own flushes.
        ctx = Mantle.vk_context()
        Mantle.drain_validation_messages!(ctx)

        # The drained list collects setup-noise warnings (validation layer
        # adjusts settings on init) alongside real errors.  Filter to actual
        # validation issues — the setup noise is harmless.
        real_messages = filter(ctx.validation.messages) do m
            !occursin("adjusting settings", m) &&
            !occursin("VALIDATION-SETTINGS", m) &&
            !occursin("Loader Message", m)
        end

        # Log every captured message verbatim before asserting, so a
        # regression points at the offending shader text immediately.
        for m in real_messages
            @info "GPU-AV captured" m
        end

        # Contract: zero validation messages from a real Hikari HW RT render.
        # If this fails, the SPIR-V emitter has regressed on alignment-tracking
        # for byte-offset GEPs — re-investigate via:
        #   ENV["LAVA_SPIRV_DUMP_DIR"] = "/tmp/lava_spv_debug"
        #   Mantle.reset_device!(debug = Mantle.DebugConfig(gpu_av = true))
        # then disassemble + scan via the script at the bottom of
        # docs/specs/2026-04-25-unaligned-bda-investigation.md.
        @test isempty(real_messages)
        # Also assert the device didn't actually go lost during the render.
        @test !Mantle.device_lost(ctx)
    end

    # Regression for the GPU-AV fault-readback DEADLOCK.
    #
    # The Vulkan debug-utils callback must not `@error`/`push!` (allocate + log)
    # from inside the driver's GPU-AV error readback, which the layer invokes
    # re-entrantly during `vkWaitSemaphores`.  Logging there yields to the Julia
    # scheduler while the driver holds an internal lock, hanging the process
    # forever after the FIRST caught fault (reproduced on the RTX 4000 Ada:
    # the OOB printed once, then the session wedged).  The callback is now
    # async-safe — it only `memcpy`s the message into a preallocated ring and
    # the main thread drains it later — so a caught fault returns control
    # cleanly.  `verify_gpu_av` drives a known out-of-bounds BDA store and must
    # report it within a bounded time; if the callback regresses to allocating
    # or logging, this hangs (caught by a CI watchdog) or never surfaces the OOB.
    @testset "GPU-AV fault readback does not hang" begin
        Mantle.reset_device!(debug = Mantle.DebugConfig(gpu_av = true, pool_disabled = true))
        ctx = Mantle.vk_context()
        if !ctx.gpu_assisted
            @info "GPU-AV did not attach on this driver — skipping fault-readback test"
            @test_skip true
        else
            # First caught fault must be reported (this is the call that hangs
            # if the callback allocates).
            @test Mantle.verify_gpu_av(timeout=30.0) == true
            # And we must be able to KEEP GOING: a second probe proves the
            # post-fault `reset_device!` left a usable, still-instrumented
            # device rather than a wedged one.
            @test Mantle.verify_gpu_av(timeout=30.0) == true
        end
        Mantle.reset_device!(debug = Mantle.DebugConfig())
    end
end
