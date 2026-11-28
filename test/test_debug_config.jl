# `DebugConfig` — the one way to switch validation on.
#
# Everything here is about a configuration that is IMPOSSIBLE to half-set. The
# seven `LAVA_*` environment variables this replaced had two failure modes that
# each cost a debugging session, and both are asserted below to be unreachable:
#
#   1. `LAVA_GPU_AV=1` without `LAVA_VALIDATION=1` built an instance with no
#      validation layer at all. The run came back clean, which reads exactly like
#      "no fault found" rather than "the instrument was off".
#   2. `LAVA_GPU_AV=1` together with `LAVA_DEBUG_PRINTF=1` silently preferred
#      printf and warned. Same shape: the thing you asked for was not running.
#
# There is also exactly ONE way to apply a config, at device construction, and
# this file pins that there are no alternatives: five preset functions
# (`enable_gpu_av`, `disable_gpu_av`, `enable_debug_printf!`,
# `disable_debug_printf!`, `activate_all_debugging`) are deleted along with the
# env vars, because "which preset do I call" was itself a way to end up
# instrumented for something other than what you were hunting.
#
# Host-only: the config is core's and needs no device. What a device does with it
# — the layer attached, its messages drained, nothing reported for a clean run —
# is `test_arena_memory_device_address.jl` and `test_tolerated_alloc_failure.jl`,
# on every backend, each in a process started for it.
#
# The Vulkan file this came from also pinned the context's `Diagnostics` switches
# (`slab_dump_target` typed `Any` with a guard that read `nothing != 0` as on, 41
# µs a dispatch). That struct is the Vulkan context's and has no counterpart on
# another backend; the guard went with the move.

using Test, Mantle

@testset "DebugConfig" begin
    @testset "off by default" begin
        c = DebugConfig()
        @test !c.validation
        @test !c.gpu_av
        @test !c.sync_val
        @test !c.best_practices
        @test !c.printf
        @test isempty(c.gpu_av_shaders)
        @test !c.pool_disabled
        # The one exception, and it defaults ON: Safe Mode exists because GPU-AV
        # crashes, and shipping without it is why reaching for GPU-AV has mostly
        # produced a SIGSEGV instead of a report.
        @test c.gpu_av_safe
    end

    @testset "every feature implies the layer" begin
        # Failure mode 1. There is no way to spell "instrument the shaders but do
        # not load the layer that does the instrumenting".
        for kw in (:gpu_av, :sync_val, :best_practices, :printf)
            @test DebugConfig(; kw => true).validation
        end
        # …and the layer alone is still a valid, cheap mode: core API checks with
        # no shader instrumentation.
        c = DebugConfig(validation = true)
        @test c.validation
        @test !c.gpu_av && !c.printf && !c.sync_val && !c.best_practices
    end

    @testset "gpu_av and printf are refused together" begin
        # Failure mode 2, now an error rather than a warning and a silent choice.
        @test_throws ArgumentError DebugConfig(gpu_av = true, printf = true)
        # The message has to say which one to pick — this is the failure an agent
        # reading it is in the middle of.
        e = try
            DebugConfig(gpu_av = true, printf = true)
        catch err
            err
        end
        msg = sprint(showerror, e)
        @test occursin("gpu_av", msg) && occursin("printf", msg)
    end

    @testset "a modified copy re-checks the rules" begin
        c = DebugConfig(gpu_av = true, gpu_av_shaders = ["step_kernel"])
        @test c.gpu_av_shaders == ["step_kernel"]
        # A copy with nothing replaced is the same configuration, compared by
        # value: two configs built apart from the same settings are equal.
        @test DebugConfig(c) == c
        @test DebugConfig(validation = true) == DebugConfig(validation = true)
        @test hash(DebugConfig(validation = true)) == hash(DebugConfig(validation = true))
        off = DebugConfig(c; gpu_av = false)
        @test !off.gpu_av
        # Copying INTO the forbidden pair must fail the same way as building it.
        @test_throws ArgumentError DebugConfig(c; printf = true)
    end

    @testset "there is one way in, and no presets beside it" begin
        # A device is built or rebuilt with a configuration; nothing applies one
        # to a device that exists. `any` over the method table: the keyword lives
        # on the backends' methods, not on the generic declaration.
        @test any(m -> :debug in Base.kwarg_decl(m), methods(Mantle.reset_device!))
        # The five presets must not come back — each was a different default
        # combination, and `enable_gpu_av`'s was `pool_disabled = false`, blind to
        # exactly the sub-pool overruns it was reached for.
        for name in (:enable_gpu_av, :disable_gpu_av, :activate_all_debugging,
                     :enable_debug_printf!, :disable_debug_printf!)
            @test !isdefined(Mantle, name)
        end
        # …and `pool_disabled` is a field of the config rather than a second step
        # that a caller has to remember.
        @test DebugConfig(pool_disabled = true).pool_disabled
    end
end
