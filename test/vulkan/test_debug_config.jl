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
# There is also exactly ONE way to apply a config, and this file pins that the
# alternatives are gone: five preset functions (`enable_gpu_av`, `disable_gpu_av`,
# `enable_debug_printf!`, `disable_debug_printf!`, `activate_all_debugging`) are
# deleted along with the env vars, because "which preset do I call" was itself a
# way to end up instrumented for something other than what you were hunting.
#
# The device-building half is deliberately NOT tested here — `vk_reset_device!`
# invalidates every live `LavaArray`, so a suite that ran it mid-file would break
# whatever came after. `test_gpuav_clean.jl` covers that, gated on
# `LAVA_TEST_GPU_AV=1`.

using Test, Lava, Mantle
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
        # …and the layer alone is still a valid, cheap mode: core spec checks with
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
        @test DebugConfig(c) == c || (DebugConfig(c).gpu_av && DebugConfig(c).validation)
        off = DebugConfig(c; gpu_av = false)
        @test !off.gpu_av
        # Copying INTO the forbidden pair must fail the same way as building it.
        @test_throws ArgumentError DebugConfig(c; printf = true)
    end

    @testset "there is one way in, and the presets are gone" begin
        # `vk_reset_device!` takes it; `VkContext` takes it. Nothing else does.
        # `any` over the method table, not `first`: `VkContext`'s first method is
        # the inner positional constructor, whose `kwarg_decl` is empty.
        for f in (Mantle.vk_reset_device!, Mantle.VkContext)
            @test any(m -> :debug in Base.kwarg_decl(m), methods(f))
        end
        # The five presets must not come back — each was a different default
        # combination, and `enable_gpu_av`'s was `pool_disabled = false`, blind to
        # exactly the sub-pool overruns it was reached for.
        for name in (:enable_gpu_av, :disable_gpu_av, :activate_all_debugging,
                     :enable_debug_printf!, :disable_debug_printf!)
            @test !isdefined(Lava, name)
        end
        # …and `pool_disabled` is a field of the config rather than a second step
        # (`pool(ctx).disabled = true`) that a caller has to remember.
        @test Mantle.DebugConfig(pool_disabled = true).pool_disabled
    end

    @testset "the device carries what it was asked for" begin
        ctx = Mantle.vk_context()
        @test ctx.debug isa DebugConfig
        # `debug` is the request; `gpu_assisted` is what was achieved. They can
        # differ — the extension or the layer may be missing — which is why both
        # exist and why `verify_gpu_av` exists on top of them.
        ctx.debug.gpu_av || @test !ctx.gpu_assisted
    end

    @testset "an off switch is off in the form its guard tests" begin
        # `slab_dump_target` was typed `Any` and defaulted to `nothing`. Its guard
        # in `submit!` reads
        #
        #     ctx.diag.slab_dump_target != UInt64(0)
        #
        # and `nothing != UInt64(0)` is TRUE, so the debug scan of the argument
        # slab ran on EVERY submit — a loop over `arg_slab_offset ÷ 8` words of
        # host-visible memory, growing with every dispatch recorded. It cost ~41 µs
        # of host CPU per dispatch (SAM 2 decode 7.1 -> 25 ms, encode 103 -> 131)
        # and said nothing, because `target` was `nothing` so no word ever matched
        # and the hit list stayed empty. The global it replaced was a
        # `Ref{UInt64}(0)`, where the same guard read `0 != 0`.
        #
        # So: assert the GUARD, not the value. `=== nothing` would have passed
        # throughout, and `isnothing(...) == false` would too — only evaluating
        # what the code actually branches on catches a field whose type drifted.
        d = Mantle.Diagnostics()
        @test d.slab_dump_target isa UInt64
        @test !(d.slab_dump_target != UInt64(0))

        # The other `Any`-typed diagnostic. It is compared with `=== nothing`
        # (`gpuarrays.jl`'s `probe_broadcast!`), so `nothing` is the right default
        # here — the pair is the point: what a field may hold depends on how its
        # guard asks.
        @test d.broadcast_probe === nothing

        # Every remaining switch is a Bool, and off means `false`. A Bool cannot
        # develop this defect, which is why they are listed rather than trusted.
        for f in (:alloc_debug, :free_debug, :freed_bda_scan, :destroy_freed_bdas_throws,
                  :presubmit_scan, :presubmit_scan_throws, :pack_arg_assert_live,
                  :batch_timing, :dispatch_logging, :dispatch_timing, :frozen_log_misses)
            v = getfield(d, f)
            @test v isa Bool
            @test !v
        end
    end
end
