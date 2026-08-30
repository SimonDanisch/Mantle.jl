# @lava_printf — on-kernel printing via NonSemantic.DebugPrintf.
#
# The SPIR-V-emission tests are always safe to run (they only compile + spirv-val,
# no device reset). The live-output test enables the validation layer's debug
# printf feature, which RESETS the Vulkan device, so it would disturb other tests
# sharing the global device — it's opt-in via LAVA_PRINTF_LIVE=1.

using Test, Lava, Mantle
using Lava.KernelAbstractions

@testset "@lava_printf SPIR-V emission" begin
    # Format + args of several widths exercise i32 / i64 / float / double operands.
    function pf_dev(out::Lava.LavaDeviceArray{Float32,1})
        i = Lava.lava_global_invocation_id_x()
        @lava_printf "i=%d u=%u big=%ld f=%f d=%lf\n" Int32(i) UInt32(i) Int64(i)*7 out[i] Float64(i)
        @inbounds out[i] = Float32(i)
        return nothing
    end

    result = Lava.lava_compile_gpu(pf_dev, Tuple{Lava.LavaDeviceArray{Float32,1}};
                                   workgroup_size=(64,1,1), validate=true)  # validate=true runs spirv-val
    d = Lava.disassemble_spirv(result.spirv_bytes)

    @test occursin("SPV_KHR_non_semantic_info", d)   # extension declared
    @test occursin("NonSemantic.DebugPrintf", d)      # ext-inst set imported
    @test occursin("OpString", d)                     # format string present
    @test occursin("i=%d u=%u big=%ld f=%f d=%lf", d) # exact format recovered

    # No-arg format must also compile + validate.
    function pf_noarg(out::Lava.LavaDeviceArray{Float32,1})
        Lava.lava_global_invocation_id_x()
        @lava_printf "hello from a kernel\n"
        return nothing
    end
    r2 = Lava.lava_compile_gpu(pf_noarg, Tuple{Lava.LavaDeviceArray{Float32,1}};
                               workgroup_size=(64,1,1), validate=true)
    @test occursin("hello from a kernel", Lava.disassemble_spirv(r2.spirv_bytes))
end

@testset "KernelAbstractions.@print (backend-independent) on Lava" begin
    # The portable KA API: string literals interleaved with values. Lava's
    # __print override auto-selects specifiers from the arg types and routes
    # through the same DebugPrintf path.
    function kp_dev(out::Lava.LavaDeviceArray{Float32,1})
        i = Lava.lava_global_invocation_id_x()
        KernelAbstractions.@print("tid=", UInt32(i), " big=", Int64(i)*3,
                                  " val=", out[i], "\n")
        @inbounds out[i] = Float32(i)
        return nothing
    end
    r = Lava.lava_compile_gpu(kp_dev, Tuple{Lava.LavaDeviceArray{Float32,1}};
                              workgroup_size=(64,1,1), validate=true)
    d = Lava.disassemble_spirv(r.spirv_bytes)
    @test occursin("NonSemantic.DebugPrintf", d)
    # literals + auto-selected specifiers (u32→%u, i64→%ld, f32→%f) assembled in order
    @test occursin("tid=%u big=%ld val=%f", d)
end

if get(ENV, "LAVA_PRINTF_LIVE", "0") == "1"
    @testset "@lava_printf live output (resets device)" begin
        Mantle.reset_device!(debug = Mantle.DebugConfig(printf = true))
        try
            backend = Mantle.LavaBackend()
            bq = Mantle.vk_context().default_bq
            @kernel function pf_live!(out)
                i = @index(Global)
                @lava_printf "tid=%u val=%f\n" UInt32(i) Float32(i) * 10f0
                @inbounds out[i] = Float32(i)
            end
            Mantle.clear_printf_output!()
            out = Mantle.LavaArray(zeros(Float32, 4))
            pf_live!(backend)(out; ndrange = 4)
            Mantle.vk_flush!(bq)
            KernelAbstractions.synchronize(backend)
            msgs = Mantle.get_printf_output()
            @test length(msgs) == 4
            @test any(m -> occursin("tid=1 val=10.000000", m), msgs)
            @test Array(out) == Float32[1, 2, 3, 4]   # kernel still computes correctly
        finally
            Mantle.reset_device!(debug = Mantle.DebugConfig())
        end
    end
end
