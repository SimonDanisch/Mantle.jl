"""
The context carries the record the compiler takes.

`Lava.TargetFeatures` is a record with no Vulkan in it, and the emitter reads it
to decide whether a module may declare `ShaderInvocationReorderNV` or the
ray-query capabilities. Declaring one the device lacks is a validation error, not
a slow path, so what is ON that record has to be what the device actually said.

It used to be a process global that `bind_context!` pushed, which answered for
the BOUND device: a kernel compiled for a second device while the RTX was bound
was shaped by the RTX. Now each `VkContext` carries its own `features`, every
compile the context runs passes it in the job, and every frozen key it reads
mixes it in. Nothing is pushed and nothing is reset on unbind, because there is
nothing global to reset.

The compiler's half, given a record, what does it emit, is
`Lava/test/test_target_features.jl` and needs no device.
"""

using Test, Mantle, Lava

@testset "the context carries what the device reports" begin
    ctx = MVE.vk_context()
    @test ctx.features.ser === ctx.ser_available
    @test ctx.features.ray_query === ctx.ray_query_available
    @test Lava.FROZEN_LOG_MISSES[] === ctx.diag.frozen_log_misses

    # The global is gone, not merely unused.
    @test !isdefined(Lava, :targetfeatures)
    @test !isdefined(Lava, :targetfeatures!)
    @test !isdefined(Lava, :TARGET_FEATURES)

    # A compile this context runs is keyed on ITS record: the same kernel for a
    # device with the opposite SER flag is a different frozen entry.
    other = Lava.TargetFeatures(; ser = !ctx.features.ser, ray_query = ctx.features.ray_query)
    tt = Tuple{Lava.LavaDeviceArray{Float32,1}}
    @test Lava.frozen_key(identity, tt, (64, 1, 1), ctx.features) !=
          Lava.frozen_key(identity, tt, (64, 1, 1), other)
    @test Lava.frozen_rt_key(identity, tt, :raygen, :f32, 8, ctx.features) !=
          Lava.frozen_rt_key(identity, tt, :raygen, :f32, 8, other)
end
