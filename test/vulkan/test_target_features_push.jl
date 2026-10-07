"""
The context carries the record the compiler takes.

`Lava.TargetFeatures` is a record with no Vulkan in it, and the emitter reads it
to decide whether a module may declare `ShaderInvocationReorderNV` or the
ray-query capabilities. Declaring one the device lacks is a validation error, not
a slow path, so what is ON that record has to be what the device actually said.

A process global pushed by `bind_context!` answers for the BOUND device: a
kernel compiled for a second device while the RTX is bound
was shaped by the RTX. Now each `VkContext` carries its own `features`, every
compile the context runs passes it in the job's compiler configuration, which
the kernel cache keys on. Nothing is pushed and nothing is reset on unbind,
because there is nothing global to reset.

The compiler's half, given a record, what does it emit, is
`Lava/test/test_target_features.jl` and needs no device.
"""

using Test, Mantle, Lava

@testset "the context carries what the device reports" begin
    ctx = Mantle.vk_context()
    @test ctx.features.ser === ctx.ser_available
    @test ctx.features.ray_query === ctx.ray_query_available

    # There is no such global, not merely an unused one.
    @test !isdefined(Lava, :targetfeatures)
    @test !isdefined(Lava, :targetfeatures!)
    @test !isdefined(Lava, :TARGET_FEATURES)

    # A compile this context runs is keyed on ITS record: the same kernel for a
    # device with the opposite SER flag is a different compiler configuration,
    # and so a different cache entry.
    other = Lava.TargetFeatures(; ser = !ctx.features.ser, ray_query = ctx.features.ray_query)
    tt = Tuple{Lava.LavaDeviceArray{Float32,1}}
    @test Lava.lava_kernel_job(identity, tt; workgroup_size = (64, 1, 1), features = ctx.features).config !=
          Lava.lava_kernel_job(identity, tt; workgroup_size = (64, 1, 1), features = other).config
end
