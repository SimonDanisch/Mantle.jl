"""
The runtime tells the compiler what the device allows.

`Lava.TargetFeatures` is a record with no Vulkan in it, and the emitter reads it
to decide whether a module may declare `ShaderInvocationReorderNV` or the
ray-query capabilities. Declaring one the device lacks is a validation error, not
a slow path — so what is ON that record has to be what the device actually said.

`bind_context!` is the only thing that writes it, and it writes it together with
the context, which is the invariant here: there is no third place the answer can
come from and no window where the two disagree.

The compiler's half — given a record, what does it emit — is
`Lava/test/test_target_features.jl`, and it needs no device. This half is the
inverse and needs one, which is why the two are in different packages.
"""

using Test, Mantle, Lava

@testset "the runtime pushes what the device reports" begin
    ctx = Mantle.vk_context()

    # Already bound by the time any test runs, so this reads what `bind_context!`
    # left rather than arranging for it.
    @test Lava.targetfeatures().ser === ctx.ser_available
    @test Lava.targetfeatures().ray_query === ctx.ray_query_available
    @test Lava.FROZEN_LOG_MISSES[] === ctx.diag.frozen_log_misses

    # Releasing the device resets the record. Without this an emitter test that
    # ran afterwards would emit a module shaped by hardware that is gone — and it
    # would look right, because the capability it declares is one this machine
    # happens to have.
    saved = Lava.targetfeatures()
    try
        Mantle.bind_context!(nothing)
        @test Lava.targetfeatures() == Lava.TargetFeatures()
        @test Lava.FROZEN_LOG_MISSES[] === false
    finally
        Mantle.bind_context!(ctx)
    end
    @test Lava.targetfeatures() == saved
end
