"""
A backend knows which device it runs on, and pins it when it is built.

`vk_context(backend)` and `vk_context(array)` name a path that already existed:
`VulkanBatchQueue.ctx` and `Buffer.ctx`. The value is the name, not new state; a
second copy of a fact the queue already holds could only disagree with it.

`LavaBackend()` used to store `nothing` in both queue fields and resolve the
process default at every property access, so that a module-level
`const BACKEND = LavaBackend()` survived `reset_device!`. That made "which
device" a global lookup on every launch, and a backend handed to code that also
held a second device dispatched on whichever was current. Every spelling pins
now, the no-argument one included: it reads the default ONCE, at the call.
"""

using Test, Lava, KernelAbstractions
const KA = KernelAbstractions

@testset "a backend knows its device" begin
    ctx = Mantle.vk_context()

    # ── Every construction form resolves to the device it dispatches on.
    @test Mantle.vk_context(LavaBackend(ctx))                          === ctx
    @test Mantle.vk_context(LavaBackend(ctx.default_bq))               === ctx
    @test Mantle.vk_context(LavaBackend(ctx.default_bq, ctx.default_bq)) === ctx

    # ── Including the no-argument one, which pins the default device at the
    #    call. There is no field left that a later access could resolve
    #    differently.
    b = LavaBackend()
    @test b.dispatch_bq === ctx.default_bq
    @test b.upload_bq === ctx.default_bq
    @test Mantle.vk_context(b) === ctx
    @test !any(t -> Nothing <: t, fieldtypes(LavaBackend))

    # ── The round trip that makes it useful: an array knows its device, and the
    #    backend derived from it agrees. This is the path a per-device cache key
    #    travels.
    a = Mantle.LavaArray(zeros(Float32, 8))
    @test Mantle.vk_context(a) === ctx
    @test Mantle.vk_context(KA.get_backend(a)) === ctx

    # ── And it still works as a backend.
    fill!(a, 2.0f0)
    KA.synchronize(b)
    @test Array(a) == fill(2.0f0, 8)
end
