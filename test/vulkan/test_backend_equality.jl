"""
A backend equals every spelling of its DEVICE.

`==` used to compare queues: `LavaBackend()` resolved its queues at each access,
`LavaBackend(ctx)` pinned them, and the default `==` compared `nothing` against a
queue, so an array allocated by the default backend did not belong to it as far
as `==` could tell (RayMakie's meshscatter tests failed on exactly that).

Now `==` is "the same device". That is the question both callers ask: Raycore
refuses a cross-backend `adapt` of a TLAS, and RayMakie asks whether an array is
the screen's. A backend on a second queue of the same device answers them the
same way as the primary one does; which queue a backend uses is a scheduling
choice, not an identity. Two devices are two backends, which
`test_device_identity.jl` asserts against lavapipe.
"""

using Test, Mantle, Lava, KernelAbstractions
const KA = KernelAbstractions

@testset "a backend equals every spelling of its device" begin
    ctx = MVE.vk_context()
    default = MVE.LavaBackend()
    pinned = MVE.LavaBackend(ctx)
    @test default == pinned
    @test pinned == default
    @test hash(default) == hash(pinned)
    a = KA.allocate(default, Float32, 8)
    @test KA.get_backend(a) == Mantle.defaultbackend()
    @test KA.get_backend(a) == default

    bq2 = Mantle.allocate_batch_queue!(ctx)
    try
        other = MVE.LavaBackend(bq2)
        @test other == pinned
        @test other !== pinned
        @test hash(other) == hash(pinned)
        @test MVE.LavaBackend(bq2, bq2) == other
        # The split spelling, another upload queue on the same device, is the
        # same device too.
        @test MVE.LavaBackend(ctx.default_bq, bq2) == pinned
    finally
        Mantle.release_batch_queue!(bq2)
    end
end
