"""
One backend, however it was spelled, compares equal to itself.

`MVE.LavaBackend()` resolves its queues through the live context at each access;
`MVE.LavaBackend(ctx)`, `MVE.LavaBackend(bq)` and `KA.get_backend(array)` pin the same
queue. The default `==` compared the unresolved fields — `nothing` against a
queue — so an array allocated by the default backend did not belong to it, as
far as `==` could tell (RayMakie's meshscatter tests asserted exactly that and
failed). Equality resolves the queues now, and a backend pinned to a DIFFERENT
queue stays unequal.
"""

using Test, Mantle, Lava, KernelAbstractions
const KA = KernelAbstractions

@testset "a backend equals every spelling of itself" begin
    ctx = MVE.vk_context()
    unpinned = MVE.LavaBackend()
    pinned = MVE.LavaBackend(ctx)
    @test unpinned == pinned
    @test pinned == unpinned
    @test hash(unpinned) == hash(pinned)
    a = KA.allocate(unpinned, Float32, 8)
    @test KA.get_backend(a) == Mantle.defaultbackend()
    @test KA.get_backend(a) == unpinned

    bq2 = Mantle.allocate_batch_queue!(ctx)
    try
        other = MVE.LavaBackend(bq2)
        @test other != unpinned
        @test other != pinned
        @test MVE.LavaBackend(bq2, bq2) == other
        # The split spelling: same dispatch queue, another upload queue, is not
        # the same backend.
        @test MVE.LavaBackend(ctx.default_bq, bq2) != pinned
    finally
        Mantle.release_batch_queue!(bq2)
    end
end
