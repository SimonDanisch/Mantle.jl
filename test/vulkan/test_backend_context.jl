"""
A backend knows which device it runs on, and it always did.

`vk_context(backend)` and `vk_context(array)` name a path that already existed:
`VulkanBatchQueue.ctx` and `Buffer.ctx`. The value here is the name, not new state —
the four module-scope caches that hold device-owned handles (`PIPELINE_CACHE`,
`LINKED_KERNEL_CACHE`, `LAUNCH_PLAN_CACHE`, `GFX_PIPELINE_CACHE`) need a device
to key by, and every call site reaching for `b.dispatch_bq.ctx` by hand is a
place that can quietly reach for `vk_context()` instead.

**A correction is recorded here rather than in a commit message, because the
wrong version briefly shipped.** A `ctx` field was added to `LavaBackend` on the
belief that no path existed — read out of the first half of `VulkanBatchQueue`'s field
list, where `ctx::Any` sits sixty-odd lines down past the command-buffer pools.
Both project briefs said the carrier existed and both were right. The field was
removed: a second copy of a fact the queue already holds can only disagree with
it, and the disagreement would be silent.

Nothing here needs a second GPU. What it pins is that a backend, however
constructed, resolves to the device it will actually dispatch on.
"""

using Test, Lava, KernelAbstractions
const KA = KernelAbstractions

@testset "a backend knows its device" begin
    ctx = MVE.vk_context()

    # ── Every construction form resolves to the device it dispatches on.
    @test MVE.vk_context(LavaBackend(ctx))                          === ctx
    @test MVE.vk_context(LavaBackend(ctx.default_bq))               === ctx
    @test MVE.vk_context(LavaBackend(ctx.default_bq, ctx.default_bq)) === ctx

    # ── Including the unpinned one, which stores no queue at all and resolves
    #    through `vk_context()` at access — the property that lets a
    #    module-level `const BACKEND = LavaBackend()` survive `reset_device!`.
    unpinned = LavaBackend()
    pinned   = LavaBackend(ctx)
    @test getfield(unpinned, :dispatch_bq) === nothing
    @test MVE.vk_context(unpinned) === ctx

    # ── The round trip that makes it useful: an array knows its device, and the
    #    backend derived from it agrees. This is the path a per-device cache key
    #    would travel.
    a = MVE.LavaArray(zeros(Float32, 8))
    @test MVE.vk_context(a) === ctx
    @test MVE.vk_context(KA.get_backend(a)) === ctx

    # ── And it still works as a backend.
    fill!(a, 2.0f0)
    KA.synchronize(pinned)
    @test Array(a) == fill(2.0f0, 8)
end
