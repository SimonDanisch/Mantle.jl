# A buffer pinned by a live batch must never be freed underneath it.
#
# HW-accel BLAS/HWTLAS teardown (`destroy_now!` -> `unsafe_free!(as.storage)` and
# its `preserves`) frees arrays that a still-open batch has already pinned.
# Before the pin accounting that path broke submit in two different ways:
#
#   * it dropped the array's own DataRef, so `submit!` -> `sync_access!` threw
#     `ArgumentError: Attempt to use a freed reference` (DataRef throws on its
#     per-ref `freed` flag regardless of refcount), and
#   * it marked the VkManagedBuffer DEFERRED, so once the DataRef was retained
#     the next failure was `AssertionError: sync_access!: buffer is not ALIVE
#     (state=1)`.
#
# `vk_free!`'s timeline guard cannot cover this: it keys off `last_write`, which
# `sync_access!` only writes at submit, so a buffer pinned into an unsubmitted
# batch reads as never-written and looks idle to the timeline check.

using Test, Lava, KernelAbstractions
const KA = KernelAbstractions

@testset "pinned buffer survives unsafe_free! until its batch releases" begin
    be = LavaBackend()
    ctx = MVE.vk_context()
    bq = ctx.default_bq

    a = KA.allocate(be, Float32, 64)
    buf = a.buf[]                    # hold the VkManagedBuffer itself
    @test (@atomic buf.state) == MVE.BUF_STATE_ALIVE
    @test (@atomic buf.pins) == 0

    batch = Mantle.ensure_active_batch!(bq)
    MVE.pin!(batch, a)
    @test (@atomic buf.pins) == 1

    # The teardown that used to corrupt the batch.
    Mantle.unsafe_free!(a)

    # The batch can still submit, so the buffer stays ALIVE and syncable. Both
    # assertions below are exactly what used to fail.
    @test (@atomic buf.state) == MVE.BUF_STATE_ALIVE
    @test (@atomic buf.pins) == 1
    MVE.sync_access!(batch, buf)    # must not throw or assert

    # Batch retires -> pin drops -> refcount reaches zero -> buffer is freed.
    MVE.release_pinned_refs!(batch)
    @test (@atomic buf.pins) == 0
    @test (@atomic buf.state) != MVE.BUF_STATE_ALIVE
end

# The retained DataRef keeps the refcount above zero, so `unsafe_free!(::LavaArray)`
# never reaches `vk_free!` while pinned.  Callers that free the buffer *directly*
# bypass that, so the pin gate has to hold on its own.
@testset "direct vk_free! on a pinned buffer is owed, not performed" begin
    be = LavaBackend()
    bq = MVE.vk_context().default_bq

    a = KA.allocate(be, Float32, 64)
    buf = a.buf[]
    batch = Mantle.ensure_active_batch!(bq)
    MVE.pin!(batch, a)

    MVE.vk_free!(buf)               # e.g. a teardown path that owns the buffer

    # Must not be marked DEFERRED: that is what trips sync_access!'s ALIVE assert.
    @test (@atomic buf.state) == MVE.BUF_STATE_ALIVE
    @test (@atomic buf.free_requested)
    MVE.sync_access!(batch, buf)

    MVE.release_pinned_refs!(batch)
    @test (@atomic buf.pins) == 0
    @test (@atomic buf.state) != MVE.BUF_STATE_ALIVE
    @test !(@atomic buf.free_requested)
end

@testset "unpinned buffer is freed immediately (no regression in the normal path)" begin
    be = LavaBackend()
    a = KA.allocate(be, Float32, 64)
    buf = a.buf[]
    @test (@atomic buf.pins) == 0
    Mantle.unsafe_free!(a)
    # Nothing pinned it, so nothing defers it.
    @test (@atomic buf.state) != MVE.BUF_STATE_ALIVE
end
