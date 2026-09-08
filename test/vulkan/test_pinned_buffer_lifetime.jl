# A buffer pinned by a live one-shot must never be freed underneath it.
#
# HW-accel BLAS/HWTLAS teardown (`destroy_now!` -> `unsafe_free!(as.storage)` and
# its `preserves`) frees arrays that a still-open one-shot has already pinned.
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
# one-shot reads as never-written and looks idle to the timeline check.

using Test, Lava, KernelAbstractions
const KA = KernelAbstractions

@testset "pinned buffer survives unsafe_free! until its one-shot releases" begin
    be = LavaBackend()
    ctx = MVE.vk_context()
    bq = ctx.default_bq

    a = KA.allocate(be, Float32, 64)
    buf = a.buf[]                    # hold the VkManagedBuffer itself
    @test (@atomic buf.state) == MVE.BUF_STATE_ALIVE
    @test (@atomic buf.pins) == 0

    o = MVE.oneshot(bq) do e
        MVE.pin!(e.owner, a)
    end
    @test (@atomic buf.pins) == 1

    # The teardown that used to corrupt the one-shot.
    Mantle.unsafe_free!(a)

    # The one-shot can still submit, so the buffer stays ALIVE and syncable. Both
    # assertions below are exactly what used to fail.
    @test (@atomic buf.state) == MVE.BUF_STATE_ALIVE
    @test (@atomic buf.pins) == 1
    MVE.sync_access!(MVE.Submission(bq), buf)    # must not throw or assert

    # One-shot retires -> pin drops -> refcount reaches zero -> buffer is freed.
    MVE.recycle!(bq, o)
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
    o = MVE.oneshot(bq) do e
        MVE.pin!(e.owner, a)
    end

    MVE.vk_free!(buf)               # e.g. a teardown path that owns the buffer

    # Must not be marked DEFERRED: that is what trips sync_access!'s ALIVE assert.
    @test (@atomic buf.state) == MVE.BUF_STATE_ALIVE
    @test (@atomic buf.free_requested)
    MVE.sync_access!(MVE.Submission(bq), buf)

    MVE.recycle!(bq, o)
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
