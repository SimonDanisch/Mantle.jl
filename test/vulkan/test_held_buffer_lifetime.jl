# A buffer a live recording names must never be destroyed underneath it.
#
# HW-accel BLAS/HWTLAS teardown (`destroy_now!` -> `unsafe_free!(as.storage)` and
# its `preserves`) frees arrays that a still-open one-shot has already named.
# Before the claim `hold!` takes, that path broke submit in two ways: it dropped
# the array's own `DataRef`, and it destroyed the `VkManagedBuffer` while a
# command buffer that names it could still be submitted.
#
# The timeline cannot answer this case on its own: a stamp is written at SUBMIT,
# so a buffer named by a closed-but-unsubmitted one-shot reads as never
# submitted and looks idle to every timeline check. `Stamp.holders` is the fact
# that closes it, and it is core's (`graph/lifetime.jl`): a destroy REQUESTED
# while something can still submit is recorded and run later, never refused and
# never performed early.
#
# This was `test_pinned_buffer_lifetime.jl`, over `pins` / `free_requested` /
# `deferred_frees` in the backend.

using Test, Lava, Mantle, KernelAbstractions
const KA = KernelAbstractions

state(buf) = @atomic :acquire buf.state
holders(buf) = @atomic Mantle.stampof(buf).holders

@testset "a held buffer survives unsafe_free! until its submission passes" begin
    be = LavaBackend()
    bq = MVE.vk_context().default_bq

    a = KA.allocate(be, Float32, 64)
    buf = a.buf[]                    # hold the VkManagedBuffer itself
    @test state(buf) == MVE.BUF_STATE_ALIVE
    @test holders(buf) == 0

    # A one-shot, written and sealed but NOT submitted — the window the
    # timeline cannot see.
    o = MVE.oneshot(bq) do e
        Mantle.hold!(e, a)
    end
    @test holders(buf) == 1

    # The teardown that used to corrupt the one-shot.
    Mantle.unsafe_free!(a)

    # The request is recorded, not performed: nothing has been destroyed, and a
    # drain that asks now declines because something can still submit.
    Mantle.drain!(bq)
    @test state(buf) != MVE.BUF_STATE_DEAD
    @test holders(buf) == 1

    # Submit it, wait, and drain: the hold goes with the submission, and the
    # destroy that was owed runs.
    Mantle.handover!(bq, Mantle.submit!(bq, o), o)
    MVE.vk_flush!(bq)
    @test holders(buf) == 0
    Mantle.drain!(bq)
    @test state(buf) == MVE.BUF_STATE_DEAD
end

@testset "a direct vk_free! on a held buffer is recorded, not performed" begin
    # `unsafe_free!(::LavaArray)` goes through the DataRef refcount; a caller
    # that frees the BUFFER directly bypasses it, so the claim has to hold on
    # its own.
    be = LavaBackend()
    bq = MVE.vk_context().default_bq

    a = KA.allocate(be, Float32, 64)
    buf = a.buf[]
    o = MVE.oneshot(bq) do e
        Mantle.hold!(e, a)
    end

    MVE.vk_free!(buf)               # e.g. a teardown path that owns the buffer
    Mantle.drain!(bq)
    @test state(buf) != MVE.BUF_STATE_DEAD
    @test holders(buf) == 1

    Mantle.handover!(bq, Mantle.submit!(bq, o), o)
    MVE.vk_flush!(bq)
    Mantle.drain!(bq)
    @test holders(buf) == 0
    @test state(buf) == MVE.BUF_STATE_DEAD
end

@testset "an unheld buffer is destroyed at the first drain" begin
    be = LavaBackend()
    a = KA.allocate(be, Float32, 64)
    buf = a.buf[]
    @test holders(buf) == 0
    Mantle.unsafe_free!(a)
    # Nothing named it, so nothing defers it beyond the drain that runs the
    # requests: no submission, no wait.
    Mantle.drain!(MVE.vk_context().default_bq)
    @test state(buf) == MVE.BUF_STATE_DEAD
end
