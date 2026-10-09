# Who owns a queue handed out by `allocate_batch_queue!`, and what happens when
# nobody gives it back.
#
# A buffer records the queue that last wrote it, and its finalizer asks that
# queue's timeline whether the GPU has passed the write. The queue must not be
# reachable only from the buffers naming it: when a caller (a RayMakie Screen)
# and its buffers become garbage together, the queue's own finalizer could run
# first — and on Vulkan, `vkGetSemaphoreCounterValue` on a destroyed semaphore
# segfaults inside the driver rather than returning an error. That is the crash
# RayMakie's runtests.jl documents as the reason test_caching_gc_correctness.jl
# is excluded.
#
# The invariant that closes it: the device holds every handed-out queue, so a
# queue outlives every buffer allocated on it. `release_batch_queue!` is the only
# way out, and it drains first.
#
# Only on a device that hands out a second queue at all (`supports_batch_queue`);
# Metal has one submission channel per device and refuses to allocate another.
# The Vulkan version also asserted the context's list of handed-out queues and
# that a released hardware queue slot is reused; both are that backend's
# bookkeeping, and what a user sees of them is what is asserted below: queues
# come and go without error, and a dropped one does not take the process down.

using Test, Mantle, KernelAbstractions
include(joinpath(@__DIR__, "testbackend.jl"))
const KA = KernelAbstractions

@testset "batch queue lifetime" begin
    dev = Mantle.Device(TESTBACKEND)
    if !Mantle.supports_batch_queue(dev)
        # Asked anyway, the answer is a refusal rather than a stub that records nothing.
        @test_throws ArgumentError Mantle.allocate_batch_queue!(dev)
    else
        @testset "a queue is handed out and given back" begin
            bq = Mantle.allocate_batch_queue!(dev)
            @test bq !== Mantle.batchqueue(dev)
            @test Mantle.todevice(bq) === dev
            @test Mantle.release_batch_queue!(bq) === nothing
            # The slot it held can be handed out again.
            again = Mantle.allocate_batch_queue!(dev)
            @test Mantle.release_batch_queue!(again) === nothing
        end

        @testset "releasing twice is a no-op, releasing the primary is an error" begin
            bq = Mantle.allocate_batch_queue!(dev)
            Mantle.release_batch_queue!(bq)
            @test Mantle.release_batch_queue!(bq) === nothing
            # Core's contract for `release_batch_queue!`: a backend must refuse to
            # release the device's primary queue.
            @test_throws Exception Mantle.release_batch_queue!(Mantle.batchqueue(dev))
            # …and the primary still works after the refusal.
            a = KA.allocate(TESTBACKEND, Float32, 64)
            fill!(a, 2f0)
            KA.synchronize(TESTBACKEND)
            @test all(==(2f0), Array(a))
        end

        @testset "a buffer written while a dropped queue lives survives GC" begin
            # The crash, as directly as it can be staged: take a queue, upload a
            # buffer, flush the queue, drop every reference to both, then force
            # collection, so the queue and the buffer are garbage in the same
            # collection. The device's reference is what keeps the queue alive;
            # nothing here gives it back.
            let bq = Mantle.allocate_batch_queue!(dev)
                a = Mantle.devicearray(TESTBACKEND, ones(Float32, 4096))
                Mantle.flush!(bq)
                a = nothing
            end
            GC.gc(true)
            GC.gc(true)
            # Reaching here at all is the assertion — the failure mode is SIGSEGV,
            # not a thrown exception — and the device still works.
            b = Mantle.devicearray(TESTBACKEND, fill(3f0, 64))
            KA.synchronize(TESTBACKEND)
            @test all(==(3f0), Array(b))
        end
    end
end
