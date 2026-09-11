"""
The suballocator, on a real Metal device.

`test/test_pool.jl` drives the same allocator against a fake device that
allocates nothing, and that file is the specification: the pool's behaviour is
backend-independent, so a Metal device must produce the same answers. This one
re-asks the questions that a real driver could plausibly answer differently —
the ones about identity, reuse and coalescing, where "it allocated something and
handed back bytes" is not enough.

What is NOT re-tested here is the internals: bin arithmetic, the randomised
partition property and the double-release guard belong to `Pool` and are already
covered device-free. Re-running them against Metal would test Metal for nothing.
"""

using Test, Mantle, Metal
const MTL = Metal.MTL

@testset "Metal: the pool primitives" begin
    d = Mantle.Device(Mantle.MetalAPI())
    # A pool of this testset's OWN, not the device's.
    #
    # `Mantle.pool(d)` is process-wide, and once the second device cache was
    # removed (phase 1.6) it really is one per process — so anything that
    # rendered earlier in the session had already grown it, and three testsets
    # here silently depended on running first: "a new block was allocated",
    # "the acquire lands at offset 0", "trim! leaves nothing reserved". None of
    # those is a statement about the pool; they are statements about being
    # early. A private `Pool` over the same real device asserts the properties
    # exactly and in any order.
    p = Mantle.Pool()
    B = Mantle.Buffers()

    @testset "budgets are real numbers from the device" begin
        # Not defaults. `capacity`'s core fallback is `typemax(Int)` and
        # `maxalloc`'s is the same, so a backend that failed to answer would
        # sail through anything that only checked for "positive".
        @test 0 < Mantle.maxalloc(d) < typemax(Int)
        @test 0 < Mantle.capacity(d) < typemax(Int)
        @test Mantle.maxalloc(d) <= Mantle.capacity(d)
        @test Mantle.blocksize(d) > 0
    end

    @testset "a second acquire does not reach the device" begin
        # An absolute, because the pool is this testset's own: one block, and
        # both regions inside it.
        before = Mantle.reserved(p)
        @test before == 0
        r1 = Mantle.acquire!(p, d, B, nothing, 1000; blocksize = 4096)
        r2 = Mantle.acquire!(p, d, B, nothing, 1000; blocksize = 4096)
        # One new block, both regions in it — the headline property, and the
        # only one that says the pool is a pool.
        @test Mantle.reserved(p) == before + 4096
        @test Mantle.memoryof(r1) === Mantle.memoryof(r2)
        @test Mantle.offset(r1) != Mantle.offset(r2)
        @test Mantle.memoryof(r1) isa Metal.MTL.MTLBuffer
        Mantle.release!(r1); Mantle.release!(r2)
    end

    @testset "release coalesces" begin
        a = Mantle.acquire!(p, d, B, nothing, 1000; blocksize = 4096)
        b = Mantle.acquire!(p, d, B, nothing, 1000; blocksize = 4096)
        Mantle.release!(a); Mantle.release!(b)
        # Only correct if the two freed spans merged: neither alone has room
        # for 2048 at offset 0.
        c = Mantle.acquire!(p, d, B, nothing, 2048; blocksize = 4096)
        @test Mantle.offset(c) == 0
        Mantle.release!(c)
    end

    @testset "trim! hands the memory back" begin
        r = Mantle.acquire!(p, d, B, nothing, 8; blocksize = 4096)
        Mantle.release!(r)
        Mantle.trim!(p, d)
        @test Mantle.reserved(p) == 0
    end

    @testset "the legacy timeline only clears a region after a real sync" begin
        # On the LEGACY queue specifically, built here by name rather than taken
        # from the process device: what a fence means is the one thing the two
        # Metal generations disagree about, and this is the older answer. The
        # MTL4 counterpart is the testset below.
        d = Mantle.MetalDevice(Metal.device(), Mantle.LegacyQueue(Metal.device()))
        # This testset used to assert the opposite — that a region retired with
        # no command buffer recording is immediately reusable — on the reasoning
        # that `fence` reports what the GPU has already finished. That reasoning
        # holds for Vulkan, where this backend owns the submission. It does not
        # hold here: compute goes out through Metal.jl's own queue, so nothing
        # in Mantle ever advanced the counter, `fence` returned 0, and
        # `passed(d, 0)` was `0 >= 0` for every region ever retired. The pool
        # duly recycled memory that running kernels were still reading, and a
        # second scene in the same session rendered NaN.
        #
        # The contract now: a fence is fresh and unpassed until something has
        # actually waited for the device.
        f1 = Mantle.fence(d)
        f2 = Mantle.fence(d)
        @test f2 > f1                      # each retirement gets its own value
        @test !Mantle.passed(d, f2)        # …and nothing has cleared it yet

        @test Mantle.waitfor(d, f2)        # a real `Metal.synchronize()`
        @test Mantle.passed(d, f2)         # …after which it is genuinely done
        @test Mantle.passed(d, f1)         # and so is everything before it

        # A fence taken after the wait is again unpassed: the sync covered what
        # had been issued, not what will be.
        @test !Mantle.passed(d, Mantle.fence(d))
    end

    # The same question of an MTL4 queue, which answers it without touching the
    # device. This is what the whole generation is for: on the legacy path every
    # pooled region costs a `Metal.synchronize()` to reuse, which is why a
    # session that had opened and closed a few screens carried hundreds of
    # retired regions it could not release.
    if MTL.supports_mtl4(Metal.device())
        @testset "the MTL4 timeline clears a region without a sync" begin
            d4 = Mantle.MetalDevice(Metal.device(), Mantle.MTL4Queue(Metal.device()))
            # A READ, so two calls with nothing submitted between them agree.
            f1 = Mantle.fence(d4)
            @test Mantle.fence(d4) == f1
            # Nothing has been submitted, so nothing can be reading those bytes
            # and the region is reusable NOW — no drain, no wait, no submission.
            @test Mantle.passed(d4, f1)
            @test Mantle.waitfor(d4, f1)
            @test d4.queue.event.signaledValue == 0

            # And a token that a submission WILL signal is unpassed until it
            # does. One empty submission, through the same verbs a replay uses.
            sub = Mantle.opensubmit!(d4, MTL.MTLBuffer[])
            tok = Mantle.closesubmit!(d4, sub)
            @test tok == f1
            @test Mantle.waitfor(d4, tok)
            @test Mantle.passed(d4, tok)
            @test d4.queue.event.signaledValue >= tok
        end
    end

    @testset "storage modes order by accessibility" begin
        # The Metal analogue of Vulkan's usage bitmask. A `Shared` block can
        # serve a request that wanted `Private`; the reverse must not hold, or
        # a transient the CPU has to read lands in GPU-only memory.
        @test Mantle.compatible(d, Metal.SharedStorage, Metal.PrivateStorage)
        @test !Mantle.compatible(d, Metal.PrivateStorage, Metal.SharedStorage)
        @test Mantle.compatible(d, Metal.SharedStorage, Metal.SharedStorage)
        # An arena reconciles to the most accessible mode any tenant asked for.
        @test Mantle.mergeconstraints(d, B, Metal.PrivateStorage, Metal.SharedStorage) ===
              Metal.SharedStorage
        @test Mantle.mergeconstraints(d, B, Metal.SharedStorage, Metal.PrivateStorage) ===
              Metal.SharedStorage
    end
end

@testset "Metal: deviceview borrows, it does not own" begin
    d = Mantle.Device(Mantle.MetalAPI())
    # A pool of this testset's OWN, not the device's.
    #
    # `Mantle.pool(d)` is process-wide, and once the second device cache was
    # removed (phase 1.6) it really is one per process — so anything that
    # rendered earlier in the session had already grown it, and three testsets
    # here silently depended on running first: "a new block was allocated",
    # "the acquire lands at offset 0", "trim! leaves nothing reserved". None of
    # those is a statement about the pool; they are statements about being
    # early. A private `Pool` over the same real device asserts the properties
    # exactly and in any order.
    p = Mantle.Pool()
    B = Mantle.Buffers()

    a = Mantle.allocate(p, d, B, Float32, 64; blocksize = 4096)
    Mantle.upload!(d, a, 1, collect(1.0f0:64.0f0))

    mtl = Mantle.deviceview(d, a)
    @test mtl isa MtlArray{Float32,1}
    @test size(mtl) == (64,)
    # The GPU reads what the host wrote, through the same bytes.
    @test sum(mtl) == sum(1:64)

    # …and writes back through them. `download` must synchronise: unified memory
    # shares the bytes, not the ordering, and without the wait this reads what
    # was there before the kernel ran.
    mtl .*= 2.0f0
    @test Mantle.download(d, a) == collect(2.0f0:2.0f0:128.0f0)

    # A second region in the SAME block, to prove the view is offset-correct and
    # that writing one does not touch the other. This is the failure a borrowed
    # view gets wrong: an ignored offset reads the block's start every time and
    # every single-region test still passes.
    b = Mantle.allocate(p, d, B, Float32, 64; blocksize = 4096)
    @test Mantle.memoryof(Mantle.region(b)) === Mantle.memoryof(Mantle.region(a))
    @test Mantle.offset(b) != Mantle.offset(a)
    Mantle.upload!(d, b, 1, fill(-1.0f0, 64))
    vb = Mantle.deviceview(d, b)
    vb .= 5.0f0
    Metal.synchronize()
    @test all(==(5.0f0), Mantle.download(d, b))
    @test Mantle.download(d, a) == collect(2.0f0:2.0f0:128.0f0)   # untouched

    # Dropping the borrowed view frees nothing: the pool owns the buffer, and a
    # second owner would free it out from under every other tenant of the block.
    mtl = nothing; vb = nothing; GC.gc()
    @test Mantle.download(d, a) == collect(2.0f0:2.0f0:128.0f0)
    @test Mantle.reserved(p) > 0
end

# ── What `reclaim!` costs a frame that cannot release anything ───────────────
#
# `run!` calls `reclaim!` at its head, every frame, and the walk covers every
# retired region the device has not finished with. On THIS device that set is
# long and stays long: the timeline counts retirements, and `passed` only moves
# when something asks `waitfor` for a full `Metal.synchronize()` — so everything
# dropped since the last wait is still on the list, and the walk runs over all of
# it again on the next frame.
#
# `test_pool.jl` pins the same property against a device double, where the
# timeline is a field the test writes. This asks it of the real one, because that
# is where it was found: four RayMakie screens created and closed left ~600
# regions retired, and every frame after that paid 1200 allocations and 48 KB to
# walk them — on a frame whose own cost is 240 bytes. The list being long is not
# the defect and is not fixed here; walking it costing anything was.
@testset "Metal: reclaim! allocates nothing for a backlog it cannot release" begin
    # A LEGACY device, by name: a backlog that cannot be released without a drain
    # is that generation's property, and it is the one the 48 KB-per-frame walk
    # was found on.
    d = Mantle.MetalDevice(Metal.device(), Mantle.LegacyQueue(Metal.device()))
    p = Mantle.Pool()                       # this testset's own, as above
    B = Mantle.Buffers()
    rs = [Mantle.acquire!(p, d, B, nothing, 256; blocksize = 1 << 22) for _ in 1:200]
    for r in rs
        Mantle.retire!(p, r)
    end
    # First call stamps them with a fence this device has not passed. Asserted,
    # not assumed: if they were released here the measurement below would be of
    # an empty list and would pass for the wrong reason.
    Mantle.reclaim!(p, d)
    @test length(p.retiring) == 200
    @test length(p.retiring_at) == 200

    Mantle.reclaim!(p, d)                   # warm the walk
    bytes = [(@allocated Mantle.reclaim!(p, d)) for _ in 1:5]
    @test maximum(bytes) == 0
    @test length(p.retiring) == 200         # still nothing released

    # And it does release, once the device is asked to catch up — which is what
    # `wait = true` is for, and the only place in Mantle that waits at all.
    @test Mantle.reclaim!(p, d; wait = true) == 200
    @test isempty(p.retiring)
    @test isempty(p.retiring_at)
end

# The same measurement against the device the process actually runs on, which on
# this machine is the MTL4 one. The backlog is the legacy queue's problem — an
# MTL4 timeline releases a region as soon as the submission that could have been
# reading it has signalled, so the list never grows in the first place — but the
# property being measured is the WALK, and it has to cost nothing either way.
@testset "Metal: reclaim! allocates nothing, on the process device" begin
    d = Mantle.Device(Mantle.MetalAPI())
    p = Mantle.Pool()
    B = Mantle.Buffers()
    rs = [Mantle.acquire!(p, d, B, nothing, 256; blocksize = 1 << 22) for _ in 1:200]
    for r in rs
        Mantle.retire!(p, r)
    end
    Mantle.reclaim!(p, d)
    Mantle.reclaim!(p, d)                   # warm the walk
    bytes = [(@allocated Mantle.reclaim!(p, d)) for _ in 1:5]
    @test maximum(bytes) == 0
    # Whatever is left is releasable once the device has caught up, and asking
    # never leaves anything behind.
    Mantle.reclaim!(p, d; wait = true)
    @test isempty(p.retiring)
    @test isempty(p.retiring_at)
end
