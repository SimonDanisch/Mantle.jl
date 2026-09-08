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

    @testset "the timeline only clears a region after a real sync" begin
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
