"""
The suballocator, on a real device.

`test/test_pool.jl` drives the same allocator against a fake device that
allocates nothing, and that file is the specification: the pool's behaviour is
backend-independent, so a real device must produce the same answers. This one
re-asks the questions a real driver could plausibly answer differently — the ones
about identity, reuse and coalescing, where "it allocated something and handed
back bytes" is not enough — and the two a device answers itself: its budgets, and
a borrowed view of pool memory.

What is NOT re-tested here is the internals: bin arithmetic, the randomised
partition property and the double-release guard belong to `Pool` and are covered
device-free. Giving empty blocks back on the device's own pool is
`test_pool_trim.jl`.

Each testset takes a pool of its OWN, not the device's. `Mantle.pool(dev)` is one
per process, so anything that ran earlier has already grown it, and three
testsets here silently depended on running first: "a new block was allocated",
"the acquire lands at offset 0", "trim! leaves nothing reserved". None of those is
a statement about the pool; they are statements about being early.

The regions are `Persistent()`: host-visible blocks every backend allocates with a
constraint it knows without being shown a transient. A `Buffers()` block's
constraint is derived FROM its transients on Vulkan (the usage flags they need),
so it cannot be asked for with none.
"""

using Test, Mantle
import KernelAbstractions as KA
include(joinpath(@__DIR__, "testbackend.jl"))

const KIND = Mantle.Persistent()

@testset "the pool, on a real device" begin
    d = Mantle.Device(TESTBACKEND)

    @testset "budgets are real numbers from the device" begin
        # Not defaults. `capacity`'s core fallback is `typemax(Int)` and
        # `maxalloc`'s is the same, so a backend that failed to answer would sail
        # through anything that only checked for "positive".
        @test 0 < Mantle.maxalloc(d) < typemax(Int)
        @test 0 < Mantle.capacity(d) < typemax(Int)
        @test Mantle.maxalloc(d) <= Mantle.capacity(d)
        @test Mantle.blocksize(d) > 0
    end

    @testset "a second acquire does not reach the device" begin
        p = Mantle.Pool()
        @test Mantle.reserved(p) == 0
        r1 = Mantle.acquire!(p, d, KIND, nothing, 1000; blocksize = 4096)
        r2 = Mantle.acquire!(p, d, KIND, nothing, 1000; blocksize = 4096)
        # One new block, both regions in it — the headline property, and the only
        # one that says the pool is a pool.
        @test Mantle.reserved(p) == 4096
        @test Mantle.memoryof(r1) === Mantle.memoryof(r2)
        @test Mantle.offset(r1) != Mantle.offset(r2)
        Mantle.release!(r1); Mantle.release!(r2)
    end

    @testset "release coalesces" begin
        p = Mantle.Pool()
        a = Mantle.acquire!(p, d, KIND, nothing, 1000; blocksize = 4096)
        b = Mantle.acquire!(p, d, KIND, nothing, 1000; blocksize = 4096)
        Mantle.release!(a); Mantle.release!(b)
        # Only correct if the two freed spans merged: neither alone has room for
        # 2048 at offset 0.
        c = Mantle.acquire!(p, d, KIND, nothing, 2048; blocksize = 4096)
        @test Mantle.offset(c) == 0
        @test Mantle.reserved(p) == 4096
        Mantle.release!(c)
    end

    @testset "trim! hands the memory back" begin
        p = Mantle.Pool()
        r = Mantle.acquire!(p, d, KIND, nothing, 8; blocksize = 4096)
        @test Mantle.reserved(p) == 4096
        Mantle.release!(r)
        Mantle.trim!(p, d)
        @test Mantle.reserved(p) == 0
    end

    @testset "deviceview borrows, it does not own" begin
        p = Mantle.Pool()
        a = Mantle.allocate(p, d, KIND, Float32, 64; blocksize = 4096)
        Mantle.upload!(d, a, 1, collect(1.0f0:64.0f0))

        v = Mantle.deviceview(d, a)
        @test v isa AbstractVector{Float32}
        @test size(v) == (64,)
        # The device reads what the host wrote, through the same bytes.
        @test sum(v) == sum(1:64)

        # …and writes back through them. `download` must synchronise: the view
        # shares the bytes, not the ordering, and without the wait this reads what
        # was there before the kernel ran.
        v .*= 2.0f0
        @test Mantle.download(d, a) == collect(2.0f0:2.0f0:128.0f0)

        # A second region in the SAME block, to prove the view is offset-correct
        # and that writing one does not touch the other. This is the failure a
        # borrowed view gets wrong: an ignored offset reads the block's start every
        # time and every single-region test still passes.
        b = Mantle.allocate(p, d, KIND, Float32, 64; blocksize = 4096)
        @test Mantle.memoryof(Mantle.region(b)) === Mantle.memoryof(Mantle.region(a))
        @test Mantle.offset(b) != Mantle.offset(a)
        Mantle.upload!(d, b, 1, fill(-1.0f0, 64))
        vb = Mantle.deviceview(d, b)
        vb .= 5.0f0
        KA.synchronize(TESTBACKEND)
        @test all(==(5.0f0), Mantle.download(d, b))
        @test Mantle.download(d, a) == collect(2.0f0:2.0f0:128.0f0)   # untouched

        # Dropping the borrowed view frees nothing: the pool owns the memory, and a
        # second owner would free it out from under every other tenant of the
        # block.
        v = nothing; vb = nothing; GC.gc()
        @test Mantle.download(d, a) == collect(2.0f0:2.0f0:128.0f0)
        @test Mantle.reserved(p) > 0
    end

    # What `reclaim!` costs a frame that cannot release anything.
    #
    # `run!` calls `reclaim!` at its head, every frame, and the walk covers every
    # retired region the device has not finished with. `test_pool.jl` pins the
    # same property against a device double, where the timeline is a field the
    # test writes. This asks it of a real device, because that is where it was
    # found: four RayMakie screens created and closed on Metal left ~600 regions
    # retired, and every frame after that paid 1200 allocations and 48 KB to walk
    # them — on a frame whose own cost is 240 bytes. The list being long was not
    # the defect; walking it costing anything was.
    @testset "reclaim! allocates nothing" begin
        p = Mantle.Pool()
        rs = [Mantle.acquire!(p, d, KIND, nothing, 256; blocksize = 1 << 22) for _ in 1:200]
        for r in rs
            Mantle.retire!(p, d, r)
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
        Mantle.trim!(p, d)
    end
end
