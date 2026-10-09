# GPU Memory Safety & GC Correctness Regression Tests
#
# What this file guards against — every section corresponds to a class of bug
# we have actually hit or are likely to hit:
#
#   1. Use-after-free at dispatch time
#   2. Double-free safety
#   3. GPU memory leak across many dispatch cycles (the 40 GiB bug)
#   4. `GC.gc()` with dispatches in flight does not crash or UAF
#   5. Atomic add + subgroup reduce produce correct results under contention
#   6. A reduction does not leak per call (its scratch is reused)
#   7. Pool-block growth is bounded
#   8. Deferred frees actually drain
#   9. Derived arrays (views/reshape) keep parent alive via DataRef refcount
#  11. Launch bookkeeping (argument memory) stays bounded over a long session
#
# Written against the Vulkan backend (`vk_context()`, its batch queue's lists,
# `gpu_live_bytes`, `live_buffer_count`, `poolblocks`), and asserted now through
# what those stood for: the bytes the device's pool has handed out, and the
# blocks it holds. A backend array is the pool's on Vulkan and Metal.jl's own on
# Metal, so the allocator sections also run on `Mantle.Buffer`, which is the
# pool's on every backend.

using Test
using Mantle
using KernelAbstractions
using Atomix
using GPUArrays
import KernelInterface as KI
include(joinpath(@__DIR__, "testbackend.jl"))
const KA = KernelAbstractions

"""
Hand back everything dropped or freed so far: collect, so finalizers retire what
nothing references; a launch and a synchronize, which is where a backend runs the
destroys it has been asked for; and the pool's two phases around a wait, so a
retired region is released once the device is past it.
"""
function settle!(dev, scratch)
    GC.gc(true); GC.gc(true)
    fill!(scratch, 0f0)
    KA.synchronize(TESTBACKEND)
    sp = Mantle.pool(dev)
    Mantle.reclaim!(sp, dev)
    Mantle.waitidle(dev)
    Mantle.reclaim!(sp, dev)
    return nothing
end

"""Bytes the device's pool has handed out and not had back."""
livebytes(dev) = sum(b -> sum(((lo, hi),) -> hi - lo, b.live; init = 0),
                     Iterators.flatten(values(Mantle.pool(dev).blocks)); init = 0)

"""How many blocks the device's pool holds, of every kind."""
nblocks(dev) = sum(length, values(Mantle.pool(dev).blocks); init = 0)

# Each loop below is a function of its own, so that once it returns nothing it
# allocated is still reachable from a stack slot of the caller: the assertions
# after it are about memory coming back, and a binding the compiler kept alive
# would read as a leak.

"""`n` cycles of allocating and explicitly freeing a 16 KiB backend array."""
@noinline function arraycycles(n)
    for _ in 1:n
        a = Mantle.devicearray(TESTBACKEND, Float32.(ones(4096)))
        Mantle.unsafe_free!(a)
    end
end

"""`n` cycles of allocating and explicitly freeing a 16 KiB `Mantle.Buffer`."""
@noinline function buffercycles(dev, n)
    for _ in 1:n
        b = Mantle.Buffer(dev, Float32.(ones(4096)))
        Mantle.free!(b)
    end
end

@kernel function touch_k!(a)
    i = @index(Global, Linear)
    @inbounds a[i] = Float32(i)
end

"""Fifty rounds of allocate, dispatch, synchronize, free."""
@noinline function dispatchcycles()
    for _ in 1:50
        a = Mantle.devicearray(TESTBACKEND, Float32.(ones(4096)))
        touch_k!(TESTBACKEND)(a; ndrange=4096)
        KA.synchronize(TESTBACKEND)   # so the launch has let go of it
        Mantle.unsafe_free!(a)
    end
end

"""Fifty 8 MiB backend arrays, each freed as soon as it exists."""
@noinline function burstarrays()
    for _ in 1:50
        a = Mantle.devicearray(TESTBACKEND, Float32, 2_000_000)
        Mantle.unsafe_free!(a)
    end
end

"""Fifty 8 MiB `Mantle.Buffer`s, each freed as soon as it exists."""
@noinline function burstbuffers(dev)
    for _ in 1:50
        b = Mantle.Buffer(dev, Float32, 2_000_000)
        Mantle.free!(b)
    end
end

"""Ten arrays nothing references, left to the collector."""
@noinline function orphans()
    for _ in 1:10
        Mantle.devicearray(TESTBACKEND, Float32[1, 2, 3, 4])  # orphaned, will be GC'd + deferred
    end
end

@kernel function sg_atomic_add!(out)
    x = 1f0
    s = KI.sub_group_reduce_add(x)
    if KI.get_sub_group_local_id(UInt32) == UInt32(1)
        Atomix.@atomic out[1] += s
    end
end

@testset "GPU Memory Safety" begin
    dev = Mantle.Device(TESTBACKEND)
    scratch = KA.allocate(TESTBACKEND, Float32, 1024)

    # ── 1. Use-after-free at dispatch time ──
    @testset "use-after-free detection" begin
        @kernel function noop_k!(x)
            i = @index(Global, Linear)
            @inbounds x[i] = Float32(i)
        end

        # The exception is the backend's (a `LavaError` on Vulkan, Metal.jl's
        # freed-reference error on Metal); what matters is that the launch is
        # refused rather than run on freed memory.
        @testset "freed array rejected at launch" begin
            a = Mantle.devicearray(TESTBACKEND, Float32[1, 2, 3])
            Mantle.unsafe_free!(a)
            @test_throws Exception noop_k!(TESTBACKEND)(a; ndrange=3)
        end

        # Synthetic tests that poked buffer.address/size directly were removed
        # after the move to atomic lifecycle state (`@atomic state::UInt8`):
        # flipping the field manually violates the CAS invariants and crashes
        # the finalizer. The single `unsafe_free!` test above exercises the
        # real UAF-detection path that's actually reachable from user code.
    end

    # ── 2. Double-free safety ──
    @testset "double-free safety" begin
        a = Mantle.devicearray(TESTBACKEND, Float32[1, 2, 3])
        Mantle.unsafe_free!(a)
        # Second unsafe_free! on the same array must be a no-op, not a crash.
        @test nothing === Mantle.unsafe_free!(a)
    end

    # ── 3. GPU memory leak across many alloc+free cycles ──
    #
    # The 40 GiB regression: pool_alloc was cutting a new 64 MiB block every
    # few cycles instead of reusing freed chunks. Fixed by the GC-retry path
    # in pool_alloc. This test is a pure allocator stress — no kernels — so
    # it isolates pool behavior from anything else.
    @testset "no GPU memory leak across 500 alloc+free cycles" begin
        for (what, cycles) in (("backend arrays", () -> arraycycles(500)),
                               ("Buffers", () -> buffercycles(dev, 500)))
            @testset "$what" begin
                settle!(dev, scratch)
                baseline_bytes  = livebytes(dev)
                baseline_blocks = nblocks(dev)

                cycles()
                settle!(dev, scratch)

                # A leak is GROWTH, and the 40 GiB regression grew both numbers:
                # it was `pool_alloc` cutting a new block per cycle. `<=` on the
                # bytes rather than `==`: a deferred free from an earlier file
                # landing inside the loop makes the bytes go DOWN (it once read 16
                # MiB less after the loop than before), and pinning that exactly
                # tests submission sequencing and not the allocator.
                @test livebytes(dev) <= baseline_bytes
                # Pool blocks can grow once or twice under transient pressure but
                # must not keep growing — anything looser stops being a real leak
                # test.
                @test nblocks(dev) <= baseline_blocks + 2
            end
        end
    end

    # ── 3b. Dispatch-then-free stress (the actual WaterLily pattern) ──
    #
    # Unlike #3 which is pure allocator, this exercises the full launch
    # lifecycle: allocate, dispatch, synchronize, free, repeat. The launch holds
    # the array until its submission has passed; the free that follows must
    # then give it back.
    @testset "dispatch + free + synchronize cycles" begin
        settle!(dev, scratch)
        baseline_bytes = livebytes(dev)

        dispatchcycles()
        settle!(dev, scratch)

        # Allow a small one-time bump for argument-memory growth on the first
        # dispatch — it is allocated once then reused forever. What we're catching
        # is UNBOUNDED growth (50 × 16 KiB = 800 KiB per leak), not steady-state
        # allocations.
        @test livebytes(dev) <= baseline_bytes + 16 * 1024^2   # 16 MiB ceiling
    end

    # ── 4. GC.gc() with dispatches in flight must not crash or UAF ──
    #
    # We launch 64 dispatches WITHOUT synchronizing, collecting after each, then
    # force a full GC. Every launch must hold its array strongly enough that no
    # finalizer fires on a live buffer. If it ever does, the read below will see
    # garbage or segfault.
    @testset "GC with dispatches in flight is safe" begin
        @kernel function inc_k!(a)
            i = @index(Global, Linear)
            @inbounds a[i] += 1.0f0
        end

        a = Mantle.devicearray(TESTBACKEND, zeros(Float32, 256))
        for _ in 1:64
            inc_k!(TESTBACKEND)(a; ndrange=256)
            GC.gc(false)            # force a young-gen sweep mid-flight
        end
        GC.gc(true)                 # full sweep too
        KA.synchronize(TESTBACKEND)
        @test all(x -> x ≈ 64.0f0, Array(a))
        Mantle.unsafe_free!(a)
    end

    # ── 5. Atomic add + subgroup reduce correctness under contention ──
    #
    # This is the kernel pattern a reduction uses (the Vulkan `vk_reduce_sum`).
    # Every lane contributes, subgroup-reduces, and one atomic per subgroup hits
    # a shared counter. If the atomic path is broken or the subgroup reduce
    # returns the wrong lane's value, the counter disagrees with
    # `total_threads`. Written with KernelInterface's `sub_group_reduce_add` and
    # sub-group lane (it named Lava's `subgroup_add` and `subgroup_elect`).
    @testset "atomic add + subgroup reduce correctness" begin
        for n in (64, 256, 1024, 16384, 131072)
            out = Mantle.devicearray(TESTBACKEND, Float32[0f0])
            sg_atomic_add!(TESTBACKEND, 64)(out; ndrange=n)
            KA.synchronize(TESTBACKEND)
            @test Array(out)[1] ≈ Float32(n)
            Mantle.unsafe_free!(out)
        end
    end

    # ── 6. A reduction does not leak per call ──
    #
    # The Vulkan reduction's scratch output is cached on the device. If it were
    # re-allocated per call, the pool would grow. This catches the regression
    # where scratch allocation was inside the hot function. (It called
    # `vk_reduce_sum`; `sum` is the portable reduction.)
    @testset "a reduction's scratch is reused" begin
        settle!(dev, scratch)
        baseline = livebytes(dev)
        a = Mantle.devicearray(TESTBACKEND, rand(Float32, 100_000))
        for _ in 1:1000
            sum(a)
        end
        Mantle.unsafe_free!(a)
        settle!(dev, scratch)
        # One call may allocate a 4-byte scratch the first time. After that it
        # must reuse the same array forever.
        @test livebytes(dev) <= baseline + 1 << 20  # generous 1 MiB ceiling
    end

    # ── 7. Pool-block growth is bounded even under bursty allocation ──
    #
    # The ω=2 bug: high tail velocity made the sim allocate large temp buffers
    # that forced new 64 MiB pool blocks every frame. The GC-retry path in
    # pool_alloc should now bound this. Stress it.
    @testset "pool blocks bounded under bursty alloc" begin
        # Allocate + free 8 MiB arrays repeatedly. Each alloc hits the pool
        # (< 64 MiB pool block size), and frees are recorded on the retired list.
        for (what, burst) in (("backend arrays", burstarrays),
                              ("Buffers", () -> burstbuffers(dev)))
            @testset "$what" begin
                settle!(dev, scratch)
                baseline_blocks = nblocks(dev)
                burst()
                settle!(dev, scratch)
                @test nblocks(dev) <= baseline_blocks + 1
            end
        end
    end

    # ── 8. Deferred frees drain ──
    # This read the Vulkan queue's `pending` and `retiring` lists; what they hold
    # is memory not yet given back, so the pool's ledger says the same thing.
    @testset "the retired list drains" begin
        settle!(dev, scratch)
        baseline = livebytes(dev)
        orphans()
        settle!(dev, scratch)
        @test livebytes(dev) <= baseline
    end

    # ── 9. Derived arrays (views/reshape) keep parent alive via DataRef ──
    #
    # It also checked that parent and child share one Vulkan buffer and that the
    # child's buffer was not poisoned; what a user sees of that is the child
    # still reading the parent's data after the parent is freed.
    @testset "derived arrays keep parent alive" begin
        @testset "GPUArrays.derive shares the DataRef" begin
            parent = Mantle.devicearray(TESTBACKEND, Float32[10, 20, 30, 40, 50])
            child = GPUArrays.derive(Float32, parent, (3,), 1)
            Mantle.unsafe_free!(parent)           # drops parent's refcount
            # Child must still be usable.
            @test Array(child) == Float32[20, 30, 40]
            Mantle.unsafe_free!(child)
            settle!(dev, scratch)
        end

        @testset "reshape shares the DataRef" begin
            a = Mantle.devicearray(TESTBACKEND, Float32[1, 2, 3, 4, 5, 6])
            b = reshape(a, 2, 3)
            Mantle.unsafe_free!(a)
            @test Array(b) == Float32[1 3 5; 2 4 6]
            Mantle.unsafe_free!(b)
            settle!(dev, scratch)
        end
    end

    # ── 10. Launch arg validation toggle ──
    # Dropped: it set and read back the Vulkan backend's
    # `diag.launch_arg_validation` flag, and nothing else.

    # ── 11. Launch bookkeeping stays bounded across a long session ──
    @testset "launch bookkeeping bounded" begin
        @kernel function cheap_k!(a)
            i = @index(Global, Linear)
            @inbounds a[i] += 1.0f0
        end

        a = Mantle.devicearray(TESTBACKEND, zeros(Float32, 64))
        settle!(dev, scratch)
        # What the pool holds BEFORE the burst. Not zero, and asserting zero is
        # wrong: a recorded plan owns its argument memory there for as long as it
        # lives, so any plan an earlier file left alive makes an absolute count
        # fail on file order. The claim is about these 500 dispatches — their
        # argument regions belong to their submissions, and every submission is
        # reclaimed — so it is a DELTA.
        before = livebytes(dev)
        # 50 synchronizes × 10 dispatches = 500 total dispatches
        for _ in 1:50
            for _ in 1:10
                cheap_k!(TESTBACKEND)(a; ndrange=64)
            end
            KA.synchronize(TESTBACKEND)
        end
        settle!(dev, scratch)

        # The argument memory those 500 dispatches read is back: every region
        # belonged to a submission, and every submission has been reclaimed. A
        # handful of blocks of it, not one per hundred launches — where a backend
        # packs launch arguments into the pool at all (Vulkan's unified arena).
        @test length(Mantle.blocksof(Mantle.pool(dev), Mantle.Unified())) <= 4
        @test livebytes(dev) <= before

        @test Array(a)[1] ≈ 500f0  # sanity: kernel did run 500 times
        Mantle.unsafe_free!(a)
    end

    # ── 12. Reset to clean state at end of suite ──
    settle!(dev, scratch)
end
