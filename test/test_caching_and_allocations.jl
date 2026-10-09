# Caching, GC, and allocation tests.
#
# The kernel cache, argument memory reuse, what a launch holds once it has run,
# the pool's accounting of an allocation and its free, and that long runs of
# compile/dispatch cycles and broadcasts give their memory back.
#
# Written against the Vulkan backend's internals (`vk_context().caches`, the
# batch queue's one-shots, `gpu_live_bytes`, `live_buffer_count`), and asserted
# now through what those stood for: the compile counter, and the bytes the
# device's pool has handed out.

using Test, Mantle, KernelAbstractions
include(joinpath(@__DIR__, "testbackend.jl"))
const KA = KernelAbstractions

"""
Hand back everything dropped or freed so far: collect, so finalizers retire what
nothing references; a launch and a synchronize, which is where a backend runs the
destroys it has been asked for; and the pool's two phases around a wait, so a
retired region is released once the device is past it.
"""
function settle!(dev, scratch)
    GC.gc(true)
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

"""Twenty rounds of two uploads, a broadcast over them, and an explicit free of
all three. A function, so nothing it made is still reachable once it returns."""
@noinline function broadcastrounds()
    for _ in 1:20
        a = Mantle.devicearray(TESTBACKEND, Float32.(rand(128)))
        b = Mantle.devicearray(TESTBACKEND, Float32.(rand(128)))
        c = a .+ b
        KA.synchronize(TESTBACKEND)
        Mantle.unsafe_free!(a)
        Mantle.unsafe_free!(b)
        Mantle.unsafe_free!(c)
    end
end

@testset "Caching & Allocations" begin
    dev = Mantle.Device(TESTBACKEND)
    scratch = KA.allocate(TESTBACKEND, Float32, 1024)

    # ── 1. Kernel cache ──
    @testset "kernel cache" begin
        # FIFO-eviction test removed: the hand-rolled LRU was replaced by
        # `GPUCompiler.cached_compilation`, which uses its own unbounded
        # MethodInstance-keyed Dict. If cache growth ever becomes a problem,
        # wire eviction back into `LINKED_KERNEL_CACHE` directly.

        @testset "cache hit produces correct results" begin
            a = Mantle.devicearray(TESTBACKEND, Float32, 64)
            @kernel function double_k!(x)
                i = @index(Global, Linear)
                @inbounds x[i] = Float32(i) * 2.0f0
            end

            # First call: compile
            double_k!(TESTBACKEND)(a; ndrange=64)
            KA.synchronize(TESTBACKEND)
            r1 = Array(a)

            # Second call: cache hit
            fill!(a, 0.0f0)
            KA.synchronize(TESTBACKEND)
            Mantle.resetkernelcompiles!(dev)
            double_k!(TESTBACKEND)(a; ndrange=64)
            KA.synchronize(TESTBACKEND)
            r2 = Array(a)

            @test r1 == r2
            @test r1[32] ≈ 64.0f0
            # …and it was a hit: the second launch compiled nothing.
            @test Mantle.kernelcompiles(dev).misses == 0

            Mantle.unsafe_free!(a)
        end
    end

    # ── 2. Pipeline cache eviction ──
    # Dropped: it bounded the Vulkan backend's in-memory pipeline dict
    # (`MAX_PIPELINE_CACHE_SIZE`, `vk_context().caches.pipelines`), a knob no other
    # backend has. That the latest of five distinct kernels still runs is what
    # every other testset here already shows.

    # ── 3. What a launch holds once it has run ──
    # Every launch is its own submission, and what it held goes with it once the
    # device is past it. This read the Vulkan queue (`outstanding` empty, every
    # free one-shot's `sync` list empty); the observable is that an array a
    # synchronized launch named is nobody's but its owner's: freeing it gives its
    # bytes back at the next reclaim.
    @testset "a synchronized launch holds nothing" begin
        @kernel function noop_cb!(x)
            i = @index(Global, Linear)
        end
        a = Mantle.Buffer(dev, Float32[1, 2, 3])
        noop_cb!(TESTBACKEND)(Mantle.storage(a); ndrange=3)
        KA.synchronize(TESTBACKEND)

        held = livebytes(dev)
        Mantle.free!(a)
        sp = Mantle.pool(dev)
        Mantle.reclaim!(sp, dev)
        Mantle.waitidle(dev)
        Mantle.reclaim!(sp, dev)
        @test livebytes(dev) <= held - 3 * sizeof(Float32)
    end

    # ── 4. Argument memory reuse ──
    # A launch's arguments are a `Region` of the device's pool where the backend
    # packs them itself (Vulkan's unified arena), owned by the submission that
    # recorded the dispatch and released when it is reclaimed. The second hundred
    # launches reuse the bytes the first hundred gave back, rather than growing
    # the pool. (Measured as the pool's whole footprint; it read the count of
    # unified blocks.)
    @testset "argument memory reuse" begin
        @testset "regions reused across synchronizes" begin
            a = Mantle.devicearray(TESTBACKEND, Float32, 16)
            @kernel function slab_k!(x)
                i = @index(Global, Linear)
                @inbounds x[i] = Float32(i)
            end
            sp = Mantle.pool(dev)

            for _ in 1:100
                slab_k!(TESTBACKEND)(a; ndrange=16)
            end
            KA.synchronize(TESTBACKEND)
            n_after_first = Mantle.reserved(sp)

            for _ in 1:100
                slab_k!(TESTBACKEND)(a; ndrange=16)
            end
            KA.synchronize(TESTBACKEND)
            n_after_second = Mantle.reserved(sp)

            @test n_after_second == n_after_first

            Mantle.unsafe_free!(a)
        end
    end

    # ── 5. Live memory accounting ──
    # The Vulkan backend's `maybe_collect` reads pressure from its live-bytes
    # counter, so the counter has to move with an allocation and back with its
    # free. The pool's ledger is the portable form of that counter.
    @testset "live bytes track an allocation and its free" begin
        settle!(dev, scratch)
        before = livebytes(dev)
        buf = Mantle.Buffer(dev, Float32, 256)        # 1024 bytes
        after_alloc = livebytes(dev)
        @test after_alloc >= before + 1024
        Mantle.free!(buf)
        # Freed is retired, not released: the region goes back once the device is
        # past it, which the reclaim around the wait establishes.
        sp = Mantle.pool(dev)
        Mantle.reclaim!(sp, dev)
        Mantle.waitidle(dev)
        Mantle.reclaim!(sp, dev)
        @test livebytes(dev) <= before
    end

    # Note: the original `staging buffer lifecycle` and `indirect slabs reset`
    # testsets checked `STAGING_BUF` / `INDIRECT_SLAB_{OFFSET,IDX}`, global
    # trackers that were fully removed in the VulkanBatchQueue-ownership refactor —
    # there is no current equivalent to assert against, and dispatch
    # correctness is already covered by the "correctness across many dispatch
    # cycles" testset below.

    # ── 8. Correctness across many compile-dispatch cycles ──
    @testset "correctness across many dispatch cycles" begin
        @testset "varied kernels produce correct results" begin
            N = 256
            a = Mantle.devicearray(TESTBACKEND, Float32, N)
            b = Mantle.devicearray(TESTBACKEND, Float32, N)

            # Kernel 1: identity
            @kernel function id_k!(dst, src)
                i = @index(Global, Linear)
                @inbounds dst[i] = src[i]
            end

            # Kernel 2: negate
            @kernel function neg_k!(dst, src)
                i = @index(Global, Linear)
                @inbounds dst[i] = -src[i]
            end

            # Kernel 3: add constant
            @kernel function add_k!(dst, src, c)
                i = @index(Global, Linear)
                @inbounds dst[i] = src[i] + c
            end

            fill!(a, 7.0f0)

            id_k!(TESTBACKEND)(b, a; ndrange=N)
            KA.synchronize(TESTBACKEND)
            @test all(Array(b) .≈ 7.0f0)

            neg_k!(TESTBACKEND)(b, a; ndrange=N)
            KA.synchronize(TESTBACKEND)
            @test all(Array(b) .≈ -7.0f0)

            add_k!(TESTBACKEND)(b, a, 3.0f0; ndrange=N)
            KA.synchronize(TESTBACKEND)
            @test all(Array(b) .≈ 10.0f0)

            Mantle.unsafe_free!(a)
            Mantle.unsafe_free!(b)
        end
    end

    # ── 9. Broadcast allocation stability ──
    # This counted the Vulkan backend's live buffers; the pool's ledger is the same
    # statement wherever the backend's arrays are the pool's.
    @testset "broadcast allocation stability" begin
        @testset "repeated broadcasts don't leak" begin
            settle!(dev, scratch)
            baseline = livebytes(dev)
            broadcastrounds()
            settle!(dev, scratch)
            @test livebytes(dev) <= baseline
        end
    end

    # ── 10. Unified buffer allocation ──
    # Dropped: it checked a raw `vk_alloc(…; unified = true)` buffer's mapped
    # pointer, device address and size, which are fields of the Vulkan buffer. A
    # host-visible device array is `devicearray(backend, T, dims; unified = true)`
    # where a backend offers one, and its contents are what a caller sees.
end
