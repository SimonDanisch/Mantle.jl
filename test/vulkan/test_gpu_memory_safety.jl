# GPU Memory Safety & GC Correctness Regression Tests for Lava.jl
#
# What this file guards against — every section corresponds to a class of bug
# we have actually hit or are likely to hit:
#
#   1. Use-after-free at dispatch time
#   2. Double-free safety
#   3. GPU memory leak across many dispatch cycles (the 40 GiB bug)
#   4. `GC.gc()` during an open batch does not crash or UAF
#   5. Hardware atomic_fadd + subgroup_add produce correct results under contention
#   6. `vk_reduce_sum` does not leak per call (scratch buffer is reused)
#   7. Pool-block growth is bounded
#   8. Deferred-free lists actually drain
#   9. Derived arrays (views/reshape) keep parent alive via DataRef refcount
#  10. Arg-validation at launch catches freed/poisoned buffers
#  11. `VulkanBatchQueue` state (pinned set, deferred lists, slabs) stays bounded

using Test
using Lava, Mantle
using KernelAbstractions
using Atomix
using GPUArrays

const CTX = MVE.vk_context()
const BQ  = CTX.default_bq

function drain!()
    MVE.vk_flush!(BQ)
    GC.gc(true); GC.gc(true)
    MVE.drain!(BQ)
end

@testset "GPU Memory Safety" begin

    # ── 1. Use-after-free at dispatch time ──
    @testset "use-after-free detection" begin
        @kernel function noop_k!(x)
            i = @index(Global, Linear)
            @inbounds x[i] = Float32(i)
        end

        @testset "freed LavaArray rejected at launch" begin
            a = MVE.LavaArray(Float32[1, 2, 3])
            Mantle.unsafe_free!(a)
            @test_throws Lava.LavaError noop_k!(MVE.LavaBackend())(a; ndrange=3)
        end

        # Synthetic tests that poked buffer.address/size directly were removed
        # after the move to atomic lifecycle state (`@atomic state::UInt8`):
        # flipping the field manually violates the CAS invariants and crashes
        # the finalizer. The single `unsafe_free!` test above exercises the
        # real UAF-detection path that's actually reachable from user code.
    end

    # ── 2. Double-free safety ──
    @testset "double-free safety" begin
        a = MVE.LavaArray(Float32[1, 2, 3])
        Mantle.unsafe_free!(a)
        # Second unsafe_free! on the same LavaArray must be a no-op, not a crash.
        @test nothing === Mantle.unsafe_free!(a)
    end

    # ── 3. GPU memory leak across many alloc+free cycles ──
    #
    # The 40 GiB regression: pool_alloc was cutting a new 64 MiB block every
    # few cycles instead of reusing freed chunks. Fixed by the GC-retry path
    # in pool_alloc. This test is a pure allocator stress — no kernels — so
    # it isolates pool behavior from anything else.
    @testset "no GPU memory leak across 500 alloc+free cycles" begin
        drain!()
        baseline_bytes    = MVE.gpu_live_bytes()
        baseline_buffers  = MVE.live_buffer_count()
        baseline_pool     = length(MVE.poolblocks(MVE.vk_context()))

        for _ in 1:500
            a = MVE.LavaArray(Float32.(ones(4096)))
            Mantle.unsafe_free!(a)
        end
        drain!()

        @test MVE.gpu_live_bytes()      == baseline_bytes
        @test MVE.live_buffer_count()  == baseline_buffers
        # Pool blocks can grow once or twice under transient pressure but must
        # not keep growing — anything looser stops being a real leak test.
        @test length(MVE.poolblocks(MVE.vk_context()))   <= baseline_pool + 2
    end

    # ── 3b. Dispatch-then-free stress (the actual WaterLily pattern) ──
    #
    # Unlike #3 which is pure allocator, this exercises the full batch
    # lifecycle: allocate, dispatch, free, flush, repeat. The pin in
    # `batch.pinned` should keep the underlying VkManagedBuffer alive until
    # after submit; the DataRef finalizer that fires mid-batch should defer
    # the vk_free.
    @testset "dispatch + free + flush cycles" begin
        @kernel function touch_k!(a)
            i = @index(Global, Linear)
            @inbounds a[i] = Float32(i)
        end

        drain!()
        baseline_bytes = MVE.gpu_live_bytes()

        for _ in 1:50
            a = MVE.LavaArray(Float32.(ones(4096)))
            touch_k!(MVE.LavaBackend())(a; ndrange=4096)
            MVE.vk_flush!(BQ)          # submit so the pin releases
            Mantle.unsafe_free!(a)
        end
        drain!()

        # Allow a small one-time bump for arg-slab / indirect-slab pool growth
        # on the first dispatch — those allocate once then reuse forever.
        # What we're catching is UNBOUNDED growth (50 × 16 KiB = 800 KiB per
        # leak), not steady-state slab allocations.
        @test MVE.gpu_live_bytes() <= baseline_bytes + 16 * 1024^2   # 16 MiB ceiling
    end

    # ── 4. GC.gc() during an open batch must not crash or UAF ──
    #
    # We record 64 dispatches WITHOUT flushing, then force a full GC. The
    # batch's `pinned` set must hold the LavaArray strongly enough that no
    # finalizer fires on live buffers. If it ever does, vk_flush + Array()
    # will read garbage or segfault.
    @testset "GC during open batch is safe" begin
        @kernel function inc_k!(a)
            i = @index(Global, Linear)
            @inbounds a[i] += 1.0f0
        end

        a = MVE.LavaArray(zeros(Float32, 256))
        for _ in 1:64
            inc_k!(MVE.LavaBackend())(a; ndrange=256)
            GC.gc(false)            # force a young-gen sweep mid-batch
        end
        GC.gc(true)                 # full sweep too
        MVE.vk_flush!(BQ)
        @test all(x -> x ≈ 64.0f0, Array(a))
        Mantle.unsafe_free!(a)
    end

    # ── 5. Hardware atomic_fadd + subgroup_add correctness under contention ──
    #
    # This is the kernel pattern vk_reduce_sum uses. Every lane contributes,
    # subgroup-reduces, and one atomic per subgroup hits a shared counter. If
    # the hardware atomic path is broken or the subgroup reduce returns the
    # wrong lane's value, the counter disagrees with `total_threads`.
    @testset "atomic_fadd + subgroup_add correctness" begin
        @kernel function sg_atomic_add!(out)
            x = 1f0
            s = Lava.subgroup_add(x)
            if Lava.subgroup_elect()
                Atomix.@atomic out[1] += s
            end
            nothing
        end

        for n in (64, 256, 1024, 16384, 131072)
            out = MVE.LavaArray(Float32[0f0])
            sg_atomic_add!(MVE.LavaBackend(), 64)(out; ndrange=n)
            MVE.vk_flush!(BQ)
            @test Array(out)[1] ≈ Float32(n)
            Mantle.unsafe_free!(out)
        end
    end

    # ── 6. vk_reduce_sum does not leak per call ──
    #
    # The scratch output is cached on `ctx.caches.reduce_scratch`. If we
    # accidentally re-allocate per call, GPU_LIVE_BYTES grows. This catches
    # the regression where scratch allocation was inside the hot function.
    @testset "vk_reduce_sum scratch is cached" begin
        drain!()
        baseline = MVE.gpu_live_bytes()
        a = MVE.LavaArray(rand(Float32, 100_000))
        for _ in 1:1000
            MVE.vk_reduce_sum(a)
        end
        drain!()
        # One call may allocate a 4-byte scratch the first time. After that it
        # must reuse the same LavaArray forever.
        @test MVE.gpu_live_bytes() <= baseline + 1 << 20  # generous 1 MiB ceiling
        Mantle.unsafe_free!(a)
    end

    # ── 7. Pool-block growth is bounded even under bursty allocation ──
    #
    # The ω=2 bug: high tail velocity made the sim allocate large temp buffers
    # that forced new 64 MiB pool blocks every frame. The GC-retry path in
    # pool_alloc should now bound this. Stress it.
    @testset "pool blocks bounded under bursty alloc" begin
        drain!()
        baseline_pool = length(MVE.poolblocks(MVE.vk_context()))
        # Allocate + free 8 MiB arrays repeatedly. Each alloc hits the pool
        # (< 64 MiB pool block size), and frees are recorded on the retired list.
        for _ in 1:50
            a = MVE.LavaArray{Float32}(undef, 2_000_000)
            Mantle.unsafe_free!(a)
        end
        drain!()
        @test length(MVE.poolblocks(MVE.vk_context())) <= baseline_pool + 1
    end

    # ── 8. Deferred-free lists drain on submit ──
    @testset "the retired list drains" begin
        drain!()
        for _ in 1:10
            MVE.LavaArray(Float32[1, 2, 3, 4])  # orphaned, will be GC'd + deferred
        end
        GC.gc(true)
        drain!()
        @test length(BQ.pending)  == 0
        @test length(BQ.retiring) == 0
    end

    # ── 9. Derived arrays (views/reshape) keep parent alive via DataRef ──
    @testset "derived arrays keep parent alive" begin
        @testset "GPUArrays.derive shares the DataRef" begin
            parent = MVE.LavaArray(Float32[10, 20, 30, 40, 50])
            child = GPUArrays.derive(Float32, parent, (3,), 1)
            @test parent.buf[] === child.buf[]
            Mantle.unsafe_free!(parent)           # drops parent's refcount
            # Child must still be usable.
            @test child.buf[].size > 0
            @test child.buf[].address != MVE.BDA_POISON
            @test Array(child) == Float32[20, 30, 40]
            Mantle.unsafe_free!(child)
            drain!()
        end

        @testset "reshape shares the DataRef" begin
            a = MVE.LavaArray(Float32[1, 2, 3, 4, 5, 6])
            b = reshape(a, 2, 3)
            @test a.buf[] === b.buf[]
            Mantle.unsafe_free!(a)
            @test Array(b) == Float32[1 3 5; 2 4 6]
            Mantle.unsafe_free!(b)
            drain!()
        end
    end

    # ── 10. Arg-validation on/off toggle ──
    @testset "launch arg validation toggle" begin
        @test MVE.vk_context().diag.launch_arg_validation == true
        MVE.vk_context().diag.launch_arg_validation = false
        try
            @test MVE.vk_context().diag.launch_arg_validation == false
        finally
            MVE.vk_context().diag.launch_arg_validation = true
        end
        @test MVE.vk_context().diag.launch_arg_validation == true
    end

    # ── 11. VulkanBatchQueue state stays bounded across a long session ──
    @testset "VulkanBatchQueue state bounded" begin
        @kernel function cheap_k!(a)
            i = @index(Global, Linear)
            @inbounds a[i] += 1.0f0
        end

        a = MVE.LavaArray(zeros(Float32, 64))
        drain!()
        # What the unified arena holds BEFORE the burst. Not zero, and asserting
        # zero is what this used to do: a recorded plan owns its argument memory
        # there for as long as it lives, so any plan an earlier file left alive
        # makes an absolute count fail on file order. The claim is about these
        # 500 dispatches — their regions belong to batches, and every batch is
        # reclaimed — so it is a DELTA.
        before = sum(b -> length(b.live), MVE.unifiedblocks(MVE.ctxof(BQ)); init = 0)
        # 50 flushes × 10 dispatches = 500 total dispatches across many batches
        for _ in 1:50
            for _ in 1:10
                cheap_k!(MVE.LavaBackend())(a; ndrange=64)
            end
            MVE.vk_flush!(BQ)
        end
        drain!()

        @test length(BQ.outstanding)       == 0
        @test length(BQ.pending)  == 0
        @test length(BQ.retiring) == 0
        # And the argument memory those 500 dispatches read is back: every
        # region belonged to a batch, and every batch has been reclaimed. A
        # handful of blocks, not one per hundred launches.
        @test length(MVE.unifiedblocks(MVE.ctxof(BQ))) <= 4
        @test sum(b -> length(b.live), MVE.unifiedblocks(MVE.ctxof(BQ)); init = 0) == before

        @test Array(a)[1] ≈ 500f0  # sanity: kernel did run 500 times
        Mantle.unsafe_free!(a)
    end

    # ── 12. Reset to clean state at end of suite ──
    drain!()
end
