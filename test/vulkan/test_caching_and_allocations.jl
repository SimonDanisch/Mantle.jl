# Caching, GC, and Allocation Tests for Lava.jl
#
# Tests kernel cache eviction, pipeline cache, staging buffer lifecycle,
# CB batching, and reset_device cleanup of global state.

using Test
using Lava, Mantle
using KernelAbstractions

@testset "Caching & Allocations" begin

    # ── 1. Kernel cache ──
    @testset "kernel cache" begin
        # FIFO-eviction test removed: the hand-rolled LRU was replaced by
        # `GPUCompiler.cached_compilation`, which uses its own unbounded
        # MethodInstance-keyed Dict. If cache growth ever becomes a problem,
        # wire eviction back into `LINKED_KERNEL_CACHE` directly.

        @testset "cache hit produces correct results" begin
            a = Mantle.LavaArray{Float32}(undef, 64)
            @kernel function double_k!(x)
                i = @index(Global, Linear)
                @inbounds x[i] = Float32(i) * 2.0f0
            end

            # First call: compile
            double_k!(Mantle.defaultbackend())(a; ndrange=64)
            Mantle.flush!(Mantle.Device())
            r1 = Array(a)

            # Second call: cache hit
            fill!(a, 0.0f0)
            double_k!(Mantle.defaultbackend())(a; ndrange=64)
            Mantle.flush!(Mantle.Device())
            r2 = Array(a)

            @test r1 == r2
            @test r1[32] ≈ 64.0f0

            Mantle.unsafe_free!(a)
        end
    end

    # ── 2. Pipeline cache eviction ──
    @testset "pipeline cache eviction" begin
        old_max = Mantle.MAX_PIPELINE_CACHE_SIZE[]
        Mantle.MAX_PIPELINE_CACHE_SIZE[] = 3

        try
            a = Mantle.LavaArray{Float32}(undef, 16)

            # Generate distinct pipelines via different kernel functions
            @kernel function pk1!(a)
                i = @index(Global, Linear)
                @inbounds a[i] = 1.0f0
            end
            @kernel function pk2!(a)
                i = @index(Global, Linear)
                @inbounds a[i] = 2.0f0
            end
            @kernel function pk3!(a)
                i = @index(Global, Linear)
                @inbounds a[i] = 3.0f0
            end
            @kernel function pk4!(a)
                i = @index(Global, Linear)
                @inbounds a[i] = 4.0f0
            end
            @kernel function pk5!(a)
                i = @index(Global, Linear)
                @inbounds a[i] = 5.0f0
            end

            pk1!(Mantle.defaultbackend())(a; ndrange=16)
            pk2!(Mantle.defaultbackend())(a; ndrange=16)
            pk3!(Mantle.defaultbackend())(a; ndrange=16)
            pk4!(Mantle.defaultbackend())(a; ndrange=16)
            pk5!(Mantle.defaultbackend())(a; ndrange=16)
            Mantle.flush!(Mantle.Device())

            @test length(Mantle.vk_context().caches.pipelines) <= 3
            @test length(Mantle.vk_context().caches.pipeline_order) <= 3

            # Latest pipeline still works
            @test Array(a)[1] ≈ 5.0f0

            Mantle.unsafe_free!(a)
        finally
            Mantle.MAX_PIPELINE_CACHE_SIZE[] = old_max
        end
    end

    # ── 3. One-shot data_refs lifecycle ──
    # Post-refactor: there is no open command buffer on the VulkanBatchQueue;
    # every launch is its own one-shot, submitted immediately.
    @testset "one-shot data_refs" begin
        @testset "data_refs cleared after flush" begin
            a = Mantle.LavaArray(Float32[1, 2, 3])
            @kernel function noop_cb!(x)
                i = @index(Global, Linear)
            end
            noop_cb!(Mantle.defaultbackend())(a; ndrange=3)

            ctx = Mantle.vk_context()
            Mantle.flush!(ctx.default_bq)
            bq = ctx.default_bq
            @test isempty(bq.outstanding)
            @test all(o -> isempty(o.sync), bq.free)

            Mantle.unsafe_free!(a)
        end
    end

    # ── 4. Argument memory reuse ──
    # A launch's arguments are a `Region` of the unified arena, owned by the
    # batch that recorded the dispatch and released when the batch is reclaimed.
    # Asserting block-count stability across two runs of dispatches is the same
    # signal the old `arg_slabs` length gave: the second hundred launches reuse
    # the bytes the first hundred gave back, rather than growing the pool.
    @testset "argument memory reuse" begin
        @testset "regions reused across flushes" begin
            a = Mantle.LavaArray{Float32}(undef, 16)
            @kernel function slab_k!(x)
                i = @index(Global, Linear)
                @inbounds x[i] = Float32(i)
            end
            ctx = Mantle.vk_context()

            for _ in 1:100
                slab_k!(Mantle.defaultbackend())(a; ndrange=16)
            end
            Mantle.flush!(ctx.default_bq)
            n_after_first = length(Mantle.unifiedblocks(ctx))

            for _ in 1:100
                slab_k!(Mantle.defaultbackend())(a; ndrange=16)
            end
            Mantle.flush!(ctx.default_bq)
            n_after_second = length(Mantle.unifiedblocks(ctx))

            @test n_after_second == n_after_first

            Mantle.unsafe_free!(a)
        end
    end

    # ── 5. GC pressure tracking ──
    # After the AMDGPU.jl-style refactor of `maybe_collect`, the byte counter
    # `GPU_BYTES_SINCE_LAST_GC` is gone — pressure is read directly from
    # `GPU_LIVE_BYTES / heap_size`.  Verify the live-bytes accounting still
    # increments on `vk_alloc` and decrements on `vk_free!`.
    @testset "GC pressure tracking" begin
        @testset "live_bytes tracks vk_alloc / vk_free!" begin
            bq = Mantle.batchqueue(Mantle.Device())
            before = Mantle.gpu_live_bytes()
            buf = Mantle.vk_alloc(bq, 1024)
            after_alloc = Mantle.gpu_live_bytes()
            @test after_alloc >= before + 1024
            Mantle.vk_free!(buf)
            # Buffer may be deferred (timeline gate); a sync ensures destroy
            # actually runs and decrements GPU_LIVE_BYTES.
            Mantle.flush!(Mantle.Device())
            Mantle.drain!(bq)
            after_free = Mantle.gpu_live_bytes()
            @test after_free <= after_alloc
        end
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
            a = Mantle.LavaArray{Float32}(undef, N)
            b = Mantle.LavaArray{Float32}(undef, N)

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

            id_k!(Mantle.defaultbackend())(b, a; ndrange=N)
            Mantle.flush!(Mantle.Device())
            @test all(Array(b) .≈ 7.0f0)

            neg_k!(Mantle.defaultbackend())(b, a; ndrange=N)
            Mantle.flush!(Mantle.Device())
            @test all(Array(b) .≈ -7.0f0)

            add_k!(Mantle.defaultbackend())(b, a, 3.0f0; ndrange=N)
            Mantle.flush!(Mantle.Device())
            @test all(Array(b) .≈ 10.0f0)

            Mantle.unsafe_free!(a)
            Mantle.unsafe_free!(b)
        end
    end

    # ── 9. Broadcast allocation stability ──
    # The live-buffer set is `mempolicy(ctx).live_buffers` now. The old
    # `flush_deferred_frees!` wrapper was replaced by per-channel
    # `Mantle.drain!`, which sweeps and reclaims in one call.
    @testset "broadcast allocation stability" begin
        @testset "repeated broadcasts don't leak" begin
            ctx = Mantle.vk_context()
            bq = ctx.default_bq
            GC.gc(true)
            Mantle.flush!(ctx.default_bq)
            Mantle.drain!(bq)
            baseline = Mantle.live_buffer_count()

            for _ in 1:20
                a = Mantle.LavaArray(Float32.(rand(128)))
                b = Mantle.LavaArray(Float32.(rand(128)))
                c = a .+ b
                Mantle.flush!(ctx.default_bq)
                Mantle.unsafe_free!(a)
                Mantle.unsafe_free!(b)
                Mantle.unsafe_free!(c)
            end

            Mantle.flush!(ctx.default_bq)
            Mantle.drain!(bq)
            after = Mantle.live_buffer_count()
            @test after == baseline
        end
    end

    # ── 10. Unified buffer allocation ──
    @testset "unified buffer allocation" begin
        @testset "mapped ptr is non-null" begin
            bq = Mantle.batchqueue(Mantle.Device())
            buf = Mantle.vk_alloc(bq, 256; unified=true)
            @test buf.mapped_ptr != Ptr{UInt8}(0)
            @test buf.address != 0
            @test buf.size >= 256
            Mantle.vk_free!(buf)
        end
    end
end
