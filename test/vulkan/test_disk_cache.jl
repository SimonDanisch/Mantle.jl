# Disk Cache Persistence Tests
#
# Tests that GPUCompiler's disk cache works for Lava's SPIR-V compilation:
# 1. Kernels defined in packages (not REPL) get cached to disk
# 2. Cache hit skips LLVM + SPIR-V compilation on second load
# 3. Cache invalidation works when kernel code changes
#
# NOTE: Disk caching only works for kernels defined in precompiled packages.
# REPL-defined kernels get build_id=nothing and are not disk-cached.
# These tests verify the in-memory two-tier cache and GPUCompiler integration.

using Test
using Lava, Mantle
using KernelAbstractions
using GPUCompiler

@testset "Kernel Cache" begin

    @testset "two-tier cache structure" begin
        # Compile a kernel
        @kernel function tier_test_kernel(dst, src)
            i = @index(Global)
            @inbounds dst[i] = src[i] + 1f0
        end

        backend = MVE.LavaBackend()
        a = MVE.LavaArray(Float32[1, 2, 3, 4])
        b = MVE.LavaArray(zeros(Float32, 4))

        # Clear caches
        empty!(MVE.vk_context().caches.linked)

        # First dispatch populates both tiers
        tier_test_kernel(backend)(b, a; ndrange=4)
        KernelAbstractions.synchronize(backend)

        @test length(MVE.vk_context().caches.linked) >= 1
        # Disk cache should have written files
        dir = MVE.lava_disk_cache_dir()
        @test isdir(dir) && !isempty(filter(f -> endswith(f, ".jls"), readdir(dir)))
        @test Array(b) == Float32[2, 3, 4, 5]
    end

    @testset "Tier 1 hit is fast" begin
        @kernel function speed_test_kernel(dst, src)
            i = @index(Global)
            @inbounds dst[i] = src[i] * 2f0
        end

        backend = MVE.LavaBackend()
        a = MVE.LavaArray(Float32[1, 2, 3, 4])
        b = MVE.LavaArray(zeros(Float32, 4))

        # Warm up (compile)
        speed_test_kernel(backend)(b, a; ndrange=4)
        KernelAbstractions.synchronize(backend)

        # Tier 1 hit should be very fast (< 1ms for the cache lookup alone)
        t = @elapsed for _ in 1:100
            speed_test_kernel(backend)(b, a; ndrange=4)
        end
        avg_us = t / 100 * 1e6
        @test avg_us < 500  # < 500μs per dispatch (including arg pack + vkCmd record)
    end

    @testset "Tier 2 repopulates Tier 1" begin
        @kernel function repop_test_kernel(dst, val::Float32)
            i = @index(Global)
            @inbounds dst[i] = val
        end

        backend = MVE.LavaBackend()
        c = MVE.LavaArray(zeros(Float32, 4))

        # Compile once
        repop_test_kernel(backend)(c, 42f0; ndrange=4)
        KernelAbstractions.synchronize(backend)

        # Simulate session-restart without disk: drop Tier 1 (the linked
        # kernel table). Tier 2 (on-disk or serialized) must repopulate it.
        # `KERNEL_INSERTION_ORDER` was removed when the insertion-order
        # tracking was folded into LINKED_KERNEL_CACHE itself; clearing the
        # cache is the only access we need.
        empty!(MVE.vk_context().caches.linked)
        # LAUNCH_PLAN_CACHE sits ABOVE both tiers: `launch_plan` returns a
        # cached LaunchPlan without ever calling
        # `get_compiled_kernel_and_pipeline`, so with only Tier 1 cleared the
        # next dispatch never reaches the compile path and Tier 1 stays empty —
        # the assertion below failed with `0 >= 1`. The plan cache postdates
        # this test (added by "perf: overlap recording with execution"), so drop
        # it too or the test measures nothing.
        empty!(MVE.vk_context().caches.launchplans)

        # Next dispatch should hit Tier 2 and repopulate Tier 1
        repop_test_kernel(backend)(c, 99f0; ndrange=4)
        KernelAbstractions.synchronize(backend)

        @test length(MVE.vk_context().caches.linked) >= 1
        @test Array(c) == fill(99f0, 4)
    end

    @testset "different workgroup sizes get separate cache entries" begin
        @kernel function wg_test_kernel(dst)
            i = @index(Global)
            @inbounds dst[i] = Float32(i)
        end

        backend = MVE.LavaBackend()
        d = MVE.LavaArray(zeros(Float32, 64))
        before = length(MVE.vk_context().caches.linked)

        wg_test_kernel(backend)(d; ndrange=64, workgroupsize=32)
        KernelAbstractions.synchronize(backend)
        after_32 = length(MVE.vk_context().caches.linked)

        wg_test_kernel(backend)(d; ndrange=64, workgroupsize=64)
        KernelAbstractions.synchronize(backend)
        after_64 = length(MVE.vk_context().caches.linked)

        # Two different workgroup sizes should create two cache entries
        @test after_64 > after_32
    end

    @testset "LavaLinkedKernel has correct fields" begin
        # Check that the linked kernel has all expected data
        for (key, linked) in MVE.vk_context().caches.linked
            @test linked isa MVE.LavaLinkedKernel
            @test !isempty(linked.compiled.spirv_bytes)
            @test !isempty(linked.compiled.entry_name)
            @test linked.compiled.workgroup_size isa NTuple{3, Int}
            @test !isempty(linked.offsets)
            @test length(linked.byval_sizes) == length(linked.offsets)
            break  # just check first entry
        end
    end

    # GPUCompiler 2.0 removed its disk cache entirely — `disk_cache_path` and
    # `cache_file` no longer exist, and there is no `disk_cache` anywhere in
    # 2.1.1 (the replacement direction is `cached_results`). This used to assert
    # on `GPUCompiler.disk_cache_path()` and became an UndefVarError on the
    # upgrade from 1.23.
    #
    # Nothing in Lava's src depended on it: Lava's own two-tier cache is
    # `lava_disk_cache_*` in launch.jl and is unaffected. What is still worth
    # testing is the behaviour the removed assertions surrounded — that a kernel
    # defined outside a package (so never precompiled, and never disk-cacheable
    # under any scheme) still compiles and runs.
    @testset "REPL-defined kernels compile without any disk cache" begin
        @kernel function repl_kernel(dst)
            i = @index(Global)
            @inbounds dst[i] = 1f0
        end

        backend = MVE.LavaBackend()
        e = MVE.LavaArray(zeros(Float32, 4))
        repl_kernel(backend)(e; ndrange=4)
        KernelAbstractions.synchronize(backend)

        # REPL kernels should work even without disk cache
        @test Array(e) == fill(1f0, 4)
    end
end
