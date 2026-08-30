# Test rapid GPU allocation/deallocation cycles that mirror the reference test pattern:
# - Many LavaArrays created per "test" (positions, colors, quad_offsets, etc.)
# - Each "test" creates and discards arrays in rapid succession
# - GC pressure from old arrays while new dispatches are recording
# - Simulates the pattern that crashes RayMakie reference tests on APU

#
# **This file called `Mantle.flush_deferred_frees!()` and read a global
# `Mantle.DEFERRED_FREES` until 2026-08-23.** Neither has existed since the
# deferred-free list became per-VulkanBatchQueue, so every testset below threw
# `UndefVarError` on its first line — and `runtests.jl` did not include the file,
# so nothing said so. It is included now. A test nothing runs is not a test.
#
# The replacements are the per-BQ API: `quiesce_before_reclaim!` for the flush +
# GC + drain that `flush_deferred_frees!` used to do (the drain on its own is not
# safe with a batch open — see its docstring), and the two `deferred_*` lists
# under the lock that guards them for the count.

using Test, Lava, KernelAbstractions

@kernel function fill_kernel!(dst, val)
    i = @index(Global, Linear)
    @inbounds dst[i] = val
end

"""Pending deferred frees on `bq`, read under the lock the finalizers push with."""
pendingfrees(bq) = lock(bq.deferred_frees_lock) do
    length(bq.deferred_frees) + length(bq.deferred_as_frees)
end

@testset "Rapid allocation/free cycles" begin
    backend = Mantle.LavaBackend()
    bq = Mantle.vk_context().default_bq
    drainfrees!() = Mantle.quiesce_before_reclaim!(bq)

    @testset "many small arrays created and discarded" begin
        # Simulate 50 "test iterations" each creating 10 arrays
        for iter in 1:50
            arrays = [Mantle.LavaArray(rand(Float32, 100)) for _ in 1:10]

            # Dispatch on each array
            for a in arrays
                fill_kernel!(backend)(a, Float32(iter); ndrange=100)
            end
            Mantle.vk_flush!(Mantle.vk_context())

            # Verify last array
            result = Array(arrays[end])
            @test all(x -> x == Float32(iter), result)

            # Let arrays go out of scope — GC should handle cleanup
        end
        GC.gc(true)
        drainfrees!()
        @test true  # Didn't crash
    end

    @testset "allocation during active recording" begin
        # Create arrays and dispatch without flushing between iterations
        for iter in 1:20
            a = Mantle.LavaArray(rand(Float32, 200))
            b = Mantle.LavaArray(zeros(Float32, 200))
            fill_kernel!(backend)(b, 1f0; ndrange=200)
            # Don't flush — batches accumulate
        end
        Mantle.vk_flush!(Mantle.vk_context())
        GC.gc(true)
        drainfrees!()
        @test true
    end

    @testset "interleaved alloc/free/dispatch" begin
        # Mirrors the reference test pattern: create figure data, render, discard, repeat
        for iter in 1:30
            # "Scatter plot" data
            positions = Mantle.LavaArray(rand(Float32, 50))
            colors = Mantle.LavaArray(rand(Float32, 50))
            sizes = Mantle.LavaArray(rand(Float32, 50))

            # "Line plot" data
            line_pos = Mantle.LavaArray(rand(Float32, 100))
            line_colors = Mantle.LavaArray(rand(Float32, 100))

            # Dispatch
            fill_kernel!(backend)(positions, Float32(iter); ndrange=50)
            fill_kernel!(backend)(line_pos, Float32(iter); ndrange=100)

            # Flush every 5 iterations (like the reference test GC between test files)
            if iter % 5 == 0
                Mantle.vk_flush!(Mantle.vk_context())
                drainfrees!()
                GC.gc(true)
            end
        end
        Mantle.vk_flush!(Mantle.vk_context())
        drainfrees!()
        GC.gc(true)
        @test true
    end

    @testset "graphics pipeline alloc pattern" begin
        # Simulate the geometry shader prep: create temp arrays, dispatch, discard
        for iter in 1:20
            # Mimic prep_sprite_gfx allocations
            n = 50
            positions = Mantle.LavaArray(rand(Float32, n * 3))
            quad_offsets = Mantle.LavaArray(rand(Float32, n * 2))
            quad_scales = Mantle.LavaArray(rand(Float32, n * 2))
            rotations = Mantle.LavaArray(rand(Float32, n * 4))
            colors = Mantle.LavaArray(rand(Float32, n * 4))
            uv_rects = Mantle.LavaArray(rand(Float32, n * 4))
            shapes = Mantle.LavaArray(rand(UInt8, n))
            stroke_colors = Mantle.LavaArray(zeros(Float32, n * 4))
            glow_colors = Mantle.LavaArray(zeros(Float32, n * 4))

            # Dispatch a compute kernel on some of them
            fill_kernel!(backend)(positions, 1f0; ndrange=n*3)
            fill_kernel!(backend)(colors, 0.5f0; ndrange=n*4)

            Mantle.vk_flush!(Mantle.vk_context())

            # Explicit cleanup (what the reference tests should do)
            for arr in [positions, quad_offsets, quad_scales, rotations, colors,
                        uv_rects, stroke_colors, glow_colors]
                Mantle.unsafe_free!(arr)
            end
            # shapes is UInt8, unsafe_free! it too
            Mantle.unsafe_free!(shapes)

            drainfrees!()
        end
        @test true
    end

    @testset "deferred free count stays bounded" begin
        initial_deferred = pendingfrees(bq)

        for iter in 1:100
            a = Mantle.LavaArray(rand(Float32, 50))
            fill_kernel!(backend)(a, 1f0; ndrange=50)
            # Don't explicitly free — let GC handle it
        end

        Mantle.vk_flush!(Mantle.vk_context())
        GC.gc(true)
        drainfrees!()

        final_deferred = pendingfrees(bq)
        @test final_deferred <= initial_deferred + 10  # Should be mostly cleaned up
    end

    @testset "proactive deferred free flush in vk_alloc" begin
        # Deferred frees should be flushed automatically by vk_alloc when safe
        # (not recording, no in-flight batches)
        for iter in 1:30
            a = Mantle.LavaArray(rand(Float32, 100))
            fill_kernel!(backend)(a, Float32(iter); ndrange=100)
            Mantle.vk_flush!(Mantle.vk_context())
            # Let a go out of scope without explicit free — GC finalizer should handle it
        end
        GC.gc(true)
        # Next allocation should proactively flush any accumulated deferred frees
        b = Mantle.LavaArray(rand(Float32, 10))
        @test pendingfrees(bq) == 0
        Mantle.unsafe_free!(b)
    end

    @testset "no crash after many flush cycles" begin
        for cycle in 1:50
            arrays = [Mantle.LavaArray(rand(Float32, 100)) for _ in 1:5]
            for a in arrays
                fill_kernel!(backend)(a, Float32(cycle); ndrange=100)
            end
            Mantle.vk_flush!(Mantle.vk_context())
        end
        GC.gc(true)
        drainfrees!()
        @test true
    end
end
