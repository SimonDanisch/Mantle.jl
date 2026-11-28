# Test rapid GPU allocation/deallocation cycles that mirror the reference test pattern:
# - Many device arrays created per "test" (positions, colors, quad_offsets, etc.)
# - Each "test" creates and discards arrays in rapid succession
# - GC pressure from old arrays while new dispatches are in flight
# - Simulates the pattern that crashes RayMakie reference tests on APU

#
# **This file called `Mantle.flush_deferred_frees!()` and read a global
# `Mantle.DEFERRED_FREES` until 2026-08-23.** Neither has existed since the
# deferred-free list became per-VulkanBatchQueue, so every testset below threw
# `UndefVarError` on its first line — and `runtests.jl` did not include the file,
# so nothing said so. It is included now. A test nothing runs is not a test.
#
# It then counted the Vulkan queue's pending destroys. What that count stood for
# is memory coming back, and that is what is asserted now, on any backend: the
# bytes the device's pool has handed out return to where they were. A backend
# whose arrays the pool does not hold (Metal.jl's own) leaves the pool untouched,
# and there the dispatch results and the absence of a crash are the assertion.
#
# This file also replaces `mwe_alloc_dispatch_free_loop.jl`, a script that asked
# whether a tight per-iteration allocate, dispatch, drop and collect loop loses
# the device without hardware ray tracing. The first testset is that loop, with
# every iteration's result checked.

using Test, Mantle, KernelAbstractions
include(joinpath(@__DIR__, "testbackend.jl"))
const KA = KernelAbstractions

@kernel function fill_kernel!(dst, val)
    i = @index(Global, Linear)
    @inbounds dst[i] = val
end

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

# Each loop is a function of its own, so that once it returns nothing it
# allocated is still reachable from a stack slot of the caller: the assertions
# after it are about memory coming back, and a binding the compiler kept alive
# would read as a leak.

"""Fifty "test iterations" of ten arrays each: dispatch on every one, check the
last, and let them all go out of scope."""
@noinline function smallarrays(backend)
    for iter in 1:50
        arrays = [Mantle.devicearray(TESTBACKEND, rand(Float32, 100)) for _ in 1:10]
        for a in arrays
            fill_kernel!(backend)(a, Float32(iter); ndrange=100)
        end
        KA.synchronize(TESTBACKEND)
        # Verify last array
        @test all(x -> x == Float32(iter), Array(arrays[end]))
        # Let arrays go out of scope — GC should handle cleanup
    end
end

"""Arrays created and dropped while dispatches are in flight, never synchronized
in between."""
@noinline function inflightarrays(backend)
    for iter in 1:20
        a = Mantle.devicearray(TESTBACKEND, rand(Float32, 200))
        b = Mantle.devicearray(TESTBACKEND, zeros(Float32, 200))
        fill_kernel!(backend)(b, 1f0; ndrange=200)
        # Don't synchronize — submissions accumulate
    end
    KA.synchronize(TESTBACKEND)
end

"""The reference test pattern: create figure data, render, discard, repeat."""
@noinline function interleaved(backend, drainfrees!)
    for iter in 1:30
        # "Scatter plot" data
        positions = Mantle.devicearray(TESTBACKEND, rand(Float32, 50))
        colors = Mantle.devicearray(TESTBACKEND, rand(Float32, 50))
        sizes = Mantle.devicearray(TESTBACKEND, rand(Float32, 50))

        # "Line plot" data
        line_pos = Mantle.devicearray(TESTBACKEND, rand(Float32, 100))
        line_colors = Mantle.devicearray(TESTBACKEND, rand(Float32, 100))

        # Dispatch
        fill_kernel!(backend)(positions, Float32(iter); ndrange=50)
        fill_kernel!(backend)(line_pos, Float32(iter); ndrange=100)

        # Synchronize every 5 iterations (like the reference test GC between test files)
        if iter % 5 == 0
            KA.synchronize(TESTBACKEND)
            drainfrees!()
        end
    end
    KA.synchronize(TESTBACKEND)
end

"""The geometry shader prep: temp arrays, a dispatch, and an explicit free of
every one."""
@noinline function spriteprep(backend, drainfrees!)
    for iter in 1:20
        # Mimic prep_sprite_gfx allocations
        n = 50
        positions = Mantle.devicearray(TESTBACKEND, rand(Float32, n * 3))
        quad_offsets = Mantle.devicearray(TESTBACKEND, rand(Float32, n * 2))
        quad_scales = Mantle.devicearray(TESTBACKEND, rand(Float32, n * 2))
        rotations = Mantle.devicearray(TESTBACKEND, rand(Float32, n * 4))
        colors = Mantle.devicearray(TESTBACKEND, rand(Float32, n * 4))
        uv_rects = Mantle.devicearray(TESTBACKEND, rand(Float32, n * 4))
        shapes = Mantle.devicearray(TESTBACKEND, rand(UInt8, n))
        stroke_colors = Mantle.devicearray(TESTBACKEND, zeros(Float32, n * 4))
        glow_colors = Mantle.devicearray(TESTBACKEND, zeros(Float32, n * 4))

        # Dispatch a compute kernel on some of them
        fill_kernel!(backend)(positions, 1f0; ndrange=n*3)
        fill_kernel!(backend)(colors, 0.5f0; ndrange=n*4)

        KA.synchronize(TESTBACKEND)

        # Explicit cleanup (what the reference tests should do)
        for arr in [positions, quad_offsets, quad_scales, rotations, colors,
                    uv_rects, stroke_colors, glow_colors]
            Mantle.unsafe_free!(arr)
        end
        # shapes is UInt8, unsafe_free! it too
        Mantle.unsafe_free!(shapes)

        drainfrees!()
    end
end

"""A hundred arrays, each dispatched on and left to the collector."""
@noinline function collectedarrays(backend)
    for iter in 1:100
        a = Mantle.devicearray(TESTBACKEND, rand(Float32, 50))
        fill_kernel!(backend)(a, 1f0; ndrange=50)
        # Don't explicitly free — let GC handle it
    end
    KA.synchronize(TESTBACKEND)
end

"""Thirty arrays, each dispatched on, synchronized and left to the collector."""
@noinline function syncedarrays(backend)
    for iter in 1:30
        a = Mantle.devicearray(TESTBACKEND, rand(Float32, 100))
        fill_kernel!(backend)(a, Float32(iter); ndrange=100)
        KA.synchronize(TESTBACKEND)
        # Let a go out of scope without explicit free — GC finalizer should handle it
    end
end

"""Fifty synchronize cycles of five arrays, checking the last of each."""
@noinline function manycycles(backend)
    for cycle in 1:50
        arrays = [Mantle.devicearray(TESTBACKEND, rand(Float32, 100)) for _ in 1:5]
        for a in arrays
            fill_kernel!(backend)(a, Float32(cycle); ndrange=100)
        end
        KA.synchronize(TESTBACKEND)
        @test all(==(Float32(cycle)), Array(arrays[end]))
    end
end

@testset "Rapid allocation/free cycles" begin
    backend = TESTBACKEND
    dev = Mantle.Device(TESTBACKEND)
    scratch = KA.allocate(TESTBACKEND, Float32, 1024)
    drainfrees!() = settle!(dev, scratch)

    @testset "many small arrays created and discarded" begin
        drainfrees!()
        before = livebytes(dev)
        smallarrays(backend)
        drainfrees!()
        @test livebytes(dev) <= before    # and every one of them came back
    end

    @testset "allocation while dispatches are in flight" begin
        drainfrees!()
        before = livebytes(dev)
        inflightarrays(backend)
        drainfrees!()
        @test livebytes(dev) <= before
    end

    @testset "interleaved alloc/free/dispatch" begin
        drainfrees!()
        before = livebytes(dev)
        interleaved(backend, drainfrees!)
        drainfrees!()
        @test livebytes(dev) <= before
    end

    @testset "graphics pipeline alloc pattern" begin
        drainfrees!()
        before = livebytes(dev)
        spriteprep(backend, drainfrees!)
        @test livebytes(dev) <= before
    end

    @testset "deferred frees stay bounded" begin
        drainfrees!()
        initial = livebytes(dev)
        collectedarrays(backend)
        drainfrees!()
        # Mostly cleaned up: at most ten of the hundred arrays still held.
        @test livebytes(dev) <= initial + 10 * 50 * sizeof(Float32)
    end

    @testset "the next allocation hands back what the collector dropped" begin
        drainfrees!()
        before = livebytes(dev)
        # Dropped arrays should be released by the next allocation, unasked, once
        # the device is past them: an allocator that grows while dead memory waits
        # for an explicit flush is the RayMakie-on-APU failure this file mirrors.
        syncedarrays(backend)
        GC.gc(true)
        # No upload, so the new array is the only thing the allocation adds.
        b = KA.allocate(TESTBACKEND, Float32, 1024)
        @test livebytes(dev) <= before + sizeof(b)
        Mantle.unsafe_free!(b)
    end

    @testset "no crash after many synchronize cycles" begin
        drainfrees!()
        before = livebytes(dev)
        manycycles(backend)
        drainfrees!()
        @test livebytes(dev) <= before
    end
end
