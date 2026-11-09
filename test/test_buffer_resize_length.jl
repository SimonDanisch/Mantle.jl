using Test, Mantle
include(joinpath(@__DIR__, "testbackend.jl"))

function buffer_resize_copy_kernel!(out, input)
    i = Mantle.KI.get_global_id().x
    i <= 2 || return nothing
    @inbounds out[i] = input[i]
    return nothing
end

@testset "resize! sets the length and keeps the prefix" begin
    dev = Mantle.Device(TESTBACKEND)
    b = Mantle.Buffer(dev, Int32[11, 22, 33])
    try
        resize!(b, 7)
        @test length(b) == 7
        @test size(b) == (7,)
        @test Array(Mantle.storage(b))[1:3] == Int32[11, 22, 33]
        resize!(b, 2)
        @test length(b) == 2
        @test size(b) == (2,)
        resize!(b, 6)                       # within capacity: the length grows too
        @test length(b) == 6
        @test size(b) == (6,)
        @test Mantle.capacity(b) >= 7
        @test Array(Mantle.storage(b))[1:2] == Int32[11, 22]
        resize!(b, 0)
        @test isempty(b)
        @test_throws ArgumentError resize!(b, -1)
    finally
        Mantle.free!(b)
    end
end

@testset "a shrink trims queued stores, not the caller's vector" begin
    dev = Mantle.Device(TESTBACKEND)
    b = Mantle.Buffer(dev, Int32, 5)
    out = Mantle.Buffer(dev, Int32, 2)
    queued = Int32[17, 18, 19]
    b[1:3] = queued
    b[4:5] = Int32[99, 100]
    resize!(b, 2)
    @test queued == Int32[17, 18, 19]
    graph = Mantle.Graph(dev)
    Mantle.dispatch!(graph, buffer_resize_copy_kernel!, (out, b), 2)
    plan = Mantle.record!(Mantle.Plan(graph))
    passes = length(graph.passes)
    try
        Mantle.run!(plan); Mantle.waitfor!(plan)
        @test length(b) == 2
        @test Array(b) == Int32[17, 18]
        @test Array(out) == Int32[17, 18]
        # Grow past the capacity, which moves the store, then shrink back: the
        # recorded plan follows the move without being recorded again.
        resize!(b, 12)
        @test length(b) == 12 && Mantle.capacity(b) >= 12
        @test Array(b)[1:2] == Int32[17, 18]
        resize!(b, 2)
        b[:] = Int32[31, 32]
        Mantle.run!(plan); Mantle.waitfor!(plan)
        @test Array(out) == Int32[31, 32]
        @test length(graph.passes) == passes
    finally
        Mantle.free!(plan); Mantle.free!(out); Mantle.free!(b)
    end
end
