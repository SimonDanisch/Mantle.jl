using Test, Mantle
import KernelInterface as KI
include(joinpath(@__DIR__, "testbackend.jl"))

const M = Mantle

function ro_add_one!(dst, src)
    i = KI.get_global_id().x
    i <= length(dst) && (@inbounds dst[i] = src[i] + 1f0)
    return
end

@testset "read-only arguments: results on every launch path" begin
    dev = M.Device(TESTBACKEND)
    a = M.devicearray(dev, zeros(Float32, 16))
    b = M.devicearray(dev, Float32.(1:16))

    KI.@launch dev ndrange=16 ro_add_one!(a, b)
    M.waitidle(dev)
    @test Array(a) == Float32.(2:17)
    KI.@launch dev ndrange=16 ro_add_one!(a, a)
    M.waitidle(dev)
    @test Array(a) == Float32.(3:18)

    # Broadcasting is the eager KA path, and the common way into it.
    c = M.devicearray(dev, zeros(Float32, 16))
    c .= b .* 2f0
    @test Array(c) == Float32.(1:16) .* 2
    c .= c .+ b
    @test Array(c) == Float32.(1:16) .* 3

    g = M.Graph(dev)
    out = M.devicearray(dev, zeros(Float32, 16))
    M.dispatch!(g, ro_add_one!, (out, b), 16)
    M.runonce!(g)
    M.waitidle(dev)
    @test Array(out) == Float32.(2:17)

    alias_graph = M.Graph(dev)
    M.dispatch!(alias_graph, ro_add_one!, (a, view(a, 1:16)), 16)
    M.runonce!(alias_graph)
    M.waitidle(dev)
    @test Array(a) == Float32.(4:19)

    recorded_graph = M.Graph(dev)
    M.dispatch!(recorded_graph, ro_add_one!, (a, a), 16)
    plan = M.record!(M.Plan(recorded_graph))
    try
        M.run!(plan)
        M.waitfor!(plan)
        @test Array(a) == Float32.(5:20)
        M.run!(plan)
        M.waitfor!(plan)
        @test Array(a) == Float32.(6:21)
    finally
        M.free!(plan)
    end
end
