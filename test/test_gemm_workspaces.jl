using Test, Mantle, KernelAbstractions
include(joinpath(@__DIR__,"testbackend.jl"))
include(joinpath(@__DIR__,"split_workspace_helpers.jl"))

function workspace_replay_bytes(plan)
    for _ in 1:20
        Mantle.run!(plan); Mantle.waitfor!(plan)
    end
    return minimum(1:10) do _
        bytes = @allocated Mantle.run!(plan)
        Mantle.waitfor!(plan)
        bytes
    end
end

@testset "independent split programs own independent workspace" begin
    dev = Mantle.Device(TESTBACKEND)
    with_split_programs(TESTBACKEND) do state
        @test !Base.mightalias(state.workspaces...)
        Mantle.runlaunches!(TESTBACKEND,state.launches)
        KernelAbstractions.synchronize(TESTBACKEND)
        for (out,expected) in zip(state.outputs,state.expected)
            @test Array(out)==expected
        end
        graph = Mantle.Graph(dev)
        Mantle.dispatchlaunches!(graph,state.launches;name="interleaved split GEMV")
        plan = Mantle.Plan(graph)
        try
            Mantle.record!(plan)
            for _ in 1:3
                foreach(out->fill!(out,0f0),state.outputs)
                KernelAbstractions.synchronize(TESTBACKEND)
                GC.gc(true)
                Mantle.run!(plan); Mantle.waitfor!(plan)
                for (out,expected) in zip(state.outputs,state.expected)
                    @test Array(out)==expected
                end
            end
            if Mantle.recorded(plan)
                @test workspace_replay_bytes(plan) == 0
            end
        finally
            Mantle.free!(plan)
        end
    end
end
