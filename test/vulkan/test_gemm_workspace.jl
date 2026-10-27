using Test, Mantle, KernelAbstractions
include(joinpath(@__DIR__,"..","split_workspace_helpers.jl"))

function check_workspace_retirement(ctx)
    dev = ctx
    backend = dev
    @testset "unreferenced split workspace returns its pool regions" begin
        regions = with_split_programs(backend) do state
            regions = map(a->a.buf[].region::Mantle.Region,state.workspaces)
            Mantle.runlaunches!(backend,state.launches)
            KernelAbstractions.synchronize(backend)
            return regions
        end
        for _ in 1:3
            GC.gc(true); Mantle.drain!(dev.default_bq)
        end
        for region in regions
            @test !haskey(region.block.live,Mantle.offset(region))
        end
        @test isempty(Mantle.validationmessages!(dev))
    end
end
