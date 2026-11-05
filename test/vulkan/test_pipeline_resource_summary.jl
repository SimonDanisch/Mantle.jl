using Test, Mantle

@testset "pipeline resource statistics preserve their units" begin
    stat(name, value) = (; name, value)
    radv = [stat("SGPRs", 108), stat("VGPRs", 240), stat("Spilled SGPRs", 51),
        stat("Spilled VGPRs", 0), stat("Scratch size", 55296), stat("Pre-Sched VGPRs", 227)]
    for stats in (radv, reverse(radv))
        @test Mantle.pipeline_resource_summary(stats) == (; registers=240, scratch_bytes=55296)
    end
    @test Mantle.pipeline_resource_summary([stat("Spilled Registers", 7)]) ==
        (; registers=nothing, scratch_bytes=nothing)
    @test Mantle.pipeline_resource_summary([stat("Register Count", 168), stat("Local Memory Size", 16)]) ==
        (; registers=168, scratch_bytes=nothing)
    @test Mantle.pipeline_resource_summary([stat("Spilled VGPRs", 0), stat("Scratch size", 0)]) ==
        (; registers=nothing, scratch_bytes=0)
    @test Mantle.pipeline_resource_summary(NamedTuple[]) == (; registers=nothing, scratch_bytes=nothing)
end
