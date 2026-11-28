"""
A host wait does not hold the thread.

A task waiting for the device yields, so the other tasks on its thread keep running: a
window's input beside its render loop, an editor's handlers beside an analysis. Vulkan
blocked in `vkWaitSemaphores` instead, and an input task beside RayMakie's render loop
ran once a frame there (MakieOrg/Makie.jl#5856). Pinned as progress: a second task counts
while the first waits on GPU work long enough to tell the two apart. The window's waits
are in `test_window_portable.jl`.
"""

using Test, Mantle
include(joinpath(@__DIR__, "testbackend.jl"))
const M = Mantle
include(joinpath(@__DIR__, "spinwork.jl"))

@testset "a host wait lets the thread's other tasks run" begin
    dev = M.Device(TESTBACKEND)
    # About 0.2 s: long enough that a held thread shows, short of any watchdog.
    pl = spinplan(dev, spinsteps(dev, 0.2))
    M.run!(pl); M.waitfor!(pl)

    stop = Ref(false)
    ticks, task = ticker(stop)
    try
        @testset "waitfor!" begin
            M.run!(pl)
            k, s = progress_during(() -> M.waitfor!(pl), ticks)
            # A wait long enough to tell, and at least one tick per 10 ms of it;
            # the ticker manages about five, a held thread none.
            @test s > 0.05
            @test k >= s / 0.01
        end
        @testset "flush!" begin
            M.run!(pl)
            k, s = progress_during(() -> M.flush!(dev), ticks)
            @test s > 0.05
            @test k >= s / 0.01
        end
    finally
        stop[] = true
        wait(task)
        M.free!(pl)
    end
end
