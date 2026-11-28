"""
A Metal 4 queue's wait does not hold the thread, and handing it to a worker allocates
nothing.

The Metal 4 queue (`MetalDevice(dev, MTL4Queue(dev))`, opt-in) waits on a shared event
of its own rather than through Metal.jl, so `test_hostwait.jl` and
`test_run_allocates_nothing.jl`, which run on the default device, never reach it. It
blocked in `waitUntilSignaledValue` on the calling thread: a second task on that thread
counted 0 ticks through a 0.2 s wait on an M5. Then its wait held the event and the
feedback, so it was boxed, 32 bytes, every time a worker took it.
"""

using Test, Mantle, Metal, Profile
const M = Mantle
include(joinpath(@__DIR__, "..", "spinwork.jl"))
include(joinpath(@__DIR__, "..", "runallocations.jl"))

# What 200 waits allocate, per wait. `run!` stays outside the profile: what it allocates
# is `test_run_allocates_nothing.jl`'s question.
function allocations_per_wait(pl)
    for _ in 1:100
        M.run!(pl); M.waitfor!(pl)
    end
    Profile.Allocs.clear()
    for _ in 1:200
        M.run!(pl)
        Profile.Allocs.@profile sample_rate = 1 M.waitfor!(pl)
    end
    return sum(a -> a.size, filter(byrun, Profile.Allocs.fetch().allocs); init = 0) / 200
end

@testset "a Metal 4 queue's wait" begin
    dev = M.MetalDevice(Metal.device(), M.MTL4Queue(Metal.device()))
    @testset "lets the thread's other tasks run" begin
        pl = spinplan(dev, spinsteps(dev, 0.2))
        M.run!(pl); M.waitfor!(pl)
        stop = Ref(false)
        ticks, task = ticker(stop)
        try
            M.run!(pl)
            k, s = progress_during(() -> M.waitfor!(pl), ticks)
            # As in `test_hostwait.jl`: at least one tick per 10 ms of the wait.
            @test s > 0.05
            @test k >= s / 0.01
        finally
            stop[] = true
            wait(task)
            M.free!(pl)
        end
    end
    @testset "allocates nothing when it reaches a worker" begin
        # Outlasts the poll, which an M5 gives up after about 2.6 ms.
        pl = spinplan(dev, spinsteps(dev, 0.01))
        @test allocations_per_wait(pl) == 0
        M.free!(pl)
    end
end
