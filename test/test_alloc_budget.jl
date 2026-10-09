# An allocation past the device's memory budget is out of memory (docs/api.md),
# also on a driver that would accept it.
#
# RADV accepts it: it moves buffers out of VRAM into system memory (GTT), and
# once that passes about half the RAM, amdgpu's `ttm_global_swapout` (kernel
# 7.2.6) corrupts a list under a spinlock and the whole machine locks up. That
# happened twice on 2026-10-06, a 7900 XTX filled by one Julia process
# rendering a 100k-instance meshscatter. The Vulkan backend's `overbudget` is the
# check that turns such a request into an ordinary out-of-memory error. Metal
# would page instead of refusing, which is the same contract broken more quietly.
#
# Nothing here fills the device: each path is asked for more than the WHOLE
# budget in ONE request, which is refused before the driver is asked. The budget
# is `Mantle.capacity(dev)`: `VK_EXT_memory_budget` summed over the device-local
# heaps, Metal's `recommendedMaxWorkingSetSize`.
#
# Not "more than the budget has left": usage includes the pool's own empty blocks,
# which the pool hands back before it retries — measured on LapWin (Radeon 8060S,
# 2026-10-07): 0.38 GB in use, a request of budget - 0.38 + 0.25 GB refused, the
# empty blocks returned, the retry granted. The test then held 30 GB and every
# later file in the suite ran out of memory.
#
# The rule itself (`overbudget(heap, bytes)`: usage plus the request against the
# budget, exactly at the budget allowed) was also tested here on a made-up heap
# snapshot. It is the Vulkan backend's helper; what it decides is asserted below.

using Test, Mantle, KernelAbstractions
include(joinpath(@__DIR__, "testbackend.jl"))
const KA = KernelAbstractions

@testset "memory budget" begin
    dev = Mantle.Device(TESTBACKEND)
    over = Mantle.capacity(dev) + (256 << 20)

    # The displayed message, matched case-insensitively: an out-of-memory error
    # that does not say so sends whoever reads it to the wrong place.
    @testset "a Buffer past the budget is refused as out of memory" begin
        @test_throws r"memory"i Mantle.Buffer(dev, UInt8, over)
    end

    @testset "a backend array past the budget is refused as out of memory" begin
        @test_throws r"memory"i KA.allocate(TESTBACKEND, UInt8, over)
    end

    @testset "the device works afterwards" begin
        a = Mantle.devicearray(TESTBACKEND, Float32[1, 2, 3, 4])
        b = a .+ 10.0f0
        KA.synchronize(TESTBACKEND)
        @test Array(b) == Float32[11, 12, 13, 14]
    end

    # A request granted above (a failure already reported) must not hold the
    # device's memory for every file after this one.
    Mantle.makeroom!(Mantle.pool(dev), dev)
end
