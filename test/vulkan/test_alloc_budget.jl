# An allocation past the driver's memory budget is out of memory (docs/api.md),
# also on a driver that would accept it.
#
# RADV accepts it: it moves buffers out of VRAM into system memory (GTT), and
# once that passes about half the RAM, amdgpu's `ttm_global_swapout` (kernel
# 7.2.6) corrupts a list under a spinlock and the whole machine locks up. That
# happened twice on 2026-10-06, a 7900 XTX filled by one Julia process
# rendering a 100k-instance meshscatter. `overbudget` is the check that turns
# such a request into an ordinary out-of-memory error.
#
# Nothing here fills the device: the rule is tested on a made-up snapshot, and
# the device paths are asked for more than the budget in ONE request, which the
# check refuses before the driver is asked.

using Test, Mantle

@testset "memory budget" begin
    @testset "the rule: usage plus the request against the budget" begin
        heap = (heap = 0, device_local = true, size = 24 << 30, budget = 20 << 30, usage = 19 << 30)
        @test !Mantle.overbudget(heap, 1 << 30)            # exactly at the budget
        @test Mantle.overbudget(heap, (1 << 30) + 1)
        @test !Mantle.overbudget(heap, 0)
    end

    ctx = Mantle.vk_context()
    if !ctx.memory_budget_available
        @info "VK_EXT_memory_budget is not available; only the driver's own refusal applies"
        @test_skip ctx.memory_budget_available
    else
        heaps = Mantle.probe_device_memory_budget(ctx)
        # More than any heap has left, so whichever heap a memory type maps to,
        # the request is over its budget.
        over = maximum(h -> h.budget - h.usage, heaps) + (256 << 20)

        @testset "images and graph arenas (device_memory) throw" begin
            bits = typemax(UInt32)                         # any memory type
            @test_throws Mantle.LavaError Mantle.device_memory(ctx, over, bits)
            @test_throws r"budget" Mantle.device_memory(ctx, over, bits)
        end

        @testset "buffers (try_vk_alloc) report a failure, not a buffer" begin
            res = Mantle.try_vk_alloc(ctx.default_bq, over)
            @test res isa Mantle.AllocFailure
            # `:Buffer` where the driver refuses the buffer's size before memory is
            # chosen (RADV's maxBufferSize is about 4 GB); `:budget` where it
            # would have created it.
            @test res.op in (:budget, :Buffer)
        end

        @testset "the device works afterwards" begin
            a = Mantle.LavaArray(Float32[1, 2, 3, 4])
            b = a .+ 10.0f0
            Mantle.flush!(ctx.default_bq)
            @test Array(b) == Float32[11, 12, 13, 14]
        end
    end
end
