"""
A buffer written on one `VulkanBatchQueue` and then used on another makes `sync_access!`
push a timeline wait onto the submission's `wait_semaphores`. That vector is
sync2-typed (`Vulkan.PipelineStageFlag2`), but `Vulkan.PIPELINE_STAGE_2_ALL_COMMANDS_BIT`
is — despite the `_2_` in its name — a *sync1* `PipelineStageFlag`, so the push
threw a `MethodError` on every cross-queue wait there has ever been.

It did not present as a MethodError. `submit!` calls `sync_access!` *after*
`vkEndCommandBuffer`, so the throw used to leave a closed command buffer stuck
half inside the submission that failed. Now a throwing `sync_access!` drops the
whole submission: every one-shot it carried goes back to `bq.free_oneshots`
(pins emptied) rather than being left in some ended, half-submitted state, and
the error propagates to the caller. This is how the underlying bug actually
surfaced: a segfault inside `vkCmdPipelineBarrier` while SAM 2's weights
uploaded in the video editor, three frames removed from the real bug, and only
once enough dispatches had queued up to put a submit in the middle of what the
caller thought was still being recorded.

So this checks both halves: that the wait can be pushed at all, and that a
throwing `sync_access!` can never again leave a closed command buffer stuck in
limbo — the submission it belonged to is dropped cleanly and the queue keeps
working.
"""

using Test, Lava, KernelAbstractions
const KA = KernelAbstractions

@kernel function xqfill!(a, v)
    i = @index(Global)
    @inbounds a[i] = v
end

# Pinned object whose access semantics throw, to drive `submit!` into its
# post-`vkEndCommandBuffer` failure path on demand.
struct ThrowsOnSync end
MVE.sync_access!(::MVE.Submission, ::ThrowsOnSync) = error("sync_access! test failure")

@testset "cross-queue sync" begin
    @testset "the stage flag is sync2-typed" begin
        # The bug in one line: right value, wrong wrapper type.
        # `MVE.VK`. This file said `import Lava: Vulkan`, which resolved
        # while Lava depended on Vulkan.jl and afterwards bound the name to
        # nothing — Julia reports it as "defined but not assigned a value", so
        # the two assertions below errored rather than failing.
        @test MVE.STAGE2_ALL_COMMANDS isa MVE.VK.PipelineStageFlag2
        @test UInt64(MVE.STAGE2_ALL_COMMANDS.val) ==
              UInt64(MVE.VK.PIPELINE_STAGE_2_ALL_COMMANDS_BIT.val)
    end

    @testset "a buffer crossing queues submits" begin
        ctx = MVE.vk_context()
        b1 = LavaBackend()                             # ctx.default_bq
        bq2 = Mantle.allocate_batch_queue!(ctx)
        b2 = LavaBackend(bq2)

        n = 256
        a = KA.allocate(b1, Float32, n)
        xqfill!(b1, 64)(a, 1.0f0; ndrange = n)
        KA.synchronize(b1)
        @test all(Array(a) .== 1.0f0)

        # Same buffer, other queue: `sync_access!` must push a wait on bq1's
        # timeline. This is the call that used to throw inside `submit!`.
        xqfill!(b2, 64)(a, 2.0f0; ndrange = n)
        KA.synchronize(b2)
        @test all(Array(a) .== 2.0f0)

        # And back again, so the wait is exercised in the other direction.
        xqfill!(b1, 64)(a, 3.0f0; ndrange = n)
        KA.synchronize(b1)
        @test all(Array(a) .== 3.0f0)
    end

    @testset "a throwing submit! drops the submission and the queue still works" begin
        bq = MVE.vk_context().default_bq
        b = LavaBackend()
        a = KA.allocate(b, Float32, 64)
        xqfill!(b, 64)(a, 1.0f0; ndrange = 64)

        o = MVE.oneshot(bq) do e
            push!(e.owner.pinned, ThrowsOnSync())
        end
        @test_throws Exception Mantle.submit!(bq, o)
        @test !o.open
        @test any(x -> x === o, bq.free_oneshots)   # dropped back to the pool
        @test isempty(o.pinned)                     # ...with its pins emptied

        # The queue must still work afterwards.
        xqfill!(b, 64)(a, 4.0f0; ndrange = 64)
        KA.synchronize(b)
        @test all(Array(a) .== 4.0f0)
    end
end
