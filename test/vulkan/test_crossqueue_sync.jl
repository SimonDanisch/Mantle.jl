"""
A buffer written on one `BatchQueue` and then used on another makes `sync_access!`
push a timeline wait onto `batch.wait_semaphores`. That vector is sync2-typed
(`Vulkan.PipelineStageFlag2`), but `Vulkan.PIPELINE_STAGE_2_ALL_COMMANDS_BIT` is
— despite the `_2_` in its name — a *sync1* `PipelineStageFlag`, so the push threw
a `MethodError` on every cross-queue wait there has ever been.

It did not present as a MethodError. `submit!` calls `sync_access!` *after*
`vkEndCommandBuffer`, so the throw left the batch still `bq.active_batch` with
`recording == true` and an ENDED command buffer; the next `ensure_active_batch!`
handed it straight back and the caller recorded into it. That is undefined
behaviour, and the NVIDIA driver takes it as a SIGSEGV — which is how it was
actually found: a segfault inside `vkCmdPipelineBarrier` while SAM 2's weights
uploaded in the video editor, three frames removed from the real bug, and only
once `bq.auto_submit_threshold` (64) put a submit in the middle of a recording.

So this checks both halves: that the wait can be pushed at all, and that a
throwing `sync_access!` can never again leave a live batch pointing at an ended
command buffer.
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
Mantle.sync_access!(::Mantle.CommandBatch, ::ThrowsOnSync) = error("sync_access! test failure")

@testset "cross-queue sync" begin
    @testset "the stage flag is sync2-typed" begin
        # The bug in one line: right value, wrong wrapper type.
        # `Mantle.VK`. This file said `import Lava: Vulkan`, which resolved
        # while Lava depended on Vulkan.jl and afterwards bound the name to
        # nothing — Julia reports it as "defined but not assigned a value", so
        # the two assertions below errored rather than failing.
        @test Mantle.STAGE2_ALL_COMMANDS isa Mantle.VK.PipelineStageFlag2
        @test UInt64(Mantle.STAGE2_ALL_COMMANDS.val) ==
              UInt64(Mantle.VK.PIPELINE_STAGE_2_ALL_COMMANDS_BIT.val)
    end

    @testset "a buffer crossing queues submits" begin
        ctx = Mantle.vk_context()
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

    @testset "a throwing submit! leaves no batch on an ended command buffer" begin
        bq = Mantle.vk_context().default_bq
        b = LavaBackend()
        a = KA.allocate(b, Float32, 64)
        xqfill!(b, 64)(a, 1.0f0; ndrange = 64)

        batch = Mantle.ensure_active_batch!(bq)
        push!(batch.pinned, ThrowsOnSync())
        @test_throws Exception Mantle.submit!(bq)
        @test bq.active_batch === nothing      # dropped, not left recording
        @test !batch.recording

        # The queue must still work afterwards.
        xqfill!(b, 64)(a, 4.0f0; ndrange = 64)
        KA.synchronize(b)
        @test all(Array(a) .== 4.0f0)
    end
end
