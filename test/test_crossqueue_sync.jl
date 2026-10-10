"""
A buffer written on one channel and then used on another has to wait for the
first.

A channel is an independent stream of submissions on one device:
`allocate_batch_queue!(dev)` hands one out, and `backend(channel)` is the
KernelAbstractions backend that launches on it. Two channels have no order between
them, so the order a buffer needs is derived: on Vulkan core stamps every
resource a submission names with its (channel, token) and `crosswaits!` turns a
use on another channel into a timeline-semaphore wait; on Metal, Metal.jl's
ownership check makes a launch that names a buffer another queue still uses
wait for that queue.

The Vulkan original found this through a wrong-typed stage flag: the cross-queue
wait pushed a sync1 `PipelineStageFlag` where a sync2 one belonged, after
`vkEndCommandBuffer`, so the throw left a closed command buffer half inside a
submission — a segfault in `vkCmdPipelineBarrier` while SAM 2's weights
uploaded, three frames from the cause. The flag's type was asserted there; what
a user sees of it is a cross-channel launch that throws or races, which is what
is asserted here, with NO wait between the two channels' launches: the first
kernel is slow on purpose, so a second channel that did not wait would overtake
it and its result would be overwritten.

"A build that throws leaves nothing in limbo" was the Vulkan file's other half;
it is a property of core's `oneshot!` on a `SubmitChannel` and is
`test_submit_channel.jl`'s.
"""

using Test, Mantle, KernelAbstractions
include(joinpath(@__DIR__, "testbackend.jl"))
const KA = KernelAbstractions

# Slow on purpose: `spin` dependent iterations before the store. The comparison
# is never true for these inputs, and the compiler cannot know that, so the loop
# stays.
@kernel function xq_slowfill!(a, v, spin)
    i = @index(Global)
    acc = 0f0
    for k in Int32(1):spin
        acc = muladd(acc, 0.999f0, sin(Float32(k)))
    end
    @inbounds a[i] = acc == -12345f0 ? 0f0 : v
end

@kernel function xq_add!(a, v)
    i = @index(Global)
    @inbounds a[i] += v
end

@testset "cross-channel ordering" begin
    dev = Mantle.Device(TESTBACKEND)
    if !Mantle.supports_batch_queue(dev)
        # Asked anyway, the answer is a refusal rather than a stub.
        @test_throws ArgumentError Mantle.allocate_batch_queue!(dev)
    else
        ch = Mantle.allocate_batch_queue!(dev)
        @test Mantle.todevice(ch) === dev
        b1 = Mantle.backend(dev)
        b2 = Mantle.backend(ch)
        @test Mantle.Device(b2) === dev
        n = 4096
        spin = Int32(20_000)
        a = KA.allocate(b1, Float32, n)

        @testset "a buffer one channel writes, the other reads" begin
            xq_slowfill!(b1, 64)(a, 1f0, spin; ndrange = n)
            xq_add!(b2, 64)(a, 1f0; ndrange = n)
            KA.synchronize(b2)
            @test all(==(2f0), Array(a))
        end

        @testset "and back the other way" begin
            xq_slowfill!(b2, 64)(a, 5f0, spin; ndrange = n)
            xq_add!(b1, 64)(a, 1f0; ndrange = n)
            KA.synchronize(b1)
            @test all(==(6f0), Array(a))
        end

        @testset "a second channel is a second stream of submissions" begin
            # Its work is the device's: it counts among the device's submissions.
            s0 = Mantle.submissions(dev)
            xq_add!(b2, 64)(a, 1f0; ndrange = n)
            KA.synchronize(b2)
            @test Mantle.submissions(dev) > s0
            @test all(==(7f0), Array(a))
        end

        @test Mantle.release_batch_queue!(ch) === nothing
        # The device's own channel still works after a channel is given back.
        xq_add!(b1, 64)(a, 1f0; ndrange = n)
        KA.synchronize(b1)
        @test all(==(8f0), Array(a))
    end
end
