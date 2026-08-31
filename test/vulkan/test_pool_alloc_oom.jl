"""
A request the device cannot satisfy is refused by name, and the device survives it.

This is the one path in `pool_alloc` that no other test reaches, because reaching
it means being out of VRAM. It is also the path most likely to be wrong without
anyone noticing: it only runs under memory pressure, and if it throws the wrong
thing it does so at the moment a workload is already in trouble.

Three separate claims, and the middle one is the reason the file exists.

**The OOM guard names the right exception.** `Mantle.Pool`'s growth calls
`rawalloc`, which reaches `VK.DeviceMemory(device, bytes, type)` — a constructor
that THROWS rather than returning a `ResultTypes.Result`, unlike most of
Vulkan.jl's API, which is why `device_memory` has no `unwrap` around it.
`acquire_or_reclaim!` catches `VK.VulkanError` by name and re-raises anything
else, so if either the type or the `code` field were wrong the escalation would
never run and a lost device would be misreported as an OOM. Verified against a
real refusal rather than a constructed exception.

**A refusal is a `LavaError` that says what happened**, after one escalation:
collect, drain the deferred frees, quiesce the queue, hand back empty blocks, try
again. A raw `VulkanError` reaching the caller here would mean the escalation was
skipped.

**The device still works afterwards.** A failed allocation must leave nothing
half-carved: no block with a region nobody holds, no lock still taken. The
allocation after the refusal is the assertion, and it is the one that would catch
a `carve!` that had already committed when growth threw.

`400 GiB` because it has to exceed the largest heap on any machine this runs on
while staying inside `Int`. It is refused before any GPU work is queued, so this
costs about 0.2 s.
"""

using Test, Mantle, KernelAbstractions
const KA = KernelAbstractions

@testset "an impossible allocation is refused, and the device survives" begin
    ctx = MVE.vk_context()
    bq  = ctx.default_bq
    dev = MVE.lavadevice(ctx)
    be  = MVE.LavaBackend()
    huge = 400 * 1024^3

    @testset "the driver refuses with the exception the guard names" begin
        err = try
            Mantle.rawalloc(dev, Mantle.Buffers(), huge, UInt32(0))
            nothing
        catch e
            e
        end
        @test err isa MVE.VK.VulkanError
        # `.code`, and one of the two the guard accepts. A different code here
        # means `acquire_or_reclaim!` rethrows instead of escalating.
        @test err.code == MVE.VK.ERROR_OUT_OF_DEVICE_MEMORY ||
              err.code == MVE.VK.ERROR_OUT_OF_HOST_MEMORY
    end

    @testset "pool_alloc turns it into a LavaError that says why" begin
        err = try
            MVE.pool_alloc(bq, huge)
            nothing
        catch e
            e
        end
        @test err isa MVE.LavaError
        msg = sprint(showerror, err)
        @test occursin("out of memory", msg)
        # The footprint is in the message: without it "out of memory" does not
        # distinguish a leak from a request that was always too large.
        @test occursin("reserved by this pool", msg)
    end

    @testset "and the allocator is intact" begin
        # The refusal must not have left a partially carved block behind. An
        # allocation that works, and reads back what was written to it, is what
        # says so — a `carve!` that committed before growth threw would hand out
        # a region inside a block that does not exist.
        a = KA.allocate(be, Float32, 4096)
        fill!(a, 7.0f0)
        KA.synchronize(be)
        @test all(==(7.0f0), Array(a))

        blocks = MVE.poolblocks(ctx)
        @test !isempty(blocks)
        # Every block still partitions into what is live and what is free.
        for b in blocks
            pieces = vcat([(lo, hi) for (lo, hi) in b.free.bylower],
                          [(lo, hi) for (lo, hi) in b.live])
            sort!(pieces)
            at = 0
            for (lo, hi) in pieces
                @test lo == at
                at = hi
            end
            @test at == b.bytes
        end
    end
end
