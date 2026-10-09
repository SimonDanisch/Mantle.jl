"""
A request the device cannot satisfy is refused by name, and the device survives it.

This is the one allocation path no other test reaches, because reaching it means
being out of device memory. It is also the path most likely to be wrong without
anyone noticing: it only runs under memory pressure, and if it throws the wrong
thing it does so at the moment a workload is already in trouble.

Two claims, on both ways a caller allocates: a `Mantle.Buffer` (the device's
pool, every backend) and a backend array (`KernelAbstractions.allocate`, which is
the pool on Vulkan and Metal.jl's allocator on Metal).

**A refusal says it is out of memory.** On Vulkan the pool's growth reaches
`VK.DeviceMemory(device, bytes, type)`, a constructor that THROWS rather than
returning a `ResultTypes.Result`, and `acquire_or_reclaim!` catches
`VK.VulkanError` by name, escalates once (collect, drain the deferred frees,
quiesce the queue, hand back empty blocks, try again) and then raises a
`LavaError` that names the pool's footprint. With either the type or the `code`
wrong the escalation never runs and a lost device is misreported as an OOM, or an
OOM as something else. What a caller can rely on, whatever the backend, is the
message: an error that does not say "memory" sends whoever reads it to the wrong
place.

**The device still works afterwards.** A failed allocation must leave nothing
half-carved: no block with a region nobody holds, no lock still taken. The
allocation after the refusal is the assertion, and it is the one that would catch
a `carve!` that had already committed when growth threw — together with every
block of the pool still partitioning into what is live and what is free.

`400 GiB` because it has to exceed the largest heap on any machine this runs on
while staying inside `Int`. It is refused before any GPU work is queued, so this
costs about 0.2 s.
"""

using Test, Mantle, KernelAbstractions
include(joinpath(@__DIR__, "testbackend.jl"))
const KA = KernelAbstractions

"""
Does `blk` partition into its live regions and its free spans: disjoint, in
range, and covering every byte? An overlap is corruption and a gap a leak.
"""
function partitions(blk)
    pieces = sort!(vcat([(lo, hi) for (lo, hi) in blk.free.bylower],
                        [(lo, hi) for (lo, hi) in blk.live]))
    at = 0
    for (lo, hi) in pieces
        lo == at || return false
        at = hi
    end
    return at == blk.bytes
end

@testset "an impossible allocation is refused, and the device survives" begin
    dev = Mantle.Device(TESTBACKEND)
    huge = 400 * 1024^3

    @testset "a Buffer from the pool is refused as out of memory" begin
        # The displayed message, matched case-insensitively: Vulkan's driver code
        # is `ERROR_OUT_OF_DEVICE_MEMORY`, Metal's refusal an `OutOfMemoryError`.
        @test_throws r"memory"i Mantle.Buffer(dev, UInt8, huge)
    end

    @testset "a backend array is refused as out of memory" begin
        @test_throws r"memory"i KA.allocate(TESTBACKEND, UInt8, huge)
    end

    @testset "and the allocator is intact" begin
        # The refusal must not have left a partially carved block behind. An
        # allocation that works, and reads back what was written to it, is what
        # says so — a `carve!` that committed before growth threw would hand out
        # a region inside a block that does not exist.
        a = KA.allocate(TESTBACKEND, Float32, 4096)
        fill!(a, 7.0f0)
        KA.synchronize(TESTBACKEND)
        @test all(==(7.0f0), Array(a))

        b = Mantle.Buffer(dev, fill(3.0f0, 4096))
        @test all(==(3.0f0), Array(b))

        blocks = collect(Iterators.flatten(values(Mantle.pool(dev).blocks)))
        @test !isempty(blocks)
        # Every block still partitions into what is live and what is free.
        @test all(partitions, blocks)
        Mantle.free!(b)
    end
end
