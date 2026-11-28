"""
A request the device cannot satisfy is refused by name, and the device survives it.

This exercises oversized-request rejection and recovery. It does not establish
that the GPU is out of memory: 400 GiB can also exceed its maximum buffer size.
The pool must reject that before calling the driver with an invalid size.

Two claims, on both ways a caller allocates: a `Mantle.Buffer` (the device's
pool, every backend) and a backend array (`KernelAbstractions.allocate`, which is
the pool on Vulkan and Metal.jl's allocator on Metal).

**A refusal names the memory request.** Core's single-allocation bound rejects
an impossible size without GC or a retry that could not help. Genuine allocation
exhaustion, reclaim and bounded retry are exercised separately by `BudgetDev` in
`test_pool.jl`. A Vulkan error other than native OOM must propagate unchanged.

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

@testset "an oversized allocation is refused, and the device survives" begin
    dev = Mantle.Device(TESTBACKEND)
    huge = 400 * 1024^3

    @testset "a Buffer beyond the allocation limit is refused" begin
        # Core refuses an oversized pooled request before native allocation.
        # Backend-owned arrays may report their native memory refusal instead.
        @test_throws r"memory"i Mantle.Buffer(dev, UInt8, huge)
    end

    @testset "a backend array beyond the allocation limit is refused" begin
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
