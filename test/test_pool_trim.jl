# Empty pool blocks must go back to the device when a caller asks for it, without
# waiting for an out-of-memory error to force it.
#
# A pool only ever grows on its own: blocks handed back only on an
# allocation-failure retry ratchet up to the high-water mark of the largest
# workload and stay there. Across a multi-scene run that is gigabytes of memory
# the rest of the machine still needs, and on Vulkan it showed up as a driver
# timeout partway through a 5-scene sweep.
#
# The explicit verb is `Mantle.makeroom!(pool, dev)`: collect, wait for what the
# device has been given, release what it has finished with, and hand every block
# nothing uses back (`trim!`). The Vulkan backend's automatic trigger on top of it
# (`maybe_trim_pool!`'s absolute-capacity threshold, its rate limit, and the cost
# budget of a soft-cap collection) is that backend's allocation policy and is
# tested beside it, in `vulkan/test_pool_trim_policy.jl`.
#
# `Mantle.Buffer`s, because they are the pool's on every backend; a backend array
# is the pool's on Vulkan and Metal.jl's own on Metal.

using Test, Mantle, KernelAbstractions
include(joinpath(@__DIR__, "testbackend.jl"))
const KA = KernelAbstractions

const MiB = 1024 * 1024

@kernel function grind!(a)
    i = @index(Global)
    x = a[i]
    for _ in 1:2000
        x = x * 1.0000001f0 + 1f-7
    end
    a[i] = x
end

@testset "freed buffers' blocks go back on a trim" begin
    dev = Mantle.Device(TESTBACKEND)
    sp = Mantle.pool(dev)
    Mantle.makeroom!(sp, dev)               # from a known floor
    # Against this file's own baseline, not zero: the pool is the device's, and
    # whatever another file still holds is in it. What this checks is that the
    # capacity THIS testset created comes back.
    base = Mantle.reserved(sp)

    let bufs = Mantle.Buffer[]
        while Mantle.reserved(sp) - base < 320MiB
            push!(bufs, Mantle.Buffer(dev, Float32, 4_000_000))   # 16 MB each
        end
        foreach(b -> fill!(b, 1.0f0), bufs)
        KA.synchronize(TESTBACKEND)
        foreach(Mantle.free!, bufs)
    end

    grown = Mantle.reserved(sp)
    @test grown - base >= 256MiB

    Mantle.makeroom!(sp, dev)
    trimmed = Mantle.reserved(sp)
    @test trimmed < grown                     # capacity actually came back
    @test trimmed - base < (grown - base) ÷ 2 # and it was THIS testset's

    # And the allocator still works afterwards — blocks were returned, not corrupted.
    b = Mantle.Buffer(dev, Float32, 1024)
    fill!(b, 2.0f0)
    KA.synchronize(TESTBACKEND)
    @test all(Array(b) .== 2.0f0)
    Mantle.free!(b)
end

# The case the testset above cannot reach, and the one that mattered.
#
# It synchronizes before freeing, so every region is past the device already and
# the trim finds empty blocks. Free them while the work that names them is still
# running and nothing is released at all: each region is retired, and the block
# keeps it as live until the device is past the work. A trim that looks at the
# blocks BEFORE it has waited — the Vulkan `trim_gpu_pool!` gated on
# `any(b -> isempty(b.live), blocks)`, a precondition it establishes itself —
# finds nothing to do and keeps everything. A graph evaluator is nothing but
# this shape, dispatches recorded and not waited for until the output is read:
# TRELLIS.2's 30-block torso left 190 blocks and 12 410 MiB resident with 0
# blocks empty, and 12 750 MiB of it was reclaimable. Nothing had leaked; the
# trim was refusing to look.
#
# 60 unsynchronised dispatches is the smallest thing that reproduces it, and the
# assertion is on the state *before* the trim as well as the bytes after, so this
# fails loudly if a future change makes the workload stop reproducing rather than
# passing on a technicality.
@testset "a trim waits for work in flight before it decides there is nothing to do" begin
    dev = Mantle.Device(TESTBACKEND)
    sp = Mantle.pool(dev)
    Mantle.makeroom!(sp, dev)               # from a known floor
    # …which is a floor and not zero in the full suite; see the first testset.
    base = Mantle.reserved(sp)

    let bufs = Mantle.Buffer[]
        for _ in 1:60
            b = Mantle.Buffer(dev, Float32, 4_000_000)          # 16 MB
            fill!(b, 1.0f0)
            grind!(TESTBACKEND, 256)(Mantle.storage(b); ndrange = length(b))   # NOT synchronised
            push!(bufs, b)
        end
        foreach(Mantle.free!, bufs)
        empty!(bufs)
    end
    GC.gc(true)

    grown = Mantle.reserved(sp)
    @test grown - base > 256MiB
    # The state an early gate mishandles: every block these buffers sat in still
    # counts as live even though every reference to its contents is gone.
    @test !any(b -> isempty(b.live), Mantle.blocksof(sp, Mantle.Persistent()))

    Mantle.makeroom!(sp, dev)
    # Half of what THIS testset added, for the reason the first one gives.
    @test Mantle.reserved(sp) - base < (grown - base) ÷ 2

    # And the allocator still works — blocks were returned, not corrupted.
    b = Mantle.Buffer(dev, Float32, 1024)
    fill!(b, 2.0f0)
    KA.synchronize(TESTBACKEND)
    @test all(Array(b) .== 2.0f0)
    Mantle.free!(b)
end
