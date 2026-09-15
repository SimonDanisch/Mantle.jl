# A recorded plan on Metal survives its buffers MOVING.
#
# This did not work at all until 2026-09-14, and it failed in the worst way
# available: `resize!` past capacity gives a `Buffer` fresh storage and RETIRES the
# old region rather than freeing it, so a recording left pointing at the old
# address kept reading well-formed, stale bytes. No fault, no warning, a number
# that is merely wrong.
#
# The cause was an abstraction that let a backend not answer. Core announced the
# move through `resource_moved!`, whose default was `nothing`; Vulkan implemented
# it, Metal never did, and nothing anywhere said so. Both halves are gone now: the
# address is `deviceaddress`, which has one right answer, and WHERE each packed
# pointer landed is found by core from the argument's type (`notepacked!`), so a
# backend is not asked and cannot forget.
#
# Pinned from the outside, on values: the same plan object, run again after the
# move, has to see the new bytes.

using Test
using Mantle
using Metal
using KernelAbstractions: @kernel, @index, @Const
const M = Mantle
isdefined(@__MODULE__, :DEV_SEAM) || (DEV_SEAM = M.Device(Mantle.MetalAPI()))

@kernel function seam_move_add!(dst, @Const(src))
    i = @index(Global)
    @inbounds dst[i] += src[i]
end

@testset "a recorded Metal plan follows its buffers when they move" begin
    n = 256
    dev = DEV_SEAM
    a = M.Buffer(dev, zeros(Float32, n))
    b = M.Buffer(dev, fill(1.0f0, n))
    g = M.Graph(dev)
    M.dispatch!(g, seam_move_add!, (a,
                                        b), n; name = "add")
    pl = M.record!(M.Plan(g))

    # The table is core's and is built from the packed arguments' types, so it is
    # not empty on a plan holding two device arrays. Before this existed it was,
    # and every assertion below passed for the wrong reason.
    @test !isempty(pl.patchtab)

    M.run!(pl); M.waitfor!(pl)
    @test Array(M.storage(a))[1] == 1.0f0

    # Past capacity: new storage, and the old region retired rather than freed —
    # which is exactly why an unpatched recording reads 1.0f0 here instead of
    # faulting.
    before = b.store.region
    resize!(b, 4n)
    @test b.store.region !== before
    M.update!(b, fill(2.0f0, 4n))
    @test !isempty(pl.pending_patches)      # queued by the move, not yet written

    M.run!(pl); M.waitfor!(pl)
    @test Array(M.storage(a))[1] == 3.0f0   # 2.0f0 would be the retired storage
    @test isempty(pl.pending_patches)       # and the run consumed them

    # Still patched on the run after, with nothing left pending: a patch is eight
    # bytes in the argument memory, not a state the next run has to redo.
    M.run!(pl); M.waitfor!(pl)
    @test Array(M.storage(a))[1] == 5.0f0
end
