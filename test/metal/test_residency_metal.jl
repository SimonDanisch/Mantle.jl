# What ends a membership of the device's residency set, on Metal.
#
# Two owners end memberships: the pool, which forgets a block when it frees it, and the
# use record of a Metal.jl array a recording reaches, whose collection queues a removal
# for after the next drain. The pool frees first, the address goes back to Metal, and a
# new buffer there joins the set; the queued removal must not end that one. It did: a
# removal named only the address, and in RayDemo's benchmark the scene rendered after
# two others lost its mesh under hardware traversal.

using Test, Mantle, Metal
const MB = Mantle.MetalBindings

@testset "Metal: a queued removal ends only the membership it was queued for" begin
    dev = Metal.device()
    r = MB.Residency(dev)
    buf = Metal.MTL.MTLBuffer(dev, 256; storage = Metal.SharedStorage)
    owner = Ref(0)
    MB.resident!(r, buf, owner)
    first = r.members[UInt(pointer(buf))]
    # The other owner ends it now, and the buffer joins again.
    MB.forget!(r, buf)
    @test !haskey(r.members, UInt(pointer(buf)))
    MB.resident!(r, (buf,))
    @test r.members[UInt(pointer(buf))] != first
    # The first owner's removal arrives late.
    MB.forget_after_drain!(r, buf, first)
    MB.forget_drained!(r)
    @test haskey(r.members, UInt(pointer(buf)))
    # A removal of the current membership ends it.
    MB.forget_after_drain!(r, buf, r.members[UInt(pointer(buf))])
    MB.forget_drained!(r)
    @test !haskey(r.members, UInt(pointer(buf)))
    Metal.free(buf)
end

@testset "Metal: a membership ends after the drain that follows its owner's collection" begin
    dev = Metal.device()
    r = MB.Residency(dev)
    buf = Metal.MTL.MTLBuffer(dev, 256; storage = Metal.SharedStorage)
    key = UInt(pointer(buf))
    owner, other = Ref(0), Ref(0)
    MB.resident!(r, buf, owner)
    gen = r.members[key]
    # Known already: the membership and its removal stay the first owner's.
    MB.resident!(r, buf, other)
    @test r.members[key] == gen
    finalize(other)
    MB.forget_drained!(r)
    @test haskey(r.members, key)
    # Collected, the owner queues the removal; the drain applies it.
    finalize(owner)
    @test haskey(r.members, key)
    MB.forget_drained!(r)
    @test !haskey(r.members, key)
    Metal.free(buf)
end
