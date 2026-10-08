"""
Two `Buffer`s the pool places in one block are two arrays, not one.

`dataids` of a `LavaArray` names the block, which every `Buffer` placed in it
shares, so alias detection keyed on it alone took disjoint buffers for the same
memory: AcceleratedKernels' `accumulate!` refused a scan from one `Buffer` into
another, and `Base.unalias` copied ahead of broadcasts between them. Views of one
array still alias where their bytes meet, and only there.
"""

using Test, Mantle, Lava

@testset "pooled buffers alias only where their bytes meet" begin
    dev = Mantle.Device(Mantle.VulkanAPI())
    a, b = Mantle.Buffer(dev, Int32, 1000), Mantle.Buffer(dev, Int32, 1000)
    sa, sb = Mantle.storage(a), Mantle.storage(b)
    # The premise: one block, so `dataids` cannot tell them apart.
    @test Base.dataids(sa) == Base.dataids(sb)
    @test !Base.mightalias(sa, sb)
    @test Base.mightalias(sa, sa)

    # Views of one array: overlapping ranges alias, neighbouring ones do not.
    @test Base.mightalias(view(sa, 1:10), view(sa, 10:20))
    @test !Base.mightalias(view(sa, 1:10), view(sa, 11:20))
    # A reinterpret covers the same bytes under another element type.
    @test Base.mightalias(reinterpret(Int64, view(sa, 1:4)), view(sa, 4:4))
    @test !Base.mightalias(reinterpret(Int64, view(sa, 1:4)), view(sa, 5:5))

    # What it was for: a scan from one pooled buffer into another.
    fill!(sa, Int32(1))
    accumulate!(+, sb, sa)
    @test Array(sb) == Int32.(1:1000)
    foreach(Mantle.free!, (a, b))
end
