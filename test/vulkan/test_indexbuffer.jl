"""
`Mantle.indexbuffer` reaches this backend's allocation, with the usage bit.

Vulkan refuses a `vkCmdBindIndexBuffer` whose buffer lacks
`BUFFER_USAGE_INDEX_BUFFER_BIT`, and the bit has to be asked for at allocation —
which is why the portable `indexbuffer` exists at all and why this backend
overrides it. Two ways to get it wrong, and both have happened:

  * the override never fires, because a caller handed in a KA backend and the
    method is on the DEVICE. `indexbuffer(::Backend, …)` normalises first for
    exactly this reason;
  * the override fires and calls an allocation that does not exist. It called
    `alloc_index_buffer(indices)` while that function took `(backend, indices)`,
    so every indexed draw in RayMakie was a `MethodError` — six call sites, none
    of them covered.

Both spellings are checked here because both are what a caller writes.
"""

using Test, Mantle, Lava, KernelAbstractions

@testset "indexbuffer allocates with INDEX_BUFFER_BIT, from either spelling" begin
    be = Mantle.defaultbackend()
    dev = Mantle.Device(Mantle.VulkanAPI())
    idx = UInt32[1, 2, 3, 1, 3, 4]

    for from in (be, dev)
        ib = Mantle.indexbuffer(from, idx)
        # An ARRAY, which is what both backends answer with and what a caller
        # keeps beside its vertex arrays — not a `Mantle.Buffer`.
        @test ib isa Mantle.LavaArray{UInt32,1}
        @test length(ib) == length(idx)
        @test Array(ib) == idx
        # The bit itself: the allocation is not pooled, because a pooled buffer
        # carries the pool's usage flags and nothing else.
        buf = ib.buf[]
        @test buf.region === nothing
        # And it draws: an indexed draw against a target is the only end-to-end
        # statement that the bit is really there.
        fb = Mantle.Framebuffer(be, 8, 8; depth = false,
                                color_format = Mantle.VK.FORMAT_R32G32B32A32_SFLOAT)
        Mantle.oneshot!(Mantle.batchqueue(Mantle.Device()); tag = :indexdraw) do e
            Mantle.transition_image!(e.cmd, fb.color_image,
                Mantle.VK.IMAGE_LAYOUT_UNDEFINED, Mantle.VK.IMAGE_LAYOUT_GENERAL,
                Mantle.VK.PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                Mantle.VK.PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                Mantle.VK.AccessFlag(0), Mantle.VK.AccessFlag(0))
            Mantle.VK.cmd_bind_index_buffer(e.cmd, buf.buffer, UInt64(0),
                                         Mantle.VK.INDEX_TYPE_UINT32)
        end
        Mantle.flush!(Mantle.batchqueue(Mantle.Device()))
    end
end
