"""
A buffer that outlives a device reset must not call into the device it outlived.

    d = KA.allocate(LavaBackend(), Float32, 1000); fill!(d, 1f0)
    Mantle.vk_reset_device!(); d = nothing; GC.gc()      # <- SIGSEGV

Ten lines, and it took down the whole suite. `vk_reset_device!` drops
`VK_CONTEXT_REF`, which makes the old context garbage — and its buffers garbage
in the **same collection**, where Julia does not order finalizers. Run the
context's first and `Vulkan.Device`'s own finalizer destroys the device; the
buffer's `vk_free!` then calls `query_timeline` on it and the driver takes the
fault inside `vkGetSemaphoreCounterValue`.

`vk_free!` does gate on `device_lost`, and the reset's comment cited that gate as
the reason this was safe. The gate was simply never true here: a reset *caused by*
`ERROR_DEVICE_LOST` finds the flag already set, and a voluntary
`vk_reset_device!()` — which is what the suite does — left it false. So the fix is
to set it, and what this test pins is that a retired context stays retired.

It reproduces at `046b1ed`, before any of the per-device work, and the AMD
laptop hit the same thing from three different call sites and filed it as "a
floating GC race". It floats because the crash lands wherever the next GC does.

**This test crashes the process when it regresses**, rather than failing. That is
the nature of the bug and there is no way to observe a finalizer-thread segfault
from inside Julia — the same reason `twodevice_probe.jl` was held out of the
suite while it was broken.
"""

using Test, Lava, KernelAbstractions
const KA = KernelAbstractions

@testset "a buffer may outlive a device reset" begin
    ctx0 = Mantle.vk_context()
    id0 = ctx0.id

    b = LavaBackend()
    d = KA.allocate(b, Float32, 1000)
    fill!(d, 1.0f0)
    KA.synchronize(b)
    @test Array(d) == fill(1.0f0, 1000)

    Mantle.vk_reset_device!()

    # The reset installs a NEW context, and retires the old one. Both halves
    # matter: a fresh id is what every per-device cache keys on, and the flag is
    # what every finalizer gates on.
    ctx1 = Mantle.vk_context()
    @test ctx1.id != id0
    @test Mantle.device_lost(ctx0)
    @test !Mantle.device_lost(ctx1)

    # The finalizer for a pre-reset buffer, run on purpose. Twice, because the
    # first collection may only queue it.
    d = nothing
    GC.gc()
    GC.gc()
    @test true      # reaching here at all is the assertion

    # And the new device still works, which rules out "retired everything".
    b2 = LavaBackend()
    e = KA.allocate(b2, Float32, 64)
    fill!(e, 3.0f0)
    KA.synchronize(b2)
    @test Array(e) == fill(3.0f0, 64)
end
