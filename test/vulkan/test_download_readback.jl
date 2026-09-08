# A download comes back through host-cached memory, at cached speed.
#
# `copy_buffer!(:download, …)` stages the device bytes into a region the host
# then memcpys out of. For a while that region came from the `Unified` arena —
# BAR memory, device-local and host-visible, which the host WRITES well and
# READS at write-combined speed: 29 MB/s measured on an RTX 4000 Ada and a RADV
# RX 7900 XTX alike, 330 ms for one 1368x1026 RGBA{Float32} frame, a third of
# a 32-sample render in the RayDemo harness. The staging is a `Readback` region
# now, which the Vulkan backend backs with HOST_CACHED memory where the device
# has it.
#
# Pinned two ways: the pool holds a `Readback` block after a download, and the
# download itself runs at a rate no write-combined read can reach. The bound is
# loose on purpose — 16 MB in under 160 ms is 100 MB/s, three times the
# write-combined figure and well under what a cached memcpy does — so it holds
# on a slow PCIe link and on lavapipe, and fails only for the mistake it pins.
using Test, Mantle, Lava
import KernelAbstractions as KA

@testset "a download stages through Readback, at cached speed" begin
    dev = Mantle.Device(Mantle.VulkanAPI())
    backend = Mantle.defaultbackend()
    n = 4 * 1024 * 1024                      # 16 MB of Float32
    host = rand(Float32, n)
    a = KA.allocate(backend, Float32, n)
    copyto!(a, host)
    Mantle.waitidle(backend)
    # Warm: the first download grows the pool.
    @test Array(a) == host
    @test !isempty(Mantle.blocksof(Mantle.pool(dev), Mantle.Readback()))
    t = minimum((@elapsed Array(a)) for _ in 1:3)
    @test t < 0.16
    @test Array(a) == host
end
