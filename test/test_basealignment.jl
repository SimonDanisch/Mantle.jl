# `basealignment`: how far a kernel may trust an operand's base address.
#
# Asked by DNNKernels' `flashrows_plan`, which stages K and V in 16-byte chunks.
# Its first guess was the element offset `stridedroot` reports, and a contiguous
# view of a device array defeats that: `view(a, 5:n)` is another device array,
# with its 8-byte offset inside it, so the reported offset is 0. The answer has
# to come from the operand.

using Test
import Mantle as M

@testset "base alignment" begin
    t = M.TransientBuffer{Float16,1}((64,), typemax(Int), 0, nothing, 0)
    # A transient's offset is decided at placement; the floor is what every
    # backend's placement rule promises.
    @test M.basealignment(t) == M.TRANSIENT_ALIGN_FLOOR
    @test M.alignment(M.Device(M.HostAPI()), t) >= M.TRANSIENT_ALIGN_FLOOR
    # Nothing is known about an arbitrary array, so nothing is claimed.
    @test M.basealignment(zeros(Float16, 8)) == 1
    # A view is as aligned as its parent and its offset both allow.
    @test M.basealignment(M.viewof(t, 32; offset = 4)) == 8
    @test M.basealignment(M.viewof(t, 32; offset = 8)) == 16
    @test M.basealignment(M.viewof(t, 32)) == M.TRANSIENT_ALIGN_FLOOR

    if _VULKAN_OK
        dev = M.Device(M.LavaBackend())
        @test M.alignment(dev, t) >= M.TRANSIENT_ALIGN_FLOOR
        a = M.LavaArray(zeros(Float16, 64))
        @test M.basealignment(a) >= 16
        # The offset lives inside the view: 4 halves in is 8 bytes.
        @test M.basealignment(view(a, 5:64)) == 8
        @test M.basealignment(view(a, 9:64)) >= 16
        b = M.Buffer(dev, Float16, (64,))
        @test M.basealignment(b) >= 16
        @test M.basealignment(M.viewof(b, 32; offset = 2)) == 4
    end
end
