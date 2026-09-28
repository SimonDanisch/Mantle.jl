# `basealignment` of the Vulkan backend's own resources: a `LavaArray` carries a
# view's offset inside it, so only the array can say how aligned its base is.
# The core half, and why DNNKernels asks, is `test/test_basealignment.jl`.

using Test
import Mantle as M

@testset "base alignment of a device array" begin
    t = M.TransientBuffer{Float16,1}((64,), typemax(Int), 0, nothing, 0)
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
