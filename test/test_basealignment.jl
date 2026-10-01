# `basealignment`: how far a kernel may trust an operand's base address.
#
# Asked by DNNKernels' `flashrows_plan`, which stages K and V in 16-byte chunks.
# Its first guess was the element offset `stridedroot` reports, and a contiguous
# view of a device array defeats that: `view(a, 5:n)` is another device array,
# with its 8-byte offset inside it, so the reported offset is 0. The answer has
# to come from the operand.
#
# The backend's own device array is `vulkan/test_basealignment_lava.jl`: the
# shared test layer names no backend-only binding.

using Test
import Mantle as M

@testset "base alignment" begin
    t = M.TransientBuffer{Float16,1}((64,), typemax(Int), 0, nothing, 0)
    # A transient's offset is decided at placement; the floor is what every
    # backend's placement rule promises.
    @test M.basealignment(t) == M.TRANSIENT_ALIGN_FLOOR
    # Nothing is known about an arbitrary array, so nothing is claimed.
    @test M.basealignment(zeros(Float16, 8)) == 1
    # A view is as aligned as its parent and its offset both allow.
    @test M.basealignment(M.viewof(t, 32; offset = 4)) == 8
    @test M.basealignment(M.viewof(t, 32; offset = 8)) == 16
    @test M.basealignment(M.viewof(t, 32)) == M.TRANSIENT_ALIGN_FLOOR
end
