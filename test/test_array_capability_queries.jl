# The capability queries a portable array algorithm sizes its kernels with
# answer for the device the array lives on.
#
# `gemv.jl` and `fft.jl` left the Vulkan backend once their capability queries
# stopped going through a `VkContext`:
#
#     workgroup_limit(vk_context(A))   ->   workgrouplimit(A)
#     max_shared_memory(vk_context(A)) ->   sharedbudget(A)
#
# array to backend to `KernelInterface.caps`, instead of array to context to
# caps. `test_array_algorithm_portability.jl` is the source-level ratchet on
# that move; this is the device half. It compared the portable answers with the
# Vulkan context's, which is a middle step another backend does not have. What
# every backend has is its own `caps` and its own `KI.max_work_group_size`, and
# the array's queries have to agree with those. If they ever disagree, a kernel
# sized through `caps` is being launched against a different device's limits —
# which is silent, and wrong.

using Test, Mantle, KernelAbstractions
import KernelInterface as KI
include(joinpath(@__DIR__, "testbackend.jl"))
const KA = KernelAbstractions

@testset "the array's capability queries answer for its device" begin
    a = KA.allocate(TESTBACKEND, Float32, 16)
    c = Mantle.caps(TESTBACKEND)
    @test Mantle.workgrouplimit(a) == KI.max_work_group_size(TESTBACKEND)
    @test Mantle.workgrouplimit(a) == c.workgrouplimit
    @test Mantle.sharedbudget(a) == c.sharedbudget
    @test Mantle.coopmatgemm(a) == c.coopmat
    # …and they are real numbers, so "they agree" is not two zeros agreeing.
    @test Mantle.workgrouplimit(a) > 0
    @test Mantle.sharedbudget(a) > 0
end
