# A device array wrapped in a view, a reshape, a permute, a transpose or an
# adjoint is still on that device, and broadcasting through the wrapper works.
#
# The Vulkan backend resolves "which device" by walking a wrapper to its parent
# (`vk_context`), and that list of methods has to stay in step with the wrappers
# its broadcast machinery admits (`AnyLavaArray`). It did not: the Union listed
# six wrappers and the methods covered three. The consequence was not subtle:
# broadcasting into a transposed device array threw
#
#     MethodError: no method matching vk_context(::Adjoint{Int16, LavaArray{Int16, 1}})
#
# GPUArrays' own broadcasting suite caught it — 12 errors in "Adjoint and
# Transpose" — because broadcasting over a transposed array is an ordinary thing
# to do, not a corner.
#
# What a user sees is asked of every wrapper here: its backend is the device's,
# and a broadcast over it computes the right numbers.

using Test, Mantle, LinearAlgebra, KernelAbstractions
include(joinpath(@__DIR__, "testbackend.jl"))
const KA = KernelAbstractions

@testset "device arrays through wrappers" begin
    dev = Mantle.Device(TESTBACKEND)
    h = collect(Int16, 1:8)
    a = Mantle.devicearray(TESTBACKEND, h)

    wrap = Dict(
        "SubArray"          => x -> view(x, 1:4),
        "ReshapedArray"     => x -> reshape(x, 2, 4),
        "PermutedDimsArray" => x -> PermutedDimsArray(reshape(x, 2, 4), (2, 1)),
        "Transpose"         => x -> transpose(x),
        "Adjoint"           => x -> adjoint(x),
    )
    for (name, f) in wrap
        @testset "$name" begin
            w = f(a)
            @test KA.get_backend(w) == TESTBACKEND
            @test Mantle.Device(KA.get_backend(w)) === dev
            # A broadcast over the wrapper, and one INTO it.
            @test Array(w .+ Int16(1)) == f(h) .+ Int16(1)
            b = Mantle.devicearray(TESTBACKEND, copy(h))
            wb = f(b)
            wb .= wb .* Int16(2)
            KA.synchronize(TESTBACKEND)
            want = copy(h)
            wh = f(want)
            wh .= wh .* Int16(2)
            @test Array(b) == want
        end
    end

    # The end-to-end shape that failed: broadcasting over a transposed array.
    b = adjoint(a) .+ Int16(1)
    @test Array(b)[1:4] == Int16[2, 3, 4, 5]
end
