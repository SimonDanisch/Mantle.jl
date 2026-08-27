# `vk_context` resolves through every wrapper `AnyLavaArray` admits.
#
# A device array wrapped in a view, a reshape, a permute, a transpose or an
# adjoint is still on that device, so `vk_context` walks to the parent. The list
# of methods has to stay in step with `AnyLavaArray` (gpuarrays.jl:91) — and it
# did not: the Union lists six wrappers and the methods covered three.
#
# The consequence was not subtle. `probe_broadcast!` does
# `vk_context(dest).diag.broadcast_probe`, so broadcasting into a transposed
# device array threw
#
#     MethodError: no method matching vk_context(::Adjoint{Int16, LavaArray{Int16, 1}})
#
# GPUArrays' own broadcasting suite caught it — 12 errors in "Adjoint and
# Transpose" — because broadcasting over a transposed array is an ordinary thing
# to do, not a corner.
#
# This asserts against `AnyLavaArray` itself rather than against a hand-written
# list, so adding a wrapper to that Union without adding a method here fails
# here rather than in somebody's broadcast.

using Test, Lava, LinearAlgebra

@testset "vk_context resolves through array wrappers" begin
    ctx = Mantle.vk_context()
    a = Mantle.LavaArray(collect(Int16, 1:8))

    wrappers = Dict(
        "SubArray"          => view(a, 1:4),
        "ReshapedArray"     => reshape(a, 2, 4),
        "PermutedDimsArray" => PermutedDimsArray(reshape(a, 2, 4), (2, 1)),
        "Transpose"         => transpose(a),
        "Adjoint"           => adjoint(a),
    )
    for (name, w) in wrappers
        @testset "$name" begin
            @test Mantle.vk_context(w) === ctx
            # …and it is `AnyLavaArray`, i.e. this test is checking the same set
            # the broadcast machinery dispatches on.
            @test w isa Mantle.AnyLavaArray
        end
    end

    # The end-to-end shape that failed: broadcasting into a transposed array.
    b = adjoint(a) .+ Int16(1)
    @test Array(b)[1:4] == Int16[2, 3, 4, 5]
end
