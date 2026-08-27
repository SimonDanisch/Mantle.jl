"""
`mul!` on a matrix whose elements are not numbers.

Lava's GEMM kernels accumulate in the destination's element type and scale by it
— `muladd(T(A[…]), T(B[…]), acc)` and `T(α)` — so they need `T` to be
constructible from the operand and scalar types. `C = A * B` enters as
`mul!(C, A, B, true, false)`, so a matrix of a plain two-field struct used to
reach that and throw `MethodError: no method matching Duo{Float16}(::Bool)`.

The signature is bounded to `T<:Number` instead, which lets `LinearAlgebra.mul!`
reach `GPUArrays.generic_matmatmul!` — a GPU kernel that multiplies with the
types' own `*` and promotes the accumulator, which is the right behaviour for
these types. The two assertions below are the two halves of that: the fallback
actually computes the right answer, and the numeric path is still Lava's own.
"""

using Test, Lava, LinearAlgebra, GPUArrays

# Everything `GPUArrays.generic_matmatmul!`'s kernel asks of an element type.
# `zero` is needed for an INSTANCE, not just the type: the kernel seeds its
# accumulator with `zero(A[i,1]*B[1,j] + A[i,1]*B[1,j])`, and Base has no generic
# `zero(x) = zero(typeof(x))` for non-`Number`s.
struct Pair2{T}
    a::T
    b::T
end
Base.zero(::Type{Pair2{T}}) where {T} = Pair2(zero(T), zero(T))
Base.zero(x::Pair2) = zero(typeof(x))
Base.:(+)(x::Pair2, y::Pair2) = Pair2(x.a + y.a, x.b + y.b)
Base.:(*)(x::Pair2, y::Number) = Pair2(x.a * y, x.b * y)
Base.:(*)(x::Number, y::Pair2) = Pair2(x * y.a, x * y.b)

@testset "mul! with a non-numeric element type" begin
    @testset "$T" for T in (Float16, Float32, Float64)
        n = 4
        A = [Pair2(T(i), T(2i)) for i in 1:n, _ in 1:n]
        B = T.(reshape(1:n^2, n, n))

        @test Array(LavaArray(A) * LavaArray(B)) == A * B
        @test Array(LavaArray(B) * LavaArray(A)) == B * A
    end

    # Which method handles what. Narrowing the bound too far would send numeric
    # GEMMs to the generic elementwise kernel — still correct, and a large silent
    # performance regression, so it is worth pinning rather than inferring from a
    # timing.
    @testset "dispatch stays where it belongs" begin
        numeric = Base.which(LinearAlgebra.mul!,
                             Tuple{LavaArray{Float32,2}, LavaArray{Float32,2},
                                   LavaArray{Float32,2}, Bool, Bool})
        @test parentmodule(numeric) === Lava

        generic = Base.which(LinearAlgebra.mul!,
                             Tuple{LavaArray{Pair2{Float32},2}, LavaArray{Pair2{Float32},2},
                                   LavaArray{Float32,2}, Bool, Bool})
        @test parentmodule(generic) !== Lava
    end
end
