"""
`mapreducedim!` into a transposed destination.

`Base.mapreducedim!` routes every GPU-array destination to
`GPUArrays.mapreducedim!`, but `transpose(::LavaArray)` is a `Transpose`, not a
`LavaArray`, so it used to miss Lava's methods and hit GPUArrays' generic
`error("Not implemented")`. The GPUArrays conformance suite reduces into
`transpose(...)`/`adjoint(...)` for every eltype it tests, so this one gap
accounted for all 133 errors it reported.

Every assertion below compares the destination's PARENT STORAGE, not the
logical values it presents. That distinction is the whole point: reducing
straight into `parent(R)` produces the right logical answer for `transpose` and
a conjugated one for `adjoint`, so a test that only reads `collect(R)`, or that
only uses real eltypes, passes against the broken implementation.
"""

using Test, Lava, LinearAlgebra, GPUArrays

@testset "mapreducedim! into transposed destinations" begin
    # The shapes the conformance suite uses: a vector parent presented as a row,
    # and a matrix parent presented with its strides swapped.
    CASES = [((2, 2), (2,), identity), ((3, 2, 10), (2, 3), abs2)]

    @testset "$ET" for ET in (Float32, Int64, Complex{Int64}, ComplexF32)
        for (szA, szR, f) in CASES, wrap in (transpose, adjoint)
            A = ET <: Complex ? convert(Array{ET}, rand(1:5, szA) .+ rand(1:5, szA) .* im) :
                                convert(Array{ET}, rand(1:5, szA))

            P_cpu, P_gpu = zeros(ET, szR), LavaArray(zeros(ET, szR))
            Base.mapreducedim!(f, +, wrap(P_cpu), A)
            Base.mapreducedim!(f, +, wrap(P_gpu), LavaArray(A))

            @test Array(P_gpu) == P_cpu
        end
    end

    # `Adjoint` conjugates on write, so for a complex eltype the two wrappers
    # must leave *different* bytes behind even though they present the same
    # values. Without this, an implementation that ignores the distinction still
    # passes everything above on real eltypes.
    @testset "adjoint conjugates on write" begin
        A = Complex{Int64}[1+1im 2+2im; 3+3im 4+4im]
        store(wrap) = (P = LavaArray(zeros(Complex{Int64}, 2));
                       Base.mapreducedim!(identity, +, wrap(P), LavaArray(A));
                       Array(P))

        @test store(transpose) == Complex{Int64}[4+4im, 6+6im]
        @test store(adjoint)   == Complex{Int64}[4-4im, 6-6im]
        @test store(transpose) != store(adjoint)
    end

    # `init === nothing` accumulates into the destination rather than
    # overwriting it, and the seed has to be read back *through* the wrapper.
    @testset "accumulates into existing contents" begin
        A = Float32[1 2; 3 4]
        P = LavaArray(Float32[10, 20])
        Base.mapreducedim!(identity, +, transpose(P), LavaArray(A))
        @test Array(P) == Float32[14, 26]
    end

    @testset "returns the destination it was given" begin
        R = transpose(LavaArray(ones(Int64, (2, 3))))
        @test Base.mapreducedim!(identity, *, R, LavaArray(rand(1:5, (3, 2, 10)))) === R
    end
end

# `Base.mapreducedim!` accumulates into whatever the destination already holds; a
# supplied `init` is the opposite instruction, meaning "R is uninitialized
# scratch, overwrite it". Lava implemented both as overwrite, so
# `sum!`/`reducedim!` into a non-empty destination dropped its contents.
#
# The GPUArrays conformance suite cannot catch this. It only ever seeds a
# destination with the neutral element (`zeros` for `+`, `ones` for `*`), and
# `op(neutral, x) == x` makes accumulating and overwriting indistinguishable.
# Every seed below is deliberately NOT neutral.
@testset "mapreducedim! accumulates into a dense destination" begin
    @testset "$op $szA -> $szR" for (op, szA, szR) in [
            (+, (2, 2),    (1,)),        # full reduction, and the Float32 fast path
            (+, (2, 2),    (2,)),        # implicit trailing singleton
            (+, (2, 2),    (1, 2)),      # dims=1
            (+, (2, 3),    (2, 1)),      # dims=2
            (*, (2, 2),    (2,)),        # a non-additive op
            (+, (2, 3, 4), (2, 1, 1)),   # multiple dims reduced
        ]
        for ET in (Float32, Int64)
            A = convert(Array{ET}, reshape(collect(1:prod(szA)), szA))
            seed = convert(Array{ET}, reshape(collect(10:10:10*prod(szR)), szR))

            R_cpu, R_gpu = copy(seed), LavaArray(copy(seed))
            Base.mapreducedim!(identity, op, R_cpu, A)
            Base.mapreducedim!(identity, op, R_gpu, LavaArray(A))

            @test Array(R_gpu) == R_cpu
        end
    end

    # The other half of the contract: an explicit `init` must ignore the
    # destination's contents, because `_mapreduce` hands over uninitialized
    # memory. Breaking this would break every `sum`/`prod` on the package.
    @testset "an explicit init overwrites instead" begin
        A = LavaArray(Float32[1 2; 3 4])
        R = LavaArray(Float32[999, 999])
        GPUArrays.mapreducedim!(identity, +, R, A; init = 0.0f0)
        @test Array(R) == Float32[3, 7]

        @test sum(LavaArray(Float32[1, 2, 3, 4])) == 10.0f0
        @test Array(sum(LavaArray(Float32[1 2; 3 4]); dims = 2)) == reshape(Float32[3, 7], 2, 1)
        @test prod(LavaArray(Float32[1, 2, 3, 4])) == 24.0f0
    end
end
