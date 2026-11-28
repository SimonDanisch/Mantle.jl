# Reductions and scans that AcceleratedKernels runs on lanes, and a triangular product
# whose diagonal is an argument, computed on the device.
#
# `istril` reduces with lanes: two `Bool`s that SROA keeps in one 16-bit variable,
# written a byte at a time and read whole. The SPIR-V emitter wrote the second byte
# into the first, and before that its splice had a 32-bit mask on a 16-bit variable,
# which spirv-val refused: GPUArrays' `istril`, `istriu` and `isdiag` tests failed on
# every element type. A scan along a matrix's long rows with an operator without a
# neutral element splits each row into chunks whose totals are lanes, a value and a
# `valid` flag; the padding after the flag was copied as a word that claimed four-byte
# alignment at byte 9, which NVIDIA wrote over the flag, and every chunk total read as
# empty. A unit triangular product of complex matrices, `generic_trimatmul!`, reads
# `unit ? oA : A[i, i]`: a choice between an argument copied to a private variable and
# a device element, which no SPIR-V pointer can be, and the argument arrives as one
# 64-bit store into its two floats. Lava's `test_subword_integer_alloca.jl`,
# `test_memcpy_chunk_alignment.jl` and `test_float_slot_pun.jl` pin the patterns.

using Test, Mantle, LinearAlgebra
include(joinpath(@__DIR__, "testbackend.jl"))

compose((a1, b1), (a2, b2)) = (a2 * a1, a2 * b1 + b2)

@testset "lane reductions and scans, and a unit triangular product" begin
    @testset "$f of $(eltype(M))" for T in testeltypes(Float32, Int32, ComplexF32, UInt8, Int64),
                                      f in (istril, istriu, isdiag),
                                      M in (A -> A, tril, triu, A -> Matrix(Diagonal(diag(A))))
        A = M(rand(T, 7, 7))
        @test f(Mantle.devicearray(TESTBACKEND, A)) == f(A)
    end

    # Long rows, so that each is scanned in chunks.
    @testset "a scan along rows with $name" for (name, op, A) in
            (("the first operand", (a, b) -> a, rand(1:10, 7, 10_000)),
             ("a non-commutative operator", compose, [(rand(-1:1), rand(-3:3)) for _ in 1:7, _ in 1:10_000]))
        @test Array(accumulate(op, Mantle.devicearray(TESTBACKEND, A); dims = 2)) == accumulate(op, A; dims = 2)
    end

    @testset "$W" for W in (UpperTriangular, LowerTriangular, UnitUpperTriangular, UnitLowerTriangular)
        A, B = rand(ComplexF32, 8, 8), rand(ComplexF32, 8, 3)
        got = Array(W(Mantle.devicearray(TESTBACKEND, A)) * Mantle.devicearray(TESTBACKEND, B))
        @test got ≈ W(A) * B rtol = 1f-5
    end
end
