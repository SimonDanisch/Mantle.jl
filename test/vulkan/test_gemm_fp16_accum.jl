# The scalar GEMM must not accumulate in the destination's precision.
#
# `strided_gemm_kernel!` took `T = eltype(C)` and accumulated in it, so an fp16
# destination gave an fp16 accumulator over K = 1280 or 5120. Found by the
# Whisper port on block 1's `fc2` (K = 5120), measured against an fp64 reference
# over the *same* fp16 inputs:
#
#     fp16 destination -> fp16 accumulator                 rel rms  4.83e-2
#     predicted by sqrt(K) * eps(Float16) / 2                       3.49e-2
#     same kernel, fp32 destination, rounded to fp16 after          2.06e-4
#     Lava's coopmat path (fp32 accumulator)                        2.13e-4
#     PyTorch's own fp16 addmm                                      3.96e-4
#
# 234x, from one line — and the agreement between the predicted and the measured
# figure is what makes it a diagnosis rather than a correlation.
#
# The scope is wider than one model: this is every fp16 matmul that misses the
# cooperative-matrix path, and `mm_coopmat_applicable` declines whenever an
# operand arrives wrapped, which in a raw exported graph is every `addmm`.
#
# The threshold below is deliberately far from both numbers — 1e-3 is 5x above
# the fixed error and 48x below the broken one — so it pins the *behaviour*
# rather than a measurement of this machine.

using Test, Lava, KernelAbstractions
using LinearAlgebra
using Random
const KA = KernelAbstractions
# `mul!` explicitly, and by name. This file has no `using` of its own — it is
# `include`d into whatever `runtests.jl` has in scope — and both LinearAlgebra and
# LLVM put a `mul!` there, so the bare name resolves to neither and the whole
# testset errors out before a single assertion runs. It reported as a failing GEMM
# for as long as that was true.
using LinearAlgebra: mul!

relrms(got, want) = sqrt(sum(abs2, Float64.(got) .- want) / sum(abs2, want))

"Force the scalar path: a wrapped operand is what makes `mm_coopmat_applicable` decline."
function scalar_mul(::Type{T}, A::Matrix{Float16}, B::Matrix{Float16}) where {T}
    be = LavaBackend()
    M, K = size(A); N = size(B, 2)
    # `transpose(Aᵀ)` is strided and wrapped: same numbers, scalar kernel.
    At = KA.allocate(be, Float16, K, M); copyto!(At, Matrix(transpose(A)))
    Bg = KA.allocate(be, Float16, K, N); copyto!(Bg, B)
    C  = KA.allocate(be, T, M, N);       fill!(C, zero(T))
    mul!(C, transpose(At), Bg, one(T), zero(T))
    KA.synchronize(be)
    Array(C)
end

@testset "fp16 GEMM accumulates wider than fp16" begin
    rng = Random.MersenneTwister(20260802)
    M, N = 64, 8
    for K in (1280, 5120)
        A = Float16.(randn(rng, Float32, M, K) .* 0.1f0)
        B = Float16.(randn(rng, Float32, K, N) .* 0.1f0)
        # Reference: exact arithmetic over the very same fp16 inputs, so the only
        # thing under test is the accumulator, not the inputs' rounding.
        want = Float64.(A) * Float64.(B)

        e16 = relrms(scalar_mul(Float16, A, B), want)

        # An fp16 accumulator lands near sqrt(K)*eps(Float16)/2; a wide one is
        # ~1e-4 and must not depend on the destination type.
        naive = sqrt(K) * Float64(eps(Float16)) / 2
        @test e16 < 1e-3
        @test e16 < naive / 10           # nowhere near the fp16-accumulator law

        # The floor an fp16 *destination* cannot go below: take the exact answer
        # and merely store it as Float16. A wide accumulator should land on this
        # number, because after it the only rounding left is the single store.
        #
        # Do NOT compare against the fp32-destination run instead. That was the
        # first form of this test and it cannot pass: the fp32 run never rounds
        # to half at all, so it sits at 2.5e-8 — four orders below — and `e16 <
        # 20 * e32` is unsatisfiable however wide the accumulator is. It read as
        # a failure of the fix on a machine where the fix was already perfect.
        floor16 = relrms(Float16.(want), want)
        @test e16 < 1.05 * floor16       # measured: equal to 8 significant digits
    end
end

@testset "gemmaccum widens only the half type" begin
    @test MVE.gemmaccum(Float16) === Float32
    @test MVE.gemmaccum(Float32) === Float32
    @test MVE.gemmaccum(Float64) === Float64
    @test MVE.gemmaccum(Int32)   === Int32
end
