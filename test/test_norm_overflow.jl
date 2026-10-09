# Regression test for 2-norm rescaling under flush-to-zero.
#
# Bug: `norm(v, 2)` rescales by `maxabs = maximum(abs, v)` to avoid overflow,
# computing `sum((abs(x)/maxabs)^2)`. Writing the division in the source is NOT
# enough: GPU drivers (and LLVM arcp fast-math) lower Float64 `x / m` to
# `x * (1/m)`, so for `maxabs` near `floatmax` the reciprocal `1/maxabs` is
# subnormal and is flushed to zero under FTZ (the default on lavapipe AND RDNA3),
# collapsing the whole norm to 0.0. Fixed by dividing by `sqrt(maxabs)` twice
# (`1/sqrt(m)` is always normal) in `lava_norm_p2_rescale`.
#
# These are exactly the cases that failed in the GPUArrays `linalg/norm` suite
# (2-norm, sizes (2,) and (2,2,2), Float64 and ComplexF64, overflow + underflow
# rescaling edges). The single-precision types have the same edge, at their own
# `floatmax`, and are the only ones a device without `Float64` can run.

using Test
using Mantle
using LinearAlgebra
include(joinpath(@__DIR__, "testbackend.jl"))

@testset "norm 2-norm FTZ rescaling" begin
    types = Mantle.supports_float64(TESTBACKEND) ? (Float32, ComplexF32, Float64, ComplexF64) :
                                                   (Float32, ComplexF32)
    for T in types
        R = real(T)
        for sz in [(2,), (2, 2, 2)]
            # Overflow edge: one element at floatmax/2 forces rescaling, and the
            # naive `x/maxabs` reciprocal would be subnormal → 0 under FTZ.
            arr = rand(T, sz)
            arr[1] = T(floatmax(R) / 2)
            @test norm(Mantle.devicearray(TESTBACKEND, arr), 2) ≈ norm(arr, 2)

            # Underflow edge: all elements subnormal-adjacent.
            arr_lo = fill(T(floatmin(R) * 2), sz)
            @test norm(Mantle.devicearray(TESTBACKEND, arr_lo), 2) ≈ norm(arr_lo, 2)

            # Normal magnitudes must remain exact.
            arr_n = rand(T, sz)
            @test norm(Mantle.devicearray(TESTBACKEND, arr_n), 2) ≈ norm(arr_n, 2)
        end
    end

end
