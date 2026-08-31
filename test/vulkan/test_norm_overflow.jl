# Regression test for Float64/ComplexF64 2-norm rescaling under flush-to-zero.
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
# rescaling edges).

using Test
using Lava, Mantle
using LinearAlgebra

@testset "norm 2-norm FTZ rescaling (Float64/ComplexF64)" begin
    for T in (Float64, ComplexF64)
        R = real(T)
        for sz in [(2,), (2, 2, 2)]
            # Overflow edge: one element at floatmax/2 forces rescaling, and the
            # naive `x/maxabs` reciprocal would be subnormal → 0 under FTZ.
            arr = rand(T, sz)
            arr[1] = T(floatmax(R) / 2)
            @test norm(MVE.LavaArray(arr), 2) ≈ norm(arr, 2)

            # Underflow edge: all elements subnormal-adjacent.
            arr_lo = fill(T(floatmin(R) * 2), sz)
            @test norm(MVE.LavaArray(arr_lo), 2) ≈ norm(arr_lo, 2)

            # Normal magnitudes must remain exact.
            arr_n = rand(T, sz)
            @test norm(MVE.LavaArray(arr_n), 2) ≈ norm(arr_n, 2)
        end
    end

    # Direct check of the failing kernel value: sum of (x/maxabs)^2 must be ~1
    # for a single dominant element, not 0.
    let v = MVE.LavaArray([floatmax(Float64) / 2, 0.5])
        maxabs = convert(Float64, maximum(abs, v))
        @test MVE.lava_norm_p2_rescale(v, maxabs) ≈ floatmax(Float64) / 2
    end
end
