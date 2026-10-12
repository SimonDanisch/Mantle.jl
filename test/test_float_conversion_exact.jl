# `Float32(Float16(x))` keeps its rounding on the device.
#
# A driver may treat any SPIR-V instruction without `NoContraction` as inexact,
# and Mesa's NIR folds a narrowing and a widening back into their input: the
# rounding is gone and the kernel computes something its source does not say.
# It surfaced as a fused norm + rotary kernel for Qwen-Image 2.1 that disagreed
# in a third of its outputs, one fp16 ulp each, with the two kernels it
# replaced, which round through memory. Lava now marks every `OpFConvert`
# exact (see Lava's `test_fconvert_exact.jl` for the emitted form).
#
# `x * r` is an fp32 product the conversion has to round, and `* c` afterwards
# only keeps the value from being stored straight back out. The comparison is
# against the host doing the same arithmetic, with and without the rounding:
# the device must be the first, and far from the second.
#
# NVIDIA drops the rounding of a narrowed value it keeps computing with, whatever
# the SPIR-V says (`NoContraction`, `FPFastMathMode None` from float_controls2,
# rounding and denormal modes, a bitcast round trip, a volatile variable: all
# measured on an RTX 4000 Ada, 595.99), and DNNKernels' fused norm + rotary lost
# it where the narrowed value fed a half multiply. So Lava narrows such a value in
# integer operations (`narrow_in_integers!`), on every device. The testsets below
# run that sequence on the device: every half, every tie between neighbouring
# halves and the floats either side of it, the subnormal range, overflow, and the
# narrowed value multiplied in half precision before it is widened.

using Test, Mantle, Random
import KernelInterface as KI
include(joinpath(@__DIR__, "testbackend.jl"))

function fce_roundtrip!(out, a, c, r::Float32, n::Int32)
    i = Int32(KI.get_global_id().x)
    if i <= n
        @inbounds out[i] = Float16(Float32(Float16(Float32(a[i]) * r)) * c[i])
    end
    return nothing
end

@testset "a float conversion keeps its rounding" begin
    backend = TESTBACKEND
    rng = MersenneTwister(5)
    n = 1 << 14
    ah = Float16.(randn(rng, Float32, n))
    ch = randn(rng, Float32, n)
    r = 0.7310585f0
    out = Mantle.devicearray(TESTBACKEND, zeros(Float16, n))
    KI.@launch backend ndrange = n workgroupsize = 64 fce_roundtrip!(
        out, Mantle.devicearray(TESTBACKEND, ah), Mantle.devicearray(TESTBACKEND, ch), r, Int32(n))
    got = Array(out)
    rounded = [Float16(Float32(Float16(Float32(ah[i]) * r)) * ch[i]) for i in 1:n]
    unrounded = [Float16(Float32(ah[i]) * r * ch[i]) for i in 1:n]
    # The unrounded chain differs in about a quarter of the elements; the device
    # may differ from the rounded one only where it fuses the final multiply,
    # which it has no add to fuse with.
    @test count(got .!= unrounded) > n ÷ 10
    @test got == rounded
end

function fce_widen_narrowed!(out, x, n::Int32)
    i = Int32(KI.get_global_id().x)
    if i <= n
        @inbounds out[i] = Float32(Float16(x[i]))
    end
    return nothing
end

@testset "a narrowed value is rounded exactly: every half, every tie" begin
    backend = TESTBACKEND
    halves = reinterpret.(Float16, UInt16(0):typemax(UInt16))
    # Each half as a Float32 (exact), and between each pair of neighbouring
    # finite halves the tie and the floats on either side of it: the inputs where
    # round-to-nearest-even decides. Plus zeros, subnormals, both sides of the
    # overflow threshold 65520, infinities and NaNs, which the halves include.
    finite = sort(filter(isfinite, Float64.(halves)))
    ties = Float32.((finite[1:end-1] .+ finite[2:end]) ./ 2)
    x = [Float32.(halves); ties; prevfloat.(ties); nextfloat.(ties);
         Float32[65519.996, 65520, 65536, -65520, 2.0^-25, nextfloat(2f0^-25), 2.0^-26]]
    n = length(x)
    out = Mantle.devicearray(TESTBACKEND, zeros(Float32, n))
    KI.@launch backend ndrange = n workgroupsize = 64 fce_widen_narrowed!(
        out, Mantle.devicearray(TESTBACKEND, x), Int32(n))
    got = Array(out)
    want = Float32.(Float16.(x))
    same(a, b) = isnan(a) ? isnan(b) : a === b
    @test count(.!same.(want, got)) == 0
    # And values that round, in the subnormal range too.
    rng = MersenneTwister(7)
    y = [randn(rng, Float32, 1 << 14); ldexp.(rand(rng, Float32, 1 << 14), -14)]
    m = length(y)
    out2 = Mantle.devicearray(TESTBACKEND, zeros(Float32, m))
    KI.@launch backend ndrange = m workgroupsize = 64 fce_widen_narrowed!(
        out2, Mantle.devicearray(TESTBACKEND, y), Int32(m))
    @test Array(out2) == Float32.(Float16.(y))
end

function fce_halfmul!(out, x, g, r::Float32, n::Int32)
    i = Int32(KI.get_global_id().x)
    if i <= n
        @inbounds out[i] = Float32(Float16(x[i] * r) * g[i])
    end
    return nothing
end

# The shape that broke DNNKernels' fused norm + rotary on NVIDIA: the narrowed
# value is not widened itself but multiplied in half precision first.
@testset "a narrowed value multiplied in half keeps its rounding" begin
    backend = TESTBACKEND
    rng = MersenneTwister(8)
    n = 1 << 15
    x = 3f0 .* randn(rng, Float32, n)
    g = Float16.(1 .+ 0.3f0 .* randn(rng, Float32, n))
    r = 0.7310585f0
    out = Mantle.devicearray(TESTBACKEND, zeros(Float32, n))
    KI.@launch backend ndrange = n workgroupsize = 64 fce_halfmul!(
        out, Mantle.devicearray(TESTBACKEND, x), Mantle.devicearray(TESTBACKEND, g), r, Int32(n))
    got = Array(out)
    rounded = [Float32(Float16(x[i] * r) * g[i]) for i in 1:n]
    unrounded = [Float32(Float16(x[i] * r * Float32(g[i]))) for i in 1:n]
    @test count(rounded .!= unrounded) > n ÷ 20     # the rounding is visible here
    @test got == rounded
end
