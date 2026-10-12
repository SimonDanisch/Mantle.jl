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
# NVIDIA folds the pair whatever the SPIR-V says (`NoContraction`,
# `FPFastMathMode None` from float_controls2, rounding and denormal modes, a
# bitcast round trip, a volatile variable: all measured on an RTX 4000 Ada,
# 595.99). So Lava widens a value it just narrowed in integer operations on the
# narrowed bits (`widen_narrowed_floats!`), on every device, and the second
# testset checks that widening against the host for every half there is.

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

@testset "a narrowed value widens exactly, every half" begin
    backend = TESTBACKEND
    halves = reinterpret.(Float16, UInt16(0):typemax(UInt16))
    # Each half as a Float32, so the narrowing is exact and the widening alone is
    # under test: zeros, subnormals, normals, infinities and NaNs of both signs.
    x = Float32.(halves)
    n = length(x)
    out = Mantle.devicearray(TESTBACKEND, zeros(Float32, n))
    KI.@launch backend ndrange = n workgroupsize = 64 fce_widen_narrowed!(
        out, Mantle.devicearray(TESTBACKEND, x), Int32(n))
    got = Array(out)
    same(a, b) = isnan(a) ? isnan(b) : a === b
    @test all(same.(x, got))
    # And between them: values that round, in the subnormal range too.
    rng = MersenneTwister(7)
    y = [randn(rng, Float32, 1 << 14); ldexp.(rand(rng, Float32, 1 << 14), -14)]
    m = length(y)
    out2 = Mantle.devicearray(TESTBACKEND, zeros(Float32, m))
    KI.@launch backend ndrange = m workgroupsize = 64 fce_widen_narrowed!(
        out2, Mantle.devicearray(TESTBACKEND, y), Int32(m))
    @test Array(out2) == Float32.(Float16.(y))
end
