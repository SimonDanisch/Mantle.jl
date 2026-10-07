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

using Test, Lava, Mantle, Random
import KernelInterface as KI

function fce_roundtrip!(out, a, c, r::Float32, n::Int32)
    i = Int32(KI.get_global_id().x)
    if i <= n
        @inbounds out[i] = Float16(Float32(Float16(Float32(a[i]) * r)) * c[i])
    end
    return nothing
end

@testset "a float conversion keeps its rounding" begin
    backend = LavaBackend()
    rng = MersenneTwister(5)
    n = 1 << 14
    ah = Float16.(randn(rng, Float32, n))
    ch = randn(rng, Float32, n)
    r = 0.7310585f0
    out = LavaArray(zeros(Float16, n))
    KI.@kernel backend ndrange = n workgroupsize = 64 fce_roundtrip!(
        out, LavaArray(ah), LavaArray(ch), r, Int32(n))
    got = Array(out)
    rounded = [Float16(Float32(Float16(Float32(ah[i]) * r)) * ch[i]) for i in 1:n]
    unrounded = [Float16(Float32(ah[i]) * r * ch[i]) for i in 1:n]
    # The unrounded chain differs in about a quarter of the elements; the device
    # may differ from the rounded one only where it fuses the final multiply,
    # which it has no add to fuse with.
    #
    # `broken` on NVIDIA: its compiler folds the narrowing and the widening away
    # despite `NoContraction` — on 2026-10-06 an RTX 4000 Ada (595.99) and an RTX
    # 3070 Laptop (595.91) returned the UNROUNDED chain in every element. Mesa
    # honours the decoration; NVIDIA would need something else (untried:
    # `FPFastMathMode` from `VK_KHR_shader_float_controls2`). An unexpected pass on
    # NVIDIA fails, which is the signal that something now holds the rounding.
    nvidia = Mantle.VK.get_physical_device_properties(
        Mantle.vk_context().physical_device).vendor_id == 0x10de
    @test count(got .!= unrounded) > n ÷ 10 broken = nvidia
    @test got == rounded broken = nvidia
end
