# Byte swap and bit reverse compute on the device what they compute on the host, at
# every integer width.
#
# Lava had no lowering of `llvm.bswap`, which LLVM forms from byte-shuffling code of
# its own accord: Hikari's Sobol sampler failed to compile on every Vulkan device.
# And `bitreverse` was `OpBitReverse` at any width, which Vulkan allows on 32-bit
# integers only. Lava's `test_bit_intrinsics.jl` checks the emitted form; this
# checks the values.

using Test, Mantle, Random
import KernelAbstractions as KA
include(joinpath(@__DIR__, "testbackend.jl"))

KA.@kernel function bits_swap_reverse!(swapped, reversed, @Const(x))
    i = KA.@index(Global)
    @inbounds swapped[i] = bswap(x[i])
    @inbounds reversed[i] = bitreverse(x[i])
end

@testset "bswap and bitreverse of a $T" for T in (UInt16, UInt32, UInt64)
    h = rand(Xoshiro(1), T, 1000)
    x = Mantle.devicearray(TESTBACKEND, h)
    swapped = Mantle.devicearray(TESTBACKEND, zeros(T, 1000))
    reversed = Mantle.devicearray(TESTBACKEND, zeros(T, 1000))
    bits_swap_reverse!(TESTBACKEND)(swapped, reversed, x; ndrange = length(h))
    @test Array(swapped) == bswap.(h)
    @test Array(reversed) == bitreverse.(h)
end
