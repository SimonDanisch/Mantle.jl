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

# Both directions, distinct operands and counts at zero, width and beyond.
# Comparing the full intrinsic name with "llvm.fshl" selected the right-shift
# lowering for every left shift; the suffix carries the integer width.
for (T, width) in ((UInt16, 16), (UInt32, 32), (UInt64, 64)), (fn, intrinsic) in
        ((:funnel_left, "llvm.fshl"), (:funnel_right, "llvm.fshr"))
    ir = """
    declare i$width @$intrinsic.i$width(i$width, i$width, i$width)
    define i$width @entry(i$width %a, i$width %b, i$width %s) {
      %r = call i$width @$intrinsic.i$width(i$width %a, i$width %b, i$width %s)
      ret i$width %r
    }
    """
    @eval @inline $fn(a::$T, b::$T, s::$T) =
        Base.llvmcall(($ir, "entry"), $T, Tuple{$T, $T, $T}, a, b, s)
end
KA.@kernel function funnel_kernel!(left, right, a, b, s)
    i = KA.@index(Global)
    @inbounds begin
        left[i] = funnel_left(a[i], b[i], s[i])
        right[i] = funnel_right(a[i], b[i], s[i])
    end
end

@testset "funnel shifts reduce counts modulo the width" begin
    backend = TESTBACKEND
    @testset "funnel shifts of a $T" for T in (UInt16, UInt32, UInt64)
        width = 8 * sizeof(T)
        counts = T[0, 1, width-1, width, width+1, 2width-1, 2width, typemax(T)]
        a = repeat(T[0x1234, 0x8a57, typemax(T), 0]; inner = length(counts))
        b, s = reverse(a), repeat(counts, 4)
        mask = (big(1) << width) - 1
        # Concatenate in a wider integer, shift, then take the requested half.
        left = [T(((((big(x) << width) | big(y)) << Int(z % T(width))) >> width) & mask)
                for (x, y, z) in zip(a, b, s)]
        right = [T((((big(x) << width) | big(y)) >> Int(z % T(width))) & mask)
                 for (x, y, z) in zip(a, b, s)]
        da, db, ds = map(h -> Mantle.devicearray(backend, h), (a, b, s))
        dl = Mantle.devicearray(backend, zeros(T, length(s)))
        dr = Mantle.devicearray(backend, zeros(T, length(s)))
        funnel_kernel!(backend)(dl, dr, da, db, ds; ndrange = length(s))
        Mantle.waitidle(Mantle.Device(backend))
        @test Array(dl) == left
        @test Array(dr) == right
    end
end
