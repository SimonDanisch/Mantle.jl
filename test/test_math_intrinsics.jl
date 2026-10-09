# Math on the device computes what the host computes.
#
# This was a Tier 1 emission test: it compiled each function and checked the
# GLSL.std.450 instruction it became (`Sin`, `Cos`, `FClamp`, ...) and that a
# Float32 kernel declared no `OpTypeFloat 64`. Which instruction a function
# lowers to is the Vulkan compiler's business; what a caller sees is the value,
# so the same kernels now run and are compared with the host. A Float64 leak in
# a Float32 kernel shows up too, as a kernel that does not compile on a device
# without Float64.
#
# The rest are regressions that always ran: each pins a Base function the
# compiler has to override for the device (`copysign`, `min`/`max`, `^`,
# `round`) or an LLVM intrinsic it has to emulate (the saturating integer ops),
# and each checks the value.

using Test
using Mantle
using KernelAbstractions
using KernelAbstractions: @kernel, @index
include(joinpath(@__DIR__, "testbackend.jl"))
const KA = KernelAbstractions

@kernel function unary_f32!(B, @Const(A), f)
    i = @index(Global, Linear)
    @inbounds B[i] = f(A[i])
end

clamp01(x) = clamp(x, 0.0f0, 1.0f0)

@testset "Math Intrinsics" begin

    # Inputs where each function is defined. The tolerances are at or above the
    # precision the Vulkan spec requires — absolute 2^-11 for `sin`/`cos` on
    # [-π, π], a few ulp for `sqrt`, 3 + 2|x| ulp for `exp`/`exp2`, absolute
    # 2^-21 near 1 for `log`/`log2` — so a conforming driver passes and a wrong
    # function does not; they are not tuned to any driver. The rest are exact.
    n = 1024
    signed = Float32.(range(-10, 10; length = n))
    f32_math = [
        (sin,   Float32.(range(-π, π; length = n)), 2f0^-11, 0f0),
        (cos,   Float32.(range(-π, π; length = n)), 2f0^-11, 0f0),
        (sqrt,  Float32.(range(0, 100; length = n)), 0f0, 1f-5),
        (abs,   signed, 0f0, 0f0),
        (floor, signed, 0f0, 0f0),
        (ceil,  signed, 0f0, 0f0),
        (exp,   Float32.(range(-4, 4; length = n)), 0f0, 1f-5),
        (log,   Float32.(range(0.25, 8; length = n)), 1f-6, 1f-5),
        (exp2,  Float32.(range(-4, 4; length = n)), 0f0, 1f-5),
        (log2,  Float32.(range(0.25, 8; length = n)), 1f-6, 1f-5),
        (sign,  signed, 0f0, 0f0),
        (clamp01, Float32.(range(-2, 2; length = n)), 0f0, 0f0),
    ]

    @testset "f32 $f" for (f, xs, atol, rtol) in f32_math
        A = Mantle.devicearray(TESTBACKEND, xs)
        B = Mantle.devicearray(TESTBACKEND, zeros(Float32, length(xs)))
        unary_f32!(TESTBACKEND)(B, A, f; ndrange = length(xs))
        KA.synchronize(TESTBACKEND)
        got = Array(B)
        want = f.(xs)
        # Named, not counted: a failure says which inputs.
        bad = [(x, g, w) for (x, g, w) in zip(xs, got, want)
               if !isapprox(g, w; atol, rtol)]
        @test bad == Tuple{Float32,Float32,Float32}[]
    end

    @testset "copysign zero-preserving (bitwise sign-copy)" begin
        # copysign(x, 0) must return +|x|, not 0. GLSL.std.450 FSign(0) is 0,
        # so `FAbs(x) * FSign(y)` produces 0 for any y==0 — breaks ray-tracing
        # `safe_invdir` clamps and similar IEEE-dependent code.
        #
        # Lava's emitter errors on any `llvm.copysign` that slips past the
        # `Base.copysign` overlay in its `device/math.jl`. Each kernel below
        # compiles (so no leak to the emitter) AND returns the IEEE result.

        @kernel function copysign_kernel!(out, x_arr, y_arr)
            i = @index(Global, Linear)
            @inbounds out[i] = copysign(x_arr[i], y_arr[i])
        end

        x = Float32[1f-5,  1f-5,  1f-5, -3f0, 2f0, 2f0]
        y = Float32[ 0f0, -0f0, -1f0,  2f0, Inf32, -Inf32]
        expected = Float32[1f-5, -1f-5, -1f-5, 3f0, 2f0, -2f0]

        x_arr = Mantle.devicearray(TESTBACKEND, x)
        y_arr = Mantle.devicearray(TESTBACKEND, y)
        out = Mantle.devicearray(TESTBACKEND, zeros(Float32, length(x)))
        copysign_kernel!(TESTBACKEND)(out, x_arr, y_arr; ndrange=length(x))
        KA.synchronize(TESTBACKEND)
        @test reinterpret(UInt32, Array(out)) == reinterpret(UInt32, expected)

        # Float64, where the device has it
        for T in testeltypes(Float64)
            xd = T[1e-10,  1e-10,  1e-10, -3.0]
            yd = T[  0.0,   -0.0,   -1.0,  2.0]
            expected_d = T[1e-10, -1e-10, -1e-10, 3.0]
            x_arr64 = Mantle.devicearray(TESTBACKEND, xd); y_arr64 = Mantle.devicearray(TESTBACKEND, yd)
            out64 = Mantle.devicearray(TESTBACKEND, zeros(T, length(xd)))
            copysign_kernel!(TESTBACKEND)(out64, x_arr64, y_arr64; ndrange=length(xd))
            KA.synchronize(TESTBACKEND)
            @test reinterpret(UInt64, Array(out64)) == reinterpret(UInt64, expected_d)
        end
    end

    @testset "copysign leak paths" begin
        # Exercises the ways `llvm.copysign` could slip past Lava's Base.copysign
        # overlay. On Vulkan each kernel here FAILS TO COMPILE if it does,
        # because Lava's emitter errors on `llvm.copysign`; on every backend
        # the numeric checks confirm correctness.

        # 1. @fastmath: fastmath rewrites `copysign` but still hits Base.copysign.
        @kernel function fastmath_cs!(out, x, y)
            i = @index(Global, Linear)
            @inbounds out[i] = @fastmath copysign(x[i], y[i])
        end
        x = Mantle.devicearray(TESTBACKEND, Float32[1f-5, 2f0])
        y = Mantle.devicearray(TESTBACKEND, Float32[0f0, -1f0])
        out = Mantle.devicearray(TESTBACKEND, zeros(Float32, 2))
        fastmath_cs!(TESTBACKEND)(out, x, y; ndrange=2)
        KA.synchronize(TESTBACKEND)
        @test Array(out) == Float32[1f-5, -2f0]

        # 2. Vectorizable inner loop — gives LLVM a chance to recognize the
        #    bitwise sign-copy pattern and canonicalize back to `llvm.copysign`.
        @kernel function loop_cs!(out, xs, ys, n::Int32)
            i = @index(Global, Linear)
            acc = 0f0
            @inbounds for j in Int32(1):n
                acc += copysign(xs[i, j], ys[i, j])
            end
            @inbounds out[i] = acc
        end
        xs = Mantle.devicearray(TESTBACKEND, fill(1f-5, 16, 8))
        ys = Mantle.devicearray(TESTBACKEND, zeros(Float32, 16, 8))  # all zero y's — the bug case
        lout = Mantle.devicearray(TESTBACKEND, zeros(Float32, 16))
        loop_cs!(TESTBACKEND)(lout, xs, ys, Int32(8); ndrange=16)
        KA.synchronize(TESTBACKEND)
        @test all(Array(lout) .≈ 8f0 * 1f-5)  # would be 0 if old bug reappeared

        # 3. Exact `safe_invdir` pattern — the original bug trigger.
        @kernel function safe_invdir_kernel!(out, d)
            i = @index(Global, Linear)
            ooeps = 1f-5
            @inbounds dv = d[i]
            clamped = abs(dv) > ooeps ? dv : copysign(ooeps, dv)
            @inbounds out[i] = 1f0 / clamped
        end
        d = Mantle.devicearray(TESTBACKEND, Float32[0f0, -0f0, 1f0, -1f0])
        sout = Mantle.devicearray(TESTBACKEND, zeros(Float32, 4))
        safe_invdir_kernel!(TESTBACKEND)(sout, d; ndrange=4)
        KA.synchronize(TESTBACKEND)
        result = Array(sout)
        # +0 → clamp to +1e-5 → 1/1e-5 = 1e5   (NOT Inf)
        # -0 → clamp to -1e-5 → 1/-1e-5 = -1e5 (NOT Inf)
        @test result[1] == 1f5
        @test result[2] == -1f5
        @test result[3] == 1f0
        @test result[4] == -1f0
    end

    @testset "min/max non-propagating (matches GPU convention)" begin
        # Julia 1.12's `Base.min`/`Base.max` lower to `llvm.minimum`/`maximum`
        # (NaN-propagating). GLSL.std.450 has no propagating op; overlaying
        # to `llvm.minnum`/`maxnum` (non-propagating, IEEE 754-2008) matches
        # CUDA/ROCm/SPIRVIntrinsics. Lava's emitter errors on any leaked
        # `llvm.minimum`/`maximum`, so these kernels must compile AND return
        # the non-NaN operand when one input is NaN.
        @kernel function min_kernel!(out, xs, ys)
            i = @index(Global, Linear)
            @inbounds out[i] = min(xs[i], ys[i])
        end
        @kernel function max_kernel!(out, xs, ys)
            i = @index(Global, Linear)
            @inbounds out[i] = max(xs[i], ys[i])
        end
        xs = Mantle.devicearray(TESTBACKEND, Float32[1f0, NaN32, NaN32, 3f0, -1f0])
        ys = Mantle.devicearray(TESTBACKEND, Float32[NaN32, 2f0, NaN32, 5f0, -2f0])
        mi = Mantle.devicearray(TESTBACKEND, zeros(Float32, 5))
        ma = Mantle.devicearray(TESTBACKEND, zeros(Float32, 5))
        min_kernel!(TESTBACKEND)(mi, xs, ys; ndrange=5)
        max_kernel!(TESTBACKEND)(ma, xs, ys; ndrange=5)
        KA.synchronize(TESTBACKEND)
        r_min = Array(mi); r_max = Array(ma)
        # Non-NaN wins where exactly one operand is NaN; both-NaN → NaN.
        @test r_min[1] == 1f0
        @test r_min[2] == 2f0
        @test isnan(r_min[3])
        @test r_min[4] == 3f0
        @test r_min[5] == -2f0
        @test r_max[1] == 1f0
        @test r_max[2] == 2f0
        @test isnan(r_max[3])
        @test r_max[4] == 5f0
        @test r_max[5] == -1f0

        # @fastmath min/max must also reach the overlay, not llvm.minimum.
        @kernel function fastmath_min!(out, xs, ys)
            i = @index(Global, Linear)
            @inbounds out[i] = @fastmath min(xs[i], ys[i])
        end
        fout = Mantle.devicearray(TESTBACKEND, zeros(Float32, 5))
        fastmath_min!(TESTBACKEND)(fout, xs, ys; ndrange=5)
        KA.synchronize(TESTBACKEND)
        @test Array(fout)[1] == 1f0  # compiles without emitter error
    end

    @testset "saturating integer arith" begin
        # LLVM's InstCombine synthesizes llvm.usub.sat from patterns like
        # `a - min(a, b)` — Julia's `Base.rem_internal` produces it during
        # float exponent arithmetic. AMDGPU/CUDA rely on native LLVM backend
        # support; for SPIR-V we emulate in the emitter.

        # Unsigned saturating subtract / add via a kernel that receives
        # inputs from device arrays (so LLVM can't constant-fold).
        @kernel function usub_sat_kernel!(out, a, b)
            i = @index(Global, Linear)
            @inbounds out[i] = Base.llvmcall(("""
                declare i32 @llvm.usub.sat.i32(i32, i32)
                define i32 @entry(i32 %a, i32 %b) #0 {
                    %r = call i32 @llvm.usub.sat.i32(i32 %a, i32 %b)
                    ret i32 %r
                }
                attributes #0 = { alwaysinline }
            """, "entry"), UInt32, Tuple{UInt32, UInt32}, a[i], b[i])
        end
        a = Mantle.devicearray(TESTBACKEND, UInt32[10, 5, 0, typemax(UInt32), 3])
        b = Mantle.devicearray(TESTBACKEND, UInt32[3, 7, 1, 1, typemax(UInt32)])
        out = Mantle.devicearray(TESTBACKEND, zeros(UInt32, 5))
        usub_sat_kernel!(TESTBACKEND)(out, a, b; ndrange=5)
        KA.synchronize(TESTBACKEND)
        @test Array(out) == UInt32[7, 0, 0, typemax(UInt32) - 1, 0]

        @kernel function uadd_sat_kernel!(out, a, b)
            i = @index(Global, Linear)
            @inbounds out[i] = Base.llvmcall(("""
                declare i32 @llvm.uadd.sat.i32(i32, i32)
                define i32 @entry(i32 %a, i32 %b) #0 {
                    %r = call i32 @llvm.uadd.sat.i32(i32 %a, i32 %b)
                    ret i32 %r
                }
                attributes #0 = { alwaysinline }
            """, "entry"), UInt32, Tuple{UInt32, UInt32}, a[i], b[i])
        end
        a = Mantle.devicearray(TESTBACKEND, UInt32[10, typemax(UInt32), typemax(UInt32) - 5, 0, 1])
        b = Mantle.devicearray(TESTBACKEND, UInt32[3, 1, 10, 0, typemax(UInt32)])
        out = Mantle.devicearray(TESTBACKEND, zeros(UInt32, 5))
        uadd_sat_kernel!(TESTBACKEND)(out, a, b; ndrange=5)
        KA.synchronize(TESTBACKEND)
        @test Array(out) == UInt32[13, typemax(UInt32), typemax(UInt32), 0, typemax(UInt32)]

        # Signed saturating add/sub
        @kernel function sadd_sat_kernel!(out, a, b)
            i = @index(Global, Linear)
            @inbounds out[i] = Base.llvmcall(("""
                declare i32 @llvm.sadd.sat.i32(i32, i32)
                define i32 @entry(i32 %a, i32 %b) #0 {
                    %r = call i32 @llvm.sadd.sat.i32(i32 %a, i32 %b)
                    ret i32 %r
                }
                attributes #0 = { alwaysinline }
            """, "entry"), Int32, Tuple{Int32, Int32}, a[i], b[i])
        end
        a = Mantle.devicearray(TESTBACKEND, Int32[10, typemax(Int32), typemin(Int32), -5, 100])
        b = Mantle.devicearray(TESTBACKEND, Int32[3, 1, -1, -typemax(Int32), -50])
        out = Mantle.devicearray(TESTBACKEND, zeros(Int32, 5))
        sadd_sat_kernel!(TESTBACKEND)(out, a, b; ndrange=5)
        KA.synchronize(TESTBACKEND)
        @test Array(out) == Int32[13, typemax(Int32), typemin(Int32), typemin(Int32), 50]

        @kernel function ssub_sat_kernel!(out, a, b)
            i = @index(Global, Linear)
            @inbounds out[i] = Base.llvmcall(("""
                declare i32 @llvm.ssub.sat.i32(i32, i32)
                define i32 @entry(i32 %a, i32 %b) #0 {
                    %r = call i32 @llvm.ssub.sat.i32(i32 %a, i32 %b)
                    ret i32 %r
                }
                attributes #0 = { alwaysinline }
            """, "entry"), Int32, Tuple{Int32, Int32}, a[i], b[i])
        end
        a = Mantle.devicearray(TESTBACKEND, Int32[10, typemax(Int32), typemin(Int32), 5, -100])
        b = Mantle.devicearray(TESTBACKEND, Int32[3, -1, 1, -5, 50])
        out = Mantle.devicearray(TESTBACKEND, zeros(Int32, 5))
        ssub_sat_kernel!(TESTBACKEND)(out, a, b; ndrange=5)
        KA.synchronize(TESTBACKEND)
        @test Array(out) == Int32[7, typemax(Int32), typemin(Int32), 10, -150]

        # End-to-end: Base.rem on Float32 (the use case that triggered this).
        @kernel function rem_kernel!(out, x, y)
            i = @index(Global, Linear)
            @inbounds out[i] = rem(x[i], y[i])
        end
        rx = Mantle.devicearray(TESTBACKEND, Float32[7f0, -7f0, 5.5f0, 100f0])
        ry = Mantle.devicearray(TESTBACKEND, Float32[3f0, 3f0, 2.0f0, 0.3f0])
        rout = Mantle.devicearray(TESTBACKEND, zeros(Float32, 4))
        rem_kernel!(TESTBACKEND)(rout, rx, ry; ndrange=4)
        KA.synchronize(TESTBACKEND)
        gpu = Array(rout)
        cpu = Float32[rem(x, y) for (x, y) in zip(
            Float32[7, -7, 5.5, 100], Float32[3, 3, 2.0, 0.3])]
        @test gpu ≈ cpu
    end

    @testset "^ overlay (Base.:(^) → llvm.pow → GLSL Pow)" begin
        # `Base.:(^)(::Float32, ::Float32)` is overlaid in device/math.jl via
        # `@_lava_binary_intrinsic ... "llvm.pow"`, matching SPIRVIntrinsics'
        # `Base.:(^) → OpenCL pow` and AMDGPU's `Base.:(^) → __ocml_pow`.
        # Lock that in: bypass Julia's CPU `^` (which uses Float64 intermediate
        # `power_by_squaring` and a `throw_exp_domainerror` stub) and hit the
        # direct llvm.pow → GLSL Pow path.
        @kernel function pow_kernel!(out, x, y)
            i = @index(Global, Linear)
            @inbounds out[i] = x[i] ^ y[i]
        end

        # Float32, positive base — all cases well-defined in GLSL Pow.
        x = Mantle.devicearray(TESTBACKEND, Float32[2f0, 3f0, 0.5f0, 10f0, 1f0])
        y = Mantle.devicearray(TESTBACKEND, Float32[3f0, 0.5f0, -2f0, 0f0, 100f0])
        out = Mantle.devicearray(TESTBACKEND, zeros(Float32, 5))
        pow_kernel!(TESTBACKEND)(out, x, y; ndrange=5)
        KA.synchronize(TESTBACKEND)
        cpu = Float32[2f0^3f0, 3f0^0.5f0, 0.5f0^-2f0, 10f0^0f0, 1f0^100f0]
        @test Array(out) ≈ cpu rtol=1f-5

        # Float64, where the device has it. On Vulkan this routes through Lava's
        # downcast overlay (`^(::Float64, ::Float64) = Float64(Float32(x) ^
        # Float32(y))`), because GLSL.std.450 Pow does not support f64, which is
        # why the tolerance is single precision.
        for T in testeltypes(Float64)
            xd = Mantle.devicearray(TESTBACKEND, T[2.0, 3.0, 0.5])
            yd = Mantle.devicearray(TESTBACKEND, T[3.0, 0.5, -2.0])
            outd = Mantle.devicearray(TESTBACKEND, zeros(T, 3))
            pow_kernel!(TESTBACKEND)(outd, xd, yd; ndrange=3)
            KA.synchronize(TESTBACKEND)
            @test Array(outd) ≈ [8.0, sqrt(3.0), 4.0] rtol=1e-5
        end

        # Negative base with integer exponent — must take the
        # `power_by_squaring` path, NOT llvm.pow (GLSL Pow is spec-undefined
        # for x<0). Julia's `Base.:(^)(::Float32, ::Integer)` handles this
        # correctly; the overlay does not touch the Integer method.
        @kernel function pow_int_kernel!(out, x, y::Int32)
            i = @index(Global, Linear)
            @inbounds out[i] = x[i] ^ y
        end
        xn = Mantle.devicearray(TESTBACKEND, Float32[-2f0, -3f0, 2f0, 0f0, -1f0])
        on = Mantle.devicearray(TESTBACKEND, zeros(Float32, 5))
        pow_int_kernel!(TESTBACKEND)(on, xn, Int32(3); ndrange=5)
        KA.synchronize(TESTBACKEND)
        @test Array(on) == Float32[-8, -27, 8, 0, -1]

        # Even integer exponent — sign must flip back
        oe = Mantle.devicearray(TESTBACKEND, zeros(Float32, 5))
        pow_int_kernel!(TESTBACKEND)(oe, xn, Int32(4); ndrange=5)
        KA.synchronize(TESTBACKEND)
        @test Array(oe) == Float32[16, 81, 16, 0, 1]

        # `@fastmath ^` on Julia 1.12 rewrites to a fast-math `Base.:(^)`
        # call, which hits the same overlay. Compiles AND gives numeric result.
        @kernel function fastmath_pow!(out, x, y)
            i = @index(Global, Linear)
            @inbounds out[i] = @fastmath x[i] ^ y[i]
        end
        xf = Mantle.devicearray(TESTBACKEND, Float32[2f0, 3f0])
        yf = Mantle.devicearray(TESTBACKEND, Float32[3f0, 2f0])
        of = Mantle.devicearray(TESTBACKEND, zeros(Float32, 2))
        fastmath_pow!(TESTBACKEND)(of, xf, yf; ndrange=2)
        KA.synchronize(TESTBACKEND)
        @test Array(of) ≈ Float32[8f0, 9f0] rtol=1f-5
    end

    @testset "round halfway (round-to-nearest-even)" begin
        # Julia's `round(::Float32)` lowers to `llvm.rint` (round-half-to-even).
        # GLSL.std.450 opcode 1 (`Round`) has implementation-defined halfway
        # behavior per SPIR-V spec; opcode 2 (`RoundEven`) is the IEEE-correct
        # round-to-nearest-even. Mapping `llvm.rint → RoundEven` keeps GPU
        # parity with Julia CPU across drivers.
        @kernel function round_kernel!(out, xs)
            i = @index(Global, Linear)
            @inbounds out[i] = round(xs[i])
        end

        xs = Mantle.devicearray(TESTBACKEND, Float32[0.5, 1.5, 2.5, 3.5, -0.5, -1.5, -2.5, 4.5])
        out = Mantle.devicearray(TESTBACKEND, zeros(Float32, 8))
        round_kernel!(TESTBACKEND)(out, xs; ndrange=8)
        KA.synchronize(TESTBACKEND)
        gpu = Array(out)
        cpu = Float32[round(x) for x in Float32[0.5, 1.5, 2.5, 3.5, -0.5, -1.5, -2.5, 4.5]]
        # Expected (round-to-even): 0, 2, 2, 4, -0, -2, -2, 4
        @test gpu == cpu
    end
end


