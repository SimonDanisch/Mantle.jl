# Tier 1: Math intrinsic emission tests
# Tests GLSL.std.450 mapping, no f64 leakage for f32 math

using Test
using Lava, Mantle
using KernelAbstractions
using KernelAbstractions: @kernel, @index
if !@isdefined(SPIRVTestUtils)
    include(joinpath(@__DIR__, "..", "spirv_test_utils.jl"))
end
import .SPIRVTestUtils: check, check_not, check_dag, check_sequence, check_count, check_regex, normalize_spirv, compare_golden, compile_and_disasm, spirv_opt_roundtrip, check_vendor_safety, compile_with_llc

@testset "Math Intrinsics" begin

    # GLSL.std.450 extended instruction mappings for f32
    f32_math = [
        (sin,   "Sin"),
        (cos,   "Cos"),
        (sqrt,  "Sqrt"),
        (abs,   "FAbs"),
        (floor, "Floor"),
        (ceil,  "Ceil"),
        (exp,   "Exp"),
        (log,   "Log"),
        (exp2,  "Exp2"),
        (log2,  "Log2"),
    ]

    @testset "f32 $jl_name → GLSL.std.450 $spirv_name" for (jl_name, spirv_name) in f32_math
        fn = let f = jl_name
            function(A, B)
                i = Lava.lava_global_invocation_id_x()
                @inbounds B[i] = f(A[i])
                return nothing
            end
        end
        d, _ = compile_and_disasm(fn, Tuple{Lava.LavaDeviceArray{Float32,1},
                                             Lava.LavaDeviceArray{Float32,1}})
        # GLSL.std.450 import and OpExtInst are on separate lines
        check(d, "GLSL.std.450")
        check_regex(d, "OpExtInst.*$spirv_name")
        check_not(d, "OpTypeFloat 64")
    end

    @testset "min/max" begin
        function minmax(A, B, C)
            i = Lava.lava_global_invocation_id_x()
            @inbounds C[i] = min(A[i], B[i])
            return nothing
        end
        d, _ = compile_and_disasm(minmax, Tuple{Lava.LavaDeviceArray{Float32,1},
                                                 Lava.LavaDeviceArray{Float32,1},
                                                 Lava.LavaDeviceArray{Float32,1}})
        # min should map to FMin or NMin
        @test occursin("FMin", d) || occursin("NMin", d)
    end

    @testset "clamp" begin
        function clamp_kernel(A, B)
            i = Lava.lava_global_invocation_id_x()
            @inbounds B[i] = clamp(A[i], 0.0f0, 1.0f0)
            return nothing
        end
        d, _ = compile_and_disasm(clamp_kernel, Tuple{Lava.LavaDeviceArray{Float32,1},
                                                       Lava.LavaDeviceArray{Float32,1}})
        @test occursin("FClamp", d) || occursin("NClamp", d) ||
              (occursin("FMax", d) && occursin("FMin", d)) ||
              (occursin("NMax", d) && occursin("NMin", d))
    end

    @testset "sign" begin
        function sign_kernel(A, B)
            i = Lava.lava_global_invocation_id_x()
            @inbounds B[i] = sign(A[i])
            return nothing
        end
        d, _ = compile_and_disasm(sign_kernel, Tuple{Lava.LavaDeviceArray{Float32,1},
                                                      Lava.LavaDeviceArray{Float32,1}})
        @test occursin("FSign", d) || occursin("OpFOrdGreaterThan", d) || occursin("OpSelect", d)
    end

    @testset "copysign zero-preserving (bitwise sign-copy)" begin
        # copysign(x, 0) must return +|x|, not 0. GLSL.std.450 FSign(0) is 0,
        # so `FAbs(x) * FSign(y)` produces 0 for any y==0 — breaks ray-tracing
        # `safe_invdir` clamps and similar IEEE-dependent code.
        #
        # The emitter errors on any `llvm.copysign` that slips past the
        # `Base.copysign` overlay in `device/math.jl`. Each kernel below
        # compiles (so no leak to the emitter) AND returns the IEEE result.

        @kernel function copysign_kernel!(out, x_arr, y_arr)
            i = @index(Global, Linear)
            @inbounds out[i] = copysign(x_arr[i], y_arr[i])
        end

        x = Float32[1f-5,  1f-5,  1f-5, -3f0, 2f0, 2f0]
        y = Float32[ 0f0, -0f0, -1f0,  2f0, Inf32, -Inf32]
        expected = Float32[1f-5, -1f-5, -1f-5, 3f0, 2f0, -2f0]

        x_arr = Mantle.LavaArray(x)
        y_arr = Mantle.LavaArray(y)
        out = Mantle.LavaArray(zeros(Float32, length(x)))
        copysign_kernel!(Mantle.LavaBackend())(out, x_arr, y_arr; ndrange=length(x))
        Mantle.vk_flush!(Mantle.vk_context())
        @test reinterpret(UInt32, Array(out)) == reinterpret(UInt32, expected)

        # Float64
        xd = Float64[1e-10,  1e-10,  1e-10, -3.0]
        yd = Float64[  0.0,   -0.0,   -1.0,  2.0]
        expected_d = Float64[1e-10, -1e-10, -1e-10, 3.0]
        x_arr64 = Mantle.LavaArray(xd); y_arr64 = Mantle.LavaArray(yd)
        out64 = Mantle.LavaArray(zeros(Float64, length(xd)))
        copysign_kernel!(Mantle.LavaBackend())(out64, x_arr64, y_arr64; ndrange=length(xd))
        Mantle.vk_flush!(Mantle.vk_context())
        @test reinterpret(UInt64, Array(out64)) == reinterpret(UInt64, expected_d)
    end

    @testset "copysign leak paths (must not reach emitter)" begin
        # Exercises the ways `llvm.copysign` could slip past the Base.copysign
        # overlay. Each kernel here will FAIL TO COMPILE if it does, because
        # the emitter errors on `llvm.copysign`. Numeric checks additionally
        # confirm correctness.

        # 1. @fastmath: fastmath rewrites `copysign` but still hits Base.copysign.
        @kernel function fastmath_cs!(out, x, y)
            i = @index(Global, Linear)
            @inbounds out[i] = @fastmath copysign(x[i], y[i])
        end
        x = Mantle.LavaArray(Float32[1f-5, 2f0])
        y = Mantle.LavaArray(Float32[0f0, -1f0])
        out = Mantle.LavaArray(zeros(Float32, 2))
        fastmath_cs!(Mantle.LavaBackend())(out, x, y; ndrange=2)
        Mantle.vk_flush!(Mantle.vk_context())
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
        xs = Mantle.LavaArray(fill(1f-5, 16, 8))
        ys = Mantle.LavaArray(zeros(Float32, 16, 8))  # all zero y's — the bug case
        lout = Mantle.LavaArray(zeros(Float32, 16))
        loop_cs!(Mantle.LavaBackend())(lout, xs, ys, Int32(8); ndrange=16)
        Mantle.vk_flush!(Mantle.vk_context())
        @test all(Array(lout) .≈ 8f0 * 1f-5)  # would be 0 if old bug reappeared

        # 3. Exact `safe_invdir` pattern — the original bug trigger.
        @kernel function safe_invdir_kernel!(out, d)
            i = @index(Global, Linear)
            ooeps = 1f-5
            @inbounds dv = d[i]
            clamped = abs(dv) > ooeps ? dv : copysign(ooeps, dv)
            @inbounds out[i] = 1f0 / clamped
        end
        d = Mantle.LavaArray(Float32[0f0, -0f0, 1f0, -1f0])
        sout = Mantle.LavaArray(zeros(Float32, 4))
        safe_invdir_kernel!(Mantle.LavaBackend())(sout, d; ndrange=4)
        Mantle.vk_flush!(Mantle.vk_context())
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
        # CUDA/ROCm/SPIRVIntrinsics. The emitter errors on any leaked
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
        xs = Mantle.LavaArray(Float32[1f0, NaN32, NaN32, 3f0, -1f0])
        ys = Mantle.LavaArray(Float32[NaN32, 2f0, NaN32, 5f0, -2f0])
        mi = Mantle.LavaArray(zeros(Float32, 5))
        ma = Mantle.LavaArray(zeros(Float32, 5))
        min_kernel!(Mantle.LavaBackend())(mi, xs, ys; ndrange=5)
        max_kernel!(Mantle.LavaBackend())(ma, xs, ys; ndrange=5)
        Mantle.vk_flush!(Mantle.vk_context())
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
        fout = Mantle.LavaArray(zeros(Float32, 5))
        fastmath_min!(Mantle.LavaBackend())(fout, xs, ys; ndrange=5)
        Mantle.vk_flush!(Mantle.vk_context())
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
        a = Mantle.LavaArray(UInt32[10, 5, 0, typemax(UInt32), 3])
        b = Mantle.LavaArray(UInt32[3, 7, 1, 1, typemax(UInt32)])
        out = Mantle.LavaArray(zeros(UInt32, 5))
        usub_sat_kernel!(Mantle.LavaBackend())(out, a, b; ndrange=5)
        Mantle.vk_flush!(Mantle.vk_context())
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
        a = Mantle.LavaArray(UInt32[10, typemax(UInt32), typemax(UInt32) - 5, 0, 1])
        b = Mantle.LavaArray(UInt32[3, 1, 10, 0, typemax(UInt32)])
        out = Mantle.LavaArray(zeros(UInt32, 5))
        uadd_sat_kernel!(Mantle.LavaBackend())(out, a, b; ndrange=5)
        Mantle.vk_flush!(Mantle.vk_context())
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
        a = Mantle.LavaArray(Int32[10, typemax(Int32), typemin(Int32), -5, 100])
        b = Mantle.LavaArray(Int32[3, 1, -1, -typemax(Int32), -50])
        out = Mantle.LavaArray(zeros(Int32, 5))
        sadd_sat_kernel!(Mantle.LavaBackend())(out, a, b; ndrange=5)
        Mantle.vk_flush!(Mantle.vk_context())
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
        a = Mantle.LavaArray(Int32[10, typemax(Int32), typemin(Int32), 5, -100])
        b = Mantle.LavaArray(Int32[3, -1, 1, -5, 50])
        out = Mantle.LavaArray(zeros(Int32, 5))
        ssub_sat_kernel!(Mantle.LavaBackend())(out, a, b; ndrange=5)
        Mantle.vk_flush!(Mantle.vk_context())
        @test Array(out) == Int32[7, typemax(Int32), typemin(Int32), 10, -150]

        # End-to-end: Base.rem on Float32 (the use case that triggered this).
        @kernel function rem_kernel!(out, x, y)
            i = @index(Global, Linear)
            @inbounds out[i] = rem(x[i], y[i])
        end
        rx = Mantle.LavaArray(Float32[7f0, -7f0, 5.5f0, 100f0])
        ry = Mantle.LavaArray(Float32[3f0, 3f0, 2.0f0, 0.3f0])
        rout = Mantle.LavaArray(zeros(Float32, 4))
        rem_kernel!(Mantle.LavaBackend())(rout, rx, ry; ndrange=4)
        Mantle.vk_flush!(Mantle.vk_context())
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
        x = Mantle.LavaArray(Float32[2f0, 3f0, 0.5f0, 10f0, 1f0])
        y = Mantle.LavaArray(Float32[3f0, 0.5f0, -2f0, 0f0, 100f0])
        out = Mantle.LavaArray(zeros(Float32, 5))
        pow_kernel!(Mantle.LavaBackend())(out, x, y; ndrange=5)
        Mantle.vk_flush!(Mantle.vk_context())
        cpu = Float32[2f0^3f0, 3f0^0.5f0, 0.5f0^-2f0, 10f0^0f0, 1f0^100f0]
        @test Array(out) ≈ cpu rtol=1f-5

        # Float64 — routes through the downcast overlay at math.jl:179
        # (`^(::Float64, ::Float64) = Float64(Float32(x) ^ Float32(y))`),
        # which is required because GLSL.std.450 Pow does not support f64.
        xd = Mantle.LavaArray(Float64[2.0, 3.0, 0.5])
        yd = Mantle.LavaArray(Float64[3.0, 0.5, -2.0])
        outd = Mantle.LavaArray(zeros(Float64, 3))
        pow_kernel!(Mantle.LavaBackend())(outd, xd, yd; ndrange=3)
        Mantle.vk_flush!(Mantle.vk_context())
        @test Array(outd) ≈ [8.0, sqrt(3.0), 4.0] rtol=1e-5

        # Negative base with integer exponent — must take the
        # `power_by_squaring` path, NOT llvm.pow (GLSL Pow is spec-undefined
        # for x<0). Julia's `Base.:(^)(::Float32, ::Integer)` handles this
        # correctly; the overlay does not touch the Integer method.
        @kernel function pow_int_kernel!(out, x, y::Int32)
            i = @index(Global, Linear)
            @inbounds out[i] = x[i] ^ y
        end
        xn = Mantle.LavaArray(Float32[-2f0, -3f0, 2f0, 0f0, -1f0])
        on = Mantle.LavaArray(zeros(Float32, 5))
        pow_int_kernel!(Mantle.LavaBackend())(on, xn, Int32(3); ndrange=5)
        Mantle.vk_flush!(Mantle.vk_context())
        @test Array(on) == Float32[-8, -27, 8, 0, -1]

        # Even integer exponent — sign must flip back
        oe = Mantle.LavaArray(zeros(Float32, 5))
        pow_int_kernel!(Mantle.LavaBackend())(oe, xn, Int32(4); ndrange=5)
        Mantle.vk_flush!(Mantle.vk_context())
        @test Array(oe) == Float32[16, 81, 16, 0, 1]

        # `@fastmath ^` on Julia 1.12 rewrites to a fast-math `Base.:(^)`
        # call, which hits the same overlay. Compiles AND gives numeric result.
        @kernel function fastmath_pow!(out, x, y)
            i = @index(Global, Linear)
            @inbounds out[i] = @fastmath x[i] ^ y[i]
        end
        xf = Mantle.LavaArray(Float32[2f0, 3f0])
        yf = Mantle.LavaArray(Float32[3f0, 2f0])
        of = Mantle.LavaArray(zeros(Float32, 2))
        fastmath_pow!(Mantle.LavaBackend())(of, xf, yf; ndrange=2)
        Mantle.vk_flush!(Mantle.vk_context())
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

        xs = Mantle.LavaArray(Float32[0.5, 1.5, 2.5, 3.5, -0.5, -1.5, -2.5, 4.5])
        out = Mantle.LavaArray(zeros(Float32, 8))
        round_kernel!(Mantle.LavaBackend())(out, xs; ndrange=8)
        Mantle.vk_flush!(Mantle.vk_context())
        gpu = Array(out)
        cpu = Float32[round(x) for x in Float32[0.5, 1.5, 2.5, 3.5, -0.5, -1.5, -2.5, 4.5]]
        # Expected (round-to-even): 0, 2, 2, 4, -0, -2, -2, 4
        @test gpu == cpu
    end
end


