# test_spirv_pattern_correctness.jl
#
# Each testset pins a kernel pattern that the Vulkan compiler (Lava's SPIR-V
# emitter) has historically miscompiled on one or more drivers. The TEST is "this
# source pattern runs correctly" — vendor-neutral and backend-neutral. The comment
# block above each testset attributes which driver first surfaced the issue
# (NVIDIA, RADV, lavapipe, AMD Windows, …) and what the emitter got wrong, as
# historical context only; the assertion is the same on every platform. If any
# test fails on a new vendor or backend, that is a real correctness bug in a
# compiler or a driver — investigate it the same way.

using Test
using Mantle
using KernelAbstractions
using Atomix
include(joinpath(@__DIR__, "testbackend.jl"))
const KA = KernelAbstractions

# ═══════════════════════════════════════════════════════════════════════
# Fix 1: OpBitcast on PSB pointers crashes NVIDIA RT shader compiler
#
# Symptom: Segfault in libnvidia-glvkspirv.so during vkCreateRayTracingPipelinesKHR
# Fix: emit_psb_ptr_reinterpret!() uses ConvertPtrToU+ConvertUToPtr instead of OpBitcast
#
# This scanned the module for an `OpBitcast` on a PhysicalStorageBuffer pointer,
# which is the emitter's business. The kernel that forced the reinterpretation
# runs here instead, and has to read both struct types right.
# ═══════════════════════════════════════════════════════════════════════

@testset "Fix 1: two struct types read through device pointers" begin
    # A kernel that reads two different struct types from device memory —
    # forces pointer reinterpretation between the two pointer types
    struct Fix1_A
        x::Float32
        y::Float32
    end
    struct Fix1_B
        a::Float32
        b::Float32
        c::Float32
    end

    @kernel function fix1_kernel!(out, @Const(as), @Const(bs))
        i = @index(Global)
        @inbounds out[i] = as[i].x + bs[i].a
    end

    N = 256
    ah = [Fix1_A(Float32(i), -1f0) for i in 1:N]
    bh = [Fix1_B(Float32(10i), -2f0, -3f0) for i in 1:N]
    out = Mantle.devicearray(TESTBACKEND, zeros(Float32, N))
    fix1_kernel!(TESTBACKEND, 64)(out, Mantle.devicearray(TESTBACKEND, ah),
                                  Mantle.devicearray(TESTBACKEND, bh); ndrange=N)
    KA.synchronize(TESTBACKEND)
    @test Array(out) == [ah[i].x + bh[i].a for i in 1:N]
end

# ═══════════════════════════════════════════════════════════════════════
# Fix 3: Non-aligned byte GEP truncation on PSB pointers
#
# Symptom: byte_offset=21 → 21/4=5 (truncated) → byte 20 instead of 21
# Fix: Integer arithmetic (ConvertPtrToU + IAdd + ConvertUToPtr) for non-divisible offsets
# ═══════════════════════════════════════════════════════════════════════

@testset "Fix 3: Non-aligned byte offsets in structs" begin
    # 13-byte struct: last field at byte 9 (not divisible by 4)
    struct Fix3_Struct
        a::Float32   # bytes 0-3
        b::Float32   # bytes 4-7
        c::UInt8     # byte 8
        d::Float32   # bytes 9-12 (not 4-aligned in memory!)
    end

    @kernel function fix3_kernel!(dst, src)
        i = @index(Global)
        @inbounds dst[i] = src[i]
    end

    N = 256
    data = [Fix3_Struct(Float32(i), Float32(i*2), UInt8(i % 128), Float32(i*3)) for i in 1:N]
    src = Mantle.devicearray(TESTBACKEND, data)
    dst = Mantle.devicearray(TESTBACKEND, Fix3_Struct, N)
    fix3_kernel!(TESTBACKEND)(dst, src; ndrange=N)
    KA.synchronize(TESTBACKEND)
    result = Array(dst)
    @test result == data
end

# ═══════════════════════════════════════════════════════════════════════
# Fix 4/5/8/9: Misaligned i64 PSB loads/stores
#
# Symptom: DEVICE_LOST or wrong values when LLVM packs i32 pairs into i64
# at non-8-aligned offsets. NVIDIA silently rounds down to 8-byte boundary.
#
# Fix 4: Constant offset misalignment (e.g., BVHNode2 at byte 12)
# Fix 5: Non-8-aligned stride (e.g., 60-byte struct, 60%8=4)
# Fix 8: ptrtoint+add+inttoptr patterns at non-aligned offsets
# Fix 9: Universal check via LLVM.alignment(inst) < type_align
# ═══════════════════════════════════════════════════════════════════════

@testset "Fix 4/5/8/9: Misaligned i64 PSB loads" begin

    @testset "60-byte struct (stride%8=4)" begin
        # 60 bytes = 15 Float32 fields → stride alignment is 4, not 8
        # LLVM may pack adjacent i32 into i64 loads, which NVIDIA rounds down
        struct Struct60
            f1::Float32;  f2::Float32;  f3::Float32;  f4::Float32;  f5::Float32
            f6::Float32;  f7::Float32;  f8::Float32;  f9::Float32;  f10::Float32
            f11::Float32; f12::Float32; f13::Float32; f14::Float32; f15::Float32
        end
        @assert sizeof(Struct60) == 60

        @kernel function copy60!(dst, src)
            i = @index(Global)
            @inbounds dst[i] = src[i]
        end

        N = 128
        data = [Struct60(ntuple(j -> Float32(i * 100 + j), 15)...) for i in 1:N]
        src = Mantle.devicearray(TESTBACKEND, data)
        dst = Mantle.devicearray(TESTBACKEND, Struct60, N)
        copy60!(TESTBACKEND)(dst, src; ndrange=N)
        KA.synchronize(TESTBACKEND)
        result = Array(dst)
        @test result == data
    end

    @testset "52-byte struct (stride%8=4)" begin
        struct Struct52
            f1::Float32;  f2::Float32;  f3::Float32;  f4::Float32
            f5::Float32;  f6::Float32;  f7::Float32;  f8::Float32
            f9::Float32;  f10::Float32; f11::Float32; f12::Float32
            f13::Float32
        end
        @assert sizeof(Struct52) == 52

        @kernel function copy52!(dst, src)
            i = @index(Global)
            @inbounds dst[i] = src[i]
        end

        N = 128
        data = [Struct52(ntuple(j -> Float32(i * 100 + j), 13)...) for i in 1:N]
        src = Mantle.devicearray(TESTBACKEND, data)
        dst = Mantle.devicearray(TESTBACKEND, Struct52, N)
        copy52!(TESTBACKEND)(dst, src; ndrange=N)
        KA.synchronize(TESTBACKEND)
        result = Array(dst)
        @test result == data
    end

    @testset "mixed-alignment struct (i64 at non-8-aligned offset)" begin
        # Field layout forces i64 (or i32 pair packed as i64) at odd offset
        struct MixedAlign
            a::Float32   # 0
            b::Float32   # 4
            c::Float32   # 8
            d::UInt32    # 12 — next pair (d,e) may be packed as i64 at offset 12
            e::UInt32    # 16
            f::Float32   # 20
        end

        @kernel function copy_mixed!(dst, src)
            i = @index(Global)
            @inbounds dst[i] = src[i]
        end

        N = 256
        data = [MixedAlign(Float32(i), Float32(2i), Float32(3i),
                           UInt32(i), UInt32(i+1), Float32(4i)) for i in 1:N]
        src = Mantle.devicearray(TESTBACKEND, data)
        dst = Mantle.devicearray(TESTBACKEND, MixedAlign, N)
        copy_mixed!(TESTBACKEND)(dst, src; ndrange=N)
        KA.synchronize(TESTBACKEND)
        result = Array(dst)
        @test result == data
    end

    @testset "struct with Bool/UInt8 creating odd offsets" begin
        struct OddOffsets
            flag::UInt8   # byte 0
            x::Float32    # byte 1 (or padded to 4?)
            y::Float32    # byte 5 (or padded to 8?)
            z::Float32    # byte 9 (or padded to 12?)
        end

        @kernel function copy_odd!(dst, src)
            i = @index(Global)
            @inbounds dst[i] = src[i]
        end

        N = 128
        data = [OddOffsets(UInt8(i % 256), Float32(i), Float32(2i), Float32(3i)) for i in 1:N]
        src = Mantle.devicearray(TESTBACKEND, data)
        dst = Mantle.devicearray(TESTBACKEND, OddOffsets, N)
        copy_odd!(TESTBACKEND)(dst, src; ndrange=N)
        KA.synchronize(TESTBACKEND)
        result = Array(dst)
        @test result == data
    end
end

# ═══════════════════════════════════════════════════════════════════════
# Fix 6: LLVM/Julia struct size mismatch in BDA arg buffer packing
#
# Symptom: broadcast of structs produces corrupted data because LLVM struct
# is larger than Julia sizeof (Nothing fields, type parameters)
# Fix: Use LLVM byval sizes for arg buffer packing
# (Detailed tests in test_struct_broadcast.jl — this is a quick smoke test)
# ═══════════════════════════════════════════════════════════════════════

@testset "Fix 6: Struct size mismatch smoke test" begin
    struct Fix6_S
        x::Float32
        y::Float32
        z::Float32
    end

    # The critical n=10 case where CompilerMetadata LLVM=24 vs Julia=16
    n = 10
    src = Mantle.devicearray(TESTBACKEND, [Fix6_S(Float32(i), Float32(2i), Float32(3i)) for i in 1:n])
    dst = Mantle.devicearray(TESTBACKEND, Fix6_S, n)
    dst .= src
    KA.synchronize(TESTBACKEND)
    @test Array(dst) == Array(src)
end

# ═══════════════════════════════════════════════════════════════════════
# Fix 7: PHI cycle detection infinite loop
#
# Symptom: trace_to_non_alloca() mishandled PHI cycles, returning wrong
# storage class. GPU crash on multi-material scenes.
# Fix: Return true (=PSB) for cycles, since PHI cycles only occur with PSB pointers
# ═══════════════════════════════════════════════════════════════════════

@testset "Fix 7: PHI cycles in complex control flow" begin
    # Simulate the pattern: loop with conditional pointer selection (PHI of pointers)
    @kernel function phi_cycle_kernel!(output, a, b, selector)
        i = @index(Global)
        @inbounds begin
            val = 0.0f0
            for j in Int32(1):Int32(4)
                src = selector[i] > 0.5f0 ? a[i] : b[i]
                val += src * Float32(j)
            end
            output[i] = val
        end
    end

    N = 256
    a = Mantle.devicearray(TESTBACKEND, ones(Float32, N) .* 2.0f0)
    b = Mantle.devicearray(TESTBACKEND, ones(Float32, N) .* 3.0f0)
    sel = Mantle.devicearray(TESTBACKEND, rand(Float32, N))
    output = Mantle.devicearray(TESTBACKEND, zeros(Float32, N))

    phi_cycle_kernel!(TESTBACKEND)(output, a, b, sel; ndrange=N)
    KA.synchronize(TESTBACKEND)
    result = Array(output)
    sel_h = Array(sel)
    for i in 1:N
        v = sel_h[i] > 0.5f0 ? 2.0f0 : 3.0f0
        expected = v * (1 + 2 + 3 + 4)
        @test result[i] ≈ expected
    end
end

# ═══════════════════════════════════════════════════════════════════════
# Zero-alloc push constants
#
# Symptom: Indirect dispatch overwrites shared Vector{UInt8} push data
# Fix: Pass UInt64 BDA through call chain; use module-level Ref for ccall
#
# The Vulkan launch path's history; on any backend the property is the same:
# two kernels launched alternately with different scalar arguments each see
# their own.
# ═══════════════════════════════════════════════════════════════════════

@testset "Zero-alloc push constants: no corruption under rapid dispatch" begin
    # Rapidly alternate between two kernels to stress push constant path
    @kernel function fill_val!(A, val::Float32)
        i = @index(Global)
        @inbounds A[i] = val
    end

    N = 1024
    a = Mantle.devicearray(TESTBACKEND, zeros(Float32, N))
    b = Mantle.devicearray(TESTBACKEND, zeros(Float32, N))

    # 20 rapid dispatches alternating targets
    for _ in 1:10
        fill_val!(TESTBACKEND)(a, 42.0f0; ndrange=N)
        fill_val!(TESTBACKEND)(b, 99.0f0; ndrange=N)
    end
    KA.synchronize(TESTBACKEND)

    @test all(Array(a) .== 42.0f0)
    @test all(Array(b) .== 99.0f0)
end

# ═══════════════════════════════════════════════════════════════════════
# Piggybacked download (_append_copy_and_flush!)
#
# Optimization: GPU→staging copy appended to active command batch,
# avoiding a second fence roundtrip. That is the Vulkan backend's download; the
# property on any backend is that a download right after a dispatch, with no
# synchronize between, sees what the dispatch wrote.
# ═══════════════════════════════════════════════════════════════════════

@testset "Piggybacked download: correct data after dispatch+copy" begin
    @kernel function iota!(A)
        i = @index(Global)
        @inbounds A[i] = Float32(i)
    end

    N = 512
    a = Mantle.devicearray(TESTBACKEND, zeros(Float32, N))

    # Dispatch a kernel, then download — should piggyback copy onto batch
    iota!(TESTBACKEND)(a; ndrange=N)
    # No synchronize: the download itself has to order after the dispatch
    result = Array(a)

    @test result[1] == 1.0f0
    @test result[N] == Float32(N)
    @test result == Float32.(1:N)
end

# ═══════════════════════════════════════════════════════════════════════
# Stress Tests
# ═══════════════════════════════════════════════════════════════════════

@testset "Stress: many small dispatches (100)" begin
    a = Mantle.devicearray(TESTBACKEND, ones(Float32, 64))
    for _ in 1:100
        a = a .+ 1.0f0
    end
    KA.synchronize(TESTBACKEND)
    @test all(Array(a) .== 101.0f0)
end

@testset "Stress: large array operations" begin
    N = 4_000_000
    a = Mantle.devicearray(TESTBACKEND, ones(Float32, N))
    b = Mantle.devicearray(TESTBACKEND, fill(2.0f0, N))
    c = a .+ b .* 3.0f0
    KA.synchronize(TESTBACKEND)
    @test Array(c)[1] == 7.0f0
    @test Array(c)[N] == 7.0f0
end

@testset "Stress: GC pressure during recording" begin
    # Allocate and discard many arrays during a recording batch
    # to exercise deferred free + data_refs lifetime tracking
    result = Mantle.devicearray(TESTBACKEND, zeros(Float32, 256))
    for i in 1:50
        tmp = Mantle.devicearray(TESTBACKEND, fill(Float32(i), 256))
        result = result .+ tmp
        # Let tmp go out of scope — GC may try to free it
    end
    GC.gc()  # Force GC while batch is still recording
    KA.synchronize(TESTBACKEND)
    r = Array(result)
    # Sum of 1..50 = 1275
    @test r[1] ≈ 1275.0f0
end

@testset "Stress: rapid alloc/free/dispatch cycle" begin
    for i in 1:20
        a = Mantle.devicearray(TESTBACKEND, fill(Float32(i), 1024))
        b = a .* 2.0f0
        KA.synchronize(TESTBACKEND)
        @test Array(b)[1] == Float32(2i)
        # a and b go out of scope each iteration
    end
    GC.gc()
end

@testset "Stress: 2D dispatch" begin
    @kernel function fill_2d!(A)
        i, j = @index(Global, NTuple)
        @inbounds A[i, j] = Float32(i * 100 + j)
    end

    M, N = 128, 64
    a = Mantle.devicearray(TESTBACKEND, zeros(Float32, M, N))
    fill_2d!(TESTBACKEND)(a; ndrange=(M, N))
    KA.synchronize(TESTBACKEND)
    result = Array(a)
    @test result[1, 1] == 101.0f0
    @test result[M, N] == Float32(M * 100 + N)
end

@testset "Stress: reduction correctness" begin
    for N in [7, 63, 127, 255, 1023, 4096, 100_000]
        a = Mantle.devicearray(TESTBACKEND, ones(Float32, N))
        s = sum(a)
        @test s ≈ Float32(N) atol=max(1.0f0, Float32(N) * 1f-5)
    end
end

@testset "Stress: mixed types" begin
    # Int32
    a = Mantle.devicearray(TESTBACKEND, Int32[1, 2, 3, 4])
    b = a .+ Int32(10)
    KA.synchronize(TESTBACKEND)
    @test Array(b) == Int32[11, 12, 13, 14]

    # UInt32
    a = Mantle.devicearray(TESTBACKEND, UInt32[10, 20, 30, 40])
    b = a .- UInt32(5)
    KA.synchronize(TESTBACKEND)
    @test Array(b) == UInt32[5, 15, 25, 35]

    # Float64, where the device has it
    for T in testeltypes(Float64)
        a = Mantle.devicearray(TESTBACKEND, T[1.0, 2.0, 3.0])
        b = a .* T(2.0)
        KA.synchronize(TESTBACKEND)
        @test Array(b) ≈ T[2.0, 4.0, 6.0]
    end
end

@testset "Stress: struct array of arrays pattern" begin
    # Nested struct with NTuple (fixed-size array) — common in Hikari
    struct SpectrumData
        data::NTuple{4, Float32}
    end

    @kernel function spectrum_add!(dst, a, b)
        i = @index(Global)
        @inbounds begin
            av = a[i]
            bv = b[i]
            dst[i] = SpectrumData(ntuple(j -> av.data[j] + bv.data[j], Val(4)))
        end
    end

    N = 256
    a_data = [SpectrumData(ntuple(j -> Float32(i * 10 + j), 4)) for i in 1:N]
    b_data = [SpectrumData(ntuple(j -> Float32(j), 4)) for i in 1:N]
    a = Mantle.devicearray(TESTBACKEND, a_data)
    b = Mantle.devicearray(TESTBACKEND, b_data)
    dst = Mantle.devicearray(TESTBACKEND, SpectrumData, N)

    spectrum_add!(TESTBACKEND)(dst, a, b; ndrange=N)
    KA.synchronize(TESTBACKEND)

    result = Array(dst)
    for i in 1:N
        for j in 1:4
            @test result[i].data[j] == Float32(i * 10 + j) + Float32(j)
        end
    end
end

# ═══════════════════════════════════════════════════════════════════════
# Fix 10: large dispatch counts submit correctly
#
# Symptom (historical): DEVICE_LOST at vkQueueSubmit when a single command
# buffer accumulated 30k+ dispatches (e.g. Hikari volpath 10spp: 50 bounces ×
# 60 dispatches/bounce × 10 samples = 30,000). NVIDIA driver's internal
# command buffer processing failed on very large CBs.
#
# There is no long-lived open command buffer to overflow on Vulkan now: every
# launch is its own one-shot, sealed and submitted immediately. How another
# backend groups launches into submissions is its own; this checks the property
# that survives on all of them — many dispatches in a row still produce correct
# results.
# ═══════════════════════════════════════════════════════════════════════

@testset "5000 dispatches in a row" begin
    @kernel function cb_split_inc!(a)
        i = @index(Global)
        @inbounds a[i] += 1.0f0
    end

    backend = TESTBACKEND
    a = Mantle.devicearray(TESTBACKEND, zeros(Float32, 256))
    kernel = cb_split_inc!(backend)
    for _ in 1:5000
        kernel(a; ndrange=256)
    end

    KA.synchronize(TESTBACKEND)
    @test Array(a) == fill(5000.0f0, 256)
end
