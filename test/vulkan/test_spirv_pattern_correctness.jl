# test_spirv_pattern_correctness.jl
#
# Each testset pins a SPIR-V emitter pattern that has historically miscompiled on
# one or more drivers. The TEST is "this source pattern produces correct SPIR-V
# and runs correctly" — vendor-neutral. The comment block above each testset
# attributes which driver first surfaced the issue (NVIDIA, RADV, lavapipe, AMD
# Windows, …) as historical context only; the assertion is the same on every
# platform. If any test fails on a new vendor, that is a real correctness bug
# either in Lava's emitter or in the driver — investigate it the same way.
#
# Run: julia --project=. -e 'include("test/test_spirv_pattern_correctness.jl")'
#
# These tests exercise GPU execution — they require a Vulkan device.

using Test
using Lava, Mantle
using KernelAbstractions
using Atomix

# ═══════════════════════════════════════════════════════════════════════
# Fix 1: OpBitcast on PSB pointers crashes NVIDIA RT shader compiler
#
# Symptom: Segfault in libnvidia-glvkspirv.so during vkCreateRayTracingPipelinesKHR
# Fix: emit_psb_ptr_reinterpret!() uses ConvertPtrToU+ConvertUToPtr instead of OpBitcast
# ═══════════════════════════════════════════════════════════════════════

@testset "Fix 1: No OpBitcast on PSB pointers (SPIR-V)" begin
    # Compile a kernel that reads two different struct types from BDA —
    # forces pointer reinterpretation between PSB pointer types
    struct Fix1_A
        x::Float32
        y::Float32
    end
    struct Fix1_B
        a::Float32
        b::Float32
        c::Float32
    end

    function fix1_kernel(as::Lava.LavaDeviceArray{Fix1_A,1},
                         bs::Lava.LavaDeviceArray{Fix1_B,1},
                         out::Lava.LavaDeviceArray{Float32,1})
        i = Lava.lava_global_invocation_id_x()
        @inbounds out[i] = as[i].x + bs[i].a
        return nothing
    end

    result = Lava.lava_compile_gpu(fix1_kernel,
        Tuple{Lava.LavaDeviceArray{Fix1_A,1}, Lava.LavaDeviceArray{Fix1_B,1},
              Lava.LavaDeviceArray{Float32,1}}; workgroup_size=(64,1,1), validate=true)
    d = Lava.disassemble_spirv(result.spirv_bytes)

    # Must NOT have OpBitcast on PhysicalStorageBuffer pointers
    for line in split(d, '\n')
        if occursin("OpBitcast", line) && occursin("PhysicalStorageBuffer", line)
            @test false  # OpBitcast on PSB pointer found — this crashes NVIDIA
        end
    end
    @test true  # passed if we get here
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
    src = Mantle.LavaArray(data)
    dst = Mantle.LavaArray{Fix3_Struct}(undef, N)
    fix3_kernel!(Mantle.LavaBackend())(dst, src; ndrange=N)
    Mantle.vk_flush!(Mantle.vk_context())
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
        src = Mantle.LavaArray(data)
        dst = Mantle.LavaArray{Struct60}(undef, N)
        copy60!(Mantle.LavaBackend())(dst, src; ndrange=N)
        Mantle.vk_flush!(Mantle.vk_context())
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
        src = Mantle.LavaArray(data)
        dst = Mantle.LavaArray{Struct52}(undef, N)
        copy52!(Mantle.LavaBackend())(dst, src; ndrange=N)
        Mantle.vk_flush!(Mantle.vk_context())
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
        src = Mantle.LavaArray(data)
        dst = Mantle.LavaArray{MixedAlign}(undef, N)
        copy_mixed!(Mantle.LavaBackend())(dst, src; ndrange=N)
        Mantle.vk_flush!(Mantle.vk_context())
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
        src = Mantle.LavaArray(data)
        dst = Mantle.LavaArray{OddOffsets}(undef, N)
        copy_odd!(Mantle.LavaBackend())(dst, src; ndrange=N)
        Mantle.vk_flush!(Mantle.vk_context())
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
    src = Mantle.LavaArray([Fix6_S(Float32(i), Float32(2i), Float32(3i)) for i in 1:n])
    dst = Mantle.LavaArray{Fix6_S}(undef, n)
    dst .= src
    Mantle.vk_flush!(Mantle.vk_context())
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
    a = Mantle.LavaArray(ones(Float32, N) .* 2.0f0)
    b = Mantle.LavaArray(ones(Float32, N) .* 3.0f0)
    sel = Mantle.LavaArray(rand(Float32, N))
    output = Mantle.LavaArray(zeros(Float32, N))

    phi_cycle_kernel!(Mantle.LavaBackend())(output, a, b, sel; ndrange=N)
    Mantle.vk_flush!(Mantle.vk_context())
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
# ═══════════════════════════════════════════════════════════════════════

@testset "Zero-alloc push constants: no corruption under rapid dispatch" begin
    # Rapidly alternate between two kernels to stress push constant path
    @kernel function fill_val!(A, val::Float32)
        i = @index(Global)
        @inbounds A[i] = val
    end

    N = 1024
    a = Mantle.LavaArray(zeros(Float32, N))
    b = Mantle.LavaArray(zeros(Float32, N))

    # 20 rapid dispatches alternating targets
    for _ in 1:10
        fill_val!(Mantle.LavaBackend())(a, 42.0f0; ndrange=N)
        fill_val!(Mantle.LavaBackend())(b, 99.0f0; ndrange=N)
    end
    Mantle.vk_flush!(Mantle.vk_context())

    @test all(Array(a) .== 42.0f0)
    @test all(Array(b) .== 99.0f0)
end

# ═══════════════════════════════════════════════════════════════════════
# Piggybacked download (_append_copy_and_flush!)
#
# Optimization: GPU→staging copy appended to active command batch,
# avoiding a second fence roundtrip
# ═══════════════════════════════════════════════════════════════════════

@testset "Piggybacked download: correct data after dispatch+copy" begin
    @kernel function iota!(A)
        i = @index(Global)
        @inbounds A[i] = Float32(i)
    end

    N = 512
    a = Mantle.LavaArray(zeros(Float32, N))

    # Dispatch a kernel, then download — should piggyback copy onto batch
    iota!(Mantle.LavaBackend())(a; ndrange=N)
    # The download triggers _append_copy_and_flush! if batch is active
    result = Array(a)

    @test result[1] == 1.0f0
    @test result[N] == Float32(N)
    @test result == Float32.(1:N)
end

# ═══════════════════════════════════════════════════════════════════════
# Stress Tests
# ═══════════════════════════════════════════════════════════════════════

@testset "Stress: many small dispatches (100)" begin
    a = Mantle.LavaArray(ones(Float32, 64))
    for _ in 1:100
        a = a .+ 1.0f0
    end
    Mantle.vk_flush!(Mantle.vk_context())
    @test all(Array(a) .== 101.0f0)
end

@testset "Stress: large array operations" begin
    N = 4_000_000
    a = Mantle.LavaArray(ones(Float32, N))
    b = Mantle.LavaArray(fill(2.0f0, N))
    c = a .+ b .* 3.0f0
    Mantle.vk_flush!(Mantle.vk_context())
    @test Array(c)[1] == 7.0f0
    @test Array(c)[N] == 7.0f0
end

@testset "Stress: GC pressure during recording" begin
    # Allocate and discard many arrays during a recording batch
    # to exercise deferred free + data_refs lifetime tracking
    result = Mantle.LavaArray(zeros(Float32, 256))
    for i in 1:50
        tmp = Mantle.LavaArray(fill(Float32(i), 256))
        result = result .+ tmp
        # Let tmp go out of scope — GC may try to free it
    end
    GC.gc()  # Force GC while batch is still recording
    Mantle.vk_flush!(Mantle.vk_context())
    r = Array(result)
    # Sum of 1..50 = 1275
    @test r[1] ≈ 1275.0f0
end

@testset "Stress: rapid alloc/free/dispatch cycle" begin
    for i in 1:20
        a = Mantle.LavaArray(fill(Float32(i), 1024))
        b = a .* 2.0f0
        Mantle.vk_flush!(Mantle.vk_context())
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
    a = Mantle.LavaArray(zeros(Float32, M, N))
    fill_2d!(Mantle.LavaBackend())(a; ndrange=(M, N))
    Mantle.vk_flush!(Mantle.vk_context())
    result = Array(a)
    @test result[1, 1] == 101.0f0
    @test result[M, N] == Float32(M * 100 + N)
end

@testset "Stress: reduction correctness" begin
    for N in [7, 63, 127, 255, 1023, 4096, 100_000]
        a = Mantle.LavaArray(ones(Float32, N))
        s = sum(a)
        @test s ≈ Float32(N) atol=max(1.0f0, Float32(N) * 1f-5)
    end
end

@testset "Stress: mixed types" begin
    # Int32
    a = Mantle.LavaArray(Int32[1, 2, 3, 4])
    b = a .+ Int32(10)
    Mantle.vk_flush!(Mantle.vk_context())
    @test Array(b) == Int32[11, 12, 13, 14]

    # UInt32
    a = Mantle.LavaArray(UInt32[10, 20, 30, 40])
    b = a .- UInt32(5)
    Mantle.vk_flush!(Mantle.vk_context())
    @test Array(b) == UInt32[5, 15, 25, 35]

    # Float64
    a = Mantle.LavaArray(Float64[1.0, 2.0, 3.0])
    b = a .* 2.0
    Mantle.vk_flush!(Mantle.vk_context())
    @test Array(b) ≈ Float64[2.0, 4.0, 6.0]
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
    a = Mantle.LavaArray(a_data)
    b = Mantle.LavaArray(b_data)
    dst = Mantle.LavaArray{SpectrumData}(undef, N)

    spectrum_add!(Mantle.LavaBackend())(dst, a, b; ndrange=N)
    Mantle.vk_flush!(Mantle.vk_context())

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
# There is no long-lived open command buffer to overflow now: every launch is
# its own one-shot, sealed and submitted immediately, so this checks the
# vendor-neutral property that survives — many dispatches in a row, each its
# own submission, still produce correct results.
# ═══════════════════════════════════════════════════════════════════════

@testset "5000 dispatches, one submission each" begin
    @kernel function cb_split_inc!(a)
        i = @index(Global)
        @inbounds a[i] += 1.0f0
    end

    backend = Mantle.LavaBackend()
    a = Mantle.LavaArray(zeros(Float32, 256))
    kernel = cb_split_inc!(backend)
    for _ in 1:5000
        kernel(a; ndrange=256)
    end

    Mantle.vk_flush!(Mantle.vk_context())
    @test Array(a) == fill(5000.0f0, 256)
end
