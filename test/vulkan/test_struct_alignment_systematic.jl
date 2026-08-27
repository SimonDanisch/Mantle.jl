# Systematic struct alignment & pointer access tests.
#
# Combinatorial matrix: 16 struct types x 7 contexts = ~100 GPU-vs-CPU tests.
# Each test writes known values on GPU and compares against CPU reference.
#
# Struct types exercise different alignment/padding corner cases.
# Contexts exercise different SPIR-V emission paths (PSB, Function, Workgroup).

using Test
using Lava, Mantle
using KernelAbstractions

# ── Struct type definitions (Axis 1: ~16 types) ──

struct S01_ThreeF32
    x::Float32; y::Float32; z::Float32
end

struct S02_LeadingBool
    flag::Bool; x::Float32; y::Float32
end

struct S03_MiddleBool
    x::Float32; flag::Bool; y::Float32
end

struct S04_TrailingBool
    x::Float32; y::Float32; flag::Bool
end

struct S05_TwoBools
    f1::Bool; f2::Bool; x::Float32
end

struct S06_MixedI64
    a::Float32; b::Int64; c::Float32
end

struct S07_LeadingI64
    a::Int64; b::Float32
end

struct S08_TinyBytes
    a::UInt8; b::UInt8; c::UInt8
end

struct S09_ShortFields
    a::Int16; b::Int16; c::Int16
end

struct S10_Nested
    inner::S01_ThreeF32; w::Float32
end

struct S11_DeepNested
    mid::S10_Nested; flag::Bool; extra::Float32
end

# S12 (WithPtr) skipped for now - requires special setup for device pointers

struct S13_TupleField
    data::NTuple{4, Float32}; id::Int32
end

struct S14_KitchenSink
    flag1::Bool; small::UInt8; pad16::Int16
    f32val::Float32; i32val::Int32; flag2::Bool
    f64val::Float64; i64val::Int64; final_f32::Float32
end

struct S15_AllF64
    a::Float64; b::Float64; c::Float64
end

struct S16_MixedF32F64
    a::Float32; b::Float64; c::Float32
end

# ── Test data generators ──
# Each returns a Vector{T} of n elements with deterministic, distinguishable values.

make_data(::Type{S01_ThreeF32}, n) =
    [S01_ThreeF32(Float32(i), Float32(i + 0.25), Float32(i + 0.5)) for i in 1:n]

make_data(::Type{S02_LeadingBool}, n) =
    [S02_LeadingBool(isodd(i), Float32(i), Float32(i + 0.5)) for i in 1:n]

make_data(::Type{S03_MiddleBool}, n) =
    [S03_MiddleBool(Float32(i), isodd(i), Float32(i + 0.5)) for i in 1:n]

make_data(::Type{S04_TrailingBool}, n) =
    [S04_TrailingBool(Float32(i), Float32(i + 0.5), isodd(i)) for i in 1:n]

make_data(::Type{S05_TwoBools}, n) =
    [S05_TwoBools(isodd(i), iseven(i), Float32(i)) for i in 1:n]

make_data(::Type{S06_MixedI64}, n) =
    [S06_MixedI64(Float32(i), Int64(i * 1000 + 7), Float32(i + 0.5)) for i in 1:n]

make_data(::Type{S07_LeadingI64}, n) =
    [S07_LeadingI64(Int64(i * 1000 + 7), Float32(i + 0.25)) for i in 1:n]

make_data(::Type{S08_TinyBytes}, n) =
    [S08_TinyBytes(UInt8(i % 200 + 1), UInt8((i * 7) % 200 + 1), UInt8((i * 13) % 200 + 1)) for i in 1:n]

make_data(::Type{S09_ShortFields}, n) =
    [S09_ShortFields(Int16(i), Int16(i * 3), Int16(i * 7)) for i in 1:n]

make_data(::Type{S10_Nested}, n) =
    [S10_Nested(S01_ThreeF32(Float32(i), Float32(i+0.25), Float32(i+0.5)), Float32(i+0.75)) for i in 1:n]

make_data(::Type{S11_DeepNested}, n) =
    [S11_DeepNested(S10_Nested(S01_ThreeF32(Float32(i), Float32(i+0.1), Float32(i+0.2)), Float32(i+0.3)), isodd(i), Float32(i+0.4)) for i in 1:n]

make_data(::Type{S13_TupleField}, n) =
    [S13_TupleField(ntuple(j -> Float32(i * 10 + j), 4), Int32(i)) for i in 1:n]

make_data(::Type{S14_KitchenSink}, n) =
    [S14_KitchenSink(isodd(i), UInt8(i % 200 + 1), Int16(i * 3),
        Float32(i), Int32(i * 7), iseven(i),
        Float64(i * 1.5), Int64(i * 1000), Float32(i + 0.25)) for i in 1:n]

make_data(::Type{S15_AllF64}, n) =
    [S15_AllF64(Float64(i), Float64(i + 0.5), Float64(i + 0.75)) for i in 1:n]

make_data(::Type{S16_MixedF32F64}, n) =
    [S16_MixedF32F64(Float32(i), Float64(i + 0.5), Float32(i + 0.75)) for i in 1:n]

# ── Checksum functions for field-read verification ──
# Each takes a struct and returns a Float64 combining all fields.
# The checksum must be unique for each distinct struct value.

checksum(s::S01_ThreeF32) = Float64(s.x) + Float64(s.y) * 1000 + Float64(s.z) * 1e6
checksum(s::S02_LeadingBool) = Float64(s.flag) + Float64(s.x) * 1000 + Float64(s.y) * 1e6
checksum(s::S03_MiddleBool) = Float64(s.x) + Float64(s.flag) * 1000 + Float64(s.y) * 1e6
checksum(s::S04_TrailingBool) = Float64(s.x) + Float64(s.y) * 1000 + Float64(s.flag) * 1e6
checksum(s::S05_TwoBools) = Float64(s.f1) + Float64(s.f2) * 10 + Float64(s.x) * 1000
checksum(s::S06_MixedI64) = Float64(s.a) + Float64(s.b) + Float64(s.c) * 1e6
checksum(s::S07_LeadingI64) = Float64(s.a) + Float64(s.b) * 1e6
checksum(s::S08_TinyBytes) = Float64(s.a) + Float64(s.b) * 256 + Float64(s.c) * 65536
checksum(s::S09_ShortFields) = Float64(s.a) + Float64(s.b) * 1000 + Float64(s.c) * 1e6
checksum(s::S10_Nested) = checksum(s.inner) + Float64(s.w) * 1e9
checksum(s::S11_DeepNested) = checksum(s.mid) + Float64(s.flag) * 1e12 + Float64(s.extra) * 1e13
checksum(s::S13_TupleField) = sum(Float64, s.data) + Float64(s.id) * 1e6
checksum(s::S14_KitchenSink) = Float64(s.flag1) + Float64(s.small) + Float64(s.pad16) +
    Float64(s.f32val) * 100 + Float64(s.i32val) + Float64(s.flag2) * 10 +
    s.f64val * 1e4 + Float64(s.i64val) + Float64(s.final_f32) * 1e7
checksum(s::S15_AllF64) = s.a + s.b * 1000 + s.c * 1e6
checksum(s::S16_MixedF32F64) = Float64(s.a) + s.b * 1000 + Float64(s.c) * 1e6

# ── List of all testable struct types ──

const ALL_STRUCT_TYPES = [
    S01_ThreeF32, S02_LeadingBool, S03_MiddleBool, S04_TrailingBool,
    S05_TwoBools, S06_MixedI64, S07_LeadingI64, S08_TinyBytes,
    S09_ShortFields, S10_Nested, S11_DeepNested,
    S13_TupleField, S14_KitchenSink, S15_AllF64, S16_MixedF32F64,
]

# ── Context C3: Whole-struct copy (simplest, test first) ──

@testset "C3: Whole-struct copy - $S" for S in ALL_STRUCT_TYPES
    n = 64
    src_data = make_data(S, n)
    src = Mantle.LavaArray(src_data)
    dst = Mantle.LavaArray{S}(undef, n)
    dst .= src
    Mantle.vk_flush!(Mantle.vk_context())
    result = Array(dst)
    @test result == src_data
end

# ── Context C1: PSB read all fields (checksum) ──

# We define per-type GPU kernels that compute checksum on GPU
# and compare against CPU checksum.

@kernel function checksum_kernel_S01(dst, src)
    i = @index(Global)
    @inbounds begin
        s = src[i]
        dst[i] = Float64(s.x) + Float64(s.y) * 1000.0 + Float64(s.z) * 1e6
    end
end

@kernel function checksum_kernel_S02(dst, src)
    i = @index(Global)
    @inbounds begin
        s = src[i]
        dst[i] = Float64(s.flag) + Float64(s.x) * 1000.0 + Float64(s.y) * 1e6
    end
end

@kernel function checksum_kernel_S03(dst, src)
    i = @index(Global)
    @inbounds begin
        s = src[i]
        dst[i] = Float64(s.x) + Float64(s.flag) * 1000.0 + Float64(s.y) * 1e6
    end
end

@kernel function checksum_kernel_S04(dst, src)
    i = @index(Global)
    @inbounds begin
        s = src[i]
        dst[i] = Float64(s.x) + Float64(s.y) * 1000.0 + Float64(s.flag) * 1e6
    end
end

@kernel function checksum_kernel_S05(dst, src)
    i = @index(Global)
    @inbounds begin
        s = src[i]
        dst[i] = Float64(s.f1) + Float64(s.f2) * 10.0 + Float64(s.x) * 1000.0
    end
end

@kernel function checksum_kernel_S06(dst, src)
    i = @index(Global)
    @inbounds begin
        s = src[i]
        dst[i] = Float64(s.a) + Float64(s.b) + Float64(s.c) * 1e6
    end
end

@kernel function checksum_kernel_S07(dst, src)
    i = @index(Global)
    @inbounds begin
        s = src[i]
        dst[i] = Float64(s.a) + Float64(s.b) * 1e6
    end
end

@kernel function checksum_kernel_S08(dst, src)
    i = @index(Global)
    @inbounds begin
        s = src[i]
        dst[i] = Float64(s.a) + Float64(s.b) * 256.0 + Float64(s.c) * 65536.0
    end
end

@kernel function checksum_kernel_S09(dst, src)
    i = @index(Global)
    @inbounds begin
        s = src[i]
        dst[i] = Float64(s.a) + Float64(s.b) * 1000.0 + Float64(s.c) * 1e6
    end
end

@kernel function checksum_kernel_S10(dst, src)
    i = @index(Global)
    @inbounds begin
        s = src[i]
        dst[i] = Float64(s.inner.x) + Float64(s.inner.y) * 1000.0 + Float64(s.inner.z) * 1e6 + Float64(s.w) * 1e9
    end
end

@kernel function checksum_kernel_S11(dst, src)
    i = @index(Global)
    @inbounds begin
        s = src[i]
        dst[i] = Float64(s.mid.inner.x) + Float64(s.mid.inner.y) * 1000.0 +
                 Float64(s.mid.inner.z) * 1e6 + Float64(s.mid.w) * 1e9 +
                 Float64(s.flag) * 1e12 + Float64(s.extra) * 1e13
    end
end

@kernel function checksum_kernel_S13(dst, src)
    i = @index(Global)
    @inbounds begin
        s = src[i]
        dst[i] = Float64(s.data[1]) + Float64(s.data[2]) + Float64(s.data[3]) + Float64(s.data[4]) + Float64(s.id) * 1e6
    end
end

@kernel function checksum_kernel_S14(dst, src)
    i = @index(Global)
    @inbounds begin
        s = src[i]
        dst[i] = Float64(s.flag1) + Float64(s.small) + Float64(s.pad16) +
                 Float64(s.f32val) * 100.0 + Float64(s.i32val) + Float64(s.flag2) * 10.0 +
                 s.f64val * 1e4 + Float64(s.i64val) + Float64(s.final_f32) * 1e7
    end
end

@kernel function checksum_kernel_S15(dst, src)
    i = @index(Global)
    @inbounds begin
        s = src[i]
        dst[i] = s.a + s.b * 1000.0 + s.c * 1e6
    end
end

@kernel function checksum_kernel_S16(dst, src)
    i = @index(Global)
    @inbounds begin
        s = src[i]
        dst[i] = Float64(s.a) + s.b * 1000.0 + Float64(s.c) * 1e6
    end
end

const CHECKSUM_KERNELS = Dict{DataType, Any}(
    S01_ThreeF32 => checksum_kernel_S01,
    S02_LeadingBool => checksum_kernel_S02,
    S03_MiddleBool => checksum_kernel_S03,
    S04_TrailingBool => checksum_kernel_S04,
    S05_TwoBools => checksum_kernel_S05,
    S06_MixedI64 => checksum_kernel_S06,
    S07_LeadingI64 => checksum_kernel_S07,
    S08_TinyBytes => checksum_kernel_S08,
    S09_ShortFields => checksum_kernel_S09,
    S10_Nested => checksum_kernel_S10,
    S11_DeepNested => checksum_kernel_S11,
    S13_TupleField => checksum_kernel_S13,
    S14_KitchenSink => checksum_kernel_S14,
    S15_AllF64 => checksum_kernel_S15,
    S16_MixedF32F64 => checksum_kernel_S16,
)

@testset "C1: PSB read fields - $S" for S in ALL_STRUCT_TYPES
    n = 64
    src_data = make_data(S, n)
    src = Mantle.LavaArray(src_data)
    dst = Mantle.LavaArray{Float64}(undef, n)

    kern = CHECKSUM_KERNELS[S]
    kern(Mantle.LavaBackend(), 64)(dst, src; ndrange=n)
    Mantle.vk_flush!(Mantle.vk_context())

    gpu_result = Array(dst)
    cpu_result = [checksum(s) for s in src_data]
    @test gpu_result == cpu_result
end

# ── Context C2: PSB write all fields ──
# Write a struct with modified fields to test the SROA byte-offset GEP path.

@kernel function write_modified_S01(dst, src, offset::Float32)
    i = @index(Global)
    @inbounds begin
        s = src[i]
        dst[i] = S01_ThreeF32(s.x + offset, s.y + offset, s.z + offset)
    end
end

@kernel function write_modified_S03(dst, src, offset::Float32)
    i = @index(Global)
    @inbounds begin
        s = src[i]
        dst[i] = S03_MiddleBool(s.x + offset, s.flag, s.y + offset)
    end
end

@kernel function write_modified_S06(dst, src, offset::Float32)
    i = @index(Global)
    @inbounds begin
        s = src[i]
        dst[i] = S06_MixedI64(s.a + offset, s.b + Int64(1), s.c + offset)
    end
end

@kernel function write_modified_S14(dst, src, offset::Float32)
    i = @index(Global)
    @inbounds begin
        s = src[i]
        dst[i] = S14_KitchenSink(s.flag1, s.small, s.pad16,
            s.f32val + offset, s.i32val, s.flag2,
            s.f64val + Float64(offset), s.i64val, s.final_f32 + offset)
    end
end

@kernel function write_modified_S15(dst, src, offset::Float32)
    i = @index(Global)
    @inbounds begin
        s = src[i]
        dst[i] = S15_AllF64(s.a + Float64(offset), s.b + Float64(offset), s.c + Float64(offset))
    end
end

@testset "C2: PSB write fields" begin
    n = 64
    offset = 100.0f0

    @testset "S01_ThreeF32" begin
        src_data = make_data(S01_ThreeF32, n)
        src = Mantle.LavaArray(src_data)
        dst = Mantle.LavaArray{S01_ThreeF32}(undef, n)
        write_modified_S01(Mantle.LavaBackend(), 64)(dst, src, offset; ndrange=n)
        Mantle.vk_flush!(Mantle.vk_context())
        result = Array(dst)
        expected = [S01_ThreeF32(s.x + offset, s.y + offset, s.z + offset) for s in src_data]
        @test result == expected
    end

    @testset "S03_MiddleBool" begin
        src_data = make_data(S03_MiddleBool, n)
        src = Mantle.LavaArray(src_data)
        dst = Mantle.LavaArray{S03_MiddleBool}(undef, n)
        write_modified_S03(Mantle.LavaBackend(), 64)(dst, src, offset; ndrange=n)
        Mantle.vk_flush!(Mantle.vk_context())
        result = Array(dst)
        expected = [S03_MiddleBool(s.x + offset, s.flag, s.y + offset) for s in src_data]
        @test result == expected
    end

    @testset "S06_MixedI64" begin
        src_data = make_data(S06_MixedI64, n)
        src = Mantle.LavaArray(src_data)
        dst = Mantle.LavaArray{S06_MixedI64}(undef, n)
        write_modified_S06(Mantle.LavaBackend(), 64)(dst, src, offset; ndrange=n)
        Mantle.vk_flush!(Mantle.vk_context())
        result = Array(dst)
        expected = [S06_MixedI64(s.a + offset, s.b + 1, s.c + offset) for s in src_data]
        @test result == expected
    end

    @testset "S14_KitchenSink" begin
        src_data = make_data(S14_KitchenSink, n)
        src = Mantle.LavaArray(src_data)
        dst = Mantle.LavaArray{S14_KitchenSink}(undef, n)
        write_modified_S14(Mantle.LavaBackend(), 64)(dst, src, offset; ndrange=n)
        Mantle.vk_flush!(Mantle.vk_context())
        result = Array(dst)
        expected = [S14_KitchenSink(s.flag1, s.small, s.pad16,
            s.f32val + offset, s.i32val, s.flag2,
            s.f64val + Float64(offset), s.i64val, s.final_f32 + offset) for s in src_data]
        @test result == expected
    end

    @testset "S15_AllF64" begin
        src_data = make_data(S15_AllF64, n)
        src = Mantle.LavaArray(src_data)
        dst = Mantle.LavaArray{S15_AllF64}(undef, n)
        write_modified_S15(Mantle.LavaBackend(), 64)(dst, src, offset; ndrange=n)
        Mantle.vk_flush!(Mantle.vk_context())
        result = Array(dst)
        expected = [S15_AllF64(s.a + Float64(offset), s.b + Float64(offset), s.c + Float64(offset)) for s in src_data]
        @test result == expected
    end
end

# ── Context C5: Function alloca + field modify ──
# Same as C2 but explicitly exercises the alloca path.
# KA kernels with struct construction implicitly use allocas.
# The C2 tests above already exercise this path after SROA.

# ── Context C6: Broadcast fill (closure capture) ──

@testset "C6: Broadcast fill - $S" for S in ALL_STRUCT_TYPES
    n = 64
    test_vals = make_data(S, 1)
    fill_val = test_vals[1]
    dst = Mantle.LavaArray{S}(undef, n)
    fill!(dst, fill_val)
    Mantle.vk_flush!(Mantle.vk_context())
    result = Array(dst)
    @test all(x -> x == fill_val, result)
end

# ── Context C4: PSB array-of-struct with dynamic index ──
# Tests stride alignment. Use array sizes that create misalignment:
# - n=5 exercises strides that may not be power-of-2 aligned
# - n=64 normal case

@testset "C4: PSB array-of-struct dynamic index - $S" for S in ALL_STRUCT_TYPES
    for n in [5, 7, 13, 64]
        @testset "n=$n" begin
            src_data = make_data(S, n)
            src = Mantle.LavaArray(src_data)
            dst = Mantle.LavaArray{Float64}(undef, n)

            kern = CHECKSUM_KERNELS[S]
            wg = min(n, 64)
            kern(Mantle.LavaBackend(), wg)(dst, src; ndrange=n)
            Mantle.vk_flush!(Mantle.vk_context())

            gpu_result = Array(dst)
            cpu_result = [checksum(s) for s in src_data]
            @test gpu_result == cpu_result
        end
    end
end

# ── Context C7: Workgroup shared memory ──
# Load from global -> shared -> barrier -> read from shared -> write to global.
# Tests workgroup type IDs, Block decoration, MemberOffset.

# We only test struct types that can be stored in shared memory
# (no pointers, reasonable size).

@kernel function shared_copy_S01(dst, src)
    i = @index(Global)
    lid = @index(Local)
    shared = @localmem S01_ThreeF32 64
    @inbounds shared[lid] = src[i]
    @synchronize()
    @inbounds dst[i] = shared[lid]
end

@kernel function shared_copy_S03(dst, src)
    i = @index(Global)
    lid = @index(Local)
    shared = @localmem S03_MiddleBool 64
    @inbounds shared[lid] = src[i]
    @synchronize()
    @inbounds dst[i] = shared[lid]
end

@kernel function shared_copy_S06(dst, src)
    i = @index(Global)
    lid = @index(Local)
    shared = @localmem S06_MixedI64 64
    @inbounds shared[lid] = src[i]
    @synchronize()
    @inbounds dst[i] = shared[lid]
end

@kernel function shared_copy_S14(dst, src)
    i = @index(Global)
    lid = @index(Local)
    shared = @localmem S14_KitchenSink 64
    @inbounds shared[lid] = src[i]
    @synchronize()
    @inbounds dst[i] = shared[lid]
end

@kernel function shared_copy_S15(dst, src)
    i = @index(Global)
    lid = @index(Local)
    shared = @localmem S15_AllF64 64
    @inbounds shared[lid] = src[i]
    @synchronize()
    @inbounds dst[i] = shared[lid]
end

const SHARED_COPY_KERNELS = Dict{DataType, Any}(
    S01_ThreeF32 => shared_copy_S01,
    S03_MiddleBool => shared_copy_S03,
    S06_MixedI64 => shared_copy_S06,
    S14_KitchenSink => shared_copy_S14,
    S15_AllF64 => shared_copy_S15,
)

@testset "C7: Workgroup shared memory - $S" for S in keys(SHARED_COPY_KERNELS)
    n = 64
    src_data = make_data(S, n)
    src = Mantle.LavaArray(src_data)
    dst = Mantle.LavaArray{S}(undef, n)

    kern = SHARED_COPY_KERNELS[S]
    kern(Mantle.LavaBackend(), 64)(dst, src; ndrange=n)
    Mantle.vk_flush!(Mantle.vk_context())

    result = Array(dst)
    @test result == src_data
end
