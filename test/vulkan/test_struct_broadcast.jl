# Test struct broadcast correctness — regression tests for LLVM/Julia struct size mismatch.
#
# Bug: LLVM's struct layout can be larger than Julia's sizeof for types with zero-sized
# fields (Nothing, type parameters). The host-side arg packing used Julia's sizeof to
# allocate inline struct data, but LLVM-generated kernel code reads at LLVM's offsets.
# When LLVM's struct is larger, inline data for successive args overlaps, corrupting values.
#
# The fix: use LLVM byval type sizes (from function parameter attributes) for inline
# struct allocation instead of Julia's sizeof. See entry_wrapper.jl and launch.jl.
#
# These tests specifically target the n=10 case where CompilerMetadata has LLVM size 24
# but Julia sizeof 16, causing the Broadcasted closure's inline data to be read incorrectly.

using Test
using Lava, Mantle
using KernelAbstractions

# ── Test structs ──

struct TestS12
    a::Float32
    b::Float32
    c::Float32
end

struct TestS52
    f1::Float32; f2::Float32; f3::Float32; f4::Float32
    f5::Float32; f6::Float32; f7::Float32; f8::Float32
    f9::Float32; f10::Float32; f11::Float32; f12::Float32
    f13::Float32
end

struct TestS24
    a::Float64
    b::Float64
    c::Float64
end

@testset "Struct Broadcast Correctness" begin
    # The key sizes that trigger different CompilerMetadata LLVM types:
    # - n=10 → workgroup_size=10, CompilerMetadata LLVM=24 vs Julia=16 (the original bug)
    # - n=1,5 → small sizes
    # - n=32,64,256,1000 → typical sizes with workgroup_size=32/64/256
    @testset "S12 broadcast n=$n" for n in [1, 5, 10, 32, 64, 100, 256, 1000]
        src = Mantle.LavaArray(TestS12[TestS12(Float32(i), Float32(i+0.5), Float32(i+0.25)) for i in 1:n])
        dst = Mantle.LavaArray{TestS12}(undef, n)
        dst .= src
        Mantle.vk_flush!(Mantle.vk_context())
        result = Array(dst)
        expected = Array(src)
        @test result == expected
    end

    @testset "S52 broadcast n=$n" for n in [1, 10, 64, 256]
        src = Mantle.LavaArray(TestS52[TestS52(ntuple(j -> Float32(i*100 + j), 13)...) for i in 1:n])
        dst = Mantle.LavaArray{TestS52}(undef, n)
        dst .= src
        Mantle.vk_flush!(Mantle.vk_context())
        result = Array(dst)
        expected = Array(src)
        @test result == expected
    end

    @testset "S24 (Float64 fields) broadcast n=$n" for n in [1, 10, 64]
        src = Mantle.LavaArray(TestS24[TestS24(Float64(i), Float64(i+0.5), Float64(i+0.25)) for i in 1:n])
        dst = Mantle.LavaArray{TestS24}(undef, n)
        dst .= src
        Mantle.vk_flush!(Mantle.vk_context())
        result = Array(dst)
        expected = Array(src)
        @test result == expected
    end

    # Test that byval_llvm_sizes are populated during compilation
    @testset "byval_llvm_sizes populated" begin
        # After running broadcasts above, linked cache should have entries.
        # Two levels: the cache is keyed by device first, then by kernel.
        @test !isempty(Mantle.linked_kernel_cache(Mantle.vk_context()))
        # All byval_sizes should be non-negative. One level now: the cache is a
        # field on the context, so there is no outer dict to iterate by mistake.
        for (_, linked) in Mantle.vk_context().caches.linked
            @test all(s -> s >= 0, linked.byval_sizes)
        end
    end
end

@testset "KA Kernel with Struct Args" begin
    @kernel function copy_structs_ka(dst, src)
        i = @index(Global)
        @inbounds dst[i] = src[i]
    end

    @testset "KA copy S12 n=$n" for n in [10, 64, 256]
        src = Mantle.LavaArray(TestS12[TestS12(Float32(i), Float32(2i), Float32(3i)) for i in 1:n])
        dst = Mantle.LavaArray{TestS12}(undef, n)
        kernel = copy_structs_ka(Mantle.LavaBackend())
        kernel(dst, src; ndrange=n)
        Mantle.vk_flush!(Mantle.vk_context())
        @test Array(dst) == Array(src)
    end

    @kernel function transform_structs_ka(dst, src, scale::Float32)
        i = @index(Global)
        s = @inbounds src[i]
        @inbounds dst[i] = TestS12(s.a * scale, s.b * scale, s.c * scale)
    end

    @testset "KA transform S12 with scalar arg n=$n" for n in [10, 64]
        src = Mantle.LavaArray(TestS12[TestS12(1f0, 2f0, 3f0) for _ in 1:n])
        dst = Mantle.LavaArray{TestS12}(undef, n)
        kernel = transform_structs_ka(Mantle.LavaBackend())
        kernel(dst, src, 2f0; ndrange=n)
        Mantle.vk_flush!(Mantle.vk_context())
        result = Array(dst)
        @test all(r -> r == TestS12(2f0, 4f0, 6f0), result)
    end
end

# ── Bool-padding alignment regression tests ──
# Bug: structs with Bool fields followed by Float32/Int32 have alignment padding
# (3 bytes after Bool). The SPIR-V emitter computed byte offsets for struct fields
# by summing compute_type_size without padding, producing addresses shifted by
# 1-3 bytes. This caused GPUVM faults on RADV and unaligned BDA access errors
# in GPU-assisted validation.

struct BoolPadStruct
    x::Float32
    flag::Bool      # offset 4, 1 byte + 3 padding
    y::Float32      # offset 8
end

struct TwoBoolStruct
    a::Float32
    b::Float32
    c::Float32
    flag1::Bool     # offset 12
    flag2::Bool     # offset 13, 2 bytes padding
    d::Float32      # offset 16
end

struct BoolHeavyStruct
    v1::Float32; v2::Float32; v3::Float32
    active::Bool     # offset 12, padding to 16
    w1::Float32; w2::Float32; w3::Float32; w4::Float32
    done::Bool       # offset 32, padding to 36
    result::Float32  # offset 36
end

@testset "Bool Padding Alignment" begin
    backend = Mantle.LavaBackend()

    @kernel function read_bool_pad(dst, @Const(src))
        i = @index(Global)
        @inbounds begin
            s = src[i]
            dst[i] = s.flag ? s.x + s.y : s.x - s.y
        end
    end

    @testset "BoolPadStruct read n=$n" for n in [64, 256, 1024]
        src = Mantle.LavaArray([BoolPadStruct(Float32(i), isodd(i), Float32(i+1)) for i in 1:n])
        dst = Mantle.LavaArray(zeros(Float32, n))
        read_bool_pad(backend)(dst, src; ndrange=n)
        Mantle.vk_flush!(Mantle.vk_context())
        result = Array(dst)
        expected = [isodd(i) ? Float32(2i+1) : Float32(-1) for i in 1:n]
        @test result ≈ expected
    end

    @kernel function write_bool_pad(dst, scale::Float32)
        i = @index(Global)
        @inbounds dst[i] = BoolPadStruct(Float32(i) * scale, isodd(i), Float32(i) + scale)
    end

    @testset "BoolPadStruct write n=$n" for n in [64, 256]
        dst = Mantle.LavaArray{BoolPadStruct}(undef, n)
        write_bool_pad(backend)(dst, 2f0; ndrange=n)
        Mantle.vk_flush!(Mantle.vk_context())
        result = Array(dst)
        for i in 1:n
            @test result[i].x ≈ Float32(i) * 2f0
            @test result[i].flag == isodd(i)
            @test result[i].y ≈ Float32(i) + 2f0
        end
    end

    @kernel function read_two_bools(dst, @Const(src))
        i = @index(Global)
        @inbounds begin
            s = src[i]
            val = s.a + s.b + s.c + s.d
            if s.flag1; val += 100f0; end
            if s.flag2; val += 1000f0; end
            dst[i] = val
        end
    end

    @testset "TwoBoolStruct (consecutive bools) n=$n" for n in [64, 256]
        src = Mantle.LavaArray([TwoBoolStruct(1f0, 2f0, 3f0, isodd(i), i % 3 == 0, 4f0) for i in 1:n])
        dst = Mantle.LavaArray(zeros(Float32, n))
        read_two_bools(backend)(dst, src; ndrange=n)
        Mantle.vk_flush!(Mantle.vk_context())
        result = Array(dst)
        for i in 1:n
            expected = 10f0 + (isodd(i) ? 100f0 : 0f0) + (i % 3 == 0 ? 1000f0 : 0f0)
            @test result[i] ≈ expected
        end
    end

    @kernel function read_bool_heavy(dst, @Const(src))
        i = @index(Global)
        @inbounds begin
            s = src[i]
            val = s.v1 + s.v2 + s.v3 + s.w1 + s.w2 + s.w3 + s.w4 + s.result
            if s.active; val *= 2f0; end
            if s.done; val *= -1f0; end
            dst[i] = val
        end
    end

    @testset "BoolHeavyStruct (bools at different offsets) n=$n" for n in [64, 256]
        src = Mantle.LavaArray([BoolHeavyStruct(1f0,2f0,3f0, isodd(i), 4f0,5f0,6f0,7f0, i%3==0, 8f0) for i in 1:n])
        dst = Mantle.LavaArray(zeros(Float32, n))
        read_bool_heavy(backend)(dst, src; ndrange=n)
        Mantle.vk_flush!(Mantle.vk_context())
        result = Array(dst)
        for i in 1:n
            base = 36f0  # 1+2+3+4+5+6+7+8
            val = isodd(i) ? base * 2f0 : base
            val = i % 3 == 0 ? val * -1f0 : val
            @test result[i] ≈ val
        end
    end
end

# ── CPU-vs-GPU roundtrip tests ──
# The ultimate correctness check: run the same kernel on CPU (KernelAbstractions.CPU()) and GPU,
# compare results. This catches any SPIR-V emission bug that produces wrong values
# (not just alignment faults). Uses struct types modeled after the actual VolPath
# work items (164-byte VPRayWorkItem, 320-byte VPMediumSampleWorkItem, etc.).

struct RoundtripSmall
    pos::NTuple{3, Float32}  # 12 bytes
    dir::NTuple{3, Float32}  # 12 bytes
    t::Float32               # 4 bytes
    depth::Int32             # 4 bytes = 32 total
end

struct RoundtripWithBool
    pos::NTuple{3, Float32}
    dir::NTuple{3, Float32}
    t_max::Float32
    time::Float32
    active::Bool              # offset 32, 3 bytes padding
    origin::NTuple{3, Float32}  # offset 36
    target::NTuple{3, Float32}  # offset 48
    scale::Float32            # offset 60
end

struct RoundtripTwoBool
    beta::NTuple{4, Float32}      # 16 bytes
    r_u::NTuple{4, Float32}       # 16 bytes
    prev_p::NTuple{3, Float32}    # 12 bytes
    prev_n::NTuple{3, Float32}    # 12 bytes
    eta::Float32                  # 4 bytes
    specular::Bool                # offset 60, +1 byte
    any_nonspecular::Bool         # offset 61, 2 bytes padding
    medium_type::UInt32           # offset 64
    medium_idx::UInt32            # offset 68 = 72 total with padding
end

struct RoundtripNested
    inner::RoundtripSmall
    weight::Float32
    flag::Bool
    extra::Float32
end

@testset "CPU-vs-GPU Roundtrip" begin
    backend = Mantle.LavaBackend()
    N = 512

    @kernel function roundtrip_small_kernel(dst, @Const(src), scale::Float32)
        i = @index(Global)
        @inbounds begin
            s = src[i]
            p = s.pos; d = s.dir
            dst[i] = RoundtripSmall(
                (p[1]*scale, p[2]*scale, p[3]*scale),
                (d[1]+1f0, d[2]+1f0, d[3]+1f0),
                s.t * 2f0,
                s.depth + Int32(1)
            )
        end
    end

    @testset "RoundtripSmall (no Bool)" begin
        data = [RoundtripSmall(ntuple(j->Float32(i*10+j), 3), ntuple(j->Float32(j), 3), Float32(i), Int32(i)) for i in 1:N]
        src_cpu = copy(data); dst_cpu = similar(data)
        roundtrip_small_kernel(KernelAbstractions.CPU())(dst_cpu, src_cpu, 0.5f0; ndrange=N)
        src_gpu = Mantle.LavaArray(data); dst_gpu = Mantle.LavaArray{RoundtripSmall}(undef, N)
        roundtrip_small_kernel(backend)(dst_gpu, src_gpu, 0.5f0; ndrange=N)
        Mantle.vk_flush!(Mantle.vk_context())
        @test Array(dst_gpu) == dst_cpu
    end

    @kernel function roundtrip_bool_kernel(dst, @Const(src))
        i = @index(Global)
        @inbounds begin
            s = src[i]
            val = s.active ? s.t_max + s.scale : s.time - s.scale
            dst[i] = RoundtripWithBool(
                s.pos, s.dir, s.t_max, s.time,
                !s.active,
                (s.origin[1]+val, s.origin[2]+val, s.origin[3]+val),
                s.target, s.scale * 2f0
            )
        end
    end

    @testset "RoundtripWithBool (1 Bool + padding)" begin
        data = [RoundtripWithBool(
            ntuple(j->Float32(i+j), 3), ntuple(j->Float32(j), 3),
            Float32(i), 0.5f0, isodd(i),
            ntuple(j->Float32(10+j), 3), ntuple(j->Float32(20+j), 3), Float32(i)*0.1f0
        ) for i in 1:N]
        src_cpu = copy(data); dst_cpu = similar(data)
        roundtrip_bool_kernel(KernelAbstractions.CPU())(dst_cpu, src_cpu; ndrange=N)
        src_gpu = Mantle.LavaArray(data); dst_gpu = Mantle.LavaArray{RoundtripWithBool}(undef, N)
        roundtrip_bool_kernel(backend)(dst_gpu, src_gpu; ndrange=N)
        Mantle.vk_flush!(Mantle.vk_context())
        @test Array(dst_gpu) == dst_cpu
    end

    @kernel function roundtrip_twobool_kernel(dst_f, @Const(src))
        i = @index(Global)
        @inbounds begin
            s = src[i]
            val = s.beta[1] + s.r_u[1] + s.prev_p[1] + s.prev_n[1] + s.eta
            if s.specular; val *= 2f0; end
            if s.any_nonspecular; val += Float32(s.medium_type); end
            val += Float32(s.medium_idx)
            dst_f[i] = val
        end
    end

    @testset "RoundtripTwoBool (consecutive Bools at offset 60-61)" begin
        data = [RoundtripTwoBool(
            ntuple(j->Float32(i+j), 4), ntuple(j->Float32(j*0.1f0), 4),
            ntuple(j->Float32(j), 3), ntuple(j->Float32(j+3), 3),
            Float32(i)*0.01f0, isodd(i), i%3==0, UInt32(i%10), UInt32(i)
        ) for i in 1:N]
        src_cpu = copy(data); dst_cpu = zeros(Float32, N)
        roundtrip_twobool_kernel(KernelAbstractions.CPU())(dst_cpu, src_cpu; ndrange=N)
        src_gpu = Mantle.LavaArray(data); dst_gpu = Mantle.LavaArray(zeros(Float32, N))
        roundtrip_twobool_kernel(backend)(dst_gpu, src_gpu; ndrange=N)
        Mantle.vk_flush!(Mantle.vk_context())
        @test Array(dst_gpu) ≈ dst_cpu
    end

    @kernel function roundtrip_nested_kernel(dst, @Const(src), offset::Float32)
        i = @index(Global)
        @inbounds begin
            s = src[i]
            inner = s.inner
            val = inner.pos[1] + inner.t + s.weight + s.extra + offset
            if s.flag; val *= -1f0; end
            dst[i] = val
        end
    end

    @testset "RoundtripNested (Bool inside nested struct)" begin
        data = [RoundtripNested(
            RoundtripSmall(ntuple(j->Float32(i+j),3), ntuple(j->1f0,3), Float32(i), Int32(0)),
            Float32(i)*0.5f0, isodd(i), Float32(i)*0.1f0
        ) for i in 1:N]
        src_cpu = copy(data); dst_cpu = zeros(Float32, N)
        roundtrip_nested_kernel(KernelAbstractions.CPU())(dst_cpu, src_cpu, 100f0; ndrange=N)
        src_gpu = Mantle.LavaArray(data); dst_gpu = Mantle.LavaArray(zeros(Float32, N))
        roundtrip_nested_kernel(backend)(dst_gpu, src_gpu, 100f0; ndrange=N)
        Mantle.vk_flush!(Mantle.vk_context())
        @test Array(dst_gpu) ≈ dst_cpu
    end
end

# ── Struct layout fuzzing ──
# Generate random struct types with Bool/Int8/UInt8/Int16/Float32/Int32/Float64 fields
# at various positions. For each type, compile a kernel that reads all fields,
# combines them into a Float64 checksum, and compare CPU vs GPU results.
# This catches any alignment/padding mismatch in the SPIR-V emitter.

const FUZZ_FIELD_TYPES = [Bool, UInt8, Int16, Float32, Int32, Float64]

# Build struct types at eval time so they're real concrete types
module FuzzStructs
    using Random

    const generated = Dict{UInt64, DataType}()

    function make_fuzz_struct(field_types::Vector{DataType}, id::Int)
        name = Symbol("FuzzStruct_$(id)")
        fields = [Symbol("f$i") for i in 1:length(field_types)]
        # Build the struct expression
        field_exprs = [:($(fields[i])::$(field_types[i])) for i in 1:length(field_types)]
        eval(:(struct $name; $(field_exprs...); end))
        return eval(name)
    end
end

random_fuzz_value(::Type{Bool}) = rand(Bool)
random_fuzz_value(::Type{UInt8}) = rand(UInt8)
random_fuzz_value(::Type{Int16}) = rand(Int16)
random_fuzz_value(::Type{Float32}) = randn(Float32)
random_fuzz_value(::Type{Int32}) = rand(Int32(1):Int32(1000))
random_fuzz_value(::Type{Float64}) = randn(Float64)

@testset "Struct Layout Fuzzing" begin
    import Random
    backend = Mantle.LavaBackend()
    N = 128
    rng = Random.MersenneTwister(42)  # deterministic seed

    # Generate 20 random struct layouts
    for trial in 1:20
        n_fields = rand(rng, 2:8)
        field_types = [FUZZ_FIELD_TYPES[rand(rng, 1:length(FUZZ_FIELD_TYPES))] for _ in 1:n_fields]

        # Ensure at least one Bool or sub-4-byte field (the interesting cases)
        if !any(t -> t in (Bool, UInt8, Int16), field_types)
            field_types[rand(rng, 1:n_fields)] = Bool
        end

        T = FuzzStructs.make_fuzz_struct(field_types, trial)

        # Create test data
        data = [T((random_fuzz_value(ft) for ft in field_types)...) for _ in 1:N]

        # Build a checksum function that works on both CPU and GPU:
        # sum all fields converted to Float64
        # We use broadcast copy (dst .= src) as the simplest possible kernel --
        # if the struct layout is wrong, the copied values will be corrupted.
        src_cpu = copy(data)
        dst_cpu = similar(data)
        dst_cpu .= src_cpu

        src_gpu = Mantle.LavaArray(data)
        dst_gpu = Mantle.LavaArray{T}(undef, N)
        dst_gpu .= src_gpu
        Mantle.vk_flush!(Mantle.vk_context())

        result_gpu = Array(dst_gpu)

        @testset "fuzz #$trial: $(join(string.(field_types), ",")) ($(sizeof(T)) bytes)" begin
            @test result_gpu == dst_cpu
        end
    end
end

# ═══════════════════════════════════════════════════════════════════════
# Constant Struct Broadcasting
# Regression tests for OpStore type mismatch when broadcasting a constant
# struct value into a LavaArray. The broadcast kernel's closure captures
# the constant as a struct alloca, and the BDA-loaded data must correctly
# match the alloca's SPIR-V type.
# ═══════════════════════════════════════════════════════════════════════

using ColorTypes: RGB

@testset "Constant struct broadcast" begin
    @testset "RGB{Float32} constant" begin
        a = Mantle.LavaArray(fill(RGB{Float32}(1,1,1), 64))
        a .= RGB{Float32}(0, 0, 0)
        Mantle.vk_flush!(Mantle.vk_context())
        @test all(Array(a) .== RGB{Float32}(0, 0, 0))
    end

    @testset "NTuple{3,Float32} via fill!" begin
        a = Mantle.LavaArray(fill(NTuple{3,Float32}((1,1,1)), 64))
        fill!(a, NTuple{3,Float32}((0,0,0)))
        Mantle.vk_flush!(Mantle.vk_context())
        @test all(x -> x == NTuple{3,Float32}((0,0,0)), Array(a))
    end

    @testset "TestS12 Ref broadcast" begin
        a = Mantle.LavaArray(fill(TestS12(1,1,1), 64))
        a .= Ref(TestS12(0, 0, 0))
        Mantle.vk_flush!(Mantle.vk_context())
        @test all(x -> x == TestS12(0,0,0), Array(a))
    end

    @testset "BoolPadStruct Ref broadcast" begin
        a = Mantle.LavaArray(fill(BoolPadStruct(1f0, true, 2f0), 64))
        a .= Ref(BoolPadStruct(0f0, false, 0f0))
        Mantle.vk_flush!(Mantle.vk_context())
        @test all(x -> x == BoolPadStruct(0f0, false, 0f0), Array(a))
    end

    @testset "2D RGB array constant" begin
        a = Mantle.LavaArray(fill(RGB{Float32}(1,1,1), 16, 16))
        a .= RGB{Float32}(0.5, 0.5, 0.5)
        Mantle.vk_flush!(Mantle.vk_context())
        @test all(Array(a) .== RGB{Float32}(0.5, 0.5, 0.5))
    end

    @testset "Ref-wrapped struct with nonzero values" begin
        a = Mantle.LavaArray(fill(TestS12(0,0,0), 64))
        a .= Ref(TestS12(42, 43, 44))
        Mantle.vk_flush!(Mantle.vk_context())
        @test all(x -> x == TestS12(42,43,44), Array(a))
    end

    @testset "non-zero value preserves data" begin
        a = Mantle.LavaArray(fill(TestS12(0,0,0), 128))
        a .= Ref(TestS12(1.5f0, 2.5f0, 3.5f0))
        Mantle.vk_flush!(Mantle.vk_context())
        result = Array(a)
        @test result[1].a == 1.5f0
        @test result[1].b == 2.5f0
        @test result[1].c == 3.5f0
        @test result[128] == TestS12(1.5f0, 2.5f0, 3.5f0)
    end
end

@testset "Broadcast with closure-captured struct array" begin
    # Array-to-array broadcast through closure (exercises BDA struct copy pattern)
    src = Mantle.LavaArray(rand(Float32, 256))
    dst = Mantle.LavaArray(zeros(Float32, 256))
    dst .= src
    Mantle.vk_flush!(Mantle.vk_context())
    @test Array(dst) == Array(src)

    # Same with struct elements
    src_s = Mantle.LavaArray([TestS12(rand(Float32, 3)...) for _ in 1:64])
    dst_s = Mantle.LavaArray(fill(TestS12(0,0,0), 64))
    dst_s .= src_s
    Mantle.vk_flush!(Mantle.vk_context())
    @test Array(dst_s) == Array(src_s)
end
