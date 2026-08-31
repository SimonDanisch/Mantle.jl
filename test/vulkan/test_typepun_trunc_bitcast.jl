# Test: OpUConvert must always produce integer result type, even when the
# trunc's destination type is resolved to float by PTM cache contamination.
#
# Regression test for: trunc i64 → i32 followed by bitcast i32 → float
# was emitting OpUConvert %float (invalid) instead of OpUConvert %uint + OpBitcast %float.
# Root cause: map_type! cache mapped i32 → %float due to type-punned bitcast users.

using Test
using Lava, Mantle
using KernelAbstractions

ENV["VK_ICD_FILENAMES"] = get(ENV, "VK_ICD_FILENAMES", "/usr/share/vulkan/icd.d/lvp_icd.x86_64.json")

@testset "Type-punned trunc+bitcast SPIR-V emission" begin
    backend = MVE.LavaBackend()

    # Struct with mixed int/float fields packed into i64 words (like StaticMultiTypeSet)
    struct PackedFields
        data::NTuple{8, UInt64}  # packed bits
    end

    # Kernel that extracts a float from packed i64 via shift+trunc+bitcast
    # This is the pattern that triggered the bug in multitypeset dispatch
    @kernel function extract_float_kernel!(out, packed::PackedFields)
        i = @index(Global)
        # Load i64, shift right, trunc to i32, bitcast to float
        word = packed.data[1]
        hi_bits = word >> 32
        # This trunc+bitcast pattern triggers the bug:
        float_val = reinterpret(Float32, UInt32(hi_bits & 0xFFFFFFFF))
        out[i] = float_val
    end

    # Pack a known float value into the high 32 bits
    test_val = 3.14f0
    hi = UInt64(reinterpret(UInt32, test_val)) << 32
    packed = PackedFields(ntuple(i -> i == 1 ? hi : UInt64(0), 8))

    out = MVE.LavaArray{Float32}(undef, 4)
    fill!(out, 0f0)

    extract_float_kernel!(backend)(out, packed; ndrange=4)
    KernelAbstractions.synchronize(backend)

    result = Array(out)
    @test result[1] ≈ test_val atol=1f-6
    @test result[4] ≈ test_val atol=1f-6
end
