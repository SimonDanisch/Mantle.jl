# VulkanInstanceRecord — a 64-byte plain-bits mirror of VkAccelerationStructureInstanceKHR.
#
# Field layout (matches Vulkan exactly):
#   bytes  0-47   transform: 12 × Float32 (row-major 3×4) — see `Mat3x4f`
#   bytes 48-50   instanceCustomIndex (24 bits)
#   byte  51      mask (8 bits)
#   bytes 52-54   instanceShaderBindingTableRecordOffset (24 bits)
#   byte  55      flags (8 bits)
#   bytes 56-63   accelerationStructureReference: UInt64 (BLAS device address)
#
# Plain-bits means LavaArray{VulkanInstanceRecord, 1} is a clean GPU buffer.
# The two packed UInt32 fields combine the (24 + 8)-bit pairs Vulkan defines.

using StaticArrays: SMatrix

# `Mat3x4f` and `identity_transform` are Mantle's, in `geometry/transform.jl`:
# the 3×4 affine is common to every backend's instance descriptor, while the
# record below is a byte-exact mirror of one API's. Construct the transform from
# a `Mat4f` via [`mat4_to_vk_transform`](@ref), which repacks the upper 3×4 into
# the row-major Vulkan layout.

struct VulkanInstanceRecord
    transform::Mat3x4f
    custom_index_and_mask::UInt32      # 24 bits index | 8 bits mask
    sbt_offset_and_flags::UInt32        # 24 bits sbt   | 8 bits flags
    blas_address::UInt64
end

"""
    VulkanInstanceRecord(transform, blas_address;
                       custom_index=UInt32(0), mask=UInt8(0xff),
                       sbt_offset=UInt32(0), flags=UInt8(0)) -> VulkanInstanceRecord

Construct a HWTLAS instance record. `transform` is a `Mat3x4f` (Vulkan-layout
row-major 3×4). `mask` is the cull mask the driver tests against `cullMask`
at trace time.
"""
function VulkanInstanceRecord(transform::Mat3x4f, blas_address::UInt64;
                             custom_index::UInt32 = UInt32(0),
                             mask::UInt8 = UInt8(0xff),
                             sbt_offset::UInt32 = UInt32(0),
                             flags::UInt8 = UInt8(0))
    cim = (custom_index & 0x00FFFFFF) | (UInt32(mask) << 24)
    sof = (sbt_offset & 0x00FFFFFF) | (UInt32(flags) << 24)
    return VulkanInstanceRecord(transform, cim, sof, blas_address)
end

# Backward-compatibility constructor: NTuple{12, Float32} → Mat3x4f.
# Existing GPU kernels (instance_writer.jl, narrow_phase.jl) build the
# 12-float row-major payload as a tuple; reinterpret as Mat3x4f is free.
function VulkanInstanceRecord(transform::NTuple{12, Float32}, blas_address::UInt64;
                             custom_index::UInt32 = UInt32(0),
                             mask::UInt8 = UInt8(0xff),
                             sbt_offset::UInt32 = UInt32(0),
                             flags::UInt8 = UInt8(0))
    return VulkanInstanceRecord(Mat3x4f(transform), blas_address;
                               custom_index, mask, sbt_offset, flags)
end

# Backward-compatibility for the packed-UInt32 form too (4-arg call).
function VulkanInstanceRecord(transform::NTuple{12, Float32},
                             cim::UInt32, sof::UInt32, blas_address::UInt64)
    return VulkanInstanceRecord(Mat3x4f(transform), cim, sof, blas_address)
end

# Sanity checks at module load time. The `Mat3x4f` half of these lives beside
# the type in `geometry/transform.jl`; what is asserted here is the part this
# file is responsible for, which is the Vulkan record's own layout.
@assert sizeof(VulkanInstanceRecord) == 64   "VulkanInstanceRecord must be exactly 64 bytes (Vulkan layout); got $(sizeof(VulkanInstanceRecord))"
@assert isbitstype(VulkanInstanceRecord)     "VulkanInstanceRecord must be plain bits for LavaArray storage"
