# VulkanInstanceRecord: a 64-byte plain-bits mirror of VkAccelerationStructureInstanceKHR.
#   bytes  0-47   transform, 12 Float32, row-major 3x4 (`Mat3x4f`)
#   bytes 48-51   instanceCustomIndex (low 24 bits) | mask (high 8)
#   bytes 52-55   instanceShaderBindingTableRecordOffset (low 24) | flags (high 8)
#   bytes 56-63   accelerationStructureReference (BLAS device address)
# Only this backend sees one: `vkinstances_kernel!` (`hwtlas.jl`) writes them from
# the portable `Raycore.InstanceRecord`s at every build and refit.

using StaticArrays: SMatrix

# `Mat3x4f` is core's (`geometry/transform.jl`); `mat4_to_vk_transform` makes one
# from a `Mat4f`.

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

`transform` is a row-major 3x4 `Mat3x4f`; `mask` is tested against the ray's cull
mask. `custom_index` and `sbt_offset` keep their low 24 bits.
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

# Layout checks; those for `Mat3x4f` live in `geometry/transform.jl`.
@assert sizeof(VulkanInstanceRecord) == 64   "VulkanInstanceRecord must be exactly 64 bytes (Vulkan layout); got $(sizeof(VulkanInstanceRecord))"
@assert isbitstype(VulkanInstanceRecord)     "VulkanInstanceRecord must be plain bits for LavaArray storage"
