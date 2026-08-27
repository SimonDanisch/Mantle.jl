# LavaInstanceRecord — a 64-byte plain-bits mirror of VkAccelerationStructureInstanceKHR.
#
# Field layout (matches Vulkan exactly):
#   bytes  0-47   transform: 12 × Float32 (row-major 3×4) — see `Mat3x4f`
#   bytes 48-50   instanceCustomIndex (24 bits)
#   byte  51      mask (8 bits)
#   bytes 52-54   instanceShaderBindingTableRecordOffset (24 bits)
#   byte  55      flags (8 bits)
#   bytes 56-63   accelerationStructureReference: UInt64 (BLAS device address)
#
# Plain-bits means LavaArray{LavaInstanceRecord, 1} is a clean GPU buffer.
# The two packed UInt32 fields combine the (24 + 8)-bit pairs Vulkan defines.

using StaticArrays: SMatrix

"""
    Mat3x4f

Vulkan-compatible 3×4 transform matrix, stored as `SMatrix{4, 3, Float32, 12}`.

The naming follows Vulkan's spec ("3×4 row-major float[12]"); the SMatrix
type parameters look transposed because `SMatrix` storage is column-major
and a column-major 4×3 has byte-for-byte the same layout as a row-major
3×4. That equivalence is the whole point — `LavaInstanceRecord.transform`
is read by the driver as `VkTransformMatrixKHR` without any reinterpret or
per-store conversion.

Construct from a `Mat4f` affine transform via [`mat4_to_vk_transform`](@ref):
the upper 3×4 of the 4×4 is repacked into the row-major Vulkan layout. Pure
linear indexing (`T[1]..T[12]`) returns the Vulkan T0..T11 floats in order,
which is what every kernel that reads transforms (GJK / EPA / narrow_phase /
instance_writer) actually wants.
"""
const Mat3x4f = SMatrix{4, 3, Float32, 12}

struct LavaInstanceRecord
    transform::Mat3x4f
    custom_index_and_mask::UInt32      # 24 bits index | 8 bits mask
    sbt_offset_and_flags::UInt32        # 24 bits sbt   | 8 bits flags
    blas_address::UInt64
end

"""
    LavaInstanceRecord(transform, blas_address;
                       custom_index=UInt32(0), mask=UInt8(0xff),
                       sbt_offset=UInt32(0), flags=UInt8(0)) -> LavaInstanceRecord

Construct a TLAS instance record. `transform` is a `Mat3x4f` (Vulkan-layout
row-major 3×4). `mask` is the cull mask the driver tests against `cullMask`
at trace time.
"""
function LavaInstanceRecord(transform::Mat3x4f, blas_address::UInt64;
                             custom_index::UInt32 = UInt32(0),
                             mask::UInt8 = UInt8(0xff),
                             sbt_offset::UInt32 = UInt32(0),
                             flags::UInt8 = UInt8(0))
    cim = (custom_index & 0x00FFFFFF) | (UInt32(mask) << 24)
    sof = (sbt_offset & 0x00FFFFFF) | (UInt32(flags) << 24)
    return LavaInstanceRecord(transform, cim, sof, blas_address)
end

# Backward-compatibility constructor: NTuple{12, Float32} → Mat3x4f.
# Existing GPU kernels (instance_writer.jl, narrow_phase.jl) build the
# 12-float row-major payload as a tuple; reinterpret as Mat3x4f is free.
function LavaInstanceRecord(transform::NTuple{12, Float32}, blas_address::UInt64;
                             custom_index::UInt32 = UInt32(0),
                             mask::UInt8 = UInt8(0xff),
                             sbt_offset::UInt32 = UInt32(0),
                             flags::UInt8 = UInt8(0))
    return LavaInstanceRecord(Mat3x4f(transform), blas_address;
                               custom_index, mask, sbt_offset, flags)
end

# Backward-compatibility for the packed-UInt32 form too (4-arg call).
function LavaInstanceRecord(transform::NTuple{12, Float32},
                             cim::UInt32, sof::UInt32, blas_address::UInt64)
    return LavaInstanceRecord(Mat3x4f(transform), cim, sof, blas_address)
end

"""
    identity_transform() -> Mat3x4f

Row-major 3×4 identity transform. Useful default for instance records that
want pure translation specified later.
"""
identity_transform() = Mat3x4f(1f0, 0f0, 0f0, 0f0,
                                0f0, 1f0, 0f0, 0f0,
                                0f0, 0f0, 1f0, 0f0)

# Sanity checks at module load time.
@assert sizeof(Mat3x4f) == 48              "Mat3x4f must be 48 bytes (12 × Float32)"
@assert isbitstype(Mat3x4f)                "Mat3x4f must be plain bits for kernel use"
@assert sizeof(LavaInstanceRecord) == 64   "LavaInstanceRecord must be exactly 64 bytes (Vulkan layout); got $(sizeof(LavaInstanceRecord))"
@assert isbitstype(LavaInstanceRecord)     "LavaInstanceRecord must be plain bits for LavaArray storage"
