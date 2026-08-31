using Test, Lava, Mantle
using Mantle: identity_transform
using .MVE: VulkanInstanceRecord

@testset "VulkanInstanceRecord — size & isbits" begin
    @test sizeof(VulkanInstanceRecord) == 64
    @test isbitstype(VulkanInstanceRecord)
end

@testset "VulkanInstanceRecord — constructor packs custom_index + mask" begin
    rec = VulkanInstanceRecord(identity_transform(), UInt64(0xDEADBEEFCAFEBABE);
                              custom_index = UInt32(0x123456),
                              mask = UInt8(0x02))
    @test rec.blas_address == UInt64(0xDEADBEEFCAFEBABE)
    # custom_index in low 24 bits, mask in high 8 bits
    @test (rec.custom_index_and_mask & 0x00FFFFFF) == UInt32(0x123456)
    @test (rec.custom_index_and_mask >> 24) == UInt32(0x02)
end

@testset "VulkanInstanceRecord — byte layout matches pack_as_instance!" begin
    # Build the same instance via the existing CPU packer and via the new struct,
    # and compare byte-for-byte. This locks the layout to Vulkan's expectations.
    transform = (2f0, 0f0, 0f0, 1f0,
                 0f0, 2f0, 0f0, 2f0,
                 0f0, 0f0, 2f0, 3f0)
    blas_addr = UInt64(0x123456789ABCDEF0)
    custom_idx = UInt32(0x000042)
    mask_v = UInt8(0x04)
    sbt = UInt32(0x000007)
    flags_v = UInt8(0x01)

    # Packed via the struct
    rec = VulkanInstanceRecord(transform, blas_addr;
                              custom_index=custom_idx, mask=mask_v,
                              sbt_offset=sbt, flags=flags_v)
    rec_bytes = reinterpret(UInt8, [rec])

    # Packed via the existing pack_as_instance! helper
    expected = Vector{UInt8}(undef, 64)
    MVE.pack_as_instance!(expected, 0, blas_addr;
                            transform=transform,
                            custom_index=custom_idx,
                            mask=mask_v,
                            sbt_offset=sbt,
                            flags=flags_v)

    @test rec_bytes == expected
end

