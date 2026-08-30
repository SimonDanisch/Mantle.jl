# Texture types and management for Lava.jl
#
# VkImage + VkSampler + descriptor set management for texture sampling in shaders.

"""2D texture backed by VkImage."""
struct VulkanTexture2D{T} <: Texture2D{T}
    image::VK.Image
    memory::VK.DeviceMemory
    view::VK.ImageView
    width::Int
    height::Int
    format::VK.Format
    ctx::VkContext
end

"""1D texture backed by VkImage."""
struct VulkanTexture1D{T} <: Texture1D{T}
    image::VK.Image
    memory::VK.DeviceMemory
    view::VK.ImageView
    width::Int
    format::VK.Format
    ctx::VkContext
end

"""Reusable sampler configuration."""
struct VulkanSampler <: Sampler
    handle::VK.Sampler
    filter::Symbol
    wrap::Symbol
    anisotropy::Float32
    ctx::VkContext
end

# `SampledTexture` is Mantle's: it pairs a `Texture` with a `Sampler`, both of
# which are now the API's abstract types, so there is nothing backend-specific
# left in the pairing itself.

# ── Sampler Construction ──

function VulkanSampler(; ctx::VkContext=vk_context(), filter::Symbol=:linear, wrap::Symbol=:repeat, anisotropy::Real=0.0f0)
    dev = ctx.device

    vk_filter = filter == :nearest ? VK.FILTER_NEAREST :
                filter == :linear  ? VK.FILTER_LINEAR :
                error("Unknown filter: $filter")

    vk_wrap = wrap == :repeat      ? VK.SAMPLER_ADDRESS_MODE_REPEAT :
              wrap == :clamp       ? VK.SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE :
              wrap == :mirror      ? VK.SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT :
              wrap == :clamp_border ? VK.SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER :
              error("Unknown wrap mode: $wrap")

    sampler = VK.Sampler(dev,
        vk_filter, vk_filter,
        VK.SAMPLER_MIPMAP_MODE_LINEAR,
        vk_wrap, vk_wrap, vk_wrap,
        0.0f0,  # mip LOD bias
        anisotropy > 0,
        Float32(anisotropy),
        false, VK.COMPARE_OP_ALWAYS,
        0.0f0, 0.0f0,
        VK.BORDER_COLOR_FLOAT_TRANSPARENT_BLACK,
        false,
    )

    VulkanSampler(sampler, filter, wrap, Float32(anisotropy), ctx)
end

# ── Texture Construction ──

"""Create a 2D texture from a matrix of data."""
function VulkanTexture2D(data::Matrix{T}; ctx::VkContext=vk_context(), filter=:linear, wrap=:repeat) where T
    dev = ctx.device

    h, w = size(data)
    format = julia_to_vk_format(T)

    image = VK.Image(dev,
        VK.IMAGE_TYPE_2D, format,
        VK.Extent3D(UInt32(w), UInt32(h), UInt32(1)),
        UInt32(1), UInt32(1),
        VK.SAMPLE_COUNT_1_BIT,
        VK.IMAGE_TILING_OPTIMAL,
        VK.IMAGE_USAGE_SAMPLED_BIT | VK.IMAGE_USAGE_TRANSFER_DST_BIT,
        VK.SHARING_MODE_EXCLUSIVE, UInt32[],
        VK.IMAGE_LAYOUT_UNDEFINED,
    )

    memory = alloc_image_memory(ctx, image)

    view = VK.ImageView(dev, image, VK.IMAGE_VIEW_TYPE_2D, format,
        VK.ComponentMapping(
            VK.COMPONENT_SWIZZLE_IDENTITY, VK.COMPONENT_SWIZZLE_IDENTITY,
            VK.COMPONENT_SWIZZLE_IDENTITY, VK.COMPONENT_SWIZZLE_IDENTITY),
        VK.ImageSubresourceRange(VK.IMAGE_ASPECT_COLOR_BIT,
            UInt32(0), UInt32(1), UInt32(0), UInt32(1)))

    tex = VulkanTexture2D{T}(image, memory, view, w, h, format, ctx)

    # Upload data
    upload_texture_data!(tex, data)

    return tex
end

"""Upload pixel data to a texture via staging buffer."""
function upload_texture_data!(tex::VulkanTexture2D{T}, data::Matrix{T}) where T
    ctx = tex.ctx
    bq = ctx.default_bq
    dev = ctx.device

    # Use staging buffer for upload
    bytes = reinterpret(UInt8, vec(collect(data)))
    nbytes = length(bytes)
    staging_buf, _, mapped_ptr, _ = get_staging(bq, nbytes)
    unsafe_copyto!(Ptr{UInt8}(mapped_ptr), pointer(bytes), nbytes)

    # Record image upload into the active batch.  Staging buffer is owned
    # by `bq.staging`; no pin required (the field holds a strong ref).
    batch = ensure_active_batch!(bq)
    cmd = batch.cmd_buf

    transition_image!(cmd, tex.image,
        VK.IMAGE_LAYOUT_UNDEFINED, VK.IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        VK.PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK.PIPELINE_STAGE_TRANSFER_BIT,
        VK.AccessFlag(0), VK.ACCESS_TRANSFER_WRITE_BIT)

    region = VK.BufferImageCopy(
        UInt64(0), UInt32(0), UInt32(0),
        VK.ImageSubresourceLayers(VK.IMAGE_ASPECT_COLOR_BIT,
            UInt32(0), UInt32(0), UInt32(1)),
        VK.Offset3D(0, 0, 0),
        VK.Extent3D(UInt32(tex.width), UInt32(tex.height), UInt32(1)),
    )
    VK.cmd_copy_buffer_to_image(cmd, staging_buf, tex.image,
        VK.IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, [region])

    transition_image!(cmd, tex.image,
        VK.IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK.IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        VK.PIPELINE_STAGE_TRANSFER_BIT, VK.PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
        VK.ACCESS_TRANSFER_WRITE_BIT, VK.ACCESS_SHADER_READ_BIT)

    # Pin the texture so the batch keeps it alive until the fence.
    pin!(batch, tex)
    # Flush now — upload!(tex) is a synchronous API (caller expects the texture
    # to be ready on return).
    flush!(bq, dev)
end

# ── Format Mapping ──

function julia_to_vk_format(::Type{T}) where T
    T == Float32      ? VK.FORMAT_R32_SFLOAT :
    T == NTuple{4, Float32} ? VK.FORMAT_R32G32B32A32_SFLOAT :
    T == NTuple{3, Float32} ? VK.FORMAT_R32G32B32_SFLOAT :
    T == NTuple{2, Float32} ? VK.FORMAT_R32G32_SFLOAT :
    T == UInt8        ? VK.FORMAT_R8_UNORM :
    T == NTuple{4, UInt8} ? VK.FORMAT_R8G8B8A8_UNORM :
    error("No VkFormat mapping for Julia type: $T")
end

# ── Descriptor Set for Textures ──

struct VulkanTextureBindings <: TextureBindings
    layout::VK.DescriptorSetLayout
    pool::VK.DescriptorPool
    set::VK.DescriptorSet
    textures::Vector{Any}  # Keep texture + sampler refs alive while descriptor set is in use
end

"""Create a descriptor set binding combined image samplers."""
function bind_textures(textures::Vector{<:SampledTexture})
    isempty(textures) && error("bind_textures: cannot bind an empty texture list")
    ctx = textures[1].texture.ctx
    dev = ctx.device

    n = length(textures)
    bindings = [VK.DescriptorSetLayoutBinding(
        UInt32(i - 1),
        VK.DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        VK.SHADER_STAGE_FRAGMENT_BIT | VK.SHADER_STAGE_VERTEX_BIT;
        descriptor_count=UInt32(1),
    ) for i in 1:n]

    layout = VK.DescriptorSetLayout(dev, bindings)

    pool = VK.DescriptorPool(dev, UInt32(1),
        [VK.DescriptorPoolSize(VK.DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, UInt32(n))])

    alloc_info = VK.DescriptorSetAllocateInfo(pool, [layout])
    sets = @vk_checked "vkAllocateDescriptorSets (textures)" VK.allocate_descriptor_sets(dev, alloc_info)
    dset = sets[1]

    # Write descriptors
    writes = VK.WriteDescriptorSet[]
    for (i, st) in enumerate(textures)
        img_info = VK.DescriptorImageInfo(
            st.sampler.handle,
            st.texture.view,
            VK.IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        )
        push!(writes, VK.WriteDescriptorSet(
            dset, UInt32(i - 1), UInt32(0),
            VK.DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            [img_info],
            VK.DescriptorBufferInfo[],
            VK.BufferView[];
            descriptor_count=UInt32(1),
        ))
    end
    VK.update_descriptor_sets(dev, writes, [])

    VulkanTextureBindings(layout, pool, dset, Any[textures...])
end

# Convenience
Base.:*(tex::Texture, sam::VulkanSampler) = SampledTexture(tex, sam)
LavaTexture(data::Matrix{T}; kw...) where T = VulkanTexture2D(data; kw...)
