# Textures, samplers and the descriptor sets that bind them.

"""2D texture backed by VkImage, with `levels` mip levels (1: none)."""
struct VulkanTexture2D{T} <: Texture2D{T}
    image::VK.Image
    memory::VK.DeviceMemory
    view::VK.ImageView
    width::Int
    height::Int
    format::VK.Format
    levels::Int
    ctx::VkContext
end

# Lets a caller check that new data fits before uploading in place, which keeps a
# recorded plan's baked descriptor set pointing at the right image.
Base.size(t::VulkanTexture2D) = (t.width, t.height)
Base.eltype(::VulkanTexture2D{T}) where {T} = T
mipmapped(t::VulkanTexture2D) = t.levels > 1

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

function VulkanSampler(; ctx::VkContext, filter::Symbol=:linear, wrap::Symbol=:repeat, anisotropy::Real=0.0f0)
    dev = ctx.device

    vk_filter = filter == :nearest ? VK.FILTER_NEAREST :
                filter == :linear  ? VK.FILTER_LINEAR :
                error("Unknown filter: $filter")

    vk_wrap = wrap == :repeat      ? VK.SAMPLER_ADDRESS_MODE_REPEAT :
              wrap == :clamp       ? VK.SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE :
              wrap == :mirror      ? VK.SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT :
              wrap == :clamp_border ? VK.SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER :
              error("Unknown wrap mode: $wrap")

    # No upper LOD clamp, so a mipmapped texture samples every level; one
    # without mipmaps has only level 0.
    sampler = VK.Sampler(dev,
        vk_filter, vk_filter,
        VK.SAMPLER_MIPMAP_MODE_LINEAR,
        vk_wrap, vk_wrap, vk_wrap,
        0.0f0,  # mip LOD bias
        anisotropy > 0,
        Float32(anisotropy),
        false, VK.COMPARE_OP_ALWAYS,
        0.0f0, VK.LOD_CLAMP_NONE,
        VK.BORDER_COLOR_FLOAT_TRANSPARENT_BLACK,
        false,
    )

    VulkanSampler(sampler, filter, wrap, Float32(anisotropy), ctx)
end

"""
    VulkanTexture2D(data::AbstractMatrix; ctx, filter=:linear, wrap=:repeat, mipmap=false)

A 2D texture from `data[x, y]`, which may already be on the device
([`upload_texture_data!`](@ref) dispatches on it). `mipmap = true` builds the full
mip chain on the device at every upload, so a texture shown smaller than its size
does not alias; off by default because it mixes neighbouring glyphs in an atlas.
"""
function VulkanTexture2D(data::AbstractMatrix{T}; ctx::VkContext, filter=:linear, wrap=:repeat, mipmap::Bool=false) where T
    dev = ctx.device

    # The first index is horizontal (as on Metal and in GLMakie): the upload copies
    # the column-major bytes as they are, so a Julia column is one image row.
    w, h = size(data)
    format = julia_to_vk_format(T)
    levels = mipmap ? floor(Int, log2(max(w, h))) + 1 : 1
    mipmap && checkmipmappable(ctx, format)

    image = VK.Image(dev,
        VK.IMAGE_TYPE_2D, format,
        VK.Extent3D(UInt32(w), UInt32(h), UInt32(1)),
        UInt32(levels), UInt32(1),
        VK.SAMPLE_COUNT_1_BIT,
        VK.IMAGE_TILING_OPTIMAL,
        VK.IMAGE_USAGE_SAMPLED_BIT | VK.IMAGE_USAGE_TRANSFER_DST_BIT |
            (mipmap ? VK.IMAGE_USAGE_TRANSFER_SRC_BIT : VK.ImageUsageFlag(0)),
        VK.SHARING_MODE_EXCLUSIVE, UInt32[],
        VK.IMAGE_LAYOUT_UNDEFINED,
    )

    memory = alloc_image_memory(ctx, image)

    view = VK.ImageView(dev, image, VK.IMAGE_VIEW_TYPE_2D, format,
        VK.ComponentMapping(
            VK.COMPONENT_SWIZZLE_IDENTITY, VK.COMPONENT_SWIZZLE_IDENTITY,
            VK.COMPONENT_SWIZZLE_IDENTITY, VK.COMPONENT_SWIZZLE_IDENTITY),
        VK.ImageSubresourceRange(VK.IMAGE_ASPECT_COLOR_BIT,
            UInt32(0), UInt32(levels), UInt32(0), UInt32(1)))

    tex = VulkanTexture2D{T}(image, memory, view, w, h, format, levels, ctx)

    upload_texture_data!(tex, data)

    return tex
end

"""Throw unless `format` supports the linear-filtered blits that make mip levels."""
function checkmipmappable(ctx::VkContext, format::VK.Format)
    features = VK.get_physical_device_format_properties(ctx.physical_device, format).optimal_tiling_features
    needed = VK.FORMAT_FEATURE_BLIT_SRC_BIT | VK.FORMAT_FEATURE_BLIT_DST_BIT | VK.FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT
    needed & features == needed ||
        throw(ArgumentError("mipmap = true: $format cannot be blitted with linear filtering on $(ctx.device_name)"))
    return nothing
end

"""A barrier taking mip levels `first:first+count-1` of `image` from one layout to another."""
function transition_levels!(cmd, image::VK.Image, first::Integer, count::Integer, old_layout, new_layout,
                            src_stage, dst_stage, src_access, dst_access)
    barrier = VK.ImageMemoryBarrier(src_access, dst_access, old_layout, new_layout,
        VK.QUEUE_FAMILY_IGNORED, VK.QUEUE_FAMILY_IGNORED, image,
        VK.ImageSubresourceRange(VK.IMAGE_ASPECT_COLOR_BIT, UInt32(first), UInt32(count), UInt32(0), UInt32(1)))
    VK.cmd_pipeline_barrier(cmd, [], [], [barrier]; src_stage_mask = src_stage, dst_stage_mask = dst_stage)
end

"""
Fill levels 1 and up of `tex` from level 0, each a linear blit of the one before,
and leave every level ready to sample. Level 0 arrives in TRANSFER_DST.
"""
function record_mipmaps!(cmd, tex::VulkanTexture2D)
    w, h = tex.width, tex.height
    for level in 1:tex.levels-1
        transition_levels!(cmd, tex.image, level - 1, 1,
            VK.IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK.IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            VK.PIPELINE_STAGE_TRANSFER_BIT, VK.PIPELINE_STAGE_TRANSFER_BIT,
            VK.ACCESS_TRANSFER_WRITE_BIT, VK.ACCESS_TRANSFER_READ_BIT)
        w2, h2 = max(1, w ÷ 2), max(1, h ÷ 2)
        blit = VK.ImageBlit(
            VK.ImageSubresourceLayers(VK.IMAGE_ASPECT_COLOR_BIT, UInt32(level - 1), UInt32(0), UInt32(1)),
            (VK.Offset3D(0, 0, 0), VK.Offset3D(w, h, 1)),
            VK.ImageSubresourceLayers(VK.IMAGE_ASPECT_COLOR_BIT, UInt32(level), UInt32(0), UInt32(1)),
            (VK.Offset3D(0, 0, 0), VK.Offset3D(w2, h2, 1)))
        VK.cmd_blit_image(cmd, tex.image, VK.IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            tex.image, VK.IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, [blit], VK.FILTER_LINEAR)
        transition_levels!(cmd, tex.image, level - 1, 1,
            VK.IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK.IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK.PIPELINE_STAGE_TRANSFER_BIT, VK.PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            VK.ACCESS_TRANSFER_READ_BIT, VK.ACCESS_SHADER_READ_BIT)
        w, h = w2, h2
    end
    transition_levels!(cmd, tex.image, tex.levels - 1, 1,
        VK.IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK.IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        VK.PIPELINE_STAGE_TRANSFER_BIT, VK.PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
        VK.ACCESS_TRANSFER_WRITE_BIT, VK.ACCESS_SHADER_READ_BIT)
    return nothing
end

"""
    copy_into_texture!(srcfor, tex)

Record `tex`'s upload in one one-shot: the transitions, the copy into level 0 from
`srcfor(e) -> (VkBuffer, byte_offset)`, then the mip levels. `srcfor` runs inside
the one-shot and must hold whatever it names.
"""
function copy_into_texture!(srcfor, tex::VulkanTexture2D)
    bq = tex.ctx.default_bq
    oneshot!(bq; tag = :upload) do e
    cmd = e.cmd
    staging_buf, src_offset = srcfor(e)

    transition_levels!(cmd, tex.image, 0, tex.levels,
        VK.IMAGE_LAYOUT_UNDEFINED, VK.IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        VK.PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK.PIPELINE_STAGE_TRANSFER_BIT,
        VK.AccessFlag(0), VK.ACCESS_TRANSFER_WRITE_BIT)

    region = VK.BufferImageCopy(
        UInt64(src_offset), UInt32(0), UInt32(0),
        VK.ImageSubresourceLayers(VK.IMAGE_ASPECT_COLOR_BIT,
            UInt32(0), UInt32(0), UInt32(1)),
        VK.Offset3D(0, 0, 0),
        VK.Extent3D(UInt32(tex.width), UInt32(tex.height), UInt32(1)),
    )
    VK.cmd_copy_buffer_to_image(cmd, staging_buf, tex.image,
        VK.IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, [region])

    record_mipmaps!(cmd, tex)

    # The source was held by `srcfor`.
    hold!(e, tex)
    end
    # No wait: a draw that samples the texture on this queue is ordered behind
    # the copy.
    return nothing
end

"""Upload host pixel data through staging scratch the one-shot owns; `data` is free
again when this returns."""
function upload_texture_data!(tex::VulkanTexture2D{T}, data::AbstractMatrix{T}) where T
    # The copy covers the texture's extent; smaller data would be read past its end.
    size(data) == size(tex) || throw(DimensionMismatch(
        "upload_texture_data!: $(size(data)) texels for a $(size(tex)) texture"))
    bytes = reinterpret(UInt8, vec(collect(data)))
    nbytes = length(bytes)
    copy_into_texture!(tex) do e
        r = scratch!(e.owner, nbytes)
        mb = (memoryof(r)::BufferBlock).ref[]::VkManagedBuffer
        GC.@preserve bytes unsafe_copyto!(mb.mapped_ptr + offset(r), pointer(bytes), nbytes)
        hold!(e, mb)
        (mb.buffer, pool_offset(mb) + offset(r))
    end
end

"""
Upload pixel data already on the device, without staging: the copy reads the
array's own buffer, contiguous because a `LavaArray` has an offset and no strides.
The array is held, not its `VkManagedBuffer`: its `DataRef` refcount keeps the
memory alive.
"""
function upload_texture_data!(tex::VulkanTexture2D{T}, data::LavaArray{T,2}) where T
    # Otherwise the GPU reads past the array's end.
    size(data) == size(tex) || throw(DimensionMismatch(
        "upload_texture_data!: $(size(data)) texels for a $(size(tex)) texture"))
    copy_into_texture!(tex) do e
        mb = data.buf[]::VkManagedBuffer
        hold!(e, data)
        (mb.buffer, pool_offset(mb) + data.offset)
    end
end

function julia_to_vk_format(::Type{T}) where T
    T == Float32      ? VK.FORMAT_R32_SFLOAT :
    T == NTuple{4, Float32} ? VK.FORMAT_R32G32B32A32_SFLOAT :
    T == NTuple{3, Float32} ? VK.FORMAT_R32G32B32_SFLOAT :
    T == NTuple{2, Float32} ? VK.FORMAT_R32G32_SFLOAT :
    T == UInt8        ? VK.FORMAT_R8_UNORM :
    T == NTuple{4, UInt8} ? VK.FORMAT_R8G8B8A8_UNORM :
    error("No VkFormat mapping for Julia type: $T")
end

struct VulkanTextureBindings <: TextureBindings
    layout::VK.DescriptorSetLayout
    pool::VK.DescriptorPool
    set::VK.DescriptorSet
    textures::Vector{Any}  # Keep texture + sampler refs alive while descriptor set is in use
end

"""Create a descriptor set binding combined image samplers. Typed on this backend's
textures so a Metal method sits beside it without ambiguity."""
function bind_textures(textures::Vector{<:SampledTexture{<:Any,<:Any,<:VulkanTexture2D}})
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

Base.:*(tex::Texture, sam::VulkanSampler) = SampledTexture(tex, sam)
LavaTexture(data::Matrix{T}; kw...) where T = VulkanTexture2D(data; kw...)
