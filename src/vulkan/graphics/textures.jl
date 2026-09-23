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

# `size` and `eltype` so a caller can ask whether new data FITS an existing
# texture without naming the backend's fields. `update_texture!` in RayMakie is
# the caller that matters: uploading into the texture it already has is what
# keeps a recorded plan's baked descriptor set pointing at the right image.
Base.size(t::VulkanTexture2D) = (t.width, t.height)
Base.eltype(::VulkanTexture2D{T}) where {T} = T

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

"""Create a 2D texture from a matrix of data.

`AbstractMatrix`, not `Matrix`: the pixels may already be ON the device — a
decoded video frame, the output of a compute pass — and a texture built from
those should not go out to the host and back. Which of the two it is, is
[`upload_texture_data!`](@ref)'s question, answered by dispatch.
"""
function VulkanTexture2D(data::AbstractMatrix{T}; ctx::VkContext, filter=:linear, wrap=:repeat) where T
    dev = ctx.device

    # `data[x, y]`: the FIRST index is the horizontal one, as on the Metal side and
    # as GLMakie's `Texture(::Matrix)` reads one.
    #
    # NOT `h, w = size(data)`: the upload below copies the column-major bytes
    # untouched, so a Julia COLUMN is one row of the image and the width has to
    # be `size(data, 1)`. The other way round the copy is right only when the two
    # are equal — a square glyph atlas — or when there is a single row, where the
    # whole matrix is one contiguous run either way. Anything else uploads
    # scrambled, an N-by-M texture reading its texels along the wrong stride.
    w, h = size(data)
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

"""
    copy_into_texture!(srcfor, tex)

Record `tex`'s upload: the two layout transitions and the buffer-to-image copy,
around whatever buffer `srcfor(e)` answers with.

ONE recording for both sources. `srcfor` returns `(VkBuffer, byte_offset)` and is
responsible for keeping whatever it named alive for the submission — it runs
INSIDE the one-shot, which is what lets the host path acquire its staging
scratch from the emitter that will own it.
"""
function copy_into_texture!(srcfor, tex::VulkanTexture2D)
    bq = tex.ctx.default_bq
    oneshot!(bq; tag = :upload) do e
    cmd = e.cmd
    staging_buf, src_offset = srcfor(e)

    transition_image!(cmd, tex.image,
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

    transition_image!(cmd, tex.image,
        VK.IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK.IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        VK.PIPELINE_STAGE_TRANSFER_BIT, VK.PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
        VK.ACCESS_TRANSFER_WRITE_BIT, VK.ACCESS_SHADER_READ_BIT)

    # The submission names the texture; the SOURCE was held by `srcfor`.
    hold!(e, tex)
    end
    # No wait: a draw that samples the texture on this queue is ordered behind
    # the copy.
    return nothing
end

"""Upload pixel data to a texture from the HOST, via staging."""
function upload_texture_data!(tex::VulkanTexture2D{T}, data::AbstractMatrix{T}) where T
    bytes = reinterpret(UInt8, vec(collect(data)))
    nbytes = length(bytes)
    # The bytes go into scratch the one-shot owns, and the sweep gives them back
    # once the copy has passed. Copied in before the submit, so the caller's
    # array is free the moment this returns.
    copy_into_texture!(tex) do e
        r = scratch!(e.owner, nbytes)
        mb = (memoryof(r)::BufferBlock).ref[]::VkManagedBuffer
        GC.@preserve bytes unsafe_copyto!(mb.mapped_ptr + offset(r), pointer(bytes), nbytes)
        hold!(e, mb)
        (mb.buffer, pool_offset(mb) + offset(r))
    end
end

"""
Upload pixel data that is ALREADY on the device: no staging, no host round trip.

`vkCmdCopyBufferToImage` reads the array's own buffer, so a frame produced by a
compute pass or a video decoder becomes a texture without ever being seen by the
host. A `LavaArray` carries a byte `offset` and no strides, so it is contiguous
by construction and the whole image is one region.

The ARRAY is held, not its `VkManagedBuffer`: what keeps the memory ours is the
`DataRef` refcount the array owns, not reachability of the buffer.
"""
function upload_texture_data!(tex::VulkanTexture2D{T}, data::LavaArray{T,2}) where T
    copy_into_texture!(tex) do e
        mb = data.buf[]::VkManagedBuffer
        hold!(e, data)
        (mb.buffer, pool_offset(mb) + data.offset)
    end
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
# On THIS backend's textures, spelled concretely: `SampledTexture` carries its
# types, so a Metal method sits beside this one without ambiguity.
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
