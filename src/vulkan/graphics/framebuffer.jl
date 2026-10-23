# Offscreen render targets: images and views for dynamic rendering, no
# VkRenderPass or VkFramebuffer.

"""
    VulkanFramebuffer

Offscreen render target with a color and an optional depth image.
"""
mutable struct VulkanFramebuffer <: Framebuffer
    width::Int
    height::Int
    color_image::VK.Image
    color_memory::VK.DeviceMemory
    color_view::VK.ImageView
    color_format::VK.Format
    # Depth attachment, or nothing.
    depth_image::Union{Nothing, VK.Image}
    depth_memory::Union{Nothing, VK.DeviceMemory}
    depth_view::Union{Nothing, VK.ImageView}
    depth_format::VK.Format
    ctx::LavaDevice
end

"""Default usage of a framebuffer's color attachment."""
const COLOR_USAGE = VK.IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                    VK.IMAGE_USAGE_TRANSFER_SRC_BIT |
                    VK.IMAGE_USAGE_SAMPLED_BIT

"""
    image_2d(ctx, width, height, format, usage; levels = 1) -> Image

A 2D image with no memory bound. Bind it with `alloc_image_memory` for one of its
own, or with `bind_image!` to place it in a shared allocation.
"""
image_2d(ctx::LavaDevice, width::Integer, height::Integer, format::VK.Format,
         usage::VK.ImageUsageFlag; levels::Integer = 1) =
    VK.Image(ctx.device, VK.IMAGE_TYPE_2D, format,
        VK.Extent3D(UInt32(width), UInt32(height), UInt32(1)), UInt32(levels), UInt32(1),
        VK.SAMPLE_COUNT_1_BIT, VK.IMAGE_TILING_OPTIMAL, usage,
        VK.SHARING_MODE_EXCLUSIVE, UInt32[], VK.IMAGE_LAYOUT_UNDEFINED)

"""A 2D view of `levels` mip levels. Must be created after the image has memory bound."""
image_view(ctx::LavaDevice, image::VK.Image, format::VK.Format,
           aspect::VK.ImageAspectFlag = VK.IMAGE_ASPECT_COLOR_BIT; levels::Integer = 1) =
    VK.ImageView(ctx.device, image, VK.IMAGE_VIEW_TYPE_2D, format,
        VK.ComponentMapping(VK.COMPONENT_SWIZZLE_IDENTITY, VK.COMPONENT_SWIZZLE_IDENTITY,
                            VK.COMPONENT_SWIZZLE_IDENTITY, VK.COMPONENT_SWIZZLE_IDENTITY),
        VK.ImageSubresourceRange(aspect, UInt32(0), UInt32(levels), UInt32(0), UInt32(1)))

"""
    VulkanFramebuffer(width, height; ctx, depth=true, color_format=VK.FORMAT_B8G8R8A8_SRGB, srgb=false)

An offscreen framebuffer with a color and an optional `D32_SFLOAT` depth attachment.
`color_format` may also be a Julia element type (`BGRA{N0f8}`, `RGBA{Float32}`),
lowered by `vkformat` as for transient images.
"""
function VulkanFramebuffer(width::Integer, height::Integer;
                          ctx::LavaDevice,
                          depth::Bool=true,
                          color_format::Union{VK.Format,Type}=VK.FORMAT_B8G8R8A8_SRGB,
                          srgb::Bool=false)
    color_format isa Type && (color_format = vkformat(VulkanAPI(), color_format; srgb))
    color_image = image_2d(ctx, width, height, color_format, COLOR_USAGE)
    color_memory = alloc_image_memory(ctx, color_image)
    color_view = image_view(ctx, color_image, color_format)

    depth_format = VK.FORMAT_D32_SFLOAT
    depth_img = nothing
    depth_mem = nothing
    depth_vw = nothing
    if depth
        depth_img = image_2d(ctx, width, height, depth_format,
                             VK.IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT)
        depth_mem = alloc_image_memory(ctx, depth_img)
        depth_vw = image_view(ctx, depth_img, depth_format, VK.IMAGE_ASPECT_DEPTH_BIT)
    end

    VulkanFramebuffer(Int(width), Int(height),
        color_image, color_memory, color_view, color_format,
        depth_img, depth_mem, depth_vw, depth_format,
        ctx)
end

"""
    image_requirements(ctx, image) -> (; size, alignment, type_bits)

What an image needs from an allocation, for choosing offsets when several
resources share one.
"""
function image_requirements(ctx::LavaDevice, image::VK.Image)
    r = VK.get_image_memory_requirements(ctx.device, image)
    (size = Int(r.size), alignment = Int(r.alignment), type_bits = r.memory_type_bits)
end

"""
    device_memory(ctx, bytes, type_bits) -> DeviceMemory

Device-local memory that can back any image whose requirements include
`type_bits`. `type_bits` is the *intersection* over everything to be placed in it.
"""
function device_memory(ctx::LavaDevice, bytes::Integer, type_bits::Integer)
    type_bits == 0 && error("no memory type satisfies every resource in this allocation")
    idx = find_memory_type(ctx, UInt32(type_bits), VK.MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
    # Always `MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT`: this also backs graph arenas,
    # whose buffers kernels address by device address, and the spec requires the
    # flag for that. Without it RADV segfaults much later, in
    # `vkCreateRayTracingPipelinesKHR` (NVIDIA tolerates it).
    flags = VK.MemoryAllocateFlagsInfo(UInt32(0);
        flags = VK.MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT)
    # Over the driver's budget is out of memory (`overbudget`). Thrown as the
    # driver's `VulkanError`, because `acquire_or_reclaim!` retries only on an
    # out-of-memory `VulkanError`.
    overbudget(ctx, idx, bytes) && throw(VK.VulkanError(
        format_oom_error(ctx, AllocFailure(VK.ERROR_OUT_OF_DEVICE_MEMORY, :budget, Int(bytes), Int(idx))),
        VK.ERROR_OUT_OF_DEVICE_MEMORY))
    VK.DeviceMemory(ctx.device, UInt64(bytes), idx; next = flags)
end

"""
    bind_image!(ctx, image, memory, offset)

Bind an image to a byte offset in an existing allocation. Two images may share
bytes only if their contents never have to survive each other; the caller is
responsible for that and for transitioning the second from `UNDEFINED`.
"""
bind_image!(ctx::LavaDevice, image::VK.Image, memory::VK.DeviceMemory, offset::Integer) =
    unwrap(VK.bind_image_memory(ctx.device, image, memory, UInt64(offset)))

"""
    unbound_buffer(ctx, bytes, usage) -> Buffer

A `VkBuffer` with no memory bound, for placing into an allocation someone else
owns with [`bind_buffer!`](@ref).
"""
unbound_buffer(ctx::LavaDevice, bytes::Integer, usage::VK.BufferUsageFlag) =
    VK.Buffer(ctx.device, UInt64(max(bytes, 1)), usage,
                  VK.SHARING_MODE_EXCLUSIVE, UInt32[])

"""Memory requirements of `buf`, shaped like [`image_requirements`](@ref)."""
function buffer_requirements(ctx::LavaDevice, buf::VK.Buffer)
    r = VK.get_buffer_memory_requirements(ctx.device, buf)
    (size = Int(r.size), alignment = Int(r.alignment), type_bits = r.memory_type_bits)
end

"""
    bind_buffer!(ctx, buffer, memory, offset)

Bind a buffer to a byte offset in an existing allocation, under the rule of
[`bind_image!`](@ref): resources may share bytes only if their contents never have
to survive each other, which the caller guarantees.
"""
bind_buffer!(ctx::LavaDevice, buf::VK.Buffer, memory::VK.DeviceMemory, offset::Integer) =
    throw_if_error(ctx, "vkBindBufferMemory",
                   VK.bind_buffer_memory(ctx.device, buf, memory, UInt64(offset)))

"""Allocate device-local memory for an image and bind it."""
function alloc_image_memory(ctx::LavaDevice, image::VK.Image)
    req = image_requirements(ctx, image)
    mem = device_memory(ctx, req.size, req.type_bits)
    bind_image!(ctx, image, mem, 0)
    return mem
end

Base.size(fb::VulkanFramebuffer) = (fb.width, fb.height)

"""
    readback_framebuffer(fb::VulkanFramebuffer) -> Matrix

The color attachment as a `(width, height)` host matrix of
`eltypeof(fb.color_format)`, a ColorTypes colorant as on Metal. The first index is
x, so an image indexed `(row, column)` needs `permutedims`.
"""
function readback_framebuffer(fb::VulkanFramebuffer)
    bq = fb.ctx.default_bq
    return readbackimage(fb.ctx, bq, eltypeof(fb.color_format), fb.width, fb.height) do mb, off
        oneshot!(bq; tag = :readback) do e
            copyattachment!(e.cmd, fb, mb.buffer, off)
            hold!(e, mb)
            hold!(e, fb)
        end
    end
end

"""
    readbackimage(f, dev, bq, T, w, h) -> Matrix{T}

Read a `w × h` image of `T` back through a host-cached `Readback` region (reading
the BAR-backed `Unified` arena runs at write-combined speed). `f(buffer, offset)`
records the copy into the region and returns the token `bq` waits on.
"""
function readbackimage(f, dev, bq, ::Type{T}, w::Integer, h::Integer) where {T}
    nbytes = w * h * pixelbytes(T)
    r = acquire!(pool(dev), dev, Readback(), nothing, max(Int(nbytes), 16);
                 align = ARG_ALIGN, blocksize = READBACK_BLOCK_SIZE)
    mb = (memoryof(r)::BufferBlock).ref[]::VkManagedBuffer
    waitfor!(bq, f(mb, pool_offset(mb) + offset(r)))
    pixels = Matrix{T}(undef, w, h)
    unsafe_copyto!(Ptr{UInt8}(pointer(pixels)), mb.mapped_ptr + offset(r), nbytes)
    release!(r)
    return pixels
end

"""
    copyimage!(cmd, buffer, offset, image, (x, y, w, h); aspect = COLOR)

Record a tightly packed copy of a region of `image`'s first mip level and layer,
which must be in `TRANSFER_SRC_OPTIMAL`, to `buffer` at byte `offset`.
"""
function copyimage!(cmd, buffer::VK.Buffer, offset::Integer, image::VK.Image, (x, y, w, h);
                    aspect::VK.ImageAspectFlag = VK.IMAGE_ASPECT_COLOR_BIT)
    region = VK.BufferImageCopy(UInt64(offset), UInt32(0), UInt32(0),
        VK.ImageSubresourceLayers(aspect, UInt32(0), UInt32(0), UInt32(1)),
        VK.Offset3D(Int32(x), Int32(y), Int32(0)), VK.Extent3D(UInt32(w), UInt32(h), UInt32(1)))
    VK.cmd_copy_image_to_buffer(cmd, image, VK.IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, buffer, [region])
    return nothing
end

# Copy `fb`'s colour attachment out and return it to `COLOR_ATTACHMENT_OPTIMAL`, the
# layout a pass that loads it expects.
function copyattachment!(cmd, fb::VulkanFramebuffer, buffer::VK.Buffer, offset::Integer)
    imagebarrier!(cmd, fb.color_image,
        VK.IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK.IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        VK.PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK.PIPELINE_STAGE_TRANSFER_BIT,
        VK.ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK.ACCESS_TRANSFER_READ_BIT)
    copyimage!(cmd, buffer, offset, fb.color_image, (0, 0, fb.width, fb.height))
    imagebarrier!(cmd, fb.color_image,
        VK.IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK.IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        VK.PIPELINE_STAGE_TRANSFER_BIT, VK.PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        VK.ACCESS_TRANSFER_READ_BIT,
        VK.ACCESS_COLOR_ATTACHMENT_READ_BIT | VK.ACCESS_COLOR_ATTACHMENT_WRITE_BIT)
    return nothing
end

"""
    copy_framebuffer!(dst::LavaArray{UInt8, 1}, fb::VulkanFramebuffer) -> dst

Copy `fb`'s colour attachment into a device buffer for a kernel or
[`blit!`](@ref): [`readback_framebuffer`](@ref) without the host round trip. `dst`
must hold `width * height * pixelbytes(eltypeof(fb.color_format))` bytes; pixels
are tightly packed rows, so `(x, y)` of a `(width, height)` view is at linear
index `(y - 1) * width + x`.
"""
function copy_framebuffer!(dst::LavaArray{UInt8, 1}, fb::VulkanFramebuffer)
    nbytes = fb.width * fb.height * pixelbytes(eltypeof(fb.color_format))
    length(dst) >= nbytes || error("destination holds $(length(dst)) bytes, need $nbytes")
    oneshot!(fb.ctx.default_bq; tag = :copy) do e
        copyattachment!(e.cmd, fb, dst.buf[].buffer, pool_offset(dst.buf[]) + dst.offset)
        hold!(e, dst)
        hold!(e, fb)
    end
    return dst
end

"""
    copy_image_to_buffer!(e, dst, image, width, height, format; aspect, region = nothing)

Record an image-to-buffer copy and no transitions, so a graph places its own
barriers: the image must be in `TRANSFER_SRC_OPTIMAL`. `aspect` is the caller's
because a depth-stencil format has two aspects that cannot share a region.
`region` is `(x, y, w, h)`, default the whole image.
"""
function copy_image_to_buffer!(e::Emitter, dst::LavaArray{T, 1}, image::VK.Image,
                               width::Integer, height::Integer, format::VK.Format;
                               aspect::VK.ImageAspectFlag=VK.IMAGE_ASPECT_COLOR_BIT,
                               region = nothing) where {T}
    # Any element type: the copy moves bytes, `T` says what they mean.
    rect = region === nothing ? (0, 0, width, height) : region
    nbytes = rect[3] * rect[4] * pixelbytes(eltypeof(format))
    sizeof(T) * length(dst) >= nbytes ||
        error("destination holds $(sizeof(T) * length(dst)) bytes, need $nbytes")
    copyimage!(e.cmd, dst.buf[].buffer, pool_offset(dst.buf[]) + dst.offset, image, rect; aspect)
    hold!(e, dst)
    return dst
end

"""
    readback_window(win::VulkanWindow) -> Matrix{BGRA{N0f8}}

Read back a window image a frame drew, between frames (see the generic
[`readback_window`](@ref)). A `(width, height)` matrix of `eltypeof(win.format)`,
`BGRA{N0f8}` for every swapchain format this backend picks, as on Metal.
"""
function readback_window(win::VulkanWindow)
    checkopen(win)
    ctx = win.ctx
    bq = ctx.default_bq

    # A presentable image may only be touched between acquire and present, so
    # this acquires one; it holds an earlier frame. Mid-frame there is no command
    # buffer to add the copy to: read back after the frame is presented.
    checkreadable(win.acquirer)
    i = acquireframe!(win)
    image = win.images[i]
    # A drawn image is in `PRESENT_SRC` (`presentready!`); an undrawn one is in
    # its creation layout, from which `UNDEFINED` is valid.
    from = win.holdsframe[i] ? VK.IMAGE_LAYOUT_PRESENT_SRC_KHR : VK.IMAGE_LAYOUT_UNDEFINED

    # Read after `acquireframe!`, which may have rebuilt the swapchain.
    w, h = size(win)
    return readbackimage(ctx, bq, eltypeof(win.format), w, h) do mb, off
        frame = oneshot(bq) do e
            cmd = e.cmd
            # srcStage COLOR_ATTACHMENT_OUTPUT matches the acquire-semaphore wait
            # `submit_and_present!` adds; TOP_OF_PIPE would not depend on it.
            imagebarrier!(cmd, image,
                from, VK.IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                VK.PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK.PIPELINE_STAGE_TRANSFER_BIT,
                VK.AccessFlag(0), VK.ACCESS_TRANSFER_READ_BIT)
            copyimage!(cmd, mb.buffer, off, image, (0, 0, w, h))
            # Not `presentready!`: a read does not draw, so `holdsframe` stays.
            imagebarrier!(cmd, image,
                VK.IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK.IMAGE_LAYOUT_PRESENT_SRC_KHR,
                VK.PIPELINE_STAGE_TRANSFER_BIT, VK.PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                VK.ACCESS_TRANSFER_READ_BIT, VK.AccessFlag(0))
            hold!(e, win)
        end
        # The acquired image must be presented back, or a few readbacks without a
        # frame block forever in vkAcquireNextImageKHR. The copy rides in the submit
        # that presents: it waits on the binary acquire semaphore, which a separate
        # submit would consume.
        submit_and_present!(bq, win, frame)
    end
end

"""
    acquireframe!(win; attempts = 4 * length(win.images)) -> Int

Acquire an image a frame drew, for a readback, and return its 1-based index. After
a resize some images hold nothing yet (Mesa's headless surface hands out
0, 1, 0, 2, 0, 1, 0, 3); those are presented back untouched and the next is
requested. A window no frame has drawn since its swapchain was built takes the
first image. Throws after `attempts` acquires.
"""
function acquireframe!(win::VulkanWindow; attempts::Int = 4 * length(win.images))
    bq = win.ctx.default_bq
    for _ in 1:attempts
        # May rebuild the swapchain, which leaves no image holding a frame.
        acquire_next_image!(win)
        i = Int(win.current_image_idx) + 1
        (win.holdsframe[i] || !any(win.holdsframe)) && return i
        back = oneshot(bq) do e
            presentuntouched!(e, win)
        end
        waitfor!(bq, submit_and_present!(bq, win, back))
    end
    error("readback_window: $attempts acquires returned no image a frame has drawn " *
          "since the swapchain was built")
end
