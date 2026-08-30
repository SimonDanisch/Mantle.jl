# Offscreen framebuffer and render targets for Lava.jl
#
# Uses VK_KHR_dynamic_rendering — no VkRenderPass/VkFramebuffer needed.
# Just images + views as render targets.

"""
    VulkanFramebuffer

Offscreen render target with color and optional depth images.
Used for render-to-texture or offscreen rendering.
"""
mutable struct VulkanFramebuffer <: Framebuffer
    width::Int
    height::Int
    # Color attachment
    color_image::VK.Image
    color_memory::VK.DeviceMemory
    color_view::VK.ImageView
    color_format::VK.Format
    # Depth attachment (optional)
    depth_image::Union{Nothing, VK.Image}
    depth_memory::Union{Nothing, VK.DeviceMemory}
    depth_view::Union{Nothing, VK.ImageView}
    depth_format::VK.Format
    # Owning context — used for readback / ownership decisions.
    ctx::VkContext
end

"""Default usage of a framebuffer's color attachment."""
const COLOR_USAGE = VK.IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                    VK.IMAGE_USAGE_TRANSFER_SRC_BIT |
                    VK.IMAGE_USAGE_SAMPLED_BIT

"""
    image_2d(ctx, width, height, format, usage) -> Image

A 2D image with no memory bound. Bind it with `alloc_image_memory` for one of its
own, or with `bind_image!` to place it in a shared allocation.
"""
image_2d(ctx::VkContext, width::Integer, height::Integer,
         format::VK.Format, usage::VK.ImageUsageFlag) =
    VK.Image(ctx.device,
        VK.IMAGE_TYPE_2D, format,
        VK.Extent3D(UInt32(width), UInt32(height), UInt32(1)),
        UInt32(1), UInt32(1),
        VK.SAMPLE_COUNT_1_BIT,
        VK.IMAGE_TILING_OPTIMAL,
        usage,
        VK.SHARING_MODE_EXCLUSIVE, UInt32[],
        VK.IMAGE_LAYOUT_UNDEFINED,
    )

"""A full-subresource 2D view. Must be created after the image has memory bound."""
image_view(ctx::VkContext, image::VK.Image, format::VK.Format,
           aspect::VK.ImageAspectFlag = VK.IMAGE_ASPECT_COLOR_BIT) =
    VK.ImageView(ctx.device, image, VK.IMAGE_VIEW_TYPE_2D, format,
        VK.ComponentMapping(
            VK.COMPONENT_SWIZZLE_IDENTITY, VK.COMPONENT_SWIZZLE_IDENTITY,
            VK.COMPONENT_SWIZZLE_IDENTITY, VK.COMPONENT_SWIZZLE_IDENTITY,
        ),
        VK.ImageSubresourceRange(aspect, UInt32(0), UInt32(1), UInt32(0), UInt32(1)),
    )

"""
    VulkanFramebuffer(width, height; depth=true, color_format=VK.FORMAT_B8G8R8A8_SRGB)

Create an offscreen framebuffer with color and optional depth attachments.
"""
# `color_format` is also accepted as a Julia element type — `BGRA{N0f8}`,
# `RGBA{Float32}` — which is Mantle's portable spelling for a pixel format and
# the only one a caller outside this backend can write. `vkformat` is the same
# lowering the graph uses for transient images, so the two cannot disagree.
function VulkanFramebuffer(width::Integer, height::Integer;
                          ctx::VkContext=vk_context(),
                          depth::Bool=true,
                          color_format::Union{VK.Format,Type}=VK.FORMAT_B8G8R8A8_SRGB,
                          srgb::Bool=false)
    color_format isa Type && (color_format = vkformat(VulkanAPI(), color_format; srgb))
    return _vulkan_framebuffer(width, height, ctx, depth, color_format)
end

function _vulkan_framebuffer(width::Integer, height::Integer, ctx::VkContext,
                             depth::Bool, color_format::VK.Format)
    dev = ctx.device

    color_image = image_2d(ctx, width, height, color_format, COLOR_USAGE)
    color_memory = alloc_image_memory(ctx, color_image)
    color_view = image_view(ctx, color_image, color_format)

    # Create depth image (optional)
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

What an image needs from an allocation. Split out from `alloc_image_memory` so a
caller that wants to place several images in one allocation can ask before
committing to an offset.
"""
function image_requirements(ctx::VkContext, image::VK.Image)
    r = VK.get_image_memory_requirements(ctx.device, image)
    (size = Int(r.size), alignment = Int(r.alignment), type_bits = r.memory_type_bits)
end

"""
    device_memory(ctx, bytes, type_bits) -> DeviceMemory

Device-local memory that can back any image whose requirements include
`type_bits`. `type_bits` is the *intersection* over everything to be placed in it.
"""
function device_memory(ctx::VkContext, bytes::Integer, type_bits::Integer)
    type_bits == 0 && error("no memory type satisfies every resource in this allocation")
    idx = find_memory_type(ctx, UInt32(type_bits), VK.MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
    VK.DeviceMemory(ctx.device, UInt64(bytes), idx)
end

"""
    bind_image!(ctx, image, memory, offset)

Bind an image to a byte offset in an existing allocation. Two images may share
bytes only if their contents never have to survive each other; the caller is
responsible for that and for transitioning the second from `UNDEFINED`.
"""
bind_image!(ctx::VkContext, image::VK.Image, memory::VK.DeviceMemory, offset::Integer) =
    unwrap(VK.bind_image_memory(ctx.device, image, memory, UInt64(offset)))

"""
    unbound_buffer(ctx, bytes, usage) -> Buffer

A `VkBuffer` with no memory behind it yet, for binding into an allocation
somebody else owns.

Every other buffer path here allocates memory per buffer and binds at offset 0,
which is right when the buffer owns its bytes and useless when a suballocator
does. This is the half that lets a caller place buffers the way `bind_image!`
already lets it place images.
"""
unbound_buffer(ctx::VkContext, bytes::Integer, usage::VK.BufferUsageFlag) =
    VK.Buffer(ctx.device, UInt64(max(bytes, 1)), usage,
                  VK.SHARING_MODE_EXCLUSIVE, UInt32[])

"""Memory requirements of `buf`, shaped like [`image_requirements`](@ref)."""
function buffer_requirements(ctx::VkContext, buf::VK.Buffer)
    r = VK.get_buffer_memory_requirements(ctx.device, buf)
    (size = Int(r.size), alignment = Int(r.alignment), type_bits = r.memory_type_bits)
end

"""
    bind_buffer!(ctx, buffer, memory, offset)

Bind a buffer to a byte offset in an existing allocation — [`bind_image!`](@ref)
for buffers, and subject to the same rule: two resources may share bytes only if
their contents never have to survive each other, and the caller owns that.

It did not exist because nothing needed it: every `bind_buffer_memory` call in
this package passes 0, since each buffer allocated its own memory. A suballocator
above Lava has nowhere to put a buffer without it, which is how one ended up
suballocating a `LavaArray` instead — i.e. stacked on top of Lava's own pool
rather than on the device.
"""
bind_buffer!(ctx::VkContext, buf::VK.Buffer, memory::VK.DeviceMemory, offset::Integer) =
    throw_if_error(ctx, "vkBindBufferMemory",
                   VK.bind_buffer_memory(ctx.device, buf, memory, UInt64(offset)))

"""Allocate device-local memory for an image and bind it."""
function alloc_image_memory(ctx::VkContext, image::VK.Image)
    req = image_requirements(ctx, image)
    mem = device_memory(ctx, req.size, req.type_bits)
    bind_image!(ctx, image, mem, 0)
    return mem
end

Base.size(fb::VulkanFramebuffer) = (fb.width, fb.height)

"""Bytes per pixel for a Vulkan format."""
function format_pixel_size(fmt::VK.Format)
    fmt == VK.FORMAT_B8G8R8A8_SRGB    && return 4
    fmt == VK.FORMAT_B8G8R8A8_UNORM   && return 4
    fmt == VK.FORMAT_R8G8B8A8_SRGB    && return 4
    fmt == VK.FORMAT_R8G8B8A8_UNORM   && return 4
    fmt == VK.FORMAT_R32G32B32A32_SFLOAT && return 16
    fmt == VK.FORMAT_R16G16B16A16_SFLOAT && return 8
    fmt == VK.FORMAT_D32_SFLOAT       && return 4
    error("Unknown pixel size for format $fmt")
end

"""Julia element type for readback of a Vulkan format."""
function format_element_type(fmt::VK.Format)
    fmt == VK.FORMAT_B8G8R8A8_SRGB      && return NTuple{4, UInt8}
    fmt == VK.FORMAT_B8G8R8A8_UNORM     && return NTuple{4, UInt8}
    fmt == VK.FORMAT_R8G8B8A8_SRGB      && return NTuple{4, UInt8}
    fmt == VK.FORMAT_R8G8B8A8_UNORM     && return NTuple{4, UInt8}
    fmt == VK.FORMAT_R32G32B32A32_SFLOAT && return NTuple{4, Float32}
    fmt == VK.FORMAT_R16G16B16A16_SFLOAT && return NTuple{4, Float16}
    # One component, not a tuple of one: a depth buffer reads back as the depth.
    fmt == VK.FORMAT_D32_SFLOAT          && return Float32
    error("Unknown element type for format $fmt")
end

"""
    readback_framebuffer(fb::VulkanFramebuffer) -> Matrix

Read back the color attachment pixels to CPU memory.
Returns a width x height matrix with element type matching the framebuffer format:
- `FORMAT_B8G8R8A8_SRGB` / `_UNORM`: `NTuple{4, UInt8}` (BGRA bytes)
- `FORMAT_R32G32B32A32_SFLOAT`: `NTuple{4, Float32}` (RGBA float)
- `FORMAT_R16G16B16A16_SFLOAT`: `NTuple{4, Float16}` (RGBA half)
"""
function readback_framebuffer(fb::VulkanFramebuffer)
    ctx = fb.ctx
    bq = ctx.default_bq
    dev = ctx.device

    bpp = format_pixel_size(fb.color_format)
    T = format_element_type(fb.color_format)
    nbytes = fb.width * fb.height * bpp
    staging_buf, _, mapped_ptr, _ = get_staging(bq, nbytes)

    # Record image→buffer copy into the active batch.  flush! at the end
    # blocks until the GPU finishes, then we read the mapped staging bytes.
    batch = ensure_active_batch!(bq)
    cmd = batch.cmd_buf

    transition_image!(cmd, fb.color_image,
        VK.IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK.IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        VK.PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK.PIPELINE_STAGE_TRANSFER_BIT,
        VK.ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK.ACCESS_TRANSFER_READ_BIT)

    region = VK.BufferImageCopy(
        UInt64(0), UInt32(0), UInt32(0),
        VK.ImageSubresourceLayers(VK.IMAGE_ASPECT_COLOR_BIT,
            UInt32(0), UInt32(0), UInt32(1)),
        VK.Offset3D(0, 0, 0),
        VK.Extent3D(UInt32(fb.width), UInt32(fb.height), UInt32(1)),
    )
    VK.cmd_copy_image_to_buffer(cmd, fb.color_image,
        VK.IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, staging_buf, [region])

    pin!(batch, fb)
    flush!(bq, dev)

    pixels = Matrix{T}(undef, fb.width, fb.height)
    unsafe_copyto!(Ptr{UInt8}(pointer(pixels)), Ptr{UInt8}(mapped_ptr), nbytes)
    return pixels
end

"""
    copy_framebuffer!(dst::LavaArray{UInt8, 1}, fb::VulkanFramebuffer) -> dst

Copy `fb`'s colour attachment into a DEVICE-LOCAL buffer — the same image copy
[`readback_framebuffer`](@ref) does, without the host round trip.

`readback_framebuffer` targets host-visible staging and returns a `Matrix`, which
is what a screenshot wants and exactly what a second GPU pass does not: a
postprocessing pass that reads its input back through the CPU pays a full
download and upload per frame. This writes straight into a buffer a kernel (or
[`blit!`](@ref)) can read.

`dst` must hold `width * height * format_pixel_size(fb.color_format)` bytes; the
pixels land tightly packed in row order, so element `(x, y)` of a
`(width, height)` view is at linear index `(y - 1) * width + x`.
"""
function copy_framebuffer!(dst::LavaArray{UInt8, 1}, fb::VulkanFramebuffer)
    ctx = fb.ctx
    bq = ctx.default_bq
    nbytes = fb.width * fb.height * format_pixel_size(fb.color_format)
    length(dst) >= nbytes ||
        error("destination holds $(length(dst)) bytes, need $nbytes")

    batch = ensure_active_batch!(bq)
    cmd = batch.cmd_buf

    transition_image!(cmd, fb.color_image,
        VK.IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK.IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        VK.PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK.PIPELINE_STAGE_TRANSFER_BIT,
        VK.ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK.ACCESS_TRANSFER_READ_BIT)

    managed = dst.buf[]
    region = VK.BufferImageCopy(
        UInt64(pool_offset(managed) + dst.offset), UInt32(0), UInt32(0),
        VK.ImageSubresourceLayers(VK.IMAGE_ASPECT_COLOR_BIT,
            UInt32(0), UInt32(0), UInt32(1)),
        VK.Offset3D(0, 0, 0),
        VK.Extent3D(UInt32(fb.width), UInt32(fb.height), UInt32(1)),
    )
    VK.cmd_copy_image_to_buffer(cmd, fb.color_image,
        VK.IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, managed.buffer, [region])

    pin!(batch, fb)
    pin!(batch, dst)
    return dst
end

"""
    copy_image_to_buffer!(bq, dst, image, width, height, format; aspect)

Record an image-to-buffer copy and nothing else. The image must already be in
`TRANSFER_SRC_OPTIMAL`, and what layout it is left in is the caller's business.

`aspect` is the caller's because the format does not settle it: `D32_SFLOAT` has
only a depth aspect, and a combined depth-stencil format has two that cannot be
copied in one region.

`copy_framebuffer!` is the same copy with the two transitions built in. That is
convenient standalone and wrong under a graph, which knows what the image was
doing before and what it will do next and can often need no barrier at all.
"""
function copy_image_to_buffer!(bq, dst::LavaArray{T, 1}, image::VK.Image,
                               width::Integer, height::Integer, format::VK.Format;
                               aspect::VK.ImageAspectFlag=VK.IMAGE_ASPECT_COLOR_BIT) where {T}
    # Any element type, because the copy moves bytes and the destination's is the
    # caller's way of saying what the pixels mean: `UInt8` for a BGRA target read
    # back as bytes, `Float32` for a depth target read back as depth.
    nbytes = width * height * format_pixel_size(format)
    sizeof(T) * length(dst) >= nbytes ||
        error("destination holds $(sizeof(T) * length(dst)) bytes, need $nbytes")
    batch = ensure_active_batch!(bq)
    managed = dst.buf[]
    region = VK.BufferImageCopy(
        UInt64(pool_offset(managed) + dst.offset), UInt32(0), UInt32(0),
        VK.ImageSubresourceLayers(aspect,
            UInt32(0), UInt32(0), UInt32(1)),
        VK.Offset3D(0, 0, 0),
        VK.Extent3D(UInt32(width), UInt32(height), UInt32(1)),
    )
    VK.cmd_copy_image_to_buffer(batch.cmd_buf, image,
        VK.IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, managed.buffer, [region])
    pin!(batch, dst)
    return dst
end

"""
    readback_window(win::VulkanWindow) -> Matrix{NTuple{4, UInt8}}

Read back the current swapchain image to CPU memory.
Must be called after rendering but BEFORE present_frame!.
Returns a width x height matrix of BGRA byte tuples.
"""
function readback_window(win::VulkanWindow)
    checkopen(win)
    ctx = win.ctx
    bq = ctx.default_bq
    dev = ctx.device

    w, h = size(win)
    bpp = format_pixel_size(win.format)
    T = format_element_type(win.format)
    nbytes = w * h * bpp
    staging_buf, _, mapped_ptr, _ = get_staging(bq, nbytes)

    # A presentable image may only be touched between acquire and present. Called
    # after a present, this used to transition an image it did not own, which
    # synchronization validation reports as three separate errors and which the
    # driver happens to tolerate: the pixels came back looking right for as long
    # as nobody turned validation on.
    #
    # When the caller is mid-frame the image is already acquired and sits in
    # COLOR_ATTACHMENT_OPTIMAL. Otherwise acquire one first, and note that it
    # then holds an EARLIER frame's contents, since acquire returns whichever
    # image the presentation engine has freed.
    fresh = !win.acquired
    fresh && acquire_next_image!(win)
    image = win.images[win.current_image_idx + 1]

    # Record image→buffer copy into the active batch, blocking flush at end.
    batch = ensure_active_batch!(bq)
    cmd = batch.cmd_buf

    # A freshly acquired image is still being read by the presentation engine
    # until its acquire semaphore signals, so the submit that transitions it has
    # to wait on that semaphore, at COLOR_ATTACHMENT_OUTPUT to match what the
    # acquire signals. Without that the transition races the present, which
    # synchronization validation calls WRITE_AFTER_PRESENT.
    #
    # The wait is not added here: `present_frame!` below is the submit, and it
    # adds exactly this wait itself (api.jl:401). Adding it in both places makes
    # one submit wait twice on one binary semaphore, which signals once — the
    # submit then never runs and the next `vkWaitSemaphores` blocks forever, in
    # the kernel, where no Julia interrupt reaches it.

    # srcStage is COLOR_ATTACHMENT_OUTPUT, never TOP_OF_PIPE: on a freshly
    # acquired image the submit waits on the acquire semaphore at that stage, and
    # a barrier from TOP_OF_PIPE would create no dependency with it.
    transition_image!(cmd, image,
        fresh ? VK.IMAGE_LAYOUT_PRESENT_SRC_KHR : VK.IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        VK.IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        VK.PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK.PIPELINE_STAGE_TRANSFER_BIT,
        fresh ? VK.AccessFlag(0) : VK.ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
        VK.ACCESS_TRANSFER_READ_BIT)

    region = VK.BufferImageCopy(
        UInt64(0), UInt32(0), UInt32(0),
        VK.ImageSubresourceLayers(VK.IMAGE_ASPECT_COLOR_BIT,
            UInt32(0), UInt32(0), UInt32(1)),
        VK.Offset3D(0, 0, 0),
        VK.Extent3D(UInt32(w), UInt32(h), UInt32(1)),
    )
    VK.cmd_copy_image_to_buffer(cmd, image,
        VK.IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, staging_buf, [region])

    # Transition back to COLOR_ATTACHMENT_OPTIMAL so present_frame! can transition to PRESENT_SRC
    transition_image!(cmd, image,
        VK.IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK.IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        VK.PIPELINE_STAGE_TRANSFER_BIT, VK.PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
        VK.ACCESS_TRANSFER_READ_BIT, VK.AccessFlag(0))

    pin!(batch, win)

    # An acquire this function made is an acquire this function returns. Leaving
    # it outstanding is invisible once — the program usually closes the window
    # next — and a deadlock in a loop: a swapchain has two or three images, so
    # the fourth readback with no frame in flight waits forever inside
    # vkAcquireNextImageKHR for an image that is never handed back.
    #
    # The present has to be the submit that carries this copy, not one after it:
    # it waits on the acquire semaphore, which is binary, and a `flush!` here
    # would consume that wait first and leave the present waiting on a semaphore
    # nothing will signal again. So submit through `present_frame!` and wait on
    # the timeline value it signals. When the caller was mid-frame the acquire was
    # theirs and so is the present, and this is an ordinary flush.
    if fresh
        signal = batch.signal_value
        present_frame!(bq, win)
        wait_semaphores!(bq, VK.SemaphoreWaitInfo([bq.timeline_sem], [signal]))
    else
        flush!(bq, dev)
    end

    pixels = Matrix{T}(undef, w, h)
    unsafe_copyto!(Ptr{UInt8}(pointer(pixels)), Ptr{UInt8}(mapped_ptr), nbytes)
    return pixels
end

# ── Render Target Subtypes ──

# `WindowTarget` and `OffscreenTarget` are Mantle's for the same reason
# `SampledTexture` is: they wrap a `Window` and a `Framebuffer`, and both of
# those are the API's abstract types now.
