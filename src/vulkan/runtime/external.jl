# External-memory interop: share Lava-produced images with other APIs.
#
# An `ExternalImage` owns a dedicated, exportable Vulkan image allocation on
# Lava's device. Its memory can be handed to another API as an opaque fd
# (`memoryfd`) — e.g. imported into OpenGL via GL_EXT_memory_object_fd — so
# a frame computed in a Lava kernel reaches the other API with **zero
# copies across PCIe** (one device-local blit, no host roundtrip).
#
# Everything here is additive and opt-in: nothing runs unless an
# ExternalImage is created. The allocation is deliberately dedicated and
# outside the pooled allocator — external memory must not be sub-allocated
# (importers see the whole allocation), and NVIDIA requires dedicated
# allocations for exported images anyway.
#
# See VideoEdit/interop_mwe.jl for a full Vulkan↔GL validation including
# the GL import side.

"""
    ExternalImage(width, height; format = VK.FORMAT_R8G8B8A8_UNORM)

A GPU image whose memory can be exported to other APIs (see [`memoryfd`](@ref)).
OPTIMAL tiling, `TRANSFER_DST | SAMPLED` usage. Fill it from a `LavaArray`
with `copyto!(img, array)`. Requires `VK_KHR_external_memory_fd` (enabled
automatically at device creation when the driver offers it; check
`vk_context().external_memory_available`).
"""
mutable struct ExternalImage
    image::VK.Image
    memory::VK.DeviceMemory
    width::Int
    height::Int
    allocation_size::Int   # importers must import exactly this many bytes
    layout_initialized::Bool
end

function ExternalImage(width::Integer, height::Integer;
                       format::VK.Format = VK.FORMAT_R8G8B8A8_UNORM)
    ctx = vk_context()
    ctx.external_memory_available ||
        error("this device was created without VK_KHR_external_memory_fd support")
    handle_types = VK.EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT
    image = @vk_checked "external_image_create" VK.create_image(ctx.device, VK.IMAGE_TYPE_2D, format,
        VK.Extent3D(width, height, 1), 1, 1, VK.SAMPLE_COUNT_1_BIT,
        VK.IMAGE_TILING_OPTIMAL,
        VK.IMAGE_USAGE_TRANSFER_DST_BIT | VK.IMAGE_USAGE_SAMPLED_BIT,
        VK.SHARING_MODE_EXCLUSIVE, UInt32[], VK.IMAGE_LAYOUT_UNDEFINED;
        next = VK.ExternalMemoryImageCreateInfo(; handle_types))
    req = VK.get_image_memory_requirements(ctx.device, image)
    mtype = findfirst(0:(length(ctx.memory_properties.memory_types) - 1)) do i
        (req.memory_type_bits >> i) & 1 == 1 &&
            ctx.memory_properties.memory_types[i + 1].property_flags &
            VK.MEMORY_PROPERTY_DEVICE_LOCAL_BIT != VK.MemoryPropertyFlag(0)
    end
    mtype === nothing && error("no device-local memory type for external image")
    memory = @vk_checked "external_image_alloc" VK.allocate_memory(ctx.device, req.size, mtype - 1;
        next = VK.ExportMemoryAllocateInfo(;
            next = VK.MemoryDedicatedAllocateInfo(; image),
            handle_types))
    @vk_checked "external_image_bind" VK.bind_image_memory(ctx.device, image, memory, 0)
    return ExternalImage(image, memory, Int(width), Int(height), Int(req.size), false)
end

"""
    memoryfd(img::ExternalImage) -> Int

Export the image's memory as an opaque file descriptor
(`VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT`). Each call duplicates a
new fd; the importer takes ownership (e.g. `glImportMemoryFdEXT` consumes
it). Importers must import exactly `img.allocation_size` bytes and, for
GL, mark the memory object dedicated and the texture OPTIMAL-tiled.
"""
function memoryfd(img::ExternalImage)
    ctx = vk_context()
    # VK.jl's high-level get_memory_fd_khr passes a Ref{Int64} where the
    # C signature wants int*, so call the low-level entry point directly.
    fptr = VK.function_pointer(ctx.device, "vkGetMemoryFdKHR")
    pfd = Ref{Int32}(-1)
    info = VK._MemoryGetFdInfoKHR(img.memory,
        VK.EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT)
    res = VK.vkGetMemoryFdKHR(ctx.device, info, pfd, fptr)
    res == VK.VkCore.VK_SUCCESS || error("vkGetMemoryFdKHR failed: $res")
    return Int(pfd[])
end

"""
    copyto!(img::ExternalImage, a::LavaArray) -> img

Blit the array's bytes into the external image (device-local copy — the
data never leaves the GPU). The array must hold at least
`4 * width * height` bytes of pixel data in x-contiguous row order matching
the image format. Waits for pending Lava kernels writing `a`, then blocks
until the copy has landed, so the importer may sample immediately.

Runs as a one-shot submission on the context's secondary compute queue —
it never interleaves with the BatchQueue's batched submissions.
"""
function Base.copyto!(img::ExternalImage, a::LavaArray)
    nbytes = length(a) * sizeof(eltype(a))
    needed = 4 * img.width * img.height
    nbytes >= needed ||
        error("array holds $nbytes bytes; image needs $needed")
    ctx = (a.buf[].ctx)::VkContext
    KA.synchronize(LavaBackend(ctx))  # pending kernel writes must land first

    managed = a.buf[]
    src_offset = UInt64(managed.pool_offset + a.offset)
    # VK.jl handles are refcounted and destroy themselves via finalizers —
    # no manual destroy (a second vkDestroy* on the same handle segfaults).
    pool = @vk_checked "external_copy_pool" VK.create_command_pool(ctx.device, ctx.queue_family_index)
    let
        cb = first(@vk_checked "external_copy_cb" VK.allocate_command_buffers(ctx.device,
            VK.CommandBufferAllocateInfo(pool, VK.COMMAND_BUFFER_LEVEL_PRIMARY, 1)))
        @vk_checked "external_copy_begin" VK.begin_command_buffer(cb, VK.CommandBufferBeginInfo())
        range = VK.ImageSubresourceRange(VK.IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1)
        oldlayout = img.layout_initialized ? VK.IMAGE_LAYOUT_GENERAL :
                                             VK.IMAGE_LAYOUT_UNDEFINED
        VK.cmd_pipeline_barrier(cb, [], [],
            [VK.ImageMemoryBarrier(VK.AccessFlag(0), VK.ACCESS_TRANSFER_WRITE_BIT,
                oldlayout, VK.IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK.QUEUE_FAMILY_IGNORED, VK.QUEUE_FAMILY_IGNORED, img.image, range)];
            src_stage_mask = VK.PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            dst_stage_mask = VK.PIPELINE_STAGE_TRANSFER_BIT)
        VK.cmd_copy_buffer_to_image(cb, managed.buffer, img.image,
            VK.IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            [VK.BufferImageCopy(src_offset, 0, 0,
                VK.ImageSubresourceLayers(VK.IMAGE_ASPECT_COLOR_BIT, 0, 0, 1),
                VK.Offset3D(0, 0, 0), VK.Extent3D(img.width, img.height, 1))])
        VK.cmd_pipeline_barrier(cb, [], [],
            [VK.ImageMemoryBarrier(VK.ACCESS_TRANSFER_WRITE_BIT, VK.AccessFlag(0),
                VK.IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK.IMAGE_LAYOUT_GENERAL,
                VK.QUEUE_FAMILY_IGNORED, VK.QUEUE_FAMILY_IGNORED, img.image, range)];
            src_stage_mask = VK.PIPELINE_STAGE_TRANSFER_BIT,
            dst_stage_mask = VK.PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT)
        @vk_checked "external_copy_end" VK.end_command_buffer(cb)
        @vk_checked "external_copy_submit" VK.queue_submit(ctx.compute_queue, [VK.SubmitInfo([], [], [cb], [])])
        @vk_checked "external_copy_wait" VK.queue_wait_idle(ctx.compute_queue)
        img.layout_initialized = true
    end
    return img
end
