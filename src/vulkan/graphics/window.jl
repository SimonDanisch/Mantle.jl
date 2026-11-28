# Vulkan's half of core's `Window` (`graphics/window.jl`): a surface on the GLFW window
# and its swapchain; needs VK_KHR_surface and the platform's surface extension.

"""
    VulkanSwapchain

A Vulkan surface on a platform window and the swapchain presenting into it.
"""
mutable struct VulkanSwapchain
    surface::VK.SurfaceKHR
    swapchain::Union{Nothing, VK.SwapchainKHR}
    images::Vector{VK.Image}
    views::Vector{VK.ImageView}
    format::VK.Format
    # The requested format, kept for rebuilds on resize; `format` is what the
    # surface offered. `_SRGB` encodes on write (shaders write linear), `_UNORM`
    # stores as-is (shaders write display-referred).
    preferred_format::VK.Format
    # Kept for rebuilds on resize, as the format is.
    vsync::Bool
    extent::VK.Extent2D
    # `image_available` and `slot_tokens` per frame slot, `render_finished` per image.
    image_available::Vector{VK.Semaphore}
    render_finished::Vector{VK.Semaphore}
    # The default channel's token for each slot's last frame, `0` for none. A
    # slot is reused once its token has passed; the frame's submission signals the
    # timeline, so it needs no fence of its own, and waiting for the token yields.
    slot_tokens::Vector{UInt64}
    current_frame::Int  # index into sync arrays (1-based, wraps)
    current_image_idx::UInt32
    # Per image: whether a frame drew it since the swapchain was built. Set by
    # `presentready!`, cleared by `presentuntouched!` and `create_swapchain!`;
    # `readback_window` hands undrawn images back.
    holdsframe::Vector{Bool}
    acquired::Bool
    # Framebuffer size the swapchain was built for (`fitsurface!`).
    fb_size::Tuple{Int,Int}
    # Swapchain and surface belong to this device.
    ctx::LavaDevice
end

"""
A surface on the GLFW window `handle` and a swapchain of `width` x `height`. `T`
decides what shaders write: an `_SRGB` format (`srgb = true`) takes linear values
and encodes them, `_UNORM` stores already gamma-encoded values as they are. Falls
back to the surface's first format when the requested one is not offered.
"""
function windowsurface(ctx::LavaDevice, handle::GLFW.Window, ::Type{T}, width::Int, height::Int;
                       vsync::Bool, srgb::Bool) where {T}
    color_format = pixelformat(VulkanAPI(), T; srgb)
    surface_ptr = GLFW.CreateWindowSurface(ctx.instance.vks, handle)
    surface_ptr == C_NULL && throw(LavaError("window creation", "GLFW CreateWindowSurface failed",
                                "Ensure Vulkan drivers support window surfaces"))
    surface = VK.SurfaceKHR(surface_ptr, ctx.instance, ctx.instance.refcount)
    sc = VulkanSwapchain(surface, nothing, VK.Image[], VK.ImageView[], color_format, color_format,
                         vsync, VK.Extent2D(width, height), VK.Semaphore[], VK.Semaphore[], UInt64[],
                         1, UInt32(0), Bool[], false, (width, height), ctx)
    create_swapchain!(sc, width, height)
    return sc
end

"""
    create_swapchain!(win::VulkanSwapchain, width, height)

Create or recreate the swapchain, for a framebuffer of `width` x `height` pixels.
"""
function create_swapchain!(win::VulkanSwapchain, fbw::Int, fbh::Int)
    vsync = win.vsync
    ctx = win.ctx
    dev = ctx.device
    phys = ctx.physical_device

    caps = unwrap(VK.get_physical_device_surface_capabilities_khr(phys, win.surface))

    # The requested format if offered, else the surface's first; read from the
    # window so a rebuild on resize keeps the caller's choice.
    formats = unwrap(VK.get_physical_device_surface_formats_khr(phys; surface=win.surface))
    chosen_format = formats[1]
    for f in formats
        if f.format == win.preferred_format &&
           f.color_space == VK.COLOR_SPACE_SRGB_NONLINEAR_KHR
            chosen_format = f
            break
        end
    end
    win.format = chosen_format.format

    present_modes = unwrap(VK.get_physical_device_surface_present_modes_khr(phys; surface=win.surface))
    present_mode = VK.PRESENT_MODE_FIFO_KHR  # vsync, always supported
    if !vsync
        for pm in present_modes
            if pm == VK.PRESENT_MODE_MAILBOX_KHR
                present_mode = pm
                break
            elseif pm == VK.PRESENT_MODE_IMMEDIATE_KHR
                present_mode = pm
            end
        end
    end

    win.fb_size = (fbw, fbh)      # what this swapchain is built for
    if caps.current_extent.width != typemax(UInt32)
        win.extent = caps.current_extent
    else
        win.extent = VK.Extent2D(
            clamp(UInt32(fbw), caps.min_image_extent.width, caps.max_image_extent.width),
            clamp(UInt32(fbh), caps.min_image_extent.height, caps.max_image_extent.height),
        )
    end

    image_count = caps.min_image_count + 1
    if caps.max_image_count > 0
        image_count = min(image_count, caps.max_image_count)
    end

    # `readback_window` copies from swapchain images, which needs TRANSFER_SRC
    # (RADV tolerates its absence, validation does not). Requested where supported.
    usage = VK.IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK.IMAGE_USAGE_TRANSFER_DST_BIT
    if (caps.supported_usage_flags & VK.IMAGE_USAGE_TRANSFER_SRC_BIT) != VK.ImageUsageFlag(0)
        usage |= VK.IMAGE_USAGE_TRANSFER_SRC_BIT
    end

    old_swapchain = win.swapchain

    kw = Dict{Symbol,Any}()
    if old_swapchain !== nothing
        kw[:old_swapchain] = old_swapchain
    end

    swapchain = VK.SwapchainKHR(
        dev, win.surface,
        image_count, chosen_format.format, chosen_format.color_space,
        win.extent, UInt32(1),
        usage,
        VK.SHARING_MODE_EXCLUSIVE, UInt32[],
        caps.current_transform,
        VK.COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
        present_mode, true;
        kw...,
    )

    win.swapchain = swapchain
    win.images = unwrap(VK.get_swapchain_images_khr(dev, swapchain))
    win.holdsframe = fill(false, length(win.images))

    # Destroy the old swapchain and views now; left to finalizers they pile up
    # across resizes. Wait first: this runs from `acquire_next_image!` with frames
    # in flight, and destroying images or views still in use is a use-after-free
    # (a GPUVM fault frames later).
    if old_swapchain !== nothing
        deviceidle(win.ctx)
    end
    for v in win.views
        finalize(v)
    end
    if old_swapchain !== nothing
        finalize(old_swapchain)
    end

    win.views = VK.ImageView[image_view(win.ctx, img, win.format) for img in win.images]

    # One of each sync object per swapchain image.
    n = length(win.images)
    win.image_available = [VK.Semaphore(dev) for _ in 1:n]
    # Indexed by image, not frame slot: a semaphore used in a present is reusable
    # only once its image is re-acquired, and images come back in any order.
    # Keyed by slot, one is signalled while the presentation engine owns it.
    win.render_finished = [VK.Semaphore(dev) for _ in 1:length(win.images)]
    win.slot_tokens = zeros(UInt64, n)   # every frame in flight has passed (see above)
    win.current_frame = 1
end

# What a render pass asks any attachment, so `Window` forwards and core never sees a
# swapchain. `current_image_idx` changes per frame and is set by
# `acquire_next_image!`.
target_view(w::VulkanSwapchain)   = w.views[w.current_image_idx + 1]
target_image(w::VulkanSwapchain)  = w.images[w.current_image_idx + 1]
target_extent(w::VulkanSwapchain) = (Int(w.extent.width), Int(w.extent.height))
target_format(w::VulkanSwapchain) = w.format

# The pixel type the swapchain's images hold, from the format it actually has.
Base.eltype(w::VulkanSwapchain) = eltypeof(w.format)

"""
Rebuild the swapchain if the framebuffer no longer matches it. Waits for the device
first: destroying images or views in flight is a use-after-free.
"""
function fitsurface!(win::VulkanSwapchain, fbw::Int, fbh::Int)
    (fbw, fbh) == win.fb_size && return false
    rebuild!(win, fbw, fbh)
    return true
end

"""Rebuild the swapchain at `width` x `height` once nothing in flight uses it."""
function rebuild!(win::VulkanSwapchain, fbw::Int = win.fb_size[1], fbh::Int = win.fb_size[2])
    VK.device_wait_idle(win.ctx.device)
    # Everything in flight has passed; give the frames' one-shots back now
    # rather than at the next sweep.
    drain!(win.ctx.default_bq)
    create_swapchain!(win, fbw, fbh)
    return nothing
end

"""
Acquire the next image after the previous frame in this slot has passed.
"""
function acquire_next_image!(win::VulkanSwapchain)
    waitfor!(win.ctx.default_bq, win.slot_tokens[win.current_frame])
    win.current_image_idx = acquireimage!(win)
    win.acquired = true
    return nothing
end

"""
One image acquire from `win`'s swapchain into a frame slot's semaphore, as a
worker thread performs it (see `TimelineWait`).
"""
struct ImageAcquire
    device::VK.vk.VkDevice
    swapchain::VK.vk.VkSwapchainKHR
    semaphore::VK.vk.VkSemaphore
    index::Base.RefValue{UInt32}
    fptr::Ptr{Cvoid}                # vkAcquireNextImageKHR
end

ImageAcquire(win::VulkanSwapchain) =
    ImageAcquire(win.ctx.device.vks, win.swapchain.vks,
                 win.image_available[win.current_frame].vks, Ref(UInt32(0)),
                 VK.function_pointer(VK.global_dispatcher[], win.ctx.device,
                                     :vkAcquireNextImageKHR))

# GC-safe: on a worker it blocks until the presentation engine gives an image back.
function acquirecall(a::ImageAcquire, timeout::UInt64)
    fptr = a.fptr
    GPUToolbox.@gcsafe_ccall $fptr(a.device::VK.vk.VkDevice,
        a.swapchain::VK.vk.VkSwapchainKHR, timeout::UInt64, a.semaphore::VK.vk.VkSemaphore,
        VK.vk.VkFence(C_NULL)::VK.vk.VkFence, a.index::Ptr{UInt32})::VK.vk.VkResult
end

blockon(a::ImageAcquire) = acquirecall(a, typemax(UInt64))

"""
    acquireimage!(win) -> UInt32

Acquire the next image and return its 0-based index. Tried without waiting first,
since an image is usually free; one the presentation engine still holds (a frame
paced by the display) is waited for on a worker ([`hostwait`](@ref)). A stale
swapchain is rebuilt and acquired from once more, into the rebuilt slot's
semaphore.
"""
function acquireimage!(win::VulkanSwapchain)
    res, a = tryacquire(win)
    if res == VK.vk.VK_ERROR_OUT_OF_DATE_KHR
        rebuild!(win)
        res, a = tryacquire(win)
    end
    throw_if_error(win.ctx, "vkAcquireNextImageKHR", checkedresult(res, "vkAcquireNextImageKHR"))
    return a.index[]
end

function tryacquire(win::VulkanSwapchain)
    a = ImageAcquire(win)
    res = acquirecall(a, UInt64(0))
    if res == VK.vk.VK_NOT_READY || res == VK.vk.VK_TIMEOUT
        res = something(hostwait(blockon, nothing, a))
    end
    return res, a
end

"""
One present of `win`'s acquired image, as a worker thread performs it (see
`TimelineWait`). `info` points into the three vectors, which live as long as it.
"""
struct PresentCall
    queue::VK.vk.VkQueue
    info::Base.RefValue{VK.vk.VkPresentInfoKHR}
    semaphores::Vector{VK.vk.VkSemaphore}
    swapchains::Vector{VK.vk.VkSwapchainKHR}
    indices::Vector{UInt32}
    fptr::Ptr{Cvoid}                # vkQueuePresentKHR
end

function PresentCall(win::VulkanSwapchain)
    sems = [win.render_finished[win.current_image_idx + 1].vks]
    swaps = [win.swapchain.vks]
    idxs = UInt32[win.current_image_idx]
    info = Ref(VK.vk.VkPresentInfoKHR(VK.vk.VK_STRUCTURE_TYPE_PRESENT_INFO_KHR, C_NULL,
        UInt32(1), pointer(sems), UInt32(1), pointer(swaps), pointer(idxs), C_NULL))
    return PresentCall(vkqueue(win.ctx.default_bq).vks, info, sems, swaps, idxs,
                       VK.function_pointer(VK.global_dispatcher[], win.ctx.device,
                                           :vkQueuePresentKHR))
end

function blockon(p::PresentCall)
    fptr = p.fptr
    GPUToolbox.@gcsafe_ccall $fptr(p.queue::VK.vk.VkQueue,
        p.info::Ptr{VK.vk.VkPresentInfoKHR})::VK.vk.VkResult
end

"""
    present!(win::VulkanSwapchain)

Present the acquired image, waiting on its `render_finished` semaphore. Call after
the frame's submit.
"""
function present!(win::VulkanSwapchain)
    win.acquired || error("Cannot present: no image acquired (call acquire_next_image! first)")
    ctx = win.ctx
    fi = win.current_frame
    p = PresentCall(win)
    # On a worker, holding the queue: some drivers wait in the present for the
    # frame's rendering and for the display (NVIDIA on X11 waits out the whole GPU
    # frame there, whatever the present mode), and the queue is held because
    # another task on this thread may submit to it meanwhile.
    raw = GC.@preserve p @lock queuelock(ctx.default_bq) something(hostwait(blockon, nothing, p))
    # OUT_OF_DATE and SUBOPTIMAL are not failures (a resize, or a compositor
    # remap, often on the first present): rebuild the swapchain.
    result = checkedresult(raw, "vkQueuePresentKHR")
    if iserror(result)
        code = unwrap_error(result).code
        if code == VK.ERROR_OUT_OF_DATE_KHR || code == VK.SUBOPTIMAL_KHR
            win.acquired = false
            rebuild!(win)                 # the caller draws the next frame
            return nothing
        end
        throw_if_error(ctx, "vkQueuePresentKHR", result)
    end

    win.acquired = false
    win.current_frame = mod1(fi + 1, length(win.slot_tokens))
end

# Destroyed here, children first: finalizers do not run at exit, and a surface
# outliving the instance is a leak (VUID-vkDestroyInstance-instance-00629).
function closesurface!(win::VulkanSwapchain)
    ctx = win.ctx
    deviceidle(ctx)
    # Everything in flight has passed; give the frames' one-shots back now
    # rather than at the next sweep.
    drain!(ctx.default_bq)
    for v in win.views
        finalize(v)
    end
    empty!(win.views)
    empty!(win.images)                  # owned by the swapchain, not destroyed
    for s in win.image_available; finalize(s); end
    for s in win.render_finished; finalize(s); end
    empty!(win.image_available); empty!(win.render_finished); empty!(win.slot_tokens)
    if win.swapchain !== nothing
        finalize(win.swapchain)
        win.swapchain = nothing
    end
    # Not `finalize`: the surface wraps GLFW's pointer by hand, so it has no VK.jl
    # destructor and no finalizer.
    VK.destroy_surface_khr(ctx.instance, win.surface)
    return nothing
end
