# GLFW windows with Vulkan surfaces and swapchains; needs VK_KHR_surface and the
# platform's surface extension.

import GLFW

"""
    VulkanWindow

A GLFW window with a Vulkan surface and swapchain.
"""
mutable struct VulkanWindow <: Window
    handle::GLFW.Window
    surface::VK.SurfaceKHR
    swapchain::Union{Nothing, VK.SwapchainKHR}
    images::Vector{VK.Image}
    views::Vector{VK.ImageView}
    format::VK.Format
    # The requested format, kept for rebuilds on resize; `format` is what the
    # surface offered. `_SRGB` encodes on write (shaders write linear), `_UNORM`
    # stores as-is (shaders write display-referred).
    preferred_format::VK.Format
    extent::VK.Extent2D
    # `image_available` and `in_flight` per frame slot, `render_finished` per image.
    image_available::Vector{VK.Semaphore}
    render_finished::Vector{VK.Semaphore}
    in_flight::Vector{VK.Fence}
    current_frame::Int  # index into sync arrays (1-based, wraps)
    current_image_idx::UInt32
    # Per image: whether a frame drew it since the swapchain was built. Set by
    # `presentready!`, cleared by `presentuntouched!` and `create_swapchain!`;
    # `readback_window` hands undrawn images back.
    holdsframe::Vector{Bool}
    acquired::Bool
    # The task holding the acquired image, `nothing` between frames. While set,
    # another acquire and a readback are refused.
    acquirer::Union{Nothing, Task}
    # Framebuffer size the swapchain was built for. Resizes are not reliably
    # reported (OUT_OF_DATE is optional, SUBOPTIMAL is a success code, Xwayland
    # sends neither), so `sync_swapchain!` compares against this.
    fb_size::Tuple{Int,Int}
    # Swapchain and surface belong to this device.
    ctx::LavaDevice
end

"""
    VulkanWindow(width, height; ctx, title="Lava", vsync=true, color_format=VK.FORMAT_B8G8R8A8_SRGB, srgb=false)

A window with a Vulkan surface and swapchain. `color_format`, a `VK.Format` or a
Julia element type, decides what shaders write: `_SRGB` takes linear values and
encodes them, `_UNORM` stores already gamma-encoded values as they are. Falls back
to the surface's first format when the requested one is not offered.
"""
function VulkanWindow(width::Integer, height::Integer;
                      ctx::LavaDevice,
                      title::String="Lava", vsync::Bool=true,
                      color_format::Union{VK.Format,Type}=VK.FORMAT_B8G8R8A8_SRGB,
                      srgb::Bool=false)
    # Lowered as `VulkanFramebuffer` lowers it, for the portable `Mantle.Window`.
    color_format isa Type && (color_format = vkformat(VulkanAPI(), color_format; srgb))

    # No OpenGL context.
    GLFW.Init()
    GLFW.WindowHint(GLFW.CLIENT_API, GLFW.NO_API)
    GLFW.WindowHint(GLFW.RESIZABLE, true)

    handle = GLFW.CreateWindow(width, height, title)
    # Window hints are process-global: left at `NO_API`, the next OpenGL window
    # (GLMakie's) has no context to make current.
    GLFW.DefaultWindowHints()

    surface_ptr = GLFW.CreateWindowSurface(ctx.instance.vks, handle)
    surface_ptr == C_NULL && throw(LavaError("window creation", "GLFW CreateWindowSurface failed",
                                "Ensure Vulkan drivers support window surfaces"))
    surface = VK.SurfaceKHR(surface_ptr, ctx.instance, ctx.instance.refcount)

    win = VulkanWindow(
        handle, surface, nothing,
        VK.Image[], VK.ImageView[],
        color_format, color_format,
        VK.Extent2D(width, height),
        VK.Semaphore[], VK.Semaphore[], VK.Fence[],
        1,  # current_frame
        UInt32(0), Bool[], false, nothing,
        (Int(width), Int(height)),       # fb_size, corrected by create_swapchain!
        ctx,
    )

    create_swapchain!(win; vsync)
    return win
end

"""
    create_swapchain!(win::VulkanWindow; vsync=true)

Create or recreate the swapchain for the window.
"""
function create_swapchain!(win::VulkanWindow; vsync::Bool=true)
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

    fbw, fbh = GLFW.GetFramebufferSize(win.handle)
    win.fb_size = (Int(fbw), Int(fbh))      # what this swapchain is built for
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
        VK.device_wait_idle(dev)
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
    win.in_flight = [VK.Fence(dev; flags=VK.FENCE_CREATE_SIGNALED_BIT) for _ in 1:n]
    win.current_frame = 1
end

# What a render pass asks any attachment, so `WindowSurface` forwards and core
# never sees a swapchain. `current_image_idx` changes per frame and is set by
# `acquire_next_image!`.
target_view(w::VulkanWindow)   = w.views[w.current_image_idx + 1]
target_image(w::VulkanWindow)  = w.images[w.current_image_idx + 1]
target_extent(w::VulkanWindow) = (Int(w.extent.width), Int(w.extent.height))
target_format(w::VulkanWindow) = w.format

# The pixel type shaders write (`attachment_format` reads it), from the format the
# swapchain actually has.
Base.eltype(w::VulkanWindow) = eltypeof(w.format)

"""
    checkopen(win)

Throw if `win` is closed: touching a destroyed GLFW handle segfaults in the
driver. Checks the handle, not `isopen`, since a window whose close button was
clicked can still be drawn to.
"""
checkopen(win::VulkanWindow) =
    win.handle.handle == C_NULL &&
        error("this window has been closed; nothing can be drawn to it or read from it")

"""
    sync_swapchain!(win) -> Bool

Rebuild the swapchain if the framebuffer size no longer matches it (resizes are
not reliably reported, see `fb_size`) and return whether it did. Separate from
`acquire_next_image!` so a graph can learn the size before recording. A zero size
is a minimised window and rebuilds nothing.
"""
function sync_swapchain!(win::VulkanWindow)
    checkopen(win)
    fbw, fbh = GLFW.GetFramebufferSize(win.handle)
    (fbw, fbh) == win.fb_size && return false
    (fbw > 0 && fbh > 0) || return false
    resize!(win)
    return true
end

"""
    acquire_next_image!(win::VulkanWindow) -> UInt32

Acquire the next swapchain image and return its 0-based index. Call before
recording the frame.
"""
function acquire_next_image!(win::VulkanWindow)
    checkopen(win)
    ctx = win.ctx
    dev = ctx.device

    sync_swapchain!(win)

    # Refused while an image is held: the fence below is signalled by that
    # frame's `submit_and_present!`, and waiting on it blocks in a foreign call
    # the holding task can never run past. Compiling a kernel mid-frame yields,
    # so a second task can arrive between acquire and present.
    checkacquirable(win.acquirer)

    fi = win.current_frame
    # The slot's previous frame has passed once its fence has; the next sweep
    # returns its one-shot.
    wait_for_fences!(ctx.default_bq, [win.in_flight[fi]])
    unwrap(VK.reset_fences(dev, [win.in_flight[fi]]))

    # A stale swapchain on acquire is rebuilt, not an error.
    acq = VK.acquire_next_image_khr(dev, win.swapchain,
        typemax(UInt64); semaphore=win.image_available[fi])
    if iserror(acq) && unwrap_error(acq).code == VK.ERROR_OUT_OF_DATE_KHR
        resize!(win)
        acq = VK.acquire_next_image_khr(dev, win.swapchain,
            typemax(UInt64); semaphore=win.image_available[fi])
    end
    idx, _ = throw_if_error(ctx, "vkAcquireNextImageKHR", acq)

    win.current_image_idx = idx
    win.acquired = true
    win.acquirer = current_task()
    return idx
end

"""
    present!(win::VulkanWindow)

Present the acquired image, waiting on its `render_finished` semaphore. Call after
the frame's submit.
"""
function present!(win::VulkanWindow)
    checkopen(win)
    win.acquired || error("Cannot present: no image acquired (call acquire_next_image! first)")
    ctx = win.ctx
    fi = win.current_frame
    present_info = VK.PresentInfoKHR(
        [win.render_finished[win.current_image_idx + 1]],
        [win.swapchain],
        [win.current_image_idx],
    )
    # OUT_OF_DATE and SUBOPTIMAL are not failures (a resize, or a compositor
    # remap, often on the first present): rebuild the swapchain.
    result = VK.queue_present_khr(vkqueue(ctx.default_bq), present_info)
    if iserror(result)
        code = unwrap_error(result).code
        if code == VK.ERROR_OUT_OF_DATE_KHR || code == VK.SUBOPTIMAL_KHR
            win.acquired = false
            win.acquirer = nothing
            resize!(win)                 # rebuild the swapchain; caller draws the next frame
            return nothing
        end
        throw_if_error(ctx, "vkQueuePresentKHR", result)
    end

    win.acquired = false
    win.acquirer = nothing
    win.current_frame = mod1(fi + 1, length(win.in_flight))
end

"""
    resize!(win::VulkanWindow)

Handle window resize by recreating the swapchain.
"""
function Base.resize!(win::VulkanWindow)
    checkopen(win)
    ctx = win.ctx
    VK.device_wait_idle(ctx.device)
    # Everything in flight has passed; give the frames' one-shots back now
    # rather than at the next sweep.
    drain!(ctx.default_bq)
    create_swapchain!(win)
end

function Base.isopen(win::VulkanWindow)
    win.handle.handle != C_NULL && !GLFW.WindowShouldClose(win.handle)
end

function Base.close(win::VulkanWindow)
    # Idempotent.
    win.handle.handle == C_NULL && return
    ctx = win.ctx
    VK.device_wait_idle(ctx.device)
    # Everything in flight has passed; give the frames' one-shots back now
    # rather than at the next sweep.
    drain!(ctx.default_bq)
    # Destroyed here, children first: finalizers do not run at exit, and a surface
    # outliving the instance is a leak (VUID-vkDestroyInstance-instance-00629).
    for v in win.views
        finalize(v)
    end
    empty!(win.views)
    empty!(win.images)                  # owned by the swapchain, not destroyed
    for s in win.image_available; finalize(s); end
    for s in win.render_finished; finalize(s); end
    for f in win.in_flight; finalize(f); end
    empty!(win.image_available); empty!(win.render_finished); empty!(win.in_flight)
    if win.swapchain !== nothing
        finalize(win.swapchain)
        win.swapchain = nothing
    end
    # Not `finalize`: the surface wraps GLFW's pointer by hand, so it has no VK.jl
    # destructor and no finalizer.
    VK.destroy_surface_khr(ctx.instance, win.surface)

    GLFW.DestroyWindow(win.handle)
    win.handle = GLFW.Window(C_NULL)
end

Base.size(win::VulkanWindow) = (Int(win.extent.width), Int(win.extent.height))
