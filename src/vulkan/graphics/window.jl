# Window and swapchain management for Lava.jl
#
# Uses GLFW for window creation, Vulkan for surface/swapchain.
# Requires VK_KHR_surface + platform-specific surface extension.

import GLFW

"""
    VulkanWindow

A window with Vulkan surface and swapchain for presenting rendered frames.
Uses GLFW for cross-platform window management.
"""
mutable struct VulkanWindow <: Window
    handle::GLFW.Window
    surface::VK.SurfaceKHR
    swapchain::Union{Nothing, VK.SwapchainKHR}
    images::Vector{VK.Image}
    views::Vector{VK.ImageView}
    format::VK.Format
    # What the caller asked for, as opposed to `format`, which is what the
    # surface actually offered. Kept because `create_swapchain!` runs again on
    # every resize and would otherwise fall back to the default preference.
    #
    # This is the choice of what a shader's output MEANS. `_SRGB` makes the
    # hardware encode on write, so shaders write linear; `_UNORM` stores the
    # value as-is, so shaders write display-referred. Either lands on screen
    # correctly, because the presentation engine decodes sRGB in both cases —
    # they differ only in which space the renderer works in.
    preferred_format::VK.Format
    extent::VK.Extent2D
    # Per-frame-in-flight sync (rotating ring buffer)
    image_available::Vector{VK.Semaphore}
    render_finished::Vector{VK.Semaphore}
    in_flight::Vector{VK.Fence}
    current_frame::Int  # index into sync arrays (1-based, wraps)
    # Current frame state
    current_image_idx::UInt32
    acquired::Bool
    # The task that acquired the image still in flight, for the message in
    # `acquire_next_image!`. Only ever read to report a bug.
    acquirer::Union{Nothing, Task}
    # The framebuffer size the current swapchain was built for. A resize is not
    # reliably reported: OUT_OF_DATE is what the spec allows a driver to return,
    # not what it must, and SUBOPTIMAL is a *success* code that never reaches an
    # error branch. On Xwayland a shrunk window keeps presenting at the old size
    # and the compositor scales it, which looks like blur and reads back at the
    # wrong resolution. Comparing against this is what actually notices.
    fb_size::Tuple{Int,Int}
    # Owning context — swapchain/surface are bound to a specific device.
    ctx::VkContext
end

"""
    VulkanWindow(width, height; ctx, title="Lava", vsync=true, color_format=FORMAT_B8G8R8A8_SRGB)

Create a new window with Vulkan surface and swapchain.

`color_format` picks the swapchain format, which decides what a shader writing
to this window is expected to produce: `_SRGB` takes linear values and encodes
them in hardware, `_UNORM` takes display-referred values and stores them
verbatim. Pass `_UNORM` when the pixels being presented have already been
gamma-encoded, or they get encoded a second time. The surface may not offer the
requested format, in which case its first advertised one is used.
"""
function VulkanWindow(width::Integer, height::Integer;
                      ctx::VkContext,
                      title::String="Lava", vsync::Bool=true,
                      color_format::Union{VK.Format,Type}=VK.FORMAT_B8G8R8A8_SRGB,
                      srgb::Bool=false)
    # A Julia element type, lowered here exactly as `VulkanFramebuffer` lowers
    # one: `Mantle.Window(backend, w, h; color_format = BGRA{N0f8})` is the
    # portable spelling, and never naming `VK.Format` is the point of the
    # portable constructor. This took only `VK.Format`, so every window
    # RayMakie opened since the split failed here with a `TypeError`.
    color_format isa Type && (color_format = vkformat(VulkanAPI(), color_format; srgb))

    # Initialize GLFW (no OpenGL context — we use Vulkan)
    GLFW.Init()
    GLFW.WindowHint(GLFW.CLIENT_API, GLFW.NO_API)
    GLFW.WindowHint(GLFW.RESIZABLE, true)

    handle = GLFW.CreateWindow(width, height, title)

    # Create Vulkan surface via GLFW
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
        UInt32(0), false, nothing,
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

    # Query surface capabilities
    caps = unwrap(VK.get_physical_device_surface_capabilities_khr(phys, win.surface))

    # Pick the format the window was asked for, falling back to whatever the
    # surface lists first. Read from the window rather than a default, because
    # a resize comes back through here and must not undo the caller's choice.
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

    # Pick present mode
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

    # Determine extent
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

    # Image count
    image_count = caps.min_image_count + 1
    if caps.max_image_count > 0
        image_count = min(image_count, caps.max_image_count)
    end

    # readback_window copies out of a swapchain image, which needs TRANSFER_SRC.
    # Without it the copy and its layout transitions are spec violations that
    # only show up once validation is on: RADV tolerates them and returns
    # plausible pixels, so the readback looked correct for as long as nobody
    # checked. Only request it when the surface actually supports it.
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

    # Destroy what the old swapchain owned, now that the new one exists.
    # Reassigning the fields would leave these to finalizers, which run whenever
    # the GC gets to them, so a window resized a few times accumulates whole
    # swapchains' worth of images. Passing `old_swapchain` retires it; retiring
    # is not destroying.
    #
    # The wait is not optional. `vkDestroySwapchainKHR` requires every use of a
    # presentable image acquired from it to have completed, and an image view
    # must not be destroyed while a pending command references it — and this
    # function is reached from `acquire_next_image!` mid-render, with frames in
    # flight. Without it a resize is a use-after-free that surfaces as a GPUVM
    # fault several frames later, nowhere near the resize.
    if old_swapchain !== nothing
        VK.device_wait_idle(dev)
    end
    for v in win.views
        finalize(v)
    end
    if old_swapchain !== nothing
        finalize(old_swapchain)
    end

    # Create image views
    win.views = VK.ImageView[]
    for img in win.images
        view = VK.ImageView(dev, img, VK.IMAGE_VIEW_TYPE_2D, win.format,
            VK.ComponentMapping(
                VK.COMPONENT_SWIZZLE_IDENTITY,
                VK.COMPONENT_SWIZZLE_IDENTITY,
                VK.COMPONENT_SWIZZLE_IDENTITY,
                VK.COMPONENT_SWIZZLE_IDENTITY,
            ),
            VK.ImageSubresourceRange(
                VK.IMAGE_ASPECT_COLOR_BIT,
                UInt32(0), UInt32(1), UInt32(0), UInt32(1),
            ),
        )
        push!(win.views, view)
    end

    # Create per-frame sync primitives (one set per swapchain image)
    n = length(win.images)
    win.image_available = [VK.Semaphore(dev) for _ in 1:n]
    # One per swapchain IMAGE, not per frame-in-flight slot. A binary semaphore
    # used in a present cannot be reused until the image it was presented with is
    # re-acquired, and acquire hands back image indices in whatever order it
    # likes (observed: 3, 2, 1, 0, 3, 2, 1, 3). Keying this by frame slot signals
    # a semaphore the presentation engine still owns — sync validation calls it
    # "may still be in use by VkSwapchainKHR", and on screen it is tearing into
    # bands of stale pixels that any full GPU sync hides.
    win.render_finished = [VK.Semaphore(dev) for _ in 1:length(win.images)]
    win.in_flight = [VK.Fence(dev; flags=VK.FENCE_CREATE_SIGNALED_BIT) for _ in 1:n]
    win.current_frame = 1
end

"""
    checkopen(win)

A closed window has a destroyed GLFW handle and no swapchain, and asking it
anything dereferences that handle inside the driver: a segfault with a native
stack, not an error anyone can act on. Every entry point that touches the handle
checks first.

This is the handle, not `isopen`: a window whose close button has been clicked is
still a perfectly good window to draw to, and the loop condition is what decides
to stop.
"""
# ── What a render pass asks a window ─────────────────────────────────────────
#
# The same four questions it asks any attachment, so `WindowSurface` forwards
# and the shared graph never learns what a swapchain is. These bodies were IN
# `src/graph/build.jl`, reaching into `views`, `current_image_idx`, `extent` and
# `format` — this backend's fields, named in core.
#
# `current_image_idx` is what makes a window different from an image: the
# presentation engine chooses which of the swapchain's images the next frame
# writes, so the answer changes per frame and `acquire_next_image!` is what
# settles it.
target_view(w::VulkanWindow)   = w.views[w.current_image_idx + 1]
target_image(w::VulkanWindow)  = w.images[w.current_image_idx + 1]
target_extent(w::VulkanWindow) = (Int(w.extent.width), Int(w.extent.height))
target_format(w::VulkanWindow) = w.format

checkopen(win::VulkanWindow) =
    win.handle.handle == C_NULL &&
        error("this window has been closed; nothing can be drawn to it or read from it")

"""
    sync_swapchain!(win) -> Bool

Rebuild the swapchain if the window no longer matches it, and say whether it did.

A resize is not reliably reported: `OUT_OF_DATE` is what a driver *may* return and
`SUBOPTIMAL` is a success code that never reaches an error branch. On Xwayland
neither arrives — the window changes size and the swapchain keeps presenting at
the old one, which the compositor scales. Comparing the framebuffer size against
what the swapchain was built for is what notices.

Separate from `acquire_next_image!` so a caller that has to know the size *before*
it starts recording — a graph, whose other attachments have to cover the render
area — can bring the swapchain up to date first and bail out without a half
recorded frame. Zero is a minimised window and has no swapchain to build.
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

Acquire the next swapchain image. Returns the image index.
Must be called before recording rendering commands.
"""
function acquire_next_image!(win::VulkanWindow)
    checkopen(win)
    ctx = win.ctx
    dev = ctx.device

    sync_swapchain!(win)

    # The fence below is signalled by the submit `present!` makes, so waiting on
    # it while an image is still acquired waits for a submit the caller has not
    # recorded yet. That wait is a `vkWaitForFences` — a blocking foreign call,
    # so the task holding the unpresented frame can never be scheduled again, GC
    # cannot run, and the process is left unkillable with no stack to look at.
    #
    # How a frame gets left open: recording is not atomic against the Julia
    # scheduler. Anything that compiles a kernel mid-frame yields (the launch
    # plan is keyed on the world counter, so one method definition anywhere in
    # the session is enough), and a second task that renders in that window then
    # arrives here between the acquire and the present.
    if win.acquired
        who = win.acquirer === current_task() ?
              "the same task that is asking for another one" :
              "another task ($(win.acquirer)) that has not presented it yet"
        error("acquire_next_image!: this window already holds an image, acquired by " *
              who * ". A window is rendered by one task: pass it around, or hand it " *
              "over once the frame that owns it has been presented.")
    end

    fi = win.current_frame
    # The frame that last used this slot has passed once its fence has: the
    # one-shot it drew with is a submission like any other, in `bq.outstanding`,
    # and the next sweep gives it back.
    wait_for_fences!(ctx.default_bq, [win.in_flight[fi]])
    unwrap(VK.reset_fences(dev, [win.in_flight[fi]]))

    # Same on acquire: a stale swapchain is a condition to recover from, not an
    # error to die on.
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

Present the rendered frame to the screen.
Must be called after recording and submitting rendering commands.
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
    # OUT_OF_DATE / SUBOPTIMAL are not failures: the surface changed under us
    # (a resize, or the compositor remapping the window — it routinely happens on
    # the very first present) and the swapchain has to be rebuilt. Throwing here
    # turned an ordinary condition into a crash on startup.
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
    # Advance to next frame-in-flight slot
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
    # Idempotent -- safe to call multiple times
    win.handle.handle == C_NULL && return
    ctx = win.ctx
    VK.device_wait_idle(ctx.device)
    # Everything in flight has passed; give the frames' one-shots back now
    # rather than at the next sweep.
    drain!(ctx.default_bq)
    # Destroy the Vulkan objects here rather than leaving them to finalizers.
    # Julia does not run finalizers at exit, so the surface outlives the
    # instance and the validation layer reports it as leaked on every process
    # teardown (VUID-vkDestroyInstance-instance-00629). Child before parent:
    # views, then swapchain, then surface.
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
    # Explicitly, not `finalize`. The surface is wrapped by hand around the
    # pointer GLFW returns, so it never went through VK.jl's `init_handle!`:
    # its `destructor` field is `UndefInitializer()` and it shares the instance's
    # refcounter. Nothing would ever have destroyed it, which is why every run
    # ended with one leaked object. No finalizer is registered, so there is no
    # double-free to guard against.
    VK.destroy_surface_khr(ctx.instance, win.surface)

    GLFW.DestroyWindow(win.handle)
    win.handle = GLFW.Window(C_NULL)
end

Base.size(win::VulkanWindow) = (Int(win.extent.width), Int(win.extent.height))
