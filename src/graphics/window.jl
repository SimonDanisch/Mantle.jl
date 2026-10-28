# A window on every backend. Core creates, polls, sizes and closes the platform window
# (GLFW) and owns whose frame it is; the backend's surface, a swapchain or a
# `CAMetalLayer`, is what frames draw into, through the verbs below and the frame verbs
# (`acquire_next_image!`, `present_frame!`, `readback_window`, the `target_*` queries)
# on the surface itself.

"""
    windowsurface(device, handle, T, width, height; vsync, srgb) -> surface

The backend's presentation surface on the platform window `handle`, of element type
`T` (`srgb` picks the encoding form), `width` x `height` pixels: the window's
framebuffer.
"""
function windowsurface end

"""
    fitsurface!(surface, width, height) -> Bool

Size `surface` to its window's framebuffer, `width` x `height` pixels, both nonzero,
and say whether anything changed. Called before every acquire, since resizes are not
reliably reported.
"""
function fitsurface! end

"""
    closesurface!(surface)

Release `surface` once nothing in flight uses it, before its platform window goes.
"""
function closesurface! end

"""
    platformwindow(width, height, title) -> GLFW.Window

A window without a graphics API context whose framebuffer is `width` x `height`
pixels. GLFW sizes windows in points; the framebuffer's ratio to the window, not the
content scale, converts: a scaled X11 desktop reports a content scale and still draws
one pixel per point, macOS draws two.
"""
function platformwindow(width::Integer, height::Integer, title::AbstractString)
    GLFW.Init()
    GLFW.WindowHint(GLFW.CLIENT_API, GLFW.NO_API)
    GLFW.WindowHint(GLFW.RESIZABLE, true)
    handle = GLFW.CreateWindow(Int(width), Int(height), String(title))
    # Hints are process-global; left at `NO_API`, the next OpenGL window (GLMakie's)
    # gets no context.
    GLFW.DefaultWindowHints()
    fw, fh = framebuffersize(handle)
    if (fw, fh) != (width, height)
        ws = GLFW.GetWindowSize(handle)
        GLFW.SetWindowSize(handle, round(Int, width * ws.width / fw), round(Int, height * ws.height / fh))
    end
    return handle
end

"""
The framebuffer of a platform window in pixels. `GetWindowSize` is in points, and a
minimised window reports zero, at which no surface can be made.
"""
function framebuffersize(handle::GLFW.Window)
    fb = GLFW.GetFramebufferSize(handle)
    return (Int(fb.width), Int(fb.height))
end
framebuffersize(w::Window) = framebuffersize(w.handle)

function Window(dev::Device, width::Integer, height::Integer; title::AbstractString = "",
                vsync::Bool = false, color_format::Type = BGRA{N0f8}, srgb::Bool = false)
    handle = platformwindow(width, height, title)
    fw, fh = framebuffersize(handle)
    surface = windowsurface(dev, handle, color_format, fw, fh; vsync, srgb)
    return Window(handle, surface, nothing, true)
end

"""
    checkopen(window)

Throw if `window` was closed: its platform window is gone, and touching a destroyed
handle crashes in the driver. Not `isopen`: a window whose close button was clicked
can still be drawn to.
"""
checkopen(w::Window) =
    w.open || error("this window has been closed; nothing can be drawn to it or read from it")

"""Size the surface to the framebuffer, unless the window is minimised; whether it changed."""
function fitwindow!(w::Window)
    checkframeowner(w.acquirer, "resize!")
    fw, fh = framebuffersize(w)
    (fw > 0 && fh > 0) || return false
    return fitsurface!(w.surface, fw, fh)
end

"""Poll events and adopt the framebuffer size."""
function beginframe!(w::Window)
    checkopen(w)
    GLFW.PollEvents()
    fitwindow!(w)
    return nothing
end

"""
Take the next image, refusing a second one while a task holds an image (see
[`checkacquirable`](@ref)). The size is compared first on every acquire: OUT_OF_DATE
is optional, SUBOPTIMAL a success and Xwayland reports neither.
"""
function acquire_next_image!(w::Window)
    checkopen(w)
    checkacquirable(w.acquirer)
    w.acquirer = current_task()
    try
        fitwindow!(w)
        acquire_next_image!(w.surface)
    catch
        w.acquirer = nothing
        rethrow()
    end
    return nothing
end

"""A frame the window's image went back to presentation in: nobody holds it any more."""
presented!(w::Window) = (w.acquirer = nothing; nothing)

"""Present the acquired image of a frame nothing drew into."""
function present_frame!(dev::Device, w::Window)
    checkopen(w)
    present_frame!(dev, w.surface)
    presented!(w)
    return nothing
end

# Mid-frame there is no command buffer to add the copy to: read back after the frame
# is presented.
function readback_window(w::Window)
    checkopen(w)
    checkreadable(w.acquirer)
    fitwindow!(w)
    return readback_window(w.surface)
end

screenshot(w::Window) = readback_window(w)

Base.isopen(w::Window) = w.open && !GLFW.WindowShouldClose(w.handle)

"""
Close the surface, then the platform window, which would otherwise stay on screen
frozen on its last frame. Idempotent.
"""
function Base.close(w::Window)
    w.open || return nothing
    checkframeowner(w.acquirer, "close")
    w.open = false
    w.acquirer = nothing
    closesurface!(w.surface)
    GLFW.DestroyWindow(w.handle)
    # macOS takes the window off the screen when the event queue next runs.
    GLFW.PollEvents()
    return nothing
end

# What a pass asks of an attachment, answered by the surface.
target_view(w::Window) = target_view(w.surface)
target_image(w::Window) = target_image(w.surface)
target_extent(w::Window) = target_extent(w.surface)
target_format(w::Window) = target_format(w.surface)
currentimage(w::Window) = currentimage(w.surface)
Base.eltype(w::Window) = eltype(w.surface)
Base.size(w::Window) = target_extent(w.surface)
