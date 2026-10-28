# A window on Metal: a `CAMetalLayer` that hands out one drawable at a time
# (`nextDrawable` blocks while all are in flight), so there is no image index.
# The layer works without a platform window, which keeps this path testable
# without a window server.

"""
A Metal presentation surface. `handle` is the platform window, or `nothing` for
an offscreen layer. `drawable` is the frame in flight's target and `nothing`
between frames, so a view requested outside a frame is an error.
"""
mutable struct MetalWindow{T} <: Mantle.Window
    handle::Any
    layer::MTL.CAMetalLayer
    width::Int
    height::Int
    drawable::Union{Nothing,MTL.CAMetalDrawable}
    open::Bool
    # The task that took `drawable`, until it presents (`checkacquirable`).
    acquirer::Union{Nothing,Task}
end

"""
    MetalWindow(T, width, height; vsync = true, srgb = false, handle = nothing)

A presentation surface of element type `T` (lowered by `mtlformat`, e.g.
`BGRA{N0f8}`), sized in pixels.
"""
function MetalWindow(::Type{T}, width::Integer, height::Integer;
                     vsync::Bool = true, srgb::Bool = false, handle = nothing) where {T}
    dev = Metal.device()
    layer = MTL.CAMetalLayer(dev, width, height; format = mtlformat(T; srgb), vsync)
    return MetalWindow{T}(handle, layer, Int(width), Int(height), nothing, true, nothing)
end

"""
    attach!(w::MetalWindow, glfw_window)

Put the layer on a GLFW window's view. `setWantsLayer:` must precede `setLayer:`,
or the view replaces the layer with its own. `drawableSize` is set again after
attaching, because a view gives the layer bounds that replace its size with
points times the backing scale. The window then adopts the size the layer
actually has, so its extent never disagrees with its drawables.
"""
function attach!(w::MetalWindow, glfw_window)
    nswindow = ccall((:glfwGetCocoaWindow, GLFW.libglfw), Ptr{Cvoid},
                     (Ptr{Cvoid},), glfw_window.handle)
    nswindow == C_NULL &&
        error("this GLFW window has no NSWindow; GLFW was not built for Cocoa")
    # `id` takes the address as an integer, not as a `Ptr`.
    ns = id{OCObject}(UInt(nswindow))
    view = @objc [ns::id{OCObject} contentView]::id{OCObject}
    @objc [view::id{OCObject} setWantsLayer:true::Bool]::Nothing
    @objc [view::id{OCObject} setLayer:(w.layer)::id{MTL.CAMetalLayer}]::Nothing
    w.layer.contentsScale =
        @objc [ns::id{OCObject} backingScaleFactor]::Cdouble
    w.layer.drawableSize = MTL.CGSize(Cdouble(w.width), Cdouble(w.height))
    got = w.layer.drawableSize
    w.width, w.height = round(Int, got.width), round(Int, got.height)
    w.handle = glfw_window
    return w
end

"""
    Window(backend, width, height; title = "", vsync = false,
           color_format = BGRA{N0f8}, srgb = false)

A GLFW window with a Metal layer, `width` x `height` in pixels. The GLFW size is
in points, so the request is divided by the content scale. Throws if the layer's
drawable size differs from the request. `color_format` is the portable keyword
the Vulkan backend also takes.
"""
function Mantle.Window(::Metal.MetalBackend, width::Integer, height::Integer;
                       title::AbstractString = "", vsync::Bool = false,
                       color_format::Type = BGRA{N0f8}, srgb::Bool = false)
    GLFW.WindowHint(GLFW.CLIENT_API, GLFW.NO_API)
    gw = GLFW.CreateWindow(Int(width), Int(height), String(title))
    # Hints are process-global; left at `NO_API`, the next OpenGL window (e.g.
    # GLMakie's) gets no context.
    GLFW.DefaultWindowHints()
    sx, sy = GLFW.GetWindowContentScale(gw)
    GLFW.SetWindowSize(gw, round(Int, width / sx), round(Int, height / sy))
    w = attach!(MetalWindow(color_format, width, height; vsync, srgb), gw)
    # `attach!` adopts the layer's actual size; passes that stride a buffer by
    # hand rely on the requested extent.
    size(w) == (Int(width), Int(height)) || error(
        "asked for a $((Int(width), Int(height))) drawable, the layer gave $(size(w))")
    return w
end

"""The texture this frame draws into. Only valid between `acquire_next_image!` and `present_frame!`."""
function Mantle.target_view(w::MetalWindow)
    d = w.drawable
    d === nothing && error("no drawable: a window's texture only exists between " *
                           "`acquire_next_image!` and `present_frame!`, and `run!` brackets them")
    return d.texture
end

# A `MTLTexture` is both the image and the view; a manual pass draws into it too.
Mantle.target_image(w::MetalWindow) = Mantle.target_view(w)
Mantle.currentimage(w::MetalWindow) = Mantle.target_view(w)

Mantle.target_extent(w::MetalWindow) = (w.width, w.height)

Mantle.target_format(w::MetalWindow) = w.layer.pixelFormat

Base.eltype(::MetalWindow{T}) where {T} = T

Base.size(w::MetalWindow) = (w.width, w.height)

# Closed by `close` or by the platform window's close button.
Base.isopen(w::MetalWindow) =
    w.open && (w.handle === nothing || !GLFW.WindowShouldClose(w.handle))

function Base.close(w::MetalWindow)
    Mantle.checkframeowner(w.acquirer, "close")
    w.open = false
    w.drawable = nothing
    w.acquirer = nothing
    # Destroy the platform window too, or it stays on screen frozen on its last
    # frame and its close button does nothing.
    h = w.handle
    if h !== nothing
        w.handle = nothing
        GLFW.DestroyWindow(h)
        # macOS takes the window off the screen when the event queue next runs.
        GLFW.PollEvents()
    end
    return nothing
end

"""
    resize!(w, width, height)

Set the layer's `drawableSize` in pixels (it does not follow its view); return
whether it changed. Attachments that follow the surface are refit by `run!`
after `beforeframe!` and before recording.
"""
function Base.resize!(w::MetalWindow, width::Integer, height::Integer)
    (width, height) == (w.width, w.height) && return false
    # A worker may be inside `nextDrawable` on this layer for another task's frame.
    Mantle.checkframeowner(w.acquirer, "resize!")
    w.width, w.height = Int(width), Int(height)
    w.layer.drawableSize = MTL.CGSize(Cdouble(width), Cdouble(height))
    return true
end

"""
Poll events and adopt the framebuffer size. A layer without a platform window
keeps its size until `resize!`.
"""
function Mantle.beginframe!(w::MetalWindow)
    w.open || error("this window is closed")
    h = w.handle
    if h !== nothing
        GLFW.PollEvents()
        # A user-closed window still draws; the loop condition stops it.
        # `GetFramebufferSize` is in pixels, `GetWindowSize` is not.
        fw, fh = GLFW.GetFramebufferSize(h)
        # Minimised windows report zero, which Metal cannot allocate.
        (fw > 0 && fh > 0) && resize!(w, fw, fh)
    end
    return nothing
end

"""
Take the next drawable. Throws if every drawable is still in flight rather than
skipping the frame silently.
"""
function Mantle.acquire_next_image!(w::MetalWindow)
    Mantle.checkacquirable(w.acquirer)
    # Claimed before the wait for a drawable, which yields: a task arriving
    # meanwhile finds the window taken (`checkacquirable`, `checkframeowner`).
    w.acquirer = current_task()
    try
        w.drawable = nextdrawable(w)
    catch
        w.acquirer = nothing
        rethrow()
    end
    return nothing
end

function nextdrawable(w::MetalWindow)
    # `nextDrawable` blocks, up to a second, while every drawable is in flight,
    # and nothing can ask the layer first, so it always waits on a worker
    # (`hostwait`); the call is GC-safe, as every `@objc` is.
    d = something(hostwait(MTL.next_drawable, nothing, w.layer))
    d === nothing &&
        error("no drawable became available; every one is still in flight. " *
              "Present the frames already recorded before starting another.")
    return d
end

"""
Present the frame through a command buffer committed behind the frame's passes
on the same queue. `[drawable present]` would show the texture immediately,
torn if the GPU is still writing it.
"""
function Mantle.present_frame!(d::MetalDevice, w::MetalWindow)
    dr = w.drawable
    dr === nothing && return nothing
    cb = framebuffer!(d.dev)
    MTL.present_drawable!(cb, dr)
    commit!(cb)
    w.drawable = nothing
    w.acquirer = nothing
    return nothing
end

"""
    readback_window(w) -> Matrix

Read back the next drawable between frames; it holds an earlier frame, as on
Vulkan. The drawable is presented afterwards: an unpresented one returns to the
layer only when the GC frees it, and the layer has three. Requires
`framebufferOnly` unset, which this backend never sets.
"""
function Mantle.readback_window(w::MetalWindow{T}) where {T}
    Mantle.checkreadable(w.acquirer)
    d = nextdrawable(w)
    tex = d.texture
    # The texture's extent, not the window's, so callers can check they agree.
    return readbackimage(cmd -> MTL.present_drawable!(cmd, d), Mantle.Device(MetalAPI()),
                         tex, T, Int(tex.width), Int(tex.height))
end

Mantle.screenshot(w::MetalWindow) = Mantle.readback_window(w)

# A failed frame drops its drawable; an unpresented drawable is released.
function Mantle.abandonframe!(::MetalDevice, pl)
    for s in pl.graph.surfaces
        s.win.drawable = nothing
        s.win.acquirer = nothing
    end
    return nothing
end
