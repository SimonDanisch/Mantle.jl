# A window on Metal: a `CAMetalLayer` and one drawable at a time.
#
# Not a swapchain of images the caller indexes. Vulkan hands out N images and
# says which one is next; a `CAMetalLayer` hands out ONE drawable, and
# `nextDrawable` blocks when the outstanding ones are all in flight — which is
# where the frame rate comes from. So `current_image_idx` has no counterpart
# here, and the accessors answer from the drawable rather than from an array.
#
# The layer works with no window at all. That is not a curiosity: it is what
# makes this path testable without a window server, and it is the same code
# either way — attaching the layer to a view is one call that changes nothing
# about how frames are drawn.

"""
A Metal presentation surface.

`handle` is the platform window, or `nothing` for a layer that is not on screen.
`drawable` is the texture the frame in flight is rendering into and is `nothing`
between frames — asking for a view outside a frame is a mistake with an answer
rather than a stale texture.
"""
mutable struct MetalWindow{T} <: Mantle.Window
    handle::Any
    layer::MTLm.CAMetalLayer
    width::Int
    height::Int
    drawable::Union{Nothing,MTLm.CAMetalDrawable}
    open::Bool
end

"""
    MetalWindow(T, width, height; vsync = true, handle = nothing)

A presentation surface of pixel type `T`, sized in PIXELS.

`T` is the element type, as everywhere a caller names a format in Mantle —
`BGRA{N0f8}` is what a display wants and what `mtlformat` lowers it to.
"""
function MetalWindow(::Type{T}, width::Integer, height::Integer;
                     vsync::Bool = true, handle = nothing) where {T}
    dev = Metal.device()
    layer = MTLm.CAMetalLayer(dev, width, height; format = mtlformat(T), vsync)
    return MetalWindow{T}(handle, layer, Int(width), Int(height), nothing, true)
end

"""
    attach!(w::MetalWindow, glfw_window)

Put the layer on a GLFW window's view, so the frames reach a screen.

Three calls and all three are needed. `glfwGetCocoaWindow` is in the GLFW
library but not in GLFW.jl's wrapper, so it is reached by `ccall`. A view is
not layer-backed by default — `setWantsLayer:` is what makes it so — and
`setLayer:` before that is silently undone when the view creates its own.

The size has to be re-asserted afterwards, and that is not a detail. A layer
with no EXPLICIT `drawableSize` derives one from its bounds times
`contentsScale`, and putting it on a view is what first gives it bounds — so
whatever size the layer was built with is silently replaced by the view's, in
points times the Retina factor. A 1600x900 window then hands out 2940x1602
drawables while `target_extent` still says 1600x900, and every pass that
indexes a buffer by hand with a stride reads far outside it.

What the layer ends up at is adopted rather than assumed: a window that cannot
have the size it asked for reports the size it got, because a `Mantle.Window`
that lies about its extent is worse than one that is the wrong size.
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
    @objc [view::id{OCObject} setLayer:(w.layer)::id{MTLm.CAMetalLayer}]::Nothing
    w.layer.contentsScale =
        @objc [ns::id{OCObject} backingScaleFactor]::Cdouble
    w.layer.drawableSize = MTLm.CGSize(Cdouble(w.width), Cdouble(w.height))
    got = w.layer.drawableSize
    w.width, w.height = round(Int, got.width), round(Int, got.height)
    w.handle = glfw_window
    return w
end

"""
    Window(backend, width, height; title = "", vsync = false)

The portable spelling, answered on this backend. `width` and `height` are
PIXELS.

`BGRA{N0f8}` is the element type, because that is what a display wants and
what `runtime/format.jl` documents as the portable name for it. Naming it here
was said to be impossible for three separate reasons, all of which read
"ColorTypes is not a dependency of Mantle"; it is, and this method is what was
missing because of that. A caller who wants another type reaches for
`MetalWindow(T, w, h)`.

The GLFW window is asked for in POINTS while the drawable is in PIXELS, so the
request is divided by the display's content scale. Ask for 1600x900 on a Retina
panel without it and the drawable is 3200x1800 — which is right for a renderer
that follows its surface and wrong for one built at a fixed resolution, and in
neither case what the caller asked for. It cost a smeared, distorted frame once,
because the composite pass indexed `hdr[py * 1600 + px]` over a 2940-wide target.
"""
function Mantle.Window(::Metal.MetalBackend, width::Integer, height::Integer;
                       title::AbstractString = "", vsync::Bool = false)
    GLFW.WindowHint(GLFW.CLIENT_API, GLFW.NO_API)
    gw = GLFW.CreateWindow(Int(width), Int(height), String(title))
    sx, sy = GLFW.GetWindowContentScale(gw)
    GLFW.SetWindowSize(gw, round(Int, width / sx), round(Int, height / sy))
    w = attach!(MetalWindow(BGRA{N0f8}, width, height; vsync), gw)
    # `attach!` ADOPTS whatever the layer could be given, so this is a real
    # check and not a restatement: a window that reports an extent its drawables
    # do not have breaks every pass that strides a buffer by hand.
    size(w) == (Int(width), Int(height)) || error(
        "asked for a $((Int(width), Int(height))) drawable, the layer gave $(size(w))")
    return w
end

# ── What a render pass asks a window ─────────────────────────────────────────

"""The texture this frame draws into. Only valid between `acquire_next_image!` and `present_frame!`."""
function Mantle.target_view(w::MetalWindow)
    d = w.drawable
    d === nothing && error("no drawable: a window's texture only exists between " *
                           "`acquire_next_image!` and `present_frame!`, and `run!` brackets them")
    return d.texture
end

# The same object. Vulkan distinguishes the image (what a barrier names) from the
# view (what a draw binds); a `MTLTexture` is both.
Mantle.target_image(w::MetalWindow) = Mantle.target_view(w)

Mantle.target_extent(w::MetalWindow) = (w.width, w.height)

Mantle.target_format(w::MetalWindow) = w.layer.pixelFormat

Base.eltype(::MetalWindow{T}) where {T} = T

Base.size(w::MetalWindow) = (w.width, w.height)

Base.isopen(w::MetalWindow) = w.open

function Base.close(w::MetalWindow)
    w.open = false
    w.drawable = nothing
    return nothing
end

# ── The frame ────────────────────────────────────────────────────────────────

"""
    resize!(w, width, height)

Give the layer a new size in PIXELS, and say whether it changed.

A `CAMetalLayer`'s `drawableSize` does not follow its view: the layer's bounds
are in POINTS and its drawables are in PIXELS, so a layer that is never told
keeps handing out the size it was made with and the display scales it up.

Only the surface moves here. Every attachment that FOLLOWS the surface moves
when `refit!` asks it to, and the plan is recompiled because the placement it
had was for the old size — which is why `run!` refits right after
`beforeframe!` has polled, resized and acquired, and before anything is
recorded: a half-recorded frame holding the old offsets is what that order
rules out.
"""
function Base.resize!(w::MetalWindow, width::Integer, height::Integer)
    (width, height) == (w.width, w.height) && return false
    w.width, w.height = Int(width), Int(height)
    w.layer.drawableSize = MTLm.CGSize(Cdouble(width), Cdouble(height))
    return true
end

"""
Pump the event queue and pick up a resize.

Without a platform window there is neither: a layer-only surface is whatever
size it was made, and `resize!` is how a caller changes it.
"""
function Mantle.beginframe!(w::MetalWindow)
    w.open || error("this window is closed")
    h = w.handle
    if h !== nothing
        GLFW.PollEvents()
        # A window the user closed is still fine to draw to — the loop
        # condition decides to stop, not this. In PIXELS, which is what
        # `GetFramebufferSize` gives and `GetWindowSize` does not.
        fw, fh = GLFW.GetFramebufferSize(h)
        # A minimised window reports zero, and a zero-sized drawable is not a
        # thing Metal will make. Keep the last good size until it comes back.
        (fw > 0 && fh > 0) && resize!(w, fw, fh)
    end
    return nothing
end

"""
Take the next drawable.

`nothing` back from the layer is a normal answer — every drawable is still in
flight — but a frame has nothing to draw into then, so it is an error here
rather than a silently skipped frame that looks like a stall.
"""
function Mantle.acquire_next_image!(w::MetalWindow)
    d = MTLm.next_drawable(w.layer)
    d === nothing &&
        error("no drawable became available; every one is still in flight. " *
              "Present the frames already recorded before starting another.")
    w.drawable = d
    return nothing
end

"""
Show the frame, ordered behind the work that drew it.

On a command buffer rather than on the drawable: `[drawable present]` shows
whatever is in the texture the moment it is called, which for a frame the GPU
is still writing is a torn one.
"""
function Mantle.present_frame!(d::MetalDevice, w::MetalWindow)
    dr = w.drawable
    dr === nothing && return nothing
    # On a command buffer of its own, committed behind the frame's passes — on
    # one queue that is the ordering — and committed here, because a frame that
    # has been presented is over.
    cb = framebuffer!(d.dev)
    MTLm.present_drawable!(cb, dr)
    commit!(cb)
    w.drawable = nothing
    return nothing
end

"""
    readback_window(w) -> Matrix

The last frame, read back.

Only between `acquire_next_image!` and `present_frame!` — after presenting, the
drawable belongs
to the compositor and its texture is not the caller's to read. A layer made
with `readable = false` (`framebufferOnly`) cannot be read at all, which is why
this backend does not set that.
"""
function Mantle.readback_window(w::MetalWindow{T}) where {T}
    tex = Mantle.target_view(w)
    dev = Mantle.Device(MetalAPI())
    row = w.width * sizeof(T)
    buf = MTLm.MTLBuffer(dev.dev, row * w.height; storage = Metal.SharedStorage)
    cmd = framebuffer!(dev.dev)
    MTLm.MTLBlitCommandEncoder(cmd) do enc
        MTLm.append_copy!(enc, buf, 0, row, 0, tex,
                          MTLm.MTLOrigin(0, 0, 0), MTLm.MTLSize(w.width, w.height, 1))
    end
    submitwait!(cmd)
    out = Matrix{T}(undef, w.width, w.height)
    unsafe_copyto!(pointer(out), convert(Ptr{T}, buf), length(out))
    return out
end

# A frame that failed after its drawable was taken: dropping the reference is
# all Metal needs — a `CAMetalDrawable` that is never presented is simply
# released, and the next `nextDrawable` hands out another.
function Mantle.abandonframe!(::MetalDevice, pl)
    for s in pl.graph.surfaces
        s.win.drawable = nothing
    end
    return nothing
end
