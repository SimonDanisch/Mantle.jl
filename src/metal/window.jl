# Metal's half of core's `Window` (`graphics/window.jl`): a `CAMetalLayer` on the GLFW
# window's view, which hands out one drawable at a time (`nextDrawable` blocks while
# all are in flight), so there is no image index.

"""
A Metal presentation surface. `drawable` is the frame in flight's target and
`nothing` between frames, so a view requested outside a frame is an error.
"""
mutable struct MetalLayer{T,D<:MetalDevice}
    device::D
    layer::MTL.CAMetalLayer
    width::Int
    height::Int
    drawable::Union{Nothing,MTL.CAMetalDrawable}
end

"""
A layer of element type `T` (lowered by `pixelformat`, e.g. `BGRA{N0f8}`) on the GLFW
window `handle`'s view, `width` x `height` pixels. `setWantsLayer:` must precede
`setLayer:`, or the view replaces the layer with its own. `drawableSize` is set after
attaching, because a view gives the layer bounds that replace its size with points
times the backing scale; the surface then adopts the size the layer actually has, so
its extent never disagrees with its drawables.
"""
function windowsurface(d::MetalDevice, handle::GLFW.Window, ::Type{T}, width::Int, height::Int;
                       vsync::Bool, srgb::Bool) where {T}
    layer = MTL.CAMetalLayer(d.dev, width, height; format = pixelformat(MetalAPI(), T; srgb), vsync)
    nswindow = ccall((:glfwGetCocoaWindow, GLFW.libglfw), Ptr{Cvoid}, (Ptr{Cvoid},), handle.handle)
    nswindow == C_NULL &&
        error("this GLFW window has no NSWindow; GLFW was not built for Cocoa")
    # `id` takes the address as an integer, not as a `Ptr`.
    ns = id{OCObject}(UInt(nswindow))
    view = @objc [ns::id{OCObject} contentView]::id{OCObject}
    @objc [view::id{OCObject} setWantsLayer:true::Bool]::Nothing
    @objc [view::id{OCObject} setLayer:layer::id{MTL.CAMetalLayer}]::Nothing
    layer.contentsScale = @objc [ns::id{OCObject} backingScaleFactor]::Cdouble
    layer.drawableSize = MTL.CGSize(Cdouble(width), Cdouble(height))
    got = layer.drawableSize
    return MetalLayer{T,typeof(d)}(d, layer, round(Int, got.width), round(Int, got.height), nothing)
end

"""The texture this frame draws into. Only valid between `acquire_next_image!` and `present_frame!`."""
function Mantle.target_view(w::MetalLayer)
    d = w.drawable
    d === nothing && error("no drawable: a window's texture only exists between " *
                           "`acquire_next_image!` and `present_frame!`, and `run!` brackets them")
    return d.texture
end

# A `MTLTexture` is both the image and the view; a manual pass draws into it too.
Mantle.target_image(w::MetalLayer) = Mantle.target_view(w)
Mantle.currentimage(w::MetalLayer) = Mantle.target_view(w)

Mantle.target_extent(w::MetalLayer) = (w.width, w.height)

Mantle.target_format(w::MetalLayer) = w.layer.pixelFormat

Base.eltype(::MetalLayer{T}) where {T} = T

# The layer's `drawableSize` does not follow its view: set it in pixels.
function fitsurface!(w::MetalLayer, width::Int, height::Int)
    (width, height) == (w.width, w.height) && return false
    w.width, w.height = width, height
    w.layer.drawableSize = MTL.CGSize(Cdouble(width), Cdouble(height))
    return true
end

# The drawable in flight, if any, is released.
closesurface!(w::MetalLayer) = (w.drawable = nothing; nothing)

"""
Take the next drawable. Throws if every drawable is still in flight rather than
skipping the frame silently.
"""
function Mantle.acquire_next_image!(w::MetalLayer)
    w.drawable = nextdrawable(w)
    return nothing
end

function nextdrawable(w::MetalLayer)
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
function Mantle.present_frame!(d::MetalDevice, w::MetalLayer)
    dr = w.drawable
    dr === nothing && return nothing
    cb = framebuffer!(d.dev)
    MTL.present_drawable!(cb, dr)
    commit!(cb)
    w.drawable = nothing
    return nothing
end

"""
    readback_window(w) -> Matrix

Read back the next drawable between frames; it holds an earlier frame, as on
Vulkan. The drawable is presented afterwards: an unpresented one returns to the
layer only when the GC frees it, and the layer has three. Requires
`framebufferOnly` unset, which this backend never sets.
"""
function Mantle.readback_window(w::MetalLayer{T}) where {T}
    d = nextdrawable(w)
    tex = d.texture
    # The texture's extent, not the window's, so callers can check they agree.
    return readbackimage(cmd -> MTL.present_drawable!(cmd, d), w.device,
                         tex, T, Int(tex.width), Int(tex.height))
end

# A failed frame drops its drawable; an unpresented drawable is released.
function Mantle.abandonframe!(::MetalDevice, pl)
    for s in pl.graph.surfaces
        s.win.surface.drawable = nothing
    end
    return nothing
end
