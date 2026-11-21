# The presentation layer: `CAMetalLayer` and its drawables, which Metal.jl does not
# wrap. Their types are Mantle's own.
#
# This is how anything reaches a screen on Apple platforms. It is QuartzCore rather
# than Metal (a `CAMetalLayer` is a `CALayer` that hands out textures), but
# everything it hands out is a `MTLTexture`. QuartzCore is loaded by
# `Mantle.initbackend!`, so the classes resolve wherever they are named.
#
# The model is not a swapchain of images the caller indexes. There is ONE drawable at
# a time: ask for the next, render into its texture, hand it back to a command buffer
# to present. `maximumDrawableCount` bounds how many can be outstanding, and
# `nextDrawable` BLOCKS when they all are, which is the frame pacing, and the reason
# it must not be called until the frame is ready to be recorded.
#
# Both wrappers are managed: `[CAMetalLayer layer]` and `nextDrawable` return
# autoreleased objects, which ObjectiveC.jl does not pool when they are wrapped as raw
# `id`s. Without a pool drained on the calling thread, every drawable and with it the
# layer stayed alive for the rest of the process, together with the layer's drawable
# textures. Managed wrappers make the calls pooled and their results retained, and
# released when the wrapper is collected.

using Metal.ObjectiveC.Foundation: NSObject, NSUInteger
using Metal.ObjectiveC: @objcproperties

"""A size in points, as Core Graphics spells one."""
struct CGSize
    width::Cdouble
    height::Cdouble
end

@objcwrapper CAMetalLayer <: NSObject

@objcproperties CAMetalLayer begin
    @autoproperty device::id{MTL.MTLDevice} setter = setDevice
    @autoproperty pixelFormat::MTL.MTLPixelFormat setter = setPixelFormat
    @autoproperty drawableSize::CGSize setter = setDrawableSize
    @autoproperty framebufferOnly::Bool setter = setFramebufferOnly
    @autoproperty maximumDrawableCount::NSUInteger setter = setMaximumDrawableCount
    @autoproperty displaySyncEnabled::Bool setter = setDisplaySyncEnabled
    @autoproperty contentsScale::Cdouble setter = setContentsScale
end

@objcwrapper CAMetalDrawable <: MTL.MTLDrawable

@objcproperties CAMetalDrawable begin
    @autoproperty texture::id{MTL.MTLTexture}
    @autoproperty layer::id{CAMetalLayer}
end

@objcwrapper managed = false CATransaction <: NSObject

"""
    flush_transactions()

Commit the implicit `CATransaction` that setting a layer property opens.

That transaction retains the layer until a run loop commits it, and nothing runs a
Core Animation run loop on a Julia thread: without a flush after the setters, the
layer and every drawable texture it allocated live until a run loop does commit it.
"""
flush_transactions() = @objc [CATransaction flush]::Nothing

"""
    CAMetalLayer(dev, width, height; format, vsync, readable) -> CAMetalLayer

A layer that hands out textures of `dev`, sized in PIXELS.

`readable` (the default) leaves `framebufferOnly` unset so the frame can be read
back: a layer that is only ever presented can let the driver pick a more compressed
layout, but a demo that also writes a PNG cannot. `drawableSize` is in pixels while
a layer's own `bounds` are in points, which is the whole of the Retina story: a
layer attached to a view has to be told the backing size or it renders at half
resolution and is scaled up.
"""
function CAMetalLayer(dev::MTL.MTLDevice, width::Integer, height::Integer;
                      format::MTL.MTLPixelFormat = MTL.MTLPixelFormatBGRA8Unorm,
                      vsync::Bool = true, readable::Bool = true)
    l = @objc [CAMetalLayer layer]::Union{Nothing,CAMetalLayer}
    l === nothing && error("Core Animation created no CAMetalLayer")
    l.device = dev
    l.pixelFormat = format
    l.drawableSize = CGSize(Cdouble(width), Cdouble(height))
    l.framebufferOnly = !readable
    l.displaySyncEnabled = vsync
    flush_transactions()
    return l
end

"""
    next_drawable(layer) -> Union{CAMetalDrawable,Nothing}

The next texture to render into, or `nothing` if none became free in time.

`nothing` is a normal answer, not an error: the layer bounds how many drawables are
outstanding and `nextDrawable` gives up rather than waiting for ever.
"""
next_drawable(layer::CAMetalLayer) =
    @objc [layer::id{CAMetalLayer} nextDrawable]::Union{Nothing,CAMetalDrawable}

"""
    present_drawable!(cmdbuf, drawable)

Show `drawable` when `cmdbuf` completes.

On the command buffer rather than on the drawable, so presentation is ordered behind
the work that drew it. `[drawable present]` exists and is the racy version: it shows
whatever is in the texture at the moment it is called.
"""
present_drawable!(cmdbuf::MTL.MTLCommandBuffer, drawable::CAMetalDrawable) =
    @objc [cmdbuf::id{MTL.MTLCommandBuffer} presentDrawable:drawable::id{CAMetalDrawable}]::Nothing
