# Manual render passes share native recording primitives with graph execution.

"""An open render pass containing the backend’s opaque recording handle."""
struct PassRecorder{H}
    handle::H
end

"""viewport!(p::PassRecorder, x, y, width, height)

Set viewport and matching scissor in pixels. Negative height puts clip-space +y
at the top, for a top-left pixel convention."""
viewport!(p::PassRecorder, x::Real, y::Real, w::Real, h::Real) =
    setviewport!(p.handle, x, y, w, h)

"""Set viewport and derived scissor on an open backend pass."""
function setviewport! end

"""draw!(p::PassRecorder, compiled, args, count; instances=1, indices=nothing)

Record a compiled draw. count is vertices, indices when indexed, or mesh workgroups."""
function draw!(p::PassRecorder, compiled, args, count::Integer;
               instances::Integer = 1, indices = nothing)
    record_draw!(p.handle, compiled, args, count; instances, indices)
    return nothing
end

"""
    bindings!(p::PassRecorder, compiled, bindings)

Make `bindings` the texture table the following draws sample from. Built by
[`bind_textures`](@ref).
"""
bindings!(p::PassRecorder, compiled, b) = use_bindings!(p.handle, compiled, b)

"""Resolve target attachments and load operations for begin_render_pass!."""
function passtargets(t::OffscreenTarget, clear, depth_clear = 1.0f0)
    load = clear === nothing ? Keep : Clear(clear)
    fb = t.fb
    d = depthimage(fb)
    # depth_clear=nothing loads the existing depth attachment.
    return ((colorimage(fb),), (load,), d,
            d === nothing ? nothing : (depth_clear === nothing ? Keep : Clear(depth_clear)))
end

function passtargets(t::WindowTarget, clear, depth_clear = nothing)
    load = clear === nothing ? Keep : Clear(clear)
    return ((currentimage(t.window),), (load,), nothing, nothing)
end

"""colorimage(fb) -> color attachment
    depthimage(fb) -> depth attachment or nothing

Framebuffer attachment accessors used by manual passes."""
function colorimage end
@doc (@doc colorimage) function depthimage end

"""
    currentimage(window) -> the image this frame draws into

The acquired swapchain image. A [`Window`](@ref)'s, because which one is current
is the window's bookkeeping and changes every frame.
"""
function currentimage end

"""pass!(f, dev, target; clear=nothing, depth_clear=1f0) -> result

Open a pass, call f(PassRecorder), and close even if f throws. clear=nothing
loads existing color; depth_clear=nothing similarly preserves depth."""
function pass!(f, device, target::RenderTarget; clear = nothing, depth_clear = 1.0f0)
    dev = todevice(device)
    targets, loads, depth, depth_load = passtargets(target, clear, depth_clear)
    h = begin_render_pass!(dev, targets, loads, depth, depth_load)
    p = PassRecorder(h)
    result = try
        f(p)
    finally
        end_render_pass!(h)
    end
    return result
end

# Fullscreen blit shaders use Julia column-major pixel order.

"""Where in the source a fragment at (x, y) is: column-major, as Julia stores it."""
@inline blitindex(x::Int32, y::Int32, height::Int32) = x * height + y + Int32(1)

"""A fullscreen triangle in clip space. Three vertices, no buffer."""
function blit_vertex()
    vid = vertex_index() - Int32(1)
    x = Float32(Int32(vid & Int32(1)) * 4 - 1)
    y = Float32(Int32((vid >> Int32(1)) & Int32(1)) * 4 - 1)
    return (position = Vec4f(x, clip_y(y), 0f0, 1f0),)
end

"""One pixel of `source`, premultiplied as it is stored."""
function blit_fragment(_inputs, source, width::Int32, height::Int32)
    ix = unsafe_trunc(Int32, frag_coord_x())
    iy = unsafe_trunc(Int32, frag_coord_y())
    return blitcolor(source[blitindex(ix, iy, height)])
end

"""Convert Vec4f or RGBA/BGRA-like source values to four attachment floats."""
@inline blitcolor(v::Vec4f) = v
@inline blitcolor(c) = Vec4f(c.r, c.g, c.b, c.alpha)

const BLIT_PIPELINE = GraphicsPipeline(;
    vertex = VertexShader(blit_vertex),
    fragment = FragmentShader(blit_fragment),
    blend = Opaque(), cull = NoCull(), depth = DepthOff())

"""checkblitsize(source, w, h)

Images use (height, width) column-major order; flattened pixel (x,y) is
x * height + y + 1. Reject a transposed matrix or insufficient pixels.
Other size mismatches with enough storage are allowed during resize transitions."""
function checkblitsize(a::AbstractMatrix, w::Integer, h::Integer)
    size(a) == (h, w) && return nothing
    # Permit temporary resize mismatches when the source still has enough pixels.
    size(a) == (w, h) && throw(DimensionMismatch(
        "blit source is $(size(a)) for a $(w)x$(h) target, which is the transpose of " *
        "the $(h)x$(w) it wants. A source is a (height, width) matrix; an image read " *
        "back with `copy_image_to_buffer!` packs rows and so comes out the other way."))
    length(a) >= w * h || throw(DimensionMismatch(
        "blit source holds $(length(a)) pixels and a $(w)x$(h) target needs $(w * h)"))
    return nothing
end

checkblitsize(a::AbstractVector, w::Integer, h::Integer) =
    length(a) >= w * h ? nothing : throw(DimensionMismatch(
        "blit source holds $(length(a)) pixels and a $(w)x$(h) target needs $(w * h)"))

"""blit!(dev, target, source; clear=true)

Draw a column-major device image over the target using a fullscreen triangle.
clear=false preserves the previous attachment contents before the draw."""
function blit!(device, target::RenderTarget, source; clear::Bool = true)
    dev = todevice(device)
    w, h = target_extent(target)
    checkblitsize(source, w, h)
    args = (resolve(dev, source), Int32(w), Int32(h))
    compiled = compile_draw(dev, BLIT_PIPELINE, (blittarget(target),), nothing, (), args)
    pass!(dev, target; clear = clear ? (0f0, 0f0, 0f0, 1f0) : nothing) do p
        viewport!(p, 0, 0, w, h)
        draw!(p, compiled, args, 3)
    end
    return nothing
end

"""draw!(device, pipeline, target, count; args=(), frag_args=(), instances=1,
          clear_color=(0f0,0f0,0f0,1f0), depth_clear=1f0, indices=nothing, bindings=nothing)

Compile and draw once through pass! and record_draw!. count is vertices,
indices when indexed, or workgroups for MeshPipeline."""
function draw!(device, pipeline::Union{GraphicsPipeline, MeshPipeline},
               target::RenderTarget, count::Integer;
               args = (), frag_args = (), instances::Integer = 1,
               clear_color::Union{Nothing, NTuple{4, Float32}} = (0f0, 0f0, 0f0, 1f0),
               depth_clear::Union{Nothing, Float32} = 1f0,
               indices = nothing, bindings = nothing)
    dev = todevice(device)
    # Manual draws resolve arguments here; graph draws resolve them during placement.
    args = argmap(a -> resolve(dev, a), args)
    frag_args = argmap(a -> resolve(dev, a), frag_args)
    compiled = compile_draw(dev, pipeline, (blittarget(target),), depthtarget(target),
                            args, frag_args; bindings)
    pass!(dev, target; clear = clear_color, depth_clear) do p
        bindings === nothing || bindings!(p, compiled, bindings)
        draw!(p, compiled, args, count; instances, indices)
    end
    return nothing
end

"""depthtarget(target) -> element type or nothing

Depth format for compile_draw. Existing framebuffer depth attachments are Float32."""
depthtarget(t::OffscreenTarget) = depthimage(t.fb) === nothing ? nothing : Float32
depthtarget(::WindowTarget) = nothing

"""blittarget(target) -> element type

Color element type for compile_draw; target_format returns the native API format."""
function blittarget end
