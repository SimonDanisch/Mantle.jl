# Vulkan implementation of portable pass recording; closing the pass submits its one-shot.
"""Native attachment view, transition image and extent for a manual render pass."""
struct VulkanAttachment
    view::VK.ImageView
    image::VK.Image
    width::Int
    height::Int
end

colorimage(fb::VulkanFramebuffer) =
    VulkanAttachment(fb.color_view, fb.color_image, fb.width, fb.height)
depthimage(fb::VulkanFramebuffer) =
    fb.depth_view === nothing ? nothing :
    VulkanAttachment(fb.depth_view, fb.depth_image, fb.width, fb.height)
currentimage(w::VulkanWindow) =
    VulkanAttachment(target_view(w), target_image(w),
                     Int(w.extent.width), Int(w.extent.height))

# Target extent/format forwarding is defined by the resource wrappers.
"""eltypeof(::VK.Format) -> Type

Element type of a supported attachment format. This projection loses sRGB versus
UNORM; graph compilation must retain the native format rather than round-trip it."""
function eltypeof(f::VK.Format)
    f === VK.FORMAT_B8G8R8A8_SRGB   && return BGRA{N0f8}
    f === VK.FORMAT_B8G8R8A8_UNORM  && return BGRA{N0f8}
    f === VK.FORMAT_R8G8B8A8_SRGB   && return RGBA{N0f8}
    f === VK.FORMAT_R8G8B8A8_UNORM  && return RGBA{N0f8}
    f === VK.FORMAT_R16G16B16A16_SFLOAT && return RGBA{Float16}
    f === VK.FORMAT_R32G32B32A32_SFLOAT && return RGBA{Float32}
    f === VK.FORMAT_D32_SFLOAT      && return Float32
    f === VK.FORMAT_R32_UINT       && return UInt32
    f === VK.FORMAT_R32G32_UINT    && return SVector{2,UInt32}
    throw(ArgumentError("no Julia element type is registered for $f — add it " *
                        "here and to `vkformat`, which is the other direction"))
end

"""Compiled native pipeline plus the shader defining its packed argument layout."""
struct VulkanDraw
    pipeline::VulkanCompiledGraphicsPipeline
    shader::LavaGfxShader
end

# Raster and mesh draws share preparation; stage wrappers select their native compilation.
function compile_draw(dev::LavaDevice, p::Union{GraphicsPipeline, MeshPipeline},
                      color_formats, depth_format, vargs, fargs; bindings = nothing)
    return compile_vulkan_draw(dev, p,
        VK.Format[vkformat(VulkanAPI(), T) for T in color_formats],
        depth_format === nothing ? VK.FORMAT_UNDEFINED : vkformat(VulkanAPI(), depth_format),
        vargs, fargs; bindings)
end

"""Prepare graph/manual draws against exact native attachment formats.
The caller chooses argument storage and retains the returned pipeline/shader."""
function compile_vulkan_draw(dev::LavaDevice, p, color_formats, depth_format,
                             vargs, fargs; bindings=nothing)
    ctx = dev.ctx
    ad = adaptor(dev.bq)
    vtt = typeof(devargs(ad, rawargs(vargs)))
    ftt = typeof(devargs(ad, rawargs(fargs)))
    vfn, vtt, ffn, ftt = resolve_shader_pair(p, vtt, ftt)
    shader, compiled = ensure_compiled_with_shader!(p, vfn, ffn, vtt, ftt; ctx,
        color_format = color_formats, depth_format,
        # Sampled stages need the binding table's descriptor layout.
        descriptor_set_layout = bindings === nothing ? nothing : bindings.layout)
    # Fragment arguments choose the packed layout when present.
    isempty(fargs) || (shader = get_or_compile_gfx(ffn, ftt, :fragment))
    return VulkanDraw(compiled, shader)
end

"""A manual pass owns one one-shot and its emitter until end_render_pass! submits it."""
struct VulkanPassHandle{E}
    e::E
    o::OneShot
end

function begin_render_pass!(dev::LavaDevice, targets, loads, depth, depth_load)
    isempty(targets) && throw(ArgumentError(
        "begin_render_pass!: a pass needs at least one colour attachment"))
    bq = dev.bq
    o = openoneshot(bq)
    e = Emitter(o, nothing)
    first_t = first(targets)
    begin_pass!(e,
                VK.ImageView[t.view for t in targets],
                VK.Image[t.image for t in targets],
                VK.Extent2D(UInt32(first_t.width), UInt32(first_t.height));
                # Vectors: `perattachment` treats any other container as one value for all.
                clear_color = [clearvalue(l) for l in loads],
                load_op = [loadop(l) for l in loads],
                depth_view = depth === nothing ? nothing : depth.view,
                depth_image = depth === nothing ? nothing : depth.image,
                depth_clear = depth === nothing ? nothing : depthclearvalue(depth_load),
                depth_load_op = depth === nothing ? nothing : loadop(depth_load))
    # Whole-attachment viewport by default: a draw without the dynamic viewport
    # and scissor silently rasterises nothing.
    set_viewport!(e, 0, 0, first_t.width, first_t.height)
    return VulkanPassHandle(e, o)
end

"""The depth clear a load carries, as this backend's `Float32`."""
depthclearvalue(::Nothing) = nothing
depthclearvalue(l) = depthclear(l)

function end_render_pass!(h::VulkanPassHandle)
    end_pass!(h.e)
    bq = queueof(h.o)
    seal!(h.o)
    handover!(bq, submit!(bq, h.o), h.o; tag = :pass)
    return nothing
end

setviewport!(h::VulkanPassHandle, x::Real, y::Real, w::Real, hgt::Real) =
    set_viewport!(h.e, x, y, w, hgt)

# The set binds against the pipeline's layout, not the `VulkanDraw`.
use_bindings!(h::VulkanPassHandle, d::VulkanDraw, bindings) =
    use_bindings!(h.e, d.pipeline, bindings)

function record_draw!(h::VulkanPassHandle, d::VulkanDraw, args, count::Integer;
                      instances::Integer = 1, indices = nothing)
    e = h.e
    addr = pack_gfx_args_bda(e.owner, args, d.shader.push_info)
    if ismeshpipeline(d.pipeline)
        indices === nothing || throw(ArgumentError(
            "record_draw!: a mesh pipeline takes no index buffer — the mesh stage " *
            "writes its own primitive indices. `count` is its workgroup count."))
        draw_mesh_in_pass!(e, d.pipeline, count; push_bda = addr)
    elseif indices === nothing
        draw_in_pass!(e, d.pipeline, count; push_bda = addr, instances)
    else
        # Held because the device reads the indices; the offset is the view's,
        # since an index buffer may be a slice.
        hold!(e, indices)
        mb = indices.buf[]::VkManagedBuffer
        draw_indexed_in_pass!(e, d.pipeline, count; push_bda = addr, instances,
                              indices_buffer = mb.buffer,
                              indices_offset = UInt64(pool_offset(mb) + indices.offset))
    end
    return nothing
end
