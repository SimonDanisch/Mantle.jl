# The hand-recorded render pass, on this backend.
#
# Core's `pass!` (`graphics/record.jl`) is the imperative half of the graph:
# same three primitives, with the ORDER decided by a caller instead of by a
# plan. RayMakie's overlay compositor is written against it, and on this backend
# every one of its verbs was a `MethodError` — the family that existed here took
# an `Emitter` and was reached through `Base.get_extension`.
#
# What the walk uses is unchanged: `beginrender!`/`emitdraw!`/`endrender!` write
# into the recording a plan owns (`graph.jl`). What is here is the other caller,
# and it owns its own one-shot: `begin_render_pass!` opens one, `end_render_pass!`
# seals and submits it, so a hand-recorded pass reaches the device the moment it
# closes, exactly as every other ad hoc piece of work does.

"""
What a colour or depth attachment is to a hand-recorded pass: the view it draws
through, and the image whose layout it transitions.

Two handles and not one, which is why this type exists rather than passing the
view: dynamic rendering binds a `VkImageView`, and the transition into
`COLOR_ATTACHMENT_OPTIMAL` names the `VkImage`. Metal's attachment is one
texture and needs no wrapper, which is why the value core passes through is
opaque to it.
"""
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

# `target_extent` is already answered for a framebuffer (`graph.jl`) and for a
# window (`graphics/window.jl`); the two TARGET wrappers are core's and forward
# to those.

"""
The element type of a target's colour attachment — what a pipeline is compiled
against. `target_format` answers with the `VkFormat`; this is the Julia type it
was lowered from, which is what every caller of `compile_draw` names.
"""
blittarget(t::OffscreenTarget) = eltypeof(target_format(t.fb))
blittarget(t::WindowTarget)    = eltypeof(target_format(t.window))

"""
    eltypeof(::VK.Format) -> Type

The Julia element type a format stores, for the formats a render target can
have. The inverse of `vkformat`, and only over the attachment formats: a caller
names a target's element type to compile a pipeline for it, and the target
carries a driver format.
"""
function eltypeof(f::VK.Format)
    f === VK.FORMAT_B8G8R8A8_SRGB   && return BGRA{N0f8}
    f === VK.FORMAT_B8G8R8A8_UNORM  && return BGRA{N0f8}
    f === VK.FORMAT_R8G8B8A8_SRGB   && return RGBA{N0f8}
    f === VK.FORMAT_R8G8B8A8_UNORM  && return RGBA{N0f8}
    f === VK.FORMAT_R16G16B16A16_SFLOAT && return RGBA{Float16}
    f === VK.FORMAT_R32G32B32A32_SFLOAT && return RGBA{Float32}
    f === VK.FORMAT_D32_SFLOAT      && return Float32
    throw(ArgumentError("no Julia element type is registered for $f — add it " *
                        "here and to `vkformat`, which is the other direction"))
end

"""
A draw compiled for a set of attachments: the pipeline, and the shader whose
push constant block its arguments are packed against.

`compile_draw` is core's name and takes ELEMENT TYPES for the attachments, like
everywhere a caller names a format in Mantle; the lowering to `VkFormat` is
here. The graph's `compiledraw` builds the same pair for a recorded pass and
carries the plan's argument offset with it, which a hand-recorded draw has no
use for — it packs into scratch the one-shot owns.
"""
struct VulkanDraw
    pipeline::VulkanCompiledGraphicsPipeline
    shader::LavaGfxShader
end

function compile_draw(dev::LavaDevice, p::GraphicsPipeline, color_formats,
                      depth_format, vargs, fargs; bindings = nothing)
    ctx = dev.ctx
    ad = adaptor(dev.bq)
    vtt = typeof(convert_args(devargs(ad, rawargs(vargs))))
    ftt = typeof(convert_args(devargs(ad, rawargs(fargs))))
    vfn, vtt, ffn, ftt = resolve_shader_pair(p, vtt, ftt)
    shader, compiled = ensure_compiled_with_shader!(p, vfn, ffn, vtt, ftt; ctx,
        color_format = VK.Format[vkformat(VulkanAPI(), T) for T in color_formats],
        depth_format = depth_format === nothing ? VK.FORMAT_UNDEFINED :
                                                  vkformat(VulkanAPI(), depth_format),
        # The layout the sampled textures are bound through. A pipeline built
        # without it has no set 0, and `vkCmdBindDescriptorSets` against it is
        # invalid — which is why this travels with the compile and not only with
        # the bind.
        descriptor_set_layout = bindings === nothing ? nothing : bindings.layout)
    # Whichever stage's shader reads the block. The same rule the graph's
    # `compiledraw` applies, and for the same reason: a fullscreen pass whose
    # vertex stage takes nothing and whose fragment stage reads a g-buffer is
    # the case that makes the usual answer wrong.
    isempty(fargs) || (shader = get_or_compile_gfx(ffn, ftt, :fragment; ctx))
    return VulkanDraw(compiled, shader)
end

"""
A pass open right now: the one-shot its commands are going into, and the emitter
over it. Core holds it as an opaque handle.
"""
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
                # VECTORS, one entry per attachment: `perattachment` reads any
                # other container as a single value repeated, so a tuple of one
                # `nothing` arrives as the clear colour itself.
                clear_color = [clearvalue(l) for l in loads],
                load_op = [loadop(l) for l in loads],
                depth_view = depth === nothing ? nothing : depth.view,
                depth_image = depth === nothing ? nothing : depth.image,
                depth_clear = depth === nothing ? nothing : depthclearvalue(depth_load),
                depth_load_op = depth === nothing ? nothing : loadop(depth_load))
    # The viewport covers the whole attachment until a caller says otherwise:
    # viewport and scissor are dynamic state, and a pipeline drawn without them
    # rasterises nothing without raising anything.
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

# The set is bound against the PIPELINE's layout, so the draw's pipeline is what
# `use_bindings!` needs — not the `VulkanDraw` that wraps it.
use_bindings!(h::VulkanPassHandle, d::VulkanDraw, bindings) =
    use_bindings!(h.e, d.pipeline, bindings)

function record_draw!(h::VulkanPassHandle, d::VulkanDraw, args, count::Integer;
                      instances::Integer = 1, indices = nothing)
    e = h.e
    addr = pack_gfx_args_bda(e.owner, args, d.shader.push_info)
    if indices === nothing
        draw_in_pass!(e, d.pipeline, count; push_bda = addr, instances)
    else
        # The device READS the indices, so the submission has to outlive them —
        # and the offset is the view's, because an index buffer may be a slice.
        hold!(e, indices)
        mb = indices.buf[]::VkManagedBuffer
        draw_indexed_in_pass!(e, d.pipeline, count; push_bda = addr, instances,
                              indices_buffer = mb.buffer,
                              indices_offset = UInt64(pool_offset(mb) + indices.offset))
    end
    return nothing
end
