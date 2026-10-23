# Native shader/pipeline preparation, shared graphics argument packing and presentation.
"""get_or_compile_gfx(f, tt, stage; config) -> LavaGfxShader

Cache device-independent SPIR-V against current CodeInstance identity."""
get_or_compile_gfx(@nospecialize(f), @nospecialize(tt), stage::Symbol; config=nothing) =
    cached_gfx_shader(f, tt, stage; config)

"""
The key of a compiled pipeline: its shaders by identity (not their functions, so a
stage recompiled after an edit gives a new pipeline) and the state baked in.
"""
function pipeline_cache_key(shaders::Tuple, pipeline, color_format, depth_format, descriptor_set_layout)
    formats = color_format isa Union{Tuple,AbstractVector} ? Tuple(color_format) : (color_format,)
    return (CacheIdentity(shaders), formats, depth_format, pipeline_state_key(pipeline),
            descriptor_set_layout !== nothing)
end

"""Compile native raster stages/pipeline and return (vertex_shader, pipeline)."""
function ensure_compiled_with_shader!(pipeline::GraphicsPipeline, vert_fn, frag_fn, tt_vertex, tt_fragment;
                              color_format=VK.FORMAT_B8G8R8A8_SRGB,
                              depth_format=VK.FORMAT_UNDEFINED,
                              descriptor_set_layout=nothing,
                              ctx::VkContext)
    vert = get_or_compile_gfx(vert_fn, tt_vertex, :vertex)
    frag = get_or_compile_gfx(frag_fn, tt_fragment, :fragment)

    geom = nothing
    if pipeline.geometry !== nothing
        # Geometry wrappers supply the emitter and primitive inputs before user arguments.
        geom_cfg = Mantle.stageconfig(pipeline.geometry)
        geom = get_or_compile_gfx(geometrystage(pipeline, geom_cfg), tt_vertex,
                                  :geometry; config=geom_cfg)
    end
    tc = nothing
    tess_cfg = nothing
    if pipeline.tess_control !== nothing
        tc_fn, tess_cfg = pipeline.tess_control
        tc = get_or_compile_gfx(tc_fn, tt_vertex, :tess_control; config=tess_cfg)
    end
    te = pipeline.tess_eval === nothing ? nothing :
         get_or_compile_gfx(pipeline.tess_eval, tt_vertex, :tess_eval; config=tess_cfg)

    cache_key = pipeline_cache_key((vert, frag, geom, tc, te), pipeline, color_format,
                                   depth_format, descriptor_set_layout)
    compiled = get!(ctx.caches.gfx_pipelines, cache_key) do
        return create_graphics_pipeline(vert.spirv_bytes, frag.spirv_bytes;
            ctx, blend=pipeline.blend, cull=pipeline.cull,
            topology=pipeline.topology, depth=pipeline.depth,
            color_format=color_format, depth_format=depth_format,
            push_constant_size=max(vert.push_info.push_size, frag.push_info.push_size),
            geometry_spirv=geom === nothing ? nothing : geom.spirv_bytes,
            tess_ctrl_spirv=tc === nothing ? nothing : tc.spirv_bytes,
            tess_eval_spirv=te === nothing ? nothing : te.spirv_bytes,
            tess_config=tess_cfg,
            descriptor_set_layout=descriptor_set_layout)
    end
    return vert, compiled
end

"""Compile native mesh stages/pipeline and return (mesh_shader, pipeline)."""
function ensure_compiled_with_shader!(pipeline::MeshPipeline, mesh_fn, frag_fn, tt_mesh, tt_fragment;
                              color_format=VK.FORMAT_B8G8R8A8_SRGB,
                              depth_format=VK.FORMAT_UNDEFINED,
                              descriptor_set_layout=nothing,
                              ctx::VkContext)
    pipeline.object === nothing || throw(ArgumentError(
        "ensure_compiled!: this pipeline has an object stage, and the Vulkan " *
        "backend dispatches its mesh threadgroups from the host. The task stage " *
        "is what an object stage lowers onto here, and nothing emits one yet."))
    mesh = get_or_compile_gfx(mesh_fn, tt_mesh, :mesh; config = Mantle.meshconfig(pipeline))
    frag = get_or_compile_gfx(frag_fn, tt_fragment, :fragment)

    cache_key = pipeline_cache_key((mesh, frag), pipeline, color_format, depth_format,
                                   descriptor_set_layout)
    compiled = get!(ctx.caches.gfx_pipelines, cache_key) do
        # No `topology`: a mesh stage's output primitive is an execution mode of
        # its entry point, carried by `MeshConfig`.
        return create_graphics_pipeline(nothing, frag.spirv_bytes;
            ctx, mesh_spirv=mesh.spirv_bytes,
            blend=pipeline.blend, cull=pipeline.cull, depth=pipeline.depth,
            color_format=color_format, depth_format=depth_format,
            push_constant_size=max(mesh.push_info.push_size, frag.push_info.push_size),
            descriptor_set_layout=descriptor_set_layout)
    end
    return mesh, compiled
end

ensure_compiled!(pipeline::Union{GraphicsPipeline,MeshPipeline}, args...; kw...) =
    last(ensure_compiled_with_shader!(pipeline, args...; kw...))

"""Resolve wrapped vertex/fragment functions and type tuples. Varyings come from
the last geometry-producing stage; empty outputs support depth-only passes."""
function resolve_shader_pair(pipeline, vert_tt::Type, frag_tt::Type)
    last = Mantle.lastgeometrystage(pipeline)
    vout = Mantle.outputtype(last)
    # Both stages must agree on Flat interpolation, including integer varyings.
    flats = Mantle.flatoutputs(last)
    wrapped_vert = VertexWrapper{typeof(Mantle.stagefunction(pipeline.vertex)), flats}()
    wrapped_frag = FragmentWrapper{typeof(Mantle.stagefunction(pipeline.fragment)), vout, flats}()
    return wrapped_vert, vert_tt, wrapped_frag, frag_tt
end

"""Resolve mesh/fragment wrappers against one output declaration. Flat outputs
use primitive storage and PerPrimitiveEXT on both producing/consuming stages."""
function resolve_shader_pair(pipeline::MeshPipeline, mesh_tt::Type, frag_tt::Type)
    vout = Mantle.outputtype(pipeline.mesh)
    flats = Mantle.flatoutputs(pipeline.mesh)
    topology = typeof(Mantle.meshconfig(pipeline).topology)
    wrapped_mesh = MeshWrapper{typeof(Mantle.stagefunction(pipeline.mesh)), vout, flats, topology}()
    wrapped_frag = FragmentWrapper{typeof(Mantle.stagefunction(pipeline.fragment)), vout,
                                   flats, flats}()
    return wrapped_mesh, mesh_tt, wrapped_frag, frag_tt
end

"""Wrap geometry inputs from the vertex output declaration, outputs from its own
declaration, and primitive width from input topology."""
function geometrystage(pipeline::GraphicsPipeline, cfg)
    gs = pipeline.geometry
    return GeometryWrapper{typeof(Mantle.stagefunction(gs)),
                           Mantle.outputtype(pipeline.vertex),
                           Mantle.outputtype(gs),
                           Mantle.flatoutputs(gs),
                           Mantle.primitivevertices(cfg.input_topology)}()
end

# Portable draw sequencing lives in graphics/record.jl.
"""Hold original argument leaves, adapt to device values, and pack into caller-owned
mapped argument memory. Shared by graph slots and manual draw scratch."""
function pack_gfx_args!(owner::Closed, args, ptr::Ptr{UInt8}, address::UInt64,
                        info::PushConstantInfo)
    holdleaves!(owner, args)
    converted = devargs(LavaAdaptor(owner), args)
    pack_args_direct!(owner, ptr, address, info.arg_offsets, info.arg_buffer_size,
                      info.byval_llvm_sizes, converted)
    return nothing
end

"Pack manual draw arguments into the owner's scratch and return its device address."
function pack_gfx_args_bda(batch::O, args, push_info::PushConstantInfo) where {O<:Closed}
    (push_info.push_size == 0 || isempty(args)) && return UInt64(0)
    total_size = push_info.arg_buffer_size + compute_inline_extra_from_byval(push_info.byval_llvm_sizes)
    arg_buf = get_arg_buffer(batch, total_size)
    pack_gfx_args!(batch, args, arg_buf.mapped_ptr, arg_buf.address, push_info)
    return arg_buf.address
end

"""Transition the drawn swapchain image from COLOR_ATTACHMENT_OPTIMAL to PRESENT_SRC; retain its window."""
function presentready!(e::Emitter, win::VulkanWindow)
    image = win.images[win.current_image_idx + 1]
    imagebarrier!(e.cmd, image,
        VK.IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK.IMAGE_LAYOUT_PRESENT_SRC_KHR,
        VK.PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK.PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
        VK.ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK.AccessFlag(0))
    win.holdsframe[win.current_image_idx + 1] = true
    # The swapchain and its images are named by this frame until it has passed.
    hold!(e, win)
    return nothing
end

"""Discard an acquired, undrawn image and transition UNDEFINED to PRESENT_SRC; retain its window."""
function presentuntouched!(e::Emitter, win::VulkanWindow)
    image = win.images[win.current_image_idx + 1]
    imagebarrier!(e.cmd, image,
        VK.IMAGE_LAYOUT_UNDEFINED, VK.IMAGE_LAYOUT_PRESENT_SRC_KHR,
        VK.PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK.PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
        VK.AccessFlag(0), VK.AccessFlag(0))
    win.holdsframe[win.current_image_idx + 1] = false
    hold!(e, win)
    return nothing
end

"""submit_and_present!(bq, win, frame) -> token

Submit a sealed frame with image-available wait and per-image render-finished
signal, keep its token for the frame slot, then present. The caller records the
final image transition."""
function submit_and_present!(bq::SubmitChannel{<:VulkanQueue}, win::VulkanWindow,
                             frame::OneShot)
    win.acquired || error("present_frame!: no image acquired (call acquire_next_image! first)")
    fi = win.current_frame
    tok = submit!(bq, frame;
        # Wait for swapchain image availability before color attachment writes:
        waits = ((win.image_available[fi], UInt64(0),
                  VK.PipelineStageFlag2(VK.PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT)),),
        # Signal render_finished for the present; indexed by image, not frame
        # slot (see `create_swapchain!`).
        signals = ((win.render_finished[win.current_image_idx + 1], UInt64(0),
                    VK.PipelineStageFlag2(VK.PIPELINE_STAGE_2_ALL_COMMANDS_BIT)),))
    win.slot_tokens[fi] = tok
    handover!(bq, tok, frame; tag = :present)
    drain!(bq)
    present!(win)
    return tok
end

# Native Vulkan graphics capabilities.
Mantle.supports_geometry_stage(::LavaBackend) = true
Mantle.supports_tessellation(::LavaBackend) = true

# `VK_EXT_mesh_shader` is optional, so the backend's own device answers.
Mantle.supports_mesh_pipeline(b::LavaBackend) = vk_context(b).mesh_shader_available
