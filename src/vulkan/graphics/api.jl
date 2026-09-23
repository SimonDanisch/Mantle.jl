# High-level graphics API for Lava.jl
#
# GraphicsPipeline — the central type for configuring and dispatching
# graphics shaders. Supports lazy compilation, caching, and multiple
# draw dispatch via RenderTarget types.


# ── Lazy Compilation ──


# The graphics shader cache is a `VkContext` field; a reset makes a new one.

"""
Everything that changes the created `VkPipeline` but is not the shader pair.

Blend, cull, topology and depth mode are singletons, so `typeof(pipeline)` carries
them exactly. The stages are hashed as values, because each one now carries its
function, its configuration and its interface together — three things that used
to be keyed separately and could disagree about which pipeline they belonged to.
Leaving state out of the key made two pipelines that differ only in, say, blend
mode or depth mode share one compiled pipeline — whichever was compiled first
won, and the second draw silently rendered with the wrong state.
"""
pipeline_state_key(p::GraphicsPipeline) =
    (typeof(p), p.vertex, p.fragment, p.geometry, p.tess_control)

pipeline_state_key(p::MeshPipeline) = (typeof(p), p.mesh, p.fragment, p.object)

"""Return (vert_shader::LavaGfxShader, compiled::VulkanCompiledGraphicsPipeline)."""
function ensure_compiled_with_shader!(pipeline::GraphicsPipeline,
                              vert_fn, frag_fn, tt_vertex, tt_fragment;
                              ctx::VkContext,
                              color_format=VK.FORMAT_B8G8R8A8_SRGB,
                              depth_format=VK.FORMAT_UNDEFINED,
                              descriptor_set_layout=nothing)
    vert = get_or_compile_gfx(vert_fn, tt_vertex, :vertex; ctx)
    compiled = ensure_compiled!(pipeline, vert_fn, frag_fn, tt_vertex, tt_fragment;
        ctx, color_format, depth_format, descriptor_set_layout)
    return vert, compiled
end

function ensure_compiled!(pipeline::GraphicsPipeline, vert_fn, frag_fn, tt_vertex, tt_fragment;
                              color_format=VK.FORMAT_B8G8R8A8_SRGB,
                              depth_format=VK.FORMAT_UNDEFINED,
                              descriptor_set_layout=nothing,
                              ctx::VkContext)
    # Cache key includes type tuples — different arg types get different compiled
    # pipelines — and the pipeline state, which is the rest of what is baked in.
    cache_key = hash((vert_fn, frag_fn, tt_vertex, tt_fragment, color_format, depth_format,
                       pipeline_state_key(pipeline), descriptor_set_layout !== nothing))
    cached = get(ctx.caches.gfx_pipelines, cache_key, nothing)
    cached !== nothing && return cached::VulkanCompiledGraphicsPipeline

    # Compile vertex shader
    vert = get_or_compile_gfx(vert_fn, tt_vertex, :vertex; ctx)

    # Compile fragment shader
    frag = get_or_compile_gfx(frag_fn, tt_fragment, :fragment; ctx)

    # Optional stages
    geom_spirv = nothing
    geom_config = nothing
    if pipeline.geometry !== nothing
        # WRAPPED, like the other two stages: a geometry body takes
        # `(emitter, primitive, args...)` and emits through `emit!`, and the
        # wrapper is what reads the arrayed inputs into `primitive` and turns an
        # emit into "write the outputs, then `emit_vertex!`". Compiling the bare
        # function reaches it with the vertex stage's argument tuple and no
        # method — `lines_geometry` has two leading parameters that nothing
        # supplies.
        geom_cfg = Mantle.stageconfig(pipeline.geometry)
        geom = get_or_compile_gfx(geometrystage(pipeline, geom_cfg), tt_vertex,
                                  :geometry; config=geom_cfg, ctx)
        geom_spirv = geom.spirv_bytes
        geom_config = geom_cfg
    end

    tc_spirv = nothing
    te_spirv = nothing
    tess_cfg = nothing
    if pipeline.tess_control !== nothing
        tc_fn, tc_cfg = pipeline.tess_control
        tc = get_or_compile_gfx(tc_fn, tt_vertex, :tess_control; config=tc_cfg, ctx)
        tc_spirv = tc.spirv_bytes
        tess_cfg = tc_cfg
    end
    if pipeline.tess_eval !== nothing
        te = get_or_compile_gfx(pipeline.tess_eval, tt_vertex, :tess_eval; ctx,
            config=tess_cfg)
        te_spirv = te.spirv_bytes
    end

    compiled = create_graphics_pipeline(vert.spirv_bytes, frag.spirv_bytes;
        ctx, blend=pipeline.blend, cull=pipeline.cull,
        topology=pipeline.topology, depth=pipeline.depth,
        color_format=color_format, depth_format=depth_format,
        push_constant_size=max(vert.push_info.push_size, frag.push_info.push_size),
        geometry_spirv=geom_spirv,
        tess_ctrl_spirv=tc_spirv, tess_eval_spirv=te_spirv,
        tess_config=tess_cfg,
        descriptor_set_layout=descriptor_set_layout)

    ctx.caches.gfx_pipelines[cache_key] = compiled
    return compiled
end

"""
The mesh-pipeline half of [`ensure_compiled_with_shader!`](@ref).

Returns the MESH stage's shader as the first element, for the same reason the
classic one returns the vertex stage's: it is the stage the argument block is
laid out to unless the fragment stage takes arguments of its own.
"""
function ensure_compiled_with_shader!(pipeline::MeshPipeline,
                              mesh_fn, frag_fn, tt_mesh, tt_fragment;
                              ctx::VkContext,
                              color_format=VK.FORMAT_B8G8R8A8_SRGB,
                              depth_format=VK.FORMAT_UNDEFINED,
                              descriptor_set_layout=nothing)
    mesh = get_or_compile_gfx(mesh_fn, tt_mesh, :mesh; config = Mantle.meshconfig(pipeline), ctx)
    compiled = ensure_compiled!(pipeline, mesh_fn, frag_fn, tt_mesh, tt_fragment;
        ctx, color_format, depth_format, descriptor_set_layout)
    return mesh, compiled
end

function ensure_compiled!(pipeline::MeshPipeline, mesh_fn, frag_fn, tt_mesh, tt_fragment;
                              color_format=VK.FORMAT_B8G8R8A8_SRGB,
                              depth_format=VK.FORMAT_UNDEFINED,
                              descriptor_set_layout=nothing,
                              ctx::VkContext)
    pipeline.object === nothing || throw(ArgumentError(
        "ensure_compiled!: this pipeline has an object stage, and the Vulkan " *
        "backend dispatches its mesh threadgroups from the host. The task stage " *
        "is what an object stage lowers onto here, and nothing emits one yet."))
    cache_key = hash((mesh_fn, frag_fn, tt_mesh, tt_fragment, color_format, depth_format,
                       pipeline_state_key(pipeline), descriptor_set_layout !== nothing))
    cached = get(ctx.caches.gfx_pipelines, cache_key, nothing)
    cached !== nothing && return cached::VulkanCompiledGraphicsPipeline

    mesh = get_or_compile_gfx(mesh_fn, tt_mesh, :mesh; config = Mantle.meshconfig(pipeline), ctx)
    frag = get_or_compile_gfx(frag_fn, tt_fragment, :fragment; ctx)

    # No `topology`: a mesh stage has no input stream to assemble, and what it
    # EMITS is an execution mode of its own entry point rather than pipeline
    # state — `MeshConfig` carries it and the compile above reads it there.
    compiled = create_graphics_pipeline(nothing, frag.spirv_bytes;
        ctx, mesh_spirv=mesh.spirv_bytes,
        blend=pipeline.blend, cull=pipeline.cull, depth=pipeline.depth,
        color_format=color_format, depth_format=depth_format,
        push_constant_size=max(mesh.push_info.push_size, frag.push_info.push_size),
        descriptor_set_layout=descriptor_set_layout)

    ctx.caches.gfx_pipelines[cache_key] = compiled
    return compiled
end

function get_or_compile_gfx(@nospecialize(f), @nospecialize(tt), stage::Symbol;
                            config=nothing, ctx::VkContext)
    key = hash((f, tt, stage, config))
    cached = get(ctx.caches.gfx_shaders, key, nothing)
    cached !== nothing && return cached
    shader = lava_compile_gfx_shader(f, tt; stage, config)
    ctx.caches.gfx_shaders[key] = shader
    return shader
end

# ── Draw API ──

"""Convert args to device-side values (LavaArray → LavaDeviceArray, scalars pass through)."""
convert_args(args::Tuple) = map(arg -> arg isa LavaArray ? LavaDeviceArray(arg) : arg, args)
convert_args(::Tuple{}) = ()

"""
Resolve the vertex and fragment callables and their type tuples.

Always wrapped, and there is no second path: a `varyings` list beside numbered
`gfx_output`/`gfx_input` locations is two ways to declare one interface, and the
numbered one cannot express a pipeline with a geometry stage at all.
`Mantle.outputtype` answers it from the stage that actually feeds the
rasteriser, which is the geometry stage when there is one.

An empty output list is a real answer: a shadow pass writes only its clip
position, and that is a pipeline whose fragment stage reads nothing.

Returns (vert_fn, vert_tt, frag_fn, frag_tt).
"""
function resolve_shader_pair(pipeline, vert_tt::Type, frag_tt::Type)
    last = Mantle.lastgeometrystage(pipeline)
    vout = Mantle.outputtype(last)
    # Which names the declaration marked `Flat`. Both wrappers get the same
    # tuple, because Vulkan requires the producing and consuming stage to agree
    # on the interpolation of every location: a disagreement is a link failure
    # rather than a wrong picture, and an integer varying has no other legal
    # form. `outputtype` strips the marker — flatness says WHERE a value is
    # delivered, never what it is — so it has to travel beside the type.
    flats = Mantle.flatoutputs(last)
    wrapped_vert = VertexWrapper{typeof(Mantle.stagefunction(pipeline.vertex)), flats}()
    wrapped_frag = FragmentWrapper{typeof(Mantle.stagefunction(pipeline.fragment)), vout, flats}()
    return wrapped_vert, vert_tt, wrapped_frag, frag_tt
end

"""
The mesh and fragment callables of a [`MeshPipeline`](@ref), and their type
tuples.

Same shape as the classic pair so every caller — `compile_draw`, the graph's
`compiledraw`, `vertextouches` — stays one function. The mesh stage needs no
output type parameter: it writes its varyings by name and the numbering follows
declaration order, so only the CONSUMER has to be told what it is reading, and
`stageoutputs(p.mesh)` is what tells it.
"""
function resolve_shader_pair(pipeline::MeshPipeline, mesh_tt::Type, frag_tt::Type)
    vout = Mantle.outputtype(pipeline.mesh)
    flats = Mantle.flatoutputs(pipeline.mesh)
    wrapped_mesh = MeshWrapper{typeof(Mantle.stagefunction(pipeline.mesh))}()
    wrapped_frag = FragmentWrapper{typeof(Mantle.stagefunction(pipeline.fragment)), vout, flats}()
    return wrapped_mesh, mesh_tt, wrapped_frag, frag_tt
end

"""
The geometry stage, wrapped: the vertex stage's outputs are what the primitive
arrays, the geometry stage's own outputs are what an emit writes, and the input
topology says how many vertices a primitive has.

`Mantle.outputtype` twice over, because the two stages' declarations are
different things: `VIn` is what this stage READS per vertex and `Out` what it
WRITES per emit.
"""
function geometrystage(pipeline::GraphicsPipeline, cfg)
    gs = pipeline.geometry
    return GeometryWrapper{typeof(Mantle.stagefunction(gs)),
                           Mantle.outputtype(pipeline.vertex),
                           Mantle.outputtype(gs),
                           Mantle.flatoutputs(gs),
                           Mantle.primitivevertices(cfg.input_topology)}()
end

# `draw!(device, pipeline, target, count)` is CORE's — `src/graphics/record.jl`.
# Two methods stood here and did their own image transitions, their own
# `vkCmdBeginRendering` and their own bind through `vk_draw!`, sharing nothing
# with `begin_pass!`/`record_draw!`. That is why they never learned mesh
# pipelines and why their barrier logic had to be fixed a second time.

"""
    pack_gfx_args_bda(owner, args, push_info) -> UInt64

Pack a draw's arguments and return the buffer device address to push, rather than
an eight-byte `Vector` holding it.

The push constant is one BDA and always has been; wrapping it in a heap array
allocated 48 bytes of header plus the payload *per draw per frame*, which at a
hundred plots is most of a frame's garbage. `pack_gfx_args` still returns the
vector for callers written against it.
"""
function pack_gfx_args_bda(batch::O, args, push_info::PushConstantInfo) where {O<:Closed}
    (push_info.push_size == 0 || isempty(args)) && return UInt64(0)
    adaptor = LavaAdaptor(batch)
    converted = map(a -> Adapt.adapt(adaptor, a), args)
    byval_sizes = push_info.byval_llvm_sizes
    total_size = push_info.arg_buffer_size + compute_inline_extra_from_byval(byval_sizes)
    arg_buf = get_arg_buffer(batch, total_size)
    pack_args_direct!(batch, arg_buf.mapped_ptr, arg_buf.address, push_info.arg_offsets,
                      push_info.arg_buffer_size, byval_sizes, converted)
    return arg_buf.address
end

function pack_gfx_args(batch::O, args, push_info::PushConstantInfo) where {O<:Closed}
    push_info.push_size == 0 && return UInt8[]
    isempty(args) && return UInt8[]

    # LavaAdaptor is the single point that strips LavaArray → LavaDeviceArray
    # AND pins the original into the owner's pinned set.  Adapt.jl's recursion
    # handles wrapper structs / Broadcasted / NamedTuple, so any nested
    # LavaArray gets both its pointer strip and its pin in the same pass.
    adaptor = LavaAdaptor(batch)
    converted = map(a -> Adapt.adapt(adaptor, a), args)

    # Use the same arg buffer packing as compute: inline byval structs into
    # the arg buffer with self-referencing BDA pointers.
    offsets = push_info.arg_offsets      # precomputed: this ran per draw per frame
    byval_sizes = push_info.byval_llvm_sizes

    inline_extra = compute_inline_extra_from_byval(byval_sizes)
    total_size = push_info.arg_buffer_size + inline_extra

    arg_buf = get_arg_buffer(batch, total_size)

    pack_args_direct!(batch, arg_buf.mapped_ptr, arg_buf.address, offsets,
                       push_info.arg_buffer_size, byval_sizes, converted)

    # Push constant = BDA of arg buffer
    push_data = Vector{UInt8}(undef, 8)
    GC.@preserve push_data begin
        unsafe_store!(Ptr{UInt64}(pointer(push_data)), arg_buf.address)
    end
    return push_data
end

# Empty-args variant
function pack_gfx_args(::Closed, args, ::Nothing=nothing)
    isempty(args) && return UInt8[]
    error("pack_gfx_args requires push_info for non-empty args.")
end

"""
    presentready!(e, win)

Transition the acquired swapchain image to `PRESENT_SRC`, as the last command
of the frame's one-shot. The frame draws into the image in
`COLOR_ATTACHMENT_OPTIMAL`; the presentation engine wants it in this one.
"""
function presentready!(e::Emitter, win::VulkanWindow)
    image = win.images[win.current_image_idx + 1]
    transition_image!(e.cmd, image,
        VK.IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK.IMAGE_LAYOUT_PRESENT_SRC_KHR,
        VK.PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK.PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
        VK.ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK.AccessFlag(0))
    # The swapchain and its images are named by this frame until it has passed.
    hold!(e, win)
    return nothing
end

"""
    presentuntouched!(e, win)

The whole of an abandoned frame's one-shot: the acquired image to `PRESENT_SRC`
from `UNDEFINED`, its contents discarded, so it can be presented without having
been drawn into. From `UNDEFINED` and not from the layout the last frame left it
in, because an acquired image's contents are undefined by the specification and
a transition from `UNDEFINED` is valid whatever the presentation engine did.
"""
function presentuntouched!(e::Emitter, win::VulkanWindow)
    image = win.images[win.current_image_idx + 1]
    transition_image!(e.cmd, image,
        VK.IMAGE_LAYOUT_UNDEFINED, VK.IMAGE_LAYOUT_PRESENT_SRC_KHR,
        VK.PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK.PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
        VK.AccessFlag(0), VK.AccessFlag(0))
    hold!(e, win)
    return nothing
end

"""
    submit_and_present!(bq, win, frame) -> token

Submit the frame's one-shot and present.

Internal to this backend: the portable verb is `present_frame!(device, window)`,
which builds the one-shot and calls this. A caller outside the backend has no
one-shot to pass, and which channel a frame submits on is Mantle's to decide.

One submission, through the same `submit!` everything else goes through, with
the two extra semaphores a swapchain image needs — wait for the image to be
available before colour attachment writes, signal render-finished for the
present — and the frame slot's fence, which `acquire_next_image!` waits on
before it reuses the slot. The one-shot must end with [`presentready!`](@ref).

One route: nothing takes an open batch over and submits it by a second one,
with a list of sealed segments and bookkeeping of which of them a recording
lent it.
"""
function submit_and_present!(bq::SubmitChannel{<:VulkanQueue}, win::VulkanWindow,
                             frame::OneShot)
    win.acquired || error("present_frame!: no image acquired (call acquire_next_image! first)")
    fi = win.current_frame
    tok = submit!(bq, frame;
        # Wait for swapchain image availability before color attachment writes:
        waits = ((win.image_available[fi], UInt64(0),
                  VK.PipelineStageFlag2(VK.PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT)),),
        # Signal render_finished for the subsequent present operation. Indexed
        # by IMAGE, not frame slot — see the comment where these are created.
        signals = ((win.render_finished[win.current_image_idx + 1], UInt64(0),
                    VK.PipelineStageFlag2(VK.PIPELINE_STAGE_2_ALL_COMMANDS_BIT)),),
        fence = win.in_flight[fi])
    handover!(bq, tok, frame; tag = :present)
    drain!(bq)
    present!(win)
    return tok
end

# ── Mantle's shader builtins, on this backend ────────────────────────────────
#
# The counterpart to the Metal backend's block, and the reason both exist: a
# shader names `Mantle.vertex_index()` and the compiler that is running decides
# what that means. `@lava_device_override` is Lava's wrapper around the same
# method-table overlay `@device_override` is on the other side.
#
# forgotten here is a `MethodError` naming it rather than a shader that reads
# the wrong thing.
#

# Both stages exist here. Declared in `graphics/commands.jl` with a `false`
# default, so this is the opt-in and Metal's silence is its answer.
Mantle.supports_geometry_stage(::LavaBackend) = true
Mantle.supports_tessellation(::LavaBackend) = true

# Unlike the two above, this one is a property of the DEVICE rather than of the
# backend: `VK_EXT_mesh_shader` is an extension, and the geometry chain is still
# the only front end on hardware without it. Answered from the live context's
# probe, the way `supportspredicate` answers for conditional rendering.
Mantle.supports_mesh_pipeline(::LavaBackend) = vk_context().mesh_shader_available
