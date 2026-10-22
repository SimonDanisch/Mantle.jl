# Graphics pipelines over VK_KHR_dynamic_rendering (no VkRenderPass). Blend, cull,
# depth and topology lower by dispatch on their config types. The pipeline cache
# is a `VkContext` field, so a reset makes a new one.

"""
    create_graphics_pipeline(vertex_spirv, fragment_spirv; ctx, mesh_spirv=nothing,
        blend=Opaque(), cull=CullBack(), topology=TriangleList(), depth=DepthLess(),
        color_format=VK.FORMAT_B8G8R8A8_SRGB, depth_format=VK.FORMAT_UNDEFINED,
        push_constant_size=8, geometry_spirv=nothing, tess_ctrl_spirv=nothing,
        tess_eval_spirv=nothing, tess_config=nothing,
        descriptor_set_layout=nothing) -> VulkanCompiledGraphicsPipeline

A dynamic-rendering graphics pipeline from SPIR-V. Exactly one of `vertex_spirv`
and `mesh_spirv` is the front end; a mesh pipeline needs `VK_EXT_mesh_shader` and
has no geometry or tessellation stage.
"""
function create_graphics_pipeline(vertex_spirv::Union{Vector{UInt8}, Nothing},
                                    fragment_spirv::Vector{UInt8};
                                    ctx::VkContext,
                                    mesh_spirv::Union{Nothing, Vector{UInt8}}=nothing,
                                    blend::BlendMode=Opaque(),
                                    cull::CullFace=CullBack(),
                                    topology::Topology=TriangleList(),
                                    depth::DepthMode=DepthLess(),
                                    color_format=VK.FORMAT_B8G8R8A8_SRGB,
                                    depth_format::VK.Format=VK.FORMAT_UNDEFINED,
                                    push_constant_size::Integer=8,
                                    geometry_spirv::Union{Nothing, Vector{UInt8}}=nothing,
                                    tess_ctrl_spirv::Union{Nothing, Vector{UInt8}}=nothing,
                                    tess_eval_spirv::Union{Nothing, Vector{UInt8}}=nothing,
                                    tess_config::Union{Nothing, TessConfig}=nothing,
                                    descriptor_set_layout::Union{Nothing, VK.DescriptorSetLayout}=nothing)
    dev = ctx.device

    # A mesh stage replaces the whole vertex/geometry/tessellation chain.
    (vertex_spirv === nothing) == (mesh_spirv === nothing) && throw(ArgumentError(
        "a graphics pipeline needs exactly one front end: pass `vertex_spirv` " *
        "for the classic chain, or `mesh_spirv` for a mesh pipeline. " *
        "Got $(vertex_spirv === nothing ? "neither" : "both")."))
    if mesh_spirv !== nothing
        ctx.mesh_shader_available || throw(ArgumentError(
            "this device has no VK_EXT_mesh_shader, so a mesh pipeline cannot " *
            "be created on it. Ask `supports_mesh_pipeline(backend)` first: the " *
            "classic vertex chain is still there on hardware without it."))
        geometry_spirv === nothing && tess_ctrl_spirv === nothing || throw(ArgumentError(
            "a mesh pipeline has no geometry or tessellation stage: the mesh " *
            "stage is what replaces them."))
    end

    (tess_ctrl_spirv === nothing) == (tess_eval_spirv === nothing) || throw(ArgumentError(
        "tessellation needs both a control and an evaluation shader"))
    specs = [mesh_spirv === nothing ? VK.SHADER_STAGE_VERTEX_BIT => vertex_spirv :
                                      VK.SHADER_STAGE_MESH_BIT_EXT => mesh_spirv,
             VK.SHADER_STAGE_FRAGMENT_BIT => fragment_spirv]
    geometry_spirv === nothing || push!(specs, VK.SHADER_STAGE_GEOMETRY_BIT => geometry_spirv)
    tess_ctrl_spirv === nothing || push!(specs,
        VK.SHADER_STAGE_TESSELLATION_CONTROL_BIT => tess_ctrl_spirv,
        VK.SHADER_STAGE_TESSELLATION_EVALUATION_BIT => tess_eval_spirv)
    modules = [create_shader_module(dev, spirv) for (_, spirv) in specs]
    stages = [VK.PipelineShaderStageCreateInfo(st, m, "main") for ((st, _), m) in zip(specs, modules)]
    all_stage_flags = reduce(|, first.(specs))

    # Empty vertex input (vertices are pulled by device address), and none for a
    # mesh pipeline, whose topology is the shader's own `OutputTrianglesEXT`.
    vertex_input = mesh_spirv === nothing ?
        VK.PipelineVertexInputStateCreateInfo([], []) : C_NULL
    input_assembly = mesh_spirv === nothing ?
        VK.PipelineInputAssemblyStateCreateInfo(vk_topology(topology), false) : C_NULL

    tess_state = C_NULL
    if tess_config !== nothing
        tess_state = VK.PipelineTessellationStateCreateInfo(UInt32(tess_config.patch_vertices))
    end

    # Viewport and scissor are dynamic state; these placeholders are overridden.
    dummy_viewport = VK.Viewport(0.0f0, 0.0f0, 1.0f0, 1.0f0, 0.0f0, 1.0f0)
    dummy_scissor = VK.Rect2D(VK.Offset2D(0, 0), VK.Extent2D(1, 1))
    viewport_state = VK.PipelineViewportStateCreateInfo(;
        viewports=[dummy_viewport], scissors=[dummy_scissor])

    cull_mode = vk_cull(cull)
    rasterization = VK.PipelineRasterizationStateCreateInfo(
        false, false,
        VK.POLYGON_MODE_FILL,
        VK.FRONT_FACE_COUNTER_CLOCKWISE,
        false, 0.0f0, 0.0f0, 0.0f0,
        1.0f0;  # line width
        cull_mode=cull_mode,
    )

    # No MSAA.
    multisample = VK.PipelineMultisampleStateCreateInfo(
        VK.SAMPLE_COUNT_1_BIT, false, 1.0f0, false, false)

    # Whether the pipeline has a depth attachment follows the render target, not
    # the depth mode: this format must equal the view bound at `vkCmdBeginRendering`,
    # UNDEFINED for none (VUID-vkCmdDraw-pDepthAttachment-06181).
    has_depth = depth_format != VK.FORMAT_UNDEFINED
    depth_enable, depth_compare = vk_depth(depth)
    keep = VK.StencilOpState(VK.STENCIL_OP_KEEP, VK.STENCIL_OP_KEEP, VK.STENCIL_OP_KEEP,
                             VK.COMPARE_OP_ALWAYS, 0, 0, 0)
    # No depth-bounds or stencil test.
    depth_stencil = VK.PipelineDepthStencilStateCreateInfo(
        depth_enable, depth_enable && depthwrites(depth), depth_compare,
        false, false, keep, keep, 0.0f0, 1.0f0)

    # Integer attachments are unblended: IDs must remain exact even when the
    # colour target uses alpha blending. Device creation enables independentBlend.
    color_formats = colorformats(color_format)
    color_blend = VK.PipelineColorBlendStateCreateInfo(
        false, VK.LOGIC_OP_COPY,
        [vk_blend(attachmentblend(blend, i, f in (VK.FORMAT_R32_UINT, VK.FORMAT_R32G32_UINT)))
         for (i, f) in enumerate(color_formats)],
        (0.0f0, 0.0f0, 0.0f0, 0.0f0),
    )

    dynamic_states = [VK.DYNAMIC_STATE_VIEWPORT, VK.DYNAMIC_STATE_SCISSOR]
    dynamic_state = VK.PipelineDynamicStateCreateInfo(dynamic_states)

    push_ranges = push_constant_size > 0 ?
        [VK.PushConstantRange(all_stage_flags, UInt32(0), UInt32(push_constant_size))] :
        VK.PushConstantRange[]
    ds_layouts = descriptor_set_layout === nothing ? VK.DescriptorSetLayout[] : [descriptor_set_layout]
    layout = VK.PipelineLayout(dev, ds_layouts, push_ranges)

    rendering_info = VK.PipelineRenderingCreateInfo(
        UInt32(0),           # view mask
        color_formats,       # color attachment formats
        depth_format,
        VK.FORMAT_UNDEFINED,  # stencil format
    )

    pipeline = createpipeline(ctx, VK.PipelineCreateFlag(0), "graphics", fragment_spirv) do flags
        ci = VK.GraphicsPipelineCreateInfo(
            stages,
            rasterization,
            layout, UInt32(0), Int32(-1);
            flags,
            vertex_input_state=vertex_input,
            input_assembly_state=input_assembly,
            tessellation_state=tess_state,
            viewport_state=viewport_state,
            multisample_state=multisample,
            depth_stencil_state=depth_stencil,
            color_blend_state=color_blend,
            dynamic_state=dynamic_state,
            next=rendering_info,
        )
        pipelines, code = @vk_checked ctx "vkCreateGraphicsPipelines" VK.create_graphics_pipelines(
            dev, [ci]; pipeline_cache=ctx.pipeline_cache)
        pipelines[1], code
    end

    return VulkanCompiledGraphicsPipeline(
        pipeline, layout, modules,
        UInt32(push_constant_size),
        descriptor_set_layout,
        all_stage_flags,
        color_formats, has_depth,
        mesh_spirv === nothing ? :vertex : :mesh,
    )
end

"""
    ismeshpipeline(p) -> Bool

Whether `p`'s front end is a mesh stage, so a draw against it counts workgroups
and issues `vkCmdDrawMeshTasksEXT`.
"""
ismeshpipeline(p::VulkanCompiledGraphicsPipeline) = p.front === :mesh

"""
One colour attachment or several, normalised. A single format is the common case
and stays spellable as itself rather than as a one-element vector.
"""
colorformats(f::VK.Format) = [f]
colorformats(f::AbstractVector{VK.Format}) = collect(f)

# Julia element types (`RGBA{Float32}`, `BGRA{N0f8}`), lowered by `vkformat` as
# for transient images, so every `color_format=` here accepts both spellings.
colorformats(T::Type) = [vkformat(VulkanAPI(), T)]
colorformats(ts::AbstractVector{<:Type}) = [vkformat(VulkanAPI(), T) for T in ts]

vk_topology(::TriangleList)  = VK.PRIMITIVE_TOPOLOGY_TRIANGLE_LIST
vk_topology(::TriangleStrip) = VK.PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP
vk_topology(::LineList)      = VK.PRIMITIVE_TOPOLOGY_LINE_LIST
vk_topology(::LineStrip)     = VK.PRIMITIVE_TOPOLOGY_LINE_STRIP
vk_topology(::PointList)     = VK.PRIMITIVE_TOPOLOGY_POINT_LIST
vk_topology(::PatchList)              = VK.PRIMITIVE_TOPOLOGY_PATCH_LIST
vk_topology(::LineListAdjacency)      = VK.PRIMITIVE_TOPOLOGY_LINE_LIST_WITH_ADJACENCY
vk_topology(::LineStripAdjacency)     = VK.PRIMITIVE_TOPOLOGY_LINE_STRIP_WITH_ADJACENCY

vk_cull(::NoCull)    = VK.CULL_MODE_NONE
vk_cull(::CullBack)  = VK.CULL_MODE_BACK_BIT
vk_cull(::CullFront) = VK.CULL_MODE_FRONT_BIT

# (enabled, compare op) of a depth mode.
function vk_depth(mode::DepthMode)
    test = depthtest(mode)
    ops = (less = VK.COMPARE_OP_LESS, lessequal = VK.COMPARE_OP_LESS_OR_EQUAL,
           greater = VK.COMPARE_OP_GREATER, always = VK.COMPARE_OP_ALWAYS)
    return test !== nothing, getproperty(ops, something(test, :always))
end

function vk_blend(mode::BlendMode)
    mask = writescolor(mode) ? VK.COLOR_COMPONENT_R_BIT | VK.COLOR_COMPONENT_G_BIT |
                               VK.COLOR_COMPONENT_B_BIT | VK.COLOR_COMPONENT_A_BIT :
                               VK.ColorComponentFlag(0)
    f = blendfactors(mode)
    vk = (one = VK.BLEND_FACTOR_ONE, srcalpha = VK.BLEND_FACTOR_SRC_ALPHA,
          oneminussrcalpha = VK.BLEND_FACTOR_ONE_MINUS_SRC_ALPHA)
    src, dst, srca, dsta = f === nothing ? (VK.BLEND_FACTOR_ONE, VK.BLEND_FACTOR_ZERO,
                                            VK.BLEND_FACTOR_ONE, VK.BLEND_FACTOR_ZERO) :
                           map(s -> getproperty(vk, s), f)
    return VK.PipelineColorBlendAttachmentState(f !== nothing, src, dst, VK.BLEND_OP_ADD,
                                                srca, dsta, VK.BLEND_OP_ADD, mask)
end

"""Record a layout transition of `image`'s mip `levels` (0-based) of its first layer."""
function imagebarrier!(cmd, image::VK.Image, old_layout, new_layout, src_stage, dst_stage,
                       src_access, dst_access; aspect = VK.IMAGE_ASPECT_COLOR_BIT, levels = 0:0)
    barrier = VK.ImageMemoryBarrier(src_access, dst_access, old_layout, new_layout,
        VK.QUEUE_FAMILY_IGNORED, VK.QUEUE_FAMILY_IGNORED, image,
        VK.ImageSubresourceRange(aspect, UInt32(first(levels)), UInt32(length(levels)), UInt32(0), UInt32(1)))
    VK.cmd_pipeline_barrier(cmd, [], [], [barrier]; src_stage_mask = src_stage, dst_stage_mask = dst_stage)
end

"""One value for every attachment (anything but a vector), or a vector with one
value each. Any other length is an error."""
function perattachment(x, n::Integer, what::String)
    x isa AbstractVector || return [x for _ in 1:n]
    length(x) == n && return x
    error("$what has $(length(x)) entries but the pass has $n colour attachments")
end

"""
    begin_pass!(e, color_views, color_images, extent; clear_color, load_op,
                depth_view, depth_image, depth_clear=1f0, depth_load_op, transition=true)

Begin a dynamic rendering pass; record draws with `draw_in_pass!` and finish with
`end_pass!`. One colour attachment or vectors of several (fragment location `i`
writes attachment `i`), with `clear_color` and `load_op` once or once each. A
clear value means CLEAR, `nothing` means LOAD, and only an explicit load op
reaches DONT_CARE; `depth_clear` and `depth_load_op` work the same way.
"""
function begin_pass!(e::Emitter,
                         color_views::AbstractVector{<:VK.ImageView},
                         color_images::AbstractVector{<:VK.Image},
                         extent::VK.Extent2D;
                         clear_color=(0f0, 0f0, 0f0, 1f0),
                         depth_view::Union{Nothing, VK.ImageView}=nothing,
                         depth_image::Union{Nothing, VK.Image}=nothing,
                         depth_clear::Union{Nothing, Float32}=1.0f0,
                         depth_load_op::Union{Nothing, VK.AttachmentLoadOp}=nothing,
                         transition::Bool=true,
                         load_op=nothing)
    cmd = e.cmd
    n = length(color_views)
    n == length(color_images) ||
        error("$n colour views but $(length(color_images)) colour images")
    clears = perattachment(clear_color, n, "clear_color")
    ops = perattachment(load_op, n, "load_op")

    color_attachments = map(1:n) do i
        clear, given = clears[i], ops[i]
        # Only an explicit `load_op` reaches DONT_CARE; inference picks CLEAR or LOAD.
        op = given !== nothing ? given :
             clear !== nothing ? VK.ATTACHMENT_LOAD_OP_CLEAR :
                                 VK.ATTACHMENT_LOAD_OP_LOAD

        # `transition=false` leaves the layout transition to a caller that derives
        # its own barriers; the transitions below would double up with them.
        if !transition
            clear_val = clear !== nothing ?
                VK.ClearValue(VK.ClearColorValue(clear)) :
                VK.ClearValue(VK.ClearColorValue((0f0, 0f0, 0f0, 0f0)))
        elseif clear !== nothing
            # Clear: the old contents are discarded.
            imagebarrier!(cmd, color_images[i],
                VK.IMAGE_LAYOUT_UNDEFINED, VK.IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                # srcStage COLOR_ATTACHMENT_OUTPUT, not TOP_OF_PIPE: the submit waits
                # on the acquire semaphore at that stage, and a TOP_OF_PIPE barrier
                # does not depend on the wait, so a swapchain image would be written
                # while the presentation engine still owns it (bands of stale pixels).
                VK.PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                VK.PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                VK.ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK.ACCESS_COLOR_ATTACHMENT_WRITE_BIT)

            clear_val = VK.ClearValue(VK.ClearColorValue(clear))
        else
            # Load: keep the previous contents.
            imagebarrier!(cmd, color_images[i],
                VK.IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK.IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                VK.PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK.PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                VK.ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK.ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK.ACCESS_COLOR_ATTACHMENT_READ_BIT)

            clear_val = VK.ClearValue(VK.ClearColorValue((0f0, 0f0, 0f0, 0f0)))
        end

        VK.RenderingAttachmentInfo(
            VK.IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            VK.IMAGE_LAYOUT_UNDEFINED,
            op,
            VK.ATTACHMENT_STORE_OP_STORE,
            clear_val;
            image_view=color_views[i],
            resolve_mode=VK.RESOLVE_MODE_NONE,
        )
    end

    # Typed for depth-only passes: `map` over an empty range may not infer it.
    color_attachments = convert(Vector{VK.RenderingAttachmentInfo}, color_attachments)

    depth_attachment = C_NULL
    if depth_view !== nothing
        dop = depth_load_op !== nothing ? depth_load_op :
              depth_clear !== nothing ? VK.ATTACHMENT_LOAD_OP_CLEAR :
                                        VK.ATTACHMENT_LOAD_OP_LOAD
        # As for colour: a discarding op transitions from UNDEFINED, a LOAD from the
        # layout the previous pass left. `transition = false` leaves both to the caller.
        if transition && depth_image !== nothing
            old = dop === VK.ATTACHMENT_LOAD_OP_LOAD ?
                  VK.IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL :
                  VK.IMAGE_LAYOUT_UNDEFINED
            # Ordered after the previous depth write either way: the transition is
            # a write itself, so a discarding one is a hazard too.
            imagebarrier!(cmd, depth_image, old, VK.IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                VK.PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT, VK.PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
                VK.ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                VK.ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK.ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
                aspect = VK.IMAGE_ASPECT_DEPTH_BIT)
        end
        depth_clear_val = VK.ClearValue(VK.ClearDepthStencilValue(
            depth_clear === nothing ? 1.0f0 : depth_clear, UInt32(0)))
        depth_attachment = VK.RenderingAttachmentInfo(
            VK.IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            VK.IMAGE_LAYOUT_UNDEFINED,
            dop,
            VK.ATTACHMENT_STORE_OP_STORE,
            depth_clear_val;
            image_view=depth_view,
            resolve_mode=VK.RESOLVE_MODE_NONE,
        )
    end

    render_area = VK.Rect2D(VK.Offset2D(0, 0), extent)
    rendering_info = VK.RenderingInfo(
        render_area,
        UInt32(1), UInt32(0),
        color_attachments;
        depth_attachment=depth_attachment,
    )
    VK.cmd_begin_rendering(cmd, rendering_info)
end

"""
Bind the pipeline and push the argument address. A graph gives every draw an
argument-memory offset, including shaders whose ABI has no arguments, whose layout
has no push-constant range at all.
"""
function bind_for_draw!(e::Emitter, pipeline::VulkanCompiledGraphicsPipeline, push_bda::UInt64)
    VK.cmd_bind_pipeline(e.cmd, VK.PIPELINE_BIND_POINT_GRAPHICS, pipeline.pipeline)
    if pipeline.push_constant_size > 0 && push_bda != UInt64(0)
        bda = Ref(push_bda)
        GC.@preserve bda VK.cmd_push_constants(e.cmd, pipeline.pipeline_layout,
            pipeline.push_stage_flags, UInt32(0), UInt32(8),
            Ptr{Nothing}(Base.unsafe_convert(Ptr{UInt64}, bda)))
    end
    # The command buffer names the pipeline until its submission passes.
    hold!(e, pipeline)
    return nothing
end

"""
    draw_in_pass!(e, pipeline, vertex_count; push_bda, instances = 1)

Record a draw inside an active rendering pass; `push_bda` is the argument buffer's
device address.
"""
function draw_in_pass!(e::Emitter, pipeline::VulkanCompiledGraphicsPipeline, vertex_count::Integer;
                       push_bda::UInt64 = UInt64(0), instances::Integer = 1)
    bind_for_draw!(e, pipeline, push_bda)
    VK.cmd_draw(e.cmd, UInt32(vertex_count), UInt32(instances), UInt32(0), UInt32(0))
end

"""
    draw_mesh_in_pass!(e, pipeline, groups; push_bda)

Record a mesh draw inside an active rendering pass. `groups` is a workgroup count;
the mesh stage's `set_mesh_outputs!` says how much geometry each group emits.
"""
function draw_mesh_in_pass!(e::Emitter, pipeline::VulkanCompiledGraphicsPipeline, groups::Integer;
                            push_bda::UInt64 = UInt64(0))
    bind_for_draw!(e, pipeline, push_bda)
    VK.cmd_draw_mesh_tasks_ext(e.cmd, UInt32(groups), UInt32(1), UInt32(1))
end

"""
    draw_indirect_in_pass!(e, pipeline, commands; first = 1, count = 1, push_bda)

Record `count` draws whose parameters are in device memory, from `commands` (a
`LavaArray{DrawIndirectCommand}`) starting at the 1-based `first`, inside an
active rendering pass. The host never reads them, so a compute pass of the same
frame can write them.
"""
function draw_indirect_in_pass!(e::Emitter, pipeline::VulkanCompiledGraphicsPipeline,
                                commands::LavaArray{DrawIndirectCommand,1};
                                first::Integer = 1, count::Integer = 1, push_bda::UInt64 = UInt64(0))
    bind_for_draw!(e, pipeline, push_bda)
    managed = commands.buf[]
    offset = pool_offset(managed) + commands.offset + (first - 1) * sizeof(DrawIndirectCommand)
    VK.cmd_draw_indirect(e.cmd, managed.buffer, UInt64(offset), UInt32(count),
                         UInt32(sizeof(DrawIndirectCommand)))
    # The device reads the commands, so the array must outlive the draw.
    hold!(e, commands)
end

"""
    draw_indexed_in_pass!(e, pipeline, index_count; push_bda, indices_buffer,
                          indices_offset, instances)

Record an indexed draw with `UInt32` indices at byte `indices_offset` of
`indices_buffer` (a `LavaArray` may be a slice of its buffer), inside an active
rendering pass. Holds only the pipeline: the caller keeps the index buffer alive.
"""
function draw_indexed_in_pass!(e::Emitter, pipeline::VulkanCompiledGraphicsPipeline,
                               index_count::Integer; push_bda::UInt64 = UInt64(0),
                               indices_buffer::VK.Buffer, indices_offset::UInt64 = UInt64(0),
                               instances::Integer = 1)
    bind_for_draw!(e, pipeline, push_bda)
    VK.cmd_bind_index_buffer(e.cmd, indices_buffer, indices_offset, VK.INDEX_TYPE_UINT32)
    VK.cmd_draw_indexed(e.cmd, UInt32(index_count), UInt32(instances), UInt32(0), Int32(0), UInt32(0))
end

"""
    end_pass!(emitter)

End the dynamic rendering pass the emitter opened.
"""
end_pass!(e::Emitter) = VK.cmd_end_rendering(e.cmd)

"""
    set_viewport!(e, x, y, w, h)

The portable viewport verb (`graphics/commands.jl`): sets the viewport and the
scissor it implies ([`scissorrect`](@ref)).
"""
function set_viewport!(e::Emitter, x::Real, y::Real, w::Real, h::Real)
    VK.cmd_set_viewport(e.cmd, [VK.Viewport(Float32(x), Float32(y), Float32(w), Float32(h), 0f0, 1f0)])
    sx, sy, sw, sh = scissorrect(x, y, w, h)
    VK.cmd_set_scissor(e.cmd, [VK.Rect2D(VK.Offset2D(sx, sy), VK.Extent2D(UInt32(sw), UInt32(sh)))])
    return nothing
end

"""Select a prepared descriptor set for the next draw. See `use_bindings!`."""
function use_bindings!(e::Emitter, compiled, bindings)
    VK.cmd_bind_descriptor_sets(e.cmd, VK.PIPELINE_BIND_POINT_GRAPHICS,
                                compiled.pipeline_layout, UInt32(0),
                                [bindings.set], UInt32[])
    # The set, its textures and samplers are named by every following draw.
    hold!(e, bindings)
    return nothing
end

