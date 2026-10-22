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

    frag_mod = create_shader_module(dev, fragment_spirv)
    front_mod = create_shader_module(dev, mesh_spirv === nothing ? vertex_spirv : mesh_spirv)
    front_bit = mesh_spirv === nothing ? VK.SHADER_STAGE_VERTEX_BIT : VK.SHADER_STAGE_MESH_BIT_EXT
    modules = VK.ShaderModule[front_mod, frag_mod]

    stages = VK.PipelineShaderStageCreateInfo[
        VK.PipelineShaderStageCreateInfo(front_bit, front_mod, "main"),
        VK.PipelineShaderStageCreateInfo(
            VK.SHADER_STAGE_FRAGMENT_BIT, frag_mod, "main"),
    ]

    if geometry_spirv !== nothing
        geom_mod = create_shader_module(dev, geometry_spirv)
        push!(modules, geom_mod)
        push!(stages, VK.PipelineShaderStageCreateInfo(
            VK.SHADER_STAGE_GEOMETRY_BIT, geom_mod, "main"))
    end

    if tess_ctrl_spirv !== nothing && tess_eval_spirv !== nothing
        tc_mod = create_shader_module(dev, tess_ctrl_spirv)
        te_mod = create_shader_module(dev, tess_eval_spirv)
        push!(modules, tc_mod, te_mod)
        push!(stages, VK.PipelineShaderStageCreateInfo(
            VK.SHADER_STAGE_TESSELLATION_CONTROL_BIT, tc_mod, "main"))
        push!(stages, VK.PipelineShaderStageCreateInfo(
            VK.SHADER_STAGE_TESSELLATION_EVALUATION_BIT, te_mod, "main"))
    end

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
    checkdepthtarget(depth, has_depth)
    depth_enable, depth_compare = vk_depth(depth)
    depth_stencil = VK.PipelineDepthStencilStateCreateInfo(
        depth_enable, depth_enable && vk_depth_write(depth),
        depth_compare,
        false,    # depth bounds test
        false,    # stencil test
        VK.StencilOpState(VK.STENCIL_OP_KEEP, VK.STENCIL_OP_KEEP,
            VK.STENCIL_OP_KEEP, VK.COMPARE_OP_ALWAYS, 0, 0, 0),
        VK.StencilOpState(VK.STENCIL_OP_KEEP, VK.STENCIL_OP_KEEP,
            VK.STENCIL_OP_KEEP, VK.COMPARE_OP_ALWAYS, 0, 0, 0),
        0.0f0, 1.0f0,
    )

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

    all_stage_flags = front_bit | VK.SHADER_STAGE_FRAGMENT_BIT
    if geometry_spirv !== nothing
        all_stage_flags |= VK.SHADER_STAGE_GEOMETRY_BIT
    end
    if tess_ctrl_spirv !== nothing
        all_stage_flags |= VK.SHADER_STAGE_TESSELLATION_CONTROL_BIT |
                            VK.SHADER_STAGE_TESSELLATION_EVALUATION_BIT
    end

    push_ranges = VK.PushConstantRange[]
    if push_constant_size > 0
        push!(push_ranges, VK.PushConstantRange(
            all_stage_flags, UInt32(0), UInt32(push_constant_size)))
    end

    ds_layouts = VK.DescriptorSetLayout[]
    if descriptor_set_layout !== nothing
        push!(ds_layouts, descriptor_set_layout)
    end

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

function vk_depth(::DepthLess)
    (true, VK.COMPARE_OP_LESS)
end
function vk_depth(::DepthLessEq)
    (true, VK.COMPARE_OP_LESS_OR_EQUAL)
end
function vk_depth(::DepthGreater)
    (true, VK.COMPARE_OP_GREATER)
end
function vk_depth(::DepthAlways)
    (true, VK.COMPARE_OP_ALWAYS)
end
function vk_depth(::DepthLessEqReadOnly)
    (true, VK.COMPARE_OP_LESS_OR_EQUAL)
end
function vk_depth(::DepthOff)
    (false, VK.COMPARE_OP_ALWAYS)
end

vk_depth_write(::DepthMode) = true
vk_depth_write(::DepthLessEqReadOnly) = false

function vk_blend(::Opaque)
    VK.PipelineColorBlendAttachmentState(
        false,
        VK.BLEND_FACTOR_ONE, VK.BLEND_FACTOR_ZERO, VK.BLEND_OP_ADD,
        VK.BLEND_FACTOR_ONE, VK.BLEND_FACTOR_ZERO, VK.BLEND_OP_ADD,
        VK.COLOR_COMPONENT_R_BIT | VK.COLOR_COMPONENT_G_BIT |
        VK.COLOR_COMPONENT_B_BIT | VK.COLOR_COMPONENT_A_BIT,
    )
end

function vk_blend(::AlphaBlend)
    VK.PipelineColorBlendAttachmentState(
        true,
        VK.BLEND_FACTOR_SRC_ALPHA, VK.BLEND_FACTOR_ONE_MINUS_SRC_ALPHA, VK.BLEND_OP_ADD,
        VK.BLEND_FACTOR_ONE, VK.BLEND_FACTOR_ZERO, VK.BLEND_OP_ADD,
        VK.COLOR_COMPONENT_R_BIT | VK.COLOR_COMPONENT_G_BIT |
        VK.COLOR_COMPONENT_B_BIT | VK.COLOR_COMPONENT_A_BIT,
    )
end

function vk_blend(::Additive)
    VK.PipelineColorBlendAttachmentState(
        true,
        VK.BLEND_FACTOR_ONE, VK.BLEND_FACTOR_ONE, VK.BLEND_OP_ADD,
        VK.BLEND_FACTOR_ONE, VK.BLEND_FACTOR_ONE, VK.BLEND_OP_ADD,
        VK.COLOR_COMPONENT_R_BIT | VK.COLOR_COMPONENT_G_BIT |
        VK.COLOR_COMPONENT_B_BIT | VK.COLOR_COMPONENT_A_BIT,
    )
end

function vk_blend(::NoWrite)
    VK.PipelineColorBlendAttachmentState(
        false,
        VK.BLEND_FACTOR_ONE, VK.BLEND_FACTOR_ZERO, VK.BLEND_OP_ADD,
        VK.BLEND_FACTOR_ONE, VK.BLEND_FACTOR_ZERO, VK.BLEND_OP_ADD,
        VK.ColorComponentFlag(0),
    )
end

function vk_blend(::Premultiplied)
    VK.PipelineColorBlendAttachmentState(
        true,
        VK.BLEND_FACTOR_ONE, VK.BLEND_FACTOR_ONE_MINUS_SRC_ALPHA, VK.BLEND_OP_ADD,
        VK.BLEND_FACTOR_ONE, VK.BLEND_FACTOR_ONE_MINUS_SRC_ALPHA, VK.BLEND_OP_ADD,
        VK.COLOR_COMPONENT_R_BIT | VK.COLOR_COMPONENT_G_BIT |
        VK.COLOR_COMPONENT_B_BIT | VK.COLOR_COMPONENT_A_BIT,
    )
end

"""Record a layout transition of `image`'s first color mip level and layer."""
function imagebarrier!(cmd, image::VK.Image,
                              old_layout, new_layout,
                              src_stage, dst_stage,
                              src_access, dst_access)
    barrier = VK.ImageMemoryBarrier(
        src_access, dst_access,
        old_layout, new_layout,
        VK.QUEUE_FAMILY_IGNORED, VK.QUEUE_FAMILY_IGNORED,
        image,
        VK.ImageSubresourceRange(VK.IMAGE_ASPECT_COLOR_BIT,
            UInt32(0), UInt32(1), UInt32(0), UInt32(1)),
    )
    VK.cmd_pipeline_barrier(cmd, [], [], [barrier];
        src_stage_mask=src_stage, dst_stage_mask=dst_stage)
end

"""
    begin_pass!(e, color_view(s), color_image(s), extent; clear_color, load_op,
                depth_view, depth_image, depth_clear=1f0, depth_load_op,
                depth_store_op, transition=true)

Begin a dynamic rendering pass; record draws with `draw_in_pass!` and finish with
`end_pass!`. One colour attachment or vectors of several (fragment location `i`
writes attachment `i`), with `clear_color` and `load_op` once or once each. A
clear value means CLEAR, `nothing` means LOAD, and only an explicit load op
reaches DONT_CARE; `depth_clear` and `depth_load_op` work the same way.
"""
begin_pass!(target, color_view::VK.ImageView, color_image::VK.Image,
               extent::VK.Extent2D; kw...) =
    begin_pass!(target, [color_view], [color_image], extent; kw...)

"""One value for every attachment (anything but a vector), or a vector with one
value each. Any other length is an error."""
function perattachment(x, n::Integer, what::String)
    x isa AbstractVector || return [x for _ in 1:n]
    length(x) == n && return x
    error("$what has $(length(x)) entries but the pass has $n colour attachments")
end

function begin_pass!(e::Emitter,
                         color_views::AbstractVector{<:VK.ImageView},
                         color_images::AbstractVector{<:VK.Image},
                         extent::VK.Extent2D;
                         clear_color=(0f0, 0f0, 0f0, 1f0),
                         depth_view::Union{Nothing, VK.ImageView}=nothing,
                         depth_image::Union{Nothing, VK.Image}=nothing,
                         depth_clear::Union{Nothing, Float32}=1.0f0,
                         depth_load_op::Union{Nothing, VK.AttachmentLoadOp}=nothing,
                         depth_store_op::VK.AttachmentStoreOp=VK.ATTACHMENT_STORE_OP_STORE,
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
            depth_barrier = VK.ImageMemoryBarrier(
                # Ordered after the previous depth write either way: the transition
                # is a write itself, so a discarding one is a hazard too.
                VK.ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                VK.ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                    VK.ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                old, VK.IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                VK.QUEUE_FAMILY_IGNORED, VK.QUEUE_FAMILY_IGNORED,
                depth_image,
                VK.ImageSubresourceRange(VK.IMAGE_ASPECT_DEPTH_BIT,
                    UInt32(0), UInt32(1), UInt32(0), UInt32(1)),
            )
            VK.cmd_pipeline_barrier(cmd, [], [], [depth_barrier];
                src_stage_mask=VK.PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
                dst_stage_mask=VK.PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT)
        end
        depth_clear_val = VK.ClearValue(VK.ClearDepthStencilValue(
            depth_clear === nothing ? 1.0f0 : depth_clear, UInt32(0)))
        depth_attachment = VK.RenderingAttachmentInfo(
            VK.IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            VK.IMAGE_LAYOUT_UNDEFINED,
            dop,
            depth_store_op,
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
    draw_in_pass!(e, pipeline, vertex_count; push_data, push_bda, instances,
                  viewport, scissor)

Record a draw inside an active rendering pass. `viewport` and `scissor` are Vulkan
structs, set only when given.
"""
function draw_in_pass!(e::Emitter,
                           pipeline::VulkanCompiledGraphicsPipeline,
                           vertex_count::Integer;
                           push_data::Vector{UInt8}=UInt8[],
                           # The device address by value; `push_data` allocates per draw.
                           push_bda::UInt64=UInt64(0),
                           instances::Integer=1,
                           viewport::Union{Nothing, VK.Viewport}=nothing,
                           scissor::Union{Nothing, VK.Rect2D}=nothing)
    bind_for_draw!(e, pipeline; push_data, push_bda, viewport, scissor)
    VK.cmd_draw(e.cmd, UInt32(vertex_count), UInt32(instances), UInt32(0), UInt32(0))
    # Hold the pipeline: the command buffer names it until its submission passes.
    hold!(e, pipeline)
end

"""
    draw_mesh_in_pass!(e, pipeline, groups; push_data, push_bda, viewport, scissor)

Record a mesh draw inside an active rendering pass. `groups` is a workgroup count;
the mesh stage's `set_mesh_outputs!` says how much geometry each group emits.
"""
function draw_mesh_in_pass!(e::Emitter,
                            pipeline::VulkanCompiledGraphicsPipeline,
                            groups::Integer;
                            push_data::Vector{UInt8}=UInt8[],
                            push_bda::UInt64=UInt64(0),
                            viewport::Union{Nothing, VK.Viewport}=nothing,
                            scissor::Union{Nothing, VK.Rect2D}=nothing)
    bind_for_draw!(e, pipeline; push_data, push_bda, viewport, scissor)
    VK.cmd_draw_mesh_tasks_ext(e.cmd, UInt32(groups), UInt32(1), UInt32(1))
    hold!(e, pipeline)
end

# A graph gives every draw an argument-memory offset, including shaders whose
# ABI has no arguments. Such a layout has no push-constant range at all.
function push_draw_args!(cmd, pipeline::VulkanCompiledGraphicsPipeline,
                         push_bda::UInt64, push_data::Vector{UInt8})
    pipeline.push_constant_size == 0 && return nothing
    # `push_bda` by value if given, else the bytes of `push_data`.
    if push_bda != UInt64(0)
        bda = Ref(push_bda)
        GC.@preserve bda begin
            VK.cmd_push_constants(cmd, pipeline.pipeline_layout,
                pipeline.push_stage_flags, UInt32(0), UInt32(8),
                Ptr{Nothing}(Base.unsafe_convert(Ptr{UInt64}, bda)))
        end
    elseif !isempty(push_data)
        GC.@preserve push_data begin
            VK.cmd_push_constants(cmd, pipeline.pipeline_layout,
                pipeline.push_stage_flags, UInt32(0), UInt32(length(push_data)),
                Ptr{Nothing}(pointer(push_data)))
        end
    end
    return nothing
end

"""
Bind the pipeline and set everything a draw reads before the draw command:
viewport, scissor and the push constant block.
"""
function bind_for_draw!(e::Emitter, pipeline::VulkanCompiledGraphicsPipeline;
                        push_data::Vector{UInt8}=UInt8[],
                        push_bda::UInt64=UInt64(0),
                        viewport::Union{Nothing, VK.Viewport}=nothing,
                        scissor::Union{Nothing, VK.Rect2D}=nothing)
    cmd = e.cmd

    VK.cmd_bind_pipeline(cmd, VK.PIPELINE_BIND_POINT_GRAPHICS, pipeline.pipeline)

    if viewport !== nothing
        VK.cmd_set_viewport(cmd, [viewport])
    end
    if scissor !== nothing
        VK.cmd_set_scissor(cmd, [scissor])
    end

    push_draw_args!(cmd, pipeline, push_bda, push_data)
    return nothing
end

"""
    indirect_buffer(bq, n = 1) -> LavaArray{DrawIndirectCommand,1}

Room for `n` draw commands, with the `BUFFER_USAGE_INDIRECT_BUFFER_BIT` ordinary
allocations lack; without it the error only surfaces at the draw.
"""
indirect_buffer(bq::SubmitChannel{<:VulkanQueue}, n::Integer=1) =
    LavaArray{DrawIndirectCommand,1}(undef, (Int(n),);
        bq, extra_usage=UInt32(VK.BUFFER_USAGE_INDIRECT_BUFFER_BIT))

"""
    draw_indirect_in_pass!(e, pipeline, commands; first = 1, count = 1, push_bda)

Record `count` draws whose parameters are in device memory, from `commands` (a
`LavaArray{DrawIndirectCommand}` made by [`indirect_buffer`](@ref)) starting at the
1-based `first`, inside an active rendering pass. The host never reads them, so a
compute pass of the same frame can write them.
"""
function draw_indirect_in_pass!(e::Emitter,
                                   pipeline::VulkanCompiledGraphicsPipeline,
                                   commands::LavaArray{DrawIndirectCommand,1};
                                   first::Integer=1,
                                   count::Integer=1,
                                   push_bda::UInt64=UInt64(0))
    cmd = e.cmd

    VK.cmd_bind_pipeline(cmd, VK.PIPELINE_BIND_POINT_GRAPHICS, pipeline.pipeline)

    if push_bda != UInt64(0)
        bda = Ref(push_bda)
        GC.@preserve bda begin
            VK.cmd_push_constants(cmd, pipeline.pipeline_layout,
                pipeline.push_stage_flags, UInt32(0), UInt32(8),
                Ptr{Nothing}(Base.unsafe_convert(Ptr{UInt64}, bda)))
        end
    end

    managed = commands.buf[]
    offset = pool_offset(managed) + commands.offset +
             (first - 1) * sizeof(DrawIndirectCommand)
    VK.cmd_draw_indirect(cmd, managed.buffer, UInt64(offset),
                             UInt32(count), UInt32(sizeof(DrawIndirectCommand)))
    hold!(e, pipeline)
    # The device reads the commands, so the array must outlive the draw.
    hold!(e, commands)
end

"""
    draw_indexed_in_pass!(e, pipeline, index_count; push_data, push_bda,
                          indices_buffer, indices_offset, instances)

Record an indexed draw with `UInt32` indices at byte `indices_offset` of
`indices_buffer`, inside an active rendering pass. Holds only the pipeline: the
caller keeps the index buffer alive.
"""
function draw_indexed_in_pass!(e::Emitter,
                                   pipeline::VulkanCompiledGraphicsPipeline,
                                   index_count::Integer;
                                   push_data::Vector{UInt8}=UInt8[],
                                   push_bda::UInt64=UInt64(0),
                                   indices_buffer::VK.Buffer,
                                   # A `LavaArray` may be a slice of its buffer.
                                   indices_offset::UInt64=UInt64(0),
                                   instances::Integer=1)
    cmd = e.cmd

    VK.cmd_bind_pipeline(cmd, VK.PIPELINE_BIND_POINT_GRAPHICS, pipeline.pipeline)

    push_draw_args!(cmd, pipeline, push_bda, push_data)

    VK.cmd_bind_index_buffer(cmd, indices_buffer, indices_offset, VK.INDEX_TYPE_UINT32)
    VK.cmd_draw_indexed(cmd, UInt32(index_count), UInt32(instances),
                             UInt32(0), Int32(0), UInt32(0))
    hold!(e, pipeline)
end

"""
    set_viewport!(e, viewport, scissor)

Set viewport and scissor once for every following draw in the pass, rather than
per draw (two commands and two vector allocations each).
"""
function set_viewport!(e::Emitter, viewport::VK.Viewport, scissor::VK.Rect2D)
    VK.cmd_set_viewport(e.cmd, [viewport])
    VK.cmd_set_scissor(e.cmd, [scissor])
    nothing
end

"""
    end_pass!(emitter)

End the dynamic rendering pass the emitter opened.
"""
end_pass!(e::Emitter) = VK.cmd_end_rendering(e.cmd)

"""
    set_viewport!(e, x, y, w, h)

The portable viewport verb (`graphics/commands.jl`): sets the viewport and the
scissor it implies. The scissor is clamped to non-negative offsets, which Vulkan
requires, and a flipped viewport (negative `h`, for Y-down clip space) puts its
origin at `y + h`.
"""
function set_viewport!(e::Emitter, x::Real, y::Real, w::Real, h::Real)
    cmd = e.cmd
    VK.cmd_set_viewport(cmd, [VK.Viewport(Float32(x), Float32(y), Float32(w), Float32(h), 0f0, 1f0)])

    sx = Int32(floor(x))
    sy = h < 0 ? Int32(floor(y + h)) : Int32(floor(y))
    sw = Int32(ceil(abs(w)))
    sh = Int32(ceil(abs(h)))
    if sx < Int32(0)
        sw = max(Int32(0), sw + sx); sx = Int32(0)
    end
    if sy < Int32(0)
        sh = max(Int32(0), sh + sy); sy = Int32(0)
    end
    VK.cmd_set_scissor(cmd, [VK.Rect2D(VK.Offset2D(sx, sy), VK.Extent2D(UInt32(sw), UInt32(sh)))])
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

# Portable form: the area as `w`, `h` instead of a `VK.Extent2D`.
begin_pass!(e::Emitter, view, image, w::Integer, h::Integer; kw...) =
    begin_pass!(e, view, image, VK.Extent2D(UInt32(w), UInt32(h)); kw...)
