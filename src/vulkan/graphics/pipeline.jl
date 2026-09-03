# Graphics pipeline creation for Lava.jl
#
# Creates VkGraphicsPipeline using VK_KHR_dynamic_rendering (no VkRenderPass).
# Pipeline state (blend, cull, depth, topology) configured via dispatch on config types.


# The graphics pipeline cache is a `VkContext` field; a reset makes a new one.

"""
    create_graphics_pipeline(vertex_spirv, fragment_spirv;
        blend=Opaque(), cull=CullBack(), topology=TriangleList(), depth=DepthLess(),
        color_format=VK.FORMAT_B8G8R8A8_SRGB,
        depth_format=VK.FORMAT_UNDEFINED, push_constant_size=8,
        geometry_spirv=nothing, geometry_config=nothing,
        tess_ctrl_spirv=nothing, tess_eval_spirv=nothing, tess_config=nothing,
        descriptor_set_layout=nothing) -> VulkanCompiledGraphicsPipeline

Create a graphics pipeline from SPIR-V shader binaries.
Uses VK_KHR_dynamic_rendering — no VkRenderPass needed.
"""
function create_graphics_pipeline(vertex_spirv::Vector{UInt8},
                                    fragment_spirv::Vector{UInt8};
                                    ctx::VkContext=vk_context(),
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

    # Create shader modules
    vert_mod = create_gfx_shader_module(dev, vertex_spirv)
    frag_mod = create_gfx_shader_module(dev, fragment_spirv)
    modules = VK.ShaderModule[vert_mod, frag_mod]

    stages = VK.PipelineShaderStageCreateInfo[
        VK.PipelineShaderStageCreateInfo(
            VK.SHADER_STAGE_VERTEX_BIT, vert_mod, "main"),
        VK.PipelineShaderStageCreateInfo(
            VK.SHADER_STAGE_FRAGMENT_BIT, frag_mod, "main"),
    ]

    # Optional: geometry shader
    if geometry_spirv !== nothing
        geom_mod = create_gfx_shader_module(dev, geometry_spirv)
        push!(modules, geom_mod)
        push!(stages, VK.PipelineShaderStageCreateInfo(
            VK.SHADER_STAGE_GEOMETRY_BIT, geom_mod, "main"))
    end

    # Optional: tessellation shaders
    if tess_ctrl_spirv !== nothing && tess_eval_spirv !== nothing
        tc_mod = create_gfx_shader_module(dev, tess_ctrl_spirv)
        te_mod = create_gfx_shader_module(dev, tess_eval_spirv)
        push!(modules, tc_mod, te_mod)
        push!(stages, VK.PipelineShaderStageCreateInfo(
            VK.SHADER_STAGE_TESSELLATION_CONTROL_BIT, tc_mod, "main"))
        push!(stages, VK.PipelineShaderStageCreateInfo(
            VK.SHADER_STAGE_TESSELLATION_EVALUATION_BIT, te_mod, "main"))
    end

    # Vertex input: EMPTY (BDA vertex pulling)
    vertex_input = VK.PipelineVertexInputStateCreateInfo([], [])

    # Input assembly
    vk_topo = vk_topology(topology)
    input_assembly = VK.PipelineInputAssemblyStateCreateInfo(vk_topo, false)

    # Tessellation state (if applicable)
    tess_state = C_NULL
    if tess_config !== nothing
        tess_state = VK.PipelineTessellationStateCreateInfo(UInt32(tess_config.patch_vertices))
    end

    # Viewport/scissor: dynamic (set via cmd_set_viewport/cmd_set_scissor)
    # Pass dummy viewport/scissor — will be overridden by dynamic state
    dummy_viewport = VK.Viewport(0.0f0, 0.0f0, 1.0f0, 1.0f0, 0.0f0, 1.0f0)
    dummy_scissor = VK.Rect2D(VK.Offset2D(0, 0), VK.Extent2D(1, 1))
    viewport_state = VK.PipelineViewportStateCreateInfo(;
        viewports=[dummy_viewport], scissors=[dummy_scissor])

    # Rasterization
    cull_mode = vk_cull(cull)
    rasterization = VK.PipelineRasterizationStateCreateInfo(
        false, false,
        VK.POLYGON_MODE_FILL,
        VK.FRONT_FACE_COUNTER_CLOCKWISE,
        false, 0.0f0, 0.0f0, 0.0f0,
        1.0f0;  # line width
        cull_mode=cull_mode,
    )

    # Multisampling: no MSAA
    multisample = VK.PipelineMultisampleStateCreateInfo(
        VK.SAMPLE_COUNT_1_BIT, false, 1.0f0, false, false)

    # Depth/stencil.
    #
    # Whether the pipeline declares a depth attachment is a property of the RENDER
    # TARGET, not of the depth mode: with dynamic rendering the format baked in here
    # must equal the format of the view bound at `vkCmdBeginRendering`, and NULL
    # there means UNDEFINED here (VUID-vkCmdDraw-pDepthAttachment-06181). Deriving
    # it from `depth isa DepthOff` instead got both mismatches wrong — a DepthOff
    # draw into a framebuffer that has depth, and a DepthLess draw into a window
    # that has none — because neither combination is about the test being on.
    has_depth = depth_format != VK.FORMAT_UNDEFINED
    depth_enable, depth_compare = vk_depth(depth)
    if depth_enable && !has_depth
        throw(ArgumentError(
            "$(typeof(depth)) needs a depth attachment, and this render target has none. " *
            "Pass `depth_format` for the target's depth view, or build the pipeline with " *
            "`depth = DepthOff()`."))
    end
    depth_stencil = VK.PipelineDepthStencilStateCreateInfo(
        depth_enable, depth_enable,
        depth_compare,
        false,    # depth bounds test
        false,    # stencil test
        VK.StencilOpState(VK.STENCIL_OP_KEEP, VK.STENCIL_OP_KEEP,
            VK.STENCIL_OP_KEEP, VK.COMPARE_OP_ALWAYS, 0, 0, 0),
        VK.StencilOpState(VK.STENCIL_OP_KEEP, VK.STENCIL_OP_KEEP,
            VK.STENCIL_OP_KEEP, VK.COMPARE_OP_ALWAYS, 0, 0, 0),
        0.0f0, 1.0f0,
    )

    # Color blend, one attachment state per colour target: with several targets
    # Vulkan requires the counts to match, and every target blends the same way
    # because blending is pipeline state and the pipeline is one.
    color_formats = colorformats(color_format)
    blend_attachment = vk_blend(blend)
    color_blend = VK.PipelineColorBlendStateCreateInfo(
        false, VK.LOGIC_OP_COPY,
        [blend_attachment for _ in color_formats],
        (0.0f0, 0.0f0, 0.0f0, 0.0f0),
    )

    # Dynamic state
    dynamic_states = [VK.DYNAMIC_STATE_VIEWPORT, VK.DYNAMIC_STATE_SCISSOR]
    dynamic_state = VK.PipelineDynamicStateCreateInfo(dynamic_states)

    # Pipeline layout
    all_stage_flags = VK.SHADER_STAGE_VERTEX_BIT | VK.SHADER_STAGE_FRAGMENT_BIT
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

    # Dynamic rendering info (Vulkan 1.3+)
    rendering_info = VK.PipelineRenderingCreateInfo(
        UInt32(0),           # view mask
        color_formats,       # color attachment formats
        depth_format,
        VK.FORMAT_UNDEFINED,  # stencil format
    )

    # Create graphics pipeline (dynamic rendering — null render pass)
    gfx_flags = VK.PipelineCreateFlag(0)
    if PIPELINE_NO_COMPILE[]
        gfx_flags |= VK.PipelineCreateFlag(VK.PIPELINE_CREATE_FAIL_ON_PIPELINE_COMPILE_REQUIRED_BIT) |
                     VK.PipelineCreateFlag(VK.PIPELINE_CREATE_EARLY_RETURN_ON_FAILURE_BIT)
    end
    ci = VK.GraphicsPipelineCreateInfo(
        stages,
        rasterization,
        layout, UInt32(0), Int32(-1);
        flags=gfx_flags,
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

    pipelines, _ = @vk_checked "vkCreateGraphicsPipelines" VK.create_graphics_pipelines(
        dev, [ci]; pipeline_cache=vk_context().pipeline_cache)
    pipeline = pipelines[1]

    return VulkanCompiledGraphicsPipeline(
        pipeline, layout, modules,
        UInt32(push_constant_size),
        descriptor_set_layout,
        all_stage_flags,
        color_formats, has_depth,
    )
end

"""
One colour attachment or several, normalised. A single format is the common case
and stays spellable as itself rather than as a one-element vector.
"""
colorformats(f::VK.Format) = [f]
colorformats(f::AbstractVector{VK.Format}) = collect(f)

# …and the portable spelling. A caller outside this backend names an attachment
# format by its Julia element type — `RGBA{Float32}`, `BGRA{N0f8}` — because
# that is the vocabulary `vkformat` already lowers for transient images. Funnel
# it here so every `color_format=` keyword in this file accepts both without
# each one needing its own conversion.
colorformats(T::Type) = [vkformat(VulkanAPI(), T)]
colorformats(ts::AbstractVector{<:Type}) = [vkformat(VulkanAPI(), T) for T in ts]

function create_gfx_shader_module(dev::VK.Device, spirv_bytes::Vector{UInt8})
    @assert length(spirv_bytes) % 4 == 0 "SPIR-V binary must be 4-byte aligned"
    code_u32 = reinterpret(UInt32, spirv_bytes)
    VK.ShaderModule(dev, length(spirv_bytes), code_u32)
end

# ── Pipeline State Dispatch ──

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
function vk_depth(::DepthOff)
    (false, VK.COMPARE_OP_ALWAYS)
end

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

function vk_blend(::Premultiplied)
    VK.PipelineColorBlendAttachmentState(
        true,
        VK.BLEND_FACTOR_ONE, VK.BLEND_FACTOR_ONE_MINUS_SRC_ALPHA, VK.BLEND_OP_ADD,
        VK.BLEND_FACTOR_ONE, VK.BLEND_FACTOR_ONE_MINUS_SRC_ALPHA, VK.BLEND_OP_ADD,
        VK.COLOR_COMPONENT_R_BIT | VK.COLOR_COMPONENT_G_BIT |
        VK.COLOR_COMPONENT_B_BIT | VK.COLOR_COMPONENT_A_BIT,
    )
end

# ── Draw Recording ──

"""
    vk_draw!(pipeline::VulkanCompiledGraphicsPipeline, color_view::VK.ImageView,
             color_image::VK.Image, extent::VK.Extent2D,
             vertex_count::Integer; push_data=UInt8[], instances=1,
             depth_view=nothing, clear_color=nothing,
             indices_buffer=nothing, index_count=0)

Record a draw command using dynamic rendering.
"""
function vk_draw!(bq::VulkanBatchQueue,
                   pipeline::VulkanCompiledGraphicsPipeline,
                   color_view::VK.ImageView,
                   color_image::VK.Image,
                   extent::VK.Extent2D,
                   vertex_count::Integer;
                   push_data::Vector{UInt8}=UInt8[],
                   instances::Integer=1,
                   depth_view::Union{Nothing, VK.ImageView}=nothing,
                   depth_image::Union{Nothing, VK.Image}=nothing,
                   depth_clear::Union{Nothing, Float32}=1.0f0,
                   depth_store_op::VK.AttachmentStoreOp=VK.ATTACHMENT_STORE_OP_STORE,
                   clear_color::Union{Nothing, NTuple{4, Float32}}=nothing,
                   indices_buffer::Union{Nothing, VK.Buffer}=nothing,
                   index_count::Integer=0,
                   descriptor_set::Union{Nothing, VK.DescriptorSet}=nothing)
    # Route through record_dispatch! so the prior-dispatch → draw barrier,
    # dispatch_count bookkeeping, CB-split logic, and dispatch-log accounting
    # all come from the single shared helper.  The do-block handles the
    # graphics-specific work (image transitions, dynamic rendering scope,
    # bind + draw + end_rendering).
    record_dispatch!(bq;
        dst_stage = VK.PIPELINE_STAGE_VERTEX_SHADER_BIT |
                    VK.PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        extra_dst_access = VK.ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
        info = "draw vtx=$vertex_count",
    ) do batch
        cmd = batch.cmd_buf

        # Transition color image to COLOR_ATTACHMENT_OPTIMAL.
        #
        # `clear_color = nothing` is a LOAD, so the previous contents have to
        # survive: the old layout is what the last pass left the image in, and the
        # destination has to allow the read the load op performs. From UNDEFINED
        # the driver is free to throw those contents away, which is the whole
        # point of drawing without a clear. Synchronization validation reports the
        # missing halves as a WRITE_AFTER_WRITE on the transition and a
        # READ_AFTER_WRITE at vkCmdBeginRendering; `begin_pass!` has always got
        # this right and this path did not.
        loads_color = clear_color === nothing
        transition_image!(cmd, color_image,
            loads_color ? VK.IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL :
                          VK.IMAGE_LAYOUT_UNDEFINED,
            VK.IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            # srcStage is COLOR_ATTACHMENT_OUTPUT, not TOP_OF_PIPE. On a swapchain
            # image the submit waits on the semaphore `vkAcquireNextImageKHR`
            # signals, at COLOR_ATTACHMENT_OUTPUT — and a barrier whose srcStage is
            # TOP_OF_PIPE creates NO execution dependency with that wait, so this
            # transition (and the writes behind it) can run while the presentation
            # engine still owns the image. Synchronization validation reports it as
            # "WRITE_AFTER_READ hazard … previously accessed by vkAcquireNextImageKHR";
            # on screen it is bands of stale pixels that vanish under any full sync.
            VK.PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            VK.PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            VK.ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
            loads_color ? VK.ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                          VK.ACCESS_COLOR_ATTACHMENT_READ_BIT :
                          VK.ACCESS_COLOR_ATTACHMENT_WRITE_BIT)

        # Transition depth image to DEPTH_STENCIL_ATTACHMENT_OPTIMAL.
        #
        # `depth_clear = nothing` means this draw builds on the depth the last one
        # left, so the transition comes from DEPTH_STENCIL_ATTACHMENT_OPTIMAL and
        # waits on that write. From UNDEFINED it would be free to throw the depth
        # away, which is exactly what a second draw testing against the first needs
        # to keep.
        if depth_image !== nothing
            loads = depth_clear === nothing
            # The source side names the previous depth write whichever way the load
            # op goes: a layout transition is itself a write, so discarding one
            # still has to be ordered after the store op of the pass before it.
            depth_barrier = VK.ImageMemoryBarrier(
                VK.ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                VK.ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK.ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                loads ? VK.IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL :
                        VK.IMAGE_LAYOUT_UNDEFINED,
                VK.IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                VK.QUEUE_FAMILY_IGNORED, VK.QUEUE_FAMILY_IGNORED,
                depth_image,
                VK.ImageSubresourceRange(VK.IMAGE_ASPECT_DEPTH_BIT,
                    UInt32(0), UInt32(1), UInt32(0), UInt32(1)),
            )
            VK.cmd_pipeline_barrier(cmd, [], [], [depth_barrier];
                src_stage_mask=VK.PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
                dst_stage_mask=VK.PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT)
        end

        # Color attachment for dynamic rendering
        clear_val = if clear_color !== nothing
            VK.ClearValue(VK.ClearColorValue(clear_color))
        else
            VK.ClearValue(VK.ClearColorValue((0.0f0, 0.0f0, 0.0f0, 1.0f0)))
        end

        load_op = clear_color !== nothing ? VK.ATTACHMENT_LOAD_OP_CLEAR : VK.ATTACHMENT_LOAD_OP_LOAD

        color_attachment = VK.RenderingAttachmentInfo(
            VK.IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            VK.IMAGE_LAYOUT_UNDEFINED,  # resolve image layout (unused)
            load_op,
            VK.ATTACHMENT_STORE_OP_STORE,
            clear_val;
            image_view=color_view,
            resolve_mode=VK.RESOLVE_MODE_NONE,
        )

        # Depth attachment (optional)
        depth_attachment = C_NULL
        if depth_view !== nothing
            depth_clear_val = VK.ClearValue(VK.ClearDepthStencilValue(
                depth_clear === nothing ? 1.0f0 : depth_clear, UInt32(0)))
            depth_attachment = VK.RenderingAttachmentInfo(
                VK.IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                VK.IMAGE_LAYOUT_UNDEFINED,
                depth_clear === nothing ? VK.ATTACHMENT_LOAD_OP_LOAD :
                                          VK.ATTACHMENT_LOAD_OP_CLEAR,
                depth_store_op,
                depth_clear_val;
                image_view=depth_view,
                resolve_mode=VK.RESOLVE_MODE_NONE,
            )
        end

        render_area = VK.Rect2D(VK.Offset2D(0, 0), extent)

        rendering_info = VK.RenderingInfo(
            render_area,
            UInt32(1),  # layer count
            UInt32(0),  # view mask
            [color_attachment];
            depth_attachment=depth_attachment,
        )

        VK.cmd_begin_rendering(cmd, rendering_info)

        # Bind pipeline + pin for batch lifetime
        VK.cmd_bind_pipeline(cmd, VK.PIPELINE_BIND_POINT_GRAPHICS, pipeline.pipeline)
        pin!(batch, pipeline)

        # Bind descriptor set (for textures)
        if descriptor_set !== nothing
            VK.cmd_bind_descriptor_sets(cmd, VK.PIPELINE_BIND_POINT_GRAPHICS,
                pipeline.pipeline_layout, UInt32(0), [descriptor_set], UInt32[])
        end

        # Dynamic viewport + scissor
        viewport = VK.Viewport(0.0f0, 0.0f0,
            Float32(extent.width), Float32(extent.height),
            0.0f0, 1.0f0)
        VK.cmd_set_viewport(cmd, [viewport])

        scissor = VK.Rect2D(VK.Offset2D(0, 0), extent)
        VK.cmd_set_scissor(cmd, [scissor])

        # Push constants
        if !isempty(push_data)
            GC.@preserve push_data begin
                VK.cmd_push_constants(cmd, pipeline.pipeline_layout,
                    pipeline.push_stage_flags, UInt32(0), UInt32(length(push_data)),
                    Ptr{Nothing}(pointer(push_data)))
            end
        end

        # Draw
        if indices_buffer !== nothing
            VK.cmd_bind_index_buffer(cmd, indices_buffer, UInt64(0), VK.INDEX_TYPE_UINT32)
            VK.cmd_draw_indexed(cmd, UInt32(index_count), UInt32(instances),
                                     UInt32(0), Int32(0), UInt32(0))
        else
            VK.cmd_draw(cmd, UInt32(vertex_count), UInt32(instances), UInt32(0), UInt32(0))
        end

        VK.cmd_end_rendering(cmd)
    end
end

"""Transition an image layout using a pipeline barrier."""
function transition_image!(cmd, image::VK.Image,
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

# =============================================================================
# Multi-draw rendering API
# =============================================================================
# Separates begin/draw/end for rendering multiple plots in a single pass.

"""
    begin_pass!(color_view(s), color_image(s), extent; clear_color, depth_view)

Begin a dynamic rendering pass. Call `draw_in_pass!` for each draw,
then `end_pass!` to finish.

One colour attachment or several: pass vectors of views and images, and
`clear_color` and `load_op` either once for all of them or once each. The
fragment shader writes location `i` to attachment `i`.

The depth attachment takes the same three-way answer as the colour one:
`depth_clear` gives a value to clear to, `depth_clear = nothing` loads what is
there, and `depth_load_op` states it outright, which is the only way to reach
DONT_CARE.
"""
begin_pass!(target, color_view::VK.ImageView, color_image::VK.Image,
               extent::VK.Extent2D; kw...) =
    begin_pass!(target, [color_view], [color_image], extent; kw...)

# The four render-pass verbs take an EMITTER. A queue is accepted too and means
# "whatever batch is open", which is what every caller outside a plan wants —
# RayMakie's overlay, `blit!`, a test drawing a triangle. One line each, so
# there is one implementation and the queue form cannot drift from it.
begin_pass!(bq::VulkanBatchQueue, views::AbstractVector{<:VK.ImageView},
            images::AbstractVector{<:VK.Image}, extent::VK.Extent2D; kw...) =
    begin_pass!(emitter(bq), views, images, extent; kw...)
end_pass!(bq::VulkanBatchQueue) = end_pass!(emitter(bq))
set_viewport!(bq::VulkanBatchQueue, args...) = set_viewport!(emitter(bq), args...)
draw_in_pass!(bq::VulkanBatchQueue, pipeline, n::Integer; kw...) =
    draw_in_pass!(emitter(bq), pipeline, n; kw...)
draw_indirect_in_pass!(bq::VulkanBatchQueue, pipeline, commands; kw...) =
    draw_indirect_in_pass!(emitter(bq), pipeline, commands; kw...)

"""One value for every attachment, or one value each. Anything else is a mistake
worth naming rather than a silent recycle."""
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
        # Resolved once, so it means the same thing on every path. An explicit
        # `load_op` is the only way to reach DONT_CARE: inferring it from
        # `clear_color === nothing` can only ever pick CLEAR or LOAD, so a pass that
        # overwrites every pixel had to pay for a load whose result it discards.
        op = given !== nothing ? given :
             clear !== nothing ? VK.ATTACHMENT_LOAD_OP_CLEAR :
                                 VK.ATTACHMENT_LOAD_OP_LOAD

        # `transition=false` hands the layout transition to the caller. A caller that
        # derives barriers from declared usage has to own this one too, because the
        # transition below assumes it is the only writer of that image's layout and
        # would double up with, or contradict, a derived barrier.
        if !transition
            clear_val = clear !== nothing ?
                VK.ClearValue(VK.ClearColorValue(clear)) :
                VK.ClearValue(VK.ClearColorValue((0f0, 0f0, 0f0, 0f0)))
        elseif clear !== nothing
            # Clear mode: discard old content, start fresh
            transition_image!(cmd, color_images[i],
                VK.IMAGE_LAYOUT_UNDEFINED, VK.IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                # srcStage is COLOR_ATTACHMENT_OUTPUT, not TOP_OF_PIPE. On a swapchain
                # image the submit waits on the semaphore `vkAcquireNextImageKHR`
                # signals, at COLOR_ATTACHMENT_OUTPUT — and a barrier whose srcStage is
                # TOP_OF_PIPE creates NO execution dependency with that wait, so this
                # transition (and the writes behind it) can run while the presentation
                # engine still owns the image. Synchronization validation reports it as
                # "WRITE_AFTER_READ hazard … previously accessed by vkAcquireNextImageKHR";
                # on screen it is bands of stale pixels that vanish under any full sync.
                VK.PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                VK.PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                VK.ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK.ACCESS_COLOR_ATTACHMENT_WRITE_BIT)

            clear_val = VK.ClearValue(VK.ClearColorValue(clear))
        else
            # Load mode: preserve existing content (for drawing on top of previous pass)
            transition_image!(cmd, color_images[i],
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

    # Typed even when empty: a depth-only pass has no colour attachments, and
    # `map` over an empty range does not always infer an element type the Vulkan
    # constructor will take.
    color_attachments = convert(Vector{VK.RenderingAttachmentInfo}, color_attachments)

    depth_attachment = C_NULL
    if depth_view !== nothing
        dop = depth_load_op !== nothing ? depth_load_op :
              depth_clear !== nothing ? VK.ATTACHMENT_LOAD_OP_CLEAR :
                                        VK.ATTACHMENT_LOAD_OP_LOAD
        # Same rule as the colour attachment: a discarding op does not need the old
        # contents, so it transitions from UNDEFINED; a LOAD does, so it has to come
        # from the layout the previous pass left the image in. `transition = false`
        # hands both to the caller.
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
    draw_in_pass!(pipeline, vertex_count; push_data, instances,
                     viewport, scissor)

Record a draw command within an active rendering pass.
Viewport/scissor are Vulkan structs for per-scene rendering.
"""
function draw_in_pass!(e::Emitter,
                           pipeline::VulkanCompiledGraphicsPipeline,
                           vertex_count::Integer;
                           push_data::Vector{UInt8}=UInt8[],
                           # The push constant is one buffer device address. Taking
                           # it as a value costs nothing; taking it as an eight-byte
                           # `Vector` costs an allocation per draw per frame.
                           push_bda::UInt64=UInt64(0),
                           instances::Integer=1,
                           viewport::Union{Nothing, VK.Viewport}=nothing,
                           scissor::Union{Nothing, VK.Rect2D}=nothing,
                           # A caller that owns the pipeline for longer than the
                           # frame does not need it pinned into every batch.
                           pin::Bool=true)
    cmd = e.cmd

    VK.cmd_bind_pipeline(cmd, VK.PIPELINE_BIND_POINT_GRAPHICS, pipeline.pipeline)

    # Set viewport and scissor
    if viewport !== nothing
        VK.cmd_set_viewport(cmd, [viewport])
    end
    if scissor !== nothing
        VK.cmd_set_scissor(cmd, [scissor])
    end

    # Push constants. The BDA path takes the address by value; the vector path is
    # for callers written before it existed.
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

    VK.cmd_draw(cmd, UInt32(vertex_count), UInt32(instances), UInt32(0), UInt32(0))
    drawn!(e.owner)
    # Pin the pipeline — prevents GC from destroying it while the command buffer references it
    pin && pin!(e, pipeline)
end

"""
    indirect_buffer(n = 1) -> LavaArray{DrawIndirectCommand,1}

Room for `n` draw commands, allocated so a draw may read them.

`BUFFER_USAGE_INDIRECT_BUFFER_BIT` is not in the flags an ordinary allocation
gets, and a buffer without it is a validation error at the draw rather than at
the allocation, which is a long way from the mistake.
"""
indirect_buffer(n::Integer=1) =
    LavaArray{DrawIndirectCommand,1}(undef, (Int(n),);
        extra_usage=UInt32(VK.BUFFER_USAGE_INDIRECT_BUFFER_BIT))

"""
    draw_indirect_in_pass!(bq, pipeline, commands; first, count, push_bda, pin)

Record a draw whose counts are in device memory, inside an active rendering pass.

`commands` is a `LavaArray` of `DrawIndirectCommand`, and `first` is which of
them to start at. Nothing here reads them: that is the point, because the numbers
are written by a compute pass in the same frame and reading them on the host
would mean waiting for it.

The array has to have been allocated with `BUFFER_USAGE_INDIRECT_BUFFER_BIT`,
which `indirect_buffer` does.
"""
function draw_indirect_in_pass!(e::Emitter,
                                   pipeline::VulkanCompiledGraphicsPipeline,
                                   commands::LavaArray{DrawIndirectCommand,1};
                                   first::Integer=1,
                                   count::Integer=1,
                                   push_bda::UInt64=UInt64(0),
                                   pin::Bool=true)
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
    drawn!(e.owner)
    pin && pin!(e, pipeline)
    pin!(e, commands)
end

"""
    draw_indexed_in_pass!(pipeline, index_count; push_data, indices_buffer)

Draw indexed geometry inside an active render pass (between begin_pass!/end_pass!).
Uses the provided index buffer for indexed drawing.
"""
function draw_indexed_in_pass!(bq::VulkanBatchQueue,
                                   pipeline::VulkanCompiledGraphicsPipeline,
                                   index_count::Integer;
                                   push_data::Vector{UInt8}=UInt8[],
                                   indices_buffer::VK.Buffer,
                                   instances::Integer=1)
    batch = bq.active_batch
    batch === nothing && error("draw_indexed_in_pass! called without an active rendering pass")
    cmd = batch.cmd_buf

    VK.cmd_bind_pipeline(cmd, VK.PIPELINE_BIND_POINT_GRAPHICS, pipeline.pipeline)

    if !isempty(push_data)
        GC.@preserve push_data begin
            VK.cmd_push_constants(cmd, pipeline.pipeline_layout,
                pipeline.push_stage_flags, UInt32(0), UInt32(length(push_data)),
                Ptr{Nothing}(pointer(push_data)))
        end
    end

    VK.cmd_bind_index_buffer(cmd, indices_buffer, UInt64(0), VK.INDEX_TYPE_UINT32)
    VK.cmd_draw_indexed(cmd, UInt32(index_count), UInt32(instances),
                             UInt32(0), Int32(0), UInt32(0))
    batch.dispatch_count += 1
    batch.last_was_rt = false
    pin!(batch, pipeline)
end

"""
    set_viewport!(bq, viewport, scissor)

Set viewport and scissor once, for every draw that follows in the pass.

They are dynamic pipeline state and every draw in a pass shares them, so setting
them per draw records two extra commands and allocates two vectors per draw per
frame — at a hundred plots, a third of the frame's garbage for a value that does
not change.
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
Mantle's portable viewport verb: plain numbers in, viewport AND the scissor it
implies out.

The five-argument form is the one `graphics/commands.jl` declares, and it exists
so a caller never has to build a `VK.Viewport`/`VK.Rect2D` — RayMakie used to,
which made every overlay draw a Vulkan-only line of code in a package that is
supposed to be portable.

The scissor is derived rather than passed because the clamping below is not a
choice: Vulkan rejects a negative offset, and a flipped viewport (negative
height, which is how Y-down clip space is handled) puts the rectangle's origin
at `y + h`. Deriving it in one place is what stops each caller getting that
wrong in its own way.
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
function use_bindings!(bq::VulkanBatchQueue, compiled, bindings)
    batch = bq.active_batch
    VK.cmd_bind_descriptor_sets(batch.cmd_buf, VK.PIPELINE_BIND_POINT_GRAPHICS,
                                compiled.pipeline_layout, UInt32(0),
                                [bindings.set], UInt32[])
    pin!(batch, bindings)
    return nothing
end

# Portable spelling: a pass is opened over a WxH area, not over a
# `VK.Extent2D`. The views and images stay backend-typed — they come from a
# `Framebuffer` or `Window` the backend made — but the size does not have to.
begin_pass!(bq::VulkanBatchQueue, view, image, w::Integer, h::Integer; kw...) =
    begin_pass!(bq, view, image, VK.Extent2D(UInt32(w), UInt32(h)); kw...)
