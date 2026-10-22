# Ray-tracing pipelines from SPIR-V, their shader binding tables, and trace dispatch.

const VK_SHADER_UNUSED_KHR = ~UInt32(0)  # 0xFFFFFFFF

"""
    LavaRTPipeline

A compiled ray-tracing pipeline with its shader binding table. Subtypes
`CompiledRTPipeline` (`runtime/coretypes.jl`) because it holds the `VkContext`
whose `DeviceCaches` store it, so the cache cannot name the concrete type.
"""
struct LavaRTPipeline <: CompiledRTPipeline
    pipeline::VK.Pipeline
    pipeline_layout::VK.PipelineLayout
    descriptor_set_layout::VK.DescriptorSetLayout
    # Shader modules (kept alive)
    shader_modules::Vector{VK.ShaderModule}
    # SBT
    sbt_buffer::VkManagedBuffer
    raygen_region::VK.StridedDeviceAddressRegionKHR
    miss_region::VK.StridedDeviceAddressRegionKHR
    hit_region::VK.StridedDeviceAddressRegionKHR
    callable_region::VK.StridedDeviceAddressRegionKHR
    push_constant_size::UInt32
    # Stage flags for push constants and descriptor sets
    stage_flags::VK.ShaderStageFlag
    # The context this pipeline was built against.
    ctx::VkContext
end

"""
    create_rt_pipeline(ctx, raygen_spirv, miss_spirv, chit_spirv(s);
                       anyhit_spirv=nothing, intersection_spirv=nothing,
                       push_constant_size=8) -> LavaRTPipeline

A ray-tracing pipeline from raygen, miss and one or more closest-hit SPIR-V
binaries. Groups: 0 raygen, 1 miss, 2..1+N one hit group per closest-hit, all
sharing the optional any-hit and intersection shaders; an intersection shader
makes them `PROCEDURAL_HIT_GROUP`s over AABBs. Set 0, binding 0 is the TLAS; the
push constant is the argument buffer's device address.
"""
function create_rt_pipeline(ctx::VkContext,
                            raygen_spirv::Vector{UInt8},
                            miss_spirv::Vector{UInt8},
                            chit_spirv::Vector{UInt8};
                            kwargs...)
    return create_rt_pipeline(ctx, raygen_spirv, miss_spirv,
                              Vector{UInt8}[chit_spirv]; kwargs...)
end

function create_rt_pipeline(ctx::VkContext,
                            raygen_spirv::Vector{UInt8},
                            miss_spirv::Vector{UInt8},
                            chit_spirvs::Vector{Vector{UInt8}};
                            anyhit_spirv::Union{Nothing, Vector{UInt8}}=nothing,
                            intersection_spirv::Union{Nothing, Vector{UInt8}}=nothing,
                            push_constant_size::Integer=8)
    dev = ctx.device
    rt_props = ctx.rt_pipeline_properties
    rt_props === nothing && throw(LavaError(
        "RT pipeline creation",
        "Ray tracing not supported on this device",
        "Ensure VK_KHR_ray_tracing_pipeline is available"))
    n_chits = length(chit_spirvs)
    n_chits >= 1 || throw(ArgumentError("create_rt_pipeline: at least one closest-hit shader required"))

    raygen_mod = create_shader_module(dev, raygen_spirv)
    check_validation_errors!("vkCreateShaderModule (raygen)", ctx)
    miss_mod = create_shader_module(dev, miss_spirv)
    check_validation_errors!("vkCreateShaderModule (miss)", ctx)
    shader_modules = [raygen_mod, miss_mod]
    chit_mods = VK.ShaderModule[]
    for (i, chit_spirv) in enumerate(chit_spirvs)
        m = create_shader_module(dev, chit_spirv)
        check_validation_errors!("vkCreateShaderModule (closest-hit $i)", ctx)
        push!(chit_mods, m)
        push!(shader_modules, m)
    end

    has_anyhit = anyhit_spirv !== nothing
    # An intersection shader makes every hit group procedural: the BLAS holds
    # AABBs, and the shader decides whether and where a ray hits inside one.
    has_isect = intersection_spirv !== nothing

    # Stages: 0 raygen, 1 miss, 2..1+N closest-hits, then the optional any-hit
    # and intersection.
    stages = VK.PipelineShaderStageCreateInfo[
        VK.PipelineShaderStageCreateInfo(
            VK.SHADER_STAGE_RAYGEN_BIT_KHR, raygen_mod, "main"),
        VK.PipelineShaderStageCreateInfo(
            VK.SHADER_STAGE_MISS_BIT_KHR, miss_mod, "main"),
    ]
    chit_stage_indices = Int[]
    for chit_mod in chit_mods
        push!(stages, VK.PipelineShaderStageCreateInfo(
            VK.SHADER_STAGE_CLOSEST_HIT_BIT_KHR, chit_mod, "main"))
        push!(chit_stage_indices, length(stages) - 1)   # 0-based
    end

    anyhit_stage_index = VK_SHADER_UNUSED_KHR
    if has_anyhit
        anyhit_mod = create_shader_module(dev, anyhit_spirv)
        check_validation_errors!("vkCreateShaderModule (any-hit)", ctx)
        push!(stages, VK.PipelineShaderStageCreateInfo(
            VK.SHADER_STAGE_ANY_HIT_BIT_KHR, anyhit_mod, "main"))
        push!(shader_modules, anyhit_mod)
        anyhit_stage_index = UInt32(length(stages) - 1)
    end

    isect_stage_index = VK_SHADER_UNUSED_KHR
    if has_isect
        isect_mod = create_shader_module(dev, intersection_spirv)
        check_validation_errors!("vkCreateShaderModule (intersection)", ctx)
        push!(stages, VK.PipelineShaderStageCreateInfo(
            VK.SHADER_STAGE_INTERSECTION_BIT_KHR, isect_mod, "main"))
        push!(shader_modules, isect_mod)
        isect_stage_index = UInt32(length(stages) - 1)
    end

    # All stage flags (for descriptor set and push constant visibility)
    all_stage_flags = VK.SHADER_STAGE_RAYGEN_BIT_KHR |
                      VK.SHADER_STAGE_CLOSEST_HIT_BIT_KHR |
                      VK.SHADER_STAGE_MISS_BIT_KHR
    if has_anyhit
        all_stage_flags |= VK.SHADER_STAGE_ANY_HIT_BIT_KHR
    end
    if has_isect
        all_stage_flags |= VK.SHADER_STAGE_INTERSECTION_BIT_KHR
    end

    # Groups: raygen (0), miss (1), then one hit group per closest-hit.
    groups = VK.RayTracingShaderGroupCreateInfoKHR[
        VK.RayTracingShaderGroupCreateInfoKHR(
            VK.RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR,
            UInt32(0),
            VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR,
        ),
        VK.RayTracingShaderGroupCreateInfoKHR(
            VK.RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR,
            UInt32(1),
            VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR,
        ),
    ]
    group_type = has_isect ?
        VK.RAY_TRACING_SHADER_GROUP_TYPE_PROCEDURAL_HIT_GROUP_KHR :
        VK.RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR
    for chit_idx in chit_stage_indices
        push!(groups, VK.RayTracingShaderGroupCreateInfoKHR(
            group_type,
            VK_SHADER_UNUSED_KHR,
            UInt32(chit_idx),
            anyhit_stage_index,
            isect_stage_index,
        ))
    end

    # Descriptor set layout: binding 0 = HWTLAS
    ds_layout = VK.DescriptorSetLayout(dev, [
        VK.DescriptorSetLayoutBinding(
            UInt32(0),
            VK.DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR,
            all_stage_flags;
            descriptor_count=1,
        ),
    ])

    # Pipeline layout: descriptor set + push constants
    push_ranges = if push_constant_size > 0
        [VK.PushConstantRange(
            all_stage_flags,
            UInt32(0),
            UInt32(push_constant_size),
        )]
    else
        VK.PushConstantRange[]
    end

    layout = VK.PipelineLayout(dev, [ds_layout], push_ranges)

    pipeline = createpipeline(ctx, VK.PipelineCreateFlag(0), "ray tracing", raygen_spirv) do flags
        rt_ci = VK.RayTracingPipelineCreateInfoKHR(
            stages, groups,
            UInt32(1),  # max_pipeline_ray_recursion_depth
            layout,
            Int32(-1);  # base_pipeline_index
            flags,
        )
        pipelines, code = @vk_checked ctx "vkCreateRayTracingPipelinesKHR" VK.create_ray_tracing_pipelines_khr(
            dev, [rt_ci]; pipeline_cache=ctx.pipeline_cache)
        pipelines[1], code
    end

    # SBT: raygen, miss, N hit groups; any-hit and intersection belong to the
    # hit groups, not separate entries.
    sbt_buf, raygen_region, miss_region, hit_region, callable_region =
        build_sbt(ctx, pipeline, rt_props, n_chits)

    return LavaRTPipeline(
        pipeline, layout, ds_layout,
        shader_modules,
        sbt_buf,
        raygen_region, miss_region, hit_region, callable_region,
        UInt32(push_constant_size),
        all_stage_flags,
        ctx,
    )
end

"""Cache binding-0 descriptors per TLAS/layout. The TLAS owns each pool through destruction."""
function get_rt_descriptor_set(pipeline::LavaRTPipeline, tlas::LavaTLAS)
    layout_handle = UInt64(pipeline.descriptor_set_layout.vks)
    cached = get(tlas.desc_sets, layout_handle, nothing)
    if cached !== nothing
        return cached[2]
    end

    desc_pool, desc_set = alloc_tlas_descriptor_set(pipeline.ctx,
                                                   pipeline.descriptor_set_layout, tlas.accel)
    tlas.desc_sets[layout_handle] = (desc_pool, desc_set)
    return desc_set
end

"""Bind, retain and timestamp one trace. Launch is a (width,height,depth) tuple or a GPU command."""
function emit_trace!(e::Emitter, pipeline::LavaRTPipeline, tlas::LavaTLAS,
                     push_bda::UInt64, launch, name::AbstractString="")
    cmd = e.cmd
    desc_set = get_rt_descriptor_set(pipeline, tlas)
    VK.cmd_bind_pipeline(cmd, VK.PIPELINE_BIND_POINT_RAY_TRACING_KHR, pipeline.pipeline)
    hold!(e, pipeline)
    VK.cmd_bind_descriptor_sets(cmd, VK.PIPELINE_BIND_POINT_RAY_TRACING_KHR,
        pipeline.pipeline_layout, UInt32(0), [desc_set], UInt32[])
    # TLAS ownership retains every BLAS; syncbuf! registers their storage uses.
    hold!(e, tlas)
    push_constants_bda!(cmd, pipeline.pipeline_layout, pipeline.stage_flags, push_bda)
    ts_slot = maybe_write_dispatch_start_timestamp!(e.ctx, cmd, name;
        stage=VK.PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR)
    tracecommand!(e, pipeline, launch)
    maybe_write_dispatch_end_timestamp!(e.ctx, cmd, ts_slot, e.ctx.cmd_pipeline_barrier_fptr;
        stage=VK.PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
        stage_mask=UInt32(VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR))
    emitted!(e, name)
    return nothing
end

function tracecommand!(e::Emitter, p::LavaRTPipeline, dims::NTuple{3,Integer})
    VK.cmd_trace_rays_khr(e.cmd, p.raygen_region, p.miss_region, p.hit_region,
                        p.callable_region, UInt32(dims[1]), UInt32(dims[2]), UInt32(dims[3]))
    return nothing
end

function tracecommand!(e::Emitter, p::LavaRTPipeline, indirect::LavaArray{UInt32,1})
    hold!(e, indirect)
    # Includes the command view's element offset; no count readback.
    VK.cmd_trace_rays_indirect_khr(e.cmd, p.raygen_region, p.miss_region, p.hit_region,
                                 p.callable_region, bda_address(indirect))
    return nothing
end

"""Build the SBT for raygen, miss and `n_hit_groups` hit groups, in the group order
of `create_rt_pipeline`. An instance picks its hit group with
`instanceShaderBindingTableRecordOffset` (0..N-1).
"""
function build_sbt(ctx::VkContext, pipeline::VK.Pipeline, rt_props::RTPipelineProperties,
                    n_hit_groups::Int)
    n_hit_groups >= 1 || throw(ArgumentError("build_sbt: n_hit_groups must be >= 1"))
    dev = ctx.device
    handle_size = rt_props.shader_group_handle_size
    base_align = rt_props.shader_group_base_alignment

    # Entry strides are multiples of shader_group_handle_alignment; regions
    # start at shader_group_base_alignment.
    handle_size_aligned = align_up(handle_size, rt_props.shader_group_handle_alignment)

    # Pull every pipeline group handle: raygen + miss + N hit groups = 2 + N
    n_groups = 2 + n_hit_groups
    total_handle_data = handle_size * n_groups
    handles = Vector{UInt8}(undef, total_handle_data)
    unwrap(VK.get_ray_tracing_shader_group_handles_khr(
        dev, pipeline, UInt32(0), UInt32(n_groups), total_handle_data, Ptr{Nothing}(pointer(handles))))

    raygen_stride = align_up(handle_size_aligned, base_align)
    miss_stride   = handle_size_aligned
    hit_stride    = handle_size_aligned

    raygen_size = raygen_stride                      # exactly 1 raygen entry
    miss_size   = align_up(miss_stride, base_align)  # 1 miss entry, region aligned
    # N hit entries, then round the region up to base_align.
    hit_size_raw = hit_stride * n_hit_groups
    hit_size     = align_up(hit_size_raw, base_align)

    total_sbt_size = align_up(raygen_size, base_align) +
                     align_up(miss_size,   base_align) +
                     align_up(hit_size,    base_align)

    sbt_buf = vk_alloc(ctx.default_bq, total_sbt_size;
        extra_usage=UInt32(VK.BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR))

    sbt_data = zeros(UInt8, total_sbt_size)
    offset = 0

    # Raygen (group 0)
    raygen_offset = offset
    copyto!(sbt_data, offset + 1, handles, 1, handle_size)
    offset = raygen_offset + align_up(raygen_size, base_align)

    # Miss (group 1)
    miss_offset = offset
    copyto!(sbt_data, offset + 1, handles, handle_size + 1, handle_size)
    offset = miss_offset + align_up(miss_size, base_align)

    # Hit groups (groups 2..1+N). Pack one handle every `hit_stride` bytes;
    # an SBT entry is indexed by `instanceShaderBindingTableRecordOffset`.
    hit_offset = offset
    for i in 1:n_hit_groups
        src_start = (1 + i) * handle_size + 1     # 1-based start of group (1+i)'s handle
        dst_start = hit_offset + (i - 1) * hit_stride + 1
        copyto!(sbt_data, dst_start, handles, src_start, handle_size)
    end

    upload!(sbt_buf, sbt_data)

    raygen_region = VK.StridedDeviceAddressRegionKHR(
        sbt_buf.address + UInt64(raygen_offset),
        UInt64(raygen_stride),
        UInt64(raygen_size),
    )
    miss_region = VK.StridedDeviceAddressRegionKHR(
        sbt_buf.address + UInt64(miss_offset),
        UInt64(miss_stride),
        UInt64(miss_size),
    )
    hit_region = VK.StridedDeviceAddressRegionKHR(
        sbt_buf.address + UInt64(hit_offset),
        UInt64(hit_stride),
        UInt64(hit_size),
    )
    callable_region = VK.StridedDeviceAddressRegionKHR(
        UInt64(0), UInt64(0), UInt64(0),
    )

    return sbt_buf, raygen_region, miss_region, hit_region, callable_region
end

align_up(x, align) = ((x + align - 1) ÷ align) * align
