# Ray-tracing pipelines from SPIR-V, their shader binding tables, and trace dispatch.

const VK_SHADER_UNUSED_KHR = ~UInt32(0)  # 0xFFFFFFFF

"""
    LavaRTPipeline

A compiled ray-tracing pipeline with its shader binding table. Subtypes
`CompiledRTPipeline` (`runtime/coretypes.jl`) because it holds the `LavaDevice`
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
    ctx::LavaDevice
end

"""
    create_rt_pipeline(ctx, raygen_spirv, miss_spirv, chit_spirvs;
                       anyhit_spirv = nothing, push_constant_size = 8) -> LavaRTPipeline

A ray-tracing pipeline from SPIR-V. Groups: 0 raygen, 1 miss, 2..1+N one triangle
hit group per closest-hit, all sharing the optional any-hit. Set 0, binding 0 is
the TLAS; the push constant is the argument buffer's device address.
"""
function create_rt_pipeline(ctx::LavaDevice, raygen_spirv::Vector{UInt8}, miss_spirv::Vector{UInt8},
                            chit_spirvs::Vector{Vector{UInt8}};
                            anyhit_spirv::Union{Nothing,Vector{UInt8}} = nothing,
                            push_constant_size::Integer = 8)
    dev = ctx.device
    rt_props = ctx.rt_pipeline_properties
    rt_props === nothing && throw(LavaError("RT pipeline creation",
        "Ray tracing not supported on this device", "Ensure VK_KHR_ray_tracing_pipeline is available"))
    n_chits = length(chit_spirvs)
    n_chits >= 1 || throw(ArgumentError("create_rt_pipeline: at least one closest-hit shader required"))
    # Stages in group order: raygen, miss, the closest-hits, the optional any-hit.
    specs = [VK.SHADER_STAGE_RAYGEN_BIT_KHR => raygen_spirv, VK.SHADER_STAGE_MISS_BIT_KHR => miss_spirv,
             (VK.SHADER_STAGE_CLOSEST_HIT_BIT_KHR => c for c in chit_spirvs)...]
    anyhit_spirv === nothing || push!(specs, VK.SHADER_STAGE_ANY_HIT_BIT_KHR => anyhit_spirv)
    shader_modules = map(specs) do (stage, spirv)
        m = create_shader_module(dev, spirv)
        check_validation_errors!("vkCreateShaderModule ($stage)", ctx)
        m
    end
    stages = [VK.PipelineShaderStageCreateInfo(st, m, "main") for ((st, _), m) in zip(specs, shader_modules)]
    all_stage_flags = reduce(|, first.(specs))
    anyhit = anyhit_spirv === nothing ? VK_SHADER_UNUSED_KHR : UInt32(length(stages) - 1)
    group(type, general, chit = VK_SHADER_UNUSED_KHR, any = VK_SHADER_UNUSED_KHR) =
        VK.RayTracingShaderGroupCreateInfoKHR(type, general, chit, any, VK_SHADER_UNUSED_KHR)
    groups = [group(VK.RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR, UInt32(0)),
              group(VK.RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR, UInt32(1)),
              (group(VK.RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR, VK_SHADER_UNUSED_KHR,
                     UInt32(1 + i), anyhit) for i in 1:n_chits)...]

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
`instanceShaderBindingTableRecordOffset` (0..N-1). Entry strides are multiples of
the handle alignment, regions start at the base alignment, and the raygen
region's stride equals its size.
"""
function build_sbt(ctx::LavaDevice, pipeline::VK.Pipeline, rt_props::RTPipelineProperties,
                   n_hit_groups::Int)
    hsize = Int(rt_props.shader_group_handle_size)
    base = Int(rt_props.shader_group_base_alignment)
    stride = alignup(hsize, Int(rt_props.shader_group_handle_alignment))
    ngroups = 2 + n_hit_groups
    handles = Vector{UInt8}(undef, hsize * ngroups)
    unwrap(VK.get_ray_tracing_shader_group_handles_khr(ctx.device, pipeline, UInt32(0),
        UInt32(ngroups), length(handles), Ptr{Nothing}(pointer(handles))))
    # (entries, stride) of the raygen, miss and hit regions.
    regions = ((1, alignup(stride, base)), (1, stride), (n_hit_groups, stride))
    sizes = [alignup(n * s, base) for (n, s) in regions]
    offsets = cumsum([0; sizes[1:end-1]])
    sbt_buf = vk_alloc(ctx.default_bq, sum(sizes);
        extra_usage = UInt32(VK.BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR))
    data = zeros(UInt8, sum(sizes))
    g = 0
    for ((n, s), off) in zip(regions, offsets), k in 0:n-1
        copyto!(data, off + k * s + 1, handles, g * hsize + 1, hsize)
        g += 1
    end
    upload!(sbt_buf, data)
    rr, mr, hr = (VK.StridedDeviceAddressRegionKHR(sbt_buf.address + UInt64(off), UInt64(s), UInt64(size))
                  for ((_, s), off, size) in zip(regions, offsets, sizes))
    return sbt_buf, rr, mr, hr, VK.StridedDeviceAddressRegionKHR(UInt64(0), UInt64(0), UInt64(0))
end
