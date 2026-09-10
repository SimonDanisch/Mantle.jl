# Ray tracing pipeline creation and dispatch for Lava.jl
#
# Creates VkRayTracingPipelineKHR from SPIR-V shader modules.
# Manages shader binding table (SBT) layout and dispatch.

const VK_SHADER_UNUSED_KHR = ~UInt32(0)  # 0xFFFFFFFF

"""
    LavaRTPipeline

A compiled ray tracing pipeline with shader binding table.

Subtypes `CompiledRTPipeline` (`runtime/coretypes.jl`) so `DeviceCaches` can
cache one: this type holds the `VkContext` that holds those caches, so the
concrete name cannot appear in the cache's field type.
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
    # Push constant size
    push_constant_size::UInt32
    # Stage flags for push constants and descriptor sets
    stage_flags::VK.ShaderStageFlag
    # Context this pipeline was built against (for descriptor-set creation,
    # SBT uploads, etc. — no global lookups needed).
    ctx::VkContext
end

"""
    create_rt_pipeline(raygen_spirv, miss_spirv, chit_spirvs;
                       push_constant_size=8) -> LavaRTPipeline

Create a ray tracing pipeline from raygen + miss + N closest-hit SPIR-V
binaries.

Layout:
  - Group 0:                raygen (GENERAL)
  - Group 1:                miss (GENERAL)
  - Group 2 .. 1+N:         closest-hit hit groups (TRIANGLES_HIT_GROUP)
  - Optional any-hit:       shared across every hit group when supplied
  - Descriptor set 0/0:     AccelerationStructure (HWTLAS)
  - Push constant:          BDA pointer (8 bytes by default)
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
                            push_constant_size::Integer=8)
    dev = ctx.device
    rt_props = ctx.rt_pipeline_properties
    rt_props === nothing && throw(LavaError(
        "RT pipeline creation",
        "Ray tracing not supported on this device",
        "Ensure VK_KHR_ray_tracing_pipeline is available"))
    n_chits = length(chit_spirvs)
    n_chits >= 1 || throw(ArgumentError("create_rt_pipeline: at least one closest-hit shader required"))

    # Create shader modules
    raygen_mod = create_shader_module(dev, raygen_spirv)
    check_validation_errors!("vkCreateShaderModule (raygen)")
    miss_mod = create_shader_module(dev, miss_spirv)
    check_validation_errors!("vkCreateShaderModule (miss)")
    shader_modules = [raygen_mod, miss_mod]
    chit_mods = VK.ShaderModule[]
    for (i, chit_spirv) in enumerate(chit_spirvs)
        m = create_shader_module(dev, chit_spirv)
        check_validation_errors!("vkCreateShaderModule (closest-hit $i)")
        push!(chit_mods, m)
        push!(shader_modules, m)
    end

    has_anyhit = anyhit_spirv !== nothing

    # Shader stages: 0=raygen, 1=miss, 2..1+N=closest-hit per chit_spirvs,
    # then optional anyhit at the next index.
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
        check_validation_errors!("vkCreateShaderModule (any-hit)")
        push!(stages, VK.PipelineShaderStageCreateInfo(
            VK.SHADER_STAGE_ANY_HIT_BIT_KHR, anyhit_mod, "main"))
        push!(shader_modules, anyhit_mod)
        anyhit_stage_index = UInt32(length(stages) - 1)
    end

    # All stage flags (for descriptor set and push constant visibility)
    all_stage_flags = VK.SHADER_STAGE_RAYGEN_BIT_KHR |
                      VK.SHADER_STAGE_CLOSEST_HIT_BIT_KHR |
                      VK.SHADER_STAGE_MISS_BIT_KHR
    if has_anyhit
        all_stage_flags |= VK.SHADER_STAGE_ANY_HIT_BIT_KHR
    end

    # Shader groups: raygen (0), miss (1), then one triangles-hit-group per chit.
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
    for chit_idx in chit_stage_indices
        push!(groups, VK.RayTracingShaderGroupCreateInfoKHR(
            VK.RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR,
            VK_SHADER_UNUSED_KHR,
            UInt32(chit_idx),
            anyhit_stage_index,
            VK_SHADER_UNUSED_KHR,
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

    # Create RT pipeline
    # Same no-compile verification the compute path uses: with PIPELINE_NO_COMPILE
    # set the driver refuses to build a binary rather than compiling one, so a run
    # that completes provably did zero driver compilation. RT is the path that
    # matters most for startup — a RayDemo scene is mostly RT pipelines.
    rt_flags = VK.PipelineCreateFlag(0)
    if PIPELINE_NO_COMPILE[]
        rt_flags |= VK.PipelineCreateFlag(VK.PIPELINE_CREATE_FAIL_ON_PIPELINE_COMPILE_REQUIRED_BIT) |
                    VK.PipelineCreateFlag(VK.PIPELINE_CREATE_EARLY_RETURN_ON_FAILURE_BIT)
    end
    rt_ci = VK.RayTracingPipelineCreateInfoKHR(
        stages, groups,
        UInt32(1),  # max_pipeline_ray_recursion_depth
        layout,
        Int32(-1);  # base_pipeline_index
        flags = rt_flags,
    )

    pipelines, _ = @vk_checked "vkCreateRayTracingPipelinesKHR" VK.create_ray_tracing_pipelines_khr(
        dev, [rt_ci]; pipeline_cache=ctx.pipeline_cache)
    pipeline = pipelines[1]

    # Build SBT (raygen + miss + N hit groups; any-hit is shared inside each
    # triangles-hit-group, not a separate SBT entry).
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

"""
    get_rt_descriptor_set(pipeline, tlas) -> VK.DescriptorSet

Look up (or lazily allocate) the RT descriptor set that binds `tlas.accel` for
the layout used by `pipeline`.  The pool + set are stored on `tlas.desc_sets`,
keyed by descriptor-set-layout Vulkan handle, so their lifetime tracks the
LavaTLAS exactly: `destroy_now!(tlas)` destroys the pool, releasing the set.

The previous implementation used a global cache keyed by `objectid(tlas)`,
which broke after a freed LavaTLAS was GC'd and a new LavaTLAS reused that
objectid: the cache returned a descriptor set bound to the destroyed
VkAccelerationStructureKHR.  Per-HWTLAS storage eliminates that hazard
entirely — no global cache, no objectid keying, no eviction policy.
"""
function get_rt_descriptor_set(pipeline::LavaRTPipeline, tlas::LavaTLAS)
    layout_handle = UInt64(pipeline.descriptor_set_layout.vks)
    cached = get(tlas.desc_sets, layout_handle, nothing)
    if cached !== nothing
        return cached[2]
    end

    dev = pipeline.ctx.device
    desc_pool = VK.DescriptorPool(dev, UInt32(1), [
        VK.DescriptorPoolSize(
            VK.DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, UInt32(1)),
    ])
    desc_sets = @vk_checked "vkAllocateDescriptorSets (RT)" VK.allocate_descriptor_sets(dev,
        VK.DescriptorSetAllocateInfo(desc_pool, [pipeline.descriptor_set_layout]))
    desc_set = desc_sets[1]

    as_write = VK.WriteDescriptorSetAccelerationStructureKHR([tlas.accel])
    write_ds = VK.WriteDescriptorSet(
        desc_set, UInt32(0), UInt32(0),
        VK.DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR,
        VK.DescriptorImageInfo[],
        VK.DescriptorBufferInfo[],
        VK.BufferView[];
        descriptor_count=UInt32(1),
        next=as_write,
    )
    VK.update_descriptor_sets(dev, [write_ds], [])

    tlas.desc_sets[layout_handle] = (desc_pool, desc_set)
    return desc_set
end

"""
    emit_trace!(emitter, pipeline, tlas, push_bda, width, height, depth, name)
    emit_trace_indirect!(emitter, pipeline, tlas, push_bda, indirect, name)

Write one ray-tracing dispatch, and nothing else. The descriptor set comes from
`get_rt_descriptor_set`, which has always cached per (TLAS, layout) on the TLAS
itself — so unlike the compute HWTLAS path there was never a per-dispatch pool
here. The unmodelled forms (`trace_rays!`, `trace_rays_indirect!`) write these
into a one-shot of their own; a plan's `trace!` writes them into its recording.
"""
function emit_trace!(e::Emitter, pipeline::LavaRTPipeline, tlas::LavaTLAS,
                     push_bda::UInt64, width::Integer, height::Integer,
                     depth::Integer, name::AbstractString = "")
    cmd = e.cmd
    desc_set = get_rt_descriptor_set(pipeline, tlas)
    VK.cmd_bind_pipeline(cmd, VK.PIPELINE_BIND_POINT_RAY_TRACING_KHR, pipeline.pipeline)
    hold!(e, pipeline)
    VK.cmd_bind_descriptor_sets(cmd, VK.PIPELINE_BIND_POINT_RAY_TRACING_KHR,
        pipeline.pipeline_layout, UInt32(0), [desc_set], UInt32[])
    # One hold covers BOTH levels: the top level references every bottom level it
    # instances, so keeping it keeps them, and `syncbuf!(::LavaTLAS)` records
    # each level's storage for ordering. `pintrace!` was a walk from outside
    # doing the same thing to a list it could not see the end of.
    hold!(e, tlas)
    push_constants_bda!(cmd, pipeline.pipeline_layout, pipeline.stage_flags, push_bda)
    # Same optional GPU timestamps as the compute paths.  Without these the
    # profiler is blind to hardware ray tracing — on an hw_accel=true frame
    # that is where nearly all the GPU time goes, so a report built only
    # from `cmd_dispatch` accounts for a small fraction of the frame and
    # invites the wrong conclusion about what is slow.
    ts_slot = maybe_write_dispatch_start_timestamp!(e.ctx, cmd, name;
                                             stage = VK.PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR)
    VK.cmd_trace_rays_khr(cmd,
        pipeline.raygen_region, pipeline.miss_region,
        pipeline.hit_region, pipeline.callable_region,
        UInt32(width), UInt32(height), UInt32(depth))
    maybe_write_dispatch_end_timestamp!(e.ctx, cmd, ts_slot, e.ctx.cmd_pipeline_barrier_fptr;
        stage = VK.PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
        stage_mask = UInt32(VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR))
    emitted!(e, name)
    return nothing
end

function emit_trace_indirect!(e::Emitter, pipeline::LavaRTPipeline, tlas::LavaTLAS,
                              push_bda::UInt64, indirect::LavaArray{UInt32,1},
                              name::AbstractString = "")
    cmd = e.cmd
    desc_set = get_rt_descriptor_set(pipeline, tlas)
    VK.cmd_bind_pipeline(cmd, VK.PIPELINE_BIND_POINT_RAY_TRACING_KHR, pipeline.pipeline)
    hold!(e, pipeline)
    VK.cmd_bind_descriptor_sets(cmd, VK.PIPELINE_BIND_POINT_RAY_TRACING_KHR,
        pipeline.pipeline_layout, UInt32(0), [desc_set], UInt32[])
    # One hold covers BOTH levels: the top level references every bottom level it
    # instances, so keeping it keeps them, and `syncbuf!(::LavaTLAS)` records
    # each level's storage for ordering. `pintrace!` was a walk from outside
    # doing the same thing to a list it could not see the end of.
    hold!(e, tlas)
    push_constants_bda!(cmd, pipeline.pipeline_layout, pipeline.stage_flags, push_bda)
    # bda_address(indirect) includes the view's element offset, so the address
    # passed to Vulkan points exactly at the 3-UInt32 command.
    ts_slot = maybe_write_dispatch_start_timestamp!(e.ctx, cmd, name;
                                             stage = VK.PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR)
    VK.cmd_trace_rays_indirect_khr(cmd,
        pipeline.raygen_region, pipeline.miss_region,
        pipeline.hit_region, pipeline.callable_region,
        bda_address(indirect))
    maybe_write_dispatch_end_timestamp!(e.ctx, cmd, ts_slot, e.ctx.cmd_pipeline_barrier_fptr;
        stage = VK.PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
        stage_mask = UInt32(VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR))
    # The ray count is READ BY THE DEVICE from here.
    hold!(e, indirect)
    emitted!(e, name)
    return nothing
end

# ── Internal helpers ──

function create_shader_module(dev, spirv_bytes::Vector{UInt8})
    @assert length(spirv_bytes) % 4 == 0 "SPIR-V must be 4-byte aligned"
    code_u32 = reinterpret(UInt32, spirv_bytes)
    return VK.ShaderModule(dev, length(spirv_bytes), code_u32)
end

"""Build the SBT for raygen + miss + `n_hit_groups` triangles-hit-groups.

Pipeline group ordering matches `create_rt_pipeline`:
  - group 0           : raygen
  - group 1           : miss
  - groups 2..1+N     : the N hit groups, in chit order.

Per-instance dispatch picks an SBT hit-group entry by
`instanceShaderBindingTableRecordOffset` (0..N-1).
"""
function build_sbt(ctx::VkContext, pipeline::VK.Pipeline, rt_props::RTPipelineProperties,
                    n_hit_groups::Int)
    n_hit_groups >= 1 || throw(ArgumentError("build_sbt: n_hit_groups must be >= 1"))
    dev = ctx.device
    handle_size = rt_props.shader_group_handle_size
    base_align = rt_props.shader_group_base_alignment

    # Each group entry must be aligned to shader_group_handle_alignment
    # but stride must be multiple of shader_group_handle_alignment
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
