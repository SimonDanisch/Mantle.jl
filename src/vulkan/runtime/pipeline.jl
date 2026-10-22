"""Create a native shader module from aligned SPIR-V bytes."""
function create_shader_module(dev::VK.Device, bytes::Vector{UInt8})
    @assert length(bytes) % 4 == 0 "SPIR-V must be 4-byte aligned"
    return VK.ShaderModule(dev, length(bytes), reinterpret(UInt32, bytes))
end

# Compute pipeline creation and the per-device pipeline cache. On Windows,
# vkCreateComputePipelines runs on a thread with a 256 MB stack: AMD's
# proprietary driver overflows its compiler stack on some SPIR-V.

const LARGE_STACK_PIPELINE = Sys.iswindows()
const LARGE_STACK_SIZE = 256 * 1024 * 1024  # 256 MB

if LARGE_STACK_PIPELINE

struct _VkCreatePipelineArgs
    device::Ptr{Cvoid}
    PIPELINE_CACHE::Ptr{Cvoid}
    create_info_count::UInt32
    _pad::UInt32
    p_create_infos::Ptr{Cvoid}
    p_allocator::Ptr{Cvoid}
    p_pipelines::Ptr{Cvoid}
    result::Int32
    _pad2::Int32
end

# Runs on a thread Julia did not create; entering adopts it into the runtime,
# which allocates and can start a GC (hence the GC-safe wait in
# `create_compute_pipeline_large_stack`). The driver call is GC-safe so a long
# compile does not block collections on other threads.
function vk_pipeline_thread_callback(args_ptr::Ptr{Cvoid})::UInt32
    args = unsafe_load(Ptr{_VkCreatePipelineArgs}(args_ptr))
    result = @ccall gc_safe = true "vulkan-1".vkCreateComputePipelines(
        args.device::Ptr{Cvoid}, args.PIPELINE_CACHE::Ptr{Cvoid}, args.create_info_count::UInt32,
        args.p_create_infos::Ptr{Cvoid}, args.p_allocator::Ptr{Cvoid}, args.p_pipelines::Ptr{Cvoid})::Int32
    unsafe_store!(Ptr{Int32}(args_ptr + 48), result)
    return UInt32(0)
end

end # if LARGE_STACK_PIPELINE

const PIPELINE_THREAD_CFUNC = Ref{Ptr{Nothing}}(C_NULL)

if LARGE_STACK_PIPELINE
function init_pipeline_thread!()
    PIPELINE_THREAD_CFUNC[] = @cfunction(vk_pipeline_thread_callback, UInt32, (Ptr{Cvoid},))
end
else
init_pipeline_thread!() = nothing
end

function create_compute_pipeline_large_stack(ctx::VkContext, device::Ptr{Cvoid},
                                               pipeline_cache::Ptr{Cvoid},
                                               create_info_ptr::Ptr{Cvoid},
                                               p_pipelines::Ptr{Cvoid})
    args = Ref(_VkCreatePipelineArgs(
        device, pipeline_cache, UInt32(1), UInt32(0),
        create_info_ptr, C_NULL, p_pipelines,
        Int32(0), Int32(0),
    ))
    GC.@preserve args begin
        args_ptr = Ptr{Cvoid}(pointer_from_objref(args))
        thread_id = Ref(UInt32(0))
        handle = ccall((:CreateThread, "kernel32"), Ptr{Cvoid},
            (Ptr{Cvoid}, Csize_t, Ptr{Cvoid}, Ptr{Cvoid}, UInt32, Ptr{UInt32}),
            C_NULL, LARGE_STACK_SIZE, PIPELINE_THREAD_CFUNC[], args_ptr,
            UInt32(0), thread_id)
        handle == C_NULL && error("CreateThread failed for vkCreateComputePipelines")
        # GC-safe wait: the adopted thread can start a collection, which waits
        # for every thread to reach a safepoint, and a plain `ccall` never does,
        # so the two deadlock until the timeout. 600 s bounds a driver that
        # really hangs; the thread is then terminated, not abandoned, because
        # it would later write into `args`, freed on return.
        wait_result = @ccall gc_safe = true "kernel32".WaitForSingleObject(
            handle::Ptr{Cvoid}, UInt32(600_000)::UInt32)::UInt32
        if wait_result == 0x00000102  # WAIT_TIMEOUT
            ccall((:TerminateThread, "kernel32"), Cint,
                (Ptr{Cvoid}, UInt32), handle, UInt32(1))
            ccall((:CloseHandle, "kernel32"), Cint, (Ptr{Cvoid},), handle)
            # Best effort: the driver is in an unknown state.
            mark_device_lost!(ctx)
            error("vkCreateComputePipelines timed out after 600s — AMD Windows " *
                  "driver shader compiler hung. Device marked DEVICE_LOST; " *
                  "call reset_device!() to recover.")
        end
        ccall((:CloseHandle, "kernel32"), Cint, (Ptr{Cvoid},), handle)
        wait_result != 0 && error("WaitForSingleObject failed: $wait_result")
        return args[].result
    end
end

# Success-class (positive) VkResult, so `result != 0` alone would report it as
# a generic failure.
const VK_PIPELINE_COMPILE_REQUIRED = 1000297000

# Forbid the driver to compile; EARLY_RETURN_ON_FAILURE alongside, as the spec pairs them.
const NO_COMPILE_FLAGS = VK.PipelineCreateFlag(VK.PIPELINE_CREATE_FAIL_ON_PIPELINE_COMPILE_REQUIRED_BIT) |
                         VK.PipelineCreateFlag(VK.PIPELINE_CREATE_EARLY_RETURN_ON_FAILURE_BIT)

"""
    createpipeline(create, ctx, flags, name, spirv) -> pipeline

`create(flags)` returns `(pipeline, VkResult)`. Inside
[`no_pipeline_compilation`](@ref) the driver is first forbidden to compile; a
refusal is counted in `ctx.diag`, `name` and the SPIR-V's hash are recorded, and
the pipeline is created with `flags`, so the workload finishes and reports
every miss.
"""
function createpipeline(create, ctx::VkContext, flags::VK.PipelineCreateFlag, name, spirv::Vector{UInt8})
    d = ctx.diag
    if d.no_pipeline_compile
        pipeline, code = create(flags | NO_COMPILE_FLAGS)
        code == VK.PIPELINE_COMPILE_REQUIRED || return pipeline
        d.pipeline_compiles_refused += 1
        # `spirv_content_hash`, not `Base.hash`, which samples large vectors.
        push!(d.pipeline_compile_misses, string(name, " spirv=", string(spirv_content_hash(spirv), base=16),
                                                " (", length(spirv), " bytes)"))
    end
    return first(create(flags))
end

"""
    no_pipeline_compilation(f; ctx = vk_context()) -> f()

Run `f` with `ctx`'s driver forbidden to compile pipelines, which proves the
`VkPipelineCache` seeded from disk serves them (a fast run alone could be the
in-memory caches). Clears `ctx`'s kernel caches and its refusal counters first.
A refusal does not throw: [`createpipeline`](@ref) records it and compiles.

    no_pipeline_compilation() do
        run_the_workload()
    end
    vk_context().diag.pipeline_compiles_refused == 0    # nothing needed the driver's compiler
"""
function no_pipeline_compilation(f; ctx::VkContext = vk_context())
    d = ctx.diag
    old = d.no_pipeline_compile
    clear_kernel_cache!(ctx)
    d.pipeline_compiles_refused = 0
    empty!(d.pipeline_compile_misses)
    d.no_pipeline_compile = true
    try
        return f()
    finally
        d.no_pipeline_compile = old
    end
end

const MAX_PIPELINE_CACHE_SIZE = Ref(1024)

# The device's default subgroup width, not the min/max it can be pinned to.
# Cached per device: it differs between GPUs (64 on RDNA 3.5), and another
# device's value silently mistiles every kernel sized by it.
function device_subgroup_size(ctx::VkContext)
    # 0 means not yet queried.
    ctx.caches.subgroup_size != 0 && return ctx.caches.subgroup_size
    props = VK.get_physical_device_properties_2(ctx.physical_device,
                                                    VK.PhysicalDeviceSubgroupProperties)
    ctx.caches.subgroup_size = Int(props.next.subgroup_size)
end

"""
    max_workgroup_dims(ctx) -> NTuple{3,Int}

The device's `maxComputeWorkGroupSize`: workitems per workgroup along each axis,
queried once. `workgroup_limit(ctx)` bounds their product.
"""
function max_workgroup_dims(ctx::VkContext)
    d = ctx.caches.wg_size_dims
    d[1] != 0 && return d
    s = VK.get_physical_device_properties(ctx.physical_device).limits.max_compute_work_group_size
    ctx.caches.wg_size_dims = (Int(s[1]), Int(s[2]), Int(s[3]))
end

"""
    subgroup_size_control(ctx) -> SubgroupSizeControl

The device's `VkPhysicalDeviceSubgroupSizeControlProperties`, queried once.
`compute` says whether a compute pipeline may pin its own subgroup size.
"""
function subgroup_size_control(ctx::VkContext)
    c = ctx.caches.subgroup_control
    c === nothing || return c
    p = VK.get_physical_device_properties_2(
            ctx.physical_device, VK.PhysicalDeviceSubgroupSizeControlProperties).next
    ctx.caches.subgroup_control = SubgroupSizeControl(
        Int(p.min_subgroup_size), Int(p.max_subgroup_size),
        (UInt32(p.required_subgroup_size_stages) &
         UInt32(VK.SHADER_STAGE_COMPUTE_BIT)) != 0)
end

"""Whether a compute pipeline on this device can be pinned to `n` lanes per subgroup."""
function can_require_subgroup_size(ctx::VkContext, n::Integer)
    c = subgroup_size_control(ctx)
    c.compute && c.min <= n <= c.max
end

# Whether the module declares `cap` via OpCapability. Capabilities precede
# OpMemoryModel in a valid module, so the scan stops there.
function spirv_declares_capability(spirv::Vector{UInt8}, cap::UInt32)
    length(spirv) >= 20 && length(spirv) % 4 == 0 || return false
    w = reinterpret(UInt32, spirv)
    w[1] == 0x07230203 || return false          # magic, matching our byte order
    i = 6                                       # first word after the 5-word header
    @inbounds while i <= length(w)
        count = w[i] >> 16
        count == 0 && break                     # malformed; refuse to loop forever
        opcode = UInt16(w[i] & 0xffff)
        opcode == Op.OpMemoryModel && break      # past the capability section
        if opcode == Op.OpCapability && count >= 2 && i + 1 <= length(w)
            w[i+1] == cap && return true
        end
        i += count
    end
    return false
end

"""
    spirv_local_size(spirv) -> Int

Threads per workgroup from the module's `OpExecutionMode LocalSize`, or `0`
when not stated literally (e.g. `OpExecutionModeId` with specialisation
constants). Callers must treat `0` as unknown, not small.
"""
function spirv_local_size(spirv::Vector{UInt8})
    length(spirv) >= 20 && length(spirv) % 4 == 0 || return 0
    w = reinterpret(UInt32, spirv)
    w[1] == 0x07230203 || return 0
    i = 6
    @inbounds while i <= length(w)
        count = w[i] >> 16
        count == 0 && break
        opcode = UInt16(w[i] & 0xffff)
        # Past the debug/annotation sections there are no more execution modes.
        opcode == Op.OpTypeVoid && break
        # OpExecutionMode <entry> LocalSize x y z: mode 17, 6 words.
        if opcode == Op.OpExecutionMode && count >= 6 && i + 5 <= length(w) &&
           w[i+2] == UInt32(17)
            return Int(w[i+3]) * Int(w[i+4]) * Int(w[i+5])
        end
        i += count
    end
    return 0
end

"""
    get_compute_pipeline(ctx, spirv_bytes::Vector{UInt8}, entry_name::String;
                         push_constant_size::Integer=8,
                         needs_tlas_descriptor::Bool=false) -> LavaComputePipeline

Get or create `ctx`'s compute pipeline for the SPIR-V, cached by content and
layout. A cooperative-matrix module is pinned to `COOPMAT_SUBGROUP` lanes where
the device's default differs and it allows pinning.
"""
function get_compute_pipeline(ctx::VkContext, spirv_bytes::Vector{UInt8}, entry_name::String;
                               push_constant_size::Integer=8,
                               needs_tlas_descriptor::Bool=false)
    cache_key = hash((spirv_content_hash(spirv_bytes), length(spirv_bytes),
                      entry_name, push_constant_size, needs_tlas_descriptor))
    cached = get(ctx.caches.pipelines, cache_key, nothing)
    if cached !== nothing
        return cached
    end

    dev = ctx.device

    shader_mod = create_shader_module(dev, spirv_bytes)
    check_validation_errors!("vkCreateShaderModule (compute)", ctx)

    push_ranges = if push_constant_size > 0
        [VK.PushConstantRange(
            VK.SHADER_STAGE_COMPUTE_BIT,
            UInt32(0),
            UInt32(push_constant_size)
        )]
    else
        VK.PushConstantRange[]
    end

    # Descriptor set layout: binding 0 = HWTLAS (only when needs_tlas_descriptor)
    ds_layout = nothing
    ds_layouts = VK.DescriptorSetLayout[]
    if needs_tlas_descriptor
        ds_layout = VK.DescriptorSetLayout(
            dev,
            [VK.DescriptorSetLayoutBinding(
                UInt32(0),
                VK.DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR,
                VK.SHADER_STAGE_COMPUTE_BIT;
                descriptor_count = UInt32(1)
            )]
        )
        push!(ds_layouts, ds_layout)
    end

    layout = VK.PipelineLayout(dev, ds_layouts, push_ranges)

    # Cooperative-matrix kernels index subgroups as `tid ÷ COOPMAT_SUBGROUP`, so
    # on a device with another default width the pipeline is pinned to
    # COOPMAT_SUBGROUP. That is a function of (SPIR-V, device), so `cache_key`
    # need not name it. A device already that wide gets an untouched create-info.
    stage_next = C_NULL
    if spirv_declares_capability(spirv_bytes, Cap.CooperativeMatrixKHR) &&
       device_subgroup_size(ctx) != COOPMAT_SUBGROUP
        if can_require_subgroup_size(ctx, COOPMAT_SUBGROUP)
            stage_next = VK.PipelineShaderStageRequiredSubgroupSizeCreateInfo(
                UInt32(COOPMAT_SUBGROUP))
        else
            # Cannot pin. Only a workgroup of more than one subgroup reads wrong
            # fragments (with one, `tid ÷ 32` is 0 for every lane), so warn
            # rather than refuse: lavapipe has coopmat, 8-lane subgroups and no
            # pinning, and runs valid single-subgroup kernels. Routing gates such
            # as `coopmat_gemm_available` check `can_require_subgroup_size`
            # themselves. A workgroup size of 0 (not literal) counts as unsafe.
            wg = spirv_local_size(spirv_bytes)
            sg = device_subgroup_size(ctx)
            if !(0 < wg <= sg) && !ctx.caches.coopmat_warned
                ctx.caches.coopmat_warned = true
                @warn """
                    $(ctx.device_name) runs $sg-lane subgroups and cannot pin them to \
                    $COOPMAT_SUBGROUP, and this module's workgroup is \
                    $(wg == 0 ? "not a literal" : string(wg)) — more than one subgroup. A \
                    kernel that indexes `tid ÷ $COOPMAT_SUBGROUP` will read another \
                    subgroup's fragment here and return wrong results silently. Kernels \
                    that derive their subgroup from `device_subgroup_size` are fine."""
            end
        end
    end

    stage = VK.PipelineShaderStageCreateInfo(
        VK.SHADER_STAGE_COMPUTE_BIT,
        shader_mod,
        entry_name;
        next = stage_next
    )
    # The capture bits give `pipeline_exec_stats` / `pipeline_exec_ir` data to read.
    flags = VK.PipelineCreateFlag(VK.PIPELINE_CREATE_DISPATCH_BASE_BIT)
    if ctx.pipeline_exec_props_available
        flags |= VK.PipelineCreateFlag(VK.PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR) |
                 VK.PipelineCreateFlag(VK.PIPELINE_CREATE_CAPTURE_INTERNAL_REPRESENTATIONS_BIT_KHR)
    end
    pipeline = createpipeline(ctx, flags, entry_name, spirv_bytes) do flags
        create_compute_pipeline(ctx, VK.ComputePipelineCreateInfo(stage, layout, -1; flags))
    end

    result = LavaComputePipeline(shader_mod, layout, pipeline, UInt32(push_constant_size),
                                  needs_tlas_descriptor, ds_layout)
    ctx.caches.pipelines[cache_key] = result
    push!(ctx.caches.pipeline_order, cache_key)
    while length(ctx.caches.pipeline_order) > MAX_PIPELINE_CACHE_SIZE[]
        old_key = popfirst!(ctx.caches.pipeline_order)
        delete!(ctx.caches.pipelines, old_key)
        # Safe to evict: every dispatch `hold!`s its pipeline, so submissions
        # using it keep it alive until they complete.
    end
    return result
end

"""Allocate a compute binding for tlas.accel. The caller must hold the pool through
submission completion (or reusable-recording release); this path does not cache it."""
alloc_compute_tlas_descriptor_set(ctx::VkContext, pipeline::LavaComputePipeline, tlas) =
    alloc_tlas_descriptor_set(ctx, pipeline.descriptor_set_layout::VK.DescriptorSetLayout, tlas.accel)

"""Allocate and write a binding-0 acceleration-structure descriptor; caller owns pool/set lifetime."""
function alloc_tlas_descriptor_set(ctx::VkContext, ds_layout::VK.DescriptorSetLayout, accel)
    dev = ctx.device
    desc_pool = VK.DescriptorPool(dev, UInt32(1), [
        VK.DescriptorPoolSize(
            VK.DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, UInt32(1)),
    ])
    desc_sets = @vk_checked ctx "vkAllocateDescriptorSets (TLAS)" VK.allocate_descriptor_sets(dev,
        VK.DescriptorSetAllocateInfo(desc_pool, [ds_layout]))
    desc_set = desc_sets[1]

    as_write = VK.WriteDescriptorSetAccelerationStructureKHR([accel])
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

    return desc_pool, desc_set
end

"""
    create_compute_pipeline(ctx, ci) -> (pipeline, VkResult)

One compute pipeline through `ctx`'s pipeline cache. The result is
`PIPELINE_COMPILE_REQUIRED`, with no usable pipeline, when `ci` forbids
compilation and the cache misses.
"""
function create_compute_pipeline(ctx::VkContext, ci::VK.ComputePipelineCreateInfo)
    dev = ctx.device
    if LARGE_STACK_PIPELINE
        # `.vks` holds the non-dispatchable handle's UInt64, passed as a pointer.
        raw = ctx.pipeline_cache.vks
        cache_handle = raw isa Ptr ? raw : Ptr{Cvoid}(UInt(raw))
        ci_low = convert(VK._ComputePipelineCreateInfo, ci)
        vk_ci_ref = Ref(ci_low.vks)
        pipeline_out = Ref(Ptr{Cvoid}(C_NULL))
        vk_result = GC.@preserve ci_low vk_ci_ref pipeline_out create_compute_pipeline_large_stack(
            ctx, dev.vks, cache_handle,
            Ptr{Cvoid}(pointer_from_objref(vk_ci_ref)),
            Ptr{Cvoid}(pointer_from_objref(pipeline_out)))
        vk_result == VK_PIPELINE_COMPILE_REQUIRED && return nothing, VK.PIPELINE_COMPILE_REQUIRED
        vk_result != 0 && error("vkCreateComputePipelines failed with VkResult $vk_result")
        parent = VK.handle(dev)
        return VK.Pipeline(pipeline_out[], x -> VK._destroy_pipeline(parent, x), dev), VK.SUCCESS
    end
    pipelines, code = @vk_checked ctx "vkCreateComputePipelines" VK.create_compute_pipelines(
        dev, [ci]; pipeline_cache = ctx.pipeline_cache)
    return pipelines[1], code
end
