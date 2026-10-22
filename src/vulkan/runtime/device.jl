# Vulkan device creation, capability queries and validation-message capture.
# `vk_context()` lazily builds the process-default `VkContext`;
# `VkContext(; select, debug)` builds others.

"""
    RTPipelineProperties

Ray tracing pipeline properties of the physical device. `VkContext.rt_pipeline_properties`
is `nothing` without the RT extensions.
"""
struct RTPipelineProperties
    shader_group_handle_size::UInt32
    shader_group_base_alignment::UInt32
    shader_group_handle_alignment::UInt32
    max_ray_recursion_depth::UInt32
    max_ray_hit_attribute_size::UInt32
end

# The two kinds of GPU work, both closed: no command buffer is open when a
# Mantle call returns. A `Recording` is a plan's, written once at `record!` and
# submitted every run; a `OneShot` is written, sealed and submitted within one
# call. `submit!(bq, closed...)` is the only path to the driver and submits
# immediately. See `oneshot!` in `command.jl`.

abstract type Closed end

"""
    OneShot

Work submitted once (an ad hoc launch, an upload, a readback, a run's host
stores and pointer patches, a frame), written and sealed inside one call. Owned
by the submission that carries it; core's `sweep!` hands it back to the
channel's free list (`finishrecording!`) once the timeline passes it, and its
scratch regions go back then. What the commands must outlive is held on the
channel (`hold!`), not here.

`bq` is core's `SubmitChannel` over a `VulkanQueue`, typed `Any` because the
queue is declared below; `queueof` asserts the concrete type at every read.
"""
mutable struct OneShot <: Closed
    bq::Any
    cmd::VK.CommandBuffer
    # The `Unified` regions the commands read arguments and workgroup counts
    # from. An address from `get_arg_buffer` is valid as long as these commands
    # can run.
    regions::Vector{Region}
    # The buffers these commands name, recorded by `syncbuf!` and read once at
    # `submit!`, where core's `crosswaits!` derives the waits and `stamp!`
    # records the submission (`graph/lifetime.jl`).
    sync::Vector{VkManagedBuffer}
    open::Bool
end

"""
    Recording

A plan's work, written once into one command buffer and submitted every run.
The graph decides pass order, barriers and argument placement and
[`Emitter`](@ref) writes exactly that, so no submit threshold cuts it into
several buffers. The command buffer, argument regions, descriptor sets and a
strong reference to every resource it names stay alive until `release!`.
"""
mutable struct Recording <: Closed
    bq::Any
    cmd::VK.CommandBuffer
    # `Unified` regions the commands read. Usually empty: a plan's arguments
    # live in its own `ArgMemory`.
    regions::Vector{Region}
    # Descriptor sets the commands bind, one per (pipeline layout, acceleration
    # structure) pair; see [`tlasset!`](@ref).
    sets::Vector{Any}
    # Timeline value of the last submission, 0 if never submitted (a queue's
    # first signal is 1). `release!` waits on it before the command buffer goes
    # back to the free list; freeing one the device still reads aborts in the
    # driver. `UInt64`, not `Any`: assigned on the allocation-free `run!` path.
    token::UInt64
    open::Bool
    # (device address, host pointer) for each address the pack wrote. The
    # pointer is into the plan's argument memory or this recording's scratch
    # regions; `closerecording!` resolves which (`patchtarget`) when it builds
    # the patch table. Filled by `recpatch!`; one-shots never need it.
    patches::Vector{Tuple{UInt64,Ptr{UInt8}}}
    # Everything the commands name and must outlive, taken from the channel's
    # hold frame at seal (`takeholds!`) and dropped by `release!`. Held by the
    # recording, not each submission, because it is submitted every run.
    # `nothing` before seal and after `release!`.
    holds::Union{Nothing,Vector{Any}}
    # The buffers a submission must order against (every array the pack walker
    # strips, every acceleration structure a trace binds), collected while
    # emitting and read by every run through core's `crosswaits!` / `stamp!`.
    # Fixed for the recording's life.
    sync::Vector{VkManagedBuffer}
end

"""
Per-queue scratch for `wait_timeline!` and `query_timeline`: one semaphore slot,
one value slot, one packed wait info and one counter cell, refilled per call so
a wait allocates nothing. Only the channel's owning thread (`bq.thread`) uses
it; other threads take the allocating path.
"""
mutable struct QueueSlots
    wait_sems::Vector{VK.vk.VkSemaphore}
    wait_values::Vector{UInt64}
    wait_info::Base.RefValue{VK.vk.VkSemaphoreWaitInfo}
    counter::Base.RefValue{UInt64}
end
QueueSlots() = QueueSlots(Vector{VK.vk.VkSemaphore}(undef, 1), Vector{UInt64}(undef, 1),
                          Ref(VK.vk.VkSemaphoreWaitInfo(VK.vk.VkStructureType(0), C_NULL,
                                                        VK.vk.VkSemaphoreWaitFlags(0),
                                                        UInt32(0), C_NULL, C_NULL)),
                          Ref(UInt64(0)))

"""
    VulkanQueue{C}

The driver side of core's `SubmitChannel`: the `VkQueue`, its command pool and
timeline semaphore, the context, and per-queue scratch the submit and wait
paths refill. In-flight submissions, hold frames and pooled recordings are the
channel's.
"""
mutable struct VulkanQueue{C}
    device::VK.Device
    queue::VK.Queue
    family_index::UInt32
    cmd_pool::VK.CommandPool
    # Each submit increments `next_timeline` and signals the new value.
    timeline_sem::VK.Semaphore
    next_timeline::UInt64
    # The owning VkContext; a type parameter because `VkContext` is declared
    # after this struct.
    ctx::C
    # How long `flush!` waits before deciding a dispatch is not completing.
    flush_timeout_ns::UInt64
    # The last two dispatches on this queue, for the dispatch log and
    # DEVICE_LOST diagnostics. Per queue so a crash is attributed to the
    # right queue's kernel.
    last_dispatch_info::String
    prev_dispatch_info::String
    # Hardware queue index within `family_index`, returned by
    # `release_batch_queue!`. -1 for the primary queue and for queues sharing it.
    queue_index::Int
    slots::QueueSlots
    # Scratch for `submit!`, refilled per call (single writer, synchronous).
    #
    # Cross-channel waits filled by core's `crosswaits!`: (channel, token)
    # pairs, lowered to a timeline semaphore and value. `Any` because the
    # channel type names this one.
    waits::Vector{Tuple{Any,UInt64}}
    # Raw VulkanCore structs, because Vulkan.jl's wrappers allocate a `deps`
    # vector per struct. `raw_submits[1]` points into these vectors' data, so a
    # refill must stay within capacity and `vkQueueSubmit2` runs under
    # `GC.@preserve`.
    raw_cb_infos::Vector{VK.vk.VkCommandBufferSubmitInfo}
    raw_wait_infos::Vector{VK.vk.VkSemaphoreSubmitInfo}
    raw_signal_infos::Vector{VK.vk.VkSemaphoreSubmitInfo}
    raw_submits::Vector{VK.vk.VkSubmitInfo2}               # always length 1
end

# The channel type is spelled out at each use, not aliased, because the channel
# is core's. It pools `OneShot`s; a plan's `Recording` is never pooled
# (`release!`). A submission's payload is core's `Submission` over its one-shot,
# or `nothing` when a run submits its recording alone.

"""The queue a closed command buffer was taken from, as the concrete type."""
@inline queueof(c::Closed) = c.bq::SubmitChannel{VulkanQueue{VkContext},OneShot,UInt64,Submission{Union{Nothing,OneShot}}}


function SubmitChannel(device::VK.Device, queue::VK.Queue, qf_idx::UInt32, ctx;
                    queue_index::Int=-1)
    cmd_pool = VK.CommandPool(device, qf_idx;
        flags=VK.COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT)
    # Per-queue timeline semaphore for cross-queue ordering.
    type_info = VK.SemaphoreTypeCreateInfo(VK.SEMAPHORE_TYPE_TIMELINE, UInt64(0))
    timeline_sem = unwrap(VK.create_semaphore(device,
        VK.SemaphoreCreateInfo(; next=type_info)))
    q = VulkanQueue{typeof(ctx)}(device, queue, qf_idx, cmd_pool,
                    timeline_sem, UInt64(0),
                    ctx,
                    UInt64(120) * 1_000_000_000,            # flush timeout
                    "", "",                                 # last / prev dispatch info
                    queue_index,
                    QueueSlots(),
                    Tuple{Any,UInt64}[],
                    sizehint!(VK.vk.VkCommandBufferSubmitInfo[], 8),
                    sizehint!(VK.vk.VkSemaphoreSubmitInfo[], 4),
                    sizehint!(VK.vk.VkSemaphoreSubmitInfo[], 4),
                    Vector{VK.vk.VkSubmitInfo2}(undef, 1))
    return SubmitChannel{VulkanQueue{typeof(ctx)},OneShot,UInt64,
                         Submission{Union{Nothing,OneShot}}}(q)
end

# The driver bundle behind a channel (`bq.channel`, which core never reads)
# and its fields.
@inline driver(bq::SubmitChannel{VulkanQueue{C}}) where {C} = bq.channel::VulkanQueue{C}
@inline ctxof(bq::SubmitChannel{VulkanQueue{C}}) where {C} = driver(bq).ctx::C
@inline vkdevice(bq::SubmitChannel{<:VulkanQueue}) = driver(bq).device
@inline vkqueue(bq::SubmitChannel{<:VulkanQueue}) = driver(bq).queue
@inline timelineof(bq::SubmitChannel{<:VulkanQueue}) = driver(bq).timeline_sem

# The Mantle device this channel submits to, which core passes to `rawfree`
# and `flush!`. Not a field: a `LavaDevice` holds its default channel, so the
# channel is built first. `lavadevice` caches one device per context.
deviceof(bq::SubmitChannel{<:VulkanQueue}) = lavadevice(ctxof(bq))

"""
    CoopMat2Caps

The `VK_NV_cooperative_matrix2` sub-features enabled on this device. Without
the extension `available` and every other field are false, so a kernel tests
only the field it needs. NVIDIA-only: kernels keep a plain
`VK_KHR_cooperative_matrix` path for AMD (RDNA3 WMMA).

`workgroup_scope`: matrices spanning the workgroup. `flexible_dimensions`: M/N/K
outside the fixed shape list. `reductions`: in-place row/column reduce.
`conversions`: element conversion without a memory round trip.
`per_element_operations`: a per-element `(row, col)` callback.
`tensor_addressing`, `block_loads`: descriptor-driven addressing and tile loads.
"""
struct CoopMat2Caps
    available::Bool
    workgroup_scope::Bool
    flexible_dimensions::Bool
    reductions::Bool
    conversions::Bool
    per_element_operations::Bool
    tensor_addressing::Bool
    block_loads::Bool
end

CoopMat2Caps() = CoopMat2Caps(false, false, false, false, false, false, false, false)

"""
    query_shader_cores(phys_dev) -> (cores, warps)

SM/CU count and warps per SM, or `(0, 0)` where the device does not report
them; kernels size grids to fill the device from it. Physical-device property
queries (vendor coverage as in llama.cpp), so neither extension is enabled, but
the chained struct is only filled for an advertised extension. The chain also
needs an instance of API 1.1 or later (`VkContext` asks for 1.4); below that
it is dropped silently, since `vkGetPhysicalDeviceProperties2` returns void.
"""
function query_shader_cores(phys_dev)
    smcount, warps = 0, 0
    if has_extension(phys_dev, "VK_NV_shader_sm_builtins")
        # No try: the extension is advertised, so a failure is a bug, and a
        # swallowed one would read as "not reported" (`0`).
        p = VK.get_physical_device_properties_2(
                phys_dev, VK.PhysicalDeviceShaderSMBuiltinsPropertiesNV).next
        smcount, warps = Int(p.shader_sm_count), Int(p.shader_warps_per_sm)
    elseif has_extension(phys_dev, "VK_AMD_shader_core_properties2")
        p = VK.get_physical_device_properties_2(
                phys_dev, VK.PhysicalDeviceShaderCoreProperties2AMD).next
        smcount = Int(p.active_compute_unit_count)
    end
    return smcount, warps
end


"""
    VkContext

One Vulkan device: instance, physical and logical device, queues, capability
flags and per-device caches. `default_bq` is the primary submit channel;
[`allocate_batch_queue!`](@ref) hands out more.
"""
mutable struct VkContext
    instance::VK.Instance
    physical_device::VK.PhysicalDevice
    device::VK.Device
    queue_family_index::UInt32
    device_name::String
    # Primary channel; set by the inner constructor once `ctx` exists. Fully
    # concrete so `ctx.default_bq` infers.
    default_bq::SubmitChannel{VulkanQueue{VkContext},OneShot,UInt64,Submission{Union{Nothing,OneShot}}}
    # Queue 1 of the primary family, or the primary queue when the family has
    # only one. Used by the external-image copy.
    compute_queue::VK.Queue
    # Ray tracing (nothing if not available)
    rt_pipeline_properties::Union{Nothing, RTPipelineProperties}
    # `nothing` without `VK_EXT_debug_utils`.
    debug_messenger::Any
    # The messenger's ring and its `pUserData`; see `ValidationRing`.
    validation::ValidationRing
    # Next hardware queue index of the primary family, and how many were created.
    next_queue_index::Int
    max_queue_count::Int
    # Queues from `allocate_batch_queue!` not yet released. Holds them alive: a
    # buffer's stamp names its channel and core's `reclaim!` queries that
    # channel's timeline. Reachable only through the buffers, channel and
    # buffers can be collected together, the semaphore finalized first, and the
    # query segfaults in the driver.
    extra_queues::Vector{SubmitChannel{VulkanQueue{VkContext},OneShot,UInt64,Submission{Union{Nothing,OneShot}}}}
    # Hardware queue slots handed back by `release_batch_queue!`, reused before
    # `next_queue_index` advances.
    free_queue_indices::Vector{Int}
    # A compute+transfer family other than the primary (RADV: family 1, 4
    # queues). Its queues are created; nothing submits to them.
    async_queue_family_index::Union{Nothing, UInt32}
    async_queue_count::Int
    # Set on `ERROR_DEVICE_LOST` and on retirement (`mark_device_lost!`).
    # Finalizers check it to skip Vulkan calls.
    device_lost::Bool
    # Physical-device properties, cached for the alloc and dispatch paths.
    memory_properties::VK.PhysicalDeviceMemoryProperties
    max_wg_dims::NTuple{3, Int}
    # Byte alignment for BDAs passed as `pScratchData` in AS builds
    # (`bda_alignment_for(ctx, true)`).
    as_scratch_align::UInt64
    # VK_KHR_ray_query enabled.
    ray_query_available::Bool
    # VK_NV_ray_tracing_invocation_reorder (SER, NVIDIA-only) enabled: the
    # emitter may declare ShaderInvocationReorderNV and raygen may use
    # `lava_rt_hit_object_*` / `lava_rt_reorder_thread_*`.
    ser_available::Bool
    # The two flags above as passed to every compile this context runs: what a
    # module compiled for this device may declare.
    features::TargetFeatures
    # VK_EXT_conditional_rendering enabled. `repeat!` needs it to let a
    # device-written count skip iterations; without it the graph is rejected at
    # compile rather than running every iteration.
    conditional_rendering_available::Bool
    # VK_KHR_cooperative_matrix enabled. `coopmat_shapes` lists the driver's
    # legal (M, N, K, A/B type, C type, scope) combinations, the only tile
    # shapes a kernel may use; empty when unsupported.
    coopmat_available::Bool
    coopmat_shapes::Vector{NamedTuple{(:M, :N, :K, :ab_type, :c_type, :scope),
                                      Tuple{Int, Int, Int, UInt32, UInt32, UInt32}}}
    # VK_NV_cooperative_matrix2 sub-features. NVIDIA-only: kernels branching on
    # them keep a path for all-false (AMD).
    coopmat2::CoopMat2Caps
    # VK_NV_cooperative_vector (NVIDIA-only): subgroup matrix x vector, for
    # shapes where a cooperative matrix wastes most of its tile.
    coopvec_available::Bool
    # VK_KHR_shader_maximal_reconvergence + subgroup_uniform_control_flow:
    # together they make a spin-wait on a shared flag well-defined, which
    # producer/consumer kernels need since Vulkan has no named barriers.
    maximal_reconvergence_available::Bool
    subgroup_uniform_control_flow_available::Bool
    # VK_KHR_shader_subgroup_rotate: shuffle by a subgroup-uniform delta.
    subgroup_rotate_available::Bool
    # VK_EXT_mesh_shader: mesh (and task) stages.
    mesh_shader_available::Bool
    # VK_EXT_memory_budget enabled: OOM reports read the per-heap budget and usage.
    memory_budget_available::Bool
    # VK_KHR_pipeline_executable_properties enabled on this device: requested
    # (`enable_pipeline_executable_properties!`) before creation and offered.
    # Pipeline creation and the queries check this, not the global request.
    pipeline_exec_props_available::Bool
    # VK_KHR_external_memory(_fd) enabled: `VulkanExternalImage` can export
    # allocations as opaque fds (OpenGL via GL_EXT_memory_object_fd).
    external_memory_available::Bool
    # `minImportedHostPointerAlignment` for `importhost`, or 0 without
    # VK_EXT_external_memory_host.
    host_import_alignment::Int
    # VK_KHR_video_decode_queue + h264 enabled. Queue and family index are
    # `nothing` without video-decode support.
    video_decode_available::Bool
    video_decode_queue::Union{Nothing, VK.Queue}
    video_decode_queue_family_index::Union{Nothing, UInt32}
    # Whether GPU-assisted validation attached: the layer or extension can be
    # missing, so this can be false where `debug.gpu_av` is true. Even `true`
    # is no proof it fires; see `verify_gpu_av`.
    gpu_assisted::Bool
    # The configuration requested at construction; `reset_device!` carries it
    # across a recovery reset.
    debug::DebugConfig
    # Whether `VK_LAYER_KHRONOS_validation` loaded; `debug.validation` is what
    # was asked.
    validating::Bool
    # Host-visible word a kernel sets when it throws (`LavaRuntime.signal_exception`).
    # Its address is the second push-constant word of every kernel that can
    # throw; `checkexceptions!` reads and clears it.
    exception_flag::VkManagedBuffer
    # Keys the on-disk VkPipelineCache file.
    driver_version::String
    # Seeded from disk at init, passed to every vkCreate*Pipelines call, saved
    # on `reset_device!` and at exit.
    pipeline_cache::VK.PipelineCache
    # `vkCmdPipelineBarrier` resolved for this device. A device function pointer
    # is valid only for its device; as a module global a second context
    # overwrote it and the first device's commands went through the second
    # device's driver (segfault in lavapipe while dispatching on NVIDIA).
    cmd_pipeline_barrier_fptr::Ptr{Nothing}
    # Identity for diagnostics, not a cache key. A counter, not `objectid`,
    # because object ids are reused after collection.
    id::UInt64

    # Per-device caches: they die with the context, so nothing outlives the
    # handles they hold. See `DeviceCaches`.
    caches::DeviceCaches

    # Debugging and instrumentation toggles for this device; see `Diagnostics`.
    diag::Diagnostics

    # Two-phase init via `new()` so `SubmitChannel` gets the live `ctx`; no
    # constructor leaves `default_bq` unset. `primary_queue` is the default
    # channel's queue; the other arguments map to fields.
    function VkContext(instance::VK.Instance,
                       physical_device::VK.PhysicalDevice,
                       device::VK.Device,
                       queue_family_index::UInt32,
                       device_name::String,
                       primary_queue::VK.Queue,
                       compute_queue::VK.Queue,
                       rt_pipeline_properties::Union{Nothing, RTPipelineProperties},
                       debug_messenger::Any,
                       validation::ValidationRing,
                       next_queue_index::Int,
                       max_queue_count::Int,
                       async_queue_family_index::Union{Nothing, UInt32},
                       async_queue_count::Int,
                       device_lost::Bool,
                       memory_properties::VK.PhysicalDeviceMemoryProperties,
                       max_wg_dims::NTuple{3, Int},
                       as_scratch_align::UInt64,
                       ray_query_available::Bool=false,
                       ser_available::Bool=false,
                       conditional_rendering_available::Bool=false,
                       coopmat_available::Bool=false,
                       coopmat_shapes=nothing,
                       coopmat2::CoopMat2Caps=CoopMat2Caps(),
                       coopvec_available::Bool=false,
                       maximal_reconvergence_available::Bool=false,
                       subgroup_uniform_control_flow_available::Bool=false,
                       subgroup_rotate_available::Bool=false,
                       mesh_shader_available::Bool=false,
                       memory_budget_available::Bool=false,
                       external_memory_available::Bool=false,
                       gpu_assisted::Bool=false,
                       debug::DebugConfig=DebugConfig(),
                       driver_version::AbstractString="unknown",
                       video_decode_available::Bool=false,
                       video_decode_queue::Union{Nothing, VK.Queue}=nothing,
                       video_decode_queue_family_index::Union{Nothing, UInt32}=nothing,
                       pipeline_exec_props_available::Bool=false)
        ctx = new()
        ctx.instance = instance
        ctx.physical_device = physical_device
        ctx.device = device
        ctx.queue_family_index = queue_family_index
        ctx.device_name = device_name
        ctx.compute_queue = compute_queue
        ctx.rt_pipeline_properties = rt_pipeline_properties
        ctx.debug_messenger = debug_messenger
        ctx.validation = validation
        ctx.next_queue_index = next_queue_index
        ctx.max_queue_count = max_queue_count
        ctx.extra_queues = SubmitChannel{VulkanQueue{VkContext},OneShot,UInt64,Submission{Union{Nothing,OneShot}}}[]
        ctx.free_queue_indices = Int[]
        ctx.async_queue_family_index = async_queue_family_index
        ctx.async_queue_count = async_queue_count
        ctx.device_lost = device_lost
        ctx.memory_properties = memory_properties
        ctx.max_wg_dims = max_wg_dims
        ctx.as_scratch_align = as_scratch_align
        ctx.ray_query_available = ray_query_available
        ctx.ser_available = ser_available
        ctx.features = TargetFeatures(; ser = ser_available, ray_query = ray_query_available)
        ctx.conditional_rendering_available = conditional_rendering_available
        ctx.coopmat_available = coopmat_available
        ctx.coopmat_shapes = coopmat_shapes === nothing ?
            eltype(fieldtype(VkContext, :coopmat_shapes))[] : coopmat_shapes
        ctx.coopmat2 = coopmat2
        ctx.coopvec_available = coopvec_available
        ctx.maximal_reconvergence_available = maximal_reconvergence_available
        ctx.subgroup_uniform_control_flow_available = subgroup_uniform_control_flow_available
        ctx.subgroup_rotate_available = subgroup_rotate_available
        ctx.mesh_shader_available = mesh_shader_available
        ctx.memory_budget_available = memory_budget_available
        ctx.pipeline_exec_props_available = pipeline_exec_props_available
        ctx.external_memory_available = external_memory_available
        ctx.video_decode_available = video_decode_available
        ctx.video_decode_queue = video_decode_queue
        ctx.video_decode_queue_family_index = video_decode_queue_family_index
        ctx.gpu_assisted = gpu_assisted
        ctx.debug = debug
        # Set by `VkContext(; …)` once it knows whether the layer loaded.
        ctx.validating = false
        ctx.driver_version = driver_version
        ctx.id = (VK_CONTEXT_COUNTER[] += 1)
        # Registered before anything can allocate against it, so the exit hook
        # marks it lost (`LIVE_CONTEXTS`).
        push!(LIVE_CONTEXTS, WeakRef(ctx))
        ctx.caches = DeviceCaches()
        ctx.diag = Diagnostics()
        # Null until `VkContext(; …)` sets it, so a barrier recorded earlier is
        # skipped rather than jumping to garbage.
        ctx.cmd_pipeline_barrier_fptr = C_NULL
        ctx.host_import_alignment = 0
        # Seeded from disk; the header is validated against this physical
        # device first, since `vkCreatePipelineCache` is no safe place to find
        # a mismatch.
        ctx.pipeline_cache = create_lava_pipeline_cache(
            device, lava_pipeline_cache_path(device_name, driver_version), physical_device)
        register_pipeline_cache_atexit!()
        ctx.default_bq = SubmitChannel(device, primary_queue, queue_family_index, ctx)
        return ctx
    end
end

# Compact: the default walks the cycle context -> channels -> queues -> context
# and took half an hour for one backend.
Base.show(io::IO, ctx::VkContext) = print(io, "VkContext(", repr(ctx.device_name), ")")
Base.show(io::IO, q::VulkanQueue{VkContext}) =
    print(io, "VulkanQueue(", repr(q.ctx.device_name), ", family ", q.family_index, ")")

"""
    caps(ctx) -> DeviceCaps

What kernels ask this device, queried on first call and cached on the context.
Lazy because part of it (cooperative-matrix GEMM, subgroup width) is answered
against the logical device, which does not exist while the context is built.
Tests may assign a modified copy:
`ctx.caches.caps = DeviceCaps(caps(ctx); workgrouplimit = 512)`.
"""
function caps(ctx::VkContext)
    c = ctx.caches.caps
    c === nothing || return c
    limits = VK.get_physical_device_properties(ctx.physical_device).limits
    cores, warps = query_shader_cores(ctx.physical_device)
    # The tile comes from the driver's table: a kernel striding by the wrong
    # tile reads another fragment's registers without crashing.
    shapes = matrixshapes(ctx)
    square = bestshape(shapes, Float16, Float32)
    ctx.caches.caps = DeviceCaps(
        coopmat_gemm_available(ctx),
        square === nothing ? 0 : square.M,
        device_subgroup_size(ctx),
        COOPMAT_SUBGROUP,
        Int(limits.max_compute_shared_memory_size),
        Int(limits.max_compute_work_group_invocations),
        cores, warps,
        workgroup_matrix_granularity(ctx),
        shapes)
end

"""
    workgroup_matrix_granularity(ctx) -> Vector{(invocations, M, N, K)}

Granularities of workgroup-scope fp16 x fp16 -> fp32 cooperative matrices, one
row per workgroup size, sorted by size. Empty without
`VK_NV_cooperative_matrix2` workgroup scope, so it doubles as the capability
test. `M`, `N`, `K` are multiples, not sizes, and coarsen as the workgroup
grows (RTX 4000 Ada: 16/16/16 at 64 invocations, 32/32/16 at 256), so padding
grows with the workgroup. Only fp16 -> fp32 rows are kept; the query also
returns others.
"""
function workgroup_matrix_granularity(ctx::VkContext)
    rows = NTuple{4,Int}[]
    ctx.coopmat2.workgroup_scope || return rows
    props = unwrap(VK.get_physical_device_cooperative_matrix_flexible_dimensions_properties_nv(
                       ctx.physical_device))
    for p in props
        p.scope == VK.SCOPE_WORKGROUP_KHR || continue
        p.a_type == VK.COMPONENT_TYPE_FLOAT16_KHR || continue
        p.b_type == VK.COMPONENT_TYPE_FLOAT16_KHR || continue
        p.c_type == VK.COMPONENT_TYPE_FLOAT32_KHR || continue
        p.result_type == VK.COMPONENT_TYPE_FLOAT32_KHR || continue
        push!(rows, (Int(p.workgroup_invocations), Int(p.m_granularity),
                     Int(p.n_granularity), Int(p.k_granularity)))
    end
    sort!(rows; by = first)
    return rows
end

"""
    shader_warps_per_sm(ctx) -> Union{Nothing,Int}

Maximum resident subgroups per SM, or `nothing` when unreported (only
`VK_NV_shader_sm_builtins` reports it).
"""
shader_warps_per_sm(ctx::VkContext) =
    (w = caps(ctx).warps) == 0 ? nothing : w

const VK_CONTEXT_REF = Ref{Union{Nothing, VkContext}}(nothing)

"""
Every `VkContext` ever built, weakly. Julia runs `atexit` hooks before the final
finalizer sweep; the hook in `initbackend!(::VulkanAPI)` marks every context
here lost (`mark_all_devices_lost!`), not only the bound one, so no finalizer
calls into a torn-down driver (a second lavapipe context segfaulted at exit).
Weak, so the list keeps no context alive; never pruned, since a process makes
one or two devices.
"""
const LIVE_CONTEXTS = WeakRef[]

"""
    mark_all_devices_lost!()

Mark every live context lost, so no finalizer reaches a driver afterwards.
Shutdown only: every later Vulkan call on these devices becomes a no-op.
"""
function mark_all_devices_lost!()
    for r in LIVE_CONTEXTS
        c = r.value
        c isa VkContext && mark_device_lost!(c)
    end
    return nothing
end

"""
    bind_context!(ctx)

Bind the process-default context. Pushes no emitter features: those travel with
the context (`ctx.features`) and its compile jobs, so a kernel for a second
device is not shaped by the bound one.
"""
function bind_context!(ctx::Union{Nothing, VkContext})
    VK_CONTEXT_REF[] = ctx
    return ctx
end
# Guards the lazy init in `vk_context`: `VkContext(; …)` is not idempotent, and
# two threads arriving together built two devices.
const VK_CONTEXT_LOCK = ReentrantLock()

"""
    device_lost(ctx::VkContext)  ->  Bool

Whether this context's device has been marked lost. The no-argument form reads
the bound context (`false` if none).
"""
device_lost(ctx::VkContext) = ctx.device_lost
device_lost() = let ctx = VK_CONTEXT_REF[]
    ctx === nothing ? false : ctx.device_lost
end

"""
    mark_device_lost!(ctx)

Mark `ctx`'s device lost: finalizers and later callers skip Vulkan calls on it.
Set on `ERROR_DEVICE_LOST` (`mark_if_lost!`) and on retirement of a context
whose buffers outlive it. `reset_device!` retires the context it replaces;
whoever builds one with `VkContext(; select)` must retire it. A context dropped
unretired hands its finalizers a device VK.jl may already have destroyed, and
the driver crashes at the next GC.
"""
mark_device_lost!(ctx::VkContext) = (ctx.device_lost = true; nothing)

"""
Source of `VkContext.id`: monotonic, never reused. Nothing in `src/` keys on
the id; it identifies contexts in diagnostics.
"""
const VK_CONTEXT_COUNTER = Ref{UInt64}(0)

"""
    vk_context() -> VkContext

The process-default context, built and bound on first call. Thread-safe.
"""
function vk_context()
    ctx = VK_CONTEXT_REF[]
    ctx === nothing || return ctx::VkContext
    # Double-checked under a lock: two contexts built concurrently left buffers
    # stamped with the loser's channel, and a cross-queue wait then passed one
    # device's semaphore to the other (driver segfault).
    lock(VK_CONTEXT_LOCK)
    try
        ctx = VK_CONTEXT_REF[]
        ctx === nothing || return ctx::VkContext
        # `invokelatest` so inference records no backedge from `vk_context`,
        # which every Lava operation reaches, to the constructor. Otherwise
        # invalidating the constructor (loading GLMakie does) re-infers the
        # whole package on the next GPU call (~41 s).
        ctx = Base.invokelatest(VkContext)::VkContext
        bind_context!(ctx)
        return ctx::VkContext
    finally
        unlock(VK_CONTEXT_LOCK)
    end
end

"""
    reset_device!(; select = nothing,
                       debug = <the outgoing device's config>)

Replace the process-default device: save the old context's pipeline cache,
destroy its pool, mark it lost, and bind a new one. Per-device caches go with
the old context. Every existing `LavaArray` becomes invalid and must be
reallocated.

`select` picks the new device as [`selectdevice`](@ref) does (`nothing`: best
ranked) and does not carry across. `debug` defaults to the outgoing device's
configuration, so a recovery reset after `ERROR_DEVICE_LOST` keeps it. Changing
validation needs a reset, since it is fixed at `vkCreateInstance`:

    reset_device!(debug = DebugConfig(gpu_av = true, pool_disabled = true))
    verify_gpu_av()
    reset_device!(debug = DebugConfig())

For a device that is not the default, use `VkContext(; select, debug)`.
"""
function reset_device!(; select = nothing,
                            debug::Union{Nothing,DebugConfig} = nothing)
    cfg = debug !== nothing ? debug :
          let old = VK_CONTEXT_REF[]
              old === nothing ? DebugConfig() : old.debug
          end
    # Saved so the next session skips the SPIR-V to ISA recompile.
    let old = VK_CONTEXT_REF[]
        old === nothing || save_pipeline_cache!(old)
    end
    # Retire the old context before dropping it, also on a voluntary reset.
    # Its buffers become garbage in the same collection, where Julia does not
    # order finalizers: if `VK.Device`'s finalizer runs first, a buffer that
    # still calls into the device segfaults in the driver. `device_lost` makes
    # `destroy_buffer!` skip Vulkan calls.
    let old = VK_CONTEXT_REF[]
        if old !== nothing
            # Its blocks, while it still owns them.
            destroy_pool!(old)
            mark_device_lost!(old)
        end
    end
    bind_context!(nothing)
    # Built here, not by `vk_context()`'s lazy path, because `cfg` must reach
    # `vkCreateInstance`. `invokelatest` for the same backedge reason.
    ctx = Base.invokelatest(VkContext; select, debug = cfg)::VkContext
    bind_context!(ctx)
    # Logs what attached, not what was asked: a run without GPU-AV reads like
    # a clean run.
    @info "Lava: device reset complete" device=ctx.device_name validation=cfg.validation gpu_av=ctx.gpu_assisted pool_disabled=cfg.pool_disabled
    if cfg.gpu_av && !ctx.gpu_assisted
        @warn "GPU-AV was requested but did NOT attach (gpu_assisted=false) — a clean run now means the instrument was off, not that there is no fault. Check VK_EXT_validation_features + VK_LAYER_KHRONOS_validation on this loader/driver."
    end
    return nothing
end


"""
    VkContext(; select = nothing, debug = DebugConfig()) -> VkContext

Build a device without binding it as the process default (`vk_context`,
`reset_device!` and `defaultdevice!` bind). The caller retires it
(`mark_device_lost!`).

`debug` ([`DebugConfig`](@ref)) holds every validation and instrumentation
setting; they are fixed at `vkCreateInstance`. `select` picks the physical
device as `Mantle.selectdevice` does: `nothing` for the ranking, a name
substring, an index into `devices(VulkanAPI())`, or a predicate over
`DeviceInfo`. The loader enumerates the GPU and lavapipe from one instance, so
`VkContext(select = "llvmpipe")` beside `vk_context()` gives two devices on a
single-GPU machine.
"""
function VkContext(; select = nothing, debug::DebugConfig = DebugConfig())
    # API 1.4, for the 1.3/1.4 core features enabled below.
    app_info = VK.ApplicationInfo(
        v"0.1.0", v"0.1.0", v"1.4.0";
        application_name="Lava.jl",
        engine_name="Lava"
    )
    want_validation = debug.validation
    layers = String[]
    if want_validation
        available_layers = unwrap(VK.enumerate_instance_layer_properties())
        for l in available_layers
            name = String(filter(!=('\0'), collect(l.layer_name)))
            if name == "VK_LAYER_KHRONOS_validation"
                push!(layers, "VK_LAYER_KHRONOS_validation")
                break
            end
        end
    end

    # Instance extensions from the drivers and the validation layer.
    inst_extensions = String[]
    available_ext = unwrap(VK.enumerate_instance_extension_properties())
    ext_names = Set(String(filter(!=('\0'), collect(e.extension_name))) for e in available_ext)
    has_validation = !isempty(layers)
    if has_validation
        layer_ext = unwrap(VK.enumerate_instance_extension_properties(; layer_name="VK_LAYER_KHRONOS_validation"))
        for e in layer_ext
            push!(ext_names, String(filter(!=('\0'), collect(e.extension_name))))
        end
    end
    has_debug_utils = "VK_EXT_debug_utils" in ext_names
    if has_debug_utils
        push!(inst_extensions, "VK_EXT_debug_utils")
    end
    # Surface extensions only where offered: requesting a missing one fails the
    # instance (lavapipe on macOS has no `VK_EXT_metal_surface`).
    "VK_KHR_surface" in ext_names && push!(inst_extensions, "VK_KHR_surface")
    if Sys.islinux()
        if "VK_KHR_xcb_surface" in ext_names
            push!(inst_extensions, "VK_KHR_xcb_surface")
        elseif "VK_KHR_xlib_surface" in ext_names
            push!(inst_extensions, "VK_KHR_xlib_surface")
        end
        if "VK_KHR_wayland_surface" in ext_names
            push!(inst_extensions, "VK_KHR_wayland_surface")
        end
    elseif Sys.iswindows()
        "VK_KHR_win32_surface" in ext_names && push!(inst_extensions, "VK_KHR_win32_surface")
    elseif Sys.isapple()
        "VK_EXT_metal_surface" in ext_names && push!(inst_extensions, "VK_EXT_metal_surface")
    end
    # GLFW's null platform (`JULIA_GLFW_PLATFORM=null`, for sessions where no
    # window can be mapped) creates its surfaces from this extension.
    "VK_EXT_headless_surface" in ext_names && push!(inst_extensions, "VK_EXT_headless_surface")

    # Extended validation, all needing VK_EXT_validation_features (all slow):
    # gpu_av instruments shaders (OOB descriptor access, ray query misuse; not
    # plain BDA ranges), sync_val checks synchronization, best_practices warns
    # on API misuse, printf delivers `@lava_printf` output. `gpu_assisted` and
    # `sync_val` record what attached, since the extension may be missing.
    gpu_assisted = false
    sync_val     = false
    want_gpu_av     = debug.gpu_av
    want_printf     = debug.printf
    want_sync_val   = debug.sync_val
    want_best       = debug.best_practices
    want_gpuav_safe = debug.gpu_av_safe
    gpuav_shaders   = debug.gpu_av_shaders
    validation_features_reqs = VK.ValidationFeatureEnableEXT[]
    if want_printf
        push!(validation_features_reqs,
              VK.VALIDATION_FEATURE_ENABLE_DEBUG_PRINTF_EXT)
    end
    if want_gpu_av
        push!(validation_features_reqs,
              VK.VALIDATION_FEATURE_ENABLE_GPU_ASSISTED_EXT)
        push!(validation_features_reqs,
              VK.VALIDATION_FEATURE_ENABLE_GPU_ASSISTED_RESERVE_BINDING_SLOT_EXT)
    end
    if want_sync_val
        push!(validation_features_reqs,
              VK.VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT)
    end
    if want_best
        push!(validation_features_reqs,
              VK.VALIDATION_FEATURE_ENABLE_BEST_PRACTICES_EXT)
    end
    if !isempty(validation_features_reqs) && has_validation && "VK_EXT_validation_features" in ext_names
        push!(inst_extensions, "VK_EXT_validation_features")
        validation_features = VK.ValidationFeaturesEXT(validation_features_reqs, [])
        # Safe Mode and selective instrumentation exist only in
        # `VK_EXT_layer_settings`; its struct is chained in front and the layer
        # reads both.
        nextchain = validation_features
        if want_gpu_av && (want_gpuav_safe || !isempty(gpuav_shaders)) &&
           "VK_EXT_layer_settings" in ext_names
            push!(inst_extensions, "VK_EXT_layer_settings")
            settings = VK.LayerSettingEXT[]
            if want_gpuav_safe
                push!(settings, layer_setting_bool("gpuav_safe_mode", true))
            end
            if !isempty(gpuav_shaders)
                push!(settings, layer_setting_bool("gpuav_select_instrumented_shaders", true))
            end
            nextchain = VK.LayerSettingsCreateInfoEXT(settings; next = validation_features)
            @info "GPU-AV layer settings" safe_mode=want_gpuav_safe shaders=gpuav_shaders
        elseif want_gpu_av && (want_gpuav_safe || !isempty(gpuav_shaders))
            @warn "GPU-AV Safe Mode / selective instrumentation requested but VK_EXT_layer_settings is not available; GPU-AV runs unconfigured and may crash."
        end
        instance = VK.Instance(
            layers,
            inst_extensions;
            application_info=app_info,
            next=nextchain
        )
        gpu_assisted = want_gpu_av
        sync_val     = want_sync_val
    else
        instance = VK.Instance(
            layers,
            inst_extensions;
            application_info=app_info
        )
        if has_validation && (want_gpu_av || want_sync_val || want_best)
            @warn "Vulkan validation layers active but VK_EXT_validation_features not available; extended validation disabled."
        end
    end

    # Created before the messenger, which is given its address.
    validation = ValidationRing()
    debug_messenger = nothing
    if has_debug_utils
        debug_messenger = setup_debug_messenger(instance, validation, debug)
    end

    phys_devs = unwrap(VK.enumerate_physical_devices(instance))
    isempty(phys_devs) && throw(LavaError(
        "device initialization",
        "No Vulkan-capable GPU found",
        "Ensure Vulkan drivers are installed"))

    phys_dev = phys_devs[selectdevice(select, deviceinfos(phys_devs))]
    props = VK.get_physical_device_properties(phys_dev)
    dev_name = String(filter(!=('\0'), collect(props.device_name)))

    # Prefers a graphics+compute family.
    qf_idx = find_graphics_compute_queue_family(phys_dev)

    qf_props = VK.get_physical_device_queue_family_properties(phys_dev)
    max_queues = qf_props[qf_idx + 1].queue_count
    n_queues = min(4, Int(max_queues))
    queue_priorities = ones(Float32, n_queues)
    queue_ci = [VK.DeviceQueueCreateInfo(qf_idx, queue_priorities)]

    # Queues from a second compute+transfer family, if any (RADV: family 0 is
    # graphics+compute with 1 queue, family 1 compute-only with 4). Nothing
    # submits to them.
    async_qf_idx = find_async_compute_queue_family(phys_dev, qf_idx)
    async_n_queues = 0
    if async_qf_idx !== nothing
        async_max = qf_props[async_qf_idx + 1].queue_count
        async_n_queues = min(4, Int(async_max))
        push!(queue_ci, VK.DeviceQueueCreateInfo(async_qf_idx, ones(Float32, async_n_queues)))
    end

    # One video-decode queue whenever offered: extensions and queue have no
    # effect until a decode session exists.
    advertised = deviceextensions(phys_dev)
    has(e) = e in advertised
    video_qf_idx = find_video_decode_queue_family(phys_dev)
    videoexts = ["VK_KHR_video_queue", "VK_KHR_video_decode_queue", "VK_KHR_video_decode_h264"]
    has_video_decode = video_qf_idx !== nothing && all(has, videoexts)
    has_video_decode || (video_qf_idx = nothing)
    video_qf_idx === nothing ||
        push!(queue_ci, VK.DeviceQueueCreateInfo(video_qf_idx, Float32[1.0]))

    rtexts = ("VK_KHR_acceleration_structure", "VK_KHR_ray_tracing_pipeline",
              "VK_KHR_deferred_host_operations")
    has_rt = all(has, rtexts)
    # Pipeline executable properties (registers, scratch, ISA) for profiling: opt-in
    # via `enable_pipeline_executable_properties!()` before device creation, because
    # enabling it changes pipeline creation on some drivers.
    exec_props = has("VK_KHR_pipeline_executable_properties") && PIPELINE_EXEC_PROPERTIES_REQUESTED[]
    # Every optional extension is enabled where advertised; enabling one never fails
    # device creation, a feature bit the device lacks does. `VK_KHR_portability_subset`
    # must be enabled when advertised; headless lavapipe on macOS has no swapchain.
    # Opaque-fd export is for GL interop, host import for `importhost` (weights
    # mapped from a file).
    extensions = filter(has, [
        "VK_KHR_swapchain", "VK_KHR_portability_subset",
        "VK_KHR_workgroup_memory_explicit_layout", "VK_EXT_shader_atomic_float",
        "VK_EXT_memory_budget", "VK_EXT_conditional_rendering", "VK_KHR_cooperative_matrix",
        "VK_NV_cooperative_matrix2", "VK_NV_cooperative_vector",
        "VK_KHR_shader_maximal_reconvergence", "VK_KHR_shader_subgroup_uniform_control_flow",
        "VK_KHR_shader_subgroup_rotate", "VK_EXT_mesh_shader",
        "VK_KHR_external_memory_fd", "VK_EXT_external_memory_host"])
    has_rt && append!(extensions, rtexts, filter(has, ["VK_KHR_ray_query", "VK_NV_ray_tracing_invocation_reorder"]))
    exec_props && push!(extensions, "VK_KHR_pipeline_executable_properties")
    any(has, ("VK_KHR_external_memory_fd", "VK_EXT_external_memory_host")) &&
        push!(extensions, "VK_KHR_external_memory")
    has_video_decode && append!(extensions, videoexts, filter(has, ["VK_KHR_video_decode_h265"]))
    has_external_memory = has("VK_KHR_external_memory_fd")
    host_import_alignment = has("VK_EXT_external_memory_host") ?
        Int(VK.get_physical_device_properties_2(phys_dev,
            VK.PhysicalDeviceExternalMemoryHostPropertiesEXT).next.min_imported_host_pointer_alignment) : 0
    has_memory_budget = has("VK_EXT_memory_budget")

    # The feature chain, each struct enabling what Mantle uses and the device reports.
    chain = Any[]
    feature!(T, want = fieldnames(T); required = ()) = last(push!(chain,
        devicefeatures(phys_dev, T, want; required, next = isempty(chain) ? C_NULL : last(chain))))
    feature!(VK.PhysicalDeviceVariablePointersFeatures)
    # Required: Float16 and i8 in SPIR-V, the BDA struct layout, cross-queue sync, BDA
    # and QueueFamily memory scope. Wanted: the emitter declares Int64Atomics (also
    # for @localmem), 64-bit subgroup shuffles need extended types
    # (VUID-RuntimeSpirv-None-06275), and Device scope is not QueueFamily scope when
    # work runs on the async compute family.
    feature!(VK.PhysicalDeviceVulkan12Features,
        (:shader_buffer_int_64_atomics, :shader_shared_int_64_atomics,
         :shader_subgroup_extended_types, :vulkan_memory_model_device_scope);
        required = (:shader_float_16, :shader_int_8, :scalar_block_layout,
                    :timeline_semaphore, :buffer_device_address, :vulkan_memory_model))
    # All required by Vulkan 1.3: `synchronization2` for `vkQueueSubmit2` and
    # `vkCmdPipelineBarrier2`, `dynamic_rendering` instead of render pass objects.
    feature!(VK.PhysicalDeviceVulkan13Features, ();
        required = (:synchronization2, :dynamic_rendering, :maintenance4,
                    :subgroup_size_control, :compute_full_subgroups,
                    :pipeline_creation_cache_control, :shader_demote_to_helper_invocation,
                    :shader_terminate_invocation, :shader_integer_dot_product,
                    :shader_zero_initialize_workgroup_memory, :private_data))
    # Mixed-type shared-memory structs.
    has("VK_KHR_workgroup_memory_explicit_layout") &&
        feature!(VK.PhysicalDeviceWorkgroupMemoryExplicitLayoutFeaturesKHR)
    exec_props && feature!(VK.PhysicalDevicePipelineExecutablePropertiesFeaturesKHR)
    # Lava's reductions use OpAtomicFAdd, on storage buffers and workgroup memory.
    has("VK_EXT_shader_atomic_float") && feature!(VK.PhysicalDeviceShaderAtomicFloatFeaturesEXT,
        (:shader_buffer_float_32_atomics, :shader_buffer_float_32_atomic_add,
         :shader_shared_float_32_atomics, :shader_shared_float_32_atomic_add))
    if has_rt
        feature!(VK.PhysicalDeviceAccelerationStructureFeaturesKHR, (); required = (:acceleration_structure,))
        feature!(VK.PhysicalDeviceRayTracingPipelineFeaturesKHR, (:ray_tracing_pipeline_trace_rays_indirect,);
                 required = (:ray_tracing_pipeline,))
    end
    enabled(ext, T, f, want = fieldnames(T)) = has(ext) && getfield(feature!(T, want), f)
    has_ray_query = has_rt && enabled("VK_KHR_ray_query", VK.PhysicalDeviceRayQueryFeaturesKHR, :ray_query)
    # SER (NVIDIA): `reorderThreadWithHitObjectNV` regroups divergent closest-hit
    # shading into coherent warps.
    has_ser = has_rt && enabled("VK_NV_ray_tracing_invocation_reorder",
        VK.PhysicalDeviceRayTracingInvocationReorderFeaturesNV, :ray_tracing_invocation_reorder)
    # The feature, not only the extension: lavapipe on macOS lists the extension
    # with `cooperative_matrix` off.
    has_coopmat = enabled("VK_KHR_cooperative_matrix", VK.PhysicalDeviceCooperativeMatrixFeaturesKHR,
                          :cooperative_matrix, (:cooperative_matrix,))
    coopmat2_caps = CoopMat2Caps()
    if has_coopmat && has("VK_NV_cooperative_matrix2")
        q = feature!(VK.PhysicalDeviceCooperativeMatrix2FeaturesNV)
        coopmat2_caps = CoopMat2Caps(true, (getfield(q, f) for f in fieldnames(typeof(q)) if f !== :next)...)
    end
    has_coopvec = enabled("VK_NV_cooperative_vector", VK.PhysicalDeviceCooperativeVectorFeaturesNV,
                          :cooperative_vector)
    # Conditional rendering: a device-buffer predicate, read at execution, discards
    # the commands in its scope; barriers still execute inside a discarded scope.
    has_cond_render = enabled("VK_EXT_conditional_rendering",
        VK.PhysicalDeviceConditionalRenderingFeaturesEXT, :conditional_rendering)
    has_max_reconv = enabled("VK_KHR_shader_maximal_reconvergence",
        VK.PhysicalDeviceShaderMaximalReconvergenceFeaturesKHR, :shader_maximal_reconvergence)
    has_subgroup_ucf = enabled("VK_KHR_shader_subgroup_uniform_control_flow",
        VK.PhysicalDeviceShaderSubgroupUniformControlFlowFeaturesKHR, :shader_subgroup_uniform_control_flow)
    has_subgroup_rotate = enabled("VK_KHR_shader_subgroup_rotate",
        VK.PhysicalDeviceShaderSubgroupRotateFeaturesKHR, :shader_subgroup_rotate)
    # The task stage as reported: the geometry lowering dispatches mesh workgroups directly.
    has_mesh_shader = enabled("VK_EXT_mesh_shader", VK.PhysicalDeviceMeshShaderFeaturesEXT,
                              :mesh_shader, (:task_shader, :mesh_shader))

    device = VK.Device(phys_dev, queue_ci, [], extensions;
        enabled_features = devicefeatures(phys_dev, VK.PhysicalDeviceFeatures,
            (:shader_float_64, :shader_int_16, :independent_blend, :geometry_shader,
             :tessellation_shader, :fill_mode_non_solid, :wide_lines, :large_points);
            required = (:shader_int_64,)),
        next = last(chain))

    queue = VK.get_device_queue(device, qf_idx, 0)
    # Queue 1 of the primary family, or the primary queue if it has only one.
    compute_queue = n_queues >= 2 ?
        VK.get_device_queue(device, qf_idx, 1) : queue
    video_decode_queue = video_qf_idx === nothing ? nothing :
        VK.get_device_queue(device, video_qf_idx, 0)

    rt_props = nothing
    if has_rt
        props2 = VK.get_physical_device_properties_2(phys_dev,
            VK.PhysicalDeviceRayTracingPipelinePropertiesKHR)
        rtp = props2.next
        rt_props = RTPipelineProperties(
            rtp.shader_group_handle_size,
            rtp.shader_group_base_alignment,
            rtp.shader_group_handle_alignment,
            rtp.max_ray_recursion_depth,
            rtp.max_ray_hit_attribute_size,
        )
    end

    # Vulkan has no usage flag for AS scratch, so `LavaArray(...; scratch=true)`
    # gets this alignment through `bda_alignment_for`.
    as_scratch_align = UInt64(1)
    if has_rt
        as_props2 = VK.get_physical_device_properties_2(phys_dev,
            VK.PhysicalDeviceAccelerationStructurePropertiesKHR)
        as_scratch_align = max(UInt64(1),
            UInt64(as_props2.next.min_acceleration_structure_scratch_offset_alignment))
    end

    has_validation = !isempty(layers)
    if has_rt
        @info "Lava: initialized Vulkan device with RT" device=dev_name queue_family=qf_idx handle_size=rt_props.shader_group_handle_size max_recursion=rt_props.max_ray_recursion_depth validation=has_validation gpu_assisted=gpu_assisted sync_val=sync_val debug_utils=has_debug_utils
    else
        @info "Lava: initialized Vulkan device (no RT)" device=dev_name queue_family=qf_idx validation=has_validation gpu_assisted=gpu_assisted sync_val=sync_val debug_utils=has_debug_utils
    end
    if !has_validation && want_validation
        @warn "Vulkan validation layers not found. Install vulkan-validationlayers for GPU error diagnostics."
    end

    # Per device (see the `VkContext` field); assigned once the context exists.
    cmd_barrier_fptr = VK.function_pointer(device, "vkCmdPipelineBarrier")

    mem_props = VK.get_physical_device_memory_properties(phys_dev)
    phys_props = VK.get_physical_device_properties(phys_dev)
    wgc = phys_props.limits.max_compute_work_group_count
    max_wg = (Int(wgc[1]), Int(wgc[2]), Int(wgc[3]))

    # The fixed set of (M, N, K, type) tiles a kernel may choose from.
    CMShape = eltype(fieldtype(VkContext, :coopmat_shapes))
    coopmat_shapes = CMShape[]
    if has_coopmat
        try
            for p in unwrap(VK.get_physical_device_cooperative_matrix_properties_khr(phys_dev))
                push!(coopmat_shapes, (M=Int(p.m_size), N=Int(p.n_size), K=Int(p.k_size),
                                       ab_type=UInt32(p.a_type), c_type=UInt32(p.c_type),
                                       scope=UInt32(p.scope)))
            end
        catch err
            # Only a `VulkanError` means unsupported; anything else is a bug
            # and rethrows rather than becoming a silently slower path.
            err isa VK.VulkanError || rethrow()
            @warn "cooperative-matrix property query failed; treating as unsupported" exception = err
        end
    end

    ctx = VkContext(
        instance, phys_dev, device, qf_idx, dev_name,
        queue, compute_queue,
        rt_props, debug_messenger, validation,
        2, n_queues,  # next_queue_index=2 (0=primary, 1=compute), max=n_queues
        async_qf_idx, async_n_queues,
        false,        # device_lost (fresh context)
        mem_props, max_wg,
        as_scratch_align,
        has_ray_query,
        has_ser,
        has_cond_render,
        has_coopmat,
        coopmat_shapes,
        coopmat2_caps,
        has_coopvec,
        has_max_reconv,
        has_subgroup_ucf,
        has_subgroup_rotate,
        has_mesh_shader,
        has_memory_budget,
        has_external_memory,
        gpu_assisted,
        debug,
        string(phys_props.driver_version),
        has_video_decode, video_decode_queue, video_qf_idx,
        exec_props,
    )
    ctx.cmd_pipeline_barrier_fptr = cmd_barrier_fptr
    ctx.host_import_alignment = host_import_alignment
    ctx.validating = has_validation
    flag = try_vk_alloc(ctx.default_bq, 16; unified = true)
    flag isa AllocFailure && throw(LavaError("device initialization",
        "could not allocate the 16-byte exception flag: $(flag.code) from $(flag.op)",
        "The device has no host-visible memory left; this is not recoverable here."))
    unsafe_store!(Ptr{UInt32}(flag.mapped_ptr), UInt32(0))
    ctx.exception_flag = flag
    # Clear what device creation reported (GPU-AV's harmless "adjusting
    # settings" warnings) so the first shader compile does not see it. This
    # context's ring: the no-argument form clears the bound context's.
    clear_validation_messages!(ctx)
    # Applied here, not by the caller: with the pool on, GPU-AV cannot see
    # sub-pool overruns.
    mempolicy(ctx).disabled = debug.pool_disabled
    return ctx
end

# The `LavaBackend` method is in `array/ka_backend.jl`: `LavaBackend` is defined
# after this file.
supports_batch_queue(::VulkanAPI) = true

"""
    allocate_batch_queue!(ctx) -> SubmitChannel

A new submit channel on its own hardware queue of the primary family, or on the
primary queue with its own command pool once all are taken. The context holds
it until [`release_batch_queue!`](@ref); a dropped one stays alive with the
device.
"""
function allocate_batch_queue!(ctx::VkContext)
    idx = isempty(ctx.free_queue_indices) ? ctx.next_queue_index : pop!(ctx.free_queue_indices)
    if idx < ctx.max_queue_count
        queue = VK.get_device_queue(ctx.device, ctx.queue_family_index, UInt32(idx))
        idx == ctx.next_queue_index && (ctx.next_queue_index += 1)
    else
        # All hardware queues taken: share the primary queue.
        queue = vkqueue(ctx.default_bq)
        idx = -1
    end
    bq = SubmitChannel(ctx.device, queue, ctx.queue_family_index, ctx; queue_index = idx)
    push!(ctx.extra_queues, bq)
    return bq
end

"""
    release_batch_queue!(bq::SubmitChannel{<:VulkanQueue})

Give a queue from [`allocate_batch_queue!`](@ref) back: flush and drain it,
drop it from the context and free its hardware slot. Call from the queue's
owning thread once nothing will record on it again. Throws for the primary
queue; a second release is a no-op.
"""
function release_batch_queue!(bq::SubmitChannel{<:VulkanQueue})
    ctx = ctxof(bq)
    bq === ctx.default_bq &&
        throw(LavaError("release_batch_queue!", "the context's primary queue cannot be released",
                        "Only queues from `allocate_batch_queue!` can be given back."))
    i = findfirst(q -> q === bq, ctx.extra_queues)
    i === nothing && return nothing   # already released

    # On a lost device every call would fail, so the queue is dropped as is.
    if !device_lost(ctx)
        flush!(bq)
        drain!(bq)
    end

    deleteat!(ctx.extra_queues, i)
    driver(bq).queue_index >= 0 && push!(ctx.free_queue_indices, driver(bq).queue_index)
    return nothing
end

"""
    queue_released(bq::SubmitChannel{<:VulkanQueue}) -> Bool

Whether `bq` has been handed back by [`release_batch_queue!`](@ref), i.e. is no
longer in `ctx.extra_queues`. A released queue must not be used or queried: a
buffer's stamp may still name it while its `timeline_sem` has been finalized.
It has no in-flight work (released after flush and drain), so nothing naming it
needs a wait.
"""
function queue_released(bq::SubmitChannel{<:VulkanQueue})
    ctx = ctxof(bq)
    bq === ctx.default_bq && return false
    return findfirst(q -> q === bq, ctx.extra_queues) === nothing
end

"""
    deviceinfos(phys_devs) -> Vector{DeviceInfo}

What `devices(VulkanAPI())` answers, from an enumeration: index, name, kind and
driver, so a selector is resolved in core without naming a `VkPhysicalDevice`.
"""
function deviceinfos(phys_devs)
    infos = DeviceInfo[]
    for (i, pd) in enumerate(phys_devs)
        props = VK.get_physical_device_properties(pd)
        name = String(filter(!=('\0'), collect(props.device_name)))
        t = props.device_type
        kind = t == VK.PHYSICAL_DEVICE_TYPE_DISCRETE_GPU   ? :discrete :
               t == VK.PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU ? :integrated :
               t == VK.PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU    ? :virtual :
               t == VK.PHYSICAL_DEVICE_TYPE_CPU            ? :cpu : :other
        # `VK_KHR_driver_properties` is core in 1.2.
        driver = if props.api_version >= v"1.2"
            d = VK.get_physical_device_properties_2(pd, VK.PhysicalDeviceDriverProperties).next
            description = strip(String(filter(!=('\0'), collect(d.driver_name))) * " " *
                                String(filter(!=('\0'), collect(d.driver_info))))
            d.driver_id == VK.DRIVER_ID_MESA_LLVMPIPE ? "lavapipe " * description : description
        else
            "unknown"
        end
        push!(infos, DeviceInfo(i, name, kind, driver))
    end
    return infos
end

"""
    devices(VulkanAPI()) -> Vector{DeviceInfo}

Every physical device the loader enumerates, in the order `select` indexes.
Through the bound instance when there is one, through a throwaway instance when
this is asked before any device exists.
"""
function devices(::VulkanAPI)
    bound = VK_CONTEXT_REF[]
    bound === nothing || return deviceinfos(unwrap(VK.enumerate_physical_devices(bound.instance)))
    # API 1.2 so `VK_KHR_driver_properties` answers; a 1.0 instance reports a
    # blank driver for every device.
    app = VK.ApplicationInfo(v"0.0.0", v"0.0.0", v"1.2.0")
    instance = VK.Instance(String[], String[]; application_info = app)
    return deviceinfos(unwrap(VK.enumerate_physical_devices(instance)))
end


function find_graphics_compute_queue_family(phys_dev)
    qf_props = VK.get_physical_device_queue_family_properties(phys_dev)
    for (i, qfp) in enumerate(qf_props)
        if (qfp.queue_flags & VK.QUEUE_COMPUTE_BIT) != 0 &&
           (qfp.queue_flags & VK.QUEUE_GRAPHICS_BIT) != 0
            return UInt32(i - 1)
        end
    end
    # Fall back to compute only; graphics pipelines will not work.
    for (i, qfp) in enumerate(qf_props)
        if (qfp.queue_flags & VK.QUEUE_COMPUTE_BIT) != 0
            return UInt32(i - 1)
        end
    end
    throw(LavaError(
        "device initialization",
        "No compute-capable queue family found",
        "Ensure your GPU supports Vulkan compute"))
end

"""
A compute+transfer queue family other than `primary_qf_idx`, or `nothing`.
"""
function find_async_compute_queue_family(phys_dev, primary_qf_idx::UInt32)
    qf_props = VK.get_physical_device_queue_family_properties(phys_dev)
    for (i, qfp) in enumerate(qf_props)
        family = UInt32(i - 1)
        family == primary_qf_idx && continue
        if (qfp.queue_flags & VK.QUEUE_COMPUTE_BIT) != 0 &&
           (qfp.queue_flags & VK.QUEUE_TRANSFER_BIT) != 0
            return family
        end
    end
    return nothing
end

"""
A queue family with `VK_QUEUE_VIDEO_DECODE_BIT_KHR` (0x20, not named in VK.jl),
or `nothing`. A dedicated transfer + decode family on NVIDIA.
"""
function find_video_decode_queue_family(phys_dev)
    qf_props = VK.get_physical_device_queue_family_properties(phys_dev)
    for (i, qfp) in enumerate(qf_props)
        if (UInt32(qfp.queue_flags) & 0x00000020) != 0
            return UInt32(i - 1)
        end
    end
    return nothing
end

"""The device extensions `phys_dev` advertises."""
deviceextensions(phys_dev) = Set(String(filter(!=('\0'), collect(e.extension_name)))
                                 for e in unwrap(VK.enumerate_device_extension_properties(phys_dev)))

"""Whether `phys_dev` advertises the device extension `name`."""
has_extension(phys_dev, name::String) = name in deviceextensions(phys_dev)

"""
    devicefeatures(phys_dev, T, want; required = (), next = C_NULL) -> T

The feature struct `T` with each field in `want` and `required` enabled where the
device reports it; a missing `required` field throws. `vkCreateDevice` fails on
any unsupported bit, and an advertised extension does not imply every bit.
"""
function devicefeatures(phys_dev, ::Type{T}, want; required = (), next = C_NULL) where {T}
    have = supportedfeatures(phys_dev, T)
    for f in required
        getfield(have, f) || throw(LavaError("device initialization",
            "the device lacks the feature $T.$f", "Mantle needs it on every Vulkan device."))
    end
    bits = Tuple(getfield(have, f) && (f in want || f in required) for f in fieldnames(T) if f !== :next)
    return featurestruct(T, bits, next)
end
supportedfeatures(phys_dev, ::Type{VK.PhysicalDeviceFeatures}) = VK.get_physical_device_features(phys_dev)
supportedfeatures(phys_dev, T::Type) = VK.get_physical_device_features_2(phys_dev, T).next
featurestruct(::Type{VK.PhysicalDeviceFeatures}, bits, next) = VK.PhysicalDeviceFeatures(bits...)
featurestruct(T::Type, bits, next) = T(bits...; next)

# Debug-utils callback. Runs re-entrantly inside blocking driver calls with
# driver locks held (GPU-AV reads back its log during `vkWaitSemaphores`), so it
# must not allocate, log, lock or yield: an `@error` here hung the process on an
# RTX 4000 Ada. It only copies the message into the preallocated ring with raw
# ccalls; `drain_validation_messages!` classifies and logs on the main thread.
# Flags arrive as raw `UInt32` so no flag wrapper is built.
function debug_callback(
    severity::UInt32,
    type::UInt32,
    p_callback_data::Ptr{VK.VkCore.VkDebugUtilsMessengerCallbackDataEXT},
    p_user_data::Ptr{Cvoid},
)
    (p_callback_data == C_NULL || p_user_data == C_NULL) && return UInt32(0)
    # The ring of this messenger's device, from `pUserData`; one isbits load.
    r = unsafe_load(Ptr{ValidationRingRaw}(p_user_data))
    data = unsafe_load(p_callback_data)        # isbits, no heap allocation
    msg_ptr = Ptr{UInt8}(data.pMessage)
    idx  = unsafe_load(r.write)
    slot = idx % VAL_RING_SLOTS
    dst  = r.buf + slot * VAL_RING_SLOT_BYTES
    n = 0
    if msg_ptr != C_NULL
        n = Int(ccall(:strlen, Csize_t, (Ptr{UInt8},), msg_ptr))
        n > VAL_RING_SLOT_BYTES - 1 && (n = VAL_RING_SLOT_BYTES - 1)
        ccall(:memcpy, Ptr{Cvoid}, (Ptr{Cvoid}, Ptr{Cvoid}, Csize_t), dst, msg_ptr, n % Csize_t)
    end
    unsafe_store!(r.len, Cint(n), slot + 1)
    unsafe_store!(r.sev, severity,  slot + 1)
    unsafe_store!(r.typ, type,      slot + 1)
    unsafe_store!(r.write, idx + 1)
    # VK_FALSE; throwing from a @cfunction would corrupt Vulkan state.
    return UInt32(0)
end

# Classify one drained message and capture hard validation errors. Called from
# `drain_validation_messages!` on the main thread, so it may allocate and log.
function handle_validation_message(r::ValidationRing, message::String, sev_u::UInt32, type_u::UInt32)
    # Only VALIDATION-typed (0x2) errors are captured; GENERAL (0x1) errors are
    # driver notes, not spec checks.
    is_error = (sev_u & UInt32(VK.DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT)) != 0
    is_warning = (sev_u & UInt32(VK.DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT)) != 0
    is_validation = (type_u & UInt32(VK.DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT)) != 0
    # Benign device-creation messages: logged as warnings, never captured.
    # "Device Layers have never worked" (VUID-VkDeviceCreateInfo-ppEnabledLayerNames-12385)
    # fires because VK.jl passes a non-null `ppEnabledLayerNames` for an empty
    # layer list.
    is_setup_noise = contains(message, "adjusting settings") ||
                     contains(message, "VALIDATION-SETTINGS") ||
                     contains(message, "Device Layers have never worked")
    # `@lava_printf` output (INFO severity), wrapped by the layer as
    # "vkQueueSubmit2(): pSubmits[0] DebugPrintf:\n<user text>"; only the user
    # text goes to `r.printf`.
    if contains(message, "DebugPrintf")
        marker = findlast("DebugPrintf:", message)
        text = marker === nothing ? message :
               lstrip(message[last(marker) + 1 : end], ['\n', '\r', ' '])
        if length(r.printf) >= MAX_PRINTF_MESSAGES
            popfirst!(r.printf)
        end
        push!(r.printf, String(text))
        @info "lava_printf" text
        return nothing
    end
    if is_error && is_validation && !is_setup_noise
        if length(r.messages) >= MAX_VALIDATION_MESSAGES
            popfirst!(r.messages)
        end
        push!(r.messages, message)
        @error "Vulkan validation error" message
    elseif is_error && !is_validation
        # Driver note (e.g. lavapipe SPIR-V notes): logged, not captured; the
        # call's VkResult is the authoritative signal.
        @warn "Vulkan driver note (type=$(type_u), not a validation error)" message
    elseif is_warning || is_setup_noise
        @warn "Vulkan validation warning" message
    end
    return nothing
end

"""
    drain_validation_messages!(ctx = <bound context>)

Turn the ring slots written since the last drain into Strings, capture hard
validation errors in `ctx.validation.messages` and log them. Main thread only
(it allocates and logs). Call before reading `ctx.validation.messages`.
"""
function drain_validation_messages!(ctx::Union{Nothing,VkContext} = VK_CONTEXT_REF[])
    ctx === nothing && return nothing
    r = ctx.validation
    write_idx = @inbounds r.write[1]
    read_idx  = r.read
    # If the callback lapped us, skip the slots it overwrote.
    if write_idx - read_idx > VAL_RING_SLOTS
        read_idx = write_idx - VAL_RING_SLOTS
    end
    while read_idx < write_idx
        slot = read_idx % VAL_RING_SLOTS
        n    = Int(@inbounds r.len[slot + 1])
        sev  = @inbounds r.sev[slot + 1]
        typ  = @inbounds r.typ[slot + 1]
        message = n == 0 ? "(no message)" :
            unsafe_string(pointer(r.buf) + slot * VAL_RING_SLOT_BYTES, n)
        handle_validation_message(r, message, sev, typ)
        read_idx += 1
    end
    r.read = write_idx
    return nothing
end

# Keeps every `VkLayerSettingEXT` value alive for the process: `pValues` is a
# raw pointer that `vkCreateInstance` reads.
const LAYER_SETTING_STORAGE = Any[]

"""
    layer_setting_bool(name, value) -> VK.LayerSettingEXT

One `VK_EXT_layer_settings` boolean for the Khronos validation layer, its value
kept in `LAYER_SETTING_STORAGE`.
"""
function layer_setting_bool(name::AbstractString, value::Bool)
    r = Ref(VK.VkBool32(value))
    push!(LAYER_SETTING_STORAGE, r)
    VK.LayerSettingEXT("VK_LAYER_KHRONOS_validation", String(name),
                           VK.LAYER_SETTING_TYPE_BOOL32_EXT,
                           Base.unsafe_convert(Ptr{Nothing}, Base.pointer_from_objref(r)))
end

function setup_debug_messenger(instance::VK.Instance, ring::ValidationRing,
                               debug::DebugConfig)
    callback_ptr = @cfunction(
        debug_callback,
        UInt32,
        (UInt32,
         UInt32,
         Ptr{VK.VkCore.VkDebugUtilsMessengerCallbackDataEXT},
         Ptr{Cvoid})
    )

    # Debug printf arrives at INFO severity; otherwise WARNING and up. Read from
    # this device's config, which matches the features its instance has.
    min_sev = debug.printf ?
        VK.DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT :
        VK.DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT
    index = findfirst(==(min_sev), VK.message_severities)
    severity = |(VK.message_severities[index:end]...)
    # `user_data` is handed back to every callback, which makes the ring per device.
    messenger = VK.DebugUtilsMessengerEXT(
        instance, severity,
        VK.DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
            VK.DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
            VK.DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT,
        callback_ptr;
        user_data = ring_user_data(ring),
    )
    return messenger
end

"""
    get_validation_messages(ctx = <bound context>) -> Vector{String}

Recent captured validation errors, after draining the ring.
"""
get_validation_messages(ctx::Union{Nothing,VkContext} = VK_CONTEXT_REF[]) =
    ctx === nothing ? String[] :
        (drain_validation_messages!(ctx); copy(ctx.validation.messages))

"""
    clear_validation_messages!(ctx = <bound context>)

Drain the ring (so errors are logged), then empty the captured list.
"""
clear_validation_messages!(ctx::Union{Nothing,VkContext} = VK_CONTEXT_REF[]) =
    ctx === nothing ? String[] :
        (drain_validation_messages!(ctx); empty!(ctx.validation.messages))

"""
    check_validation_errors!(context::String, ctx::VkContext)

Throw a `LavaError` listing up to five validation errors captured since the
last check, and clear them. Call after operations that may trigger validation
errors (shader module or pipeline creation, dispatch recording).
"""
function check_validation_errors!(context::String, ctx::VkContext)
    drain_validation_messages!(ctx)
    msgs = ctx.validation.messages
    isempty(msgs) && return
    # `messages` holds only ERROR-severity validation messages
    # (`handle_validation_message`), whatever their text says.
    errors = copy(msgs)
    isempty(errors) && return
    n = min(length(errors), 5)
    detail = join(["  [$i] $(first(errors[i], 1000))" for i in 1:n], "\n")
    empty!(msgs)
    throw(LavaError(
        context,
        "Vulkan validation error(s):\n$detail",
        "Fix the validation errors above before proceeding."
    ))
end

# `Mantle.waitidle` (declared in `graph/queue.jl`); the `LavaBackend` method is
# in `array/ka_backend.jl`.
function waitidle(ctx::VkContext)
    # Checked: a discarded Result let a lost device look idle.
    throw_if_error(ctx, "vkDeviceWaitIdle", VK.device_wait_idle(ctx.device))
    return nothing
end
waitidle(d::VK.Device) = VK.device_wait_idle(d)

supports_graphics(::VkContext) = true

"""
    vulkan_available() -> Bool

Whether a Vulkan device can be created. A loader exists once this code is
loaded; whether it exposes a device is found by building the default context
(cached by `vk_context`). Only `LavaError` and `VK.VulkanError` (no ICD, no
suitable device) mean "not available"; anything else rethrows.
"""
function vulkan_available()
    VK_CONTEXT_REF[] === nothing || return true
    try
        vk_context()
        return true
    catch e
        e isa LavaError || e isa VK.VulkanError || rethrow()
        @debug "Mantle: a Vulkan loader is present but no usable device was found" exception = e
        return false
    end
end
