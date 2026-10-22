# Types `VkContext` names, declared before it so the context holds its
# per-device state in concretely typed fields instead of module-level
# dictionaries. Only types are hoisted; their methods stay with their behaviour.

"""
    LavaComputePipeline

A compiled compute pipeline ready for dispatch.
"""
struct LavaComputePipeline
    shader_module::VK.ShaderModule
    pipeline_layout::VK.PipelineLayout
    pipeline::VK.Pipeline
    push_constant_size::UInt32
    needs_tlas_descriptor::Bool
    descriptor_set_layout::Union{Nothing, VK.DescriptorSetLayout}
end

struct SubgroupSizeControl
    min::Int
    max::Int
    compute::Bool   # COMPUTE present in requiredSubgroupSizeStages
end

mutable struct VkManagedBuffer
    buffer::VK.Buffer
    memory::VK.DeviceMemory
    address::UInt64     # BDA for PhysicalStorageBuffer access
    mapped_ptr::Ptr{UInt8}  # Non-null for unified/BAR memory
    size::Int
    # The pool region the bytes were carved from; `nothing` when the buffer owns
    # its `VkDeviceMemory` (mapped, unified or extra-usage buffers).
    region::Union{Nothing, Region}
    # Core's lifetime record (`graph/lifetime.jl`): last submitting channel and
    # token, and how many unsubmitted recordings hold the buffer. `holders` is
    # `@atomic` because two channels on two threads can hold one buffer.
    stamp::Stamp{UInt64}
    # `BUF_STATE_*` (memory.jl), changed only by CAS so a double free is a no-op.
    @atomic state::UInt8
    # Owning `VkContext`. `Any` because `VkContext` is declared in device.jl,
    # included after this file.
    ctx::Any
    # Position of this buffer in the owner `sync` list it last joined, so
    # `syncbuf!` checks membership with one comparison. A hint only: a stale
    # value reads as "not listed".
    syncpos::Int
end

VkManagedBuffer(buffer, memory, address, mapped_ptr, size, region, stamp, state, ctx) =
    VkManagedBuffer(buffer, memory, address, mapped_ptr, size, region, stamp, state, ctx, 0)

mutable struct MemoryStats
    # Device-local heap bytes, probed lazily and refreshed every 10 s by `maybe_collect`.
    @atomic size::Int
    @atomic last_updated::Float64
end

MemoryStats() = MemoryStats(0, 0.0)

"""
Per-device allocation policy, and accounting for buffers that own their memory.

Not the allocator: memory comes from `Mantle.Pool`, one per `LavaDevice` and one
`LavaDevice` per `VkContext` (`DEVICES`). Must stay per device: shared, it
divided two devices' usage by one heap, and a block served one device's
allocation from another device's memory (reads of 0.0, or a segfault).
"""
mutable struct MemoryPolicy
    # Trim threshold, soft cap and collection pacing are core's `PoolPolicy`.
    disabled::Bool
    track_allocs::Bool
    # Whether `maybe_collect` runs, and its cached heap size.
    eager_gc::Bool
    stats::MemoryStats
    # Dedicated buffers only; pooled bytes are `reserved(spans(ctx))`, and
    # `gpu_live_bytes` adds both. Atomic because `destroy_buffer!` can run off
    # the allocating thread.
    live_bytes::Threads.Atomic{Int}
    live_buffers::Set{VkManagedBuffer}
end

# A compiled kernel and this device's pipeline for it (`DeviceCaches.linked`).
struct LavaLinkedKernel
    compiled::LavaGPUKernel        # SPIR-V bytes + push_info (serializable)
    pipeline::LavaComputePipeline  # VkPipeline (session-dependent, NOT serializable)
    offsets::Vector{Int}           # arg layout offsets (derived from push_info)
    byval_sizes::Vector{Int}      # LLVM byval sizes (derived from push_info)
end

"""
Everything a dispatch needs that depends only on the kernel and argument types,
cached under `Tuple{typeof(f), typeof(all_args)}` in an `IdDict` (a pointer
hash), not under a rebuilt signature, whose interning and lookups were the
largest host cost at ~2000 dispatches per step. `world` is checked on each hit,
so a redefined kernel rebuilds once and a stale pipeline is never served.
"""
struct LaunchPlan
    compiled::LavaGPUKernel
    pipeline::LavaComputePipeline
    offsets::Vector{Int}
    byval_sizes::Vector{Int}
    arg_buffer_size::Int
    total_size::Int
    world::UInt64
    wg::NTuple{3,Int}
    ray_query::Bool
end

"""
    VulkanCompiledGraphicsPipeline

A compiled graphics pipeline ready for draw commands.
"""
struct VulkanCompiledGraphicsPipeline <: CompiledGraphicsPipeline
    pipeline::VK.Pipeline
    pipeline_layout::VK.PipelineLayout
    modules::Vector{VK.ShaderModule}
    push_constant_size::UInt32
    descriptor_set_layout::Union{Nothing, VK.DescriptorSetLayout}
    push_stage_flags::VK.ShaderStageFlag
    # For inspection. A vector: with dynamic rendering a pipeline targets a set
    # of colour attachment formats.
    color_formats::Vector{VK.Format}
    has_depth::Bool
    # `:vertex` or `:mesh`, which decides the legal draw command: `vkCmdDraw`
    # against a mesh pipeline is undefined behaviour, not a reported error.
    front::Symbol
end

"""
    Diagnostics

Every debugging and instrumentation toggle, with its counters and logs, per
context so two devices are instrumented independently. Off by default except
`freed_bda_scan_poisons` and `launch_arg_validation`; each read is a field load
behind a branch.
"""
mutable struct Diagnostics
    # ── allocation / free
    alloc_debug::Bool
    free_debug::Bool
    freed_bda_scan::Bool
    # Whether the scan zeroes the stale BDAs it finds, so the GPU faults on null.
    # Off makes it observe only, to tell a prevented fault from a timing change.
    freed_bda_scan_poisons::Bool
    destroy_freed_bdas_throws::Bool
    presubmit_scan::Bool
    presubmit_scan_throws::Bool
    pack_arg_assert_live::Bool
    # `UInt64`, not `Any`: `submit!` scans when `slab_dump_target != UInt64(0)`,
    # which `nothing` passes, and then every submit scanned the arg slab
    # (~41 µs per dispatch).
    slab_dump_target::UInt64
    # ── dispatch / batch
    batch_timing::Bool
    dispatch_logging::Bool
    dispatch_log_file::Union{Nothing,String}
    dispatch_timing::Bool
    # ── compiler / cache
    launch_arg_validation::Bool
    spirv_dump_dir::Union{Nothing,String}
    spirv_dump_counter::Int
    kernel_debug_counter::Int
    broadcast_probe::Any

    # ── logs the flags above fill
    alloc_log::Vector{NamedTuple}
    free_log::Vector{NamedTuple}
    freed_bda_scan_log::Vector{NamedTuple}
    slab_dump_log::Vector{NamedTuple}
    # Call site => (count, bytes), filled when `track_allocs` is on.
    alloc_trace::Dict{Symbol,Tuple{Int,Int}}
    alloc_trace_lock::ReentrantLock
    # The rolling dispatch log `throw_with_validation_context` prints, and the
    # per-batch wait times behind `batch_timing`.
    dispatch_log::Vector{String}
    batch_wait_times::Vector{Float64}
    batch_wait_info::Vector{String}
    batch_wait_dispatches::Vector{Int}
    # Pipelines this device compiled or refused to.
    pipeline_compile_misses::Vector{String}
    pipeline_compiles_refused::Int
    # Atomics: `submit!` may run off the main thread.
    flush_counter::Threads.Atomic{Int}
    total_dispatches::Threads.Atomic{Int}
    # Closed command buffers that began with the global barrier (one per
    # one-shot and per recording); see `headbarrier!`.
    head_barriers::Threads.Atomic{Int}
end

Diagnostics() = Diagnostics(false, false, false, true, false, false, false, false, UInt64(0),
                            false, false, nothing, false,
                            true, nothing, 0, 0, nothing,
                            NamedTuple[], NamedTuple[], NamedTuple[], NamedTuple[],
                            Dict{Symbol,Tuple{Int,Int}}(), ReentrantLock(),
                            String[], Float64[], String[], Int[],
                            String[], 0,
                            Threads.Atomic{Int}(0), Threads.Atomic{Int}(0),
                            Threads.Atomic{Int}(0))

const VAL_RING_SLOTS      = 64
const VAL_RING_SLOT_BYTES = 2048
const MAX_VALIDATION_MESSAGES = 50
const MAX_PRINTF_MESSAGES     = 4096

"""
    ValidationRingRaw

The pointers `debug_callback` reads from `pUserData` with one `unsafe_load`.
`isbits` because the callback runs on a driver thread, re-entrantly inside a
blocking `ccall` (GPU-AV reads back during `vkWaitSemaphores` with driver locks
held), where allocating or entering the Julia runtime deadlocks or corrupts.
"""
struct ValidationRingRaw
    buf::Ptr{UInt8}
    len::Ptr{Cint}
    sev::Ptr{UInt32}
    typ::Ptr{UInt32}
    write::Ptr{Int}
end

"""
    ValidationRing

One device's validation messages: a slot ring the debug-utils callback fills on
a driver thread, reached through the messenger's `pUserData`, and the drained
strings the main thread reads. Per context because each context has its own
instance and messenger; a shared ring reported one device's errors at another's
call site. The arrays are never resized, so the pointers in `raw` stay valid;
`raw`'s address is the `pUserData`, and the context keeps it alive.
"""
mutable struct ValidationRing
    buf::Vector{UInt8}
    len::Vector{Cint}
    sev::Vector{UInt32}
    typ::Vector{UInt32}
    write::Vector{Int}      # 1 element; total messages written by the callback
    read::Int               # main-thread drain cursor
    raw::Base.RefValue{ValidationRingRaw}
    # Drained, classified output. Hard errors for `check_validation_errors!`;
    # `@lava_printf` text kept separate so a debug print is not an error.
    messages::Vector{String}
    printf::Vector{String}
end

function ValidationRing()
    buf = zeros(UInt8,  VAL_RING_SLOTS * VAL_RING_SLOT_BYTES)
    len = zeros(Cint,   VAL_RING_SLOTS)
    sev = zeros(UInt32, VAL_RING_SLOTS)
    typ = zeros(UInt32, VAL_RING_SLOTS)
    wr  = zeros(Int, 1)
    raw = Ref(ValidationRingRaw(pointer(buf), pointer(len), pointer(sev),
                                pointer(typ), pointer(wr)))
    ValidationRing(buf, len, sev, typ, wr, 0, raw, String[], String[])
end

"""The `pUserData` this ring is reached through. Stable for the ring's lifetime."""
ring_user_data(r::ValidationRing) = Ptr{Cvoid}(pointer_from_objref(r.raw))

"""
    IterPlan

A kernel's launch decomposition: KA context, 3-D block grid, workgroup size and
block count. Declared here so `DeviceCaches.iterplans` can name it. Device
specific: `block_dims` comes from `pad_to_3d(ctx, …)`.
"""
struct IterPlan{Ctx}
    ka_ctx::Ctx
    block_dims::NTuple{3, Int}
    ws_3d::NTuple{3, Int}
    nblocks::Int
end

"""
One cached `IterPlan` and the `(ndrange, workgroupsize)` it was built for.
`iterplans` maps the kernel type to entries scanned linearly: with `K` known at
the call site, `q isa IterEntry{K} && q.key === k` compares an isbits tuple,
where a `Dict` on the heterogeneous key paid ~8 ns of dynamic `hash`/`isequal`
per dispatch. `plan` is `Any`, boxed once at build time and typed behind a
function barrier.
"""
struct IterEntry{K}
    key::K
    plan::Any
end

# `DeviceCaps` is `KernelInterface`'s; `caps(ctx)` fills it. On this backend
# `cores` and `warps` are 0 when unreported (only `VK_NV_shader_sm_builtins` and
# `VK_AMD_shader_core_properties2` report them); read them through
# `shader_core_count` / `shader_warps_per_sm`, which return `nothing` then.
# `coopmatsubgroup` is not `subgroup`: coopmat pipelines are pinned to
# `COOPMAT_SUBGROUP` (`pipeline.jl`), so on RDNA 3.5 (default 64, pinned 32) a
# coopmat workgroup sized in `subgroup` units has twice the threads the kernel indexes.

"""
    CompiledRTPipeline

Supertype of the backend's compiled ray-tracing pipeline, so `DeviceCaches` can
name it: `LavaRTPipeline` holds a `VkContext`, which holds `DeviceCaches`. Not in
core, which never names a compiled RT pipeline.
"""
abstract type CompiledRTPipeline end

"""
    DeviceCaches

Everything a `VkContext` caches. Fields rather than module-level globals: a
cache of device-owned handles must die with its device, so a reset needs no
clearing.
"""
mutable struct DeviceCaches
    # Compute pipelines by SPIR-V content hash; `pipeline_order` is insertion
    # order, for eviction.
    pipelines::Dict{UInt64,LavaComputePipeline}
    pipeline_order::Vector{UInt64}
    # Keyed on the kernel by identity: a recompiled kernel is a new key.
    linked::IdDict{LavaGPUKernel,LavaLinkedKernel}
    # `Vector{Any}`, not `Vector{LaunchPlan}`: a typed vector stores the
    # immutable struct inline and reading one boxes 64 bytes per dispatch, an
    # `Any` element is already a box. (A mutable `LaunchPlan` also avoids it but
    # failed a Lava test.) See `launch_plan`.
    launchplans::IdDict{DataType,Vector{Any}}
    pool::MemoryPolicy
    # 0 means "not yet queried"; the device never reports 0.
    subgroup_size::Int
    subgroup_control::Union{Nothing,SubgroupSizeControl}
    # `maxComputeWorkGroupSize`; (0, 0, 0) until `max_workgroup_dims` queries it
    wg_size_dims::NTuple{3,Int}
    # `nothing` until first read.
    caps::Union{Nothing,DeviceCaps}
    # One warning per device about a subgroup width that cannot be pinned.
    coopmat_warned::Bool
    # Keyed on the shaders a pipeline is made of (`pipeline_cache_key`); the
    # shaders live with their compiled code (`GfxShaders`).
    gfx_pipelines::CompilationCache{Tuple,VulkanCompiledGraphicsPipeline}
    # Keyed like `gfx_pipelines`. Here and not on the user's
    # `RayTracingPipeline`, so a reset drops the handles and identical
    # pipelines compile once.
    rt_pipelines::CompilationCache{UInt64,Tuple{CompiledRTPipeline,LavaRTShader,Vector{Int},Vector{Int}}}
    # The `rt_pipelines` entry a pipeline description with these argument types
    # draws with, and the world it was checked in. A newer world looks the
    # stages up again (`rt_compiled_for`); old entries stay, as a recording may
    # hold their pipeline.
    rt_current::Dict{UInt64,Tuple{UInt,UInt64}}
    timestamp_pool::Union{Nothing,VK.QueryPool}
    timestamp_next_slot::Int
    timestamp_period_ns::Float64
    # `DispatchTiming`s (declared in `profiling.jl`), indexing `timestamp_pool` slots.
    recorded_dispatches::Vector{Any}
    # Kernel type => `IterEntry`s (see there). Per device because `block_dims`
    # depends on the device's workgroup limits.
    iterplans::IdDict{DataType,Vector{Any}}
    reduce_scratch::Any          # one unified cell for `vk_reduce_sum`
    # `Mantle.AccessCache`: per device because the answers come through this
    # device's method table and target features.
    accesses::Any
end

# `MemoryPolicy()` resolves at call time, long after `memory.jl` is loaded.
DeviceCaches() = DeviceCaches(
    Dict{UInt64,LavaComputePipeline}(), UInt64[],
    IdDict{LavaGPUKernel,LavaLinkedKernel}(), IdDict{DataType,Vector{Any}}(),
    MemoryPolicy(), 0, nothing, (0, 0, 0), nothing, false,
    CompilationCache{Tuple,VulkanCompiledGraphicsPipeline}(),
    CompilationCache{UInt64,Tuple{CompiledRTPipeline,LavaRTShader,Vector{Int},Vector{Int}}}(),
    Dict{UInt64,Tuple{UInt,UInt64}}(),
    nothing, 0, 1.0, Any[],
    IdDict{DataType,Vector{Any}}(), nothing, AccessCache())
