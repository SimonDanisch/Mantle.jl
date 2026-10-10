"""
The types `VkContext` has to name, hoisted ahead of it.

Definitions only, no behaviour: these `struct` blocks sit ahead of the functions
that use them so `VkContext` can hold its per-device state in **concrete typed
fields** rather than module-level dictionaries keyed by `ctx.id`. Such a
dictionary's entries outlive the device they describe and need a reset hook to
empty them; state owned by the context needs neither.

Only types have to be hoisted. Constructors and methods stay with their
behaviour, because include order constrains types and not methods — Julia
resolves a call at the first invocation, long after every file is loaded.

One `Any` field: `VkManagedBuffer.ctx` / `last_write`, because a `VkContext`
owns the queue that owns the buffer. That is a genuine cycle rather than an
ordering problem.
"""

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

# `PoolBlock` was here: one 64 MiB slab carved by a bump pointer, with a count of
# its live sub-allocations and a back-reference to the pool that owned it. It is
# `Mantle.Block` now, and the count is `Block.live` — the same fact kept as the
# ledger of what was handed out rather than as a number that has to be kept in
# step with it.

mutable struct VkManagedBuffer
    buffer::VK.Buffer
    memory::VK.DeviceMemory
    address::UInt64     # BDA for PhysicalStorageBuffer access
    mapped_ptr::Ptr{UInt8}  # Non-null for unified/BAR memory
    size::Int
    # Where this buffer's bytes came from, when they came from the suballocator.
    #
    # Two fields before — `pool_offset::Int` and `pool_block::Union{Nothing,
    # PoolBlock}` — which together are what `Mantle.Region` already is: a block
    # and a span inside it. They were the last piece of a SECOND allocator
    # living beside `Mantle.Pool`, with its own blocks, its own free lists and
    # its own idea of what a suballocation is.
    #
    # `nothing` means the buffer owns a `VkDeviceMemory` of its own rather than
    # a slice of one, which is still how a mapped, unified or unusually-flagged
    # buffer is served. That case is what `pool_offset` returning 0 says.
    region::Union{Nothing, Region}
    # What core knows about this buffer's lifetime: which channel last submitted
    # work naming it, under which token, and how many recordings that can still
    # submit hold it. Read and written ONLY by `graph/lifetime.jl` — the backend
    # supplies the storage and `stampof` says where it is, and nothing in this
    # package decides anything from it.
    #
    # Three fields before — `last_write_bq::Any`, `last_write_val::UInt64` and
    # `@atomic pins::Int`, with a `free_requested::Bool` beside them for the
    # free the last unpin owed — each read by backend code that decided
    # something with it: whether to insert a cross-queue wait, whether a
    # destructor could run, whether a free had to be deferred. Those are
    # decisions about SUBMISSIONS and they are core's now.
    #
    # A heap object rather than plain fields, and it costs nothing per store:
    # writing `s.channel` and `s.token` is a pointer store and a word store, the
    # boxing the two-field version existed to avoid (`Union{Nothing,Tuple{Any,
    # UInt64}}`, 48 bytes a submission) is not reintroduced, and `holders` is
    # `@atomic` because two channels on two threads can hold one buffer.
    stamp::Stamp{UInt64}
    # Lifecycle state — see BUF_STATE_* constants above. @atomic CAS is the
    # single point where double-free / use-after-free is ruled out.
    @atomic state::UInt8
    # Owning VkContext — so upload!/download!/vk_free! don't need the global.
    # Loose type because VkContext is declared in device.jl, included first.
    ctx::Any
    # Where this buffer last went into an owner's `sync` list, so `syncbuf!`
    # answers "already listed?" with one comparison instead of a scan. Only a
    # hint: the list is checked at that position, so a stale value (the list was
    # emptied, or the buffer went into another list since) reads as "not listed".
    syncpos::Int
end

VkManagedBuffer(buffer, memory, address, mapped_ptr, size, region, stamp, state, ctx) =
    VkManagedBuffer(buffer, memory, address, mapped_ptr, size, region, stamp, state, ctx, 0)

"""
When one device grows, trims and collects, and what it currently holds.

**Not where the memory comes from.** That is `Mantle.Pool`, reached through this
context's `LavaDevice` — the same pool every graph arena is placed in, with one
free list and one set of blocks for the whole device.

A second allocator here would mean two over one `VkDevice`: each peak would
describe only its own bookkeeping, and an arena and an array could not reuse
each other's bytes however idle one was.

What is left is policy, and it stays per device for the reason the split of these
fields out of eleven module-level `Ref`s was made in the first place: a second
device would otherwise be trimmed, capped and garbage-collected according to the
first one's numbers, and the pressure ratio would divide the SUM of two heaps by
the capacity of one.

**Keep this per device, whatever else changes.** `PoolBlock` once carried no
device, and an allocation on a second device was served out of the first
device's block — measured: allocate on the GPU (one block created), then
allocate on lavapipe, and the block count was *still 1*. The buffer's `ctx` was
right and the memory under it belonged to the other device, which is the worst
shape a bug can have: `fill!` on the second context read back **0.0**, and the
same sequence in another order segfaulted instead. `Mantle.Pool` inherits that
requirement — one pool per `LavaDevice`, one `LavaDevice` per `VkContext`, which
is what `DEVICES` is for.
"""
mutable struct MemoryStats
    # Estimated maximum bytes available to us on the device-local heap.
    # Probed lazily from `ctx.memory_properties` and refreshed every 10s.
    @atomic size::Int
    @atomic last_updated::Float64

    # Last `maybe_collect` run + the rolling cost of that GC.
    @atomic last_time::Float64
    @atomic last_gc_time::Float64
    # Bytes freed by the most recent `maybe_collect`-triggered GC.
    @atomic last_freed::Int
end

MemoryStats() = MemoryStats(0, 0.0, 0.0, 0.0, 0)

mutable struct MemoryPolicy
    # ── Policy. These were module-level `Ref`s, which is the same mistake as the
    # caches one level up: a second device would have been collected according
    # to the first one's numbers. The trim threshold, the soft cap and the
    # collection budget are not here: they are the pool's own policy, core's
    # `PoolPolicy`, which every backend's pool applies.
    disabled::Bool
    track_allocs::Bool
    # Whether the pressure-driven `maybe_collect` runs at all, and the state it
    # rate-limits itself with. Both were module-level — `EAGER_GC` a `Ref{Bool}`
    # and `MEMORY_STATS` a `MemoryStats()` — which made a BACKEND read
    # process-global state to decide what to do while the context sat in its
    # argument list.
    eager_gc::Bool
    stats::MemoryStats

    # ── Accounting for buffers that own their memory: staging, mapped, and
    # anything whose usage flags the pool cannot host. The SUBALLOCATED bytes are
    # `Mantle.reserved(spans(ctx))`, and `gpu_live_bytes` is the two added — a
    # chunk costs nothing beyond the block it sits in, so counting both would
    # double every byte.
    #
    # Per device, as everything here is: module-level, these were one number for
    # two heaps, so the pressure ratio and the OOM retry read the SUM of both
    # devices against ONE device's capacity, and a busy discrete GPU drove
    # collection on an idle integrated one. Atomics because `destroy_buffer!` is
    # reachable from a finalizer.
    live_bytes::Threads.Atomic{Int}
    live_buffers::Set{VkManagedBuffer}
    # Guards re-entry into reclamation through `flush!`'s own allocation path.
    # Per pool: one device quiescing must not make another's reclaim a no-op.
    reclaiming::Threads.Atomic{Bool}
end

# Linked result: session-dependent, stored in the cache Dict.
struct LavaLinkedKernel
    compiled::LavaGPUKernel        # SPIR-V bytes + push_info (serializable)
    pipeline::LavaComputePipeline  # VkPipeline (session-dependent, NOT serializable)
    offsets::Vector{Int}           # arg layout offsets (derived from push_info)
    byval_sizes::Vector{Int}      # LLVM byval sizes (derived from push_info)
end

"""
Everything a dispatch needs that depends only on the *types* of its arguments.

Keyed on `typeof(all_args)` and not on a rebuilt
`Tuple{map(arg_sigtype, tail(all_args))...}`: that interns a fresh `Type`, does
a method lookup and then two hash lookups on slow-hashing types, all to
rediscover an already-compiled pipeline. At ~2000 dispatches per MatAnyone
inference step it is the largest single host cost in the loop.

`typeof(all_args)` is available for free and types are interned, so an `IdDict`
keyed on it is a pointer hash. Everything downstream — pipeline, arg layout,
buffer size — is a function of exactly that, so it is all cached together.

The world counter is stored with the entry and checked on each hit. It moves on
any method definition, so redefining a kernel (Revise, or a first-time
specialisation) drops back to the slow path for one call and re-caches; a stale
pipeline can never be served.
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
    # Pipeline state (for debug/inspection). Plural: with dynamic rendering a
    # pipeline is built for a whole set of colour attachment formats.
    color_formats::Vector{VK.Format}
    has_depth::Bool
    # `:vertex` or `:mesh` — which front end this pipeline was built with, and
    # therefore which draw command is legal against it. Carried on the pipeline
    # because that is where the answer is true: a caller holding one has no
    # other way to ask, and `vkCmdDraw` against a mesh pipeline is undefined
    # behaviour rather than an error the driver reports.
    front::Symbol
end

"""
    Diagnostics

Every debugging and instrumentation toggle Lava has, owned by the context they
describe.

Per context and not per process, like the caches and the pool policy one level
up: a module-level toggle turns allocation tracing or dispatch logging on for
*every* device, so a second device cannot be instrumented independently of the
first, and the counters two of them carry (`spirv_dump_counter`,
`kernel_debug_counter`) would be shared across devices that emit different
kernels. Two differently-instrumented runs at once are possible this way, and
nothing has to be reset.

Off by default, and free when off: every read is a field load behind a branch
the compiler hoists.
"""
mutable struct Diagnostics
    # ── allocation / free
    alloc_debug::Bool
    free_debug::Bool
    freed_bda_scan::Bool
    # Whether the scan POISONS what it finds. On by default because a stale BDA
    # left in a slab is a use-after-free waiting to be submitted, and zeroing it
    # faults on null instead. Off makes the scan a pure observer — which is what
    # separates "the poisoning prevented the hang" from "the scan slowed the
    # render down enough to hide it".
    freed_bda_scan_poisons::Bool
    destroy_freed_bdas_throws::Bool
    presubmit_scan::Bool
    presubmit_scan_throws::Bool
    pack_arg_assert_live::Bool
    # `UInt64`, NOT `Any`. `submit!` gates the slab scan on
    # `slab_dump_target != UInt64(0)`, and the global this replaced was a
    # `Ref{UInt64}(0)`, so that guard read `0 != 0` — false, block skipped. Typed
    # `Any` and defaulted to `nothing`, the same guard reads
    # `nothing != UInt64(0)` — TRUE — and every submit scanned the whole used arg
    # slab with uncached host-visible reads. Silent, because `target` was
    # `nothing` so nothing ever matched and `hits` stayed empty; it only showed up
    # as ~41 us of host CPU per dispatch at submit (SAM 2 decode 7.1 -> 25 ms).
    # A field that is compared against a number has to hold one.
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

    # ── The buffers the flags above fill. They were module-level `Vector`s and
    # `Dict`s, which the census that drove this refactor could not see: it
    # matched `= Ref` and these are collections. Same defect all the same — with
    # two devices every log interleaved both, so the record of what one device
    # allocated, dispatched or waited on was a record of neither.
    alloc_log::Vector{NamedTuple}
    free_log::Vector{NamedTuple}
    freed_bda_scan_log::Vector{NamedTuple}
    slab_dump_log::Vector{NamedTuple}
    # Allocation totals by tag, filled when `track_allocs` is on.
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
    # Lifetime dispatch counters. Atomics: `submit!` may run off the main thread.
    flush_counter::Threads.Atomic{Int}
    total_dispatches::Threads.Atomic{Int}
    # How many closed command buffers opened with the global barrier — one
    # per one-shot and per recording — so a test can say that a launch began
    # with it. See `headbarrier!`.
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

# ── Per-device state ─────────────────────────────────────────────────────────

const VAL_RING_SLOTS      = 64
const VAL_RING_SLOT_BYTES = 2048
const MAX_VALIDATION_MESSAGES = 50
const MAX_PRINTF_MESSAGES     = 4096

# `DebugConfig` is core's: `runtime/lifecycle.jl`. What it does on this backend
# is `VkContext`'s business, which reads every field at instance creation.

"""
    ValidationRingRaw

The five pointers `debug_callback` needs, in a layout it can read off
`pUserData` with one `unsafe_load`.

`isbits`, so that load is a plain memory read: the callback runs on a driver
thread, re-entrantly from inside a blocking `ccall` (GPU-AV reads back during
`vkWaitSemaphores` with driver locks held), where allocating or entering the
Julia runtime deadlocks or corrupts. That constraint is why the ring is raw
preallocated arrays, and not a reason for them to be *module-level*.
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
a driver thread, and the drained strings the main thread reads.

**Per context, not module-level: globals here are a two-device bug.**
`create_vulkan_context` builds a *fresh* `VK.Instance` and
`DebugUtilsMessengerEXT` per context — `VkContext` holds both as fields —
so two contexts meant two messengers writing into ONE ring. Device A's validation
errors surfaced in device B's `get_validation_messages()`, and
`check_validation_errors!` would raise one device's fault at the other device's
call site. The drain cursor was shared too, so whichever device drained first
consumed the other's messages.

`VkDebugUtilsMessengerCreateInfoEXT` carries a `pUserData` pointer for exactly
this, and `debug_callback` already took (and ignored) it.

The arrays are allocated once and never resized, so the pointers cached in `raw`
stay valid; `raw` is a `RefValue` because its own address is what gets handed to
the driver, and Julia's GC does not move objects that something roots — here, the
context.
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

A kernel's launch decomposition: the KA context, the 3-D block grid, the
workgroup size, and the block count.

Lives here rather than beside its only user in `ka_backend.jl` so
`DeviceCaches.iterplans` can name its element type. `block_dims` comes from
`pad_to_3d(ctx, …)`, which reads `ctx.max_wg_dims` — so a plan describes one
device, and the cache holding it has to belong to that device.
"""
struct IterPlan{Ctx}
    ka_ctx::Ctx
    block_dims::NTuple{3, Int}
    ws_3d::NTuple{3, Int}
    nblocks::Int
end

"""
One cached `IterPlan` plus the `(ndrange, workgroupsize)` it was built for.

**Why an entry type instead of a `Dict` key.** A `Dict{Any,IterPlan}` keyed on
`(typeof(obj), ndrange, workgroupsize)` hashes a HETEROGENEOUS tuple per
dispatch — a `DataType` beside an `Int` beside a `Nothing` — through dynamic
`hash`/`isequal`. Measured against the alternatives:

    Dict{Any,IterPlan}       (heterogeneous key)   7.9 ns
    Dict{Any,IterPlan{Ctx}}  (concrete VALUES)     8.1 ns   <- values do not help
    IdDict + linear scan     (pointer compare)     3.2 ns
    concrete-field compare                         ~0 ns

Making the *value* type concrete does nothing for lookup time; the key is the
cost. `K` is a type parameter so the scan can test `q isa IterEntry{K}` — and `K`
is known at the call site, because the caller has `ndrange` in hand — after which
`q.key === k` is an `===` on a concrete isbits tuple rather than a hash.

`plan` stays `Any` on purpose: it is boxed exactly once at build time, and the
launch path types it with a single function barrier. A `Vector{IterEntry{K}}`
would store entries INLINE and re-box on every read, which is the trap
`DeviceCaches.launchplans` documents.
"""
struct IterEntry{K}
    key::K
    plan::Any
end

# `DeviceCaps` and `supports`/`bestshape` are `KernelInterface`'s, the module
# Lava and Mantle both implement, so there is one definition of the type and
# `Lava.jl` imports it. `caps(ctx)` below fills it in.
#
# The docstring that stood here argued the case for the fields, and it went with
# the type. Two Lava-specific notes it carried did not, so they are kept:
#
#   * `cores` and `warps` are 0 when the device does not report them. Vulkan has
#     no core query: NVIDIA exposes it through `VK_NV_shader_sm_builtins`, AMD
#     through `VK_AMD_shader_core_properties2`, everyone else through nothing.
#     Read them via `shader_core_count` / `shader_warps_per_sm`, which return
#     `nothing` for unknown — a missing fallback is then a `MethodError` at the
#     first arithmetic rather than a silently empty grid.
#
#   * `coopmatsubgroup` is NOT `subgroup`. Lava pins any module declaring
#     `CooperativeMatrixKHR` to `COOPMAT_SUBGROUP` through
#     `VK_EXT_subgroup_size_control` (`pipeline.jl`), because a cooperative
#     matrix is subgroup-scoped and the kernels index subgroups as `tid ÷ 32`. On
#     RDNA 3.5 the device default is 64 while the pipeline still runs 32 — a
#     coopmat workgroup sized in units of `subgroup` would ask for twice the
#     threads the kernel indexes, and write part of its tile.

"""
    CompiledRTPipeline

Supertype of a backend's compiled ray-tracing pipeline, declared here so
`DeviceCaches` can name it.

The forward reference is unavoidable rather than incidental: `LavaRTPipeline`
holds the `VkContext` it was built against, `VkContext` holds a `DeviceCaches`,
and `DeviceCaches` caches `LavaRTPipeline`s — so one of the three has to be
named before it is defined. `IterPlan{Ctx}` in this same file breaks the same
cycle with a type parameter; a supertype is the cheaper break here, because the
cache is a `Dict` value read a handful of times per frame and nothing indexes it
in a hot loop.

Not in Mantle core beside `CompiledGraphicsPipeline`: core never names a
compiled RT pipeline, and Metal compiles its intersection functions without
caching them at all.
"""
abstract type CompiledRTPipeline end

"""
    DeviceCaches

Everything a `VkContext` caches, owned by the context that owns the handles.

Fields and not module-level globals, keyed by `ctx.id` or otherwise: a cache
holding a device-owned handle outlives the device it describes, which is the
shape that produces the function-pointer crash and the memory-pool corruption. A
field dies with its context, so nothing has to empty these on reset.

Core's `BLIT_PIPELINE` is a pipeline DESCRIPTION and holds no handle; the
compiled pipeline it turns into is keyed on the device in `gfx_pipelines`, like
every other draw.
"""
mutable struct DeviceCaches
    # Compute pipelines, keyed by SPIR-V content hash. No `ctx.id` in the key:
    # two devices cannot collide when they do not share a dict.
    pipelines::Dict{UInt64,LavaComputePipeline}
    pipeline_order::Vector{UInt64}
    # `Dict{Any,…}` because GPUCompiler.cached_compilation derives the key itself.
    # This device's pipeline for each compiled kernel, keyed on the kernel by
    # identity: the SPIR-V is kept with its `CodeInstance` (`compile_or_lookup`),
    # so a kernel compiled again is a new key.
    linked::IdDict{LavaGPUKernel,LavaLinkedKernel}
    # `Vector{Any}`, NOT `Vector{LaunchPlan}` — and the difference is one heap
    # allocation on EVERY dispatch, on the path that HITS this cache.
    #
    # `LaunchPlan` is an immutable struct with reference fields, so it is not
    # `isbitstype`. Julia stores such structs INLINE in a typed `Vector`, which
    # means pulling one out has to materialise it — a 64-byte box per read, even
    # though the plan was already built and nothing is being constructed:
    #
    #     scan over Vector{Immut}  ->  64 bytes
    #     scan over Vector{Any}    ->   0 bytes   (element is already a box;
    #                                             reading hands back a pointer)
    #
    # A `Vector{Any}` holding one concrete type reads as sloppy, so: the
    # alternative is making `LaunchPlan` mutable, which also measures 0 bytes but
    # changes the type's semantics for every holder and failed a Lava test. This
    # keeps the struct immutable and pays a `::LaunchPlan` typeassert on read,
    # which is free. See `launch_plan`.
    launchplans::IdDict{DataType,Vector{Any}}
    pool::MemoryPolicy
    # 0 means "not yet queried" — the device never reports 0.
    subgroup_size::Int
    subgroup_control::Union{Nothing,SubgroupSizeControl}
    # `maxComputeWorkGroupSize`; (0, 0, 0) until `max_workgroup_dims` queries it
    wg_size_dims::NTuple{3,Int}
    # What kernels ask the device — see `DeviceCaps`. `nothing` until first read;
    # built from this context, so a second device cannot be handed the first's.
    caps::Union{Nothing,DeviceCaps}
    # One warning per device about a subgroup width that cannot be pinned.
    coopmat_warned::Bool
    # Keyed on the shaders a pipeline is made of (`pipeline_cache_key`). The
    # shaders themselves are not a device's: GPUCompiler keeps them with the
    # code they were compiled from (`GfxShaders`).
    gfx_pipelines::Dict{UInt64,VulkanCompiledGraphicsPipeline}
    # Ray-tracing pipelines, here for the same reason the graphics ones are: a
    # compiled pipeline is a device object, so it belongs to the device.
    #
    # Not on the user's `RayTracingPipeline`: one that outlives a
    # `reset_device!` would hold handles from a dead device, and two identical
    # pipelines would compile twice.
    #
    # Keyed like `gfx_pipelines`: the shader identities plus the argument types,
    # since the argument types alone do not identify a pipeline.
    #
    # `CompiledRTPipeline` and not `LavaRTPipeline`: see the supertype above for
    # why the concrete name cannot be spelled here.
    rt_pipelines::Dict{UInt64,Tuple{CompiledRTPipeline,LavaRTShader,Vector{Int},Vector{Int}}}
    # Which entry of `rt_pipelines` a pipeline description, called with these
    # argument types, draws with now, and the world age that was checked in.
    # In a newer world its stages are looked up again (`rt_compiled_for`), and an
    # edited stage makes a new entry; the old one stays, since a recording may
    # still hold its pipeline.
    rt_current::Dict{UInt64,Tuple{UInt,UInt64}}
    timestamp_pool::Union{Nothing,VK.QueryPool}
    timestamp_next_slot::Int
    timestamp_period_ns::Float64
    # The dispatches this device timestamped. `Any` because `DispatchTiming` is
    # declared in `profiling.jl`; the slots it indexes are `timestamp_pool`'s, so
    # a module-level vector was a list of one device's slot numbers read against
    # whichever device's pool happened to be current.
    recorded_dispatches::Vector{Any}
    # Launch decompositions, keyed by (kernel type, ndrange, workgroupsize) — a
    # key that does NOT name the device, while the value does: `block_dims` is
    # `pad_to_3d(ctx, …)` over `ctx.max_wg_dims`. Two devices with different
    # `maxComputeWorkGroupCount` therefore handed the second one the first one's
    # block grid. Same class as the caches above; it survived the first sweep
    # because that census matched `= Ref` and this is a `Dict`.
    # `IdDict` keyed by kernel TYPE (a pointer compare), holding `IterEntry`
    # values scanned linearly — not a `Dict` on a heterogeneous tuple. See
    # `IterEntry` for the measurements.
    iterplans::IdDict{DataType,Vector{Any}}
    # Scratch buffers. These hold DEVICE MEMORY, so a process-wide one is the
    # defect the memory pool had: the second device is handed the first's buffer.
    # Both were already keyed by context, which is the surrogate a field replaces.
    reduce_scratch::Any          # one unified cell for `vk_reduce_sum`
    gemm_split_scratch::Any      # split-K partials; grows, never shrinks
    # Grown-past scratch, retained rather than dropped: dispatches already
    # recorded point into the old buffer and its finalizer would pull it out
    # from under them.
    gemm_split_retired::Vector{Any}
    # What each kernel signature does to its arguments — see `Mantle.AccessCache`.
    # Keyed by signature and not by device, like the pipeline caches above, and a
    # field for the same reason: the answers are derived through THIS device's
    # method table and its target features.
    accesses::Any
end

# `MemoryPolicy()` resolves at call time, long after `memory.jl` is loaded.
DeviceCaches() = DeviceCaches(
    Dict{UInt64,LavaComputePipeline}(), UInt64[],
    IdDict{LavaGPUKernel,LavaLinkedKernel}(), IdDict{DataType,Vector{Any}}(),
    MemoryPolicy(), 0, nothing, (0, 0, 0), nothing, false,
    Dict{UInt64,VulkanCompiledGraphicsPipeline}(),
    Dict{UInt64,Tuple{CompiledRTPipeline,LavaRTShader,Vector{Int},Vector{Int}}}(),
    Dict{UInt64,Tuple{UInt,UInt64}}(),
    nothing, 0, 1.0, Any[],
    IdDict{DataType,Vector{Any}}(), nothing, nothing, Any[], AccessCache())
