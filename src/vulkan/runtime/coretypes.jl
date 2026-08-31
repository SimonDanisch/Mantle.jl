"""
The types `VkContext` has to name, hoisted ahead of it.

Nothing here is new and nothing here has behaviour — these are the `struct`
blocks that used to sit beside the functions that use them, moved so that
`VkContext` can hold its per-device state in **concrete typed fields** instead of
module-level dictionaries keyed by `ctx.id`.

That keying was the shortcut: it made two devices work without touching call
sites, and the probe that followed found seven real defects. But it left twelve
globals whose entries outlive the device they describe, and a `RESET_CALLBACKS`
mechanism whose entire job was emptying them. State owned by the context needs
neither.

Only the definitions moved. Every constructor, method and comment about
*behaviour* stayed where it was, because include order constrains types and not
methods — Julia resolves a call at the first invocation, long after every file
is loaded.

One `Any` field survives, and it predates this: `VkManagedBuffer.ctx` /
`last_write`, because a `VkContext` is what owns the queue that owns the buffer.
That is a genuine cycle; the rest were only ordering.

`PoolBlock` used to be the second, and it is gone — with the size-class
allocator it belonged to. A suballocation is a `Mantle.Region` now, which is the
one Mantle already hands to every graph arena.
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
    # Cross-queue synchronization: records which VulkanBatchQueue last wrote to
    # this buffer, at which timeline value. Consumed by sync_access! to
    # auto-insert semaphore waits when a dispatch on a different queue
    # takes this buffer as an argument. Nothing = never written.
    # Typed as Any so VulkanBatchQueue (defined later) doesn't force a cyclic include.
    # @atomic so the finalizer thread (vk_free!) and main thread (record /
    # sync_access!) can read/write it safely.
    @atomic last_write::Union{Nothing, Tuple{Any, UInt64}}
    # Lifecycle state — see BUF_STATE_* constants above.  @atomic CAS is the
    # single point where double-free / use-after-free is ruled out.
    @atomic state::UInt8
    # Number of live CommandBatches that have `pin!`ed an array backed by this
    # buffer.  Incremented at pin time, decremented when the batch releases its
    # pins (`release_pinned_refs!`, i.e. the batch completed or its submit
    # failed).  A buffer with pins > 0 is REACHABLE BY A BATCH THAT CAN STILL
    # SUBMIT, so `vk_free!` must not touch it — not even to mark it DEFERRED,
    # because `sync_access!` asserts the buffer is ALIVE at submit.
    #
    # This is what makes the guarantee structural rather than a timing accident:
    # `last_write` only tells us about work already *submitted*, so a buffer
    # pinned into a still-open batch looks idle to the timeline check.
    @atomic pins::Int
    # A free was requested while pins > 0.  The free is not lost, just owed: the
    # last `unpin_buffer!` performs it.
    @atomic free_requested::Bool
    # Owning VkContext — so upload!/download!/vk_free! don't need the global.
    # Loose type because VkContext is declared in device.jl, included first.
    ctx::Any
end

"""
When one device grows, trims and collects, and what it currently holds.

**Not where the memory comes from.** That is `Mantle.Pool`, reached through this
context's `LavaDevice` — the same pool every graph arena is placed in, with one
free list and one set of blocks for the whole device.

This type used to be that pool as well: `blocks::Vector{PoolBlock}` and
`free_lists`, a bump allocator with 153 size classes, sitting beside
`Mantle.Pool` and invisible to it. Two allocators over one `VkDevice` meant the
peak either could report was about its own bookkeeping, and an arena and an array
could not reuse each other's bytes however idle one of them was.

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
mutable struct MemoryPolicy
    # ── Policy. These were eleven module-level `Ref`s, which is the same mistake
    # as the caches one level up: a second device would have been trimmed,
    # capped and garbage-collected according to the first one's numbers. They are
    # defaults, so they stay mutable — but they are this pool's defaults.
    disabled::Bool
    soft_cap::Int
    trim_threshold::Int
    trim_min_interval::Float64
    trim_full_gc_interval::Float64
    gc_mingap::Float64
    gc_full_mingap::Float64
    track_allocs::Bool

    # ── Bookkeeping: when this pool last trimmed or collected, and how long it
    # has spent doing it. Per device for the obvious reason — one device's
    # collection must not suppress another's.
    last_trim::Float64
    last_full_gc::Float64
    gc_last::Float64
    gc_full_last::Float64
    gc_seconds::Float64

    # ── Accounting for buffers that own their memory: staging, mapped, and
    # anything whose usage flags the pool cannot host. The SUBALLOCATED bytes are
    # `Mantle.reserved(spans(ctx))`, and `gpu_live_bytes` is the two added — a
    # chunk costs nothing beyond the block it sits in, so counting both would
    # double every byte.
    #
    # Per device, as everything here is: module-level, these were one number for
    # two heaps, so the pressure ratio, the trim threshold and the OOM retry all
    # read the SUM of both devices against ONE device's capacity, and a busy
    # discrete GPU drove collection on an idle integrated one. Atomics because
    # `destroy_buffer!` is reachable from a finalizer.
    live_bytes::Threads.Atomic{Int}
    live_buffers::Set{VkManagedBuffer}
    # `requested` / `rounded` / `nalloc` and the `accounting` flag that gated
    # them are gone with the size classes. They existed to measure rounding
    # waste — how much bigger a size class was than the request — and
    # `Mantle.carve!` splits at exactly the requested length, so the ratio they
    # reported is now 1.0 by construction. `pool_gc_stats` is what is left.
    gc_count::Threads.Atomic{Int}
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

`ka_launch!` used to rebuild `Tuple{map(arg_sigtype, tail(all_args))...}` on
every single dispatch and hand it to `GPUCompiler.methodinstance`: that interns
a fresh `Type` object, does a method lookup, and then two hash lookups keyed on
that type and a freshly built compiler config — all to rediscover a pipeline it
had already compiled. Types hash and compare slowly, and at ~2000 dispatches per
MatAnyone inference step this was the largest single host cost in the loop.

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
The compiled prepare-indirect kernel for one device.

It owns a `VkPipeline`, so it is per device (`GUARDRAILS.md` §8) — four separate
`Ref`s before, which meant four things that had to be reset together and were
reachable from the wrong device in exactly the same way. Bundling them makes the
per-device dict hold one value instead of four parallel ones.
"""
struct PrepareIndirect
    pipeline::LavaComputePipeline
    offsets::Vector{Int}
    byval_sizes::Vector{Int}
    arg_buffer_size::Int
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
end

"""
    Diagnostics

Every debugging and instrumentation toggle Lava has, owned by the context they
describe.

**These were eighteen module-level `Ref`s.** They are the same mistake as the
caches and the pool policy one level up, with a milder symptom: turning on
allocation tracing or dispatch logging did it for *every* device in the process,
and a second device could not be instrumented independently of the first. Two of
them carry counters (`spirv_dump_counter`, `kernel_debug_counter`) whose values
were shared across devices that emit different kernels.

`DNNKernels` did exactly this in its own step 3 — `OPTIMES`, `OPDOUBLE`,
`OPDOUBLEFILTER`, `PLAN_MISSES` and `LAUNCH_PROBE` became `Ctx.diag` — and the
argument there applies here: two differently-instrumented runs at once become
possible, and nothing has to be reset.

Off by default, and free when off: every read is a field load behind a branch the
compiler hoists, which is what the `Ref`s cost too.
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
    frozen_log_misses::Bool
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
end

Diagnostics() = Diagnostics(false, false, false, true, false, false, false, false, UInt64(0),
                            false, false, nothing, false,
                            false, true, nothing, 0, 0, nothing,
                            NamedTuple[], NamedTuple[], NamedTuple[], NamedTuple[],
                            Dict{Symbol,Tuple{Int,Int}}(), ReentrantLock(),
                            String[], Float64[], String[], Int[],
                            String[], 0,
                            Threads.Atomic{Int}(0), Threads.Atomic{Int}(0))

# ── Per-device state ─────────────────────────────────────────────────────────

const VAL_RING_SLOTS      = 64
const VAL_RING_SLOT_BYTES = 2048
const MAX_VALIDATION_MESSAGES = 50
const MAX_PRINTF_MESSAGES     = 4096

"""
    DebugConfig(; validation, gpu_av, gpu_av_safe, gpu_av_shaders,
                  sync_val, best_practices, printf, pool_disabled)

Every validation and instrumentation setting, chosen **at device construction**.

    reset_device!(debug = DebugConfig(gpu_av = true))   # replace the default device
    ctx = VkContext(debug = DebugConfig(gpu_av = true))         # or build a separate one

**That is the whole API.** There is no `enable_gpu_av()`, no environment
variable, and no way to switch any of this on after the fact — all of these are
properties of the `VkInstance`, fixed by `vkCreateInstance`, so a setting applied
to a device that already exists cannot take effect. There were five preset
functions and seven `LAVA_*` variables; every one of them is deleted, because
each was another way to reach a configuration slightly different from the one you
asked for.

The two recipes worth knowing:

    DebugConfig(validation = true)                       # core spec checks, cheap
    DebugConfig(gpu_av = true, pool_disabled = true)     # + shader OOB, sub-pool visible

After the second, call [`verify_gpu_av`](@ref). "GPU-AV is enabled" and "GPU-AV
is catching errors" are not the same thing on every driver, and a clean run under
an instrument that never fired reads exactly like a clean run.

This was seven environment variables — `LAVA_VALIDATION`, `LAVA_GPU_AV`,
`LAVA_GPU_AV_SAFE`, `LAVA_GPU_AV_SHADERS`, `LAVA_SYNC_VAL`, `LAVA_BEST`,
`LAVA_DEBUG_PRINTF` — read here at instance creation. They are **deleted**, not
deprecated, because the failure they produced was not occasional: setting one in
an already-running session did nothing at all, and setting `LAVA_GPU_AV` without
`LAVA_VALIDATION` produced a *clean run with the instrument switched off*, which
reads exactly like "no fault found". Both are unrepresentable here.

## The two rules, enforced in the constructor rather than warned about

`validation` is implied by everything else. GPU-AV, sync validation,
best practices and debug printf are all features **of** the Khronos validation
layer, so asking for one turns the layer on. Passing `validation = true` alone
means core spec checks with no shader instrumentation, which is the cheap mode.

`gpu_av` and `printf` are mutually exclusive and **throw** together. The layer
instruments shaders for each and cannot do both; the previous behaviour was to
silently prefer printf and warn, which is one more way to get a clean run out of
a disabled instrument.

## Two settings that exist because GPU-AV crashes

`gpu_av_safe` defaults to **true**, unlike everything else here. Khronos' own
documentation: "Safe Mode will have GPU-AV try and prevent crashes, but will be
much slower to validate, and when using Safe Mode, selective shader
instrumentation is recommended to only instrument the shaders/pipelines causing
issues." Lava shipped GPU-AV with neither for a long time, which is most of why
reaching for it produced a SIGSEGV rather than a report.

`gpu_av_shaders` names the kernels to instrument; empty means all of them, which
on a ~99-kernel model is both very slow and the configuration most likely to fall
over. Both need `VK_EXT_layer_settings` — `VkValidationFeaturesEXT` cannot
express either.

## `pool_disabled`, which is here because it is a debugging setting

GPU-AV tracks buffer-device-address bounds **per `VkBuffer`**, and Lava's pool
puts many `LavaArray`s into one shared 64 MiB block. So with the pool on, GPU-AV
sees overruns past the whole block and is *blind to sub-pool overruns* — which
are most of the bugs anyone turns it on to find. `pool_disabled = true` gives
every array its own `VkBuffer`.

Debug-only: allocation is much slower and the path is less exercised than the
pooled one (the host-upload and flush-after-error paths may misbehave). It lives
on this struct rather than as a separate `mempolicy(ctx).disabled = true` step because
a second step is a second way to get it wrong, and it applies to the device being
built.
"""
struct DebugConfig
    validation::Bool
    gpu_av::Bool
    gpu_av_safe::Bool
    gpu_av_shaders::Vector{String}
    sync_val::Bool
    best_practices::Bool
    printf::Bool
    pool_disabled::Bool

    function DebugConfig(; validation::Bool = false,
                           gpu_av::Bool = false,
                           gpu_av_safe::Bool = true,
                           gpu_av_shaders::AbstractVector{<:AbstractString} = String[],
                           sync_val::Bool = false,
                           best_practices::Bool = false,
                           printf::Bool = false,
                           pool_disabled::Bool = false)
        if gpu_av && printf
            throw(ArgumentError("""
                DebugConfig: `gpu_av` and `printf` cannot both be on — the validation
                layer instruments shaders for each and does not do both at once.
                Pick one: `DebugConfig(gpu_av = true)` to hunt out-of-bounds accesses,
                or `DebugConfig(printf = true)` to read `@lava_printf` output."""))
        end
        # Implied, not required: every feature below is a feature OF the layer, so
        # asking for one without it was the half-configuration that produced a
        # clean run from a disabled instrument.
        validation |= gpu_av || sync_val || best_practices || printf
        new(validation, gpu_av, gpu_av_safe, collect(String, gpu_av_shaders),
            sync_val, best_practices, printf, pool_disabled)
    end
end

"""
    DebugConfig(c::DebugConfig; kw...) -> DebugConfig

`c` with named settings replaced. The two rules above are re-checked, so a copy
cannot reach a state the constructor refuses.
"""
DebugConfig(c::DebugConfig;
            validation = c.validation, gpu_av = c.gpu_av,
            gpu_av_safe = c.gpu_av_safe, gpu_av_shaders = c.gpu_av_shaders,
            sync_val = c.sync_val, best_practices = c.best_practices,
            printf = c.printf, pool_disabled = c.pool_disabled) =
    DebugConfig(; validation, gpu_av, gpu_av_safe, gpu_av_shaders,
                  sync_val, best_practices, printf, pool_disabled)

"""
    ValidationRingRaw

The five pointers `debug_callback` needs, in a layout it can read off
`pUserData` with one `unsafe_load`.

`isbits`, so that load is a plain memory read: the callback runs on a driver
thread, re-entrantly from inside a blocking `ccall` (GPU-AV reads back during
`vkWaitSemaphores` with driver locks held), where allocating or entering the
Julia runtime deadlocks or corrupts. That constraint is why the ring was raw
preallocated arrays — it is not a reason for them to be *module-level*, which is
what this replaces.
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

**This was eight module-level globals, and that was a two-device bug rather than
untidiness.** `create_vulkan_context` builds a *fresh* `VK.Instance` and
`DebugUtilsMessengerEXT` per context — `VkContext` already holds both as fields —
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

**Why an entry type instead of a `Dict` key.** The cache was
`Dict{Any,IterPlan}` keyed on `(typeof(obj), ndrange, workgroupsize)`, so every
dispatch hashed a HETEROGENEOUS tuple — a `DataType` beside an `Int` beside a
`Nothing` — through dynamic `hash`/`isequal`. Measured against the alternatives:

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

# `DeviceCaps` used to be defined here, together with `supports`/`bestshape` over
# it. It moved to `KernelInterface`: it was written twice, field for field — once
# here and once in Mantle — and bridged by a positional copy, because neither
# could depend on the other. KI is the module both already implement, so the type
# is there and `Lava.jl` imports it. `caps(ctx)` below still fills it in; what
# left is the definition, not the query.
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

**These were twelve module-level globals.** Ten had been keyed by `ctx.id` to
make two devices work, which fixed the sharing but not the ownership: entries
outlived the device they described, `ctx.id` was a surrogate for "the object I
should have stored this on", and `RESET_CALLBACKS` existed almost entirely to
empty them. A field dies with its context, so none of that is needed.

The last two — `BLIT_PIPELINE` and `TIMESTAMP_POOL` — were never keyed at all.
They are module-level `Ref`s holding device-owned handles, which is precisely the
shape that produced the function-pointer crash and the memory-pool corruption;
they survived only because the two-device probe's path (dispatch, reduction,
GEMM) reaches neither graphics nor dispatch profiling.

`blit` is `Any` because `GraphicsPipeline` is defined near the end of the include
list — no worse than the `Ref{Any}` it replaces, and the only field that is not
concrete.
"""
mutable struct DeviceCaches
    # Compute pipelines, keyed by SPIR-V content hash. The key no longer needs
    # `ctx.id` mixed in: two devices cannot collide when they do not share a dict.
    pipelines::Dict{UInt64,LavaComputePipeline}
    pipeline_order::Vector{UInt64}
    # `Dict{Any,…}` because GPUCompiler.cached_compilation derives the key itself.
    linked::Dict{Any,LavaLinkedKernel}
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
    prepare_indirect::Union{Nothing,PrepareIndirect}
    pool::MemoryPolicy
    # 0 means "not yet queried" — the device never reports 0.
    subgroup_size::Int
    subgroup_control::Union{Nothing,SubgroupSizeControl}
    # What kernels ask the device — see `DeviceCaps`. `nothing` until first read;
    # built from this context, so a second device cannot be handed the first's.
    caps::Union{Nothing,DeviceCaps}
    # One warning per device about a subgroup width that cannot be pinned.
    coopmat_warned::Bool
    gfx_pipelines::Dict{UInt64,VulkanCompiledGraphicsPipeline}
    gfx_shaders::Dict{UInt64,LavaGfxShader}
    # Ray-tracing pipelines, here for the same reason the graphics ones are: a
    # compiled pipeline is a device object, so it belongs to the device.
    #
    # It used to be a `PIPELINE_CACHE` field on the user's `RayTracingPipeline`,
    # which is the one place it could not correctly live — two identical
    # pipelines compiled twice, and a pipeline outliving a `reset_device!` held
    # handles from a dead device. The no-op `invalidate_stale_rt_cache!` beside
    # it already said so: "cache is tied to the current VkContext's lifetime".
    #
    # Keyed like `gfx_pipelines`: the shader identities plus the argument types.
    # The per-object dict got away with keying on argument types alone because
    # the object WAS the rest of the key.
    #
    # `CompiledRTPipeline` and not `LavaRTPipeline`: see the supertype above for
    # why the concrete name cannot be spelled here.
    rt_pipelines::Dict{UInt64,Tuple{CompiledRTPipeline,LavaRTShader,Vector{Int},Vector{Int}}}
    blit::Any
    timestamp_pool::Union{Nothing,VK.QueryPool}
    timestamp_next_slot::Int
    timestamp_period_ns::Float64
    # The dispatches this device timestamped. `Any` because `DispatchTiming` is
    # declared in `profiling.jl`; the slots it indexes are `timestamp_pool`'s, so
    # a module-level vector was a list of one device's slot numbers read against
    # whichever device's pool happened to be current.
    recorded_dispatches::Vector{Any}
    # The frozen kernel cache's session memo, holding `LavaLinkedKernel`s — each
    # of which owns a `VkPipeline`. Same §8 class as the caches above, and missed
    # for the same reason `BLIT_PIPELINE` was: nothing on the two-device probe's
    # path enables the frozen cache. Its RT sibling `FROZEN_RT_MEM` stays a
    # global on purpose — a `LavaRTShader` is bytes and metadata, no handle.
    frozen_mem::Dict{Tuple{DataType,DataType,Any},Any}
    # A baton, not a cache: `frozen_link_recording` leaves the per-kernel
    # `VkPipelineCache` here for `frozen_store` to snapshot. Narrow window, but a
    # driver object all the same, and two devices recording at once would swap it.
    frozen_last_pcache::Any
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
end

# `MemoryPolicy()` resolves at call time, long after `memory.jl` is loaded.
DeviceCaches() = DeviceCaches(
    Dict{UInt64,LavaComputePipeline}(), UInt64[],
    Dict{Any,LavaLinkedKernel}(), IdDict{DataType,Vector{Any}}(),
    nothing, MemoryPolicy(), 0, nothing, nothing, false,
    Dict{UInt64,VulkanCompiledGraphicsPipeline}(), Dict{UInt64,LavaGfxShader}(),
    Dict{UInt64,Tuple{CompiledRTPipeline,LavaRTShader,Vector{Int},Vector{Int}}}(),
    nothing, nothing, 0, 1.0, Any[],
    Dict{Tuple{DataType,DataType,Any},Any}(), nothing,
    IdDict{DataType,Vector{Any}}(), nothing, nothing, Any[])
