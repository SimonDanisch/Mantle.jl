# Vulkan device initialization for Lava.jl
#
# Singleton VkContext holds all persistent Vulkan state.
# Lazy initialization: first use triggers device creation.
#
# Required features: BufferDeviceAddress, VariablePointers, Int64, Float64
# Optional features: AccelerationStructure, RayTracingPipeline

"""
    RTPipelineProperties

Ray tracing pipeline properties queried from the physical device.
`nothing` if RT extensions are not available.
"""
struct RTPipelineProperties
    shader_group_handle_size::UInt32
    shader_group_base_alignment::UInt32
    shader_group_handle_alignment::UInt32
    max_ray_recursion_depth::UInt32
    max_ray_hit_attribute_size::UInt32
end

# ── What the queue is handed ─────────────────────────────────────────────────
#
# Two kinds of GPU work, and both are CLOSED: a command buffer is never open
# when a Mantle call returns. A `Recording` is a plan's, written once at
# `record!`, owned by the plan and submitted every run; a `OneShot` is written,
# sealed and handed over inside one call by whoever needed it, owned by the
# submission that carries it and given back to the queue's pool when the
# timeline passes. `submit!(bq, closed...)` is the only way either reaches the
# driver, and it goes the moment it is called.
#
# This replaces the command batch: one command buffer open across call boundaries
# that anything could append to — an ad hoc launch, an upload, a readback, a
# per-run store, a present transition — with a dispatch count deciding whether
# the next launch needed a barrier, a segment count deciding when to cut the
# buffer, a threshold deciding when to submit it, and three lists of command
# buffers saying which of them it owned. What the open buffer contained
# depended on every call since it was opened, from any package, in any order;
# a plan is the opposite, and the batching the buffer bought is the graph's
# job. See `oneshot!` in `command.jl`.

abstract type Closed end

"""
    OneShot

Work submitted once: an ad hoc launch, an upload, a readback, a run's host
stores and pointer patches, a frame. Written and sealed inside one call
(`oneshot`), owned by the [`Submission`](@ref) that carries it, and returned to
the queue's pool by the sweep once the timeline has passed it — which is when
its pins, its retained `DataRef`s and its scratch regions go back too.

`bq` is the `VulkanBatchQueue{VkContext}` it was taken from; typed `Any`
because the queue is declared below this, and asserted at every read
(`queueof`) so the field is a static load and not a boxed one.
"""
mutable struct OneShot <: Closed
    bq::Any
    cmd::VK.CommandBuffer
    # Everything the commands name, kept alive until the submission that
    # carries this one-shot has passed. `pinned_refs` are the retained
    # `DataRef`s of every pinned `LavaArray` — see `pin!(::LavaArray)`.
    pinned::Base.IdSet{Any}
    pinned_refs::Vector{Any}
    # The `Unified` regions the commands read their arguments and workgroup
    # counts from. Owned here, so an address handed to `get_arg_buffer` is
    # valid for exactly as long as these commands can run.
    regions::Vector{Region}
    open::Bool
end

"""
    Recording

A plan's work, written into ONE command buffer once and handed to the queue
every time the plan runs. Nothing here is heuristic: the graph decided the pass
order, the barriers and where the arguments live, and [`Emitter`](@ref) writes
exactly that. There is no threshold to cut the buffer on, because a recording
is not being submitted while it is being written.

What a recording OWNS is the whole of its lifetime rule. The command buffer,
the argument regions its dispatches read, the descriptor sets they bind and a
strong reference to every resource they name stay alive until `release!`,
because until then the recording may be submitted again.

This replaced `CapturedSequence`, which held a LIST of command buffers because
the thing that produced it was the open batch's `submit!`: a capture ran the
ordinary recorder and collected whatever segments the queue's split and submit
thresholds happened to cut. Measured on Hikari's fused sample:
five command buffers per recording, decided by a threshold about when to
submit, applied while nothing was being submitted.
"""
mutable struct Recording <: Closed
    bq::Any
    cmd::VK.CommandBuffer
    pinned::Base.IdSet{Any}
    pinned_refs::Vector{Any}
    # The `Unified` regions the emitted commands read. A plan's arguments live in
    # its own `ArgMemory`, so this is usually empty — it is here because "who may
    # hand these bytes out again" has to have exactly one answer per owner.
    regions::Vector{Region}
    # Descriptor sets the emitted commands bind, one per (pipeline layout,
    # acceleration structure) pair rather than one per dispatch — see
    # [`tlasset!`](@ref).
    sets::Vector{Any}
    # What covers the last submission of this recording, or 0 if it has never
    # been handed over. `release!` waits on it, which is the whole reason it is
    # kept: the command buffer goes back to the device there, and freeing one
    # the device is still reading is undefined behaviour that presents as a
    # driver-side abort with no Julia frame. UInt64, not Any — the assignment
    # is on the `run!` path, which allocates nothing, and an `Any` field boxes
    # the timeline value it is handed. 0 is safe as "never": the first signal
    # a queue emits is 1.
    token::UInt64
    open::Bool
    # Where each device address the pack wrote landed: (address, host pointer).
    # The pointer is into the plan's argument memory or into one of this
    # recording's own scratch regions (`emitkernel!` packs a prepare kernel's
    # arguments there) — `record!` sorts out which when it builds the patch
    # table (`patchtarget`). Filled by `recpatch!` while the commands are
    # emitted, read once by `record!`. A one-shot never needs it — its commands
    # run once — so the hook is a no-op there.
    patches::Vector{Tuple{UInt64,Ptr{UInt8}}}
    # The VkManagedBuffers a submission of this recording has to `sync_access!`,
    # snapshotted once at `record!` from `pinned` and `pinned_refs`. A run
    # iterates THIS — a concrete vector — rather than the `IdSet{Any}` of pins,
    # which boxed on every element: ~640 bytes a submission on a plan that pins
    # a ray-tracing acceleration structure's two dozen handles. The set of
    # buffers is fixed for the recording's life (pins happen during `emit!`),
    # so the snapshot cannot drift the way a per-run walk of `pinned` could not
    # either — it is the same buffers, un-boxed.
    sync::Vector{VkManagedBuffer}
end

"""
    Submission

What one `vkQueueSubmit2` carried: the one-shots it OWNS, given back to the
queue's pool when the timeline passes `signal_value`; the recordings it PINNED,
a plan's, submitted again on the next run, of which the sweep only drops the
reference; the cross-queue waits `sync_access!` collected over their pins; and
the refill storage the raw submit call reads, so a submission allocates
nothing.

The batch with the open command buffer and everything that existed to manage
it deleted, and the name saying what is left.
"""
mutable struct Submission
    signal_value::UInt64
    oneshots::Vector{OneShot}
    recordings::Vector{Recording}
    # Cross-queue dependencies, built up by `sync_access!(::VkManagedBuffer)` at
    # submit.
    wait_semaphores::Vector{Tuple{VK.Semaphore, UInt64, VK.PipelineStageFlag2}}
    bq::Any
    # Raw VulkanCore structs: Vulkan.jl's `_`-level wrappers each box a `deps`
    # vector for GC rooting — right once at setup, wrong fifty times a frame.
    # `raw_submits[1]` points into these vectors' data, so a refill resizes only
    # within capacity and the `vkQueueSubmit2` ccall runs under GC.@preserve.
    raw_cb_infos::Vector{VK.vk.VkCommandBufferSubmitInfo}
    raw_wait_infos::Vector{VK.vk.VkSemaphoreSubmitInfo}
    raw_signal_infos::Vector{VK.vk.VkSemaphoreSubmitInfo}
    raw_submits::Vector{VK.vk.VkSubmitInfo2}               # always length 1
end

Submission(bq) = Submission(UInt64(0), OneShot[], Recording[],
                            Tuple{VK.Semaphore, UInt64, VK.PipelineStageFlag2}[], bq,
                            sizehint!(VK.vk.VkCommandBufferSubmitInfo[], 8),
                            sizehint!(VK.vk.VkSemaphoreSubmitInfo[], 4),
                            sizehint!(VK.vk.VkSemaphoreSubmitInfo[], 4),
                            Vector{VK.vk.VkSubmitInfo2}(undef, 1))

# `BatchQueue` is Mantle's — see `src/graph/queue.jl`. This alias pins the ten
# driver parameters to Vulkan's types so every existing `VulkanBatchQueue{VkContext}`
# still names exactly what it did.
const VulkanBatchQueue{C} = BatchQueue{VK.Device, VK.Queue, VK.CommandPool,
                                       Submission, OneShot, VK.Semaphore, C, UInt64}

"""The queue a closed command buffer was taken from, as the concrete type — the
field is `Any` only because the queue is declared after the closed types."""
@inline queueof(c::Closed) = c.bq::VulkanBatchQueue{VkContext}


"""
Per-queue scratch for the wait and timeline-query fast paths: one semaphore
slot, one value slot, one packed wait info and one counter cell, refilled per
call. A fresh `Vector` + wrapper + `Ref` per wait cost ~300 bytes of every
`waitfor!`; these cost nothing after construction. A heap object (mutable) so
the `Any`-typed `BatchQueue.slots` field hands out a REFERENCE — an isbits
bundle would box on every read.
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

function VulkanBatchQueue(device::VK.Device, queue::VK.Queue, qf_idx::UInt32, ctx;
                    queue_index::Int=-1)
    cmd_pool = VK.CommandPool(device, qf_idx;
        flags=VK.COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT)
    # Per-queue timeline semaphore for cross-queue ordering.
    type_info = VK.SemaphoreTypeCreateInfo(VK.SEMAPHORE_TYPE_TIMELINE, UInt64(0))
    timeline_sem = unwrap(VK.create_semaphore(device,
        VK.SemaphoreCreateInfo(; next=type_info)))
    # `{typeof(ctx)}`, and the parameter is not optional. `BatchQueue` gained a
    # context TYPE PARAMETER when it moved to `src/graph/queue.jl` — `ctx::C`,
    # so `bq.ctx.caches` infers — and `VulkanBatchQueue{C}` leaves it free.
    # Julia generates a field-wise constructor for the bare name and for all
    # parameters given, never for a partial application, so the unparameterised
    # call this was landed on the four-argument method above and raised a
    # `MethodError` listing forty-one arguments.
    bq = VulkanBatchQueue{typeof(ctx)}(device, queue, qf_idx, cmd_pool,
                    Submission[], OneShot[],                # the two pools
                    timeline_sem, UInt64(0),
                    Any[], Any[],    # deferred_frees, deferred_as_frees
                    Base.Threads.SpinLock(),  # deferred_frees_lock
                    ctx,             # owning VkContext (required)
                    Threads.threadid(),  # owning_thread
                    UInt64(120) * 1_000_000_000,            # flush timeout
                    Outstanding{UInt64,Submission}[],       # outstanding submissions
                    "", "",                                 # last / prev dispatch info
                    queue_index,
                    QueueSlots())                           # fast-path scratch
    return bq
end

"""
    CoopMat2Caps

The sub-features of `VK_NV_cooperative_matrix2`, as enabled on this device.
`available` is false when the extension itself is missing, in which case every
other field is false too — so a kernel may test the single field it needs.

The extension is NVIDIA-only. On AMD (RDNA3 exposes WMMA through the portable
`VK_KHR_cooperative_matrix`) all of these are false and kernels must fall back
to the plain KHR path, which is the one that ships today.

| field | what it adds | who wants it |
|:--|:--|:--|
| `workgroup_scope` | matrices spanning the whole workgroup, not one subgroup | wider GEMM tiles |
| `flexible_dimensions` | M/N/K free of the driver's fixed shape list | `E = 72` without padding to 80 |
| `reductions` | row/column reduce on an accumulator in place | flash softmax row max/sum |
| `conversions` | element conversion without a round trip through memory | mixed-precision staging |
| `per_element_operations` | a callback per element, given `(row, col)` | the flash rescale of a held `O` |
| `tensor_addressing` | a tensor descriptor drives the addressing | hand-coded root+stride staging |
| `block_loads` | load a tile straight from a descriptor | ditto |
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

The SM/CU count and warps per SM, or `(0, 0)` where the device will not say.

A grid that does not fill the device is the dominant cost on small shapes — SAM
2's decode runs one attention at 8 workgroups on 48 SMs — so "how many workgroups
before the device is busy" is a number kernels need, and before it was queried
they hardcoded it.

Vendor coverage follows llama.cpp's `ggml-vulkan.cpp:6138-6146`, which is the
same question asked by the same kind of code. Neither extension is *enabled* on
the logical device — these are physical-device property queries and enabling
would be pointless — but the chained struct is only filled for an extension the
device advertises, so each is guarded by `has_extension`.

The chain is also only filled when the instance opted into
`vkGetPhysicalDeviceProperties2`: core in Vulkan 1.1, or the
`VK_KHR_get_physical_device_properties2` instance extension before that. Lava's
instance asks for 1.4 (see `create_vulkan_context`), so this holds — but note the
failure is silent if it ever stops holding. The base properties still come back
fully populated and only the `next` chain is dropped, with no return code to
check, because `vkGetPhysicalDeviceProperties2` returns void.
"""
function query_shader_cores(phys_dev)
    smcount, warps = 0, 0
    if has_extension(phys_dev, "VK_NV_shader_sm_builtins")
        # No try: the extension was just confirmed present by `has_extension`,
        # so a failure here is a bug in the query, not an absent capability.
        # Swallowing it produced `sm_count = 0`, which every caller reads as "the
        # device does not report it" — a wrong answer dressed as a fact.
        p = VK.get_physical_device_properties_2(
                phys_dev, VK.PhysicalDeviceShaderSMBuiltinsPropertiesNV).next
        smcount, warps = Int(p.shader_sm_count), Int(p.shader_warps_per_sm)
    elseif has_extension(phys_dev, "VK_AMD_shader_core_properties2")
        # Same reasoning as the NV branch above: advertised means queryable.
        p = VK.get_physical_device_properties_2(
                phys_dev, VK.PhysicalDeviceShaderCoreProperties2AMD).next
        smcount = Int(p.active_compute_unit_count)
    end
    # MERGE NOTE (2026-08-05): `sd/nvidia` restructured this as
    # `query_device_compute(phys_dev, phys_props) -> DeviceCompute`, which
    # `DeviceCaps` has since absorbed (see its docstring: "three things became
    # this one"). Git matched the two bodies as one hunk and spliced theirs into
    # this shell, leaving `return 0, 0` below the assignments — a function that
    # always reported "no SM count", which is exactly the wrong-answer-dressed-
    # as-a-fact their comment above warns about, reintroduced by the merge that
    # removed it. Their bodies are the improvement and are kept.
    return smcount, warps
end


"""
    VkContext

Persistent Vulkan context. Batch-based command recording goes through
`default_bq::VulkanBatchQueue` (the primary queue). Use `VulkanBatchQueue(...)` to
create additional independent queues (e.g., for async compute/RT).
"""
mutable struct VkContext
    instance::VK.Instance
    physical_device::VK.PhysicalDevice
    device::VK.Device
    queue_family_index::UInt32
    device_name::String
    # Primary batch queue — all global API functions delegate here.  Always
    # non-nothing after the inner constructor returns (VulkanBatchQueue is built
    # using `new()`-based two-phase init to break the chicken-and-egg with
    # VulkanBatchQueue.ctx).
    # `VulkanBatchQueue{VkContext}`, not the UnionAll — otherwise `ctx.default_bq` is
    # abstract and the parameter above buys nothing at this end of the cycle.
    default_bq::VulkanBatchQueue{VkContext}
    # Secondary compute queue (async RT) — same family, separate queue object
    compute_queue::VK.Queue
    # Ray tracing (nothing if not available)
    rt_pipeline_properties::Union{Nothing, RTPipelineProperties}
    # Debug messenger (nothing if validation layers not available)
    debug_messenger::Any
    # The messenger's ring, and its `pUserData`. Per context because the
    # messenger is — see `ValidationRing`.
    validation::ValidationRing
    # Queue allocation: next available index + total requested from device
    next_queue_index::Int
    max_queue_count::Int
    # Every queue `allocate_batch_queue!` has handed out and that nobody has
    # released yet.
    #
    # OWNERSHIP, not bookkeeping. A `ManagedBuffer` records `last_write = (bq,
    # value)` and its finalizer asks that queue's timeline semaphore whether the
    # GPU is done. Without this list the queue is reachable only from the buffers
    # that name it, so when a screen and its buffers become garbage together the
    # semaphore's own finalizer can run FIRST — and then
    # `vkGetSemaphoreCounterValue` is a use-after-free that segfaults inside the
    # driver. Holding the queue here means it outlives every buffer allocated on
    # it, the same argument `caches` makes further down: a field dies with its
    # context, so nothing outlives the handles it describes.
    extra_queues::Vector{VulkanBatchQueue{VkContext}}
    # Hardware queue slots handed back by `release_batch_queue!`, reused before
    # `next_queue_index` advances.
    free_queue_indices::Vector{Int}
    # Async compute family (distinct from primary). RADV family 1: 4 queues,
    # compute+transfer. Used by the explicit-queue refactor for upload_bq.
    async_queue_family_index::Union{Nothing, UInt32}
    async_queue_count::Int
    # Per-device state (was previously a global Ref).
    # Set to `true` after vkQueueSubmit returns DEVICE_LOST. Finalizers
    # holding a ref to the context check this to skip Vulkan calls on
    # invalid handles.
    device_lost::Bool
    # Cached physical-device properties (avoid re-querying on every alloc/dispatch).
    memory_properties::VK.PhysicalDeviceMemoryProperties
    max_wg_dims::NTuple{3, Int}
    # Alignment (bytes) for BDAs passed as `pScratchData` in AS builds.
    # Picked by `bda_alignment_for(ctx, scratch=true)`.
    as_scratch_align::UInt64
    # Whether VK_KHR_ray_query is available on this device.
    # Set to true by B1 (device extension probe). False until proven otherwise.
    ray_query_available::Bool
    # Whether VK_NV_ray_tracing_invocation_reorder (SER) is available.
    # When true, the SPIR-V emitter declares the ShaderInvocationReorderNV
    # capability and the raygen can use `lava_rt_hit_object_*` /
    # `lava_rt_reorder_thread_*` intrinsics.  NVIDIA-only.
    ser_available::Bool
    # The two flags above as the record the compiler takes: what a module
    # compiled FOR THIS DEVICE may declare. Every compile this context runs
    # passes it, and every frozen key it reads mixes it in.
    features::TargetFeatures
    # Whether VK_EXT_conditional_rendering is enabled. `repeat!` needs it — it is
    # how a device-written count decides which iterations of a recorded loop
    # actually run — and a graph that asks for one on a device without it has to
    # be told so at compile, not silently run every iteration.
    conditional_rendering_available::Bool
    # Whether VK_KHR_cooperative_matrix is enabled: subgroup-scope matrix
    # multiply-accumulate (tensor cores). `coopmat_shapes` holds the driver's
    # legal (M, N, K, A/B type, C/result type, saturating) combinations -- the
    # hardware only implements a fixed set, so a kernel must pick one of these
    # rather than any tile size it likes. Empty when unsupported.
    coopmat_available::Bool
    coopmat_shapes::Vector{NamedTuple{(:M, :N, :K, :ab_type, :c_type, :scope),
                                      Tuple{Int, Int, Int, UInt32, UInt32, UInt32}}}
    # VK_NV_cooperative_matrix2, per sub-feature. NVIDIA-only; every kernel that
    # branches on one of these must keep a path that runs with all of them false,
    # because that is what AMD (RDNA3 WMMA via the plain KHR extension) gets.
    coopmat2::CoopMat2Caps
    # VK_NV_cooperative_vector: matrix x vector at subgroup scope, for the shapes
    # where a cooperative *matrix* would waste most of its tile (`Lq = 23`).
    # NVIDIA-only.
    coopvec_available::Bool
    # VK_KHR_shader_maximal_reconvergence + VK_KHR_shader_subgroup_uniform_control_flow.
    # Together they make a spin-wait on a shared flag well-defined, which is how
    # a producer/consumer (warp-specialised) kernel is written without Vulkan
    # having named barriers. KHR, so RDNA3 has them too.
    maximal_reconvergence_available::Bool
    subgroup_uniform_control_flow_available::Bool
    # VK_KHR_shader_subgroup_rotate: OpGroupNonUniformRotateKHR, a shuffle by a
    # subgroup-uniform delta. KHR, promoted to Vulkan 1.4.
    subgroup_rotate_available::Bool
    # Whether VK_EXT_memory_budget is enabled. When true, OOM error reporting
    # queries the driver's real per-heap budget vs usage via
    # VkPhysicalDeviceMemoryBudgetPropertiesEXT.
    memory_budget_available::Bool
    # Whether VK_KHR_external_memory(_fd) is enabled. When true,
    # `VulkanExternalImage` can export allocations as opaque fds for zero-copy
    # sharing with other APIs (OpenGL via GL_EXT_memory_object_fd).
    external_memory_available::Bool
    # Whether VK_KHR_video_decode_queue + h264 are enabled (hardware decode into
    # Vulkan images). The decode queue + its family index are `nothing` when the
    # device has no video-decode support. No effect until a decode session is made.
    video_decode_available::Bool
    video_decode_queue::Union{Nothing, VK.Queue}
    video_decode_queue_family_index::Union{Nothing, UInt32}
    # Whether GPU-Assisted Validation is ACTIVE on this instance — what was
    # achieved, which `debug` below is not: the layer or the extension can be
    # missing, and then a requested `gpu_av` yields `gpu_assisted = false`.
    # `verify_gpu_av` exists because even `true` here is not proof it fires.
    gpu_assisted::Bool
    # What this device was ASKED for at construction. Kept so a caller can see
    # the configuration without reconstructing it, and so `reset_device!` can
    # carry it across a recovery reset instead of silently disarming it.
    debug::DebugConfig
    # Driver version (used to key the on-disk VkPipelineCache file).
    driver_version::String
    # Persistent VkPipelineCache. Seeded from disk on init, passed to every
    # vkCreate*Pipelines call, snapshotted back on reset_device! + atexit.
    pipeline_cache::VK.PipelineCache
    # `vkCmdPipelineBarrier`, resolved for THIS device.
    #
    # Device function pointers are per device — `vkGetDeviceProcAddr` returns a
    # pointer valid only for the device it was asked about. This was a module
    # global (`CMD_PIPELINE_BARRIER_FPTR`), so creating a second context
    # overwrote it, and the FIRST device's command buffers were then recorded
    # through the SECOND device's driver.
    #
    # That is an immediate segfault rather than a wrong answer, which makes it
    # the most dangerous piece of module-scope device state in the library and
    # the one `GUARDRAILS.md` §8 did not list — it names four caches holding
    # handles and does not mention the function table. Found the first time two
    # contexts existed at once: the crash was inside `libvulkan_lvp.so` while
    # dispatching on the NVIDIA context.
    cmd_pipeline_barrier_fptr::Ptr{Nothing}
    # A stable identity for this device, for as long as it exists.
    #
    # Four caches hold device-owned Vulkan handles at module scope and were keyed
    # without the device, so two devices running the same kernel produced the
    # SAME key and the second got the first's `VkPipeline` — bound into a command
    # buffer on a different `VkDevice`, which is undefined behaviour
    # (`GUARDRAILS.md` §8). This is the missing part of every one of those keys.
    #
    # A counter rather than `objectid(ctx)`: object ids are reused after the
    # collector reclaims, and a *reused* id is exactly the failure this exists to
    # prevent — a fresh device silently inheriting a dead one's pipelines. It
    # also survives `reset_device!`, which builds a new context, so entries
    # from before a reset can never be handed to after it.
    #
    # It survives as an identity — for logging, and for the probe's assertion
    # that two contexts are distinct — but it is no longer a cache key. It was
    # one, and this comment used to argue that keying was as good as ownership
    # because the cached value types are defined in files included after this
    # one, so a field would have to be `Any` and cost inference on a lookup per
    # dispatch. The premise was true and the conclusion was wrong: the fix is to
    # move the nine type definitions ahead of this file (`coretypes.jl`), not to
    # accept a surrogate key. `caches` below is concrete.
    id::UInt64

    # Per-device state, owned by the device. See `DeviceCaches` for what was
    # global before and why keying it by `id` was not the same thing: a field
    # dies with its context, so nothing outlives the handles it describes and
    # `reset_device!` has nothing to clear.
    caches::DeviceCaches

    # Every debugging and instrumentation toggle, owned by the device it
    # instruments. Was eighteen module-level `Ref`s; see `Diagnostics`.
    diag::Diagnostics

    # Inner constructor: two-phase init via `new()` so we can hand a live
    # `ctx` reference to `VulkanBatchQueue(...)` while finishing the ctx's own
    # field assignments.  There is no public ctor that can leave `default_bq`
    # unset.  `primary_queue` is the raw `VK.Queue` for the default bq;
    # everything else maps directly to a field.
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
                       memory_budget_available::Bool=false,
                       external_memory_available::Bool=false,
                       gpu_assisted::Bool=false,
                       debug::DebugConfig=DebugConfig(),
                       driver_version::AbstractString="unknown",
                       video_decode_available::Bool=false,
                       video_decode_queue::Union{Nothing, VK.Queue}=nothing,
                       video_decode_queue_family_index::Union{Nothing, UInt32}=nothing)
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
        ctx.extra_queues = VulkanBatchQueue{VkContext}[]
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
        ctx.memory_budget_available = memory_budget_available
        ctx.external_memory_available = external_memory_available
        ctx.video_decode_available = video_decode_available
        ctx.video_decode_queue = video_decode_queue
        ctx.video_decode_queue_family_index = video_decode_queue_family_index
        ctx.gpu_assisted = gpu_assisted
        ctx.debug = debug
        ctx.driver_version = driver_version
        # Seed a persistent VkPipelineCache from disk (if any). The header is
        # validated against this physical device before the driver sees it —
        # `vkCreatePipelineCache` is not a safe place to discover a mismatch.
        ctx.id = (VK_CONTEXT_COUNTER[] += 1)
        # Registered before anything can allocate against it — see
        # `LIVE_CONTEXTS`. A context that is not in this list is one whose
        # buffers will call into a dead driver during Julia's shutdown sweep.
        push!(LIVE_CONTEXTS, WeakRef(ctx))
        ctx.caches = DeviceCaches()
        ctx.diag = Diagnostics()
        # Filled by the constructor once the device exists; null until then so a
        # barrier recorded before that point is skipped rather than jumping to
        # whatever the field happened to contain.
        ctx.cmd_pipeline_barrier_fptr = C_NULL
        ctx.pipeline_cache = create_lava_pipeline_cache(
            device, lava_pipeline_cache_path(device_name, driver_version), physical_device)
        _register_pipeline_cache_atexit!()
        # Now build the default VulkanBatchQueue with the live ctx.  Sets the
        # remaining field; no nullable slot, no post-hoc mutation.
        ctx.default_bq = VulkanBatchQueue(device, primary_queue, queue_family_index, ctx)
        return ctx
    end
end

"""
    caps(ctx) -> DeviceCaps

What kernels ask this device. Queried on first call and cached on the context.

Lazy rather than filled by the constructor because half of it — whether a
cooperative-matrix GEMM is usable, and at what subgroup width — is answered
against the *logical* device and its enabled extensions, which do not exist while
the context is still being built.

Tests that want a kernel's decision for a device they do not have assign a
modified copy: `ctx.caches.caps = DeviceCaps(caps(ctx); workgrouplimit = 512)`.
Per context, so it cannot leak into another device the way the module-level
`WORKGROUP_LIMIT` it replaces did.
"""
function caps(ctx::VkContext)
    c = ctx.caches.caps
    c === nothing || return c
    limits = VK.get_physical_device_properties(ctx.physical_device).limits
    cores, warps = query_shader_cores(ctx.physical_device)
    # The driver's own table, not a constant. `tile` was `GEMM_TILE = 16` — a
    # comment reading "the cooperative-matrix tile this device implements" on a
    # module-level binding, which is a device fact that no device answered for.
    # On a card reporting anything but 16 the kernels strided by 16 and read
    # another fragment's registers, and nothing would have crashed.
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

# `wggranularity(caps, nt)` moved to `KernelInterface` with `DeviceCaps`. It is a
# lookup into `caps.wggran` and nothing else, so it belongs with the table rather
# than with the Vulkan query that fills it — which is `workgroup_matrix_granularity`
# just below, and stays here because it is a driver call.

"""
    workgroup_matrix_granularity(ctx) -> Vector{(invocations, M, N, K)}

The shapes a **workgroup-scope** fp16 x fp16 -> fp32 cooperative matrix may have
on this device, one row per workgroup size, sorted by size.

Empty means no workgroup-scope matrices — which is every device that is not
NVIDIA with `VK_NV_cooperative_matrix2`, and is why this doubles as the
capability test: a kernel cannot ask "may I?" without also being handed "at what
shapes?", and those two drifting apart is how a kernel ends up asking for a tile
the device refuses at pipeline creation.

The pairing is the part that is not guessable. `M`, `N` and `K` are multiples,
not sizes, and they COARSEN as the workgroup grows — on an RTX 4000 Ada,
16/16/16 at 32 and 64 invocations, 32/16/16 at 128, 32/32/16 at 256. So a
head dimension of 72 pads to 80 at 128 invocations and to 96 at 256, and the
padding is 33% of both products in the second case. Measured: every tiling was
faster at 128 than at 256 for exactly that reason.

fp16 x fp16 -> fp32 only, because that is what every kernel here multiplies.
Widen it when something needs bf16 or int8; the query returns those rows too.
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
    shader_core_count(ctx) -> Union{Nothing,Int}

Streaming multiprocessors (NVIDIA) or active compute units (AMD), or `nothing`
when the device does not report it.

`nothing` rather than `0` on purpose. The number's whole job is to be a
denominator — llama.cpp derives its flash-attention split count as
`shader_core_count * 2 / total_workgroups` — and a `0` there yields zero splits
silently, which is a wrong launch that still produces a plausible-looking answer.
`nothing` cannot be divided, so an unguarded use fails immediately.

Supply the fallback explicitly:

    cores = something(shader_core_count(ctx), 16)
"""
shader_core_count(ctx::VkContext) =
    (c = caps(ctx).cores) == 0 ? nothing : c

"""
    shader_warps_per_sm(ctx) -> Union{Nothing,Int}

Maximum resident subgroups per SM — the denominator for an occupancy figure.
`nothing` when unreported; NVIDIA-only in practice (`VK_NV_shader_sm_builtins`).
"""
shader_warps_per_sm(ctx::VkContext) =
    (w = caps(ctx).warps) == 0 ? nothing : w

"""
    max_shared_memory(ctx) -> Int

`maxComputeSharedMemorySize`: the per-workgroup shared-memory ceiling, in bytes.
Core Vulkan, so this one is always a real number — a kernel that sizes its
`@localmem` against a budget should read it here rather than assume 48 KB.
"""
max_shared_memory(ctx::VkContext) = caps(ctx).sharedbudget

# ── Async-safe validation message capture ──────────────────────────────────
#
# The Vulkan debug-utils callback can be invoked re-entrantly from inside a
# blocking driver ccall — notably GPU-Assisted Validation reading back its
# error log during `vkWaitSemaphores`.  In that context the calling thread is
# mid-ccall with driver locks held; doing ANY Julia work that can allocate,
# log, or hit a scheduler yield/safepoint deadlocks the runtime.  (Observed on
# the RTX 4000 Ada: GPU-AV catches a BDA OOB, `@error` prints it once, then the
# process hangs forever — the `@error` yielded while the driver held a lock.)
#
# So the callback writes ONLY into preallocated memory via raw ccalls
# (strlen/memcpy) and `unsafe_store!`/`@inbounds` array writes — no allocation,
# no logging, no `push!`.  The main thread later calls
# `drain_validation_messages!(ctx)` to turn raw slots into Strings, capture the
# hard errors in `ctx.validation.messages`, and log everything. See
# `ValidationRing` for why the arrays live on the context and how the callback
# reaches them.

const VK_CONTEXT_REF = Ref{Union{Nothing, VkContext}}(nothing)

"""
Every `VkContext` ever built, weakly.

`VK_CONTEXT_REF` is the BOUND one, and at shutdown that is not the interesting
set. Julia's `atexit` hooks run before the final finalizer sweep, and the hook in
`MantleVulkanExt.__init__` marks the bound context lost so that a `LavaArray`
finalizer running afterwards skips `query_timeline` instead of calling
`vkGetSemaphoreCounterValue` on a driver that has been torn down.

With one device that is enough. With two it is not, and Mantle's own suite makes
two: `twodevice_probe.jl` builds a second context, and the software rasterizer
answers it. Measured 2026-08-30 — the suite finished every test and then took a
SIGSEGV at exit:

    pthread_mutex_lock                      libc
    …                                       libvulkan_lvp.so     <- lavapipe
    vkGetSemaphoreCounterValue
    query_timeline                          runtime/command.jl:944
    vk_free!                                runtime/memory.jl:742
    unsafe_free!(::LavaArray)
    run_finalizer / ijl_atexit_hook

Weak, so being in this list never keeps a context alive: a device that is
collected during the session drops out of it, and marking a dead entry is a
no-op. Never cleaned up in normal running — the list is bounded by how many
devices a process makes, which is one, or two when something is probing.
"""
const LIVE_CONTEXTS = WeakRef[]

"""
    mark_all_devices_lost!()

Mark every live context lost, so no finalizer reaches a driver afterwards.

Shutdown only. Marking a device lost makes every subsequent Vulkan call on it a
no-op, which is what the process wants once it has decided to exit and is
exactly what it must not do while it is still rendering.
"""
function mark_all_devices_lost!()
    for r in LIVE_CONTEXTS
        c = r.value
        c isa VkContext && mark_device_lost!(c)
    end
    return nothing
end

"""
Bind the process-wide default context.

What the emitter may declare for a device is NOT pushed from here any more: it
used to be a process global the compiler read, so a kernel compiled for a second
device was shaped by whichever device was bound. The record travels with the
context (`ctx.features`) and with every compile job the context runs.
"""
function bind_context!(ctx::Union{Nothing, VkContext})
    VK_CONTEXT_REF[] = ctx
    # The frozen cache's miss logging, for the same reason: the half of that
    # cache the compiler consults is `compiler/frozen_spirv.jl`, and it was
    # reading `ctx.diag.frozen_log_misses` through `VK_CONTEXT_REF[]` for a
    # `println`. It is a boolean, so it is pushed like one.
    FROZEN_LOG_MISSES[] = ctx !== nothing && ctx.diag.frozen_log_misses
    return ctx
end
# Guards the lazy init below. `VkContext(; …)` builds a whole VkDevice and has no
# idempotency guard of its own, so two threads arriving together built TWO.
const VK_CONTEXT_LOCK = ReentrantLock()

"""
    device_lost(ctx::VkContext)  ->  Bool

Whether this context's device has been marked lost. Prefer passing `ctx`
explicitly; the no-arg form looks up the current default context.
"""
device_lost(ctx::VkContext) = ctx.device_lost
device_lost() = let ctx = VK_CONTEXT_REF[]
    ctx === nothing ? false : ctx.device_lost
end

"""
Mark `ctx`'s device as lost. All subsequent finalizers will skip Vulkan calls.

Two things reach this state, and only the first is a fault:

 1. `ERROR_DEVICE_LOST` from any Vulkan call — see `mark_if_device_lost!`.
 2. **Retirement**: a context nobody will call into again, whose buffers are
    still alive in Julia and whose finalizers must therefore not touch it.
    `reset_device!` retires the context it replaces; anything that builds a
    context of its own with `VkContext(; select)` owns retiring it.

The flag means the same thing to every consumer either way — *do not call into
this device* — which is why the second case reuses it rather than adding a
parallel one. A context that is dropped without being retired hands its
finalizers a device that VK.jl's own handle finalizers may already have
destroyed, and the crash lands inside the driver at whatever GC runs next.
"""
mark_device_lost!(ctx::VkContext) = (ctx.device_lost = true; nothing)

"""
Hands out `VkContext.id`. Monotonic and never reused, so a device destroyed and
recreated cannot inherit cache entries keyed to its predecessor.

Nothing in `src/` reads `ctx.id` any more — the dictionaries it keyed are
`VkContext` fields now, which is what made it redundant. It survives as a cheap
identity for diagnostics and for `twodevice_probe.jl`'s "two contexts must not
share an id" assertion, and this counter with it.
"""
const VK_CONTEXT_COUNTER = Ref{UInt64}(0)

# `RESET_CALLBACKS` was here: a list every later-included file pushed onto so
# `reset_device!` could empty its module-level caches. Deleted rather than
# emptied — its entries were the symptom this refactor was diagnosing. State
# that outlives the device it describes has to be told to go away; state a
# `VkContext` owns simply does not. The last four went with the pool accounting,
# the dispatch counters, the capture handle and the timing records.

"""
    vk_context() -> VkContext

Get or create the global Vulkan context. Lazily initializes on first call.
"""
function vk_context()
    ctx = VK_CONTEXT_REF[]
    ctx === nothing || return ctx::VkContext
    # Double-checked under a lock. Unlocked, two threads both saw `nothing` and
    # both ran the constructor, leaving two live VkDevices: the loser's context is
    # still reachable from every buffer it allocated (`buf.last_write` retains
    # its VulkanBatchQueue), so the next cross-queue wait passed a semaphore from one
    # device to the other and the driver segfaulted with no Julia frame to show.
    lock(VK_CONTEXT_LOCK)
    try
        ctx = VK_CONTEXT_REF[]
        ctx === nothing || return ctx::VkContext
        # `invokelatest`, not a direct call: a direct one makes inference record
        # a backedge from `vk_context` to the `VkContext(; …)` constructor, and
        # `getproperty(::LavaBackend, :bq)` goes through here, so EVERY Lava
        # operation ends up inferring through Vulkan initialisation. Anything
        # that invalidates the constructor then invalidates the whole package —
        # loading GLMakie does exactly that (FreeType adds a `Base.unsafe_load`
        # method), and it cost 50 506 Lava MethodInstances and ~41 s of
        # re-inference on the first GPU call afterwards. The dynamic dispatch is
        # paid once, on the single call that creates the context.
        ctx = Base.invokelatest(VkContext; select = defaultselector())::VkContext
        bind_context!(ctx)
        return ctx::VkContext
    finally
        unlock(VK_CONTEXT_LOCK)
    end
end

vk_device() = vk_context().device

"""
    reset_device!(; select = <MANTLE_DEVICE, or the ranking>,
                       debug = <the outgoing device's config>)

Replace the process-default Vulkan device. Destroys the old context and creates a
fresh one; clears all caches (pipelines, kernels, arg buffers).

Two reasons to call it.

1. **Recovery**, after `ERROR_DEVICE_LOST` or another unrecoverable error:
   `reset_device!()`. The debugging configuration carries across, so a reset in
   the middle of a session does not silently turn the instruments off.

2. **Switching validation on or off** — and this is the *only* way, because those
   settings are fixed at `vkCreateInstance` and cannot be applied to a device
   that already exists:

       reset_device!(debug = DebugConfig(gpu_av = true, pool_disabled = true))
       verify_gpu_av()                     # prove the layer actually fires
       reset_device!(debug = DebugConfig())   # …and back to the fast path

   See [`DebugConfig`](@ref). There is nothing else: no `enable_gpu_av`, no
   environment variable, no post-hoc toggle. If you want a device *without*
   installing it as the default — a second device, a lavapipe reference — build
   it with `VkContext(; select, debug)` instead.

**WARNING**: All existing `LavaArray`s become INVALID after reset — their backing
GPU buffers no longer exist. You must reallocate all GPU data.
"""
function reset_device!(; select = defaultselector(),
                            debug::Union{Nothing,DebugConfig} = nothing)
    cfg = debug !== nothing ? debug :
          let old = VK_CONTEXT_REF[]
              old === nothing ? DebugConfig() : old.debug
          end
    # Persist the VkPipelineCache before tearing the device down so the
    # next session can skip AMDVLK's SPIR-V → ISA recompile.
    let old = VK_CONTEXT_REF[]
        old === nothing || save_pipeline_cache!(old)
    end
    # Retire the old context BEFORE dropping it. Pre-reset `VkManagedBuffer`s
    # hold a strong ref to it, and their finalizers gate every Vulkan call on
    # `device_lost` — so marking it here is what makes that gate true.
    #
    # It used to be assumed rather than set, and the assumption only held on the
    # path that *caused* it: a reset after `ERROR_DEVICE_LOST` finds the flag
    # already true, while a voluntary `reset_device!()` left it false. Then
    # dropping the ref below made the old context garbage — and its buffers
    # garbage in the SAME collection, where Julia does not order finalizers. Run
    # the context's first and `VK.Device`'s own finalizer destroys the
    # device; the buffer's `vk_free!` then calls `query_timeline` on it and the
    # driver takes a SIGSEGV inside `vkGetSemaphoreCounterValue`.
    #
    #     d = KA.allocate(LavaBackend(), Float32, 1000); fill!(d, 1f0)
    #     reset_device!(); d = nothing; GC.gc()   # <- segfault
    #
    # A retired context is one nothing may call into: every array that predates
    # the reset holds memory belonging to a device that is gone, so there is no
    # case where skipping is wrong. `mark_device_lost!` is the same flag the
    # `VkResult` rule sets, and this is the second way to reach it.
    let old = VK_CONTEXT_REF[]
        if old !== nothing
            # Its blocks, while it is still the thing that owns them. This used
            # to be a reset callback walking a global `POOLS` dict.
            destroy_pool!(old)
            mark_device_lost!(old)
        end
    end
    bind_context!(nothing)
    # Don't destroy old Vulkan handles — they're invalid after DEVICE_LOST.
    # GC will eventually try to destroy them; _destroy_buffer! skips when
    # DEVICE_LOST was true (and we set it false only after clearing context).
    # Nothing to clear: the ring, its cursor and the drained messages are all
    # fields of the context being retired, and the new one brings a fresh
    # `ValidationRing` that its own messenger writes into.
    #
    # Built here rather than left to `vk_context()`'s lazy path, because the
    # configuration has to reach `vkCreateInstance` and the lazy path has no way
    # to be told about it. `invokelatest` for the same backedge reason as there.
    ctx = Base.invokelatest(VkContext; select, debug = cfg)::VkContext
    bind_context!(ctx)
    # `gpu_av` is what was ACHIEVED, not what was asked — the layer or the
    # extension can be missing, and a run under an instrument that never attached
    # reads exactly like a clean run. Logged so that is visible without asking.
    @info "Lava: device reset complete" device=ctx.device_name validation=cfg.validation gpu_av=ctx.gpu_assisted pool_disabled=cfg.pool_disabled
    if cfg.gpu_av && !ctx.gpu_assisted
        @warn "GPU-AV was requested but did NOT attach (gpu_assisted=false) — a clean run now means the instrument was off, not that there is no fault. Check VK_EXT_validation_features + VK_LAYER_KHRONOS_validation on this loader/driver."
    end
    return nothing
end


"""
    VkContext(; select = nothing, debug = DebugConfig()) -> VkContext

Build a device. **Does not install it** as the process default — `vk_context()`
is what does that, and it is the only caller that should.

This is the explicit constructor. Reach for it when you want a *specific* device,
or a device with validation on; `vk_context()` is the lazy default for everything
else.

`debug` is where every validation and instrumentation setting is chosen, and the
only place they can be: they configure the `VkInstance` and are fixed by
`vkCreateInstance`, so nothing set afterwards reaches a device that already
exists. See [`DebugConfig`](@ref) — it replaced seven environment variables that
were read here, two of whose failure modes were silent.

    ctx = VkContext(debug = DebugConfig(gpu_av = true, gpu_av_shaders = ["step_kernel"]))

`select` names the physical device the way `Device(VulkanAPI(); select)` does
(`Mantle.selectdevice`): `nothing` for the ranking, a name substring, an index
into `devices(VulkanAPI())`, or a predicate over `DeviceInfo`. That is what makes
a two-device test possible on a single-GPU machine: the loader enumerates the
real GPU *and* lavapipe from one instance, so

    gpu = vk_context()
    cpu = VkContext(select = "llvmpipe")

gives two live contexts with two distinct `id`s, which is the pair every
per-device cache key has to be checked against.
"""
function VkContext(; select = nothing, debug::DebugConfig = DebugConfig())
    # Create instance — target Vulkan 1.4 (device supports 1.4.335 on RADV).
    # Bumping API version unlocks 1.3/1.4 core features we enable below.
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

    # Collect all available instance extensions (driver + layer-provided)
    inst_extensions = String["VK_KHR_surface"]
    available_ext = unwrap(VK.enumerate_instance_extension_properties())
    ext_names = Set(String(filter(!=('\0'), collect(e.extension_name))) for e in available_ext)
    # Also collect extensions provided by the validation layer
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
    # Platform-specific surface extension (ext_names already computed above)
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
        push!(inst_extensions, "VK_KHR_win32_surface")
    elseif Sys.isapple()
        push!(inst_extensions, "VK_EXT_metal_surface")
    end

    # Extended validation, all of it requiring VK_EXT_validation_features:
    #   gpu_av         → shader instrumentation (OOB descriptor access, ray query
    #                    misuse); does NOT cover plain BDA ranges in current layers.
    #   sync_val       → synchronization validation (reads before writes, missing
    #                    barriers, cross-submit hazards). Catches shadow-ownership UAFs.
    #   best_practices → API-misuse / perf warnings.
    #   printf         → NonSemantic.DebugPrintf output from `@lava_printf`.
    # All are very slow; use one or a combination as needed for triage. See
    # `DebugConfig` — it enforces the two rules that used to be warnings here.
    #
    # `gpu_assisted` and `sync_val` below record what was ACHIEVED, which is not
    # the same as `debug`, what was asked for: the extension may be missing.
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
        # `VkValidationFeaturesEXT` cannot express Safe Mode or selective
        # instrumentation — those are `VK_EXT_layer_settings` only, which is why
        # they were missing. Chain the settings struct in front when either is
        # asked for; the layer reads both.
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

    # Set up debug messenger to capture validation/driver error messages
    # Before the messenger, because the messenger is given its address.
    validation = ValidationRing()
    debug_messenger = nothing
    if has_debug_utils
        debug_messenger = setup_debug_messenger(instance, validation, debug)
    end

    # Pick physical device (prefer discrete GPU)
    phys_devs = unwrap(VK.enumerate_physical_devices(instance))
    isempty(phys_devs) && throw(LavaError(
        "device initialization",
        "No Vulkan-capable GPU found",
        "Ensure Vulkan drivers are installed"))

    phys_dev = phys_devs[selectdevice(select, deviceinfos(phys_devs))]
    props = VK.get_physical_device_properties(phys_dev)
    dev_name = String(filter(!=('\0'), collect(props.device_name)))

    # Find queue family (prefer graphics+compute for graphics pipeline support)
    qf_idx = find_graphics_compute_queue_family(phys_dev)

    # Create logical device with required features
    qf_props = VK.get_physical_device_queue_family_properties(phys_dev)
    max_queues = qf_props[qf_idx + 1].queue_count
    n_queues = min(4, Int(max_queues))
    queue_priorities = ones(Float32, n_queues)
    queue_ci = [VK.DeviceQueueCreateInfo(qf_idx, queue_priorities)]

    # Additionally, request queues from any compute-capable family that
    # isn't the primary one (RADV exposes graphics+compute on family 0 with
    # 1 queue, and compute-only on family 1 with 4 queues). We need these
    # for async upload/dispatch in the explicit-queue design.
    async_qf_idx = find_async_compute_queue_family(phys_dev, qf_idx)
    async_n_queues = 0
    if async_qf_idx !== nothing
        async_max = qf_props[async_qf_idx + 1].queue_count
        async_n_queues = min(4, Int(async_max))
        push!(queue_ci, VK.DeviceQueueCreateInfo(async_qf_idx, ones(Float32, async_n_queues)))
    end

    # Hardware video decode (opt-in): request one queue from the VIDEO_DECODE
    # family. Enabling the extensions + queue has no effect until a decode
    # session is created, so it's safe to always request when the device offers it.
    video_qf_idx = find_video_decode_queue_family(phys_dev)
    has_video_decode = video_qf_idx !== nothing &&
        has_extension(phys_dev, "VK_KHR_video_queue") &&
        has_extension(phys_dev, "VK_KHR_video_decode_queue") &&
        has_extension(phys_dev, "VK_KHR_video_decode_h264")
    has_video_decode || (video_qf_idx = nothing)
    video_qf_idx === nothing ||
        push!(queue_ci, VK.DeviceQueueCreateInfo(video_qf_idx, Float32[1.0]))

    # Check for RT extension support
    has_rt = has_rt_extensions(phys_dev)
    has_ray_query = has_rt && has_extension(phys_dev, "VK_KHR_ray_query")
    # SER (Shader Execution Reordering) — NVIDIA-specific extension that
    # exposes the `HitObject*` API and `reorderThreadWithHitObjectNV` for
    # warp-level work reordering between traceRay and shading.  Lets
    # divergent path-tracer chits ((`vp_closesthit_shade`) execute in
    # coherent warps.  Optional; we still ship a working RT pipeline path
    # without it.
    has_ser = has_rt && has_extension(phys_dev, "VK_NV_ray_tracing_invocation_reorder")

    # Check for workgroup memory explicit layout (needed for mixed-type shared memory structs)
    has_wg_explicit = has_extension(phys_dev, "VK_KHR_workgroup_memory_explicit_layout")
    has_atomic_float = has_extension(phys_dev, "VK_EXT_shader_atomic_float")
    # 64-bit atomics. These were hardcoded `false` below while the emitter happily
    # declared the `Int64Atomics` SPIR-V capability, so any kernel using one built a
    # module whose capability had no enabled feature behind it — undefined, and
    # reported by the validation layer as
    # VUID-VkShaderModuleCreateInfo-pCode-08740. Ask the device instead of
    # assuming, in both directions: enabling an unsupported feature fails device
    # creation, and leaving a supported one off is what caused this.
    has_int64_atomics = let
        q = VK.get_physical_device_features_2(phys_dev,
                VK.PhysicalDeviceVulkan12Features).next
        (buffer = q.shader_buffer_int_64_atomics, shared = q.shader_shared_int_64_atomics)
    end
    # Subgroup ops on anything other than 32-bit types. `subgroup_shuffle` and the
    # rest are bound for Int64/UInt64/Float64 (test_subgroup_shuffle.jl's
    # "every element type round-trips"), and SPIR-V group operations on 8-, 16- or
    # 64-bit types REQUIRE this feature: VUID-RuntimeSpirv-None-06275. Hardcoded
    # false, so those shuffles were undefined — GPU-assisted validation reports
    # "OpGroupNonUniformShuffle is using a 64-bit int scalar but
    # shaderSubgroupExtendedTypes was not enabled". Asked, not assumed, for the
    # same reason as the atomics above.
    has_subgroup_extended = let
        q = VK.get_physical_device_features_2(phys_dev,
                VK.PhysicalDeviceVulkan12Features).next
        q.shader_subgroup_extended_types
    end
    has_pipeline_exec_props = has_extension(phys_dev, "VK_KHR_pipeline_executable_properties")
    # VK_EXT_memory_budget — lets us read VkPhysicalDeviceMemoryBudgetPropertiesEXT
    # for real heap utilisation, used in OOM error reporting.
    has_memory_budget = has_extension(phys_dev, "VK_EXT_memory_budget")
    # Device-generated commands: a shader writes the command stream and the host
    # issues one execute whose SEQUENCE COUNT comes from a device address. That
    # is the only way either backend lets the device decide how much work runs —
    # neither Vulkan nor Metal has device-side enqueue.
    #
    # It is NOT what `repeat!` lowers to, and the reason is worth recording here
    # because the two extensions look interchangeable and are not: DGC sequences
    # carry no barriers between them, so N sequences are N dispatches that may
    # overlap. That expresses "do this independent thing N times"; it cannot
    # express a loop whose iteration k+1 reads what k wrote, which is every
    # bounce loop.
    has_dgc = has_extension(phys_dev, "VK_EXT_device_generated_commands")
    # Conditional rendering: a 32-bit predicate in a device buffer, read at
    # EXECUTION time, that discards the commands inside its scope. That is what
    # `repeat!` lowers to — the body is recorded `maxiters` times with ordinary
    # barriers between iterations, and the device decides how many of them run.
    # Barriers still execute inside a discarded scope; only the work is skipped.
    has_cond_render = has_extension(phys_dev, "VK_EXT_conditional_rendering")
    # Cooperative matrix — subgroup-scope matrix multiply-accumulate (tensor
    # cores). Probed once here; kernels pick a coopmat or a scalar
    # instantiation from `ctx.coopmat_available` / `ctx.coopmat_shapes`.
    #
    # Uses the portable VK_KHR_cooperative_matrix.
    has_coopmat = has_extension(phys_dev, "VK_KHR_cooperative_matrix")
    # VK_NV_cooperative_matrix2 layers per-element operations, row reductions,
    # flexible dimensions and tensor addressing onto the KHR matrices. Which of
    # its seven sub-features are actually on is read back below, after device
    # creation, rather than assumed from the extension being present.
    # NVIDIA-only; kernels keep their KHR path for everyone else.
    has_coopmat2 = has_coopmat && has_extension(phys_dev, "VK_NV_cooperative_matrix2")
    # Matrix x vector at subgroup scope. Wanted for the shapes where a
    # cooperative matrix wastes most of its tile. NVIDIA-only.
    has_coopvec = has_extension(phys_dev, "VK_NV_cooperative_vector")
    # A defined reconvergence point (KHR, and RDNA3 has it). Needed before a
    # kernel may spin on a shared flag, i.e. before producer/consumer staging.
    has_max_reconv = has_extension(phys_dev, "VK_KHR_shader_maximal_reconvergence")
    has_subgroup_ucf = has_extension(phys_dev, "VK_KHR_shader_subgroup_uniform_control_flow")
    # OpGroupNonUniformRotateKHR: shuffle by a subgroup-uniform delta, which is
    # the cheap form of a butterfly reduction. KHR, promoted to Vulkan 1.4.
    has_subgroup_rotate = has_extension(phys_dev, "VK_KHR_shader_subgroup_rotate")

    # Device extensions
    extensions = String[
        "VK_KHR_swapchain",
    ]
    if has_rt
        append!(extensions, [
            "VK_KHR_acceleration_structure",
            "VK_KHR_ray_tracing_pipeline",
            "VK_KHR_deferred_host_operations",
        ])
    end
    if has_ray_query
        push!(extensions, "VK_KHR_ray_query")
    end
    if has_ser
        push!(extensions, "VK_NV_ray_tracing_invocation_reorder")
    end
    if has_wg_explicit
        push!(extensions, "VK_KHR_workgroup_memory_explicit_layout")
    end
    if has_atomic_float
        push!(extensions, "VK_EXT_shader_atomic_float")
    end
    # Opt-in: pipeline executable properties (register count, scratch, vendor
    # ISA) for the profiling API.  Off by default because enabling the
    # extension changes pipeline-creation behavior on some drivers; user opts
    # in via `enable_pipeline_executable_properties!()` BEFORE first
    # device creation.
    if has_pipeline_exec_props && PIPELINE_EXEC_PROPERTIES_REQUESTED[]
        push!(extensions, "VK_KHR_pipeline_executable_properties")
    end
    if has_memory_budget
        push!(extensions, "VK_EXT_memory_budget")
    end
    if has_dgc
        push!(extensions, "VK_EXT_device_generated_commands")
    end
    if has_cond_render
        push!(extensions, "VK_EXT_conditional_rendering")
    end
    if has_coopmat
        push!(extensions, "VK_KHR_cooperative_matrix")
    end
    if has_coopmat2
        push!(extensions, "VK_NV_cooperative_matrix2")
    end
    if has_coopvec
        push!(extensions, "VK_NV_cooperative_vector")
    end
    if has_max_reconv
        push!(extensions, "VK_KHR_shader_maximal_reconvergence")
    end
    if has_subgroup_ucf
        push!(extensions, "VK_KHR_shader_subgroup_uniform_control_flow")
    end
    if has_subgroup_rotate
        push!(extensions, "VK_KHR_shader_subgroup_rotate")
    end
    # External-memory export (opaque fds for GL/other-API interop). Enabling
    # the extension has no effect until an VulkanExternalImage is created.
    has_external_memory = has_extension(phys_dev, "VK_KHR_external_memory_fd")
    if has_external_memory
        push!(extensions, "VK_KHR_external_memory")
        push!(extensions, "VK_KHR_external_memory_fd")
    end
    # Hardware video decode extensions (family already probed + queue requested above).
    if has_video_decode
        append!(extensions, ["VK_KHR_video_queue", "VK_KHR_video_decode_queue",
                             "VK_KHR_video_decode_h264"])
        has_extension(phys_dev, "VK_KHR_video_decode_h265") &&
            push!(extensions, "VK_KHR_video_decode_h265")
    end

    # Chain required features — all Vulkan 1.2 promoted features go in Vulkan12Features
    # (can't mix Vulkan12Features with separate promoted structs like BDA/VariablePointers)
    var_ptr_features = VK.PhysicalDeviceVariablePointersFeatures(
        true,   # variable_pointers_storage_buffer
        true,   # variable_pointers
    )
    # Vulkan 1.2 features: BDA, VulkanMemoryModel, shaderInt8, scalarBlockLayout
    vulkan12_features = VK._PhysicalDeviceVulkan12Features(
        false,  # sampler_mirror_clamp_to_edge
        false,  # draw_indirect_count
        false,  # storage_buffer_8_bit_access
        false,  # uniform_and_storage_buffer_8_bit_access
        false,  # storage_push_constant_8
        has_int64_atomics.buffer,  # shader_buffer_int_64_atomics ← the emitter declares Int64Atomics
        has_int64_atomics.shared,  # shader_shared_int_64_atomics ← ditto, for @localmem
        true,   # shader_float_16  ← REQUIRED (Float16 types in SPIR-V)
        true,   # shader_int_8  ← REQUIRED (i8 types in SPIR-V)
        false,  # descriptor_indexing
        false,  # shader_input_attachment_array_dynamic_indexing
        false,  # shader_uniform_texel_buffer_array_dynamic_indexing
        false,  # shader_storage_texel_buffer_array_dynamic_indexing
        false,  # shader_uniform_buffer_array_non_uniform_indexing
        false,  # shader_sampled_image_array_non_uniform_indexing
        false,  # shader_storage_buffer_array_non_uniform_indexing
        false,  # shader_storage_image_array_non_uniform_indexing
        false,  # shader_input_attachment_array_non_uniform_indexing
        false,  # shader_uniform_texel_buffer_array_non_uniform_indexing
        false,  # shader_storage_texel_buffer_array_non_uniform_indexing
        false,  # descriptor_binding_uniform_buffer_update_after_bind
        false,  # descriptor_binding_sampled_image_update_after_bind
        false,  # descriptor_binding_storage_image_update_after_bind
        false,  # descriptor_binding_storage_buffer_update_after_bind
        false,  # descriptor_binding_uniform_texel_buffer_update_after_bind
        false,  # descriptor_binding_storage_texel_buffer_update_after_bind
        false,  # descriptor_binding_update_unused_while_pending
        false,  # descriptor_binding_partially_bound
        false,  # descriptor_binding_variable_descriptor_count
        false,  # runtime_descriptor_array
        false,  # sampler_filter_minmax
        true,   # scalar_block_layout  ← BDA struct layout
        false,  # imageless_framebuffer
        false,  # uniform_buffer_standard_layout
        has_subgroup_extended,  # shader_subgroup_extended_types ← 64-bit subgroup shuffles
        false,  # separate_depth_stencil_layouts
        false,  # host_query_reset
        true,   # timeline_semaphore  ← REQUIRED for explicit-queue cross-queue sync
        true,   # buffer_device_address  ← REQUIRED (BDA)
        false,  # buffer_device_address_capture_replay
        false,  # buffer_device_address_multi_device
        true,   # vulkan_memory_model  ← REQUIRED (QueueFamily scope)
        false,  # vulkan_memory_model_device_scope
        false,  # vulkan_memory_model_availability_visibility_chains
        false,  # shader_output_viewport_index
        false,  # shader_output_layer
        false;  # subgroup_broadcast_dynamic_id
        next=var_ptr_features
    )
    # Vulkan 1.3 core features, bundled.  We turn on the set that's
    # guaranteed by the 1.3 core spec and useful for a compute/RT backend:
    #   - synchronization2        (REQUIRED: we use vkQueueSubmit2 / cmd_pipeline_barrier_2)
    #   - dynamic_rendering       (no VkRenderPass/VkFramebuffer boilerplate)
    #   - maintenance4            (relaxes shader requirements, e.g. storage-buffer layout)
    #   - subgroup_size_control   (query/set subgroup size)
    #   - compute_full_subgroups  (FULL_SUBGROUPS flag in pipeline create)
    #   - pipeline_creation_cache_control (pipeline-cache hints in create)
    #   - shader_demote_to_helper_invocation (OpDemoteToHelperInvocation in SPIR-V)
    #   - shader_terminate_invocation         (OpTerminateInvocation in SPIR-V)
    #   - shader_integer_dot_product          (OpSDot / OpUDot in SPIR-V)
    #   - shader_zero_initialize_workgroup_memory (zero-init shared mem)
    #   - private_data            (VkPrivateDataSlot for driver-side tagging)
    vulkan13_features = VK.PhysicalDeviceVulkan13Features(
        :synchronization2,
        :dynamic_rendering,
        :maintenance4,
        :subgroup_size_control,
        :compute_full_subgroups,
        :pipeline_creation_cache_control,
        :shader_demote_to_helper_invocation,
        :shader_terminate_invocation,
        :shader_integer_dot_product,
        :shader_zero_initialize_workgroup_memory,
        :private_data;
        next=vulkan12_features,
    )

    # Chain workgroup explicit layout if available
    feature_chain = vulkan13_features
    if has_wg_explicit
        wg_explicit_features = VK.PhysicalDeviceWorkgroupMemoryExplicitLayoutFeaturesKHR(
            true,   # workgroup_memory_explicit_layout
            true,   # workgroup_memory_explicit_layout_8_bit_access
            true,   # workgroup_memory_explicit_layout_16_bit_access
            true;   # workgroup_memory_explicit_layout_scalar_block_layout
            next=feature_chain
        )
        feature_chain = wg_explicit_features
    end

    # Chain pipeline-executable-properties features (opt-in profiling).
    if has_pipeline_exec_props && PIPELINE_EXEC_PROPERTIES_REQUESTED[]
        exec_props_features = VK.PhysicalDevicePipelineExecutablePropertiesFeaturesKHR(
            true;  # pipeline_executable_info
            next=feature_chain
        )
        feature_chain = exec_props_features
    end

    # Chain atomic-float features (used by Lava native reductions via OpAtomicFAdd)
    if has_atomic_float
        atomic_float_features = VK.PhysicalDeviceShaderAtomicFloatFeaturesEXT(
            true,   # shader_buffer_float_32_atomics        — OpAtomicStore/Load/Exchange on f32 SSBO
            true,   # shader_buffer_float_32_atomic_add     — REQUIRED: OpAtomicFAdd on f32 SSBO
            false,  # shader_buffer_float_64_atomics
            false,  # shader_buffer_float_64_atomic_add
            true,   # shader_shared_float_32_atomics        — atomics on workgroup/shared memory
            true,   # shader_shared_float_32_atomic_add
            false,  # shader_shared_float_64_atomics
            false,  # shader_shared_float_64_atomic_add
            false,  # shader_image_float_32_atomics
            false,  # shader_image_float_32_atomic_add
            false,  # sparse_image_float_32_atomics
            false;  # sparse_image_float_32_atomic_add
            next=feature_chain
        )
        feature_chain = atomic_float_features
    end

    # Chain RT features if available
    if has_rt
        as_features = VK.PhysicalDeviceAccelerationStructureFeaturesKHR(
            true,   # acceleration_structure
            false,  # acceleration_structure_capture_replay
            false,  # acceleration_structure_indirect_build
            false,  # acceleration_structure_host_commands
            false;  # descriptor_binding_acceleration_structure_update_after_bind
            next=feature_chain
        )
        rt_features = VK.PhysicalDeviceRayTracingPipelineFeaturesKHR(
            true,   # ray_tracing_pipeline
            false,  # ray_tracing_pipeline_shader_group_handle_capture_replay
            false,  # ray_tracing_pipeline_shader_group_handle_capture_replay_mixed
            true,   # ray_tracing_pipeline_trace_rays_indirect
            false;  # ray_traversal_primitive_culling
            next=as_features
        )
        feature_chain = rt_features
    end
    if has_ray_query
        rq_features = VK.PhysicalDeviceRayQueryFeaturesKHR(
            true;   # ray_query
            next=feature_chain
        )
        feature_chain = rq_features
    end
    if has_ser
        ser_features = VK.PhysicalDeviceRayTracingInvocationReorderFeaturesNV(
            true;   # ray_tracing_invocation_reorder
            next=feature_chain
        )
        feature_chain = ser_features
    end
    if has_coopmat
        cm_features = VK.PhysicalDeviceCooperativeMatrixFeaturesKHR(
            true,   # cooperative_matrix
            false;  # cooperative_matrix_robust_buffer_access
            next=feature_chain
        )
        feature_chain = cm_features
    end
    # Ask the driver which sub-features it actually has before requesting them:
    # `vkCreateDevice` fails outright if a feature struct turns on a bit the
    # device does not support, and an advertised extension does not imply
    # every bit. Everything below follows that shape.
    coopmat2_caps = CoopMat2Caps()
    if has_coopmat2
        q = VK.get_physical_device_features_2(phys_dev,
            VK.PhysicalDeviceCooperativeMatrix2FeaturesNV).next
        coopmat2_caps = CoopMat2Caps(
            true,
            q.cooperative_matrix_workgroup_scope,
            q.cooperative_matrix_flexible_dimensions,
            q.cooperative_matrix_reductions,
            q.cooperative_matrix_conversions,
            q.cooperative_matrix_per_element_operations,
            q.cooperative_matrix_tensor_addressing,
            q.cooperative_matrix_block_loads,
        )
        feature_chain = VK.PhysicalDeviceCooperativeMatrix2FeaturesNV(
            coopmat2_caps.workgroup_scope,
            coopmat2_caps.flexible_dimensions,
            coopmat2_caps.reductions,
            coopmat2_caps.conversions,
            coopmat2_caps.per_element_operations,
            coopmat2_caps.tensor_addressing,
            coopmat2_caps.block_loads;
            next=feature_chain
        )
    end
    if has_coopvec
        q = VK.get_physical_device_features_2(phys_dev,
            VK.PhysicalDeviceCooperativeVectorFeaturesNV).next
        has_coopvec = q.cooperative_vector
        if has_coopvec
            feature_chain = VK.PhysicalDeviceCooperativeVectorFeaturesNV(
                true,                             # cooperative_vector
                q.cooperative_vector_training;    # (inference-only is legal)
                next=feature_chain
            )
        end
    end
    if has_dgc
        q = VK.get_physical_device_features_2(phys_dev,
            VK.PhysicalDeviceDeviceGeneratedCommandsFeaturesEXT).next
        has_dgc = q.device_generated_commands
        has_dgc && (feature_chain = VK.PhysicalDeviceDeviceGeneratedCommandsFeaturesEXT(
            true, q.dynamic_generated_pipeline_layout; next = feature_chain))
    end
    if has_cond_render
        q = VK.get_physical_device_features_2(phys_dev,
            VK.PhysicalDeviceConditionalRenderingFeaturesEXT).next
        has_cond_render = q.conditional_rendering
        # `inherited_conditional_rendering` is for secondary command buffers,
        # which nothing here records into — asked for as reported, not as `true`.
        has_cond_render && (feature_chain = VK.PhysicalDeviceConditionalRenderingFeaturesEXT(
            true, q.inherited_conditional_rendering; next = feature_chain))
    end
    if has_max_reconv
        q = VK.get_physical_device_features_2(phys_dev,
            VK.PhysicalDeviceShaderMaximalReconvergenceFeaturesKHR).next
        has_max_reconv = q.shader_maximal_reconvergence
        has_max_reconv && (feature_chain =
            VK.PhysicalDeviceShaderMaximalReconvergenceFeaturesKHR(true; next=feature_chain))
    end
    if has_subgroup_ucf
        q = VK.get_physical_device_features_2(phys_dev,
            VK.PhysicalDeviceShaderSubgroupUniformControlFlowFeaturesKHR).next
        has_subgroup_ucf = q.shader_subgroup_uniform_control_flow
        has_subgroup_ucf && (feature_chain =
            VK.PhysicalDeviceShaderSubgroupUniformControlFlowFeaturesKHR(true; next=feature_chain))
    end
    if has_subgroup_rotate
        q = VK.get_physical_device_features_2(phys_dev,
            VK.PhysicalDeviceShaderSubgroupRotateFeaturesKHR).next
        has_subgroup_rotate = q.shader_subgroup_rotate
        has_subgroup_rotate && (feature_chain =
            VK.PhysicalDeviceShaderSubgroupRotateFeaturesKHR(
                true,                             # shader_subgroup_rotate
                q.shader_subgroup_rotate_clustered;
                next=feature_chain))
    end

    # Enable shader int64, float64, geometry/tessellation shaders, wide lines
    core_features = VK.PhysicalDeviceFeatures(
        :shader_int_64, :shader_float_64,
        :shader_int_16,
        :geometry_shader, :tessellation_shader,
        :fill_mode_non_solid, :wide_lines, :large_points,
    )

    device = VK.Device(
        phys_dev,
        queue_ci,
        [],         # layers
        extensions;
        enabled_features=core_features,
        next=feature_chain
    )

    queue = VK.get_device_queue(device, qf_idx, 0)
    # Second queue for async compute/RT (falls back to same queue if only 1 available)
    compute_queue = n_queues >= 2 ?
        VK.get_device_queue(device, qf_idx, 1) : queue
    video_decode_queue = video_qf_idx === nothing ? nothing :
        VK.get_device_queue(device, video_qf_idx, 0)

    # Query RT pipeline properties
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

    # Query AS scratch alignment once and cache on the context — used by
    # `bda_alignment_for(::VkContext, scratch::Bool)` to pick the right
    # alignment for `LavaArray(...; scratch=true)` allocations.  Vulkan has no
    # usage flag for "AS scratch", so callers signal via the `scratch` kwarg.
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

    # Clear validation messages accumulated during device creation.
    # GPU-assisted validation emits harmless "adjusting settings" warnings during
    # vkCreateDevice that would otherwise block the first shader compilation.
    clear_validation_messages!()

    # Zero-alloc Vulkan function pointers for hot paths. Per device — see the
    # field's comment on `VkContext`; a global here crashed the first two-device
    # run. Kept in a local until the context exists, and assigned below.
    cmd_barrier_fptr = VK.function_pointer(device, "vkCmdPipelineBarrier")

    mem_props = VK.get_physical_device_memory_properties(phys_dev)
    phys_props = VK.get_physical_device_properties(phys_dev)
    wgc = phys_props.limits.max_compute_work_group_count
    max_wg = (Int(wgc[1]), Int(wgc[2]), Int(wgc[3]))

    # VkContext's inner constructor builds its own default_bq via `new()`-
    # based two-phase init.  Pass the raw primary queue and all other ctx
    # fields; the ctor wires VulkanBatchQueue(device, queue, qfi, ctx) internally.
    # The hardware implements a fixed set of (M, N, K, dtype) tiles; a kernel
    # must choose one of these, it cannot pick an arbitrary tile size.
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
            # "Treating as unsupported" on ANY error is how a real fault becomes a
            # silently slower path. Only an actual absence of the feature is
            # tolerated; anything else is this library's bug and must surface.
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
        has_memory_budget,
        has_external_memory,
        gpu_assisted,
        debug,
        string(phys_props.driver_version),
        has_video_decode, video_decode_queue, video_qf_idx,
    )
    # After construction, because it is resolved from THIS device and the inner
    # constructor has no access to the local. Per device — a module-global one
    # sent the first device's command buffers through the second device's driver.
    ctx.cmd_pipeline_barrier_fptr = cmd_barrier_fptr
    # The pool belongs to this context, so its debug setting is applied here
    # rather than by the caller. It used to be a separate `mempolicy(ctx).disabled =`
    # line every caller had to remember after the reset — and forgetting it left
    # GPU-AV blind to exactly the sub-pool overruns it was turned on to find.
    mempolicy(ctx).disabled = debug.pool_disabled
    return ctx
end

# This backend has one; see `supports_batch_queue` in `graph/queue.jl` for why
# that is a separate question from whether it can rasterise.
#
# Only the `VulkanAPI` method here. The `LavaBackend` one is in
# `array/ka_backend.jl` beside the type: `LavaBackend` is defined seven includes
# after this file, and a method signature needs its argument type to exist when
# the method is DEFINED, not when it is called.
supports_batch_queue(::VulkanAPI) = true

"""
    allocate_batch_queue!(ctx) -> VulkanBatchQueue

Create a new independent VulkanBatchQueue on a separate Vulkan queue (if available).
Falls back to a separate command pool on the primary queue if all queues are taken.
Used by Screen for isolated graphics rendering.

The context holds the returned queue until [`release_batch_queue!`](@ref) gives
it back. Call that when done — a caller that just drops the reference keeps the
command pool, semaphore and slabs alive for the life of the device.
"""
function allocate_batch_queue!(ctx::VkContext)
    idx = isempty(ctx.free_queue_indices) ? ctx.next_queue_index : pop!(ctx.free_queue_indices)
    if idx < ctx.max_queue_count
        queue = VK.get_device_queue(ctx.device, ctx.queue_family_index, UInt32(idx))
        idx == ctx.next_queue_index && (ctx.next_queue_index += 1)
    else
        # All hardware queues taken — reuse primary queue with separate command pool
        queue = ctx.default_bq.queue
        idx = -1
    end
    bq = VulkanBatchQueue(ctx.device, queue, ctx.queue_family_index, ctx; queue_index = idx)
    push!(ctx.extra_queues, bq)
    return bq
end

"""
    release_batch_queue!(bq::VulkanBatchQueue)

Give a queue from [`allocate_batch_queue!`](@ref) back: drain it, destroy what it
still holds, and make its hardware slot available again.

Must be called from the queue's owning thread, and only once nothing will record
on it again — `flush!` below waits for everything already submitted. Releasing
the context's primary queue is an error; it is not one of the handed-out ones.

Dropping a queue without this is what made the finalizer crash possible: the
context keeps it alive (see `extra_queues`), so an unreleased queue is a leak of
a command pool, a semaphore and its argument slabs rather than a dangling
handle.
"""
function release_batch_queue!(bq::VulkanBatchQueue)
    ctx = bq.ctx::VkContext
    bq === ctx.default_bq &&
        throw(LavaError("release_batch_queue!", "the context's primary queue cannot be released",
                        "Only queues from `allocate_batch_queue!` can be given back."))
    i = findfirst(q -> q === bq, ctx.extra_queues)
    i === nothing && return nothing   # already released; releasing twice is a no-op

    # Drain before letting go. On a lost device there is nothing to wait for and
    # every call would fail, so the queue is dropped as-is.
    if !device_lost(ctx)
        flush!(bq, ctx.device)
        drain_deferred_frees!(bq)
        drain_deferred_as_frees!(bq)
    end

    deleteat!(ctx.extra_queues, i)
    bq.queue_index >= 0 && push!(ctx.free_queue_indices, bq.queue_index)
    return nothing
end

"""
    queue_released(bq::VulkanBatchQueue) -> Bool

Whether `bq` has been handed back by [`release_batch_queue!`](@ref).

A released queue is not safe to QUERY. Its `timeline_sem` is a
`VK.Semaphore` whose own finalizer destroys it, and a buffer's `last_write`
names the queue, so buffer and queue can become garbage in the same GC cycle —
where Julia runs finalizers in UNSPECIFIED order. If the semaphore goes first,
`vk_free!`'s `query_timeline` reads a dangling handle and segfaults inside
`run_finalizers`, with no Julia frame naming the buffer. Holding a reference
cannot fix an ordering problem; the query itself has to be safe.

It is, and semantically rather than defensively: `release_batch_queue!` flushes
and drains BEFORE letting go, so a released queue has no in-flight work and any
buffer still naming it can be destroyed immediately instead of waited on.

Membership in `ctx.extra_queues` is already the liveness record, so this needs no
flag — `release_batch_queue!` removing the entry IS the transition.
"""
function queue_released(bq::VulkanBatchQueue)
    ctx = bq.ctx::VkContext
    bq === ctx.default_bq && return false      # the primary queue is never released
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
        # `VK_KHR_driver_properties` is core since 1.2, which every device here is.
        driver = if props.api_version >= v"1.2"
            d = VK.get_physical_device_properties_2(pd, VK.PhysicalDeviceDriverProperties).next
            strip(String(filter(!=('\0'), collect(d.driver_name))) * " " *
                  String(filter(!=('\0'), collect(d.driver_info))))
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
    # A throwaway instance at API 1.2 so `VK_KHR_driver_properties` (core since
    # 1.2) answers; a 1.0 instance reports a blank driver for every device.
    app = VK.ApplicationInfo(v"0.0.0", v"0.0.0", v"1.2.0")
    instance = VK.Instance(String[], String[]; application_info = app)
    return deviceinfos(unwrap(VK.enumerate_physical_devices(instance)))
end

# The process default is chosen by `MANTLE_DEVICE` when it is set (a name
# substring, the same selector `Device(VulkanAPI(); select)` takes) and by the
# kind ranking in `selectdevice` otherwise: the best GPU there is. This replaces
# pinning the loader to one ICD through `VK_DRIVER_FILES`, which also hid every
# other device from the two-device tests.
defaultselector() = get(ENV, "MANTLE_DEVICE", nothing)

function find_graphics_compute_queue_family(phys_dev)
    qf_props = VK.get_physical_device_queue_family_properties(phys_dev)
    # Prefer graphics+compute (needed for graphics pipeline support)
    for (i, qfp) in enumerate(qf_props)
        if (qfp.queue_flags & VK.QUEUE_COMPUTE_BIT) != 0 &&
           (qfp.queue_flags & VK.QUEUE_GRAPHICS_BIT) != 0
            return UInt32(i - 1)
        end
    end
    # Fall back to any compute-capable queue (graphics won't work but compute will)
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
Find a secondary queue family distinct from `primary_qf_idx` that supports
compute + transfer. On RDNA/RADV the primary family is graphics+compute with
1 queue; the async family is compute-only with 4 queues — ideal for the
upload/dispatch split. Returns `nothing` if no such family exists.
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
Find a queue family that supports hardware video decode
(`VK_QUEUE_VIDEO_DECODE_BIT_KHR`, bit 0x20 — not surfaced as a named flag in
VK.jl). Dedicated on NVIDIA (transfer + video-decode, no graphics/compute).
Returns `nothing` if the device has none.
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

"""Check if the physical device supports the RT extensions we need."""
function has_extension(phys_dev, ext_name::String)
    available = unwrap(VK.enumerate_device_extension_properties(phys_dev))
    for ext in available
        name = String(filter(!=('\0'), collect(ext.extension_name)))
        name == ext_name && return true
    end
    return false
end

function has_rt_extensions(phys_dev)
    available = unwrap(VK.enumerate_device_extension_properties(phys_dev))
    names = Set{String}()
    for ext in available
        push!(names, String(filter(!=('\0'), collect(ext.extension_name))))
    end
    return "VK_KHR_acceleration_structure" in names &&
           "VK_KHR_ray_tracing_pipeline" in names &&
           "VK_KHR_deferred_host_operations" in names
end

# ── Validation layer debug messenger ──

# Async-safe Vulkan debug callback.  MUST NOT allocate, log, take locks, or
# hit a scheduler yield/safepoint — it can run re-entrantly inside a blocking
# driver ccall (GPU-AV readback during vkWaitSemaphores) with driver locks
# held, where any of those deadlocks the runtime.  All it does is copy the
# message bytes into the preallocated ring via raw ccalls + @inbounds stores.
# Classification + logging happens later on the main thread in
# `drain_validation_messages!`.  The flag args are received as raw UInt32 (the
# C ABI for VkDebugUtilsMessageSeverity/TypeFlags) so no flag-wrapper is built.
function debug_callback(
    severity::UInt32,
    type::UInt32,
    p_callback_data::Ptr{VK.VkCore.VkDebugUtilsMessengerCallbackDataEXT},
    p_user_data::Ptr{Cvoid},
)
    (p_callback_data == C_NULL || p_user_data == C_NULL) && return UInt32(0)
    # One isbits load, no allocation: which device's ring this messenger belongs
    # to is the pointer the driver hands back, not a global.
    r = unsafe_load(Ptr{ValidationRingRaw}(p_user_data))
    data = unsafe_load(p_callback_data)        # isbits C struct → stack value, no heap alloc
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
    # Return VK_FALSE — can't throw from a @cfunction callback (would corrupt
    # Vulkan state).  Errors are surfaced via drain_validation_messages!.
    return UInt32(0)
end

# Classify one drained message and, for hard validation errors, capture it.
# Runs ONLY on the main thread (called from drain), so logging/alloc is safe.
function handle_validation_message(r::ValidationRing, message::String, sev_u::UInt32, type_u::UInt32)
    # Message-type bits classify the source (Vulkan spec):
    #   VALIDATION (0x2) — VK_LAYER_KHRONOS_validation + driver spec checks
    #   GENERAL    (0x1) — driver runtime notes (not spec checks)
    #   PERFORMANCE (0x4), DEVICE_ADDRESS_BINDING (0x8) — self-explanatory
    # Only VALIDATION-typed errors are captured as hard failures.
    is_error = (sev_u & UInt32(VK.DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT)) != 0
    is_warning = (sev_u & UInt32(VK.DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT)) != 0
    is_validation = (type_u & UInt32(VK.DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT)) != 0
    # Known-benign device-creation chatter that must NOT be captured as a hard
    # validation error.  The "Device Layers have never worked" notice is a VVL
    # deprecation (VUID-VkDeviceCreateInfo-ppEnabledLayerNames-12385) triggered
    # because VK.jl's generated `_DeviceCreateInfo` always passes a non-null
    # `ppEnabledLayerNames` even for an empty layer list (count is 0, so it is
    # functionally ignored by the loader).  It is a dependency-side quirk with
    # no runtime effect; log it as a warning but don't treat it as a failure.
    is_setup_noise = contains(message, "adjusting settings") ||
                     contains(message, "VALIDATION-SETTINGS") ||
                     contains(message, "Device Layers have never worked")
    # @lava_printf output arrives at INFO severity. The Khronos layer wraps it as
    #   "vkQueueSubmit2(): pSubmits[0] DebugPrintf:\n<user text>"
    # Capture just the user text into `r.printf`, separate from validation.
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
        # Driver-general error (e.g. chatty lavapipe SPIR-V notes). Log but
        # don't capture — the authoritative signal is the VkResult of the
        # surrounding API call, which @vk_checked already unwraps.
        @warn "Vulkan driver note (type=$(type_u), not a validation error)" message
    elseif is_warning || is_setup_noise
        @warn "Vulkan validation warning" message
    end
    return nothing
end

"""
    drain_validation_messages!()

Convert any messages the async callback wrote into the ring into Strings,
capture hard validation errors in `ctx.validation.messages`, and log them. Must be
called on the main thread (it allocates + logs). Idempotent — only processes
slots written since the last drain. Call this before reading
`ctx.validation.messages` and while polling for a GPU-AV fault.
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

"""
    layer_setting_bool(name, value) -> VK.LayerSettingEXT

One `VK_EXT_layer_settings` boolean for the Khronos validation layer.

`VkLayerSettingEXT` takes a raw `pValues` pointer, so the storage has to outlive
the `vkCreateInstance` call that reads it. `LAYER_SETTING_STORAGE` keeps every
value alive for the process — a handful of bytes, set once at instance creation,
and the alternative is a pointer into a collected `Ref`.
"""
const LAYER_SETTING_STORAGE = Any[]

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

    # Debug printf is delivered at INFO severity, so subscribe down to INFO when
    # it's enabled; otherwise stay at WARNING to avoid info-level chatter.
    #
    # Read from the config this device was built with, not from the environment.
    # As `LAVA_DEBUG_PRINTF` it was the one env read that happened AFTER the
    # instance existed, so exporting it mid-session moved the severity floor on
    # a device whose instance had no printf feature — a subscription to messages
    # that could never arrive.
    min_sev = debug.printf ?
        VK.DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT :
        VK.DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT
    # `user_data` is what makes the ring per-device: the driver hands this
    # pointer back to every invocation, so the callback never has to guess which
    # context it is reporting for.
    index = findfirst(==(min_sev), VK.message_severities)
    severity = |(VK.message_severities[index:end]...)
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
    get_validation_messages() -> Vector{String}

Return recent validation layer messages. Useful for diagnosing DEVICE_LOST errors.
"""
get_validation_messages(ctx::Union{Nothing,VkContext} = VK_CONTEXT_REF[]) =
    ctx === nothing ? String[] :
        (drain_validation_messages!(ctx); copy(ctx.validation.messages))

"""
    clear_validation_messages!()

Clear the validation message buffer. Drains the async ring first (so captured
hard errors are logged), then empties the capture list.
"""
clear_validation_messages!(ctx::Union{Nothing,VkContext} = VK_CONTEXT_REF[]) =
    ctx === nothing ? String[] :
        (drain_validation_messages!(ctx); empty!(ctx.validation.messages))

"""
    get_printf_output() -> Vector{String}

Return captured `@lava_printf` output since the last clear. Drains the async
callback ring first. Requires a device built with `DebugConfig(printf = true)`.
"""
get_printf_output(ctx::Union{Nothing,VkContext} = VK_CONTEXT_REF[]) =
    ctx === nothing ? String[] :
        (drain_validation_messages!(ctx); copy(ctx.validation.printf))

"""
    clear_printf_output!()

Drop captured `@lava_printf` output (drains the ring first).
"""
clear_printf_output!(ctx::Union{Nothing,VkContext} = VK_CONTEXT_REF[]) =
    ctx === nothing ? String[] :
        (drain_validation_messages!(ctx); empty!(ctx.validation.printf))

"""
    check_validation_errors!(context::String)

Check if any validation errors were captured since the last check.
Throws `LavaError` with the error messages if any errors are found.
Call this after Vulkan operations that may trigger validation errors
(shader module creation, pipeline creation, dispatch recording).
"""
function check_validation_errors!(context::String,
                                 ctx::Union{Nothing,VkContext} = VK_CONTEXT_REF[])
    ctx === nothing && return
    drain_validation_messages!(ctx)
    msgs = ctx.validation.messages
    isempty(msgs) && return
    # Separate true errors from warnings using the severity recorded by the callback.
    # GPU-AV messages like "Unaligned pointer access" are errors, not warnings,
    # even though their text may contain strings like "WARNING-Validation".
    # We rely on the callback storing only ERROR-severity messages.
    errors = copy(msgs)
    isempty(errors) && return
    n = min(length(errors), 5)
    detail = join(["  [$i] $(first(errors[i], 1000))" for i in 1:n], "\n")
    # Clear after reporting to avoid re-triggering
    empty!(msgs)
    throw(LavaError(
        context,
        "Vulkan validation error(s):\n$detail",
        "Fix the validation errors above before proceeding."
    ))
end

# Mantle's device-idle wait, in Vulkan's terms. Declared in
# `src/graph/queue.jl`; this is the method. The `LavaBackend` spelling — for a
# caller who holds the KA backend and should not have to reach for the context
# to get at the handle underneath it — is in `array/ka_backend.jl`, with the
# type.
waitidle(ctx::VkContext) = VK.device_wait_idle(ctx.device)
waitidle(d::VK.Device) = VK.device_wait_idle(d)

# Vulkan rasterizes. See `supports_graphics` in `src/graphics/commands.jl`.
supports_graphics(::VkContext) = true

"""
    vulkan_available() -> Bool

Whether this machine can actually give us a Vulkan device.

`MantleVulkanExt` only loads when `using Vulkan` succeeded, and VulkanCore's own
`__init__` raises when it cannot `dlopen` a loader — so reaching this function
means a loader exists. What is still open is whether it exposes a device, and
the only honest way to find out is to build the context, which `vk_context`
caches, so the cost is paid once either way.

The `catch` converts exactly two expected failures — no ICD, no suitable device
— into "not available", which is the question being asked. Anything else is a
real bug and is rethrown rather than reported as a missing GPU.
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
