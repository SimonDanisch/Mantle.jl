# Review of recording-plan.md against the code

2026-09-26. Nine reviewers read all of `src/` and `ext/` (52.6k lines) at
7e64cd5 plus the uncommitted changes in `src/graph/{backend,build,kalaunch,types}.jl`,
`src/metal/record.jl`, `src/vulkan/graph.jl`. Code only; comments were not used
as evidence. Paths are relative to `src/` unless stated.

Items marked (verified) were re-checked by hand after the review. Everything
else is from reading code, not from running it.

## 1. Bugs found on the way

| where | what | status |
|---|---|---|
| `ext/MantleROCmExt.jl:873` vs `graph/kalaunch.jl:966,1017` | `openrecording(d, pl)` has 2 args, core calls 3 since 2dc58eb; the default returns `nothing`, so ROCm never records and `run!` throws "not recorded" on every headless plan | verified |
| `vulkan/runtime/memory.jl:1023` | `destroy_pool!` calls `empty!(p.pending)` on an `Inbox`; no such method, so `reset_device!` with a live context throws | verified |
| `vulkan/runtime/command.jl:1147-1150` | `flush_stall_report` reads `wait_semaphores`/`signal_value` from core `Submission{R}` (fields `recording`, `holds`); the timeout diagnostic throws | verified |
| `vulkan/array/gemm_cm2.jl:324,362` | `gemm_cm2_sg!(backend, n)(…)`/`gemm_cm2!(backend, NT)(…)` call plain functions as kernel factories: MethodError | verified |
| `graph/build.jl:342-354` + `graph/phases.jl:556-679` | `when!`: Barriers assume every pass runs. W writes B, P (conditional) reads B, Q reads B: read-after-read gets no barrier (`sync/transition.jl:98-99`), so only P's piece carries W→read. Skipping P leaves Q unordered against W; layout transitions inside a skipped piece break the same way | rule verified, scenario derived |
| `metal/device.jl:448,1141`, `metal/record.jl:1219` | `Metal.synchronize()` with no device in `waitfor(::LegacyQueue)`/`waitidle`/`openrun`: with two devices completion is credited from the wrong queue and the pool reuses in-flight memory | derived |
| `metal/record.jl:153-156,345,368` | walked Metal plans keep compile-time arrays, `patchable` is `true`, no move listener: stale reads after `resize!` or arena growth (the host backend's old bug) | derived |
| `vulkan/graphics/framebuffer.jl:254-257` vs `vulkan/graphics/pipeline.jl:449-452` | readback/copy leave the image in TRANSFER_SRC; a later LOAD pass assumes COLOR_ATTACHMENT_OPTIMAL | derived |
| `vulkan/runtime/device.jl:1735`, `command.jl:1257,1330`, `vulkan/raytracing/pipeline.jl:86-133` | validation checks read `VK_CONTEXT_REF[]` inside functions that hold a context: they check (or clear) the wrong device | derived |
| `memory/pool.jl:920-922` | weak reference dereferenced without the `nothing` check that 1082 has | derived |
| `vulkan/runtime/memory.jl:675` | unlocked `delete!` on `live_buffers` from finalizer context races `push!` at 635 | derived |
| `vulkan/array/gpuarrays.jl:695` | `append!` writes at `old_len*sizeof(T)` ignoring `a.offset` | derived |
| `vulkan/runtime/video_h265.jl:605,635` | aligns bitstream offsets to 256 where the shared core sizes by `bsalign` | derived |
| `vulkan/graphics/textures.jl:211-219` vs `vulkan/lowering.jl:239` | `Float32` maps to `R32_SFLOAT` in one table, `D32_SFLOAT` in the other | derived |
| `metal/trace.jl:162-163` | `unsafe_convert(Ptr, Ref(n))` without `GC.@preserve` | derived |
| `metal/trace.jl:90` | custom index returned as `instance_id` while TLASes are built with UserID descriptors | derived |
| `vulkan/runtime/device.jl:1672-1677` | geometry/tessellation/wide lines/float64 requested without querying support | derived |

## 2. Uncommitted work already implements item 8, differently

`graph/kalaunch.jl:1059-1121`, `graph/types.jl:669-686`, `graph/build.jl:1444-1462`,
`graph/backend.jl:606-632`, `vulkan/graph.jl:1473`. `maxpasses`/`record_maxpasses`
are gone; core cuts by `budget` and measured `passcost`. As built:

- an unmeasured pass costs `Inf`, so a plan's first run (and every `runonce!`)
  submits once per pass (`kalaunch.jl:1068`);
- `measure!` invalidates and `run!` re-records (`kalaunch.jl:633-637,1119`),
  which the plan forbids;
- `when!` passes are never measured (`kalaunch.jl:1092-1094,1108`), so a
  conditional region stays one submission per pass forever;
- every Vulkan plan gets a query pool, and `Plan()` throws on a device with
  `timestampPeriod == 0` even without profiling (`vulkan/graph.jl:559-561`);
- Metal and ROCm are never budget-split (`timestamps(::Device) = false`).

The plan's status line ("Nothing below is implemented") and its `maxpasses`
references are stale.

## 3. What the plan does not cover (structural)

### 3.1 Memory: most device memory is outside the pool

- **Vulkan's second allocator.** `vk_alloc` (`vulkan/runtime/memory.jl:431-644`)
  gives each buffer its own `VkDeviceMemory`, accounted in `live_bytes`, freed
  through `retire!(::SubmitChannel)`. It serves every `LavaArray` with
  `extra_usage`/`scratch`/`unified` (`vulkan/array/lavaarray.jl:68-71`), every
  allocation of at most 64 B (`vulkan/array/ka_backend.jl:156`), index buffers
  (`:183-188`), `reduce_scratch` (`vulkan/array/mapreduce.jl:60`), BLAS/TLAS
  storage, build and refit scratch, instance buffers
  (`vulkan/raytracing/acceleration.jl:218-617`, `hwtlas.jl:389,723`), the SBT
  (`vulkan/raytracing/pipeline.jl:409`) and indirect draw buffers
  (`vulkan/graphics/pipeline.jl:621-623`). Textures and framebuffers get
  dedicated memory per image (`vulkan/graphics/framebuffer.jl:87,98,201-206`,
  `textures.jl:112`); video and external memory too (`vulkan/runtime/video.jl:597,691,724`,
  `external.jl:55`).
- **Metal.** Argument memory (`metal/record.jl:62-67`, retired through
  Metal.jl batch roots at `:80-92`), ICBs, aux and gate vectors (`:726-750`),
  textures and framebuffers (`metal/graphics.jl:129,264,273`), acceleration
  structures, scratch, instance and payload buffers (`metal/raytracing.jl:122-284`,
  `hwtlas.jl:382,543-557`, `procedural.jl:253`), `indexbuffer`/`devicearray`
  as `MtlArray` (`metal/graphics.jl:1270-1294`), `KI.allocate`
  (`metal/kernelinterface.jl:27-29`).
- **ROCm.** `rawalloc` is AMDGPU's pool with its own last-stream tracking
  (`ext/MantleROCmExt.jl:353-369,917`).
- **Core.** `devicearray`/`Adapt` via `KA.allocate` (`memory/array.jl:166-198`);
  `repeat!`'s predicate `Buffer` per loop (`graph/build.jl:374`); a query pool
  per plan (`graph/build.jl:1454`).
- **Pressure policy exists only in Vulkan.** GC pressure, soft cap, OOM retry
  with `flush!` + `GC.gc(true)` on the allocation path
  (`vulkan/runtime/memory.jl:176-366,498-541,1079-1298`). Core's pool has none.
- **Nothing is given back.** `trim!` (`memory/pool.jl:816-836`) has no core
  caller; its one caller is Vulkan's OOM path, gated by `movable`, which is
  false whenever any plan is recorded. An arena keeps its largest-ever size
  until its last tenant leaves (`pool.jl:891-895,961-976`); superseded growth
  blocks stay reserved (`:610-614`). Sparse arenas as planned also only grow.
- **Placement is stored in the graph.** Compile writes `first/last/block/offset`
  into the `TransientBuffer` declarations (`graph/phases.jl:382-388,461`,
  `graph/build.jl:1683`): two plans compiled from one graph overwrite each other.
- **Wrong refusals under pressure.** `headroom` counts the arena's own region
  as reserved (`memory/pool.jl:481-482`, `graph/phases.jl:433`); `capacity`
  means "budget incl. ours" on Vulkan and "free now" on ROCm/host.
- **Hoarding caches.** `gemm_split_scratch`/`gemm_split_retired` only grow
  (`vulkan/array/gemm.jl:2391-2405`); `reduce_scratch`; pipeline caches whose
  count bound is defeated by `linked`/`launchplans`/`frozen_mem`
  (`vulkan/runtime/pipeline.jl:482-491`); `iterplans` per ndrange.
- **Non-region objects lose their engine in phase 3.** AS handles, dedicated
  buffers, images, descriptor pools are freed through the channel's
  `retire!`/`reclaim!`/`rawfree` (`graph/lifetime.jl:481-560`); the pool's
  `retire!` takes only `Region`s. VK.jl handle finalizers call the driver from
  the GC (`VK.Image`, `DeviceMemory`, `DescriptorPool`, pipelines).
- **Device identity.** `Device(MetalAPI(); select)`, `Device(ROCm…)`,
  `Device(HostAPI())` build a new device with a new `Pool` per call
  (`metal/device.jl:281`, ROCm `:241`, `host/host.jl:40,70-71`): two pools over
  one GPU cannot share or trim together. `DEVICES` never releases
  (`vulkan/graph.jl:51`).
- **Pool lock held across a GPU wait** (`memory/pool.jl:590-619`), shared with
  every `run!`'s patch step.

### 3.2 Ordering between submissions

"Ordered by submission" orders execution, not memory. Every plan's transients
start at offset 0 of one shared arena. Today the only thing ordering them is
Vulkan's `emithead!`, an ALL_COMMANDS barrier at the head of every recording and
one-shot (`vulkan/graph.jl:1262-1269`, `vulkan/runtime/command.jl:360-376,441-447`);
core's default is a no-op (`graph/backend.jl:127`); Metal's `emithead!` is a
range reset, not a barrier (`metal/record.jl:1127-1141`). The Barriers phase
seeds only from the plan's own final usages (`graph/phases.jl:610-621`). The
builder verbs have no head barrier, and the full barrier serialises independent
plans side by side.

The Dag and Barriers also disagree: the Dag adds an edge for two `Unordered`
writers (`graph/phases.jl:198-212`), Barriers skip them (`sync/transition.jl:95-105`).
`needs_transition(::ROCmAPI) = false` (`sync/backend.jl:79`) assumes one
in-order stream, which HIP graph branches break.

### 3.3 Acceleration structures and textures are not graph resources

`trace!` never declares the accel (`raytracing/api.jl:103-104`); `TraceRead`/
`TraceBuild` (`sync/usage.jl:62-63`) are declared by nothing. Builds and refits
are one-shots with backend-chosen barriers and host waits
(`vulkan/raytracing/acceleration.jl:989-1034`, `metal/raytracing.jl:84-99`).
`Raycore.sync!` swaps `hw_tlas` and `resize!`s `tri_gpu`/`off_gpu`
(`vulkan/raytracing/hwtlas.jl:703,884,891`), which recordings bake as
descriptor-set handles (`vulkan/runtime/pipeline.jl:513-538`, `vulkan/graph.jl:1176`),
and an address patch cannot rebind a handle. Metal bakes `gpuResourceID`s into
`scene_buf` (`metal/hwtlas.jl:551-566`). `Adapt.adapt_structure(::VulkanTLAS)`
and `(::MetalHWTLAS)` can trigger a rebuild. HWTLAS bookkeeping and rebuild/refit
policy exist twice (`vulkan/raytracing/hwtlas.jl:109-924`, `metal/hwtlas.jl:265-586`),
and `build_accel!`/`refit_tlas!` have different signatures per backend. The
Metal bake's "builds as commands of the plan" has no declaration verb. Texture
bindings (`bind_textures`, one descriptor pool per call) sit outside plans.

### 3.4 Second paths beside the graph

- Hand-recorded render passes: `pass!`/`blit!`/`draw!` on a device
  (`graphics/record.jl:164-321`) over `begin_render_pass!`…`end_render_pass!`,
  which submits (Vulkan via `openoneshot`/`handover!`, deleted in phase 1;
  Metal one `commit!` per pass). RayMakie composites through it every frame.
- A second render verb family exported from core: `begin_pass!`,
  `draw_in_pass!`, `set_viewport!` (`graphics/commands.jl:12-60`).
- Immediate ray tracing: `trace_rays!`, `trace_closest_hits!` and variants
  (`raytracing/api.jl:59-171`), with backend-specific signatures and hand-placed
  barriers.
- Eager array library: `fft`/`rfft`/`stft`/`gemv` (`array/fft.jl`, `array/gemv.jl`),
  GPUArrays broadcast/`fill!`/`copyto!`/`resize!`, `mapreducedim!`,
  `accumulate!`, `mul!`, `norm` (`vulkan/array/*`). FFT's eager and graph forms
  already diverge (`rfft!` packs by reinterpret, `rfft_dispatch!` by kernel).
  `array/launch.jl`'s `ArrayLaunch` competes with the plan's `Op`/`lower`.
- Eager device-sized dispatch `ka_launch_indirect!` (`vulkan/array/ka_backend.jl:924-997`)
  beside `DeviceRange`; the `WORKGROUP_FALLBACK` codegen workaround exists on
  the eager path only.
- `LavaBackend` holds a dispatch queue and an upload queue
  (`ka_backend.jl:54-61`).

### 3.5 Windows and present

Fences and binary semaphores per frame slot, `vkQueuePresentKHR`, implicit
swapchain rebuild with `vkDeviceWaitIdle` inside acquire/present
(`vulkan/graphics/window.jl:211,240-249,311-438`). `submitnative!` has no
signals, fence or present. The Threads claim is wrong: `vkQueuePresentKHR`,
`vkDeviceWaitIdle` and `vkQueueBindSparse` need external synchronisation too.
GLFW calls must run on the main thread. Metal has no image index (a new
drawable per frame).

### 3.6 Queues

Decision 4 ("nothing outside the tests allocates a second queue") is false:
`VkContext` requests up to 4+4 queues and a video queue
(`vulkan/runtime/device.jl:1213-1240`); `compute_queue` is submitted to by
`external.jl:135-136`; the video queue by `video.jl:106` (with no ordering
against main-queue writes to `dst`); `LavaBackend`'s upload queue; Metal's
legacy queue beside the MTL4 queue for passes, present, uploads, AS builds and
traces (`metal/graphics.jl:789-794`, `raytracing.jl:85`, `trace.jl:153`).

### 3.7 Backend decisions and duplication the plan does not name

- GEMM/GEMV selection in `vulkan/array/gemm.jl` (tilings, variants, split-K
  with `cores = 48`, padding, GEMV split), `SGEMM_*` in `Mantle.jl:278-293`
  read by DNNKernels, DNNKernels calling Vulkan-only `coopmat_gemm_dispatch!`
  etc. directly. `native_batched_gemm_dispatch!` is not in the delete list and
  `Gemm` has no batch or `alpha`.
- `compile_dispatch` written three times (`vulkan/graph.jl:958-990`,
  `metal/record.jl:321-410`, ROCm `:650-726`) with diverging launch geometry
  (`ki_launch_extents` vs `KI.auto_launch_sizes`), autotune on two of three,
  and different `DeviceRange`-without-max handling.
- Patch table built twice: `notepacked!` (`graph/packing.jl:276-285`) and
  `recpatchfields!` (`vulkan/runtime/command.jl:147-156`).
- `openrecording` returning `nothing` lets a backend decide walked vs
  recorded; Metal reads the graph to segment and gate
  (`metal/record.jl:588-636,961-1050`) and refuses core's partition after the
  first piece (`:699-705`).
- The host backend replaces core's Barriers phase
  (`run!(::Barriers, ::Compile{HostDevice}) = c`, `host/host.jl:215`).
- Metal chooses the geometry→mesh fallback (`metal/mesh.jl:475-479,563-646`)
  and compiles it during `record_draw!` because `compile_draw` is not told
  whether a draw is indexed.
- `closerun!`/`submitrun!`/present loops in three backends plus core.
- KernelInterface host methods duplicated Vulkan/Metal; GEMV twice (core
  `array/gemv.jl` and `vulkan/array/gemm.jl:2664-2787`); window/GLFW, sampler
  and format tables per backend.
- Flush timeout policy, device-lost state machine, `indirectbarrier!`,
  `headbarrier!` in Vulkan.

### 3.8 Dispatch instead of probing

`buildskernel` = `hasmethod` + file-path string match (`graph/kalaunch.jl:166-168`);
`kisupported` (`:191-192`); `hasmethod(target_image, …)` (`graph/build.jl:1148`);
`wantsvertexindex`; TLAS found by `hasfield(T, :hwtlas)`
(`vulkan/array/ka_backend.jl:372,385`); `try/catch` probing
(`metal/kernelinterface.jl:57-64`); `Pass.kind::Symbol` branched in core and
Metal; `front === :mesh`; `copy_buffer!(::Symbol)`. Zero-argument hooks answered
by whichever backend was `@static include`d (`staged_gemm_tile()`,
`use_frozen_kernels`, `initbackend!`, `intrinsic_usage(::Symbol)`): with the ROCm
extension loaded a ROCm device gets Vulkan's GEMM tile. Backends are chosen by
`@static Sys.isapple()` (`Mantle.jl:555-581`), so CUDA or HIP cannot load
beside Vulkan.

### 3.9 Global and ambient state not in phase 4

`DEVICES`/`lavadevice(ctx)`; `VK_CONTEXT_REF` in `@vk_checked`,
`check_validation_errors!`, atexit cache save, diagnostics; `device()`,
`device(select)`, `defaultdevice!` (exported); `LIVE_CONTEXTS`,
`LAYER_SETTING_STORAGE`, `PIPELINE_NO_COMPILE`, `WORKGROUP_FALLBACK`,
`KERNEL_RECORDERS`, `BACKEND_PROBES`, `FROZEN_*`; ROCm's `ROCM_DEVICE` and
task-local `adopt!` (ROCm has no phase at all); Metal's `Metal.synchronize()`,
task-local `Metal.device()` compiles, Metal.jl's linked/table function
registries (`metal/hwtlas.jl:106`, `procedural.jl:219,333`), `COLLECT[]` never
reset; graphics constructors keyed on the KA backend with no device
(`Texture2D`, `Sampler`, `Framebuffer`, `Window`, `readback_*`); `Window(w, h)`
calling `defaultbackend()` (`runtime/api.jl:102-103`); `frozen_last_pcache` as
a channel through a context field. Phase 4's "roughly 30 sites" is about 75-86.

Unlocked shared state under "record on any thread": `iterplans`, `launchplans`,
`AccessCache`, `gfx_pipelines`/`rt_pipelines` (hash-keyed; the graphics key
stores the descriptor layout as a Bool), `reduce_scratch` written from the host,
`dispatch_log`/`last_dispatch_info` written at emit, `LavaTLAS.desc_sets`.

## 4. Contradictions inside the plan

1. "`run!` never records" and phase 2's deletion of the re-record vs item 8's
   split by measured timings (and the working tree re-records in `run!`).
2. `Recording` is `mutable` with `last`; `finish(b)` returns "an immutable
   Recording"; `Recording(sub.queue, finish(b), pl)` has 3 args for 4 fields.
3. One queue per device vs "every submission names its queue" and
   `Recording.queue`.
4. `run!` "allocates 0 bytes" and "submits exactly once" vs the per-run front
   `OneShot`, Metal run staging, budget cuts and foreign-node segments.
5. Loop body "submitted n times in one submission" vs "loop iterations are
   natural cut points".
6. Metal `Builder` = "new `MTLCommandBuffer`", `recycle!` = "drop" vs the MTL4
   allocator/command-buffer free list; "an ICB sized from the submission's pass
   range" has the backend inspect passes.
7. Phase 1 deletes `Closed`/`hold!(::Closed)`, the only input `crosswaits!`/
   `stamp!` have until phase 3 deletes them.
8. Phase 2 deletes `Plan.stale`, but resize (`graph/build.jl:721`), image-arena
   growth (`memory/pool.jl:907`), non-patchable devices (`:1106-1108`) and
   `measure!` rely on it, and their replacements come later or never.
9. Phase 1 "adds `oneshot`" and "deletes the non-bang `oneshot`"; the existing
   `oneshot!` is never mentioned.
10. The new names collide with existing ones: `Submission{T}` vs core
    `Submission{R}` (exported), `recycle!(device, n)` vs `recycle!(q, payload)`/
    `recycle!(ch, s)`, `completed`/`waituntil` vs `fence`/`passed`/`waitfor`/
    `waitfor!`. `SubmitChannel`, `Outstanding`, `sweep!`, `drain!`, `flush!`,
    `batchqueue`, `supports_batch_queue`, `release_batch_queue!` have no fate.
11. "Finalizers push onto a locked pending list" would undo 1e26aeb (lock-free
    `Inbox` because a lock in a finalizer fails).
12. The host backend: "refused, not walked" vs phase 6 deleting `Immediate`
    only on devices that record; the Host column vs the "Four backends" table
    naming CUDA/HIP; host thunks reading `DeviceRange` counts and predicates at
    run.
13. `forbid_defaults!()` is a process-global switch to test for no globals;
    it misses `device()`/`defaultdevice!`.
14. `test_record_is_pure` counts compiles with `kernelcompiles`, which reads
    Lava's process-global frozen counters and stays 0 in the default
    configuration (`vulkan/graph.jl:168`, `vulkan/runtime/launch.jl:636`): the
    test cannot fail.
15. Decision 1 schedules growth as "its own phase, after phase 3"; no such
    phase exists, and phase 5 lists sparse arenas inside the Metal work. Item 5
    grows persistent KV caches "in the sparse arena", which holds transients;
    the doubling fallback makes sharing a sum, not a max.
16. "Out of scope: caches keyed by device": already true for `DeviceCaches`
    and `AccessCache`; what is global is `FROZEN_*`; thread safety of the caches
    is needed by `test_any_thread` and not mentioned.
17. `emit_trace!` → `emittrace!` "to match the rest": `emit_dispatch!`,
    `emit_dispatch_indirect!`, `emit_barrier!`, `emit_draw!`,
    `emit_trace_indirect!` also have underscores.

## 5. Stale references

`graph/backend.jl:881-888` (now a comment in the vocabulary; `CompiledDispatch`
is `graph/types.jl:484`); `kalaunch.jl:634` → 637, `679-690` → 682-693;
`pool.jl:565,735-741` → 764-768 (inside `reclaim!`), `844-906` → 872-934;
`build.jl:356-400` → 356-413; `access.jl:148-153` is `argument_usage`,
`kerneltouches` is 1160-1173; `get_or_build_iter_plan` is defined at
`vulkan/array/ka_backend.jl:402`; Metal `graphics.jl:742` → 787,
`window.jl:255` → 268; `emit.jl:3980-4045` → 4101-4111; `test_hold_lifetime.jl`
is under `test/vulkan/`, assertions at 47-52 (also `OneShot.sync`,
`Recording.holds`, `VkManagedBuffer.stamp`); ROCm "re-records on every move" is
"recompiles via `refit!`, moved persistent Buffers refused"; ROCm's own library
is hipBLASLt, rocBLAS only via a consumer's `mul!`; Metal "five measurements"
are six, four already resolved; `HWTLAS_FUNCS`/`PROC_FUNCS` already key by
device.
