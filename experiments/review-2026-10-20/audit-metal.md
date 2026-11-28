# Audit: `src/metal/` against `docs/api.md` and `docs/recording-plan.md`

Historical snapshot: counts, locations and deletion hypotheses below predate
consolidation. Follow [current ownership and deletion gates](consolidation-order.md).

Scope: every line of the 17 files under `src/metal/` (8,649 lines), compared
against `src/vulkan/graph.jl`, `src/vulkan/runtime/*.jl`,
`src/vulkan/graphics/*.jl`, `src/vulkan/raytracing/*.jl`,
`src/vulkan/array/ka_backend.jl`, `src/vulkan/access.jl`,
`src/vulkan/lowering.jl` and the core directories `src/graph/`,
`src/graphics/`, `src/raytracing/`. Target: api.md section 4 (device-type
verbs), Appendix A, the plan's "Metal" section (recording-plan.md:1537-1592)
and "9. Metal on the core model" (recording-plan.md:2013-2024).

Line classes (measured by a script over `src/metal`, each file's total
counted once; the per-file figure is the file length):

| class | lines | share |
|---|---|---|
| docstrings (`"""…"""`) | 2,446 | 28% |
| `#` comment lines | 2,091 | 24% |
| blank | 967 | 11% |
| code | ~3,145 | 36% |

So 4,537 of 8,649 lines are prose. The code alone is 3,145 lines, of which
section 5 estimates ~1,800 are the Metal device type's share in the target.

Every number marked "est." is an estimate from reading, not a measurement.

---

## 1. Inventory

Buckets per api.md: (a) belongs in core (backend independent), (b) belongs
in a device-type verb, (c) belongs nowhere (dead, superseded, compat shim,
test-only, debug-only, duplicate). Bucket figures include the prose attached
to the code they classify, so they sum to the file length. All est.

| file | lines | what it does | (a) core | (b) verb | (c) nowhere | api.md verb / Appendix A row |
|---|---|---|---|---|---|---|
| `metal.jl` | 118 | `using`s, include order, `initbackend!` (registers a KA backend factory + an `atexit` retire loop over `METAL.live`), `kernelcompiles`/`resetkernelcompiles!` forwarding to `Metal.compile_stats` | 12 (atexit retire loop → registry; compile counter → core) | 10 (`hardwarekeys(T, registry)`, `T(key)` have no code here yet) | 96 (70 comment lines; `initbackend!` "goes: hooks take the device"; `kernelcompiles` "core: the per-device compile counter") | §4.1 `hardwarekeys`, `T(key)`; App. A `initbackend!`, `kernelcompiles` |
| `caps.jl` | 108 | `simd_width`/`max_threads` by compiling a probe kernel, `matrixshapes`, `caps(d)` lazily cached in `d.caps`, `caps(::MetalBackend)` | 0 | 35 (the `DeviceCaps` record + the two pipeline probes) | 73 (lazy `=== nothing` cache: §4.1 says caps is an immutable record, built at `T(key)`; `caps(::MetalBackend)` "goes"; `bestshape` row → selection table) | §4.1 `caps(dev)` |
| `memory.jl` | 102 | `hostspan`, `deviceview` (borrowed `MtlArray` over a pool buffer), `upload!`/`download`/`devicecopy!` forwarders to core's host helpers, `bufferusage = nothing`, `deviceaddress` | 0 | 8 (`hostspan` 2 lines, `deviceview` → vendor view §4.6, `deviceaddress` → cell write) | 94 (`upload!`/`download`/`devicecopy!`/`bufferusage` all "goes" in App. A; 79 lines prose) | §4.6 vendor view; App. A `hostspan`, `deviceaddress`, `upload!`, `download`, `devicecopy!`, `bufferusage` |
| `debug.jl` | 103 | `debugenv`, `classname` (ObjC `object_getClassName` ccall), `validating`, `checkdebug` (refuses `DebugConfig` fields Metal lacks; installs the report block), `capturereports!` (`MTLSetReportFailureBlock` via `dlsym`), `validationmessages!` draining `METAL.reports` | 0 | 40 (validation is not in §4 at all; see section 4 item 11) | 63 (process-global report list + lock + retained block in `METAL`; §9 allowlist is `REGISTRY` only) | none (gap) |
| `trace.jl` | 170 | A second MSL traversal kernel (`TRACE_MSL`) and the immediate `trace_closest_hits!(hits, rays, tlas, n)` with its own command buffer and wait | 0 | 0 | 170 (App. A: `trace_closest_hits!` "goes: the immediate trace_* verbs"; `TRACE_MSL` duplicates `HWTLAS_MSL`, hwtlas.jl:28-81) | App. A `trace_closest_hits!` |
| `kernelinterface.jl` | 47 | `KI.matrix_shapes`/`supports`/`bestshape` forwarders to `caps`; `KI.launch` for `KI.Kernel{MB,<:Function}` mapping `storage` over args | 0 | 5 (`devicearg` is the one conversion; `KI.launch` is Metal.jl's) | 42 (`bestshape` → selection table; `argmap(storage, args)` is `devicearg`) | App. A `bestshape`, `argtype` |
| `device_intrinsics.jl` | 43 | `const StageOut{T} = Core.LLVMPtr{T,0}` | 0 | 1 | 42 (history of removed overrides) | part of `compiledraw` |
| `device.jl` | 1468 | `MetalDevice{Q}`; two submission generations (`LegacyQueue` over Metal.jl's `BatchedCommandQueue`, `MTL4Queue` with a 3-slot allocator ring, shared event, feedback, bridge command buffers between two queues); `METAL` process global; `Device(::MetalAPI)`, `devices`, `defaultdevice!` (task-local queue adoption); storage-mode constraints; `rawalloc` ×2, `rawfree` ×2; `managedbuffer`/`claimbuffers!` (Metal.jl ownership records); fence/passed/waitfor/waitidle ×2 generations; `opensubmit!`/`closesubmit!`/`suspendsubmit!`/`nextpiece!`/`executesegment!`/`closeframe!`/`ownflush!`/`flushlegacy!`; `encodeplan!`/`argtable` (MTL4 encode route); `canreplay`/`encodes`/`canrun`/`refusereplay`; channels (`allocate_batch_queue!`, `deviceof`, `release_batch_queue!`); `submissions`/`retire!`/`reset_device!`/`tryallocate` | 40 (`compatible`/`mergeconstraints`/`permissiveness` → core kind logic; `deviceof` → registry; `submissions` counter) | 150 (`T(key)` 276-283; `rawalloc` 403-436; `rawfree` 446-454, 487-488; MTL4 `passed`/`waitfor` 1133-1165; `submitnative!` core of `closesubmit!(::MTL4Queue)` 1019-1050; `budget` 1314; `waitidle` 1342) | 1278 (plan §9 deletes `LegacyQueue`, `LegacySubmission`, `MTL4Submission`, `opensubmit!`/`closesubmit!`/`suspendsubmit!`, `replaywithcalls!`, the slot ring, `encodes`/`canreplay`, `ownflush!`; App. A deletes `defaultdevice!`, `backend`, `batchqueue`, `allocate_batch_queue!`, `release_batch_queue!`, `supports_batch_queue`, `tryallocate` is not a verb; 941 lines prose) | §4.1 `T(key)`, `caps`, `budget`; §4.2 `submitnative!`, `passed`, `waitfor`, `recycle!`; §4.3 `rawalloc`, `rawfree` |
| `record.jl` | 1690 | Recording a plan into an `MTLIndirectCommandBuffer`: `argbytes`, `MetalIndirectSlot`/`writegroups!` (a prepare kernel rewrites an ICB command), `RecordedArg`/`recorded_args`/`pack_recorded!` (argument layout), `MetalRecordedDispatch` + call operator (the walked path), `compile_dispatch` (KA + KI), segments/gates (`MetalSegment`, two writer types, three gate kernels), `MetalRecorder`/`MetalRecording` (two parallel resource lists), `planshape`, `openrecording` (115 lines of policy + allocation), the emit verbs, `ensureresident!` (run-time residency rebuild), `submitrun!`/`replay!`/`encodepiece!`/`replaywithcalls!` | 230 (`compile_dispatch` 386-479 = Vulkan's; `recorded_args`/`pack_recorded!` = core `bindings!`; `planshape`; `passbarriers`/`emitbarriers!`/`emitupdate!`/`profiled!`; gate-kernel logic = core loop head/prepare) | 210 (`emitdispatch!` ICB encode 1038-1085; `emitloop!` = range writer 922-942, 1247-1262; `openrecording`'s ICB creation 845-864; `Builder`/`finish`; `submitnative!` = `replay!`'s commit) | 1250 (plan §9 deletes `MetalRecordedDispatch` as a type, `ensureresident!`'s run-time rebuild, `make_persistently_resident!`, `walkedplan`/`planshape`, `replaywithcalls!`, per-run encoding; App. A deletes `argbytes`, `indirectslot`, `passedas`, `argsize`, `openrun`/`closerun!`/`submitrecording!`/`recordsplans`, `holdleaves!` at emit, `emithead!`/`emitkernel!`/`emitpreparebarrier!`/`withpredicate` fold into builder verbs; 866 lines prose) | §4.5 `Builder`, `emitdispatch!`, `emitloop!`, `finish`; §4.2 `submitnative!`; §4.4 `compilekernel` |
| `graphics.jl` | 1623 | State tables (cull/primitive/depth/blend), `MetalTexture2D`/`MetalSampler`/`Texture2D`/`Sampler`/`upload_texture_data!` ×2/`blit_into_texture!`, `MetalFramebuffer`/`Framebuffer`, `mtlformat`/`mtlsampledformat`/`eltypeof`, stage wrappers (`MetalVertexStage`, `MetalFragmentStage`, `PackedArgs`, `to_stage_field`, `stage_fragment_out`), `GFX_CACHE`, `compile_pipeline`, `MetalStages`/`compile_stage_function`, `framebuffer!`/`commit!`/`submitwait!` (one command buffer per call), profiling globals (`COMMITTED`/`COLLECT`), `readback_framebuffer`, `StageArgs`/`rebake!`/`bindall!`/`bind_arg!`, `compile_draw`, `begin_render_pass!`, `record_draw!`, `setviewport!`, `indexbuffer`/`devicearray`/`buffercopy`, `target_*`, `end_render_pass!`, `draw_with_count!`, `metal_buffer`/`metal_texture`, `mtl_loadaction`, KI builtin overlays, `clip_y`/`flip_clip`, `MetalTextureBindings`/`bind_textures`/`use_bindings!` | 110 (`StageArgs` re-bake = core's argument block; `to_stage_field`/`stage_fragment_out`/`unpacked`/`bodytypes` are generic tuple code; the two `upload_texture_data!` are stores/copy nodes) | 520 (state tables 15-74; `mtlformat` 296-331 → `imageformat`; stage wrappers 385-596 and `compile_pipeline`/`compile_stage_function` 665-804 → `compiledraw`; `begin_render_pass!`/`record_draw!`/`end_render_pass!`/`setviewport!`/`draw_with_count!` → `beginrender!`/`recordpiece`/`emitdraw!`/`endrender!`; KI overlays 1506-1573; texture upload blit → `emitcopy!`) | 993 (App. A deletes `Framebuffer`, `Texture2D`, `Sampler` as types, `readback_framebuffer`, `target_view`/`target_image`/`target_extent`/`target_format`, `colorimage`/`depthimage`, `indexbuffer`, `devicearray`, `bind_textures`'s pool, `use_bindings!`, `gpupasstime!`/`makeprofiler` (profiling open), `begin_render_pass!`/`end_render_pass!`; `GFX_CACHE` → the `Kernel` at its `CodeInstance`; `framebuffer!`/`commit!`/`submitwait!` go with "no legacy MTLCommandQueue"; 850 lines prose) | §4.4 `compiledraw`; §4.5 `beginrender!`, `recordpiece`, `emitdraw!`, `endrender!`, `bindindices!`; §4.8 `tableentry!`, `samplerentry!`, `imageformat` |
| `hwtlas.jl` | 760 | `HWTLAS_MSL` (linked `__metal_linked_closest`/`_any`), `MetalHit`, per-device linked-function registration behind `HWTLAS_FUNCS`, `make_resident!` (a per-call residency set commit), device-side `closest_hit`/`any_hit` ×2 accel types with `_unpack`/`_unpack_procedural`, `MetalInstanceBatch`/`MetalHWTLAS` (10 `::Any` fields), `add_geometry!`, `addbatch!`, `addgeometry!`, `push!` ×5, `update_transforms!` ×3, `update_transform!`, `delete!`, `set_visible!`, `instance_buffer`, `refit!`, `mtlinstances_kernel!`, `writeinstances!` (+ `KA.synchronize`), `sync!` (refit vs rebuild, triangle concatenation, argument-buffer assembly, residency), `supports_*` ×5, procedural candidate overlays | 410 (everything from 239 to 711 except the Metal build calls: the same bookkeeping as `VulkanTLAS`, which §2.6 replaces by core `TLAS{P}`/`BLAS`/`InstanceRange`/`RangeEntry` and `BuildAccel` changes) | 190 (MSL + registration 28-111; `closest_hit`/`any_hit` ccall lowering 173-237; `instancerecordtype`/`emitbuild!`/`acceladdress` pieces of 589-598, 667-705; caps flags; three candidate overlays 743-760) | 160 (`HWTLAS_FUNCS` globals; `make_resident!` → the one residency helper; `wait_for_gpu!`/`KA.synchronize` host waits; prose) | §2.6 core TLAS; §4.1 `instancerecordtype`, `countsource`; §4.3 `placeaccel`, `accelsizes`, `acceladdress`; §4.5 `emitbuild!`; App. A `AdaptedAccel`, `build_accel!`, `refit_tlas!` |
| `mesh.jl` | 662 | KI mesh-verb overlays onto `air.set_*_mesh` (`_write_mesh_vertex!` splits a NamedTuple by name into vertex/primitive planes with the provoking-vertex rule, index writers, counts, builtins); `mesh_object_of`/`mesh_topology_symbol`/`mesh_verts_per_primitive`/`mesh_stage_lead`; `MetalCompiledMeshPipeline`, `MESH_CACHE`, `MetalMeshStage`, `compile_pipeline(::MeshPipeline)`; `supports_mesh_pipeline` ×2; geometry-as-mesh: `MetalCompiledGeometryDraw` (caches two lowered pipelines per draw), `MetalCompiledMeshDraw`, `compile_draw` ×2, `record_draw!` ×3 (one compiles the pipeline at record time) | 40 (the lowering cache belongs to compile; `lower_geometry_to_mesh` is already core `graphics/lowering.jl:256`) | 320 (overlays 41-232 are the device compiler's ABI and stay; `compile_pipeline(::MeshPipeline)` 350-425 → `compiledraw`; mesh dispatch in `emitdraw!` 536-548) | 302 (`MESH_CACHE`; `compile_pipeline` duplicates graphics.jl's 40 lines; third `record_draw!` bind copy; compile-at-record 605-613 forbidden by §4.5; `supports_*` double definitions; prose 308 lines) | §4.4 `compiledraw` (`MeshPipeline`); §4.5 `emitdraw!`; §4.1 `caps.supports_mesh_pipeline` |
| `ka.jl` | 520 | `syncbackend`, `intrinsic_usage` (AIR symbol table), `deviceslice` (identical to `deviceview`), `alignment = 16`, `argtype`/`devicebuffertype`/`isdevicearray`/`gemmstrides` ×3/`basealignment`, `native_gemm_available`/`native_gemm_dispatch!`/`native_batched_gemm_dispatch!`, MPSGraph callables (`MPSGraphGemm`/`Attention`/`Conv2D`) with `argument_usage`, `librarygemm`, `denseoperands` (isa chain), `native_conv2d_dispatch!`/`native_attention_dispatch!`, `accesscache`, `metalinterpreter`/`kernelinterpreter`/`kakernelaccesssignature`, `vertextouches` ×2/`fragmenttouches`/`stagetouches` | 10 (`basealignment`, `gemmstrides` → selection key) | 130 (`intrinsic_usage` 56-94 "stays"; `kernelinterpreter`/`kakernelaccesssignature`/`vertextouches`/`fragmenttouches` 444-520 "stay"; `devicearg` from `deviceslice`/`argtype`; `candidates(dev, f)` from the library hooks ~30) | 380 (App. A: `argtype`, `devicebuffertype`, `isdevicearray`, `syncbackend`, `accesscache`, `librarygemm`, `native_*_dispatch!`, `native_gemm_available`, `staged_gemm_tile` all go; plan §9 deletes every `MPSGraph*` callable MLX's kernels match; `runscalls` goes; 296 lines prose) | App. A `intrinsic_usage`, `kernelinterpreter`, `kakernelaccesssignature`, `vertextouches`, `fragmenttouches`; §4.4 `candidates`, §4.6 `devicearg` |
| `raytracing.jl` | 382 | `packedtransform` (unused), `MetalInstanceDescriptor` (UserID layout) + kernel-side constructor, `MetalTLAS`/`MetalBLAS`, `withaccelencoder` (command buffer + wait per build), `build_blas` (device-buffer form), `MetalAccelBuildContext`/`build_accel!(f, bq)`, `build_blas(ctx, verts, indices)`, `Raycore.refit!(blas)`, `build_blas_aabb`, `build_accel!(dev, blases, descriptors, n)`, `refit_tlas!` | 30 (`MetalTLAS`/`MetalBLAS` fields `refittable`/`count`/`vertices` → core `BLAS`/`TLAS` capacity + `builtcount`; refit = an Update change) | 130 (`MetalInstanceDescriptor` 50-71 → `instancerecordtype`; sizes/alloc/build/refit encodes 158-180, 293-316, 337-360, 378-380 → `accelsizes`/`placeaccel`/`emitbuild!`) | 222 (`packedtransform` dead; `withaccelencoder` synchronous; `MetalAccelBuildContext`/`build_accel!` "goes"; `MTLBuffer`s outside the pool 173-179, 234-236, 289-290 → `rawalloc`; 181 lines prose) | §4.1 `instancerecordtype`; §4.3 `placeaccel`, `accelsizes`, `acceladdress`; §4.5 `emitbuild!`; App. A `build_accel!`, `refit_tlas!` |
| `procedural.jl` | 346 | `_as_f32`/`_from_f32`/`packbest`/`unpackbest` (a 4-float hit record across a visible-function return), `ProceduralOut`/`ProceduralVisible{P,B}` callable, `PROCEDURAL_MSL` (intersection_query loop + a host kernel to own the table pipeline), `PROC_FUNCS` globals/`ensure_procedural_linked!`, `procedural_payload_bytes` (struct → bytes via `Ref` copy), `procedural_resident!`, `procedural_table` (compiles the MSL library a second time), `compile_procedural_candidate` (`register_table_function!`) | 0 | 210 (the visible-function ABI 33-97, the MSL loop 111-205, the candidate compile 317-344 are the device compiler's; no §4 verb names them — section 4 item 9) | 136 (`PROC_FUNCS`; second `MTLLibrary(d.dev, PROCEDURAL_MSL)` at 295; per-sync residency; `procedural_payload_bytes` is a `devicearg`; prose) | none (gap) |
| `window.jl` | 319 | `MetalWindow{T}` (CAMetalLayer + one drawable + acquirer task), `attach!` (ObjC: `contentView`, `setWantsLayer:`, `setLayer:`, `backingScaleFactor`), `Mantle.Window(::MetalBackend)` (GLFW `NO_API`, points vs pixels), `target_*`, `eltype`/`size`/`isopen`/`close`, `resize!`, `beginframe!` (polls GLFW, resizes), `acquire_next_image!`/`nextdrawable`, `present_frame!` (own command buffer), `readback_window` (blit + present), `screenshot`, `abandonframe!` | 10 (`checkacquirable`/acquirer task → core window lock) | 90 (window open verb 42-47, 71-88, 116-133; `createswapchain` = `drawableSize` 197-202; `acquirenext` = `next_drawable` 240-246; present inside `submitnative!` 261-263) | 219 (App. A: `target_*`, `beginframe!`, `present_frame!`, `readback_window`/`screenshot` → `Array(win)`, `abandonframe!` go; prose 161 lines) | §4.7 window open verb, `createswapchain`, `acquirenext`, `surfacesize` |
| `images.jl` | 188 | `imageusage`, `MetalImageReq`/`MetalTransientImage`, `image_descriptor`, `makeimage`/`remakeimage!` (heap size/align query), `materialize!` (texture on a placement heap at an offset), `readback_target` (blit + wait), `copy_target!` (blit, own command buffer), `byteoffset` ×3 | 5 (`TransientImage` is core's type) | 50 (`placeimage` from `image_descriptor`+`makeimage`+`materialize!` 68-113; image→buffer blit 168-171 → `emitcopy!`; `imageusage` → Images constraint) | 133 (App. A: `remakeimage!`, `readback_target`, `copy_target!` go; `byteoffset` is `devicearg`; 103 lines prose) | §4.3 `placeimage`; §4.5 `emitcopy!` |
| **total** | **8,649** | | **897** | **2,099** | **5,653** | |

Reading of the totals: 2,099 lines (24%) are in the vicinity of a verb and
still carry their current prose and current shape; 897 lines re-implement
core algorithms; 5,653 lines (65%) have no place in the target. The
verb-shaped code shrinks further in section 5 (~1,800 code lines) because
today's forms carry the two-generation branches and per-frame rebuilds.

---

## 2. Duplication: Metal against Vulkan and core

The target design (api.md §1 rule 1, §4) has a device type implementing only
verbs. Everything below exists in `src/metal` AND in `src/vulkan` in
different words; core already owns or will own the shared part. "Recoverable"
is est. lines that leave each side when the shared part is written once in
core.

### 2.1 Cross-backend table

| # | concern | Metal (file:lines) | Vulkan (file:lines) | core today / target | recoverable Metal | recoverable Vulkan |
|---|---|---|---|---|---|---|
| 1 | Device object, default device, device list, "device for a KA backend" | device.jl:30-61 (`MetalDevice`), 276-344 (`MetalDevice` ctor, `METAL`, `Device(::MetalAPI)`, `devices`, `defaultdevice!`, `Device(::MetalBackend)`, `pool`, `backend`) | graph.jl:32-90 (`LavaDevice`, `DEVICES` IdDict, `Device(::VulkanAPI)`, `defaultdevice!`, `Device(::LavaBackend)`, `backend`), runtime/device.jl:819-958 (`VK_CONTEXT_REF`, `LIVE_CONTEXTS`, `bind_context!`, `vk_context()`), 2012-2021 (`devices`) | `REGISTRY` (§3), `hardwarekeys`/`T(key)` (§4.1); plan §8 deletes `METAL_DEVICE`, `DEVICES`, `VK_CONTEXT_REF`, `LIVE_CONTEXTS`, `adopt!` | 60 | 100 |
| 2 | Capabilities | caps.jl:22-108; `supports_graphics` ×2 graphics.jl:1019-1020; `supports_mesh_pipeline` ×2 mesh.jl:439-440; `supports_rt_pipeline`/`supports_procedural_traversal`/`supports_hwtlas` ×5 hwtlas.jl:716-729; `supports_batch_queue` device.jl:1358 | runtime/device.jl:695-800 (`caps(ctx)`, `workgroup_matrix_granularity`, `shader_core_count`, `shader_warps_per_sm`), runtime/matrixshapes.jl:40-84; ka_backend.jl:81-83, 111-112; graphics/api.jl:396-403; hwtlas.jl:1076-1084; graph.jl:136-142 | one immutable `caps(dev)` (§4.1); App. A `supports_*` → fields | 25 (the ::Backend/::Device doubles and the lazy cache) | 30 |
| 3 | Memory constraints and raw allocation | device.jl:346-389 (`constraintof` ×3, `permissiveness` ×4, `compatible`, `mergeconstraints`), 391-454 (`rawalloc` ×2, `rawfree`), 485-488 (`rawfree` heap) | graph.jl:254-256, 732-939 (`rawalloc` ×4, `rawfree` ×4, `constraintof` ×5, `compatible` ×2, `mergeconstraints` ×5, `materialize!`, `maxalloc`) | memory/pool.jl:20-96 declares; App. A: `compatible`, `mergeconstraints`, `constraintof` become core; `alignment`/`bufferusage`/`extrausage`/`imageusage` become the kind's constraint | 40 | 40 |
| 4 | Host view / upload / download / device copy | memory.jl:13-102 (`hostspan`, `deviceview`, `upload!`, `download`, `devicecopy!`, `bufferusage`, `deviceaddress`); ka.jl:111-115 (`deviceslice`, same body as `deviceview`) | graph.jl:315-336 (`deviceview`, `deviceaddress`, `upload!` ×2, `download`, `devicecopy!`), 239 (`bufferusage`), 499-535 (`storebytes!` ×2) | memory/resources.jl:93-124, 642 (`hostupload!`, `hostdownload`, `hostdevicecopy!`); App. A: all go (stores, copy nodes, `Array(a)`) | 95 | 30 |
| 5 | Timeline (`fence`/`passed`/`waitfor`/`waitidle`) | device.jl:490-570 (Legacy: retirement counter, `quiet`, `waitidle`), 1096-1165 (MTL4: `next + 1` fence, `min(f, next)` passed, bounded wait with feedback), 1325-1343 | graph.jl:271-313 (`fence = next_timeline`, `passed`, `waitfor` with the same "`f > next` → decline" rule at 300-313), runtime/command.jl (channel timeline) | `Token = (channel, value)` (§3); verbs `passed(ch, v)`, `waitfor(ch, v)` (§4.2); `fence` is a field read | 140 | 20 |
| 6 | Channels (batch queues) | device.jl:1347-1414 (`allocate_batch_queue!` ×2, `deviceof`, `todevice`, `backend(bq)`, `flush!(bq)`, `reclaim!`, `release_batch_queue!`) | runtime/device.jl:1886-1980 (`allocate_batch_queue!`, `release_batch_queue!`, `queue_released`), graph.jl:87-88, 120-126, 274 | graph/queue.jl:12-120 declares; `SubmitChannel` is core (§3); App. A: `allocate_batch_queue!`, `release_batch_queue!`, `supports_batch_queue`, `batchqueue`, `backend`, `syncbackend` go | 68 | 95 |
| 7 | Device lifecycle (`submissions`, `retire!`, `reset_device!`, `tryallocate`) | device.jl:1416-1468 | graph.jl:150-215, runtime/device.jl:993-1084 (`reset_device!`), runtime/lifecycle.jl:168-290 (core declarations) | registry; `reset_device!` open (§10); `tryallocate` is `acquire!` pressure (§7) | 50 | 90 |
| 8 | `compile_dispatch` (KA kernel object, closure check, resolve, dispatch range, `callgroup`, `launch_config`/`mkcontext`, KI path) | record.jl:377-479; identical error text 432-436 | graph.jl:1032-1115 (`compile_dispatch`, `kicompile_dispatch`); identical error text 1038-1042 | graph/kalaunch.jl:657 (`compile_dispatch` generic), 536-586; target `compilekernel(dev, kernel, config)` + core `compile!` (§4.4) | 85 | 80 |
| 9 | Argument layout and packing | record.jl:151-342 (`RecordedArg`, `recorded_args`, `pack_recorded!`, `passedas` ×2, `boundbuffer`, `checkresident`), 56-67 (`argbytes`) | graph.jl:538-558 (`argbytes`, `indirectslot`), 2016-2060 (`packdispatch!`, `emitlaunch!`), runtime/launch.jl:286 (`passedas`) | graph/packing.jl:82-101 (`eachpassedarg`, `passedas`); target: `bindings!` lays out the block, `devicearg` converts (App. A `argbytes`, `argidentity`, `argsize`, `passedas`, `indirectslot`) | 100 | 60 |
| 10 | Walk primitives (barriers, predicates, head, loop head, updates, profiling hooks) | record.jl:962-1397 (`emitdispatch!` ×2, `encode!`, `emitkernel!`, `emitpreparebarrier!`, `passbarriers`, `emitbarriers!`, `emitupdate!`, `profiled!`, `withpredicate` ×2, `closesegment!`, `emitwriter!`/`emitrangewriter!`/`emitgridwriter!`, `patchwriter!` ×2, `emithead!`, `emitloophead!`, `closerecording!`, `markgates!`, error stubs) | graph.jl:1015-1030 (`passbarriers`), 1249-1262, 1324-1345 (`profiled!`), 1347-1430 (`emithead!`, `emitloophead!`, `emitbarriers!`, `withpredicate`, `emitupdate!`), 1486-1515, 1560-1585 (`emitdispatch!`), 1671-1760 (`openrecording`, `closerecording!`, `patchtarget`) | graph/backend.jl:96-247 declares 14 hooks; kalaunch.jl:1441-1527 (`Immediate` versions); target §4.5: `emitdispatch!`/`emitcopy!`/`emitbuild!`/`emitloop!`/`finish` with `deps` complete and core's, `when!` indirect (no predicate scope) | 250 | 60 |
| 11 | Run and submit | record.jl:1399-1690 (`recordsplans`, `recordable`, `openrun`, `enterrun!`, `closerun!` ×2, `submitrun!`, `ensureresident!`, `submitrecording!` ×3, `replay!`, `encodepiece!`, `replaywithcalls!`); device.jl:572-1094 (`opensubmit!`/`closesubmit!`/`suspendsubmit!`/`nextpiece!`/`executesegment!`/`closeframe!`/`flushlegacy!`/`encodeplan!` on two queue types) | graph.jl:1773-1935 (`recordsplans`, `openrun`, `closerun!`, `submitrecording!` ×3, `abandonrecording!`, `abandonrun!`, `present_frame!`, `abandonframe!`), runtime/command.jl (`submit!`, `submitlist!`, `handover!`) | kalaunch.jl:1336-1391 (`submitplan!`, `RecordingParts`); target: `submitnative!(ch, token, parts, waits, signals, present)` only (§4.2); plan §9 deletes `opensubmit!`/`closesubmit!`/`suspendsubmit!`, `replaywithcalls!`, the slot ring | 700 | 120 |
| 12 | Fixed-function state translation | graphics.jl:15-74 (`mtl_cull` ×3, `mtl_primitive` ×4, `mtl_depth_compare` ×6, `depth_writes` ×3, `apply_blend!` ×5) | graphics/pipeline.jl:260-345 (`vk_topology` ×8, `vk_cull` ×3, `vk_depth` ×6, `vk_depth_write` ×2, `vk_blend` ×5) | graphics/state.jl:25-133 (the portable values). Inherent per API; the five blend modes are four factor pairs + a write mask that one core table could hold with a 4-line device method | 30 | 30 |
| 13 | Format tables | graphics.jl:103-105 (`mtlsampledformat`), 296-331 (`mtlformat`), 1385-1392 (`eltypeof` reverse) | lowering.jl:219-250 (`vkformat`), graphics/record.jl:57-70 (`eltypeof` reverse), graphics/textures.jl:218-226 (`julia_to_vk_format`, a third table) | §4.8 `imageformat(dev, T)`: "the one format table … for targets and textures alike" | 25 | 25 |
| 14 | Textures and samplers | graphics.jl:76-235 (`MetalTexture2D`, `MetalSampler`, `Texture2D`, `upload_texture_data!` ×2, `blit_into_texture!`, `Sampler`), 1575-1623 (`MetalTextureBindings`, `bind_textures`, `use_bindings!`) | graphics/textures.jl:6-286 (`VulkanTexture2D`, `VulkanTexture1D`, `VulkanSampler`, `copy_into_texture!`, `upload_texture_data!` ×2, `VulkanTextureBindings`, `bind_textures`), graph.jl:1606-1607, graphics/pipeline.jl:776-790 (`use_bindings!`) | graphics/resources.jl:28-123 (abstract types); target §2.6 `Image` + §4.8 `tableentry!`/`samplerentry!`; App. A: `Texture2D`, `bind_textures`'s pool, `use_bindings!` go | 150 | 200 |
| 15 | Offscreen framebuffer and readback | graphics.jl:237-283 (`MetalFramebuffer`, `Framebuffer`), 969-1014 (`readback_framebuffer`), 1370-1392 (`target_*`, `colorimage`, `depthimage`, `eltypeof`) | graphics/framebuffer.jl:12-306 (`VulkanFramebuffer`, `_vulkan_framebuffer`, memory helpers, `readback_framebuffer`, `copy_framebuffer!`), graph.jl:347-362, 1598, graphics/record.jl:25-40 (`VulkanAttachment`, `colorimage`, `depthimage`) | App. A: `Framebuffer`, `readback_framebuffer`, `colorimage`, `depthimage`, `target_*` go; render targets are `Image`s, `Array(img)` | 90 | 250 |
| 16 | Shader stage wrappers (return-value stages → pointer/varying ABI) | graphics.jl:385-596 (`to_stage_field`, `stage_fragment_out`, `PackedArgs`/`packargs`/`unpacked`/`bodytypes`, `MetalVertexStage`, `MetalFragmentStage`, `varying_markers`, `texture_markers`), mesh.jl:326-336 (`MetalMeshStage`) | Lava/src/device/gfx_intrinsics.jl:635 (`emit_vertex_outputs`), 844-1017 (`VertexWrapper`, `GeometryWrapper`, `FragmentWrapper`, `MeshWrapper`), used from graphics/api.jl:179-228 (`resolve_shader_pair`) | nothing in core. Both are `@generated` wrappers that call `F.instance` with unpacked args and assemble outputs by name; inherent per compiler, but the by-name assembly and tuple↔`Vec` coercion are one function (guess) | 40 | 30 |
| 17 | Pipeline compile and cache | graphics.jl:598-804 (`GFX_CACHE`, `varying_type`, `stage_signatures`, `compile_pipeline`, `MetalStages`, `identifier`, `stagename`, `compile_stage_function`), mesh.jl:323-425 (`MESH_CACHE`, `compile_pipeline(::MeshPipeline)`: attachments loop 398-403 = graphics.jl:720-725, depth state 411-417 = 730-736, cache 388-391/421-423 = 708-711/741-743) | graphics/api.jl:31-160 (`pipeline_state_key`, `pipeline_cache_key`, `ensure_compiled!` ×2, `ensure_compiled_with_shader!` ×2), runtime/pipeline_cache.jl:1-185, graphics/pipeline.jl:21-250 (`create_graphics_pipeline`) | §4.4 `compiledraw(dev, pipeline, argtypes, formats)` keyed by core; "the `Kernel` at its `CodeInstance` … no other kernel table, nothing global" (plan rule 18) | 70 | 120 |
| 18 | Draw recording (compile draw, begin/end pass, bind, draw) | graphics.jl:1022-1488 (`MetalPassHandle`, `StageArgs`, `rebake!`, `collectbuffers!`, `MetalCompiledDraw`, `compile_draw`, `begin_render_pass!`, `bind_stage!`/`bindall!`/`bind_arg!`, `record_draw!`, `setviewport!`, `end_render_pass!`, `draw_with_count!` ×2, `draw_indirect_with!`, `mtl_loadaction`), mesh.jl:442-662 (`MetalCompiledGeometryDraw`, `loweredmesh`, `MetalCompiledMeshDraw`, `compile_draw` ×2, `record_draw!` ×3). The bind sequence (`set_pipeline!`, depth state, cull, winding, `bind_stage!` ×2, draw) is written three times: graphics.jl:1259-1275, mesh.jl:536-547, 616-648 | graphics/record.jl:81-191 (`VulkanDraw`, `compile_draw`, `VulkanPassHandle`, `begin_render_pass!`, `end_render_pass!`, `setviewport!`, `use_bindings!`, `record_draw!`), graphics/pipeline.jl:354-790 (`transition_image!`, `begin_pass!`, `draw_in_pass!`, `draw_mesh_in_pass!`, `bind_for_draw!`, `draw_indirect_in_pass!`, `draw_indexed_in_pass!`, `set_viewport!` ×2, `end_pass!`), graph.jl:1956-2014 (`emitdraw!`, `emit_draw!` ×4) | graphics/commands.jl:21-492 (declarations), graphics/record.jl:48-346, kalaunch.jl:1469-1506 (`Immediate` `beginrender!`/`emitdraw!`/`endrender!`); target §4.5 `beginrender!`/`emitpieces!`/`endrender!`/`recordpiece`/`emitdraw!`/`bindindices!`; App. A deletes the `begin_pass!`/`draw_*_in_pass!` family, `setviewport!`, `record_draw!` as public | 250 | 200 |
| 19 | Mesh-shader vocabulary (NamedTuple → per-vertex/per-primitive planes, 1-based → 0-based, provoking vertex) | mesh.jl:41-232 | Lava/src/device/gfx_intrinsics.jl:999-1160 (`MeshWrapper`, `mesh_location`, `set_mesh_outputs!`, `set_mesh_vertex!`, `set_mesh_primitive_data!`, `set_mesh_triangle!`/`line!`/`point!`, `mesh_thread_index`, `mesh_group_index`) | nothing in core. The name→(plane, index) split and the provoking-vertex rule are the same algorithm on both sides (guess ~60 lines each); the intrinsic calls are inherent | 40 | 40 |
| 20 | Geometry→mesh lowering use | mesh.jl:442-662 (per-draw lowered-pipeline cache, compile at record) | Vulkan has a native geometry stage (graphics/api.jl:230-254 `geometrystage`, 396) | graphics/lowering.jl:256-339 (`lower_geometry_to_mesh`, core). Metal's per-draw cache + compile-at-record violate §4.5 ("Record compiles nothing"); lowering is a compile-time key of `compiledraw` | 120 | 0 |
| 21 | HWTLAS bookkeeping (incremental scene) | hwtlas.jl:239-711. Pairs: `MetalInstanceBatch` 242-258 ≡ `InstanceBatch` 42-50; `MetalHWTLAS` 274-311 ≡ `VulkanTLAS` 110-140; ctor 315-325 ≡ 183-211; `world_bound`/`n_geometries`/`n_instances`/`wait_for_gpu!` 327-330 ≡ 278-280, 303-310; `AdaptedAccel`/`adapt_structure` ×2/`holdleaves!` 332-365 ≡ 213-275; `add_geometry!` 371-412 ≡ `hwtlas_add_geometry!` 312-370 (identical up to `isempty(cpu_triangles) && error`, then flat vs indexed vertices); `addbatch!`/`addgeometry!` 416-437 ≡ 375-403; `push!` ×5 460-522 ≡ 404-520 (same five signatures incl. the ignored `sbt_offset`); `update_transforms!` ×3/`update_transform!` 526-551 ≡ 521-625; `delete!`/`set_visible!` 553-570 ≡ 627-660; `instance_buffer`/`refit!` 574-577 ≡ 284-287; records kernel 589-598 ≡ `vkinstances_kernel!` 685-697; `writeinstances!` 608-620 ≡ 706-733; `sync!` 622-711 ≡ 847-940 + `rebuild_hw_tlas_from_batch!` 769-846 (triangle concatenation 655-663 ≡ 783-795); `_unpack` 137-155 ≡ `rq_collect` 961-1003 | raytracing/hwtlas.jl:42-181, 213-310, 312-660, 664-940, 961-1003 | raytracing/batches.jl (`InstanceBatches`, `register!`, `batchof`, `delete!`, `sethidden!`, `recordsof`), instances.jl:19-64 (`instancerecords`, `update_instance_records_kernel!`), accel.jl:29-123 (`AdaptedAccel`, `hittriangle`) already core; target §2.6 `TLAS{P}`, `BLAS`, `InstanceRange`, `RangeEntry`, `push!`/`delete!`/`build!`/`refit!` as pending changes; device verbs `accelsizes`, `placeaccel`, `acceladdress`, `instancerecordtype`, `emitbuild!` | 400 | 600 |
| 22 | BLAS/TLAS build and refit | raytracing.jl:118-382 (`withaccelencoder`, `build_blas` ×2, `MetalAccelBuildContext`, `build_accel!(f, bq)`, `refit!(blas)`, `build_blas_aabb`, `build_accel!(dev, …)`, `refit_tlas!`); AABB flattening 283-288 | raytracing/acceleration.jl:178-700 (`VulkanAccelBuildContext`, `build_blas` ×3, `refit_blas!`, `refit!`, `build_blas_aabb` with the same `(min, max)` 24-byte packing at 391-460, `build_tlas` ×2, `refit_tlas!`, `pack_as_instance!`), 997-1145 (`build_accel!`, `build_as_on_gpu` ×3) | raytracing/api.jl:19-83 (declarations), accel.jl:126-133 (`AccelBuildContext`); App. A: `build_accel!`, `refit_tlas!` go; `BuildAccel` changes + `BuildRecording` (§3) | 120 | 300 |
| 23 | Procedural (AABB) traversal | procedural.jl:1-346, hwtlas.jl:157-171, 185-237, 743-760 | raytracing/hwtlas.jl:941-1003 (`CANDIDATE_*`, `candidate_primitive_index`, `candidate_object_ray`, `commit_intersection!` overrides, `rq_collect` loop) | accel.jl:202-227 declares the three candidate verbs. Vulkan loops in Julia over an inline ray query; Metal must loop in MSL and call a visible function; the hit unpack/`procedural_commit`/`procedural_bary` sequence (hwtlas.jl:205-213) is the same as `rq_collect`'s generated branch | 30 | 20 |
| 24 | Immediate trace verbs | trace.jl:1-170 (`TRACE_MSL` duplicates `HWTLAS_MSL` hwtlas.jl:28-81: same ray setup, same `intersector<instancing, triangle_data>`, same miss/hit packing) | raytracing/raycore_compat.jl:26-247 (`HardwareAccel`, `set_anyhit_pipeline!`, `trace_closest_hits!` ×4, `hw_raygen`, `hw_closesthit`, `hw_miss`) | raytracing/api.jl:159-195 declares; App. A: all `trace_closest_hits*` and `set_anyhit_pipeline!` go (`trace!`) | 170 | 240 |
| 25 | Window and present | window.jl:1-319. Pairs: GLFW `NO_API` + `DefaultWindowHints` 119-123 ≡ window.jl:80-85; `target_*` 138-155 ≡ 286-299; `isopen`/`close` 159-177 ≡ 433-470; acquire 233-246 ≡ 336-360; present 255-267 ≡ 386-421 + graph.jl:1905-1915; `readback_window` 284-305 ≡ framebuffer.jl:387-470; `abandonframe!` 313-319 ≡ graph.jl:1917-1930; `beginframe!` 210-224 ≡ graph.jl:1767-1771 | graphics/window.jl:14-475, graphics/framebuffer.jl:378-514 (`readback_window`, `acquireframe!`), graph.jl:218, 1767-1771, 1890-1935, graphics/api.jl:313-395 (`presentready!`, `presentuntouched!`, `submit_and_present!`) | graphics/commands.jl:63-154 (declarations, `checkacquirable`, `checkreadable`), runtime/api.jl:69-128; target §2.7 + §4.7 (window open verb, `createswapchain`, `acquirenext`, `surfacesize`; present inside `submitnative!`; `Array(win)`) | 150 | 300 |
| 26 | Transient images | images.jl:1-188 | graph.jl:398-476 (`VulkanTransientImage`, `alignment`, `imageusage` ×2, `aspect`, `checkusage`, `makeimage`, `remakeimage!`), 919-937 (`materialize!`), 1451-1484 (`readback_target`), 1432-1449 (`emitcopy!`) | graph/backend.jl:35-66 (declarations), kalaunch.jl:1459-1467 (`Immediate` copy → `copy_target!`); target `placeimage` (§4.3), `copyto!` copy nodes, `Array(img)`; App. A: `makeimage`, `remakeimage!`, `materialize!`, `readback_target`, `copy_target!`, `imageusage` | 80 | 80 |
| 27 | Access walk glue | ka.jl:127-157, 427-520. `stagetouches` 512-520 ≡ Vulkan `vertextouches`/`fragmenttouches` 94-106 (same `accessof` call, same slice); `kakernelaccesssignature` 460-467 ≡ access.jl:36-42, and both rebuild the KA context exactly as `compile_dispatch` does (third copy record.jl:447-449) | vulkan/access.jl:12-140 (`argtype`, `accesscache`, `devicebuffertype`, `kernelinterpreter`, `kakernelaccesssignature`, `isdevicearray`, `kerneltouches`, `shadertouches`, `vertextouches`, `fragmenttouches`, `intrinsic_usage`) | graph/access.jl:74-1339; App. A: `argtype`, `devicebuffertype`, `isdevicearray`, `accesscache` go; `kernelinterpreter`, `kakernelaccesssignature`, `vertextouches`, `fragmenttouches`, `intrinsic_usage` stay | 30 | 20 |
| 28 | Library dispatch (GEMM, batched GEMM, attention, conv) | ka.jl:159-425 | not in the compared set (Vulkan's are `src/vulkan/array/*.jl`); graph/backend.jl:293-430 declares the hooks with `nothing`/`false` defaults | target: selection table + `candidates(dev, f)` (§2.4, §4.4); MPSGraph as `ForeignNode`; plan §9 deletes "every `MPSGraph*` callable that MLX's kernels match" | 200 | — |
| 29 | Validation and debug | debug.jl:1-103 | runtime/debug.jl:1-126, runtime/device.jl:2110-2353 (`debug_callback`, `handle_validation_message`, `drain_validation_messages!`, `setup_debug_messenger`, `check_validation_errors!`, `LAYER_SETTING_STORAGE`), graph.jl:150-160 | runtime/lifecycle.jl:66-165 (`DebugConfig`, `debugenv`, `validating`, `validationmessages!`); not in api.md (section 4 item 11) | 20 | 60 |
| 30 | Per-pass GPU profiling | graphics.jl:879-967 (`COMMITTED`, `COLLECT`, `COMMITTED_MAX`, `committed!`, `makeprofiler`, `gpupasstime!`), record.jl:1176-1183 (`profiled!`) | graph.jl:626-650 (`Profiler`), 1264-1345 (`collect!`, `profiled!`), 1619-1625 (`makeprofiler`, `timestamps`), runtime/profiling.jl:155-440 (dispatch timing, timestamp pool) | graph/backend.jl:507-688; open (§10 Profiling); globals forbidden (§9) | 95 | 150 |

Sum of "recoverable" (est.): Metal ≈ 3,460 lines of the 8,649 by deletion
or motion into core, before the prose attached to them (section 1 puts the
total that leaves at 5,653); Vulkan ≈ 3,400 in the compared files.

### 2.2 Duplication inside `src/metal`

| what | copies | lines |
|---|---|---|
| "the `MTLBuffer` and byte offset behind an `MtlArray`" | `_mtlbuffer` trace.jl:124; `metal_buffer` + `byteoffset` graphics.jl:1460-1467, images.jl:185-188; `pointer(data).buffer/.offset` graphics.jl:188-189; `x.data[].buffer` / `x.offset` inline record.jl:1362-1363, 1543, hwtlas.jl:692-693, procedural.jl:271; `MtlArray{T,N}(ref, dims; maxsize, offset)` memory.jl:38 and ka.jl:114 | 25 |
| `deviceview` (memory.jl:33-39) and `deviceslice` (ka.jl:111-115) | identical bodies, two names | 7 |
| "open a command buffer, encode, commit, wait" | `blit_into_texture!` graphics.jl:194-211; `withaccelencoder` raytracing.jl:125-138; `trace_closest_hits!` trace.jl:153-168; `readback_window` window.jl:294-301; `readback_target` images.jl:132-140; `copy_target!` images.jl:165-175; `present_frame!` window.jl:261-263 | 60 |
| residency, five mechanisms | `make_resident!` (a residency set installed and committed per call) hwtlas.jl:117-125, 697, 705, procedural.jl:271; `Metal.make_persistently_resident!` record.jl:64, 284, 357, 864, 872, 1554; `ensureresident!` run-time list rebuild record.jl:1509-1562; `MTL.use!` per encoder device.jl:631, 660, graphics.jl:1212, 1421, 1446, mesh.jl:630; MTL4 `resset` device.jl:779-783, 970 | 110 |
| traversal MSL | `TRACE_MSL` trace.jl:30-97 and `HWTLAS_MSL` hwtlas.jl:28-81 (same ray setup, intersector, hit packing); `HitOut` declared a third time in `PROCEDURAL_MSL` procedural.jl:131 | 60 |
| `compile_pipeline` ×2 | graphics.jl:665-745 and mesh.jl:350-425 share the attachment loop, depth-state block, cache lookup/store, library retention | 40 |
| `record_draw!` bind sequence ×3 | graphics.jl:1259-1275; mesh.jl:536-547; mesh.jl:616-648 | 40 |
| `supports_*` defined for both `::MetalBackend` and `::MetalDevice` | graphics.jl:1019-1020; mesh.jl:439-440; hwtlas.jl:727-728; device.jl:1358 | 8 |
| per-queue-generation doubles | `executesegment!`, `opensubmit!`, `closesubmit!`, `nextpiece!`, `suspendsubmit!`, `closeframe!`, `ownflush!`, `batchqueue`, `cmdqueue`, `fence`, `passed`, `waitfor`, `waitidle`, `submissions` each ×2 (device.jl:508-570, 598-695, 793-1094, 1113-1165, 1192-1211, 1256, 1289, 1342, 1429-1430) + the 12 `f(d) = f(d.queue)` forwarders | 400 |
| gate machinery ×2 (replay vs encode) | `MetalRangeWriter`/`MetalGridWriter` record.jl:506-529; `metal_write_range!`/`metal_write_grids!` 937-960; `emitrangewriter!`/`emitgridwriter!` 1247-1283; `patchwriter!` ×2 1286-1300; `ranges`/`grids`/`templ`/`templhost` fields 592-601; `markgates!` 1381-1389; `encodes(...)` branches 878, 991, 1056-1072, 1245, 1314-1315, 1355-1360 | 150 |
| `ensure_traversal_linked!` / `ensure_procedural_linked!` | hwtlas.jl:100-111 and procedural.jl:213-224: same body over two global `Dict`s; `procedural_table` compiles the same MSL library again at procedural.jl:295 | 25 |
| `closest_hit`/`any_hit` × triangle/procedural | hwtlas.jl:173-237: four 12-line `ccall` bodies differing in the symbol name and the unpack function | 36 |
| `GFX_CACHE`/`MESH_CACHE` + locks; `HWTLAS_FUNCS`/`PROC_FUNCS` + locks | graphics.jl:599-600, mesh.jl:323-324, hwtlas.jl:97-98, procedural.jl:210-211 | 8 |
| readback bodies ×3 | `readback_framebuffer` graphics.jl:992-1014 (`getBytes!`), `readback_target` images.jl:127-144 (blit), `readback_window` window.jl:284-305 (blit + present) | 50 |

---

## 3. Patterns that inflate the code

Each with 2-3 `file:line` examples and an est. total cost in `src/metal`.

### 3.1 Two submission generations behind a type parameter (≈ 900 lines)

`MetalDevice{Q}` (device.jl:30) carries `LegacyQueue` (85-145) or `MTL4Queue`
(731-791); every submission verb exists twice (`opensubmit!` 609/944,
`closesubmit!` 683/1019, `executesegment!` 635/930, `nextpiece!` 659/974,
`closeframe!` 1208/1089, `fence` 508/1113, `passed` 531/1133, `waitfor`
541/1147, `waitidle` 565/1342, `submissions` 1429/1430); the recorder captures
a second, encode-shaped form of every command (`MetalEncodedCommand`
record.jl:542-560, `argtable` device.jl:842-851, `encodeplan!` 864-886) and
doubles its gate machinery (section 2.2). `canreplay`/`encodes`/`canrun`/
`refusereplay` (device.jl:821-919) are hand-written dispatch on the parameter.
Neither generation is the target: the plan is MTL4-only with ICB replay
(recording-plan.md:1539-1542, 1559-1563). `executesegment!(::MTL4Queue)` is
documented as "NOT reached, and deliberately" (device.jl:921-929). No file
under `test/` constructs an `MTL4Queue` (grep over `test/`: zero constructor
calls; two comment mentions, `test_recorded_plan_runs.jl:162,218`), so the
entire MTL4 branch is untested on this suite.

### 3.2 History and measurement narrative in comments and docstrings (≈ 3,000 lines over a contract budget)

4,537 of 8,649 lines are prose. A 128-line comment (device.jl:147-274)
precedes the one-line `metalqueue(dev) = LegacyQueue(dev)`; it holds six
measurement tables. Other examples: device.jl:113-145 (frame-rate tables
above `pinpipeline!`), record.jl:1-55, metal.jl:1-25 and 102-116, window.jl:96-114
("said to be impossible for three separate reasons … the third keyword this
session found"), graphics.jl:984-990, caps.jl:102-106 (a comment that begins
mid-sentence: `device per process" beside …` — a fragment left by an edit),
device_intrinsics.jl:1-24 (what is no longer in the file). The plan's rule
(memory: "docs state the target only") puts provenance in git. A contract
docstring per verb would be ~1,200 lines (guess).

### 3.3 Globals and process state (≈ 120 lines, plus the bugs they caused)

Mutable globals: `METAL` (device.jl:297-306: default device, live devices,
validation reports, their lock, the retained ObjC block), `GFX_CACHE`/
`GFX_CACHE_LOCK` (graphics.jl:599-600), `MESH_CACHE`/`MESH_CACHE_LOCK`
(mesh.jl:323-324), `COMMITTED`/`COLLECT`/`COMMITTED_MAX` (graphics.jl:891-911),
`HWTLAS_FUNCS`/`HWTLAS_FUNCS_LOCK` (hwtlas.jl:97-98), `PROC_FUNCS`/
`PROC_FUNCS_LOCK` (procedural.jl:210-211). Implicit globals: `Metal.device()`
(graphics.jl:131, 164, 234, 692, 782; mesh.jl:359; window.jl:44; ka.jl:438),
`Device(MetalAPI())` inside functions that have or could take a device
(graphics.jl:200, 851; hwtlas.jl:316; raytracing.jl:217, 259; trace.jl:146;
window.jl:292), `Metal.synchronize()` on the current task's queue
(graphics.jl:187, 1004; hwtlas.jl:330; record.jl:1420), and Metal.jl's
task-local queue made "ours" by `adoptqueue!` at every run and record
(device.jl:337, 1298; record.jl:789, 1446). What should own each: the device
object (caches → the `Kernel` at its `CodeInstance`, plan rule 18; linked
functions, report list and reporter → device fields; the default device →
`REGISTRY`). api.md §9 allows one mutable global. Three separate docstrings
describe first-frame-empty bugs caused by the task-local queue
(device.jl:326-337, record.jl:776-789, 1432-1440).

### 3.4 Defensive checks on values whose type is known (≈ 150 lines)

116 `=== nothing`/`!== nothing` sites and 20 `isa` sites. `::Any` fields on
hot structs: `MetalHWTLAS` has ten (hwtlas.jl:275, 286-300, 307) and
`records::Any` (251); `MetalTLAS.desc`/`instances` (raytracing.jl:85, 89);
`MetalRecordedDispatch.ndrange` (record.jl:194); `MetalRecorder.pred` (586);
`calls::Vector{Pair{Int,Any}}` (581, 612); `MetalWindow.handle` (window.jl:23);
`libs::Vector{Any}` (graphics.jl:344, mesh.jl:320); `lowered_plain::Any`
(mesh.jl:468-469); `MetalProcess.reporter::Any` (device.jl:304). A required
answer returning `nothing`: `metal_buffer(@nospecialize(x)) = nothing`
(graphics.jl:1467) followed by `=== nothing && error` at four call sites
(graphics.jl:1417, 1437; images.jl:163; mesh.jl:625) — api.md rule 5. Type
tests in place of methods: `denseoperands` (ka.jl:307-311), `d isa
MetalRecordedDispatch || d isa Mantle.Call` (record.jl:708), `d isa Mantle.Call ?`
(741), `b.memory isa MTL.MTLBuffer` (1524), `d isa MetalDevice` over `WeakRef`s
(device.jl:96, 1378, 1402), `passing isa Mantle.DeviceAllocation` (record.jl:281,
332), `hasproperty(nmesh, :face_meta)` (hwtlas.jl:379), `data isa Matrix{T} ?
data : collect(data)` (graphics.jl:162), `x isa NamedTuple` (ka.jl:419),
`color_format === nothing ? … : mtlformat(…)` (graphics.jl:266).

### 3.5 `try`/`catch` (2 sites)

device.jl:1457-1462 catches `OutOfMemoryError` and returns `nothing`
(`tryallocate`); the verb goes with the pool's pressure steps (§7).
raytracing.jl:128-132 is `try`/`finally` (fine). Separately, `capturereports!`
(debug.jl:77-93) turns Metal's abort-on-validation-failure into an append to a
global list that nobody raises; a failure is only seen by a test that drains
`validationmessages!`.

### 3.6 Hand-rolled vector math and manual loops (≈ 60 lines)

raytracing.jl:32-38 `packedtransform` builds `_MTLPackedFloat4x3` from
reinterpreted 12-byte tuples, and is called from nowhere (grep: definition
and docstring only); raytracing.jl:283-288 flattens `AABB`s into six floats by
index (a `reinterpret(Float32, aabbs)` if `AABB` is two `Vec3f`); hwtlas.jl:393-399
triple loop flattening triangle vertices (`reinterpret(Float32, reduce(vcat,
…))`); hwtlas.jl:407-409 bounds union loop (`mapreduce`); raytracing.jl:256
`NTuple{3,Float32}[Tuple(Float32.(v)) for v in vertices]`; graphics.jl:1572-1573
`flip_clip` for `Vec4f` and for `NTuple{4,Float32}`; mesh.jl:145-156
`_mesh_vec4` ×2 and `_mesh_component` ×4 converting tuples and `Vec`s by
element; record.jl:1557-1559 manual marshal loop (`map!(pointer, …)`);
window.jl:302-303 and images.jl:141-142 `unsafe_copyto!` into a fresh `Matrix`
(a `copyto!` over `unsafe_wrap`, or an `MtlArray` view and `Array(...)`).

### 3.7 Near-copies instead of one generic method (≈ 350 lines)

Section 2.2 lists them: `compile_pipeline` ×2, `record_draw!` ×3,
`upload_texture_data!` ×2 (graphics.jl:157-191: host source vs device source,
both a copy node in the target), range/grid writer ×2 with their kernels and
patchers, every queue verb ×2, `closest_hit`/`any_hit` × triangle/procedural
(four bodies, hwtlas.jl:173-237, differing only in a symbol string and the
unpack call), three readback bodies, three "command buffer + wait" blocks,
`ensure_*_linked!` ×2, `_mtlbuffer`/`metal_buffer`+`byteoffset`, `supports_*`
×2 receivers.

### 3.8 Compat shims and abandoned approaches (≈ 1,100 lines)

`LegacyQueue` and everything over Metal.jl's `BatchedCommandQueue` (plan
§9 deletes it; recording-plan.md:1543-1547 "No Mantle path uses Metal.jl's
`BatchedCommandQueue`"): device.jl:85-145, 572-695, 1167-1298, 1347-1414.
The MTL4 encode route kept against an ICB stall (device.jl:147-274, 812-942;
record.jl gate doubles). `MetalVertexStage{F,Out}()` and
`MetalFragmentStage{F,VIn,Out}()` "old spelling" constructors
(graphics.jl:503-505, 541-542). `icb_name(name, i)` kept "for the recorder's
own helpers" (record.jl:247-251). trace.jl whole (superseded by the linked
traversal; verb deleted). `packedtransform` (unused). Stubs that exist to
satisfy a hook: `Mantle.release!(::MetalRecording) = nothing` (record.jl:688),
`reclaim!(::BatchedCommandQueue) = 0` (device.jl:1393), `bufferusage(…) = nothing`
(memory.jl:90), `emitupdate!(::MetalRecorder, …) = nothing` (record.jl:1161),
`emitcopy!`/`beginrender!` error stubs (record.jl:1394-1397), `canrun`/
`refusereplay` (device.jl:903-919), `COMMITTED_MAX` (graphics.jl:901-911: a
bound on a leak rather than removing the global), `runscalls`/
`replaywithcalls!`/`suspendsubmit!` (MPSGraph holes; the plan makes MPSGraph a
foreign node). `Mantle.runscalls(::MetalDevice{LegacyQueue}) = true`
(device.jl:1245) is a capability that depends on the queue generation.

### 3.9 Long functions (none over 100 code lines; seven over 60)

`openrecording` 115 (record.jl:774-888; six `throw` sites of policy: which
piece may be gated, buffer-slot limits, residency-set presence — rule 1 says
a verb decides none of these), `compile_dispatch` 94 (386-479), `sync!` 90
(hwtlas.jl:622-711), `compile_pipeline` 81 (graphics.jl:665-745),
`_write_mesh_vertex!` 78 (mesh.jl:54-131), `compile_pipeline` 76 (mesh.jl:350-425),
`record_draw!` 62 (mesh.jl:589-650).

### 3.10 Keyword arguments accepted and ignored (≈ 40 lines plus contract holes)

`sbt_offset` on five `push!` methods, "accepted and has nothing to do here"
(hwtlas.jl:439-442, 463, 471, 480, 499, 516); `opaque` ignored on
`build_blas`/`build_blas_aabb` (raytracing.jl:226-230, 270-276);
`bindings = nothing` "accepted and ignored" on three `compile_draw` methods
(graphics.jl:1161-1166, mesh.jl:505, 570); `instances` throws on two of three
`record_draw!` methods (mesh.jl:530, 591); `build_blas(d, vertices, ntriangles;
stride, refittable, indices, nvertices)` (raytracing.jl:154-157);
`native_conv2d_dispatch!` with nine keywords (ka.jl:347-350); `Window(…; title,
vsync, color_format, srgb)` + `MetalWindow(T, w, h; vsync, srgb, handle)`
(window.jl:42-43, 116-118).

### 3.11 Parallel data structures kept in sync by hand (≈ 200 lines)

`MetalRecording.resources`/`residentids`/`managed`/`nblocks`/`argmanaged`
(record.jl:631-646) rebuilt together in `ensureresident!` (1509-1562);
`MetalRecorder.grids`/`templ`/`templhost`/`gridcursor`/`gridfirst` (597-601)
patched by `patchwriter!` and `markgates!`; `MetalHWTLAS.blas_list`/
`blas_triangles` (hwtlas.jl:278-279) plus `tri_gpu`/`off_gpu`/`records`/
`scene_buf`/`static_tlas`/`proc_*` (287-300) under two dirty flags (309-310)
that the target's pending-change queue replaces; `MTL4Queue.allocs`/`cbs`/
`at`/`slot` ring (device.jl:733-737); `MetalDevice.channels` +
`releasedsubmits` to keep `submissions` monotone (58-60, 1409, 1427);
`StageArgs.source`/`device`/`buffers`/`scratch` re-baked on every draw
(graphics.jl:1064-1072, 1124-1133); `MetalTLAS.blases` beside
`desc.instancedAccelerationStructures` (raytracing.jl:86, 345); `d.managed`
keyed by `Base.bitcast(UInt, pointer(buf))` (device.jl:48, 451, 465).

### 3.12 Code downstream of a cause that is elsewhere (≈ 300 lines)

`ensureresident!` (record.jl:1509-1562) exists because residency is granted
per recording at run time; api.md §8 puts it in `rawalloc`/`rawfree` through
one helper. `managedbuffer`/`claimbuffers!`/`ManagedLeaves`/`argmanaged`
(device.jl:456-484; record.jl:649-683, 1614-1616) re-implement Metal.jl's
ownership tracking because recorded commands bypass Metal.jl's launch path;
with no `BatchedCommandQueue` the whole mechanism goes. `openrun`'s
`Metal.synchronize()` when a recording exists (record.jl:1420) because host
stores are written through a pointer into memory a frame may be reading; the
target makes stores a change prefix in the submission. Unconditional
`rebake!` per draw (graphics.jl:1095-1133) because "`resize!` on a `MtlVector`
reallocates the `MTLBuffer`"; the target's cell is the one slot a plan bakes.
`checkresident` (record.jl:296-313) refuses sub-8-byte buffers that a pool
never produces. `flushlegacy!` and the bridge command buffers
(device.jl:997-1017, 1070-1074) exist because launches and blits still go to
a second queue. `adoptqueue!` at every entry (section 3.3). `KA.synchronize`
in `writeinstances!` (hwtlas.jl:618) and `Metal.synchronize()` in
`upload_texture_data!` (graphics.jl:187) because builds and blits are separate
command buffers nothing orders. `MTL.use!` per indirect draw with a four-line
warning (graphics.jl:1438-1448) because residency is per encoder.

### 3.13 Forwarding layers (≈ 130 lines)

Twelve `f(d::MetalDevice, …) = f(d.queue, …)` one-liners with docstrings
(device.jl:498-500, 598, 657, 674, 681, 1190, 1206, 1255, 1277, 1288);
backend-axis forwarders the plan deletes (`caps(::MetalBackend)` caps.jl:108,
`Device(::MetalBackend)` device.jl:342, `waitidle(::MetalBackend)` 1335,
`allocate_batch_queue!(::MetalBackend)` 1369, `debugenv(::Union{…})`
debug.jl:27, `framebuffer!(::MTLDevice)` graphics.jl:851); aliases
(`screenshot` window.jl:308, `todevice` device.jl:1384); `KI.matrix_shapes`/
`supports`/`bestshape` (kernelinterface.jl:22-25); `upload!`/`download`/
`devicecopy!` (memory.jl:51-78); `Mantle.Window(::MetalBackend)` →
`MetalWindow` → `attach!` (window.jl:42-47, 71-88, 116-133).

### 3.14 Branching by hand instead of dispatch (≈ 120 lines)

`encodes(e.dev.queue) ? … : …` (record.jl:878, 991, 1056, 1245, 1314-1315,
1355) and the `if !canreplay(d.queue) … elseif isempty(rec.calls) … else`
chain (1644-1652) where the queue's type is known; `walked ? 0 : indirect`
and `d.ndrange isa DeviceRange && walked && …` (396-399); `seg.slot < 0`
(device.jl:636-641, 931); `a.buffer === nothing ? store : a.buffer` twice
(record.jl:846-847, 1047-1048: `RecordedArg` should be two types);
`c.slotoff >= 0 … elseif c.gridoff < 0` (device.jl:871-883); `if indices ===
nothing … else` (graphics.jl:1413-1425; mesh.jl:605-611, 623-638); Symbol
kinds, which api.md rule 2 forbids: `filter === :nearest ? … : filter ===
:linear ? … : error` and `wrap === :repeat …` (graphics.jl:221-227),
`mesh_verts_per_primitive(kind::Symbol)` (mesh.jl:295-299), `mesh_topology_symbol`
(282-284), `MPSGraphGemm.act::Symbol` (ka.jl:227-229), `p.kind === :compute ||
p.kind === :update` (record.jl:702, 739); `mtl_loadaction` mixing a type
method and a predicate branch (graphics.jl:1485-1488); `t.procedural ===
nothing` if/else in `sync!` (hwtlas.jl:678-701).

### 3.15 Compile, allocate and upload inside record and run (structural; ≈ 150 lines directly)

api.md §4.5: "Record compiles nothing, allocates no device memory, uploads
nothing, submits nothing". `openrecording` allocates an ICB, an aux buffer and
three `MtlVector`s and fills them with kernels (record.jl:858-882);
`emitkernel!`, `emitrangewriter!`, `emitgridwriter!`, `emithead!` call
`Metal.mtlfunction` (a compile) during record (1102, 1253, 1274, 1318);
`closerecording!` uploads the template (1358); `record_draw!` for a geometry
pipeline compiles a pipeline per draw (mesh.jl:612-613); `argbytes`
(record.jl:63) and every `MTLBuffer(d.dev, …)` in raytracing.jl:173-179,
234-236, 289-290, 306-312, 350-357, graphics.jl:164, window.jl:294,
images.jl:132 allocate outside the pool; `recorded_state` allocates per
dispatch through `Metal.malloc_buffer_and_gpu_address` (record.jl:354-355);
`ensureresident!` rebuilds at run; `replay!` holds Metal.jl's
`submission_lock` across the whole encode (1628-1636).

### 3.16 Pointers and unsafe calls where a Julian type works (86 sites; ≈ 40 avoidable lines)

Avoidable: `pack_recorded!` copies each argument through `Ref` +
`unsafe_copyto!` (record.jl:333-338) — core's `bindings!` writes typed slots;
`bind_arg!` stores into a `Vector{UInt8}` scratch (graphics.jl:1241-1248) — a
`Ref` is what `setBytes` needs; `unsafe_copyto!` into `MTL.contents`
(hwtlas.jl:401, raytracing.jl:257) — `copyto!` on an `MtlArray`;
`procedural_payload_bytes` (procedural.jl:246-253) — an `MtlArray([dev_side])`
of the struct's own type; `unsafe_copyto!` into a `Matrix` (window.jl:303,
images.jl:142); `Base.bitcast(UInt, pointer(buf))` as a `Dict` key
(device.jl:451, 465) — an `IdDict` on the object; `unsafe_store!(Ptr{UInt64}(p
+ 16), …)` slot writes (record.jl:992-994) — a struct type for the slot.
Inherent: the ObjC bridging in window.jl:72-83 and debug.jl:34, 88-90, and
the device-side `LLVMPtr` code.

### 3.17 Owner's Julia rules, every violation found

- `_name` functions (visibility is by export): `_caps_probe` caps.jl:29;
  `_mtlbuffer` trace.jl:124; `_unpack` hwtlas.jl:137; `_unpack_procedural`
  205; `_as_f32` ×3 and `_from_f32` ×3 procedural.jl:33-38;
  `_write_mesh_vertex!` mesh.jl:54; `_write_mesh_primitive!` 133;
  `_write_mesh_indices!` 181; `_mesh_vec4` ×2 145-146; `_mesh_component` ×3
  147-156; parameter `_compiled` graphics.jl:1613. 18 definitions.
- Globals: section 3.3 (all uppercase, but 15 mutable ones against an
  allowlist of one; state belongs in the device object).
- `try … catch` without rethrow or report: device.jl:1457-1462 (returns
  `nothing`; by contract, but the verb goes).
- Name variants instead of dispatch: `deviceview`/`deviceslice`;
  `readback_framebuffer`/`readback_target`/`readback_window`;
  `emitrangewriter!`/`emitgridwriter!`; `_unpack`/`_unpack_procedural`;
  `closesegment!`/`closerecording!`/`closesubmit!`/`closeframe!`/`closerun!`
  (five "close" verbs on one object graph); `compile_pipeline`/`compile_draw`/
  `compile_stage_function`/`compile_dispatch`/`compile_procedural_candidate`.
- GeometryBasics `Vec`/`Mat`: section 3.6.
- Pointers/unsafe: section 3.16.
- Defensive code on known types: section 3.4.
- One rule the files mostly keep: no `hasmethod`/`applicable` (the one
  `hasmethod` mention is a comment, graphics.jl:498); `hasfield` appears only
  inside `@generated` bodies (mesh.jl:57, 90, 99) where the type is the input.

---

## 4. What the design docs miss or map wrongly for Metal

1. **Device-side overlays have no verb.** api.md §4 lists host-side verbs
   only. Required Metal device-type code with no row: the KI builtin overlays
   and `clip_y` (graphics.jl:1506-1573), the mesh verbs (mesh.jl:163-227),
   the procedural candidate verbs (hwtlas.jl:743-760), the
   `Raycore.closest_hit`/`any_hit` lowering (hwtlas.jl:173-237), `StageOut`
   (device_intrinsics.jl:43), `intrinsic_usage` (ka.jl:56-94; Appendix A says
   "stays" but §4 has no entry). ≈ 600 lines the vocabulary check (§6) would
   reject as not in §4.

2. **No ray-tracing pipeline and no SBT on Metal.** §4.4 `compiletrace`
   returns "a ray-tracing pipeline and its shader binding table layout"; §2.6
   has `RayTracingPipeline(; raygen, closest_hit, miss, any_hit)`; §4.5 has
   `emittrace!`. Metal has none of these (`supports_rt_pipeline = false`,
   hwtlas.jl:716-717): traversal is a linked MSL function called from a compute
   kernel (hwtlas.jl:59-80, 176-181), any-hit is `accept_any_intersection`
   (71-80), and a per-material `closest_hit` tuple (raytracing/pipeline.jl:42-58)
   has no lowering. The plan's Metal section (recording-plan.md:1537-1592) never
   mentions `trace!`. Needed: on Metal `compiletrace` compiles the raygen as a
   compute kernel with the traversal linked (what `hwtlas.jl` does today) and
   `emittrace!` is `emitdispatch!`; or the docs say `trace!` is a core
   operation over `closest_hit` kernels on every device type.

3. **TLAS instance descriptor.** §4.1 names
   `MTLIndirectAccelerationStructureInstanceDescriptor`; the code uses
   `MTLAccelerationStructureUserIDInstanceDescriptor` (raytracing.jl:50-58,
   341) and names BLASes by index into `instancedAccelerationStructures`
   (raytracing.jl:14-19, 70, 86, 345; hwtlas.jl:596, 613-615). The indirect
   descriptor is the right choice (it carries a 64-bit `accelerationStructureID`,
   so a kernel writes `acceladdress`, and `MTLIndirectInstanceAccelerationStructureDescriptor`
   reads its count from a buffer — `CountFromMemory()`), but nothing says the
   index scheme goes, that `acceladdress(dev, as)` is `gpuResourceID`
   (hwtlas.jl:679), or that the transform is Metal's column-major 4x3
   (raytracing.jl:9-13, 69-71 `transpose(r.transform)`) where Vulkan's record is
   row-major 3x4 — so "the TLAS records kernel" (§4.1) must be specialised per
   `instancerecordtype`, not only its element type.

4. **Argument buffers vs descriptors.** §4.4 `compilebindings` "Descriptor
   objects"; §4.8 `tableentry!` "a host write of a descriptor into … the
   device-wide texture table (descriptorsets.adoc:793-807)". Metal has no
   descriptor sets: a texture is an entry-function parameter (graphics.jl:575-596
   `texture_markers`; `Metal/src/compiler/texture.jl`) bound per draw by slot
   (1613-1623). A device-wide table on Metal is an argument buffer of
   `MTLResourceID`s sampled by ID, which needs a compiler change (sample by
   resource ID, not bound slot); the plan's Metal section does not cover it.
   Also unstated: the 31-buffer-slot limit per ICB command (record.jl:898,
   803-813) and the 4 KB `setBytes` limit (graphics.jl:447-470) decide how
   "the plan's argument memory" binds — today one slot per argument
   (record.jl:1046-1050) or one packed `setBytes` block; the plan says only
   "the argument layout measured on the M5" (recording-plan.md:2016).

5. **Residency.** §8 names one helper used by `rawalloc`/`rawfree`/
   `createreserve`/`bindpages!`/`unbindpages!` and compile; the plan puts the
   pool's blocks, reserves and acceleration structures in one set per device
   on every queue. That replaces the five mechanisms in section 2.2. Not
   stated: a structure reached only through an argument buffer must be in the
   set (hwtlas.jl:113-125, 695-700) — covered by "a BLAS reached only through
   a TLAS is resident" if the set holds every pool block, which it does; the
   per-`commit` cost of a residency set per `rawalloc`; that
   `MTL_SHADER_VALIDATION=1` makes everything resident and hides residency
   bugs (device.jl:950-952) — a note §6's tests need.

6. **MTL4 command-buffer semantics.** The plan (recording-plan.md:1559-1567)
   compiles kernel nodes as "one compute ICB per segment …
   `executeCommandsInBuffer`". That is the exact call this backend refuses on
   MTL4 because it stops completing after ~1 s of replayed work
   (device.jl:154-200, 812-822, 906-913; `Metal/research/mtl4_icb_stall.jl`);
   the plan does not mention the stall or what resolved it. Other MTL4 facts
   the docs do not state: an MTL4 command buffer is single-use, so "recorded
   once" means the ICB, not the command buffer (record.jl:1-9); an ICB command
   has no indirect-dispatch form, so a `launchrange(a)` over a resizable array
   is a device-side command rewrite from a prepare kernel (record.jl:69-124
   `writegroups!` via `concurrent_dispatch_threadgroups!`) that the command
   processor sees only from the next `execute` — a forced segment cut
   (record.jl:1116-1124) the "submission budget decides segments" rule does
   not account for; failures arrive through commit feedback, not a command
   buffer status (device.jl:749-753, 1157-1163), so `waitfor` must poll it;
   presenting from an MTL4 queue is `waitForDrawable`/`signalDrawable` on the
   queue plus `[drawable present]`, not `presentDrawable:` on a command buffer
   (window.jl:255-267 uses the legacy form) — "present … through MTL4"
   (recording-plan.md:1541) names no mechanism.

7. **No 64-bit atomics, no Float64, no 8/16-bit atomics.** §2.2 `lengthof(a)`,
   kernel-written lengths, `while_nonzero` counts and TLAS totals
   (`CountFromMemory`) are written by kernels; a count accumulated atomically
   must be 32-bit on Metal. `Cell` (§3) is "address and dims" with no element
   widths. Recommend the docs fix: counts and lengths in cells are `UInt32` on
   every device type; 64-bit addresses are written only by the head node with
   plain stores (legal on Metal); no `Float64` in any shared kernel.

8. **`emitstore!` limit 0** (§4.5) is consistent with Metal (a blit from the
   upload region). Note today's `argbytes` is host-written Shared memory
   (record.jl:56-67), which the `unified = true` path (§2.2) keeps.

9. **Procedural geometry.** No verb for compiling a candidate as a visible
   function, building its table, or registering linked functions per device
   (procedural.jl:213-224, 280-344; hwtlas.jl:100-111, 342). §10 "the SBT
   layout for procedural hit groups" is Vulkan wording. Appendix A maps
   `set_anyhit_pipeline!` to "a field of `RayTracingPipeline`"; Metal
   implements any-hit without a shader.

10. **Appendix A rows that mislabel Metal.** `AdaptedAccel` "goes with
    Metal's HWTLAS bookkeeping" — `AdaptedAccel` is core's (raytracing/accel.jl:29),
    used by Vulkan (hwtlas.jl:213-275) and by Hikari's kernels; what goes is
    `MetalHWTLAS`, and the kernel form still needs the scene pointer,
    triangles, offsets and records (hwtlas.jl:332-334, 355-365).
    `transition_image!` "no implementation on either device type" — Vulkan has
    one (graphics/pipeline.jl:354). `makeimage` → `placeimage`: on Metal
    `rawalloc(dev, Images, …)` returns a placement `MTLHeap`, not a buffer
    (device.jl:425-436), so "one block of kind" has two driver types and
    `createreserve` for images is undefined. Metal-private names with no row
    at all: `simd_width`, `max_threads`, `matrixshapes`, `tryallocate`,
    `blocksize`, `managedbuffer`, `claimbuffers!`, `adoptqueue!`, `ownflush!`,
    `closeframe!`, `canreplay`, `encodes`, `norecordreason`, `walkedplan`,
    `enterrun!`, `framebuffer!`, `commit!`, `submitwait!`, `buffercopy`,
    `attach!`, `nextdrawable`, `make_resident!`, `withaccelencoder`,
    `hostupload!`/`hostdownload`/`hostdevicecopy!` (core helpers Metal forwards
    to, memory/resources.jl:93-124, 642). The vocabulary check must scan
    definitions, not only calls, to catch them.

11. **Validation and debug.** No row for `DebugConfig`, `debugenv`,
    `validating`, `validationmessages!` (debug.jl; core runtime/lifecycle.jl:66-165).
    Metal decides validation per process at start (debug.jl:3-11), so `T(key)`
    cannot honour a `DebugConfig`; a `caps`-style fact ("validating") plus the
    fresh-process test harness is the mechanism. §10 lists only device teardown.

12. **Windows on Metal.** §2.7/§4.7 are written for swapchains:
    "swapchain generation", "per-image present segments",
    `acquirenext(sc) -> (result, index, semaphore)`, `Image(g, T, win)`
    "recreated per swapchain generation … on the same arena memory". A
    `CAMetalLayer` hands out one drawable at a time (window.jl:1-12), the
    drawable's texture is not placeable on arena memory, `next_drawable` blocks
    (240-246), `framebufferOnly` must be false for `Array(win)` (280-282),
    `createswapchain` is `drawableSize` (197-202), and sizes are points vs
    pixels (109-114, 124-125). The plan's Metal section has no window paragraph.

13. **Library kernels.** The plan's "MLX's Metal kernels compiled from
    `.metal` source" exist nowhere yet; what exists is MPSGraph as calls
    (ka.jl:208-425) measured 1.6-4x faster than the in-tree kernels
    (ka.jl:269-272, 380-383, 409-413). "`MPSGraphExecutable` … never encoded into
    our command buffer" (recording-plan.md:1578-1580) contradicts today's
    `replaywithcalls!`, which encodes MPSGraph into the same command buffer
    (record.jl:1656-1690); `runscalls` is `true` only on the legacy queue
    (device.jl:1245) because "MPSGraph has no MTL4 encode entry point" — so on
    MTL4-only every MPSGraph op is a cross-queue hand-off (plan open question 1),
    and SAM 2.1's 130 ms number depends on that answer.

---

## 5. Minimal form: the Metal device type as verbs only

Estimated code lines per verb (no history prose; one-paragraph contract
docstrings would add ~25%). "Guess" marks verbs the docs leave open.

| verb (api.md) | Metal implementation | est. lines |
|---|---|---|
| `hardwarekeys(T, registry)`, `T(key)` | `Metal.devices()`; `MTLDevice` + `MTL4CommandQueue`s + one `MTLSharedEvent` each + one `MTLResidencySet` | 30 |
| `caps(dev)` | `threadExecutionWidth`/`maxTotalThreadsPerThreadgroup` from one probe pipeline, `maxThreadgroupMemoryLength`, `maxBufferLength`, `recommendedMaxWorkingSetSize`, `supportsPlacementSparse`, `num_gpu_cores`, 8x8 shapes, inline store 0, flags | 45 |
| `config(dev, f)`, `queuekinds`, `countsource`, `budget`, `instancerecordtype` | one-liners + `MetalInstanceDescriptor` (indirect layout) struct | 25 |
| `submitnative!(ch, token, parts, waits, signals, present)` | `waitForEvent:value:` per wait, `commit:count:` of the parts' command buffers, `signalEvent:value:`, `waitForDrawable`/`signalDrawable` + `present` | 40 |
| `passed(ch, v)`, `waitfor(ch, v)` | `signaledValue` read; bounded `waitUntilSignaledValue` loop checking commit feedback | 15 |
| `recycle!(ch, objects)` | allocator reset once its token passed; pair back to the free list | 12 |
| `bindpages!`/`unbindpages!` | placement-sparse mapping on the sparse queue kind (guess; API surface unverified) | 40 |
| `rawalloc`, `rawfree` | `MTLBuffer` per kind, placement `MTLHeap` for `Images`; add/remove in the device set through the residency helper | 30 |
| `createreserve` | placement-sparse buffer over a reserved range, added to the set | 15 |
| residency helper | `addAllocation`/`removeAllocation`/`commit` on the device set; a per-plan set for ICBs, argument memory, pipelines | 20 |
| `placeimage`, `placeaccel`, `accelsizes`, `acceladdress`, counted objects | texture on heap at offset; `alloc_acceleration_structure` on a region (`newAccelerationStructureWithDescriptor:offset:` on a heap); `accelerationStructureSizes`; `gpuResourceID`; bytes + destroy for ICB/pipeline | 45 |
| `compilekernel(dev, kernel, config)`, `kernelinfo` | `mtlfunction(…; indirect = true)`; `KernelState` once per device; registers/threads from the pipeline | 50 |
| `compiledraw(dev, pipeline, argtypes, formats)` | state tables (55), format table (25), vertex/fragment/mesh wrappers (120), pipeline + depth-stencil descriptors (40), mesh pipeline descriptor (30) | 270 |
| `compiletrace(dev, pipeline, argtypes)` | the raygen as a compute kernel with the linked traversal (section 4 item 2) | 20 |
| `compilebindings(dev, s)` | `MTL4ArgumentTable` or address writes for the plan's argument block | 15 |
| `candidates(dev, f)` | MLX kernel entries + `MPSGraphExecutable` foreign-node candidates | 40 |
| `Builder(dev, ch)`, `finish(b)` | allocator + command buffer from the free list; `endCommandBuffer` | 20 |
| `emitdispatch!`/`emittrace!` | ICB command: pipeline, buffer slots, grid or execution-range slot, barrier bit; the prepare-rewrite for `launchrange` | 40 |
| `emitcopy!` | blit encoder: buffer↔buffer, buffer↔texture | 25 |
| `emitstore!` | limit 0: none | 0 |
| `emitbuild!` | acceleration-structure encoder build/refit with the count from a cell | 30 |
| `emitloop!` | execution-range writer kernel + the reset kernel, segment ranges | 50 |
| `beginrender!`, `emitpieces!`, `endrender!`, `recordpiece`, `emitdraw!`, `bindindices!` | render pass descriptor from targets and loads; render ICB per piece; per-draw `MTLIndirectRenderCommand` + the state it cannot hold on the encoder; `drawMeshThreadgroups` for mesh | 140 |
| `emitlibrary!` | foreign node: event-chained `MPSGraphExecutable` | 15 |
| `devicearg(dev, a)`, vendor view | `MtlDeviceArray` from region + offset + dims; borrowed `MtlArray` over a block | 25 |
| window open verb, `createswapchain`, `acquirenext`, `surfacesize` | GLFW + `CAMetalLayer` attach (ObjC), `drawableSize`, `nextDrawable`, framebuffer size | 65 |
| `tableentry!`, `samplerentry!`, `imageformat` | argument-buffer entry of a resource ID; `MTLSamplerState`; the one format table | 45 |
| device-side (no verb yet; section 4 item 1) | KI builtins + `clip_y` (15), mesh verbs (150), `closest_hit`/`any_hit` + `HWTLAS_MSL` (170), procedural visible function + `PROCEDURAL_MSL` + table (190), `intrinsic_usage` (35) | 560 |
| **total** | | **≈ 1,790 code; ≈ 2,200 with contract docstrings** |

Of today's 8,649 lines this keeps the state tables, the format table, the
stage and mesh wrappers, the MSL sources, the overlay vocabulary, the ICB
encode, the allocation calls and the window attach; it drops both queue
generations, every forwarder, the three caches, the HWTLAS bookkeeping, the
library call types, the readback bodies, the per-frame rebake and residency
rebuild, and ~4,500 lines of narrative.
