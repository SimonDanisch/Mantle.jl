# Audit: src/graphics, src/raytracing, ext/MantleROCmExt.jl

Historical snapshot: counts, locations and deletion hypotheses below predate
consolidation. Follow [current ownership and deletion gates](consolidation-order.md).

Scope: every line of `src/graphics/*.jl` (1,986 lines), `src/raytracing/*.jl`
(781 lines) and `ext/MantleROCmExt.jl` (1,478 lines); 4,245 lines in all.
Compared against `src/vulkan/graphics/*.jl`, `src/vulkan/raytracing/*.jl`,
`src/vulkan/lowering.jl`, `src/metal/{graphics,hwtlas,raytracing,mesh,procedural}.jl`
and, for the ROCm concerns, `src/vulkan/graph.jl`, `src/vulkan/access.jl`,
`src/metal/{device,memory,record,ka,caps}.jl`. Consumers checked:
`dev/Makie/RayMakie/src`, `dev/Hikari/src`, `dev/VideoEditor/src`, `dev/Raycore/src`.
Target: `docs/api.md` sections 2.6, 2.6b, 4, Appendix A; `docs/recording-plan.md`
"Acceleration structures and textures" (998-1105), "Graphics" (1107-1222),
"What a device type implements" (1381-1409), "RayMakie on the core model"
(1471-1536), "10. CUDA and HIP" (2030-2042), "Forbidden moves" (1632-1652).

Line counts per bucket are from reading, not from a tool; they are estimates
to within about 10%. Where a number is a guess it says so.

Line-number note: `src/graphics/pipeline.jl`, `src/raytracing/instances.jl`,
`src/raytracing/pipeline.jl` and `src/graphics/builtins.jl` are cited by their
own file lines.

---

## 1. Inventory

Buckets: (a) survives into the target core; (b) becomes a device-type verb
(the declaration survives, under an api.md name); (c) belongs nowhere (dead,
superseded, compat shim, test-only, duplicate, narrative).

| file | lines | what it does | a | b | c | api.md type/verb it maps to |
|---|---|---|---|---|---|---|
| `src/graphics/commands.jl` | 492 | Declares 25 backend verbs for passes, draws, windows, readback, capability predicates, `compile_draw`; two core helpers (`checkacquirable` 102-110, `readback_target`/`restingusage` 177-209) | ~10 | ~15 | ~467 | `compile_draw`→`compiledraw` (A); `begin_render_pass!`/`record_draw!`/`end_render_pass!`→`beginrender!`/`emitdraw!`/`endrender!` (4.5); `acquire_next_image!`→`acquirenext` (4.7); `supports_*`→`caps` fields (4.1); `readback_*`→`Array(img)`/`Array(win)` (2.6b); `begin_pass!` family, `present_frame!`, `beginframe!`, `transition_image!`, `copy_target!`, `use_bindings!`, `argidentity`: "goes" (A) |
| `src/graphics/record.jl` | 346 | The hand-recorded pass DSL (`PassRecorder`, `pass!`, `viewport!`, `bindings!`, `draw!(p,…)`), target resolution (`passtargets`, `colorimage`/`depthimage`/`currentimage`), `blit!` with its two shaders, eager `draw!(dev, pipeline, target, count; 8 kwargs)` | ~40 | 0 | ~306 | `draw!(dev, target, pipeline, args, count; instances, indices)` (2.6b, eager form); `clear!`; the rest "goes" (`setviewport!`, `colorimage`, `depthimage`, `currentimage`, `blittarget`: A) |
| `src/graphics/lowering.jl` | 339 | Rewrites `GraphicsPipeline` + `GeometryShader` into a `MeshPipeline` (`GeometryAsMesh`, `lower_geometry_to_mesh`, `requirevertexindex`); Metal's `compile_draw` applies it (`src/metal/mesh.jl:567-589`) | ~200 | 0 | ~140 | none; api.md 2.6b lists `GeometryShader` and `MeshShader` as stage types and says nothing about a device type without a geometry stage. See section 4. |
| `src/graphics/resources.jl` | 246 | Abstract `Texture{1,2}D`, `Sampler`, `SampledTexture`, `TextureBindings`, `bind_textures`, `upload_texture_data!`, `Framebuffer`, `WindowTarget`/`OffscreenTarget`, `target_extent`/`target_format`/`eltypeof`/`blittarget` forwarders, `CompiledGraphicsPipeline`, `HWTLAS`, `ExternalImage` | ~5 | 0 | ~241 | `Image`, `Sampler` value, texture table (2.6b); `Framebuffer`, `Texture2D`, `Surface`, `target_*`: "goes" (A); `HWTLAS`→core `TLAS{P}` (3); `ExternalImage`: no row (section 4) |
| `src/graphics/stages.jl` | 226 | `ShaderStage` and the five stage structs with constructors and accessors (`stageconfig`, `stageoutputs`, `flatoutputs`, `outputtype`, `ntextures`) | ~120 | 0 | ~106 | `VertexShader`, `FragmentShader`, `GeometryShader`, `MeshShader`, `ObjectShader` (2.6b). `FragmentShader.textures` (84-104) goes with the device-wide table. |
| `src/graphics/state.jl` | 133 | `BlendMode` (incl. `PerAttachment`, `NoWrite`), `attachmentblend`, `CullFace`, `DepthMode`, abstract `RenderTarget` | ~100 | 0 | ~33 | the fixed state of `GraphicsPipeline(; blend, cull, depth)` (2.6b); `RenderTarget` goes (targets are images and windows) |
| `src/graphics/pipeline.jl` | 115 | `GraphicsPipeline` struct + kw constructor, `Rasterizer`, `TrianglePipeline`/`LinePipeline`, `lastgeometrystage`, `fragmentinputs`, `fragmentinputtype`, `DrawIndirectCommand` | ~55 | 0 | ~60 | `GraphicsPipeline`/`Rasterizer` (2.6b). `tess_control::TC`/`tess_eval::TE` as untyped `(func, TessConfig)`/func (34-35) contradict "tessellation as stage types" (2.6b). `DrawIndirectCommand` (109-114): no row. |
| `src/graphics/mesh.jl` | 79 | `MeshPipeline` struct + kw constructor, `meshconfig`, `objectconfig`, `fragmentinputs` | ~45 | 0 | ~34 | `MeshPipeline` (2.6b) |
| `src/graphics/builtins.jl` | 10 | A comment stating that nothing is declared here (the builtins are KernelInterface's) | 0 | 0 | 10 | — |
| `src/raytracing/api.jl` | 290 | Declares `build_accel!`, `build_blas`, `build_blas_aabb`, `refit_tlas!`, `trace_rays!`×2, `trace_closest_hits!`×4, `set_anyhit_pipeline!`; the graph `trace!` (122-139); `shaders`/`shadertouches`; `supports_rt_pipeline`/`supports_procedural_traversal`/`supports_hwtlas` | ~40 | 0 | ~250 | `trace!(g, …)` → `TraceNode` (3); `shadertouches` stays (A); `BLAS(dev, vertices, faces)`, `build!`, `refit!` (2.6); `emitbuild!`, `emittrace!` (4.5); `supports_*`→`caps`; `trace_*`/`set_anyhit_pipeline!`/`build_accel!`/`refit_tlas!`: "goes" (A) |
| `src/raytracing/accel.jl` | 237 | `AdaptedAccel` (kernel-side accel with concatenated triangles + offsets + instances), `holdleaves!`, `NOTRIANGLES`/`hittriangle`/`toworld`, `AccelBuildContext`, the procedural protocol (`procedural_miss/candidate/commit/bary`, `candidate_primitive_index`, `candidate_object_ray`, `commit_intersection!`) | ~40 | 0 | ~197 | `AdaptedAccel`: "goes" (A); `holdleaves!`: "goes at emit" (A); range table replaces triangles/offsets (2.6 "reached through the range table, never through a scene-wide concatenation"); procedural protocol: no row (section 4) |
| `src/raytracing/batches.jl` | 118 | `InstanceBatches{B}`: handle allocator, `Vector` of batches + `Dict` of positions, `register!`, `batchof`, `delete!` with reindex, `sethidden!`, `recordsof`, `ninstances` | ~40 | 0 | ~78 | `InstanceRange`, `RangeEntry` (`mask`, `active`), slot free-list (3; 2.6 `push!`/`delete!`/`tlas[h] = t`) |
| `src/raytracing/instances.jl` | 76 | `instancerecords` (host vector of `Raycore.InstanceRecord` from `Mat4f`s), `update_instance_records_kernel!`, `write_grain_instances_kernel` (a grain-physics consumer kernel) | ~10 | 0 | ~66 | the records kernel (2.6: "a records kernel writes all instance records"); `push!(tlas, blas, transforms::MantleArray{Mat4f,1})`; `write_grain_instances_kernel` is test-only in this tree (`test/test_instance_writer_kernel.jl:37`, `test/test_hwtlas_refit_stress.jl:41`) |
| `src/raytracing/pipeline.jl` | 60 | `mutable struct RayTracingPipeline` with `Any` fields, `_normalise_chit`×3, kw constructor, `shaders` | ~25 | 0 | ~35 | `RayTracingPipeline(; raygen, closest_hit, miss, any_hit)` (2.6). `payload_type` and `chit_miss_take_args` (26-37) have no row; Hikari sets both (`Hikari/src/integrators/volpath/rt-pipeline.jl:569-570`). |
| `ext/MantleROCmExt.jl` | 1478 | The HIP device type: device singleton, pool verbs, retirement-counter timeline, borrowed `ROCArray` views, transfers, dispatch compilation with launch geometry, stream-capture recording with its own invalidation and warm-up, profiling, hipBLASLt/MIOpen/rocBLAS library hooks that declare passes into the user's graph, caps | 0 | ~370 | ~1,100 | 4.1 `caps`, `budget`, `hardwarekeys`; 4.2 `passed`/`waitfor` (rewritten on stream counters); 4.3 `rawalloc`/`rawfree`; 4.4 `compilekernel`/`kernelinfo`/`candidates`; 4.6 `devicearg`, vendor view (`rocview` 437-460); 4.5 Builder over `hipGraph` nodes (new). Plan 10 deletes: stream capture (1078-1160), `checkresolved` (1249-1266), the retirement timeline (389-413), `ROCmCompiledDispatch` (548-595). |
| **total** | **4,245** | | **~730** | **~385** | **~3,130** | |

Reading of the totals: about 17% of these lines carry content the target core
keeps (pipeline/stage value types, the geometry-to-mesh rewrite, the trace
node, a trimmed procedural protocol, a range-table seed); about 9% are verb
declarations or verb bodies that survive under api.md names; the remaining
~74% is declarations of verbs Appendix A deletes, forwarders, two generations
of the same path kept side by side, capability predicates that become `caps`
fields, and narrative comments.

---

## 2. Duplication

### 2.1 Within the core files and against Vulkan/Metal

**D1. Two pass-verb families in core, three begin/draw/end implementations per backend.**
`commands.jl:12-60` declares `begin_pass!`/`end_pass!`/`draw_in_pass!`/`draw_indexed_in_pass!`/`draw_indirect_in_pass!`/`set_viewport!`; `commands.jl:377-405,487-492` declares `begin_render_pass!`/`record_draw!`/`end_render_pass!`; `record.jl:40-175` adds a third spelling (`PassRecorder`, `viewport!`, `draw!(p,…)`, `bindings!`, `pass!`) that forwards to the second. `record.jl:3-18` says the first family "was Vulkan's alone" and the file exists to replace it, yet `commands.jl:12-60` still declares and `src/Mantle.jl:337` still exports it. On Vulkan the first family is the only one with a body (`src/vulkan/graphics/pipeline.jl:390-402,541,647,683,730-757,741`), and both the graph walk (`src/vulkan/graph.jl:1488,1502,1507,1961,1984,1990,1999`) and the second family's bodies (`src/vulkan/graphics/record.jl:130,146,155,163,180,186`) call it. So one draw on Vulkan passes through: `draw!(p::PassRecorder)` `record.jl:85` → `record_draw!(VulkanPassHandle, VulkanDraw)` `src/vulkan/graphics/record.jl:170-191` → `draw_in_pass!(e::Emitter,…)` `src/vulkan/graphics/pipeline.jl:541` → `VK.cmd_draw`. Metal implements the second family four times for the draw (`src/metal/graphics.jl:1250`, `src/metal/mesh.jl:525,589,656`). The mesh/indexed/plain three-way `if` is written by hand in both Vulkan bodies (`src/vulkan/graphics/record.jl:177-190`, `src/vulkan/graph.jl:1961-1999`). Recoverable: `commands.jl:12-60` (49), `record.jl:40-175` (136), `src/vulkan/graphics/record.jl:120-191` (72): ~257 lines, replaced by one `emitdraw!(b, node, cmd::CompiledDraw)` per device type.

**D2. Capability predicates: 7 booleans, ~190 lines.**
Core: `commands.jl:211-295` (`supports_graphics`, `supports_geometry_stage`, `supports_tessellation`, `supports_mesh_pipeline`, each as an untyped default plus a `::Device` forwarder: 245-246, 269-277, 294-295), `api.jl:197-278` (`supports_rt_pipeline` 214-226 with three methods, `supports_procedural_traversal` 255-263, `supports_hwtlas` 278). Vulkan: `src/vulkan/array/ka_backend.jl:82`, `src/vulkan/runtime/device.jl:2359`, `src/vulkan/graphics/api.jl:396-403`, `src/vulkan/raytracing/hwtlas.jl:1076-1084`, `src/vulkan/graph.jl:142`. Metal: `src/metal/graphics.jl:1019-1020`, `src/metal/mesh.jl:439-440`, `src/metal/hwtlas.jl:716-729`. Appendix A: all become fields of `caps(dev)`. Recoverable: ~165 core + ~20 backend lines for ~7 fields of one record.

**D3. Target accessors.** `resources.jl:169-194` (`target_extent`/`target_format`/`blittarget` forwarders for the two wrappers), `record.jl:323-346` (`depthtarget`, `blittarget`), `record.jl:99-144` (`passtargets`, `colorimage`, `depthimage`, `currentimage`); backends: `src/vulkan/graph.jl:349-350`, `src/vulkan/graphics/window.jl:288-289`, `src/vulkan/graphics/record.jl:32-37,57-`, `src/metal/graphics.jl:1375-1385`, `src/metal/window.jl:149-151`. Recoverable: ~70 core + ~25 backend; an `Image` has `dims` and `T`, a `Window` its size and format table entry.

**D4. HWTLAS bookkeeping written twice over one core container.** `batches.jl` is the shared part; everything around it is duplicated:

| concern | Vulkan `src/vulkan/raytracing/hwtlas.jl` | Metal `src/metal/hwtlas.jl` |
|---|---|---|
| batch struct (records, n, handle, triangles, hidden + one backend field) | 42-58 `InstanceBatch{Tri}` | 242-258 `MetalInstanceBatch{Tri}` |
| TLAS struct with `dirty`, `transforms_dirty`, `static_tlas`, `blas_list`, `tri_gpu`, `off_gpu` | 110-181 | 274-313 |
| `AdaptedAccel(hwtlas)` + `Adapt.adapt_structure` | 213-231, 250-277 | 332-335, 336-370 |
| `push!` family (mesh/blas × one/many transforms × records) | 404-520 (5 methods) | 460-525 (5 methods) |
| `update_transforms!`/`update_transform!` | 521-626 (6 methods) | 526-552 (4 methods) |
| `delete!`, `set_visible!` | 627-663 | 553-573 |
| instance-writer kernel: read `InstanceRecord`, write the driver record + a shading copy | 685-696 `vkinstances_kernel!` | 589-599 `mtlinstances_kernel!` |
| `writeinstances!` loop over batches with offset | 706-734 | 608-620 |
| BLAS-list compaction | 735-768 `_compact_blas_list!` | inside `sync!` ~653-690 |
| `sync!`: rebuild-vs-refit decision | 847-920 | 622-710 |
| hit unpacking (`hittriangle`, procedural branch) | 961-1000 `rq_collect` | 137-240 `_unpack`, `_unpack_procedural` |
| capability predicates | 1076-1084 | 716-729 |

The two instance-writer kernels differ only in the record constructor (`VulkanInstanceRecord(...)` at 692 vs `MetalInstanceDescriptor(r, index, options, keep)` at 597): that is exactly `instancerecordtype(dev)` (api.md 4.1). The plan says "HWTLAS bookkeeping (geometry list, instance concatenation, rebuild versus refit, BLAS collection, compaction) is core's". Recoverable: of 1,084 + 760 = 1,844 backend lines, ~1,100 is this table twice, plus core `batches.jl` 118 and `accel.jl:1-111` (`AdaptedAccel`, `hittriangle`) which the range table replaces.

**D5. The 4x4-to-3x4 transform conversion exists three times.** `src/geometry/transform.jl:121-133` `mat4_to_vk_transform` (element by element, used by `instances.jl:23` and `src/metal/hwtlas.jl:539`), `Raycore.mat4_to_mat3x4` (`dev/Raycore/src/instanced-bvh.jl:140`, used by `src/vulkan/raytracing/hwtlas.jl:544`), and Metal's `packedtransform` (`src/metal/raytracing.jl:32`) transposing it again for `MTLPackedFloat4x3`. ~40 lines for one `convert(Mat3x4f, ::Mat4f)`.

**D6. The procedural intrinsics are declared in core and overridden per backend.** `accel.jl:201-227` declares `candidate_primitive_index`, `candidate_object_ray`, `commit_intersection!`; `src/vulkan/raytracing/hwtlas.jl:951-959` and `src/metal/hwtlas.jl:743-760` override them as device functions. `builtins.jl:1-10` states the rule that device builtins are KernelInterface's because Lava cannot override a Mantle name without a bridge; the raytracing builtins break it. Not duplicate bodies, but a third home for one vocabulary.

**D7. Five immediate trace entry points for one operation.** `api.jl:85-98` (`trace_rays!`, `trace_rays_indirect!`), `api.jl:158-188` (`trace_closest_hits!` × {plain, indirect} × {closest, anyhit}); Vulkan bodies `src/vulkan/raytracing/shaders.jl:44-127`, `src/vulkan/raytracing/raycore_compat.jl:126-171`; Metal `src/metal/trace.jl:134`. Appendix A: all go; `trace!(dev or g, …)` with a `launchrange` count and the `any_hit` field covers the four axes. Hikari calls `trace_closest_hits!` twice (`dev/Hikari/src`), so a device-side ray-query replacement must exist before deletion (section 4). Recoverable: ~40 core + `raycore_compat.jl` 247 + ~100 of `shaders.jl`.

**D8. Build entry points, with two forwarders and a doubled Metal definition.** `api.jl:18-83` declares `build_accel!`, `build_blas`, `build_blas_aabb`, `refit_tlas!`; `src/vulkan/graph.jl:89-90` forwards `build_accel!(f, d::LavaDevice)` to the channel; Metal defines `build_blas` twice (`src/metal/raytracing.jl:154` unqualified, `:229` as `Mantle.build_blas`) and `build_accel!` twice (`:213` `Mantle.`, `:327` unqualified), which is the "a missed `import` silently defines a new function" trap the plan names (recording-plan.md:1402-1405). Replaced by `BLAS(dev, vertices, faces)`, `build!`, `refit!`, `emitbuild!`, `accelsizes`, `placeaccel`.

**D9. `lastgeometrystage`/`fragmentinputs`/`fragmentinputtype` written twice.** `pipeline.jl:67-92` for `GraphicsPipeline` and `mesh.jl:74-79` for `MeshPipeline`, with `stageinputs` (`stages.jl:198`) as a third name for `stageoutputs`. One `fragmentinputs(p) = stageoutputs(laststage(p))` over both pipeline types: ~25 lines → 8.

### 2.2 ext/MantleROCmExt.jl: every concern, and whether Vulkan and Metal have it too

| concern | ROCm (ext) | Vulkan | Metal | target |
|---|---|---|---|---|
| module narrative (benchmarks, history) | 1-131 (131) | — | — | none (plan: docs state the target) |
| device struct, stream adoption, process singleton, `defaultdevice!`, `devices` | 148-289 (142): `ROCmDevice` 178-195, `adopt!` 244-247, `ROCM_DEVICE` 252, `Device(::ROCmAPI)` 254-261 | `src/vulkan/graph.jl:38-142`, `src/vulkan/runtime/device.jl:2012` | `src/metal/device.jl:324-345` | `hardwarekeys(T, registry)`, `T(key)` (4.1); the registry owns the singleton (3 `DeviceRegistry`) |
| `waitidle`, `awaitwrites` | 291-327 (37) | `src/vulkan/graph.jl:287`, `runtime/device.jl:2355-2356` | `src/metal/device.jl:565,1335-1342` | `waitidle` open (A); `awaitwrites` goes |
| constraints, `rawalloc`/`rawfree`, `blocksize`, `capacity`, `maxalloc` | 329-385 (57) | `src/vulkan/graph.jl:254-256,732-939` | `src/metal/device.jl:364-487,1304-1323` | `rawalloc(dev, kind, bytes, constraint)`, `rawfree`, `budget`, `caps.maxalloc` (4.1, 4.3) |
| timeline: `fence`/`passed`/`waitfor` on a retirement counter | 387-413 (27) | `src/vulkan/graph.jl:271-300` | `src/metal/device.jl:498-565,1113-1147` | `passed(ch, value)`, `waitfor(ch, value)` over `hipStreamWriteValue64` (4.2; plan 10 deletes the retirement counter) |
| borrowed `ROCArray` view, `deviceview`, `deviceslice`, `alignment` | 415-494 (80) | `src/vulkan/graph.jl:315,403` | `src/metal/memory.jl:33`, `src/metal/ka.jl:111,124` | vendor view verb (4.6, 2.9); `deviceslice` goes (A) |
| `upload!`/`download`/`devicecopy!` | 496-530 (35) | `src/vulkan/graph.jl:331-336` | `src/metal/memory.jl:51-77` | go: `Store` changes, `emitcopy!`, `Array(a)` (A) |
| compiled dispatch type + `argsize`/`indirectindex`/`devicesized` | 532-595 (64) | `src/vulkan/graph.jl:1032-1149` (`CompiledTrace` 1156) | `src/metal/record.jl:221-226,386` | goes: `CompiledDispatch` is core's (3); plan 10 deletes `ROCmCompiledDispatch` |
| `compile_dispatch`, KI path, launch geometry, `hipfunction` | 597-742 (146) | `src/vulkan/graph.jl:1032-1149` | `src/metal/record.jl:386-` | `compilekernel`, `kernelinfo` (4.4); launch geometry is "one core calculation from KernelInfo" (A `callgroup`, `workgroupsize`) |
| `recordsplans`, `recordable`, `runscalls`, `profiled!` | 744-808 (65) | `src/vulkan/graph.jl:1324,1773` | `src/metal/record.jl:1176,1405-1406`, `device.jl:1245` | go (A); profiling open (10) |
| hipBLASLt bias GEMM (hand-written ccall FFI) | 810-881 (72) | none | `src/metal/ka.jl:274` (`librarygemm`) | a library candidate (4.4 `candidates`) |
| MIOpen convolution, declaring passes into the user's graph | 883-970 (88) | none | `src/metal/ka.jl:347` | candidate; declaration is core's |
| rocBLAS batched GEMM, same | 972-1035 (64) | none | `src/metal/ka.jl:193` | candidate |
| `librarygemm` selection by eight `return nothing` tests | 1037-1048 (12) | none | `src/metal/ka.jl:274` | selection table (A) |
| stream-capture recording: `openrecording` with Launch refusal and warm-up, `closerecording!` with memory-node audit, `ROCmRecording`, `release!` | 1050-1210 (161) | `src/vulkan/graph.jl:1679-1727` | `src/metal/record.jl:774-1347` | `Builder`, emit verbs, `finish` building `hipGraph` nodes (4.5); plan 10 deletes capture |
| `abandonrun!`/`abandonrecording!`/`endcapture!`, `checkresolved` | 1212-1266 (55) | `src/vulkan/graph.jl:1878-1882` | none | go (A); plan 10 deletes `checkresolved` |
| `openrun`/`closerun!`×2/`submitrun!`/`submitrecording!` | 1268-1330 (63) | `src/vulkan/graph.jl:1782-1856` | `src/metal/record.jl:1419-1584` | `submitnative!` (4.2) |
| `deviceaddress`, `patchable`, `supportspredicate` | 1332-1377 (46) | `src/vulkan/graph.jl:233,328` | `src/metal/memory.jl:102` | cell (A `deviceaddress`); `patchable`, `supportspredicate` go |
| `caps` from HIP properties | 1379-1428 (50) | `src/vulkan/runtime/device.jl:695`, `graph.jl:136` | `src/metal/caps.jl:83-108` | `caps(dev)` (4.1) |
| `AccessCache`, `argtype`, `devicebuffertype`, `isdevicearray`, `kernelinterpreter`, `syncbackend` | 276-289 (14) | `src/vulkan/access.jl:12-44` | `src/metal/ka.jl:135-142,427,444` | `devicearg` (4.6); access into the `Kernel` (3); `kernelinterpreter` stays |
| deleted-code commentary, `__init__`, hipBLASLt `dlopen`, `register_backend!` | 1430-1478 (49) | `initbackend!` (A: goes) | — | registry loads device types |

**Does the extension do more than "memory and queues; compute"?** Yes, in eight places:

1. It owns the recording mechanism and the decision whether a plan may be recorded: `recordable` 774, `recordsplans` 764, the `Launch` refusal 1083-1087, the memory-node audit 1151-1180.
2. It owns move invalidation policy: `patchable` 1359 and `checkresolved` 1249-1266 compare resolved pointers against baked ones ("a change detected instead of queued", plan 1645).
3. It declares nodes and transients into the user's graph from inside the backend: `Mantle.dispatch!(g, MIOpenConv(…), …)` 967, `Mantle.dispatch!(g, Widen(), …)` 968, `Mantle.dispatch!(g, call, …)` 1033, `Mantle.Transient.Buffer(g, …)` 959, 966. The plan gives selection to core and a device type only `candidates(dev, f)` (4.4).
4. It computes launch geometry (`kicompile` 637-652, `hipcompile` 727-739 with the occupancy autotune), which api.md makes core's one calculation.
5. It holds a process-wide device singleton and adopts its stream into the calling task (`ROCM_DEVICE` 252, `adopt!` 244-247, called from 15 verbs).
6. It profiles (`profiled!` 781-794).
7. It executes every library call once on the host before capturing (1112-1114) and converts arguments for their side effect (1122-1129): work at record time that the plan forbids ("Record compiles nothing, allocates no device memory, uploads nothing, submits nothing", api.md 4.5).
8. It loads and binds a vendor library by hand (`hipblaslt_library` 206-214, `HIPBLASLT` 217, `__init__` 1463-1472, twelve `ccall`s 839-878).

---

## 3. Patterns that inflate the code

Each pattern: examples with file:line, then an estimate of its line cost inside the audited files.

**P1. Branching on kind by hand instead of dispatch.** `ext:581-585` (`if d.ctx === nothing … else …`), `ext:641` (`nd isa Integer ? … : …`), `ext:643-644` (nested `isa Integer`), `ext:669` (`d.ndrange isa Mantle.DeviceRange && d.ndrange.max === nothing`), `ext:1083` (`x isa Mantle.Launch`), `ext:1113` (`cd isa Mantle.Call && cd()`), `ext:1123`, `ext:1251-1255` (three `isa` in one loop), `ext:930` (`x isa Union{Buffer,TransientBuffer,ResourceView}`), `commands.jl:371` (`depth isa DepthOff ||`), `state.jl:72` (`mode isa Union{Opaque,NoWrite}`), `lowering.jl:261` (`tess_control === nothing && tess_eval === nothing && return _lower`). The plan forbids "an `isa`/`typeof` branch on a device or node type" (1640). Cost: ~60 lines plus the branches they guard.

**P2. Defensive `=== nothing` on values of known type, and three-valued returns.** `batches.jl:54-60` (`batchof` returns `nothing` for a stale handle "so a caller can ask without an exception"), `batches.jl:105-118` (`sethidden!` returns `Bool or nothing`), `batches.jl:71-80` (`delete!` returns `false` for a missing handle), `accel.jl:60-65` (three constructors filling trailing fields with `nothing`), `accel.jl:108-110` (`toworld(::Nothing,…)`), `record.jl:110,118,122,317,333`, `pipeline.jl:67-68` (`p.geometry === nothing ? …`), `mesh.jl:65`, `instances.jl:21,23`, `ext:193` (`caps::Union{Nothing,DeviceCaps}` filled lazily at 1395), `ext:899,907` (`algo::Union{Nothing,…}`), `ext:1038-1047` (eight `|| return nothing` lines: a candidate function answering "not supported" with `nothing`, the "fallback that returns `nothing`" the plan forbids at 1637), `ext:953-958` (five `|| return nothing`), `ext:1025-1031` (six `|| return false`). Cost: ~80 lines.

**P3. `try`/`catch` that swallows.** None in the audited files. `record.jl:169-173` and `ext:837-879` are `try`/`finally` (cleanup, correct). `ext:1466-1468` is a soft failure of another kind: `dlopen(…; throw_error = false)` then `@warn` and continue, so a missing or broken hipBLASLt degrades the device silently after one log line. Cost: 0 lines; one behaviour to remove.

**P4. Globals.** `record.jl:212-215` `const BLIT_PIPELINE` (a module constant used in one function, `blit!` 269; the owner's rule: a kwarg or a local), `accel.jl:88` `const NOTRIANGLES` (a sentinel in a parallel array, read at 104 and in both backends: `src/vulkan/raytracing/hwtlas.jl:813`, `src/metal/hwtlas.jl:660`), `ext:217` `const HIPBLASLT = Ref{Ptr{Cvoid}}(C_NULL)` (mutable; the device already carries `blaslt::Ptr{Cvoid}` at 184, so the handle lives in two places), `ext:252` `const ROCM_DEVICE = Ref{Union{Nothing,ROCmDevice}}(nothing)` (a process singleton with a comment at 249-251 explaining it mirrors `METAL_DEVICE`; the target's `DeviceRegistry` owns this). `ext:145-146` and `lowering.jl:71-72` are type aliases (fine). The plan's check: "No mutable globals outside a short allowlist (the device registry)" (1663). Cost: ~8 lines; the structural cost is the `adopt!` call in 15 verbs that the singleton makes necessary.

**P5. Near-copies of a function for different kinds.** `commands.jl:245-246, 269-277, 294-295` and `api.jl:214-216, 225-226, 255-263, 278` (seven predicates as 2-3 methods each, ~20 methods for seven booleans), `commands.jl:30-53` (plain/indexed/indirect draws as three verbs), `api.jl:85-98` and `158-188` (trace × indirect × anyhit as six verbs), `pipeline.jl:54-55` (`TrianglePipeline`/`LinePipeline` instead of the `topology` kwarg; zero users in the tree), `accel.jl:60-65` (three positional constructors), `src/raytracing/pipeline.jl:42-44` (`_normalise_chit` for a function, a tuple and a vector), `ext:1289-1290` (two `closerun!` methods because core's `::Immediate` method makes a `Union` ambiguous, explained at 1283-1287), `ext:1219` and `1330` ("Same body … separate method because they are separate verbs"), `instances.jl:19-25` (host records) beside `41-47` (device records). Cost: ~150 lines.

**P6. Hand-rolled vector/matrix math.** `instances.jl:71` `Mat3x4f(build_4x3(quat_to_rot3x3(q), radius, p))`: a quaternion to a 9-tuple (`src/geometry/transform.jl:70-81`), a 9-tuple times a scalar plus a `Point3f` into a 12-tuple (`:88-92`), then a 4x3 `SMatrix`; `Mat3x4f` itself is `SMatrix{4,3,Float32,12}` (`transform.jl:41`), a row-major 3x4 stored as a column-major 4x3 so the bytes match `VkTransformMatrixKHR` (`transform.jl:126-128` "the byte-equivalence trick"), which means `m[i,j]` on a `Mat3x4f` is the transpose of the matrix it represents; `instances.jl:23` and `src/metal/hwtlas.jl:539` call `mat4_to_vk_transform` (`transform.jl:121-133`, twelve `Float32(m[i,j])` conversions) while `src/vulkan/raytracing/hwtlas.jl:544` calls `Raycore.mat4_to_mat3x4`; `accel.jl:237` returns `SVector{3,Float32}` where the codebase's type is `Vec3f`. `quat_to_rot3x3`, `build_4x3`, `build_4x3_pervec` and `Mat3x4f` are exported (`src/Mantle.jl:393`), so the tuple arithmetic is public API. The target's `push!(tlas, blas, positions; scale, rotation)` (2.6) computes the transform on the device from GeometryBasics types. Cost: ~15 lines in the audited files, ~70 in `geometry/transform.jl`, plus the Metal transpose (`src/metal/raytracing.jl:22-44`).

**P7. Manual loops that are a reduce/find or a map for its side effect.** `batches.jl:76-78` (reindex every `Dict` entry after a `deleteat!`; the positions-in-a-Dict design makes `delete!` O(n)), `commands.jl:205-207` (`for pp in pl.passes, t in pp.pre; t.resource == id && (rest = t.to)`: a `findlast`), `ext:1112-1114` (execute every `Call` of a piece on the host), `ext:1122-1129` (`map(arg -> AMDGPU.rocconvert(arg, d.stream), cd.args)` with the result discarded; the comment at 1115-1121 says the map exists for the transfer it triggers). Cost: ~15 lines.

**P8. Forwarding layers.** `commands.jl:246,276-277,295`, `api.jl:216,256` (`f(dev::Device) = f(backend(dev))`, with a 28-line justification at 226-243), `resources.jl:169-170,185,192-194` (target wrappers to the thing they wrap), `record.jl:62-63` (`viewport!` → `setviewport!`), `record.jl:85-89` (`draw!` → `record_draw!`), `record.jl:97` (`bindings!` → `use_bindings!`), `commands.jl:177-191` (`readback_target(plan, t)` → `readback_target(dev, t, rest)`), `mesh.jl:57` (`meshconfig` → `stageconfig`), `stages.jl:198` (`stageinputs` → `stageoutputs`, with a docstring saying it returns the same thing), `ext:273` (`Device(::ROCBackend)` → `Device(ROCmAPI())`), `ext:327` (`awaitwrites` → `waitidle`), `ext:1428` (`caps(b)` → `caps(Device(b))`), `ext:1289-1290` → `submitrun!`. Cost: ~45 lines of forwarders, ~130 with the docstrings each one carries.

**P9. `_`-prefixed functions (owner's rule: visibility is by export).** `lowering.jl:261,270` `_lower`; `src/raytracing/pipeline.jl:42-44,49` `_normalise_chit`. In the compared backends: `src/vulkan/raytracing/hwtlas.jl:664` `_reuse_or_alloc`, `:735` `_compact_blas_list!`, `src/metal/hwtlas.jl:137` `_unpack`, `:205` `_unpack_procedural`. Cost: 2 functions in the audited files.

**P10. Compat shims and feature flags.** `commands.jl:310-326` and `427-437` ("The Vulkan backend does NOT go through these … These exist for backends that record a pass per draw"), `commands.jl:399-404` (kwargs "so the graph … calls this exactly as it did"), `src/raytracing/pipeline.jl:34-37` (`chit_miss_take_args`: "Default `false` keeps the legacy contract"), `accel.jl:57-59` ("adding a field did not touch a single existing construction site"), `lowering.jl:36-40` (a vertex body may declare `VertexIndex` or not; "a body that declares BOTH spellings runs natively where the geometry stage exists and lowers where it does not"), `ext:764,774,808,1359,1377` (five booleans: `recordsplans`, `recordable`, `runscalls`, `patchable`, `supportspredicate`; each a capability core branches on), `ext:192` (`blocking::Bool`, a field a caller toggles), `ext:1058-1062` (a signature-change history). The plan forbids "keeping an old path next to its replacement … a compatibility shim; 'for now'" (1635). Cost: ~90 lines.

**P11. Comments that narrate history instead of stating the contract.** `commands.jl:1-10, 216-243, 289-296, 399-404, 427-437, 452-457, 476-481`; `record.jl:1-38, 113-116, 289-296`; `lowering.jl:3-15, 26-40`; `resources.jl:1-16, 64-71, 78-80, 117-122, 164-168, 180-194, 229-236`; `stages.jl:1-28`; `state.jl:1-16, 126-131`; `pipeline.jl:1-7, 24-27`; `mesh.jl:1-12`; `builtins.jl:1-10` (the whole file is a comment about what is not there); `api.jl:1-16, 43-45, 61-66, 239-253, 280-290`; `accel.jl:1-5, 8-9` (core's docstring says "GPU-adapted form of `VulkanTLAS`"), `67-72, 114-124`; `batches.jl:1-12, 66-69`; `instances.jl:1-9, 50-56`; `src/raytracing/pipeline.jl:1-16, 20-29`; `ext:1-131` (a 131-line benchmark narrative with medians of fifteen runs), `150-176, 291-322, 417-435, 617-635, 699-725, 1050-1076, 1089-1111, 1162-1168, 1227-1247, 1430-1454, 1456-1462`. The plan's comment rules (1672) and the memory note "docs state the target only" rule this out. Estimate: ~1,650 of 4,245 lines (commands ~230, record ~120, lowering ~90, resources ~110, stages ~70, state ~25, pipeline ~35, mesh ~20, builtins 10, api ~150, accel ~110, batches ~35, instances ~30, rt pipeline ~25, ext ~600). This is the largest single cost.

**P12. Functions over ~100 lines.** None in the audited files. Longest: `ext:826-881` (56), `ext:1078-1137` (60 with comments), `lowering.jl:256-304` (49 across two functions). Docstrings over 100 lines: `ext:1-131`.

**P13. Keyword-argument explosions.** `record.jl:298-303` (`draw!` with `args`, `frag_args`, `instances`, `clear_color`, `depth_clear`, `indices`, `bindings`: seven keywords where the target has two plus a separate `clear!`), `commands.jl:329-330` (`compile_draw` with six positionals plus `bindings`), `ext:950-952` (`native_conv2d_dispatch!` with nine keywords). The `GraphicsPipeline` constructor's nine keywords (`pipeline.jl:44-49`) match api.md. Cost: ~20 lines, but each keyword is a branch in a body.

**P14. Parallel data structures kept in sync by hand.** `batches.jl:24-25` (`batches::Vector{B}` and `index::Dict{TLASHandle,Int}` of positions; `71-80` reindexes), `accel.jl:30-32,54` (`triangles`, `offsets`, `instances` indexed by the same `inst_id`, with `NOTRIANGLES` at 88 as the "no entry" sentinel), `ext:553-555` (`args` and `raw` parallel tuples, zipped at 1252), `ext:188-189` (`next`/`completed` counters standing in for the stream's own progress), `ext:184` and `217` (the hipBLASLt handle on the device and the library handle in a global). Cost: ~40 lines plus the invariants each pair carries.

**P15. Lowering steps and command representations.** Two files are called "lowering" and do different things: `src/graphics/lowering.jl` rewrites a pipeline description (`GraphicsPipeline` with a `GeometryShader` → `MeshPipeline`, 256-304); `src/vulkan/lowering.jl` maps usage types to `VkPipelineStageFlags2`/`VkAccessFlags2`/`VkImageLayout` (47-178), decides `needs_transition` (205-225) and holds the format table `vkformat` (227-243). The first survives as core (section 4); the second is the Vulkan barrier helper and `imageformat` of api.md (A: `stages`, `access`, `layout` "internal to the device type's barrier helper"; `needs_transition` goes).

Representations of one draw between the user's call and the driver today, on Vulkan: (1) the `Draw` call stored in `Pass` (`src/graph/types.jl:194`), (2) `CompiledDraw` (`src/graph/types.jl:490`), (3) the backend's compiled pair `VulkanDraw(compiled, shader)` (`src/vulkan/graphics/record.jl:118`) or `MetalCompiledDraw`, (4) the `Emitter`-level verb `draw_in_pass!(e, pipe, n; push_bda)` (`src/vulkan/graphics/pipeline.jl:541`), then `VK.cmd_draw`. The hand path has its own top two: `draw!(p::PassRecorder, compiled, args, count)` (`record.jl:85`) → `record_draw!(VulkanPassHandle, VulkanDraw, …)` (`src/vulkan/graphics/record.jl:170`) → (4). Four representations on the graph path, three on the hand path, two paths that agree only at (4). The target keeps two core representations and one verb: `DrawNode` → `CompiledDraw` (the device type's compiled pipeline inside it, from `compiledraw`) → `emitdraw!(b, node, cmd)` → driver (api.md 3, 4.5).

**P16. Mutable value types with `Any` fields.** `src/raytracing/pipeline.jl:18-37` (`mutable struct RayTracingPipeline` with `raygen_func::Any`, `miss_func::Any`, `anyhit_func::Any`, `closesthit_funcs::Tuple`); its header (1-16) says it is "exactly what `GraphicsPipeline` next door is", but `GraphicsPipeline` is an immutable parametric struct (`pipeline.jl:29-40`). api.md 2.6 says a `RayTracingPipeline` is a value keyed on its shaders, which an `Any`-typed mutable struct cannot be without hashing its fields. `accel.jl:131` `preserves::Vector{Any}`. Cost: no lines; every field read is a dynamic dispatch, and the cache key for `compiletrace` is not derivable from the type.

**P17. Pointers and unsafe calls where a Julian type would do.** `ext:184` (`blaslt::Ptr{Cvoid}`), `217` (`Ref{Ptr{Cvoid}}`), `220` (`dlsym` per call), `833-836` (four `Ref{Ptr{Cvoid}}` descriptors), `839-878` (twelve `ccall`s to hipBLASLt with integer enum literals `2, 0`, `4`, `2`, `3` commented inline at 838, 841), `845` (`Ptr{Cvoid}(pointer(bias))`), `437-460` (`rocview`: 24 lines computing `convert(Ptr{T}, mg.mem) + byteoffset` to check which unit AMDGPU.jl's `ROCArray.offset` is in, because two versions document it differently), `1255` (`pointer(now) == pointer(a)`), `1342` (`UInt64(convert(Ptr{UInt8}, mg.mem))`). Cost: ~110 lines in the extension. Whether a Julia wrapper for hipBLASLt exists is not checked here (a guess: none in AMDGPU.jl, which is why the FFI was written).

**P18. A backend that writes into the user's graph.** `ext:959,966` (allocates transients in `g`), `ext:967-968,1033` (declares passes into `g`), `ext:1112-1114` (runs them on the host). The target's split (plan 1381-1409): a device type contributes `candidates(dev, f)`; core declares. Cost: ~40 lines of hooks; the structural cost is that three `native_*_dispatch!` entry points exist in the vocabulary for Metal and ROCm to fill (A lists all three as "goes").

**P19. Host-side work that is a store or a kernel in the target.** `instances.jl:19-25` builds a host `Vector{InstanceRecord}` per `push!` (one `mat4_to_vk_transform` per element), which each backend then uploads; `batches.jl` keeps the instance table on the host; `commands.jl:201-209` walks a plan's passes on the host to learn an image's layout (`restingusage`), which `barriers!` decides in the target.

---

## 4. What the design docs miss

Things the audited files do that api.md and the plan have no replacement for, and rows that map the code wrongly.

1. **Geometry stage on a device without one (`lowering.jl`).** RayMakie's scatter and lines overlays are `GeometryShader` pipelines (`dev/Makie/RayMakie/src`: 4 `GeometryShader`, 2 `lower_geometry_to_mesh`, 11 `VertexIndex`; `overlay_rendering.jl:158-185` branches on `supports_geometry_stage`/`supports_mesh_pipeline`). Metal has no geometry stage, so `src/metal/mesh.jl:567-589` lowers through `lower_geometry_to_mesh` at `compile_draw`. api.md 2.6b lists `GeometryShader` as a stage and `caps` has a flag for it (A `supports_geometry_stage`), but neither doc says what core does with a geometry pipeline on a device whose flag is false. Needed: a sentence in 2.6b that compile of a render node applies the geometry-to-mesh rewrite in core before `compiledraw` when `caps(dev).geometrystage` is false, and the `VertexIndex` contract on vertex bodies (`lowering.jl:36-40, 306-339`) stated once. Also an open point: the lowering runs one thread per threadgroup (`lowering.jl:17-57`), which the file itself says to measure before widening.

2. **The procedural (AABB) traversal protocol.** `accel.jl:135-237` is the contract Hikari's `FEMMaterial` is written against (`dev/Hikari/src`: `procedural_candidate` 5, `candidate_object_ray` 4, `commit_intersection!` 3, `procedural_commit` 2). api.md 2.6 has `BLAS(dev, vertices, faces)` only; there is no `BLAS` over AABBs, no `procedural` field on a TLAS range, and no device-side intrinsics. `caps` has `supports_procedural_traversal` (A), so the capability is foreseen but not the API. Metal's answer is a `[[visible]]` function table (`src/metal/procedural.jl:282-330`), Vulkan's an inline ray query (`src/vulkan/raytracing/hwtlas.jl:951-1000`); the plan's "ranges and shaders" paragraph (1050-1057) says nothing about either. Needed: `BLAS(dev, aabbs)`, a per-range payload, and the three intrinsics moved to KernelInterface.

3. **The kernel-side accel type Hikari takes.** Appendix A says `AdaptedAccel` "goes with Metal's HWTLAS bookkeeping". It is core's (`accel.jl:29-55`), both backends construct it (`src/vulkan/raytracing/hwtlas.jl:213-231`, `src/metal/hwtlas.jl:332-335`), and Hikari names it 13 times (`rt-pipeline.jl:590` `accel_t::Mantle.AdaptedAccel`) and calls `hittriangle` (2). The row is wrong about what it is, and no row names what replaces it: the device form of `TLAS{P}` (cell address, range table, per-range BLAS geometry cells) and the lookup a hit shader does instead of `hittriangle` (2.6 says "reached through the range table", the function is unnamed).

4. **Hikari's `RayTracingPipeline` fields.** `payload_type::Symbol` and `chit_miss_take_args::Bool` (`src/raytracing/pipeline.jl:26-37`) are set by Hikari (`rt-pipeline.jl:569-570`) and read by Vulkan's shader compile (`src/vulkan/raytracing/shaders.jl:203-221`). api.md 2.6 lists four keywords. Either the two are part of the value (and the cache key) or the doc must say how a closest-hit shader receives the raygen's arguments without them.

5. **`trace_closest_hits!` has a consumer.** Hikari calls it twice; A deletes it. The replacement, a ray-query from an ordinary kernel against the TLAS cell, is implied by 2.6 ("traces and ray queries take the TLAS by address") but no device-side function is named. Also `set_anyhit_pipeline!` (A: "any-hit is a field") today builds a second pipeline on the accel (`src/vulkan/raytracing/raycore_compat.jl:89-107`) for the ray-query path; the row covers the field, not that path.

6. **Draw pieces and batches.** 2.6b's "a group of draws" covers pieces and `sort!`. Not covered: the per-draw `sbt_offset` a batch carries (`src/vulkan/raytracing/hwtlas.jl:52-54`, `push!` keyword at 406) for per-material closest-hit groups; 2.6 says the SBT layout for many hit groups is open (section 10) but 2.6b and the RayMakie section assume per-plot materials. A range needs a material/hit-group index in its `RangeEntry`.

7. **Instance update paths.** Today: `update_transforms!(tlas, handle, transforms::DeviceArray)` on Vulkan queues the array and applies it in the next `sync!` by a kernel (`src/vulkan/raytracing/hwtlas.jl:521-532, 706-721`); Metal launches the kernel immediately (`src/metal/hwtlas.jl:526-535`); `write_grain_instances_kernel` (`instances.jl:66-76`) writes two batches' records from positions and quaternions in one kernel. api.md 2.6 has `tlas[h] = t` (one store + refit) and `push!(tlas, blas, positions; scale, rotation)` (transform computed on the device). Missing: a per-frame update of thousands of transforms that a *kernel* writes (the grain case; RayMakie `plots/mesh.jl:1121` references `update_instance_transforms!`) with no host store, i.e. a range whose source is an array a run writes. The plan's "A write reaches the structures built from it" (1078-1097) covers it through readers; 2.6's table does not say a range's `transforms` array may be kernel-written and that the refit is then pended by the run. One sentence closes it.

8. **BLAS reuse across TLAS.** Covered (2.6 "Shared by every range over the same geometry arrays"). Today's `blas_list`/`_compact_blas_list!` per TLAS (`src/vulkan/raytracing/hwtlas.jl:735-768`) goes.

9. **Pipeline state coverage.** `state.jl:46-75` `NoWrite` and `PerAttachment` (RayMakie uses both, 2 each) and `DepthLessEqReadOnly` (`state.jl:112`, RayMakie 2) are finer than 2.6b's `blend, cull, depth`; the doc should state the blend is per attachment and the depth mode has a read-only form, or they are lost in the rewrite. `attachmentblend` (`state.jl:67-75`) with its integer-attachment rule is core logic with no doc line.

10. **Vertex layouts.** None exist today: vertex stages read attributes through buffer-device-address loads (`src/vulkan/lowering.jl:36-47`), which is what 2.6b's `attribute(a, i)` keeps. Nothing missing.

11. **Uniforms and push constants.** Today a draw's arguments are packed into a block and its address pushed (`src/vulkan/graphics/record.jl:174` `pack_gfx_args_bda`; `commands.jl:412-414`). 2.6b replaces this with slots and `GPURef`. `DrawIndirectCommand` (`pipeline.jl:109-114`) is the layout a culling kernel writes today; 2.6b's `count` as a `launchrange` implies the layout is core's, unnamed. One row needed.

12. **The mesh shader path.** `MeshPipeline` is in 2.6b; `ObjectShader` is listed. Not stated: that a mesh draw's `count` is a workgroup count (`record.jl:286-287`), and whether `object` dispatches replace it. Metal's `MetalCompiledMeshDraw` and Vulkan's `draw_mesh_in_pass!` (`src/vulkan/graphics/record.jl:181`) become one `emitdraw!` with `cmd.kind`; fine.

13. **`ExternalImage`.** `resources.jl:238-246` declares it; VideoEditor constructs one (`dev/VideoEditor/src/glbridge.jl:243` `lavamod.ExternalImage(W, H)`, via the Vulkan module) for GL interop; Vulkan names it 12 times. api.md has no row; recording-plan.md mentions external memory only under Memory (1232) and `VK_EXT_external_memory_host` in open decisions (2182). A consumer in tree depends on it; it needs a row (keep as a Vulkan-only `Image` source, or delete with the GL bridge).

14. **Appendix A rows that map wrongly or are missing.**
    - `transition_image!`: "(no implementation on either device type)". Vulkan implements the name at `src/vulkan/graphics/pipeline.jl:354` (signature `(cmd, image::VK.Image, …)`, ten call sites: `api.jl:315,336`, `pipeline.jl:441,458`, `framebuffer.jl:263,316,434,451`, `textures.jl:146,161`). What has no implementation is core's `(image, from, to)` signature (`commands.jl:112-121`). RayMakie imports the name (`RayMakie.jl:40`) and does not call it.
    - `AdaptedAccel`: see item 3.
    - Five `BACKEND_VOCABULARY` names have no Appendix A row: `supports_hwtlas` (`src/graph/backend.jl:862`), `build_blas` (963), `build_blas_aabb` (815), `readback_target` (947), `upload_texture_data!` (815). The appendix claims to cover the vocabulary's 189 names.
    - `copy_target!`: the row says "copies between images and arrays"; Metal's body takes a `region` keyword (`src/metal/images.jl:158`) that 2.6b's `copyto!` does not mention (a sub-rectangle copy).
    - `argidentity`: "goes: as argbytes" is right; note that only Vulkan implements it (`src/vulkan/array/lavaarray.jl:325`), so Metal plans never detected a move through it.

15. **ROCm: what the HIP device type must provide beyond memory and queues.**
    - Kernel compilation: `compilekernel` through `AMDGPU.hipfunction` with the converted argument types (`ext:727-742`), `kernelinfo` from `AMDGPU.launch_configuration` (`ext:731`), and `config(dev, f)` from `AMDGPU.Compiler.compiler_config(d.dev)` (`ext:284`); `kernelinterpreter` (`ext:283-287`) stays per api.md.
    - A `Builder` over `hipGraph` nodes: the plan's table (1417) says "built node by node", but no api.md verb row names `hipGraphAddKernelNode`/`hipGraphAddMemcpyNode` as the one call site, as 8 does for Vulkan. The `caps` inline-store limit is 0 on HIP (api.md 4.5 `emitstore!`).
    - AMDGPU.jl interop: the vendor view (`rocview`, `ext:437-460`) depends on `ROCArray.offset`'s unit, documented differently in AMDGPU 2.2 and 2.8 (`ext:426-435`); the target's "zero-copy view not owning the region" (4.6) needs `GPUArrays.DataRef` with a no-op destructor (`ext:452`), which is what exists. `AMDGPU.Managed` stream tracking must not be used (plan 10 deletes it); `ext:360-366` returns a `Managed` from `rawalloc`, so `rawalloc` changes to a raw `HIPBuffer`.
    - Interop with Vulkan memory: nothing in the extension, nothing in the docs. The memory note `vulkan-cuda-hip-interop-measured` says HIP shares memory but not semaphores; api.md 4.2 `submitnative!` "waits on tokens of any channel, expressed natively" cannot be met across a Vulkan channel and a HIP stream without an external-semaphore import. Not a verb today; it is an open decision that the docs do not list.
    - Library candidates: hipBLASLt (`ext:813-881`), MIOpen (`ext:893-917`), rocBLAS batched (`ext:985-1009`) become `candidates(dev, f)` entries with `argument_usage` declarations (`ext:817-819, 902-903, 991-994` already have them). The `Widen` pass (`ext:920-922`) is a core elementwise op, not a candidate.
    - The host-side warm-up before capture (`ext:1089-1114`) goes with capture; with a node-built graph the library's first call still creates its handle lazily, so `candidates` must initialise (`rocblas_create_handle`) at declaration, not at record. One line in the plan would say so.

---

## 5. Minimal form

Estimates for the same functionality under api.md and the plan. Lines are code with contract comments, not narrative. Each is a guess from the current bodies minus the duplication listed above.

### Core graphics (backend independent)

| part | lines | what it holds |
|---|---|---|
| Pipeline values: `GraphicsPipeline`, `MeshPipeline`, `RayTracingPipeline` (immutable, parametric), stage structs, `BlendMode`/`CullFace`/`DepthMode`, `fragmentinputs` once | 250 | today `stages.jl` 226 + `state.jl` 133 + `pipeline.jl` 115 + `mesh.jl` 79 + `src/raytracing/pipeline.jl` 60 = 613 |
| Geometry-to-mesh rewrite (`GeometryAsMesh`, `splatcall`, `transposeprimitive`, the rewrite, the `VertexIndex` check) | 170 | today `lowering.jl` 339 |
| `DrawNode`, `ClearNode`, render-pass formation at compile (grouping by target set, load/store from liveness, pieces per group, the scene-rectangle uniform), `when!` draws | 350 | new; replaces `commands.jl`/`record.jl` 838 and the graph-side `Pass.targets/loads/depth` fields |
| Eager `draw!(dev, target, …)` and `clear!` over the same compile + one `Builder` | 60 | today `record.jl:146-321` |
| `Image` resource, texture and sampler tables with free lists, `tableentry!`/`samplerentry!` call sites, `imageformat` use | 220 | today `resources.jl` 246 + per-backend `Texture2D`/`Sampler`/`Framebuffer` types |
| `compiledraw`/`compiletrace` cache keys (stage functions, device argtypes, fixed state, formats) | 60 | today per backend |
| **graphics core total** | **~1,110** | |

### Core raytracing

| part | lines | what it holds |
|---|---|---|
| `TLAS{P}`, `BLAS`, `InstanceRange`, `RangeEntry`, capacity doubling, `builtcount`, slot free-list, `push!`/`delete!`/`tlas[h]=`/`build!`/`refit!` as pending `BuildAccel` changes, coalescing | 350 | today `batches.jl` 118 + `instances.jl` 76 + `api.jl` 290 + the two backends' ~1,100 |
| offsets kernel (prefix sum over range lengths) and records kernel (writes `instancerecordtype(dev)` records, mask 0 past the total) | 80 | today `vkinstances_kernel!`/`mtlinstances_kernel!` + `update_instance_records_kernel!` |
| `BuildRecording` and its re-record marks | 80 | today `AccelBuildContext` + per-backend build submission |
| `TraceNode`, `trace!(g,…)`, `shaders`, `shadertouches` use | 50 | today `api.jl:100-156` |
| device-side: TLAS by cell, range-table lookup for hit data, procedural payload protocol (intrinsics in KernelInterface) | 60 | today `accel.jl` 237 + `hittriangle` |
| `BLAS` over AABBs (section 4, item 2) | 30 | today `build_blas_aabb` per backend |
| **raytracing core total** | **~650** | |

Together ~1,760 lines of core for what today is 2,767 lines of core declarations (my files) plus roughly 3,700 lines of Vulkan and Metal graphics/raytracing bodies that re-implement the bookkeeping (section 2.1, D1-D9). The per-device-type remainder for graphics and raytracing is the verbs: `compiledraw`, `compiletrace`, `beginrender!`/`emitpieces!`/`endrender!`/`recordpiece`/`emitdraw!`/`bindindices!`, `emittrace!`, `emitbuild!`, `accelsizes`, `placeaccel`, `acceladdress`, `instancerecordtype`, `countsource`, `tableentry!`, `samplerentry!`, `imageformat`: roughly 600-900 lines each on Vulkan and Metal (a guess from today's `src/vulkan/graphics/pipeline.jl` 790 and `src/metal/graphics.jl` 1623, which include the driver-object plumbing that stays).

### The HIP device type, verbs only

| verb group (api.md section) | lines | notes |
|---|---|---|
| 4.1 `hardwarekeys`, `T(key)`, `caps`, `config`, `countsource`, `queuekinds`, `budget` | 90 | today `ext:148-289, 1379-1428` (~190) |
| 4.2 `submitnative!` (`hipGraphLaunch` + `hipStreamWaitValue64`/`hipStreamWriteValue64`), `passed`, `waitfor` (GC-safe), `recycle!`, `bindpages!`/`unbindpages!` (`hipMemMap`) | 110 | today `ext:387-413, 1268-1330` (~90), without VMM |
| 4.3 `rawalloc` (`hipMemCreate`), `rawfree`, `createreserve` (`hipMemAddressReserve`) | 50 | today `ext:329-385` (~57) |
| 4.4 `compilekernel`, `kernelinfo`, `candidates` (rocBLAS, hipBLASLt, MIOpen wrappers with `argument_usage`) | 60 + 180 | today `ext:597-742` (~146) + `810-1048` (~240) |
| 4.5 `Builder` over `hipGraph`, `emitdispatch!` (kernel node), `emitcopy!` (memcpy node), `emitstore!` (limit 0), `emitlibrary!`, `emitloop!` (predication: open), `finish` (`hipGraphInstantiate`) | 160 | today `ext:1050-1266` (~216) capture-based |
| 4.6 `devicearg` (`rocconvert`), vendor view | 40 | today `ext:415-494` (~80) |
| **HIP device type total** | **~690** (of which ~180 are library candidates) | today 1,478 |

The 690 excludes windows (none on HIP) and assumes the `hipGraph` node API wraps as cleanly as AMDGPU.jl's `HIP` module exposes it today (`ext:1169-1180` already calls `hipGraphGetNodes`/`hipGraphNodeGetType` directly, so the raw calls are available).

---

## Rule violations (owner's Julia rules), consolidated

- `_name` functions: `lowering.jl:261,270` (`_lower`); `src/raytracing/pipeline.jl:42-44` (`_normalise_chit`).
- Globals that should be state on an object: `ext:217` (`HIPBLASLT`), `ext:252` (`ROCM_DEVICE`); a module constant used once: `record.jl:212` (`BLIT_PIPELINE`); a sentinel constant standing for a missing entry: `accel.jl:88` (`NOTRIANGLES`).
- `try … catch` without rethrow: none. Silent degradation: `ext:1466-1468`.
- Name variants instead of dispatch: `pipeline.jl:54-55`; `api.jl:85-98, 158-188`; `commands.jl:30-53`; `src/raytracing/pipeline.jl:42-44`; `accel.jl:60-65`.
- Vector/matrix math by hand: `instances.jl:23, 71`; `accel.jl:237` (`SVector`); `src/geometry/transform.jl:41, 70-133` (reached from `instances.jl`).
- Pointers/unsafe where a Julian type works: `ext:184, 217-227, 437-460, 833-879, 1255, 1342`.
- Defensive code on known types: `batches.jl:54-60, 71-80, 105-118`; `ext:1038-1047, 953-958, 1025-1031`; `ext:193`; the `isa` chains at `ext:641-644, 669, 1083, 1113, 1123, 1251-1255`.
- Mutable/`Any` where a value is meant: `src/raytracing/pipeline.jl:18-37`; `accel.jl:131`.
