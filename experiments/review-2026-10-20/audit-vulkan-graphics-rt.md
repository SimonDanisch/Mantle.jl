# Audit: src/vulkan/graphics and src/vulkan/raytracing against docs/api.md

Scope: every line of `src/vulkan/graphics/{pipeline,framebuffer,window,api,textures,record}.jl`
(2 659 lines) and `src/vulkan/raytracing/{acceleration,hwtlas,pipeline,raycore_compat,shaders,instance_record}.jl`
(3 699 lines); 6 358 lines in total. Targets read first: `docs/api.md` sections 2.6, 2.6b, 2.7,
4 and Appendix A; `docs/recording-plan.md` "Windows and present", "Acceleration structures and
textures", "Graphics". Counts of call sites come from `grep` over `src`, `test`, `ext` and the
consumers in `dev/Hikari`, `dev/Makie/RayMakie`, `dev/Raycore`. No Julia was run; anything that
would need a run is marked **guess**.

Bucket key for section 1: (a) backend independent, belongs in core per api.md; (b) a device-type
verb; (c) belongs nowhere: dead, superseded, compat shim, test-only, debug-only, or a duplicate of
something that already exists. Line estimates are to the nearest ~5 lines.

## 1. Inventory

| file | lines | what it does | (a) core | (b) verb | (c) nowhere | api.md verb / Appendix A row |
|---|---|---|---|---|---|---|
| graphics/pipeline.jl | 790 | `VkGraphicsPipeline` from SPIR-V plus fixed state (L21-227, L260-342); dynamic-rendering pass begin/end with its own layout transitions (L354-532, L741); draw / indexed / indirect / mesh draw recording (L541-718); viewport+scissor (L730-773); descriptor-set bind (L776-784) | 15 | 300 | 475 | `compiledraw` (4.4) for L21-342; `beginrender!`/`endrender!` (4.5) for L402-532, L741; `emitdraw!`/`bindindices!` for L541-718; `recordpiece` sets viewport for L757-773. Appendix A: `begin_pass!`, `end_pass!`, the `draw_in_pass!` family, `set_viewport!`, `use_bindings!`, `transition_image!` all "go" |
| graphics/framebuffer.jl | 514 | offscreen colour+depth target that allocates its own `VkDeviceMemory` (L12-106); image/buffer memory helpers (L108-215); image→buffer readback for a framebuffer (L246-289), a device copy of the same (L307-336), a bare copy (L352-375), a window readback through an extra acquire/present (L387-508) | 36 | 93 | 385 | `placeimage` (4.3) for L40-62, L115-118; `rawalloc` is "the only call site of vkAllocateMemory" so L126-159 is a duplicate; `emitcopy!` (4.5) for L352-375; `Array(img)` / `Array(win)` (2.6b) for L246-289, L387-508. Appendix A: `Framebuffer`, `readback_framebuffer`, `readback_window` go |
| graphics/window.jl | 475 | GLFW window + surface (L72-114); swapchain creation and re-creation with per-image semaphores and per-slot fences (L121-261); `target_*` accessors (L286-299); framebuffer-size polling (L321-328); acquire with fence wait and resize retry (L336-378); present with resize on OUT_OF_DATE (L386-416); resize/close with `device_wait_idle` (L423-473) | 36 | 138 | 301 | window open verb, `createswapchain`, `acquirenext`, `surfacesize` (4.7); present is part of `submitnative!` (4.2). Appendix A: `target_extent/format/image/view` go, `acquire_next_image!` → `acquirenext`, `present_frame!` goes |
| graphics/api.jl | 403 | stage compile + pipeline cache keyed on shaders, state and formats, twice (classic L46-107, mesh L116-157); stage wrappers (L179-237); argument packing to a BDA, twice (L256-304); present transitions (L313-343); submit+present (L364-381); `supports_*` (L396-403) | 0 | 104 | 299 | `compiledraw` (4.4) for L31-237; core's `bindings!` lays out the argument block so L256-304 is a duplicate (Appendix A `argbytes`/`argsize` go); `submitnative!` for L364-381; `caps(dev)` fields for L396-403 |
| graphics/textures.jl | 286 | texture types with their own `VkDeviceMemory` (L6-31, L86-127); sampler from Symbols (L34-75); staged upload with two transitions (L140-214); a second Julia→VkFormat table (L218-226); a per-call descriptor pool/set (L230-282) | 10 | 63 | 213 | `placeimage` for L86-127; `samplerentry!` (4.8) for L48-75; `emitcopy!` for L140-172; stores `img[:] = data` (2.6b) for L175-214; `imageformat` (4.8) for L218-226; `tableentry!` for L240-282. Appendix A: `Texture2D`, `Sampler`, `bind_textures` go; `upload_texture_data!` has no row |
| graphics/record.jl | 191 | eager render pass over a one-shot: attachment wrapper (L25-39), a third format table (L57-69), `compile_draw` (L90-112), pass handle and begin/end/draw/viewport/bindings (L118-191) | 0 | 25 | 166 | `compiledraw` for L90-112; `imageformat` for L57-69; 2.6b "an eager draw!(dev, …) runs now, as its own pass" through the Builder verbs, so L118-191 go. Appendix A: `compile_draw`, `begin_render_pass!`, `record_draw!`, `end_render_pass!`, `setviewport!`, `colorimage`, `depthimage`, `currentimage` go |
| raytracing/acceleration.jl | 1614 | BLAS/TLAS objects with finalizers and per-TLAS descriptor cache (L15-166); BLAS build/refit (L203-373), AABB BLAS (L391-453), TLAS build/refit (L461-690); hand-packed instance records (L703-733); a host-visible AS input pool (L741-776); hand-packed Vulkan AS structs and raw `ccall`s (L794-975); one-shot build submission (L997-1042); build command (L1049-1133); pooled builders and Raycore bridges (L1146-1614) | 105 | 190 | 1320 | `accelsizes` (L914-975), `placeaccel` (L252-256 etc.), `acceladdress` (L275-276), `emitbuild!` (L1076-1133), `countsource` (4.1/4.3/4.5). Appendix A: `build_accel!`, `build_blas`, `refit_tlas!` go; `build_blas_aabb` is in `BACKEND_VOCABULARY` (`src/graph/backend.jl` L815) but has **no Appendix A row** |
| raytracing/hwtlas.jl | 1084 | `VulkanTLAS`: batches, five `push!` forms, delete/hide/update (L42-651); rebuild-vs-refit decision, records kernel, BLAS-list compaction, triangle concatenation (L653-921); inline ray-query traversal (L923-1019); descriptor set per (layout, TLAS) (L1021-1070); `supports_*` (L1072-1084) | 304 | 88 | 676 | core `TLAS`, `InstanceRange`, `RangeEntry`, `push!`, `tlas[h] = t`, `delete!`, `build!`/`refit!` (2.6); records kernel with `instancerecordtype(dev)` (plan, "Acceleration structures"); ray-query intrinsics stay device-type. Appendix A: `AdaptedAccel` "goes with Metal's HWTLAS bookkeeping" (mis-mapped, see 4) |
| raytracing/pipeline.jl | 458 | `VkRayTracingPipeline` + SBT (L58-236, L377-456); descriptor set cache on the TLAS (L252-282); trace / trace-indirect commands with profiler timestamps (L294-357) | 0 | 200 | 258 | `compiletrace` (4.4) for L58-236, L377-456; `emittrace!` (4.5) for L294-357; 2.6 "traces take the TLAS by address through its cell" removes L252-282 |
| raytracing/raycore_compat.jl | 247 | `HardwareAccel` (a `LavaTLAS` + CPU triangle data + a default RT pipeline, L29-93); any-hit pipeline slot (L101-110); `trace_closest_hits!` family (L126-175); built-in raygen/closest-hit/miss writing a 7-float payload (L179-247) | 40 | 0 | 207 | Appendix A: `trace_closest_hits_*` "go: the immediate trace_* verbs", `set_anyhit_pipeline!` → a field of `RayTracingPipeline`. The default closest-hit-record pipeline (L179-247) has no home in api.md (see 4) |
| raytracing/shaders.jl | 240 | eager `trace_rays!`/`trace_rays_indirect!` with argument packing (L44-75, L127-165); pipeline cache keyed on description + world age (L26-28, L96-114); stage compile (L192-237); indirect prepare kernel (L169-183) | 23 | 48 | 169 | `compiletrace` for L96-114, L202-237; `trace!(dev, …)` (2.4) for L44-75, L127-165; the prepare node is core's. Appendix A: `trace_rays!`, `trace_rays_indirect!` go |
| raytracing/instance_record.jl | 56 | 64-byte plain-bits mirror of `VkAccelerationStructureInstanceKHR` | 0 | 30 | 26 | `instancerecordtype(dev)` (4.1) |
| **total** | **6358** | | **~570** | **~1280** | **~4500** | |

Reading of the totals: about 9 % of these files is backend-independent bookkeeping written against
Vulkan types, about 20 % is what the Vulkan verbs need (and that part is itself inflated, section 3),
and about 70 % is dead, duplicated, superseded, or scaffolding around the other two.

## 2. Duplication

### 2.1 Within the audited files

| # | what | side A | side B (and C…) | recoverable |
|---|---|---|---|---|
| D1 | push-constant write (BDA by value or a byte vector) | pipeline.jl L604-617 (`bind_for_draw!`) | pipeline.jl L657-664, L699-712; api.jl L293-296; the runtime already has `push_constants!` (`src/vulkan/runtime/command.jl` L864-870) and `push_constants_bda!` (used by raytracing/pipeline.jl L308, L341) | 35 |
| D2 | image→buffer copy with the transfer transition | framebuffer.jl L263-276 (`readback_framebuffer`) | framebuffer.jl L316-330 (`copy_framebuffer!`), L364-372 (`copy_image_to_buffer!`), L434-447 (`readback_window`); `src/vulkan/graph.jl` L1462-1470 (`readback_target`) | 60 |
| D3 | `VkImage` + full-subresource `VkImageView` | framebuffer.jl L40-51, L54-62 | textures.jl L101-119; window.jl L231-246 | 30 |
| D4 | `VkDeviceMemory` allocation with the device-address flag | framebuffer.jl L126-159 (`device_memory`) | `src/vulkan/runtime/memory.jl` L578-581 (`vk_alloc`); acceleration.jl L741-776 (`create_as_input_pool`); `src/vulkan/graph.jl` L794, L838. Five `vkAllocateMemory` sites; api.md 4.3 allows one | 90 |
| D5 | element type ↔ format tables | `src/vulkan/lowering.jl` L227-242 (`vkformat`, 7 entries) | record.jl L57-69 (`eltypeof`, the inverse, 9 entries); textures.jl L218-226 (`julia_to_vk_format`, 6 entries over a different vocabulary: `NTuple{N,Float32}`/`UInt8` instead of colorants, so a texture and a target disagree about what `RGBA{N0f8}` is); framebuffer.jl L219-233 is a comment about two more that were removed | 40 |
| D6 | "accept a `VK.Format` or a Julia type" | framebuffer.jl L78 | window.jl L82; pipeline.jl L241-250 (`colorformats`, four methods) | 20 |
| D7 | BLAS build body: sizes → storage array → `AccelerationStructureKHR` → scratch → preserves → build → address → debug log → finalizer | acceleration.jl L248-287 (`build_blas_from_addresses`) | L423-452 (`build_blas_aabb`); L1231-1251 + L1259-1266 (`build_blas_pooled`); L1494-1526 (`build_hw_accel_from_tlas`) | 150 |
| D8 | TLAS build body | acceleration.jl L480-517 | L549-588 | 45 |
| D9 | build-geometry-info packing + raw `vkCmdBuildAccelerationStructuresKHR` ccall | acceleration.jl L632-688 (`refit_tlas!`) | L1086-1132 (`build_as_on_gpu_impl`) | 50 |
| D10 | hand-built AS memory barrier | acceleration.jl L352-366 | L645-659, L1094-1104, L1269-1280, L1529-1540, L1009-1020 — six copies; api.md 4.5 wants one barrier helper per device type | 60 |
| D11 | instance record layout | acceleration.jl L703-733 (`pack_as_instance!`, byte offsets by hand) | instance_record.jl L26-50 (`VulkanInstanceRecord`, the same 64 bytes as a struct) | 40 |
| D12 | mesh → (vertices, indices) de-indexing loops | acceleration.jl L1316-1330 | L1391-1404; hwtlas.jl L334-347 | 40 |
| D13 | pooled BLAS builder, passes 1-5 | acceleration.jl L1146-1299 (`build_blas_pooled`) | L1364-1597 (`build_hw_accel_from_tlas`): L1406-1510 is L1162-1252 verbatim. Neither has a caller (section 3, P10) | 400 |
| D14 | "legacy" Symbol-keyword twins of typed functions | acceleration.jl L818-851 (`pack_geometry!` typed) | L855-884 (`geometry_type::Symbol`); `query_as_build_sizes` L914-921 vs L924-930; `build_as_on_gpu` L1049-1060 vs L1063-1072 | 50 |
| D15 | eager trace, direct vs indirect | shaders.jl L50-73 | shaders.jl L142-163 (hold, adapt, pack identical); raytracing/pipeline.jl L294-325 vs L327-357 | 60 |
| D16 | `ensure_compiled!` / `ensure_compiled_with_shader!` for `GraphicsPipeline` vs `MeshPipeline` | api.jl L46-107 | api.jl L116-157; the `_with_shader!` variants compile the front stage a second time (L52 then L63; L122 then L137) | 70 |
| D17 | five `push!` methods for (mesh or BLAS) × (one transform, transforms, records) | hwtlas.jl L404-411 | L413-422, L433-439, L462-474, L489-501 | 60 |
| D18 | six transform-update methods converting `Mat4f`/`Mat3x4f`/host/device | hwtlas.jl L521-532 | L535-538, L542-545, L572-593, L599-602, L611-620 | 70 |
| D19 | SPIR-V → `VkShaderModule` | pipeline.jl L252-256 | raytracing/pipeline.jl L361-365 | 5 |
| D20 | readback region acquire + final `unsafe_copyto!` into a `Matrix` | framebuffer.jl L256-259, L285-288 | L421-424, L471-474 | 15 |

Within the files: about 1 400 lines recoverable (D13 alone is 400 and dead).

### 2.2 Against the core (`src/graphics`, `src/raytracing`)

| # | what | audited side | core side | note |
|---|---|---|---|---|
| C1 | argument packing for a draw | api.jl L256-304 (`pack_gfx_args_bda`, `pack_gfx_args`) and `convert_args` L162 | both call `pack_args_direct!` and `LavaAdaptor`, which already do the work; `BACKEND_VOCABULARY` says packing is core's (`src/graph/backend.jl` L884-887). `pack_gfx_args` has no caller | 55 lines are a forwarding layer |
| C2 | eager render pass | record.jl L123-191 (`begin_render_pass!`, `record_draw!`, `end_render_pass!`, `setviewport!`, `use_bindings!`) | `src/vulkan/graph.jl` L1486-1508 (`beginrender!`/`endrender!`) and L1956-1998 (`emitdraw!`/`emit_draw!`) call the same `begin_pass!`/`draw_*_in_pass!` with the same arguments; core `pass!` (`src/graphics/record.jl` L164) and `draw!(device, …)` (L298) are the callers. api.md 2.6b: one path, the eager draw is its own pass through the Builder | 80 lines (plus Metal's twin, M4) |
| C3 | build context | acceleration.jl L178 (`VulkanAccelBuildContext` alias) and L181-184 accessors | `src/raytracing/accel.jl` L126-133 (`AccelBuildContext`) | 17 lines |
| C4 | TLAS bookkeeping | hwtlas.jl L42-171, L375-501, L503-651, L653-921 | `InstanceBatches` (`src/raytracing/batches.jl` L23-118) already owns order, handles, reindex, hidden; `instancerecords` and `update_instance_records_kernel!` (`src/raytracing/instances.jl` L19-46) already write records. api.md 2.6 and the plan put the rest (ranges, Build vs Update, `builtcount`, release by token) in core | see 2.4 |
| C5 | capability flags | api.jl L396-403; hwtlas.jl L1076-1084 | `src/graphics/commands.jl` L245-295 and `src/raytracing/api.jl` L214-278 define the defaults; api.md 4.1 makes them `caps(dev)` fields | 20 lines; also rule 3: `vk_context()` is a "current" object |
| C6 | `Mat4f` | hwtlas.jl L20 `const Mat4f = SMatrix{4,4,Float32,16}` | `using GeometryBasics` on L14 already brings `Mat4f`; the line shadows it | 1 line, owner rule |
| C7 | format inverse | record.jl L57-69 (`eltypeof`) | `src/graphics/resources.jl` L178 declares it and L185-194 (`blittarget`, `eltype(::Framebuffer)`) exist only because a target carries a `VK.Format` instead of `Image{T}` | the whole question disappears with `Image{T}` |

### 2.3 Against Metal (`src/metal`)

| # | audited side | Metal side | note |
|---|---|---|---|
| M1 | hwtlas.jl: `InstanceBatch` L42-58; `VulkanTLAS` L110-171 + ctors L181-206; `hwtlas_add_geometry!` L312-369; `push!` L404-501; `update_transforms!`/`update_transform!` L521-620; `delete!`/`set_visible!` L627-651; `instance_buffer`/`refit!` L284-287; `vkinstances_kernel!` L685-696; `writeinstances!` L706-730; `sync!` L847-921; `AdaptedAccel(hwtlas)` L213-218; `adapt_structure` L250-272 | `src/metal/hwtlas.jl`: `MetalInstanceBatch` L242-259; `MetalHWTLAS` L261-325 (its own comment at L269: "is the same as Vulkan's and is a candidate to be" shared); `add_geometry!` L371-410; `push!` L460-520; L526-551; L553-568; L574-577; `mtlinstances_kernel!` L589-599; `writeinstances!` L608-619; `sync!` L622-709; L332-334; L356 | ~550 lines written twice; one core copy of ~300 replaces both |
| M2 | fixed-state maps pipeline.jl L260-342 | `src/metal/graphics.jl` L21-35 (`mtl_cull`, `mtl_depth_compare`) | legitimately per device type; listed so nobody "shares" them |
| M3 | `readback_framebuffer` framebuffer.jl L246-289 | `src/metal/graphics.jl` ~L973-1010 (same `(width, height)` matrix of `eltypeof(fb.color_format)` contract) | `Array(img)` in core |
| M4 | `VulkanPassHandle`, `record_draw!`, `setviewport!`, `use_bindings!`, `end_render_pass!` record.jl L118-191 | `MetalPassHandle` L1029, `record_draw!` L1250, `setviewport!` L1305, `end_render_pass!` L1400, `use_bindings!` L1613 | the eager-pass twin exists per backend |
| M5 | `VulkanTexture2D`/`VulkanSampler`/`VulkanTextureBindings`/`upload_texture_data!` textures.jl L6-40, L175-214, L230-282 | `MetalTexture2D` L79, `MetalSampler` L88, `MetalTextureBindings` L1588, `upload_texture_data!` L157-180 (same `DimensionMismatch` text) | `Image` + sampler table in core |
| M6 | ray-query candidate verbs hwtlas.jl L951-959 | `src/metal/procedural.jl` | per device type by design (core `src/raytracing/accel.jl` L206-227 declares them) |

### 2.4 What in acceleration.jl / hwtlas.jl is TLAS/BLAS bookkeeping written against Vulkan types

All of the following is backend independent (Metal has the same code, M1) and only *names* Vulkan
types; api.md 2.6 and the plan's "Acceleration structures and textures" put it in core:

- **Instance tables.** `InstanceBatch` (hwtlas.jl L42-58); `VulkanTLAS.instances`, `blas_list`,
  `blas_triangles`, `blas_offsets` (L117-127); `combined_instance_buf`/`combined_records` (L141-145);
  `writeinstances!` (L706-730); `vkinstances_kernel!` (L685-696; only the `VulkanInstanceRecord`
  constructor on L692-694 is Vulkan's, which is exactly what `instancerecordtype(dev)` abstracts).
  Target: the range table and the records kernel of 2.6.
- **Batching.** `addbatch!`/`addgeometry!` (L375-402), five `push!` (L404-501), `pending_updates`
  (L164, L529, L710-716), `update_instance_transforms!` mapping a flat index onto batches
  (L572-593). Target: `push!(tlas, blas, …)` returning a range handle; `tlas[h] = t`.
- **Rebuild vs refit.** `dirty`/`transforms_dirty` (L153-156), `sync!` (L847-921), the
  "refit before any build is a build" fallback (L897-903), the refit preconditions
  (acceleration.jl L310-322, L609-613), the `allow_update` opt-in and cached
  `update_scratch_size` (`LavaBLAS` L36-48, `LavaTLAS` L74-76, L283-285, L514, L585),
  `build_tlas(...; allow_update=true)` always (hwtlas.jl L797). Target: pending `BuildAccel`
  changes that coalesce into a Build or an Update with `builtcount` (2.6, plan "Pending builds
  coalesce").
- **Compaction of the BLAS list** (not driver compaction, which does not exist here, see 4):
  `_compact_blas_list!` (L735-767) and the release of dropped BLASes after the rebuild
  (L888-890). Target: "a removed range releases its BLAS with the token of the build that
  deactivated it".
- **Lifetime.** `preserves` lists (acceleration.jl L26, L72, L265-269, L503-504, L570-571, L688,
  L1132), the wait-and-free in `build_accel!` (L1025-1040), `unsafe_free!`/`destroy_now!`
  (L99-149), `syncbuf!` (L159-166), the `holdleaves!` stop (hwtlas.jl L232), the "fresh instance
  buffers per rebuild" rule (L781-791). Target: holds and retirement by token in the release
  engine (api.md 3, 4.2).
- **Sizing.** Nine `max(…, 16)` storage minimums (acceleration.jl L252, L258, L427, L433, L491,
  L497, L559, L565, L625) and `max(total, 1)` (hwtlas.jl L665, L789-791). Target: capacity that
  starts at 64 and doubles (2.6).
- **Triangle concatenation and per-instance offsets.** hwtlas.jl L800-819, L832-833;
  acceleration.jl L1582-1596; `HardwareAccel` fields raycore_compat.jl L30-38; `AdaptedAccel`
  `triangles`/`offsets`. api.md 2.6: "never through a scene-wide concatenation" — this goes
  rather than moves.

What is genuinely Vulkan's in those 2 698 lines: `VK.AccelerationStructureKHR` creation on a
region (L254-256 and twins), the sizes query (L932-975, replaceable by Vulkan.jl's
`get_acceleration_structure_build_sizes_khr`), the build command (L1106-1129, replaceable by
`cmd_build_acceleration_structures_khr`), the address query (L275-276), `VulkanInstanceRecord`
(instance_record.jl), the ray-query intrinsics (hwtlas.jl L951-1019), and — only while a TLAS is
bound as a descriptor — the descriptor set (hwtlas.jl L1041-1070, raytracing/pipeline.jl
L252-282). Estimate: ~700 lines of bookkeeping that belongs in core, ~200 lines of Vulkan verbs,
~1 800 lines dead/duplicate/scaffolding.

## 3. Patterns that inflate the code

Each pattern: name, 2-3 examples, estimated line cost in the audited files (costs overlap; the
sum is not the total).

**P1. Branching on type or kind by hand instead of dispatch — ~120 lines.**
`as isa LavaTLAS` acceleration.jl L125, L148 (one method per type would do); `p isa LavaArray`
L145 and L1036 over containers typed `Vector{LavaArray}`/`Any`; `prev isa LavaArray{T}`
hwtlas.jl L666; `hwtlas.hw_accel isa HardwareAccel` L821; `x isa Vector`/`Tuple` L1602-1603.
Symbol kinds (api.md rule 2): `p.front === :mesh` pipeline.jl L225/L235 and the
`ismeshpipeline` branch record.jl L174-189 (`graph.jl` L1976-1978 repeats it);
`geometry_type::Symbol` acceleration.jl L856-881; stage names `:vertex`, `:fragment`,
`:geometry`, `:tess_control`, `:mesh`, `:raygen`, `:closesthit` api.jl L17, L52, L63-64, L77-87,
L122, shaders.jl L192-221; `filter::Symbol`, `wrap::Symbol` textures.jl L36-37 with the ternary
chains L51-59; `payload_type::Symbol` (`:f32_7`, raycore_compat.jl L89, L107).

**P2. Defensive checks on values whose type is known — ~150 lines, plus 17 `Union{Nothing,…}`/`Any` fields.**
`=== nothing` ladders: api.jl L68, L82, L86, L99-101, L105; record.jl L35, L99, L105, L139-142;
pipeline.jl L42-54, L58-59, L91-94, L98, L427-429, L483-486, L491, L595-599, L604-611;
acceleration.jl L474-476, L710, L1112; hwtlas.jl L526, L613, L648, L712, L849, L884-887, L897,
L913. `checkopen` (window.jl L301-303) at four entry points; `win.acquired ||` L388 and api.jl
L366; `@assert` acceleration.jl L547, L618, L1151; `as.storage.buf.freed` guard L105 (a finalizer
double-tap, which the release engine makes impossible); `batch === nothing && continue` L712.
Fields: `VulkanFramebuffer` L21-23, `VulkanWindow` L17, L48, `LavaBLAS` L44, `LavaTLAS` L76,
`VulkanTLAS` L133-145, L150 (`static_tlas::Any`), L164 (`Dict{…,Any}`), L170
(`procedural::Any`), `HardwareAccel` L41, `VulkanTextureBindings.textures::Vector{Any}`
textures.jl L234.

**P3. `try/catch` that swallows or logs — 2 sites, ~15 lines.**
acceleration.jl L127-134 logs a destructor failure and continues; L1608-1613 catches
`MethodError` to fall back to `collect`. Both violate the rule; the second is in dead code.

**P4. Globals and module-level state — ~80 lines; the owner should be the plan, the TLAS sync state, or the device.**
`PIPELINE_NO_COMPILE[]` read in production paths pipeline.jl L195, raytracing/pipeline.jl L206
(a test flag); `vk_context()`/`vk_context(b)` api.jl L403, hwtlas.jl L1076-1077, L1083 (rule 3:
nothing is found through a "current" object); `COLOR_USAGE` framebuffer.jl L30-32 used once
(L86) — should be a kwarg; `VK_STYPE_*` L803-808, `C_SIZEOF_AS_GEOMETRY_KHR` L800, `VkBRI` L591,
`VK_SHADER_UNUSED_KHR` raytracing/pipeline.jl L6, `CANDIDATE_*`/`COMMITTED_*` hwtlas.jl
L941-945, `AS_*_USAGE` L781-792, `const Mat4f`/`const KA` hwtlas.jl L18-20. Caches that are
globals in disguise: `LavaTLAS.desc_sets::Dict{UInt64,…}` keyed by a raw layout handle
(acceleration.jl L84, raytracing/pipeline.jl L253-280); `ctx.caches.gfx_pipelines` (api.jl L91,
L105); the double cache `ctx.caches.rt_current`/`rt_pipelines` keyed on world age
(shaders.jl L101-112 — the `Kernel` at its `CodeInstance` replaces it); `pending_updates`
hwtlas.jl L164; `ctx.diag.alloc_log` pushed from five build sites (acceleration.jl L277-280,
L446-449, L770-773, L1238-1241, L1501-1504).

**P5. Near-copies for a second type, dimension or extension — ~900 lines** (D7, D8, D13-D18 above).

**P6. Hand-rolled vector/matrix math — ~50 lines.**
`(Float32(v[1]), Float32(v[2]), Float32(v[3]))` acceleration.jl L380, L1322, L1398, hwtlas.jl
L341 (`Vec3f(v)`); transforms as `NTuple{12,Float32}` L462, L704, L711-713, L1556, L1563 while
`Mat3x4f` exists; the per-component store loop L717-719; `Vec3f(get_origin(false,1), …(2),
…(3))` hwtlas.jl L953-958 (the intrinsic should return a `Vec3f`); `SVector{3,Float32}` bary
L987, L997 (`Vec3f`); the `Bounds3` union loop L361-365; `Mat4f` redefined L20.

**P7. Manual loops that are a map, broadcast or reinterpret — ~90 lines.**
Index ramps acceleration.jl L1328-1330, hwtlas.jl L344-347; vertex de-indexing L1317-1324,
L1392-1401, hwtlas.jl L336-343; AABB byte packing with `unsafe_store!` L402-414 (24-byte records:
`reinterpret(UInt8, aabbs)`); instance packing L471-478; `for v in views; finalize(v); end`
window.jl L223-225, L450-457; linear searches for a format/present mode L134-140, L147-155
(`findfirst`); running offsets hwtlas.jl L810-819 (`cumsum`); `T[p for p in all_primitives]`
L1583-1588.

**P8. Forwarding layers and one-field wrappers — ~200 lines.**
`get_or_compile_gfx` api.jl L17-18 → `cached_gfx_shader`; `get_or_compile_rt` shaders.jl
L192-194 → `cached_rt_shader`; `ensure_compiled_with_shader!` → `ensure_compiled!` (api.jl L46-56,
L116-126); `build_blas` ×2 → `build_blas_from_addresses` (acceleration.jl L203-224);
`create_rt_pipeline` 3-arg → vector form (raytracing/pipeline.jl L58-65); `query_as_build_sizes`
→ `_impl` (L914-930), `build_as_on_gpu` → `_impl` (L1049-1072); the four one-line accessors
L181-184; `begin_pass!` shims pipeline.jl L390-392, L789-790; `setviewport!`/`use_bindings!`
record.jl L162-168; `depthclearvalue` L151-152; `Raycore.update_instance_transforms!` 3-arg
hwtlas.jl L591-593; `trace_closest_hits!` ×4 → `trace_rays!` (raycore_compat.jl L126-175).
Wrapper structs: `VulkanAttachment` (view + image, record.jl L25-30), `VulkanDraw` (pipeline +
shader, L81-84), `VulkanPassHandle` (emitter + one-shot, L118-121), `HardwareAccel`
(raycore_compat.jl L29-45), `VulkanAccelBuildContext` (acceleration.jl L178).

**P9. `_`-prefixed functions — 3 definitions, 7 uses.**
`_vulkan_framebuffer` framebuffer.jl L79, L82; `_reuse_or_alloc` hwtlas.jl L664, L832-833;
`_compact_blas_list!` L735, L771. (`_normalise_chit`, mentioned at shaders.jl L30, lives in core
`src/raytracing/pipeline.jl` L42-44.)

**P10. Compat shims, dead code and abandoned approaches — ~900 lines.**
- `raycore_compat.jl` (247) is compatible with Raycore's *software BVH* API: "a drop-in
  replacement for Raycore's software BVH traversal closest_hit()" (L3-4), using Raycore's
  `RTRay`/`RTHitResult` (L7). Still needed in part: Hikari calls `trace_closest_hits!`
  (`Hikari/src/integrators/volpath/hw-rt.jl` L42) and `trace_closest_hits_indirect!`
  (`volpath.jl` L58), and `hwtlas.jl` L821-830 makes one `HardwareAccel` per `VulkanTLAS` to hold
  the RT pipeline. Not needed: the any-hit variants (L101-110, L157-175; callers only in Mantle
  tests), and `HardwareAccel(tlas; ctx)` (L57-62), which converts a Raycore CPU `TLAS` and has no
  caller.
- Dead chain in acceleration.jl: `build_hw_accel_from_tlas` L1364-1597, `build_blas_pooled`
  L1146-1299, `build_blas_from_primitives` L1312-1335, `to_cpu_vector` L1601-1614,
  `create_as_input_pool` L741-776 — no call site in `src`, `test`, Hikari, RayMakie, Raycore
  (≈ 530 lines).
- `pack_gfx_args` vector form api.jl L268-304 ("for callers written against it", L253-254): no
  caller. `LavaTexture` textures.jl L286 and `Base.:*(tex, sam)` L285: no caller.
  `VulkanTexture1D` L23-31: no constructor, no caller (only a docstring mention in
  `src/graphics/resources.jl` L33). `invalidate_stale_rt_cache!` shaders.jl L32: a no-op.
- "Legacy" twins kept on purpose: acceleration.jl L853-854 "not being migrated in this task —
  left intact" (D14).
- Debug-only in production: `ctx.diag.alloc_debug` logging (five sites, P4), with the comment
  L767-769 naming the fault hunt it served; `PIPELINE_NO_COMPILE`.
- Two regimes in one function: `transition::Bool` pipeline.jl L412, L431-438, L491 (the graph
  passes `false`, `graph.jl` L1496; the eager pass passes `true`).
- Workarounds kept as features: `holdsframe` + `acquireframe!` (window.jl L39-44,
  framebuffer.jl L477-508, api.jl L319, L340) for Mesa's headless image order; `fb_size` +
  `sync_swapchain!` (window.jl L49-55, L305-328) for Xwayland not reporting OUT_OF_DATE.
- Flags for approaches api.md abandons: `allow_update` as an opt-in (acceleration.jl L28-35,
  L206, L465) — api.md 2.6 makes Update the default; `opaque` per build; `chit_miss_take_args`
  (core `src/raytracing/pipeline.jl` L36).

**P11. Comments that narrate history instead of stating the contract — ~350 lines.**
pipeline.jl L346-351 ("`vk_draw!` is GONE. It was 178 lines…"), L124-132, L443-450;
framebuffer.jl L129-146, L199-203, L219-233, L512-514; window.jl L77-81 ("every window RayMakie
opened since the split failed"), L208-219, L251-257, L343-353, L396-399, L445-449, L463-468;
api.jl L21-29, L69-75, L239-243, L250-254, L383-392; textures.jl L176-177 ("did not until
2026-09-26"); record.jl L1-13, L41-47; acceleration.jl L96-97, L175-177, L579-583, L794-798,
L853-854, L1027-1034, L1246-1250, L1286-1294, L1571-1578; hwtlas.jl L5-10, L22-24, L220-231,
L559-564 (a ten-line anecdote about a still picture), L657-663, L773-779, L781-787, L853-858,
L898-900, L1031-1039; raytracing/pipeline.jl L245-250, L288-292; shaders.jl L1-16, L30,
L104-107, L239-240; raycore_compat.jl L152-155, L169.

**P12. Functions over ~100 lines — six, 1 037 lines.**
`create_graphics_pipeline` 207 (pipeline.jl L21-227); `begin_pass!` 131 (L402-532);
`create_swapchain!` 141 (window.jl L121-261); `create_rt_pipeline` 170
(raytracing/pipeline.jl L67-236); `build_blas_pooled` 154 (acceleration.jl L1146-1299);
`build_hw_accel_from_tlas` 234 (L1364-1597).

**P13. Keyword-argument explosions.**
`create_graphics_pipeline` 14 kwargs (pipeline.jl L21-36); `begin_pass!` 9 (L402-413), each a
`Union{Nothing,…}`; `pack_geometry!` Symbol form 10 (acceleration.jl L855-862);
`pack_build_geometry_info!` 8 (L887-893); `build_as_on_gpu` 6 (L1049-1055); `build_tlas` 4
(L461-465); `push!` ×5 with `instance_id`/`instance_mask`/`sbt_offset`/`n`/`triangles`
(hwtlas.jl L404-501); `ensure_compiled!` 4 kwargs × 4 methods (api.jl L46-157).

**P14. Parallel data structures kept in sync by hand — ~120 lines.**
`blas_list`/`blas_triangles`/`blas_offsets` pushed together (hwtlas.jl L353-359, L396-400) and
rewritten together (L763-765); six derived views of one instance list refreshed by `sync!`:
batches, `combined_instance_buf`, `combined_records`, `hw_accel.{triangle_data, blas_offsets,
per_instance_tri_offsets}`, `tri_gpu`/`off_gpu`, `static_tlas` (L133-150, L821-833, L874-882);
window.jl `images`/`views`/`holdsframe`/`image_available`/`render_finished`/`in_flight`
(L18-19, L33-35, L44, L205-206, L223-260) and `extent` vs `fb_size` vs GLFW (L31, L55,
L158-167); `LavaBLAS.vertex_arr/index_addr/n_triangles/max_vertex/geo_flags` cached from the
build for a later refit (acceleration.jl L44-48, L281-285).

**P15. Vulkan structs built by hand where Vulkan.jl wrappers exist — ~230 lines.**
acceleration.jl L794-975: `pack_geometry!` ×3, `pack_build_geometry_info!`,
`query_as_build_sizes_impl` with a raw `ccall(fptr, …)` and `unsafe_load` from a byte buffer
(L954-970); L1086-1129 and L635-686 raw `ccall` of `vkCmdBuildAccelerationStructuresKHR`;
`VkBRI` L591; `pack_as_instance!` L703-733. Vulkan.jl (`~/.julia/packages/Vulkan/12azN`) provides
`AccelerationStructureGeometryKHR` (generated/linux.jl:23510), `AccelerationStructureInstanceKHR`
(:21028), `DeviceOrHostAddressConstKHR` (:4145), `get_acceleration_structure_build_sizes_khr`
(:75971) and `cmd_build_acceleration_structures_khr` (:74976). The hand packing is justified by a
"VulkanCore.jl alignment bug" (L794-798). **Guess:** that bug is from an older VulkanCore; whether
the current wrappers lower the address union correctly was not verified (no Julia run) — it is
one MWE away. Smaller cases: the 15-positional-argument `VK.Sampler` call textures.jl L61-72; two
full `StencilOpState`s for a pipeline that never uses stencil, pipeline.jl L141-144.

**P16. Code that works around a bug downstream of its real cause — ~250 lines.**
The hand-packed structs (P15); `acquireframe!` and `holdsframe` (P10); `fb_size` polling (P10);
`GLFW.DefaultWindowHints()` so GLMakie can open a window afterwards (window.jl L90-92);
`checkacquirable` (L343-354: a second task rendering mid-frame — the real cause is no graph lock);
`VK.device_wait_idle` on resize, close and swapchain re-creation (L221, L426, L441) standing in
for retirement by token (the plan forbids both); the nine `max(…, 16)` and two `max(…, 1)`
minimum sizes standing in for a capacity; `_reuse_or_alloc` (hwtlas.jl L664-673) standing in for
`resize!`; and 30+ `hold!`/`preserves` sites doing the release engine's job per command
(pipeline.jl L555, L578, L671-674, L717, L782; framebuffer.jl L280-281, L332-333, L373, L455;
textures.jl L167, L189, L211; api.jl L321, L341; raytracing/pipeline.jl L300, L307, L333, L340,
L354; acceleration.jl L269, L337, L436, L504, L571, L628, L688, L1132; hwtlas.jl L1055, L1068).
api.md 3: holds are core data recorded at declaration.

**P17. Pointers and unsafe calls where a Julian type works — ~120 lines.**
`unsafe_copyto!`/`pointer`: framebuffer.jl L286, L472; textures.jl L188; acceleration.jl
L1219-1224, L1479-1485. `unsafe_store!`: acceleration.jl L407-412, L718-730, L822-834,
L842-849, L864-882, L895-905, L940; api.jl L295. `unsafe_load`: L968-970.
`Ptr{Nothing}(Base.unsafe_convert(Ptr{UInt64}, bda))`: pipeline.jl L609, L662, L704.
`pointer(handles)` raytracing/pipeline.jl L393. Raw `.vks` handles: acceleration.jl L668-669,
L682, L959, L1108, L1125; `UInt64(layout.vks)` as a `Dict` key raytracing/pipeline.jl L253.

## 4. What the design docs miss

Checked against api.md sections 2-4 and Appendix A, and the plan's three sections.

**Swapchain recreation and resize.** Covered in principle (2.7, 4.7, plan "Windows and present").
Missing: present-mode selection from `vsync` (window.jl L143-155: MAILBOX, else IMMEDIATE, else
FIFO) — `createswapchain` must own it but nothing says what `vsync = false` means when MAILBOX is
absent; surface-format negotiation (L129-141: the surface may not offer the requested format and
the window's actual format differs from the one asked for) — api.md 2.7 leaves the default colour
format open and has no way to report the format a window ended up with; surface capabilities
(image count L169-173, transform L198, composite alpha L199); `Array(win)` needs
`IMAGE_USAGE_TRANSFER_SRC_BIT` on swapchain images, which a surface may not support (L175-183) —
2.6b does not say what `Array(win)` does then. The Xwayland case (L49-55, L310-314: no
OUT_OF_DATE, no SUBOPTIMAL) is covered only if `resize!(win, w, h)` is reliably called from the
GLFW event; the docs should state that the event, not the driver, is the trigger.

**MSAA and depth attachments.** No MSAA exists today (pipeline.jl L120-122 hard-codes
`SAMPLE_COUNT_1_BIT`, resolve mode NONE L473, L520). Depth is `D32_SFLOAT` only (framebuffer.jl
L91, lowering.jl L239) and there is no stencil. api.md 2.6b (`target` = colour images then one
`Float32` depth image) matches today and should say explicitly that MSAA, resolve and stencil are
out of scope (or add a `samples` parameter to the target). The depth/target mismatch rule
(`checkdepthtarget`, pipeline.jl L133-134, `src/graphics/commands.jl` L370) — a `DepthLess`
pipeline into a target without depth, or `DepthOff` into one with — is enforced today and not
stated in 2.6b.

**Render pass vs dynamic rendering.** Consistent: today is dynamic rendering (pipeline.jl L3,
L185-191, L525-531) and 4.5 wants dynamic rendering with secondary command buffers. One thing
to state: 4.5 says a piece "sets viewport and scissor itself" while Appendix A says `set_viewport!`
goes — a viewport set remains in `recordpiece`; and the Y-flip convention. Today Vulkan's
Y-down clip space is handled by a negative-height viewport with the scissor re-derived from it
(pipeline.jl L757-773). 2.6b's "scene rectangle … a remap and four clip distances" says nothing
about which clip-space convention the shared vertex stage targets, and RayMakie must render the
same on Metal.

**Pipeline state coverage.** Today: blend `Opaque`, `AlphaBlend`, `Additive`, `NoWrite`,
`Premultiplied`, `PerAttachment` (pipeline.jl L295-342, state.jl L25-68) with integer attachments
forced unblended (L148-156); cull 3; depth 6 including `DepthLessEqReadOnly` (write mask off,
L292-293); topology 8 including adjacency and patch list (L260-267); tessellation
`patch_vertices` (L96-100); polygon mode FILL, front face CCW, line width 1, no depth bias, no
stencil, no logic op (L111-118). api.md 2.6b names only `blend, cull, topology, depth`. Missing
from the docs: `PerAttachment` blend and the integer-attachment rule, the depth-write mask, and
a statement that line width / polygon mode / depth bias / stencil are out of scope. Push-constant
size is 8 or 16 bytes (BDA plus the exception-flag address, `src/vulkan/runtime/command.jl`
L872-879); fine under core's argument block, but the exception flag's slot is not in 4.4.

**Texture formats and sampling.** Two disagreeing tables today (D5): targets use colorants
(`vkformat`), textures use `NTuple{N,Float32}`/`UInt8` (`julia_to_vk_format`). `imageformat(dev,
T)` (4.8) fixes that, but one question it cannot answer from `T` alone: sRGB vs UNORM for the same
`BGRA{N0f8}` (window.jl L21-29, L65-70; `srgb::Bool` kwargs at framebuffer.jl L77-78, window.jl
L76-82, lowering.jl L229-237). The docs need to say where "encode on write" lives (a parameter of
`Image`/`Window`, or a distinct element type). Samplers today: filter nearest/linear, address
repeat/clamp/mirror/clamp-to-border, anisotropy; mipmap mode is set (textures.jl L63) but every
image has one mip level (L104, framebuffer.jl L45), so there are no mipmaps; no compare sampler;
`Texture1D` is declared and never implemented. api.md `Sampler(; filter, address, anisotropy)`
matches; it should state that mipmaps and 1D/3D/array images are out of scope or add them to
`Image(dev, T, dims...)`.

**BLAS compaction.** None today: no `COMPACTED_SIZE` query, no `ALLOW_COMPACTION` flag anywhere in
`src` (grep: 0 hits). The plan says "HWTLAS bookkeeping (…, compaction) is core's", but api.md 2.6
`BLAS(dev, vertices, faces)` has no compaction step and section 4 has no verb for the query pool +
copy it needs. Either drop the word from the plan or add the verb.

**Instance masks and SBT / hit groups.** Today: mask per record and mask 0 for hidden batches
(hwtlas.jl L692-694); an SBT offset per batch (`InstanceBatch.sbt_offset` L54, `VulkanInstanceRecord`
L29); N hit groups = N closest-hit functions (raytracing/pipeline.jl L151-175); one any-hit shared
by every hit group (L53, L172); an intersection shader turns *all* groups procedural (L98-103,
L164-166); one miss shader; empty callable region (L451-453); recursion depth 1 (L212); the
instance flags byte is always 0 (instance_record.jl L46, hwtlas.jl L692-694 — no
`TRIANGLE_FACING_CULL_DISABLE`/`FORCE_OPAQUE`); `opaque` per BLAS (acceleration.jl L206, L242); a
cull mask per trace (raycore_compat.jl L126-130, L179-198). api.md has `RangeEntry.mask` and
"the SBT layout for many hit groups is open (section 10)"; it has no SBT offset on `push!`, no
instance flags, no `opaque`, no cull mask on `trace!`, and `RayTracingPipeline` has neither
`payload_type` nor `chit_miss_take_args` (core `src/raytracing/pipeline.jl` L28, L36), which the
compile keys on today (shaders.jl L26-28). Hikari's volpath path needs the cull mask.

**Procedural (AABB) geometry.** Today: `build_blas_aabb` (acceleration.jl L391-453, used by
`Hikari/src/fem.jl` L788-789), the `procedural` payload on `AdaptedAccel` with the three candidate
verbs (hwtlas.jl L947-999; core `src/raytracing/accel.jl` L135-237), an intersection stage in the RT
pipeline (raytracing/pipeline.jl L72, L130-138) that core never uses because
`supports_rt_pipeline(::AdaptedAccel)` forces the inline path (core `src/raytracing/api.jl`
L225-226). api.md 2.6 has only `BLAS(dev, vertices, faces)`: no AABB constructor, no procedural
payload, no candidate verbs in section 4, no intersection stage in `RayTracingPipeline`.
Appendix A reduces it to a `supports_procedural_traversal` caps flag. This is a consumer (Hikari
FEM) need with no replacement.

**Motion blur / multi-frame.** Nothing today (grep: 0); docs silent. State it as out of scope.

**What the window/present verbs in 4.7 omit.** `isopen`/`close` and ordered teardown (window.jl
L433-473); who pumps GLFW events (`GLFW.PollEvents()` is in `src/vulkan/graph.jl` L1768 and
`src/metal/window.jl` L174, L214 — not in these files, not in the docs); present-mode and
surface-format negotiation and the reporting of the actual format (above); the per-image
rendered semaphore (L248-259) is in the plan but `acquirenext(sc)` only returns the acquire
semaphore — where the present-wait semaphore comes from is unsaid; headless windows (GLFW's null
platform, framebuffer.jl L484-486) used by tests; the process-global GLFW hint reset (L90-92) that
lets GLMakie coexist.

**Appendix A rows that map the audited code wrongly or are missing.**
- `transition_image!`: "goes as a call … (no implementation on either device type)" — Vulkan
  implements it (pipeline.jl L354-368) with 11 call sites in these files and 5 elsewhere.
- `AdaptedAccel`: "goes with Metal's HWTLAS bookkeeping" — it is core's
  (`src/raytracing/accel.jl` L29) and the device-side object every `Raycore.closest_hit(accel, ray,
  mask)` (hwtlas.jl L1004-1019) takes; 2.6 names no device-side TLAS type for ray queries. Its
  concatenated `triangles`/`offsets` go; what replaces the object does not exist in the docs.
- `trace_closest_hits_*`: "go: the immediate trace_* verbs" — the default raygen/closest-hit/miss
  that writes a hit record (raycore_compat.jl L179-247) and the `RTHitResult` layout Hikari reads
  have no home; the docs should say it is a core library pipeline.
- Missing rows for names in `BACKEND_VOCABULARY`: `build_blas_aabb` and `upload_texture_data!`
  (backend.jl L815), `supports_hwtlas` (L862), `readback_target` (L947).
- Raycore's mutation API on a `VulkanTLAS` — `Raycore.sync!`, `update_transform!`,
  `update_transforms!`, `update_instance_transforms!`, `set_visible!`, `instance_buffer`, `refit!`,
  `wait_for_gpu!`, `static_tlas`, `Adapt.adapt` — is what RayMakie (`plots/mesh.jl` L1121 and 11
  other calls) and Raycore (39 uses of `static_tlas`, 16 of `instance_buffer`) actually call. None
  is in Appendix A (they are not vocabulary names) and 2.6 does not say which core call each
  becomes. A second table (Raycore name → core call) is needed, or the consumers cannot be ported.

## 5. Minimal form

What the same functionality needs under api.md, by owner. Estimates assume Vulkan.jl's
acceleration-structure wrappers work (P15, guess) and that the barrier helper, release engine
and argument block exist in core as the docs state.

**Core (backend independent, shared with Metal, CUDA, HIP):**
- `TLAS`/`BLAS`/`InstanceRange`/`RangeEntry`, `push!`/`tlas[h] = t`/`delete!`/`build!`/`refit!`,
  pending `BuildAccel` coalescing with `builtcount`, the offsets and records kernels
  (parametrised by `instancerecordtype(dev)`), release of a range's BLAS by token: ~400 lines
  (replaces hwtlas.jl L42-921 and `src/metal/hwtlas.jl` L242-709 together, and the bookkeeping
  half of acceleration.jl).
- `Image`, `Sampler`, the texture and sampler tables with free lists, `copyto!`/`Array(img)`,
  stores into images: ~150 lines (replaces textures.jl, framebuffer.jl's target type and readbacks,
  Metal's twins).
- `Window` (size, pending size, generation holds, acquired images), `resize!`, present as part of
  the run, `Array(win)` through the present segment: ~200 lines (replaces the policy half of
  window.jl and api.jl L306-381).
- Draw nodes, render-pass formation, pieces, `when!` draws, the pipeline key: part of graph
  compile, ~300 lines attributable to graphics.
- `RayTracingPipeline` value plus a library "closest-hit record" pipeline for Hikari's
  `trace_closest_hits!` use: ~100 lines.
Core total for these features: ~1 150 lines, of which today's Vulkan files hold ~570 written
against Vulkan types and Metal holds a second copy.

**Vulkan verbs (`src/vulkan`):**
- `compiledraw`: `VkGraphicsPipeline` from stage SPIR-V, fixed-state maps, target formats; mesh and
  classic front ends as two methods: ~200 lines (from pipeline.jl L21-342 and api.jl L31-237).
- `compiletrace`: RT pipeline, hit groups, SBT: ~150 lines (from raytracing/pipeline.jl L58-236,
  L377-456, shaders.jl L202-237).
- `beginrender!`/`endrender!`/`recordpiece`/`emitdraw!`/`bindindices!`: ~80 lines (from
  pipeline.jl L402-532, L541-741 minus the transitions, the three push-constant copies and the
  holds).
- `emittrace!`: ~25 lines.
- window open verb, `createswapchain`, `acquirenext`, `surfacesize`, the present part of
  `submitnative!`: ~120 lines (from window.jl without fences, polling, wait-idle, `target_*`).
- `placeimage`, `imageformat` (one table), `tableentry!`, `samplerentry!`: ~80 lines.
- `accelsizes`, `placeaccel`, `acceladdress`, `emitbuild!`, `instancerecordtype`, `countsource`:
  ~120 lines with Vulkan.jl's wrappers (from acceleration.jl L226-288, L914-1133 and
  instance_record.jl).
- Ray-query intrinsic overlays and the three candidate verbs: ~40 lines (hwtlas.jl L951-1019).
Vulkan total: ~800 lines, against 6 358 today.

Together: ~800 Vulkan + ~1 150 core ≈ 1 950 lines for what these 6 358 lines (plus ~4 300 Metal
lines in the counterpart files) do now. The ratio comes from three things the refactor can forbid
by name: a second eager implementation beside the graph's (C2, M4, D15), per-backend copies of
bookkeeping (M1, 2.4), and holds/barriers/transitions written per command instead of derived
(P16, D2, D10).

## 6. Julia-rule violations, by rule (file:line)

- **`_name` functions:** framebuffer.jl L79, L82; hwtlas.jl L664, L735 (and uses L771, L832-833).
- **Globals / state not in a context object:** pipeline.jl L195 and raytracing/pipeline.jl L206
  (`PIPELINE_NO_COMPILE[]`); api.jl L403, hwtlas.jl L1076-1077, L1083 (`vk_context()`);
  framebuffer.jl L30-32 (`COLOR_USAGE`, one use); acceleration.jl L591, L800-808, L781-792;
  raytracing/pipeline.jl L6; hwtlas.jl L18-20, L941-945. Dict caches as state: acceleration.jl
  L84; shaders.jl L101-112; hwtlas.jl L164.
- **`try … catch` without rethrow/report:** acceleration.jl L127-134 (logs), L1608-1613 (fallback).
- **Name variants instead of dispatch:** `draw_in_pass!`/`draw_mesh_in_pass!`/
  `draw_indexed_in_pass!`/`draw_indirect_in_pass!` (pipeline.jl L541-718); `build_blas`/
  `build_blas_aabb`/`build_blas_pooled`/`build_blas_from_primitives`/`build_blas_from_addresses`
  (acceleration.jl L203-1335); `build_tlas` ×2 (L461-589); `pack_geometry!` typed vs Symbol
  (L818-884); `trace_rays!`/`trace_rays_indirect!`, `emit_trace!`/`emit_trace_indirect!`,
  `trace_closest_hits!` ×4; `update_transform!`/`update_transforms!`/`update_instance_transforms!`
  (hwtlas.jl L521-620); `readback_framebuffer`/`readback_window`/`copy_framebuffer!`/
  `copy_image_to_buffer!` (framebuffer.jl L246-475); `ensure_compiled!`/
  `ensure_compiled_with_shader!` (api.jl L46-157); `set_viewport!` 2-arg vs 5-arg (pipeline.jl
  L730, L757); Symbol kinds listed under P1.
- **GeometryBasics `Vec`/`Mat` for vector math:** acceleration.jl L380, L462, L704-719, L1322,
  L1398, L1556-1563; hwtlas.jl L20, L341, L361-365, L953-958, L987, L997.
- **Pointers / unsafe calls where a Julian type works:** listed under P17 (framebuffer.jl L286,
  L472; textures.jl L188; api.jl L295; pipeline.jl L609, L662, L704; acceleration.jl L402-414,
  L661-686, L703-733, L818-975, L1106-1129, L1219-1224, L1479-1485; raytracing/pipeline.jl L393).
- **Defensive code on known types:** listed under P2 (≈ 45 `=== nothing`/`isa`/`@assert` sites and
  17 `Union{Nothing,…}`/`Any` fields).
