# Audit: src/Mantle.jl, src/phases.jl, src/memory/, src/sync/, src/runtime/

Scope: 19 files, 5978 lines, every line read. Comparison files read in part for
the duplication table: `src/vulkan/runtime/memory.jl`, `src/vulkan/runtime/device.jl`,
`src/vulkan/graph.jl`, `src/metal/memory.jl`, `src/metal/device.jl`,
`src/graph/lifetime.jl`, `src/graph/access.jl`, `src/graph/backend.jl`,
`src/graph/types.jl`, `ext/MantleROCmExt.jl`. Target: `docs/api.md` (all 768
lines) and `docs/recording-plan.md` sections Rules, Devices, Arrays, Resources,
Queues, Ordering, Core types, Memory, Threads, What a device type implements,
Forbidden moves, phases 2 and 8. Every number not from a line count is marked
"estimate".

Headline measurements (awk over the 19 files, a line is comment/doc if it is a
`#` line or inside a `"""` block):

| | lines |
|---|---|
| total | 5978 |
| comment and docstring | 3070 (51 %) |
| code | 2082 |
| blank | 826 |

So the slice is 2.1k lines of code carrying 3.1k lines of prose. Of the code,
the inventory below puts ~1.6k lines in "survives" (most of it rewritten under
the target's names), ~40 in "backend verb" and ~4.3k lines (code and the prose
attached to it) in "belongs nowhere". The reasons it grew, in order of line
cost: (1) a backend axis that is not the device type (markers, probes,
accessor protocols, `@static`), with its fallbacks and its forwarding layers;
(2) three generations of memory lifetime engines kept side by side (pool
fence, channel stamp, Vulkan deferred list) and two pressure policies; (3)
prose that records measurements and bug history instead of contracts.

## 1. Inventory

Columns: lines; what it does; estimate of lines per bucket (a) survives into
the target core, (b) becomes a device-type verb, (c) belongs nowhere (dead,
superseded by the target, compat shim, test-only, debug-only, duplicate). The
bucket figures include the prose attached to the code, so they sum to the file
length. "Maps to" names the `api.md` type or verb.

| file | lines | what it does | (a) | (b) | (c) | maps to (api.md) |
|---|---|---|---|---|---|---|
| `src/Mantle.jl` | 671 | Module: KI re-exports (17-104), dependency `using`s (119-161), include order (168-274), ~150 lines of `export` (285-437), `argmap` (506-510), `basealignment` (524-536), `capacity`/`recorded`/`storage` fallbacks (443, 474, 553), `@static` backend includes (584-602), `Device()`/`devices()`/`Device(select)`/`defaultdevice!`/`__init__` platform switches (615-669). | 130 | 0 | 540 | `Device()`, `devices()` (2.1); `basealignment` (App. A "core: part of the selection key"); `argmap` (memory: "long argument tuples allocate"); the export list is mostly App. A "goes" names (`hold!`, `oneshot!`, `begin_pass!`, `trace_rays!`, `blit!`, …) |
| `src/phases.jl` | 871 | The graph compiler's phase pipeline: `Dag` (209-243), `Schedule` with RPS packed scores (253-385), `loopdeps`/`loopspans` (395-438), `Liveness` (441-530), `Place` with arena-part splitting (535-624), `giveup!` (642-649), `Aliasing` (658-709), `Barriers` (741-864), `compile!` (866-871); plus the `Compilation` accessor protocol (63-200). | 450 | 0 | 420 | `dependencies!`, `barriers!` (api.md:238), `costs!`/`schedule!` (256), `place!` (683); `Analysis.peak/naive` → `peakbytes(g)`/`naivebytes(g)` (84). `materialize!` goes (683), `deviceslice` goes (634), `syncbackend` goes (745), tenancy by id replaces `giveup!` (plan:1832) |
| `src/memory/pool.jl` | 1377 | Block suballocator: `FreeSpans` bins (158-203), `Block`/`Region` (220-245), `Arena` + weak tenants (265-313), `Inbox` (323-344), `PoolPolicy` + timers (346-411), `Pool` (423-474), `headroom`/`arenaroom`/`largestfree` (523-570), `carve!`/`acquire!`/`underpressure`/`carveorgrow!` (597-753), `release!` (762-789), `fence`/`passed`/`waitfor` decls (804-832), `retire!`/`reclaim!` (859-944), `makeroom!`/`autotrim!`/`reclaimable`/`autocollect!`/`trim!` (960-1098), `reserve!`/`tenant!`/`untenant!`/`sharing` (1134-1256), move listeners `notify_move!` (1274-1377). | 330 | 20 | 1030 | `Pool`, `Inbox` (api.md:230); `acquire!` with the pressure steps (480); `rawalloc`/`rawfree` verbs (290-291); `retire!(pool, x, tokens)` (365); `reserved(dev)` (179); `budget(dev)` replaces `capacity`/`headroom`/`arenaroom` (260, plan:1838); `caps(dev)` replaces `maxalloc` (684); cells replace `notify_move!`/`patchable` (212, 626); sparse arena parts replace `Arena`/`remap!`/`remappable`/`movable` (plan:1296-1316, 1837); `fence` goes (655); `passed`/`waitfor` become channel verbs (275-276) |
| `src/memory/resources.jl` | 650 | `Buffer{T,N}` and `GPURef{T}` persistent resources (29-90), `upload!`/`download`/`deviceview`/`hostspan`/`hostview` (92-171), host-addressable transfer trio (189-204), `bufferusage`/`Persistent` (206-221), constructors (226-274), `persistentarray` (289-292), Base forwarders (309-363), `update!` (365-376), `setindex!` stores ×3 (414-488), `isdirty`/`anydirty`/`landstores!` ×4 (491-566), `resize!` (582-593), `retireonce!`/`free!` (609-639), `devicecopy!` (650). | 110 | 10 | 530 | `MantleArray` (2.2), `GPURef` (125), `Store` change (61, 218), `hostspan` core (663), `devicearg` verb (339); `upload!`/`download`/`devicecopy!`/`update!`/`indexbuffer`/`holdleaves!`/`awaitwrites` go (759, 636, 629, 665, 662, 594); "no host-written graph arrays" (127) removes the `TransientBuffer` store path |
| `src/memory/array.jl` | 204 | `DeviceArray{T,N}` region handle (31-59), `allocate` with `autotrim!`/`autocollect!` (71-85), four `caps` field readers (111-157), `devicearray` ×4 (172-197), `Adapt.adapt_storage(::Device)` (204). | 35 | 0 | 170 | storage field of `ArrayState` (211); `caps(dev)` read directly (256); `devicearray` goes (627); `adapt(Device(), x)` (51) |
| `src/memory/placement.jl` | 99 | Interval-graph placement: `conflicts` with gaps (7-14), `LowestFit`/`BestFit` (26-35), `place` (46-76), `choose` ×2 (79-99). | 70 | 0 | 29 | `place!` (683). `BestFit` kept "not because it is recommended" (31-35): test/bench only (test/runtests.jl:142, 168, 180) |
| `src/memory/bound.jl` | 96 | `maxload` lower bound (13-27), `hmax`, `humanbytes` (35-40), `checkcapacity` error text (61-83), `fragmentation` (93-96). | 30 | 0 | 66 | `OutOfDeviceMemory` message (480). `maxload`/`fragmentation` readers: test/runtests.jl and docs/internals.jl only |
| `src/memory/model.jl` | 75 | `OffsetWindow`/`Gap` (6-27), `Item` (39-51), `segments` (54-65), `Problem`/`Placement` (67-75). | 35 | 0 | 40 | placement input. `Gap` has no producer in `src/` (only `csv.jl:21`) and says so at 19-22 |
| `src/memory/csv.jl` | 63 | MiniMalloc benchmark CSV reader (`parsegap`, `readproblem`). Readers: `test/runtests.jl:98-178` only. | 0 | 0 | 63 | none; test fixture living in the runtime and exported (`Mantle.jl:295`) |
| `src/memory/interval.jl` | 29 | `Span` half-open interval. `from_closed` (29) has no caller in src (guess from grep breadth). | 26 | 0 | 3 | liveness intervals |
| `src/sync/usage.jl` | 207 | `Access{RD,WR}` (22-36), `Src`/`Dst` (46-47), usage singletons (50-74), `Storage`/`ColorAttachment`/`Depth` (79-103), `kindof`/`reads`/`writes`/`discards` tables (106-149), `aliasable`/`evictable` (153-154), `Unordered`/`Traced` wrappers and `traced` (164-207). | 170 | 0 | 37 | `Access` (215); `TraceRead`/`TraceBuild` kept (plan:231); `Predicated` goes with conditional rendering (743); `aliasable`/`evictable` are constants nothing specialises |
| `src/sync/transition.jl` | 150 | `Transition` (16-21), `barrierhazards` (53-60), `ResourceState` (75-82), `needs_transition` default (95-105), `transition!` (118-131), `transitions` test walker (142-150). | 100 | 0 | 50 | `barriers!` core (694), `Deps` (223). The `be::Backend` parameter threaded through every call is the marker axis (section 3.1) |
| `src/sync/backend.jl` | 51 | `Backend` marker types `VulkanAPI`/`MetalAPI`/`WebGPUAPI`/`ROCmAPI` (10-31), `stages`/`access`/`layout` decls (33-35), `needs_transition` = false for WebGPU and ROCm (39, 51). | 0 | 3 | 48 | `stages`/`access`/`layout` internal to the device type's barrier helper (583, 678, 726); `backend`/`syncbackend` go (595, 745). `WebGPUAPI` has no backend in tree (comment at 22-25) |
| `src/runtime/api.jl` | 451 | `Device`/`Resource` abstracts (24-25), `todevice`/`backend` normalisation (65-66, 123-125), `Window` abstract + default-backend constructor (89-105), `screenshot` (135), `LoadOp` (149-166), `Surface`/`Attribute`/`draw!`/`render!` decls (168-199), `newpass`/`handle`/`dispatches` (216-218), `checklive` (232-238), `IdTable` (253-280), `copy!`/`run!`/`free!` decls (288-302), `PassTiming`/`timings`/`NSAMPLES` (311-366, 429), `importhost`/`importablehost`/`canimporthost` (387-424), `stride`/`count` (439-442), `Transient` module (448-451). | 30 | 0 | 420 | `Device` (199), `draw!` (158), `free!` (58, 81), `run!` (80); `Window(dev,w,h)` (173); `Array(win)` replaces `screenshot` (720); `clear!` replaces `LoadOp` (159); `attribute` replaces `stride` (163); `freed` flag replaces `checklive` (392); profiling open (541); resource ids become the array state (211, 447); `importhost` has no entry (section 4) |
| `src/runtime/dispatch.jl` | 319 | `Dispatch` record (14-19), `DeviceRange` (70-79), `refuserefs` (105-121), `passof`/`graphof`/`touch!` hooks (131-151), `d2dcopy_kernel!`/`fill_kernel!` (193-211), `dispatch!` launch and call forms (235-297), `iscall`, `indirectcount!` (306-319). | 35 | 0 | 284 | `dispatch!` (107), `KernelNode` (205); `launchrange(a)` replaces `DeviceRange` ("There is no DeviceRange", 67); GPUArrays serves `fill!`/`copyto!` (72); the call form becomes the selection table (115-119; `runscalls` goes, 718) |
| `src/runtime/lifecycle.jl` | 288 | `DebugConfig` (66-117), `debugenv`/`validating`/`validationmessages!`/`submissions` (133-187), `retire!(::Device)`/`retired`/`refuseretired`/`reset_device!` (208-251), `trydevicearray`/`tryallocate` (271-288). | 40 | 0 | 248 | device teardown open (561, 713); `submissions(g)` is a graph verb (82), not a device counter; nothing for validation or a non-throwing allocation probe (section 4) |
| `src/runtime/backendhooks.jl` | 169 | `staged_gemm_tile` (50), video decode API stubs (68-128), `supports_float64`/`supports_int64_atomics` (142-156), `isoutofmemory` (169). | 0 | 10 | 159 | selection table replaces `staged_gemm_tile` (725); `caps(dev)` fields replace `supports_*` (735); `VideoDecodeNode` (205) and the video queue kind (259) exist with no API (section 4); `isoutofmemory` goes: `rawalloc` throws only on device loss, budget decides (248, 260) |
| `src/runtime/backends.jl` | 168 | `BackendProbe`/`BACKEND_PROBES`/`register_backend!`/`defaultbackend`/`availablebackends`/`eachbackend` (13-17, 96-168), `DeviceInfo`/`devices(api)`/`defaultdevice!`/`selectdevice`/`admits` (36-94). | 25 | 0 | 143 | `REGISTRY`/`DeviceRegistry` (200), `hardwarekeys(T, registry)` (254), `devices()` (33); `defaultdevice!` goes (622); `backend` goes (595) |
| `src/runtime/format.jl` | 40 | A 30-line module-level string (1-30, attached to nothing) and `pixelbytes` (34). | 5 | 0 | 35 | `imageformat(dev, T)` verb (357) |
| **sum** | **5978** | | **1621** | **43** | **4315** | |

`src/phases.jl`, what it is: the graph compiler's phase list, run as
`compile!(Compile(g; alias, coalesce, policy))` from `src/graph/build.jl:787`
and `:1541`; `Pipelines` has its body in `src/graph/kalaunch.jl:419`. It is not
a precompile workload (PrecompileTools is only re-exported, `Mantle.jl:121-127`)
and has nothing to do with the plan's numbered phases (plan:1689). The design
needs its algorithms: the dependency DAG, the schedule, liveness, placement,
the aliasing handover barrier and the per-resource transition walk are what
api.md calls `dependencies!`, `schedule!`/`costs!`, `place!` and `barriers!`
(api.md:238, 256, 683) and the plan keeps "compile steps, placement" in core
(plan:1405). It does not need: the `Compilation` abstract type and its ten
accessor functions written so that "a backend subtypes this for its own
context" (phases.jl:63-200; the one subtype is `Compile{D}`,
`src/graph/types.jl:800`), `materialize!` and `deviceslice` (176-196; api.md:683,
634), `giveup!` (642-649; tenancy by id, plan:1832), the `alias = false`
bisection flag (149-153, 526), the `Policy`/`Compact`/`Overlap` bit-shift
scoring (255-270; the target schedules on a roofline estimate, api.md:256;
`Compact()` has no caller outside tests), and the `syncbackend` marker
(743, 811). `Schedule` (286-385, 100 lines) and `Barriers` (741-864, 124
lines) each mix three concerns and will be rewritten rather than moved.

## 2. Duplication

Pairs and groups doing the same thing. Line estimates are what a single
implementation recovers (estimate, ±30 %).

| # | what | my side | other side(s) | replacement | lines |
|---|---|---|---|---|---|
| A | OOM-retry around an allocation: try, catch "out of memory", collect/trim/drain, try again, give up | `pool.jl:687-695` `underpressure` + `:960-965` `makeroom!` | `vulkan/runtime/memory.jl:1140-1172` `acquire_or_reclaim!` (try `acquire!`, catch `isoutofmemory`, `GC.gc(true)`, `drain!`, quiesce, trim, retry, `LavaError`); `:391-440` `vk_alloc` (drain, `maybe_collect`, try, quiesce+trim, retry, `@info`) | one `acquire!` with the pressure steps in order (api.md:480); `rawalloc` is "a driver call" that checks the budget first (api.md:260, 290), so no exception probe | 90 |
| B | Proactive collection and trim policy with timers and a soft cap | `pool.jl:346-411` `PoolPolicy`, `:989-1068` `autotrim!`/`reclaimable`/`autocollect!` | `memory.jl:253-320` `maybe_collect` (its own ratio-gated collector, `MemoryStats`, 10 s heap re-probe), `:245-251` `trim_gpu_pool!`, `:909-911` `MemoryPolicy` with `eager_gc`/`track_allocs`, `:1024-1063` alloc-site tracing; `memory.jl:100-110` says `autotrim!` is "core's now" yet `:261` calls it from inside `maybe_collect` and `:1109-1110` calls `autocollect!` beside it | pressure policy only in `acquire!` (plan:1340); see section 4.1 for what is lost | 150 |
| C | Deferred release once the GPU has passed | `pool.jl:859` `retire!(::Pool)` + `:888-944` `reclaim!` keyed on a device-wide `fence` (804-832) | `graph/lifetime.jl:486-496` `retire!(::SubmitChannel, obj)`, `:518-588` `reclaim!(ch)` keyed on a per-resource `Stamp` (71-76), `:591-598` `drain!`; `memory.jl:52-54` `BUF_STATE_*`, `:637-700` `vk_free!`/`destroy_buffer!` deferred list; `memory.jl:874-881` states that the two layers coexist on purpose | one release engine: `retire!(pool, x, tokens)` with last-use tokens, checked at every `acquire!` and `run!` (api.md:365; plan:1236-1240) | 180 |
| D | Keeping a resource alive across submissions | `resources.jl:42, 48, 609-614` `@atomic retired` + `retireonce!` finalizer; `:307` `holdleaves!(holder, ::Buffer)`; `pool.jl:269` WeakRef tenants, `:453` WeakRef `movelisteners` | `lifetime.jl:74` `Stamp.holders`, `:260-425` `hold!`/`takeholds!`/`unhold!`/`pushholds!`/`dropholds!` hold-frame stack | holds as a CAS count (api.md:367), `holdleaves!` goes (662) | 100 |
| E | Host↔device transfer | `resources.jl:92-137` `upload!`/`download`/`deviceview`/`hostspan` decls, `:160-204` `hostview` + `hostupload!`/`hostdownload`/`hostdevicecopy!` | `metal/memory.jl:51-78` forwards its three verbs to those; `vulkan/graph.jl:331-339` the same three over `deviceview`; `ext/MantleROCmExt.jl:505-530` again; `memory.jl:1231-1335` `copy_buffer!`/`upload!(::VkManagedBuffer)`/`download!`/`upload_typed!`/`download_typed!`; `vulkan/array/lavaarray.jl:264` `upload!(::LavaArray)` | stores are `Store` changes written by `emitstore!`/`emitcopy!` (api.md:61, 318), host reads are `Array(a)` (64); `upload!`/`download`/`devicecopy!` go (759, 636, 629) | 120 |
| F | "Is this error out-of-memory" and "allocate or return nothing" | `backendhooks.jl:169`; `lifecycle.jl:271-288` `trydevicearray`/`tryallocate` | `memory.jl:1176-1177`; `vulkan/graph.jl:195-199`; `metal/device.jl:1456-1467` (catches `OutOfMemoryError`, returns `nothing`) | budget check before `rawalloc` (api.md:260); the probe has no replacement (section 4.5) | 25 |
| G | The default device, per backend | `backends.jl:96-135` `BACKEND_PROBES` + closures; `Mantle.jl:615-653` platform switch | `vulkan/graph.jl:51` `DEVICES` + `memory.jl:923` `lavadevice(ctx)`; `vulkan/runtime/device.jl:819, 875-881, 922-930` `VK_CONTEXT_REF`/`bind_context!`/`VK_CONTEXT_LOCK`/`VK_CONTEXT_COUNTER`/`vk_context()`; `metal/device.jl:297-306, 337` `METAL.default`; `ext:252-261` `ROCM_DEVICE`. Each `Device(::API; select, debug)` has the same "both nothing → cached, else build" shape: `vulkan/graph.jl:57-63`, `metal/device.jl:308-321`, `ext:254-261` | `REGISTRY` and `T(key)` once per key (api.md:200, 255, 524) | 150 |
| H | Enumerating hardware and matching a selector | `backends.jl:36-94` `DeviceInfo`/`selectdevice`/`admits` | `vulkan/runtime/device.jl:2012-2019`, `metal/device.jl:341-343`, `ext:266-270`; three call sites repeat `selectdevice(select, devices(API()))` | `hardwarekeys(T, registry)` (api.md:254) | 15 |
| I | Lifecycle verbs with a default and a per-backend answer | `lifecycle.jl:145, 165, 217` defaults | `vulkan/graph.jl:150-166`; `metal/device.jl:1427-1433`, `metal/debug.jl:36, 97` | a required verb per type or nothing (api.md:21) | 10 |
| J | Token wait and pass, forwarded through handles | `pool.jl:804-832` decls | `vulkan/graph.jl:271-306` (device → `bq`), `metal/device.jl:498-500` (device → queue) + two queue types `:508-541`, `:1113-1147`; `ext:389-410`; `graph/backend.jl:569-575` `waitfor!(dev,tok)`; `lifetime.jl:464-469` `waitfor!(::Stamp)`; `vulkan/runtime/command.jl:1067` `waitfor!(bq,tok)` | `Token = (channel, value)`, `passed(ch, v)`, `waitfor(ch, v)` (api.md:217, 275-276) | 60 |
| K | Memory kinds and constraints | `pool.jl:57-104` decls + defaults; `resources.jl:221` `Persistent` | `vulkan/graph.jl:254-256` (`Persistent` = `Buffers`), `:806-807`, `:847-848`, `:894-916`; `metal/device.jl:364-388`, `:403` `Union{Buffers,Persistent}`; `ext:339-342` (`nothing`/`true` stubs), `:360` | kinds `Buffers`, `Unified`, `Readback`, `Images`, `Accel` with a usage constraint (api.md:290; plan:1231) | 30 |
| L | A second type axis for barriers | `sync/backend.jl` whole; `transition.jl:95, 118, 142` `be::Backend` threading; `phases.jl:743, 811` | `syncbackend(::LavaDevice)` `vulkan/graph.jl:956`, `metal/ka.jl:51`, `ext:289`; `graph/backend.jl:524` | the device type in the barrier helper (api.md:583, 678, 726) | 60 |
| M | Two encodings of "read/write/atomic" | `usage.jl:22-36` `Access{RD,WR}` type-level, `:164-169` `Unordered{U}` | `graph/access.jl:31-54` `Touch(read, write, atomic)` value-level, converted at `:67` | one `Access` (api.md:215) | 30 |
| N | Constants repeated | `pool.jl:573` `alignup` and `placement.jl:1`; `phases.jl:147` `blocksize = 64 << 20`; `pool.jl:1109` `REGION_ALIGN = 256`, `:654`, `array.jl:72`, `resources.jl:291` `align = 256` | `metal/device.jl:1323`, `ext:380` `64 << 20`; `memory.jl:888` `POOL_BLOCK_SIZE`, `:904` `POOL_ALIGN` | core's block size (api.md:605); alignment in `rawalloc`'s constraint (587) | 10 |
| O | The pending-store protocol | `resources.jl:414-428` (GPURef seqlock), `:430-446` (Buffer), `:463-485` (TransientBuffer, identical but for the capacity check), `:520-566` four `landstores!` | fields repeated in `graph/types.jl:275-277` (`pending`, `pendinglock`, `@atomic dirty`) | one `Store` change in the sync state (api.md:61, 218); graph arrays take no stores (127) | 80 |
| P | `reset_device!` | `lifecycle.jl:232-251` | `vulkan/graph.jl:177-192` (device form), `vulkan/runtime/device.jl:993-1010` (context keyword form), `metal/device.jl:1443-1448` | open (api.md:561) | 20 |
| Q | Two handles for one device | `api.jl:65-66, 123-125` `todevice`/`backend`; `array.jl:178, 204`; `resources.jl:227, 235, 267, 273` | `vulkan/graph.jl:77` `Device(::LavaBackend)`, `:120, 126` `backend`; `metal/device.jl:341-344, 1380-1386`; `lifetime.jl:205` | `Device <: KA.Backend`, one object (api.md:34-35) | 30 |
| R | Trim wrapped twice to count | `pool.jl:1078-1098` `trim!` | `memory.jl:1191-1200` `reclaim_empty_pool_blocks!` wraps it to count, `:245-251` `trim_gpu_pool!` wraps that | `trim!` is a pressure step; `reserved(dev)` before and after is the count (api.md:179) | 15 |

Total recoverable (estimate): about 1,175 lines across my files and the
comparison files.

### 2.1 How many backend registry / hook / dispatch mechanisms exist

Seven distinct mechanisms select behaviour by backend; none of them is
"a method on the device type":

1. **Probe registry.** `BackendProbe`, `BACKEND_PROBES`, `register_backend!`,
   `defaultbackend()` (`runtime/backends.jl:13-17, 96-135`): a priority-sorted
   list of closures that return a KA backend or `nothing`; registered from
   `vulkan/vulkan.jl:147`, `metal/metal.jl:90`, `ext/MantleROCmExt.jl:1473`.
2. **"Declared here, answered there".** `function f end` with no method, a
   `MethodError` if a backend forgets: 4 in `backendhooks.jl`, 18 in
   `runtime/api.jl`, 3 in `lifecycle.jl`, 3 in `sync/backend.jl`, 9 in
   `pool.jl`, 6 in `resources.jl`, 17 in `phases.jl`, 22 in `graph/backend.jl`
   (grep `^function [a-z!]+ end`). The rationale is written at
   `backendhooks.jl:1-27`.
3. **Default-returning hooks.** The same shape with the opposite failure mode
   (silent): `recorded(plan) = false` (`Mantle.jl:443`), `capacity(dev) =
   typemax(Int)` (553), `maxalloc(dev) = typemax(Int)` (`pool.jl:54`),
   `remappable(x) = true` (313), `validating(dev) = false`
   (`lifecycle.jl:145`), `validationmessages!(dev) = String[]` (165),
   `retired(dev) = false` (217), `debugenv(x, cfg) = []` (133),
   `videodecodes(dev) = false` (`backendhooks.jl:68`), `supports_float64 =
   false` (142), `supports_int64_atomics = false` (156), `canimporthost =
   false` (`api.jl:424`), `kindof(…) = nothing` (`usage.jl:116-118`),
   `countresource(::Any) = nothing` (`dispatch.jl:79`), `patchable(::Device) =
   true` (`graph/backend.jl:449`), `needs_transition(::Backend)` (`transition.jl:95`),
   `mergeconstraints` equality default (`pool.jl:90`). Seventeen; api.md:21
   forbids every one.
4. **Marker-type dispatch.** `Backend` singletons `VulkanAPI()`, `MetalAPI()`,
   `WebGPUAPI()`, `ROCmAPI()` (`sync/backend.jl:10-31`) obtained from a device
   through `syncbackend(dev)` (`graph/backend.jl:524`) and used as the first
   argument of `stages`/`access`/`layout`/`needs_transition`
   (`vulkan/lowering.jl:47-208`), of `initbackend!(api)`
   (`graph/backend.jl:793`) and of `Device(::VulkanAPI; …)`.
5. **The vocabulary allowlist.** `BACKEND_VOCABULARY`, 189 symbols a backend
   may add methods to, checked by a test (`graph/backend.jl:795-806`).
6. **Accessor-override protocol.** `Compilation` with `analysis(c) = c.analysis`
   and nine more "a backend whose context is shaped differently overrides the
   accessor" (`phases.jl:63-200`); `passof(h) = h.pass`, `graphof(h) = h.graph`
   "a backend only overrides if it names them otherwise" (`dispatch.jl:127-132`);
   `touch!(g, x)` "a backend specialises this for its transients" (144-151).
7. **Platform switch.** `@static Sys.isapple()` / `VK.HAS_LOADER` ten times in
   `Mantle.jl` (584-602 include block, 615 `Device()`, 623 `devices()`,
   644-653 `Device(select)`, 665-669 `__init__`).

Beside these, the four "which object is the device" handles (`Device`, KA
backend, `SubmitChannel`, `VkContext`) are reconciled by conversion functions
(`todevice`/`backend`/`Device(::LavaBackend)`/`vk_context(b)`,
`api.jl:65-66, 125`; `vulkan/graph.jl:77, 120`; `metal/device.jl:341`), which
is dispatch by normalisation.

The one mechanism that replaces all seven: `abstract type Device <: KA.Backend`
with concrete `LavaDevice`, `MetalDevice`, `CudaDevice`, `ROCmDevice`
(api.md:34); every verb of api.md §4 takes the device, or a channel/builder of
it, as its first argument and has a required method per joined type
(api.md:21, 245); the registry enumerates device types and hardware keys
(api.md:200, 254-255), so there is no probe list, no marker, no accessor
protocol, no default method, and `@static Sys.is*` only in the include block
(api.md:527). The vocabulary check becomes "a device type defines only §4's
verbs" (api.md:434), a test over the spec instead of a tuple in the source.

## 3. Patterns that inflate the code

Each pattern: examples with file:line, and its line cost in my files
(estimate).

### 3.1 Branching on the backend by hand instead of dispatch on the device type

- `@static Sys.isapple()` deciding which `Device(API)` to call:
  `Mantle.jl:615, 623, 645-652, 666-667`.
- `syncbackend(g.dev)` fetched once and threaded as `be` through
  `transition!`: `phases.jl:743, 811`; `transition.jl:118-131, 142-150`.
- `patchable(pl.graph.dev)` branch deciding patch-vs-invalidate: `pool.jl:1355`.
- `basekind(kind) isa Images && invalidate!(t)` and `old === nothing ||
  basekind(kind) isa Images || notify_move!(…)`: `pool.jl:1174, 1196`.
- `supports_batch_queue(dev) && flush!(dev)`, `supports_batch_queue(dev) &&
  holdsretired(batchqueue(dev))`: `pool.jl:962, 1033` (a capability probe
  deciding a code path inside the pool).
- `t isa Plan` to decide whether a tenant can be invalidated: `pool.jl:1173`.
- Cost: ~80 lines plus all of `sync/backend.jl` (51).

### 3.2 Defensive checks on values whose type is known

Counts by grep: `=== nothing`/`!== nothing` 24 in `pool.jl`, 12 in
`phases.jl`, 3 each in `model.jl`, `transition.jl`, `backends.jl`, 2 in
`dispatch.jl`; `isa` 3 in `pool.jl`, 2 in `resources.jl`, 1 each in
`phases.jl`, `backendhooks.jl`.

- `Union{Nothing,…}` fields that make every reader branch: `Arena.region`
  (`pool.jl:266`), `InboxNode.next` (326), `ResourceState.current`
  (`transition.jl:76`), `Gap.window` and `Item.offset` (`model.jl:26, 45`),
  `DeviceRange.max` (`dispatch.jl:72`).
- `data isa Vector{T} ? data : convert(Vector{T}, data)` twice
  (`resources.jl:437, 479`); `convert` alone does both.
- `r isa BufferRange || return nothing` (`phases.jl:461`), `err isa
  OutOfMemoryError` (`backendhooks.jl:169`).
- Re-checking a WeakRef after a `filter!` because the filter "is a SNAPSHOT,
  not a guarantee" (`pool.jl:1339-1349`): the WeakRef list is the defensive
  structure; the plan's tenancy by id (plan:1832) removes it.
- `a.region === nothing ? req : mergeconstraints(…)` (`pool.jl:1147`),
  `a === nothing && return nothing` (`pool.jl:1231`), `get(byid, id, nothing)`
  chains (`phases.jl:457-463`).
- Cost: ~120 lines.

### 3.3 `try/catch` that swallows, logs, or probes

- `underpressure` catches, asks `isoutofmemory(err)`, rethrows otherwise
  (`pool.jl:688-692`): a typed catch, but the exception is the OOM signal that
  api.md:14 and :290 replace with a budget check before the driver call. The
  same probe exists twice more in `vulkan/runtime/memory.jl:1142-1147,
  1161-1167`.
- `tryallocate` on Metal returns `nothing` from a catch
  (`metal/device.jl:1457-1462`), the "merely returns a default" case.
- In my files no catch swallows or only logs; `reclaim!` (`pool.jl:896-943`)
  and `autotrim!` (1005-1009) are `try/finally`. Cost: ~15 lines, but pattern A
  of section 2 hangs on it.

### 3.4 Globals and who should own the state

In my files (all upper-case, so the naming rule holds): `BACKEND_PROBES`
(`backends.jl:96`, the only mutable one), `PHASES`, `EMPTY_HANDOVER`,
`SCORE_MAX` (`phases.jl:724, 727, 273`), `NBINS`, `BIN_SCAN`, `REGION_ALIGN`
(`pool.jl:107, 587, 1109`), `TRANSIENT_ALIGN_FLOOR` (`Mantle.jl:536`),
`NSAMPLES` (`api.jl:429`), `DeviceSelector` (`Mantle.jl:628`), `KA`, `KI`
(`Mantle.jl:104, 153`). `REGION_ALIGN`, `BIN_SCAN`, `TRANSIENT_ALIGN_FLOOR` and
`NSAMPLES` are each read in one place (`pool.jl:605-606, 610`, `Mantle.jl:535`,
`api.jl:345-349`) and are the "const used only once" the owner's rules forbid.

The state that matters is in the backends and is "current device" state four
times over: `DEVICES` (`vulkan/graph.jl:51`), `VK_CONTEXT_REF` +
`VK_CONTEXT_LOCK` + `VK_CONTEXT_COUNTER` (`vulkan/runtime/device.jl:819, 881,
922`), `METAL` (`metal/device.jl:306`), `ROCM_DEVICE`
(`ext/MantleROCmExt.jl:252`); plus 24 more `const … = Ref/Dict/Lock` in `src/`
(`vulkan/runtime/pipeline.jl:43, 122, 125, 187`, `profiling.jl:164, 440`,
`metal/graphics.jl:599-600, 899`, `metal/mesh.jl:323-324`, `metal/hwtlas.jl:97-98`,
`metal/procedural.jl:210-211`, `array/gemm.jl:615-1034` six kernel caches,
`array/gemv.jl:126, 136, 230`, `array/fft.jl:570`, `ext:217`). The plan names
most of these for deletion (plan:2001-2008). Owner in the target: `REGISTRY`
for devices (api.md:524), the device object for its pool, tables and caches
(api.md:229-230), the `Kernel` at its `CodeInstance` for compiled kernels
(api.md:208). `PoolPolicy` (`pool.jl:389-411`, 15 fields of timer state "read
and changed as `pool(dev).policy.trim_threshold = …`", 350) is per pool, but it
is configuration-as-mutable-state; the target has no knobs there
(plan:1340-1352). One `_`-prefixed global exists outside my files:
`_PIPELINE_CACHE_ATEXIT_REGISTERED` (`vulkan/runtime/pipeline_cache.jl:167`).

Cost in my files: ~45 lines; in the registries across backends ~60.

### 3.5 Near-copies of a function for different kinds

- `setindex!` for `Buffer` (`resources.jl:430-446`) and `TransientBuffer`
  (463-485): the same 17 lines with one check changed; `landstores!` ×3
  (520-560) plus a `@generated` tuple form (564-566); `isdirty`/`anydirty`.
- `Buffer(dev, data::AbstractVector)` (226-232) and `Buffer(dev,
  data::AbstractArray)` (265-270).
- `acquire!` (653-665) and `reserve!` (1134-1201), both `underpressure` around
  `carveorgrow!`; `headroom` (554) and `arenaroom` (570).
- `similar` ×6 (`resources.jl:322-328`), `devicearray` ×4 (`array.jl:172-197`),
  `reshape` ×2, `size` ×2.
- `decode_h264_gpu` and `h264decoder` throw stubs (`backendhooks.jl:77-78,
  102-103`).
- `alignup` (`pool.jl:573`, `placement.jl:1`).
- Cost: ~130 lines.

### 3.6 Manual loops that are a map, filter or reduce

- `tenants!` compaction (`pool.jl:274-283`) is `filter!(wr -> wr.value !==
  nothing, a.tenants)`, which `untenant!` (1232) and `listen_moves!` (1276)
  already write.
- `largestfree` (523-540): a `maximum` over one bin.
- `touches[id] = get(touches, id, 0) + 1` over `ps` (`phases.jl:303-306`): a
  count map; `loopdeps` (395-412) is a `union` per loop; the handover filter
  (850-855) is `filter`.
- `conflicts` (`placement.jl:10-13`) is `any`.
- Cost: ~40 lines.

### 3.7 Forwarding layers

- `todevice`/`backend` (`api.jl:65-66, 125`) and the normalising call at every
  constructor (`resources.jl:227, 235, 267, 273`; `lifecycle.jl:272`);
  `devicearray(d::Device) = devicearray(backend(d), …)` (`array.jl:178`);
  `indexbuffer(b::KA.Backend) = indexbuffer(Device(b), …)` (`resources.jl:260-261`);
  `Adapt.adapt_storage(d::Device, x) = adapt_storage(backend(d), x)` (`array.jl:204`).
- `Window(w, h) = Window(defaultbackend(), w, h)` (`api.jl:104-105`);
  `Device(select)` → `Device(API; select)` (`Mantle.jl:644-653`);
  `defaultdevice!(select) = defaultdevice!(Device(select))` (663).
- `decode_h264_luma` → `decode_h264_gpu` (`backendhooks.jl:86-89`).
- `hostupload!`/`hostdownload`/`hostdevicecopy!` exist so a backend can
  forward three verbs to them (`resources.jl:173-204`; `metal/memory.jl:51-78`).
- On the other side: `passed(d::LavaDevice) = passed(d.bq, f)`
  (`vulkan/graph.jl:291`), `fence/passed/waitfor(d::MetalDevice) = …(d.queue)`
  (`metal/device.jl:498-500`), `videodecodes(dev) = videodecodes(dev.ctx)` and
  `videodecodes(b::LavaBackend) = videodecodes(vk_context(b))`
  (`vulkan/graph.jl:137`, `vulkan/array/ka_backend.jl:115-116`).
- Root cause: four handles for one device (section 2, Q). Cost: ~60 lines in
  my files, ~40 in the backends.

### 3.8 `_`-prefixed functions

None in my files. (`_PIPELINE_CACHE_ATEXIT_REGISTERED`,
`vulkan/runtime/pipeline_cache.jl:167`, is the one hit in `src/`.)

### 3.9 Compat shims and feature flags

- `alias` "a bisection tool rather than a tuning knob" (`phases.jl:149-153,
  526`); `Policy`/`Compact`/`Overlap` (255-270; `Compact()` has no caller in
  `src/`); `BestFit` "not because it is recommended" (`placement.jl:28-35`);
  `Gap`/`OffsetWindow` for a case "none of them can express today"
  (`model.jl:16-27`; no producer in `src/`).
- `PoolPolicy`'s seven knobs (`pool.jl:389-396`), `pool_disabled`
  (`lifecycle.jl:74`), `maywait` (`pool.jl:655, 687, 730-742`: an escape for a
  hold-stack bug the comment describes, which the target removes with the
  hold stack, api.md:421).
- `WebGPUAPI` with no backend in tree (`sync/backend.jl:19, 22-25`).
- The seventeen default-returning hooks of section 2.1 item 3.
- Cost: ~150 lines including their docs.

### 3.10 Comments that narrate history or measurements instead of the contract

51 % of the slice is prose (3070 lines). Contract text is a minority of it; the
rest records how a bug was found or what a benchmark showed:

- `Mantle.jl:132-139` (the `unsafe_free!` MethodError story, "125 of them in
  one suite run"), `468-472` ("SAM2Runner simply failed to build"), `570-582`
  ("400 lines of `import Mantle:`").
- `pool.jl:109-157` (a six-row benchmark table of vector vs bins), `366-388`
  (SAM 2 encoder table), `428-445` (boxed-field measurement in RayMakie),
  `675-680` (TRELLIS.2 on a 7900 XTX), `729-738`, `840-858`, `976-987`,
  `1145-1157`, `1163-1169`, `1342-1356`.
- `resources.jl:144-149` (`LightBVHNode` padding), `276-288` (RADV vs NVIDIA
  hang), `455-462`; `array.jl:95-110, 124-138, 142-157`; `api.jl:98-103,
  246-251, 398-406` (bandwidth figures); `backends.jl:158-162` ("57 times");
  `backendhooks.jl:15-27, 56-63`; `dispatch.jl:1-6, 85-96, 183-185, 220-224`;
  `lifecycle.jl:61-64`; `phases.jl:3-6, 219, 248-251, 280-281, 600-604,
  667-669, 676`.
- `format.jl:1-30` is a 30-line string literal attached to nothing.
- Cost: ~1,300 lines (estimate; `pool.jl` alone ~400). The plan's comment rules
  (plan:1672) and "docs state the target only" are the fix; the measurements
  belong in `experiments/` or the commit that made the change.

### 3.11 Functions over ~100 lines

- `run!(::Barriers, c)` `phases.jl:741-864` (124): segment cutting over views,
  state seeding from the schedule's end, the per-resource walk, the aliasing
  handover, all in one body with two closures (`segments_of`, `state`).
- `run!(::Schedule, c)` `286-385` (100): dependency counting, byte accounting,
  loop containment and the scoring loop.
- Near the line: `run!(::Place)` 543-624 (82), `reserve!` 1134-1201 (68),
  `reclaim!` 888-944 (57), `carveorgrow!` 704-753 (50), `notify_move!`
  1329-1377 (49).

### 3.12 Keyword-argument explosions

- `acquire!(pool, dev, kind, transients, bytes; align, blocksize, constraint,
  maywait)` (`pool.jl:653-655`); `reserve!(…; align, blocksize)` (1134-1135);
  `allocate(…; align, blocksize, constraint)` (`array.jl:71-73`). `blocksize`
  is a device property (`phases.jl:139-147`) threaded through four signatures
  (`phases.jl:607-608`, `pool.jl:654, 1135`, `array.jl:72`, `resources.jl:291`).
- `DebugConfig` with eight keywords and a copy constructor that restates all
  eight (`lifecycle.jl:76-83, 106-112`).
- Cost: ~40 lines.

### 3.13 Parallel data structures kept in sync by hand

- `Pool.retiring` / `Pool.retiring_at`, documented as "PARALLEL vectors"
  (`pool.jl:429-447`, written together at 911-912, deleted together at 937-938).
- `Analysis.regions` / `arenas` / `parts` (`phases.jl:54-58`, pushed together
  at 609-610, cleared together in `giveup!` 646-647).
- `FreeSpans`'s four indexes (`pool.jl:158-163`) plus `Block.live` (226), which
  exists because the free list cannot detect a double release (207-214).
- `IdTable.ids` / `by_id` (`api.jl:253-257`); `Placement.offsets` keyed by
  `string(i)` and read back by `string(i)` (`phases.jl:526, 583, 613`).
- Two WeakRef lists pruned in five places (`pool.jl:269, 453; 274-283, 1214,
  1232, 1276-1278, 1287, 1339`).
- `Buffer.len`/`capacity` beside `DeviceArray.dims` (`resources.jl:30-32`).
- Cost: ~90 lines.

### 3.14 Abstract types with one subtype

- `Compilation` → `Compile{D}` only (`phases.jl:90`; `graph/types.jl:800`),
  carrying the ten-accessor protocol (section 2.1 item 6).
- `Window` → `VulkanWindow` only in tree (`api.jl:89`).
- `Phase`'s seven singletons exist to dispatch `run!(phase, ctx)`, whose
  abstract method throws `MethodError` by hand (`phases.jl:20-22`).
- Cost: ~150 lines (the protocol and its docs).

### 3.15 `Any`-typed fields

Thirty occurrences in my files: `Block.memory/kind/constraint`
(`pool.jl:221-224`), `Arena.constraint` (268), `InboxNode.item` (324),
`Pool.blocks::Dict{Any,…}`, `arenas::Dict{Any,Arena}` (424-425),
`Analysis.regions/arenas/parts::Vector{Any}` (`phases.jl:54-58`), `Buffer.dev`
and `GPURef.dev` (`resources.jl:33, 77`, kept "so it can free itself",
298-301), `Dispatch.kernel/ndrange/group` (`dispatch.jl:15-18`),
`BackendProbe.probe` (`backends.jl:16`), `IdTable` (`api.jl:254-255`),
`Transition.from::Type`/`waits::Vector{Type}` (`transition.jl:18-20`),
`ResourceState.current::Union{Nothing,Type}` (76). The stated reason is to
avoid a type parameter cascading from `Pool` into the device types
(`pool.jl:436-444`), which is the backend axis again: a `Pool{M}` field in
`LavaDevice` is concrete once the pool belongs to the device type (api.md:230).
Cost: boxing on the run path (the comment at 429-435 measured 48 KB per frame
from one such field), not lines.

### 3.16 CSV and diagnostic code in the runtime

- `memory/csv.jl` (63): MiniMalloc corpus reader; readers `test/runtests.jl:98,
  99, 106, 120, 141, 167, 178` and nothing else; exported (`Mantle.jl:295`).
- `bound.jl` `maxload`/`hmax`/`fragmentation` (13-30, 93-96): readers
  `test/runtests.jl` and `docs/internals.jl`.
- `PassTiming` and its two `show` methods (`api.jl:311-332`), `NSAMPLES` (429).
- `PoolPolicy.gc_seconds/gc_count` counters (`pool.jl:404-405`),
  `Pool.blockgen` for Metal's residency (454-461), `submissions(dev)` counter
  (`lifecycle.jl:168-187`), `validationmessages!` (147-165).
- Cost: ~200 lines.

### 3.17 Pointers and unsafe calls

- `hostview`'s bounds-checked `unsafe_wrap` (`resources.jl:160-171`): one
  place, host-visible memory has no Julian type; keep as `hostspan`'s core
  helper (api.md:663).
- The hand-rolled seqlock around `Base.unsafe_convert(Ptr{T}, r.pending)` with
  `GC.@preserve` (`resources.jl:414-428, 520-532`) for a scalar store: replaced
  by a scheduled `Store` (api.md:126).
- `ccall(:madvise)` (`api.jl:410-414`): belongs with the mmapped-host path if
  it survives (section 4.4).

### 3.18 Name variants where dispatch was meant (owner's rule)

`upload!`/`hostupload!`, `download`/`hostdownload`,
`devicecopy!`/`hostdevicecopy!` (`resources.jl:189-204`: the host-addressable
case is a storage property, not a name); `devicearray`/`trydevicearray`
("`devicearray`'s refusing sibling", `Mantle.jl:184-185`); `acquire!`/`reserve!`;
`retire!`/`retireonce!`/`retireplan!`; `reclaim!`/`reclaimable`;
`tenant!`/`untenant!`/`tenants!`; `listen_moves!`/`unlisten_moves!`;
`waitfor`/`waitfor!`; `update!`/`setindex!`; `Buffer`/`indexbuffer`;
`decode_h264_gpu`/`decode_h264_luma`. Twelve families.

GeometryBasics rule: no vector math in these files; nothing to flag.

## 4. What the design docs miss, and what the code shows to be harder

### 4.1 Proactive collection and trim (no replacement)

The target puts the pressure policy "only in `acquire!`" and only when "an
allocation would exceed the device's budget" (plan:1340-1352; api.md:480). The
code has two more triggers and records why: `autocollect!` collects past a
soft cap before the pool grows, because Julia's GC never sees device bytes and
"SAM 2's encoder ran to 16 136 MiB across 200 blocks with nothing asking the
collector" (`pool.jl:366-373`); `autotrim!` hands empty blocks back on a timer
because "a 5-scene sweep on an iGPU sharing system RAM hit a driver timeout
without it" (368-369) and because an incremental collection never frees old
regions (976-987). A budget-only trigger reproduces both failures: on a
unified-memory device the budget is the system's RAM, and the driver times out
before the budget is hit. The plan also does not budget the cost of a full
collection inside `acquire!` during a frame ("~63 ms" per incremental
collection on a large heap, `pool.jl:1046-1048`; "six collections a run turned
142 ms of work into a p50 of 519 ms", 384-387).

### 4.2 Finalizer-driven free of plain arrays

Today every `Buffer`/`GPURef` has a finalizer (`resources.jl:48, 88`) and
"forgetting `free!` is not a leak" (622-626). api.md lists finalizers only for
graphs (81), `TLAS`, `BLAS` and render objects (367), and says a view has "no
finalizer of their own" (69); it never says whether a `MantleArray` has one.
Without it, an array dropped without `free!` keeps its creator hold for ever
(367: holds go to zero only through `free!`), which is the leak the current
design was built to avoid. Needs one sentence either way.

### 4.3 Budget per heap, not per device

`capacity(dev::LavaDevice)` sums the device-local heaps' budgets
(`vulkan/graph.jl:107-115`), while `Readback` and `Unified` blocks come from
host-visible heaps (`vulkan/graph.jl:770, 823`) with budgets of their own
(`VK_EXT_memory_budget` reports per heap, `vulkan/runtime/memory.jl:159`).
api.md:260 defines `budget(dev)` with "one meaning" and no heap. A pool with
`Readback`, `Unified` and `Buffers` kinds (api.md:290) needs the budget per
kind's heap or it will throw `OutOfDeviceMemory` for a BAR heap that is full
while VRAM is empty (guess: this has not bitten because readback regions are
small).

### 4.4 Importing host memory in place

`importhost`/`importablehost`/`canimporthost` (`api.jl:368-424`) map a host
array into the device without a copy, with measured bandwidths for 4 KiB vs
huge pages (398-406) and a driver restriction (RADV takes anonymous memory
only, 380-382). api.md's `MantleArray(dev, host; readonly = true)` (49)
uploads a copy and restores from `host`; the zero-copy import, the huge-page
requirement and the "two thirds of the bandwidth" trade-off have no entry.
Either it is dropped (say so) or it is a third storage kind.

### 4.5 A non-throwing allocation probe

`trydevicearray` (`lifecycle.jl:254-278`) is for "a caller that expects a
refusal and has an answer to it — a smaller batch, a host fallback", refused
without asking the driver above `maxalloc`, and "no reclaim or retry is
attempted" (285-287). api.md:480 only throws `OutOfDeviceMemory`, and only
after evicting other graphs' resources. A caller probing for a batch size
cannot do that without evicting its neighbours.

### 4.6 Device life: loss, reset, exit

`retire!(::Device)`, `retired`, `refuseretired`, `reset_device!`
(`lifecycle.jl:189-251`) and the exit-order hazard they handle ("at exit the
driver can be torn down before the arrays that name it are finalized, in no
order Julia defines", 200-204; `vulkan/runtime/device.jl:822-828`) have no
rule in api.md: teardown is "open" (561), `reset_device!` is "declared, never
implemented" (713), and the lock table says device objects are released by
"process exit" (363). Two statements conflict: "Returns the same object on
every call" (32) and "a reset makes a new device object" (378). The code shows
reset needs: retire the old context before dropping it (`device.jl:1004-1010`),
save the pipeline cache (`vulkan/graph.jl:188`), release the pool's blocks
(189), and keep every finalizer from calling the driver afterwards
(`memory.jl:92-96`). An `ERROR_DEVICE_LOST` mid-run is the same path
(`device.jl:898-903`).

### 4.7 Memory-kind constraints are not core knowledge

api.md:615 and :685 make `compatible` and `mergeconstraints` core; `pool.jl:69-93`
records that the merge rule is the device type's: Vulkan buffer usage bits
union while image memory-type bits intersect (`vulkan/graph.jl:907-916`),
Metal orders storage modes by permissiveness (`metal/device.jl:383-388`), HIP
has no constraint (`ext:339-342`). Core can hold the policy only if the
constraint is a core type whose `merge` and `compatible` the device type
implements, which is a verb api.md:290's "constraint (alignment, usage)" does
not list.

### 4.8 Atomic-write inference and the ray-tracing stage

`Unordered{U}` (`usage.jl:156-169`) and `Touch.atomic`
(`graph/access.jl:24-30`) let two passes whose writes are all atomic
read-modify-writes share a resource with no barrier; `Traced{U}` (`usage.jl:171-193`)
puts storage accesses from raygen/hit/miss shaders on the ray-tracing stage,
without which NVIDIA races ("one `VK_ERROR_DEVICE_LOST`", 181-182). api.md
mentions `Unordered()` only as a test's negative control (448) and lists the
fixed barrier stages as "compute, copy and build writes to vertex, index,
indirect and fragment reads" (320), with no ray-tracing stage. Both are
semantics the access walk already produces (api.md:232-239) and the barrier
helper must consume.

### 4.9 Validation has no API but tests depend on it

api.md:453 requires "validation clean" for the composite fuzz test, yet
defines nothing for enabling a validation layer, draining its messages, or
starting a process with Metal's environment: that is `DebugConfig`,
`validating`, `validationmessages!`, `debugenv` (`lifecycle.jl:10-165`), and
the rules at 50-64 (GPU-AV and printf exclusive; safe mode default) are
hard-won. Needs a place: a keyword of `T(key)` (api.md:255) and one read verb.

### 4.10 Video decode has a node and a queue kind but no user surface

`VideoDecodeNode` (api.md:205) and the video-decode queue kind (259) exist;
`videodecodes`, `h264decoder`, `feed!`, `decodemore!`, `remaining`
(`backendhooks.jl:52-128`) are the only API the editor calls
(`backendhooks.jl:56-63`). Which of these is a verb and what the node's
declaration looks like is unspecified.

### 4.11 Harder than the plan assumes

- **The pool lock is not a leaf today.** api.md:364 and :416-418 make the pool
  lock a leaf with "no GPU wait inside". `reclaim!` holds it while calling
  `free!(node.item)` on dropped plans (`pool.jl:902-906`, which reaches
  `untenant!` → `retire!` and `unlisten_moves!`, both re-locking it) and while
  `waitfor(dev, f)` blocks on the device (932). `reserve!` holds it across
  `remap!` of every tenant (1186-1188), which records device copies. Making it
  a leaf means `reclaim!` and the arena growth move out from under it; the
  release engine must compute what to free under the lock and free outside.
- **"Nothing fallible between claiming a token and submitting."** Today the
  pending-store landing allocates staging inside the submission path
  (`resources.jl:534-546` runs `f` under `pendinglock`; the Vulkan `StoreEmitter`
  at `vulkan/graph.jl:1426-1428`). The plan moves the upload-region acquire to
  the store's call (api.md:61), which means every `a[r] = data` acquires a
  region from the pool, i.e. may run the pressure steps and wait, from any
  thread, on the caller's line. That is a behaviour change the docs should
  state: a store may block.
- **Scheduled `resize!` touches every submitter.** Today `resize!` runs at the
  call (`resources.jl:582-593`) and `notify_move!` patches recorded plans
  (`pool.jl:1329-1377`). In the target every submitter (run, eager call, host
  read, eviction copy) must apply pending changes for every resource it
  touches before its locks (api.md:59; plan:48-57). The code's one cheap test
  for "anything pending?" is a per-plan tuple walk (`resources.jl:493-505`,
  `graph/kalaunch.jl:785`), which the inbox replaces; the inbox must be
  filled by `resize!`, stores, `GPURef` stores, eviction and `BuildRecording`
  marks alike (api.md:216) or a change is silently skipped.
- **`waitfor` declining an unsubmitted token.** `pool.jl:816-832` and
  `vulkan/graph.jl:293-306` return `false` for a token no submission will
  signal, because a recording may be open. The target has no open batch
  (api.md:274), so the subtlety disappears; but Metal's `LegacyQueue`
  (`metal/device.jl:508-541`) needs an open command buffer to mint a token,
  and "one command buffer per segment" (plan:1566) is what removes it.

## 5. Minimal form

Lines the same functionality needs under api.md, with the pieces from my files
mapped in. Estimates; "today" counts code plus its docs in my files.

| part | contents | today | target |
|---|---|---|---|
| device and registry | `Device <: KA.Backend`, `DeviceRegistry`/`REGISTRY`, `Device()`, `devices()`, `caps` record type, `budget`, `show` | `backends.jl` 168, `lifecycle.jl` 288, `Mantle.jl` device part ~90, `backendhooks.jl` 169 | 120 |
| pool mechanics | `FreeSpans`, `Block`, `Region`, `carve!`/`release!`, kinds and constraints | `pool.jl` 158-245, 573-632, 762-789 (~230 with docs) | 150 |
| acquire and pressure | `acquire!`: release passed retirements, then unmap idle pages, trim, collect, wait for retiring storage, evict, throw | `pool.jl` 653-753, 960-1098 (~330) | 120 |
| release engine | `retire!(pool, x, tokens)`, `Inbox`, sweep at `acquire!`/`run!`, counted driver objects | `pool.jl` 323-344, 859-944 (~130); `graph/lifetime.jl` 440-598; `memory.jl` 637-700 | 80 |
| arena parts | `Arena` with mapped range, tenants by graph id, `bindpages!`/`unbindpages!` calls, `ArenaPart` split at `maxalloc`, host-visible part | `pool.jl` 247-313, 1134-1256 (~250); `phases.jl` Place 535-624 | 150 |
| eviction and restore | LRU list, `Evicted`/`Restore` change, read-only arrays restored from host | none today (plan:1304-1340) | 150 |
| placement | liveness intervals, `LowestFit`, per-part placement, peak/naive | `phases.jl` 441-624, `placement.jl`, `model.jl`, `interval.jl`, `bound.jl`, `csv.jl` (~470) | 200 |
| dependencies and schedule | DAG from accesses over resource ids, roofline cost, list schedule, loop bodies | `phases.jl` 209-438 (~230) | 200 |
| sync and hazards | `Access`, usage kinds, `SyncState` (accesses, namers, pending changes, lent/freed), `Token`, `barriers!` walk with the aliasing handover | `sync/` 408, `phases.jl` Barriers 741-864, `resources.jl` store protocol (~250) | 350 |
| arrays | `MantleArray{T,N,L}`, `ArrayState`, storage kinds, `Cell`, `Store`/`CellWrite`/`MoveStorage`, GPUArrays interface (`storage`, `derive`, host copies), `free!` and holds, `GPURef` | `resources.jl` 650, `array.jl` 204, `dispatch.jl` 319, `api.jl` 451 | 400 |
| module file | imports, includes, exports | `Mantle.jl` 671 | 120 |
| device life | retired flag read by allocation, atexit mark, reset as "new object, old retired" | `lifecycle.jl` 189-251 | 60 |
| **total** | | **5978** (+ the duplicated halves in the backends) | **~2,100** |

So the slice goes from 5978 to about 2,100 lines (2,082 lines of code today,
most of it rewritten; the 3,070 lines of prose shrink to the contracts), and
the ~1,175 duplicated lines in the backend and graph files go with it. The
reduction is not from dropping features: three of the twelve target parts
(eviction, sparse arena parts, the sync state with namers and inboxes) are
larger than anything the current files have, and section 4 lists what the
target has to add back before it covers today's behaviour.
