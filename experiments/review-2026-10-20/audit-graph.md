# Audit: `src/graph/` against `docs/api.md` and `docs/recording-plan.md`

Scope: every line of `src/graph/` (9 files, 7 810 lines), compared against
`src/vulkan/graph.jl`, `src/metal/record.jl`, `src/runtime/*.jl`,
`src/memory/*.jl`, `src/sync/*.jl`, `src/vulkan/runtime/coretypes.jl`,
`src/vulkan/access.jl`, `src/vulkan/array/ka_backend.jl`, `src/metal/ka.jl`,
`src/array/launch.jl`. Line numbers are of the working tree on
`sd/lava-refactor` at the time of reading. Estimates are marked "est."; the
bucket splits in section 1 are judgement calls on current line counts
(docstrings counted with the function they document).

Mechanical facts (script over the nine files):

| file | lines | code | comment | docstring | non-code % |
|---|---|---|---|---|---|
| build.jl | 1 827 | 647 | 306 | 704 | 55 |
| kalaunch.jl | 1 524 | 613 | 242 | 590 | 54 |
| access.jl | 1 339 | 593 | 165 | 499 | 49 |
| backend.jl | 966 | 188 | 139 | 583 | 74 |
| types.jl | 835 | 254 | 230 | 312 | 64 |
| lifetime.jl | 783 | 232 | 140 | 362 | 64 |
| packing.jl | 285 | 55 | 33 | 184 | 75 |
| submission.jl | 131 | 25 | 19 | 80 | 75 |
| queue.jl | 120 | 10 | 22 | 78 | 82 |
| total | 7 810 | 2 617 | 1 296 | 3 392 | 60 |

2 617 of 7 810 lines are code. The other 5 188 are prose.

## 1. Inventory

Buckets: (a) the concern exists in api.md section 3 "Core types" or section
4; (b) becomes a device-type verb; (c) belongs nowhere (dead, superseded,
compat shim, test-only, debug-only, duplicate). Numbers are est. current
lines including their docstrings.

| file | lines | what it does | (a) core | (b) verb | (c) nowhere | api.md mapping |
|---|---|---|---|---|---|---|
| types.jl | 835 | The graph's data structures: `Graph`, `Pass`, `DrawCall`, `Trace`, `Loop`, transients, arena kinds, the compiled draw/dispatch/trace, `ArgMemory`, `PassPlan`, `Profiler`, `Plan`, `Compile`, `BufferBlock` | 330 | 60 | 445 | `Graph`→`Graph` (3); `Pass`+`DrawCall`+`Trace`+`Loop`→`Node` subtypes (3); `TransientBuffer`→`MantleArray{T,N,GraphDevice}` (2.2); `TransientImage`→`Image(g,…)` (2.6b); `Buffers/Images/Unified/Readback`→`rawalloc` kinds (4.3); `ArenaPart`→`Arena` (3); `Compiled*`→`CompiledDispatch/Trace/Draw` (3); `ArgMemory`→argument block (plan "Run"); `PassPlan`→`CompiledNode` (3); `Plan`→`Plan` (3); `Compile`→`CompileState` (3). Nowhere: `WindowSurface` (api.md:173 "There is no `Surface`"), `DrawBinding`+`rebind!`+8 `bound*` forwarders (2.5 `GPURef` slots, 2.6b groups of draws), `Pass.kind::Symbol` (rule 2), `Pass.hostcond` (2.4 `when!` is a GPU flag), `Profiler` (10, open), `Immediate` (4.5 Builder is a value), `Attr/Commands/BufferRange/ResourceView` (2.2 views share state, `launchrange`, `attribute`), `Plan.{profiler,alias,coalesce,policy,budget,profile,passcost,partition,stamped,patchtab,pending_patches,hostwritten,stale,replaced}`, `TransientBuffer.{pending,pendinglock,dirty,hostwritten}` (api.md:127 no host-written graph arrays), `BufferBlock` is Vulkan's (b). |
| build.jl | 1 827 | Declaring: `dispatch!`, `declare!`+leaf walk, `repeat!`/`when!`/gates, `render!`/`draw!`/`Attribute`, `copy!`, transients, views/slices, `hostwritten!`, `Plan`/`Graph`/`Compile` ctors, `overlapping`/`usage_of`, `refit!`, `free!`, `remap!`, `runonce!` | 840 | 90 | 900 | `dispatch!(g,…)`→2.4 `dispatch!`; `dispatch!(g, call, args)`→foreign/library node (plan 277); `declare!`/`resourceleaves!`→`dependencies!` input (plan 477); `repeat!`→2.4 `repeat!(group,n)`; `render!`/`draw!`→2.6b `draw!`/`clear!`, passes formed by compile (api.md:160); `Attribute`→`attribute(a,i)` (api.md:163); `copy!`→`copyto!` copy node (api.md:113); `Transient.Buffer/Image`→`MantleArray(g,…)`/`Image(g,…)`; `refit!(pl)`→`recompile!`/`applywindows!` (api.md:390-391); `free!(pl)`→`free!(g)`; `checkextents`→compile check. Verbs: `makeimage`/`remakeimage!`→`placeimage`; `imageusage`/`extrausage`/`alignment`→`rawalloc` constraint; `clearvalue`/`depthclear`→`beginrender!`; `storage(::TransientBuffer, block)`→`devicearg`. Nowhere: `when!` host cond (354-369), `gate!`/`Predicate`/`supportspredicate` (470-523; api.md:518 conditional rendering unused), `hostwritten!`/`updates_pass!`/`hostwritable` (1140-1197; stores are pending changes api.md:61), `viewof`/`ResourceView`/`slice`/`BufferRange`/`rootresource` (613-665, 1095-1118, 1316-1320; one resource per array state api.md:447), `overlapping`/`overlapgroup`/`sliceindex`/`lastuses` (1363-1466), `remappable`/`invalidate!`/`droprecording!` (1654-1707; re-record of marked parts api.md:481), `remap!` (1776-1797), `runonce!` (1736-1742; rule 4), `npipelines`/`elapsed`/`sample!`/`lp_of`, `writes_it` (1354, zero references), `target_*` forwarders (575-593), `passname` (origin api.md:207), `Surface` (562), tombstone comments (~60 lines, see 3.P10). |
| access.jl | 1 339 | The taint walk over typed IR: `Touch`, `argument_usage` declarations, `accessof`/`signaturetouches`/`walkir`/`stmttaint!`, llvmcall/intrinsic reading, `kerneltouches`, `devicetype` | 1 150 | 40 | 150 | Stays almost whole: api.md:232-239 names `argument_usage`, `kerneltouches`, `kernelinterpreter`, `kakernelaccesssignature`, `intrinsic_usage`, `vertextouches`, `fragmenttouches`, `shadertouches`. Verbs: `argtype`/`devicebuffertype` (1326-1339)→`devicearg` (4.6). Nowhere: `AccessCache` world-keyed per device (453-490; access lives in the `Kernel` at its `CodeInstance`, api.md:208, 584), the `VulkanAPI()`/`MetalAPI()` probe chain in `intrinsic_usage(name)` (1038-1043), `devicetype` for `TransientBuffer`/`ResourceView`/`BufferRange` (1262-1303; one array type), the `kisupported || kikernel` probe (1218). |
| backend.jl | 966 | The hook declarations a backend answers (69 `function x end`/defaults) and `BACKEND_VOCABULARY` (189 names) | 60 | 330 | 576 | Core: `isdepth`→`imageformat`; `waitfor!(dev,tok)`→`waitfor(::Token)` (4.2); `submissionbudget`→core from `caps` (api.md:263); `makeargmemory`→`bindings!` (plan 487); `activationkind`→rewriter attribute (plan 505-514); `Base.show(::Device)`. Verbs: `makeimage`→`placeimage`; `openrecording`/`closerecording!`→`Builder`/`finish`; `emitdispatch!`/`emitcopy!`/`beginrender!`/`emitdraw!`/`endrender!`→4.5 (new signatures); `emitinline!`/`storebytes!`→`emitstore!`/`emitcopy!`; `emitkernel!`/`emithead!`/`emitloophead!`/`emitpreparebarrier!`/`withpredicate`→fold into builder verbs (Appendix A); `compiledraw`→`compiledraw`; `compile_dispatch`→`compilekernel`; `argbytes`/`indirectslot`/`slotgrid`→goes (`bindings!`); `native_*`→`candidates(dev,f)` (4.4); `kernelcompiles`→core counter; `devicename`→core. Nowhere: `BACKEND_VOCABULARY` (806-966, 161 lines of test scaffolding; api.md:573 "Section 4's verbs replace `BACKEND_VOCABULARY`"), `recordsplans`/`runscalls`/`patchable`/`supportspredicate`/`timestamps` capability Bools (239-271, 428-449, 624-633), `openrun`/`closerun!`/`abandonrun!`/`abandonframe!`/`abandonrecording!` (451-498), `profiled!`/`collect!`/`makeprofiler`/`hostprofiler`/`gpupasstime!` (508-542, 606-688), `emitupdate!` (155-163), `workgroupsize` decl (182-189; `KernelInfo`), `syncbackend` (517-524), `initbackend!` (781-793). |
| kalaunch.jl | 1 524 | Compile (`run!(::Pipelines)`, `bake`, `compiledraw`), the prepare kernels, `run!(pl)`/`execute!`, the walk (`emitplan!`…), stores/patches, `record!`+partition+measurement, `RecordingParts`+submit, the `Immediate` emitter | 430 | 40 | 1 054 | Core: `run!(::Pipelines)`→`compile!(s,node)`+`bindings!` (plan 478, 487); `emitprepares!`/`multiprepare!`/`writegroups!`/`gateopen`→head/prepare nodes (plan 489-497, 825-838); `emitplan!`/`emitpasses!`/`emitpass!`/`emitwork!`→`record!` walk (plan 747-770); `run!(pl)`→`run!(g)`/`run!(pl)` (plan 787-804); `partitionranges`/`submissionbatches`→`submissions!` (plan 485, by estimate not measurement); `record!`/`recordparts!`→`record!`; `RecordingParts`→segment parts (api.md:220); `ki_launch_extents`/`defaultgroup`/`callgroup`→`config(dev,f)`+"one calculation `ki_launch_extents`" (plan 400-409); `checkrenderarea`; `Call`→`ForeignNode`. Verb: `resolve`→`devicearg`. Nowhere: `Launch`/`DeferredRange`/`ndrangeof`/`awaitwrites`/`KA.launch_config(::DeviceRange)` (57-158; no interpreted launch, "awaitwrites goes" Appendix A), `buildskernel`/`kisupported`/`kikernel` hasmethod probes (160-239; rule 2), `argsize`/`indirectindex` (478-490; slots are `Bound`), `drawviewport`/`packedargs`/`DrawBinding` handling (588-656), `landhost!`/`StoreEmitter`/`emitstore!`/`storefrom!` (666-679, 839-874; `Store` change + `emitstore!(b,dst,upload)`), `emitpatches!` (819-837; patch table goes), `beforeframe!`/`execute!` branching (734-803), `timings`/`measure!`/`measurable`/`measuring` (876-895, 1254-1301; profiling open), `recordable`/`hasrebindable` (1399-1433), `submitplan!` narrowing (1361-1381), `Immediate` emitter (1435-1524). |
| lifetime.jl | 783 | `Stamp`, `SubmitChannel`, hold stack (`hold!`/`holdleaves!`/`claim!`/frames), `crosswaits!`/`stamp!`, channel-side `retire!`/`reclaim!`/`drain!`, recording free list (`acquire!`/`release!`/`finishrecording!`), `Submission`, `oneshot!`/`handover!` | 150 | 60 | 573 | Core: `SubmitChannel`→`SubmitChannel{N,T}` (api.md:219, plan 711-717: native, kind, lock, next, outstanding — 5 of today's 10 fields); `Submission`→`Submission{T,C}` (plan 724); `recycle!`→`recycle!` (4.2); `crosswaits!`/`stamp!`→`SyncState` accesses and `waits` (api.md:215); channel `retire!`/`reclaim!`→the one release engine `retire!(pool,x,tokens)` (api.md:365); `oneshot!`→an eager call's submission (plan 736). Verbs: `makerecording`/`resetrecording!`/`destroyrecording!`/`finishrecording!`/`recorder`→`Builder(dev,ch)`/`finish`/`recycle!`. Nowhere: `Stamp`/`stampof` per backend object (50-89; sync state per resource), `WrongThread`/`ownthread` single-writer thread (209-241; api.md:449 any thread), the hold stack `holds`/`spare`/`takeholds!`/`unhold!`/`pushholds!`/`dropholds!`/`emptyframe!` (243-405; plan 737 "no hold stack"), `holdsretired` (491-500), `channelof`/`deviceof`/`todevice`/`payloadtype` forwarders (170-207), header narrative (1-48). |
| packing.jl | 285 | `passedas`/`notpassed`/`passedslots`/`eachpassedarg` (which args take a slot), `devicepointeroffsets`/`notepacked!` (patch table) | 120 | 0 | 165 | Core: the NotPassed rule and slot walk→`bindings!` argument-block layout (plan 487, 813-817). Nowhere: `DeviceAllocation` (63-79; the block holds addresses on every device type), `nestinglevels`/`devptroffsets!`/`devicepointeroffsets`/`notepacked!` (180-285; a move re-records handle parts, plan 167-172, api.md:481). |
| submission.jl | 131 | `Outstanding`, `submitted!`, `recycle!`, `newest`, `sweep!`, `idle`, `outstanding` | 50 | 0 | 81 | Core: `outstanding`+`sweep!`+`recycle!`→`SubmitChannel.outstanding` and the release engine (api.md:219-221, 277). Nowhere: `tag::Any` (41), `newest`/`idle` (serve `flush!`, which goes: Appendix A `flush!`), the "five records" header (1-19). |
| queue.jl | 120 | `allocate_batch_queue!`, `supports_batch_queue`, `batchqueue`, `release_batch_queue!`, `submit!`, `flush!`, `waitidle` declarations | 10 | 0 | 110 | `submit!`→core `submit!(ch, parts;…)` (Appendix A); `waitidle`→open (api.md:544). Nowhere: `allocate_batch_queue!`/`release_batch_queue!`/`supports_batch_queue`/`batchqueue` (Appendix A: goes; queue kinds via `queuekinds(dev)` 4.1), `flush!` ×3 (goes). |
| **total** | **7 810** | | **≈3 140** | **≈620** | **≈4 050** | |

Reference counts (script, whole-word, across `src/`, `ext/`, `test/`):
definitions in `src/graph` with no reference anywhere but their own line:
`writes_it` (build.jl:1354), `hasrebindable` (kalaunch.jl:1428, called once
by `recordable` at 1416 — reachable but single-use). Test-only (zero
references in `src/` outside `src/graph`, zero in `ext/`, used by tests):
`when!` (70 test refs, 0 in sibling packages RayMakie/Hikari/DNNKernels/
VideoEditor), `DrawBinding`/`rebind!` (1 test ref, 0 consumers), `runonce!`
(0 consumers), `viewof` (2 test refs), `timings` (0 consumers),
`hostprofiler`, `submissionbudget` (no backend overrides it: 6 refs, all in
`src/graph`).

## 2. Duplication

### 2.1 types.jl vs src/vulkan/runtime/coretypes.jl

Not copies of each other; the overlap is state kept twice.

- The launch decomposition is stored twice. `IterPlan` (coretypes.jl:422-427)
  is cached per device in `DeviceCaches.iterplans` (592) *and* held in every
  `CompiledDispatch` as `iter`/`nd0`/`obj`/`group` (types.jl:538-542). The
  same for `LaunchPlan` (coretypes.jl:200-210): cached in
  `DeviceCaches.launchplans` (540) and held as `CompiledDispatch.launch`
  (types.jl:537). `ka_backend.jl:403-444` and `vulkan/graph.jl:1062-1065`
  build them; the per-device caches exist for the eager path, the fields for
  the recorded path. The target keeps one `DeviceKernel` per device at the
  kernel's `CodeInstance` (api.md:208) and one `CompiledDispatch` per node.
  Recoverable: est. 120 lines (coretypes.jl:412-455, half of 513-618;
  types.jl:533-557 shrinks to pipeline + slots).
- `DeviceCaches.accesses::Any` (coretypes.jl:606) is `AccessCache`
  (access.jl:468-472), typed `Any` and re-asserted at vulkan/access.jl:15.
  Goes with `AccessCache` (api.md:584).
- `MemoryPolicy` (coretypes.jl:140-172) beside `PoolPolicy`
  (pool.jl:389-411): two policy records for one pool; coretypes.jl:141-145
  says so. Recoverable: 33 lines.
- `VkManagedBuffer.stamp::Stamp{UInt64}` (coretypes.jl:81) is the backend
  storage for core's `Stamp` (lifetime.jl:71-75): a core concept (last
  submission, holders) stored on a driver object and reached through
  `stampof` (lifetime.jl:89). The target's `SyncState` is on the resource
  (api.md:215). Recoverable: coretypes.jl:63-81 and lifetime.jl:50-89, est.
  55 lines.

### 2.2 access.jl vs src/sync/usage.jl vs src/vulkan/access.jl vs src/metal/ka.jl

- Two encodings of read/write. `Touch(read, write, atomic)` (access.jl:31-35)
  and `Access{RD,WR}` + `Unordered{U}` + `Storage{K,A}` (usage.jl:22-33,
  79, 164); `usagetype` (access.jl:66-69) converts one into the other at
  `declare!` (build.jl:161). `Transition` then carries the second as
  `Type` values (`waits::Vector{Type}`, transition.jl:19) and sorts them by
  `string` (transition.jl:56). One value type for an access would remove the
  conversion and the `Type`-valued vectors. Recoverable: est. 40 lines, plus
  the dynamic dispatch on `Type` values in `Barriers`.
- The per-backend "touches" functions are one function written five times.
  Generic: `kerneltouches` (access.jl:1208-1221) and
  `kakernelaccesssignature` (1237-1245). Vulkan: `kerneltouches` for a
  `RayTracingPipeline` (vulkan/access.jl:55-67), `shadertouches` (70-77),
  `vertextouches` (94-99), `fragmenttouches` (101-106). Metal:
  `kakernelaccesssignature` (metal/ka.jl:460-467), `vertextouches` ×2
  (482-496), `fragmenttouches` (507-510), `stagetouches` (512-520). Every
  one is `argT = map(a -> devicetype(dev, a), args)`, build `tt`, call
  `accessof(interp, f, tt; cache)`, drop `k` leading entries. One
  `touches(dev, f, lead, args)` in core with `kernelinterpreter(dev, f, tt)`
  as the only verb covers all of them. Recoverable: est. 70 lines
  (vulkan/access.jl 55-106 → ~15; metal/ka.jl 460-520 → ~20).
- `intrinsic_usage(name)` (access.jl:1038-1043) asks `VulkanAPI()` then
  `MetalAPI()` by name — core naming its backends, which api.md rule 2
  forbids. Keyed on the interpreter it is one method per device type.
  Recoverable: 6 lines and one core→backend dependency.
- `argument_usage` declarations: core's for `mul!`/`coopmat_*`
  (access.jl:189-227), Metal's for `MPSGraph*` (metal/ka.jl:231-232,
  253-254, 328-329) — same mechanism, not a duplicate.

### 2.3 kalaunch.jl vs src/vulkan/array/ka_backend.jl vs src/metal/ka.jl vs src/array/launch.jl

"Kernel + group + ndrange → something launchable" is written nine times:

| where | lines | what it repeats |
|---|---|---|
| kalaunch.jl `bake` | 353-378 | `buildskernel` ? `kernelfor`+`callgroup` : `kikernel`; `Launch(...)` |
| kalaunch.jl `kikernel` | 230-239 | `tt = map(Core.Typeof∘KI.argconvert)`; `KI.kernel_function` |
| array/launch.jl `kilaunch!` | 88-98 | the same two lines (90-91) plus the `workgroupsize === nothing` branch |
| array/launch.jl `runlaunches!` | 49-70 | `buildskernel` branch again, `group == 0` sentinel, three `invokelatest` |
| array/launch.jl `dispatchlaunches!` | 109-120 | same sentinel, `invokelatest(dispatch!, …)` |
| ka_backend.jl `(obj::KA.Kernel)(args...)` | 466-512 | eager path: `find_tlas_in_args`, `DeviceRange` branch, `get_or_build_iter_plan`, function barrier |
| ka_backend.jl `ka_launch_indirect!` | 867-940 | its own workgroup default `(256,)` (877), its own ceiling `1024*1024*ws_prod` (888), its own `KA.partition`/`mkcontext` |
| vulkan/graph.jl `compile_dispatch` + `kicompile_dispatch` | 1032-1102 | `buildskernel` branch, `kernelfor`/`callgroup`, `get_or_build_iter_plan`, `launch_plan` |
| metal/record.jl `compile_dispatch` | 386-479 | `buildskernel` branch (412), `KA.launch_config`+`mkcontext` (447-449), `defaultgroup` then a repartition (457-468) |

The target has `compilekernel(dev, kernel, config)` and one launch-geometry
calculation (plan 390-409). Recoverable: est. 450 lines across these files
(kalaunch.jl 160-305 → ~40; array/launch.jl 39-98 → 0 after one path;
ka_backend.jl 279-314, 413-444, 466-512, 565-595, 867-940 → ~80;
vulkan/graph.jl 1032-1102 → ~30; metal/record.jl 386-479 → ~40).

Smaller pairs:

- Indirect prepare: `multiprepare!`/`multi_prepare_indirect_kernel`/
  `writegroups!` (kalaunch.jl:546-583) vs `prepare_indirect_kernel`/
  `prepare_indirect_dispatch!` (ka_backend.jl:838-856, the n=1 case with
  its own barrier at 933-934) vs Metal's `writegroups!(::ICBGridSlot)`
  (record.jl:111-124) and `metal_write_range!`/`metal_write_grids!`/
  `metal_reset_ranges!` (922-960). Recoverable: est. 40.
- `Launch`/`Call` callables (kalaunch.jl:64-158, 314-351) vs
  `MetalRecordedDispatch` and its call operator (record.jl:176-227).
  backend.jl:907-913 states the duplication: "They exist at all only because
  the Metal backend defines `MetalRecordedDispatch` instead of using core's
  `CompiledDispatch`". Recoverable: est. 60.
- `DeferredRange`/`ndrangeof`/`awaitwrites` (kalaunch.jl:76-121),
  `KA.launch_config(::Kernel, ::DeviceRange, …)` (139-143) and
  `hostdownload`'s `awaitwrites` (resources.jl:194-197): three host readbacks
  of a device count. All go (Appendix A `awaitwrites`, `devicesized`).

### 2.4 lifetime.jl vs src/memory/pool.jl (and submission.jl)

Three deferred-release engines for one fact ("this token passed"):

1. `SubmitChannel.pending`/`pendinglock`/`retiring`/`taken` with
   `retire!(ch, obj)`/`reclaim!(ch)` (lifetime.jl:135-151, 486-576): two
   phases, stamp with the channel token, hand to another channel if that one
   named it last (543-551).
2. `Pool.pending::Inbox`/`retiring`/`retiring_at`/`pendingplans` with
   `retire!(pool, dev, r)`/`reclaim!(pool, dev)` (pool.jl:445-447, 467,
   859-944): two phases, stamp with `fence(dev)`, prefix scan.
3. `Outstanding`/`sweep!`/`recycle!` (submission.jl:38-117): the recordings
   and hold lists of passed submissions.

The two `reclaim!`s differ in how a finalizer pushes: pool.jl:315-344 is a
lock-free `Inbox` "because a finalizer may not wait"; lifetime.jl:136-145 is
a `ReentrantLock` with a 10-line justification for why a finalizer may take
it. One of the two arguments is wrong, and the files do not say which. The
target has one engine, `retire!(pool, x, tokens)` with a lock-free inbox
(api.md:365, 367, 230). Recoverable: est. 150 lines (lifetime.jl:471-595
→ 0; pool.jl:859-944 → ~60; submission.jl:84-124 → ~20).

- Holds: the frame stack `holds`/`spare`/`takeholds!`/`unhold!`/
  `pushholds!`/`dropholds!` (lifetime.jl:243-405) plus `Stamp.holders`
  (71-75) plus `Pool.movelisteners` (pool.jl:453) plus `Arena.tenants`
  (pool.jl:269, both `WeakRef` lists pruned by hand) plus Metal's
  `ManagedLeaves`/`argmanaged`/`claimbuffers!` (record.jl:653-683, 1615).
  Five ways of saying "work in flight names this". The target: atomic hold
  counts on the array state and tokens of last uses (api.md:367-368).
  Recoverable: est. 160 in lifetime.jl, 30 in metal/record.jl.
- Same name, different operation: `acquire!(ch)` (lifetime.jl:655) vs
  `acquire!(pool, …)` (pool.jl:653); `release!(ch, r)` (lifetime.jl:678) vs
  `release!(::Region)` (pool.jl:762) vs `release!(::RecordingParts)`
  (kalaunch.jl:1333); `reclaim!(ch)` vs `reclaim!(pool, dev)`;
  `retire!(ch, obj)` (lifetime.jl:486) vs `retire!(pool, dev, r)`
  (pool.jl:859) vs `retire!(dev)` = "mark the device lost" (lifecycle.jl:208,
  vulkan/graph.jl:165). These are not methods of one operation; they are
  five operations sharing a word, which is what "multiple dispatch instead
  of name variants" is meant to prevent in the other direction.

### 2.5 kalaunch.jl vs src/vulkan/graph.jl vs src/metal/record.jl (walk, submit, profiling, windows, stores, patches)

- The walk itself is single (kalaunch.jl:907-1012) — the one place where the
  intended split holds. But the primitive set is implemented three times:
  Vulkan (vulkan/graph.jl:1324-1571), Metal (record.jl:967-1397) and core's
  `Immediate` (kalaunch.jl:1435-1524). The target has one builder verb set
  and no interpreted emitter (api.md:310-325). Recoverable: the 90 lines of
  `Immediate`.
- `submitrecording!` families: core ×3 (kalaunch.jl:1345-1359, 1376-1381,
  1391-1397), Vulkan ×3 (vulkan/graph.jl:1822-1873), Metal ×3 + `replay!`
  (record.jl:1569-1639). Core's is a loop that re-dispatches to the backend
  per piece; the target has `submit!(ch, parts;…)` and `submitnative!`
  (api.md:274, 732). Recoverable: est. 80 core + 60 Vulkan.
- Profiling: backend.jl:508-542, 606-688; kalaunch.jl:876-895, 1254-1301;
  types.jl:666-694; build.jl:1286-1291, 1637-1649 (≈330 lines in my files)
  vs vulkan/graph.jl:626-635, 1264-1372 vs record.jl:1176-1183. api.md:541
  leaves profiling open and says `Profiler`, `makeprofiler`, `profiled!`,
  `gpupasstime!`, `collect!`, `timestamps` are to be replaced. The
  measurement-driven partition (`measure!`, kalaunch.jl:1285-1301, re-records
  once after the first run, 1040-1043) is replaced by the estimate
  (plan 516-527). Recoverable: est. 300 pending that decision.
- Windows/frames: `beforeframe!` (kalaunch.jl:734-754),
  `openrun`/`closerun!`/`abandonrun!`/`abandonframe!` (backend.jl:451-498),
  `Surface`/`WindowSurface`/`target_*` (build.jl:562-593, types.jl:27-29) vs
  vulkan/graph.jl:1767-1927 vs record.jl:1419-1484. The target folds acquire
  and present into `run!`/`submitnative!` (api.md:80, 173-175, 274).
  Recoverable in my files: est. 110.
- Stores: `landhost!` ×3, `StoreEmitter`, `emitstore!` ×3, `storefrom!`,
  `emitupdates!` (kalaunch.jl:666-679, 805-874), `emitinline!`/`storebytes!`
  decls (backend.jl:500-506, 544-552) vs vulkan/graph.jl:499-530 (two
  `storebytes!` with the same 64 KiB/4-byte test, 502 and 522), 1424-1430,
  1937-1941 vs Metal's `Metal.synchronize()` before any store
  (record.jl:1420) vs the pending lists in `Buffer`/`GPURef`/
  `TransientBuffer` (resources.jl:414-566: two identical `setindex!` bodies at
  430-446 and 463-485, three `landstores!`). The target: one `Store` change
  with an upload region, `emitstore!` within `caps(dev).inlinestore`, else
  `emitcopy!` (api.md:61, 318). Recoverable: est. 100 in my files, 150 in
  resources.jl, 40 in vulkan/graph.jl.
- Patch table: packing.jl:180-285, build.jl:1654-1707, kalaunch.jl:819-837,
  types.jl:748-766 vs pool.jl:1258-1377 (`listen_moves!`/`unlisten_moves!`/
  `notify_move!`) vs vulkan/graph.jl:1727-1760 (`closerecording!`/
  `patchtarget`). The target replaces address patching by cells, the head
  node and re-recording of handle parts (plan 134-139, 167-172; api.md:481).
  Recoverable: est. 150 in my files, 120 in pool.jl, 35 in vulkan/graph.jl.
- Argument packing: `passedslots`+`eachpassedarg` (packing.jl:144-178, a
  type-level and a value-level walk of the same rule) vs Metal's
  `recorded_args` at compile and `pack_recorded!` at record (record.jl:277-342,
  the same walk twice as the file's own comment at 328-330 says) vs Vulkan's
  `packdraw!`/`packtrace!`/`packdispatch!` (vulkan/graph.jl:1174-1184,
  1195-1217, 2016-2031: three copies of "`am = e.args; off = d.argoff; raw =
  rawargs(…); holdleaves!(…); pack_args_direct!(e.owner, am.ptr+off,
  am.address+off, offsets, size, byval, args)`") vs `ka_launch!` and
  `ka_launch_indirect!` packing per eager launch (ka_backend.jl:818, 916).
  The target packs once at compile in `bindings!` (plan 487, 813-817).
  Recoverable: est. 100 in packing.jl, 60 in vulkan/graph.jl, 50 in
  record.jl.

Total recoverable from the pairs above: est. 2 100 lines, of which est.
1 350 are in `src/graph/`.

## 3. Patterns that inflate the code

Each pattern: examples with file:line, est. total cost in `src/graph/`.

**P1. Branching by hand on a kind or a capability Bool instead of dispatch.**
`p.kind === :compute/:render/:copy/:update` at build.jl:1143, 1183, 1510;
kalaunch.jl:467, 813, 969, 978, 992-999 (and metal/record.jl:702, 739).
Eight capability Bools each consumed by an `if` at the call site:
`supportspredicate(g.dev) || devicesized(pp)` (build.jl:423);
`kisupported(g.dev, f) ? dispatch!(…, 1) : dispatch!(…)` (495-496);
`recordsplans(dev) && recordable(dev, pl)` (kalaunch.jl:773);
`patchable(pl.graph.dev) || (pl.replaced = true)` (build.jl:1795);
`timestamps(g.dev) && isfinite(budget)` (1546); `runscalls(dev) ||`
(kalaunch.jl:361); `supports_batch_queue(dev) && flush!(dev)` (pool.jl:962);
`buildskernel(d.kernel, be) ||` (kalaunch.jl:373). The `Pass` struct carries
every kind's fields at once (`targets`, `loads`, `depth`, `dst`, `draws`,
`dispatches`, types.jl:197-206) because the kind is a Symbol. api.md rule 2
forbids Symbol kinds and `isa` chains; rule 5 forbids "not supported"
defaults. Cost: est. 200 lines (the Bool verbs with docstrings, the
branches, the union-of-fields `Pass`).

**P2. Defensive checks on values whose type is known.** 97 `=== nothing`/
`!== nothing` sites (access 19, build 28, kalaunch 35, lifetime 10, backend
3, types 2). Most come from four `Union{Nothing,…}`/`Any` fields:
`Plan.args` (types.jl:719; checks at kalaunch.jl:521, 798, backend.jl:597,
build.jl:809, 1768, record.jl:970, 1353), `Plan.profiler` (711; checks at
kalaunch.jl:1068, 1257, backend.jl:535, vulkan:1326, 1349, metal:1178),
`Plan.recording::Any` (726; checks at kalaunch.jl:772, 1046, 1298,
build.jl:1668, 1684, 1697, metal:1420), `DrawCall.indices::Any` (135; checks
at types.jl:78, 97, kalaunch.jl:649, vulkan:1004). Probes that api.md rule
2 bans outright: `hasmethod(target_image, Tuple{typeof(src)})`
(build.jl:1221), `hasmethod(f, Tuple{typeof(backend)})` followed by
`Base.infer_return_type` — inference run to answer a type question
(kalaunch.jl:190-192), `hasmethod(KI.kernel_function, …)` (217),
`isdefined(x, i)` in a field walk (build.jl:216), `hasfield(T, :hwtlas)`
(ka_backend.jl:383, 396). Cost: est. 120 lines.

**P3. try/catch.** In `src/graph` every `catch` rethrows (kalaunch.jl:727,
779, 791, 1132; lifetime.jl:732) and the `finally`s release
(access.jl:562, kalaunch.jl:1004, lifetime.jl:530, 557). No violation of
the owner's rule in my files. One outside: `rawfree(::LavaDevice,
::BufferBlock)` (vulkan/graph.jl:863-873) catches a destructor failure and
logs it. `underpressure` (pool.jl:687-695) catches, retries once, rethrows
— acceptable.

**P4. Globals.** `src/graph` has only immutable `const`s: 17 in access.jl
(`Touch` singletons, op-name tuples, regexes — these are fine as constants),
`BACKEND_VOCABULARY` (backend.jl:806), and three numbers: `LEAFDEPTH`
(build.jl:203, used once at 207), `INDIRECT_CEILING` (1491, used once at
1493), `INDIRECT_STRIDE` (1259, used at 722, 728 — and restated as the
literal `256` in metal/record.jl:751). The two single-use constants violate
the owner's rule (a kwarg of `resourceleaves!`/`dispatchrange`). Mutable
process state lives outside my files and is what the target's one
`REGISTRY` replaces: `DEVICES` (vulkan/graph.jl:51), `VK_CONTEXT_REF`
(vulkan/graph.jl:183), `BACKEND_PROBES` (runtime/backends.jl:96). Two
registries that act like globals: `Pool.movelisteners` (pool.jl:453) and
`Plan.patchtab::Dict{UInt64,…}` keyed by device address (types.jl:765) —
state that should be the array's (its cell, api.md:212). `AccessCache` is
per device but keyed on `Base.get_world_counter()` (access.jl:474-481), a
cache that empties on every method definition; api.md:208 puts the access
on the `CodeInstance`, which Julia invalidates for it. Cost in my files:
est. 30 lines; the owner-rule violations are the two single-use consts.

**P5. Near-copies for different argument kinds.** `emitstore!` ×3 +
`storefrom!` (kalaunch.jl:857-874) and `landhost!` ×3 (672-679): two triples
differing only in the sink. `devicetype` ×6 (access.jl:1262-1303), five of
them to say "a view/range/transient of T has the device type of a T array".
`resourceleaves!` ×6 (build.jl:187-220). `holdleaves!` ×6
(lifetime.jl:291-325, resources.jl:307). `submitrecording!` ×3 core.
`refit!` ×4 with three meanings (backend.jl:84 "resize in place",
build.jl:736/873 "track a window", 759 "recompile the plan"). `record!` ×2
with two meanings (access.jl:714 records a taint; kalaunch.jl:1045 records a
plan). Outside: `rawalloc` ×4 sharing 20 lines each (vulkan/graph.jl:732-845),
`pack{draw,trace,dispatch}!` ×3 (1174-2031), `emit_draw!` ×4 (1982-2002),
`setindex!`/`landstores!` ×2/×3 (resources.jl:430-566). Cost: est. 250
lines in my files.

**P6. Manual loops that are a map/filter/any/findfirst.**
`updates_pass!` (build.jl:1141-1148, `findfirst`), `hostwritten!` (1180-1197,
`unique`/`filter` by `===`), `sample!` (1287-1291, a ring written as
`push!`+`popfirst!`), `lastuses` (1453-1466), the three parallel `Any[]`
vectors turned into `ntuple`s (kalaunch.jl:522-538), `submissionbatches`
(1158-1195), the two hand compactions `taken[keep] = obj; resize!`
(lifetime.jl:533-561, 564-575), `sweep!` (submission.jl:107-117), `tenants!`
(pool.jl:274-283). Cost: est. 120 lines.

**P7. Forwarding layers.** `target_view/image/extent/format/eltype(::WindowSurface)`
(build.jl:575-593); `newpass`/`handle`/`dispatches`/`passes` (1132-1138) —
four one-liners that exist so `dispatch!(p, …)` in runtime/dispatch.jl:235
can pretend not to know `Pass`; `drawargs`/`boundcount`/`boundindices`/
`boundinstances`/`boundbindings` ×2 each (types.jl:105-114); `flush!` ×2
(queue.jl:100, 108); `channelof`/`deviceof`/`todevice`/`outstanding`
(lifetime.jl:187-207); `pool(c::Compile)`, `overlapping(c::Compile, …)`,
`overlapgroup(c::Compile, …)`, `alignment(c::Compile, t)` (build.jl:714,
1338-1341); `PassHandle` (types.jl:421-424) exists only to carry
`(graph, pass)` through `passof`/`graphof` (dispatch.jl:131-132); the nine
`Compilation` accessors (phases.jl:93-161) for an abstract type with one
subtype (`Compile`, types.jl:800). Cost: est. 90 lines in my files.

**P8. `_`-prefixed functions.** None defined in `src/graph`. Two in
ka_backend.jl:391-401 (`_find_tlas`, `_maybe_tlas`). Mixed naming within one
file is common (`ki_launch_extents` beside `kikernel`, `loop_start!` beside
`emitloophead!`, `first_target` beside `firstloop`) but is not the
underscore-prefix rule.

**P9. Compat shims and feature flags for abandoned approaches.** The
KI-vs-KA split: `buildskernel`/`kisupported`/`kikernel`/`kibackend`
(kalaunch.jl:160-239), `gate!` picking kernel-or-call (build.jl:489-496),
array/launch.jl:51-54 marked "MIGRATION SHIM … goes away with the last
`@kernel`". `supportspredicate` fallback (build.jl:506-523). `patchable`
(backend.jl:428-449). `recordsplans`/`runscalls`/`recordable` and the
walked-vs-recorded fork in `execute!` (kalaunch.jl:770-803). The `Immediate`
emitter for backends that do not record (1435-1524). `awaitwrites` defaulting
to a full synchronize (112-121). `DeferredRange` host readback (76-110).
`Plan.alias`/`coalesce`/`policy` knobs (types.jl:712-714; `Overlap`/`Compact`
phases.jl:255-270) and `Compile(g; alias, coalesce, policy)` (build.jl:1322).
`TransientBuffer.hostwritten` (types.jl:285). `Plan.profile` and `stamped`
(734, 747). `budget = Inf` as a sentinel (kalaunch.jl:1209). Cost: est. 350
lines.

**P10. Comments that narrate history instead of stating the contract.**
Grep for narration markers (was/used to/moved/measured/dates/replaced/
deleted) hits 208 lines in my files, and those are the sentence starts of
longer paragraphs. Tombstones "`X` is Mantle's now — see types.jl":
build.jl:606, 979, 981, 994, 1080, 1246, 1248, 1250, 1284, 1296, 1298 plus
orphan paragraphs about removed code at 1120-1131, 1343-1350, 1631-1633,
537-549; vulkan/graph.jl:341, 367, 384-391, 394-397, 407, 455-457, 623,
639-640, 717, 730, 1575, 1642, 1657-1660, 1703-1711. The narration has gone
wrong where it contradicts the code: build.jl:901-903 says `resourceid(::Graph)`
"is NOT defined here: `runtime/dispatch.jl` has it"; dispatch.jl:141-142 says
it is in build.jl; build.jl:534 defines it. kalaunch.jl:12-16 says the
Vulkan backend "defines its own `run!(::Pipelines, ::Compile{LavaDevice})`";
no such method exists (the only one is kalaunch.jl:419). types.jl:8 contains
a literal `\n` inside a comment. Dated measurements inside docstrings:
types.jl:398, build.jl:764, 1659 ("KNOWN GAP (2026-09-21)"),
vulkan/graph.jl:1316, 1539. api.md's own rule for docs ("state the target
only") would cut prose from 5 188 lines to est. 1 300: est. 3 800 lines
recoverable in my files from this pattern alone.

**P11. Long functions and keyword explosions.** Few functions over 100 lines:
`stmttaint!` (access.jl:757-861, 105 lines, 74 code), `calltaint!`
(938-1002), `repeat!` (build.jl:371-444), `reclaim!` (lifetime.jl:518-576).
The `Plan` constructor passes 24 positional fields in a `let` chain
(build.jl:1548-1559). Kwarg-heavy signatures: `repeat!(g, kernel, args,
ndrange; max, count, while_nonzero, group, name)` (build.jl:302-304),
`draw!(p, shader, args, n; frag_args, viewport, bindings, indices,
instances)` (1034-1036), `Plan(g; coalesce, alias, profile, policy, budget)`
(1536-1538), `native_conv2d_dispatch!(…; bias, stride, pad, dilation,
groups, epilogue, transposed, name)` (backend.jl:332-334), `acquire!(pool,
dev, kind, transients, bytes; align, blocksize, constraint, maywait)`
(pool.jl:653-655). Each kwarg is a mode the function branches on. Cost:
est. 60 lines.

**P12. Parallel data structures kept in sync by hand.** `Graph.transients`
(Vector) beside `Graph.transient_by_id` (Dict), types.jl:410-411;
`IdTable.ids` (IdDict) beside `by_id` (Dict), api.jl:253-264;
`Plan.slabs`/`arenas`/`offsets`/`parts` — four vectors indexed by transient
position (types.jl:705-708), rewritten together at build.jl:791-795 and
read by index at 1781-1782; `Plan.passcost` parallel to `passes` (738) and
remapped by identity at 790-792; `Profiler.names`/`host_ns`/`gpu_ns`/
`sampled` (689-692); `Pass.targets` beside `Pass.loads` (197-198);
`RecordingParts.parts`/`conds` and `batches`/`batchconds`
(kalaunch.jl:1317-1330); `Pool.retiring`/`retiring_at` (pool.jl:446-447);
`SubmitChannel.pending`/`retiring`/`taken` (lifetime.jl:135-151). Cost:
est. 100 lines of sync code plus the invariants nobody checks.

**P13. Abstract types with a single subtype.** `Compilation` → `Compile`
(phases.jl:90, types.jl:800) with nine accessor forwarders; `Window`
(api.jl:89) one per backend; `CompiledRTPipeline` (coretypes.jl:497).
`Phase`, `Policy`, `TransientResource`, `ArgPassing`, `LoadOp` have 2-7
subtypes and are fine.

**P14. `Any`-typed fields and the code written to recover from them.**
types.jl has 27 `::Any` fields. `Pass`: 8 of 14 fields (`targets::Vector{Any}`,
`depth`, `dst`, `viewport`, `dispatches::Vector{Any}`, `predicate`,
`hostcond`, `loop`, 197-236). `DrawCall`: 6 of 8 (117-135). `Trace`: 3 of 4.
`TransientBuffer.block::Any` (260), `TransientImage.view/memory/source`
(324-329), `Graph.views::Dict{…,Any}` (418), `Plan.slabs/arenas/parts::Vector{Any}`,
`recording::Any`, `patchtab`/`pending_patches` with `Any` targets (705-766),
`Compile.pipelines::Set{Any}` (811), `BufferBlock.buffer/memory/ref::Any`
(825-834), `PassPlan.barrier::Any` (659), `ArgMemory.indirect::Vector{Any}`
(627), `Outstanding.tag::Any`, `Stamp.channel::Any`, `Block.memory/kind/
constraint::Any` (pool.jl:221-224). The recovery code then documents itself:
`ArgMemory.token` as a `RefValue` to dodge boxing through a UnionAll field
(types.jl:620-640), `submitplan!` narrowing `Plan.recording` (kalaunch.jl:
1361-1381), `argtoken`/`setargtoken!` accessors (636-640), type parameters
on `CompiledDraw`/`CompiledDispatch` justified as "concrete throughout … a
dynamic lookup per draw per frame" (490-513, 533-557), `::LaunchPlan`/
`::IterPlan` asserts and function barriers (ka_backend.jl:497-511, 689-721),
`Vector{Any}` caches explained at coretypes.jl:523-540. Cost: est. 150 lines
of workaround and explanation in my files.

**P15. `@generated` and `Val` where a plain method would do.** Five
compile-time walkers over the same argument tree, each written to keep an
allocation off a per-run path: `holdleaves!` ×3 (lifetime.jl:307-325),
`refuserefs` (dispatch.jl:112-121), `anydirty`/`landstores!`
(resources.jl:497-505, 564-566), `devicepointeroffsets` (packing.jl:254-259),
`find_tlas_in_args` (ka_backend.jl:369-390). In the target the argument
tree is walked once at declaration (`location`, `dependencies!`) and packed
once at compile (`bindings!`); none of these is on a run path. Cost: est. 60
lines in my files.

**P16. Argument packing/encoding done more than once.** See 2.5: a
type-level walk and a value-level walk in packing.jl (144-178), a compile
walk and a record walk on Metal, three copies on Vulkan, and the patch
table (`notepacked!`) as a fourth pass over the packed bytes
(packing.jl:276-285, reading addresses back with `unsafe_load`). Cost: est.
100 lines in packing.jl (285 → ~40).

**P17. Device/backend duality.** Every verb accepts a `Device` or the KA
backend under it: `todevice`/`backend` (api.jl:65-66, 123-125),
`kibackend` as a third object (kalaunch.jl:195-209), `supports_batch_queue`
×2 (queue.jl:39-43), `native_gemm_available(::Union{Device,KA.Backend},…)`
(backend.jl:381), `videodecodes` ×3 (ka_backend.jl:115-116,
vulkan/graph.jl:137), `Compile{D}` parameterised so "a backend that needs a
different execution model says so by being more specific" (kalaunch.jl:
14-16; types.jl:800) — the escape hatch api.md rule 1 closes. The target's
`Device <: KA.Backend` (api.md:34) removes the pair. Cost: est. 40 lines in
my files, est. 150 across `src/`.

**P18. Pointers and unsafe calls where a Julian type works (owner's rule).**
`notepacked!` reads device addresses out of packed bytes with
`unsafe_load(Ptr{UInt64}(at + f))` (packing.jl:280); `emitpatches!` boxes a
`UInt64` into a `Ref` and passes `unsafe_convert(Ptr{Cvoid}, ref)` to write
8 bytes (kalaunch.jl:833-834); `storefrom!` passes `pointer(src)`
(869-873); `emitinline!(::Immediate)` is an `unsafe_copyto!` (1521-1523);
`ArgMemory.ptr::Ptr{UInt8}` (types.jl:619). The target's stores are typed
upload regions and `emitstore!(b, dst, upload)` (api.md:61, 318), which
removes every pointer from core. GeometryBasics `Vec`/`Mat`: no vector math
in my files; the viewport is an `NTuple{4,Float32}` (build.jl:604) where a
`Vec4f` is the convention.

**P19. Base names shadowed.** `count` (api.jl:442) and `stride` (439) are
Mantle functions that shadow `Base.count`/`Base.stride`; metal/record.jl:748
has to write `Base.count`, resources.jl:339-342 warns about `strides` vs
`stride`. `copy!(g, name, dst, src)` (build.jl:1220) is not `Base.copy!`'s
meaning (resources.jl:646-648 says so). `record!` means two things (P5).

## 4. What the design docs miss

Things my files do that api.md/the plan have no replacement for, and things
the plan assumes that the code shows to be harder.

1. **Sub-array hazards.** Today two disjoint slices of one buffer get no
   barrier: `BufferRange`/`slice` intern a `(resource, range)` as its own
   resource (types.jl:446-461, build.jl:1095-1118), `overlapping` tests
   range intersection (build.jl:1363-1379), and `viewof` declares a view as a
   slice of its parent (build.jl:184-188). The plan makes the resource the
   array's state, "one resource for an array, its views and reshapes"
   (plan 147-152; api.md:447 and its negative control). Under that rule a
   kernel writing `view(a, 1:n÷2)` and one writing `view(a, n÷2+1:n)` in one
   graph get a barrier they do not need, and two passes over disjoint halves
   serialise. The plan should say whether `dependencies!` keys on (state,
   byte range) or accept the lost parallelism; today's `test_window.jl`
   "disjoint slices of one buffer are not ordered against each other"
   (named at access.jl:1290-1292) pins the current behaviour.
2. **Unordered (atomic) accesses.** access.jl infers that a store at an index
   claimed by an atomic read-modify-write on the same argument cannot
   collide (`disjoint`, `storetouch`, `llvmstoretouch`, access.jl:567-588,
   1011-1015, 1155-1173) and `usagetype` turns it into `Unordered`
   (66-69; usage.jl:156-169, transition.jl:95-105). access.jl:24-30 states
   the stake: a wavefront tracer's per-pixel radiance costs "one barrier per
   stage" without it. Neither api.md nor the plan mentions unordered or
   atomic accesses; `Access` in api.md:215 is "last write and reads since".
   `dependencies!`/`barriers!` need the third bit, and `Kernel.access::Vector{Touch}`
   (plan 362) must keep it.
3. **Stage of an access per node type.** `Traced{U}` (usage.jl:171-207)
   exists because a ray-tracing launch's storage reads are at a different
   pipeline stage than a compute dispatch's and a barrier lowered to compute
   stages "is no dependency at all" on NVIDIA (usage.jl:176-182). api.md:4.5
   says `deps` carries "the derived barrier" but does not say that the
   barrier is a function of (access, writer node type, reader node type).
   `Deps` (api.md:223) needs the stage, or the device type's barrier helper
   needs the node types on both sides.
4. **Reach rule vs. structs-of-arrays and closures.** api.md:108 "The
   arguments are the only way a call reaches a graph" and 466 "an argument
   that reaches device memory but is not a `MantleArray` … throws". Today
   `resourceleaves!` walks arguments to depth 16 to find the arrays inside a
   work queue or a scene (build.jl:144-151, 170-220, `LEAFDEPTH`), and
   ray-tracing shaders are closures over material arrays whose accesses are
   the closure's own entry (access.jl:500-503; vulkan/graph.jl:1199-1207
   holds them per record). The plan must say that `location`/`dependencies!`
   descend into isbits structs and `MultiTypeSet`s to find `MantleArray`
   leaves, and where a shader closure's captured arrays are declared (they
   are not arguments). The vararg element tracking (access.jl:311-338) is
   the same question for `f(queue, args...)` wrappers.
5. **Argument block layout is per device type.** `bindings!` "lays out the
   argument block" (plan 487; api.md:813-817 "fields sit directly in the
   slots"). Today the slot offsets come from the compiled SPIR-V entry
   wrapper on Vulkan (`info.arg_offsets`, `byval_llvm_sizes`,
   vulkan/graph.jl:1180-1182, 2027-2029; packing.jl:139-143) and are chosen
   by the backend at 256-byte alignment on Metal (packing.jl:164-167,
   record.jl:287-290), and `notpassed` (packing.jl:130) says which arguments
   have no slot at all. `KernelInfo` (api.md:304) has no layout field.
   Either `compilekernel` returns the layout or `bindings!` gets a verb for
   it; the plan says neither.
6. **Host-side conditional regions.** `when!(f, g, cond)` skips whole
   recorded pieces on a host `Ref{Bool}` at submit (build.jl:311-369;
   kalaunch.jl:1345-1359): a false condition costs one boolean read. The
   target's `when!` is one launch under a GPU flag (api.md:111); the launches
   are still submitted with count 0 (plan 319-339). The plan should state
   that host-known branches are either two graphs or a GPU flag, and what a
   skipped region of 50 nodes then costs. 70 test references, 0 consumer uses
   — the decision is cheap today.
7. **Arena growth under a tenant.** build.jl:1659-1667 records a live
   correctness bug: a plan whose shared arena moved keeps writing the old
   block on a backend that cannot patch (`remap!`, build.jl:1776-1797;
   `reserve!`, pool.jl:1163-1197; `patchable`). The plan says parts are
   shared and held per run (plan 652-657, 703-706) and that a graph array
   beyond its slot marks `recompile` (api.md:390), but not what happens to
   the *other* tenants when a part grows for a new one on a device without
   sparse residency (`createreserve`, api.md:292, is for arrays, not arena
   parts). Either parts never move (pre-sized from `caps`, new tenants that
   do not fit get a new part) or every tenant is marked stale; say which.
8. **Recording from several threads.** The single-writer rule
   (`ownthread`, lifetime.jl:209-241) exists because a `VkCommandPool` is
   not thread-safe. api.md:449 wants "any thread compiles, records, runs",
   and `Builder(dev, ch)` takes "command objects from the device's free list"
   (api.md:314) with the channel lock only at submit (lock order,
   api.md:410-416). A free list of command buffers from one pool recorded on
   two threads is the race `ownthread` prevents; the device type needs a
   pool per thread or the free list must be per channel with recording
   under the channel lock. The plan does not say.
9. **Submission split by estimate.** The measured partition exists because
   an unmeasured pass "is taken to be unbounded" and the watchdog returns
   garbage, not an error (kalaunch.jl:1204-1208; backend.jl:636-646, the
   TRELLIS NaN case). The plan's roofline (plan 516-527) needs
   `caps(dev).throughput[T]`/`.bandwidth` (api.md:256), which are open
   (api.md:538), and a node whose estimate is 10× low is a lost device. The
   plan does say measurements calibrate the next compile (526); it should
   also say the first run of an unmeasured plan is cut conservatively, which
   is what today's code does.
10. **`DeviceRange` ceilings.** A kernel launched over a device count is
    compiled against a 1 Mi ceiling so KA's `__validindex` passes
    (build.jl:1483-1493; ka_backend.jl:882-891 restates it with its own
    `1024*1024*ws_prod`). The plan makes the KA ndrange a slot the head writes
    (plan 346-347, 829-831), which removes the ceiling — good — but the
    `CompilerMetadata` *type* depends on the ndrange's dimensionality and
    the group (access.jl:1237-1245, vulkan/access.jl:29-32), so the access
    walk and the compile must agree on the context type per (kernel, rank,
    group). `Kernel` at one `CodeInstance` per "compiler configuration"
    (plan 379-384) is per (f, argtypes); the plan should say the iteration
    context type is part of that key.
11. **Per-run stores vs. host-written transients.** `TransientBuffer.hostwritten`
    (types.jl:274-285) exists for a per-frame plane whose bytes may alias dead
    transients, with the 28× placement cost of registering every transient
    as writable. api.md:127 "There are no host-written graph arrays" moves
    such planes to device arrays the graph names. That is a persistent
    allocation per plane per graph where today it is arena bytes; the plan
    should say that this is accepted (it is the right trade: no aliasing
    question) so the editor's decoded-frame path is designed for it.

## 5. Minimal form

What the same functionality needs in the target design, est. lines of code
(not prose), and what each part consists of. Counted against the plan's
module boundaries, not today's files.

| part | est. lines | consists of |
|---|---|---|
| types | 300 | `Graph`, `GraphDevice`, `Plan`, `CompileState`, the `Node` subtypes (kernel, trace, clear, draw, group, host, head, prepare, copy, loop, render, build, foreign, present), `CompiledNode`, `Deps`, `Origin`, `Bound`, the `Compiled*` records, `Recording`, `Submission`, `SubmitChannel`, `Token` (api.md section 3). No `Any` fields: every node type is concrete and compile fills a `CompiledNode` per node. |
| build (declare) | 400 | `dispatch!`/`trace!`/`draw!`/`clear!`/`when!`/`repeat!`/`Group`/`delete!`/`sort!`/`HostCall`/`copyto!` nodes, `location`, the leaf walk from arguments to array states, `refusenothing`/`refuseempty`, origin capture, graph lock and `freed`/`recompile` flags. Today's build.jl:30-43, 65-96, 153-220, 302-444 (half), 949-1078 (half), 1220-1243, 1586-1629 are the seed. |
| access | 900 | access.jl minus `AccessCache`, the `intrinsic_usage` probe chain and the view `devicetype` methods, plus one `touches(dev, f, lead, args)` replacing the five per-backend copies. The llvmcall regex heuristics (access.jl:1051-1193, ~140 lines) can go once every device intrinsic is declared, but that is a decision, not a given. |
| plan (compile) | 1 200 | `copy`/`head`/`layout!`, rewriters, `dependencies!`, `compile!(s, node)` per node type (today `run!(::Pipelines)`, `bake`, `compiledraw`), `costs!`, `schedule!`, `hostcuts!`, `liveness!`/`place!` (phases.jl `Liveness`/`Place` and memory/placement.jl stay as they are), `sharing!`, `barriers!` (transition.jl's rule, keyed by memory), `submissions!` (today's `partitionranges`/`submissionbatches` reshaped around the estimate), `summary!`, `bindings!` (the NotPassed rule and slot layout, the cell copy table, the prepare tables). |
| record | 250 | One walk over the stages calling the builder verbs (today's `emitplan!`…`emitwork!` without kind branches, 907-1012), the head/prepare node kernels (`multiprepare!`/`writegroups!`, 546-583), `BuildRecording` re-record, window-sized segment re-record. |
| run | 400 | `run!(g)` loop (plan 787-795), `applywindows!`, `recompile!`, `run!(pl)`: arena-part holds, inbox/namers wait computation, pending-change prefix per stage, token claim + `submit!`, host nodes between stages, present, `Again`/`Skipped`. |
| sync + release (outside graph/, replacing lifetime/submission/queue/pool's second engine) | 500 | `SyncState`/`Access`/`Change` subtypes, `lockexpanded`, the namers/inbox lists, one release engine `retire!(pool, x, tokens)` with the lock-free inbox, `passed`/`waitfor` on tokens, `recycle!`. |
| **core graph total** | **≈3 450** | against 7 810 in `src/graph/` today, plus the 2 053 of vulkan/graph.jl and 1 690 of metal/record.jl that mostly re-implement the walk, submission, stores and patches per backend (est. 1 900 of those 3 743 go). |

The owner's 10 k figure for the whole package is consistent with this: the
graph core at ≈3.5 k, placement/pool/sync at ≈2 k, arrays/images/TLAS at
≈2 k, and two device types at ≈1.2 k each if they implement only section 4.

What survives nearly verbatim: the taint walk (access.jl:309-1193), the
transition rule (transition.jl:95-131), the placement phases (phases.jl
`Liveness`/`Place`, memory/placement.jl), the prepare kernel
(kalaunch.jl:546-583), the partition cut (kalaunch.jl:1221-1252 minus
`when!` and measurement), the NotPassed rule (packing.jl:101-130).

## Rule violations, listed

- `_name` functions: none in `src/graph`; `_find_tlas`, `_maybe_tlas`
  (vulkan/array/ka_backend.jl:391-401).
- Globals: `LEAFDEPTH` (build.jl:203) and `INDIRECT_CEILING` (build.jl:1491)
  are single-use consts that the rule says are kwargs; `INDIRECT_STRIDE`
  (1259) is restated as a literal at metal/record.jl:751. Mutable globals
  outside my files: `DEVICES` (vulkan/graph.jl:51), `VK_CONTEXT_REF` (183),
  `BACKEND_PROBES` (runtime/backends.jl:96).
- `try … catch` without rethrow or report: none in `src/graph`;
  vulkan/graph.jl:863-873 logs and continues.
- Name variants instead of dispatch: `trace_closest_hits!`/`_indirect`/
  `_anyhit`/`_anyhit_indirect`/`trace_rays!`/`trace_rays_indirect!`
  (backend.jl:963-965), `native_{gemm,batched_gemm,attention,conv2d}_dispatch!`
  (backend.jl:307-426), `draw_in_pass!`/`draw_indexed_in_pass!`/
  `draw_indirect_in_pass!`/`draw_mesh_in_pass!` (vulkan/graph.jl:1982-2002),
  `emitrangewriter!`/`emitgridwriter!` (record.jl:1247-1283). Conversely,
  one name for different operations: `acquire!`, `release!`, `reclaim!`,
  `retire!`, `refit!`, `record!` (section 2.4, P5).
- Vec/Mat: viewport as `NTuple{4,Float32}` (build.jl:604).
- Pointers/unsafe where a type works: packing.jl:276-285, kalaunch.jl:
  833-834, 869-873, 1521-1523, types.jl:619.
- Defensive code on known types: section 3 P2 (97 nothing-checks, 5 probes).
