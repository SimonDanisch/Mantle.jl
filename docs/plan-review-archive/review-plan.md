> Checked afterwards (2026-09-30):
> - Finding 25 is WRONG: `where(a) = a + 1` defines and calls fine in Julia 1.12.7 (evaluated in a test module, returned 42).
> - Finding 50 is CONFIRMED: Vulkan conditional rendering covers only draws, dispatches and vkCmdClearAttachments (refs/vkdocs chapters/drawing.adoc:2092-2104); copies, barriers, acceleration-structure builds and trace-rays calls are not skipped.

Review of `/sim/Programmieren/VideoEdit/dev/Mantle/docs/recording-plan.md`, checked against `/sim/Programmieren/VideoEdit/experiments/plan-review/decisions.md` (A and B treated as binding). I changed no files and ran no Julia or GPU code. I checked two facts outside the plan: JuliaSyntax's handling of `where` and GPUArraysCore's `allowscalar`.

## 1. Decisions (B) against the plan

1. **contradicts-decision (B14 missing).** The plan never mentions B14.
   - L301-303: "Nothing runs at declaration. The node stores the function … Compile calls `lower` with the graph's shapes; a shape change is a recompile". Also L285-296 (`select!` calls `lower` at compile), L314-315 and L485 (`CallNode`).
   - L330-332 and L1022: `DeviceRange(count; max)`. B14 has `DeviceRange(a)`.
   - L439-441: `resize!` "appends the new address". B14 says `resize!` is an eager write of the size.
   - L340-342 and L1109: selection and MPSGraph use "the plan's shapes".
   - Should say: `sum!(x)` on a graph adds the implementation's kernel nodes at the call. Sizes live on the GPU, with a capacity for placement. Launches run over `DeviceRange(a)`. `resize!` writes the size eagerly. State whether `CallNode`/`select!` survive only for selected library operations, and which size selection uses (the capacity?).

2. **contradicts-decision (B7).**
   - L3-5: "the working tree at 7e64cd5 carries an uncommitted submission split … (`measure!`, a re-record inside `run!`)". L514: "The uncommitted `measure!`/re-record path".
   - B7 says it was committed in 692a4e1, and L325-326 already cite 692a4e1.
   - Should say: "committed in 692a4e1; phase 1 replaces it".

3. **contradicts-decision (B8, B12).**
   - L1332-1334, `test_one_path.jl`: "goes through core's compiled-lowering cache (counted)". Core keeps no kernel cache and no cached one-op plans (L416, L741-743, B8, B12).
   - The same test lists `pass!` and `trace_rays!`, which L771 and L1416 delete.
   - Should say: every eager entry reaches `compilekernel(dev, …)` and one `submitnative!`, counted by the device type's compile count and the token delta. The entry list should use `render!(dev, …)` and `trace!(dev, …)`.

4. **contradicts-decision (B10, push).**
   - L784-785: "`run!` sees the new extent (the window's events, or acquire reporting the swapchain out of date), recreates the swapchain".
   - L1313-1315: "no call from the test besides `run!`".
   - L1425: "resize handled inside `run!`".
   - L443-445 follow B10 (`resize!(win, w, h)` pushes). These three passages contradict it.
   - Should say: `resize!(win, …)`, called from the event, pushes the change to the plans. `run!` only consumes what was pushed. OUT_OF_DATE from acquire is the only driver-reported input. The test calls `resize!(win)` and then `run!`.

5. **contradicts-decision (B10).**
   - L663: `function run!(pl::Plan)`. L1351 says "`run!` only ever runs a graph".
   - "Plan" also leaks into user-facing text:
     - L710-711: "handing a plan new arrays"
     - L1325: "drop every reference to a plan"
     - L1039: "one plan run"
     - L1491: "frame plans recorded"
   - Should say: `run!(g::Graph)`, with an internal verb name for the plan step. Use "graph" in all user-facing text.

6. **contradicts-decision (B11: one recording, no slots), partial.**
   - L778-779: "Vulkan: one recording per image, `run!` submits the one for the acquired index". L783: "A frame slot's fence". L621 records one recording per segment.
   - L680 says "One argument block, one recording."
   - Should say: either list per-swapchain-image recordings as an explicit exception for Simon to rule on, or record once into a Mantle image and blit to the acquired image per run. Rephrase as "each segment recorded once".

7. **contradicts-decision (B2: HostDevice deleted).** L586 `SubmitChannel.native`: "… | stream + GPU-written counter | host". The `host` alternative is left over from HostDevice. Delete it.

8. **contradicts-decision (B2: devices opened lazily).** L888-889: "A `Device` is opened explicitly and owns its pool". L126-127 and B2 say driver objects are created on first use. Say "created lazily on first use". See finding 37 for the identity hole.

9. **contradicts-decision (B3).**
   - L303-304: "graph arrays are not `AbstractArray`s and `KA.get_backend` is not defined on them".
   - L196-199 and B3 say it is an `AbstractArray` and `getindex`/`KA.get_backend` throw.
   - Separately, L190-196 ("On a device it is an `AbstractGPUArray` … On a graph it is an `AbstractArray` too") cannot hold for one parametric type: the supertype of `MantleArray{T,N,D}` cannot depend on `D`.
   - Should say: `MantleArray{T,N,D} <: AbstractGPUArray{T,N}` for every `D`. Methods on `MantleArray{T,N,<:Graph}` throw. Broadcast style dispatches on `D`.

10. **contradicts-decision (B5: compiled data in the node).** L482 is headed "produces, stored in the plan". L484 (`dependencies!`): "Stored in the plan". L490: "the resolved TLAS per dispatch" is kept at plan level. L457-462 put these in `CompiledNode`. Fix L482, L484 and L490 to say "stored in the node's `CompiledNode`".

11. **contradicts-decision (B6).**
    - L449: "Compile options move to `Graph(dev; policy)`". L486 and B6 use `Graph(dev; rewrites = (…))`.
    - L486: `rewrite(r, nodes, dag, selections) -> nodes`. L1345-1347: "a rewriter is a callable value in the same list" applied to `CompileState`.
    - Pick one signature: a rewriter is a function on `CompileState`. Delete `policy` or define it.

12. **contradicts-decision (B15).**
    - The Open decisions section (L1541-1553) lists three items, and one of them (the re-sync) is not in B15.
    - Missing from it: FuseBroadcasts (L1537 even lists it as decided), rewriter and attribute names (only inline at L486), the streamed-weights keyword (L891-892), cross-device `copyto!` (absent from the whole plan), OS memory-pressure events (inline at L878-879), descriptor-bound handles on Vulkan (inline at L715-717 and L947).
    - Add them all; remove FuseBroadcasts from Decision 11.

13. **contradicts-decision (B-list coverage).** The Decisions section (L1507-1539) has no entry for:
    - B11 (static block, head kernel, `GPURef`, stores are eager copies)
    - B10 (the graph manages its plan, fixed after its first run, changes pushed)
    - B4 (a Graph is a device)
    - B5 (nodes)
    - B8 (no core kernel cache)
    - B13's counted holds and "pressure only in `acquire!`"
    - B14

    Add them.

14. **internal-contradiction (`lower`, B12, B14).**
    - L290-292: `lower` on "a device (an eager call runs them now)".
    - L731-733: `l = lower(dev, f, args...)`, `k = kernel(dev, l)`, `submitnow!(dev, l, k)`. Here `lower` returns a value that is compiled and submitted afterwards. `kernel(dev, l)` is also singular for a multi-kernel lowering.
    - Decide one contract. Under B14 the implementation runs at the call on either target, and the separate name `lower` may go.

15. **hole (B4).** "A Graph is a device" (L244) is not placed in the type hierarchy. `Device <: KA.Backend` (L135), `dispatch!(dev::Device, …)` (L730) and `dispatch!(g, …)` need a stated relation. If `Graph <: Device`, graph arrays get a KA backend and the methods are ambiguous. Define the abstract types.

16. **cosmetic (B2).**
    - L139-141 state the device-type registry as decided; B2 and L1547 keep it open.
    - L1227: "a backend or device type". No backend types remain after deletion.

## 2. Standing rules (A)

17. **rule-violation (A4).** L718-719 and L849-850: pages "unmapped under pressure are mapped again before the submission". The plan does not say how `run!` learns this; without a push it is a per-run check of arena state. Should say: the pressure policy that unmaps pages pushes a "map before next run" item to the plans it affected.

18. **rule-violation (A4), hole.** L229-232: "Mantle's next use of the region waits on an event recorded on that stream". No call tells Mantle when vendor work on a `CuArray(a)` view ends, so Mantle would have to discover it. Should say: an explicit handback, for example a scoped `CuArray(a) do v … end` that records the event at block end.

19. **rule-violation (A3).** L1310-1312: `test_record_is_pure` "in a checking mode throws on reading `store`, `len` or `block`". A mode is ambient state; L1322-1323 rejects a process-global switch for the same reason. Make it an argument, or a checking wrapper type passed to `record!`.

20. **rule-violation (A3), low.** L758-760: "a scalar read under `allowscalar`". GPUArraysCore implements `allowscalar` with `task_local_storage(:ScalarIndexing, …)` (`GPUArraysCore.jl:97-208`). Plan Rule 5 (L43-45) says "nothing is found by task-local storage". State that this is accepted as GPUArrays' interface, or drop the scalar path.

21. **rule-violation (A2).** L665: `fill!(entry, pl)` uses `Base.fill!` for "write this run's GPURef values and patches", which is not fill semantics. Use a distinct verb.

22. **rule-violation (A2), low.** L419-421: `KernelInfo` "fields it cannot answer are `nothing`". Core must then branch on `nothing` and supply defaults, which is close to the forbidden move at L1224. Either require the fields, or make "unknown" a dispatchable type with a documented core estimate.

23. **rule-violation (A1), low.**
    - L638: `emitrender!(b, node, deps)` hands the backend a node; every other emit verb takes a `Compiled*` command. Use `cmd::CompiledRender`.
    - L911: `compile_bindings` is a device-type verb beside core's `bindings!` (L490), with no statement of who decides the argument layout.
    - Also, `compile_draw`/`compile_trace`/`compile_bindings` use underscores, unlike `compilekernel`.

24. **rule-violation (A5, plain language), cosmetic.** Metaphors:
    - L1065 "Bake" (section title)
    - L707 "ride in the run's update entry"
    - L986, L990, L1014, L1018 "glue"
    - L68 "seam"
    - L704 "The GPU queue is the queue" says nothing.

25. **rule-violation (A2 naming), cosmetic.** L214 and L283: `where(a)`. `where` is a Julia operator token (JuliaSyntax `kinds.jl:820`, level `BEGIN_WHERE`, like `in`/`isa`). Whether a function named `where` parses is unverified. Rename it (for example `home(a)`) or use `device(a)`.

## 3. Internal contradictions

26. **Graph-persistent arrays shared between graphs.** L274-276: a `MantleArray(g, …; persistent = true)` can be passed as an argument by "other graphs". L279: "Arguments from two graphs in one call are an error." L179-180: "anything shared between graphs and eager code are `MantleArray(dev, …)`". L1004-1005 connects graphs by "a persistent buffer" without saying which kind. Say that cross-graph state is `MantleArray(dev, …)`.

27. **CUDA/HIP patching.**
    - L651: "node handles kept for patches". L815: "a node parameter update on CUDA/HIP". L1474: "node handles for patches".
    - L713-714: "node parameters are never updated". The L947 table says the head kernel does it on all four backends.
    - L814 ("a descriptor write on Vulkan") also contradicts L715-717 ("which per resource kind is open").

28. **Record uploads.** L656-657: "Record … uploads nothing … reads no mutable resource field". L681: "a static block written once at record". Say the block contents come from `bindings!` (compile) and record only references them.

29. **Loops on Vulkan.**
    - L327-329: "the body's recording submitted `n` times".
    - L948: "indirect counts, conditional rendering".
    - L673-674: "submits each segment once, all up front, and nothing else".
    - L509-510 makes loop iterations cut points.
    - Pick one mechanism per capability and state `n` submissions as an exception, or remove it.

30. **Metal one command buffer vs the split.** L1081 is titled "Run: one command buffer", with segments executed inside it. L502-504 cut submissions against the driver timeout; segments inside one command buffer do not reduce the GPU work per submission. Also, L1085-1087 (compute encoder, then render encoder) cannot express compute after render inside one run.

31. **Metal open item 4.** L565: "whether that also orders memory on the same queue is Metal open item 4". Item 4 (L1152-1155) is about `barrierAfterQueueStages` and is not marked open.

32. **Compile step order differs in three places.**
    - L474-479: `select!` and rewriters first, then `dependencies!`, …, `barriers!`, `summary!`, then `pipelines!`, `costs!`, …
    - The table at L482-495 starts with `dependencies!`, has no `alias!` row, and lists `barriers!`/`summary!` after `bindings!`.
    - L1345-1347 has yet another order.

33. **Barrier emit verb.** L1251: "`vkCmdPipelineBarrier2` only in the barrier emit verb". L633-645 have no barrier verb; `emit_barrier!` is folded into the `emit*` verbs with `deps`.

34. **`Device()` platform choice.** L1451-1452 delete "`Device()`'s `@static Sys.isapple()` choice inside Mantle (it stays for application entry points only)". L122 defines `Device()` as Mantle's, and L1257-1259 allow `@static Sys.is*` "never to select behaviour". Say: `Device()` is defined in each platform's included file, with no `Sys` check.

35. **Pressure policy in `run!`.**
    - L876-877: "in `acquire!` … not in `run!`".
    - L673: "`run!` … allocates no device memory".
    - L718-719: `run!` maps arena pages, which increases memory use.
    - Define `run!`'s mapping as a call to `acquire!`, and restate L673 and `test_run_is_submit` accordingly.

36. **Phase order and definition of done.**
    - L1207-1208: every phase passes on Metal, ROCm and lavapipe. But Metal joins the core model only in phase 8 (L1456-1466), ROCm is rebuilt in phase 9 (L1473), and phase 3 deletes `openrun`/`closerun!`/`submitrun!` "in every backend" (L1386).
    - Phase 2 adds "CUDA/HIP VMM" (L1364), but the CUDA backend is new in phase 9.
    - Phase 1 adds `select!` (L1345), which calls `lower`; the registration (`opstyle`, `lower`) arrives in phase 5 (L1411-1412).
    - L87-90: the `when!` barrier bug is listed in "Bugs to fix first", which phase 0 fixes (L1302), yet it says "Fixed in phase 1".
    - The order of work (L1193-1203: tests, then bugs) differs from phase 0 (L1301-1303: bugs, then tests).

37. **Plan sections with no phase.** None of these appears in any phase's Adds or Deletes (checked by grep over L1293-1506):
    - the per-device selection table and the move of `vulkan/array/gemm.jl` selection (L363-367)
    - the deletion of `librarygemm`/`native_*_dispatch!` (L369-373)
    - the rewriters `FuseEpilogues`/`MergeRegions`
    - `HostCall` and host-signalled segments (L375-396; only CUDA/HIP foreign nodes appear, in phase 9)
    - `repeat!` with one body (L321-329)
    - `DeviceRange` refused without `max` (L330-332)
    - counted holds, `free!(buf)`/`free!(g)` (L852-858)
    - streamed weights (L891)
    - the `LavaPipe_jll` extension (L120), although lavapipe is required from phase 1

38. **Hikari lock is a forbidden workaround.** L1490 ("`render!`/`invalidate!` on one `VolPath` serialised by a lock on that `VolPath`") is a consumer-side lock, which L1229 forbids, unless the plan states that concurrent `run!` on one graph is unsupported (see finding 49).

## 4. Holes

39. **Device identity.** L888-890 ("Opening one GPU twice gives two devices") together with L122 (`Device()` in each application entry point) and L890 (`DEVICES` goes): an editor and a package that each call `Device()` get two pools on one GPU. That defeats the goal at L20-22. The plan must say whether `devices()`/`Device()` return the same object each time. If yes, there is a process-level cache (which rule allows it); if no, applications must pass the device.

40. **`resize!` of an array used by several graphs, some in flight** (L438-442, L708-712). The plan does not say:
    - that the old region is retired through its access record. `Submission.holds` holds "the arguments" (L601), meaning the `MantleArray`, which after `resize!` points at the new region, so it does not keep the old one alive;
    - whether contents are copied, and how that copy is ordered after in-flight writers;
    - that dims and ndrange change. "A sparse buffer grows in place and changes nothing" (L441) is wrong: the length changes;
    - how `resize!` on thread A is ordered against `run!` taking its entry on thread B. It needs the same lock as the access records;
    - what `resize!` means on a `MantleArray(g, …)`;
    - what `resize!` pushes to a graph that is declared but not yet compiled.

41. **Window resize mid-run.**
    - Beyond finding 4: `resize!(win)` runs in a GLFW callback on the main thread (L795), and "once that window's frames in flight are done" (L444) means it either blocks the main thread or defers. The plan does not say which.
    - Queue tokens passing do not prove the presentation engine has released the old swapchain images. `oldSwapchain` handoff and present fences are not addressed.
    - Per-image recordings: all of them are re-recorded, and nothing says who holds the old ones while they are in flight.

42. **Recompile while runs are in flight.** Compiled data lives in the nodes (L457-467), and `select!` "puts what it declares in the node's place" (L292). A window-resize recompile therefore:
    - mutates nodes that in-flight runs' block and recordings came from;
    - has lost the original `CallNode` it needs to lower again with the new shapes.

    Specify: replace `CompiledNode`s (never mutate them), retire the old ones by token, and keep the declaration.

43. **Argument-block write-after-read hazard.**
    - L688-690 claim "Consecutive runs of one plan are serialised anyway … so one block costs no overlap". This fails for a plan without transients, and for per-swapchain-image recordings.
    - The block, ring entries and host-written transient data are not listed as resources with access records (L553-554).
    - With several queues per plan (L528-530), the plan does not say which segment holds the head kernel, or how other queues wait for it.
    - If the first segment is a host-conditional `when!` segment (L494), the head kernel could be skipped.

44. **GPURef.**
    - The value is captured at `run!`, so a write while a run is in flight is safe.
    - Open: a multi-word value (a `Mat4f` camera) written on another thread while `run!` reads it can tear, and L699 rules out a seqlock.
    - Open: where a GPURef read by K kernels lives in the block (K slots, or one slot read through a pointer, which is the header cost C1 measures).
    - Open: how GPURef reaches an eager call, which has "no update ring" (L726).
    - Open: the ring depth.

45. **`free!(buf)` and `free!(g)` with in-flight or recorded use.** Holds are covered (L852-858). Not covered:
    - use of `buf` by eager code after `free!(buf)` (throw?);
    - `free!(g)` while its run is in flight: L449-450 release "the compiled state and the graph's arena tenancy" without a token;
    - `untenant!` shrinking the arena (L862-864) while the leaving tenant's run still reads the pages.

46. **Two devices.**
    - Cross-device `copyto!` is absent (B15 lists it as open; see finding 12).
    - Not covered: a graph on dev1 naming a `MantleArray(dev2)` (L279 covers only two graphs); token waits across devices (timeline semaphores are per `VkDevice`); vendor views across two drivers on one GPU (L130).

47. **Out of memory.** L871-879 cover only `acquire!`. Not specified:
    - the graph's state after compile fails with OOM (partial pipelines, block, arena growth released?);
    - OOM while `run!` maps pages after `takeentry!` and consuming queued updates;
    - OOM or submit failure after accesses were recorded (L555-558): the records then name a token that never signals, and every later user waits forever.

48. **A graph whose first run throws.** Not said: whether it counts as "run" (fixed), whether the user can retry, what is released.
    - Worse: with host nodes, "`run!` submits every segment up front" (L386) behind a host-signalled timeline value (L393).
    - If the `HostCall` throws, the value is never signalled and the queue blocks every later submission, including other graphs'.
    - A signal-on-failure path is needed that does not break the no-blanket-try/catch rule.

49. **Host nodes in a replayed graph.**
    - Calling foreign nodes "in order" requires `run!` to wait for the previous segment's token, so `run!` blocks. That contradicts L673-674.
    - `HostCall` arguments are graph transients in device-local arena memory. The plan does not say how the host reads them or writes `writes = (…)` back (staging copies, placement in `Readback`/`Unified`).
    - L387 ("never calls Mantle: no … wait") leaves no mechanism for this.
    - Not said: how "a size a host function computes reaches the device as a count" (L388).

50. **`when!` and `repeat!` with GPU-decided counts.**
    - `repeat!(g, maxiters; while_nonzero = flag)` on Vulkan: the host cannot know when to stop, so either all `maxiters` predicated submissions go out every run, or there is a readback. Neither is stated.
    - From memory of the extension spec, not verified here: `VK_EXT_conditional_rendering` predicates draws and dispatches, not copies, barriers, acceleration-structure builds or trace calls. So a `when!` region holding a `CopyNode`, `TraceNode` or build has no Vulkan lowering.
    - HIP "kernel-entry predication" adds a flag load to every kernel, the same indirection cost C1 measures.
    - `when!(g, cond)` never states cond's type: host-decided (segment skip, L494) or device-decided (gate).

51. **Threads.** L896 ("Compile, record, run and free on any thread") does not cover:
    - two threads calling `run!(g)` on the same graph (entry ring, update list, first-run compile race);
    - declaration into one graph from two threads.

52. **Transient that is also `hostwritten`.**
    - No API for supplying the data: `getindex`/`setindex!` throw on graph arrays (L197), and `copyto!(x, host)` on a graph array declares a copy node.
    - Not said: what happens if a run supplies no data.
    - "an entry is sized at compile for them" (L685-686) makes the "small" ring (L684) and the "one small dispatch" (L691) as large as all host inputs times the ring depth, for example image-sized inputs.

53. **Vendor views while a graph runs.** A recorded graph re-submitted while vendor work on the view is still queued on the vendor stream has no ordering; see finding 18. L231 ("Handing out waits") does not say whether that is a host wait or a stream wait.

54. **Recording invalidation is not enumerated.** "Fixed after its first run" (L435) needs a list of what invalidates a recording, each with its pusher:
    - window resize
    - `resize!` (C4)
    - arena migration where sparse binding is missing (L866-868: "moves to it when it is recompiled", but nothing triggers that recompile)
    - image recreation for descriptor-bound textures
    - device loss
    - Lava kernel edits

55. **Node types missing.**
    - `CompiledBuild` (L637), `CompiledLibrary` (L639) and `emitbranch!` (L641) have no node type in L314-317.
    - A closed-library call (cuBLAS on a stream) is neither a `HostNode` nor a `KernelNode`.
    - `when!` has no node.

56. **Compile counting (L424, L1311).** Core counts compiles per device but keeps no cache. `compilekernel` must report whether it compiled or hit its cache; that is not in its signature (L412).

57. **`timings(g)` (L451) and calibration (L512).** They need GPU timestamps, but the device-type verb table (L906-915) has no timestamp or query verb, and L515-516 delete the per-plan query pool.

## 5. Section C (experiments): places the results change

58. **L438-447** (plans registered per resource; `resize!` appends to each plan's updates; `movelisteners`/`notify_move!` kept). **Outdated by C2:** the header arena makes `resize!` a header write with no plan list. **Tension within C:** C4 re-records a plan's launches on a host-known resize, which again needs the list of plans using the array unless launch counts become indirect. That choice has to be put to Simon.

59. **L495 (`patches!`, "the patch table, built once from the address table"), L665 ("queued pointer patches"), L684 ("(offset, new bytes) pairs").** Outdated by C2: they become a compile-time header-to-slot copy table the head kernel applies every run. The ring then carries only GPURef values and host-written data.

60. **L691-693** ("one small dispatch per run, sized by the entry (no updates, no work) … on Vulkan it is not measured"). **Wrong now:** measured at 1.3-1.9 µs on AMD and 2.0-3.5 µs on NVIDIA for 1250-20000 slots (`v2_radv.txt`, `v2_nvidia.txt`). Under C2 the work is proportional to referenced slots, not to updates.

61. **L708-717** (patches pushed per plan). **Outdated by C2.** L713-714 (CUDA/HIP "every kernel reads its arguments through one pointer into the block") is **wrong per C3**: flat fields gain about 140 ns per small dispatch on NVIDIA. Descriptor-bound handles stay open.

62. **L813-817** (acceleration-structure rebuild as descriptor write, resource-ID store or node update) and **L947** ("patch" row). **Incomplete:** a TLAS address change becomes a header write; descriptor-bound TLAS remains open.

63. **L459-461, L680-682** (argument block, "its slice"). **Incomplete per C3 and C6:**
    - Say the layout is flat, with fields directly in the slots.
    - Say the memory is marked NonWritable (−25% at 1M elements on RADV).
    - Say KA ndrange values live in the block.
    - Today every isbits struct is passed as a pointer to an inline copy (launch.jl ~255), and RADV emits no scalar loads.
    - No phase changes the Lava entry wrapper; add it (phase 1 or 3).

64. **L323-324** ("`i` is a device value; per-step constants are a buffer indexed by it"). **Incomplete per C1:** a dependent load in every body kernel is the header read C1 measures at +120-490 ns. State that the loop node writes `i` into the slots.

65. **L192-195** (`MantleArray` as a `DataRef` over a region). **Incomplete:** add the per-array header entry in the device header arena, and its write ordering against in-flight head kernels. An eager GPU write ordered by the arena's access record is required; a host memcpy into mapped memory would race.

66. **Size handling.**
    - L303 ("a shape change is a recompile"), L295-296, L443-445, L676-678, L784-789, L1425-1426 (window resize recompiles and re-records): **incomplete per C4**, which re-records launches for host-known sizes (0.97 ms host for a 300-dispatch plan) and recompiles only for re-placement (not measured).
    - L439-441 (address-only `resize!`): **wrong**, since the launch count changes.

67. **L330-332** (`DeviceRange(count; max)`). **Incomplete per C4:** state the indirect cost (+45-60 ns NVIDIA, +190-245 ns AMD) or the grid-stride option, and that `VK_EXT_device_generated_commands` is excluded (15 µs preprocess on NVIDIA; ordering bug on 595.x, `dgc*_nvidia.txt`).

68. **L3-5, L512-516, L1316-1318, L1358.** **Incomplete per C5:** `census_radv.txt` shows SAM 2 recorded in run 1 and again in run 2 ("appended per run [624, 619, 3, 3]; last recording in run 2"). `test_run_is_submit`'s "in steady state" does not catch that. Add a test: the recording count of a graph equals 1 after runs 1..N, with SAM 2 as the case.

69. **L1085 (Metal head kernel), L1069 (Metal argument memory).** Still fine as written, but C measured only Vulkan (RADV, NVIDIA). Say so, so the Metal layout is measured rather than assumed.

## 6. Claims without a source

70. L3-5: "uncommitted". Wrong per B7.
71. L130-131: "the driver's memory budget already counts the other's usage". No source.
72. L204-210: the GPUArrays 11.4 capability lists. Version given, no file references.
73. L226-228: vendor constructor signatures (`MtlArray{T,N,S}(::DataRef{<:MTLBuffer}, dims; offset)`). No source.
74. L350: "No other vendor library has one." No source.
75. L688-690: "one block costs no overlap". Argued, not measured (see finding 43).
76. L692, L754-755, L1092, L1133: M5 costs and the "2.6x" are cited to code comments (`metal/device.jl`), not to a measurement file.
77. L895-899: `FROZEN_*` process globals, `iterplans`/`launchplans` unlocked. No file:line.
78. L958-962: ROCm 1.93 vs 2.20 µs and 322 ms, cited to a code comment.
79. L992: "about 80 MB, read from the code". No file:line.
80. L1028-1030: Horizon "timed eager dispatches, not a node". The interpretation has no source.
81. L1091: "About a dozen Objective-C calls per run". No source.
82. L1156-1173: `gemm_select.jl` has no path or commit; "within ±7%" has no file.
83. L1187-1188: ROCm "broke silently behind a fallback that returned `nothing`". No file:line for the fallback.
84. L1436-1437: "about 80 default-queue sites and 44 `vk_context()` calls in `src/vulkan`, not 30". No command, no commit, and "not 30" refers to nothing.
85. L786-787: token passing is claimed to suffice before recreating the swapchain. No source (presentation-engine release).
86. L7-11 and L1549-1553: all file:line references are to 7e64cd5 while 58a05a9 (19 commits later) has landed. The plan itself says the re-sync is pending, so every reference is unverified against current code.

## Five most important fixes

1. **Put B14 in and settle C2/C4 before rewriting the Run section** (findings 1, 14, 58-67). Sizes live on the GPU with a capacity. Kernel nodes are added at the call. `DeviceRange(a)`. `resize!` writes size and header eagerly. Decide between the header arena with a head-kernel copy table and per-plan pushed patches, including the C2/C4 tension over whether `resize!` must know its plans. Flat NonWritable argument slots, with the Lava entry-wrapper change as a phase item. Delete "Nothing runs at declaration", `lower` at compile, and "a shape change is a recompile" where they are superseded.

2. **Make window resize and every invalidation pushed, and list them** (findings 4, 6, 17, 41, 42, 54). `resize!(win)` pushes; `run!` never "sees" anything. Fix L784-789, `test_window_resize` and phase 6. Enumerate each invalidation with its pusher. Replace `CompiledNode`s instead of mutating them, and keep the declaration for recompiles. Rule explicitly on per-swapchain-image recordings against "one recording".

3. **Remove the contradictions with B8, B12, B3 and B5** (findings 3, 9, 10, 26). `test_one_path`'s "compiled-lowering cache" and the deleted `pass!`/`trace_rays!`. The `AbstractArray` vs "not `AbstractArray`" split and the parametric-supertype impossibility. Graph-persistent arrays passed to other graphs vs L279. Compile results "stored in the plan".

4. **Specify failure and concurrency semantics** (findings 40, 43-52). Host-node exceptions with up-front submitted segments (queue deadlock). OOM at compile vs at run, and records naming a token that never signals. A first run that throws. Concurrent `run!`/`resize!`/`GPURef` writes on one graph. The argument-block write-after-read hazard and the head kernel across queues and conditional segments. How `hostwritten` data enters a run. Vendor-view handback.

5. **Make the phases executable and complete** (findings 36, 37, 39, 12, 13):
   - The definition of done requires Metal, ROCm and lavapipe from phase 1, but they arrive in phases 8, 9 and nowhere.
   - Phase 2 adds CUDA VMM before the CUDA backend exists (phase 9).
   - Phase 1 uses `lower`, which phase 5 adds.
   - Assign the orphaned sections (selection table, rewriters, `HostCall` segments, one-body `repeat!`, `DeviceRange`, counted holds, `LavaPipe_jll`).
   - Settle device identity (two `Device()` calls = two pools).
   - Bring Open decisions and Decisions up to B15 and B10/B11/B14.