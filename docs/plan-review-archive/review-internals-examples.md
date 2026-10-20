> Checked afterwards (2026-09-30) against docs/internals.jl: findings 26 (one token, several channels: lines 375 vs 461), 28 (throw between reserve! and submit!), 29 (waits from firstuse, records from lastuse: lines 445, 459) and 51 (head kernel only in the first segment, skipped when that segment's when! is off: lines 330-331, 384) are confirmed. Finding 44's Julia claim is confirmed: base/lock.jl:174 disables finalizers while a ReentrantLock is held.

# Adversarial review of `internals.jl` and `examples.jl` against `recording-plan.md` and `decisions.md`

Abbreviations: **I** = `/sim/Programmieren/VideoEdit/dev/Mantle/docs/internals.jl`, **E** = `.../docs/examples.jl`, **P** = `.../docs/recording-plan.md`, **D** = `/sim/Programmieren/VideoEdit/experiments/plan-review/decisions.md`. No file was modified and no Julia was run. I read two Julia base sources to back findings 46 and 81.

## 1. Names, arguments, cross-file consistency

1. **inconsistency**, E:77-78, I:84-85, I:116-117, P:195 vs P:303-305, D B3.
   - Quote: `myhelper!(x)  # MethodError: x is not an AbstractArray`.
   - Problem: `v .= v .^ 2` is a broadcast. I:116-117 declares it into `g`, so nothing throws, and `g` gains a node that runs at E:152.
   - The plan contradicts itself: P:195 says "On a graph it is an `AbstractArray` too", while P:303-304 says "graph arrays are not `AbstractArray`s". D B3 says getindex and get_backend throw; it does not say MethodError.
   - Fix: follow D B3. Rewrite example 14 with an indexing helper and the `nodata()` error. Correct P:303-305.

2. **inconsistency**, E:85-86 vs I:132-144, I:586, P:288-298 vs P:730-734.
   - Quote (E:85-86): example 15's `lower` returns `(resizeplanar_kernel!, (…), size(dst))`.
   - Problem: in I, `lower` calls `dispatch!(t, …)`. P:731 (`l = lower(dev, …); k = kernel(dev, l)`) treats it as returning an object. E:85 and P:281 also name the target `dev`, although `lower` is called with a Graph too.
   - Fix: keep one contract, I's (`lower` dispatches on `t`), and fix E:85-86 and P:730-734.

3. **inconsistency**, E:82-83 vs I:77-79.
   - Quote (E): "one opstyle method, one lower method. Its eager form and its graph form now exist". I says the forwarding method on MantleArray is "Generated once per registered function (macro, name open)".
   - Problem: without that forward, `resizeplanar!(x, img; …)` calls GPUFiltering's own method.
   - Fix: show the macro line in E (name open).

4. **missing-case**, E:106, D B4, P:266 vs I:44-46, I:623.
   - Quote (E:106): `dispatch!(g, HostCall(align), durations, idx, nframes; writes = …)`.
   - Problem: I has no `dispatch!` for `HostCall`. The call reaches I:95, where the default `opstyle` gives `NotOp`, then I:108, which needs `args::Tuple, ndrange` and has no `writes` keyword: MethodError.
   - HostCall splats its arguments while raw kernels take a tuple (E:74).
   - `HostPart` (I:623) appears nowhere else; the node type is `HostNode`.
   - Fix: add `dispatch!(g::Graph, h::HostCall, args…; writes)` that pushes a HostNode, use one argument convention, and drop HostPart.

5. **missing-case**: `repeat!`, `render!`/`draw!`, `trace!`, `DeviceRange`, the KA launch `k(dev)(…; ndrange)`, `MantleArray(g, T, win)` and the `GPURef` constructor.
   - These are used at E:121, 130, 308, 297, 138, 36, 317 and 158 and are absent from I.
   - `LoopNode` (I:47-49) has `count::Union{Int,Resource}; maxiters` but:
     - nothing maps `while_nonzero` onto those fields;
     - nothing moves body nodes into `body` (I:99 and I:110 always push to `g.nodes`);
     - nothing says what the device value `i` is.
   - Fix: add `repeat!` using the same slice-after-`f()` technique as `when!`, and define `i`. Add `render!`/`trace!` with RenderNode/TraceNode fields.

6. **inconsistency**, E:158, P:170, I:474.
   - Quote: `GPURef(dev, Mat4f(I))`.
   - Problem: no other file gives a constructor or says where a GPURef lives. Eager use (`dispatch!(dev, k, (a, cam), n)`) is undefined.
   - Fix: define `GPURef(dev, v)` and state that an eager call packs `r.value[]` at the call.

7. **inconsistency**, P:712-714 vs P:813-815, P:1474, I:644.
   - Quote (P:712-714): "node parameters are never updated". Against it: P:813-815 "a node parameter update on CUDA/HIP", P:1474 "node handles for patches", and I:644 "kept: patches update this node".
   - Fix: pick one answer. Under D C3 (arguments passed directly, "flat") it becomes the node-parameter update (see 88).

8. **inconsistency**, I:423-430 vs P:784-787, P:1424-1426, E:314-316.
   - Quote (I): `resize!(w::Window…)` calls `recreateswapchain!` from the event callback. P, phase 6 and E say `run!` recreates the swapchain.
   - Fix: choose one. To comply with D A4, `resize!(w)` should only mark the window and its plans; the `run!` that owns the window recreates the swapchain under the window's lock (see 34).

9. **inconsistency**, P:1332-1334 vs D B12, P:741-743, P:767, P:771-772.
   - Quote: `test_one_path` "goes through core's compiled-lowering cache (counted)".
   - Problem: D B12 says "no cached one-op plans" and P:741-743 says "Core keeps no kernel cache". The test also lists `pass!` and `trace_rays!`, which P deletes.
   - Fix: count `compilekernel` calls per device instead.

10. **inconsistency**, P:3-5 vs D B7 and E:3.
    - P says the measured split is "uncommitted … at 7e64cd5". D B7 says it was "committed in 692a4e1". E refers to 58a05a9.
    - Fix: cite 692a4e1.

11. **inconsistency**: several functions have different arities in different places.
    - `Builder`: `Builder(dev, ch)` at I:329 and I:592; `Builder(pl.device, Recording, seg.channel)` at P:622; `Builder(device, kind, channel)` at P:633.
    - `emitdispatch!`: four arguments at I:349; three at I:331, I:600 and I:634.
    - `Kernel`: `Kernel(native, info)` at I:285; `Kernel(dev, native)` at I:590. The eager path never calls `kernelinfo`, which breaks "one calculation" of launch geometry (P:422-423).
    - `run!`: `run!(pl::Plan)` at P:663; `run!(g::Graph)` at I:358 and D B10.
    - Fix: one signature each.

12. **inconsistency**, P:486 vs I:210 and I:224, P:1346, D B6.
    - Rewriter contract: P:486 has `rewrite(r, nodes, dag, selections) -> nodes`; I has a callable on the CompileState.
    - `MergeRegions(kind::Symbol)` (I:223) differs from D B6's `MergeRegions(attribute)`.
    - Fix: a callable on the CompileState, with the attribute as a type.

13. **inconsistency**: names differ or are undefined.
    - `sweep!(pool)` (I:546) vs `sweep!(ch)` (P:612).
    - `trimblocks!` (I:499) vs `trim!` (P:882).
    - `rebuild!(t::HWTLAS)` (I:434) builds eagerly, but P:800-802 says builds are declared `Mantle.build!` ops and P:813 says "Rebuilds are per-run inputs".
    - `RegionNode` (I:229), `HostPart` (I:623) and `ForeignCall` (I:151) are not in D B5's node list.
    - `alias!` is listed at I:164, P:477 and P:1346 and defined nowhere.
    - `tenant!` (P:569, P:858) is missing from I, but `ensuremapped!` reads `arena.tenants[plan]` (I:525), which raises a KeyError for a plan that was never made a tenant.

14. **inconsistency**, I:41, 44, 47 vs I:50-51 and I:237.
    - KernelNode, HostNode and LoopNode are immutable `struct`s, yet they "carry a `compiled`" field set by compile, and I:237 assigns `p.compiled.queue`.
    - Fix: `mutable struct` with `const` fields, like CallNode.

15. **inconsistency**, I:20 vs I:392 and I:578.
    - Quote (I:20): "once set, the graph is fixed". A recompile replaces the plan (I:392), and `free!(g)` sets `g.plan = nothing` (I:578), which makes `g` declarable again (I:105).

16. **inconsistency**, I:12, I:15, D B4.
    - `Graph <: Device <: KA.Backend`, so every method on `::Device` also accepts a Graph: eager `dispatch!`, `render!(dev,…)`, `KA.allocate`, `scale!(g)(…)`.
    - Any Graph method that is missing silently runs the eager path, with `pool(g)` and `queue(g, :main)`.
    - This needs a decision from you: either list the Graph methods for every eager entry point, or introduce a separate location type.

## 2. Rule violations

Checked with grep: no `_`-prefixed function names (the `_` at I:151 is an argument placeholder), no try/catch, no `hasproperty`, no non-const globals, and no batching outside graphs.

17. **rule-violation (D A2)**: `isa` branches.
    - I:90 `location(a) isa Graph`; I:92 `a isa MantleArray`; I:191 `p isa CallNode || continue`; I:215 `q isa CallNode && q.f === broadcast!`.
    - Fix: `graphof(::MantleArray{T,N,<:Graph})`, and one method per node type, e.g. `select!(c, p::CallNode)` plus `select!(c, ::Node) = nothing`.

18. **rule-violation (P:37-39, rule 3)**: Symbol kinds.
    - I:241-244 and I:591 use `:transfer`, `:compute` and `:main`. I:223 and I:226 use `regionkind(...) === r.kind`.
    - Only the queue lines carry "(kinds as types in real code)".
    - Fix: types, e.g. `Transfer()`, `Compute()`, `Main()`.

19. **rule-violation (D A3, P:43-46)**, I:675.
    - Quote: `markforeign!(accessrecord(a), CUDA.stream())`.
    - Problem: CUDA.jl's stream is task-local. The stream the vendor code later uses is picked by whichever task calls that code, not by the task that made the view.
    - The mark also orders only Mantle's *next* use. Vendor work enqueued on the still-alive view afterwards is unordered.
    - Fix: pass the stream explicitly, `CuArray(a, stream)`, or scope the view: `withview(a, stream) do ca … end`, recording the event at block end.

20. **rule-violation (D A4)**, I:526.
    - Quote: `need <= arena.mapped && return`. This compares values at every `run!` to discover an unmap made by pressure.
    - Fix: `unmapidle!` and `untenant!` push a remap mark into the affected plans.

21. **rule-violation / global state**, I:661-666, D B2, P:888-890.
    - If `devices()` builds new objects per call, E:12, E:16, E:283 and E:326 each get a separate device and pool for one GPU.
    - If it returns cached objects, there is a second process-level mutable (the opened devices plus the VkInstance needed to enumerate them) that I:661 says does not exist.
    - E:22 (`using LavaPipe_jll` after E:12 and E:15 have created an instance) only works if `devices()` creates a new VkInstance, because the loader reads ICDs at instance creation.
    - "GPU 0" can be lavapipe.
    - Needs your decision: which global holds the instance and the devices, and whether `Device()` skips CPU drivers.

22. **rule-violation / bug**, I:374-378 and I:595-599.
    - Records are locked and unlocked by hand, with no try/finally. A throw in between leaves the records locked forever.
    - try/finally is not the forbidden catch.

## 3. Logic of the internals

### run!, threads, ordering

23. **bug-in-design**, I:388-396.
    - `currentplan!` is unsynchronised. Two first `run!(g)` calls on two threads both compile and record.
    - One `g.plan` is overwritten and never retired or removed from its arena (holds and tenancy leak). The other thread runs an orphaned plan.
    - Fix: a per-graph lock around `currentplan!`.

24. **bug-in-design**, I:360 and I:380.
    - `entry.token = token` is written after `submit!`. Until then the entry still shows its old, passed token, so a concurrent `run!` of the same graph can take the same entry.
    - Fix: `takeentry!` marks the entry as taken under the plan's lock.

25. **bug-in-design (use after free)**, I:363-366 vs I:373-374 and I:403-412.
    - GPURef values and updates are taken before the records are locked. Interleaving:
      1. `run!` takes an empty update list.
      2. `resize!` swaps `b.region`, pushes the patch and retires the old region.
      3. `run!` then locks the *new* region's record and records its access there.
      4. `run!` submits with the *old* address.
    - The old region's record never sees this run, so `sweep!` releases the region while the run reads it.
    - `pushupdate!` and `takeupdates!` also share one list across threads with no lock.
    - Fix: `resize!` swaps the region and pushes the patch under the record lock, and `run!` takes updates after locking. D C2 removes the patch list entirely (see 88).

26. **bug-in-design**, I:375 vs I:461 and I:240-244.
    - Quote: `token = reserve!(channel(pl))` gives one token from one channel, but I:461 records `Access(channel(pl, use), token, …)` for uses on other queues.
    - `passed(ch2, t1)` then compares numbers from two different timelines: waits end too early (a race and an early release) or never end.
    - Fix: one token per channel per segment, and each access recorded with the token of the segment that makes it.

27. **bug-in-design (GPU deadlock)**, follows from 26.
    - Graph A: segment A1 on Q1 writes `r`; segment A2 runs on Q2. Graph B, on Q2, reads `r` and waits on A's end-of-run token.
    - Token order is enforced per channel only (I:379). If B reaches Q2 before A2, Q2 blocks on B and A2 is queued behind it forever.
    - Fix: the per-segment tokens of 26.

28. **bug-in-design**, I:375-379.
    - Quote: "submissions reach the queue in token order". Any throw between `reserve!` and `submit!` leaves that token unsubmitted.
    - Every later submission on the channel then waits forever, and every record holding the token never passes (a permanent leak).
    - The wait for the predecessor token must not be taken inside the channel lock.
    - Fix: make this window non-throwing, or on failure submit an empty batch that signals the token (try/finally).

29. **bug-in-design**, I:445-449 and I:459-466.
    - The write-after-read decision uses `writes(use)` of the *first* use only, and only the *last* use is recorded.
    - A plan that reads `r` and later writes it skips the wait on other graphs' in-flight reads (a write-after-read hazard).
    - A plan that writes `r` and reads it last is recorded as a reader, so the next reader of `r` waits only for an older write (a read-after-write hazard).
    - Fix: `summary!` stores {reads, writes, first stage, last stage} per shared resource.

30. **bug-in-design (GPU race)**, P:687-690 vs I:309-313.
    - Quote (P): "Consecutive runs of one plan are serialised anyway … ordered by the access records". The argument block is not an access-recorded resource.
    - E:306-312 (`gw`) only renders into the acquired swapchain image and reads `verts`/`cam`, so run N+1's head kernel rewrites `cam` in the block while run N's draws still read it.
    - Fix: the block is a resource in the plan's own summary, or every run waits on the plan's previous token.

31. **bug-in-design**, I:363 vs I:474.
    - `run!` reads `r.value[]` while another thread may be writing it. A Mat4f is 64 bytes and is not written atomically.
    - P:697-699 rules out seqlocks, so the rule must be stated instead: set a GPURef and run on the same task, or accept torn values.

32. **bug-in-design**, I:572-579 and E:322-323.
    - Quote (E:322): "run and free a graph from any thread". `free!(g)` retires the plan while another thread is between `currentplan!` and `submit!`; with no token (see 35) the recordings are released before that run is submitted.
    - `free!(g)` twice drops every hold twice: I:574 runs before any guard.
    - A freed graph becomes declarable and runnable again, with its holds already dropped.
    - Fix: the graph lock, and a dead mark that makes later `run!`/declare throw.

33. **bug-in-design**, I:423-429 vs P:795.
    - The event thread recreates the swapchain with no lock against a `run!` on another thread that has acquired an image and not yet presented.
    - Fix: a window lock covering acquire→present and recreation.

34. **missing-case**, I:358-382 vs P:781-783.
    - `run!` has no acquire step, no per-image recording selection, no acquire-semaphore wait and no present list. `submitnative!` takes `presents` (I:649), but `run!` passes none.

35. **bug-in-design (use after free)**, I:544 and I:551.
    - Quote: `retire!(pool, x) = push!(pool.inbox, x)`. Release waits for `allpassed(accessrecord(x))`.
    - Objects without their own record are released at the next sweep: the eager argument region `packed` (I:593 and I:602; I:598 records the arguments, not `packed`), and plans, recordings and the ring (I:391 and I:577; `accessrecord(::Plan)` is undefined).
    - Fix: `retire!(pool, x, lastaccesses)`, as P:837-838 states ("retired with the channel token of their last use").

36. **bug-in-design (lock order)**.
    - `run!` takes the records in id order, then the channel. `crossgraphwaits` and `recordaccess!` lock records again per resource (I:447, I:462). That is safe only if `accessrecords(pl)` equals the union of the first-use and last-use records; otherwise records are taken lazily, out of order.
    - `unmapidle!` (I:536) takes the channel lock and reads `lasttokens` (channel before records, the reverse order).
    - Pool, arena and inbox have no locks at all: `sweep!` mutates `pool.retiring` from any thread (I:548-553).
    - Fix: state one global order: graph → records by id → channel. Pool and arena locks are leaf locks, never held across a wait.

### Release, holds, leaks

37. **bug-in-design (leak)**, I:559 and I:520-521, E:241.
    - There is no graph finalizer, so a graph dropped without `free!` never drops its holds. Weights it named stay at holds ≥ 1 forever.
    - `Arena.tenants` and `lastrun` are `Dict{Plan,…}` with strong keys, which keep plans and their resources reachable.
    - E:241 "or drop `big` and let the GC collect it" is false as written.
    - Fix: WeakKeyDict (or ids); the graph finalizer pushes onto the inbox, and `sweep!` drops the holds and removes the plan from its arena.

38. **bug-in-design (leak)**, I:391.
    - A recompile retires the old plan without `untenant!`, so the arena keeps the old peak (for example after a window shrinks).

39. **bug-in-design**, I:570 and I:563.
    - Nothing stops use of `b` after `free!(b)`. A later `name!` raises holds from 0 back to 1 on a retired region.
    - The finalizer at I:559 drops the creator's hold again after `free!` already did.
    - Fix: the dead mark throws on use, `addhold!` refuses 0→1, and the finalizer checks `markfreed!`.

40. **bug-in-design**, I:568 and I:117.
    - `name!` walks only the top-level arguments. Keyword arrays (`mask`), arrays inside nested `Broadcasted`, and view parents get no hold.

41. **inconsistency**, I:400 vs I:409-411.
    - Quote (I:400): "a graph registers when it names it". But `users` holds plans, and no plan exists at declaration.
    - If registration happens at compile after `writearguments`, a `resize!` in between pushes to nobody, and the new plan keeps the old address.

42. **bug-in-design**, I:365-366, I:341-344, P:685-686.
    - Updates pushed to a plan that is not run pile up without bound.
    - When it finally runs, they go into an entry "sized at compile", so overflow is undefined.
    - Two resizes push the same offset twice; parallel head-kernel threads then write both in no defined order.
    - Fix: one pending value per offset, last write wins. D C2 makes this moot.

### Pressure policy and arenas

43. **bug-in-design (spurious OOM)**, I:492-505.
    - `acquire!` never waits for work already submitted. Memory that frees once a token passes becomes `OutOfDeviceMemory`, for example with per-frame reallocation near the budget.
    - "Never flush" does not require "never wait".
    - Fix: after the other steps, wait for the oldest retiring token, sweep, retry, then throw.

44. **bug-in-design**, I:499-503.
    - After `collectgarbage!`, the loop retries without calling `sweep!` again, so regions the finalizers just pushed are not released.
    - Julia 1.12.7 turns off finalizers on a thread while it holds any ReentrantLock (`base/lock.jl:174`, `GC.disable_finalizers()` in `_trylock`). An `acquire!` reached under the graph, record or channel lock runs no finalizers.
    - `@ccall` defaults to `gc_safe = false` (`base/c.jl:278`), so a thread blocked in a driver wait stalls `GC.gc()` in another thread's `acquire!`.

45. **bug-in-design**, I:499 vs E:231-232 and P:22-24.
    - `sweep!` runs only as the first pressure step. In the normal path, retired memory is never released until the budget is hit.
    - E:232 "R if the GPU has finished 7" is false.
    - Fix: `sweep!` at the start of `tryacquire!`.

46. **bug-in-design**, I:531-538 vs I:524-529.
    - `arena.mapped` drops immediately, but the unmap runs later. No arena lock and no writer of `lastrun` exist, so:
      - a `run!` can pass `ensuremapped!` and then submit onto pages being unmapped;
      - a later `ensuremapped!` remaps pages that the deferred unmap then removes;
      - the immediate retry gains nothing, because unbinding frees no pages.
    - Fix: an arena lock, map and unmap as operations recorded on the arena's access record, and physical page release by token.

47. **bug-in-design (unverified)**, I:527.
    - My recollection of the Vulkan spec: `vkQueueBindSparse` batches are not ordered against command-buffer submissions by submission order, so the bind would need to signal the timeline and the run wait on it. Check against the spec's sparse-binding chapter before relying on this.

48. **inconsistency**, D B13, P:847-851, E:219-221 vs I.
    - Quote (D B13): "an idle graph holds address space, not memory". In I, pages are unmapped only under pressure in this process; pressure from other processes is still open (P:878-879).

### resize!, update ring, head kernel

49. **bug-in-design**, I:403-412 and E:171-173.
    - Quote (E): "the next run of gc patches the new address". Sizes are not patched: the argument dims and the ndrange were written at record for 1000 elements, so `shade!` processes 1000 of 2000.
    - I:404 "sparse: same address, nothing to tell anyone" is false for the same reason.
    - E:309 `length(verts)` is fixed when the draw is declared.
    - Fix: the size is pushed too (D B14, D C2, D C4).

50. **bug-in-design**, E:257-258, I:404-407, P:869-870.
    - `(64,448,20,32)` → `(64,896,20,32)` grows a dimension that is not last. That changes every element's linear offset, so growing in place or a byte copy scrambles the cache.
    - I:406 also takes an Int, not a tuple.
    - Fix: grow in place only along the last dimension; otherwise use a strided copy kernel.

51. **bug-in-design**, I:330-331, I:384, I:305.
    - The head kernel is emitted only in the first segment, and inactive `when!` segments are skipped. If the plan starts with a `when!` region, runs with it off apply no GPURef values, and the drained patches are lost for good.
    - This is exactly D B1's SAM 2 case, `when!(g, newimage)` with the encoder first: a click without a new image decodes with the previous click's coordinates.
    - Fix: put the head kernel in an unconditional segment of its own.

52. **bug-in-design**, I:331 and I:642-646.
    - On CUDA/HIP the head kernel has no graph edges to the other nodes, so root nodes can start before it has written the argument block.
    - Segments on other queues do not wait for it.
    - On Vulkan its barrier must cover every consumer: compute, `INDIRECT_COMMAND_READ` (I:638), vertex/fragment and ray tracing.

53. **bug-in-design**, P:684-686 vs I:340-345.
    - Host-written transient data belongs in the arena, but `applyupdates!` writes only into the argument block, and copies one update per thread (MB-sized for images).
    - Fix: a copy command from the entry to the arena offset.

54. **bug-in-design**, I:483-487.
    - `st[:] = src` goes through I:479, which calls the same `lower` again with a new staging array: infinite recursion.
    - On a graph, `lower` runs once at compile, so the host data is copied only once, into a transient that is lost after the run.
    - Fix: the store into host-visible staging memory is a plain host memcpy.

55. **contradicts-decision (D B10)**, E:271-274.
    - `pts` is declared on `gc` after `gc` has run, and I:479 would declare a node, which `declarable` rejects (I:105).
    - No method in I stages a hostwritten transient into the next entry.

### Compile order

56. **bug-in-design**, I:164, I:166, I:240-244, I:200-206.
    - `schedule!` (step 2) needs `overlapgain`, which needs the per-node costs that `costs!` only produces in step 3.
    - `select!` (step 1) calls `estimate` on candidates, which needs the KernelInfo that `pipelines!` produces in step 3.

57. **bug-in-design (memory race)**, I:254-258 vs I:175-186 and I:261-272.
    - `place!` lets transients with non-overlapping lifetimes share bytes (E:196-202, p1 and p3). But dependencies and barriers are keyed by ResourceId, and `dependencies!` runs before `place!`.
    - Nothing orders p3's first write after p1's last read, and the CUDA/HIP predecessor lists never contain that edge.
    - `alias!` is undefined.
    - Fix: after `place!`, add edges and barriers for every pair of transients that share bytes, then reduce.

58. **bug-in-design**, I:161-162, I:189-195, I:215, E:97.
    - `FuseEpilogues` runs after `select!` has replaced every CallNode, yet it matches on `CallNode`, so it never fires. E:97's "one kernel with FuseEpilogues" is false.
    - It also removes from `c.nodes` while iterating over it (I:211, I:219).
    - Fix: lowered nodes keep their original op and elementwise function, and the rewriter matches on those.

59. **bug-in-design**, I:193 → I:108-111.
    - At compile, `lower(g, …)` goes through the user-facing `dispatch!`:
      - `declarable(g)` throws on every recompile, so a window resize always fails;
      - `push!(g.nodes, …)` appends lowered nodes to the declared graph on every compile.
    - Fix: the compile target is the CompileState (or a capture type) with its own `dispatch!` methods.

60. **bug-in-design**, I:190-193.
    - `select!` iterates only top-level nodes, so CallNodes inside loop bodies (E:122-123) are never lowered.
    - The `when!` condition of a replaced CallNode is not carried over to its lowering.

61. **bug-in-design**, I:123 and I:263-270.
    - With nested `when!`, the outer call overwrites the inner condition. `barriers!` keeps one saved state instead of a stack.

62. **bug-in-design**, I:151 vs I:95 and I:583.
    - Quote (I:151): `dispatch!(t, k::ForeignCall, args, _)`. For a call like (Graph, ForeignCall, Tuple, Any), neither this method nor `dispatch!(g::Graph, f, args...)` is more specific: an ambiguity error on every closed-library selection.
    - Fix: route through `opstyle(::ForeignCall) = Foreign()`.

63. **bug-in-design**, I:92 vs E:325-328.
    - `where` takes the first array's device and never checks that all arguments share it, so arrays from two devices are mixed silently.
    - Fix: throw on two devices, as it already does for two graphs.

64. **missing-case**, E:103-106, I:626, P:386-389.
    - The host node's arguments are device-local transients, but `runhost!` reads and writes host-visible memory. No compile step places those arguments in host-visible memory or adds copies around the node.
    - "Submits every segment up front" cannot hold for a host node inside a device-decided `repeat!`.

65. **inconsistency**, P:327-329 vs I:350-353, P:673-675, P:1316-1318.
    - P:327-329 says the loop body's recording is "submitted n times"; I records the loop inside one recording, and P says each segment is submitted once.

## 4. Examples

66. **inconsistency**, E:112-124.
    - `transformer(noise, h)` takes two arguments; `transformer(lat, h, i)` takes three.
    - The loop is declared after `vae(lat)`. The write-after-read dependency on `lat` orders `vae` first, so `img` decodes the initial latent.

67. **cosmetic**, E:120: `schedule(50)` calls `Base.schedule` (task scheduling), which fails.

68. **inconsistency**, E:128-130: `running` is never initialised, so the loop may run 0 iterations.

69. **inconsistency**, E:136-138: `nframes` is a transient and is gone after the run, so the host cannot know `audio`'s valid length. It must be persistent.

70. **contradicts-decision (D B1)**, E:10-23: Qwen, Whisper, Kokoro and MatAnyone are all declared into one `g`, so `run!(g)` at E:152 runs all of them.

71. **contradicts-decision (D B10)**, E:317: `blur` is declared after `gw` ran (E:311-312), and no node uses it.

72. **inconsistency**, E:308-312.
    - `length(verts)` is frozen at declaration.
    - "acquire, … present" has no counterpart in I's `run!` (see 34).
    - "one recording per swapchain image" is an exception to D B11 ("one recording per plan") and should be stated there.

73. **inconsistency**, E:204-209 vs E:211-213.
    - Example 32 has a DNN and a ray graph sharing one arena (900 MB). Example 33 says overlapping graphs get separate arenas (1200 MB). No function in I chooses between them.
    - I keeps one AccessRecord per arena, so "ray waits only on sam's work in the same bytes" really means ray waits on all of sam's arena work.

74. **inconsistency**, E:299-301.
    - `push!(tlas, newmesh)` changes the instance count, which needs a recompile or re-record, not only a patched handle.
    - I:434's eager `rebuild!` contradicts P:813 ("Rebuilds are per-run inputs").

75. **inconsistency**, E:166-167 "One submission": as written it is at least two and recurses (see 54).

76. **cosmetic**, E:196-202: no node uses p1, p2 or p3, so `peakbytes` is 0.

77. **inconsistency**, E:36 vs E:74: example 6 uses the constructed KA kernel, example 13 passes the constructor.

78. **inconsistency**, E:269 vs E:66-68 and D B4: `streamed = true` gives a device array graph semantics. Eager use is undefined (the keyword is open in D B15).

79. **inconsistency**, E:323: the example is sequential, so it does not show the concurrent use it claims (23, 24, 32).

80. **inconsistency**, E:248: `feat` can still be used after `free!` (see 39).

81. **cosmetic**, E:7, E:162, E:283.
    - NNlib, CUDA and ColorTypes are not imported.
    - `lookat` comes from Makie (`Makie/src/camera/projection_math.jl:132`), not GeometryBasics.
    - The device filter uses `isa`; a `devices(CudaDevice)` would avoid it.

82. **missing-case**: use cases from the plan with no example.
    - Simulation: time stepping with `repeat!`, and a device-decided convergence loop.
    - Video decode queue feeding a graph.
    - Editor frame beside a background DNN, with queue priorities.
    - D B1's SAM 2 `when!(g, newimage)`.
    - RayMakie: textures in a graph, offscreen render plus readback, overlay compositing (P:1492).
    - Hikari: a ray-query dispatch with `TraceRead`, `postprocess!` as a node.
    - Two windows; `readback_window`.
    - Two threads running two graphs that share a buffer; `resize!` concurrent with `run!`.
    - LavaDevice and CudaDevice on one GPU (P:130-131).
    - A closed library as a foreign node; `MtlArray`/`ROCArray` views.
    - Eager `render!(dev,…)` and `trace!(dev,…)` (P:752).
    - A host node writing a `repeat!` flag (P:1026-1028).
    - `Array(a)` of a persistent graph output after a run.

## 5. What section C changes

83. **outdated-by-experiments**, I:638 (`vkCmdDispatchIndirect` for every dispatch) and P:634.
    - D C4: fixed-size launches are direct. Indirect launches cost +190–245 ns on AMD and +45–60 ns on NVIDIA.
    - Keep indirect (or a grid-stride loop on AMD) only for sizes known only on the GPU (E:138).
    - CompiledDispatch must carry the direct/indirect choice as core's decision.

84. **outdated-by-experiments**, I:327, I:637, I:593, P:712-714 vs D C3 and D C6.
    - Arguments move to a layout where fields sit directly in the slots, with the argument memory marked read-only (NonWritable).
    - On CUDA, arguments passed by value cannot be written by a head kernel. Patches there become node-parameter updates (my inference, not measured), which resolves 7.

85. **outdated-by-experiments**, I:398-419, I:365-366, I:166 (`patches!`), I:321-324; P:495 and P:708-712; comments at E:171, E:255 and E:299.
    - D C2 replaces the pushed patches with a header per array (address, dims), which `resize!` writes eagerly, and a per-run head-kernel copy from a table built at compile.
    - This removes findings 25, 41 and 42 and fixes 49's dims.
    - "(no updates, no work)" at I:339 and P:691 becomes false: the copy costs 1.3–1.9 µs per run on AMD and 2.0–3.5 µs on NVIDIA (D C2).
    - The ring stays for GPURef values and hostwritten data.
    - New requirement: the eager header write must be ordered after in-flight head kernels that read the old header. Otherwise a submitted run that has not started yet reads the new address before `copyregion!` has filled it.

86. **outdated-by-experiments**, I:388-396, I:423-429, P:1316-1318, E:314-316.
    - D C4: when the host knows the size, re-record the launches (0.97 ms for 300 dispatches); re-placing transients is a recompile and was not measured.
    - `resize!` must push "re-record" to its user plans, so the `users` list that D C2 otherwise removes comes back for this.
    - A window resize becomes a re-record when placement is unchanged. `test_run_is_submit` must list this as an exception.

87. **contradicts-decision / needs your decision**: D B14 vs I:37-40, I:97-102, I:146-149, I:189-196, E:138.
    - D B14: kernel nodes are added at declaration, sizes live on the GPU, `DeviceRange(a)`, and `resize!` writes the size.
    - The docs instead defer lowering to compile via CallNodes and use `DeviceRange(nframes; max)`.
    - D C4 partly reverses D B14 for host-known sizes. Someone has to rule which scheme applies to fixed, host-resized and GPU-sized arrays.

88. **outdated-by-experiments (minor)**, D C5: plans are recorded twice today. I's estimated split removes the cause; add "one recording per plan structure" to `test_record_is_pure`.

## Names used in one file and not defined or described in the others

- **In E, not in I** (P describes them unless noted): the `GPURef(dev, v)` constructor (location not in P), `HostCall`, `repeat!` with `while_nonzero`, `render!`, `draw!`, `trace!`, `DeviceRange`, `Window(dev, w, h)`, `MantleArray(g, T, win)`, the KA launch, `Mantle.refit!`, `HWTLAS` (open), `submissions(g)`, and `resize!` with a tuple.
  - The introspection names `reserved`, `mapped`, `naivebytes`, `peakbytes` and `budget` are open.
  - Not in P either: `push!(::HWTLAS)` and `Texture2D` (P only says "Image / texture").
- **In I, not in P, E or D:**
  - Node types: `RegionNode`, `HostPart`, `ForeignCall`/`foreignnode!`.
  - Update ring: `Entry`, `currentplan!`, `takeentry!`, `takeupdates!`, `pushupdate!`, `readdress!`, `addressoffsets`, `users`, the `recompile` flag.
  - Acceleration structures: `rebuild!`.
  - Arenas and pool: `ensuremapped!`, `unmapidle!`, `recentlyrun`, `lasttokens`, `lastrun`, `trimblocks!`, `collectgarbage!`, `tryacquire!`, `suballocate!`.
  - Holds: `addhold!`, `drophold!`, `markfreed!`, `name!`, `firstnamedby!`, `named`.
  - Declaration and conditions: `where`, `location`, `nodata`, `declarable`, `setcondition!`, `entering`, `leaving`, `mergestates`.
  - Selection and queues: `SelectionKey`, `selectiontable`, `candidates`, `supports`, `estimate` (two arities), `queuefor`, `overlapgain`, `crossqueuecost`.
  - Head kernel and eager path: `headkernel`, `headbarrier`, `packarguments!`, `waitsfor`, `recordaccesses!`.
  - Vendor views: `markforeign!`, `keepalive!`.
  - Devices and segments: `DEVICE_TYPES`, `enumeratedevices`, `defaultdevice`, `VirtualRange`, `pageceil`, `Segment`, `mustcut`, `gate`.
- **Listed but defined nowhere:** `alias!`.
- **In P or D, in neither file:**
  - Verbs and names: `tenant!`, `devicearg`, `fence`, `recycle!`, `rawfree`, `placeimage`, `placeaccel`, `TraceRead`/`TraceBuild`, `Mantle.build!`, `FuseBroadcasts`, `MtlArray(a)`/`ROCArray(a)`, `timings(g)`, `Graph(dev; policy)`, `readback_window`.
  - The `emit*` verbs other than `emitdispatch!`/`emitloop!`.
  - `compile_draw`, `compile_trace`, `accelsizes`, `compile_bindings`.

## Five most important fixes

1. **Tokens per channel and per segment, with a submit window that cannot throw** (26–29). Record each access with the token of the segment that makes it. Record reads and writes over the whole plan, not just the first and last use. Release the locks with try/finally. As written, a multi-queue plan mixes timelines, can deadlock the GPU across two graphs, and one exception wedges a channel.
2. **Release engine: `retire!` carries the tokens of last use** (35); holds and tenancy are released through a graph finalizer and weak tenancy (37, 38); a per-graph lock covers `currentplan!`, `takeentry!` and `free!` (23, 24, 32). Today the argument regions of eager calls and retired plans are freed while the GPU still uses them, and graphs dropped without `free!` leak shared weights.
3. **Head kernel and argument block** (30, 51, 52, 53):
   - an unconditional first step that is a predecessor of every root node;
   - the argument block recorded as a resource the plan writes;
   - hostwritten data copied into the arena.
   Adopt D C2's header copy so that `resize!` no longer pushes per-plan patches (removes 25, 41, 42).
4. **`resize!` must carry sizes, not only addresses** (49, 50, 86, 87). You need to decide D B14 vs D C4 per array class; multi-dimensional growth must be limited to the last dimension.
5. **Compile pipeline correctness** (56–60):
   - `lower` at compile targets the CompileState, not the declared graph (today every recompile throws);
   - transients that share bytes get ordering edges and barriers after `place!`;
   - fix the step order (queue choice needs costs; selection needs KernelInfo);
   - `FuseEpilogues` matches on the nodes that actually exist after `select!`.