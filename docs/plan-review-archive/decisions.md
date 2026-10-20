# Decisions and rules for the Mantle redesign (reference for reviewing the plan)

Collected from the design conversation of 2026-09-26..30 between Simon (the owner)
and the assistant. "Simon:" marks his own words or rulings; everything else was
agreed in the conversation or measured. The plan documents under review are
/sim/Programmieren/VideoEdit/dev/Mantle/docs/{recording-plan.md, internals.jl,
examples.jl, recording-plan-review.md}; they were last edited 2026-09-28 16:26,
BEFORE the experiment results in section C.

## A. Standing rules (apply to every line of the plan)

1. Core decides everything backend-independently. A backend (device type) only
   implements verbs; it never "gets a graph to execute" and never makes a
   decision or implements an algorithm core could own. Zero duplication across
   backends and across packages.
2. Multiple dispatch, never `isa`/type-switch branches. No `_`-prefixed function
   names. No globals unless unavoidable (then ALL_CAPS const); a const used in
   one place becomes a keyword argument. No blanket try/catch. No defensive
   `hasproperty` on values whose type is known.
3. No implicit or ambient state (Simon: "I dont want anything that uses global
   implicit state"; an "open eager command buffer" was rejected for this). A call
   reaches a graph only through its arguments; a "current graph" that assigns
   calls implicitly is "ein RIESEN NOGO" and "ein super mieses anti pattern"
   (2026-10-19).
4. Changes are PUSHED by the function that makes them. Never detected: no version
   counters, no value comparison at run!, no walk-and-diff/reconcile passes
   (Simon: "ITS DETECTED WHEN THE FUNCTION THAT CHANGES SOMETHING IS CALLED").
5. Plain technical language; no metaphors; no time estimates; every claim sourced.

## B. Design decisions

1. One call, one graph. A model call is ONE Mantle graph; consumers never
   partition it; core schedules, allocates and splits. Whether an application
   uses one graph or two is decided by its calls (SAM 2: one graph with
   `when!(g, newimage)`, or two graphs if the encoder runs on its own).
2. Devices, not backends. Device types: LavaDevice (default `Device()` = GPU 0 on
   Linux/Windows), MetalDevice (macOS), CudaDevice/ROCmDevice only after
   `using CUDA`/`using AMDGPU` (package extensions). lavapipe via
   `using LavaPipe_jll` or system Mesa. HostDevice is deleted. `devices()` lists
   devices (opened lazily). The device IS the KA backend (`Device <: KA.Backend`).
   Backend/API tags, `syncbackend`, `LavaBackend`, `backend(dev)`, `kibackend`,
   `defaultbackend` are deleted. Loading stays `@static include` per platform.
   Behaviour by dispatch on device type; one load-time device-type registry
   (open decision).
3. One array type `MantleArray{T,N,D}`. `MantleArray(dev, …)` runs now (eager);
   `MantleArray(g, …)` declares on a graph (transient by default;
   `persistent = true`; `hostwritten = true`; window-sized form). Eager
   temporaries `MantleArray(dev, …) do a … end`. Replaces LavaArray, DeviceArray,
   Buffer, Transient/TransientBuffer. getindex / KA.get_backend on a graph array
   throw (no tracing). Zero-copy vendor views `CuArray(a)`, `MtlArray(a)`,
   `ROCArray(a)` at the boundary, with handover ordering. ("DeviceArray" is also
   a bad name because in GPUArrays/CUDA/AMDGPU it means the in-kernel array.)
4. A Graph is a device. Dispatch on where arguments live: arrays on a graph
   declare, on a device run now. `fill!(x, v)` etc. are overloaded for
   MantleArray. `dispatch!(g, kernel, args, ndrange)` for raw kernels;
   `dispatch!(g, HostCall(f), …; writes = …)` for host functions; closed
   libraries are foreign nodes; `render!`, `trace!`, `repeat!` (one body),
   `when!`.
5. Nodes, not passes: CallNode, KernelNode, RenderNode, TraceNode, CopyNode,
   HostNode, LoopNode. Compiled data lives in the node (`CompiledNode`); no
   tables keyed by node.
6. Compile = rewrite / analyse / build steps as functions on a CompileState.
   Rewriters are opt-in (`Graph(dev; rewrites = (…))`), none by default
   (RayMakie must not pay for broadcast fusion). No backend-specific passes:
   backends declare attributes, core has generic rewriters such as
   `MergeRegions(attribute)`.
7. Submission split is ESTIMATED at compile (roofline from KernelInfo, declared
   bytes) against a device budget. The measure-then-re-record path committed in
   692a4e1 is to be replaced.
8. Kernel compilation: `compilekernel(dev, …)` answers from the device type's
   compiler cache; core keeps no kernel cache. `kernelinfo(dev, native)`.
9. Queues: core chooses how many queues from the graph ("one queue" was never
   the rule). Cross-graph and cross-queue ordering by per-resource access records
   turned into token waits; accesses recorded before submit under locks.
10. `run!(g)`: the graph manages its plan; no user-facing Plan/record!. A graph is
    fixed after its first run. Changes are pushed by the function that makes them
    (`resize!(buf)`, `resize!(window)`, TLAS rebuild).
11. Static argument block, one recording per plan, and a recorded head kernel
    that applies this run's updates at the start of every run (Simon's own design
    from 2026-09-03: "a patch_arguments! kernel run on a fixed gpuvector of
    offset, ptr, newptr, which just always runs first (recorded)"). GPURef:
    `x[] = v; run!(g)`. Stores into arrays that live on a device are eager
    copies through Mantle's own verbs. No slots, no OneShot, no hold!.
12. Eager path: `dispatch!(dev, f, …)` → kernel from the device's compiler cache
    → one command buffer → submit. No batching outside graphs, no cached one-op
    plans.
13. Memory: Mantle bypasses backend allocators. One pool per device with kinds
    (Buffers, Unified, Readback, Images, Accel); backends only implement
    rawalloc / rawfree / mapsparse! (renamed bindpages!/unbindpages! 2026-10-02) / placeimage / placeaccel. One release engine
    by token. Sparse arenas grow and shrink; an idle graph holds address space,
    not memory. Shared resources are counted (holds: creator + each graph;
    `free!(buf)`, `free!(g)`). Pressure policy only in `acquire!`.
14. Latest direction before the experiments (not yet in the docs): `sum!(x)` on a
    graph adds the implementation's kernel nodes directly; sizes live on the GPU
    with a capacity for placement; launches over `DeviceRange(a)`; `resize!` is
    an eager write of the size.
15. Open decisions in the plan: AK kernels first for reduce/scan/sort;
    device-type registry; FuseBroadcasts; rewriter/attribute names; streamed
    weights keyword; cross-device copyto!; OS memory-pressure events;
    descriptor-bound handles on Vulkan.

## B2. Rulings of 2026-09-30 (Simon, after the review)

1. Runs of one graph never overlap ("I dont think we should try to overlap
   runs"). One argument block per graph and ONE per-run entry: a host write into
   it (GPURef value, size) waits for the previous run of that graph to finish.
   No double buffering for now ("I dont think it would even give us that much";
   changes like resize! are queued anyway).
2. `resize!` (and every change to a resource) is QUEUED, always; never a host
   side swap or an eager patch.
3. Sizes: to be measured per strategy on a whole recorded plan (done:
   plan_bench.jl; sizes on the device cost ~1% GPU per run on a SAM 2 decoder
   shaped plan and ~0 host; re-record costs 50-400 µs host per change).
4. Argument updates: the array owns a shared cell (address, size) that every
   graph uses ("have the array own a GPURef or something, which is simply shared").
5. Device identity: the same Julia object per hardware device.
6. Graph vs device: a `Graph` plus a `GraphDevice` that graph arrays point at
   (not Graph <: Device).
7. Swapchain: the most efficient option (proposed: record only the segment that
   writes the swapchain image once per swapchain image).
8. `when!`/`repeat!`: kernel launches only; builds etc. are explicit
   constructions. Possibly `when!` on single kernels only.
   HostCall STAYS: "the user should not manage graphs, so if they need host
   calls inside a graph, Mantle should make sure to schedule it correctly".
   Mantle cuts the recording around host nodes and orders the submissions; the
   user never splits a graph. (Earlier: the cost of host work should be clear
   to the user.)
9. Compile is `compile!(state, node)` per node type; no lowering at compile
   (operations add their kernel nodes at the call).
10. `run!(g)` dispatches to `run!(g.plan)`; Plan stays internal.
11. `MantleArray{T,N,D} <: AbstractGPUArray{T,N}` for every location.
12. Commits 8d95632 and 58a05a9 going against the plan: "doesn't matter".

## B3. Rulings of 2026-10-02 (Simon)

1. Conditional launches are a block, `when!(g, flag) do … end` (plan open
   decision 1, option b): every launch declared inside runs only where the
   flag is nonzero on the GPU. Launches only, like `repeat!`. Not a keyword
   per launch.
2. TLAS: no inactive padding ("Padding hört sich schlecht an, vor allem weil
   viele instance counts sich selten ändern"). Builds and refits carry the
   exact count; storage is sized for a doubling capacity; on Vulkan (every
   driver) a plan's build/refit is its own small command buffer, re-recorded
   by the run when the count or storage changed; Metal reads the count from a
   cell. Vulkan's indirect build is not used: only AMD's Windows driver has
   it, and there it is slower than the exact count (m1_lapwin.txt).
3. Host nodes (D2): one submission per stage. No memory of their own: "ich
   will so einfaches und effizientes speicher management wie moeglich, und
   lieber auf host writes warten, also die gesamte implemenation zu
   verkomplizieren". Arena
   parts stay shared; a run holds its parts until its last stage is
   submitted, so another graph's run on the same part waits on the host for
   the whole run. A host function running a graph that shares a part with
   its own throws (it would wait for itself).
4. D5 (graph lock `Base.Semaphore(1)`, GC-safe host waits) was presented as
   forced by Julia (`ReentrantLock` disables finalizers, base/lock.jl:28;
   `@ccall gc_safe = true` checked on 1.12.7) and not objected to.
5. D3 is not a decision: it follows from B2 (a host read in a graph is a copy
   node; MantleArray for every location).
6. Requirements for graphics and memory (Simon, 2026-10-02, in his words):
   RayMakie in raster and ray-trace mode "should NOT recompile plans often -
   only if new plots are added or removed, anything else should
   grow/patch/update inplace without recompile"; it "should neither rebuild
   the TLAS often, nor re-allocate buffers unecessarily"; "the inplace
   update!(plot; ...) HAS to be 100% efficient"; "the renderloop walking all
   render objects has to be efficient". Memory: no use after free, no leaks,
   no weird corner cases, no waste; "especially true for complicated stuff
   like the TLAS constructor - we need to make sure that it's easy to write
   correct code for it". Uniforms and buffers with "the same type for compile
   time optimizations" (bench/three_updates.jl, bench/two_scatters.jl). Two
   DNNs that each use 90% of VRAM must both run. A simulation's arrays drawn
   with mesh! update the render efficiently on each run!(sim).
   Accepted as proposed ("ja, hört sich gut an").

7. From the draw benchmarks (Simon, 2026-10-02: "Ich nehme jetzt erstmal
   deine Entscheidungen von den Benchmarks"; experiments/draw_bench/, NVIDIA,
   AMD Windows, Apple M5):
   - Plot attributes are `GPURef`s of arrays: `GPURef(dev, a::MantleArray)`,
     `r[] = b` rebinds without copy or recompile (launches over them are
     indirect: free for draws with work, +14-34 ns per tiny draw on Vulkan,
     nothing on the M5). Not a new name: `GPURef` holds a value or an array.
   - `attribute(a, i)` = `a[min(i, n-1)]`: one argument type for a uniform, a
     value for all elements and one per element (within 1% of `a[i]`, of a
     fixed stride and of a true uniform on all three).
   - The scene rectangle is a per-draw uniform applied in the vertex stage with
     four clip distances (within 1%; supported on all three).
   Eviction (D9) and readers after runs (D10) stay proposals.
8. (2026-10-05) One graph presents to a window ("b"): the graph that declares
   `render!` into it; a second graph declaring into the same window throws at
   declaration; other graphs render into `Image`s the presenting graph reads.
   Reason: half of the window defects the review rounds found (concurrent
   acquires, images kept by other graphs, generations created under a
   running acquire, a two-image acquire waiting for another graph's image)
   exist only with several presenting graphs; RayMakie has one frame graph
   per screen anyway. Plan open decision 20.
9. (2026-10-05) Eviction is automatic ("16a, definitiv automatisch"). Input
   modes to support it: a read-only array created from a file
   (`MantleArray(path; mmap = true)`-like) can leave even RAM (the OS drops
   its clean file pages; restore reads the file); a read-only array created
   from host data leaves only VRAM. Naming (`immutable`?) and how read-only
   is enforced are open (it must not change).
10. (2026-10-05) Readers (a): a run that writes arrays pends the updates of
    the structures built from them; "so effizient und fehlerfrei wie
    moeglich".
11. (2026-10-05) Index buffers and handle dependences: (c), re-record the
    segments that recorded the handle; benchmarks first (re-record cost, and
    pulled indices against indexed draws).
12. (2026-10-05) Pipeline-picking value changes in RayMakie never compile: a
    GPU flag that works with indirect launches (`when!`: count times flag).
    Type changes are `delete!` + `insert!` in Makie anyway.
13. (2026-10-05) Changes are SCHEDULED ("I KEEP SAYING IT SHOULD GET
    SCHEDULED"): `resize!`, stores, rebinds, TLAS push!/delete! only enqueue
    the request at the call (the requested size is what `length(a)` reports;
    a store's data is taken). New storage, pages, capacity, staging, the copy,
    the cell write and the retirement are done by the submission that applies
    the change, which allocates before its locks (the same step as restoring
    an evicted array). The call never allocates, never waits, never retries.
    (The review-round docs had a `prepare!` step allocating at the call: the
    eager drift again.)
14. (2026-10-05, "joa b b") Ordering without a per-resource walk at run.
    Between graphs: neighbors at graph level (his proposal): plans that share
    a resource where one writes it wait for each other's complete last run
    (both directions), found at registration; finer segment edges only if a
    measurement asks for them. Host side: an inbox per plan, filled by
    whoever pends a change or records an access that is not a run; a run
    handles only its inbox. The per-resource structure is `SyncState` (was
    `AccessRecord`).
15. (2026-10-05, "b hört sich gut an... wir sollten auf wenig recompile und
    hohe runtime performance optimieren") Replaces the neighbor lists of 14.
    Asked first whether a recorded launch can wait on a GPU lock: Vulkan has
    no cross-queue wait inside a command buffer (events: same queue or host
    only, synchronization.adoc:5463-5465, 6176-6178; timeline semaphore
    values are fixed by the host at submit); CUDA/HIP have
    `cuStreamWaitValue64`/`hipStreamWaitValue64`. So the host picks the
    waits at submit. Option B: no plan keeps a list of other plans; each
    resource has `namers` (plans and their use, replaced at
    register!/unregister!); a stage writing a resource another plan names
    waits for that plan's complete last run and puts the resource into the
    readers' inboxes; a reader's stage waits for the writers' complete last
    runs. Composite members reach the plans naming the composite through
    their readers snapshot (push! changes no plan; the reader is dropped only
    after the removing submission passed). Nothing recompiles when plans
    register or members change (the old register! recompiled when a plan
    registered meanwhile; now it only locks again). Option A (lists linked at
    the call, `push!(F.inputs, S)`) was not taken: the call would need the
    plans' sync locks, which come before the SyncStates `change!` holds.
16. (2026-10-05, "nen plot hinzufuegen darf niemals den kompletten graph
    wegschmeissen, das waere doch insane"; "wird nicht einfach nur der baked
    plan invalidiert?") A graph is never replaced for a structure change.
    Declaring into a compiled graph, or removing a node block from it, only
    invalidates the plan; the next run recompiles from the graph's nodes.
    The graph, its windows, its named resources and the device arrays it
    reads stay. Replaces my rule "a compiled graph is fixed; a different
    structure is a new graph" (api.md 2.3, plan "The graph and its plan",
    internals.jl section 2), which was never his. Consequence for S32: the
    film is a device array the screen holds, so accumulation survives plot
    changes.
17. (2026-10-05, "ich wuerde gerne complexe GUIs optimieren, ohne state of
    the art scatter/mesh speeds fuer grosse meshes zu kanibalisieren. Wenn
    insert plot im milisecunden bereich ist, ists ok, erst recht wenn
    visibile=true/false ziemlich billig ist"; then "ok lass machen!" on C.)
    A render pass is a list of pieces; a plot is one piece, compiled and
    recorded on its own (Vulkan secondary command buffer, Metal ICB) with its
    own argument region and cell-copy entries. Insert/delete applied by the
    next run: one piece compiled and recorded, the list rewritten, no
    recompile, no other piece re-recorded. Each draw keeps its own pipeline
    and index buffer (no batching, so large meshes are untouched); visibility
    stays a `when!` flag store. Not taken: batching into shared arrays (B: a
    copy per attribute, device arrays copied per frame) or via plot records
    (B': an extra load per vertex, shared index arrays); both also split at
    every change of kind in draw order, so interleaved GUIs gain nothing.
18. (2026-10-05, "Ja ne lass das mal lieber super ineffizient machen *sigh*",
    i.e. yes to both.) A store copies its data once, at the call, into an
    upload region (host-visible system memory acquired for it, not the
    array's storage); the applying submission only copies from it (or writes
    it inline). My reading of 13 ("the call never allocates") had extended to
    the upload buffer and made every large store copy twice. Creating an
    array from data uploads at creation, streamed in chunks from the caller's
    memory or a mapped file: creation is not a change, no graph can name the
    array yet.

## B4. Rulings of 2026-10-14 (Simon)

1. A `when!` flag is a `Bool`: a `GPURef{Bool}` or a one-element `Bool` array,
   enforced by the type annotations of `when!`'s methods, so another type is a
   `MethodError` by dispatch, no runtime check ("By dispatch and....TYPE
   ANNOTATIONS"). Values other than 0 and 1 cannot exist; the indirect count is
   count times flag. `repeat!`'s `while_nonzero` stays an integer: it is a count
   (a queue's live size). Settles pending decision 2 of 2026-10-06.

2. Launch groups are values, not `do`-blocks: `group = Group(graph, "name")`,
   launches declared into it (`dispatch!(group, …)`), then attached with
   `repeat!(group, n; while_nonzero)` ("ok lets go with group = Group(graph,
   "name") for now"). No mode state on the graph (`g.inbody`/`g.inloop` go);
   what a group may hold (launches only, no `HostCall`) is enforced by which
   methods exist for `Group`. Reason (his): avoid host code being easily
   inserted; a `do`-block invites code between dispatches, a problem shared by
   any graph builder. `when!` takes single launches with a `Bool` flag (B4.1).
   Supersedes the block form of B3.1 and settles pending decision 3.

3. KernelAbstractions launches on graph arrays DECLARE kernel nodes instead of
   throwing: `KA.get_backend` of a graph array is the graph, `KA.allocate` gives a
   graph transient, so KA libraries and GPUArrays' generic code run inside a graph.
   Host reads of contents during declaration throw at their line; launch sizes the
   traced code computed on the host are recorded per node, and a resize of an array
   behind one makes the next compile throw. Replaces the plan's "on a graph device
   `KA.get_backend` throws". (Simon's proposal, 2026-10-14.)
4. `adapt(Mantle.Device(graph), x)` re-homes arrays (or a struct of them, e.g. a
   simulation's state) as zero-copy aliases the graph names (with a hold), so code
   that finds its target through its arrays declares into the graph;
   `adapt(Mantle.Device(), x)` uploads host data to a device. A group is NOT a
   location: allocations belong to a device or a graph ("why would we ever want to
   make a device / allocation local to a group instead of the graph?"). Launches go
   into a group only by passing it, `dispatch!(group, …)` (B4.2, "you can declare
   them anywhere, you just need to pass the group to dispatch"); library code that
   finds its target through its arrays therefore declares into the graph.
5. One array type, `MantleArray{T,N,L}` with `L` the location (device or graph, per
   API); a view is the same type with its own offset and dims, sharing the parent's
   resource (no role parameter); read-only, mmap-backed, host-written, Readback
   memory and resizable are runtime properties of the resource, not types ("Yes").

6. Read-only inputs (pending decision 5 of 2026-10-06):
   - Name: `readonly = true` (5.1).
   - Enforced on the host from the per-argument access `dispatch!` already derives
     at declaration (`argument_usage`, `graph/build.jl`): a write touch on a
     read-only array throws at the call, eager or declared. No compile involved;
     the write-up's premise that per-argument access was only a proposal was
     wrong (5.2).
   - `MantleArray(dev, host; readonly = true)` keeps a reference to the caller's
     host array, no copy; eviction drops the device copy and restore uploads from
     that array (5.3).
   - No file form and no `mmap` keyword: an mmapped array is a plain `Array`
     (`Mmap.mmap` returns `Vector{T}`, checked 2026-10-14), so the caller passes
     one and Mantle does not know or care where the bytes come from; the OS drops
     and re-reads file-backed pages by itself (5.4).
   - A host array or file that changes afterwards is undefined; no checks, no
     copies into RAM (5.5).
7. A kernel writing past an array's capacity follows Julia's convention: bounds
   are checked unless the kernel says `@inbounds`, where it is undefined. No
   clamping of a kernel-written length (pending decision 6.3).

8. Pending decisions of 2026-10-06 settled by veto check (Simon reviewed the list
   2026-10-14 and objected to none of these; 9 and 13 "yes", 12 "what else would
   we do then error?"):
   - 1: `persistent = true` on graph arrays is removed; what survives runs and
     recompiles is a device array the graph names (B3.16, B4.5).
   - 4.1: any node of the presenting graph may write the window (`render!`, a
     kernel, a copy); the first graph declaring a write presents, a second throws.
     4.2: eager `render!` into a window throws; into images it is allowed.
     4.3: the binding ends when the graph is freed or collected or its render node
     is removed. 4.4: `Array(win)` throws (minimized, no presenting graph, or from
     the presenting graph's own host function), and a waiting call is woken with
     the error. 4.5: `run!` returns `Submitted()` or `Skipped()`, both public.
   - 9: `resize!` of a texture keeps the texels where old and new sizes overlap.
   - 11: pieces are reordered with `Base.sort!(pass; by, lt, rev)`.
   - 12: a TLAS total past the device's `maxInstanceCount` throws at the call that
     raises it (`push!`, `resize!`, `copy!`, rebinding a source); capacity capped.
   - 13: Mantle loading without a display is a GLFW.jl bug, fixed by lazy
     initialization (PR open); documented until released.
9. The rest of the 2026-10-06 list (Simon, 2026-10-14):
   - 7: "old or new, never mixed" holds PER STAGE. A run with host nodes is one
     submission per stage (B3.3); a change arriving while the run is between
     stages goes into the next submission, so a later stage sees it ("per stage
     sounds good"). A host function's own declared writes reach the later stages.
   - 8: no gate counters as a mechanism. A plan carries `stale::Bool`, set by a
     mutation that needs a recompile and acted on at the next run, and a count of
     its recompiles for debugging and tests (his: "Maybe that should be a second
     field, recompile_count?"); a test asserts the count did not move. Proposed
     with it (mine, not ruled): a count of re-records too, since a piece insert or
     a moved index buffer re-records without recompiling.
   - 10: a value `GPURef` is ONE shared slot in device memory whose address every
     plan and piece bakes; `r[] = v` is a scheduled store applied in the queue like
     any change, and never waits. Host-written graph arrays become device arrays
     written the same way. Replaces B2.1's per-graph, per-run entry and its
     "a host write waits for the previous run". Reading a value from a
     one-element buffer costs the same as a true uniform (within 0.8% on all
     three GPUs, `experiments/draw_bench/`). ("Ok, guess we got that then".)

10. Arrays whose length a kernel writes (pending decision 6.1/6.2) follow Julia's
    array API, implemented correctly ("We should stick to the normal julia api's
    and just implement them correctly? Most of the time we dont need the size on
    the host"; confirmed "yes. sounds good!"):
    - `length(a)`/`size(a)` return the real length; for a kernel-written one that
      waits for the kernel that writes it and reads the length back. During graph
      declaration no run has produced it yet, so the call throws (as host reads do
      under B4.3).
    - Launches over the array use the length on the device, never the host.
    - `resize!` works as on a `Vector`, scheduled as always, also beyond the
      capacity; `capacity = n` only reserves memory up front, like `sizehint!`.
      Growing past it moves the storage, and plans whose temporaries were sized
      from it recompile once.

## B5. Rulings of 2026-10-19 (Simon), on the ten questions of the contradiction review

1. **Operations take the GRAPH, never a group.** "nimmt den GRAPH!! ... Gruppe ist
   sehr speziell fuer repeat etc und sollte nur fuer das was repeated wird creiert
   werden...Ansonsten sollte immer der graph rumgereicht werden... in deinem
   beispiel [ist] das gar nicht notwendig den graph weiter zu reichen, weil der ist
   ja idealerweise im device/array/etc". So `mul!(C, A, B)` on graph arrays declares
   into their graph (B4.3, B4.4). A group exists only for the launches that are
   repeated, and B4.2 already says how they get there ("Warum nicht unserer plan
   den wir schon besprochen haben"): `dispatch!(group, …)`; a call that does not
   take the group is not repeated: it declares into a graph only if its arguments
   reach one (graph arrays, the graph device), else it runs now on the device
   (A.3). So model code that
   is to be repeated passes the group to its `dispatch!` calls (examples.jl 19,
   `transformer(step, …)`, stays); a plain `mul!(C, A, B)` on graph arrays
   inside it lands in their graph and runs once per run. My `mul!(step, C, A, B)`
   goes.
2. **One semantics on every backend, without exception.** "es ist ein UNENDLICH
   GROSSES NOGO, wenn unsere api fuer verschiedene backends eine andere semantik
   hat". `when!` means the same on Vulkan, Metal, CUDA and HIP; how a device type
   implements it is its verb's business. A question is never framed as if
   semantics could differ per backend.
3. **A change a running plan cannot follow is applied after the run's last
   stage,** and its submitter waits. Ruled before ("hab ich circa schon 3 mal so
   abgenommen"; B3.3, B4.9.7).
4. **Eviction works the same on every backend,** unified-memory devices included:
   the M5, the Ryzen AI Max+ 395 and NVIDIA's unified-memory chips. No device-type
   exception (replaces S22's "skipped on single-heap devices").
5. **A kernel-written length needs no `capacity`** (his proposal, B4.10).
6. **A run locks only its inbox and what its stages write.** A change that marks a
   build, and an eviction, put the resource into the inbox of each plan that names
   it ("haben wir so diskutiert").
7. **Not a question.** "Wer sagt denn wartet nie? Und warum sollte ein store was
   mit speicher druck zu tun haben? Die grosse regel die wir haben ist, dass wir
   vorhande arrays mutieren, und nicht double buffering machen". A store mutates
   the existing array. "Never waits" (B4.9.10) is my wording, not a rule of his;
   the docs drop it.
8. **A kernel-written TLAS total:** as discussed before (B4.8, S29).
9. **(a): core marks the build that must be re-recorded,** when its count or its
   storage changed, and every device type answers with its verb. Vulkan
   re-records the build's own small command buffer: an indirect build that reads
   the count from memory needs `accelerationStructureIndirectBuild`, which is
   `false` on all four desktop devices (7900 XTX RADV, RTX 4000 Ada, the iGPU,
   llvmpipe; vulkaninfo 2026-10-19). Only AMD's Windows driver has it, and it
   measured slower there (`experiments/tlas_sparse/m1_lapwin.txt`: 10 000
   instances 95 µs exact, 103-167 µs at capacity 2x-16x). Metal reads the count
   from a cell. The name `AccelCommand` goes ("der duemmste name den ich jeh fuer
   sowas gehoert hab"); its name is `BuildRecording` ("ok fuer BuildRecording").
10. **(a): per-argument access is the Julia-IR inference at declaration,** one
    place for every device type: `argument_usage` → `kerneltouches`, a taint walk
    over the kernel's typed Julia IR through the backend's interpreter
    (`src/graph/access.jl`, `test_access.jl`, `test_access_kernels.jl`). A
    declaration next to the function only for calls with no Julia body (`mul!`
    into a library). Not part of `KernelInfo`; F2 closes, and api.md's
    "(per-argument access: G2)" goes.
11. **One `Mantle.Kernel` per kernel, attached to its `CodeInstance`** ("es an die
    CodeInstance hoert sich gut an...das machen wir dann mit dem gesamten
    Mantle.Kernel, was alle infos enthaelt"; "ok!"), the way CUDA.jl keeps its
    kernels (`GPUCompiler.cached_results`, `CUDACompilerResults`, with each
    context's `CuFunction` in a list inside it). Its stages fill it in place:
    - at declaration: the function, the argument types and the access per
      argument (10);
    - at compile: the code (SPIR-V, AIR, PTX);
    - at first use on a device: that device's native handle and its
      `KernelInfo` (`kernel.kernelinfo`, no separate table), in a short list found
      by the device's identity.
    Each stage writes its field once, under a lock in the `Kernel` itself; nothing
    is global. Julia invalidates it with the code and keeps it in package images;
    no native handle is added while generating output. A device is never torn
    down kernel by kernel: CUDA.jl does not reset devices at all (its
    `device_reset!` is a deprecated no-op) and finds handles by context identity;
    our one reset, `vk_reset_device!(debug = …)`, makes a new device object, and
    the old entries are never found again. Replaces the per-device `AccessCache`
    and the device type's compiler cache of api.md section 6 (Kernels); core and
    device types keep no other kernel table.

## C. Experiment results (2026-09-28..30; proposals, not yet approved by Simon)

Source: /sim/Programmieren/VideoEdit/experiments/header_indirection/, trusted
files v2_radv.txt, v2_nvidia.txt, dgc_counts_*.txt, dgc2_*.txt, census_radv.txt.

1. Kernels must NOT read per-array headers: +320–490 ns per small dependent
   dispatch on AMD RADV, +120–130 ns on NVIDIA, vs today.
2. Proposal: every array has a header (address, dims) in one device header arena
   as the source of truth; `resize!`/reallocation writes it (eager write); the
   head kernel of each run copies every referenced header into the plan's
   argument slots from a table built at compile (per run: AMD 1.3–1.9 µs,
   NVIDIA 2.0–3.5 µs for 1250–20000 slots). This replaces pushing per-plan
   pointer patches (resize! does not need to know which plans use the array).
   GPURef values still need the per-run ring (host writes while runs are in
   flight).
3. Argument layout: fields directly in the slots ("flat") instead of today's
   pointer-to-inline-copy, AND argument memory marked read-only (NonWritable) so
   RADV uses scalar loads. AMD gains come from read-only (−25% at 1M elements),
   NVIDIA gains from flat (−140 ns per small dispatch). Both are Lava
   entry-wrapper changes. KA ndrange values also live in the argument block.
4. Launch counts: fixed-size arrays → direct, recorded once. Size known to the
   host (`resize!`, window) → re-record the plan's launches on resize! (Mantle
   invalidate!+record! of a 300-dispatch plan: 0.97 ms host; re-placement of
   transients would be a recompile, not measured). Size only on the GPU
   (compaction, queues) → indirect launch (+45–60 ns NVIDIA, +190–245 ns AMD) or
   a grid-stride loop over a fixed per-device grid (AMD). VK_EXT_device_generated_commands
   is not usable for this (15 µs preprocess per dispatch on NVIDIA; ordering bug
   on NVIDIA 595.x).
5. SAM 2 plans are recorded twice today (run 1 and again in run 2; consistent
   with the measured-split re-record of 692a4e1).
6. Lava today: every isbits struct argument is passed as a pointer to an inline
   copy (Mantle launch.jl ~255); on RADV all argument loads are vector loads
   (0 scalar loads), confirmed on a real Mantle broadcast kernel.

## D. Proposals in the 2026-09-30 rewrite (the assistant's, NOT ruled by Simon)

1. SUPERSEDED by Simon's ruling (2026-09-30): arrays are fixed ("immutable")
   by default and become resizable on their first `resize!`; pointer and size
   reach recorded work through the cell ("I thought we simply patch the size +
   ptr?"). Only direct counts, temporaries sized for fewer elements and
   recorded handles recompile a plan. Pseudocode:
   Mantle/docs/resizing-and-raytracing.jl.
2. RULED 2026-10-02 (B3.3), without private arena parts. A run with host nodes is one submission per stage, each stage starting with
   a head node; graphs with host nodes get private arena parts. A change queued
   during a host function is seen by the later stages. Reason: recording later
   stages in advance deadlocks in-order queues (Vulkan spec cmdbuffers.adoc
   3106-3117; Mesa vk_queue.c 745-762); host-side reservations had deadlocks.
3. Not a decision (B3.5). Outputs the host reads are device arrays; a call declares into a graph if
   any argument lives on it.
4. RULED (Simon, 2026-09-30): traces are allowed inside `repeat!` and under
   conditions. "Kernel launch" in B2.8 means launching work on the GPU
   ("when I say kernel launch, I mean launching work on the gpu").
5. Taken into the plan 2026-10-02, not objected to (B3.4). Graph lock is a Base.Semaphore(1) (ReentrantLock disables finalizers);
   host waits are `@ccall gc_safe = true`.
6. RULED 2026-10-02 (B3.7) as the `GPURef` of an array. Was `ArrayRef`: a reference to an array whose target a
   pending rebind replaces; zero-copy plot attributes.
7. RULED 2026-10-02 (B3.7) with `a[min(i, n-1)]` instead. Was `attribute(a, i)` = `a[ifelse(length(a) == 1, 1,
   i)]`: one argument type for a value and per-element attributes; replaces
   the stride fixed at declaration.
8. RULED 2026-10-02 (B3.7). The scene rectangle as a per-draw uniform applied in
   the vertex stage, so layout changes are stores.
9. (2026-10-02, not ruled; plan open decision 16; revised in the review round
   of 2026-10-02) Eviction of idle resources under pressure, restored by
   their next use. Evictable: not held by any composite (TLAS or BLAS range,
   array `GPURef`), no handle dependent, no plan that names it running, not
   lent, nothing pending; a structure's own arrays are held by it, so never.
   Accesses still in flight are allowed: the eviction copy waits for the last
   write through the resource's access record, the storage is retired with
   the tokens of every access in flight and the copy's, and `acquire!` waits
   for those tokens outside every lock. A run looks for evicted resources
   only when its plan is marked (an eviction marks the plans that name the
   resource; a new plan starts marked). Only where
   `Readback` memory is a different heap from device memory (a `caps(dev)`
   field); skipped on single-heap devices (Apple M5), where the host copy
   would take from the same heap. Acceleration structures are not evicted
   (a TLAS or BLAS would need its sources back before a rebuild).
   `restore!` of an
   already-evicted array with no prepared storage marks the plan for
   restore, then returns `Again`. At the last hold (part of the holds, phase
   2, not of this proposal): pending changes dropped under the lock, storage
   `Released()`, old storage retired with the last tokens, the array off the
   eviction list, its cell released; `ArrayBox`, `TLAS`, `BLAS` and
   `RenderObject` have finalizers that push their hold drops through the
   pool's inbox. Pressure steps in order: unmap idle arena pages, trim empty
   blocks, collect garbage, wait for retiring storage, then evict least
   recently used and wait for their retirement tokens, then throw
   `OutOfDeviceMemory`.
10. (2026-10-02, not ruled; plan open decision 16; revised in the review
    round of 2026-10-02) Readers: an array's readers are the composites built
    from it: TLAS, BLAS, an array `GPURef` (forwarding to its own readers),
    a screen (a redraw mark). Host changes reach the structures through
    readers in any design; the proposal is only the run's part below
    (instead of declaring `refit!(g, t)` in every graph that writes a
    source). They are an immutable snapshot replaced under
    the array's lock and counted per reader. A change to an array with
    readers locks the array and its readers together, prepares reader
    capacity before the locks, and returns `Again` if a total crossed a bound
    that was not prepared. A run pends `pendupdate!(reader, array)` after it
    submitted each stage. Reads through a composite are expanded
    transitively (TLAS → BLASes and range arrays, BLAS → geometry, array
    `GPURef` → target) by `lockexpanded`, and the run's waits and recorded
    accesses use the expanded uses; a build is a write of the structure and a
    read of its members, a write through an array `GPURef` a write of its
    target (this expansion is needed for ordering whether or not readers are
    ruled in: concurrency review finding 1, verification round defect 4).
11. (2026-10-02, open; plan open decision 17) Index buffers and other handle
    dependences.
    - Problem: an index buffer is bound by handle (`vkCmdBindIndexBuffer`
      records the buffer), so a move of an index array recompiles every plan
      that recorded it. RayMakie's lines draw an adjacency index list built
      from the NaN pattern (RayMakie e6e1aa9b0, `overlay/lines.jl:515-519`,
      `:568`; requirements review, scenario 13) and meshes
      draw faces as an index buffer; any move (a first `resize!` beyond the
      storage, a doubling, growth beyond a reserve) recompiles, which breaks
      B3.6 ("only if new plots are added or removed").
    - Option (a): pull indices by address in the vertex stage (no index
      buffer, no handle). Loses the post-transform vertex cache's reuse; the
      cost is not measured; measure with `experiments/draw_bench/` on the
      three devices (NVIDIA, AMD Windows, Apple M5).
    - Option (b): one device-wide index reserve bound once, each draw's first
      index written by the head into its indirect command (`firstIndex` on
      Vulkan, `indexStart` on Metal). Keeps the vertex cache; needs an
      allocator inside the reserve and covers only index buffers.
    - Option (c): a handle change re-records the segments that recorded the
      handle instead of recompiling the plan. One mechanism for every handle
      dependence (index buffers, build inputs, anything a command records by
      handle); costs a re-record of those segments per move, which the
      doubling makes rare.
    - Recommendation: (c) as the rule; (a) where measured cheaper.
12. (2026-10-02, open; plan open decision 18; corrected 2026-10-05) RayMakie
    updates that change a shader. Type changes of attributes (colour as
    numbers or colours, Point2f/Point3f, a material of another type) are a
    `delete!` + `insert!` of the plot in Makie already (Simon, 2026-10-05), so
    they recompile like any plot added (B3.6). A new Hikari material type is
    the same case: materials live in a `MultiTypeSet` (one array per type; the
    GPU form is a tuple of typed arrays the shade kernel dispatches over,
    Raycore multitypeset.jl:213-225), so a new type is a new kernel
    specialization. (The earlier text said "hit groups / SBT offset": wrong for
    Hikari.) Left open: value changes that pick another pipeline without a
    type change (`transparency`): (a) RayMakie re-inserts the render object
    (recompile), or (b) both draws recorded under `when!` flags, the change a
    store (+14-34 ns per tiny draw on Vulkan, measured). Which RayMakie
    attributes are such switches is to be listed.
13. (2026-10-02, open; plan open decision 19) Run host time.
    - Problem: a run locks every access record it names. The model of that
      loop measured 293 µs for 25 000 records (plan, "Measurements this plan
      rests on"). Array `GPURef`s add their targets' records: 5 000 plots ×
      5 attributes is about 50 000 records, which was not measured.
    - Option (a): keep the lock loop and measure the real `run!`
      (`test_run_host_time.jl`, 100 to 50 000 records) against the phase 3
      target (within 3x of the model). No design change.
    - Option (b): a per-plan inbox: writers push to the plans that name a
      resource, and a run touches only its inbox. The run's cost no longer
      grows with unchanged resources; every writer and the retirement rule
      change.
    - Recommendation: (a); (b) only if the measurement is over budget.

## Review round 2026-10-02: choices applied, open to veto

The fifth review round (reviews in `experiments/plan-review/reviews/`)
applied these choices to the plan, `api.md`, the ledger and the pseudocode
without a ruling; each stands unless Simon vetoes it.

- S1: a window resized beyond its capacity recompiles the graph at its next
  run, like a graph array beyond the size its slot was placed for; it never
  throws.
- S5: the first `resize!` moves nothing when the new size fits the current
  storage; beyond it a large array on a sparse device moves into a reserve,
  any other into a pool region of twice the need (each doubling a move).
- S13: `when!` launches and draws are indirect and whoever writes their
  commands multiplies the count by the flag; no conditional rendering, the
  same on Metal, and the flag is never a handle dependence.
- S16: `GPURef(dev, a)` is device-only; its target is a root device array (a
  view throws); rebinding to the current target is a no-op.
- S22: eviction (still a proposal, D9) is skipped on single-heap devices, and
  composite members and handle dependents are never evicted.
- S25: `lockgraph(g)` throws when the calling task holds an arena part of
  `g`'s plan; `holdparts!` throws when the task would wait for a part while
  holding one with a higher id.
- S29: a kernel-written TLAS total is built over the capacity with mask-0
  records past it (Vulkan); a kernel-written BLAS length over the geometry
  capacity with degenerate primitives past it, so later changes are Updates.
- S31: `copy!(a, data)` with a new length does not copy the old contents.
- S32 (film part corrected by B3.16: the film and sample count are device
  arrays the screen holds): a traced screen's film and work queues are
  window-sized arrays in the frame graph; the scene rectangle is one array per scene; glyph UVs are in
  texels, normalized in the shader.

Fixes applied in the same round that make the documents agree with each
other or with a ruling, listed so that the ledger's "S<n>" references
resolve:

- S2: pressure never unmaps pages beyond an array's size and never compacts
  acceleration-structure storage; `trim!(a)` (name open) is the explicit call.
- S3: a BLAS has a doubling capacity like a TLAS; a new topology is a Build
  into it, moved vertices an Update; TLASes over it pend an Update.
- S4: stores of up to 64 KiB keep their bytes on the host and are written by
  the builder verb `emitstore!(b, dst, bytes)`; larger stores use staging.
- S6: no access-record, channel, pool or window lock is held while the host
  waits; the graph lock may be.
- S7: readers as immutable snapshots (D10).
- S8: a segment is a list of parts (recordings and `AccelCommand`s).
- S9: window-sized images are recreated per swapchain generation at the
  actual size on the same arena memory.
- S14: `DeviceRange(a)` is dropped; `launchrange(a)` is the one name.
- S15: `reservesize`, `sparsethreshold` and the submission budget's policy
  are core functions of `caps(dev)` facts.
- S17: `attribute(a, i) = a[min(i, length(a))]` (1-based); the benchmark's
  0-based form is `a[min(i, n-1)]`.
- S18: the order of the pressure steps (D9).
- S19: change types include `Restore`, storage kinds `Evicted` and
  `Released`; holders are the creator, each naming graph and each composite;
  `freed` is on the access record only; a range entry has a `mask`.
- S20: nothing is detected at run; a change marks the `AccelCommand` it
  alters.
- S21: composite reads are expanded transitively in a run's waits and
  recorded accesses (D10).
- S23: the last hold and the holder finalizers (D9).
- S24: `withview` takes a hold and applies pending changes first.
- S26: later segments of a stage wait on the first segment's token when they
  touch records whose changes were taken.
- S27 (revised in the verification round): no `except`. A change that
  would make a running plan stale waits on the host until the run has
  submitted its last stage; from the plan's own host function it throws.
  `except` held the change back only from the plan's own submissions: an
  eager call or another graph's run applied it in between and the plan's
  later stages used freed storage.
- S28: build coalescing per structure (any Build → Build) and `builtcount`.
- S30: a value `GPURef` store marks the plans whose entry holds it
  (`entrychanged`).
- S33: an acquired swapchain image has one owner, the window.

