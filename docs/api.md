# Mantle API: types, verbs and their contracts

Status: written 2026-10-02 from `recording-plan.md`, `internals.jl`,
`resizing-and-raytracing.jl` and `examples.jl` as of that day, and from the
rulings in `experiments/plan-review/decisions.md` (B2, B3); revised the same
day with the fixes of the fifth review round (decisions.md, "Review round
2026-10-02"). Code references are to Mantle 5c2d4c6.

This file is the reference the refactor is checked against:

- **The vocabulary test** reads section 4: a device type (`src/vulkan`,
  `src/metal`, `ext/`) may define methods only for the verbs listed there, at
  the signature core calls. It replaces `BACKEND_VOCABULARY`
  (`src/graph/backend.jl:848`, 189 names at 5c2d4c6); what becomes of each of
  those names is in Appendix A.
- **Reviews** check a phase's diff against sections 4 to 9.
- **Changing this file** (a name, a signature, a contract) needs a line in
  `recording-plan.md` first ("Forbidden moves": a new vocabulary name without
  a line in the plan).

Marks: "(name open)" as in the plan; "(contract: phase N)" where the plan does
not yet say enough for a contract, to be written before that phase starts;
"derived" where a statement is this file's reading of a plan section, not a
sentence of the plan; "(open decision N)" for a question the plan lists as
open. Eviction and readers (open decision 16) are ruled: automatic eviction
(decisions.md B3.9) and readers (a) (B3.10).

## 1. Rules every entry obeys

From the plan's "Rules", restated as checks:

1. Core decides; a device type implements verbs. A verb never walks a plan,
   reads the dependency graph, or decides an order, a barrier, a split, a
   queue, a lifetime, a residency set or a fallback.
2. Dispatch, not probing: one function name, methods per type. No `isa`
   chains on device or node types, no `hasmethod`/`hasfield`/`applicable`, no
   `try`/`catch` probes, no Symbol kinds.
3. Every function takes its device, graph or channel. Nothing is found by
   task-local storage, a thread id or a "current" object. The device registry
   is the one process-level state.
4. One path: an operation on a device and the same operation on a graph run
   the same method; where the arrays live decides which.
5. A required verb has a method for every joined device type. A fallback
   that returns `nothing` or a default for a required verb is a forbidden move.

## 2. User API

What application code, Makie recipes and Hikari scenes call. Exported unless
marked internal.

### 2.1 Devices

| call | contract |
|---|---|
| `Device()` | This machine's default: a `LavaDevice` on GPU 0 (never a CPU driver) on Linux and Windows, a `MetalDevice` on macOS. Returns the same object on every call. |
| `devices()` | Every device: each Vulkan physical device as a `LavaDevice` (lavapipe included where Mesa installs it), plus `CudaDevice`s and `ROCmDevice`s once `using CUDA`/`using AMDGPU` loaded their extensions. Driver objects are created on a device's first use, not here. |
| `LavaDevice`, `MetalDevice`, `CudaDevice`, `ROCmDevice` | Subtypes of `abstract type Device <: KA.Backend`. One Julia object per hardware device and driver; one GPU through two drivers is two devices with two pools. |
| `KA.get_backend(a)` | `device(a)` for an array on a device; throws for an array on a graph. |
| `using Lavapipe_jll` | Makes lavapipe's ICD visible; must happen before the first `Device()`/`devices()` (the registry creates the Vulkan instance then). An extension from phase 8. |

### 2.2 Arrays

`MantleArray{T,N,D} <: AbstractGPUArray{T,N}` for every location `D` (a device
type or `GraphDevice`). The internal form carries a fourth parameter for the
role (root or view).

| call | contract |
|---|---|
| `MantleArray(dev, T, dims...)`, `MantleArray(dev, data)` | A pool region on `dev`; operations on it run now. Fixed until its first `resize!`. From data: uploaded at creation (no graph can name it yet, B3.18), streamed from the caller's memory in chunks through upload regions (each chunk one submission within the budget) or written directly into host-visible storage; returns once every chunk is submitted, later uses wait for them. `MantleArray(path; mmap = true)` (B3.9) streams from the mapped file the same way. |
| `MantleArray(g, T, dims...)` | A transient of graph `g`: contents exist only during a run, placed in `g`'s arena. |
| `MantleArray(g, …; persistent = true)` | Kept between runs, used only by `g`; its own region. |
| `MantleArray(g, …; hostwritten = true)` | A transient the host supplies each run with `x[:] = data`. |
| `MantleArray(dev, …; memory = Readback())` (name open) | In host-readable memory: a graph writes it, `Array(out)` after the run is a wait and a memcpy, no GPU copy. Usable as a `when!` flag. |
| `MantleArray(dev, data; streamed = true)` (open decision 6) | A weight placed like a transient and uploaded by a copy node before its first use in each run (`examples.jl` 41; plan, Memory, "Streamed weights"). Keyword and upload cost open. |
| `MantleArray(g, T, win)` | Sized like window `win`; placed at the window's capacity; contents undefined after a resize. |
| `MantleArray(g or dev, …; capacity = n)` | Resizable from the start with a reserve for `n` elements. Required for a length a kernel writes; without it such a launch is refused at declaration. |
| `MantleArray(location(a), T, dims...) do t … end` | A temporary inside an operation: on a device a region retired with the call's token, on a graph a transient. |
| `MantleArray(l, T, SizedBy(f, a))` (internal) | A temporary sized at compile: `f(2 * length(a))` for a host length, `f(capacity(a))` for a kernel-written one. |
| `free!(a)` | Drops the creator's hold, once whatever the number of threads. Later use through any handle throws (eager calls, declarations, stores). Graphs and composites that hold `a` keep using it. Throws for a view. At the last hold the pending changes are dropped and the storage is retired (section 5, "the last hold"). |
| `resize!(a, dims)`, `resize!(a, n)` | Scheduled: the call only enqueues the request (`size(a)` reports the requested size; nothing is allocated, copied or written on the device). The next submission that touches `a` allocates what it needs before its locks and applies it in front of its own work. The first `resize!` call makes a fixed array resizable (plans compiled after it launch over it indirectly); when applied: if the new size fits its current storage, nothing moves (only a cell write); beyond it, a large array on a device with sparse residency moves into a reserve (later growth maps pages, the address stays), and any other array moves into a pool region of twice the need (each later doubling is a move). Which arrays are large, and a reserve's size, core computes from `caps(dev)`. Plans that launched over it directly recompile once. Contents kept as linear memory (`resize!(a, n)` sets the last dimension). Shrinking unmaps nothing. Throws for a view. On a graph array: within the placed size a cell write, beyond it the graph recompiles at its next run. |
| `trim!(a)` (name open) | Pending: unmaps a reserve's pages beyond `a`'s size, retired with the token of the submission that unmaps them; afterwards a view past the size reads undefined values. The only call that unmaps an array's pages: pressure never does (plan, Arrays). Today's `trim!(pool, dev)` frees empty pool blocks. |
| `a[r] = data` | A store, scheduled: the call copies the data once into an upload region (host-visible system memory acquired for it, not the array's storage; B3.18) and enqueues the request with its byte range in the parent (a view's offset applied). The submission that applies it writes it into the storage the array has at that point of the sequence: within the device type's inline limit (`caps(dev)`: 64 KiB at 4-byte offsets and sizes on Vulkan) with `emitstore!`, else through a staging region that submission allocated. A store pended before a resize is copied along with the old contents; one after it writes the new storage. `length(data) == length(r)` or `DimensionMismatch`. A later store covering an earlier pending one replaces it. |
| `copyto!(a, data)` | A store into `a[1:length(data)]`. |
| `copy!(a, data)` | Base's `copy!`: a resize to `length(data)` that keeps no old contents and a store, enqueued together. On a view only with the view's length. |
| `Array(a)`, `copyto!(host, a)` | A host read: its own submission (the pending changes of `a`, a copy into Readback memory), a wait for that token, a memcpy. For a kernel-written length the cell is copied in the same submission. Throws for a graph array. |
| `size(a)`, `length(a)` | The host view. |
| `devicesize(a)` (name open) | The length a kernel wrote: a copy of the cell in a submission of its own, a wait, a memcpy. |
| `capacity(a)` | Only for an array whose length a kernel writes: the `n` it was created with (`capacity = n`), which bounds that length and sizes temporaries over it (`SizedBy`); 0 for every other array (`resizing-and-raytracing.jl` B.2). Not the device's memory budget, which is `budget(dev)` (section 2.8). |
| `launchrange(a)` | The one name for a launch over an array: a reference to `a` resolved at compile, direct while `a` is fixed, indirect after its first `resize!`. `DeviceRange(a)` is dropped; `DeviceRange(count; max)` goes in phase 4. |
| `lengthof(a)` (name open) | A kernel argument through which a kernel writes `a`'s length into its cell; declaring that launch adds a prepare node. |
| views, `reshape`, transposes (GPUArrays' `derive`) | Share the parent's state (storage, size, cell, sync state, holds, freed mark). No hold, no finalizer of their own. |
| `getindex`, iteration, `KA.get_backend` on a graph array | Throw: generic code raises an error instead of being traced. The one exception is `x[:] = data` on a hostwritten array. |
| broadcast, `fill!`, `copyto!`, `map!`, `mapreducedim!`, `accumulate!`, `sort!`, `sortperm!`, `mul!`, `fft`, … | Methods on `MantleArray` that launch through `dispatch!(location(a), …)`: run now on a device, declare on a graph. GPUArrays' generic code serves the rest, on devices only. |
| `allowscalar` | GPUArrays' interface, accepted as such; scalar indexing off by default. |

### 2.3 Graphs

| call | contract |
|---|---|
| `Graph(dev; rewrites = (), profile = false, priority)` (keyword names open) | An empty graph on `dev`. Rewriters run only if listed (`FuseEpilogues()`, `MergeRegions(attribute)`, names open). |
| `run!(g)` | The first call compiles, records and runs; later calls run. A run's first segment waits on the GPU for the previous run of `g`. The host waits only with the graph lock held, never with an sync-state, channel, pool or window lock: for another graph's run that holds a shared arena part, for the previous run of `g` before writing the entry (only a plan marked `entrychanged` by a value `GPURef` store writes it), for a host node's inputs, for the swapchain acquire, and in `acquire!`'s pressure steps (plan, Ordering, Run). Throws for a freed graph, when called from a host function of `g`, and when the calling task holds an arena part of `g`'s plan. |
| `free!(g)` | Releases the plan and drops the graph's holds, once; also done by the graph's finalizer. `run!` and declaring afterwards throw. Called from a host function of `g` it throws before it marks `g` freed, so the finalizer still releases it. |
| `submissions(g)` | Segments and cuts, including those host nodes cause. |
| `timings(g)` | Open decision 13. |
| `naivebytes(g)`, `peakbytes(g)` (names open) | Transient bytes without and with placement. |

A graph is never replaced for a structure change (decisions B3.16).
Declaring into a compiled graph, or removing nodes from it, only marks its
plan: the next `run!(g)` compiles a new plan from the graph's nodes and
retires the old one after its last run. The graph, its windows, the
resources it names and the device arrays it reads stay. A graph array's
contents live only during a run; what must survive runs and recompiles (a
film, an accumulation count) is a device array the graph names. Declaring
takes the graph lock (it waits while a run of `g` has stages left); from a
host function of `g` it throws.

| call | contract |
|---|---|
| `h = insert!(g) do … end` (name open) | Declares the block's nodes and returns a handle to them (kernels that belong together, such as one plot's compute; a plot's draws are a piece of its pass instead, B3.17). A throw inside the block removes what it declared. Not inside a `repeat!` or `when!` body; blocks do not nest. |
| `delete!(g, h)` | Removes the block's nodes; marks the plan. A resource no remaining node names loses the graph's hold once the plan that names it has retired. Nodes declared outside a block stay until `free!(g)`. |

### 2.4 Declaring work

| call | contract |
|---|---|
| `dispatch!(g or location, kernel, args::Tuple, ndrange; origin)` | The one launch verb. On a `GraphDevice` it adds a kernel node (inside a `when!` block tagged with its flag); on a device it compiles from the device type's cache and submits now, with what is pending on its arguments in front. |
| `location(args...)` | The one graph among the arguments if any lives on one (its device arrays are then named by that graph), else the arguments' device. Arguments from two graphs or two devices throw. |
| `dispatch!(g, HostCall(f), args; writes::Tuple)` | A host node. `f` runs on the calling thread of `run!`, between the stages before and after it, on host-visible arena memory. It may not use its own graph or a graph that shares an arena part with it (both throw: `lockgraph` throws when the calling task holds an arena part of that graph's plan). If `f` throws, the rest of the run is not submitted, the exception propagates, and an acquired swapchain image goes back to the window for the next run. Not allowed inside `repeat!` or `when!`. |
| `repeat!(body, g, n; while_nonzero = flag)` | The body is declared once and holds launches of GPU work only (dispatches and traces; copies between arrays are copy kernels); anything else throws at declaration. `body(i)` receives a placeholder; before each iteration a loop-head kernel writes `i` into the slots of the launches that take it. `while_nonzero` ends the loop on the GPU. |
| `when!(body, g, flag)` | Every launch declared in the block runs only where `flag` is nonzero on the GPU (Simon, 2026-10-02). The launches are indirect, and whoever writes their indirect commands multiplies the count by the flag: the head node for a flag the host writes, a prepare node after the kernel that writes it, like `repeat!`'s loop head. No conditional rendering, the same on every device type; the flag is never a handle dependence (decisions.md, review round 2026-10-02). Launches only; blocks do not nest; a block may sit in a `repeat!` body. `flag` is a `GPURef` or a one-element array (a `Readback` one lets the host read it after the run). Graphs only: an eager call runs. |
| `trace!(g or dev, pipeline, accel, args, ndrange)` | A ray-tracing launch; allowed in `repeat!` and `when!`: indirect, its count zeroed by the loop head when the loop ends and multiplied by the flag in a `when!` block. |
| `copyto!(out, x)` with `x` on a graph | A copy kernel node (so allowed in `repeat!` and `when!`), recorded once and run every run; how the host reads a graph result. |

Selection is a call in the method: `mul!` and the other library operations
ask the device's selection table at declaration. Inside `repeat!` and `when!`
only kernel candidates qualify.

### 2.5 Per-run values

| call | contract |
|---|---|
| `GPURef(dev, v)` | A value box: a host value with its own lock (a `ReentrantLock`, a leaf), for the scalars a kernel takes by value. A value that changes every frame and is read as data (a camera matrix) is a one-element array written by a store instead (section 2.2). `GPURef(dev, a::MantleArray)` is the array box (section 2.6b). |
| `r[] = v` | Sets the value under the lock and marks the plans whose entry holds `r` (`entrychanged`); the next run of a marked plan waits for its previous run and copies the value into the entry under the same lock (never torn). An unmarked plan writes no entry and does not wait. |
| `x[:] = data` on a hostwritten graph array | Into the entry once the graph's previous run finished (this call waits, not `run!`; decisions B2.1); while the graph has no current plan or a recompile is pending, kept on the host and written into the new plan's entry. |

### 2.6 Acceleration structures and textures

| call | contract |
|---|---|
| `BLAS(dev, vertices, faces)` | Storage on a pool region sized from its geometry with a capacity that doubles, like a TLAS. Vertices moved with the same topology: an Update (refit), pending. A new topology: a Build into its storage, pending (beyond the capacity a `MoveAccel`, the new address through its cell). TLASes over it pend an Update when its address or contents change. A geometry length a kernel writes: built over the geometry capacity with degenerate primitives past the length, so later changes are Updates. Its builds and refits are `BuildAccel` changes and, in a plan, `AccelCommand`s, like a TLAS's. Shared by every range over the same geometry arrays. A reader of those arrays: a run that writes them pends its update (open decision 16, ruled B3.9/B3.10). |
| `TLAS(dev; hardware)` | One core type for hardware and software. `TLAS(dev)` picks the hardware path where the device has it; `hardware = false` forces software. Storage sized for a capacity starting at 64 that doubles; builds and refits carry the exact count. Pending changes to one structure coalesce: any Build makes a Build, else an Update; every Update uses `builtcount`, the count of the last Build. A total a kernel writes (any device type): built over the capacity with mask-0 records past the total that reference a valid BLAS. An Update before any Build is a Build. A new TLAS pends a Build of zero instances, so a trace before any `push!` reads a built, empty structure. |
| `push!(tlas, blas, transform::Mat4f)` | One instance; returns a range handle. Pending: a range-table store and a Build. |
| `push!(tlas, blas, transforms::MantleArray{Mat4f,1})` | One instance per element. |
| `push!(tlas, blas, positions; scale, rotation)` | One instance per position, transform computed on the device; `scale` and `rotation` arrays or one value. |
| `tlas[h] = t` | Pending: a store into the range's source and an Update (refit). |
| `delete!(tlas, r)` | Pending: the range's entry inactive (`active = false`) and its slot freed, a Build; the range's BLAS and arrays released with the token of the build that deactivated it. |
| `build!(tlas)` | Pending: a full build, applied by the next submission that touches `tlas`. |
| `build!(g, tlas)`, `refit!(g, tlas)` | A build or refit every run. A TLAS whose total a kernel writes is built over its capacity (mask-0 records past the total), so a refit stays legal. A refit before any build is a Build. Never inside `repeat!` or `when!`. |
| `RayTracingPipeline(; raygen, closest_hit, miss, any_hit)` | A value; compiled through `compiletrace` from the device type's cache, keyed on its shaders and argument types. A new material type is a new hit group and a new key, so a recompile (open decision 18). |
| `Texture2D(dev, T, w, h)` | An image with an index in the device-wide texture table. An upload of the same size and type writes the image it has. `resize!(tex, w, h)` or an upload of a new size: a new image under a new index and a store into the slot; the old image retired with the token of the submission that applies the change. Never a recompile. |

In shaders, `instanceCustomIndex` is the range's slot (stable for the range's
life) and the instance's index within its range is `InstanceId -
rangeoffset(tlas, slot)`. A range's triangle data and material index are
reached through the range table, never through a scene-wide concatenation.
Hiding a range: mask 0 in its records (`RangeEntry.mask`, a refit). Moving: a
refit. A new count: a Build with the exact count. Deleting a range: its entry
inactive, a Build. None of these recompiles or drops a plan.

### 2.6b Graphics

| call | contract |
|---|---|
| `GraphicsPipeline(; vertex, fragment, geometry, blend, cull, topology, depth)` (`Rasterizer`), `MeshPipeline(; mesh, fragment, object, …)` | Values. Stages `VertexShader`, `FragmentShader`, `GeometryShader`, `MeshShader`, `ObjectShader` (tessellation as stage types, phase 6). Compiled through `compiledraw` from the device type's cache, keyed on stage functions, device-side argument types, fixed state and target formats; equal keys are one pipeline on every device type. |
| `p = render!(g or dev, targets...) do p … end` | A render node (on a device: runs now); returns the pass. Targets: `image => Clear(v)`, `Keep`, `Discard`; a depth image (`Float32`); a window. A window-sized target's segments are recorded again with each swapchain generation. The pass is a list of pieces (B3.17); the block declares the first. |
| `h = insert!(p) do … end`, `delete!(p, h)` (names open) | A piece of the pass: the draws of the block (a plot), in list order after the others. Applied by the graph's next run before its stages: the piece is compiled and recorded on its own and the pass's list rewritten; nothing recompiles and no other piece is recorded again. Draws only (inside `when!` too); anything else throws. A piece that fails to compile is taken out and `run!` throws. Visibility is a store into the draw's `when!` flag; a changed order is a list rewrite. |
| `draw!(p, pipeline, args, count; instances = 1, indices = nothing)` | A draw. `count`, `instances`: numbers or `launchrange`s (indirect over resizable arrays and array `GPURef`s). `indices`: a `MantleArray{UInt32}` bound by handle; it never moves within its capacity, and a move marks the parts that recorded the handle, which are recorded again before their next submission; nothing recompiles (open decision 17, ruled (c)). |
| `when!(p, flag) do draw!(…) end` | Conditional draws: the draws inside are indirect and the count written into their commands is multiplied by the flag, as for launches in `when!(g, flag)` (no conditional rendering); visibility is a store into `flag`. |
| `attribute(a, i)` (name open, device side) | `a[min(i, length(a))]` (1-based; the benchmark's 0-based form is `a[min(i, n-1)]`), the length from the slot: one argument type and one pipeline for a uniform, one value for all elements, or one per element (measured: within 1% of `a[i]` and of a true uniform on NVIDIA, AMD, M5). |
| `GPURef(dev, a::MantleArray)`, `r[] = b`, `r[]` | The array box: a reference whose target can be replaced between runs. Device only; the target is a root device array of the same element type and dimension count (a view, a transient, a freed or a lent array throws). `r[] = b` is a pending cell write of the box, applied like a store; rebinding to the current target is a no-op. Launches and draws over it are indirect. The box holds its target; the old target's hold is dropped with the token of the submission that applies the rebind. TLASes with a range over the box pend an Update (a Build if the length changed). A reader of its target that forwards to its own readers (open decision 16, ruled B3.9/B3.10). |
| scene rectangle | One one-element array per scene, shared by its plots, that the vertex stage applies to each draw (a remap and four clip distances); a layout change is one store. |
| `Image(dev, T, dims...)`, `Image(g, T, dims...)`, `Image(g, T, win)` | An image resource: sync state, holds, stores (pending uploads), `copyto!` with arrays (pending changes or copy nodes), `Array(img)`. Format from `T`, one table per device type (`imageformat`). Layouts are decided by `barriers!`. `Image(g, T, win)` is recreated per swapchain generation at the actual size on the same arena memory. |
| `Sampler(; filter, address, anisotropy)` | A value; a device-wide sampler table entry created on first use. |
| `Array(win)` (name open) | A screenshot: the next run presenting to the window copies the image into `Readback` memory before presenting; waits for that run. |

### 2.7 Windows

| call | contract |
|---|---|
| `Window(dev, w, h; title, vsync, capacity)` | A graph resource; its swapchain (Vulkan) or drawables (Metal) is its storage; the device type's window open verb makes the window and its surface (section 4.7). Capacity: the largest monitor's size unless given. A render target itself (`render!(g, win => Clear(c))`) of exactly one graph: a second graph declaring a render into it throws at declaration (other graphs render into `Image`s that graph reads; open decision 20, ruled); `Surface` and the `target_*` accessors go. The default colour format and the `vsync` default differ between device types today; which ones Mantle uses is open (G9). |
| `resize!(win, w, h)` | Called from the window's event: stores the size and marks the graphs that render to it; nothing else happens on the event thread. The next run of such a graph makes the new swapchain generation, recreates its window-sized images at the actual size on the same arena memory (the arena part is sparse where possible and maps only the pages the size needs), and records again the segments that draw into window-sized targets and the per-image present segments. Beyond the capacity: the graph recompiles at its next run, like a graph array beyond the size its slot was placed for; it never throws. |
| an acquired swapchain image | One owner, the window. An image acquired by a run that returns `Again` or throws goes back to the window; only real images are kept. |

### 2.8 Memory introspection (names open)

`reserved(dev)`, `mapped(dev)`, `budget(dev)`.

### 2.9 Vendor views

`withview(f, CuArray | MtlArray | ROCArray, a, stream)` (name open, open
decision 9): a zero-copy view of `a` in the vendor's type, inside the block
only. It takes a hold on `a` and applies `a`'s pending changes and any
restore first, in a submission the stream waits on; the stream also waits for
Mantle's other accesses to `a`. Mantle's later uses wait for the stream's
work, also if the block throws, and the hold is dropped at the end with the
stream's token. While lent, every Mantle call that locks `a`'s record throws.
No Mantle lock is held while `f` runs.

## 3. Core types

Internal; listed because the verbs in section 4 receive or return them.

| type | what it is | lives in | protected by |
|---|---|---|---|
| `Device` | abstract; one object per hardware device and driver | the registry | registry lock (opening) |
| `DeviceRegistry`, `REGISTRY` | loaded device types, opened devices, the Vulkan instance | process | its lock |
| `Graph` | nodes as declared, named resources, rewrites, plan; the flags `recompile`, `windowchanged`, `freed` | user | `Base.Semaphore(1)` (not reentrant; owner task checked); the flags are atomics (section 5) |
| `GraphDevice` | what graph arrays point at (a weak reference to the graph) | graph | immutable |
| `Plan` | compiled state: stages, placement, argument block, entry, tables, recordings, `lastrun`; the flags `stale`, `unmapped`, `restore`, `running`, `entrychanged` | graph | graph lock; the flags are atomics (section 5) |
| `CompileState` | the plan under construction; a copy of the nodes | compile | graph lock |
| `Node` subtypes | `KernelNode`, `TraceNode`, `LoopNode`, `HostNode`, `HeadNode`, `PrepareNode`, `CopyNode`, `RenderNode`, `BuildNode`, `ForeignNode`, `RegionNode` (a run of nodes one region compiler compiles), `VideoDecodeNode`, `PresentNode` | plan (its own copies) | graph lock |
| `CompiledNode` | queue, predecessors, barrier, command, cost, slots | node | graph lock |
| `Origin` | the Julia call that launched a node | node | immutable |
| `Kernel{N}`, `KernelInfo` | a native pipeline/function and what core needs about it | compiled node | immutable |
| `Bound{R}` | a plan's argument slots for a resource | plan | node uses |
| `MantleArray{T,N,D,R}` | location, state, role | user | — |
| `ArrayState` | storage, dims, length source, capacity, cell, sync state, holds, dependents, readers (open decision 16, ruled B3.9/B3.10); the freed mark is on the sync state only | shared by an array and its views | its sync state's lock |
| `Cell` | an array's address and dims in the device's cell arena | array state | written only by applied changes or by a kernel through `lengthof` |
| `Region`, `Reserve`, `Committed`, `Evicted`, `Released` | the storage kinds: a pool region; a sparse reserve with mapped pages; a pool region of twice the need that moves on each doubling; the host copy of an evicted array (open decision 16, ruled B3.9/B3.10); nothing, after the last hold | array state | the array's sync state (host view); the pool lock for blocks |
| `Arena` | one arena part: range, mapped bytes, tenants, sync state, holder | pool | its sync state (mapping, tenants); its condition (holder) |
| `SyncState`, `Access` (was `AccessRecord`) | per shared resource: the accesses of what is not a run (eager calls, host reads, vendor views, eviction copies, applied change steps: last write and reads since), the last tokens of plans retired while naming it, pending changes, lent and freed marks. Runs are not recorded here (their last-run tokens, found through the plans naming the resource) | every shared resource | its `ReentrantLock` (never held while `acquire!` runs) |
| `namers` (on `SyncState`), plan inbox, plan sync lock | per resource, the plans naming it with their use from compile (an immutable snapshot, replaced at registration and retirement; composite members reach the plans naming the composite through their readers); per plan, the resources written by other runs, changed or accessed eagerly since its last look (a lock-free list), and the lock that orders its submissions against the plans it shares resources with | the resource; the plan | the resource's sync state lock (`namers`); the plan's sync lock (the inbox is lock-free) |
| `Token` | `(channel, value)`: a value a channel signals. Core's `passed(t::Token)` and `waitfor(t::Token)` call the device-type verbs `passed(ch, value)` and `waitfor(ch, value)` (section 4.2) | accesses, `lastrun`, retirements | immutable |
| `Change` subtypes | `Store` (the storage it writes, a byte range in the parent, its upload region acquired at the call, B3.18; written inline within the `caps(dev)` limit, else copied), `CellWrite`, `BindPages`, `MoveStorage`, `BuildAccel`, `MoveAccel`, `Restore` (open decision 16, ruled B3.9/B3.10), each with a device-wide sequence number | sync state | sync state lock |
| `SubmitChannel{N,T}` | native queue + timeline, kind, lock, `next` (the last token handed out), outstanding submissions | device | its lock (submit, present, sparse binding, wait-idle) |
| `Recording{N}` | one recorded part of a segment, written once. A segment is a list of parts: recordings and `AccelCommand`s | plan | — |
| `Submission{T}` | a token and the command objects to recycle once it passed | channel | channel lock |
| `Builder` | a device type's recording in progress; a value that never escapes the call that made it (`record!`, an eager call, a submission's change prefix, the re-recording of an `AccelCommand` or a window-sized segment) | stack | — |
| `Deps` | the predecessor nodes and the derived barrier of a node | compiled node | immutable |
| `CompiledDispatch`, `CompiledTrace`, `CompiledCopy`, `CompiledBuild`, `CompiledRender`, `CompiledDraw`, `CompiledLibrary`, `CompiledLoop`, `CompiledHead` | what compile decided for one command | compiled node | immutable |
| `TLAS{P}`, `BLAS`, `InstanceRange`, `RangeEntry` (`mask`, `active`), `MemberSet` | resizing-and-raytracing.jl B.8; `builtcount`, the count of the last Build, which every Update uses | user | the structure's sync state |
| `AccelCommand` | a plan's TLAS or BLAS build or refit in its own command buffer, on every device type (Vulkan records the count, Metal the storage). A change that alters what it recorded, or a Build that sets a new `builtcount`, marks it; the run re-records the marked ones after the change prefix (nothing is detected at run) | segment parts | the structure's sync state, held by the run |
| `Window`, swapchain generation | size, pending size, current generation with holds; the swapchain images acquired and not yet presented (the window is their one owner) | user | the window's lock |
| `GPURef` | a value box (a host value) or an array box (a reference to a root device array) | user | value box: its own `ReentrantLock`, a leaf; array box: its sync state |
| texture table, sampler table | the device-wide descriptor tables, with free lists | device | the table's lock, a leaf (section 5) |
| `Pool`, `Inbox` | blocks per kind, retiring list, eviction list (open decision 16, ruled B3.9/B3.10); lock-free inbox for finalizers | device | pool lock (a leaf); none (inbox) |

## 4. Device-type verbs

The only functions a device type implements. For every verb: it dispatches on
the device (or an object of it) as its first argument; it takes no Mantle
lock unless stated; it does not call into core except through the functions
named here; it throws only on device loss or a driver failure, never to
signal "not supported" (capabilities are answered by `caps`).

### 4.1 Device

| verb | called by, when | contract |
|---|---|---|
| `hardwarekeys(T, registry)` (name open) | `devices()`, under the registry lock | The keys of the hardware devices this device type can open. Creates no driver objects beyond enumeration. |
| `T(key)` | the registry, once per key | The device object; driver objects are created on first use, not here. |
| `caps(dev)` | core, at compile and at declaration | An immutable record of facts the device type reports; core decides from them (field names open, G3). Fields: `sparseresidency`; `accelbyaddress`; the largest single allocation (today `maxalloc(dev)`); the sparse page size and the device's sparse address space (core's `reservesize` and `sparsethreshold`); whether `Readback` memory is a different heap from device memory (eviction is skipped where it is not); the driver's job timeout and whether the device drives a display (core's submission budget); the roofline inputs of `costs!` and `schedule!`: throughput per element type, memory bandwidth, whether a transfer queue exists, the cost of a wait between queues (internals.jl section 4: `caps(dev).throughput[T]`, `.bandwidth`, `.transfer`, `.crossqueuecost`); the inline store limit and alignment (`.inlinestore`, `.inlinealign`: 65536 and 4 on Vulkan, `vkCmdUpdateBuffer`); the fields from today's `supports_*` (phases 1 and 8). |
| `config(dev, f)` (name open) | `kernel!`, before `compilekernel` | The compile configuration of kernel `f` on `dev` (internals.jl section 4; contents open, G3). |
| `countsource(dev)` | core, at record and run | `CountRecorded()` (Vulkan) or `CountFromMemory()` (Metal): where a TLAS or BLAS build gets its count. |
| `queuekinds(dev)` (name open) | core's queue policy | The queue kinds the device has (graphics+compute, compute, transfer, sparse binding, video decode, present). |
| `budget(dev)` | `acquire!` | The device's memory budget minus other processes' usage (`VK_EXT_memory_budget`, Metal's recommended working set, `cuMemGetInfo`/`hipMemGetInfo`). One meaning. |
| `instancerecordtype(dev)` | the TLAS records kernel | `VkAccelerationStructureInstanceKHR` or `MTLIndirectAccelerationStructureInstanceDescriptor`. |

Not device-type verbs: `submissionbudget(dev)` (the GPU time one submission
may take), `sparsethreshold(dev)` (the bytes below which a resizable array
doubles in the pool instead of moving into a reserve) and `reservesize(dev,
need)` (the address space of one reserve) are core functions computed from
`caps(dev)` fields; a device type reports facts and core decides (plan,
"Submission split: estimated", Arrays; rule 1). Today `submissionbudget`
answers 0.1 s for every device (`graph/backend.jl:643`).

### 4.2 Channels and submission

| verb | called by, when | contract |
|---|---|---|
| `submitnative!(ch, token, parts, waits, signals, present)` | core's `submit!`, under the channel lock, after the token is claimed and the accesses recorded | Submits `parts` in order; waits on `waits` (tokens of any channel, expressed natively: a timeline wait, `waitForEvent:value:`, stream order or a stream value wait); signals `token` on `ch` and `signals`; presents `present` if given (binary acquire and rendered semaphores on Vulkan). Must not block on the GPU, take a Mantle lock, allocate, or throw except on device loss: nothing fallible happens between claiming a token and submitting. The only call site of `vkQueueSubmit2` and `vkQueuePresentKHR`. |
| `passed(ch, value)` | core's `passed(::Token)` (release engine, waits) | Whether the GPU signalled `value` on `ch`. Non-blocking. |
| `waitfor(ch, value)` | core's `waitfor(::Token)` (host reads, entry writes, host nodes, pressure) | Blocks until `value` passed on `ch`. A GC-safe call (`@ccall gc_safe = true`). Never called with an sync-state, channel, pool or window lock held. |
| `recycle!(ch, objects)` (contract: phase 3, G6) | the release engine, once a submission's token passed | Returns an eager call's command objects to the device's free list. |
| `bindpages!(ch, token, binds, waits)` | core's `sparsesubmit!`, under the channel lock, the token claimed | A queue operation binding pages (Vulkan and Metal through the device type's one sparse-binding helper, CUDA and HIP `cuMemMap`/`hipMemMap`), waiting on `waits`, signalling `token`. Same no-fail rule as `submitnative!`. |
| `unbindpages!(ch, token, unbinds, waits)` | as `bindpages!` (also for `trim!(a)` and idle arena pages) | The same, unbinding. |

A `Token` is `(channel, value)` (section 3); core's `passed(t::Token) =
passed(t.channel, t.value)` and `waitfor(t::Token) = waitfor(t.channel,
t.value)`. The last value handed out on a channel is core data
(`SubmitChannel.next`), not a verb: today's `fence(ch)` becomes a field read.

### 4.3 Memory

| verb | called by, when | contract |
|---|---|---|
| `rawalloc(dev, kind, bytes, constraint)` | the pool's `tryacquire!`, under the pool lock | One block of `kind` (`Buffers`, `Unified`, `Readback`, `Images`, `Accel`) meeting `constraint` (alignment, usage). A driver call, not a GPU wait; runs no finalizer. The only call site of `vkAllocateMemory` (and of `cuMemCreate`/`hipMemCreate` for blocks). |
| `rawfree(dev, block)` | the release engine | Frees a block whose last use passed. |
| `createreserve(dev, bytes)` | core's growth decision, before any lock | A `Reserve`: sparse buffer, placement-sparse buffer (added to the device's residency set), or reserved address range; no pages mapped. `bytes` is core's `reservesize`. |
| `placeimage(dev, region, …)` (contract: phase 2, G15) | the pool | An image on a pool region. |
| `placeaccel(dev, path, region)` | the TLAS/BLAS code | An acceleration structure object on a pool region (never a reserve: VUID-VkAccelerationStructureCreateInfoKHR-buffer-03615). |
| `accelsizes(dev, path, n)` | the TLAS/BLAS code | Storage and scratch bytes for `n` instances or a geometry. |
| `acceladdress(dev, as)` | the TLAS code (cell write) | `vkGetAccelerationStructureDeviceAddressKHR` or the Metal resource ID. |
| counted objects (names: phase 2, G10) | the release engine | The bytes of a driver object the API cannot place (ICB, pipeline, descriptor pool, swapchain image), and its destruction. |

### 4.4 Compile

| verb | called by, when | contract |
|---|---|---|
| `compilekernel(dev, f, argtypes, config) -> (native, compiled::Bool)` | `kernel!` at declaration (eager) and at compile, with no lock held other than the graph lock (compile runs under it) | From the device type's compiler cache, per device, keyed on the kernel body, thread-safe. `compiled` says whether this call compiled (core's per-device counter adds it up). |
| `kernelinfo(dev, native) -> KernelInfo` | `kernel!` | No optional fields: registers, shared memory, group size limits and occupancy (from the API, else the device's limits), instruction and memory-operation counts of the compiled IR. (per-argument access: G2) |
| `compiledraw(dev, pipeline, argtypes, formats)` | `compile!` of a render node | A pipeline from the device type's cache, keyed on the stage functions, `argtypes` (device side), the fixed state and the target `formats`; the same key gives the same pipeline. |
| `compiletrace(dev, pipeline, argtypes)` | `compile!` of a trace node | A ray-tracing pipeline and its shader binding table layout from the device type's cache, keyed on the shaders and `argtypes`. |
| `compilebindings(dev, s)` | `bindings!` | Descriptor objects for the layout core decided; nothing else. |
| `candidates(dev, f)` | `select` at declaration | The candidates the device type contributes for `f`, with their attributes (`implements`, region compiler). |

### 4.5 Record

| verb | contract |
|---|---|
| `Builder(dev, ch)` | Command objects from the device's free list. A value that never escapes the call that made it: `record!`, an eager call, a submission's change prefix, or the re-recording of an `AccelCommand` or a window-sized segment. |
| `emitdispatch!(b, node, cmd::CompiledDispatch, deps)` | The barrier in `deps`, then the launch, direct or indirect as compile decided (a launch in a `when!` block is indirect; the flag is in its count, not in a condition). |
| `emittrace!(b, node, cmd::CompiledTrace, deps)` | As `emitdispatch!`. |
| `emitcopy!(b, node, cmd::CompiledCopy, deps)` | As `emitdispatch!`. Also a change's copy (`node = nothing`, `deps = Deps((), ())`): an upload region to storage, a cell copy, a move, a restore. |
| `emitstore!(b, dst, upload)` | An inline store within the `caps(dev)` limit in a change prefix: the bytes of the store's upload region (written at the store's call, B3.18) written into `dst` inside the submission. Vulkan: `vkCmdUpdateBuffer` (at most 65536 bytes per call). Device types without an inline store report a limit of 0 and get a copy from the upload region (`emitcopy!`). Allocates nothing. |
| `emitbuild!(b, node, cmd::CompiledBuild, deps)` | A TLAS or BLAS Build or Update with the count in `cmd` (Vulkan: recorded, `builtcount` for an Update) or from a cell (Metal). |
| `beginrender!(b, node, cmd::CompiledRender, deps)`, `emitpieces!(b, node, natives)`, `endrender!(b)` | A render pass: its barrier (target layouts, and the fixed barrier from compute, copy and build writes to vertex, index, indirect and fragment reads), then its pieces in list order (Vulkan: one `vkCmdExecuteCommands`, the pass begun with `VK_RENDERING_CONTENTS_SECONDARY_COMMAND_BUFFERS_BIT`; Metal: `executeCommandsInBuffer` per piece). |
| `recordpiece(dev, node, draws, region) -> native` | One piece's draws with no barrier, binding its own pipelines and setting viewport and scissor itself (no state is inherited, `pipelines.adoc:131-133`). Vulkan: a secondary command buffer with `VkCommandBufferInheritanceRenderingInfo` for the pass's formats; Metal: an indirect command buffer. Uses `emitdraw!(b, node, cmd::CompiledDraw)` per draw. Allocates nothing: its region was acquired before. |
| `bindindices!(b, node, storage)` | An index-buffer bind inside a render pass, by handle (open decision 17). |
| `emitlibrary!(b, node, cmd::CompiledLibrary, deps)` | A library call written into the recording (cuDNN into the CUDA graph). |
| `emitloop!(f, b, node, cmd::CompiledLoop, deps)` | The loop start and loop head around `f()`, which emits the body. |
| `finish(b)` | The recording. |

For all builder verbs: `deps` is complete and core's (predecessors and the
derived barrier). CUDA/HIP add a node with those edges; Vulkan and Metal
record the barrier and the command, the barrier through one helper shared by
the verbs (the only call site of `vkCmdPipelineBarrier2`). `node` is the
plan's node (CUDA/HIP keep a handle per node) or `nothing` for eager calls and
changes. Record compiles nothing, allocates no device memory, uploads
nothing, submits nothing and reads no mutable resource field.

### 4.6 Arrays

| verb | contract |
|---|---|
| `devicearg(dev, a)` | The one conversion to the type the device's compiler wants (`LavaDeviceArray`, `MtlDeviceArray`, `CuDeviceArray`, `ROCDeviceArray`); views, reshapes and transposes resolve to parent region, offset and strides here. |
| vendor view (name open; phase 6 for core's `withview`, each device type's constructor when it joins: Metal phase 9, CUDA and HIP phase 10; G11) | A zero-copy `CuArray`/`MtlArray`/`ROCArray` over a region, not owning it. |

### 4.7 Windows

| verb | contract (phase 7, G9) |
|---|---|
| window open verb (name open, G13) | Makes the native window and its surface (Vulkan) or layer (Metal) for `Window(dev, w, h; …)`; on the main thread (GLFW). The plan's "windows: open, acquire, present". |
| `createswapchain(w, width, height; oldswapchain)` | A new swapchain generation. Called by the first of the window's graphs to run after a resize, under the window lock. |
| `acquirenext(sc)` | `(result, index, semaphore)`; may block; called before the run's other locks, with only the graph lock held. |
| `surfacesize(w)` | The surface's current size (for OUT_OF_DATE). |

### 4.8 Textures

| verb | contract |
|---|---|
| `tableentry!(dev, table, index, texture)` | A host write of a descriptor into an unused index of the device-wide texture table (not a recorded command; descriptorsets.adoc:793-807). Called under the table's lock. |
| `samplerentry!(dev, table, index, sampler)` (name open) | The same for the device-wide sampler table; a sampler is a value, created once per device. Called under the table's lock. |
| `imageformat(dev, T)` (name open) | The one format table: element type to the device's format, for targets and textures alike. |

## 5. Ownership and locks

| state | created by | released by | protected by |
|---|---|---|---|
| device objects, Vulkan instance | the registry | process exit | registry lock |
| pool blocks | `rawalloc` via `acquire!` | the release engine via `rawfree` | pool lock (leaf; no GPU wait inside) |
| a region, staging, a plan, a recording | `acquire!` / compile / record | `retire!(pool, x, tokens)`, released when the tokens passed | the inbox (lock-free) then the pool lock |
| an array's storage, dims, pending changes | constructor, `resize!`, stores | the last hold (below) | the array's sync state |
| an array's holds | the creator, each graph that names the array, each composite that holds it (a TLAS or BLAS range, an array `GPURef`) | `free!`, `free!(g)`, `free!` of the composite; finalizers of `ArrayBox`, `TLAS`, `BLAS` and render objects push their hold drops through the pool's inbox | atomic compare-and-swap; never 0 → 1 |
| the last hold | the holder that drops it | under the array's record lock: pending changes dropped (their staging retired), storage swapped for `Released()`, the old storage retired with the last tokens, the array taken off the eviction list, the cell released | the array's sync state |
| a cell | the array constructor | at the last hold | written only by applied changes and `lengthof` kernels |
| graph nodes, plan | declaration, compile | `free!(g)` / finalizer | graph semaphore; one declaring task |
| a plan's entry | compile | with the plan | graph lock; written only by a run of a plan marked `entrychanged` and by `x[:] = data`, both after the previous run |
| a `GPURef` value | `r[] = v` | — | the value box's `ReentrantLock` (a leaf; taken under the graph lock by the run that copies the value) |
| tokens, outstanding submissions | `submit!`, `sparsesubmit!` | `recycle!` once passed | channel lock |
| an arena's mapping and tenants | compile, `bindunmapped!`, `unmapidle!` | `free!(g)` (untenant) | the arena's sync state |
| an arena part's holder | `holdparts!` at the start of a run | `releaseparts!` after its last stage | the arena's condition |
| a window's size, swapchain generation and acquired images | `resize!(win)` / `applywindows!`; the run's acquire | holds of the window and of runs that acquired; an image goes back to the window when its run returns `Again` or throws | window lock |
| a TLAS's or BLAS's slots, members, storage, scratch, capacity, `builtcount` | `push!`, `delete!`, `resize!` of a source, geometry changes | `free!(t)` | the structure's sync state (`change!`) |
| an `AccelCommand`'s command buffer and its re-record mark | the run (marked ones); the change that alters what it recorded (the mark) | retired with its last token when replaced | the structure's sync state, held by the run |
| compiled kernels and pipelines | the device type's compiler cache | the device type | the device type (thread-safe, per device) |
| an array `GPURef`'s target | `GPURef(dev, a)`, `r[] = b` (a pending cell write of the box) | the old target's hold, with the token of the submission that applies the rebind | the box's sync state |
| texture and sampler table entries | `Texture2D`, `Image` with an index, `Sampler` on first use (`tableentry!`, `samplerentry!`) | the image's last hold (the index back to the free list with its last tokens) | the table's lock (a leaf) |
| an array's readers (open decision 16, ruled B3.9/B3.10) | the composites built from it: TLAS, BLAS, array `GPURef` (forwarding to its own readers), a screen (a redraw mark) | the reader's `delete!` or `free!` | an immutable snapshot, replaced under the array's lock and counted per reader |
| an evicted resource's host copy (open decision 16, ruled B3.9/B3.10) | eviction under pressure | `Restore` (after its upload's token), `free!` or the last hold | the resource's sync state |
| a composite's members (TLAS, BLAS, tables, render objects) | the composite (creator hold) | its `free!` | holds; changes on the composite's record |

Flags written without the graph lock are atomics (`@atomic`); each is set by
the function that makes the change and cleared by the run or compile that
consumes it:

| flag | on | set by | cleared by |
|---|---|---|---|
| `recompile` | graph | `markstale!` (a dependency changed, a graph array beyond its slot's size), a window resized beyond its capacity, a failed compile | `recompile!` |
| `windowchanged` | graph | `resize!(win)`, OUT_OF_DATE, SUBOPTIMAL | `applywindows!` |
| `freed` | graph | `free!(g)` (after its re-entry check) and the graph's finalizer | never |
| `stale` | plan | `markstale!` | the recompile that replaces the plan |
| `unmapped` | plan | `unmapidle!` under pressure | the run that binds the pages again |
| `restore` (open decision 16, ruled B3.9/B3.10) | plan | eviction of a resource it names; `restore!` of an already-evicted array with no prepared storage | the run that restores |
| `running` | plan | the run, before its first submission | the run, after its last stage |
| `entrychanged` | plan | `r[] = v` of a value `GPURef` its entry holds; compile (a new plan starts marked) | the run that writes the entry |

Changes to an array with readers (ruled 2026-10-05, B3.10) lock the array
and its readers together (`lockexpanded`) and pend each reader's update; the
capacity a new total needs is allocated by the submission that applies the
build (scheduled, like every change). A run pends `pendupdate!(reader, array)` after
it submitted each stage. Between graphs, a plan naming a composite is
among its members' plans, through their readers (open decision 19, ruled: a
stage writing a resource waits for the complete last runs of the other
plans naming it; a stage reading one from its inbox for those of its
writers). Submitters that are not runs
expand a composite to its members, transitively (TLAS to its BLASes and range
arrays, BLAS to its geometry, array `GPURef` to its target), and wait for and
record the expanded uses.

Lock order, everywhere: graph lock → the run's hold on its arena parts, by
part id → window lock → plans' sync locks by plan id (a stage: its own,
those of the plans sharing what it writes and of the plans naming its inbox
resources; anything else: those of the plans naming the resources it
touches) → sync states by
resource id (an array and its readers together, through `lockexpanded`) →
channel lock. The pool lock, the
inbox, a value `GPURef`'s lock, the texture and sampler tables' locks and a
plan's `runend` lock are leaves (`runend` guards the plan's `between` flag,
set under stage 1's record locks and cleared when the last stage is
submitted; a change that would make the plan stale waits on it, holding no
lock). `acquire!` is never called with an sync-state, channel or pool
lock held. No lock is held while vendor code runs.

Checks that keep the order: `lockgraph(g)` throws when the calling task holds
an arena part of `g`'s plan; `holdparts!` throws when the task would wait for
a part while holding one with a higher or equal id. The host waits (section 2.3,
`run!`) only with the graph lock held, never with an sync-state, channel,
pool or window lock.

## 6. Invariants and the tests that check them

| invariant | checked by |
|---|---|
| A device type defines only section 4's verbs, at their signatures | the vocabulary check (phase 0, `@test_broken` until clean; ledger P0.T16) |
| Each driver entry point is called from exactly one function (section 8) | the driver-call check (phase 0, `@test_broken` until clean; ledger P0.T17) |
| No mutable globals outside section 9's allowlist | the globals check (ledger P0.T18) |
| No probes outside section 9's allowlist; `@static Sys.is*` only in `Mantle.jl`'s include block | the probe check (ledger P0.T19) |
| Core passes Aqua and JET | the Aqua and JET check (ledger P0.T20) |
| Every function takes its device; two devices open for the whole suite never mix | `test_device_is_an_argument.jl`; the two-devices check (ledger P0.T21) |
| One path: every eager entry reaches `compilekernel`, one submission per launch | `test_one_path.jl` |
| Changes are pending until the next use; every run sees the old or the new state, never a mix | `test_resize_is_queued.jl`, `test_hostwritten.jl` |
| Runs of one graph never overlap (plan, rule 9): a run's first segment waits on the GPU for the previous run's last token on each channel | `test_runs_never_overlap.jl` (negative control: the wait removed) |
| Record is pure | `test_record_is_pure.jl` |
| A plan is recorded once; in steady state `run!` allocates, records and compiles nothing. Re-recorded are only the `AccelCommand`s a change marked and, per swapchain generation, the segments drawing into window-sized targets and the per-image present segments | `test_recorded_once.jl`, `test_run_is_submit.jl` |
| A builder is a value | `test_builder_is_a_value.jl` |
| Memory is released only after the tokens of its last uses passed | `test_lifetime.jl`, `test_memory_returns.jl` |
| Cross-graph order by sync states, per segment | `test_crossplan_order.jl` (negative control: `ordering = Unordered()`) |
| Any thread compiles, records, runs, frees | `test_any_thread.jl`, `test_interleaved_recording.jl` |
| Host nodes: stages, re-entry throws, a throwing host function leaves the device usable | `test_host_node.jl` |
| A window resize costs no device-wide wait and no recompile within capacity; beyond the capacity the graph recompiles at its next run | `test_window_resize.jl` |
| Launch kernels never read cells (except head, prepare, the TLAS offsets and records kernels, `lengthof` writers) | derived (no test in the plan's list): a check over every plan the suite compiles that only those nodes have a `CellRead` use (phase 4, G16) |
| Composites cannot free early, leak or free twice | `test_composite_fuzz.jl`: `push!`, `delete!`, `resize!` (green in phase 5), array `GPURef` rebinds (phase 6), plot updates (phase 11) interleaved with runs on several threads; validation clean; reserved memory back at baseline after `free!` |
| Two graphs that each need 90% of the device run alternately (open decision 16, ruled B3.9/B3.10) | `test_eviction.jl` (green in phase 4 once open decision 16 rules eviction in): back to back with no host wait, restore before the run, results exact, no recompile; composite members and handle dependents never evicted |
| A RayMakie screen does not recompile when plots are added or removed | RayMakie's compile-counter session (written in phase 0, green in phase 11): plot inserts and deletes (pieces, B3.17), attribute changes, value ↔ per-element, device-array rebinds, visibility, camera, layout, resizes within capacity, TLAS moves and count changes; zero compiles. Insert time per plot measured at 100, 1 000 and 5 000 plots must not grow with the plot count; a visibility change records nothing |
| A frame does no host walk over render objects; a run's host time does not grow with the number of resources it names, only with the named resources its stages write and its inbox | `test_run_host_time.jl` (green in phase 3: plans naming 100, 2 500, 25 000 and 50 000 resources with nothing changed take the same host time per run; open decision 19, ruled) |
| Nothing fallible between claiming a token and submitting | a review item, not a test: the review of `submitnative!`, `bindpages!`, `unbindpages!` and of core's `submit!` in each phase that touches them |

## 7. Corner cases

- `resize!` or `free!` of a view throws.
- Declaring into a compiled graph marks its plan for recompile (B3.16); declaring into or running a freed graph throws.
- A host function that runs, frees or writes into its own graph, or runs a graph sharing an arena part with it, throws: `lockgraph(g)` throws when the calling task holds an arena part of `g`'s plan, and `holdparts!` throws when the task would wait for a part while holding one with a higher or equal id.
- `free!(g)` from `g`'s own host function throws before it marks `g` freed (the re-entry check comes before the swap of `freed`), so the graph's finalizer still releases the plan and the holds.
- Arguments from two graphs or two devices throw at `location`.
- A device-decided length without a capacity is refused at declaration.
- `getindex`, iteration, `Array(x)`, `KA.get_backend` on a graph array throw.
- A non-launch node (build, render pass, image copy, host call) in `repeat!` or `when!` throws; nested `when!` throws.
- `refit!(g, t)` for a TLAS whose total a kernel writes is legal: the structure is built over its capacity.
- A store whose data length differs from the range throws `DimensionMismatch`.
- A call on an array lent to vendor code throws.
- `free!(a)` while graphs hold `a`: the graphs keep running; `a`'s handle throws.
- A window with a zero extent in either dimension (minimized) skips the whole run: nothing is submitted and the changes stay pending; OUT_OF_DATE is a resize to the surface's size, applied and retried by the same `run!`; SUBOPTIMAL presents and marks.
- A window resized beyond its capacity: the graph recompiles at its next run (never a throw).
- A run that returns `Again` or throws after acquiring a swapchain image gives the image back to the window, its one owner; the next run presents it.
- An allocation beyond the budget throws `OutOfDeviceMemory` only after the pressure steps: unmap idle arena pages, trim empty blocks, collect garbage, wait for retiring storage, then (open decision 16, ruled B3.9/B3.10) evict least recently used idle resources and wait, outside every lock, for the tokens their storage was retired with (the accesses in flight and the eviction copies). Pressure never unmaps pages beyond an array's size and never compacts acceleration-structure storage (`trim!(a)` is the explicit call).
- A change that would make a running plan stale (a move of storage it recorded by handle, a length it launches over directly, a temporary it sized outgrown, a rebind of a `GPURef` whose target it bound by handle, a capacity change of a structure it sized temporaries from) waits on the host until the run has submitted its last stage (only a run past its first stage is waited for; no fairness: it may wait again if the plan's next run gets past its first stage first). From inside any run (any host function) it throws: it could wait in a cycle. No submitter applies it in between.
- `r[] = b` with another element type or dimension count, or with a transient, a view, a freed or a lent array, throws (the argument type would change, or the box would hold a view's offset it cannot carry). `r[] = b` with `b` the current target is a no-op.
- `copy!(a, data)` with a new length does not copy the old contents.
- Eviction (open decision 16, ruled B3.9/B3.10): evictable is a resource no composite holds, with no handle dependent, that no running plan names, not lent, with nothing pending; accesses still in flight are allowed (the eviction copy waits for the last write; the storage is retired with the tokens of every access in flight and the copy's). A structure's own arrays are held by it and never evicted. Acceleration structures are not evicted. Eviction runs only where `Readback` memory is a different heap from device memory (`caps(dev)`); on a single-heap device (the Apple M5) the step is skipped. A host read of an evicted array reads the host copy; `free!` of an evicted resource frees the host copy; a plan that names an already-evicted array with no prepared storage is marked for restore and its run returns `Again`.
- A plot with zero elements keeps its render object; its draw count is zero.
- A pipeline key change (another shader for a plot, another attribute element type, a new Hikari material type) counts as removing and adding the plot: a recompile (open decision 18).
- An index array that moves (beyond its capacity) marks the parts that recorded its handle; they are recorded again before their next submission, nothing recompiles (open decision 17, ruled (c)).

## 8. Driver entry points in one place

The plan's rule: each driver entry point is called from exactly one function
(plan, "Checks that fail when a rule is skipped"). The plan names three; the
rest follow from section 4 (derived) and are checked the same way, per device
type, from the phase it joins. Where several verbs need one entry point, they
share a helper, as the emit verbs share the barrier helper.

| entry point | the one function that calls it |
|---|---|
| `vkAllocateMemory` | `rawalloc` (plan) |
| `vkQueueSubmit2` | `submitnative!` (plan) |
| `vkCmdPipelineBarrier2` | the Vulkan builder's barrier helper used by every emit verb (plan) |
| `vkFreeMemory` | `rawfree` |
| `vkQueuePresentKHR` | `submitnative!` |
| `vkQueueBindSparse` | the Vulkan sparse-binding helper (`bindsparse!`, name open) used by `bindpages!` and `unbindpages!` |
| `vkCmdUpdateBuffer` | `emitstore!` |
| `vkWaitSemaphores` | `waitfor` |
| `vkGetSemaphoreCounterValue` | `passed` |
| `vkAcquireNextImageKHR` | `acquirenext` |
| `vkCreateSwapchainKHR` | `createswapchain` |
| `vkCmdBuildAccelerationStructuresKHR` | `emitbuild!` |
| `vkGetAccelerationStructureBuildSizesKHR` | `accelsizes` |
| `vkCreateAccelerationStructureKHR` | `placeaccel` |
| Metal `commit` on an `MTL4CommandQueue` | `submitnative!` |
| Metal residency set `addAllocation`/`removeAllocation`/`commit` | the Metal residency helper (`residency!`, name open) used by `rawalloc`, `rawfree`, `createreserve`, `bindpages!`, `unbindpages!`, and by compile for a plan's own set (its ICBs, argument memory and pipelines; plan, Metal, "Compiled form") |
| `cuGraphLaunch`/`hipGraphLaunch` | `submitnative!` |
| `cuMemMap`/`hipMemMap` | `bindpages!` |
| `cuMemUnmap`/`hipMemUnmap` | `unbindpages!` |

`vkCmdBeginConditionalRenderingEXT` is not used: launches and draws in a
`when!` block are indirect, with the flag in their count (section 2.4).
`vkDeviceWaitIdle`: see G5.

## 9. Allowlists

- **Mutable globals:** `REGISTRY` (the device registry). Nothing else, in
  core or in a device type (phase 8 deletes the rest).
- **Probes** (`hasmethod`, `applicable`, `hasfield`): none.
- **`@static Sys.is*`:** only in `Mantle.jl`'s include block.
- **`isa`/`typeof` on device or node types:** none.

## 10. Gaps: what the plan does not specify yet

Each needs a plan line (or a ruling) before the phase named. The vocabulary
names that point here are in Appendix A.

- **G1 Graphics: what is still open** (phase 6). Specified on 2026-10-02 in
  the plan's "Graphics" section: pipelines and their keys, render passes and
  targets, draws and counts, conditional draws, attributes, images, samplers,
  screenshots. Still open: the shader binding table's layout for many
  material types and procedural hit groups (`set_anyhit_pipeline!`'s role;
  open decision 18), stencil, multisampling, blending modes beyond the four
  singletons.
- **G2 Per-argument access of a kernel or shader** (phase 1): which arguments
  a kernel reads and writes. `dependencies!` and `barriers!` need it
  (`uses(p)` in internals.jl section 4); today device types analyse kernel IR
  (`kerneltouches`, `shadertouches`, `vertextouches`, `fragmenttouches`,
  `kernelinterpreter`, `kakernelaccesssignature`, `argument_usage`,
  `intrinsic_usage`, `accesscache`). Proposal (not ruled): part of
  `KernelInfo`, answered by `kernelinfo`, and by `compiledraw`/`compiletrace`
  for shaders.
- **G3 Device facts as fields of `caps(dev)`** (phases 1 and 8): the field
  names and types. The plan says a device type reports facts and core
  decides, and names as `caps(dev)` facts `sparseresidency`,
  `accelbyaddress`, limits, throughput and bandwidth for the roofline
  estimate, the sparse page size and sparse address space (core's
  `reservesize` and `sparsethreshold`), the job timeout and whether the
  device drives a display (core's submission budget), and whether `Readback`
  memory is a separate heap (the eviction proposal). Not yet in the plan:
  the field names and types; the fields for today's `supports_*`; the
  largest single allocation (today `maxalloc`, unless "limits" covers it);
  the cost of a wait between queues (`crossqueuecost(dev)` in internals.jl
  section 4; `hastransfer(dev)` can be read from `queuekinds(dev)`); the
  kernel compile configuration `config(dev, f)`.
- **G4 Profiling** (open decision 13): `Profiler`, `makeprofiler`,
  `profiled!`, `gpupasstime!`, `collect!`, `timestamps`.
- **G5 Device wait-idle** (phase 3): the plan has wait-idle under the channel
  lock (Windows) but no verb, and deletes the device-wide wait from acquire
  and present; today's `waitidle`.
- **G6 `recycle!` on Vulkan, CUDA and HIP** (phase 3). The recording
  lifecycle that replaces `makerecording`, `finishrecording!`,
  `resetrecording!`, `destroyrecording!`, `abandon*!` and `recorder` is
  specified: `Builder`/`finish` (plan, Record), recordings retired by token
  with their plan (plan, Compile), and Metal's free list of
  allocator/command-buffer pairs (plan, Metal, "Channels"). Open: the shape
  of `recycle!` (which objects, which free list) on Vulkan, CUDA and HIP.
- **G7 Copies between devices** (open decision 7): `devicecopy!`.
- **G8 The selection table: locking and measurement** (phase 6). Specified:
  the key (function, element types, shapes, layout, alignment, with
  `basealignment`; plan, Operations, and phase 6), the table per device, and
  the rule that an unmeasured entry uses core's cost estimate. Open: the
  table's locking, and when measurement runs (`bestshape` has no definition
  at 5c2d4c6).
- **G9 Window API** (phase 7): specified in the plan (a window is a target,
  `Surface` and the `target_*` accessors go); open: GLFW hints, the default
  colour space and format (Vulkan defaults to an sRGB swapchain, Metal to
  `BGRA8Unorm`, so one shader output means different things today), and the
  `vsync` default.
- **G10 Counted driver objects** (phase 2): verb names for an object's bytes
  and its destruction.
- **G11 Vendor views per device type** (phase 6, the plan's phase list adds
  scoped vendor views there): core's `withview` and the foreign channel
  (`foreignchannel`, `streamwaits!`, `signal!` in internals.jl section 14)
  in phase 6; each device type's view constructor when it joins (Metal in
  phase 9, CUDA and HIP in phase 10).
- **G12 Device teardown for debugging** (phase 8): `reset_device!`, declared
  and never implemented; what tearing down a device means for its pool,
  graphs and the registry.
- **G13 The window open verb** (phase 7): the plan's table lists "windows:
  open, acquire, present"; the verb that makes the native window and its
  surface has no name or signature.
- **G14 Texture and sampler table locking** (phases 5 and 6): the tables are
  device-wide with free lists; this file gives each a lock that is a leaf
  (section 5, derived). Open: whether a table index is retired by token like
  memory, and how a restored texture gets its new index.
- **G15 Contracts marked "(contract: phase N)"**: `placeimage`'s arguments
  (phase 2), `recycle!` (phase 3, G6).
- **G16 An invariant with no test in the plan's list**: launch kernels never
  read cells (phase 4). Section 6 sketches a check (derived).

## Appendix A. `BACKEND_VOCABULARY` at 5c2d4c6, name by name

189 names (`Immediate`, listed here before, is not in the vocabulary; ledger
P7.D9 deletes it). Basis: **P** the plan says it (section named), **D**
derived from the section named (to be confirmed in review), **G** a gap
(section 10). Counts: 91 P, 71 D, 27 G.

| name | fate | phase | basis |
|---|---|---|---|
| `abandonframe!` | goes: recording lifecycle replaced by Builder/finish and the free list | 3 | D: Record |
| `abandonrecording!` | goes: as abandonframe! | 3 | D: Record |
| `abandonrun!` | goes: as abandonframe! | 3 | D: Record |
| `access` | internal to the device type's barrier helper (how a barrier is expressed) | 3 | D: Record, Four device types |
| `accesscache` | per-argument access of a kernel | 1 | G2 |
| `acquire_next_image!` | becomes `acquirenext(sc)` (verb), called by core's acquireimage! | 7 | D: Windows and present |
| `AdaptedAccel` | goes with Metal's HWTLAS bookkeeping | 5 | D: Acceleration structures (HWTLAS bookkeeping), phase 5 |
| `alignment` | becomes part of `rawalloc`'s constraint | 2 | D: Memory |
| `allocate_batch_queue!` | goes (Metal's uses in phase 9) | 3 | P: Queues, phase 3 |
| `argbytes` | goes: core's bindings! lays out the argument block | 4 | D: Run (argument block) |
| `argidentity` | goes: as argbytes | 4 | D: Run |
| `argsize` | goes: as argbytes | 4 | D: Run |
| `argtype` | goes: `devicearg(dev, a)` is the one conversion | 6 | D: Arrays (device side) |
| `argument_usage` | per-argument access of a kernel | 1 | G2 |
| `awaitwrites` | goes: eager calls wait per sync state, no KA synchronize | 3 | D: Eager work |
| `backend` | goes | 8 | P: Devices (deleted) |
| `basealignment` | core: part of the selection key | 6 | P: phase 6 |
| `batchqueue` | goes (Metal's uses in phase 9) | 3 | P: Queues, phase 3 |
| `begin_pass!` | goes: `render!(dev, …)` | 6 | P: Eager work (Goes) |
| `begin_render_pass!` | goes | 6 | P: Eager work (Goes) |
| `beginframe!` | goes: acquire and present are part of run! | 7 | D: Windows and present |
| `beginrender!` | verb: `beginrender!(b, node, cmd, deps)` | 3 | P: Record |
| `bestshape` | selection table (no definition found at 5c2d4c6) | 6 | G8 |
| `bind_textures` | its per-call descriptor pool goes; textures through the device-wide table | 5 | P: Acceleration structures and textures |
| `blittarget` | goes: image copies are `copyto!` operations (copy nodes) | 6 | D: Graphics |
| `blocksize` | core: the pool's block size | 2 | D: Memory |
| `bufferusage` | becomes part of the memory kind's usage constraint | 2 | D: Memory (Kinds) |
| `build_accel!` | goes: per-device-type signature; core TLAS + `emitbuild!` | 5 | P: phase 5 deletes |
| `callgroup` | goes: launch geometry is one core calculation from KernelInfo | 1 | D: Kernels |
| `capacity` | becomes `budget(dev)`: today's `capacity(dev)` is the device's memory budget (`vulkan/graph.jl:107`, Metal's `recommendedMaxWorkingSetSize`, `AMDGPU.free()`) | 2 | P: Memory, "Capacity means one thing" |
| `caps` | verb: `caps(dev)`; `caps(::KA.Backend)` goes | 8 | P: Devices, What a device type implements |
| `closerecording!` | goes: `finish(b)`; its patch filling goes | 4 | P: phase 4 deletes |
| `closerun!` | goes (Metal's in phase 9) | 3 | P: phase 3 deletes |
| `collect!` | profiling | - | G4 |
| `colorimage` | goes: render targets are images and windows | 6 | D: Graphics |
| `compatible` | core: block compatibility for a kind and constraint | 2 | D: Memory |
| `compile_dispatch` | goes | 1 | P: Kernels, phase 1 |
| `compile_draw` | becomes `compiledraw(dev, pipeline, argtypes, formats)` | 6 | D: Graphics |
| `compiledraw` | verb | 6 | P: What a device type implements, Graphics |
| `constraintof` | core: the constraint of a kind | 2 | D: Memory |
| `copy_target!` | goes: `copyto!` between images and arrays | 6 | D: Graphics |
| `currentimage` | goes: the run holds the acquired swapchain image, whose one owner is the window | 7 | D: Windows and present |
| `defaultdevice!` | goes: `Device()` and the registry | 8 | D: Devices, phase 8 (METAL_DEVICE) |
| `depthimage` | goes: a depth target is a `Float32` image | 6 | D: Graphics |
| `destroyrecording!` | goes: recordings retired by token through the release engine | 3 | D: Record, Memory |
| `Device` | user: abstract type, `Device()` | - | P: Devices |
| `deviceaddress` | core: an array's address is in its cell; AS: `acceladdress` | 4 | D: Arrays (cell) |
| `devicearray` | goes: `MantleArray(dev, data)` | 6 | D: Arrays |
| `devicebuffertype` | goes: `devicearg(dev, a)` | 6 | D: Arrays (device side) |
| `devicecopy!` | copies: `copyto!` as an operation / `emitcopy!`; cross-device open (decision 7) | 6 | G7 |
| `devicename` | core: shown by the device object | 8 | D: Devices |
| `deviceof` | core: `device(x)` | 8 | D: Devices |
| `devices` | user: `devices()` | - | P: Devices |
| `devicesized` | goes: LengthOf resolved at compile | 4 | D: resizing-and-raytracing.jl B.7 |
| `deviceslice` | goes: `devicearg` resolves views in one place | 6 | D: Arrays (device side) |
| `deviceview` | goes: as deviceslice | 6 | D: Arrays (device side) |
| `download` | goes: host reads (`Array(a)`, copy nodes) | 6 | D: Eager work (host reads) |
| `draw!` | user: `draw!(p, shader, args, n)` inside `render!`; on a device it goes | 6 | P: Operations, Eager work |
| `draw_in_pass!` | goes | 6 | P: Eager work (Goes) |
| `draw_indexed_in_pass!` | goes: the draw_in_pass! family | 6 | P: Eager work (Goes) |
| `draw_indirect_in_pass!` | goes: the draw_in_pass! family | 6 | P: Eager work (Goes) |
| `emitbarriers!` | folds into the builder verbs | 3 | P: Record |
| `emitcopy!` | verb: `emitcopy!(b, node, cmd, deps)` | 3 | P: Record |
| `emitdispatch!` | verb (the builder verb takes the name over) | 3 | P: Record |
| `emitdraw!` | verb: `emitdraw!(b, node, cmd)` | 3 | P: Record |
| `emithead!` | folds into the builder verbs (the head is a node) | 3 | P: Record |
| `emitinline!` | folds into the builder verbs: an inline store is `emitstore!` | 3 | P: Record |
| `emitkernel!` | folds into the builder verbs | 3 | P: Record |
| `emitloophead!` | folds into `emitloop!` | 1 | P: Record, phase 1 |
| `emitpreparebarrier!` | folds into the builder verbs (prepare is a node) | 3 | D: Record |
| `emitupdate!` | folds into the builder verbs: a run's stores are its change prefix (`emitcopy!`, `emitstore!`) | 3 | P: Record |
| `end_pass!` | goes: the begin_pass! family | 6 | P: Eager work (Goes) |
| `end_render_pass!` | goes | 6 | P: Eager work (Goes) |
| `endrender!` | verb: `endrender!(b)` | 3 | P: Record |
| `extrausage` | becomes part of the memory kind's usage constraint | 2 | D: Memory (Kinds) |
| `fence` | core: the channel's `next` (the last token handed out), a field read, not a device verb | 3 | D: Core types (`SubmitChannel.next`) |
| `finishrecording!` | goes: `finish(b)` | 3 | D: Record |
| `flush!` | goes as a user-facing verb | 3 | P: Core types |
| `fragmenttouches` | per-argument access of a shader | 1 | G2 |
| `Framebuffer` | goes: render targets are `Image`s and windows | 6 | D: Graphics |
| `gpupasstime!` | profiling | - | G4 |
| `hold!` | goes | 3 | P: Core types, phase 3 |
| `holdleaves!` | goes at emit | 3 | P: phase 3 |
| `hostspan` | core: a host view of host-visible memory (Readback, staging) | 2 | D: Memory |
| `imageusage` | becomes part of the Images kind's usage constraint | 2 | D: Memory (Kinds) |
| `indexbuffer` | goes: indices are a MantleArray bound by `bindindices!` (open decision 17) | 6 | D: resizing-and-raytracing.jl B.7 |
| `indirectindex` | goes: launch commands are slots core lays out | 4 | D: Run (head node) |
| `indirectslot` | goes: as indirectindex | 4 | D: Run (head node) |
| `initbackend!` | goes: hooks take the device | 8 | P: What a device type implements |
| `initial_state` | core: image layouts decided by barriers! | 7 | D: Compile (barriers!) |
| `initial_usage` | core: as initial_state | 7 | D: Compile (barriers!) |
| `intrinsic_usage` | per-argument access of a kernel | 1 | G2 |
| `isdevicearray` | goes: dispatch on MantleArray | 6 | D: Rules (dispatch) |
| `kakernelaccesssignature` | per-argument access of a kernel | 1 | G2 |
| `kernelcompiles` | core: the per-device compile counter | 1 | D: Kernels, phase 1 (the per-device compile counter) |
| `kernelinterpreter` | per-argument access of a kernel | 1 | G2 |
| `kerneltouches` | per-argument access of a kernel | 1 | G2 |
| `kibackend` | goes | 8 | P: Devices (deleted) |
| `layout` | internal to the device type's barrier helper (image layout of a use) | 3 | D: Four device types |
| `librarygemm` | goes: selection table | 6 | P: Operations |
| `makeimage` | becomes `placeimage` behind `Image(dev, T, dims...)` | 6 | D: Graphics, Memory |
| `makeprofiler` | profiling | - | G4 |
| `makerecording` | goes: `Builder(dev, ch)` | 3 | D: Record |
| `materialize!` | goes: transients placed by place! | 1 | D: Compile (place!) |
| `maxalloc` | becomes a field of `caps(dev)`: the largest single allocation, which the pool reads (section 4.1) | 2 | G3 |
| `mergeconstraints` | core: constraints of a kind | 2 | D: Memory |
| `native_attention_dispatch!` | goes: selection table candidates | 6 | P: Operations (native_*_dispatch!) |
| `native_batched_gemm_dispatch!` | goes | 6 | P: Operations |
| `native_conv2d_dispatch!` | goes | 6 | P: Operations |
| `native_gemm_available` | goes | 6 | P: Operations |
| `native_gemm_dispatch!` | goes | 6 | P: Operations |
| `needs_transition` | goes | 10 | P: Four device types |
| `openrecording` | goes | 3 | P: phase 3 |
| `openrun` | goes (Metal's in phase 9) | 3 | P: phase 3 |
| `passbarriers` | goes: barriers! is core's | 1 | D: Compile |
| `passed` | verb: `passed(ch, value)`; core's `passed(::Token)` calls it | 3 | P: Core types |
| `passedas` | goes: argument layout is core's (bindings!) | 4 | D: Run (argument block) |
| `patchable` | goes with the pushed patch mechanism | 4 | D: phase 4 deletes |
| `pool` | core: `pool(dev)` | 2 | P: Memory |
| `present_frame!` | goes: present is part of `submitnative!` | 7 | D: Windows and present |
| `profiled!` | profiling | - | G4 |
| `Profiler` | profiling | - | G4 |
| `rawalloc` | verb | 2 | P: Memory |
| `rawfree` | verb (its separate AS path goes) | 2 | P: Memory, phase 2 |
| `readback_framebuffer` | goes: `Array(img)` | 6 | P: Graphics |
| `readback_window` | goes: `Array(win)` through the present segment | 7 | P: Graphics |
| `record_draw!` | goes as a public path | 6 | P: Eager work (Goes) |
| `recorder` | goes: Builder | 3 | D: Record |
| `recordsplans` | goes | 7 | P: phase 7 |
| `refit_tlas!` | goes: per-device-type signature | 5 | P: phase 5 deletes |
| `release!` | `release!(::SubmitChannel, r)` goes; core's pool release stays internal | 3 | P: Core types |
| `release_batch_queue!` | goes | 3 | P: Queues |
| `remakeimage!` | goes: a new size is a new image under a new index | 5 | D: Graphics, B.9 |
| `reset_device!` | device teardown for debugging; declared, never implemented | 8 | G12 |
| `resetkernelcompiles!` | core: the compile counter | 1 | D: Kernels, phase 1 |
| `resetrecording!` | goes: command objects back to the free list | 3 | D: Record |
| `resourcekind` | core: resource kinds of sync states | 2 | D: Resources |
| `retire!` | core: `retire!(pool, x, tokens)`; `retire!(::SubmitChannel)` goes | 2 | P: Memory, phase 2 |
| `runscalls` | goes | 6 | P: Operations |
| `Sampler` | user: a value in the device-wide sampler table | 6 | P: Graphics |
| `screenshot` | becomes `Array(win)` (name open) | 7 | P: Graphics |
| `set_anyhit_pipeline!` | any-hit is a field of `RayTracingPipeline`; the SBT layout for many hit groups is open | 5 | G1 |
| `set_viewport!` | goes: the begin_pass! family | 6 | P: Eager work (Goes) |
| `setviewport!` | goes: the scene rectangle is a per-draw uniform | 6 | D: Graphics |
| `shadertouches` | per-argument access of a shader | 1 | G2 |
| `staged_gemm_tile` | goes: selection table | 6 | P: Operations |
| `stages` | internal to the device type's barrier helper (pipeline stages of a use) | 3 | D: Four device types |
| `stampof` | goes | 3 | P: Ordering, phase 3 |
| `storage` | core: GPUArrays' `storage(a)` | 6 | P: Arrays |
| `storebytes!` | goes: stores are pending changes (`Store`; inline ones written by `emitstore!`) | 4 | D: resizing-and-raytracing.jl B.3 |
| `submissionbudget` | core: `submissionbudget(dev)` computed from `caps(dev)` facts (job timeout, whether the device drives a display); not a device verb | 1 | P: Submission split |
| `submissionholds!` | goes | 3 | P: Core types |
| `submit!` | core: `submit!(ch, parts; …)`; the device type's part is `submitnative!` | 3 | P: Ordering, internals.jl section 8 |
| `SubmitChannel` | core type; fields change | 3 | P: Core types |
| `submitrecording!` | goes | 1 | P: Submission split (Goes) |
| `supports` | capabilities: fields of `caps(dev)` (no definition found at 5c2d4c6) | 8 | G3 |
| `supports_batch_queue` | goes | 3 | P: Queues |
| `supports_geometry_stage` | capabilities: a field of `caps(dev)` | 8 | G3 |
| `supports_graphics` | capabilities: a field of `caps(dev)` | 8 | G3 |
| `supports_mesh_pipeline` | capabilities: a field of `caps(dev)` | 8 | G3 |
| `supports_procedural_traversal` | capabilities: a field of `caps(dev)` | 8 | G3 |
| `supports_rt_pipeline` | capabilities: a field of `caps(dev)` | 8 | G3 |
| `supports_tessellation` | capabilities: a field of `caps(dev)` | 8 | G3 |
| `supportspredicate` | goes: no conditional execution; launches and draws in a `when!` block are indirect with the flag in their count (section 2.4) | 1 | D: Operations (`when!`), decisions.md review round 2026-10-02 |
| `Surface` | goes: a window is a target itself | 7 | P: Graphics |
| `syncbackend` | goes | 8 | P: Devices (deleted) |
| `target_extent` | goes | 7 | P: Graphics |
| `target_format` | goes: `imageformat(dev, T)` | 7 | P: Graphics |
| `target_image` | goes | 7 | P: Graphics |
| `target_view` | goes | 7 | P: Graphics |
| `Texture2D` | user: `Texture2D(dev, T, w, h)` | 5 | P: resizing-and-raytracing.jl A |
| `timestamps` | goes as a device hook for the split; profiling open | 1 | P: Submission split (Goes), G4 |
| `trace_closest_hits!` | goes: the immediate trace_* verbs | 6 | P: Eager work (Goes) |
| `trace_closest_hits_anyhit!` | goes | 6 | P: Eager work (Goes) |
| `trace_closest_hits_anyhit_indirect!` | goes | 6 | P: Eager work (Goes) |
| `trace_closest_hits_indirect!` | goes | 6 | P: Eager work (Goes) |
| `trace_rays!` | goes: `trace!(dev or g, …)` | 6 | P: Eager work (Goes) |
| `trace_rays_indirect!` | goes | 6 | P: Eager work (Goes) |
| `transition_image!` | goes as a call: image layouts are `barriers!` output (no implementation on either device type) | 6 | P: Graphics |
| `upload!` | goes: stores are pending changes (`Store`; inline ones written by `emitstore!`) | 4 | D: resizing-and-raytracing.jl B.3 |
| `use_bindings!` | goes: bindings compiled with the plan (`compilebindings`), textures through the device-wide table; draw-binding code (`metal/graphics.jl:1524`, `vulkan/graphics/record.jl:169`) whose last caller is the walked draw in `execute!` | 7 | D: Graphics, Windows and present |
| `use_frozen_kernels` | goes: the device type's compiler cache, per device | 8 | D: Threads, phase 8 (FROZEN_*) |
| `vertextouches` | per-argument access of a shader | 1 | G2 |
| `waitfor` | verb: `waitfor(ch, value)`, GC-safe; core's `waitfor(::Token)` calls it | 3 | P: Core types |
| `waitfor!` | `waitfor!(::Stamp)` goes | 3 | P: Core types |
| `waitidle` | device wait-idle | 3 | G5 |
| `Window` | user: `Window(dev, w, h)` | 7 | P: Windows and present |
| `withpredicate` | folds into the builder verbs: a `when!` launch is indirect with the flag in its count | 3 | P: Record |
| `workgroupsize` | goes: KernelInfo's group size | 1 | D: Kernels |
