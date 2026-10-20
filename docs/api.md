# Mantle API: types, verbs and their contracts

This document is Mantle's API after the refactor: every type and verb with its
contract. "The plan" is `recording-plan.md`; "(name open)" and "open" mark what
is not decided yet (section 10).

## 1. Rules every entry obeys

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
| `KA.get_backend(a)` | `device(a)` for an array on a device; the graph's device for an array on a graph, so a KernelAbstractions launch over it declares a kernel node (section 2.2). |
| `using Lavapipe_jll` | Makes lavapipe's ICD visible; must happen before the first `Device()`/`devices()` (the registry creates the Vulkan instance then). An extension. |

### 2.2 Arrays

`MantleArray{T,N,L} <: AbstractGPUArray{T,N}` for every location `L`: a device or
a graph, per device type. A view is the same type with its own
offset and dims, sharing the parent's resource; there is no role parameter.
Read-only, unified and resizable are runtime properties of
the array's resource, not types.

| call | contract |
|---|---|
| `MantleArray(dev, T, dims...)`, `MantleArray(dev, data)` | A pool region on `dev`; operations on it run now. Fixed until its first `resize!`. From data: uploaded at creation (no graph can name it yet), streamed from the caller's memory in chunks through upload regions (each chunk one submission within the budget) or written directly into host-visible storage; returns once every chunk is submitted, later uses wait for them. |
| `MantleArray(dev, host; readonly = true)` | As above, and read-only: keeps a reference to `host`, no copy; eviction drops the device copy and a restore uploads from `host` again. An mmapped `host` (`Mmap.mmap` returns a plain `Array`) lets the OS drop and re-read its pages, so such an array can leave even RAM: this is how device data is backed by a file, and Mantle writes no files itself. Mantle does not know where the bytes come from. A write throws at the call: Mantle's own write methods, and `dispatch!` from the per-argument access it derives at declaration, eager or declared. `host` (or its file) changing afterwards is undefined; nothing checks it. |
| `MantleArray(g, T, dims...)` | A transient of graph `g`: contents exist only during a run, placed in `g`'s arena. Graph arrays are never persistent: what must survive runs and recompiles is a device array the graph names; what the host supplies per run is a device array written by stores. |
| `adapt(Mantle.Device(graph), x)` | `x`'s arrays, or a struct of them (a simulation's state), as zero-copy aliases on the graph's device: the graph names them (with a hold), so code that finds its target through its arrays declares into the graph. `adapt(Mantle.Device(), x)` uploads host data to a device. A `Group` is not a location. |
| `KA.allocate(dev, T, dims...; unified = true)`, `MantleArray(dev, T, dims...; unified = true)` | In unified memory, readable and writable by the host without a copy (KernelInterface's `unified`; `KI.supports_unified(dev)` is true on every Mantle device): a graph writes it, `Array(out)` after the run is a wait and a memcpy, no GPU copy. A one-element `Bool` one is usable as a `when!` flag. |
| `MantleArray(dev, data; streamed = true)` (an open decision) | A weight placed like a transient and uploaded by a copy node before its first use in each run (`examples.jl` 41; plan, Memory, "Streamed weights"). Keyword and upload cost open. |
| `MantleArray(g, T, win)` | Sized like window `win`; placed at the window's capacity; contents undefined after a resize. |
| `MantleArray(g or dev, …; capacity = n)` | Resizable from the start with a reserve for `n` elements, like `sizehint!`: growing past it moves the storage, and plans whose temporaries were sized from it recompile once. Optional for a length a kernel writes: the kernel writes within the array's current storage, bounds-checked unless `@inbounds`; `capacity = n` reserves room for `n` up front. |
| `MantleArray(location(a), T, dims...) do t … end` | A temporary inside an operation: on a device a region retired with the call's token, on a graph a transient. |
| `MantleArray(l, T, SizedBy(f, a))` (internal) | A temporary sized at compile: `f(2 * length(a))` for a host length, `f(capacity(a))` for a kernel-written one. |
| `free!(a)` | Drops the creator's hold, once whatever the number of threads. Later use through any handle throws (eager calls, declarations, stores). Graphs and composites that hold `a` keep using it. Throws for a view. At the last hold the pending changes are dropped and the storage is retired (section 5, "the last hold"). |
| `resize!(a, dims)`, `resize!(a, n)` | Scheduled: the call only enqueues the request (`size(a)` reports the requested size; nothing is allocated, copied or written on the device). The next submission that touches `a` allocates what it needs before its locks and applies it in front of its own work. The first `resize!` call makes a fixed array resizable (plans compiled after it launch over it indirectly); when applied: if the new size fits its current storage, nothing moves (only a cell write); beyond it, a large array on a device with sparse residency moves into a reserve (later growth maps pages, the address stays), and any other array moves into a pool region of twice the need (each later doubling is a move). Which arrays are large, and a reserve's size, core computes from `caps(dev)`. Plans that launched over it directly recompile once. Contents kept as linear memory (`resize!(a, n)` sets the last dimension). Shrinking unmaps nothing. Throws for a view. On a graph array: within the placed size a cell write, beyond it the graph recompiles at its next run. |
| `trim!(a)` (name open) | Pending: unmaps a reserve's pages beyond `a`'s size, retired with the token of the submission that unmaps them; afterwards a view past the size reads undefined values. The only call that unmaps an array's pages: pressure never does (plan, Arrays). |
| `a[r] = data` | A store, scheduled: the call copies the data once into an upload region (host-visible system memory acquired for it, not the array's storage) and enqueues the request with its byte range in the parent (a view's offset applied). The submission that applies it writes it into the storage the array has at that point of the sequence: within the device type's inline limit (`caps(dev)`: 64 KiB at 4-byte offsets and sizes on Vulkan) with `emitstore!`, beyond it by a copy from the upload region (`emitcopy!`); no staging is allocated at apply. A store pended before a resize is copied along with the old contents; one after it writes the new storage. `length(data) == length(r)` or `DimensionMismatch`. A later store covering an earlier pending one replaces it. |
| `copyto!(a, data)` | A store into `a[1:length(data)]`. |
| `copy!(a, data)` | Base's `copy!`: a resize to `length(data)` that keeps no old contents and a store, enqueued together. On a view only with the view's length. |
| `Array(a)`, `copyto!(host, a)` | A host read: its own submission (the pending changes of `a`, a copy into Readback memory), a wait for that token, a memcpy. For a kernel-written length the cell is copied in the same submission. Throws for a graph array. |
| `size(a)`, `length(a)` | The real length, as Julia's array API defines it. For a length a kernel writes: waits for that kernel and reads the length back from its cell; launches never need it, they use the length on the device. During the declaration of a graph that writes it, no run has produced it yet, so it throws. |
| `capacity(a)` | The array's current reserve, like a `Vector`'s after `sizehint!`: `capacity = n` at creation, raised when growth moves the storage. It also sizes temporaries over a kernel-written length (`SizedBy`), which is why plans that sized one from it recompile once after a move (`resizing-and-raytracing.jl` B.2). Not the device's memory budget, which is `budget(dev)` (section 2.8). |
| `launchrange(a)` | The one name for a launch over an array: a reference to `a` resolved at compile, direct while `a` is fixed, indirect after its first `resize!`. There is no `DeviceRange`. |
| `lengthof(a)` (name open) | A kernel argument through which a kernel writes `a`'s length into its cell; declaring that launch adds a prepare node. |
| views, `reshape`, transposes (GPUArrays' `derive`) | Share the parent's state (storage, size, cell, sync state, holds, freed mark). No hold, no finalizer of their own. |
| `KA.get_backend`, `KA.allocate` on a graph array | The graph: a KernelAbstractions launch over graph arrays declares a kernel node and `KA.allocate` gives a transient, so KA libraries and GPUArrays' generic code run inside a graph. `getindex`, iteration and every other host read of contents throw at their line, since nothing exists before a run. Launch sizes such code computes on the host are recorded per node; a resize of an array behind one makes the next compile throw. |
| bounds | Kernels are bounds-checked unless they say `@inbounds`, where an access past the storage is undefined. Nothing clamps a kernel-written length. |
| broadcast, `fill!`, `copyto!`, `map!`, `mapreducedim!`, `accumulate!`, `sort!`, `sortperm!`, `mul!`, `fft`, … | Methods on `MantleArray` that launch through `dispatch!(location(a), …)`: run now on a device, declare on a graph. GPUArrays' generic code serves the rest, on devices and graphs alike. |
| `allowscalar` | GPUArrays' interface, accepted as such; scalar indexing off by default. |

### 2.3 Graphs

| call | contract |
|---|---|
| `Graph(dev; rewrites = (), profile = false, priority)` (keyword names open) | An empty graph on `dev`. Rewriters run only if listed (`FuseEpilogues()`, `MergeRegions(attribute)`, names open). |
| `run!(g)` | The first call compiles, records and runs; later calls run. A run's first segment waits on the GPU for the previous run of `g`. The host waits only with the graph lock and the run's arena-part holds held, never with a sync-state, channel, pool or window lock: for another graph's run that holds a shared arena part, for a host node's inputs, for the swapchain acquire, and in `acquire!`'s pressure steps (plan, Ordering, Run). A `GPURef` store never makes it wait: it is a scheduled change like any other. Returns `Submitted()` or `Skipped()` (a minimized window), both public. Throws for a freed graph, when called from a host function of `g`, and when the calling task holds an arena part of `g`'s plan. |
| `free!(g)` | Releases the plan and drops the graph's holds, once; also done by the graph's finalizer. `run!` and declaring afterwards throw. Called from a host function of `g` it throws before it marks `g` freed, so the finalizer still releases it. |
| `submissions(g)` | Segments and cuts, including those host nodes cause. |
| `timings(g)` | Open (section 10, Profiling). |
| `naivebytes(g)`, `peakbytes(g)` (names open) | Transient bytes without and with placement. |

A graph is never replaced for a structure change.
Declaring into a compiled graph, or removing a group from it, only marks its
plan: the next `run!(g)` compiles a new plan from the graph's nodes and
retires the old one after its last run. Groups of draws are the exception:
the next run records them on their own and recompiles nothing (2.6b). The graph, its windows, the
resources it names and the device arrays it reads stay. A graph array's
contents live only during a run; what must survive runs and recompiles (a
film, an accumulation count) is a device array the graph names. Declaring
takes the graph lock (it waits while a run of `g` has stages left); from a
host function of `g` it throws.

| call | contract |
|---|---|
| `group = Group(g or group, name)` | A value and a place in the order: the end of its parent's order (the graph, or another group) when it is made. Work is declared into it by passing it (`dispatch!(group, …)`, `trace!(group, …)`, copy kernels, `clear!(group, …)`, `draw!(group, …)`, `when!(group, …)`, `Group(group, …)`) anywhere in the code, and runs at the group's place whenever it was declared. It holds GPU work only: a host node, a build or an image copy has no method for a `Group` (a `MethodError`). Not an owner: arrays and allocations belong to the device or the graph, so library code that finds its target through its arrays declares into the graph, not a group. |
| `delete!(g, group)` | Removes the group and everything in it. A group of draws leaves its pass's list at the next run with no recompile (2.6b); any other group marks the plan. A resource no remaining node names loses the graph's hold once the plan that names it has retired. Work that will be removed is declared into a group; nodes declared into the graph itself stay until `free!(g)`. |
| `sort!(group; by, lt, rev)` | Reorders the groups a group holds (`by` sees the groups); its other nodes keep their places. Groups of draws in one pass: the next run rewrites the pass's list, nothing recompiles. Otherwise the plan is marked. |

### 2.4 Declaring work

| call | contract |
|---|---|
| `dispatch!(g or group or location, kernel, args::Tuple, ndrange; group, origin)` | The one launch verb. On a graph it adds a kernel node; on a `Group` a kernel node at the group's place; on a device it compiles from the device type's cache and submits now, with what is pending on its arguments in front. A `group` size the caller gives (here, or KA's `kernel(backend, workgroupsize)`) is kept as given; `config(dev, f)` picks one only where none is given (its default for 2-D and 3-D ranges is open). |
| `location(args...)` | The one graph among the arguments if any lives on one (its device arrays are then named by that graph), else the arguments' device. The arguments are the only way a call reaches a graph: no global, task-local or scoped "current graph" exists (plan, rule 5). Arguments from two graphs or two devices throw. An argument that reaches device memory but is not a `MantleArray` (a vendor array, an array escaping `withview`, a raw backend array) throws: the resource of an argument is the array's state, shared by its views and reshapes, never the object (plan, Arrays). |
| `dispatch!(g, HostCall(f), args; writes::Tuple)` | A host node. `f` runs on the calling thread of `run!`, between the stages before and after it, on host-visible arena memory. It may not use its own graph or a graph that shares an arena part with it (both throw: `lockgraph` throws when the calling task holds an arena part of that graph's plan). If `f` throws, the rest of the run is not submitted, the exception propagates, and an acquired swapchain image goes back to the window for the next run. Has no method for a `Group` and is not a `when!` launch. |
| `repeat!(group, n; while_nonzero = count)` | The group runs `n` times per run at its place; it holds launches only (kernels, traces, copy kernels): a draw, a clear or a group in it throws, at `repeat!` or when declared later (loops do not nest). After a compile it marks the plan; before each iteration a loop-head kernel writes the iteration index (`iteration(group)`, name open) into the slots of the launches that take it. `while_nonzero` is an integer count on the device (a queue's live size) that ends the loop on the GPU. `repeat!(g, kernel, args, ndrange; max, while_nonzero)` is the form for a loop of one launch. |
| `when!(g or group, flag, kernel, args, ndrange)` | One launch that runs only where `flag` is true on the GPU. `flag` is a `GPURef{Bool}` or a one-element `Bool` array, enforced by the method's argument types: anything else is a `MethodError`. The meaning is the same on every device type; only the mechanism is the device type's verb. Vulkan and Metal launch indirect, and whoever writes the indirect command multiplies the count by the flag: the head node for a flag the host writes, a prepare node after the kernel that writes it. CUDA and HIP reach the same result their own way (on CUDA a kernel node whose grid the device sets, or a conditional node; on HIP 7.1 a check at kernel entry). No conditional rendering; the flag is never a handle dependence. Several launches under one flag are several `when!`s. The host reads a flag array after the run like any device array. |
| `trace!(g or group or dev, pipeline, accel, args, ndrange)` | A ray-tracing launch; allowed in a `Group` and as a `when!` launch: indirect, its count zeroed by the loop head when the loop ends and multiplied by the flag under `when!`. |
| `copyto!(out, x)` with `x` on a graph | A copy kernel node (so allowed in a `Group`), recorded once and run every run; how the host reads a graph result. |

Selection is a call in the method: `mul!` and the other library operations
ask the device's selection table at declaration. The table is per device,
keyed on function, element types, shapes, layout and alignment (with
`basealignment`; plan, Operations); an entry not yet measured uses core's cost
estimate. In a `Group` and under `when!` only kernel candidates qualify.

### 2.5 Per-run values

| call | contract |
|---|---|
| `GPURef(dev, v)` | A value box: ONE slot in device memory holding `v`, whose address every plan and group of draws that uses it bakes. Reading it costs what a true uniform costs (within 0.8% on NVIDIA, AMD and the M5). Values only: an array argument is the array itself. |
| `r[] = v` | A scheduled store into the slot, applied by the next submission that touches `r`, like any change; it marks no plan. |
| per-run host data | A device array the graph names, written by stores (`x[r] = data`), scheduled like any change. There are no host-written graph arrays. |

### 2.6 Acceleration structures and textures

| call | contract |
|---|---|
| `BLAS(dev, vertices, faces)` | Storage on a pool region sized from its geometry with a capacity that doubles, like a TLAS. Vertices moved with the same topology: an Update (refit), pending. A new topology: a Build into its storage, pending (beyond the capacity a `MoveAccel`, the new address through its cell). TLASes over it pend an Update when its address or contents change. A geometry length a kernel writes: built over the geometry capacity with degenerate primitives past the length, so later changes are Updates. Its builds and refits are `BuildAccel` changes and, in a plan, `BuildRecording`s, like a TLAS's. Shared by every range over the same geometry arrays. A reader of those arrays: a run that writes them pends its update. |
| `TLAS(dev; hardware)` | One core type for hardware and software. `TLAS(dev)` picks the hardware path where the device has it; `hardware = false` forces software. Storage sized for a capacity starting at 64 that doubles; builds and refits carry the exact count. Pending changes to one structure coalesce: any Build makes a Build, else an Update; every Update uses `builtcount`, the count of the last Build. A total a kernel writes (any device type): built over the capacity with mask-0 records past the total that reference a valid BLAS. An Update before any Build is a Build. A new TLAS pends a Build of zero instances, so a trace before any `push!` reads a built, empty structure. |
| `push!(tlas, blas, transform::Mat4f)` | One instance; returns a range handle. Pending: a range-table store and a Build. A total past the device's `maxInstanceCount` (16 777 215 on every Vulkan GPU here) throws at the call that raises it: `push!`, a `resize!` or `copy!` of a source; the capacity is capped at the limit. |
| `push!(tlas, blas, transforms::MantleArray{Mat4f,1})` | One instance per element. |
| `push!(tlas, blas, positions; scale, rotation)` | One instance per position, transform computed on the device; `scale` and `rotation` arrays or one value. |
| `tlas[h] = t` | Pending: a store into the range's source and an Update (refit). |
| `delete!(tlas, r)` | Pending: the range's entry inactive (`active = false`) and its slot freed, a Build; the range's BLAS and arrays released with the token of the build that deactivated it. |
| `build!(tlas)` | Pending: a full build, applied by the next submission that touches `tlas`. |
| `build!(g, tlas)`, `refit!(g, tlas)` | A build or refit every run. A TLAS whose total a kernel writes holds exactly the instances below the total on every device type: where the device records the count (Vulkan) it is built over its capacity with mask-0 records past the total, so a refit stays legal; where the device reads the count from memory (Metal) it is built with the total from its cell. A refit before any build is a Build. Not a launch: no method for a `Group`, not a `when!` launch. |
| `RayTracingPipeline(; raygen, closest_hit, miss, any_hit)` | A value; compiled through `compiletrace` from the device type's cache, keyed on its shaders and argument types. A new Hikari material type is not a hit group: materials are a `MultiTypeSet` whose GPU form is a tuple of typed arrays, so a new type is a new specialization of the shade kernel, and the plan that declares that kernel recompiles (an open decision). |
| an `Image` a shader samples | Gets an index in the device-wide texture table on its first use by a compiled pipeline or kernel; the index is stable for the image's life. An upload of the same size and type writes the image it has. `resize!(img, w, h)` or an upload of a new size: a new image under a new index and a store into the slot, keeping the texels where the old and new sizes overlap (one GPU copy, as `resize!` keeps an array's contents); the old image retired with the token of the submission that applies the change. Never a recompile. |

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
| `GraphicsPipeline(; vertex, fragment, geometry, blend, cull, topology, depth)` (`Rasterizer`), `MeshPipeline(; mesh, fragment, object, …)` | Values. Stages `VertexShader`, `FragmentShader`, `GeometryShader`, `MeshShader`, `ObjectShader` (tessellation as stage types). Compiled through `compiledraw` from the device type's cache, keyed on stage functions, device-side argument types, fixed state and target formats; equal keys are one pipeline on every device type. |
| `draw!(g or group or dev, target, pipeline, args, count; instances = 1, indices = nothing)` | A draw node into `target`: a `Window`, an `Image`, or a tuple of them (colour images, then one `Float32` depth image). On a device it runs now, as its own pass; an eager draw into a window throws. `count`, `instances`: numbers or `launchrange`s (indirect over resizable arrays). `indices`: a `MantleArray{UInt32}` bound by handle; it never moves within its capacity, and a move marks the parts that recorded the handle, which are recorded again before their next submission; nothing recompiles. |
| `clear!(g or group or dev, target, v)` | A clear node: the load operation of `target` in the pass of the draws that follow it. A target with no clear before its draws keeps its contents; a target no later node reads is discarded at the pass's end (compile decides from liveness). |
| render passes | Formed by compile: clears and draws into the same targets, in order with no node between them that reads what they write, are one render pass. There is no pass object. A window-sized target's segments are recorded again with each swapchain generation. |
| a group of draws | A group whose nodes are draws into one pass is recorded on its own (a piece). Adding one at a place inside a pass with its targets, declaring draws into it, deleting it, or `sort!` of such groups is applied by the graph's next run before its stages: the group's draws are compiled and recorded on their own and the pass's list rewritten; nothing recompiles and no other group is recorded again. A launch, a clear, a draw into the graph itself, or a group that would start a new pass recompiles. A group of draws that fails to compile is deleted and `run!` throws. Visibility is a store into the draw's `when!` flag. `GPURef`s it reads are slots it bakes, like a plan. |
| `when!(g or group, target, flag, pipeline, args, count; instances, indices)` | One conditional draw: indirect, the count written into its command multiplied by the `Bool` flag, as for a `when!` launch (no conditional rendering); visibility is a store into `flag`. |
| `attribute(a, i)` (name open, device side) | `a[min(i, length(a))]` (1-based), the length from the slot: one argument type and one pipeline for a uniform, one value for all elements, or one per element (within 1% of `a[i]` and of a true uniform on NVIDIA, AMD and the M5). |
| scene rectangle | One one-element array per scene, shared by its plots, that the vertex stage applies to each draw (a remap and four clip distances); a layout change is one store. |
| `Image(dev, T, dims...)`, `Image(g, T, dims...)`, `Image(g, T, win)` | An image resource: sync state, holds, stores (pending uploads), `copyto!` with arrays (pending changes or copy nodes), `Array(img)`. Format from `T`, one table per device type (`imageformat`). Layouts are decided by `barriers!`. `Image(g, T, win)` is recreated per swapchain generation at the actual size on the same arena memory. |
| `Sampler(; filter, address, anisotropy)` | A value; a device-wide sampler table entry created on first use. |
| `Array(win)` (name open) | A screenshot: the next run presenting to the window copies the image into `Readback` memory before presenting; waits for that run. Throws `ArgumentError` when the window is minimized, when no graph presents to it, or from the presenting graph's own host function; a call already waiting is woken with that error when the graph's next run is skipped or the graph is freed. |

### 2.7 Windows

| call | contract |
|---|---|
| `Window(dev, w, h; title, vsync, capacity)` | A graph resource; its swapchain (Vulkan) or drawables (Metal) is its storage; the device type's window open verb makes the window and its surface (section 4.7). Capacity: the largest monitor's size unless given. Written by exactly one graph, the presenting one: any of its nodes may write the window (a draw, a clear, a kernel, a copy), the first graph declaring a write presents, and a second graph declaring one throws at declaration (other graphs render into `Image`s that graph reads). The binding ends when the graph is freed or collected or the nodes writing the window are removed; another graph may then present. An eager draw into a window throws; eager draws into images are allowed. There is no `Surface` and no `target_*` accessor. The default colour format and the `vsync` default are open (section 10). |
| `resize!(win, w, h)` | Called from the window's event: stores the size and marks the graphs that render to it; nothing else happens on the event thread. The next run of such a graph makes the new swapchain generation, recreates its window-sized images at the actual size on the same arena memory (the arena part is sparse where possible and maps only the pages the size needs), and records again the segments that draw into window-sized targets and the per-image present segments. Beyond the capacity: the graph recompiles at its next run, like a graph array beyond the size its slot was placed for; it never throws. |
| an acquired swapchain image | One owner, the window. An image acquired by a run that returns `Again` or throws goes back to the window; only real images are kept. |

### 2.8 Memory introspection (names open)

`reserved(dev)`, `mapped(dev)`, `budget(dev)`.

### 2.9 Vendor views

`withview(f, CuArray | MtlArray | ROCArray, a, stream)` (name open; an open
decision): a zero-copy view of `a` in the vendor's type, inside the block
only. It takes a hold on `a` and applies `a`'s pending changes and any
restore first, in a submission the stream waits on; the stream also waits for
Mantle's other accesses to `a`. Mantle's later uses wait for the stream's
work, also if the block throws, and the hold is dropped at the end with the
stream's token. While lent, every Mantle call that locks `a`'s record throws.
No Mantle lock is held while `f` runs. Core's side uses the foreign channel
(`foreignchannel`, `streamwaits!`, `signal!`; internals.jl section 14).

## 3. Core types

Internal; listed because the verbs in section 4 receive or return them.

| type | what it is | lives in | protected by |
|---|---|---|---|
| `Device` | abstract; one object per hardware device and driver | the registry | registry lock (opening) |
| `DeviceRegistry`, `REGISTRY` | loaded device types, opened devices, the Vulkan instance | process | its lock |
| `Graph` | nodes as declared, named resources, rewrites, plan; the flags `recompile`, `windowchanged`, `freed` | user | `Base.Semaphore(1)` (not reentrant; owner task checked); the flags are atomics (section 5) |
| `GraphDevice` | what graph arrays point at (a weak reference to the graph) | graph | immutable |
| `Plan` | compiled state: stages, placement, argument block, tables, recordings, `lastrun`; the flags `stale`, `unmapped`, `restore`, `running`; `compiles`, the number of compiles of the graph so far, carried from each plan to the one that replaces it (for debugging and tests) | graph | graph lock; the flags are atomics (section 5) |
| `CompileState` | the plan under construction; a copy of the nodes | compile | graph lock |
| `Node` subtypes | `KernelNode`, `TraceNode`, `ClearNode`, `DrawNode`, `Group`, `HostNode`, `HeadNode`, `PrepareNode`, `CopyNode`, `LoopNode` (a repeated group, made by compile), `RenderNode` (a pass, formed by compile), `BuildNode`, `ForeignNode`, `RegionNode` (a run of nodes one region compiler compiles), `VideoDecodeNode`, `PresentNode` | plan (its own copies) | graph lock |
| `CompiledNode` | queue, predecessors, barrier, command, cost, slots | node | graph lock |
| `Origin` | the Julia call that launched a node | node | immutable |
| `Kernel`, `DeviceKernel`, `KernelInfo` | one kernel: its function, the access per argument (declaration), its code (compile), and per device the native handle and `KernelInfo` (first use) | attached to the kernel's `CodeInstance` (`GPUCompiler.cached_results`), one per compiler configuration | the kernel's own lock; each field written once |
| `Bound{R}` | a plan's argument slots for a resource | plan | node uses |
| `MantleArray{T,N,L}` | location; the resource (state, runtime properties); a view's offset and dims | user | — |
| `ArrayState` | storage, dims, length source, capacity, cell, sync state, holds, dependents, readers; the freed mark is on the sync state only | shared by an array and its views | its sync state's lock |
| `Cell` | an array's address and dims in the device's cell arena | array state | written only by applied changes or by a kernel through `lengthof` |
| `Region`, `Reserve`, `Committed`, `Evicted`, `Released` | the storage kinds: a pool region; a sparse reserve with mapped pages; a pool region of twice the need that moves on each doubling; the host copy of an evicted array; nothing, after the last hold | array state | the array's sync state (host view); the pool lock for blocks |
| `Arena` | one arena part: range, mapped bytes, tenants, sync state, holder | pool | its sync state (mapping, tenants); its condition (holder) |
| `SyncState`, `Access` | per shared resource: the accesses of what is not a run (eager calls, host reads, vendor views, eviction copies, applied change steps: last write and reads since), the last tokens of plans retired while naming it, pending changes, lent and freed marks. Runs are not recorded here (their last-run tokens, found through the plans naming the resource) | every shared resource | its `ReentrantLock` (never held while `acquire!` runs) |
| `namers` (on `SyncState`), plan inbox, plan sync lock | per resource, the plans naming it with their use from compile (an immutable snapshot, replaced at registration and retirement; composite members reach the plans naming the composite through their readers); per plan, the resources written by other runs, changed or accessed eagerly since its last look (a lock-free list), and the lock that orders its submissions against the plans it shares resources with | the resource; the plan | the resource's sync state lock (`namers`); the plan's sync lock (the inbox is lock-free) |
| `Token` | `(channel, value)`: a value a channel signals. Core's `passed(t::Token)` and `waitfor(t::Token)` call the device-type verbs `passed(ch, value)` and `waitfor(ch, value)` (section 4.2) | accesses, `lastrun`, retirements | immutable |
| `Change` subtypes | `Store` (the storage it writes, a byte range in the parent, its upload region acquired at the call; written inline within the `caps(dev)` limit, else copied), `CellWrite`, `BindPages`, `MoveStorage`, `BuildAccel`, `MoveAccel`, `Restore`, each with a device-wide sequence number | sync state | sync state lock |
| `SubmitChannel{N,T}` | native queue + timeline, kind, lock, `next` (the last token handed out), outstanding submissions | device | its lock (submit, present, sparse binding, wait-idle) |
| `Recording{N}` | one recorded part of a segment, written once. A segment is a list of parts: recordings and `BuildRecording`s | plan | — |
| `Submission{T}` | a token and the command objects to recycle once it passed | channel | channel lock |
| `Builder` | a device type's recording in progress; a value that never escapes the call that made it (`record!`, an eager call, a submission's change prefix, the re-recording of a `BuildRecording` or a window-sized segment) | stack | — |
| `Deps` | the predecessor nodes and the derived barrier of a node | compiled node | immutable |
| `CompiledDispatch`, `CompiledTrace`, `CompiledCopy`, `CompiledBuild`, `CompiledRender`, `CompiledDraw`, `CompiledLibrary`, `CompiledLoop`, `CompiledHead` | what compile decided for one command | compiled node | immutable |
| `TLAS{P}`, `BLAS`, `InstanceRange`, `RangeEntry` (`mask`, `active`), `MemberSet` | resizing-and-raytracing.jl B.8; `builtcount`, the count of the last Build, which every Update uses | user | the structure's sync state |
| `BuildRecording` | a plan's TLAS or BLAS build or refit in its own command buffer, on every device type (Vulkan records the count, Metal the storage). A change that alters what it recorded, or a Build that sets a new `builtcount`, marks it; the run re-records the marked ones after the change prefix (nothing is detected at run) | segment parts | the structure's sync state, held by the run |
| `Window`, swapchain generation | size, pending size, current generation with holds; the swapchain images acquired and not yet presented (the window is their one owner) | user | the window's lock |
| `GPURef` | a value box: one slot in device memory | user | the slot's sync state |
| texture table, sampler table | the device-wide descriptor tables, with free lists | device | the table's lock, a leaf (section 5) |
| `Pool`, `Inbox` | blocks per kind, retiring list, eviction list; lock-free inbox for finalizers | device | pool lock (a leaf); none (inbox) |

The access per argument of a kernel or shader is derived at declaration,
before any compile: from a declaration next to the function for a call with
no Julia body, else by the taint walk over the kernel's typed Julia IR through
the device type's interpreter (`argument_usage`, `kerneltouches`,
`shadertouches`, `vertextouches`, `fragmenttouches`, `kernelinterpreter`,
`kakernelaccesssignature`, `intrinsic_usage`). It is stored in the `Kernel` at
its `CodeInstance`, not in `KernelInfo`. `dependencies!`, `barriers!` and the
read-only check (section 2.2) read it.

## 4. Device-type verbs

The only functions a device type (`src/vulkan`, `src/metal`, `ext/`)
implements, at the signatures core calls. For every verb: it dispatches on
the device (or an object of it) as its first argument; it takes no Mantle
lock unless stated; it does not call into core except through the functions
named here; it throws only on device loss or a driver failure, never to
signal "not supported" (capabilities are answered by `caps`).

### 4.1 Device

| verb | called by, when | contract |
|---|---|---|
| `hardwarekeys(T, registry)` (name open) | `devices()`, under the registry lock | The keys of the hardware devices this device type can open. Creates no driver objects beyond enumeration. |
| `T(key)` | the registry, once per key | The device object; driver objects are created on first use, not here. |
| `caps(dev)` | core, at compile and at declaration | An immutable record of facts the device type reports; core decides from them (field names open). Fields: `sparseresidency`; `accelbyaddress`; the device's limits; the largest single allocation; the sparse page size and the device's sparse address space (core's `reservesize` and `sparsethreshold`); the driver's job timeout and whether the device drives a display (core's submission budget); the roofline inputs of `costs!` and `schedule!`: throughput per element type, memory bandwidth, whether a transfer queue exists, the cost of a wait between queues (internals.jl section 4: `caps(dev).throughput[T]`, `.bandwidth`, `.transfer`, `.crossqueuecost`); the inline store limit and alignment (`.inlinestore`, `.inlinealign`: 65536 and 4 on Vulkan, `vkCmdUpdateBuffer`); the capability flags for geometry stages, graphics, mesh pipelines, procedural traversal, ray-tracing pipelines and tessellation (Appendix A, `supports_*`). |
| `config(dev, f)` (name open) | `kernel!`, before `compilekernel` | The compile configuration of kernel `f` on `dev` (internals.jl section 4; contents open). |
| `countsource(dev)` | core, at record and run | `CountRecorded()` (Vulkan) or `CountFromMemory()` (Metal): where a TLAS or BLAS build gets its count. |
| `queuekinds(dev)` (name open) | core's queue policy | The queue kinds the device has (graphics+compute, compute, transfer, sparse binding, video decode, present). |
| `budget(dev)` | `acquire!` | The device's memory budget minus other processes' usage (`VK_EXT_memory_budget`, Metal's recommended working set, `cuMemGetInfo`/`hipMemGetInfo`). One meaning. |
| `instancerecordtype(dev)` | the TLAS records kernel | `VkAccelerationStructureInstanceKHR` or `MTLIndirectAccelerationStructureInstanceDescriptor`. |

Not device-type verbs: `submissionbudget(dev)` (the GPU time one submission
may take), `sparsethreshold(dev)` (the bytes below which a resizable array
doubles in the pool instead of moving into a reserve) and `reservesize(dev,
need)` (the address space of one reserve) are core functions computed from
`caps(dev)` fields; a device type reports facts and core decides (plan,
"Submission split: estimated", Arrays; rule 1).

### 4.2 Channels and submission

| verb | called by, when | contract |
|---|---|---|
| `submitnative!(ch, token, parts, waits, signals, present)` | core's `submit!`, under the channel lock, after the token is claimed and the accesses recorded | Submits `parts` in order; waits on `waits` (tokens of any channel, expressed natively: a timeline wait, `waitForEvent:value:`, stream order or a stream value wait); signals `token` on `ch` and `signals`; presents `present` if given (binary acquire and rendered semaphores on Vulkan). Must not block on the GPU, take a Mantle lock, allocate, or throw except on device loss: nothing fallible happens between claiming a token and submitting. The only call site of `vkQueueSubmit2` and `vkQueuePresentKHR`. |
| `passed(ch, value)` | core's `passed(::Token)` (release engine, waits) | Whether the GPU signalled `value` on `ch`. Non-blocking. |
| `waitfor(ch, value)` | core's `waitfor(::Token)` (host reads, host nodes, pressure) | Blocks until `value` passed on `ch`. A GC-safe call (`@ccall gc_safe = true`). Never called with a sync-state, channel, pool or window lock held. |
| `recycle!(ch, objects)` (contract open) | the release engine, once a submission's token passed | Returns an eager call's command objects to the device's free list. On Metal the free list holds allocator and command-buffer pairs (plan, Metal, "Channels"). |
| `bindpages!(ch, token, binds, waits)` | core's `sparsesubmit!`, under the channel lock, the token claimed | A queue operation binding pages (Vulkan and Metal through the device type's one sparse-binding helper, CUDA and HIP `cuMemMap`/`hipMemMap`), waiting on `waits`, signalling `token`. Same no-fail rule as `submitnative!`. |
| `unbindpages!(ch, token, unbinds, waits)` | as `bindpages!` (also for `trim!(a)` and idle arena pages) | The same, unbinding. |

A `Token` is `(channel, value)` (section 3); core's `passed(t::Token) =
passed(t.channel, t.value)` and `waitfor(t::Token) = waitfor(t.channel,
t.value)`. The last value handed out on a channel is core data
(`SubmitChannel.next`), a field read, not a verb.

### 4.3 Memory

| verb | called by, when | contract |
|---|---|---|
| `rawalloc(dev, kind, bytes, constraint)` | the pool's `tryacquire!`, under the pool lock | One block of `kind` (`Buffers`, `Unified`, `Readback`, `Images`, `Accel`) meeting `constraint` (alignment, usage). A driver call, not a GPU wait; runs no finalizer. The only call site of `vkAllocateMemory` (and of `cuMemCreate`/`hipMemCreate` for blocks). |
| `rawfree(dev, block)` | the release engine | Frees a block whose last use passed. |
| `createreserve(dev, bytes)` | core's growth decision, before any lock | A `Reserve`: sparse buffer, placement-sparse buffer (added to the device's residency set), or reserved address range; no pages mapped. `bytes` is core's `reservesize`. |
| `placeimage(dev, region, …)` (contract open) | the pool | An image on a pool region. |
| `placeaccel(dev, path, region)` | the TLAS/BLAS code | An acceleration structure object on a pool region (never a reserve: VUID-VkAccelerationStructureCreateInfoKHR-buffer-03615). |
| `accelsizes(dev, path, n)` | the TLAS/BLAS code | Storage and scratch bytes for `n` instances or a geometry. |
| `acceladdress(dev, as)` | the TLAS code (cell write) | `vkGetAccelerationStructureDeviceAddressKHR` or the Metal resource ID. |
| counted objects (names open) | the release engine | The bytes of a driver object the API cannot place (ICB, pipeline, descriptor pool, swapchain image), and its destruction. |

### 4.4 Compile

| verb | called by, when | contract |
|---|---|---|
| `compilekernel(dev, kernel, config) -> (native, compiled::Bool)` | `kernel!` at declaration (eager) and at compile, with no lock held other than the graph lock (compile runs under it) and the kernel's own | Fills the kernel's code if it is missing and its entry for `dev`; the `Kernel` is the only store, no device-type cache. `compiled` says whether this call compiled (core's per-device counter adds it up). |
| `kernelinfo(dev, native) -> KernelInfo` | `kernel!` | No optional fields: registers, shared memory, group size limits and occupancy (from the API, else the device's limits), instruction and memory-operation counts of the compiled IR. Stored in the kernel's entry for the device; the access per argument is not part of it. |
| `compiledraw(dev, pipeline, argtypes, formats)` | `compile!` of a render node | A pipeline from the device type's cache, keyed on the stage functions, `argtypes` (device side), the fixed state and the target `formats`; the same key gives the same pipeline. |
| `compiletrace(dev, pipeline, argtypes)` | `compile!` of a trace node | A ray-tracing pipeline and its shader binding table layout from the device type's cache, keyed on the shaders and `argtypes`. |
| `compilebindings(dev, s)` | `bindings!` | Descriptor objects for the layout core decided; nothing else. |
| `candidates(dev, f)` | `select` at declaration | The candidates the device type contributes for `f`, with their attributes (`implements`, region compiler). |

### 4.5 Record

| verb | contract |
|---|---|
| `Builder(dev, ch)` | Command objects from the device's free list. A value that never escapes the call that made it: `record!`, an eager call, a submission's change prefix, or the re-recording of a `BuildRecording` or a window-sized segment. |
| `emitdispatch!(b, node, cmd::CompiledDispatch, deps)` | The barrier in `deps`, then the launch, direct or indirect as compile decided (a `when!` launch is indirect; the flag is in its count, not in a condition). |
| `emittrace!(b, node, cmd::CompiledTrace, deps)` | As `emitdispatch!`. |
| `emitcopy!(b, node, cmd::CompiledCopy, deps)` | As `emitdispatch!`. Also a change's copy (`node = nothing`, `deps = Deps((), ())`): an upload region to storage, a cell copy, a move, a restore. |
| `emitstore!(b, dst, upload)` | An inline store within the `caps(dev)` limit in a change prefix: the bytes of the store's upload region (written at the store's call) written into `dst` inside the submission. Vulkan: `vkCmdUpdateBuffer` (at most 65536 bytes per call). Device types without an inline store report a limit of 0 and get a copy from the upload region (`emitcopy!`). Allocates nothing. |
| `emitbuild!(b, node, cmd::CompiledBuild, deps)` | A TLAS or BLAS Build or Update with the count in `cmd` (Vulkan: recorded, `builtcount` for an Update) or from a cell (Metal). |
| `beginrender!(b, node, cmd::CompiledRender, deps)`, `emitpieces!(b, node, natives)`, `endrender!(b)` | A render pass: its barrier (target layouts, and the fixed barrier from compute, copy and build writes to vertex, index, indirect and fragment reads), then its pieces in list order (Vulkan: one `vkCmdExecuteCommands`, the pass begun with `VK_RENDERING_CONTENTS_SECONDARY_COMMAND_BUFFERS_BIT`; Metal: `executeCommandsInBuffer` per piece). |
| `recordpiece(dev, node, draws, region) -> native` | One piece's draws with no barrier, binding its own pipelines and setting viewport and scissor itself (no state is inherited, `pipelines.adoc:131-133`). Vulkan: a secondary command buffer with `VkCommandBufferInheritanceRenderingInfo` for the pass's formats; Metal: an indirect command buffer. Uses `emitdraw!(b, node, cmd::CompiledDraw)` per draw. Allocates nothing: its region was acquired before. |
| `bindindices!(b, node, storage)` | An index-buffer bind inside a render pass, by handle. |
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
| vendor view (name open) | A zero-copy `CuArray`/`MtlArray`/`ROCArray` over a region, not owning it. |

### 4.7 Windows

| verb | contract |
|---|---|
| window open verb (name open) | Makes the native window and its surface (Vulkan) or layer (Metal) for `Window(dev, w, h; …)`; on the main thread (GLFW). |
| `createswapchain(w, width, height; oldswapchain)` | A new swapchain generation. Called by the presenting graph's next run after a resize (`applywindows!`), under the window lock. |
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
| a region, an upload region, a plan, a recording | `acquire!` (an upload region at the store's call) / compile / record | `retire!(pool, x, tokens)`, released when the tokens passed | the inbox (lock-free) then the pool lock |
| an array's storage, dims, pending changes | constructor, `resize!`, stores | the last hold (below) | the array's sync state |
| an array's holds | the creator, each graph that names the array, each composite that holds it (a TLAS or BLAS range) | `free!`, `free!(g)`, `free!` of the composite; finalizers of `TLAS`, `BLAS` and render objects push their hold drops through the pool's inbox | atomic compare-and-swap; never 0 → 1 |
| the last hold | the holder that drops it | under the array's record lock: pending changes dropped (their upload regions retired), storage swapped for `Released()`, the old storage retired with the last tokens, the array taken off the eviction list, the cell released | the array's sync state |
| a cell | the array constructor | at the last hold | written only by applied changes and `lengthof` kernels |
| graph nodes, plan | declaration, compile | `free!(g)` / finalizer | graph semaphore; one declaring task |
| a `GPURef`'s slot | `GPURef(dev, v)` | its last hold | the slot's sync state; written only by applied stores (`r[] = v`, scheduled) |
| tokens, outstanding submissions | `submit!`, `sparsesubmit!` | `recycle!` once passed | channel lock |
| an arena's mapping and tenants | compile, `bindunmapped!`, `unmapidle!` | `free!(g)` (untenant) | the arena's sync state |
| an arena part's holder | `holdparts!` at the start of a run | `releaseparts!` after its last stage | the arena's condition |
| a window's size, swapchain generation and acquired images | `resize!(win)` / `applywindows!`; the run's acquire | holds of the window and of runs that acquired; an image goes back to the window when its run returns `Again` or throws | window lock |
| a TLAS's or BLAS's slots, members, storage, scratch, capacity, `builtcount` | `push!`, `delete!`, `resize!` of a source, geometry changes | `free!(t)` | the structure's sync state (`change!`) |
| a `BuildRecording`'s command buffer and its re-record mark | the run (marked ones); the change that alters what it recorded (the mark) | retired with its last token when replaced | the structure's sync state, held by the run |
| compiled kernels and pipelines | the `Kernel` at its `CodeInstance`: code at compile, a device's entry at its first use | Julia's code cache, with the code; a device's entries are never removed (a reset makes a new device object) | the kernel's own lock |
| texture and sampler table entries | an `Image` on its first use by a shader, a `Sampler` on first use (`tableentry!`, `samplerentry!`) | the image's last hold (the index back to the free list with its last tokens) | the table's lock (a leaf) |
| an array's readers | the composites built from it: TLAS, BLAS, a screen (a redraw mark) | the reader's `delete!` or `free!` | an immutable snapshot, replaced under the array's lock and counted per reader |
| an evicted resource's host copy | eviction under pressure | `Restore` (after its upload's token), `free!` or the last hold | the resource's sync state |
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
| `restore` | plan | eviction of a resource it names; `restore!` of an already-evicted array with no prepared storage | the run that restores |
| `running` | plan | the run, before its first submission | the run, after its last stage |

Changes to an array with readers lock the array and its readers together
(`lockexpanded`) and pend each reader's update; the capacity a new total needs
is allocated by the submission that applies the build (scheduled, like every
change). A run pends `pendupdate!(reader, array)` after it submitted each
stage. Between graphs, a plan naming a composite is among its members' plans,
through their readers: a stage writing a resource waits for the complete last
runs of the other plans naming it, a stage reading one from its inbox for
those of its writers. Submitters that are not runs expand a composite to its
members, transitively (TLAS to its BLASes and range arrays, BLAS to its
geometry), and wait for and record the expanded
uses.

Lock order, everywhere: graph lock → the run's hold on its arena parts, by
part id → window lock → plans' sync locks by plan id (a stage: its own,
those of the plans sharing what it writes and of the plans naming its inbox
resources; anything else: those of the plans naming the resources it
touches) → sync states by
resource id (an array and its readers together, through `lockexpanded`) →
channel lock. The pool lock, the
inbox, the texture and sampler tables' locks and a
plan's `runend` lock are leaves (`runend` guards the plan's `between` flag,
set under stage 1's record locks and cleared when the last stage is
submitted; a change that would make the plan stale waits on it, holding no
lock). `acquire!` is never called with a sync-state, channel or pool
lock held. No lock is held while vendor code runs.

Checks that keep the order: `lockgraph(g)` throws when the calling task holds
an arena part of `g`'s plan; `holdparts!` throws when the task would wait for
a part while holding one with a higher or equal id. The host waits (section 2.3,
`run!`) only with the graph lock and the run's arena-part holds held, never with
a sync-state, channel, pool or window lock.

## 6. Invariants and the tests that check them

| invariant | checked by |
|---|---|
| A device type defines only section 4's verbs, at their signatures | the vocabulary check |
| Each driver entry point is called from exactly one function (section 8) | the driver-call check |
| No mutable globals outside section 9's allowlist | the globals check |
| No probes outside section 9's allowlist; `@static Sys.is*` only in `Mantle.jl`'s include block | the probe check |
| Core passes Aqua and JET | the Aqua and JET check |
| Every function takes its device; two devices open for the whole suite never mix | `test_device_is_an_argument.jl`; the two-devices check |
| One path: every eager entry reaches `compilekernel`, one submission per launch | `test_one_path.jl` |
| Changes are pending until the next use; every STAGE sees the old or the new state, never a mix (a run with host nodes is one submission per stage, so a change arriving between stages reaches the later ones) | `test_resize_is_queued.jl` |
| Runs of one graph never overlap (plan, rule 9): a run's first segment waits on the GPU for the previous run's last token on each channel | `test_runs_never_overlap.jl` (negative control: the wait removed) |
| Record is pure | `test_record_is_pure.jl` |
| A plan is recorded once; in steady state `run!` allocates, records and compiles nothing. Re-recorded are only the `BuildRecording`s a change marked; the parts that recorded the handle of a moved array; a new or changed group of draws and the segment holding its pass's list; and, per swapchain generation, the segments drawing into window-sized targets and the per-image present segments | `test_recorded_once.jl`, `test_run_is_submit.jl` |
| A builder is a value | `test_builder_is_a_value.jl` |
| Memory is released only after the tokens of its last uses passed | `test_lifetime.jl`, `test_memory_returns.jl` |
| A write through an array and a read through a view, `reshape` or storage of it in one graph get a barrier: one resource per array state, not per object | `test_view_barriers.jl` (negative control: an object-keyed resource map, which finds three resources and no barrier) |
| Cross-graph order through namers and inboxes: a stage that writes a resource waits for the complete last runs of the other plans naming it, a stage that reads one from its inbox for those of its writers | `test_crossplan_order.jl` (negative control: `ordering = Unordered()`) |
| Any thread compiles, records, runs, frees | `test_any_thread.jl`, `test_interleaved_recording.jl` |
| Host nodes: stages, re-entry throws, a throwing host function leaves the device usable | `test_host_node.jl` |
| A window resize costs no device-wide wait and no recompile within capacity; beyond the capacity the graph recompiles at its next run | `test_window_resize.jl` |
| Launch kernels never read cells (except head, prepare, the TLAS offsets and records kernels, `lengthof` writers) | a check over every plan the suite compiles that only those nodes have a `CellRead` use (test name open) |
| Composites cannot free early, leak or free twice | `test_composite_fuzz.jl`: `push!`, `delete!`, `resize!`, plot updates interleaved with runs on several threads; validation clean; reserved memory back at baseline after `free!` |
| Two graphs that each need 90% of the device run alternately | `test_eviction.jl`: back to back with no host wait, restore before the run, results exact, no recompile; composite members and handle dependents never evicted |
| A RayMakie screen does not recompile when plots are added or removed | RayMakie's compile-counter session: plot inserts and deletes (groups of draws), attribute changes, value ↔ per-element, another device array for an attribute, visibility, camera, layout, resizes within capacity, TLAS moves and count changes; zero compiles. Insert time per plot measured at 100, 1 000 and 5 000 plots must not grow with the plot count; a visibility change records nothing |
| A frame does no host walk over render objects; a run's host time does not grow with the number of resources it names, only with the named resources its stages write and its inbox | `test_run_host_time.jl` (plans naming 100, 2 500, 25 000 and 50 000 resources with nothing changed take the same host time per run) |
| Nothing fallible between claiming a token and submitting | a review item, not a test: the review of `submitnative!`, `bindpages!`, `unbindpages!` and of core's `submit!` in every change that touches them |

## 7. Corner cases

- `resize!` or `free!` of a view throws.
- Declaring into a compiled graph marks its plan for recompile; declaring into or running a freed graph throws.
- A host function that runs, frees or writes into its own graph, or runs a graph sharing an arena part with it, throws: `lockgraph(g)` throws when the calling task holds an arena part of `g`'s plan, and `holdparts!` throws when the task would wait for a part while holding one with a higher or equal id.
- `free!(g)` from `g`'s own host function throws before it marks `g` freed (the re-entry check comes before the swap of `freed`), so the graph's finalizer still releases the plan and the holds.
- Arguments from two graphs or two devices throw at `location`.
- An argument that reaches device memory but is not a `MantleArray` (a vendor array, an array escaping `withview`, a raw backend array) throws at declaration instead of becoming a resource of its own.
- A caller's group size (`dispatch!(…; group)`, KA's `workgroupsize`) is kept as given, also for 2-D and 3-D ranges.
- A device-decided length needs no `capacity`: the kernel writes within the array's current storage, bounds-checked unless `@inbounds`.
- `getindex`, iteration, `Array(x)` and every other host read of a graph array's contents throw; `KA.get_backend` and `KA.allocate` declare into the graph.
- A build, image copy or host node has no method for a `Group`; `when!` takes one launch or one draw with a `Bool` flag, anything else is a `MethodError`; `repeat!` of a group holding a draw, a clear or a group throws (loops do not nest).
- A write to a `readonly` array throws at the call, eager or declared.
- `length(a)` of a kernel-written length during the declaration of the graph that writes it throws.
- `refit!(g, t)` for a TLAS whose total a kernel writes is legal on every device type: on Vulkan the structure is built over its capacity; whether Metal can refit after the total changed is open, and it builds where it cannot.
- A store whose data length differs from the range throws `DimensionMismatch`.
- A call on an array lent to vendor code throws.
- `free!(a)` while graphs hold `a`: the graphs keep running; `a`'s handle throws.
- A window with a zero extent in either dimension (minimized) skips the whole run: nothing is submitted and the changes stay pending; OUT_OF_DATE is a resize to the surface's size, applied and retried by the same `run!`; SUBOPTIMAL presents and marks.
- A window resized beyond its capacity: the graph recompiles at its next run (never a throw).
- A run that returns `Again` or throws after acquiring a swapchain image gives the image back to the window, its one owner; the next run presents it.
- An allocation beyond the budget throws `OutOfDeviceMemory` only after the pressure steps: unmap idle arena pages, trim empty blocks, collect garbage, wait for retiring storage, then evict least recently used idle resources and wait, outside every lock, for the tokens their storage was retired with (the accesses in flight and the eviction copies). Pressure never unmaps pages beyond an array's size and never compacts acceleration-structure storage (`trim!(a)` is the explicit call).
- A move of storage a plan recorded by handle re-records the parts that recorded the handle before their next submission; nothing recompiles and nothing waits.
- A change that would make a running plan stale (a length it launches over directly, a temporary it sized outgrown, a capacity change of a structure it sized temporaries from) is applied only after the run has submitted its last stage: the submitter that would apply it waits on the host, never the call (only a run past its first stage is waited for; no fairness: it may wait again if the plan's next run gets past its first stage first). From inside any run (any host function) it throws: it could wait in a cycle. No submitter applies it in between.
- `copy!(a, data)` with a new length does not copy the old contents.
- Eviction: evictable is a resource no composite holds, with no handle dependent, that no running plan names, not lent, with nothing pending; accesses still in flight are allowed (the eviction copy waits for the last write; the storage is retired with the tokens of every access in flight and the copy's). A structure's own arrays are held by it and never evicted. Acceleration structures are not evicted. Eviction works the same on every device type, unified-memory devices included (the Apple M5, the Ryzen AI Max+ 395, NVIDIA's unified-memory chips): the device allocation is freed and the copy goes to pageable host memory, which the OS may compress or swap to disk (how a device type writes pageable memory is open, plan, open decision 14). A read-only array copies nothing out. A host read of an evicted array reads the host copy; `free!` of an evicted resource frees the host copy; a plan that names an already-evicted array with no prepared storage is marked for restore and its run returns `Again`.
- A plot with zero elements keeps its render object; its draw count is zero.
- A pipeline key change of a drawn plot (another shader, another attribute element type) is a `delete!` and `insert!` of the plot in Makie: its group deleted and a new group of draws, no recompile. A new Hikari material type is a new specialization of the shade kernel the integrator declares: that plan recompiles (an open decision).
- An index array that moves (beyond its capacity) marks the parts that recorded its handle; they are recorded again before their next submission, nothing recompiles.

## 8. Driver entry points in one place

Each driver entry point is called from exactly one function, per device type
(plan, "Checks that fail when a rule is skipped"). Where several verbs need
one entry point, they share a helper, as the emit verbs share the barrier
helper.

| entry point | the one function that calls it |
|---|---|
| `vkAllocateMemory` | `rawalloc` |
| `vkQueueSubmit2` | `submitnative!` |
| `vkCmdPipelineBarrier2` | the Vulkan builder's barrier helper used by every emit verb |
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

`vkCmdBeginConditionalRenderingEXT` is not used: `when!` launches and draws
are indirect, with the flag in their count (section 2.4).
`vkDeviceWaitIdle`: its verb is open (section 10, Device wait-idle).

## 9. Allowlists

- **Mutable globals:** `REGISTRY` (the device registry). Nothing else, in
  core or in a device type.
- **Probes** (`hasmethod`, `applicable`, `hasfield`): none.
- **`@static Sys.is*`:** only in `Mantle.jl`'s include block.
- **`isa`/`typeof` on device or node types:** none.

## 10. Gaps: what is not specified yet

Each is to be specified before the phase named.

- **Graphics** (phase 6): the shader binding table's layout for procedural
  hit groups (`set_anyhit_pipeline!`'s role; Hikari's material types are not
  hit groups); stencil; multisampling; blending modes beyond the four
  singletons.
- **`caps(dev)` fields** (phases 1 and 8): the field names and types (whether
  a transfer queue exists can also be read from `queuekinds(dev)`); the
  contents of the compile configuration `config(dev, f)`.
- **Profiling** (an open decision): `timings(g)`, and what replaces
  `Profiler`, `makeprofiler`, `profiled!`, `gpupasstime!`, `collect!` and
  `timestamps`.
- **Device wait-idle** (phase 3): the verb's name and signature. Wait-idle
  runs under the channel lock (Windows); acquire and present do no
  device-wide wait.
- **`recycle!` on Vulkan, CUDA and HIP** (phase 3): which objects it returns
  and to which free list.
- **Copies between devices** (an open decision): `devicecopy!` across
  devices.
- **Selection table** (phase 6): its locking; when measurement runs.
- **Window defaults** (phase 7): GLFW hints; the default colour space and
  format (the Vulkan device type defaults to an sRGB swapchain, the Metal one
  to `BGRA8Unorm`, so one shader output means different things); the `vsync`
  default.
- **Counted driver objects** (phase 2): verb names for an object's bytes and
  its destruction.
- **Vendor views**: `withview` (an open decision; core's side in phase 6);
  the name of each device type's view constructor (Metal in phase 9, CUDA and
  HIP in phase 10).
- **Device teardown for debugging** (phase 8): what tearing down a device
  (`reset_device!`) means for its pool, its graphs and the registry.
- **The window open verb** (phase 7): its name and signature.
- **Texture and sampler tables** (phases 5 and 6): how a restored texture gets
  its new index.
- **`placeimage`'s arguments** (phase 2).
- **A test for "launch kernels never read cells"** (phase 4): section 6
  sketches the check; it has no test name yet.
- **Re-record count**: whether `Plan` counts re-records beside `compiles`.

## Appendix A. Today's backend vocabulary and what replaces it

Section 4's verbs replace `BACKEND_VOCABULARY` (`src/graph/backend.jl`, 189
names) as the list of functions a device type may define methods for; the
vocabulary check (section 6) enforces it. This table gives what replaces each
of today's names.

| name | fate |
|---|---|
| `abandonframe!` | goes: recording lifecycle replaced by Builder/finish and the free list |
| `abandonrecording!` | goes: as abandonframe! |
| `abandonrun!` | goes: as abandonframe! |
| `access` | internal to the device type's barrier helper (how a barrier is expressed) |
| `accesscache` | goes: into the `Kernel` at its `CodeInstance` |
| `acquire_next_image!` | becomes `acquirenext(sc)` (verb), called by core's acquireimage! |
| `AdaptedAccel` | goes with Metal's HWTLAS bookkeeping |
| `alignment` | becomes part of `rawalloc`'s constraint |
| `allocate_batch_queue!` | goes |
| `argbytes` | goes: core's bindings! lays out the argument block |
| `argidentity` | goes: as argbytes |
| `argsize` | goes: as argbytes |
| `argtype` | goes: `devicearg(dev, a)` is the one conversion |
| `argument_usage` | stays: the access per argument at declaration, stored in the `Kernel` |
| `awaitwrites` | goes: eager calls wait per sync state, no KA synchronize |
| `backend` | goes |
| `basealignment` | core: part of the selection key |
| `batchqueue` | goes |
| `begin_pass!` | goes: an eager `draw!(dev, …)` is its own pass |
| `begin_render_pass!` | goes |
| `beginframe!` | goes: acquire and present are part of run! |
| `beginrender!` | verb: `beginrender!(b, node, cmd, deps)` |
| `bestshape` | selection table (measurement open, section 10) |
| `bind_textures` | its per-call descriptor pool goes; textures through the device-wide table |
| `blittarget` | goes: image copies are `copyto!` operations (copy nodes) |
| `blocksize` | core: the pool's block size |
| `bufferusage` | becomes part of the memory kind's usage constraint |
| `build_accel!` | goes: per-device-type signature; core TLAS + `emitbuild!` |
| `callgroup` | goes: launch geometry is one core calculation from KernelInfo |
| `capacity` | becomes `budget(dev)`: today's `capacity(dev)` is the device's memory budget (Metal's `recommendedMaxWorkingSetSize`, `AMDGPU.free()`) |
| `caps` | verb: `caps(dev)`; `caps(::KA.Backend)` goes |
| `closerecording!` | goes: `finish(b)`; its patch filling goes |
| `closerun!` | goes |
| `collect!` | profiling (open, section 10) |
| `colorimage` | goes: render targets are images and windows |
| `compatible` | core: block compatibility for a kind and constraint |
| `compile_dispatch` | goes |
| `compile_draw` | becomes `compiledraw(dev, pipeline, argtypes, formats)` |
| `compiledraw` | verb |
| `constraintof` | core: the constraint of a kind |
| `copy_target!` | goes: `copyto!` between images and arrays |
| `currentimage` | goes: the run holds the acquired swapchain image, whose one owner is the window |
| `defaultdevice!` | goes: `Device()` and the registry |
| `depthimage` | goes: a depth target is a `Float32` image |
| `destroyrecording!` | goes: recordings retired by token through the release engine |
| `Device` | user: abstract type, `Device()` |
| `deviceaddress` | core: an array's address is in its cell; AS: `acceladdress` |
| `devicearray` | goes: `MantleArray(dev, data)` |
| `devicebuffertype` | goes: `devicearg(dev, a)` |
| `devicecopy!` | copies: `copyto!` as an operation / `emitcopy!`; copies between devices open (section 10) |
| `devicename` | core: shown by the device object |
| `deviceof` | core: `device(x)` |
| `devices` | user: `devices()` |
| `devicesized` | goes: LengthOf resolved at compile |
| `deviceslice` | goes: `devicearg` resolves views in one place |
| `deviceview` | goes: as deviceslice |
| `download` | goes: host reads (`Array(a)`, copy nodes) |
| `draw!` | user: `draw!(g or group or dev, target, pipeline, args, n)` |
| `draw_in_pass!` | goes |
| `draw_indexed_in_pass!` | goes: the draw_in_pass! family |
| `draw_indirect_in_pass!` | goes: the draw_in_pass! family |
| `emitbarriers!` | folds into the builder verbs |
| `emitcopy!` | verb: `emitcopy!(b, node, cmd, deps)` |
| `emitdispatch!` | verb (the builder verb takes the name over) |
| `emitdraw!` | verb: `emitdraw!(b, node, cmd)` |
| `emithead!` | folds into the builder verbs (the head is a node) |
| `emitinline!` | folds into the builder verbs: an inline store is `emitstore!` |
| `emitkernel!` | folds into the builder verbs |
| `emitloophead!` | folds into `emitloop!` |
| `emitpreparebarrier!` | folds into the builder verbs (prepare is a node) |
| `emitupdate!` | folds into the builder verbs: a run's stores are its change prefix (`emitcopy!`, `emitstore!`) |
| `end_pass!` | goes: the begin_pass! family |
| `end_render_pass!` | goes |
| `endrender!` | verb: `endrender!(b)` |
| `extrausage` | becomes part of the memory kind's usage constraint |
| `fence` | core: the channel's `next` (the last token handed out), a field read, not a device verb |
| `finishrecording!` | goes: `finish(b)` |
| `flush!` | goes as a user-facing verb |
| `fragmenttouches` | stays: a shader stage's access at declaration |
| `Framebuffer` | goes: render targets are `Image`s and windows |
| `gpupasstime!` | profiling (open, section 10) |
| `hold!` | goes |
| `holdleaves!` | goes at emit |
| `hostspan` | core: a host view of host-visible memory (Readback, upload regions) |
| `imageusage` | becomes part of the Images kind's usage constraint |
| `indexbuffer` | goes: indices are a MantleArray bound by `bindindices!` |
| `indirectindex` | goes: launch commands are slots core lays out |
| `indirectslot` | goes: as indirectindex |
| `initbackend!` | goes: hooks take the device |
| `initial_state` | core: image layouts decided by barriers! |
| `initial_usage` | core: as initial_state |
| `intrinsic_usage` | stays: part of the access walk at declaration |
| `isdevicearray` | goes: dispatch on MantleArray |
| `kakernelaccesssignature` | stays: the signature the access walk infers a KA kernel with |
| `kernelcompiles` | core: the per-device compile counter |
| `kernelinterpreter` | stays: the device type's interpreter the access walk and compile use |
| `kerneltouches` | stays: the access walk over the kernel's typed Julia IR |
| `kibackend` | goes |
| `layout` | internal to the device type's barrier helper (image layout of a use) |
| `librarygemm` | goes: selection table |
| `makeimage` | becomes `placeimage` behind `Image(dev, T, dims...)` |
| `makeprofiler` | profiling (open, section 10) |
| `makerecording` | goes: `Builder(dev, ch)` |
| `materialize!` | goes: transients placed by place! |
| `maxalloc` | becomes a field of `caps(dev)`: the largest single allocation, which the pool reads (section 4.1) |
| `mergeconstraints` | core: constraints of a kind |
| `native_attention_dispatch!` | goes: selection table candidates |
| `native_batched_gemm_dispatch!` | goes |
| `native_conv2d_dispatch!` | goes |
| `native_gemm_available` | goes |
| `native_gemm_dispatch!` | goes |
| `needs_transition` | goes |
| `openrecording` | goes |
| `openrun` | goes |
| `passbarriers` | goes: barriers! is core's |
| `passed` | verb: `passed(ch, value)`; core's `passed(::Token)` calls it |
| `passedas` | goes: argument layout is core's (bindings!) |
| `patchable` | goes with the pushed patch mechanism |
| `pool` | core: `pool(dev)` |
| `present_frame!` | goes: present is part of `submitnative!` |
| `profiled!` | profiling (open, section 10) |
| `Profiler` | profiling (open, section 10) |
| `rawalloc` | verb |
| `rawfree` | verb (its separate AS path goes) |
| `readback_framebuffer` | goes: `Array(img)` |
| `readback_window` | goes: `Array(win)` through the present segment |
| `record_draw!` | goes as a public path |
| `recorder` | goes: Builder |
| `recordsplans` | goes |
| `refit_tlas!` | goes: per-device-type signature |
| `release!` | `release!(::SubmitChannel, r)` goes; core's pool release stays internal |
| `release_batch_queue!` | goes |
| `remakeimage!` | goes: a new size is a new image under a new index |
| `reset_device!` | device teardown for debugging (open, section 10); declared, never implemented |
| `resetkernelcompiles!` | core: the compile counter |
| `resetrecording!` | goes: command objects back to the free list |
| `resourcekind` | core: resource kinds of sync states |
| `retire!` | core: `retire!(pool, x, tokens)`; `retire!(::SubmitChannel)` goes |
| `runscalls` | goes |
| `Sampler` | user: a value in the device-wide sampler table |
| `screenshot` | becomes `Array(win)` (name open) |
| `set_anyhit_pipeline!` | any-hit is a field of `RayTracingPipeline`; the SBT layout for many hit groups is open (section 10) |
| `set_viewport!` | goes: the begin_pass! family |
| `setviewport!` | goes: the scene rectangle is a per-draw uniform |
| `shadertouches` | stays: a shader's access at declaration |
| `staged_gemm_tile` | goes: selection table |
| `stages` | internal to the device type's barrier helper (pipeline stages of a use) |
| `stampof` | goes |
| `storage` | core: GPUArrays' `storage(a)` |
| `storebytes!` | goes: stores are pending changes (`Store`; inline ones written by `emitstore!`) |
| `submissionbudget` | core: `submissionbudget(dev)` computed from `caps(dev)` facts (job timeout, whether the device drives a display); not a device verb |
| `submissionholds!` | goes |
| `submit!` | core: `submit!(ch, parts; …)`; the device type's part is `submitnative!` |
| `SubmitChannel` | core type; fields change |
| `submitrecording!` | goes |
| `supports` | capabilities: fields of `caps(dev)` |
| `supports_batch_queue` | goes |
| `supports_geometry_stage` | capabilities: a field of `caps(dev)` |
| `supports_graphics` | capabilities: a field of `caps(dev)` |
| `supports_mesh_pipeline` | capabilities: a field of `caps(dev)` |
| `supports_procedural_traversal` | capabilities: a field of `caps(dev)` |
| `supports_rt_pipeline` | capabilities: a field of `caps(dev)` |
| `supports_tessellation` | capabilities: a field of `caps(dev)` |
| `supportspredicate` | goes: no conditional execution; `when!` launches and draws are indirect with the flag in their count (section 2.4) |
| `Surface` | goes: a window is a target itself |
| `syncbackend` | goes |
| `target_extent` | goes |
| `target_format` | goes: `imageformat(dev, T)` |
| `target_image` | goes |
| `target_view` | goes |
| `Texture2D` | goes: an `Image` gets its table index on its first use by a shader |
| `timestamps` | goes as a device hook for the split; profiling open (section 10) |
| `trace_closest_hits!` | goes: the immediate trace_* verbs |
| `trace_closest_hits_anyhit!` | goes |
| `trace_closest_hits_anyhit_indirect!` | goes |
| `trace_closest_hits_indirect!` | goes |
| `trace_rays!` | goes: `trace!(dev or g, …)` |
| `trace_rays_indirect!` | goes |
| `transition_image!` | goes as a call: image layouts are `barriers!` output (no implementation on either device type) |
| `upload!` | goes: stores are pending changes (`Store`; inline ones written by `emitstore!`) |
| `use_bindings!` | goes: bindings compiled with the plan (`compilebindings`), textures through the device-wide table; draw-binding code whose last caller is the walked draw in `execute!` |
| `use_frozen_kernels` | goes: the `Kernel` at its `CodeInstance` holds the code and each device's native handle |
| `vertextouches` | stays: a shader stage's access at declaration |
| `waitfor` | verb: `waitfor(ch, value)`, GC-safe; core's `waitfor(::Token)` calls it |
| `waitfor!` | `waitfor!(::Stamp)` goes |
| `waitidle` | device wait-idle (verb open, section 10) |
| `Window` | user: `Window(dev, w, h)` |
| `withpredicate` | folds into the builder verbs: a `when!` launch is indirect with the flag in its count |
| `workgroupsize` | goes: KernelInfo's group size |
