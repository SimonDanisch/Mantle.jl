# Mantle API: types, verbs and their contracts

This document is Mantle's API after the refactor: every type and verb with its
contract. This document owns public entry signatures and indexes the native
boundary; `recording-plan.md` owns the architecture and execution-contracts.md
supplies the detailed execution/ownership protocols. Pseudocode and examples
must agree with those contracts. Section 10 lists remaining protocol obligations,
implementation and qualification gates, rather than undecided API names.
Existing baseline evidence is indexed by the plan.

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
   Required means required by an enabled capability path: graphics/window
   verbs are required where graphics/display is supported, and hardware
   acceleration verbs where that hardware path is supported. Core selects a
   documented portable fallback or reports the missing feature before calling
   a device verb. Unsupported families do not acquire dummy methods.

## 2. User API

What application code, Makie recipes and Hikari scenes call. Exported unless
marked internal.

### 2.1 Devices

| call | contract |
|---|---|
| `Device()` | This machine's default: a `LavaDevice` on GPU 0 (never a CPU driver) on Linux and Windows, a `MetalDevice` on macOS. Returns the same object on every call. |
| `devices()` | Enumerates registered device types: Vulkan physical devices as `LavaDevice`s where available (including installed lavapipe), Apple GPUs as `MetalDevice`s on macOS, plus `CudaDevice`s and `ROCmDevice`s once `using CUDA`/`using AMDGPU` loaded their extensions. Driver objects are created on a device's first use, not here. |
| `LavaDevice`, `MetalDevice`, `CudaDevice`, `ROCmDevice` | Subtypes of `abstract type Device <: KA.Backend`. One Julia object per hardware device and driver; one GPU through two drivers is two devices with two pools. |
| `KA.get_backend(a)` | `device(a)` for an array on a device; its `GraphDevice{D}` location for an array on a graph, so a KernelAbstractions launch over it declares a kernel node (section 2.2). |
| `using Lavapipe_jll` | Makes lavapipe's ICD visible; must happen before the first `Device()`/`devices()` (the registry creates the Vulkan instance then). An extension. |

On a host with no hardware GPU for the platform default, Device() throws
NoDefaultGPU with the available explicit software devices listed. It never
silently selects Lavapipe. devices() and MantleArray(explicit_device, …) remain
usable on CPU-only CI. Display availability and watchdog risk are separate facts.

### 2.2 Arrays

`MantleArray{T,N,L} <: AbstractGPUArray{T,N}` for every location `L`: a device or
a graph, per device type (`GraphDevice{D}`). A view is the same type with its own
offset, dims and strides, sharing the parent's resource; there is no role parameter.
Nested views compose against the root state. Conjugating adjoints retain
their conjugation wrapper; a transpose of complex data must not conjugate it.
Read-only, unified and resizable are runtime properties of
the array's resource, not types.

| call | contract |
|---|---|
| `MantleArray(dev, T, dims...)`, `MantleArray(dev, data)` | A pool region on `dev`; operations on it run now. Fixed until its first `resize!`. From data: uploaded at creation (no graph can name it yet), streamed from the caller's memory in chunks through upload regions (each chunk one submission within the budget) or written directly into host-visible storage; returns once every chunk is submitted, later uses wait for them. |
| `MantleArray(dev, host; readonly = true)` | As above, and read-only: keeps a reference to `host`, no copy; eviction drops the device copy and a restore uploads from `host` again. An mmapped `host` (`Mmap.mmap` returns a plain `Array`) lets the OS drop and re-read its pages, so such an array can leave even RAM: this is how device data is backed by a file, and Mantle writes no files itself. Mantle does not know where the bytes come from. A write throws at the call: Mantle's own write methods, and `dispatch!` from the per-argument access it derives at declaration, eager or declared. `host` (or its file) changing afterwards is undefined; nothing checks it. |
| `MantleArray(g, T, dims...)` | A transient of graph `g`: contents exist only during a run, placed in `g`'s arena. Graph arrays are never persistent: what must survive runs and recompiles is a device array the graph names; what the host supplies per run is a device array written by stores. |
| `adapt(Mantle.Device(graph), x)` | `x`'s arrays, or a struct of them (a simulation's state), as zero-copy aliases on the graph's device: the graph names them (with a hold), so code that finds its target through its arrays declares into the graph. `adapt(Mantle.Device(), x)` uploads host data to a device. A `Group` is not a location. |
| `KA.allocate(dev, T, dims...; unified = true)`, `MantleArray(dev, T, dims...; unified = true)` | In unified memory, readable and writable by the host without a copy (KernelInterface's `unified`; `KI.supports_unified(dev)` is true on every Mantle device): a graph writes it, `Array(out)` after the run is a wait and a memcpy, no GPU copy. A one-element `Bool` one is usable as a `when!` flag. |
| `MantleArray(dev, data; streamed = true)` | A weight placed like a transient and uploaded by a copy node before its first use in each run (`examples.jl` 41; plan, Memory, "Streamed weights"). The source/interval/upload protocol is execution-contracts.md §12; bandwidth remains a measured cost. |
| `MantleArray(g, T, win)` | Sized like window `win`; placed at the window's capacity; contents undefined after a resize. |
| `MantleArray(g or dev, …; capacity = n)` | Resizable from the start with a reserve for `n` elements, like `sizehint!`: growing past it moves the storage, and plans whose temporaries were sized from it recompile once. Optional for a length a kernel writes: the kernel writes within the array's current storage, bounds-checked unless `@inbounds`; `capacity = n` reserves room for `n` up front. |
| `MantleArray(location(a), T, dims...) do t … end` | A temporary inside an operation: on a device a region retired with the call's token, on a graph a transient. |
| `MantleArray(l, T, SizedBy(f, a))` (internal) | A temporary sized at compile: `f(2 * length(a))` for a host length, `f(capacity(a))` for a kernel-written one. |
| `free!(a)` | Drops the creator's hold, once whatever the number of threads. Later use through any handle throws (eager calls, declarations, stores). Graphs and composites that hold `a` keep using it. Throws for a view. At the last hold the pending changes are dropped and the storage is retired (section 5, "the last hold"). |
| `resize!(a, dims)`, `resize!(a, n)` | Scheduled: the call only enqueues the request (`size(a)` reports the requested size; nothing is allocated, copied or written on the device). The next submission that touches `a` allocates what it needs before its locks and applies it in front of its own work. The first `resize!` call makes a fixed array resizable (plans compiled after it launch over it indirectly); when applied: if the new size fits its current storage, nothing moves (only a cell write); beyond it, a large array on a device with sparse residency moves into a reserve (later growth maps pages, the address stays), and any other array moves into a pool region of twice the need (each later doubling is a move). Which arrays are large, and a reserve's size, core computes from `caps(dev)`. Plans that launched over it directly recompile once. Contents kept as linear memory (`resize!(a, n)` sets the last dimension). Shrinking unmaps nothing. Throws for a view. On a graph array: within the placed size a cell write, beyond it the graph recompiles at its next run. |
| `trim!(a)` | Pending: unmaps a reserve's pages beyond `a`'s size, retired with the token of the submission that unmaps them; afterwards a view past the size reads undefined values. The only call that unmaps an array's pages: pressure never does (plan, Arrays). |
| `a[r] = data` | A store, scheduled: the call copies the data once into an upload region (host-visible system memory acquired for it, not the array's storage) and enqueues the request with its byte range in the parent (a view's offset applied). The submission that applies it writes it into the storage the array has at that point of the sequence: within the device type's inline limit (`caps(dev)`: 64 KiB at 4-byte offsets and sizes on Vulkan) with `emitstore!`, beyond it by a copy from the upload region (`emitcopy!`); no staging is allocated at apply. A store pended before a resize is copied along with the old contents; one after it writes the new storage. `length(data) == length(r)` or `DimensionMismatch`. A later store covering an earlier pending one replaces it. |
| `copyto!(a, data)` | A store into `a[1:length(data)]`. |
| `copy!(a, data)` | Base's `copy!`: a resize to `length(data)` that keeps no old contents and a store, enqueued together. On a view only with the view's length. |
| `Array(a)`, `copyto!(host, a)` | A host read: its own submission (the pending changes of `a`, a copy into Readback memory), a wait for that token, a memcpy. For a kernel-written length the cell is copied in the same submission. Throws for a graph array. |
| `show(io, MIME"text/plain"(), a)` on a graph array | A metadata summary (element type, requested shape when host-known, graph); reads no contents, submits nothing and never reads back a kernel-written length. Displaying the value at the REPL is allowed. |
| `size(a)`, `length(a)` | The real length, as Julia's array API defines it. For a length a kernel writes: waits for that kernel and reads the length back from its cell; launches never need it, they use the length on the device. During the declaration of a graph that writes it, no run has produced it yet, so it throws. |
| `capacity(a)` | The array's current reserve, like a `Vector`'s after `sizehint!`: `capacity = n` at creation, raised when growth moves the storage. It also sizes temporaries over a kernel-written length (`SizedBy`), which is why plans that sized one from it recompile once after a move (`resizing-and-raytracing.jl` B.2). Not the device's memory budget, which is `budget(dev)` (section 2.8). |
| `launchrange(a)` | The one name for a launch over an array: a reference to `a` resolved at compile, direct while `a` is fixed, dynamic after its first `resize!` (indirect where supported, otherwise capacity predication). There is no `DeviceRange`. |
| `lengthof(a)` | A kernel argument through which a kernel writes `a`'s length into its cell; declaring that launch adds a prepare node. |
| views, `reshape`, transposes (GPUArrays' `derive`) | Share the parent's state (storage, size, cell, sync state, holds, freed mark). No hold, no finalizer of their own. |
| `KA.get_backend`, `KA.allocate` on a graph array | Its GraphDevice: a KernelAbstractions launch over graph arrays declares a kernel node and `KA.allocate` gives a transient, so KA libraries and GPUArrays' generic code run inside a graph. `getindex`, iteration and every other host read of contents throw at their line, since nothing exists before a run. Launch sizes such code computes on the host are recorded per node; a resize of an array behind one makes the next compile throw. |
| bounds | Kernels are bounds-checked unless they say `@inbounds`, where an access past the storage is undefined. Nothing clamps a kernel-written length. |
| broadcast, `fill!`, `copyto!`, `map!`, `mapreducedim!`, `accumulate!`, `sort!`, `sortperm!`, `mul!`, `fft`, … | Methods on `MantleArray` that launch through `dispatch!(location(a), …)`: run now on a device, declare on a graph. GPUArrays' generic code serves the rest, on devices and graphs alike. |
| `allowscalar` | GPUArrays' interface, accepted as such; scalar indexing off by default. |

A noncontiguous store captures the view/index mapping and packs only its
source elements; the change prefix scatters them with a kernel. It is never
treated as a contiguous byte span. Coalescing tests actual destination
coverage, so stores into interleaved elements do not erase one another.
Preparing changes copies mutable pending lists and storage facts under short
record locks, releases the locks before allocation, and validates the snapshot
when applying. Concurrent mutation of an unlocked Vector is not a snapshot.

A unified host read waits outside the locks, then captures an idle storage,
shape and writer-token snapshot under the resource and naming-plan locks.
It allocates its host output outside them, validates the snapshot again, and
copies while holding them (no GPU wait inside). New submissions cannot race
the CPU copy. A changed snapshot retries; a bare wait followed by an unlocked
memcpy is insufficient. The direct path holds these bookkeeping locks during
the CPU copy. A staged read captures its shape in the copy submission, not
from a later `size(a)` after waiting.

### 2.3 Graphs

| call | contract |
|---|---|
| `Graph(dev; rewrites = (), profile = false, priority = 0)` | An empty graph on `dev`. Rewriters run only if listed (`FuseEpilogues()`, `MergeRegions(attribute)`). Priority is a relative scheduling hint, with no preemption guarantee; profiling is opt-in (execution-contracts.md §10). |
| `run!(g)` | The first call compiles, records and runs; later calls run. Every independent root segment waits on the GPU for all previous-run tokens of g. The host waits only with the graph lock and the run's arena-part holds held, never with a sync-state, channel, pool or window lock: for another graph's run that holds a shared arena part, for a host node's inputs, for the swapchain acquire, and in `acquire!`'s pressure steps (plan, Ordering, Run). A `GPURef` store never makes it wait: it is a scheduled change like any other. Returns `Submitted()` or `Skipped()` (a minimized window), both public. Throws for a freed graph, when called from a host function of `g`, and when the calling task holds an arena part of `g`'s plan. |
| `free!(g)` | Releases the plan and drops the graph's holds, once; also done by the graph's finalizer. `run!` and declaring afterwards throw. Called from a host function of `g` it throws before it marks `g` freed, so the finalizer still releases it. |
| `submissions(g)` | A vector of stages, cut by host nodes; each has `.segments`, cut by budget and queue. A segment has `.channel` and `.after` (its predecessor segments in this run). |
| `timings(g)` | Nonblocking snapshot of completed sampled GPU intervals and sampling counters; profile=false creates no timer objects. execution-contracts.md §10 fixes lifecycle and availability. |
| `naivebytes(g)`, `peakbytes(g)` | Transient bytes without and with placement. |

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
The previous run's tokens belong to the graph across plan replacements: the
new plan inherits them before its first submission, even when the old and
new plans name disjoint resources.

| call | contract |
|---|---|
| `group = Group(g or group, name)` | A value and a place in the order: the end of its parent's order (the graph, or another group) when it is made. Work is declared into it by passing it (`dispatch!(group, …)`, `trace!(group, …)`, copy kernels, `clear!(group, …)`, `draw!(group, …)`, `when!(group, …)`, `Group(group, …)`) anywhere in the code, and runs at the group's place whenever it was declared. It holds GPU work only: a host node, a build or an image copy has no method for a `Group` (a `MethodError`). Not an owner: arrays and allocations belong to the device or the graph, so library code that finds its target through its arrays declares into the graph, not a group. |
| `delete!(g, group)` | Removes the group and everything in it. A group of draws leaves its pass's list at the next run with no recompile (2.6b); any other group marks the plan. A resource no remaining node names loses the graph's hold once the plan that names it has retired. Work that will be removed is declared into a group; nodes declared into the graph itself stay until `free!(g)`. |
| `sort!(group; by, lt, rev)` | Reorders the groups a group holds (`by` sees the groups); its other nodes keep their places. Groups of draws in one pass: the next run rewrites the pass's list, nothing recompiles. Otherwise the plan is marked. |

### 2.4 Declaring work

| call | contract |
|---|---|
| `dispatch!(g or group or location, kernel, args::Tuple, ndrange; group, origin)` | The one launch verb. On a graph it adds a kernel node; on a `Group` a kernel node at the group's place; on a device it uses the CodeInstance-owned Kernel and submits now, with what is pending on its arguments in front. A `group` size the caller gives (here, or KA's `kernel(backend, workgroupsize)`) is kept as given; core chooses a legal default from `config(dev, f)` only where none is given (the bounded default heuristic and validation are fixed in execution-contracts.md §9). |
| `location(args...)` | The one graph among the arguments if any lives on one (its device arrays are then named by that graph), else the arguments' device. Callable fields and the explicit arguments are the only way a call reaches a graph: no global, task-local or scoped "current graph" exists (plan, rule 5). Arguments from two graphs or two devices throw. An argument that reaches device memory but is not a `MantleArray` (a vendor array, an array escaping `withview`, a raw backend array) throws: the resource of an argument is the array's state, shared by its views and reshapes, never the object (plan, Arrays). |
| `dispatch!(g, HostCall(f), args; writes::Tuple)` | A host node. `f` runs on the calling thread of `run!`, between the stages before and after it, on host-visible arena memory. It may not use its own graph or a graph that shares an arena part with it (both throw: `lockgraph` throws when the calling task holds an arena part of that graph's plan). If `f` throws, the rest of the run is not submitted, the exception propagates, and an acquired swapchain image goes back to the window for the next run. Has no method for a `Group` and is not a `when!` launch. |
| `repeat!(group, n; while_nonzero = count)` | The group runs `n` times per run at its place; it holds launches only (kernels, traces, copy kernels): a draw, a clear or a group in it throws, at `repeat!` or when declared later (loops do not nest). After a compile it marks the plan; before each iteration a loop-head kernel writes the iteration index (`iteration(group)`) into the slots of the launches that take it. `while_nonzero` is an integer count on the device (a queue's live size) that ends the loop on the GPU. `repeat!(g, kernel, args, ndrange; max, while_nonzero)` is the form for a loop of one launch. |
| `when!(g or group, flag, kernel, args, ndrange)` | One launch that runs only where `flag` is true on the GPU. `flag` is a `GPURef{Bool}` or a one-element `Bool` array, enforced by the method's argument types: anything else is a `MethodError`. The meaning is the same on every device type; only the mechanism is the device type's verb. Vulkan and Metal launch indirect, and whoever writes the indirect command multiplies the count by the flag: the head node for a flag the host writes, a prepare node after the kernel that writes it. CUDA and HIP reach the same result their own way (the baseline on CUDA/HIP is a capacity launch with device-entry predication; qualified native indirect/conditional candidates are optional). No conditional rendering; the flag is never a handle dependence. Several launches under one flag are several `when!`s. The host reads a flag array after the run like any device array. |
| `trace!(g or group or dev, pipeline, accel, args, ndrange)` | A ray-tracing launch; allowed in a `Group` and as a `when!` launch. GPU control disables it when the loop ends or the flag is false, using indirect commands/ranges or the capacity-predication baseline as applicable (execution-contracts.md §3). |
| `copyto!(out, x)` with `x` on a graph | A copy kernel node (so allowed in a `Group`), recorded once and run every run; how the host reads a graph result. |

Selection is a call in the method: `mul!` and the other library operations
ask the device's selection table at declaration. The table is per device,
keyed on function, element types, shapes, layout and alignment (with
`basealignment`; plan, Operations); an entry not yet measured uses core's cost
estimate. In a `Group` and under `when!` only kernel candidates qualify.

Mantle operations that compose launches (`fill!`, `copyto!` between arrays,
`map!`, `mul!`, reductions, scans, sorting and FFT) also accept an explicit
leading `Group`: `mul!(group, C, A, B)`, `copyto!(group, dst, src)`. They pass
that group to every internal launch and choose only kernel candidates;
temporaries still belong to the graph. Implement the body once with an
explicit target, shared with the array-inferred form. Julia broadcast syntax
and GPUArrays methods without a Group overload still target the arrays'
graph; passing a Group to such a method is a `MethodError`. Consumer model
methods must pass the group through all operations intended to repeat.

Candidates declare `requires(candidate)`: feature requirements checked
against `caps(dev)` before cost estimates or measured entries are considered.
Core contributes portable candidates; a device type contributes native
candidates. Facts include supported scalar arithmetic and atomic operations
(width, scope and ordering), subgroup operations/sizes, cooperative-matrix
types/shapes/scopes, ray query, ray pipelines and mesh stages. Users call
operations such as `mul!`; they do not select vendor extensions. A raw kernel
requesting an unavailable intrinsic fails at compile with a feature error;
Mantle does not silently weaken arithmetic or memory ordering. The required capability schema is fixed in execution-contracts.md §6;
F3 tracks its native implementation and qualification.

Reuse KernelInterface's existing `DeviceCaps`, `MatrixShape`, matrix scopes
and shape queries. Mantle's expanded runtime facts compose that kernel-facts
record rather than duplicating its matrix/subgroup fields or tables.
Because `Mantle.caps` is the imported `KI.caps` generic, core must provide
the projection/query methods for the expanded record (`matrix_shapes`,
`wggranularity`, `supports` and `bestshape`) once, for every device type.
Candidate selection and compiler configuration read the same kernel facts;
they do not maintain separate capability caches that can disagree.

The selection table belongs to the device, protected by its own lock. An
unmeasured call uses the estimate without benchmarking. `tune!(dev, f,
representative_args...)` measures eligible candidates on private inputs,
outputs and workspaces, verifies their results, then publishes entries under
that lock. Compilation, allocation and GPU waits occur outside it. Keys
include compiler configuration and the candidate implementation version;
changed code/configuration invalidates the entry. No tuning happens inside
`run!`, a host node or graph declaration. Initially the table is in memory
only. A selected candidate's supported shape/capacity is a plan dependence:
growth outside it reselects through recompile, rather than running an invalid
specialization.

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
| `TLAS(dev; hardware, raytypes = 1)` | One core type for hardware and software; raytypes fixes its logical hit-group record layout. `TLAS(dev)` picks the hardware path where the device has it; `hardware = false` forces software. Storage sized for a capacity starting at 64 that doubles; host-known counts build/refit with the exact count. Pending changes to one structure coalesce: any Build makes a Build, else an Update; every Update uses `builtcount`, the count of the last Build. A total a kernel writes (any device type): built over the capacity with mask-0 records past the total that reference a valid BLAS. An Update before any Build is a Build. A new TLAS pends a Build of zero instances, so a trace before any `push!` reads a built, empty structure. |
| `push!(tlas, blas, transform::Mat4f; hitgroup = 1)` | One instance; returns a range handle. Pending: a range-table store and a Build. A total past the device's `maxInstanceCount` (16 777 215 on every Vulkan GPU here) throws at the call that raises it: `push!`, a `resize!` or `copy!` of a source; the capacity is capped at the limit. |
| `push!(tlas, blas, transforms::MantleArray{Mat4f,1})` | One instance per element. |
| `push!(tlas, blas, positions; scale, rotation)` | One instance per position, transform computed on the device; `scale` and `rotation` arrays or one value. |
| `tlas[h] = t` | Pending: a store into the range's source and an Update (refit). |
| `delete!(tlas, r)` | Pending: the range's entry inactive (`active = false`) and its slot freed, a Build; the range's BLAS and arrays released with the token of the build that deactivated it. |
| `visible!(tlas, r, flag::Bool)` | Pending: changes the range's mask, preserving its slot and count, and pends an Update. No plan recompile. |
| `build!(tlas)` | Pending: a full build, applied by the next submission that touches `tlas`. |
| `build!(g, tlas)`, `refit!(g, tlas)` | A build or refit every run. A TLAS whose total a kernel writes holds exactly the instances below the total on every device type: the portable baseline builds over capacity with mask-0 records past the total and valid BLAS references. Native count-from-memory receives that same fixed build count, so changing the logical total cannot invalidate a refit. A refit before any build is a Build. Not a launch: no method for a `Group`, not a `when!` launch. |
| `RayTracingPipeline(; raygen, closest_hit, miss, any_hit, payload, maxdepth)` | Immutable typed program value. One callable or tuple of closest-hit groups; every shader receives the same args. General TriangleHitGroup/ProceduralHitGroup and multiple ray types use execution-contracts.md §7. Both Hikari shading queues and typed closest-hit groups are preserved. |
| `rayquery(tlas, origin, direction; tmin, tmax, mask, accept, intersect)` | Closest accepted device hit, with identical logical indices/masks/intervals across native and software traversal. RayHit fields and procedural intersections are fixed in execution-contracts.md §7. |
| `traceray(origin, direction, payload = nothing; raytype, tmin, tmax, mask)` | Callback-invoking device trace within a ray program; returns the typed payload, logical ray type and automatic optional SER; rayhit/raypayload/currentray stage IO is fixed in execution-contracts.md §7. |
| `hitgroup!(tlas, range, index)` | Scheduled logical group change; core encodes native offsets. Pipelines sharing a TLAS must agree on group meaning and ray-type count. |

| an `Image` a shader samples | Gets an index in the device-wide texture table on its first use by a compiled pipeline or kernel; the index is stable while that native image remains its storage. An upload of the same size and type writes the image it has. `resize!(img, w, h)` or an upload of a new size: a new image under a new index and a store into the slot, keeping the texels where the old and new sizes overlap (one GPU copy, as `resize!` keeps an array's contents); the old image retired with the token of the submission that applies the change. Never a recompile. |

Shader image arguments are typed `SampledImage{T,N}` values containing the
current table index and extent from the plan slots; a raw UInt32 alone loses
the result type and dimensions. `sample(image, sampler, uv; lod = 0f0, gradients = nothing)` is the
portable 2-D sampling intrinsic, usable in compute and fragment code. Coordinates
are normalized: texel (i,j) is centred at ((i-0.5)/width,(j-0.5)/height), with no
implicit vertical flip. Sampler defaults are filter=:linear, address=:clamp,
anisotropy=1; :nearest and :linear, and :clamp/:repeat/:mirror are required.
RGBA normalized formats return RGBA{Float32}, float formats their float value;
sRGB RGB channels decode to linear, alpha stays linear. Integer formats require
nearest filtering. Explicit lod without gradients permits anisotropy=1 only.
Anisotropic filtering requires explicit gradients in compute, or a qualified
implicit fragment footprint. Unsupported native anisotropy selects an eligible
core footprint sampler or rejects at selection, never invents derivatives
(execution-contracts.md §11). Mipmap support must be qualified before enabling it. Sampling an attachment
being written in the same pass is rejected unless an explicit supported feedback
path exists. `size(image)` in device code reads its slotted extent.

`Mantle.textureindex(image)` is an internal diagnostic projection for the suite;
consumers use sample, never arithmetic on descriptor indices. A backend without
descriptor indexing selects core buffer-backed sampling at compile, with native
image/buffer copies at required render boundaries. This fallback may cost extra
GPU copies and must be measured; it does not upload/read back via the host or
ask consumers to choose descriptor extensions.

In portable shaders, RayHit.customindex is the range's one-based slot (stable
for its life), and RayHit.instanceid - rangeoffset(tlas, slot) is one-based
within its range. Core decodes native zero-based instance/custom indices; raw
KI native builtins retain their documented native convention (§7). A range's triangle data and material index are
reached through the range table, never through a scene-wide concatenation.
Hiding a range: mask 0 in its records (`RangeEntry.mask`, a refit). Moving: a
refit. A new host-known count: a Build with the exact count; device-written
totals retain the fixed capacity build count. Deleting a range: its entry
inactive, a Build. None of these recompiles or drops a plan.

### 2.6b Graphics

| call | contract |
|---|---|
| `GraphicsPipeline(; vertex, fragment, geometry, blend, cull, topology, depth)` (`Rasterizer`), `MeshPipeline(; mesh, fragment, object, …)` | Values. Stages `VertexShader`, `FragmentShader`, `GeometryShader`, `MeshShader`, `ObjectShader` (tessellation as stage types). Compiled through core's device-owned pipeline cache (`pipeline!`/`compiledraw`), keyed on stage code identities, configuration, device-side argument types, fixed state and target formats; equal keys give one pipeline per device. |
| `draw!(g or group or dev, target, pipeline, args, count; instances = 1, indices = nothing)` | A draw node into `target`: a `Window`, an `Image`, or a tuple of them (colour images, then one `Float32` depth image). On a device it runs now, as its own pass; an eager draw into a window throws. `count`, `instances`: numbers or `launchrange`s (indirect over resizable arrays). `indices`: a `MantleArray{UInt32}` bound by handle; it never moves within its capacity, and a move marks the parts that recorded the handle, which are recorded again before their next submission; nothing recompiles. |
| `clear!(g or group or dev, target, v)` | A clear node: the load operation of `target` in the pass of the draws that follow it. A target with no clear before its draws keeps its contents. Discard at pass end is allowed only for a graph transient with no later read or preserving load; device images and windows are stored for host reads, other graphs and presentation. A clear after a draw starts a new pass; it cannot become the earlier draws' load operation. |
| render passes | Formed by compile: clears and draws into the same targets, in order with no node between them that reads what they write, are one render pass. There is no pass object. A window-sized target's segments are recorded again with each swapchain generation. |
| a group of draws | A group whose nodes are draws into one pass is recorded on its own (a piece). Adding one at a place inside a pass with its targets, declaring draws into it, deleting it, or `sort!` of such groups is applied by the graph's next run before its stages: the group's draws are compiled and recorded on their own and the pass's list rewritten; nothing recompiles and no other group is recorded again. A launch, a clear, a draw into the graph itself, or a group that would start a new pass recompiles. A group of draws that fails to compile is deleted and `run!` throws. Visibility is a store into the draw's `when!` flag. `GPURef`s it reads are slots it bakes, like a plan. |
| `when!(g or group, target, flag, pipeline, args, count; instances, indices)` | One conditional draw: indirect, the count written into its command multiplied by the `Bool` flag, as for a `when!` launch (no conditional rendering); visibility is a store into `flag`. |
| `attribute(a, i)` (device side) | `a[min(i, length(a))]` (1-based), the length from the slot: one argument type and one pipeline for a uniform, one value for all elements, or one per element (performance gate: within 1% of `a[i]` and of a true uniform on NVIDIA, AMD and the M5). |
| scene rectangle | One one-element array per scene, shared by its plots, that the vertex stage applies to each draw (a remap and four clip distances); a layout change is one store. |
| `Image(dev, T, dims...)`, `Image(g, T, dims...)`, `Image(g, T, win)` | An image resource: sync state, holds, stores (pending uploads), `copyto!` with arrays (pending changes or copy nodes), `Array(img)`. Format from `T`, one table per device type (`imageformat`). Layouts are decided by `barriers!`. `Image(g, T, win)` is recreated per swapchain generation at the actual size on the same arena memory. |
| `Sampler(; filter, address, anisotropy)` | A value; a device-wide sampler table entry created on first use. |
| `Array(win)` | A screenshot: the next run presenting to the window copies the image into `Readback` memory before presenting; waits for that run. Throws `ArgumentError` when the window is minimized, when no graph presents to it, or from the presenting graph's own host function; a call already waiting is woken with that error when the graph's next run is skipped or the graph is freed. |

A pass that accepts new pieces reserves dependencies on every earlier writer
and before every later writer, with cross-queue waits as well as the pass's
memory barrier. The fast path requires draws to read non-attachment resources
only and to fit the pass's existing image-layout, placement and queue
contract. Storage writes from a shader, new image-layout transitions or
attachment feedback requiring new dependencies trigger a plan recompile.
Deleting or sorting pieces preserves the same condition. A fixed memory
barrier alone cannot establish a dependency between queues. This condition
qualifies the zero-recompile claim for groups of draws above.

### 2.7 Windows

| call | contract |
|---|---|
| `Window(dev, w, h; title, vsync, capacity)` | A graph resource; its swapchain (Vulkan) or drawables (Metal) is its storage; the device type's window open verb makes the window and its surface (section 4.7). Capacity: the largest monitor's size unless given. Written by exactly one graph, the presenting one: any of its nodes may write the window (a draw, a clear, a kernel, a copy), the first graph declaring a write presents, and a second graph declaring one throws at declaration (other graphs render into `Image`s that graph reads). The binding ends when the graph is freed or collected or the nodes writing the window are removed; another graph may then present. An eager draw into a window throws; eager draws into images are allowed. There is no `Surface` and no `target_*` accessor. Defaults: linear shader output converted through an sRGB surface path, FIFO/vsync on; surface-specific negotiation is in execution-contracts.md §4. |
| `resize!(win, w, h)` | Called from the window's event: stores the size and marks the graphs that render to it; nothing else happens on the event thread. The next run of such a graph makes the new swapchain generation, recreates its window-sized images at the actual size on the same arena memory (the arena part is sparse where possible and maps only the pages the size needs), and records again the segments that draw into window-sized targets and the per-image present segments. Beyond the capacity: the graph recompiles at its next run, like a graph array beyond the size its slot was placed for; it never throws. |
| an acquired swapchain image | One owner, the window. An image acquired by a run that returns `Again` or throws goes back to the window; only real images are kept. |

### 2.8 Memory introspection

`reserved(dev)`, `mapped(dev)`, `budget(dev)`; heap/class and opaque-object
breakdowns distinguish physical allocation from logical live bytes. trim!(dev)
releases empty caches, while trim!(array) schedules live-range page trimming.
devicecopy!(dst,src) explicitly transfers device arrays across driver/device
domains with the snapshot/completion contract in execution-contracts.md §12.

### 2.9 Vendor views

`withview(f, CuArray | MtlArray | ROCArray, a, stream)`: a zero-copy view of `a` in the vendor's type, inside the block
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
| `Graph` | nodes as declared, named resources, rewrites, profile and priority settings, plan; the flags `recompile`, `windowchanged`, `freed` | user | `Base.Semaphore(1)` (not reentrant; owner task checked); the flags are atomics (section 5) |
| `GraphDevice{D}` | graph-array location, retaining the physical device type for dispatch; a weak reference to the graph | graph | immutable |
| `Plan` | compiled state: stages, placement, argument block, tables, recordings, `lastrun`; the flags `stale`, `unmapped`, `restore`, `running`; `compiles`, the number of compiles of the graph so far, carried from each plan to the one that replaces it (for debugging and tests) | graph | graph lock; the flags are atomics (section 5) |
| `CompileState` | the plan under construction; a copy of the nodes | compile | graph lock |
| `Node` subtypes | `KernelNode`, `TraceNode`, `ClearNode`, `DrawNode`, `Group`, `HostNode`, `HeadNode`, `PrepareNode`, `CopyNode`, `LoopNode` (compile-time repeated-group syntax, expanded before the execution DAG), `RenderNode` (a pass, formed by compile), `BuildNode`, `ForeignNode`, `RegionNode` (a run of nodes one region compiler compiles), `VideoDecodeNode`, `PresentNode` | plan (its own copies) | graph lock |
| `CompiledNode` | queue, predecessors, barrier, command, cost, slots | node | graph lock |
| `Origin` | the Julia call that launched a node | node | immutable |
| `Kernel`, `DeviceKernel`, `KernelInfo` | one kernel: its function, the access per argument (declaration), its code (compile), and per device the native handle and `KernelInfo` (first use) | attached to the kernel's `CodeInstance` (`GPUCompiler.cached_results`), one per compiler configuration | the kernel's own lock; code/access immutable once filled; idle native entries may be dropped and recreated under §13 of execution-contracts.md |
| `Bound{R}` | a plan's argument slots for a resource | plan | node uses |
| `MantleArray{T,N,L}` | location; the resource (state, runtime properties); a view's offset, dims and strides | user | — |
| `ArrayState` | storage, dims, length source, capacity, cell, sync state, holds, dependents, readers; the freed mark is on the sync state only | shared by an array and its views | its sync state's lock |
| `Cell`, `NoCell` | an array's address and dims in the device's cell arena; `NoCell` for a fixed transient whose descriptor is compile-time placement | array state | cells written only by applied changes or by a kernel through `lengthof` |
| `Region`, `Reserve`, `Committed`, `Evicted`, `Released`, `Unplaced`, `ArenaSlice` | device storage: pool region, sparse reserve, doubling region, host eviction copy, released; graph storage: unplaced metadata, then a plan-owned arena slice | array state | the array's sync state (host view); graph lock for placement; pool lock for blocks |
| `Arena` | one arena part: range, mapped bytes, tenants, sync state, holder | pool | its sync state (mapping, tenants); its condition (holder) |
| `SyncState`, `Access` | per shared resource: the accesses of what is not a run (eager calls, host reads, vendor views, eviction copies, applied change steps: last write and reads since), the last tokens of plans retired while naming it, pending changes, transfer lease, lent and freed marks. Runs are not recorded here (their last-run tokens, found through the plans naming the resource) | every shared resource | its `ReentrantLock` (never held while `acquire!` runs) |
| `namers` (on `SyncState`), plan inbox, plan sync lock | per resource, the plans naming it with their use from compile (an immutable snapshot, replaced at registration and retirement; composite members reach the plans naming the composite through their readers); per plan, the resources written by other runs, changed or accessed eagerly since its last look (a lock-free list), and the lock that orders its submissions against the plans it shares resources with | the resource; the plan | the resource's sync state lock (`namers`); the plan's sync lock (the inbox is lock-free) |
| `Token` | `(channel, value)`: a value a channel signals. Core's `passed(t::Token)` and `waitfor(t::Token)` call the device-type verbs `passed(ch, value)` and `waitfor(ch, value)` (section 4.2) | accesses, `lastrun`, retirements | immutable |
| `Change` subtypes | Pending requests: `Store`, `Resize`, `BuildAccel`, each with a device-wide sequence number. A Store owns its upload region and destination mapping into the parent (a byte span when contiguous). `CellWrite`, `BindPages`, `MoveStorage`, `MoveAccel` and `Restore` are prepared application steps, never pending requests. | sync state | sync state lock |
| `SubmitChannel`, `ChannelState` | abstract native endpoint and canonical shared queue state: timeline, lock, next value, failure domain and outstanding submissions. Aliases share the identical ChannelState. | device / endpoint aliases | ChannelState.lock for enqueue; waits outside locks |
| `Recording{N}` | one recorded part of a segment, written once. A segment is a list of parts: recordings and `BuildRecording`s | plan | — |
| `Submission{T}` | a token and the command objects to recycle once it passed | channel | channel lock |
| `Builder` | a device type's recording in progress; a value that never escapes the call that made it (`record!`, an eager call, a submission's change prefix, the re-recording of a `BuildRecording` or a window-sized segment) | stack | — |
| `Deps` | the predecessor nodes and the derived barrier of a node | compiled node | immutable |
| `CompiledDispatch`, `CompiledTrace`, `CompiledCopy`, `CompiledBuild`, `CompiledRender`, `CompiledDraw`, `CompiledLibrary`, `CompiledHead` | what compile decided for one command; `CompiledCopy(NoCopy())` is a barrier-only command (including the transition to presentation) | compiled node | immutable |
| `TLAS{P}`, `BLAS`, `InstanceRange`, `RangeEntry` (`mask`, `active`), `MemberSet` | resizing-and-raytracing.jl B.8; `builtcount`, the count of the last Build, which every Update uses | user | the structure's sync state |
| `BuildRecording` | a plan's TLAS or BLAS build or refit in its own command buffer, on every device type (Vulkan records the count, Metal the storage). A change that alters what it recorded, or a Build that sets a new `builtcount`, marks it; the run re-records the marked ones after the change prefix (nothing is detected at run) | segment parts | the structure's sync state, held by the run |
| `Window`, swapchain generation | size, pending size, current generation with holds; the swapchain images acquired and not yet presented (the window is their one owner) | user | the window's lock |
| `GPURef` | a value box: one slot in device memory | user | the slot's sync state |
| texture table, sampler table | the device-wide descriptor tables, with free lists | device | the table's lock, a leaf (section 5) |
| `Pool`, `Inbox` | blocks per kind, retiring list, eviction list; lock-free inbox for finalizers | device | pool lock (a leaf); none (inbox) |

Callable resource captures participate in location, holds and access inference
just like explicit arguments (execution-contracts.md §7; DECL-12/13).

The access per argument of a kernel or shader is derived at declaration,
before any compile: from a declaration next to the function for a call with
no Julia body, else by the taint walk over the kernel's typed Julia IR through
the device type's interpreter (`argument_usage`, `kerneltouches`,
`shadertouches`, `vertextouches`, `fragmenttouches`, `kernelinterpreter`,
`kakernelaccesssignature`, `intrinsic_usage`). It is stored in the `Kernel` at
its `CodeInstance`, not in `KernelInfo`. `dependencies!`, `barriers!` and the
read-only check (section 2.2) read it.

## 4. Device-type verbs

[Execution contracts](execution-contracts.md) are part of this API: §2 fixes
prepare/claim/accept/publication and terminal failure; §3 completes factories,
object destruction, foreign streams and video; §4 defines presentation retirement.
Sections 6–14 complete capability, SBT, timestamp and memory-class verbs and their
ownership. These companion signatures are part of the same exclusive boundary.
The submission and presentation operations are separate. Native calls can fail;
a no-fail driver guarantee is not an API precondition.

The only functions a device type (`src/vulkan`, `src/metal`, `ext/`)
implements, at the signatures core calls. For every verb: it dispatches on
the device (or an object of it) as its first argument; it takes no Mantle
lock unless stated; it does not call into core except through the functions
named here; it reports unsupported paths through capabilities before compilation. Preparation
and object creation may fail; errors after token claim follow the terminal
failure contract. Normal WSI status is a structured result, not device loss.

### 4.1 Device

| verb | called by, when | contract |
|---|---|---|
| `hardwarekeys(T, registry)` | `devices()`, under the registry lock | The keys of the hardware devices this device type can open. Creates no driver objects beyond enumeration. |
| `T(key)` | the registry, once per key | The device object; driver objects are created on first use, not here. |
| `caps(dev)` | core, at compile and at declaration | An immutable record of facts the device type reports; core decides from them (required schema in execution-contracts.md §6). Fields: `sparseresidency`; `accelbyaddress`; the device's limits; the largest single allocation; the sparse page size and the device's sparse address space (core's `reservesize` and `sparsethreshold`); the driver's job timeout and whether the device drives a display (core's submission budget); the roofline inputs of `costs!` and `schedule!`: throughput per element type, memory bandwidth, whether a transfer queue exists, the cost of a wait between queues (internals.jl section 4: `caps(dev).throughput[T]`, `.bandwidth`, `.transfer`, `.crossqueuecost`); the inline store limit and alignment (`.inlinestore`, `.inlinealign`: 65536 and 4 on Vulkan, `vkCmdUpdateBuffer`); the capability flags for geometry stages, graphics, mesh pipelines, procedural traversal, ray-tracing pipelines and tessellation (Appendix A, `supports_*`). |
| `config(dev, f)` | `kernel!`, before `compilekernel` | Immutable compiler/feature/ABI/group/math/specialization configuration of kernel f; execution-contracts.md §6. |
| `countsource(dev)` | core, at record and run | `CountRecorded()` (Vulkan) or `CountFromMemory()` (Metal): where a TLAS or BLAS build gets its count. |
| `queuekinds(dev)` | core's queue policy | The queue kinds the device has (graphics+compute, compute, transfer, sparse binding, video decode, present). |
| `memoryclasses(dev)`, `budget(dev, heap)` | core allocator before acquire | Native memory-class/heap facts and available-byte estimate per heap with provenance. Core budget(dev) aggregates distinct device-local heaps. Unified/host/opaque accounting and pageable copies follow execution-contracts.md §12. |
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
| `preparesubmit(ch, parts, waits, signals)` | core before claim | Fallible packet preparation; no enqueue, device allocation or pressure. Reuse prepared packet capacity in steady state. |
| `submitnative!(ch, value, packet)` | core under canonical ChannelState.lock after claim | Enqueue prepared work and signal value; no present, allocation or host wait. Any exception poisons the native failure domain; no counter rollback or reuse of potentially live objects. |
| `presentnative!(ch, request)` | run after ALL stages are accepted/published, outside record locks | Structured outcome and independent retirement evidence; may block in WSI, serialized on canonical queue lock. See execution-contracts.md §§2,4. |
| `passed(ch, value)` | core's `passed(::Token)` (release engine, waits) | Whether the GPU signalled `value` on `ch`. Non-blocking. |
| `waitfor(ch, value)` | core's `waitfor(::Token)` (host reads, host nodes, pressure) | Blocks until `value` passed on `ch`. A GC-safe call (`@ccall gc_safe = true`). Never called with a sync-state, channel, pool or window lock held. |
| `recycle!(ch, objects)` | the release engine, once a submission's token passed | Returns an eager call's command objects to the device's free list. On Metal the free list holds allocator and command-buffer pairs (plan, Metal, "Channels"). |
| `bindpages!(ch, token, binds, waits)` | core's `sparsesubmit!`, under the channel lock, the token claimed | A queue operation binding pages (Vulkan and Metal through the device type's one sparse-binding helper, only APIs that actually support queue-ordered buffer mapping), waiting on `waits`, signalling `token`. Prepared before claim; enqueue errors use the same terminal rule as `submitnative!`. |
| `unbindpages!(ch, token, unbinds, waits)` | as `bindpages!` (also for `trim!(a)` and idle arena pages) | The same, unbinding. |

A `Token` is `(channel, value)` (section 3); core's `passed(t::Token) =
passed(t.channel, t.value)` and `waitfor(t::Token) = waitfor(t.channel,
t.value)`. The last value handed out on a channel is canonical core data
(`channelstate(ch).next`), a field read, not a verb.

### 4.3 Memory

| verb | called by, when | contract |
|---|---|---|
| `rawalloc(dev, kind, bytes, constraint)` | the pool's `tryacquire!`, under the pool lock | One block of `kind` (`Buffers`, `Unified`, `Readback`, `Images`, `Accel`) meeting `constraint` (alignment, usage). A driver call, not a GPU wait; runs no finalizer. The only call site of `vkAllocateMemory` (and of `cuMemCreate`/`hipMemCreate` for blocks). |
| `rawfree(dev, block)` | the release engine | Frees a block whose last use passed. |
| `createreserve(dev, bytes)` | core's growth decision, before any lock | A `Reserve`: sparse buffer, placement-sparse buffer (added to the device's residency set), or reserved address range; no pages mapped. `bytes` is core's `reservesize`. |
| `imagerequirements(dev, spec)`, `placeimage(dev, region, spec)` | core image creation | Query native size/alignment/memory-type/dedicated constraints, then bind an image to a core-owned region. spec and lifetime: execution-contracts.md §3. |
| `placeaccel(dev, path, region)` | the TLAS/BLAS code | An acceleration structure object on a pool region (never a reserve: VUID-VkAccelerationStructureCreateInfoKHR-buffer-03615). |
| `accelsizes(dev, path, n)` | the TLAS/BLAS code | Storage and scratch bytes for `n` instances or a geometry. |
| `acceladdress(dev, as)` | the TLAS code (cell write) | `vkGetAccelerationStructureDeviceAddressKHR` or the Metal resource ID. |
| `objectbytes(dev, native)`, `destroy!(dev, native)` | accounting / release engine | Known native bytes (unknown tracked separately) and destruction after GPU and any presentation retirement. Full creation/destruction contract: execution-contracts.md §3. |

### 4.4 Compile

| verb | called by, when | contract |
|---|---|---|
| `compilekernel(dev, kernel, config) -> (native, compiled::Bool)` | `kernel!` at declaration (eager) and at compile, with no lock held other than the graph lock (compile runs under it) and the kernel's own | Fills the kernel's code if it is missing and its entry for `dev`; the `Kernel` is the only store, no device-type cache. `compiled` says whether this call compiled (core's per-device counter adds it up). |
| `kernelinfo(dev, native) -> KernelInfo` | `kernel!` | Native register/shared-memory facts, group limits and occupancy where available, plus instruction/memory-operation counts from compiled IR. Unavailable native queries yield explicitly marked estimates or bounds, not invented measurements (execution-contracts.md §6). Stored in the device entry; per-argument access is in the `Kernel`. |
| `compiledraw(dev, pipeline, argtypes, formats)` | core's pipeline cache on a render-pipeline miss | Creates the native pipeline for the supplied stages, device-side argument types, fixed state and formats. Cache lookup, invalidation and publication are core's, over device-owned storage. |
| `compiletrace(dev, pipeline, argtypes)` | core's pipeline cache on a trace-pipeline miss | Creates a native ray-tracing pipeline and its shader binding table layout. The same core cache machinery handles its shader/argument key. |
| `compilebindings(dev, s)` | `bindings!` | Descriptor objects for the layout core decided; nothing else. |
| `candidates(dev, f)` | `select` at declaration | The candidates the device type contributes for `f`, with their attributes (`implements`, region compiler). |

Core's `pipeline!(compileverb, dev, pipeline, argtypes, formats...)` owns the
shared cache algorithm. The key includes each shader's code identity and the
compiler configuration as well as the pipeline value, argument types and
formats: editing a shader invalidates its entry even when its function object
keeps the same identity. Cache bookkeeping uses a short lock; compilation,
allocation and pressure happen outside it. A per-key entry serializes the
first native compilation, as a compute `Kernel` does. Equal keys produce one
native pipeline per device. A device verb contains no second cache.

### 4.5 Record

| verb | contract |
|---|---|
| `Builder(dev, ch)` | Command objects from the device's free list. A value that never escapes the call that made it: `record!`, an eager call, a submission's change prefix, or the re-recording of a `BuildRecording` or a window-sized segment. |
| `emitdispatch!(b, node, cmd::CompiledDispatch, deps)` | The barrier in `deps`, then the launch as core decided: direct, indirect, or fixed-capacity with entry predication. A `when!` flag gates its indirect count/range or its entry predicate; no host flag read. |
| `emittrace!(b, node, cmd::CompiledTrace, deps)` | As `emitdispatch!`. |
| `emitcopy!(b, node, cmd::CompiledCopy, deps)` | As `emitdispatch!`. Also a change's copy (`node = nothing`, `deps = Deps((), ())`): an upload region to storage, a cell copy, a move, a restore. |
| `emitstore!(b, dst, upload)` | An inline store within the `caps(dev)` limit in a change prefix: the bytes of the store's upload region (written at the store's call) written into `dst` inside the submission. Vulkan: `vkCmdUpdateBuffer` (at most 65536 bytes per call). Device types without an inline store report a limit of 0 and get a copy from the upload region (`emitcopy!`). Allocates nothing. |
| `emitbuild!(b, node, cmd::CompiledBuild, deps)` | A TLAS or BLAS Build or Update with the count in `cmd` (Vulkan: recorded, `builtcount` for an Update) or from a cell (Metal). |
| `beginrender!(b, node, cmd::CompiledRender, deps)`, `emitpieces!(b, node, natives)`, `endrender!(b)` | A render pass: its barrier (target layouts, and the fixed barrier from compute, copy and build writes to vertex, index, indirect and fragment reads), then its pieces in list order (Vulkan: one `vkCmdExecuteCommands`, the pass begun with `VK_RENDERING_CONTENTS_SECONDARY_COMMAND_BUFFERS_BIT`; Metal: `executeCommandsInBuffer` per piece). |
| `recordpiece(dev, node, region, emit) -> native` | Creates a piece builder, calls core's `emit(builder)` callback, then finishes it; never walks the draws. The callback calls `emitdraw!` per compiled draw, with no barrier. Each piece binds its own pipelines and viewport/scissor (no inherited state). Vulkan: secondary command buffer with rendering inheritance formats; Metal: indirect command buffer. Allocates no device storage: its region was acquired before. |
| `bindindices!(b, node, storage)` | An index-buffer bind inside a render pass, by handle. |
| `emitdraw!(b, node, cmd::CompiledDraw)` | One draw inside a piece, binding its pipeline and argument slots; called by core's piece walk. |
| `emitlibrary!(b, node, cmd::CompiledLibrary, deps)` | A library call written into the recording (cuDNN into the CUDA graph). |
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
| `devicearg(dev, a)` | The one conversion to the type the device's compiler wants (`LavaDeviceArray`, `MtlDeviceArray`, `CuDeviceArray`, `ROCDeviceArray`). Core supplies prepared/frozen layout metadata (root, offset, dims, strides); recorded dynamic descriptors come from argument slots. This verb does not authorize reading live mutable resource state while recording. |
| `vendorview(dev, region, T, dims, strides)` | Borrowed native array over core-owned storage; lease and foreign-channel ordering are core’s. See execution-contracts.md §3. |

### 4.7 Windows

| verb | contract |
|---|---|
| `openwindow(dev, width, height, options)` | Native window + surface/layer; main-thread creation/destruction. |
| `surfacecaps(w)` | Per-surface formats/colour spaces, usage, present modes/families, image counts, extent limits and retirement mode. |
| `presentpassed(record)` | Nonblocking presentation retirement evidence; independent of rendering tokens. |
| `createswapchain(w, width, height; oldswapchain)` | A new swapchain generation. Called by the presenting graph's next run after a resize (`applywindows!`), under the window lock. |
| `acquirenext(sc)` | `(result, index, semaphore)`; may block; called before the run's other locks, with only the graph lock held. |
| `surfacesize(w)` | The surface's current size (for OUT_OF_DATE). |

### 4.8 Textures

| verb | contract |
|---|---|
| `tableentry!(dev, table, index, texture)` | A host write of a descriptor into an unused index of the device-wide texture table (not a recorded command; descriptorsets.adoc:793-807). Called under the table's lock. |
| `samplerentry!(dev, table, index, sampler)` | The same for the device-wide sampler table; a sampler value interns a held SamplerEntry with token-based retirement (execution-contracts.md §11). Called under the table's lock. |
| `imageformat(dev, T)` | The one format table: element type to the device's format, for targets and textures alike. |

### 4.9 Native contract extensions

The following sections of execution-contracts.md extend the verb tables above.
They belong to the same required native boundary; implementation may not treat
them as optional merely because their full signatures are in the companion.
The links supply the full contracts, rather than alternative versions of them.

| boundary | authoritative detail |
|---|---|
| Queue construction/identity, prepared submission, native destruction, image requirements, foreign streams/views, video | [§3: native factories and destruction](execution-contracts.md#3-native-boundary-factories-recording-and-destruction) |
| WSI requests, results and independent retirement | [§§2–4: presentation retirement](execution-contracts.md#4-presentation-retirement) |
| Enabled capability schema, compiler configuration, host visibility and external bridges | [§6: capability schema](execution-contracts.md#6-capability-schema-and-qualification-matrix) |
| Shader ABI, ray programs, groups and SBT | [§7: shader arguments](execution-contracts.md#7-shader-arguments-hits-and-ray-programs) |
| Raster fixed state and timestamp instrumentation | [§8: raster state](execution-contracts.md#8-raster-state-without-duplicated-policy), [§10: profiling](execution-contracts.md#10-profiling-and-counters) |
| Texture/sampler generations, reset, heap facts and native object accounting | [§11: texture/reset](execution-contracts.md#11-texture-restoration-samplers-and-device-reset), [§12: memory](execution-contracts.md#12-memory-classes-pageable-eviction-and-explicit-transfers), [§13: cache ownership](execution-contracts.md#13-cache-ownership-and-pressure) |

## 5. Ownership and locks

The linearization and lifetime arguments are in [execution-contracts.md §1](execution-contracts.md#1-synchronization-and-lifetime-argument). Canonical queue state and submitted-prefix publication are required even on failure.

| state | created by | released by | protected by |
|---|---|---|---|
| device objects, Vulkan instance | the registry | process exit | registry lock |
| pool blocks | `rawalloc` via `acquire!` | the release engine via `rawfree` | pool lock (leaf; no GPU wait inside) |
| a region, an upload region, a plan, a recording | `acquire!` (an upload region at the store's call) / compile / record | `retire!(pool, x, tokens)`, released when the tokens passed | the inbox (lock-free) then the pool lock |
| an array's storage, dims, pending changes | constructor, `resize!`, stores | the last hold (below) | the array's sync state |
| an array's holds | the creator, each graph that names the array, each composite that holds it (a TLAS or BLAS range) | `free!`, `free!(g)`, `free!` of the composite; finalizers of `TLAS`, `BLAS` and render objects push their hold drops through the pool's inbox | atomic compare-and-swap; never 0 → 1 |
| the last hold | the holder that drops it | under the array's record lock: pending changes dropped (their upload regions retired), storage swapped for `Released()`, the old storage retired with the last tokens, the array taken off the eviction list, the cell released | the array's sync state |
| a cell | device-array construction, or compile for a graph transient becoming resizable (fixed graph transients have NoCell) | at the last hold; arena backing remains plan-owned | written only by applied changes and `lengthof` kernels |
| graph nodes, plan | declaration, compile | `free!(g)` / finalizer | graph semaphore; one declaring task |
| a `GPURef`'s slot | `GPURef(dev, v)` | its last hold | the slot's sync state; written only by applied stores (`r[] = v`, scheduled) |
| tokens, outstanding submissions | `submit!`, `sparsesubmit!` | `recycle!` once passed; failed-domain objects quarantined until teardown | canonical ChannelState.lock shared by aliases |
| an arena's mapping and tenants | compile, `bindunmapped!`, `unmapidle!` | `free!(g)` (untenant) | the arena's sync state |
| an arena part's holder | `holdparts!` at the start of a run | `releaseparts!` after its last stage | the arena's condition |
| a window's size, swapchain generation and acquired images | `resize!(win)` / `applywindows!`; the run's acquire | holds of the window and of runs that acquired; an image not submitted for presentation goes back to the window on Again/throw; an enqueued present retains an independent completion hold | window lock |
| a TLAS's or BLAS's slots, members, storage, scratch, capacity, `builtcount` | `push!`, `delete!`, `resize!` of a source, geometry changes | `free!(t)` | the structure's sync state (`change!`) |
| a `BuildRecording`'s command buffer and its re-record mark | the run (marked ones); the change that alters what it recorded (the mark) | retired with its last token when replaced | the structure's sync state, held by the run |
| compiled compute kernels | the `Kernel` at its `CodeInstance`, with one native entry per device | code metadata follows CodeInstance; idle native entries obey the bounded core cache (execution-contracts.md §13); reset creates a new device generation | the kernel's own lock |
| graphics/ray pipelines | device-owned caches with the keys in section 4.4, including shader code identity and compiler configuration | code invalidation, bounded idle-cache pressure or device teardown; active/token holds retain native objects (execution-contracts.md §13) | cache bookkeeping under a short cache lock, native compilation outside it; publication serialized per key |
| texture table entries | an Image storage generation on first compiled use (tableentry!) | that generation's final GPU uses; restored images get fresh indices | the table's lock (a leaf) |
| sampler entries | core weak value cache; compiled users hold SamplerEntry (samplerentry!) | last active/cache/token hold, independently of Image lifetime (execution-contracts.md §§11/13) | table/cache leaf locks |
| selection table | the device; entries published by tune! after verification | device lifetime or code/configuration invalidation | its own leaf lock; no compile, allocation or GPU wait inside |
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
change). A run pends transitive readers’ updates after each accepted segment, before
releasing its expanded record locks, including when a later segment fails. Other submitters
publish their writes and reader updates under the same rule: a concurrent
trace cannot see new geometry before its BLAS/TLAS update is pending.
Between graphs, a plan naming a composite is among its members' plans,
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
inbox, the selection, texture and sampler tables' locks and a
plan's `runend` lock are leaves (`runend` guards the plan's `between` flag,
set under stage 1's record locks and cleared when the last stage is
submitted; a change that would make the plan stale waits on it, holding no
lock). `acquire!` is never called with a sync-state, channel or pool
lock held. No lock is held while vendor code runs; the caller exclusively owns the
foreign stream during the scope. Signal failures quarantine the lease hold.

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
| Every function takes its device; two devices open for the whole suite never mix | `test/suite/areas/devices.jl`; the two-devices check |
| One path: every eager entry reaches `compilekernel`, one submission per launch | `test/suite/areas/eager.jl` |
| Changes are pending until the next use; every STAGE sees the old or the new state, never a mix (a stage can contain several ordered submissions, and accepted prefixes survive later failure, so a change arriving between stages reaches the later ones) | `test/suite/areas/resize.jl` |
| Runs of one graph never overlap (plan, rule 9): each independent root waits for the previous run’s last token on every channel | `test/suite/areas/run.jl` (negative control: the wait removed) |
| Record is pure | `test/suite/areas/threads.jl` |
| A plan is recorded once; in steady state `run!` allocates, records and compiles nothing. Re-recorded are only the `BuildRecording`s a change marked; the parts that recorded the handle of a moved array; a new or changed group of draws and the segment holding its pass's list; and, per swapchain generation, the segments drawing into window-sized targets and the per-image present segments | `test/suite/areas/graphs.jl`, `test/suite/areas/run.jl` |
| A builder is a value | `test/suite/areas/threads.jl` |
| Memory is released only after the tokens of its last uses passed | `test/suite/areas/holds.jl`, `test/suite/areas/free.jl` |
| A write through an array and a read through a view, `reshape` or storage of it in one graph get a barrier: one resource per array state, not per object | `test/suite/areas/views.jl` (negative control: an object-keyed resource map, which finds three resources and no barrier) |
| Cross-graph order through namers and inboxes: a stage that writes a resource waits for the complete last runs of the other plans naming it, a stage that reads one from its inbox for those of its writers | `test/suite/areas/ordering.jl` (negative control: `ordering = Unordered()`) |
| Any thread compiles, records, runs, frees | `test/suite/areas/threads.jl` (THR-01/THR-02) |
| Host nodes: stages, re-entry throws, a throwing host function leaves the device usable | `test/suite/areas/host.jl` |
| A window resize costs no device-wide wait and no recompile within capacity; beyond the capacity the graph recompiles at its next run | `test/suite/areas/windows.jl` |
| Launch kernels never read cells (except head, prepare, the TLAS offsets and records kernels, `lengthof` writers) | compiled-origin cellaccesses audit; KL-10, with native compiler-output qualification (§14) |
| Composites cannot free early, leak or free twice | `test/suite/areas/fuzz.jl`: `push!`, `delete!`, `resize!`, plot updates interleaved with runs on several threads; validation clean; reserved memory back at baseline after `free!` |
| Two graphs that each need 90% of the device run alternately | `test/suite/areas/eviction.jl`: back to back with no host wait, restore before the run, results exact, no recompile; composite members and handle dependents never evicted |
| A RayMakie screen does not recompile when plots are added or removed | RayMakie's compile-counter session: plot inserts and deletes (groups of draws), attribute changes, value ↔ per-element, another device array for an attribute, visibility, camera, layout, resizes within capacity, TLAS moves and count changes; zero compiles. Insert/delete latency at 100, 1 000 and 5 000 plots meets RM-06's bound (at most twice the 100-plot median plus 1 ms); a visibility change records nothing |
| A frame does no host walk over render objects; a run's host time does not grow with the number of resources it names, only with the named resources its stages write and its inbox | `test/suite/areas/run.jl` (plans naming 100, 2 500, 25 000 and 50 000 resources with nothing changed take the same host time per run) |
| No fallible preparation after claim; native enqueue errors poison/quarantine | execution-contracts.md §2; RUN-07/RUN-18, actual FaultDevice host interception and disposable native failure-context qualification |

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
- `refit!(g, t)` for a TLAS whose total a kernel writes is legal on every device type: on Vulkan the structure is built over its capacity; the count-from-memory baseline also uses capacity, so every Update keeps builtcount. An exact device-count native candidate needs its own build/update legality proof.
- A store whose data length differs from the range throws `DimensionMismatch`.
- A call on an array lent to vendor code throws.
- `free!(a)` while graphs hold `a`: the graphs keep running; `a`'s handle throws.
- A window with a zero extent in either dimension (minimized) skips the whole run: nothing is submitted and the changes stay pending; OUT_OF_DATE is a resize to the surface's size, applied and retried by the same `run!`; SUBOPTIMAL presents and marks.
- A window resized beyond its capacity: the graph recompiles at its next run (never a throw).
- An incomplete run returns its acquired image to the window with tracked GPU uses and token-safe signal-semaphore retirement. A complete frame whose present was not enqueued retains its rendered request: run! retries only presentation before edits/recompile, then returns Submitted without replaying GPU work (execution-contracts.md §2).
- An allocation beyond the budget throws `OutOfDeviceMemory` only after the pressure steps: unmap idle arena pages, trim empty blocks, collect garbage, wait for retiring storage, then evict least recently used idle resources and wait, outside every lock, for the tokens their storage was retired with (the accesses in flight and the eviction copies). Pressure never unmaps pages beyond an array's size and never compacts acceleration-structure storage (`trim!(a)` is the explicit call).
- A move of storage a plan recorded by handle re-records the parts that recorded the handle before their next submission; nothing recompiles and nothing waits.
- A change that would make a running plan stale (a length it launches over directly, a temporary it sized outgrown, a capacity change of a structure it sized temporaries from) is applied only after the run has submitted its last stage: the submitter that would apply it waits on the host, never the call (only a run past its first stage is waited for; no fairness: it may wait again if the plan's next run gets past its first stage first). From inside any run (any host function) it throws: it could wait in a cycle. No submitter applies it in between.
- `copy!(a, data)` with a new length does not copy the old contents.
- Eviction: evictable is a resource no composite holds, with no handle dependent, that no running plan names, not lent, with nothing pending; accesses still in flight are allowed (the eviction copy waits for the last write; the storage is retired with the tokens of every access in flight and the copy's). A structure's own arrays are held by it and never evicted. Acceleration structures are not evicted. Eviction works the same on every device type, unified-memory devices included (the Apple M5, the Ryzen AI Max+ 395, NVIDIA's unified-memory chips): the device allocation is freed and the copy goes to pageable host memory, which the OS may compress or swap to disk (core’s bounded HostCopyJob protocol is execution-contracts.md §12). A read-only array copies nothing out. A host read of an evicted array reads the host copy; `free!` of an evicted resource frees the host copy; a plan that names an already-evicted array with no prepared storage is marked for restore and its run returns `Again`.
- A plot with zero elements keeps its render object; its draw count is zero.
- A pipeline key change of a drawn plot (another shader, another attribute element type) is a `delete!` and `insert!` of the plot in Makie: its group deleted and a new group of draws, no recompile. A new Hikari material type can specialize shade kernels, closest-hit groups or both; affected graph/pipeline specializations recompile.
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
| `vkQueuePresentKHR` | `presentnative!` |
| `vkQueueBindSparse` | the Vulkan sparse-binding helper (private `bindsparse!`) used by `bindpages!` and `unbindpages!` |
| `vkCmdUpdateBuffer` | `emitstore!` |
| `vkWaitSemaphores` | `waitfor` |
| `vkGetSemaphoreCounterValue` | `passed` |
| `vkAcquireNextImageKHR` | `acquirenext` |
| `vkCreateSwapchainKHR` | `createswapchain` |
| `vkCmdBuildAccelerationStructuresKHR` | `emitbuild!` |
| `vkGetAccelerationStructureBuildSizesKHR` | `accelsizes` |
| `vkCreateAccelerationStructureKHR` | `placeaccel` |
| Metal `commit` on an `MTL4CommandQueue` | `submitnative!` |
| Metal residency set `addAllocation`/`removeAllocation`/`commit` | the Metal residency helper (private `residency!`) used by `rawalloc`, `rawfree`, `createreserve`, `bindpages!`, `unbindpages!`, and by compile for a plan's own set (its ICBs, argument memory and pipelines; plan, Metal, "Compiled form") |
| `cuGraphLaunch`/`hipGraphLaunch` | `submitnative!` |
| `cuMemMap`/`hipMemMap` | committed unused-region preparation only; not queue bindpages! (execution-contracts.md §3) |
| `cuMemUnmap`/`hipMemUnmap` | native destruction of idle committed VMM storage; never queue unbindpages! (execution-contracts.md §3) |

`vkCmdBeginConditionalRenderingEXT` is not used: `when!` launches and draws
are indirect, with the flag in their count (section 2.4).
`vkDeviceWaitIdle` belongs only to waitnativeidle!(ch) during teardown/debugging
(execution-contracts.md §6); ordinary waitidle(dev) waits a token snapshot.

## 9. Allowlists

- **Mutable globals:** `REGISTRY` (the device registry). Nothing else, in
  core or in a device type.
- **Probes** (`hasmethod`, `applicable`, `hasfield`): none.
- **`@static Sys.is*`:** only in `Mantle.jl`'s include block.
- **`isa`/`typeof` on device or node types:** none.

## 10. Implementation and qualification gates

The linked sections specify target contracts; implementation and native/consumer
evidence must satisfy them before the phase named. Any remaining protocol design
obligation is identified explicitly below, rather than hidden as an open name.

- **Direct restore qualification** (phase 4, ledger F74/P4.A31): qualify native
  host-page access or initialization of a private CPU-visible final destination,
  source/destination retention, retry cleanup and ordered cell publication
  (execution-contracts.md §12). Core-owned HostCopy storage meets the direct
  path's requirements. Local primitive results are in experiments/direct-host-2026-10-21;
  replacement integration, capacity and performance gates remain. A failure
  must first be debugged and fixed; it does not justify a second transfer path.
  A staging fallback is excluded unless the plan's debug-first investigation
  establishes overwhelming evidence of necessity for a required workload.

- **Shader/raster implementation** (phases 5/6): execution-contracts.md §§7/8
  fixes hit groups/SBT, typed hit results, sampling, stencil, resolves and blend
  equations. Implement and qualify the shared paths; no native result is inferred
  from these contracts.
- **Capability implementation/measurement** (phases 1/8): the required schema
  and compile-key contents are fixed in execution-contracts.md §6. Supply actual
  native measurements/estimates and KI projections, then qualify each backend.
- **Profiling qualification**: implement the bounded nonblocking sampling
  contract in execution-contracts.md §10; validate timer units, wrap and native
  query/event reuse without disturbing the active plan.
- **Native qualification** (phases 3/7/9/10): execution-contracts.md now fixes
  enqueue failure, recycling, wrappers and presentation retirement. These need
  native implementations and disposable-context failure tests. Legacy Vulkan
  shutdown has the documented specification limit; idle is not a present fence.
- **Cross-device optimization**: devicecopy! has an explicit synchronous
  read/store baseline and qualified peer candidates (execution-contracts.md §12); graphs and implicit
  mixed-device arithmetic remain within one device domain.
- **Selection qualification** (phase 6): implement SelectionKey/Candidate and
  the independent-reference tuning harness in execution-contracts.md §9.
- **Native ABI implementation**: implement the exact factories, object-byte and
  destruction verbs, scoped vendor views and surface policy in
  execution-contracts.md §3; these are specified, not implemented by this review.
- **Teardown qualification** (phase 8): implement reset_device! as a closed
  generation, never a revival of old handles (execution-contracts.md §11).
- **Texture/sampler qualification** (phases 5/6): fresh indices on restore and
  weakly interned SamplerEntry holds/retirement are fixed in execution-contracts.md §11.
- **Compiler/debug qualification**: source-cell origin checks and the live-state
  record guard have concrete THR-04/KL-10 gates (execution-contracts.md §14).
- **Recording counters**: Diagnostics.records is the canonical native-record
  count; temporary encoding has separate prefixrecords/eagerrecords counters and
  there is no duplicate Plan rerecord counter (execution-contracts.md §10).

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
| `bestshape` | selection table; explicit tuning in section 2.4, SelectionKey/Candidate and tuning in execution-contracts.md §9 |
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
| `collect!` | Graph(profile=true), timings(g), shared diagnostics (execution-contracts.md §10) |
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
| `devicecopy!` | copies: `copyto!` as an operation / `emitcopy!`; explicit synchronous devicecopy! baseline (execution-contracts.md §12) |
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
| `emitloophead!`, `emitloop!` | go as backend algorithms: core lowerloops! emits ordinary prepare/dispatch nodes; optional native loop candidates use the same selection contract |
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
| `gpupasstime!` | Graph(profile=true), timings(g), shared diagnostics (execution-contracts.md §10) |
| `hold!` | goes |
| `holdleaves!` | goes at emit |
| `hostspan` | core: a host view of host-visible memory (Readback, upload regions) |
| `imageusage` | becomes part of the Images kind's usage constraint |
| `indexbuffer` | goes: indices are a MantleArray bound by `bindindices!` |
| `indirectindex` | goes: launch commands are slots core lays out |
| `indirectslot` | goes: as indirectindex |
| `initbackend!` | goes: hooks take the device |
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
| `makeprofiler` | Graph(profile=true), timings(g), shared diagnostics (execution-contracts.md §10) |
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
| `present_frame!` | goes: present is the separately retired `presentnative!` operation in run! |
| `profiled!` | Graph(profile=true), timings(g), shared diagnostics (execution-contracts.md §10) |
| `Profiler` | Graph(profile=true), timings(g), shared diagnostics (execution-contracts.md §10) |
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
| `reset_device!` | closed-generation debugging teardown (execution-contracts.md §11); implement without reviving old handles |
| `resetkernelcompiles!` | core: the compile counter |
| `resetrecording!` | goes: command objects back to the free list |
| `resourcekind` | core: resource kinds of sync states |
| `retire!` | core: `retire!(pool, x, tokens)`; `retire!(::SubmitChannel)` goes |
| `runscalls` | goes |
| `Sampler` | user: a value in the device-wide sampler table |
| `screenshot` | becomes `Array(win)` |
| `set_anyhit_pipeline!` | typed hit-group any-hit/intersection values and core SBT layout (execution-contracts.md §7) |
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
| `supportspredicate` | goes: core selects indirect counts/ranges or capacity entry predication from enabled capabilities; no separate backend policy (section 2.4) |
| `Surface` | goes: a window is a target itself |
| `syncbackend` | goes |
| `target_extent` | goes |
| `target_format` | goes: `imageformat(dev, T)` |
| `target_image` | goes |
| `target_view` | goes |
| `Texture2D` | goes: an `Image` gets its table index on its first use by a shader |
| `timestamps` | timestampinfo/preparetimestamps/readtimestamps!/emittimestamp! for optional sampled profiling (execution-contracts.md §10) |
| `trace_closest_hits!` | goes: the immediate trace_* verbs |
| `trace_closest_hits_anyhit!` | goes |
| `trace_closest_hits_anyhit_indirect!` | goes |
| `trace_closest_hits_indirect!` | goes |
| `trace_rays!` | goes: `trace!(dev or g, …)` |
| `trace_rays_indirect!` | goes |
| `upload!` | goes: stores are pending changes (`Store`; inline ones written by `emitstore!`) |
| `use_bindings!` | goes: bindings compiled with the plan (`compilebindings`), textures through the device-wide table; draw-binding code whose last caller is the walked draw in `execute!` |
| `use_frozen_kernels` | goes: the `Kernel` at its `CodeInstance` holds the code and each device's native handle |
| `vertextouches` | stays: a shader stage's access at declaration |
| `waitfor` | verb: `waitfor(ch, value)`, GC-safe; core's `waitfor(::Token)` calls it |
| `waitfor!` | `waitfor!(::Stamp)` goes |
| `waitidle` | core token snapshot wait; native teardown idle is specified in execution-contracts.md §6 |
| `Window` | user: `Window(dev, w, h)` |
| `withpredicate` | folds into builder verbs: core supplies the indirect count/range or capacity entry predicate for `when!` |
| `workgroupsize` | goes: KernelInfo's group size |
