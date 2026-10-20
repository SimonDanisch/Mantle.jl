# Mantle: one runtime model for every device

## What Mantle is for

One Julia process runs a video editor, RayMakie, GPU simulations, DNNs, ray
tracing and raster graphics side by side, on Vulkan, Metal, CUDA or HIP, at the
speed of the fastest open implementation of each piece. Two consequences:

- **All of them share VRAM.** A DNN, a ray tracer and a renderer in one process
  must not each hold their peak: memory that is free in one graph is usable by
  another, and memory nobody needs goes back to the device.
- **Every decision is made once, in core, for every device type.** The graph
  knows what runs, what it touches and in which order, so core decides
  placement, lifetime, queues, ordering, submission splits and fallbacks. A
  device type answers questions and implements small verbs on its driver
  objects.

## Rules

1. **Core decides; device types implement verbs.** A device type never walks a
   plan, reads the dependency graph, or decides an order, a barrier, a split, a
   queue, a lifetime, a residency set or a fallback. Capabilities are
   questions core asks; core picks the construction from the answer.
2. **No duplication.** Where two device types need the same thing, it is core's,
   with a verb for the part only the driver knows.
3. **Dispatch, not probing.** No `isa` chains on device or node types, no
   `hasmethod`/`hasfield`/`applicable`/`try`-`catch` probes, no Symbol kinds.
   One function name, methods per type.
4. **Mantle owns device memory.** Every allocation is placed by core's pool or
   counted and released by it. Vendor allocators (Metal.jl, AMDGPU, CUDA.jl,
   Vulkan dedicated allocations) are bypassed.
5. **No ambient state.** Every function takes its device, graph or channel.
   Nothing is found by task-local storage, a thread id or a "current" object.
   A call reaches a graph only through its arguments (graph arrays, the graph
   device, `adapt(Mantle.Device(g), x)`); with device arrays it runs now on the
   device. A global, task-local or scoped "current graph" (or group) that
   assigns calls implicitly is forbidden. The one process-level state is the
   device registry:
   `Device()` and `devices()` return the same object for the same hardware
   device.
6. **One path.** An operation on a device and the same operation on a graph run
   the same code; which of the two happens is decided by where the arrays live.
   There is no second implementation of any operation.
7. **One call, one graph.** A consumer declares one graph per call and never
   partitions it. Host work inside a call is a host node; Mantle schedules it.
8. **Changes are scheduled, not executed at the call.** `resize!`, `trim!`,
   `setindex!`, `copyto!` and `copy!` into device arrays, `GPURef` stores and
   acceleration structure changes only enqueue a request at the call (the
   data of a store is taken then): the requested size is what `length(a)`
   reports, and nothing physical changes. New storage, pages, the copy of the
   contents, the cell write and the retirement of the old storage are done by
   the next submission that touches the resource (a run, an eager call, a host
   read), which allocates what they need before its locks and applies them in
   front of its own work, inside the same submission, in call order across
   resources. A window resize marks the graphs that render to the window;
   their next run applies it. Runs submitted
   before a change see the old state, runs after it the new one. Nothing is
   detected, compared or swapped on the host.
9. **Runs of one graph never overlap.** A run's first segment waits on the GPU
   for the last tokens of the graph's previous run.

## The model

### Devices

Mantle has devices. A user with one GPU calls `Device()` and never meets another
concept.

| device type | what it is | available |
|---|---|---|
| `LavaDevice` | a Vulkan device through Lava: AMD, NVIDIA or Intel GPUs, and lavapipe where the system's Mesa installs it | always, on Linux and Windows |
| `MetalDevice` | an Apple GPU through Metal 4 | always, on macOS |
| `CudaDevice` | an NVIDIA GPU through CUDA | after `using CUDA` (package extension) |
| `ROCmDevice` | an AMD GPU through HIP | after `using AMDGPU` (package extension) |

- `Device()` is this machine's default: `LavaDevice` on GPU 0 (never a CPU
  driver) on Linux and Windows, `MetalDevice` on macOS.
- `devices()` returns every device: each Vulkan physical device as a
  `LavaDevice`, plus `CudaDevice`s and `ROCmDevice`s when their packages are
  loaded. Driver objects are created on first use.
- **One object per hardware device.** `Device()` and `devices()` return the
  same Julia object for the same hardware device and driver, so an editor and
  a package that both ask get one pool. The registry of opened devices is the
  one process-level state; it also holds the list of loaded device types and
  the Vulkan instance.
- CUDA and HIP are add-ons, used where Lava lacks an intrinsic or a library
  (cuDNN, cuBLAS, rocBLAS). One GPU through two drivers is two devices with two
  pools.
- Lavapipe (Mesa's CPU Vulkan driver) is a `LavaDevice` on another physical
  device. `using Lavapipe_jll` (26.2.2 in General) makes its ICD visible to the
  loader on machines whose system Mesa does not install it; it must be loaded
  before the first `Device()` or `devices()`, since the registry creates the
  Vulkan instance then and the loader reads driver manifests at instance
  creation (to be checked against the loader source). CI and machines without
  a GPU test the real Vulkan path on it.
- A device is the KernelAbstractions backend: `Device <: KA.Backend`,
  `KA.get_backend(a) = device(a)` for an array on a device.
- **A graph is not a device, but it is a KernelAbstractions backend.**
  Arrays declared into a graph live on its `GraphDevice` (`Mantle.Device(graph)`).
  `KA.get_backend` of a graph array is the graph: a KA launch over graph arrays
  declares a kernel node and `KA.allocate` gives a transient, so KA libraries and
  GPUArrays' generic code run inside a graph. Host reads of contents throw at
  declaration (there are none before a run). `adapt(Mantle.Device(graph), x)`
  re-homes device arrays, or a struct of them, as zero-copy aliases the graph
  names, so code that finds its target through its arrays declares into it.

"A backend" in this plan means the code for one device type (`src/vulkan`,
`src/metal`, `ext/`); it is never a user-facing concept or type.

### Arrays

`MantleArray{T,N,L} <: AbstractGPUArray{T,N}` is the one array type, for every
location `L` (a device or a graph, per device type). A view is the same
type with its own offset and dims (no role parameter); read-only, unified
and resizable are runtime properties of the resource, not types.

| constructor | lives on | contents |
|---|---|---|
| `MantleArray(dev, T, dims...)` | a device | its own pool region; operations on it run now |
| `MantleArray(location(a), …) do p … end` | where `a` lives | a temporary inside an operation: on a device a region retired with the call's token, on a graph a transient |
| `MantleArray(g, T, dims...)` | the graph | a transient: contents exist only during a run; placed in the graph's arena |
| `MantleArray(dev, host; readonly = true)` | a device | read-only: keeps a reference to `host` (no copy), from which a restore after eviction uploads again; an mmapped `host` is a plain `Array`, whose pages the OS drops and re-reads, so the array can leave even RAM; a write throws at the call |
| `KA.allocate(dev, T, dims...; unified = true)`, `MantleArray(dev, T, dims...; unified = true)` | a device | in unified memory (KernelInterface's `unified`): a graph writes it, `Array(out)` after the run is a wait and a memcpy |
| `MantleArray(g, T, win)` | the graph | sized like a window; its length follows the window |
| `MantleArray(g or dev, …; capacity = n)` | either | resizable from the start, with a reserve for `n` elements, like `sizehint!`: growing past it moves the storage; needed when a kernel writes the length (no host `resize!` tells core how much to reserve) |

- **Where contents live:** contents used only within one run are a transient;
  contents used again in a later run, by this graph or anything else (the
  previous frame, a recurrent state, a film, weights, outputs, a KV cache shared
  between calls, data the host supplies per run), are a `MantleArray(dev, …)` the
  graph names. Graph arrays have no `persistent` or `hostwritten` option.
  Transients of a graph share its arena, and graphs that do not overlap share
  arenas: they cost the largest tenant, not the sum.
- **Each array owns a cell**: its address and its size. The cell lives in
  the device's cell arena and is the only place those values are written; every
  graph that uses the array gets them through its head node (Run). This is
  how one array is shared by every graph that uses it. A transient that
  is not resizable has no cell: its address is an arena offset fixed by the
  plan.
- **Views share their parent's state.** A view, reshape or transpose
  of a fixed array (GPUArrays' `derive`) refers to its parent's state
  (storage, size, cell, holds, and the sync state with its freed mark), so a store through a view is
  ordered like one into the parent, a `resize!` of the parent reaches it, and
  a freed parent makes its views unusable. A view has no hold and no finalizer
  of its own; `free!` and `resize!` on a view throw. Its argument slots carry a
  static offset and its own dims and strides.
- **A resource is the array's state, not an object.** `dependencies!` and
  `barriers!` see one resource for an array, its views and reshapes, however
  many objects name it, so a write through one and a read through another are
  ordered. An argument that reaches device memory but is not a `MantleArray` (a
  vendor array, an array escaping `withview`, a raw backend array) throws at
  declaration; it never becomes a resource of its own.
- **Every array is fixed until its first `resize!`.** A
  fixed array lives in a pool region and launches over it are direct. The
  first `resize!` call makes it resizable (a host fact: plans compiled after
  it launch over it indirectly). What happens to the storage is decided when
  the submission that applies the resize does it: if the new size fits its
  current storage, nothing moves: the region becomes a doubling pool region in place,
  and only the cell is written. Beyond it, a large array on a device with
  sparse residency moves into a reserve: address space whose pages are mapped
  as it grows (sparse residency on Vulkan, placement-sparse buffers on Metal,
  reserved address ranges on CUDA/HIP), so later growth keeps its handle and
  address. Any other array moves into a pool region of twice the need, and
  each later doubling is a move. Which arrays are large, and how much address
  space a reserve gets, core computes from `caps(dev)` (the sparse page size,
  the device's sparse address space). Addresses and sizes reach recorded work
  through the cell. Only what a plan recorded by value recompiles it: a direct
  count (the array was fixed at compile) and a temporary sized for fewer
  elements. A buffer handle in a command (after a move) re-records the parts
  that recorded it; nothing recompiles. This re-record is the one mechanism
  for every handle dependence (index buffers, Vulkan buffer copies, restores
  of handle dependents). Shrinking unmaps nothing, so a view never reads unmapped memory; neither
  does memory pressure. `trim!(a)` (name open) is the explicit call that
  unmaps a reserve's pages beyond the array's size; after it, a view past the
  size reads undefined values. Details and the pseudocode:
  `docs/resizing-and-raytracing.jl`.
- **On a graph device** `KA.get_backend` is the graph, so KA launches, broadcasting,
  Mantle's methods and GPUArrays' generic code all declare. `getindex`,
  iteration and every other host read of contents throw at their line. Launch
  sizes such code computes on the host are recorded per node; a resize of an array
  behind one makes the next compile throw. What the host computes from its own
  values (constants, a host random draw) becomes a constant of the plan, which
  cannot be detected: values that change per run are `GPURef`s.
- **Lengths** follow Julia's API: `length(a)` is the real length; for a
  length a kernel writes it waits for that kernel and reads it back, and during
  the declaration of the graph that writes it, it throws. Launches use the length
  on the device. Kernels are bounds-checked unless `@inbounds`.
- **GPUArrays 12.1** (in the environment) gives every `AbstractGPUArray`
  broadcast, `map!`, `fill!`, `copyto!`, indexing with index arrays, `repeat`,
  `adjoint`/`transpose`, `tril`/`triu`, a generic matmul, random numbers,
  `mapreducedim!` (`host/mapreduce.jl:18`), `accumulate!`, `sort!` and `findall`
  through AcceleratedKernels, all as KA kernels on `get_backend(dest)`. It
  requires `storage`, `copy`, `derive`, host copies and `Adapt`. It has no BLAS
  or FFT.
- **Mantle's methods** on `MantleArray`: `mul!`, `fft` and the other operations
  the selection table serves, plus whatever is faster than GPUArrays' generic
  version; everything else is GPUArrays' generic code, on devices and, through
  the graph backend, on graphs.
- **Device side:** one conversion verb per device type, `devicearg(dev, a)`, to
  the type its compiler wants (`LavaDeviceArray`, `MtlDeviceArray`,
  `CuDeviceArray`, `ROCDeviceArray`). Views, reshapes and transposes resolve to
  parent region, offset and strides in one place.
- **Vendor views:** `CuArray`, `MtlArray` and `ROCArray` over the same memory,
  without copying or owning it, handed out in a scoped block with an explicit
  stream: `withview(CuArray, a, stream) do v … end` (name open). `withview`
  takes a hold on `a`, first applies its pending changes and any restore in a
  submission of its own, and makes the stream wait on that token and on
  Mantle's accesses to `a`; at the block's end the stream writes a token (the
  stream is a channel like Mantle's own), Mantle's later uses of `a` wait on
  it, and the hold is dropped with it, so a `free!(a)` during the block
  releases nothing the stream still uses. No Mantle lock is held while vendor code runs. Vendor code
  allocates from the vendor's pool; Mantle sees that only in the device's
  memory budget.

The vendor array types are not the primary array: every operation on them
that allocates (`similar`, broadcast results, `copy`, library workspaces) goes
through the vendor's allocator and stream, which breaks rules 4 and 6 for all
eager work, and Vulkan has no vendor array package.

### Resources

| resource | memory |
|---|---|
| `MantleArray` | pool region (`Buffers` kind), or an arena slice for transients |
| image / texture | pool region, `Images` kind |
| acceleration structure | pool region, `Accel` kind; build and refit scratch is the structure's own, sized for its capacity and replaced with it (not a graph transient: a capacity change after compile would outgrow it) |
| `Window` | its swapchain or drawables, counted |
| `GPURef` | a value box: ONE slot in device memory whose address every plan and group of draws bakes; `r[] = v` is a scheduled store into it, applied like any change. Values only: an array is passed as itself |

Acceleration structures are declared with `TraceRead` by traces and ray-query
dispatches, and with `TraceBuild` by builds and refits (both access kinds are
in `sync/usage.jl`).

### Operations

An operation is a Julia method on `MantleArray`. Its body launches kernels with
`dispatch!(location(a), kernel, args, ndrange)`; on a device that runs now, on a
`GraphDevice` it adds a kernel node to the graph. There is no `lower`, no
registration and no expansion at compile: the graph holds exactly the kernels
the operations launched when they were called.

```julia
g  = Graph(dev)
x  = MantleArray(g, Float32, n)
fill!(x, 0f0)                           # adds a kernel node to g
mul!(y, W, x)                           # W = MantleArray(dev, weights): selection runs here
y .= gelu.(y .+ b)                      # broadcast: one kernel node
accumulate!(+, c, y; dims = 1)
resizeplanar!(x, img; mean, std)        # GPUFiltering's own method on MantleArray
fill!(a, 0f0)                           # a = MantleArray(dev, …): runs now

dispatch!(g, kernel, (a, b), ndrange)                        # a raw kernel
when!(g, flag, kernel, (a, b), ndrange)                      # one launch, runs only where flag (a Bool on the GPU) is true
dispatch!(g, HostCall(align), (durations, idx); writes = (idx,))
step = Group(g, "step")                                      # a value; its place in the order is here
dispatch!(step, kernel, (a, b), ndrange)                     # declared into the group, anywhere; runs at its place
repeat!(step, n)                                             # the group runs n times per run
repeat!(step, maxiters; while_nonzero = queue.size)          # …until a count on the GPU is zero
clear!(g, target, (0f0, 0f0, 0f0, 1f0))                      # the load operation of the pass that draws next into target
draw!(g, target, pipeline, args, n)                          # a draw into a window, an image or a tuple of them
plot = Group(g, "plot")                                      # draws that will be removed: a group
when!(plot, target, visible, pipeline, args, n)              # a draw that runs only where visible is true
delete!(g, plot)                                             # a group of draws goes without a recompile
trace!(g, pipeline, accel, args, ndrange)
sim_g = adapt(Mantle.Device(g), sim)                         # zero-copy aliases g names
step!(sim_g)                                                 # a KA library declares into g
```

- **Selection is a call in the method.** `mul!` on MantleArrays asks the
  device's selection table (key: function, element types, shapes, layout,
  alignment) and launches the chosen candidate. The graph's device is known at
  declaration, so the choice is made there. Candidates:
  1. a kernel compiled by us (ours, MLX's Metal kernels, CUTLASS or CUB through
     NVRTC, Composable Kernel through hipRTC): a kernel node;
  2. a vendor API that writes into our graph (cuDNN's
     `cudnnBackendPopulateCudaGraph`): a library node;
  3. a closed library (cuBLAS, rocBLAS, hipBLASLt, an `MPSGraphExecutable`): a
     foreign node, whose access is declared by the method.
  A device type contributes candidates; measurement fills the table per device;
  an unmeasured entry uses core's cost estimate. In a `Group` and under `when!`
  only kernel candidates qualify, since a foreign node cannot run
  under a condition the device decides. A library's workspace is a
  transient in a graph and a retired region when eager; libraries never
  allocate for themselves.
- **Nodes keep their origin**: the Julia function and arguments of the call
  that launched them. Optional rewriters match on it (Compile).
- **No length is baked into a node.** Operations launch over
  `launchrange(a)`, a reference to the array that compile resolves: direct with
  the length while the array is fixed, indirect after its first `resize!`.
  Temporaries are sized by size expressions evaluated at compile
  (`SizedBy(nblocks, a)`: twice the current length, or the capacity for a
  length a kernel writes); `resizing-and-raytracing.jl` B.7.
- **Where a call runs:** if any argument lives on a graph, the call declares
  into that graph, and the device arrays among its arguments are named by the
  graph (they get a hold). Otherwise it runs now on the arguments' device.
  Arguments from two graphs, or from two devices, are an error. So an output
  the host reads after a run is a device array the graph writes; the host
  reads it with `Array(out)`, an eager read ordered after the run.
- **Groups are places in the order.** `Group(g or group, name)` is a value
  whose place is the end of its parent's order (the graph, or another group)
  when it is made. Work is declared into it by passing it (`dispatch!(group,
  …)`, `trace!`, copy kernels, `clear!`, `draw!`, `when!`, `Group(group, …)`)
  anywhere in the code, and runs at the group's place whenever it was declared.
  A group holds GPU work only: a build, image copy or host node has no method
  for a `Group` (a `MethodError`). A group is not an owner: arrays and
  allocations belong to the device or the graph, so library code that finds
  its target through its arrays declares into the graph. A group has three
  uses: `repeat!(group, n)` loops it; `delete!(g, group)` removes it;
  `sort!(group; by, lt, rev)` reorders the groups it holds. There is no
  `do`-block body (it invites host code between dispatches) and no mode state
  on the graph.
- **Control flow is launches only.** A repeated group holds kernel launches,
  ray-tracing pipelines (`trace!`) and copy kernels; a draw, a clear or a group
  in it throws, at `repeat!` or when declared later, so loops do not nest.
  Builds are explicit constructions; a render pass or
  image copy would need the image's layout to be the same on the way in and out
  of every iteration; a host call cuts the recording. Hikari's bounce
  loop traces inside `repeat!` (`Hikari/src/integrators/volpath/graph.jl:433`).
  A launch in a body the device may end is an indirect launch whose count a
  prepare kernel sets to zero (conditional rendering does not cover
  `vkCmdTraceRaysKHR`, and Mantle uses it for nothing): the loop head zeroes the
  body's counts once the flag is 0, and the loop start restores them in every
  run. A conditional launch works the same way (`when!`, below). `copyto!(dst,
  src)` between arrays in a graph is a copy kernel node, so it is allowed; copy
  nodes (images, host transfers) are not. Before each iteration a loop-head
  kernel writes `i` into the argument slots of the body launches that take it;
  they do not read it from a buffer. Conditional execution is one launch at a
  time, `when!(g or group, flag, kernel, args, ndrange)`: `flag` is a
  `GPURef{Bool}` or a one-element `Bool` array, enforced by the method's
  argument types. The launch is indirect, and whoever writes its launch command
  multiplies the count by the flag: the head for a flag the host writes, a
  prepare node right after the kernel that writes a kernel-written flag. This
  mechanism is the same on every device type. No conditional rendering is used,
  so the flag is never a handle dependence (it may move like any array), and
  Metal needs nothing extra. A flag in unified memory serves both sides: a
  kernel writes it, the prepare node reads it on the GPU, and the host reads it
  after the run without a copy. Several launches under one flag are several
  `when!`s. Conditions exist in graphs only. A build or copy that must be
  conditional is constructed explicitly by the application.
- **Host calls are nodes** (`HostCall(f)` with the arguments it writes). Mantle
  schedules them (Run, "Host nodes"); the user never splits a graph.
- **Lengths decided on the GPU** use a resizable array. The kernel that decides
  the length writes it into the array's cell through `lengthof(a)`, and
  declaring that launch adds a prepare node after it, which
  writes the launch commands of every later launch over the array.
  Head and prepare kernels write the launch command, the KA `ndrange` values
  and the dims slots of every launch over the array.
  `launchrange(a)` launches over the array's current length, indirectly once
  the array is resizable; it is the one name for this. A device-decided length
  needs no `capacity`: the kernel writes within the array's current
  storage, bounds-checked unless `@inbounds`. On the host, `length(a)` and
  `size(a)` are the real length: they wait for the kernel that writes it
  and read the cell back, and throw during the declaration of the graph that
  writes it. There is no `devicesize`. `Array(a)` copies the cell and the data
  in one submission.

### Kernels: `Mantle.Kernel`

```julia
mutable struct Kernel{F,N}               # N: the device type's native handle type
    f::F
    access::Vector{Touch}                 # per argument, set at declaration
    code::Union{Nothing,CompiledCode}     # SPIR-V | AIR | PTX, set at compile
    devices::Vector{DeviceKernel{N}}      # one per device it ran on, set at first use
    lock::Base.Semaphore                  # each stage writes its field once under it
end
struct DeviceKernel{N}
    device::Device                        # found by identity (===)
    native::N                             # VkPipeline | MTLComputePipelineState | CUfunction | hipFunction
    info::KernelInfo
end
```

- **Why two types:** a `Kernel` belongs to one compiler configuration, so to one
  device type, and its code serves every device of that type (two GPUs and
  lavapipe share one SPIR-V module); a `DeviceKernel` is what each device builds
  from it. One device type means one native handle type `N`, so the vector is
  concretely typed.
- **Where it lives:** attached to the kernel's `CodeInstance` through
  `GPUCompiler.cached_results`, one per compiler configuration, the way CUDA.jl
  keeps `CUDACompilerResults` with each context's `CuFunction` inside. Julia
  invalidates it with the code (an edit, Revise) and keeps it in package
  images; no native handle is added while generating output. Core and device
  types keep no other kernel table, and nothing of it is global.
- **Stages fill it in place:** the access per argument at declaration
  (`argument_usage`: a declaration next to the function for a call with no
  Julia body, else `kerneltouches`, the taint walk over the kernel's typed Julia
  IR through the device type's interpreter); the code at compile; a
  `DeviceKernel` at the first use on a device. Each is written once.
- **Device-type verbs:** `compilekernel(dev, kernel, config) -> (native, compiled::Bool)`
  fills the code if it is missing and the device's entry; `kernelinfo(dev, native) -> KernelInfo`;
  and `config(dev, f)` (name open), the launch configuration the device type
  picks for a kernel (workgroup size). `compiled` says whether this call
  compiled, which is what core's per-device compile counter adds up. No lock
  other than the graph lock and the kernel's own is held while a kernel compiles.
- **No teardown per kernel.** A device's entries stay with the kernel; a reset
  (`vk_reset_device!(debug = …)`) makes a new device object, and the old entries
  are never found again (CUDA.jl does not reset devices at all: its
  `device_reset!` is a deprecated no-op).
- **A group size the caller gives is kept**: KA's `kernel(backend, workgroupsize)`
  and `dispatch!(…; group)` launch with exactly that size; `config(dev, f)` picks
  one only where none is given. Its default for 2-D and 3-D ranges is open:
  one row of 64, or a 2-D tile.
- **`KernelInfo`** has no optional fields: registers, shared memory, group size
  limits and occupancy (from the API where it reports them, from the device's
  limits where it does not), and the instruction and memory-operation counts of
  the compiled IR.
- **Core uses it for** launch geometry (one calculation,
  `ki_launch_extents`), per-node cost estimates and the submission split.
- The per-argument access lives in the `Kernel`; there is no separate access
  cache.

### The graph and its plan

User code declares a graph and calls `run!(g)`, which is `run!(g.plan)`: the
plan is core's internal compiled state, one per graph. The first `run!`
compiles and records. A graph is never replaced for a structure change
(adding a plot never throws the graph away). Declaring into a compiled graph,
or `delete!(g, group)`, only marks the plan: the next `run!(g)` compiles a new
plan from the graph's nodes and retires the old one after its last run. Groups
of draws are the exception: the next run records them on their own and
recompiles nothing (Graphics). The graph, its windows, the resources it names and
the device arrays it reads stay; what must survive runs and recompiles (a
film) is a device array, since a graph array's contents live only during a
run. `free!(g)` releases the plan
and drops the graph's holds; a finalizer on the graph does the same for graphs
nobody freed. Introspection: `submissions(g)` (segments and cuts, including
those host nodes cause), `timings(g)`.

What can change after the first run, and what each change costs:

| change | made by | what happens |
|---|---|---|
| a `GPURef` | `x[] = v` | pending: a store into its slot, applied like any store by the next submission that touches it; it marks no plan |
| a device array's contents | `a[r] = data`, `copyto!(a, data)`, `copy!(a, data)` (resizes a vector to `length(data)`, as Base's `copy!`, without copying the old contents) | pending: a small store (within the device type's inline limit, `caps(dev)`: 64 KiB at 4-byte offsets and sizes on Vulkan) is written inline by the submission that applies it (Run); a larger one is a copy from its upload region. Either way the call copied the data once into an upload region |
| an array's size | `resize!(a, dims)` | pending: the first time, a cell write if the new size fits the storage, else a move with a copy into a reserve or a doubling pool region (Arrays); later, pages mapped in a reserve or a move to a doubled region, and a cell write; plans that recorded a direct count or sized a temporary for fewer elements are marked stale at the call; parts that recorded its handle are re-recorded when the move is applied |
| a reserve's pages beyond the size | `trim!(a)` (name open) | pending: the pages unmapped, retired with the token of the submission that unmaps them |
| a window's size | `resize!(win, w, h)` from the window's event | stored; the next run of the graph presenting to it creates the new swapchain generation with its window-sized images at the new size, queues the cell writes of window-sized arrays, and re-records the segments that draw into window-sized targets and the per-image present segments (Windows); beyond the window's capacity the graph recompiles at that run |
| an instance's transform or mask | `tlas[h] = t`, hiding an instance | pending: records rewritten, a refit (Update) with the count of the last build |
| an acceleration structure's instance count | `push!`, `delete!`, a range's array length | pending: records rewritten, a build with the exact count; beyond the instance capacity new storage (capacity doubles, capped at the device's `maxInstanceCount`); the change marks the `BuildRecording`s it alters, and plans re-record only those at their next run. A total past `maxInstanceCount` (16 777 215 on every Vulkan GPU here) throws at the call that raises it: `push!`, a `resize!` or `copy!` of a source |

Nothing on this list is detected: each is done by the function that makes the
change. A change is pending on the resource's sync state, in call order.
Whoever next submits work that touches the resource takes its pending changes
under the record's lock and records them in front of its own work in the same
submission (pages first, through the sparse queue, waited for by that
submission); their accesses carry that submission's token, and upload regions
and replaced storage are retired with it. In a run they ride in front of the
first segment of the stage; every later segment of that stage that touches a
record whose changes were taken waits on the first segment's token. A later store that covers an earlier
pending one replaces it. So a plot updated every frame costs no submission of
its own: the frame's run applies it. At the call only the requested size
changes (what `length(a)` reports), under the array's lock; the storage is
changed only by the submission that applies the change, at its commit.
Everything that submits allocates what the pending changes of what it locks
need before taking its locks (storage of the requested size or pages, a
structure's capacity, restore storage; a store's upload region was acquired
at its call), and
applies them under the locks with what it allocated; if a change arrived in
between and what it allocated no longer fits, it starts over. That
allocation is the only one a change causes, and the only place where memory
pressure from changes happens. Runs never read the host's sizes: they get
sizes from cells.
`resize!(a, dims)` keeps contents as linear memory, which preserves them when
only the last dimension changes; window-sized arrays' contents are undefined
after a resize. `resize!(a, n)` sets the last dimension.

### Compile

Compile inspects the graph and the device; record and run may not. It is a
fixed sequence of steps, each a function applied to a `CompileState`:

| step | produces |
|---|---|
| copy, head, `layout!` | the plan's own copies of the graph's nodes, a loop's body included (a recompile never touches an older plan's), the head node first and the `Bound` resources (below) |
| rewriters (only those the graph opted into) | the node list after optional rewrites, for example `FuseEpilogues`, `MergeRegions(attribute)` |
| `dependencies!` | an edge "q before p" wherever two nodes conflict on a resource, reduced so each node lists only its direct predecessors (what a CUDA/HIP graph node takes); kept in the plan |
| `compile!(s, node)` for every node | the node's compiled data: a `Kernel` per kernel node, compiled draws and traces, library setup; one method per node type, a loop node's method compiles its body |
| `costs!` | an estimated time per node from its `KernelInfo`, ndrange and declared bytes |
| `schedule!` | node order and queue per node (Queues) |
| `hostcuts!` | independent nodes moved across host nodes, adjacent host nodes merged, copy nodes (compiled here) to host-visible memory for device arrays a host node reads, none for an array already in `Readback` memory; before `liveness!`, since it changes the order |
| `liveness!`, `place!` | transient placement in the plan's arena parts, stored in the plan; arena parts shared with other graphs, host nodes or not |
| `sharing!` | edges between transients that share bytes (for CUDA/HIP graphs) |
| `barriers!` | barriers and image layouts, keyed by memory (a transient by its arena bytes, so transients sharing bytes conflict), with state merged over both outcomes of every conditional launch |
| `submissions!` | ordered segments, each with its queue, cut by the estimated budget, host nodes and present; waits between segments on different queues |
| `summary!` | every resource the plan shares with other graphs or eager code, with all its accesses per segment; composites (a TLAS, a BLAS) included, their members expanded at submit (Ordering) |
| `bindings!` | the argument block layout, the cell copy table, the prepare kernels' tables, descriptor sets, the residency list |

**The head and prepare nodes are nodes.** Compile puts a head node first, and
one after each host node (Ordering); a
launch that writes an array's length on the device is followed by a prepare
node, added at declaration. The argument slots a plan holds for an array (address, size, launch
commands, `ndrange` values) are a
resource of the plan (a `GPURef` is not among them: its slot is shared,
and the plan bakes its address): the head and prepare nodes write them, launches read
them. So the head precedes every root, barriers follow the head, and the head's
reads of cells are in the access summary, by the same rules as data.

What compile decides about a node is stored in the node (`CompiledNode`): its
queue, predecessors, barrier, compiled command, cost and argument slots. The
plan holds only what spans nodes: segments, placement, the argument block, the
cell copy table, recordings. A recompile runs the same steps and
builds a new plan; the old plan and its compiled nodes are retired by token.

**Rewriters are opt-in and generic.** `Graph(dev; rewrites = (FuseEpilogues(),))`
(names open). None run by default, so RayMakie's graphs run no rewrite. A
rewriter matches on the nodes' origin and on attributes device types declare
for their candidates ("implements exactly this Julia function as an epilogue",
"compiles consecutive operations as one region"); it is never specific to a
device type. `FuseEpilogues` merges a broadcast into the selected GEMM or
convolution candidate that implements exactly that function (hipBLASLt's GELU
is not the exact one); `MergeRegions` compiles a run of nodes whose candidates
share a region compiler (one `MPSGraphExecutable`, one cuDNN operation graph)
through a device-type verb. `FuseBroadcasts` is open.

#### Submission split: estimated

Each node's estimated time (its `KernelInfo`, ndrange, declared bytes, and known
work for library operations: 2MNK for a `mul!`) is a roofline estimate against
the device's throughput and bandwidth. `submissions!` cuts where the running
total reaches the device's budget, with margin. The budget,
`submissionbudget(dev)`, is core's: the policy (the driver's job timeout, a
smaller target on a GPU that drives a display) is computed from `caps(dev)`
fields a device type reports (its job timeout, whether it drives a display).
A device type reports facts; core decides. Measured node times, where
profiling is on (`Graph(dev; profile = true)`, name open), only calibrate the
next compile. A plan is never re-recorded because of a measurement.

### Queues

A device has queues of kinds (graphics+compute, compute, transfer, video
decode, present). Core's policy:

- Work only one kind can do goes there: video decode, present.
- Host-device copies go to a transfer queue where the device has one and the
  copy is independent of the neighbouring work.
- Parallelism inside a plan is barrier absence on one queue (Vulkan, Metal) or
  graph branches (CUDA, HIP). A second compute queue is used where independent
  branches' estimated overlap exceeds the cost of the waits between queues.
- Independent graphs can run on different queues, with priorities (the
  editor's frame above a background DNN).

### Ordering across graphs and queues

Graphs share memory: transients of graphs that share an arena alias, and
device arrays and other shared resources are named by several graphs and eager code. Inside a
graph the order is compiled and recorded once. Between graphs, and between
graphs and the host, it is settled by two things, neither of which looks at
every resource a run names:

- **Between graphs, through the resource when it is written** (few
  recompiles, a cheap run). No plan keeps a list of other plans. Each
  resource's sync state (`SyncState`) has its `namers`: the plans naming
  it with their use from compile, an immutable snapshot replaced at
  registration and retirement. A plan naming a composite reads its members:
  the plans naming a TLAS are found from a member through the member's
  readers (the frame tracing a TLAS over a simulation's positions is among
  the positions' plans). A stage that writes a resource another plan names
  waits for that plan's complete last run, and after its submission puts the
  resource into the inboxes of the other plans reading it. A stage that finds
  a resource in its inbox waits for the complete last runs of the other
  plans writing it. Whether a resource is shared is read from the snapshot
  at run, so a plan registering later needs no recompile. Two plans
  therefore overlap on the GPU unless one writes what the other uses, and
  then they wait for each other's whole last run. A composite gaining a member changes no plan: the
  member's readers snapshot is replaced at the call, and the plans naming
  the composite read the member only after the step applying its build,
  which locks the member's writers. A removed member keeps the composite as
  a reader until the submission that removed it has passed.
- **The inbox, for the host side.** What is not a run (eager calls, host
  reads, vendor views, eviction copies, sparse binding, and the steps a
  submission applies for pending changes) records its accesses on the
  resource's sync state: the last write and the reads since, each as
  (channel, token, stages). Pending a change at the call, or recording such
  an access, puts the resource into the inbox of every plan that names it
  and of every plan that names a composite built from it; a run's write of a
  shared resource puts it into the inboxes of the other plans reading it.
  Registration puts the shared resources a plan uses into its own inbox, so
  its first run waits like any later one. Each stage of a run takes the
  plan's inbox and handles only what is in it: it applies their pending
  changes in front of its first segment and waits for their other users, so
  a change made while the run is between stages reaches the later stages
  (per stage). Its own accesses are recorded
  nowhere but in its last-run tokens. A run's host time grows with the
  named resources its stages write (its outputs, mostly) and with its inbox, not with the number
  of resources it names.
- **Who waits for what.** A run: its previous run (runs never overlap), the
  last runs of the other plans naming what a stage writes, and for its inbox
  resources their recorded accesses and the last runs of the other plans
  writing them. What a stage found is kept for the run's later stages, which
  add only what arrived in the inbox since.
  Anything that is not a run, and the steps a run applies for pending
  changes (they write what they change): the recorded accesses of each
  resource, the last runs of the plans naming it where their use conflicts
  (each plan's use of a resource, read or write, is known from compile), and
  the last runs of plans retired while naming it. Retired or replaced
  storage is released after the same tokens.
- **Sync locks.** A plan's sync lock orders its submissions against those it
  shares resources with: a stage holds its own, those of the plans sharing
  what it writes and those of the plans naming its inbox resources (and the
  members their pending builds read) while it takes its inbox, computes its
  waits, submits, sets its last-run tokens and fills the readers' inboxes;
  registration and retirement hold those of the plans naming what the plan
  names; anything that reads a plan's last run (an eager call on a resource
  the plan names, an eviction, the release of storage it used) holds that
  plan's sync lock. So a last run is never read while it is about to change.
  The snapshots that chose the locks are checked under them; a change only
  makes the stage lock again, never recompile. A run locks the sync states
  of its inbox resources and of what its stages write (and, in its first
  stage, its arena parts), nothing else. A change that marks a
  `BuildRecording`, and an eviction, put the resource into the inbox of every
  plan that names it, so a run never walks the resources it names to find
  them.
- **Tokens per segment.** Each segment of a run gets a token from the channel
  it runs on. Claiming a token and submitting it happen under the channel
  lock in one step, so submissions reach each queue in token order (a
  timeline semaphore's signal value must increase with every signal). A run
  computes the waits of all its segments before it submits any, so its
  segments on different queues do not wait on each other through shared
  state; their order inside the run comes from the dependency graph.
- **Composite reads.** Inside a plan, where compile does not know a
  composite's members, a build node conflicts with every earlier write (a
  full memory barrier).
  Between graphs the members are reached through their readers (above).
  Eager calls lock a composite's members with it (`lockexpanded`) and wait
  for and record their accesses: a read of a composite is a read of each
  member, a build a write of the structure and a read of each member.
- **A run with host nodes is one submission per stage.** Each stage records
  its accesses when it is submitted, and each starts with a head node that
  copies the cells again. A change queued while a host function runs (a
  `resize!`, a store) is ordered before the later stages and seen by them,
  like any change between two runs. The call never waits: what to do about
  a running plan is decided when a change is applied.
  - A move of storage the plan recorded by handle marks the parts that
    recorded it; they are recorded again before they are next submitted, and the
    move, ordered after the run's earlier
    accesses, has copied what they wrote. Nothing waits.
  - A change the plan cannot follow mid-run (a length it launches over
    directly, a temporary it sized outgrown, a capacity change of a
    structure it sized temporaries from: compile decided those) is left
    pending by the plan's own later stages, with every later change on the
    same resource, and applied after the run (the next run recompiles).
    Any other submitter that would apply it waits on the host until the run
    has submitted its last stage, then starts over. Only a run past its first stage is
    waited for (one not yet there finds the stale mark under its stage-1
    locks and starts over); a plan with one stage is never waited for.
    From inside any run (any host function, on the run's task) such a
    submitter throws instead (a task the host function starts and waits for
    is not detected): it could wait in a cycle. No fairness: the waiting
    submitter starts over when the run ends and may wait again if the
    plan's next run got past its first stage first.
  - Arena parts stay shared: a run holds its parts from before its first
    submission until its last stage is submitted, so another graph's run on
    the same part waits on the host for the whole run, host functions
    included, and then on its tokens. A host-node run has no memory of its
    own. Parts are taken in id order, and a task that would wait for a part
    while holding one with a higher id throws (`holdparts!`; it would
    deadlock). A host function may not run, write into or free a graph that
    shares arena memory with its own graph: `lockgraph(g)` throws when the
    calling task holds an arena part of `g`'s plan. It may run another graph
    only if that run cannot wait on the parts its own graph's run holds. The
    pressure step leaves the pages of a running plan alone. A host function
    reads and writes only host-visible arena memory; copies before and after
    the cut move data between it and device arrays.
  - A later stage's accesses are not recorded in advance with a token
    signalled when that stage is submitted: a GPU wait for a signal not yet
    submitted blocks the queue it sits on (Vulkan requires the signal to
    execute before forward progress can be made,
    `refs/vkdocs/chapters/cmdbuffers.adoc:3106-3117`).
- **A lent array** (inside `withview`) makes every Mantle call that locks its
  record throw: eager calls, queued changes, runs. `withview` holds the array
  for the block, and restores it, applies its pending changes and lends it in
  one locked section, so no change, move or eviction lands in between; the
  hold drops once the stream's work has passed.
- **Nothing can fail between claiming tokens and submitting.** Everything
  fallible (compiling, allocating arena pages, acquiring a swapchain image,
  converting arguments) happens before. After that only the driver submit
  remains, and its failure is device loss. Locks are released in `finally`.
- **How a wait is expressed** is the device type's: a timeline-semaphore wait
  (Vulkan), `waitForEvent:value:` (Metal), stream order or
  `cuStreamWaitValue64`/`hipStreamWaitValue64` (CUDA, HIP). No command buffer is
  added for ordering; no conservative full barrier.
- **Lock order**, everywhere: graph lock, then the run's hold on its arena
  parts by part id (held across host functions, never across a record lock
  taken before it), then window lock, then plans' sync locks by plan id,
  then sync states by resource id, then the channel lock. The pool lock, the inbox,
  the texture and sampler tables' locks and a plan's `runend` lock are leaves
  (nothing else is taken inside them), never held across a GPU wait or a call
  that may run finalizers.
  An arena's state (mapped range, tenants) is protected by its sync state's
  lock. `acquire!` is never called with an sync-state, channel or pool lock
  held. The graph lock is a `Base.Semaphore(1)`, not a `ReentrantLock`: compile
  allocates under it, and holding a `ReentrantLock` would keep `acquire!`'s GC
  step from running finalizers (Julia 1.12, `base/lock.jl:174`).
- **Host waits and locks.** No sync-state, channel, pool or window lock is
  held while the host waits. The graph lock may be: a run waits under it for
  another run holding a shared arena part, for a host node's inputs, for the
  swapchain acquire, and in `acquire!`'s pressure steps. No store waits for a
  run, `GPURef`s included.
- **Host waits are GC-safe.** A blocking driver wait is a `@ccall gc_safe =
  true` (Julia 1.12, `base/c.jl:278-296`), so a thread waiting on the GPU does
  not hold up a GC another thread needs.
- **Arenas:** graphs core expects to overlap get separate arena parts; the rest
  share one. Tenancy is a table from graph id to that graph's peak, never
  keeping a graph alive. Which graphs are expected to overlap comes from their
  priorities (open).

### Core types

```julia
mutable struct SubmitChannel{N,T}
    native::N                       # VkQueue + timeline | MTL4CommandQueue + MTLSharedEvent | stream + counter
    kind::QueueKind
    lock::ReentrantLock             # submit, present, sparse binding, wait-idle
    next::T                         # last token handed out
    outstanding::Vector{Submission{T}}
end

struct Recording{N}                 # one part of a segment: written once by record!, submitted every run
    native::N                       # VkCommandBuffer | ICB | CUgraphExec | hipGraphExec
    channel::SubmitChannel
end

struct Submission{T,C}
    token::T
    recycle::Vector{C}              # an eager call's command objects, back to the free list once the token passes
end
```

- A segment is a list of parts submitted together: recordings written once by
  `record!`, and `BuildRecording`s (Acceleration structures). A run re-records
  only the `BuildRecording`s a change marked; the parts that recorded the
  handle of a moved array; a new or changed group of draws and the segment
  holding its pass's list; and, per swapchain generation, the segments that draw
  into window-sized targets and the per-image present segments (Windows).
- A submission is a segment of a run, an eager call, a host read or a
  pressure step; pending changes ride in the next of these. There
  is no one-shot type and no hold stack. Nothing a submission uses is kept
  alive by the submission: memory is retired with the tokens of its last uses
  (Memory), so an array dropped while the GPU still uses it is released only
  after that use.
- A token is a channel and a value. Token verbs, per channel:
  `passed(ch, value)`, `waitfor(ch, value)`; core's `passed(::Token)` and
  `waitfor(::Token)` wrap them. The last token handed out is core data (the
  channel's `next`), not a verb.

### Record

Core's walk calls builder verbs; a device type implements them on its native
object.

```julia
b = Builder(device, channel)
emitdispatch!(b, node, cmd::CompiledDispatch, deps)   # direct or indirect, as compile decided
emittrace!(b, node, cmd::CompiledTrace, deps)
emitcopy!(b, node, cmd::CompiledCopy, deps)
emitbuild!(b, node, cmd::CompiledBuild, deps)
beginrender!(b, node, cmd::CompiledRender, deps); emitdraw!(b, node, cmd::CompiledDraw); endrender!(b)
bindindices!(b, node, storage)                         # an index buffer inside a render pass, by handle
emitlibrary!(b, node, cmd::CompiledLibrary, deps)
emitloop!(b, node, cmd::CompiledLoop, deps) do … end
emitstore!(b, dst, bytes)                              # an inline store within caps(dev)'s limit, in a change prefix
recording = finish(b)
```

- `deps` is complete and core's: the predecessor nodes and the derived barrier.
  CUDA/HIP add a node with those edges; Vulkan and Metal record the barrier and
  the command. No stream capture. The barrier is recorded by the same verb as
  the command, so `vkCmdPipelineBarrier2` has one call site.
- A builder is a value that never escapes the call that made it: `record!`, an
  eager call, the change prefix of a submission, or the re-recording of an
  `BuildRecording` or of a window-sized segment. Command objects come from the
  device's free list.
- `emitstore!` writes a store's bytes into `dst` inside the submission:
  `vkCmdUpdateBuffer` on Vulkan (at most 65536 bytes, offset and size
  multiples of 4, `refs/vkdocs/chapters/clears.adoc:800-805`; a store that
  does not meet this is a copy from its upload region). Device types without
  an inline store (Metal, CUDA, HIP) report a limit of 0, and their stores are
  copies from the upload region acquired at the store's call.
- Record compiles nothing, allocates no device memory, uploads nothing, submits
  nothing and reads no mutable resource field; the argument block's contents
  come from `bindings!`.

### Run

```julia
run!(g::Graph) = lockgraph(g) do                # throws for a freed graph, on re-entry from its host function,
                                                # and for a task that holds an arena part of g's plan
    while true
        applywindows!(g)                        # a window resize, marked by resize!(win)
        (g.plan === nothing || g.recompile) && recompile!(g)   # the first run; after a failed compile
        r = run!(g.plan)                        # Again: nothing claimed; an arena range was unmapped, the window changed,
        done(r) && return r                     # the plan went stale, or a resource it names was evicted after it looked
    end                                         # returns Submitted(), or Skipped() for a minimized window
end
```

An acquired swapchain image is owned by the window: a run that returns Again
or throws before presenting gives it back, and the next run takes it there
(internals.jl section 7).

`run!(pl)` allocates what it may need before any sync-state lock
(unmapped arena pages, the swapchain image), then submits the first stage
under the records' locks, and the later stages after their host nodes.

- **Runs of one graph never overlap:** a run's first segment waits on the GPU
  for the previous run's last token on each channel. The host never waits for
  the previous run. A run's host waits (another run holding a shared
  arena part, a host node's inputs, the swapchain acquire, `acquire!`'s
  pressure steps) are under the graph lock and the run's arena-part holds,
  and no other lock (Ordering).

- **The argument block** holds every dispatch's arguments, one allocation per
  plan, in host-visible device memory. Fields sit directly in the slots
  (address, dims, scalars), and the block is marked read-only to the kernels
  (`NonWritable`), which lets RADV use scalar loads. Both are properties of
  Lava's entry wrapper. KA's `ndrange` values are slots too.
- **`GPURef`s are slots.** A
  `GPURef` is one slot in device memory; every plan and group of draws that uses it
  bakes its address, and a launch reads the value through it (a one-element
  buffer read costs what a true uniform costs). `r[] = v` is a scheduled store,
  applied by the next submission that touches the slot; nothing waits for a
  run. Data the host supplies per run is a device array the graph names,
  written by stores the same way.
- **The head node** runs first in every run, unconditionally (Compile: it is a
  node with uses, so everything that reads what it writes is ordered after it,
  on every queue; on CUDA/HIP it precedes every root). It copies
  every referenced array's cell (address, size) into the argument slots, and
  computes the launch command and `ndrange` values of every launch over a
  resizable array from the array's size and the kernel's group size, all from
  the tables `bindings!` built. A plan that names no device array or resizable array has an empty head,
  which is not recorded.
- **Launch kernels never read cells.** A launch over a fixed-size array is
  direct, recorded once. A launch over a resizable array is indirect, from the
  command the head node (or a prepare kernel, for lengths decided during the
  run) wrote. The exceptions are Mantle's own: the head and prepare nodes, the
  TLAS offsets and records kernels (once per build), and kernels that write a
  length through `lengthof(a)`.
- **`GPURef(dev, v)`** is one slot in device memory holding a value (above):
  `x[] = v` copies the value once at the call, like any store, so a
  multi-word value is never torn, and the store is applied by the next
  submission that touches the slot. It holds values only; an array argument
  is the array itself.
- **Stores into device arrays** (`a[r] = data`, `copyto!(a, data)`,
  `copy!(a, data)`) copy the data once at the call, into an upload region
  (host-visible system memory acquired for it), and are pending until
  the next submission that touches `a` (Changes table). A store within the
  device type's inline limit (`caps(dev)`; on Vulkan 64 KiB at 4-byte offsets
  and sizes, `vkCmdUpdateBuffer`) is written from the region's bytes with
  `emitstore!` (Record); a larger store is a copy from the region. A store captures the
  storage it writes and its byte range in the parent (through a view's
  offset), so a move pended after it carries the stored data along. The run applies them in front of its head node, in the
  first segment of its first stage (a later stage's, for changes made while a
  host function ran).
- **Host nodes.** Compile moves independent nodes across a host node and
  merges consecutive ones into one cut. The data a host node reads is placed in
  host-visible memory (or copied there by a node before the cut), and what it
  writes is copied to its device placement after it. `run!` submits the
  segments before the first host node, waits for their tokens, calls the host
  function on the calling thread, submits the next stage, and so on (Ordering).
  If a host function throws, the rest of the run is not submitted and the
  exception propagates; an acquired swapchain image that was not presented
  goes back to the window, which hands it to the next run (Windows). A host
  function may not use its own graph (the graph lock is not reentrant: this
  throws), nor run, write into or free a graph that shares arena memory with
  its own (Ordering); host functions of two graphs that run each other can
  deadlock otherwise, which Mantle does not detect. The cost is one
  GPU-to-host round trip per cut, shown by `submissions(g)`.
- **Failure of a run:** the plan stays compiled and the graph unchanged. An
  error before any token is claimed (an arena range unmapped meanwhile, an
  out-of-date swapchain, a stale plan) is retried by `run!` itself.

### Eager work: operations on a device

An operation whose arrays live on a real device runs through the same method
and the same `dispatch!` as on a graph, and is submitted before it returns.

```julia
function dispatch!(dev::Device, kernel, args, ndrange)
    k = kernel!(dev, kernel, argtypes(args))          # the Kernel at its CodeInstance
    ch = channel(dev, Main())
    packed = acquire!(pool(dev), Unified(), argumentbytes(k))   # fallible: before any lock
    token = nothing
    try
        token = submitapplying(uses(k, args); ch) do after   # internals.jl section 11: pending changes in front,
            foreach(usable, resources(args))          # freed through the creator's handle: throws, under the lock
            checkwritable(kernel, args)               # a write touch on a read-only array throws
            packarguments!(packed, k, args, after)    # each array's storage after the pending changes
            ext = launchextents(dev, k, resolve(ndrange, args))
            b = Builder(dev, ch)
            emitdispatch!(b, nothing, CompiledDispatch(k, ext, packed), Deps((), ()))
            [finish(b)]
        end
    finally
        retire!(pool(dev), packed, token === nothing ? () : (token,))   # after the call ran, or now after a throw
    end
end
```

- **Per launch:** a compiler-cache hit, a few commands from the device's free
  list (on CUDA and HIP the builder buffers the launch and issues it in
  `finish`, after the change prefix), one submission. A `GPURef`
  argument is packed with its slot's address; a pending store into the slot is
  applied in front of the launch by the same submission. An array lent to vendor code
  (`withview`) or freed throws.
  The packed arguments and a multi-kernel operation's temporaries are regions
  retired with the call's token. No batching outside graphs, no cached one-op
  plans, nothing left open between calls.
- **Every eager entry is this path:** Mantle's methods on `MantleArray`s on a
  device, KA and KernelInterface launches, GPUArrays' generic code through its
  KA launches, `draw!(dev, …)`, `trace!(dev, …)`, `copyto!` downloads. An
  upload into an existing device array is a store, scheduled like every
  change (Rule 8).
- **Host reads** (`Array(a)`, `copyto!(host, a)` of a device array, outside a
  run) are a submission of their own: the changes pending on `a`, a copy into
  `Readback` memory, a wait for that token and a memcpy. In a graph a read is a
  copy kernel node into a device array (`copyto!(out, x)`, Operations), recorded once and run
  every run, after the run has applied what is pending; `Array(out)` then only
  waits for the run's token, and needs no GPU copy when `out` was created in
  `Readback` memory. `Array(x)` of a graph array throws, like `getindex`. Scalar
  indexing stays off by default; `allowscalar` is GPUArrays' interface and
  accepted as such.
- Calls on one channel from two threads are ordered by the channel lock; each
  thread's program order is kept.

### Windows and present

- A `Window` is a graph resource; its swapchain or drawable is its storage.
- **Vulkan:** everything except the segment that writes the swapchain image is
  recorded once; that last segment is recorded once per swapchain image (two or
  three small recordings), and `run!` submits the one for the acquired index,
  so choosing the image adds no GPU work (no full-screen copy from a Mantle
  image into the acquired image).
- **Metal:** a new drawable per frame; the render pass's attachment is set in
  the run's command buffer, and the render ICBs execute inside it.
- Present is part of the submission (`submitnative!` takes waits, signals and a
  present). Acquire and present use binary semaphores
  (VUID-vkAcquireNextImageKHR-semaphore-03265,
  VUID-vkQueuePresentKHR-pWaitSemaphores-03267): the segment that writes the
  image waits on the acquire semaphore and signals the image's rendered
  semaphore, which the present waits on. The acquire happens under the graph
  lock, before any sync-state lock of the run and outside the window lock;
  OUT_OF_DATE is a resize to the surface's current size, applied and retried
  by the same `run!`; SUBOPTIMAL presents and marks the window. A window
  marked after the run looked, or with a zero extent in either dimension
  (minimized), acquires nothing: the first is applied and retried, the second
  skips the whole run (nothing submitted, its changes stay pending) until the
  window has an extent again.
- **One graph presents to a window:**
  any of its nodes may write the window (a draw, a clear, a kernel, a copy);
  the first graph declaring a write presents, and a second graph declaring one
  throws at declaration. Other graphs render into `Image`s this graph reads.
  The binding ends when the graph is freed or collected or the nodes writing
  the window are removed; another graph may then present. An eager draw into a
  window throws (it has no presenting graph and no run); eager draws into
  images are allowed. `run!` returns `Submitted()` or `Skipped()`, both public. So every acquire and present of a window happens under
  its graph's lock, in that graph's runs: no two acquires on one window, no
  other graph's run holding one of its images, no generation created under a
  running acquire. A graph presents to at most one window.
- **One owner for an acquired image: the window.** An image acquired by a run
  that returns Again or throws goes back to the window (`keepimage!`); the
  graph's next run takes it before acquiring and presents it. Only real
  images are kept, never the marker of a window change; a kept image of a
  retired generation is dropped with its generation. The image's generation
  is held until the run has submitted its last stage, since window-sized
  segments after the present use its recordings.
- **Resize:** `resize!(win, w, h)`, called from the window's event, stores the
  size and marks the graphs that render to the window; nothing else happens on
  the event thread. The graph's next run, under its graph lock and before
  anything else, makes the new swapchain generation (created with the old
  one as `oldSwapchain`),
  recreates its window-sized images at the window's size on the same arena
  memory, queues the cell writes of its window-sized arrays, and re-records
  the segments that draw into window-sized targets and its per-image present
  segments for the new images. A new generation retires the current
  swapchain, and acquiring from a retired swapchain is invalid
  (VUID-vkAcquireNextImageKHR-swapchain-01285); since only this graph
  acquires, under the same lock, none is in progress then. A zero
  size (a minimized window) makes no generation; runs skip until the window
  has an extent again. A host node between the first write of the swapchain
  image and its present is a compile error (the image cannot wait across a
  host function in one present segment). A swapchain generation is a resource with
  holds: one from the window while it is current, one from a run that
  acquired an image from it and has not submitted its last stage. It and its
  per-image recordings are retired with the tokens of its last run; whether a
  queue token also covers the presentation is open decision 9. A window has a capacity (the largest monitor's size unless
  given): window-sized arrays and images are placed at it. The arena part
  that holds window-sized transients is sparse where the device has sparse
  residency, and a generation maps only the pages its size needs, so a small
  window costs the capacity in address space only; without sparse residency
  the capacity is mapped. A resize beyond the capacity recompiles the graph
  at its next run, like a graph array beyond the size its slot was placed
  for. User code stays `run!(g)`. Acquire and present never rebuild the
  swapchain and never wait for the whole device.
- The channel lock covers submit, present, sparse binding and wait-idle.
- GLFW calls run on the main thread; recording and submission run anywhere.

### Acceleration structures and textures

- **Declared.** Builds and refits are operations with `TraceBuild` access;
  traces and ray queries declare `TraceRead`. Builds and refits are never
  inside `repeat!` or conditional; traces may be (Operations).
- **Placed.** Storage in the `Accel` kind; build and refit scratch the
  structure's own, sized for its capacity and replaced with it; instance
  buffers and the shader binding table as pool buffers; descriptor pools owned
  by the plan.
- **One implementation.** HWTLAS bookkeeping (geometry list, instance
  concatenation, rebuild versus refit, BLAS collection, compaction) is core's.
  Device-type verbs: `accelsizes(dev, path, n)`, `emitbuild!`, `countsource`,
  `instancerecordtype`, and the address or resource ID (`acceladdress`).
- **One core `TLAS` for hardware and software ray tracing**, with storage
  sized for an instance capacity that doubles; builds and refits carry the
  exact instance count, no padding (padding costs build time on AMD and
  NVIDIA and refit time on NVIDIA). Instances come in ranges (one instance, or one
  per element of a device array such as a meshscatter's positions); a
  device-side range table names cells. Each build, an offsets kernel computes
  each range's first record from the lengths in the source cells (a prefix
  sum) and a records kernel writes all instance records; both read cells. On
  every device type a plan's build or refit is a `BuildRecording`, its own
  small command buffer inside the segment's submission, with its barrier
  recorded first (a full memory barrier: the members are not known at
  compile). Vulkan records the count into it; on Metal, which reads the count
  from a cell, it still records the structure's storage. The structure keeps
  its `BuildRecording`s; a change that alters what one recorded (a new count, a
  `MoveAccel`, a Build that sets a new `builtcount`) marks it, and the run
  re-records only marked ones, after the submission's change prefix, so a
  refit takes the count the prefix's Build just set. A refit before any build
  is a Build. Nothing is compared at run; the plan itself is never recompiled
  or re-recorded for it.
  Vulkan's indirect build is not used: RADV and NVIDIA
  lack it, and on AMD's Windows driver it is slower than the exact count.
  Metal reads the count from a cell (`instanceCountBuffer`).
  `BuildRecording` and `BuildAccel` apply to a TLAS and a BLAS alike.
- **Pending builds coalesce per structure.** The pending `BuildAccel` changes
  of one structure taken into one submission merge into a Build if any of
  them is a Build, else into an Update; a merged change takes the latest
  sequence number of those it merged, so it comes after everything pended
  before them (a BLAS move, say). The structure keeps `builtcount`, the
  count of its last Build, and every Update uses it, never the current total.
  A change of a structure pends the Updates of the structures over it in the
  same call, with later sequence numbers. A build counts as a read of the
  structure's members (Ordering, composite reads), so a move or release of a
  member waits for builds in flight. The records kernel writes the records
  up to the build count, with mask 0 past the device total, so a count that
  ran ahead of a source's cell builds no unwritten record.
- **Kernel-written counts.** A TLAS total that a kernel writes (any device
  type) is built over the capacity, with mask-0 records past the device
  total, so later refits stay legal. A BLAS over arrays whose length a kernel
  writes is built over the geometry capacity, with zero-area primitives past
  the device length written by a prepare kernel, so a change of the vertex
  length is an Update. Vulkan constrains both: an update may change instance
  definitions, transforms and vertex positions and nothing else, and may not
  turn a primitive or instance active or inactive
  (`accelstructures.adoc:159-170`). So the mask-0 records reference a BLAS
  (an instance referencing 0 is inactive, `:268`), the padding primitives are
  zero-area, not NaN (`:261-262`), and a run or store that writes a BLAS's
  index array pends a Build, its vertices an Update.
- **Ranges and shaders.** Traces and ray queries take the TLAS by address or
  resource ID through its cell, and textures sit in a device-wide table
  indexed through a slot (open decision 3). A range holds its
  BLAS and its source array, and its entry has a `mask`: hiding an instance is
  mask 0, a refit; only a deleted range is inactive, which is a Build. A
  removed range releases its BLAS and source array with the token of the
  build that deactivated it. Pseudocode: `docs/resizing-and-raytracing.jl`.
- No `Raycore.sync!` inside `Adapt.adapt_structure`; sync is an operation.
- **What a scene change costs.** Moving instances (a transform, a
  meshscatter's positions): a TLAS refit, pending. A different number of
  instances: a TLAS build with the exact count, pending; the BLASes are kept.
  A mesh's vertices moved with the same topology: a BLAS Update and a TLAS
  refit. A new topology: a BLAS Build into its storage and a TLAS refit,
  since an update may change an instance's BLAS reference
  (`accelstructures.adoc:159-162`). A BLAS's storage is sized from its
  geometry with a capacity that doubles, like a TLAS: beyond the capacity the
  build is a `MoveAccel` to new storage, and the new address reaches the TLAS
  records through the BLAS's cell. TLASes over a BLAS pend an Update when the
  BLAS's address or contents change; the records kernel reads the BLAS
  address from its cell. Hiding an instance: mask 0 in its range entry, a
  refit. A material: a store into the material table. None of these
  recompiles a plan or drops one. BLAS
  builds follow the TLAS's count rules (a `BuildRecording`).
- **A write reaches the structures built from it.** Host changes (a
  `resize!` or store of a source) reach the structures over it
  through readers, and so do runs that write a source (the last sentences
  below); a graph that writes a source declares no `refit!(g, t)`. An
  array's readers are
  the composites built from it: a TLAS (a range
  over it), a BLAS (its geometry) and a screen (a redraw mark). They are kept as an immutable
  snapshot, replaced under the array's lock and counted per reader: a TLAS
  with two ranges over one array counts twice, and `delete!` removes one. A
  `resize!` or store of an array with readers locks the array and its
  readers together (like `lockexpanded`) and pends each reader's update; the
  capacity a new total needs is allocated by the submission that applies the
  build. A run whose plan writes such arrays (compile lists them) pends
  `pendupdate!(reader, array)` on their readers after submitting each stage,
  so the next submission that touches the structure applies it, after the
  write. A mesh plotted from a simulation's arrays then follows the
  simulation with no host work.
- **Per-instance data is found, not concatenated.** Hit shaders reach a
  range's triangle data (the BLAS's geometry arrays by cell) and its material
  index through the range table, so adding a plot writes one range entry.
  A BLAS is shared by every range over the same geometry arrays.
- **Textures** are images in the `Images` kind (Graphics); bindings are
  compiled with the plan.
- `Mantle.refit!` is the acceleration-structure refit; the transient/plan
  `refit!` (`graph/build.jl`) has another name.

### Graphics

A frame of a recorded scene costs its submission. Updates are stores and
resizes; a plot is a group of draws, so adding or removing one
recompiles nothing. An attribute whose type changes (colour as numbers or
as colours, `Point2f` or `Point3f`) is a plot deleted and inserted, since
Makie fixes a rendered plot's types: its group deleted and a new one added, no
recompile. A value that picks another pipeline without changing a type (such
as `transparency`) never recompiles: each variant is its own recorded draw,
`when!(plot, target, flag, pipeline, args, count)`, and the change is a store
into the flags. Still recompiling: kernels declared per plot, a new pass, and a new
Hikari material type (a new specialization of the shade kernel, which
recompiles the plan that declares it).

- **Pipelines are values.** `GraphicsPipeline` (`Rasterizer`) and
  `MeshPipeline` hold the stages (`VertexShader`, `FragmentShader`,
  `GeometryShader`, `MeshShader`, `ObjectShader`; tessellation stages are
  stage types too) and the fixed
  state (topology, blend, cull, depth). A draw's pipeline comes from the
  device type's cache through `compiledraw`, keyed on the stage functions, the
  device-side argument types, the fixed state and the target formats; equal
  keys give one pipeline on every device type. `RayTracingPipeline` likewise,
  through `compiletrace`.
- **One argument type for a value and a per-element attribute.** A drawn
  attribute is a `MantleArray`: length 1 for one value for every element,
  length n for one per element. It is mutated in place (`copy!`, `copyto!`,
  `resize!`); an attribute that becomes another array object (a plot made
  from a simulation's device array, then given a different one) is a new
  group of draws for the plot over the new array, recorded on its own, no
  recompile. Its device-side type is the
  same either way, and shaders read it with `attribute(a, i)` (name open),
  `a[min(i, length(a))]` (1-based), the length coming from the slot. So a
  value and a per-element attribute are one pipeline, and switching between
  them is `copy!(a, data)`: a resize and a store, pending, no recompile (the
  length is never recorded). No stride is fixed at declaration. A value that changes
  every frame (a camera matrix) is a one-element array written by a store,
  which is inline: the call copies its bytes into an upload region and the next
  submission that touches the array writes them in its change prefix
  (`emitstore!`, Record), with no host wait. A `GPURef` is such a slot:
  one slot in device memory whose address every plan and piece bakes; a write
  to it is a scheduled store.
- **Draws are nodes; core forms the render passes.** `draw!(g or group,
  target, pipeline, args, count; instances, indices)` declares a draw into `target`: a
  window, an image, or a tuple of them (color images and one depth image). There
  is no pass object. Draws into the same target set, in declaration order with
  no node between them that reads what they write, are one render pass: the
  driver learns the targets and their load and store operations, and a
  tile-based GPU keeps the framebuffer on chip across the draws. A target's
  contents are kept unless a `clear!(g or group, target, v)` comes before its draws,
  which becomes the pass's load operation; a target no later node reads is
  discarded at the pass's end (from liveness). Counts are numbers or
  `launchrange`s (indirect over resizable arrays); indices are a
  `MantleArray{UInt32}`, bound by handle (`bindindices!`), so a move of an index
  array marks the parts that recorded it, which are recorded again before their
  next submission. `when!(g or group, target, flag, pipeline, args, count)` is a
  conditional draw: indirect, and the head (or a prepare node after the kernel
  that writes the flag) writes its count multiplied by the `Bool` flag, so
  visibility is a store into the flag. State the API records by value (render
  area, viewport, scissor) is fixed per recording: a scene's rectangle inside
  the target is one array per scene, shared by its plots, which the vertex stage
  applies (a remap to the rectangle and four clip distances), so a layout change
  is one store per scene. The render area is the target's; a window-sized
  target's segments are recorded again with each swapchain generation
  (Windows).
- **Groups of draws change without a recompile.** Requirements: complex GUIs
  change cheaply without slowing large meshes and scatters; an insert may take
  time in the millisecond range; visibility is cheap. A group whose nodes are
  draws into one pass is a piece of that pass: compiled and recorded on its own
  (Vulkan: a secondary command buffer with dynamic rendering; Metal: its own
  indirect command buffer), with its own argument region and its own entries
  in the head's cell-copy table. On a graph that already has a plan, a new
  group of draws whose place is inside a pass with its targets (a RayMakie
  screen makes one group per scene, starting with the clears, and each plot is
  a group in it), draws declared into such a group, `delete!(g, group)` and
  `sort!` of such groups are applied by the graph's next run before its
  stages: the group's draws are compiled and recorded, and the pass's list of
  pieces is rewritten, which is one part recorded again. The plan is not
  recompiled and no other piece is recorded again. A launch, a clear, a draw
  into the graph itself, or a group that would start a new pass recompiles. Every pass starts with a
  fixed barrier (compute, copy and build writes before it, then vertex, index,
  indirect and fragment reads), so a new draw adds no dependency inside the
  graph. Each draw keeps its own pipeline, index buffer and argument layout: a
  large mesh draws as it would alone. Draws run in the order of their places;
  `sort!(group; by, lt, rev)` reorders the groups of draws in a group (a
  rewrite of the list, nothing compiled). `GPURef`s a piece reads are slots it bakes, like a
  plan. A kernel declared into a graph that has a plan is a recompile. Core
  Vulkan inherits no state into a secondary command buffer
  (`pipelines.adoc:131-133`), so each piece binds its pipelines and sets
  viewport and scissor itself; the pieces of a pass into a window are recorded
  per swapchain generation. Gate: adding and deleting one plot in a frame graph
  of 100, 1 000 and 5 000 plots; the time per change must not grow with the
  number of plots, and a visibility change records nothing.
- **Images are resources like arrays.** `Image(dev, T, dims...)`, transients
  `Image(g, T, dims...)` and `Image(g, T, win)`; the format comes from `T`
  through one table per device type (`imageformat(dev, T)`, name open). Stores and
  `copyto!(img, array)`/`copyto!(array, img)` are pending changes or copy
  nodes, `Array(img)` a host read; layouts come from `barriers!`. An image a
  shader samples gets an index in the device-wide texture table on its first
  use (`resizing-and-raytracing.jl` B.9); there is no separate texture type. An upload of the same size and type
  writes the image it has, a new size is a new image under a new index, a
  store into the slot, keeping the texels where the old and new sizes overlap
  (one GPU copy, as `resize!` keeps an array's contents), never a
  recompile.
- **Samplers are values** (filter, address mode, anisotropy) in a device-wide
  sampler table, created on first use; a shader takes a texture index and a
  sampler index. Texture and sampler table entries are host writes into an
  unused index (`tableentry!`, `samplerentry!`), not recorded commands.
- **Screenshots.** `Array(win)` (name open) makes the next run that presents
  to the window copy the image into `Readback` memory before presenting (a
  second recording of the per-image present segment, made on first use per
  generation) and waits for that run. It throws `ArgumentError` when the window
  is minimized, when no graph presents to it, or from the presenting graph's own
  host function, and a call already waiting is woken with that error when the
  graph's next run is skipped or the graph is freed. One behaviour on
  every device type.

### Memory

- **Device-type verbs only:** `rawalloc(dev, kind, bytes, constraint)`,
  `rawfree`, `createreserve(dev, bytes)` (address space with no pages mapped),
  `bindpages!`/`unbindpages!` (queue operations, under the channel
  lock), `placeimage`, `placeaccel`. Driver objects the API cannot place (an
  ICB, a pipeline, a descriptor pool, a swapchain image) are counted with the
  bytes the device type reports and released by the same engine.
- **Kinds:** `Buffers`, `Unified`, `Readback`, `Images`, `Accel`, with usage
  constraints. Vulkan's per-image memory, video and external memory, Metal's
  argument memory, ICBs and textures, and ROCm's memory are pool allocations;
  there are no Vulkan dedicated allocations and no use of AMDGPU's pool.
- **One release engine.** A region or counted object is retired with the
  tokens of its last use (`retire!(pool, x, tokens)`) and released as soon as
  they have passed: checked at every `acquire!` and every `run!`, not only
  under pressure. Finalizers push onto the lock-free `Inbox` and nothing else;
  they take no lock and call no driver.
- **Holds.** A device array (and every other resource graphs share: images,
  acceleration structures) has one hold for its creator, one for
  each graph that names it (including arrays inside broadcasts, keyword
  arguments, view parents and flags), and one for each composite holding it
  (a TLAS range, a BLAS). `free!(a)` drops the creator's hold (once,
  whatever the number of threads calling it), `free!(g)` and the graph's
  finalizer the graph's (one function for both); at zero the resource is
  retired with the tokens of its last uses. After `free!`, using the handle
  throws, in eager calls and in declarations; adding a hold is a
  compare-and-swap that throws at zero, so a hold never goes from zero back to
  one. `run!` of a freed graph throws.
- **The last hold.** Under the record's lock the pending changes are dropped
  (upload regions retired, prepared storage released, deferred hold drops
  done), the storage becomes `Released()`, the old storage is retired with the
  record's last tokens, the resource leaves the eviction list and its
  cell is released. Every holder type (`TLAS`, `BLAS`, a render object) has a finalizer that pushes its hold drops through the
  pool's inbox.
- **Composite resources hold; they do not allocate.** A TLAS, a BLAS, a
  texture, material or light table, a plot's render object is a struct of
  `MantleArray`s and acceleration-structure storage, each created
  and held by it. Its changes are pending changes on its record, its growth
  is a `resize!` of its arrays or a `MoveAccel` whose commit retires the old
  storage, and its `free!` drops its holds. A submission that uses a
  composite locks its members with it and waits for and records their
  accesses (Ordering, composite reads). Only the hold machinery and change
  commits call `retire!`; no finalizer frees a member. So a composite written
  with these calls alone cannot free early, leak or free twice, and the tests
  of the primitives cover it.
- **Tables free their slots.** Material, medium, light and texture tables keep
  a free list; a deleted plot's slots are reused.
- **Waste is bounded and given back.** Pool regions grow by doubling, reserves
  map at least twice what they had (`resizing-and-raytracing.jl` B.1). Under
  pressure, idle arena pages go back. Pressure never unmaps pages beyond an
  array's size (`trim!(a)` does that, explicitly; Arrays) and never compacts
  acceleration-structure storage.
- **Arenas grow and shrink.** A sparse arena is a fixed virtual range whose
  pages are mapped as tenants need them and unmapped when the largest current
  tenant needs fewer (Vulkan sparse binding, Metal placement-sparse buffers,
  CUDA/HIP `MemAddressReserve`/`MemMap`); mapping and unmapping are queue
  operations ordered by the arena's sync state, decided under its lock.
  Unbound pages are retired with the unmapping's token. A host node's
  arguments live in a host-visible arena part that is not sparse, so a host
  function gets one contiguous view. Where sparse binding is
  missing, a graph compiled later that needs more gets a new arena part
  (an `ArenaPart`, which splits past the driver's maximum allocation);
  existing parts never move.
- **An idle graph holds address space, not memory.** Under pressure, arena
  pages that only idle graphs use are unmapped, and the unmapping marks those
  graphs' plans, per arena part (a plan has several: device memory, the
  host-visible part of host nodes, the part of window-sized transients, whose
  tenancy follows the swapchain generation's size); a run binds only the
  marked parts, and a mark is cleared only for the part that now covers the
  tenant. A marked plan's next run allocates the pages before it takes
  any lock, and binds them under the arena's record lock in the same critical
  section as its submission. The unmapping marks and queues under the same
  lock, so a run is either recorded before the unmap (which then waits for it
  on the GPU) or sees the mark. Nothing at `run!` decides by comparing sizes
  whether to bind: the mark says so. The one comparison left checks that the
  pages allocated without the lock still cover the gap (another tenant's
  pressure step may have widened it), the retry check of an optimistic
  allocation, like `lockexpanded`'s snapshot check.
- **An idle resource can leave the device.** Two
  graphs that each need 90% of the device run one after the other: their
  transients share an arena part (the largest, not the sum), and where their
  persistent memory (weights) does not fit beside it, the idle graph's is
  evicted and restored by its next run. Eviction is a pressure step, least
  recently used first. A resource is evictable when no composite holds it, no
  plan has a handle dependence on it, no plan that names it is running, it is
  not lent, nothing is pending on it, and its storage is neither `Evicted` nor
  `Released`. A structure's own arrays (records, range table, offsets) are
  held by it, so they are never evicted. Accesses still in flight are
  allowed: the eviction copy waits for the last write through the access
  record (a read need not wait for reads), and the storage is retired with the
  tokens of every access in flight and the copy's. Arrays and images are
  copied to pageable host memory in a submission of their own, so the OS may
  compress the copy or swap it to disk; Mantle writes no files. A `readonly`
  array keeps a reference to the host array it was created from and restores
  from it, so its eviction copies nothing out. That is how device data is
  backed by a file: an mmapped host array (`Mmap.mmap` returns a plain
  `Array`) has the file's pages, unchanged, which the OS drops and re-reads
  without writing anything (a host array or file changed afterwards is
  undefined);
  `acquire!` waits for those retirement tokens outside every lock, then sweeps
  and retries. So `run!(dnn_1); run!(dnn_2)` back to back works while
  `dnn_1`'s weights are still in use on the GPU.
  Eviction works the same on every device type, unified-memory devices
  included (the Apple M5, the Ryzen AI Max+ 395, NVIDIA's unified-memory
  chips): the device allocation is freed, and the copy is pageable, so the OS
  can compress or swap it where it cannot touch the device allocation.
  Restores upload from pageable memory through upload regions, as stores do.
  Acceleration structures are not evicted.
  Under the resource's record lock it is marked evicted,
  and the plans that name it (all register in its dependents,
  `resizing-and-raytracing.jl` B.6) are marked for restore. An evicted resource keeps its
  handle, size and cell. Its next use restores it like an unmapped arena: a
  marked run or an eager call allocates storage before any lock (this
  allocation may evict others, never what the running plan names), and under
  the locks a `Restore` change uploads the host copy and writes the cell, in
  front of its own work, in the same submission; a resource evicted after the
  run looked makes it start over. A plan that meets an evicted array it names
  is marked for restore even when it found it already evicted (the restore
  sets the mark, then the run starts over), so a new plan, or one whose own
  compile evicted what it names, restores it at its next attempt. Only a
  marked plan looks at what it names (a new plan starts marked), so an
  unmarked run walks nothing. Plans read
  the new address through the cell and stay valid; a resource a plan recorded
  by handle is not evicted. While evicted: a store restores it first and is
  pended after the Restore, a host read copies from the host copy, a
  `resize!` restores into storage of the new size, becoming a composite's
  member (a TLAS range) restores it first, `free!` frees the host
  copy.
  Pseudocode: `internals.jl` section 9. The cost is the copies, paid only
  while both do not fit.
- **Pressure policy, only in `acquire!`.** When an allocation would exceed the
  device's budget (`VK_EXT_memory_budget`, Metal's recommended working set,
  `cuMemGetInfo`/`hipMemGetInfo`), `acquire!` releases passed retirements and
  then takes these steps in order, retrying the allocation after each: unmap
  idle arena pages, trim empty blocks, collect garbage (Julia 1.12 disables
  finalizers on a thread that holds a `ReentrantLock`, `base/lock.jl:174`, so
  `acquire!` is never called with one held; the graph lock is a semaphore),
  wait for retiring storage oldest first while any is left, then
  evict the least recently used idle resources and wait for
  their storage's retirement tokens (the accesses in flight and the eviction
  copies), and only then throw `OutOfDeviceMemory`. It
  submits nothing but the unmapping of idle pages and the eviction copies, and
  never flushes a queue. Reacting to other processes taking VRAM is open.
- **Budget means one thing:** the device's budget minus other processes'
  usage.
- **Streamed weights** (open): a weight placed like a transient and uploaded by
  a copy node before its first use.

### Threads

No owning thread. Compile, record, run and free a graph on any thread; one
graph's compile, record, run and free are serialised by the graph's lock.
Declarations into one graph are serialized by the graph lock (a nested
declaration of the same task reuses it). Compiled kernels live in
the `Mantle.Kernel` at the kernel's `CodeInstance`, each field written once under
the kernel's lock. Eager operations share no scratch; each uses its own
temporaries.

### What a device type implements

| area | verbs |
|---|---|
| device | enumerate for `devices()`, create driver objects on first use, `caps` (facts only: limits, throughput and bandwidth for the roofline estimate, the sparse page size and sparse address space, whether readback memory is a heap of its own, whether the device drives a display, the driver's job timeout, a transfer queue and the cost of a cross-queue wait, the inline store limit and alignment), queue kinds, memory budget |
| channel | `submitnative!(ch, value, parts, waits, signals, present)`, `passed`, `waitfor`, `recycle!` |
| memory | `rawalloc`, `rawfree`, `createreserve`, `bindpages!`, `unbindpages!`, `placeimage`, `placeaccel`, counted-object release |
| compile | `compilekernel`, `kernelinfo`, `config` (name open), `compiledraw`, `compiletrace`, `accelsizes`, `compilebindings` (descriptor objects for the layout core decided) |
| record | `Builder`, the emit verbs above (`bindindices!` and `emitstore!` included), `finish` |
| acceleration structures | `countsource` (where a build gets its count: recorded into the command on Vulkan, read from memory on Metal), `instancerecordtype` (the driver's instance record), `acceladdress` (the address or resource ID for a structure's cell) |
| textures and images | `tableentry!`, `samplerentry!` (name open; host writes into the device-wide tables), `imageformat` (name open; element type to format) |
| arrays | `devicearg`, the vendor view constructor |
| windows | open (name open; creates the window's surface), acquire, present (through `submitnative!`) |
| library | candidates for selected operations, with their attributes |

Everything else is core: graph, compile steps, placement, queues, ordering,
lifetime, pressure, submission split and its budget, growth policy (which
arrays go into a reserve, `sparsethreshold`, and a reserve's size,
`reservesize`, both computed from `caps(dev)`), loops and conditions and their
fallbacks, cells and the head node, residency lists, arrays and their
GPUArrays interface, eager work, HWTLAS.

Loading: Vulkan or Metal is `@static include`d by platform inside `module
Mantle` (the platform is the trigger, and as separate modules a missed
`import` silently defines a new function; the reasons are in `Mantle.jl`'s
include block). ROCm and CUDA are package extensions that define their
methods qualified. Behaviour is selected per device: every hook takes the
device and dispatches on its type.

## Four device types, one model

The table holds for CUDA 13.3, HIP 7.1, Metal from the macOS 27.0 SDK, and
Vulkan.

| model piece | Vulkan | Metal | CUDA | HIP |
|---|---|---|---|---|
| channel native | `VkQueue` + timeline semaphore | `MTL4CommandQueue` + `MTLSharedEvent` | stream + counter by `cuStreamWriteValue64` | stream + `hipStreamWriteValue64` |
| recording | command buffer, SIMULTANEOUS_USE | compute ICB + render ICBs | `CUgraphExec`, built node by node | `hipGraphExec`, built node by node |
| dependencies in a recording | barriers | ICB barrier bits, consumer barriers | graph edges | graph edges |
| argument layout | flat slots, `NonWritable` | argument memory | open (open decision 2): kernel parameters by value need a node-parameter update on change; a pointer to the block costs a load | as CUDA |
| cell copy | head node (compute) | head kernel in the compute encoder | head node every root depends on | head node |
| `repeat!` / conditional kernels (one meaning on every device type; the mechanism per device type) | indirect counts: the loop head zeroes a group's, the head or a prepare node multiplies a `when!` launch's by its flag | device-written execution ranges | a kernel node whose grid the device sets, or conditional WHILE / IF nodes (chosen in phase 10) | kernel-entry predication (no conditional nodes in HIP 7.1) |
| library operations | our kernels | MLX kernels; MPSGraph as a foreign node | CUTLASS; cuDNN into our graph; cuBLAS as a foreign node | Composable Kernel; rocBLAS/hipBLASLt as foreign nodes |
| sparse arena | sparse binding | placement-sparse buffer | `cuMemAddressReserve`/`cuMemCreate`/`cuMemMap` | `hipMemAddressReserve`/`hipMemCreate`/`hipMemMap` |
| residency | implicit | one set per device on the queues, plus per plan for ICBs and pipelines | implicit | implicit |
| vendor array view | none | `MtlArray` | `CuArray` | `ROCArray` |

Hazards are computed the same on every device type; a device type answers
only how a transition is expressed.

## One call, one graph

A graph exists so core can schedule, allocate and split a unit of work as a
whole, which only works on the unit the caller runs. A consumer declares one
graph per call (encode an image, decode a click, generate an image, transcribe
a window), compiles it once per shape and runs it. Separate calls (SAM 2's
encode per image and decode per click) are separate graphs connected by a
device array; parts that always run together are one graph.

What makes every call one graph:

1. **Composition.** DNNKernels emits each exported ATen graph into a Mantle
   graph; an exported module called on graph arrays declares into their graph
   (`emitgraph!(g, …)`, generalised).
2. **Methods on `MantleArray`** for the work between modules: GPUFiltering,
   CFG and Euler steps, Hikari's `postprocess!`. GPUFiltering depends on Mantle
   directly, its ops are KA launches, and it has no internal `synchronize` and
   no readback: `goodfeatures` and `fitaffine`/`fithomography` use device
   reductions. An op with temporaries has two methods of one name, one
   allocating them (`KA.allocate` on the output's backend, a transient on a
   graph) and one taking them.
3. **`repeat!` with one body** for diffusion steps, sweeps, prompt chunks.
4. **Resizable arrays** for lengths only the device knows.
5. **State between calls lives in device arrays**: KV caches (resizable, or
   sparse where they grow large), MatAnyone's memory bank, Bonsai's positions.
6. **Autoregressive generation** is one call with a `Group` holding one token's
   launches and `repeat!(group, maxtokens; while_nonzero = running)` where
   sampling runs on the device; where it does not, one run per token with the
   host deciding between runs (a host node has no method for a `Group`).
7. **Host work inside a call is a host node**; trivial host work (an embedding
   lookup, a causal mask) is a missing device kernel.
8. **Submissions are core's.**
9. **Streamed weights** make QwenImage one graph (open).
10. **Load is a call:** load-time transposes, `hoistconstants` and quantization
    become one load graph per model.

Every runner's suite asserts per model call: one run (one per step where a
host policy decides between steps), no bytes copied outside the graph, host
waits only for the outputs it reads and at host nodes.

## RayMakie on the core model

Requirements and how the model meets each.

| requirement | how |
|---|---|
| Raster and traced plans do not recompile when plots are added or removed | A plot is a group of draws in the frame graph: an insert or delete compiles and records that group's draws and rewrites its pass's list. Everything else is a pending change: stores and resizes of attribute arrays; visibility as a conditional draw or an instance mask; a scene's rectangle as one array per scene; TLAS, BLAS and material changes as pending structure updates (Graphics; Acceleration structures). A plot that switches to another pipeline by type is deleted and inserted (switches by value are a `when!` flag): a new group of draws. A changed draw order rewrites the list. A move of an index array re-records the groups of draws that bound it. Still recompiling: kernels declared per plot, a new pass. |
| No frequent TLAS rebuilds, no needless reallocation | Moves refit; a new count builds with the exact count into existing storage; BLASes are kept and shared; arrays resize within their reserves or doubling regions; table slots are reused. |
| `update!(plot; …)` updates in place | One call per changed attribute: host data is `copy!` into the plot's array (a store, a resize if the length changed); another device array is a new group of draws for the plot over that array (no copy, no recompile); value ↔ per-element is the same array resized; zero elements is a zero count, the render object stays. Nothing is reallocated within capacity, nothing is uploaded twice, nothing waits. |
| The render loop does not walk render objects | A frame is `run!(framegraph)`: the recording holds every draw, sizes come from cells, pending changes ride in the frame's first submission. Plots whose inputs changed are on the screen's dirty list, put there by their ComputePipeline edges; the loop resolves those and nothing else. The run looks only at its inbox and at the shared resources its stages write (Ordering): unchanged plots cost it nothing. |

A simulation drawn without copies:

```julia
sim = Graph(dev)
vertices = MantleArray(dev, Point3f, 0; capacity = 1 << 20)   # device arrays: the plot reads them after sim's runs
normals  = MantleArray(dev, Vec3f, 0; capacity = 1 << 20)
faces    = MantleArray(dev, TriangleFace{UInt32}, 0; capacity = 1 << 21)
simulate!(sim, vertices, normals, faces)        # kernels; may write the lengths (lengthof)
mesh!(ax, vertices, faces; normals)             # the plot's draws read these arrays
run!(sim)      # writes them; the screen is marked for redraw; traced: the BLAS update is pending
run!(frame)    # the screen's frame graph draws the new contents over the new lengths; nothing recompiles
```

The frame's plan names the simulation's arrays through the plot's draws, so
the frame is among those arrays' plans (Ordering):
each simulation step waits for the frame's last run and puts the arrays into
the frame's inbox, and the frame's next run waits for the step. It draws over the arrays' current lengths (indirect), and in the
traced view applies the BLAS update the simulation's run left pending; the
BLAS over these kernel-written lengths is built over the geometry capacity
(Acceleration structures). The screen is one of the arrays' readers, so a run
that writes them marks it.

- **The frame graph** is one per screen: the overlay render passes, Hikari's
  integrator for its traced scenes, the composite of traced films, the
  present. Shadow and ambient-occlusion passes run only when their inputs
  changed: their draws are `when!` draws with no
  `clear!` before them (the map keeps its contents), or their compute launches `when!` launches;
  the store that changed an input sets the flag, and a kernel at the end of
  the pass clears it.
- **Hikari's integrator** is declared into the frame graph once, outside
  any plot's group. A traced plot is a TLAS range and material
  entries, pending structure changes, so adding or removing one recompiles
  nothing and keeps the integrator's declaration; only a plot that declares
  kernels of its own recompiles the plan. Its film and
  sample count are device arrays the screen holds (accumulation survives
  runs and recompiles; a viewport change is a scheduled `resize!`); its work
  queues are window-sized graph arrays (their contents matter only within a
  run), so a window resize within capacity re-records the window-sized
  segments (Windows). Scene changes never free it (`notify_scene_changed` and
  `invalidate!` go). A new Hikari material type is not a hit group: it is a
  new specialization of the shade kernel (materials are a `MultiTypeSet`),
  so the frame graph's plan recompiles.
- **No host waits inside a frame.** `fill_aux_buffers!` and `postprocess!` lose
  their `waitidle`; host reads (`colorbuffer`) are graph copy nodes.
- **Draw order** is the order of the plots' groups in the screen's plots
  group: a plot moving in front of another in 2D is a
  `sort!(screen.plots; by, lt, rev)`; per-frame
  transparency sorting is open decision 13.
- **RayMakie's own fixes**, in phase 11: grid lines computed from the camera on
  the GPU; lines' indices stored, not recreated; `visible` honoured for traced
  plots; the dependencies on the image's camera, stroke and glow colours and
  colormaps tracked, so no plot is left stale; glyph UVs stored in texels and
  normalized in the shader by the atlas's runtime size, so atlas growth is a
  new texture index and nothing else.

## Metal

- **Target:** MTL4 only, macOS 26 or later, Apple silicon. No Mantle path
  uses the legacy `MTLCommandQueue`: render passes, present, texture uploads,
  readbacks, acceleration-structure builds and traces all go through MTL4.
- **Channels:** `MTL4CommandQueue`s owned by Mantle, each with one
  `MTLSharedEvent`; a token is an event value. The device's free list holds
  (`MTL4CommandAllocator`, `MTL4CommandBuffer`) pairs; an allocator is reset
  when its last command buffer's token has passed. No Mantle path uses
  Metal.jl's `BatchedCommandQueue`, its roots, its autoflush or its task-local
  queue.
- **Ordering:** `barrierAfterQueueStages:beforeStages:` orders against earlier
  work on the same queue; across queues, event waits.
- **Residency of pipelines:** every pipeline an ICB names is in a residency
  set in use at commit (`MTL4ComputeCommandEncoder.h`).
- **Placement-sparse buffers:** support is read from
  `supportsPlacementSparse`.

### Compiled form

| node | compiled as |
|---|---|
| kernel nodes | one compute ICB per segment; pipelines compiled with `supportIndirectCommandBuffers`; arguments in the plan's argument memory; a barrier bit where `barriers!` put one |
| conditional kernels, loops | device-written execution ranges in plan memory |
| render passes | one render ICB per piece (a group of draws), executed in list order inside the run's render encoder; per-draw state an indirect render command cannot hold (open question 2 below) is emitted directly in the run's encoder |
| copy nodes | blit commands in the run's command buffer |
| library operations | MLX kernels as ordinary pipelines inside the compute ICB |
| residency | the pool's blocks, reserves and acceleration structures in one `MTLResidencySet` per device, added to every Mantle queue (`MTL4CommandQueue.h:317-349`), so a BLAS reached only through a TLAS is resident; per plan only its ICBs, argument memory and pipelines |

### Run: one command buffer per segment

Each segment is one command buffer, which signals the segment's token. In a
compute segment a compute encoder executes the segment's range of the compute
ICB (`executeCommandsInBuffer`); the first segment starts with the head kernel
and a barrier. A render segment's render encoder sets the attachment (acquired
drawable or target) and executes the render ICB. On Metal the submission
budget, not a cost of cutting, decides the number of segments, and the number
should be small: the first dispatch of a command buffer costs far more than
each further dispatch.

### Library kernels

Which candidate runs is the selection table's.

1. **MLX's Metal kernels** (MIT), compiled from their `.metal` source into our
   pipelines, dispatched inside the ICB: GEMM with fused epilogue, split-K,
   attention, quantized matmul. MLX's runtime is not used.
2. **Our own Julia kernels** where MLX has nothing or loses.
3. **`MPSGraphExecutable`** as a foreign node where both lose, never encoded
   into our command buffer because MPS may `commitAndContinue` it.

### Open questions

1. What does one event-chained `MPSGraphExecutable` call cost?
2. Which per-draw render state does an MTL4 indirect render command hold?

## How we build it

The refactor is built so that skipping one of its rules fails a test.

### Order of work

1. **Docs:** this plan, `internals.jl` and `examples.jl` (pseudocode and
   examples consistent with it), `api.md` (every type and verb with its
   contract, the ownership table, the invariants, the corner cases),
   `refactor-ledger.md`.
2. **Comments:** every comment and docstring in Mantle checked against the code
   and this plan before implementation starts.
3. **Tests**, red where the code does not satisfy them yet.
4. **Bugs:** the bugs listed in `refactor-ledger.md` (rows P0.B1 to P0.B15),
   and the performance baseline (the whole-plan benchmark and each suite's
   model calls).
5. Phases 1 to 12.

### Definition of done, per phase

- Its property tests pass on every device type that has joined the core model
  by that phase: Vulkan on the RX 7900 XTX and the RTX 4000 Ada and lavapipe
  (system Mesa) from phase 1, Metal from phase 9, ROCm and CUDA from phase 10.
  Until a device type joins, it keeps working through its current code, and its
  old paths are deleted in its own phase. A name a device type that has not
  joined still uses stays until that phase. Metal: `openrun`/`closerun!`
  (`metal/record.jl`), `batchqueue` (`metal/device.jl`), `DeviceArray`,
  `TransientBuffer`, `Buffer`, `LavaArray` (`metal/memory.jl`, `metal/ka.jl`,
  `metal/kernelinterface.jl`); ROCm: `TransientBuffer`, `DeviceArray`
  (`ext/MantleROCmExt.jl`). Tests of a device type that has not joined are
  rewritten in its phase.
- Every add and delete item of the phase is ticked in the ledger with its
  evidence: the test that proves it and the commit.
- One real use runs: the editor opened and played, a RayMakie frame, a model
  call.
- An independent review of the phase's diff against this plan, `api.md` and
  the ledger found no forbidden move, or its findings are fixed.
- The baseline, re-measured interleaved, shows no regression beyond noise.

### Forbidden moves

Each has one alternative: stop and report the blocker with evidence.

- Keeping an old path next to its replacement on a device type that has joined
  the core model; a compatibility shim; "for now".
- A fallback that returns `nothing` or a default for a required verb.
- `try`/`catch` around something that should not throw; defensive
  `hasproperty`/`isa` for a known type.
- An `isa`/`typeof` branch on a device or node type, a `hasmethod` probe, a
  Symbol kind.
- A change detected instead of queued by the function that makes it.
- A workaround in a consumer for a Mantle problem.
- Weakening, skipping or deleting a failing test; `@test_broken` on something
  the phase claims. Tests that pin behaviour this plan deletes are rewritten in
  the phase that deletes it, listed there.
- Reverting the requested refactor because tests fail.
- A new name for a concept that already has one; a new vocabulary name without
  a line in this plan.
- Recording a measurement as a driver defect without a minimal reproducer whose
  negative control fails.

### Checks that fail when a rule is skipped

- **Vocabulary frozen to `api.md`**, covering `src/vulkan`, `src/metal` and
  `ext/` (the `ext/` check exists in `test_mantle_owns_it.jl`), and checking
  arity: every required verb has a method at the signature core calls.
- **Driver calls in one place:** each driver entry point appears in exactly one
  function (`vkAllocateMemory` only in `rawalloc`, `vkQueueSubmit2` only in
  `submitnative!`, `vkCmdPipelineBarrier2` only in the emit verbs' shared
  barrier helper; the list is in `api.md`).
- **No mutable globals** outside a short allowlist (the device registry).
- **No probes** (`hasmethod`, `applicable`, `hasfield`) outside an allowlist;
  `@static Sys.is*` only in `Mantle.jl`'s include block.
- **Aqua and JET** on core.
- **Two devices open** for the whole suite.

Each check starts as `@test_broken` with the violations listed, and is promoted
to `@test` in the commit that makes it pass.

### Comment rules

- A comment states what the code does now, and why where that is not obvious.
  A false statement is corrected, whoever wrote it.
- Transitional code says so in one line and names the replacement
  ("Replaced in phase 3, recording-plan.md, Ordering").
- A claim about a driver or a measurement names where it was measured.
- History stays only where it still constrains the code.
- Plain technical language; no emphasis by capitals.
- Comments only: no code change in the same commit.

### The ledger

`refactor-ledger.md` lists every add and delete item of every phase and every
item found while working, with status and evidence. Every session starts by
reading this plan and the ledger.

## Phases

Code that a phase deletes is named, not located, so the names can be found
with grep. Each phase adds its mechanism, migrates every user on the device
types that have joined, deletes what it replaced there, and ends green on the
tests it names.

### 0. Docs, comments, tests, bugs

In the order of "How we build it". The bugs listed in `refactor-ledger.md`
(rows P0.B1 to P0.B15) are fixed first, each with a regression test that
fails before the fix. The tests, red:

- `test_interleaved_recording.jl`: tasks on one and on four threads build and
  run graphs and eager calls with a yield between every emit, 10 000 times;
  results exact.
- `test_builder_is_a_value.jl`: two builders alternating on one thread; while
  both are open, no field of the device, channel or pool references either.
- `test_record_is_pure.jl`: `record!` makes no pool allocation, no compile (the
  per-device compile counter) and no submission, and reads no mutable resource
  field (checked by passing resources wrapped in a checking type).
- `test_recorded_once.jl`: after runs 1 to 20 of SAM 2's encoder and decoder,
  each plan has been recorded exactly once.
- `test_window_resize.jl`: `resize!(win, w, h)` from the event thread while a
  run is in flight on another thread, then `run!`: correct image at the new
  size, no device-wide wait, no recompile, the old swapchain released after its
  last frame; a size beyond the window's capacity recompiles the graph once,
  at its next run.
- `test_run_is_submit.jl`: in steady state, `run!` allocates no device memory,
  compiles nothing, records nothing except the `BuildRecording`s a change marked,
  the parts that recorded a moved handle, a new group of draws with its pass's list,
  and the segments of a new swapchain generation, and submits each segment
  once.
- `test_runs_never_overlap.jl` (rule 9): one graph whose kernels read, modify
  and write a device array it names, run from two threads and with segments on two
  queues, 10 000 times with no host wait between runs; the result counts every
  run exactly once. Negative control: with the wait on the previous run's
  tokens removed it fails.
- `test_resize_is_queued.jl`: `resize!` of an array used by two graphs, one of
  them in flight, from another thread, directly and through a view: every
  stage sees either the old or the new size and address, never a mix (per
  stage; a run with host nodes may see the new state from a later stage
  on); the old region
  is released only after the in-flight run; the first `resize!` of an array a
  recorded plan launches over directly recompiles that plan once, and later
  ones (pages in a reserve, moves of a doubling region) do not; a first
  `resize!` that fits the storage moves nothing.
- `test_device_is_an_argument.jl`: the suite with two devices open; every
  resource, array, plan and cache entry belongs to the device the test passed;
  `Device()` called twice returns the same object.
- `test_any_thread.jl`: compile on A, record on B, run on C, free on D.
- `test_lifetime.jl`: drop every reference to a graph and its arrays while a
  run is in flight, `GC.gc()`; the run completes and the pool gets the memory
  back.
- `test_crossplan_order.jl`: two graphs sharing an arena part and a device
  array, on one channel and on two, interleaved 10 000 times, including a graph
  that reads then writes a shared array; results exact. Negative control: with
  graphs built with `ordering = Unordered()` (a keyword for tests only) it
  fails.
- `test_view_barriers.jl`: in one graph a kernel writes an array and a later
  kernel reads it through a view, a `reshape`, or the array's storage; the plan
  has a barrier between the two and the result is right. Negative control: a
  resource table keyed by object (today's `Graph.ids`), which finds three
  resources and no barrier. Reproducer: `dispatch!(g, fillk!,
  (b, 2f0), n)` then `dispatch!(g, addk!, (c, Mantle.storage(b)), n)`; the
  recorded plan has no transition from the write of `b` to the read.
- `test_memory_returns.jl`: compile and free a large graph beside a small one;
  reserved memory drops to the small one's need.
- `test_host_node.jl`: a host node between kernels, reading and writing graph
  arrays; results exact while another thread resizes an array the run uses and
  runs a second graph sharing its arena part during the host function (the
  second run waits until the first has submitted its last stage; both exact);
  a throwing host function leaves the device usable, no part held, nothing
  waiting forever, and the next run correct; a host function that runs its
  own graph, or runs, writes into or frees a graph sharing an arena part with
  it, throws, also while another thread waits for that graph's lock.
- `test_one_path.jl`: every eager entry (KA launch, broadcast, `fill!`,
  `copyto!`, `draw!(dev, …)`, `trace!(dev, …)`, `fft`, `mul!`) reaches
  `compilekernel`, with one submission per kernel launch, on every device type.
- `test_gpuref_slot.jl` (renamed from `test_hostwritten.jl`): `r[] = v` before the
  first run reaches the first run; a store while a run is in flight reaches the
  next submission that touches the slot; two plans and a group of draws
  baking one slot all read the stored value.
- `test_run_host_time.jl`: the real `run!`'s host time per run over plans
  that name 100, 2 500, 25 000 and 50 000 sync states with nothing changed is
  the same for all four.
- `test_eviction.jl`: two graphs that each need
  90% of the device run alternately, back to back with no host wait; weights
  evicted and restored before the run; results exact; zero recompiles;
  composite members and resources a plan recorded by handle never evicted.
- `test_composite_fuzz.jl`: `push!`, `delete!`, `resize!` (phase 5) and plot
  updates (phase 11) interleaved with runs
  on several threads; validation clean; reserved memory back at its baseline
  after `free!` (Memory).
- RayMakie's compile-counter session: the per-device compile counter over a
  scripted session (attribute changes, value ↔ per-element, another device
  array for an attribute, visibility, camera, layout, resizes within capacity, TLAS moves and
  count changes); zero compiles.

### 1. One compile path

Adds: `Kernel`/`KernelInfo`, `compilekernel`/`kernelinfo`, the per-device
compile counter, node types with their origin, `compile!(s, node)`, the
dependency graph kept and reduced with one hazard function shared with
barriers, barrier merging for conditional kernels, single conditional launches
(`when!(g or group, flag, kernel, args, ndrange)` on a device-side `Bool` flag),
groups as places in the order (`Group`, `repeat!(group, n)` over launches
only, `delete!(g, group)`, `sort!(group)`), with hazards derived between iterations (instead of `emitloophead!`'s full barrier), the estimated
submission split, placement stored in the plan, `sharing!`, the per-segment
access summary.

Renames: the compile phases become the step functions of "Compile", applied to
a `CompileState`; `run!(phase, ctx)`, the phase marker types and `PHASES` go,
so `run!` only runs a graph. `submissionbudget(dev)` moves from a device-type
hook (which answers 0.1 s for every device) to core. `AccessCache` and
`accesscache` fold into the `Kernel`.

Deletes: the three `compile_dispatch` bodies, `buildskernel`/`kisupported`,
`Pass.kind::Symbol`, `get_or_build_iter_plan` at emit, Lava's unlocked
`iterplans`/`launchplans`, `g.inbody`/`g.inloop`, the measured split
(`measure!`, `measuring`, `partitionranges`, `submissionbatches`,
`recordparts!`, `recordplan!`, `submitrecording!(ctx, ::RecordingParts, e)`,
the `Plan` fields `passcost` (unmeasured passes at `Inf`), `partition`,
`stamped`, `stale` as it is today (it drives the re-record inside `run!`; the
name is reused for the mark a change sets on a plan that recorded it by value,
`resizing-and-raytracing.jl` B.6) and `profiler` as the split's input, the
default query pool on every Vulkan plan, `timestamps` as a device hook for the
split), the host-read `when!` (a `Ref{Bool}` read on the host; the name is
reused) with `Pass.hostcond` and `RecordingParts.conds`/`batchconds`. Rewrites
`test_partitioned_recording.jl` and `test_run_allocates_nothing.jl`, which pin
the measured split, and `test_when.jl`.

Green: `test_record_is_pure.jl` for compiles, `test_recorded_once.jl`.

### 2. Memory

Adds: sync states per shared resource (the release engine needs their
last-use tokens), pool kinds for every allocation, `placeimage`/`placeaccel`,
counted driver objects, the release engine with last-use tokens and release at every
`acquire!` and `run!`, holds and `free!`, the last hold under the record's
lock (storage `Released()`, the old storage retired with the record's last
tokens; dropping pending changes and releasing the cell come with them in
phase 4), graph finalizers, tenancy by id, sparse arenas with queued mapping (Vulkan), the
pressure policy in `acquire!` with `makeroom!` folded in (its steps up to the
wait for retiring storage; eviction is phase 4), one meaning of budget,
tables with free lists. Composites that hold and never allocate come in
phase 5, with the TLAS and BLAS.

Deletes: `vk_alloc` and its accounting, Vulkan's `MemoryPolicy` engine, the
channel's `pending`/`retiring`/`taken`, `retire!(::SubmitChannel)`,
`reclaim!(::SubmitChannel)`, `rawfree` for acceleration structures as a
separate path, `gemm_split_scratch`/`gemm_split_retired`, `reduce_scratch`,
`remap!`/`remappable`/`movable`/`Plan.replaced`, `headroom`'s and
`arenaroom`'s two meanings.

Green: `test_lifetime.jl`, `test_memory_returns.jl`.

### 3. Channels, recording and ordering

Adds: core `SubmitChannel`/`Recording`/`Submission`, `Builder` and the emit
verbs on core's walk, queue kinds and core's queue policy, tokens per segment
and channel with claim and submit in one step, waits derived for the whole run
before its accesses are recorded, the lock order with `finally`, GC-safe host
waits, the graph lock as a non-reentrant semaphore and the only lock held
while the host waits, segments as lists of parts, host nodes with
`hostcuts!`, host-visible placement of their arguments, arena parts held by a
run until its last stage is submitted (`holdparts!` throwing on an
out-of-order wait, `lockgraph` throwing for a task that holds a part of the
graph's plan), and one head per stage.

Deletes: the hold stack, `spare`/`thread`, `Closed`, the Vulkan
`OneShot`/`Recording` structs, `ownthread`/`WrongThread`, `openoneshot`,
`oneshot`, `handover!`, `sealed!`, `maywait`, `submissionholds!`, `hold!`,
`holdleaves!` at emit, `submitlist!`, `Stamp`/`stampof`/`claim!`/`unclaim!`/
`crosswaits!`/`stamp!`/`syncbuf!`, the head barrier, `indirectbarrier!`,
`openrecording` returning `nothing`, `openrun`/`closerun!`/`submitrun!`,
`batchqueue` and the batch-queue verbs (Metal's uses stay until phase 9), `LavaBackend`'s upload queue,
`Outstanding`, the names listed under "Core types" (Metal's
`openrun`/`closerun!` stay until phase 9). Rewrites
`test/vulkan/test_hold_lifetime.jl`, `test_batch_queue_lifetime.jl`,
`test_closed_command_buffers.jl`, `test_hold_trace.jl` and
`test_holdleaves_stops_at_tlas.jl`.

Green: `test_interleaved_recording.jl`, `test_builder_is_a_value.jl`,
`test_run_is_submit.jl` for headless graphs, `test_runs_never_overlap.jl`,
`test_crossplan_order.jl`, `test_any_thread.jl`, `test_host_node.jl` without
its resize part (that part in phase 4), `test_run_host_time.jl`: a run's
host time does not grow with the number of resources it names.

### 4. Arguments and sizes

Adds: Lava's entry wrapper with fields in the argument slots and the block
marked `NonWritable`; the cell arena; each array's cell, shared by its views;
the head node and prepare nodes with the plan's slots as a resource (cell copy
table, launch commands); the `GPURef` as one slot in device memory whose
address plans and pieces bake, written by scheduled stores (no per-plan
entry, no host write waiting for a run, no host-written arrays);
array state with storage kinds (pool region, reserve with pages mapped as it
grows, committed region that doubles, evicted, released), the last hold
dropping pending changes and releasing the cell, and the first
`resize!` (in place when the new size fits, else a move into a reserve or a
doubling region); `trim!(a)`; pending
changes on sync states with a device-wide sequence (`change!`, `pendone!`,
`submitwith!`, `lockexpanded` with a composite's members locked with it and
the expanded uses in waits and recorded accesses), later segments of a stage
waiting on the first segment's token, inline stores within the `caps(dev)` limit through
`emitstore!`; dependents with registration after compile
and marking at the change; `LengthOf` and `SizedBy` resolved at compile;
indirect launches only over resizable arrays; `lengthof(a)` for lengths a
kernel writes (`resizing-and-raytracing.jl` B.1-B.7); eviction of idle
resources under pressure with `Restore` on their next use (it needs the dependents and pending changes of this phase).

Deletes: the pointer-to-inline-copy argument layout in Lava's entry wrapper and
Mantle's packer, the pushed patch table (`emitpatches!`, `Plan.patchtab`,
`Plan.pending_patches`, Vulkan's `Recording.patches`, `closerecording!` filling
it, `graph/packing.jl`'s patch half), `movelisteners`/`notify_move!`/
`listen_moves!`/`unlisten_moves!` as a patch mechanism, `DeviceRange(count; max)`.
Rewrites `test/vulkan/test_recorded_move_patch.jl`
(`test/metal/test_recorded_move_metal.jl` in phase 9).

Green: `test_resize_is_queued.jl`, `test_host_node.jl` in full,
`test_gpuref_slot.jl`, `test_eviction.jl`, the whole-plan benchmark within its noise of `plan_*.txt`.

### 5. Acceleration structures and textures

Adds: builds and refits as operations, `TraceRead`/`TraceBuild` declarations,
the core `TLAS` (hardware and software) with instance ranges, a device-side
range table and a doubling capacity, builds with all their uses, the texture
table; `BuildRecording`s marked by the changes that alter them, re-recorded
only when marked; pending builds coalesced per structure, Updates with
`builtcount`; BLAS storage with a doubling capacity, refits and builds into
it under the same count rules (`BuildRecording`), `MoveAccel` beyond it;
kernel-written counts built over the capacity where the device records the
count (Vulkan), from the cell where it reads it (Metal);
a change that would make a running plan stale applied after the run's end,
its submitter waiting, never the call (throwing from the plan's own host
function); composites that hold
and never allocate, with finalizers on every holder type; readers:
a structure's update pended by host changes and by
runs that write its arrays; per-range data through the range table; BLASes
shared by geometry; a `mask` per range entry for visibility.

Deletes: both HWTLAS implementations' bookkeeping, `AccelBuildContext` and
`MetalAccelBuildContext`, per-device-type `build_accel!`/`refit_tlas!`
signatures, `find_tlas_in_args`/`hasfield(:hwtlas)`, `Raycore.sync!` inside
`adapt`, `bind_textures`' per-call descriptor pool, the host concatenation of
every triangle on each add or remove (`tri_gpu`/`off_gpu` and
`HardwareAccel.triangle_data`), `HardwareAccel`.

Green: phase 3's tests with a ray-traced graph whose TLAS is rebuilt every
run, `test_composite_fuzz.jl` over TLASes and BLASes.

### 6. One array type and operations

Adds: `MantleArray <: AbstractGPUArray` on devices and graph devices,
`GraphDevice`, core's KA backend and GPUArrays interface (`storage`, `copy`,
`derive`, `mapreducedim!`), operations as methods calling
`dispatch!(location(a), …)`, the selection table (moving the Vulkan GEMM/GEMV
selection out of `vulkan/array/gemm.jl`, including `basealignment` in the key),
optional rewriters, `devicearg`, scoped vendor views (a hold, pending
changes and restores applied first); `GPURef` as a value slot only;
`attribute(a, i)`; graphics and ray-tracing pipelines from the device type's
cache under one key on every device type; images as resources with one format
table per device type (`imageformat`), the sampler table (`samplerentry!`);
`clear!`/`draw!` as nodes and render passes formed by compile, groups of
draws recorded on their own, index arrays through `bindindices!`, `when!`
around draws (indirect, count multiplied by the flag), the scene rectangle
as one array per scene applied in the vertex stage.

Deletes: `LavaArray`, `DeviceArray`, `Buffer`, `Transient`/`TransientBuffer`,
`vulkan/array/gpuarrays.jl`'s duplicated paths, the eager `mapreduce.jl`/
`accumulate.jl` implementations, the Vulkan and Metal KernelInterface host
copies, the immediate render and trace verbs, `ka_launch_indirect!`,
`ArrayLaunch`/`runlaunches!`/`dispatchlaunches!`/`kilaunch!` as separate
mechanisms, the pass-level `dispatch!`, `gate!`, `Mantle.Call`, the op structs
and `native_*` entries listed under Operations, the names under Graphics,
"Goes", except `Surface` and the `target_*` names, which go in phase 7 with
the window code that replaces them.

Green: `test_one_path.jl`, `test_view_barriers.jl`, GPUArrays' TestSuite on
every joined device type.

### 7. Windows and present

Adds: `Window` as a resource, swapchain generations with holds, acquire before
the run's locks with binary semaphores and OUT_OF_DATE handling, per-image
present segments, present in `submitnative!`, `resize!(win, …)` as a mark the
next run applies, window capacity and window-sized arrays and images placed
at it (a sparse arena part where the device has sparse residency, mapped per
generation for its size), window-sized images recreated and window-sized
segments re-recorded per swapchain generation, a recompile beyond the
capacity, the acquired image owned by the window (`keepimage!` on Again or a
throw), screenshots through the present segment.

Deletes: the swapchain rebuild inside acquire and present, `device_wait_idle`
there, `submit_and_present!`, `Surface` and `target_view`/`target_image`/
`target_extent`/`target_format`, the walked path in `execute!` (including the
walked headless plans for RayMakie's composited frame, which becomes a recorded
graph whose changing counts are resizable arrays), `DrawBinding` (RayMakie uses
it, `src/overlay_rendering.jl`), `boundbindings`, `boundcount`,
`boundindices`, `boundinstances`, the `recordsplans` hook, `Immediate`,
`recordable`, `rebind!`. Rewrites `test/test_headless_rebindable_draw.jl` and
`test_recordable_plans.jl`.

Green: `test_window_resize.jl`, `test_run_is_submit.jl` for a windowed graph
with overlays.

### 8. Devices are arguments

Every function takes its device or channel: no `vk_context()` and no
`default_bq`. Every hook takes the device and dispatches on its type.
The device registry returns one object per hardware device. The backend axis
goes (everything under "Devices"). `Lavapipe_jll` becomes an extension.

Deletes: `HostDevice` and `src/host/`, `DEVICES`, `lavadevice(ctx)`,
`VK_CONTEXT_REF` reads inside Mantle, `METAL_DEVICE`, `ROCM_DEVICE`,
`adopt!`/`adoptqueue!`, `LIVE_CONTEXTS`, `LAYER_SETTING_STORAGE`,
`PIPELINE_NO_COMPILE`, `WORKGROUP_FALLBACK`, `KERNEL_RECORDERS`,
`BACKEND_PROBES`, `FROZEN_*` as process globals, `COMMITTED`/`COLLECT`,
`Metal.device()`/`Metal.synchronize()` reads, graphics constructors keyed on a
KA backend, hooks without a device argument.

Green: `test_device_is_an_argument.jl`, `test_any_thread.jl`.

### 9. Metal on the core model

Open items 6 and 7 first. Then channels, compiled form and run, placement-sparse
arenas, MLX's kernels, the device-wide residency set on every queue plus a set
per plan for its ICBs and pipelines, the argument layout measured on
the M5.

Deletes: `LegacyQueue`, `LegacySubmission`, `MTL4Submission`,
`opensubmit!`/`closesubmit!`/`suspendsubmit!`, `replaywithcalls!`, the slot
ring, `encodes`/`canreplay` and per-run encoding, `ownflush!`,
`ensureresident!`'s run-time rebuild, `make_persistently_resident!`,
`walkedplan`/`planshape`, `MetalRecordedDispatch` as a type, every `MPSGraph*`
callable that MLX's kernels match.

Green: phase 0's tests on the MacBook worker; SAM 2.1's frame no slower than
today; RayDemo's crown matching today's image.

### 10. CUDA and HIP

CUDA backend new; ROCm rebuilt on the core model: graphs built node by node,
the argument layout decided by measurement (kernel parameters by value with
node-parameter updates, or a pointer into the block), counters written by the
stream as tokens, streams as queue kinds, VMM arenas, library candidates
(CUTLASS, cuDNN, Composable Kernel, cuBLAS/rocBLAS/hipBLASLt as foreign nodes).

Deletes: stream capture, `checkresolved`, the retirement-counter timeline with
full-sync waits, `ROCmCompiledDispatch` as a type, AMDGPU `Managed` tracking of
Mantle memory.

Green: phase 0's tests on the workstation (CUDA) and the Bosgame (HIP).

### 11. Consumers

- **VideoEditor:** `SCENELOCK`, `mantlethread`, `renderthread`'s Mantle branch,
  `runowned`'s `WrongThread` discovery and `onthread` for Mantle renderers go;
  the `repeat!(g, 1; while_nonzero = gate)` idiom becomes `when!`;
  the engine's device is passed everywhere.
- **Hikari:** `fill_aux_buffers!` and `postprocess!` become methods; builds and
  refits are operations; one `VolPath` is rendered by one task at a time (its
  graph's lock), with no consumer-side lock.
- **RayMakie:** as "RayMakie on the core model": one frame graph per screen,
  plots as groups of draws, so adding or removing a plot
  recompiles nothing; attributes as `MantleArray`s mutated in place,
  read with `attribute(a, i)`; `update!` as stores and resizes, another
  array object as a new group of draws; a
  dirty list instead of scene-tree walks; one scene-rectangle array per scene;
  Hikari's integrator declared into the frame graph, its film and sample count
  device arrays the screen holds and its work queues window-sized graph arrays,
  kept across scene changes; shadow and ambient-occlusion passes
  under `when!`; no `waitidle` in a frame; overlay compositing through the one
  path; its own fixes listed there.
- **JuliaVision:** one graph per call ("One call, one graph"). DNNKernels'
  GEMM, attention and convolution as `mul!`, `conv!`, `attention!` on graph
  arrays. PPOCRRunner's and UVDocRunner's `Mantle.storage(...)` calls go.
- **Paperhawk** (github.com/SimonDanisch/Paperhawk.jl): `DeviceArrays`
  (`src/gpu.jl`) and every `Mantle.storage(...)` call go. Its per-step plans
  (detector frame, spans, edges, thumbnail; page warp, cleanup, rotate, pack;
  preview tiles) become one graph per call, and host readbacks after `run!`
  become unified arrays (`unified = true`) or copy nodes.
- Comments contradicting Mantle are corrected in the commit that makes them
  wrong.

Green: each consumer suite with two devices open; each runner's per-call
assertion; RayMakie's compile-counter session (phase 0) with zero compiles.

### 12. Full suites, once

Mantle, Hikari, RayMakie, VideoEditor, JuliaVision and Paperhawk on Linux
(Vulkan, CUDA, HIP; Paperhawk's suite is `Pkg.test()` in its root and needs a
display and the GPU); Mantle, Hikari, RayMakie on the Mac. Then a "State" section is written
from the results.

## The design in short

1. **Mantle owns device memory**; vendor allocators are bypassed.
2. **One array type**, `MantleArray{T,N,L} <: AbstractGPUArray{T,N}` on devices
   and graph devices, with scoped zero-copy vendor views; a view is the same
   type, and read-only, unified and resizable are runtime properties.
   Lengths follow Julia's API; kernels are bounds-checked unless `@inbounds`.
3. **Devices, not backends.** `Device()` is a `LavaDevice` on GPU 0 (a
   `MetalDevice` on macOS); `CudaDevice`/`ROCmDevice` after `using CUDA`/
   `using AMDGPU`; lavapipe replaces `HostDevice`; one Julia object per hardware
   device. Loading stays by platform.
4. **A graph is not a device**: graph arrays live on its `GraphDevice`
   (`Mantle.Device(graph)`). It is a KernelAbstractions backend: KA launches and
   `KA.allocate` on graph arrays declare; `adapt(Mantle.Device(graph), x)`
   makes zero-copy aliases the graph names.
5. **Operations are methods** that launch kernels with `dispatch!(location(a),
   …)`; nothing is lowered at compile; selection is a call in the method, from a
   per-device table.
6. **Nodes, compiled per type** (`compile!(s, node)`), compiled data stored in
   the plan's own copies of the nodes.
7. **Rewrites are optional and generic**, matching on node origin and declared
   candidate attributes.
8. **One call, one graph**; host work inside a call is a host node that Mantle
   schedules.
9. **Queues and cross-graph ordering are core's**, tracked per resource with a
   token per segment and channel.
10. **Submission split from estimated cost**; no measurement-driven re-record.
11. **Changes are queued** by the functions that make them; nothing is detected.
12. **Runs of one graph never overlap**; one argument block per plan; value
    `GPURef`s are slots written by scheduled stores, so no host write waits for
    a run.
13. **Each array owns a cell** (address, size), shared by its views; the head
    node copies cells into flat, read-only argument slots; kernels never read
    cells; arrays are fixed until their first `resize!`, which makes them
    resizable: in place if the new size fits the storage, else a large array
    on a device with sparse residency moves into a reserve that grows by
    mapping pages, and any other into a pool region that doubles, each
    doubling a move; launches are indirect only over resizable arrays.
14. **Groups and the graph, nothing else**: a group is a value and a place in
    the order, `Group(g or group, name)`, declared into anywhere; it holds GPU
    work only, by which methods exist. `repeat!(group, n)` loops a group of
    launches (loops do not nest); `delete!(g, group)` removes one; a group of
    draws is recorded on its own, so adding or removing it recompiles nothing.
    A condition is one `when!` launch or draw with a `Bool` flag. Render passes
    are formed by compile. No `do`-block bodies.
15. **Sparse arenas that grow and shrink**; where sparse binding is missing, a
    graph compiled later gets a new arena part.
16. **MTL4 is the only Metal target**; MLX's kernels inside the recording,
    `MPSGraphExecutable` only as a foreign node.
17. **Per-image present segments** on Vulkan; a window resize is a mark the
    next run applies.
18. **One `Mantle.Kernel` per kernel at its `CodeInstance`**: the access
    per argument at declaration, the code at compile, per device the native
    handle and `KernelInfo`; no other kernel table, nothing global.
19. **Eager work is one submission per launch**: no batching outside graphs, no
    cached one-op plans.
20. **Holds and pressure:** one hold for the creator, one per naming graph
    and one per composite holding the resource (a TLAS range, a BLAS);
    pressure is handled only in `acquire!`.
21. **One device registry** per process, the only process-level state;
    one object per hardware device needs it.
22. **Read-only inputs**: `MantleArray(dev, host; readonly = true)` keeps a
    reference to `host` for restores after eviction; an mmapped `host` is just an
    `Array`; writes throw at the call from the per-argument access `dispatch!`
    derives at declaration.

## Open decisions

1. Kernels behind `mapreducedim!`, `accumulate!`, `sort!`: AcceleratedKernels'
   first, ported state-of-the-art kernels later as candidates.
2. CUDA/HIP argument layout, decided by measurement in phase 10.
3. TLAS and textures in shaders. `resizing-and-raytracing.jl` uses the TLAS by
   device address (`OpConvertUToAccelerationStructureKHR`) and textures through
   a device-wide table indexed by a slot value. With descriptors instead,
   traces become handle dependents of the TLAS.
4. `FuseBroadcasts`, and the names of rewriters and attributes.
5. Streamed weights: the keyword, and upload bandwidth against step time.
6. Cross-device `copyto!`.
7. Reacting to other processes taking VRAM (macOS memory-pressure events,
   DXGI's budget notification).
8. The name of the scoped vendor view (`withview`).
9. When a retired swapchain's images are free: a queue token covers the
   rendering but not the presentation; `VK_EXT_swapchain_maintenance1`'s
   present fences cover both where the driver has them.
10. Whether a repeated group (`repeat!(group, n)`) is recorded once per
    iteration into one segment, or recorded once and submitted per iteration.
    Either way a loop whose iterations together exceed the submission budget is
    cut between iterations.
11. Priorities and expected overlap: `Graph(dev; priority)` (name open), from
    which core decides queues and arena parts.
12. The profiling switch and `timings(g)`.
13. Draw order that changes every frame (per-frame transparency sorting).
    Options: (a) recompile; (b) a sorted draw list the GPU reads (multi-draw
    indirect); (c) re-record the render segment. A change of order between
    frames is `sort!(group; by, lt, rev)`, a rewrite of the pass's list.
14. Pageable eviction copies. Whether a device type writes the copy into
    pageable host memory directly (Vulkan: a page-aligned host allocation
    imported with `VK_EXT_external_memory_host` for the copy and released
    after it; Metal: a no-copy buffer over page-aligned memory) or through a
    staging region and a memcpy. To measure on Linux (NVIDIA, RADV) and the
    M5: whether `Readback` memory and Metal shared buffers are pinned, and how
    much physical memory evicting a writable array frees once the OS
    compresses or swaps the copy.

## Out of scope

- Video decode itself (formats, conformance); only its queue, memory and
  ordering are in scope.
