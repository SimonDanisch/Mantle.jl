# Mantle: one runtime model for every device

Status: plan, rewritten 2026-09-30. It replaces the version of 2026-09-26/28,
which three reviews found contradicting its own decisions and the code
(`experiments/plan-review/review-plan.md`, `review-internals-examples.md`,
`review-plan-vs-code.md`, under `/sim/Programmieren/VideoEdit/`). The
decisions it implements, with Simon's rulings of 2026-09-30, are listed in
`experiments/plan-review/decisions.md` and in "Decisions" below. The
measurements it rests on are in `experiments/header_indirection/` (section
"Measurements this plan rests on").

References are to Mantle 90e582f and JuliaVision 8063736. Code that goes is
named, not located, so the names can be found with grep.

Already done in the code: `repeat!` declares and compiles its body once
(692a4e1), launch extents are computed in one core function
(`ki_launch_extents`, 692a4e1), `record_maxpasses` is gone from Mantle and
JuliaVision. Committed but to be replaced: the measured submission split with a
re-record inside `run!` (692a4e1).

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
   The one process-level state is the device registry: `Device()` and
   `devices()` return the same object for the same hardware device.
6. **One path.** An operation on a device and the same operation on a graph run
   the same code; which of the two happens is decided by where the arrays live.
   There is no second implementation of any operation.
7. **One call, one graph.** A consumer declares one graph per call and never
   partitions it. Host work inside a call is a host node; Mantle schedules it.
8. **Changes are scheduled, not executed at the call.** `resize!`, `trim!`,
   `setindex!`, `copyto!` and `copy!` into device arrays, `GPURef` rebinds and
   acceleration structure changes only enqueue a request at the call (the
   data of a store is taken then): the requested size is what `length(a)`
   reports, and nothing physical changes. New storage, pages, the copy of the
   contents, the cell write and the retirement of the old storage are done by
   the next submission that touches the resource (a run, an eager call, a host
   read), which allocates what they need before its locks and applies them in
   front of its own work, inside the same submission, in call order across
   resources (Simon, 2026-10-05: "it should get scheduled"). A window resize marks the
   graphs that render to the window; their next run applies it. Runs submitted
   before a change see the old state, runs after it the new one. Nothing is
   detected, compared or swapped on the host.
9. **Runs of one graph never overlap.** A run's first segment waits on the GPU
   for the last tokens of the graph's previous run.

## Why the previous attempts did not get here

1. **Verbs moved to core, types stayed in Vulkan.** `record!`/`run!` moved in
   1954417; `Closed`, `OneShot`, `Recording` and `Emitter` stayed in
   `src/vulkan/runtime/`. When ec5b20a moved lifetime into core, core had no
   recording type to keep a hold list on, so it kept one on the channel: a
   positional stack found by `last(ch.holds)`.
2. **Recording did compile work.** `preparekernel` compiled, allocated argument
   scratch and walked argument trees during `record!`; descriptor pools were
   allocated there; draw counts were read from the host there.
3. **"Done" was a grep for deleted names.** No test checked a property;
   `test/vulkan/test_hold_lifetime.jl` asserts the hold stack exists.
4. **Metal never joined** (its own recording type, lifetime through Metal.jl's
   batch, a task-local queue, `METAL_DEVICE`); ROCm repeated the pattern.
5. **Workarounds went to the consumers**: `submissionholds!`, `maywait = false`,
   VideoEditor's `SCENELOCK`, `runowned`, `mantlethread`.
6. **Plans were patched instead of rewritten**, so they contradicted
   themselves. The 2026-09-26 version of this plan did it again.

## Bugs to fix first

Independent of the design, each with a regression test that fails before the
fix. Locations at 90e582f.

- `flush_stall_report` reads fields core's `Submission` does not have
  (`wait_semaphores` and `signal_value`, read from `Outstanding.payload`;
  `vulkan/runtime/command.jl:1140-1169`).
- `gemm_cm2!`/`gemm_cm2_sg!` are called as kernel factories
  (`vulkan/array/gemm_cm2.jl:324,362`).
- A conditional region can drop a barrier: barriers assume every node runs,
  and a skipped reader can hold the only barrier against a producer; the
  docstring claiming the opposite is at `graph/build.jl:348`, the rule at
  `sync/transition.jl:98-99`. Fix: a reader after a conditional region gets its
  own barrier against the producer; phase 1's barrier merging keeps that.
- Metal waits call `Metal.synchronize()`, which waits on the current task's
  queue, not the device's: `waitfor`/`waitidle` (`metal/device.jl:448,1141-1142`),
  `upload_texture_data!` (`metal/graphics.jl:180`), `readback_framebuffer`
  (`metal/graphics.jl:930`), `metal/hwtlas.jl:317`, `metal/record.jl:1197`.
- Walked Metal plans keep compile-time arrays with `patchable = true` and no
  move listener (`metal/record.jl`).
- Readback and copy leave images in TRANSFER_SRC while a later render pass that
  loads the image assumes COLOR_ATTACHMENT_OPTIMAL
  (`vulkan/graphics/framebuffer.jl`, `vulkan/graphics/pipeline.jl`).
- Validation checks read `VK_CONTEXT_REF[]` inside functions that hold a
  context, checking the wrong device (`vulkan/runtime/device.jl`,
  `vulkan/runtime/command.jl`, `vulkan/raytracing/pipeline.jl`).
- Arena regrowth dereferences tenant weak references without a `nothing` check
  (`memory/pool.jl:956-958`).
- `append!` on a `LavaArray` view ignores the view's offset
  (`vulkan/array/gpuarrays.jl:695`).
- The unlocked `live_buffers` delete from a finalizer, H.265 bitstream
  alignment, the two disagreeing texture format tables, `Ref` without
  `GC.@preserve` and UserID vs `instance_id` in `metal/trace.jl`, device
  features enabled without querying them (`review-plan-vs-code.md`, task 2).
- Fixed since the last version of this plan, regression test still missing:
  `destroy_pool!` calling `empty!` on an `Inbox` (2c7dc14). Fixed with a test:
  ROCm's `openrecording` arity (af50bfc).

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
  creation (to be checked against the loader source). It replaces
  `HostDevice`: CI and machines without a GPU test the real Vulkan path.
- A device is the KernelAbstractions backend: `Device <: KA.Backend`,
  `KA.get_backend(a) = device(a)` for an array on a device.
- **A graph is not a device.** Arrays declared into a graph live on its
  `GraphDevice`. Methods that run work exist only for real device types, so an
  operation without a graph method is a `MethodError`, never a silent eager run.

"A backend" in this plan means the code for one device type (`src/vulkan`,
`src/metal`, `ext/`); it is never a user-facing concept or type.

Deleted: `Backend` and the API tags (`VulkanAPI`, `MetalAPI`, `ROCmAPI`,
`HostAPI`, `WebGPUAPI`), `syncbackend`, `LavaBackend`, `backend(dev)`,
`kibackend`, `todevice`, `defaultbackend`, `availablebackends`, `eachbackend`,
`register_backend!`, `BACKEND_PROBES`, `Device(::KA.Backend)`,
`caps(::KA.Backend)`, `Device(api; select)`, `HostDevice` and `src/host/`
(including 692a4e1's KernelInterface executor there).

### Arrays

`MantleArray{T,N,D} <: AbstractGPUArray{T,N}` is the one array type, for every
location `D` (a device type or a `GraphDevice`). It replaces `LavaArray`, the
non-owning `DeviceArray` handle, `Buffer`, `Transient`/`TransientBuffer` and
`GPURef`'s array storage.

| constructor | lives on | contents |
|---|---|---|
| `MantleArray(dev, T, dims...)` | a device | its own pool region; operations on it run now |
| `MantleArray(location(a), …) do p … end` | where `a` lives | a temporary inside an operation: on a device a region retired with the call's token, on a graph a transient |
| `MantleArray(g, T, dims...)` | the graph | a transient: contents exist only during a run; placed in the graph's arena |
| `MantleArray(g, …; persistent = true)` | the graph | kept between runs and used only by this graph (the previous frame, a recurrent state); its own region |
| `MantleArray(g, …; hostwritten = true)` | the graph | a transient the host supplies each run with `x[:] = data` |
| `MantleArray(dev, …; memory = Readback())` (name open) | a device | in host-readable memory: a graph writes it, `Array(out)` after the run is a wait and a memcpy |
| `MantleArray(g, T, win)` | the graph | sized like a window; its length follows the window |
| `MantleArray(g or dev, …; capacity = n)` | either | resizable from the start, with a reserve for `n` elements; needed when a kernel writes the length (no host `resize!` tells core how much to reserve) |

- **Where contents live:** contents used only within one run are a transient;
  contents only this graph uses again in a later run are persistent on the
  graph; contents anything else uses (the host, eager code, another graph:
  weights, outputs, a KV cache shared between calls) are a
  `MantleArray(dev, …)`. Transients of a graph share its arena, and graphs
  that do not overlap share arenas: they cost the largest tenant, not the sum.
- **Each array owns a cell**: its address and its size. The cell lives in
  the device's cell arena and is the only place those values are written; every
  graph that uses the array gets them through its head node (Run). This is
  what "the array owns a GPURef that is simply shared" means. A transient that
  is not resizable has no cell: its address is an arena offset fixed by the
  plan.
- **Views share their parent's state.** A view, reshape or transpose
  of a fixed array (GPUArrays' `derive`) refers to its parent's state
  (storage, size, cell, holds, and the sync state with its freed mark), so a store through a view is
  ordered like one into the parent, a `resize!` of the parent reaches it, and
  a freed parent makes its views unusable. A view has no hold and no finalizer
  of its own; `free!` and `resize!` on a view throw. Its argument slots carry a
  static offset and its own dims and strides.
- **Every array is fixed until its first `resize!`** (Simon, 2026-09-30). A
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
  count (the array was fixed at compile), a temporary sized for fewer
  elements, a buffer handle in a command (after a move; open decision 17).
  Shrinking unmaps nothing, so a view never reads unmapped memory; neither
  does memory pressure. `trim!(a)` (name open) is the explicit call that
  unmaps a reserve's pages beyond the array's size; after it, a view past the
  size reads undefined values. Details and the pseudocode:
  `docs/resizing-and-raytracing.jl`.
- **On a graph device** `getindex`, iteration and `KA.get_backend` throw, so
  generic code raises an error instead of being traced. Broadcasting and Mantle's
  methods declare. The one exception is `x[:] = data` on a `hostwritten` array.
- **GPUArrays 11.5.14** requires `storage`, `copy`, `derive` and
  `mapreducedim!` (`host/abstractarray.jl:106,320`, `host/construction.jl:164`,
  `host/mapreduce.jl:8`), plus host copies and `Adapt`. It gives broadcast,
  `map!`, `fill!`, `copyto!`, indexing with index arrays, `repeat`,
  `adjoint`/`transpose`, `tril`/`triu`, a generic matmul, random numbers, all as
  KA kernels on `get_backend(dest)`. It has no `accumulate!`, `sort!`, BLAS or FFT.
- **Mantle's methods** on `MantleArray`: `mapreducedim!`, `accumulate!`,
  `sort!`, `sortperm!`, `mul!`, `fft`, broadcast, `fill!`, `copyto!` and every
  other function a graph needs (AcceleratedKernels' kernels launched through
  `dispatch!`, so they work on graph arrays too). GPUArrays' generic code serves the rest, on devices only.
- **Device side:** one conversion verb per device type, `devicearg(dev, a)`, to
  the type its compiler wants (`LavaDeviceArray`, `MtlDeviceArray`,
  `CuDeviceArray`, `ROCDeviceArray`). Views, reshapes and transposes resolve to
  parent region, offset and strides in one place; today Vulkan has several
  parallel walks (`vulkan/array/gpuarrays.jl`, `vulkan/array/gemm.jl`).
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

Why not the vendor array types as the primary array: every operation on them
that allocates (`similar`, broadcast results, `copy`, library workspaces) goes
through the vendor's allocator and stream, so rules 4 and 6 fail for all eager
work, and Vulkan has no vendor array package.

### Resources

| resource | memory |
|---|---|
| `MantleArray` | pool region (`Buffers` kind), or an arena slice for transients |
| image / texture | pool region, `Images` kind |
| acceleration structure | pool region, `Accel` kind; build and refit scratch is the structure's own, sized for its capacity and replaced with it (not a graph transient: a capacity change after compile would outgrow it) |
| `Window` | its swapchain or drawables, counted |
| `GPURef` | a host value box, copied into the entry by the run of a plan a store marked; or the box of a device array (`GPURef(dev, a)`, an `ArrayBox`): a cell holding its target's cell index, rebound by `r[] = b` (Graphics) |

Acceleration structures are declared with `TraceRead` by traces and ray-query
dispatches, and with `TraceBuild` by builds and refits (both access kinds exist
in `sync/usage.jl`; nothing declares them today).

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
when!(g, flag) do … end                                      # launches inside run only where flag ≠ 0 on the GPU
dispatch!(g, HostCall(align), (durations, idx); writes = (idx,))
repeat!(g, n) do i … end                                     # kernel launches only
repeat!(g, maxiters; while_nonzero = flag) do i … end
render!(g, attachments...) do p; draw!(p, shader, args, n); end
trace!(g, pipeline, accel, args, ndrange)
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
  an unmeasured entry uses core's cost estimate. Inside a `repeat!` body or a
  conditional only kernel candidates qualify, since a foreign node cannot run
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
- **Control flow is launches only.** A `repeat!` body may launch kernels and
  ray-tracing pipelines (`trace!`) and nothing else; a build, render pass,
  image copy or host call inside it is refused at declaration. Builds are
  explicit constructions (Simon, 2026-09-30); a render pass or image copy
  would need the image's layout to be the same on the way in and out of every
  iteration (today's `repeat!` docstring, `graph/build.jl`); a host call cuts
  the recording. Hikari's bounce
  loop traces inside `repeat!` (`Hikari/src/integrators/volpath/graph.jl:433`).
  A launch in a body the device may end is an indirect launch whose count a
  prepare kernel sets to zero, as today for traces (`emitprepares!`;
  conditional rendering does not cover `vkCmdTraceRaysKHR`, 1954417, and
  Mantle uses it for nothing): the loop head zeroes the body's counts once the
  flag is 0, and the loop start restores them in every run. A conditional
  launch works the same way (`when!`, below). ("Kernel launches only"
  in decisions.md B2.8 means launches of GPU work, traces included; Simon,
  2026-09-30.) `copyto!(dst, src)` between arrays in a graph is a copy kernel
  node, so it is allowed; copy nodes (images, host transfers) are not. Before each
  iteration a loop-head kernel writes `i` into the argument slots of the body
  launches that take it; they do not read it from a buffer, which is what
  they do today. Conditional execution is a block, `when!(g, flag) do … end`
  (Simon, 2026-10-02): every launch declared inside is tagged with the flag;
  launches only, like `repeat!`; blocks do not nest; a `when!` block may sit in
  a `repeat!` body. The launches inside are indirect, and whoever writes their
  launch commands multiplies the count by the flag (open decision 1).
  Conditions exist in graphs only: an eager call runs. The
  VideoEditor's `repeat!(g, 1; while_nonzero = gate)` idiom, which wraps one
  to three kernels (`VideoEditor/src/gpugraph.jl`), becomes `when!`. A build or
  copy that must be conditional is constructed explicitly by the application.
- **Host calls are nodes** (`HostCall(f)` with the arguments it writes). Mantle
  schedules them (Run, "Host nodes"); the user never splits a graph.
- **Lengths decided on the GPU** use a resizable array. The kernel that decides
  the length writes it into the array's cell through `lengthof(a)`, and
  declaring that launch adds a prepare node after it, which
  writes the launch commands of every later launch over the array. Today's
  `emitprepares!` runs a prepare kernel before every indirect pass instead.
  Head and prepare kernels write the launch command, the KA `ndrange` values
  and the dims slots of every launch over the array.
  `launchrange(a)` launches over the array's current length, indirectly once
  the array is resizable; it is the one name for this. Without a capacity
  a device-decided length is refused at declaration. On the host, `size(a)` is
  the host view; the length a kernel wrote is read with `devicesize(a)` (name
  open), which reads the cell back and waits for that write. `Array(a)` copies
  the cell and the data in one submission.

Today's `ArrayLaunch`/`runlaunches!`/`dispatchlaunches!`/`kilaunch!`
(`array/launch.jl`) are the start of this: one description, launched eagerly or
declared. They become the methods' `dispatch!` calls.

Deleted: `lower`, `opstyle` and `CallNode` of the previous plan, `select!` as a
compile step and the op structs (`Gemm`, `Conv2D`, `Map`, `Fill`, `Copy`,
`BuildAccel`, `RefitAccel`, `LibraryCall`) of the previous plan (none exist in
the code), `use`, `Mantle.Call` and `gate!`'s
host-call fallback, `librarygemm`, `native_gemm_dispatch!` and the other
`native_*_dispatch!`, `native_gemm_available`, `runscalls`, `staged_gemm_tile()`,
the `SGEMM_*` constants (the selection table replaces them), DNNKernels' direct
calls of `coopmat_gemm_dispatch!`.

### Kernels: `Mantle.Kernel`

```julia
struct Kernel{N}
    native::N          # VkPipeline | MTLComputePipelineState | CUfunction | hipFunction
    info::KernelInfo
end
```

- **Device-type verbs:** `compilekernel(dev, f, argtypes, config) -> (native, compiled::Bool)`,
  `kernelinfo(dev, native) -> KernelInfo`, and `config(dev, f)` (name open), the
  launch configuration the device type picks for a kernel (workgroup size). The device type answers from its
  compiler's own cache (Lava's, Metal.jl's, CUDA.jl's `cufunction`, AMDGPU's),
  per device, keyed on the kernel body, thread-safe; `compiled` says whether
  this call compiled, which is what core's per-device compile counter adds up.
  Core keeps no kernel cache and holds no lock other than the graph lock while
  a kernel compiles.
- **`KernelInfo`** has no optional fields: registers, shared memory, group size
  limits and occupancy (from the API where it reports them, from the device's
  limits where it does not), and the instruction and memory-operation counts of
  the compiled IR.
- **Core uses it for** launch geometry (one calculation, today's
  `ki_launch_extents`), per-node cost estimates and the submission split.
- `buildskernel`, `kisupported`, the three `compile_dispatch` bodies,
  `get_or_build_iter_plan` at emit and `WORKGROUP_FALLBACK` go.

### The graph and its plan

User code declares a graph and calls `run!(g)`, which is `run!(g.plan)`: the
plan is core's internal compiled state, one per graph. The first `run!`
compiles and records. A graph is never replaced for a structure change
(Simon, 2026-10-05: adding a plot must never throw the graph away; decisions
B3.16). Declaring into a compiled graph, or `delete!(g, h)` of the handle an
`insert!(g) do … end` block returned (names open), only marks the plan: the
next `run!(g)` compiles a new plan from the graph's nodes and retires the old
one after its last run. The graph, its windows, the resources it names and
the device arrays it reads stay; what must survive runs and recompiles (a
film) is a device array, since a graph array's contents live only during a
run. `free!(g)` releases the plan
and drops the graph's holds; a finalizer on the graph does the same for graphs
nobody freed. Introspection: `submissions(g)` (segments and cuts, including
those host nodes cause), `timings(g)`.

What can change after the first run, and what each change costs:

| change | made by | what happens |
|---|---|---|
| a `GPURef` value | `x[] = v` | marks the plans whose entry holds it; the next run of a marked plan copies it into its entry (waiting for its previous run) |
| an array `GPURef`'s target | `r[] = b` | pending: a cell write of the `GPURef`; no recompile; its readers pend their update (a TLAS an Update, or a Build if the length changed); a plan that bound the old target's buffer by handle (an index buffer) records the parts that bound it again when the Rebind is applied; nothing recompiles (open decision 17, ruled (c)) |
| a hostwritten transient's data | `x[:] = data` | copied into the entry (waits for the graph's previous run); kept on the host while the graph has no current plan, and written into the entry by the next compile |
| a device array's contents | `a[r] = data`, `copyto!(a, data)`, `copy!(a, data)` (resizes a vector to `length(data)`, as Base's `copy!`, without copying the old contents) | pending: a small store (within the device type's inline limit, `caps(dev)`: 64 KiB at 4-byte offsets and sizes on Vulkan) is written inline by the submission that applies it (Run); a larger one is a copy from its upload region. Either way the call copied the data once into an upload region (B3.18) |
| an array's size | `resize!(a, dims)` | pending: the first time, a cell write if the new size fits the storage, else a move with a copy into a reserve or a doubling pool region (Arrays); later, pages mapped in a reserve or a move to a doubled region, and a cell write; plans that recorded a direct count, a handle, or sized a temporary for fewer elements are marked stale at the call |
| a reserve's pages beyond the size | `trim!(a)` (name open) | pending: the pages unmapped, retired with the token of the submission that unmaps them |
| a window's size | `resize!(win, w, h)` from the window's event | stored; the next run of each graph rendering to it creates the new swapchain generation with its window-sized images at the new size, queues the cell writes of window-sized arrays, and re-records the segments that draw into window-sized targets and the per-image present segments (Windows); beyond the window's capacity the graph recompiles at that run |
| an instance's transform or mask | `tlas[h] = t`, hiding an instance | pending: records rewritten, a refit (Update) with the count of the last build |
| an acceleration structure's instance count | `push!`, `delete!`, a range's array length | pending: records rewritten, a build with the exact count; beyond the instance capacity new storage (capacity doubles); the change marks the `AccelCommand`s it alters, and plans re-record only those at their next run |

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
at its call, B3.18), and
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
| copy, head, `layout!` | the plan's own copies of the graph's nodes, a loop's body included (a recompile never touches an older plan's), the head node first, the entry and the `Bound` resources (below) |
| rewriters (only those the graph opted into) | the node list after optional rewrites, for example `FuseEpilogues`, `MergeRegions(attribute)` |
| `dependencies!` | an edge "q before p" wherever two nodes conflict on a resource, reduced so each node lists only its direct predecessors (what a CUDA/HIP graph node takes); kept, not discarded as today |
| `compile!(s, node)` for every node | the node's compiled data: a `Kernel` per kernel node, compiled draws and traces, library setup; one method per node type, a loop node's method compiles its body |
| `costs!` | an estimated time per node from its `KernelInfo`, ndrange and declared bytes |
| `schedule!` | node order and queue per node (Queues) |
| `hostcuts!` | independent nodes moved across host nodes, adjacent host nodes merged, copy nodes (compiled here) to host-visible memory for device arrays a host node reads, none for an array already in `Readback` memory; before `liveness!`, since it changes the order |
| `liveness!`, `place!` | transient placement in the plan's arena parts, stored in the plan; a hostwritten transient lives from the head on; arena parts shared with other graphs, host nodes or not |
| `sharing!` | edges between transients that share bytes (for CUDA/HIP graphs) |
| `barriers!` | barriers and image layouts, keyed by memory (a transient by its arena bytes, so transients sharing bytes conflict), with state merged over both outcomes of every conditional launch |
| `submissions!` | ordered segments, each with its queue, cut by the estimated budget, host nodes and present; waits between segments on different queues |
| `summary!` | every resource the plan shares with other graphs or eager code, with all its accesses per segment; composites (a TLAS, an array `GPURef`) included, their members expanded at submit (Ordering) |
| `bindings!` | the argument block layout, the cell copy table, the prepare kernels' tables, descriptor sets, the residency list |

**The head and prepare nodes are nodes.** Compile puts a head node first, and
one after each host node (Ordering); a
launch that writes an array's length on the device is followed by a prepare
node, added at declaration. The argument slots a plan holds for an array or a
`GPURef` (address, size, launch commands, `ndrange` values, value) are a
resource of the plan: the head and prepare nodes write them, launches read
them. So the head precedes every root, barriers follow the head, and the head's
reads of cells and the entry and its writes of hostwritten data into the arena
are in the access summary, all by the same rules as data.

What compile decides about a node is stored in the node (`CompiledNode`): its
queue, predecessors, barrier, compiled command, cost and argument slots. The
plan holds only what spans nodes: segments, placement, the argument block, the
cell copy table, the entry, recordings. A recompile runs the same steps and
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
total reaches the device's budget, with margin. `submissionbudget(dev)` exists
(692a4e1) as a device-type hook but answers 0.1 s for every device
(`graph/backend.jl:643`). It becomes core's: the policy (the driver's job
timeout, a smaller target on a GPU that drives a display) is computed from
`caps(dev)` fields a device type reports (its job timeout, whether it drives
a display), which is new work. A device type reports facts; core decides.
Measured node times, where profiling is on (`Graph(dev; profile = true)`,
name open), only calibrate the next compile. A
plan is never re-recorded because of a measurement.

Goes: 692a4e1's `measure!`, `measuring`, `partitionranges`,
`submissionbatches`, `recordparts!`, `recordplan!`,
`submitrecording!(ctx, ::RecordingParts, e)`, the `Plan` fields `passcost`
(unmeasured passes at `Inf`), `partition`, `stamped`, `stale` as it is today
(it drives the re-record inside `run!`; the name is reused for the mark a
change sets on a plan that recorded it by value, `resizing-and-raytracing.jl`
B.6) and `profiler` as the split's input, the default query
pool on every Vulkan plan, `timestamps` as a device hook for the split. SAM 2's
encoder and decoder plans are recorded twice today, in their first and second
run (`census_radv.txt`); after phase 1 each is recorded once.

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

`LavaBackend`'s upload queue, `external.jl`'s direct use of `compute_queue`,
`allocate_batch_queue!`, `batchqueue`, `supports_batch_queue`,
`release_batch_queue!` and the queue-slot fields go.

### Ordering across graphs and queues

Graphs share memory: transients of graphs that share an arena alias, and
persistent resources are named by several graphs and eager code. Inside a
graph the order is compiled and recorded once. Between graphs, and between
graphs and the host, it is settled by two things, neither of which looks at
every resource a run names (Simon, 2026-10-05, decisions.md B3.14):

- **Between graphs, through the resource when it is written** (decisions.md
  B3.15, option B: few recompiles, a cheap run). No plan keeps a list of
  other plans. Each resource's sync state has its `namers`: the plans naming
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
  then they wait for each other's whole last run (his rule: it is not
  performance-sensitive). A composite gaining a member changes no plan: the
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
  its first run waits like any later one. A run takes its inbox and handles
  only what is in it: it applies their pending changes in front of its first
  segment and waits for their other users. Its own accesses are recorded
  nowhere but in its last-run tokens. A run's host time grows with the
  named resources its stages write (its outputs, mostly) and with its inbox, not with the number
  of resources it names.
- **Who waits for what.** A run: its previous run (runs never overlap), the
  last runs of the other plans naming what a stage writes, and for its inbox
  resources their recorded accesses and the last runs of the other plans
  writing them. What a stage found is kept for the run's later stages.
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
  of its inbox resources (and, in its first stage, its arena parts), nothing
  else.
- **Tokens per segment.** Each segment of a run gets a token from the channel
  it runs on. Claiming a token and submitting it happen under the channel
  lock in one step, so submissions reach each queue in token order (a
  timeline semaphore's signal value must increase with every signal). A run
  computes the waits of all its segments before it submits any, so its
  segments on different queues do not wait on each other through shared
  state; their order inside the run comes from the dependency graph.
- **Composite reads.** Inside a plan, where compile does not know a
  composite's members, a build node conflicts with every earlier write (a
  full memory barrier), and a write through a `GPURef` with every later node.
  Between graphs the members are reached through their readers (above).
  Eager calls lock a composite's members with it (`lockexpanded`) and wait
  for and record their accesses: a read of a composite is a read of each
  member, a build a write of the structure and a read of each member, a write
  through an array `GPURef` a write of its target.
- **A run with host nodes is one submission per stage.** Each stage records
  its accesses when it is submitted, and each starts with a head node that
  copies the cells again. A change queued while a host function runs (a
  `resize!`, a store) is ordered before the later stages and seen by them,
  like any change between two runs. The call never waits: what to do about
  a running plan is decided when a change is applied.
  - A move of storage the plan recorded by handle (or a rebind of a
    `GPURef` whose target it bound by handle) marks the parts that recorded
    it; they are recorded again before they are next submitted (open
    decision 17 (c)), and the move, ordered after the run's earlier
    accesses, has copied what they wrote. Nothing waits.
  - A change the plan cannot follow mid-run (a length it launches over
    directly, a temporary it sized outgrown, a capacity change of a
    structure it sized temporaries from: compile decided those) is left
    pending by the plan's own later stages, with every later change on the
    same resource, and applied after the run (the next run recompiles).
    Any other submitter that would apply it waits on the host until the run
    has submitted its last stage (Simon, 2026-10-02: waiting over
    complexity), then starts over. Only a run past its first stage is
    waited for (one not yet there finds the stale mark under its stage-1
    locks and starts over); a plan with one stage is never waited for.
    From inside any run (any host function, on the run's task) such a
    submitter throws instead (a task the host function starts and waits for
    is not detected): it could wait in a cycle. No fairness: the waiting
    submitter starts over when the run ends and may wait again if the
    plan's next run got past its first stage first. Arena parts stay shared: a run holds its
  parts from before its first submission until its last stage is submitted,
  so another graph's run on the same part waits on the host for the whole
  run, host functions included, and then on its tokens (Simon, 2026-10-02:
  simple memory management, waiting rather than memory of its own). Parts are
  taken in id order, and a task that would wait for a part while holding one
  with a higher id throws (`holdparts!`; it would deadlock). A host function
  may not run, write into or free a graph that shares arena memory with its
  own graph: `lockgraph(g)` throws when the calling task holds an arena part
  of `g`'s plan. It may run another graph only if that run cannot wait on the
  parts its own graph's run holds. The pressure step
  leaves the pages of a running plan alone. A host function reads and writes
  only host-visible arena memory; copies before and after the cut move data
  between it and device arrays. Recording a later stage's
  accesses in advance, with a token signalled when that stage is submitted,
  was rejected: a GPU wait for a signal not yet submitted blocks the queue it
  sits on (Vulkan requires the signal to execute before forward progress can
  be made, `refs/vkdocs/chapters/cmdbuffers.adoc:3106-3117`; the submit thread
  of Mesa's shared Vulkan runtime blocks on a submit's waits until their
  signals are submitted, `src/vulkan/runtime/vk_queue.c:745-762` in the Mesa
  26.2.2 source of the `Lavapipe_jll` build; CUDA and HIP streams run in
  order).
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
  then sync states by resource id, then the channel lock. The pool lock, the inbox, a value `GPURef`'s lock,
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
  another run holding a shared arena part, for its previous run before it
  writes the entry, for a host node's inputs, for the swapchain acquire, and
  in `acquire!`'s pressure steps; `x[:] = data` waits under it for the
  graph's previous run.
- **Host waits are GC-safe.** A blocking driver wait is a `@ccall gc_safe =
  true` (Julia 1.12, `base/c.jl:278-296`), so a thread waiting on the GPU does
  not hold up a GC another thread needs.
- **Arenas:** graphs core expects to overlap get separate arena parts; the rest
  share one. Tenancy is a table from graph id to that graph's peak, never
  keeping a graph alive. Which graphs are expected to overlap comes from their
  priorities (open).

Replaced: the head barrier, `indirectbarrier!`, `Stamp`, `stampof`,
`crosswaits!`, `stamp!`, `syncbuf!`, the per-emit claims, `claim!`/`unclaim!`
from `hold!`.

### Core types

Existing names are kept where the concept is the same; their fields change.

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

struct Submission{T}
    token::T
    recycle::Vector{Any}            # an eager call's command objects, back to the free list once the token passes
end
```

- A segment is a list of parts submitted together: recordings written once by
  `record!`, and `AccelCommand`s (Acceleration structures). A run re-records
  only the `AccelCommand`s a change marked and, per swapchain generation, the
  segments that draw into window-sized targets and the per-image present
  segments (Windows).
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
- `SubmitChannel.holds`, `.spare`, `.thread`, `Outstanding`, `Closed`, `OneShot`,
  `oneshot!` and the non-bang `oneshot`, `hold!`, `ownthread`, `WrongThread`,
  `openoneshot`, `handover!`, `sealed!`, `maywait`, `submissionholds!`,
  `submitlist!`, `RecordingParts.batches`, `drain!(::SubmitChannel)` (the
  pool inbox's `drain!` stays), `flush!` as a user-facing
  verb, `takeholds!`, `emptyframe!`, `unhold!`, `pushholds!`, `dropholds!`,
  `acquire!(::SubmitChannel)`, `release!(::SubmitChannel, r)`,
  `waitfor!(::Stamp)` and the Vulkan `Emitter` go.

### Record

Core's walk (today `emitplan!`/`emitpasses!`) calls builder verbs; a device type
implements them on its native object.

```julia
b = Builder(device, channel)
emitdispatch!(b, node, cmd::CompiledDispatch, deps)   # direct or indirect, as compile decided
emittrace!(b, node, cmd::CompiledTrace, deps)
emitcopy!(b, node, cmd::CompiledCopy, deps)
emitbuild!(b, node, cmd::CompiledBuild, deps)
beginrender!(b, node, cmd::CompiledRender, deps); emitdraw!(b, node, cmd::CompiledDraw); endrender!(b)
bindindices!(b, node, storage)                         # an index buffer inside a render pass, by handle (open decision 17)
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
  `AccelCommand` or of a window-sized segment. Command objects come from the
  device's free list.
- `emitstore!` writes a store's bytes into `dst` inside the submission:
  `vkCmdUpdateBuffer` on Vulkan (at most 65536 bytes, offset and size
  multiples of 4, `refs/vkdocs/chapters/clears.adoc:800-805`; a store that
  does not meet this goes through staging); on Metal, CUDA
  and HIP the device type copies them into upload memory of that submission.
- Record compiles nothing, allocates no device memory, uploads nothing, submits
  nothing and reads no mutable resource field; the argument block's contents
  come from `bindings!`.

Today's `emit_dispatch!`, `emit_dispatch_indirect!`, `emit_barrier!`,
`emit_draw!`, `emit_trace!`, `emit_trace_indirect!`, `emithead!`, `emitkernel!`,
`emitinline!`, `emitbarriers!`, `emitupdate!`, `emitloophead!`,
`withpredicate` and today's `emitdispatch!` (`graph/backend.jl:196`,
`vulkan/graph.jl:1443`, whose name the builder verb takes over) fold into
these.

### Run

```julia
run!(g::Graph) = lockgraph(g) do                # throws for a freed graph, on re-entry from its host function,
                                                # and for a task that holds an arena part of g's plan
    while true
        applywindows!(g)                        # a window resize, marked by resize!(win)
        (g.plan === nothing || g.recompile) && recompile!(g)   # the first run; after a failed compile
        done(run!(g.plan)) && return            # Again: nothing claimed; an arena range was unmapped, the window changed,
    end                                         # the plan went stale, or (proposal) a resource it names was evicted after it looked
end                                             # Skipped: a minimized window, nothing submitted
```

An acquired swapchain image is owned by the window: a run that returns Again
or throws before presenting gives it back, and the next run takes it there
(internals.jl section 7).

`run!(pl)` writes the entry (waiting for the previous run only if a store
marked the plan), allocates what it may need before any sync-state lock
(unmapped arena pages, the swapchain image), then submits the first stage
under the records' locks, and the later stages after their host nodes.

- **Runs of one graph never overlap:** a run's first segment waits on the GPU
  for the previous run's last token on each channel. The host waits for the
  previous run only to write the entry. A run's other host waits (another
  run holding a shared arena part, a host node's inputs, the swapchain
  acquire, `acquire!`'s pressure steps) are under the graph lock and no other
  (Ordering).

- **The argument block** holds every dispatch's arguments, one allocation per
  plan, in host-visible device memory. Fields sit directly in the slots
  (address, dims, scalars), not behind a pointer to a copy as today, and the
  block is marked read-only to the kernels (`NonWritable`), which lets RADV use
  scalar loads. Both are changes to Lava's entry wrapper (phase 4). KA's
  `ndrange` values are slots too.
- **The entry** is one small host-visible buffer per plan: the `GPURef` values
  and the hostwritten transients' data of the next run. A host write into it
  waits until the graph's previous run has finished: `x[:] = data` waits at
  the call (not `run!`), and `run!` waits only for a plan that a store into a
  value `GPURef` marked. One entry, no ring (Simon, 2026-09-30).
- **The head node** runs first in every run, unconditionally (Compile: it is a
  node with uses, so everything that reads what it writes is ordered after it,
  on every queue; on CUDA/HIP it precedes every root). It applies the entry
  (values into their slots, copies of hostwritten data into the arena), copies
  every referenced array's cell (address, size) into the argument slots, and
  computes the launch command and `ndrange` values of every launch over a
  resizable array from the array's size and the kernel's group size, all from
  the tables `bindings!` built. A plan that names no device array, `GPURef`,
  hostwritten or resizable array has an empty head, which is not recorded.
- **Launch kernels never read cells.** A launch over a fixed-size array is
  direct, recorded once. A launch over a resizable array is indirect, from the
  command the head node (or a prepare kernel, for lengths decided during the
  run) wrote. The exceptions are Mantle's own: the head and prepare nodes, the
  TLAS offsets and records kernels (once per build), and kernels that write a
  length through `lengthof(a)`.
- **`GPURef`** has two forms. The value form, `GPURef(dev, v)`, is a host value
  box with its own lock: `x[] = v` sets it under the lock and marks the plans
  whose entry holds it; only a marked plan's run copies it into the entry,
  under the same lock, so a multi-word value is never torn. The array form,
  `GPURef(dev, a)` (an `ArrayBox`, Graphics), is device-only: its target is a
  device array and a root array (a view throws); `r[] = b` is a pending cell
  write, and rebinding to the current target is a no-op.
- **Stores into device arrays** (`a[r] = data`, `copyto!(a, data)`,
  `copy!(a, data)`) copy the data once at the call, into an upload region
  (host-visible system memory acquired for it, B3.18), and are pending until
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
    k = kernel!(dev, kernel, argtypes(args))          # compilekernel: the device type's compiler cache
    ch = channel(dev, Main())
    packed = acquire!(pool(dev), Unified(), argumentbytes(k))   # fallible: before any lock
    b = Builder(dev, ch)
    token = nothing
    try
        token = lockexpanded(uses(k, args)) do recs, x   # internals.jl section 11; composites bring their members
            foreach(usable, resources(args))              # freed through the creator's handle: throws, under the lock
            packarguments!(packed, k, args)               # arguments from the host view
            ext = launchextents(dev, k, resolve(ndrange, args))
            emitdispatch!(b, nothing, CompiledDispatch(k, ext, packed), Deps((), ()))
            first(submitwith!(ch, recs, expand(x, uses(k, args)), [finish(b)]))   # pending changes in front
        end
    finally
        retire!(pool(dev), packed, token === nothing ? () : (token,))   # after the call ran, or now after a throw
    end
end
```

- **Per launch:** a compiler-cache hit, a few commands from the device's free
  list (on CUDA and HIP the builder buffers the launch and issues it in
  `finish`, after the change prefix), one submission. A `GPURef`
  argument is packed with its value at the call. An array lent to vendor code
  (`withview`) or freed throws.
  The packed arguments and a multi-kernel operation's temporaries are regions
  retired with the call's token. No batching outside graphs, no cached one-op
  plans, nothing left open between calls.
- **Every eager entry is this path:** Mantle's methods on `MantleArray`s on a
  device, KA and KernelInterface launches, GPUArrays' generic code through its
  KA launches, `render!(dev, …)`, `trace!(dev, …)`, `copyto!` downloads. An
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

Goes: `begin_render_pass!`, `end_render_pass!`, `record_draw!` as a public path,
the `begin_pass!`/`draw_in_pass!`/`set_viewport!` family, the immediate
`trace_*` verbs, `ka_launch_indirect!`, the eager `mapreducedim!`/
`accumulate!`/`mul!`/`norm` code in `vulkan/array/` (their kernels become
candidates), the pass-level `dispatch!(p, …)`, `pass!`/`blit!`/`draw!` on a
device (they are `render!(dev, …)`).

### Windows and present

- A `Window` is a graph resource; its swapchain or drawable is its storage.
- **Vulkan:** everything except the segment that writes the swapchain image is
  recorded once; that last segment is recorded once per swapchain image (two or
  three small recordings), and `run!` submits the one for the acquired index,
  so choosing the image adds no GPU work. Rendering into a Mantle image and
  copying it to the acquired image each frame would cost a full-screen copy,
  about 0.08 ms on the RX 7900 XTX and 0.21 ms on the RTX 4000 Ada at
  3840×2160 (estimated from the copy bandwidth measured on each,
  `stride_*.txt`).
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
- **One graph presents to a window** (Simon, 2026-10-05, decisions.md B3.8):
  the graph that declares `render!` into it; a second graph declaring into
  the same window throws at declaration. Other graphs render into `Image`s
  this graph reads. So every acquire and present of a window happens under
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
  queue token also covers the presentation is open decision 10. A window has a capacity (the largest monitor's size unless
  given): window-sized arrays and images are placed at it. The arena part
  that holds window-sized transients is sparse where the device has sparse
  residency, and a generation maps only the pages its size needs, so a small
  window costs the capacity in address space only; without sparse residency
  the capacity is mapped. A resize beyond the capacity recompiles the graph
  at its next run, like a graph array beyond the size its slot was placed
  for. User code stays `run!(g)`.
  Today's rebuild inside acquire and present with a device-wide
  `vkDeviceWaitIdle` goes.
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
  concatenation, rebuild versus refit, BLAS collection, compaction) moves to
  core; today it exists in `vulkan/raytracing/hwtlas.jl` and `metal/hwtlas.jl`.
  Device-type verbs: `accelsizes(dev, path, n)`, `emitbuild!`, `countsource`,
  `instancerecordtype`, and the address or resource ID (`acceladdress`).
- **One core `TLAS` for hardware and software ray tracing**, with storage
  sized for an instance capacity that doubles; builds and refits carry the
  exact instance count, no padding (Simon, 2026-10-02; padding costs build
  time on AMD and NVIDIA and refit time on NVIDIA, measured in
  `experiments/tlas_sparse/`). Instances come in ranges (one instance, or one
  per element of a device array such as a meshscatter's positions); a
  device-side range table names cells. Each build, an offsets kernel computes
  each range's first record from the lengths in the source cells (a prefix
  sum) and a records kernel writes all instance records; both read cells. On
  every device type a plan's build or refit is an `AccelCommand`, its own
  small command buffer inside the segment's submission, with its barrier
  recorded first (a full memory barrier: the members are not known at
  compile). Vulkan records the count into it; on Metal, which reads the count
  from a cell, it still records the structure's storage. The structure keeps
  its `AccelCommand`s; a change that alters what one recorded (a new count, a
  `MoveAccel`, a Build that sets a new `builtcount`) marks it, and the run
  re-records only marked ones, after the submission's change prefix, so a
  refit takes the count the prefix's Build just set. A refit before any build
  is a Build. Nothing is compared at run; the plan itself is never recompiled
  or re-recorded for it.
  Vulkan's indirect build is not used: RADV and NVIDIA
  lack it, and on AMD's Windows driver it is slower than the exact count
  (measured). Metal reads the count from a cell (`instanceCountBuffer`).
  `AccelCommand` and `BuildAccel` apply to a TLAS and a BLAS alike.
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
  length is an Update. Both follow from decisions.md B3.2 and are listed for
  confirmation. Vulkan constrains both: an update may change instance
  definitions, transforms and vertex positions and nothing else, and may not
  turn a primitive or instance active or inactive
  (`accelstructures.adoc:159-170`). So the mask-0 records reference a BLAS
  (an instance referencing 0 is inactive, `:268`), the padding primitives are
  zero-area, not NaN (`:261-262`), and a run or store that writes a BLAS's
  index array pends a Build, its vertices an Update.
- **Ranges and shaders.** Traces and ray queries take the TLAS by address or
  resource ID through its cell, and textures sit in a device-wide table
  indexed through a slot (both proposed, open decision 4). A range holds its
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
  recompiles a plan or drops one (today Hikari's `notify_scene_changed` frees
  the integrator's plans on every material, light or structure change). BLAS
  builds follow the TLAS's count rules (an `AccelCommand` on Vulkan).
- **A write reaches the structures built from it.** Host changes (a
  `resize!`, store or rebind of a source) must reach the structures over it
  in any design, and do so through readers; the proposal (open decision 16)
  is the run's part, the last sentences below, which replaces declaring
  `refit!(g, t)` in every graph that writes a source. An array's readers are
  the composites built from it: a TLAS (a range
  over it), a BLAS (its geometry), an array `GPURef` (which forwards to its
  own readers) and a screen (a redraw mark). They are kept as an immutable
  snapshot, replaced under the array's lock and counted per reader: a TLAS
  with two ranges over one array counts twice, and `delete!` removes one. A
  `resize!`, store or rebind of an array with readers locks the array and its
  readers together (like `lockexpanded`) and pends each reader's update; the
  capacity a new total needs is allocated by the submission that applies the
  build. Ruled 2026-10-05 (B3.10):
  a run whose plan writes such arrays (compile lists them) pends
  `pendupdate!(reader, array)` on their readers after submitting each stage,
  so the next submission that touches the structure applies it, after the
  write. A mesh plotted from a simulation's arrays then follows the
  simulation with no host work.
- **Per-instance data is found, not concatenated.** Hit shaders reach a
  range's triangle data (the BLAS's geometry arrays by cell) and its material
  index through the range table, so adding a plot writes one range entry.
  Today every add or remove concatenates every triangle of the scene on the
  host and uploads it (`vulkan/raytracing/hwtlas.jl:805-830`), and a
  meshscatter whose count changes builds a new BLAS for its marker
  (`RayMakie/src/plots/meshscatter.jl:205-224`). A BLAS is shared by every
  range over the same geometry arrays.
- **Textures** are images in the `Images` kind (Graphics); bindings are
  compiled with the plan. `bind_textures`' descriptor pool per call goes.
  `Mantle.refit!` as the name for acceleration-structure refit collides with
  the transient/plan `refit!` (`graph/build.jl`); the plan-internal one is
  renamed.

### Graphics

A frame of a recorded scene costs its submission. Updates are stores,
resizes and rebinds; only adding or removing what is drawn recompiles, apart
from the cases open decisions 15, 17 and 18 list.

- **Pipelines are values.** `GraphicsPipeline` (`Rasterizer`) and
  `MeshPipeline` hold the stages (`VertexShader`, `FragmentShader`,
  `GeometryShader`, `MeshShader`, `ObjectShader`; tessellation, today a tuple
  in `GraphicsPipeline.tess_control`, becomes stage types too) and the fixed
  state (topology, blend, cull, depth). A draw's pipeline comes from the
  device type's cache through `compiledraw`, keyed on the stage functions, the
  device-side argument types, the fixed state and the target formats; equal
  keys give one pipeline on every device type (`npipelines` counts draws on
  Metal today, `metal/graphics.jl:1078-1101`). `RayTracingPipeline` likewise,
  through `compiletrace`.
- **One argument type for a value and a per-element attribute.** A drawn
  attribute is a `MantleArray`, or an array `GPURef` holding one
  (`GPURef(dev, a)`, device-only, Run): length 1 for one value
  for every element, length n for one per element. Its device-side type is the
  same either way, and shaders read it with `attribute(a, i)` (name open),
  `a[min(i, length(a))]` (1-based), the length coming from the slot. So a
  value and a per-element attribute are one pipeline, and switching between
  them is `copy!(a, data)`: a resize and a store, pending, no recompile (the
  length is never recorded). This replaces the stride `Attribute` fixes at
  declaration (`graph/build.jl:987-990`) and RayMakie's three ways (scatter
  forks the pipeline by argument type, lines and text expand values with
  `fill`, mesh switches at run time over dummy buffers). A value that changes
  every frame (a camera matrix) is a one-element array written by a store,
  which is inline: its bytes stay on the host until the next submission that
  touches the array writes them in its change prefix (`emitstore!`, Record),
  with no staging region and no host wait. The value `GPURef` stays for the
  scalars a kernel takes by value; a write to it waits for the previous run.
- **Render passes are nodes.** `render!(g, targets...) do p … end`; a target is
  `image => Clear(v)`, `Keep` or `Discard`, a depth image, or a window.
  `draw!(p, pipeline, args, count; instances, indices)`: counts are numbers or
  `launchrange`s (indirect over resizable arrays and array `GPURef`s); indices are
  a `MantleArray{UInt32}`, bound by handle (`bindindices!`), so a move of an
  index array marks the parts that recorded it, which are recorded again
  before their next submission (open decision 17, ruled (c)). A draw inside
  `when!(p, flag) do … end` is conditional: it is indirect, and the head (or a
  prepare node after the kernel that writes the flag) writes its count
  multiplied by the flag (open decision 1), so visibility is a store into the
  flag. State
  the API records by value (render area, viewport, scissor) is fixed per
  recording: a scene's rectangle inside the target is one array per scene,
  shared by its plots, which the vertex stage applies (a remap to the
  rectangle and four clip distances), so a layout change is one store per
  scene. The render area is the target's; a window-sized
  target's segments are recorded again with each swapchain generation
  (Windows).
- **A render pass is a list of pieces** (decisions B3.17; Simon: optimize
  complex GUIs without slowing large meshes and scatters, an insert in the
  millisecond range is fine, visibility must be cheap). `render!` returns the
  pass; its block declares the first piece, and each `h = insert!(p) do … end`
  adds one (a plot), `delete!(p, h)` removes it. A piece is compiled and
  recorded on its own (Vulkan: a secondary command buffer with dynamic
  rendering; Metal: its own indirect command buffer), with its own argument
  region and its own entries in the head's cell-copy table (an array the
  head launches over indirectly). The graph's next run applies inserts and
  deletes before its stages: it compiles and records the new piece,
  registers its resources, and rewrites the pass's list, which is one part
  recorded again; the plan is not recompiled and no other piece is recorded
  again. Every pass starts with a fixed barrier (compute, copy and build
  writes before it, then vertex, index, indirect and fragment reads), so a new
  draw adds no dependency inside the graph. Each draw keeps its own pipeline,
  index buffer and argument layout: a large mesh draws as it would alone.
  Visibility stays a store into the draw's `when!` flag; a changed draw order
  rewrites the list. A piece holds draws only; kernels per plot go into the
  graph with `insert!(g)` (B3.16, a recompile). Core Vulkan inherits no
  state into a secondary command buffer (`pipelines.adoc:131-133`), so each
  piece binds its pipelines and sets viewport and scissor itself; the pieces
  of a pass into a window are recorded per swapchain generation. Gate:
  inserting and deleting one plot in a frame graph of 100, 1 000 and 5 000
  plots; the time per insert must not grow with the number of plots, and a
  visibility change records nothing.
- **Images are resources like arrays.** `Image(dev, T, dims...)`, transients
  `Image(g, T, dims...)` and `Image(g, T, win)`; the format comes from `T`
  through one table per device type (`imageformat(dev, T)`, name open; today Vulkan's texture and target tables
  disagree with each other and with Metal's). Stores and
  `copyto!(img, array)`/`copyto!(array, img)` are pending changes or copy
  nodes, `Array(img)` a host read; layouts come from `barriers!`. A
  `Texture2D` is an image with an index in the device-wide texture table
  (`resizing-and-raytracing.jl` B.9): an upload of the same size and type
  writes the image it has, a new size is a new image under a new index, a
  store into the slot, never a recompile.
- **Samplers are values** (filter, address mode, anisotropy) in a device-wide
  sampler table, created on first use; a shader takes a texture index and a
  sampler index. Texture and sampler table entries are host writes into an
  unused index (`tableentry!`, `samplerentry!`), not recorded commands.
- **Screenshots.** `Array(win)` (name open) makes the next run that presents
  to the window copy the image into `Readback` memory before presenting (a
  second recording of the per-image present segment, made on first use per
  generation) and waits for that run. One behaviour on every device type
  (today Vulkan's `readback_window` errors between acquire and present and
  Metal's works only there).
- **Goes:** the hand-recorded path (`pass!`, `PassRecorder`, `viewport!`, the
  immediate `draw!` and `blit!`, which become `render!(dev, …)`), the
  `begin_pass!` family, `DrawBinding`/`rebind!`, `SampledTexture`,
  `bind_textures`, `Surface` and `target_view`/`target_image`/`target_extent`/
  `target_format`, `transition_image!` (no implementation on either device
  type), `readback_window`, `readback_framebuffer`, `readback_target`, `Attr`
  with a stored stride, `HardwareAccel` (no reader in Hikari or RayMakie).

### Memory

- **Device-type verbs only:** `rawalloc(dev, kind, bytes, constraint)`,
  `rawfree`, `createreserve(dev, bytes)` (address space with no pages mapped),
  `bindpages!`/`unbindpages!` (queue operations, under the channel
  lock), `placeimage`, `placeaccel`. Driver objects the API cannot place (an
  ICB, a pipeline, a descriptor pool, a swapchain image) are counted with the
  bytes the device type reports and released by the same engine.
- **Kinds:** `Buffers`, `Unified`, `Readback`, `Images`, `Accel`, with usage
  constraints. Vulkan's dedicated allocations (`vk_alloc`), per-image memory,
  video and external memory, Metal's argument memory, ICBs and textures, and
  ROCm's use of AMDGPU's pool all become pool allocations.
- **One release engine.** A region or counted object is retired with the
  tokens of its last use (`retire!(pool, x, tokens)`) and released as soon as
  they have passed: checked at every `acquire!` and every `run!`, not only
  under pressure. Finalizers push onto the lock-free `Inbox` and nothing else;
  they take no lock and call no driver.
- **Holds.** A device array (and every other resource graphs share: images,
  acceleration structures) has one hold for its creator, one for
  each graph that names it (including arrays inside broadcasts, keyword
  arguments, view parents and flags), and one for each composite holding it
  (a TLAS range, a BLAS, an array `GPURef`). `free!(a)` drops the creator's hold (once,
  whatever the number of threads calling it), `free!(g)` and the graph's
  finalizer the graph's (one function for both); at zero the resource is
  retired with the tokens of its last uses. After `free!`, using the handle
  throws, in eager calls and in declarations; adding a hold is a
  compare-and-swap that throws at zero, so a hold never goes from zero back to
  one. `run!` of a freed graph throws.
- **The last hold.** Under the record's lock the pending changes are dropped
  (staging retired, prepared storage released, deferred hold drops done), the
  storage becomes `Released()`, the old storage is retired with the record's
  last tokens, the resource leaves the eviction list (proposal, open decision
  16) and its cell is released. Every holder type (`ArrayBox`, `TLAS`, `BLAS`,
  a render object) has a finalizer that pushes its hold drops through the
  pool's inbox.
- **Composite resources hold; they do not allocate.** A TLAS, a BLAS, a
  texture, material or light table, a plot's render object is a struct of
  `MantleArray`s, array `GPURef`s and acceleration-structure storage, each created
  and held by it. Its changes are pending changes on its record, its growth
  is a `resize!` of its arrays or a `MoveAccel` whose commit retires the old
  storage, and its `free!` drops its holds. A submission that uses a
  composite locks its members with it and waits for and records their
  accesses (Ordering, composite reads). Only the hold machinery and change
  commits call `retire!`; no finalizer frees a member. So a composite written
  with these calls alone cannot free early, leak or free twice, and the tests
  of the primitives cover it. Today Raycore's TLAS finalizer finalizes arrays
  a recording and its `StaticTLAS` still use (`instanced-bvh.jl:372,400-416`),
  `Raycore.free!` requires an idle GPU (`:390-398`), and `free!` on the Vulkan
  HWTLAS does nothing (it has no finalizer). A fuzz test drives the
  composites (`test_composite_fuzz.jl`): `push!`, `delete!` and `resize!` on
  TLASes and BLASes green in phase 5, array `GPURef` rebinds added in phase 6,
  plot updates in phase 11; interleaved with runs on several threads,
  validation clean, reserved memory back at its baseline after `free!`.
- **Tables free their slots.** Material, medium, light and texture tables keep
  a free list; a deleted plot's slots are reused. Today they only grow
  (RayMakie's surface reshapes and volume changes add slots, emissive rebuilds
  add area lights).
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
  (today's `ArenaPart`, which already splits past the driver's maximum
  allocation, 34166c3); existing parts never move.
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
- **An idle resource can leave the device** (open decision 16, ruled B3.9/B3.10). Two
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
  copied to host memory (the `Readback` kind) in a submission of their own;
  `acquire!` waits for those retirement tokens outside every lock, then sweeps
  and retries. So `run!(dnn_1); run!(dnn_2)` back to back works while
  `dnn_1`'s weights are still in use on the GPU.
  Eviction acts only where readback memory is a different heap from device
  memory (a `caps(dev)` field); on a device with one heap (the Apple M5) it
  would free nothing and is skipped. Acceleration structures are not evicted.
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
  member (a `GPURef`, a TLAS range) restores it first, `free!` frees the host
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
  wait for retiring storage oldest first while any is left, then (proposal,
  open decision 16) evict the least recently used idle resources and wait for
  their storage's retirement tokens (the accesses in flight and the eviction
  copies), and only then throw `OutOfDeviceMemory`. It
  submits nothing but the unmapping of idle pages and the eviction copies, and
  never flushes a queue. 58a05a9's `makeroom!` (called from placement, flushing a queue) folds into
  this. Reacting to other processes taking VRAM is open.
- **Capacity means one thing:** the device's budget minus other processes'
  usage (today `headroom` and `arenaroom` count an arena's own region
  differently).
- **Streamed weights** (open): a weight placed like a transient and uploaded by
  a copy node before its first use.

### Threads

No owning thread. Compile, record, run and free a graph on any thread; one
graph's compile, record, run and free are serialised by the graph's lock.
Declarations into one graph are serialized by the graph lock (a nested
declaration of the same task reuses it). The compiler caches are the
device types', per device, thread-safe (today Lava's `FROZEN_*` are process
globals and `iterplans`/`launchplans` are unlocked). Scratch that eager
operations share today (`reduce_scratch`, `gemm_split_scratch`) becomes their
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

Loading stays as it is: Vulkan or Metal is `@static include`d by platform
inside `module Mantle` (the reasons are in `Mantle.jl`'s include block: the
platform is the trigger, and as separate modules a missed `import` silently
defined a new function). ROCm and CUDA are package extensions that define their
methods qualified. What changes is behaviour selection: every hook takes the
device and dispatches on its type. Today hooks that take no device
(`staged_gemm_tile()`, `initbackend!()`) are answered by whichever backend was
included, so a ROCm device next to Vulkan gets Vulkan's GEMM tile.

## Four device types, one model

Facts from the headers on the machines that run them: CUDA 13.3 (driver 595.99,
RTX 4000 Ada), HIP 7.1 (Radeon 8060S gfx1151 on the Bosgame), Metal from the
macOS 27.0 SDK on the M5, Vulkan on the RX 7900 XTX and the RTX 4000 Ada.

| model piece | Vulkan | Metal | CUDA | HIP |
|---|---|---|---|---|
| channel native | `VkQueue` + timeline semaphore | `MTL4CommandQueue` + `MTLSharedEvent` | stream + counter by `cuStreamWriteValue64` | stream + `hipStreamWriteValue64` |
| recording | command buffer, SIMULTANEOUS_USE | compute ICB + render ICBs | `CUgraphExec`, built node by node | `hipGraphExec`, built node by node |
| dependencies in a recording | barriers | ICB barrier bits, consumer barriers | graph edges | graph edges |
| argument layout | flat slots, `NonWritable` (measured) | argument memory (not measured) | open: kernel parameters by value need a node-parameter update on change; a pointer to the block costs a load (decided by measurement in phase 10) | as CUDA |
| cell copy / entry | head node (compute) | head kernel in the compute encoder | head node every root depends on | head node |
| `repeat!` / conditional kernels | indirect counts: the loop head zeroes a body's, the head or a prepare node multiplies a `when!` block's by its flag (open decision 1) | device-written execution ranges | conditional WHILE / IF nodes | kernel-entry predication (no conditional nodes in HIP 7.1) |
| library operations | our kernels | MLX kernels; MPSGraph as a foreign node | CUTLASS; cuDNN into our graph; cuBLAS as a foreign node | Composable Kernel; rocBLAS/hipBLASLt as foreign nodes |
| sparse arena | sparse binding | placement-sparse buffer | `cuMemAddressReserve`/`cuMemCreate`/`cuMemMap` | `hipMemAddressReserve`/`hipMemCreate`/`hipMemMap` |
| residency | implicit | one set per device on the queues, plus per plan for ICBs and pipelines | implicit | implicit |
| vendor array view | none | `MtlArray` | `CuArray` | `ROCArray` |

`needs_transition(::ROCmAPI) = false` goes: hazards are computed the same
everywhere; a device type answers only how a transition is expressed.

To validate on CUDA before building (contract, reference, negative control): an
explicitly built graph with a changed kernel node, cuDNN written into our graph,
a CUTLASS GEMM through NVRTC as a node, a WHILE node, a VMM arena growing and
shrinking under a live graph, two streams with a wait between them. On HIP, the
same minus the conditional node and cuDNN, with Composable Kernel.

## Measurements this plan rests on

All in `/sim/Programmieren/VideoEdit/experiments/header_indirection/`, on the
RX 7900 XTX (RADV, Mesa 26.2.3) and the RTX 4000 Ada (driver 595.x), interleaved
arms, GPU timestamps, every arm checked for correct output. Trusted files:
`v2_*.txt`, `plan_*.txt`, `dgc2_*.txt`, `dgc_counts_*.txt`, `census_radv.txt`,
`stride_*.txt`.

| question | result | file |
|---|---|---|
| kernels reading a per-array header (plus an indirect launch), small dependent dispatch | +349–488 ns on RADV, +117–132 ns on NVIDIA over today | `v2_*.txt` |
| argument loads marked read-only | RADV: today's layout 8036 → 6018 ns at 1M elements (−25%); NVIDIA: no effect | `v2_*.txt` |
| fields in the slots instead of a pointer to a copy | NVIDIA: −138 to −154 ns per small dispatch; RADV: −4 to −55 ns once read-only, within the spread | `v2_*.txt` |
| indirect instead of direct launch, small dependent dispatch | +194–244 ns RADV, +44–62 ns NVIDIA | `v2_*.txt` |
| head node copying cells into slots, per run | 1.3–1.9 µs RADV, 1.9–3.5 µs NVIDIA (1250–20000 slots) | `v2_*.txt` |
| whole recorded plan, 302 dispatches shaped like SAM 2's decoder, 60 resizable, lengths changing every run (host / GPU µs per run) | cells: 25 / 2032 RADV, 25 / 1838 NVIDIA; re-record: 79 / 2008 RADV, 405 / 1830 NVIDIA (a minimal recording, not Mantle's `record!`) | `plan_*.txt` |
| `VK_EXT_device_generated_commands` for one dependent launch | slower than indirect on both; NVIDIA: 15 µs preprocessing per dispatch, and preprocessed executes are ordered only by a barrier whose destination access includes `INDIRECT_COMMAND_READ` (a driver bug by the spec) | `dgc2_*.txt`, `dgc_counts_*.txt` |
| SAM 2 per call | encoder 619 dispatches, decoder 302; both plans recorded twice | `census_radv.txt` |

TLAS padding, sparse binding and submission cost: `experiments/tlas_sparse/`
(AMD Radeon 8060S on Windows, RTX 3070 Laptop), summarized at the top of
`docs/resizing-and-raytracing.jl`. One `vkQueueSubmit` costs about 24 µs of
host time on both, which is why changes wait for the next submission.

A model of the former design, in which a run locked the sync state of every
shared resource it names (lock in id order, check pending, record one
access, unlock; median of 2000 runs, Julia 1.12.7 on the workstation's CPU,
2026-10-02), cost 0.7 µs for 100 resources, 22.6 µs for 2 500 and 293 µs for
25 000; with array `GPURef`s a RayMakie screen of 5 000 plots names about
50 000 (not measured). That design is replaced by waits found through the
resources a stage writes and the inbox (Ordering; open decision 19, ruled):
a run's host time no longer grows with the resources it names. Phase 3
measures it.

Draw measurements behind open decision 16 (ruled 2026-10-02),
`experiments/draw_bench/` (RTX 3070
Laptop, Radeon 8060S on Windows, Apple M5; one draw of 1M and 4M points;
GPU timestamps on Vulkan, command-buffer GPU times on Metal; images checked
per variant against each other and a negative control):

- Reading an attribute as `a[min(i, n-1)]` (the benchmark shader's 0-based
  form of `attribute(a, i)`) costs the same as `a[i]` (within
  1% on all three) and as a stride fixed at declaration (within 2.2%), inside
  the spread. One value for all elements against one per element: 12-15%
  faster on the M5, 7-9% on NVIDIA, between 1% slower and 3% faster on AMD,
  whichever way it is read. One value read from a one-element buffer costs
  the same as a true uniform (push constants on Vulkan, the constant buffer
  on Metal): within 0.8% on all three (`uniform_summary.txt`).
- The scene-rectangle remap with four clip distances: within 1% on all three;
  `shaderClipDistance` is present on both Vulkan drivers and Metal compiles
  `[[clip_distance]]`.
- Indirect instead of direct draws costs nothing while draws have work (≥ 200
  points each). For tiny draws (1-6 points) it adds 14-22 ns per draw on
  NVIDIA and 28-34 ns on AMD (10 000 draws: +0.14-0.34 ms per frame), nothing
  on the M5. One multi-draw indirect call is the cheapest on Vulkan (4.2 ns
  per draw NVIDIA, 12.5-13.5 ns AMD). On the M5 a render ICB is slower than
  direct encoding for tiny draws (60.2 vs 36.6 ns at 10 000 × 6 points) and
  equal for draws with work.

Not measured: the flat layout through Lava's real entry wrapper (emulated with
`Ptr` and `Int` arguments), the head node inside a real plan, Metal, CUDA,
HIP.

## One call, one graph

A graph exists so core can schedule, allocate and split a unit of work as a
whole, which only works on the unit the caller runs. A consumer declares one
graph per call (encode an image, decode a click, generate an image, transcribe
a window), compiles it once per shape and runs it. Separate calls (SAM 2's
encode per image and decode per click) are separate graphs connected by a
device array; parts that always run together are one graph.

### Where JuliaVision is today

Counted at 8063736 (`review-plan-vs-code.md`, task 4). DNNKernels emits each
exported ATen graph into one Mantle graph, and `emitgraph!(g, …)` already
composes into a caller's graph (Trellis2 uses it).

| pattern | runners |
|---|---|
| a model exported as several modules, joined on the host | 7 (QwenImage 10 graphs, Trellis2 5–7) |
| `replay!` copying arguments into the plan's inputs (`driver.jl`) | 8 (Whisper about 80 MB in and 18 MB out per token) |
| a host loop with a count fixed in advance | 5 |
| host compute in the middle of a call | 7 or more |
| helper work as its own plan or `runonce!` | 5 (Trellis2: 44 `runonce!` lines) |
| autoregressive loop with a readback per step | 3 |
| an output length only the device knows | 2, plus device-decided loop exits |
| `record_maxpasses` | 0 (removed) |

### What removes every split

1. **Composition.** An exported module called on graph arrays declares into
   their graph (`emitgraph!` generalised); `call`, `replay!`, `planfor` and
   `recordedplan`'s hooks go.
2. **Methods on `MantleArray`** for the work between modules: GPUFiltering,
   CFG and Euler steps, Hikari's `postprocess!`. GPUFiltering's internal
   `synchronize` and readbacks go.
3. **`repeat!` with one body** for diffusion steps, sweeps, prompt chunks
   (Trellis2 does this already).
4. **Resizable arrays** for lengths only the device knows.
5. **State between calls lives in device arrays**: KV caches (resizable, or
   sparse where they grow large), MatAnyone's memory bank, Bonsai's positions.
6. **Autoregressive generation** is one call with `repeat!(g, maxtokens;
   while_nonzero = running)` where sampling runs on the device; where it does
   not, one run per token with the host deciding between runs (a host node
   cannot sit inside `repeat!`).
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

Requirements (Simon, 2026-10-02) and how the model meets each. What RayMakie
does today is from reading `dev/Makie/RayMakie` and `dev/Hikari` at e6e1aa9b0
and c5ca85a.

| requirement | how |
|---|---|
| Raster and traced plans do not recompile when plots are added or removed | A plot is a piece of the frame graph's pass (B3.17): an insert or delete compiles and records that piece and rewrites the pass's list. Everything else is a pending change: stores, resizes and rebinds of attribute arrays; visibility as a conditional draw or an instance mask; a scene's rectangle as one array per scene; TLAS, BLAS and material changes as pending structure updates (Graphics; Acceleration structures). A plot that switches to another pipeline by type is deleted and inserted (open decision 18; switches by value are a `when!` flag): a new piece. A changed draw order rewrites the list. A move of an index array re-records the pieces that bound it (open decision 17, ruled (c)). Still recompiling: kernels declared per plot, a new pass. |
| No frequent TLAS rebuilds, no needless reallocation | Moves refit; a new count builds with the exact count into existing storage; BLASes are kept and shared; arrays resize within their reserves or doubling regions; table slots are reused. |
| `update!(plot; …)` updates in place | One call per changed attribute: host data is `copy!` into the plot's array (a store, a resize if the length changed); a device array rebinds the attribute's array `GPURef` (no copy); value ↔ per-element is the same array resized; zero elements is a zero count, the render object stays. Nothing is reallocated within capacity, nothing is uploaded twice, nothing waits. |
| The render loop does not walk render objects | A frame is `run!(framegraph)`: the recording holds every draw, sizes come from cells, pending changes ride in the frame's first submission. Plots whose inputs changed are on the screen's dirty list, put there by their ComputePipeline edges; the loop resolves those and nothing else. The run looks only at its inbox and at the shared resources its stages write (Ordering; open decision 19, ruled): unchanged plots cost it nothing. |

Today, against those: a windowed frame is never recorded and is walked and
re-packed every frame (`recordable` in `graph/kalaunch.jl:1353`); each frame
walks the scene tree four to five times and builds a `frame_signature`; one
cached plan per key, so a toggled tooltip or visibility rebuilds it; every
material, colour or light change frees and recompiles the path tracer's plans;
`fill_aux_buffers!` and `postprocess!` end in `waitidle`, twice per traced
scene per frame; axis grid lines (`linesegments`) are rebuilt and uploaded on
every camera move; lines get a new index buffer on every data change; a
scalar ↔ vector switch of a scatter attribute is not handled.

A simulation drawn without copies:

```julia
sim = Graph(dev)
vertices = MantleArray(dev, Point3f, 0; capacity = 1 << 20)   # device arrays: the plot reads them after sim's runs
normals  = MantleArray(dev, Vec3f, 0; capacity = 1 << 20)
faces    = MantleArray(dev, TriangleFace{UInt32}, 0; capacity = 1 << 21)
simulate!(sim, vertices, normals, faces)        # kernels; may write the lengths (lengthof)
mesh!(ax, vertices, faces; normals)             # the plot's array GPURefs point at these arrays
run!(sim)      # writes them; the screen is marked for redraw; traced: the BLAS update is pending
run!(frame)    # the screen's frame graph draws the new contents over the new lengths; nothing recompiles
```

The frame's plan names the plot's array `GPURef`s, whose targets are the
simulation's arrays, so the frame is among those arrays' plans (Ordering):
each simulation step waits for the frame's last run and puts the arrays into
the frame's inbox, and the frame's next run waits for the step. It draws over the arrays' current lengths (indirect), and in the
traced view applies the BLAS update the simulation's run left pending; the
BLAS over these kernel-written lengths is built over the geometry capacity
(Acceleration structures). The screen is one of the arrays' readers, so a run
that writes them marks it (readers: open decision 16, ruled B3.9/B3.10).

- **The frame graph** is one per screen: the overlay render passes, Hikari's
  integrator for its traced scenes, the composite of traced films, the
  present. Shadow and ambient-occlusion passes run only when their inputs
  changed, not every frame as today: their draws sit inside `when!(p, flag)`
  with a `Keep` target, or their compute launches inside `when!(g, flag)`;
  the store that changed an input sets the flag, and a kernel at the end of
  the pass clears it.
- **Hikari's integrator** is declared into the frame graph once, outside
  any plot's `insert!` block, so adding or removing a plot recompiles the
  frame graph's plan but keeps the integrator's declaration. Its film and
  sample count are device arrays the screen holds (accumulation survives
  runs and recompiles; a viewport change is a scheduled `resize!`); its work
  queues are window-sized graph arrays (their contents matter only within a
  run), so a window resize within capacity re-records the window-sized
  segments (Windows). Scene changes never free it (`notify_scene_changed` and
  `invalidate!` go). A new material type is a new hit group in the ray-tracing
  pipeline and recompiles the frame graph, like a plot of a new kind (open
  decision 18).
- **No host waits inside a frame.** `fill_aux_buffers!` and `postprocess!` lose
  their `waitidle`; host reads (`colorbuffer`) are graph copy nodes.
- **Draw order** is the order of the pass's pieces: a plot moving in front of
  another in 2D rewrites the list (B3.17); per-frame transparency sorting is
  open decision 15.
- **RayMakie's own fixes**, in phase 11: grid lines computed from the camera on
  the GPU; lines' indices stored, not recreated; `visible` honoured for traced
  plots; the dependencies RayMakie misses (the image's camera, stroke and
  glow colours, colormaps), which today leave plots silently stale; glyph UVs
  stored in texels and normalized in the shader by the atlas's runtime size,
  so atlas growth is a new texture index and nothing else (today UVs are
  stored normalized and the atlas width is a compile-time constant,
  `RayMakie/src/overlay/scatter.jl:195,426`).

## Metal

Verified on the MacBook worker (Apple M5, macOS 26.6.2, SDK 27.0) against the
SDK headers, MLX's source and measurements.

- **Target:** MTL4 only, macOS 26 or later, Apple silicon. The legacy
  `MTLCommandQueue` path goes, including its use for render passes, present,
  texture uploads, readbacks, acceleration-structure builds and traces.
- **Channels:** `MTL4CommandQueue`s owned by Mantle, each with one
  `MTLSharedEvent`; a token is an event value. The device's free list holds
  (`MTL4CommandAllocator`, `MTL4CommandBuffer`) pairs; an allocator is reset
  when its last command buffer's token has passed. No Mantle path uses
  Metal.jl's `BatchedCommandQueue`, its roots, its autoflush or its task-local
  queue.

### Compiled form

| node | compiled as |
|---|---|
| kernel nodes | one compute ICB per segment; pipelines compiled with `supportIndirectCommandBuffers`; arguments in the plan's argument memory; a barrier bit where `barriers!` put one |
| conditional kernels, loops | device-written execution ranges in plan memory |
| render nodes | one render ICB per render node, executed inside the run's render encoder; per-draw state an indirect render command cannot hold (open item 7) is emitted directly in the run's encoder |
| copy nodes | blit commands in the run's command buffer |
| library operations | MLX kernels as ordinary pipelines inside the compute ICB |
| residency | the pool's blocks, reserves and acceleration structures in one `MTLResidencySet` per device, added to every Mantle queue (`MTL4CommandQueue.h:317-349`), so a BLAS reached only through a TLAS is resident; per plan only its ICBs, argument memory and pipelines |

### Run: one command buffer per segment

Each segment is one command buffer, which signals the segment's token. In a
compute segment a compute encoder executes the segment's range of the compute
ICB (`executeCommandsInBuffer`); the first segment starts with the head kernel
and a barrier. A render segment's render encoder sets the attachment (acquired
drawable or target) and executes the render ICB. Measured on the M5
(`metal/device.jl:217-234`): about 30 µs for the first dispatch of a command
buffer, about 1.3 µs per further dispatch; so on Metal the submission budget,
not a cost of cutting, decides the number of segments, and it should be small.

### Library kernels

Which candidate runs is the selection table's. For fp16 GEMM on the M5 the
kernels rank MPSGraph, MLX, ours (item 5).

1. **MLX's Metal kernels** (MIT), compiled from their `.metal` source into our
   pipelines, dispatched inside the ICB: GEMM with fused epilogue, split-K,
   attention, quantized matmul. MLX's runtime is not used.
2. **Our own Julia kernels** where MLX has nothing or loses.
3. **`MPSGraphExecutable`** as a foreign node where both lose, never encoded
   into our command buffer because MPS may `commitAndContinue` it.

### Measured and open

1. **MTL4 ICB stall: solved 2026-09-25.** Every pipeline an ICB names must be
   in a residency set in use at commit (`MTL4ComputeCommandEncoder.h`). The
   comment in `metal/device.jl` calling it a driver defect is wrong.
2. **Render ICBs on MTL4: validated 2026-09-25**: 120 frames, three in flight,
   image exact to the byte.
3. **Placement-sparse buffers: validated 2026-09-25**: pages mapped while
   replays are in flight, address unchanged, data intact; a mapping call costs
   about 0.26 ms. Support is read from `supportsPlacementSparse`.
4. **Same-queue ordering:** `barrierAfterQueueStages:beforeStages:` orders
   against earlier work on the same queue; across queues, event waits.
5. **fp16 GEMM on the M5: measured 2026-09-25** over SAM 2.1's six encoder
   shapes: MPSGraph 4.00 ms (13.6 TFLOP/s), MLX 0.32.2 4.81 ms, ours 6.71 ms.
   The measurement scripts (`render_icb.jl`, `sparse_arena.jl`,
   `gemm_select.jl`) are on the MacBook worker, not in a repository here.
6. **Open:** the cost of one event-chained `MPSGraphExecutable` call.
7. **Open:** which per-draw render state an MTL4 indirect render command holds.

## How we build it

The previous attempts had checks too: `test_backend_vocabulary.jl` allowed a
backend to extend only names in `BACKEND_VOCABULARY`, and whenever a backend
needed to decide something, the name was added (the vocabulary grew from 179
names at 7e64cd5 to 189 at 90e582f). So this refactor is built so that
skipping one of its rules fails a test.

### Order of work

1. **Docs:** this plan, `internals.jl` and `examples.jl` (pseudocode and
   examples consistent with it), `api.md` (every type and verb with its
   contract, the ownership table, the invariants, the corner cases),
   `refactor-ledger.md`.
2. **Comments:** every comment and docstring in Mantle checked against the code
   and this plan before implementation starts.
3. **Tests**, red where the code does not satisfy them yet.
4. **Bugs to fix first** and the performance baseline (the whole-plan benchmark
   and each suite's model calls).
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

Each phase adds its mechanism, migrates every user on the device types that
have joined, deletes what it replaced there, and ends green on the tests it
names.

### 0. Docs, comments, tests, bugs

In the order of "How we build it". The tests, red:

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
  compiles nothing, records nothing except the `AccelCommand`s a change marked
  and the segments of a new swapchain generation, and submits each segment
  once.
- `test_runs_never_overlap.jl` (rule 9): one graph whose kernels read, modify
  and write a persistent array, run from two threads and with segments on two
  queues, 10 000 times with no host wait between runs; the result counts every
  run exactly once. Negative control: with the wait on the previous run's
  tokens removed it fails.
- `test_resize_is_queued.jl`: `resize!` of an array used by two graphs, one of
  them in flight, from another thread, directly and through a view: every run
  sees either the old or the new size and address, never a mix; the old region
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
  `copyto!`, `render!(dev, …)`, `trace!(dev, …)`, `fft`, `mul!`) reaches
  `compilekernel`, with one submission per kernel launch, on every device type.
- `test_hostwritten.jl`: `x[:] = data` before the first run reaches the first
  run.
- `test_run_host_time.jl`: the real `run!`'s host time per run over plans
  that name 100, 2 500, 25 000 and 50 000 sync states, measured against
  the phase 3 target (within 3x of the model in Measurements; open decision
  19); from phase 6 on, also with the records named through array `GPURef`s.
- `test_eviction.jl` (open decision 16, ruled B3.9/B3.10): two graphs that each need
  90% of the device run alternately, back to back with no host wait; weights
  evicted and restored before the run; results exact; zero recompiles;
  composite members and resources a plan recorded by handle never evicted.
- `test_composite_fuzz.jl`: `push!`, `delete!`, `resize!` (phase 5), array
  `GPURef` rebinds (phase 6) and plot updates (phase 11) interleaved with runs
  on several threads; validation clean; reserved memory back at its baseline
  after `free!` (Memory).
- RayMakie's compile-counter session: the per-device compile counter over a
  scripted session (attribute changes, value ↔ per-element, device-array
  rebinds, visibility, camera, layout, resizes within capacity, TLAS moves and
  count changes); zero compiles.

### 1. One compile path

Adds: `Kernel`/`KernelInfo`, `compilekernel`/`kernelinfo`, the per-device
compile counter, node types with their origin, `compile!(s, node)`, the
dependency graph kept and reduced with one hazard function shared with
barriers, barrier merging for conditional kernels, conditional blocks
(`when!(g, flag) do … end` on a device-side flag), `repeat!` bodies limited to kernels with hazards derived
between iterations (instead of `emitloophead!`'s full barrier), the estimated
submission split, placement stored in the plan, `sharing!`, the per-segment
access summary.

Renames: the compile phases become the step functions of "Compile", applied to
a `CompileState`; `run!(phase, ctx)`, the phase marker types and `PHASES` go,
so `run!` only runs a graph.

Deletes: the three `compile_dispatch` bodies, `buildskernel`/`kisupported`,
`Pass.kind::Symbol`, `get_or_build_iter_plan` at emit, 692a4e1's measured split
(the list under "Submission split: estimated"), the host-read `when!` with
`Pass.hostcond` and `RecordingParts.conds`/`batchconds`. Rewrites
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
wait for retiring storage; eviction, a proposal, is phase 4), one meaning of capacity,
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
host time does not grow with the number of resources it names (open
decision 19, ruled).

### 4. Arguments and sizes

Adds: Lava's entry wrapper with fields in the argument slots and the block
marked `NonWritable`; the cell arena; each array's cell, shared by its views;
the head node and prepare nodes with the plan's slots as a resource (entry
application, cell copy table, launch commands); the one entry per plan with host
writes waiting for the previous run; the value `GPURef` with its lock, a
store marking the plans whose entry holds it; hostwritten data
through `x[:] = data`, kept on the host while no current plan exists;
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
resources under pressure with `Restore` on their next use (proposal, open
decision 16; it needs the dependents and pending changes of this phase).

Deletes: the pointer-to-inline-copy argument layout in Lava's entry wrapper and
Mantle's packer, the pushed patch table (`emitpatches!`, `Plan.patchtab`,
`Plan.pending_patches`, Vulkan's `Recording.patches`, `closerecording!` filling
it, `graph/packing.jl`'s patch half), `movelisteners`/`notify_move!`/
`listen_moves!`/`unlisten_moves!` as a patch mechanism, `DeviceRange(count; max)`.
Rewrites `test/vulkan/test_recorded_move_patch.jl`
(`test/metal/test_recorded_move_metal.jl` in phase 9).

Green: `test_resize_is_queued.jl`, `test_host_node.jl` in full,
`test_hostwritten.jl`, `test_eviction.jl` (once open decision 16 rules
eviction in), the whole-plan benchmark within its noise of `plan_*.txt`.

### 5. Acceleration structures and textures

Adds: builds and refits as operations, `TraceRead`/`TraceBuild` declarations,
the core `TLAS` (hardware and software) with instance ranges, a device-side
range table and a doubling capacity, builds with all their uses, the texture
table; `AccelCommand`s marked by the changes that alter them, re-recorded
only when marked; pending builds coalesced per structure, Updates with
`builtcount`; BLAS storage with a doubling capacity, refits and builds into
it under the same count rules (`AccelCommand`), `MoveAccel` beyond it;
kernel-written counts built over the capacity (to be confirmed);
a change that would make a running plan stale waiting for the run's end
(throwing from the plan's own host function); composites that hold
and never allocate, with finalizers on every holder type; readers (proposal,
open decision 16): a structure's update pended by host changes and by
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
changes and restores applied first); the array `GPURef` (device-only, a root
target); `attribute(a, i)`; graphics and ray-tracing pipelines from the device type's
cache under one key on every device type; images as resources with one format
table per device type (`imageformat`), the sampler table (`samplerentry!`);
`render!`/`draw!` as nodes, index arrays through `bindindices!`, `when!`
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

Green: `test_one_path.jl`, GPUArrays' TestSuite on every joined device type.

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
`target_extent`/`target_format`, the walked path in `execute!` (including 8d95632's
walked headless plans for RayMakie's composited frame, which becomes a recorded
graph whose changing counts are resizable arrays), `DrawBinding` (RayMakie uses
it, `src/overlay_rendering.jl`) and 8d95632's `boundbindings`, `boundcount`,
`boundindices`, `boundinstances`, the `recordsplans` hook, `Immediate`,
`recordable`, `rebind!`. Rewrites `test/test_headless_rebindable_draw.jl` and
`test_recordable_plans.jl`.

Green: `test_window_resize.jl`, `test_run_is_submit.jl` for a windowed graph
with overlays.

### 8. Devices are arguments

Every function takes its device or channel (52 lines naming `vk_context()`,
about 31 of them calls, and 54 lines naming `default_bq` in `src/vulkan` at
90e582f). Every hook takes the device and dispatches on its type.
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
  plots as pieces of its pass (B3.17), so adding or removing a plot
  recompiles nothing; attributes as array `GPURef`s
  read with `attribute(a, i)`; `update!` as stores, resizes and rebinds; a
  dirty list instead of scene-tree walks; one scene-rectangle array per scene;
  Hikari's integrator declared into the frame graph with window-sized film and
  work queues, kept across scene changes; shadow and ambient-occlusion passes
  under `when!`; no `waitidle` in a frame; overlay compositing through the one
  path; its own fixes listed there.
- **JuliaVision:** one graph per call ("One call, one graph"). DNNKernels'
  GEMM, attention and convolution as `mul!`, `conv!`, `attention!` on graph
  arrays.
- Comments contradicting Mantle are corrected in the commit that makes them
  wrong.

Green: each consumer suite with two devices open; each runner's per-call
assertion; RayMakie's compile-counter session (phase 0) with zero compiles.

### 12. Full suites, once

Mantle, Hikari, RayMakie, VideoEditor, JuliaVision on Linux (Vulkan, CUDA,
HIP); Mantle, Hikari, RayMakie on the Mac. Then a "State" section is written
from the results.

## Decisions

1. **Mantle owns device memory**; vendor allocators are bypassed.
2. **One array type**, `MantleArray{T,N,D} <: AbstractGPUArray{T,N}` on devices
   and graph devices, with scoped zero-copy vendor views.
3. **Devices, not backends.** `Device()` is a `LavaDevice` on GPU 0 (a
   `MetalDevice` on macOS); `CudaDevice`/`ROCmDevice` after `using CUDA`/
   `using AMDGPU`; lavapipe replaces `HostDevice`; one Julia object per hardware
   device. Loading stays by platform.
4. **A graph is not a device**: graph arrays live on its `GraphDevice`.
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
12. **Runs of one graph never overlap**; one argument block and one entry per
    plan; host writes into the entry wait for the previous run.
13. **Each array owns a cell** (address, size), shared by its views; the head
    node copies cells into flat, read-only argument slots; kernels never read
    cells; arrays are fixed until their first `resize!`, which makes them
    resizable: in place if the new size fits the storage, else a large array
    on a device with sparse residency moves into a reserve that grows by
    mapping pages, and any other into a pool region that doubles, each
    doubling a move; launches are indirect only over resizable arrays.
14. **Control flow is launches only**: `repeat!` bodies and conditions hold
    launches of GPU work (dispatches and traces); a condition is a `when!`
    block (Simon, 2026-10-02).
15. **Sparse arenas that grow and shrink**; where sparse binding is missing, a
    graph compiled later gets a new arena part.
16. **MTL4 is the only Metal target**; MLX's kernels inside the recording,
    `MPSGraphExecutable` only as a foreign node.
17. **Per-image present segments** on Vulkan; a window resize is a mark the
    next run applies.
18. **Kernels come from the device type's compiler cache**
    (`compilekernel(dev, …)`); core keeps no kernel cache.
19. **Eager work is one submission per launch**: no batching outside graphs, no
    cached one-op plans.
20. **Holds and pressure:** one hold for the creator, one per naming graph
    and one per composite holding the resource (a TLAS range, a BLAS, an array
    `GPURef`); pressure is handled only in `acquire!`.
21. **One device registry** per process, the only process-level state
    (decisions.md B15 listed it as open; B2.5's one object per hardware device
    needs it).

## Open decisions

1. Ruled 2026-10-02: the block form, Decision 14. Today's `when!` takes a
   host-read `Ref{Bool}` and goes in phase 1; the name is reused. The
   mechanism, the same on every device type: the launches and draws inside a
   `when!` block are indirect, and whoever writes their indirect commands
   multiplies the count by the flag: the head for a flag the host writes, a
   prepare node right after the kernel that writes a kernel-written flag.
   This is the mechanism `repeat!`'s loop head already uses (it zeroes the
   body's counts). No conditional rendering is used, so the flag is never a
   handle dependence (it may move like any array) and Metal needs nothing
   extra. A flag in `Readback` memory serves both sides: a kernel writes it,
   the prepare node reads it on the GPU, and the host reads it after the run
   without a copy.
2. Kernels behind `mapreducedim!`, `accumulate!`, `sort!`: AcceleratedKernels'
   first (in use today), ported state-of-the-art kernels later as candidates.
3. CUDA/HIP argument layout (phase 10, by measurement).
4. TLAS and textures in shaders. Proposed in `resizing-and-raytracing.jl`: the
   TLAS by device address (`OpConvertUToAccelerationStructureKHR`), textures
   through a device-wide table indexed by a slot value. With descriptors
   instead, traces become handle dependents of the TLAS.
5. `FuseBroadcasts`, and the names of rewriters and attributes.
6. Streamed weights: keyword and upload bandwidth against step time.
7. Cross-device `copyto!`.
8. Reacting to other processes taking VRAM (macOS memory-pressure events,
   DXGI's budget notification).
9. The name of the scoped vendor view (`withview`).
10. When a retired swapchain's images are free: a queue token covers the
    rendering but not the presentation; `VK_EXT_swapchain_maintenance1`'s
    present fences cover both where the driver has them.
11. Whether a `repeat!` body is recorded once per iteration into one segment or
    recorded once and submitted per iteration (today's way). Either way a loop
    whose iterations together exceed the submission budget is cut between
    iterations.
12. Priorities and expected overlap: `Graph(dev; priority)` (name open), from
    which core decides queues and arena parts.
13. The profiling switch and `timings(g)`.
14. Ruled 2026-10-02 (decisions.md B3.3): one submission per stage for runs
    with host nodes, arena parts shared and held by a run until its last
    stage. The graph lock as a `Base.Semaphore(1)` with GC-safe host waits was
    presented the same day as forced by Julia's `ReentrantLock` (it disables
    finalizers, `base/lock.jl:28`) and not objected to.
15. Draw order that changes after compile (2D z order, transparency sorting).
    Options: (a) recompile; (b) a sorted draw list the GPU reads (multi-draw
    indirect); (c) a changed draw order re-records the render segment (no
    recompile), the same mechanism as 17(c). With pieces (B3.17), (c) is a
    rewrite of the pass's list only; still open for per-frame sorting.
16. Ruled 2026-10-02 from the draw benchmarks (decisions.md B3.7): the
    `GPURef` of an array (zero-copy attribute rebinding), `attribute(a, i)` as
    `a[min(i, length(a))]` (1-based; the benchmark shader's 0-based form is
    `a[min(i, n-1)]`), the scene rectangle in the vertex stage with clip
    distances. Ruled 2026-10-05 (decisions.md B3.9, B3.10): eviction is
    automatic (Memory, "An idle resource can leave the device"; phase 4),
    with input modes still to settle (a read-only array from a file or from
    host data needs no copy out); a run that writes arrays pends the updates
    of the structures built from them (Acceleration structures; phase 5),
    implemented as efficiently and carefully as the rest of the core.
17. Index buffers and other handle dependences. An index buffer is bound by
    handle (`vkCmdBindIndexBuffer`; Metal's `indexBuffer` argument), so a move
    of an index array (growth beyond its storage, growth past a reserve, a
    `GPURef` rebind of faces) recompiles the plan today; RayMakie's lines (an
    adjacency index list built from the NaN pattern,
    `RayMakie/src/overlay/lines.jl:515-519`) and meshes change those. Options:
    (a) pull indices by address in the vertex stage (a non-indexed draw of
    indexcount vertices; the shader reads the index, then the vertex): no
    handle at all; it costs post-transform vertex reuse, to be measured with
    `experiments/draw_bench/` on the three devices; (b) one device-wide index
    reserve bound once per render pass, each index array a range in it, its
    first index written by the head into the indirect command (`firstIndex`
    on Vulkan, `indexStart` in Metal's indexed indirect arguments); it needs an
    allocator inside the reserve; (c) a handle change re-records the segments
    that recorded the handle instead of recompiling the plan (compile and
    placement unchanged); this covers every handle dependence (index buffers,
    Vulkan buffer copies, restores of handle dependents) with one mechanism.
    Ruled 2026-10-05 (decisions.md B3.11): (c), after benchmarks of the
    re-record cost (100, 1 000, 5 000 draws) and of pulled indices against
    indexed draws (option a) on the three devices.
18. RayMakie changes that still change a shader. An attribute whose type
    changes (colour as numbers or as colours, `Point2f` or `Point3f`, a
    material of another type) is a `delete!` and `insert!` of the plot in
    Makie already (Simon, 2026-10-05: rendered plots fix their types), so it
    is a plot removed and added, which may recompile (B3.6). A new Hikari
    material type arrives the same way: Hikari keeps materials in a
    `MultiTypeSet`, one array per material type, whose GPU form is a tuple of
    typed arrays the shade kernel dispatches over (Raycore
    `multitypeset.jl:213-225`, Hikari `scene.jl:18-24`), so a new type is a
    new kernel specialization. What is left: a change of a value that picks
    another pipeline without changing a type, such as `transparency`. Options
    for those: (a) RayMakie treats the change as `delete!` + `insert!` of the
    render object (a recompile); (b) both draws are recorded, each inside
    `when!(p, flag)`, and the change is a store into the two flags (one more
    draw per plot with count 0: +14-34 ns per tiny draw on Vulkan, nothing on
    the M5, measured). Ruled 2026-10-05 (decisions.md B3.12): never a
    recompile; a GPU flag (`when!`, the count times the flag in the indirect
    command). In today's RayMakie raster path a plot's pipeline is chosen by
    its plot type, its argument types, whether it is indexed and whether it
    samples textures (`overlay_rendering.jl:234-256`); blending is fixed per
    plot type (premultiplied, `overlay/scatter.jl:49`), so the switches by
    value are few; each gets one recorded draw per variant.
19. Ruled 2026-10-05 (decisions.md B3.14, B3.15): no per-resource walk at
    run. Between graphs, through the resource when it is written: a stage
    writing a resource another plan names waits for that plan's complete
    last run and puts it into the readers' inboxes; a reader's stage waits
    for the writers' complete last runs (Simon: waiting for the whole run is
    fine). No neighbor lists; nothing recompiles when plans register or
    composites change members. Host side: an inbox per plan, filled by
    whoever pends a change or records an access that is not a run. The
    per-resource structure is named `SyncState` (was `AccessRecord`).
    Ordering.
20. Ruled 2026-10-05 (decisions.md B3.8): one graph presents to a window; a
    second graph declaring into it throws, other graphs render into `Image`s
    the presenting graph reads. Removes the coordination of acquires between
    graphs (an acquire counter, waits for acquires in progress, images kept
    by other graphs, an acquire timeout, a `lockgraph` check for held images)
    and the deadlocks and use-after-free the reviews found there (Windows).

## Out of scope

- Video decode itself (formats, conformance); only its queue, memory and
  ordering are in scope.
