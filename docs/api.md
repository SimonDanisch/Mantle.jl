# Mantle API: types, verbs and their contracts

This document is Mantle's API after the refactor: every type and verb with its
contract. This document owns public entry signatures and indexes the native
boundary; `recording-plan.md` owns the architecture and execution-contracts.md
supplies the detailed execution/ownership protocols. Pseudocode and examples
must agree with those contracts. Section 10 lists remaining protocol obligations,
implementation and qualification gates. Listed entries are target contracts;
additional operation families and extension details recorded as open in the ledger
are not established merely by naming them here.
Section 2 defines the consumer-facing API. Core helpers/types and native provider
verbs in later sections are internal contracts, not additional user entry points.
Mentioning a helper in a sketch does not make it public.
The refactor must reflect this boundary in code: supported consumer bindings use
`export` when intended for unqualified use, or Julia's `public` when intended for
qualified use. Core implementation helpers use neither. An unexported name alone
does not establish whether it is supported API. Julia permits qualified access
to internals; that access carries no compatibility guarantee. Native provider and
library extension contracts are documented separately from consumer entry points;
their implementation names are not automatically consumer exports.
The consolidated usage, implementation and tradeoff guide is [api-guide.md](api-guide.md).
Existing baseline evidence is indexed by the plan.

## 1. Rules every entry obeys

1. Core decides; a device type implements verbs. A verb never walks a plan,
   reads the dependency graph, or decides an order, a barrier, a split, a
   queue, a lifetime, a residency set or a fallback.
2. Dispatch, not probing: one function name, methods per type. No `isa`
   chains on device or node types, no `hasmethod`/`hasfield`/`applicable`, no
   `try`/`catch` probes, no Symbol kinds.
3. Core execution/native entries take an explicit target/channel; ordinary eager
   Julia methods obtain device ownership from their operands. No task-local,
   thread-id or ambient current graph/group/device selects execution. The device
   registry is the one process-level state.
4. One shared operation definition serves eager methods and recorded call! entries.
   The frontend selects execution; storage lifetime and array identity do not.
5. A required verb has a method for every joined device type. A fallback
   that returns `nothing` or a default for a required verb is a forbidden move.
   Required means required by an enabled capability path: graphics/window
   verbs are required where graphics/display is supported, and hardware
   acceleration verbs where that hardware path is supported. Core selects a
   documented portable fallback or reports the missing feature before calling
   a device verb. Unsupported families do not acquire dummy methods.

6. Arrays belong to one Device/domain; node uses in multiple graphs/groups retain
   that same resource. Recording ordinary operations uses call!(Graph/Group,f,args...), never
   adaptation, a backend wrapper, array membership or ambient current state.
   Standard eager methods keep Julia signatures. Return contracts belong to the
   shared operation definition; a ! suffix implies no particular return value.
7. Keep stable concrete device/element/argument representations without encoding
   graph topology, resource identity or dynamic extents in public type parameters.
   The architecture's compile/runtime rule and execution-contracts.md §16.11 govern
   bounded specialization; runtime wins a measured unavoidable tradeoff.

## 2. User API

### 2.1 Devices and declaration targets

| Call | Contract |
|---|---|
| `Device()`, `devices()` | Stable registered device objects. Platform hardware default is Vulkan on Linux/Windows, Metal on macOS; no silent software default. CUDA/HIP extensions add devices. A physical GPU through different drivers is different device domains. |
| `adapt(dev::Device, x)` | Device/data adaptation, including callable fields. Same-domain Mantle arrays preserve their device/root without copying. It does not select a graph/group. Passing a Graph/Group as the adaptation target rejects. Foreign allocation import/cross-domain transfers use their explicit contracts. |
| `KA.get_backend(a)` | The owning Mantle Device, independent of all nodes/graphs using a. Ordinary KA launches are eager; graph recording uses dispatch!(target,...), with one shared compiler/recipe path. |

Operands validate against the explicit target's owning Device/domain, including
callable captures. The same array can appear in sibling Groups or unrelated graphs
without adaptation. Each declaration retains holds/uses for its target. Only
explicit transport permits different device domains; scratch still cannot escape
its owning run. NoDefaultGPU lists available explicit software devices.

### 2.2 Arrays and storage

`MantleArray{T,N,D} <: AbstractGPUArray{T,N}` is the one family for persistent,
scratch and imported arrays. Those are resource properties, never role parameters.
Ordinary views share their logical parent's state and carry their own offsets,
dims and strides; nested views compose. Canonical allocation identity is distinct
from a logical array header: independently shaped/typed imported headers share
payload hazards/ownership, not dimensions merely because their bytes alias.
Adjoints retain conjugation, plain transposes do not. Native overlap uses checked
byte spans. View offset/stride units are explicit at the ABI boundary; reinterpret
aliases must not reuse a differently typed header's element count as their own.
Static views cannot resize; their capacity is their fixed view extent, subject to
prepared backing validation, not spare storage outside their view.

| Call | Contract |
|---|---|
| `MantleArray(target, T, dims...; scratch=false, capacity=nothing, unified=false)` | Persistent by default on a Device, Graph or Group; the returned array always belongs to the physical Mantle Device. The construction target never becomes a default destination for later operations. A graph may defer placement; initialized contents survive replay/recompile and graph teardown while held. Omitting capacity leaves fixed geometry until resize; the usable bound is still initialized to at least the initial length. Explicit capacity requests that storage bound and opts into dynamic geometry; placement/restore must back it before GPU use. readonly/streamed retention and copy=false import require persistent ownership and reject scratch=true. |
| `MantleArray(target, data; scratch=false, ...)` | Persistent data initializes once, preserving contents across replay/recompile. With explicit scratch=true, each execution of its owning region initializes the scratch instance by an ordinary recorded discard-resize/copy before use (each iteration when repeated). Host data is snapshotted at declaration; a GPU source is an explicit dependency of that initialization copy. Eager initialization happens once for that eager lifetime. |
| `MantleArray(target, T, dims...; scratch=true)` | Explicit temporary: one graph run's arena slice or eager last-token retirement. Undefined entry contents; requires a dominating initialization before read. No immediate host data/arena-metadata read, cross-run use or persistent composite membership. A helper may return its handle/view for further declarations in the same owning graph/run; this does not export live data. Zero-sized/unused scratch is legal and needs no native payload allocation; metadata/count publishers still execute when required. Use an owner-local recorded Array conversion or an explicit persistent copy when data must survive. |
| `MantleArray(target, T, dims...) do tmp ... end` | Explicit scoped scratch form, equivalent to scratch=true. finally drops the creator hold; it neither runs the graph nor frees storage held by declared nodes. Arena/retirement lifetime still follows owning runs and actual tokens, not the host callback's return. |
| `KA.allocate(dev,T,dims; scratch=false, unified=false)`, `similar(a,...)` | Return arrays on the owning device; similar(a,...) and ordinary KA.allocate are eager allocations with persistent default. A graph helper explicitly constructs workspace with MantleArray(target,...;scratch=true). A generic allocation is never guessed to be temporary or grouped. |
| `MantleArray(dev, host; readonly=true)` | Retains immutable host backing for direct restores, with no extra retained copy. Writes/resize reject at declaration or eager call from inferred effects. Caller must not mutate retained backing, including after creator free while graph holds remain. |
| `MantleArray(dev, host; streamed=true)` | Read-only weights with retained immutable source plus core-planned resident intervals/uploads (§12). Writes/resize reject through the same effect validation as readonly=true; mutable state requires an explicit persistent copy. It is not persistent resident VRAM or an untracked backend loader. Costs include paging/transfers. |
| `MantleArray(target,T,win; scratch=false)` | Tracks window dimensions/capacity. Native resize invalidates contents unless an explicit preserving operation is specified. Lifetime still comes only from scratch. |
| `MantleArray(dev, foreign; copy=false, stream)` | Import an existing compatible same-driver allocation with no copy, retained allocation owner and stream readiness. Requires exact device/context/extent/stride facts; aliases join one root. See §2.9. |
| `free!(a)` | Idempotently drops creator hold; that handle and its aliases reject new operations. Existing graph/composite holds remain usable. Views cannot be freed separately. Last hold retires data, Cell and native objects against all uses. |

Graph/Group nodes retain resource holds; they do not alter the array's device or
identity. A persistent root survives its creating graph while held and can be
passed directly to another graph. Same-domain adapt(dev,a) never adds/removes graph
membership. Adapting host data uploads it; foreign import is explicit (§2.9).
Persistent construction from ordinary data captures one-time initialization at its
ordered construction point. Host data is snapshotted unless an explicit immutable
readonly/streamed source contract applies. A GPU source uses the shared ordered
copy/snapshot protocol: later source writes cannot change that captured result
merely because destination placement was deferred. Such deferral requires source
version/ordering holds, not a first-use read of whatever bytes then exist. In
contrast, explicit scratch-with-GPU-data initialization reads its declared source
at every owner-region invocation. copy=false is an explicit alias/import contract,
not the data-copy constructor. These rules do not select a recording target from
the resulting array.

No pointer/Cell/slot may accidentally point into expired scratch.

### 2.3 Size, changes and snapshots

| Call | Contract |
|---|---|
| `length(a)`, `size(a)` | The only logical-size queries. Kernels read prepared descriptors. Persistent-array host queries of GPU-written size synchronize/read metadata; host-known shapes are ordinary integers. GPU-written scratch metadata cannot be read on the host outside its owner run; inspect an owner-local recorded snapshot instead. A returned host Int is an explicit snapshot, not deferred graph geometry. Core graph recipes use prepared size/shape dependencies and launchrange without hidden scalar readback. |
| `capacity(a)` | Usable allocation element bound, independent of live size, virtual address reservation and device budget. For deferred/evicted storage it is a retained backing requirement, not resident-byte accounting; core must back it before publishing a device descriptor. Querying it neither allocates nor runs work. A GPU-visible capacity never exceeds the backing held for that use. Sparse trim reduces the bound only when unmapping commits. Used for maximum scratch/eligibility and initialized even without an explicit reserve keyword. |
| `resize!(a,n)`, `resize!(a,dims)` on host | Schedule size/growth; requested host size updates immediately, storage changes at the next ordered use. Preserve linear contents using the ordered old logical extent (prepared on GPU when necessary), change the last dimension for the n form, never resize a view. A specialization/placement dependence may recompile once; compatible handle moves only rerecord their dependents. |
| `resize!(a,n)`, `resize!(a,dims)` in a kernel | Publish logical dimensions within allocated capacity, with no allocation/growth. Compiler infers a metadata-write effect; core prepares subsequent descriptors/geometry. One invocation publishes each update unless a specified collective elects it. |
| `launchrange(a)` | Deferred geometry for dispatch/draw; direct when fixed, dynamic after inferred device resize or host resizing. It is not a length-query alternative. |
| `a[r]=hostdata`, `copyto!(a,hostdata)` | Snapshot source once into upload storage at the call; schedule actual store. Respect view/index mapping and sequence-relative storage. Inline/copy only when logical bounds are known valid in the ordered state. GPU-written logical bounds use the shared checked-store kernel/prepare path; invalid logical ranges report a deferred bounds error and perform no write. Physical-capacity violations reject at the call. |
| `copy!(a,hostdata)` | Atomically enqueue discard-resize and store; on a view only equal-size copy. No old-content copy when discarded. |
| `Array(a)`, `copyto!(host,a)` | Immediate coherent persistent-resource snapshot, including size/payload and pending ordered host changes. Array(a) is the blocking convenience over the same readback protocol. These calls never run a graph or infer a graph position. Wait outside locks; unified CPU copies exclude concurrent submissions. Run-local scratch requires an owner-local recorded Array conversion. |
| `call!(target,Array,a)`, `Readback(previous::ReadbackFuture)` | The recorded Array conversion declares a new snapshot position and requests its next execution. Readback(previous) requests another execution of that exact existing output without adding a node. There is no public eager Readback(a) or separate Readback(target,a) operation; ordinary eager Array(a) uses the shared copy protocol. Payload/size stay coherent; scratch requires its owning run. |
| `fetch(future)`, `wait(future)`, `isready(future)` | Submit no GPU work. Submitted futures retain token/storage after Graph teardown. Before staging reuse, preserve any outstanding preceding result. |
| `show(a)` | Metadata only; no GPU read/submission. GPU-written extents may be shown as device-resident rather than downloaded. |

Kernel resize updates the publishing invocation's local descriptor; other invocations
retain their entry snapshots. It is not a grid synchronization operation. A later
dependent dispatch sees the prepared new dimensions. Device views cannot resize.
Checked operations reject invalid dimensions/capacity before authorizing memory
access; no clamping, implicit GPU allocation or overflow-retry mechanism is added.
Raw @inbounds code retains its ordinary explicit unsafe semantics.

An application builder such as `app_compact!(target,active,particles,predicate)`
plans capacity from its input bound,
uses checked internal capacity descriptors for scatter, and publishes the final
size with resize!. Empty inputs publish zero unconditionally. Internal helpers
must not expose another public metadata handle or thread a second independent
counter through ordinary array operations. This is an example composition, not a
Mantle `compact!` overload or a function that requires `call!` registration.
Detailed protocol: execution-contracts §15. Array shrinking retains capacity;
there is no public `trim!(a)` in the initial API. Explicit array-backing trimming
is deferred until a concrete consumer justifies its portable contract.

### 2.4 Graphs and regions

| Call | Contract |
|---|---|
| `Graph(dev; rewrites=(), profile=false, priority=0, share_scratch=true)` | Explicit device owner and declaration tree. Rewriters/profile opt in. Priority is urgency; scratch sharing is a separate memory choice. Shared tenants serialize, independent placement can overlap equal-priority graphs subject to budget/hazards. |
| `Group(parent,name)` | Structured operation-node container at a stable parent position, supporting nested Groups. Its explicit recorded-call/node arguments determine membership; arrays are never adapted/bound to it. |
| `repeat!(group,maxiters; while_nonzero=count)` | Bounded, nestable region. Reset each enclosing invocation; latch GPU stop until reset. Internal prepares, eligible libraries/copies/renders/derived builds are part of the body. Reject negative bounds before mutation. |
| `when!(group,flag)` | Evaluate the condition once at region entry and latch it for the complete body, including reset/clear/render work. A false branch preserves outputs and does not advance RNG; changing the source flag inside the body affects later region entries only. |
| `iteration(group)` | Typed placeholder for that enclosing loop's one-based iteration, available only within its lexical descendant regions. |
| `delete!(g,group)`, `sort!(group;by,lt,rev)` | Delete/reorder region Groups. Compatible draw pieces update locally; other structural changes rebuild plan. Retire removed holds/recordings after actual uses. |
| `run!(g)` | Compile/record if needed, then accept work. Return Submitted/Skipped. Every root follows the graph's previous-run tokens. The same stateful graph's runs never overlap. |
| `free!(g)` | Idempotently retire plan and drop graph holds; persistent arrays with other holds survive. Re-entry from its hostcall! throws before marking freed. |
| `submissions(g)`, `timings(g)` | Stages/segments including queue lane, predecessors, estimates/host cuts; bounded nonblocking sampled timings under §10. |
| `naivebytes(g)`, `peakbytes(g)` | Explicit scratch placement bytes without/with aliasing; persistent/imported accounting is separate. |

Declarations validate the complete expansion, target/capture/lifetime/access
requirements and controls BEFORE mutating nodes/holds. A failure leaves the graph
unchanged. Compile preserves Graph identity and output roots across replacements.
Plan, compile/record methods and native executables are internal.

### 2.5 Declaring and composing work

| Call | Contract |
|---|---|
| `dispatch!(target,kernel,args::Tuple,ndrange;group=nothing,origin=nothing,split=false)` | One raw launch; callable structs/closures are supported. origin is optional diagnostic provenance, derived from the call/code when available; callers need not construct Origin. Preserve explicit workgroup dimensions. Complete resource/metadata effects include callable fields. |
| `hostcall!(target,f,args::Tuple;writes::Tuple)` | Host node on staged host-visible arguments, outside record locks. Legal in an ordinary Group, rejected in GPU-controlled regions. Re-entry/arena rules follow §5; failure retains accepted prefixes. |
| `when!(target,flag,kernel,args,ndrange)` | Convenience single conditional launch; flag is GPURef{Bool} or a validated singleton Bool array. |
| `repeat!(g,kernel,args,ndrange;max,while_nonzero)` | Convenience one-launch region with the same nested-control semantics. |
| `fill!`, `copyto!`, `map!`, reductions/scans/sort, `mul!`, FFT, convolution, attention, broadcast | Ordinary eager array/library interfaces retain standard Julia signatures. Supported recorded forms use call!(Graph/Group,f,args...); they share one Mantle definition/recipe and capability selector. No extra public device argument or graph/group overload is added to numerical functions. |
| `tune!(dev,f,args...;check=referencecheck(f))` | Private alias-preserving data/workspace/RNG state, independent correctness check and completed timing; publish only matching code/config versions. No tuning inside declaration/run!. |
| `call!(target,f,args...;kwargs...)` | Record a supported ordinary operation in an explicit Graph/Group. Known definitions lower to retained GPU/copy/library recipes. Returns follow each definition: mul! returns C itself; Array returns a typed future; ! alone implies nothing. Unknown calls reject before node/hold publication, never eager-execute or fall back to hostcall!. No arbitrary Julia tracing/capture is promised. |
| `GPURef(dev,value)`, `ref[]=value` | One stable pool-owned isbits value slot; arrays are passed as arrays, not hidden inside the value. Host assignment schedules a source snapshot/store without patching recordings. Kernel read/write effects participate in ordinary hazards. |
| `RNG(dev,seed)`, `Random.seed!(rng,seed)`, `rand!(rng,a)`, `call!(target,rand!,rng,a)` | Explicit core counter resource. Ordinary random fill is eager; the recorded call explicitly names Graph/Group. Replay/iterations advance declared effects, skipped work does not; recompile preserves state. Reset is explicit and qualified per algorithm/version/type. |

Core selects operation candidates at materialization using immutable requirements,
shape/capacity/layout/alias classes and compiler/ABI keys. Growth can change the
complete command sequence and workspace while keeping public output identity.
Kernel bodies, native library subgraphs and foreign candidates share this contract.
Native-library capture is restricted to a closed eligible recipe with caller-owned
workspace and explicit resources/control constraints. It is not consumer capture.

A target-aware graph builder, an ordinary array operation and a host controller
have different contracts; see recording-plan's Composition boundaries. A host
controller that updates inputs/runs a plan is not a recorded operation merely
because its name is step!.

Target-aware graph-building helpers are called directly during construction:
PrintSim.step!(group,world,state,dt) emits its timestep into the supplied Group,
using call! for supported array operations and dispatch! for kernels inside it.
Such a helper needs no call! wrapper or operation registration; its Julia body
builds nodes once, and run! replays their GPU work. It is not a CPU callback during
replay. A recorded ordinary operation and a graph-building helper are distinct
interfaces; no target is inferred from arrays in either case.

### Minimum recorded array-operation contracts

The initial required catalog is intentionally concrete. Each uses the same shared
Mantle operation body/candidate policy as its ordinary eager form. A composed
application graph builder calls these entries; it is not itself wrapped in call!.

| Recorded call | Declaration-time return | Device effect |
|---|---|---|
| call!(g,mul!,C,A,B) | The identical C | Read A/B and write C; validate precision/layout/shape and alias restrictions. |
| call!(g,broadcast!,f,dst,args...) | The identical dst | Inferred input/capture reads and destination writes; ordinary broadcast shape/alias semantics. |
| call!(g,copyto!,dst,src) | The identical dst | Ordered source read/destination write; explicit transport is the qualified cross-device case. |
| call!(g,fill!,dst,value) | The identical dst | Write destination, with the declared conversion semantics. |
| call!(g,sum!,f,r,A;init=true) | The identical r | Read A and mapped captures; write the reduced-shape device array r. init=false also reads its prior contents. |
| call!(g,rand!,rng,a) | The identical a | Write a and advance the explicit RNG resource according to its qualified stream contract. |
| call!(g,Array,a) | A one-shot ReadbackFuture | Snapshot read plus independent staging/host visibility; forbidden inside GPU control. Windows use their presenting-graph/frame contract. |

For a recorded copyto! from host data, capture an owned source snapshot when the
operation is declared; replay copies those captured bytes. Mutating the original
host array later does not patch baked work. Changing per-run inputs uses explicit
ordered stores to the existing device input before run!, or replaces that declared
copy; no implicit host-array watcher or per-replay host callback is introduced.
A device source is instead read at the copy node's declared execution position,
with ordinary resource dependencies. This source distinction is in the shared
copy definition, not a backend-specific policy.

These are minimum supported signatures, not arbitrary-call capture. Other variants
and operations require their own qualified definitions. In particular, naming
reductions in a family list does not automatically define call!(g,sum,A)'s scalar
return or any other new recorded allocating/CPU-result contract.

GPU-resident scalar reductions need no new result type. Inside a solver builder,
a one-element reduction destination is an ordinary MantleArray:

```julia
# residual was written by earlier nodes in this Krylov iteration.
r2 = MantleArray(krylov, Float32, 1; scratch=true)
call!(krylov, sum!, abs2, r2, residual)
dispatch!(krylov, app_update_convergence!, (active, r2, tolerance), 1)
# The application kernel reads r2[1] and updates the device active flag.
```

The reduction overwrites r2 by default, including writing zero for an empty sum;
init=false folds into initialized prior contents instead. Preserve Julia's supported
mapping/type/promotion/reduced-shape semantics. Consume GPU-written input lengths
from prepared descriptors; do not run a host isempty/length check while recording.
Plan scratch from capacity and publish the result even when live input is empty.
Reuse GPUArrays/AcceleratedKernels algorithms where their interfaces permit explicit
emission; their current eager host branching is not a recording implementation.
PrintSim's existing target-aware world_dot! device-output kernel chain remains a
valid domain-specific path; this catalog does not require replacing it with extra
passes or inventing a public dot! operation. The Library builders and custom operators section defines the extension boundary;
F115/F118 implementation/native correctness/performance gates remain open.

### Library builders and custom operators

A library that already accepts a Graph/Group composes directly into that target;
PrintSim.step! and DNN emitgraph! need Group-aware emission, not whole-library
call! adapters. The same rule applies to an LLM decoder or renderer's builder;
host controllers that run/cycle cached plans remain host code.

A library that supplies a supported ordinary operation extends call! by Julia
dispatch on that function and its argument types. Its eager and recording
frontends delegate to one private shared definition, as _mul! does in internals.jl.
The definition validates domains/shapes/effects, explicitly states returns and
retains a typed recipe with stable outputs. Recipe/candidate lookup may dispatch
on operand types, including custom operator/preconditioner types, not only the
function name. There is no mutable registration service, backend overload matrix
or generic invocation/tracing of the eager function. Candidate policy remains in
the shared Mantle layer; definitions supply eligible emission bodies/requirements.
Changing shape may rematerialize that recipe; replay invokes neither frontend.

The ordinary operator boundaries can therefore be:

```julia
call!(physics, mul!, q, operator, p)
call!(physics, ldiv!, z, preconditioner, r)
```

These forms require the operator library's qualified definition; arbitrary existing
mul!/ldiv! methods are not automatically recordable. The returned destination,
alias/shape rules and all captured array/derived-workspace effects must be explicit.
A target-aware matrix-free or preconditioner builder emitting kernels directly is
also valid. Mantle does not require a new Solver, sparse-array family or materialized
matrix for these algorithms. A library owns its domain representation/strategy;
the shared operation layer owns portable execution and capability selection.

The minimum catalog uses caller-supplied device destinations, plus the explicit
Array snapshot future. No generic recorded allocating/view/scalar-return family is
implied. An additional allocating definition must specify element type/rank,
initial logical dimensions, bounded output capacity, metadata-readiness/lifetime
and stable output identity before admission; device code cannot allocate beyond
backed capacity. Unknown variants reject. Host-known views are ordinary aliases;
GPU-produced offsets/lengths form views inside device code from explicit bound
resources, as in examples.jl. Neither a host Int nor a static view secretly becomes
a deferred GPU-bound expression. DNN exports with known view geometry need only
ordinary views of their classified root.

Atomic declaration applies to each primitive/supported-operation expansion.
An ordinary Julia helper composing several declarations is not implicitly one
transaction. Builder authors validate configuration before emission and clean up
an incomplete owned region on failure; callers must not run a partial solver.
This is distinct from promising rollback of arbitrary Julia effects.

### 2.6 Acceleration structures and shaders

| Call | Contract |
|---|---|
| `BLAS(target,vertices,faces;scratch=false)`, `BLAS(target,bounds;scratch=false)` | One core structure over triangle or procedural geometry. Member holds, capacities, topology/position dirtiness and retirement are shared policy. |
| `TLAS(target;hardware=nothing,scratch=false)` | The default selects hardware by enabled capabilities/eligibility, otherwise shared software traversal with equivalent logical conventions. Explicit hardware=true requires the hardware path; false selects the software reference path. Neither requires backend-specific consumer code. Host totals build exactly; GPU totals use capacity with valid mask-zero records. |
| `push!(tlas,blas,transform_or_array;hitgroup=1,...)` | Add retained range, return stable range handle. Exact native representational/count limits are checked; no workstation constants. |
| `delete!(tlas,range)`, `tlas[range]=transform`, `visible!(tlas,range,bool)`, `hitgroup!(tlas,range,index)` | Scheduled core range/member changes; delete changes activity (Build), hide changes mask (eligible Update), transforms/refits obey native legality. |
| `build!(tlas)` | Schedule explicit full build. |
| `compact!(as)` | Explicit scheduled AS storage compaction/rebuild maintenance; preserve logical scene/handles, native readiness and old in-flight generations. Charge peak memory and any size checkpoint; no automatic compaction under pressure. Does not promise cheap refit afterward. |
| `free!(as)` | Drop creator hold; existing graph/member/native-generation holds remain valid. Last-hold retirement includes child objects, descriptors and residency after all uses. |
| `build!(target,tlas)`, `refit!(target,tlas)` | Declare derived work in a Graph/Group. Region eligibility selects legal native or software construction; no backend-specific consumer branch. Automatic geometry-write -> AS-read updates require no manual refit. |
| `RayTracingPipeline(;raygen,closest_hit,miss,any_hit=nothing,payload=Nothing,maxdepth=1)` | Immutable typed callable program; one-ray-type convenience form; it normalizes to the explicit hit-function table and callback ABI in §7. |
| `trace!(target,pipeline,accel,args,ndrange)` | Recorded ray launch, including eligible conditional/repeated regions. |
| `rayquery(tlas,origin,direction;tmin,tmax,mask,accept,intersect)`, `traceray(origin,direction,payload;...)` | KI/compiler device operations with the typed hit/filter/payload contracts in §7. |

The logical table/query/reporter contracts below resolve F124–F126 at design level;
implementation and independent Metal/Vulkan qualification remain open. The current
Hikari query/wavefront path stays supported; a programmable table is optional.

| Call/value | Contract |
|---|---|
| `TriangleGeometry(vertices,faces)`, `ProceduralGeometry(bounds)`, `BLAS(target,geometries::AbstractVector;scratch=false)` | Logical ordered geometry list; core handles native grouping/splitting and preserves logical IDs. Single-geometry forms normalize to it. |
| `TriangleHitGroup(closest_hit=nothing;any_hit=nothing)` | Typed triangle handlers. Query-only tables may omit shading. |
| `ProceduralHitGroup(intersection,closest_hit=nothing;any_hit=nothing,attributes=Nothing)` | Declared concrete isbits attribute schema and streaming intersection callback. |
| `HitFunctionTable(groups)` | Flat group tuple for one ray type, or rectangular tuple of group rows × ray-type columns. Native layouts/handles stay internal. |
| `RayTracingPipeline(;raygen,hitgroups::HitFunctionTable,miss::Tuple,payload=Nothing,maxdepth=1)` | General logical program; one miss function per ray type. Convenience form normalizes to this and cannot be combined with it. |
| `hitgroup!(tlas,range,selector)` | Integer shared group, per-geometry host IDs, or device integer matrix [geometry,instance]. Selector resources/versions are explicit. No native record offsets. |
| `rayquery(tlas,o,d,args::Tuple=();tmin=0f0,tmax=Inf32,mask=0xff,accept=AcceptAll(),intersect=BoxIntersection(),hitgroups=nothing,raytype=1,anyhit=false,attributes=Nothing,capture=nothing)` | Closest accepted inline hit, or any accepted hit with anyhit=true; table or shorthand intersection/filter. No shading handler invocation. Typed optional capture result; filters precede commits. |
| `RayHit{Q}`, `hitattributes(hit)` | Common hit IDs/distance/barycentrics and concrete captured Q. Attribute access checks that a hit exists. |
| `intersection(report,candidate,args...)`, `report(t,attributes)` | Callback protocol, not a global registration API. Emit zero/many roots; report returns a continue flag (false only after early-accept termination) and cannot escape the invocation. Candidate exposes already-transformed object ray and logical IDs. |

Callback signatures, space conventions, attribute capture, per-geometry native
mapping and failure/resource rules are normative in execution-contracts §7.
TLAS has no public ray-type-count layout option; core owns any native binding
views required by a program/table. Host selectors are snapshotted, device selectors
are read at their execution point. All paths use the same handlers without device
or backend tests in application code.

A geometry write is followed by an internal build/refit before its first AS read,
even in the same segment. Native representation-only builds may execute on a
disabled region if proven observationally equivalent and native-valid; logical
geometry/payload writes remain gated. Otherwise select a qualified kernel/native
region construction or reject that raw combination before mutation.
Accepted geometry versions remain dirty until an accepted matching build consumes
them. Builds elsewhere in a plan and future failed stages cannot clear dirtiness.

### 2.6b Images and graphics

| Call | Contract |
|---|---|
| `Image(target,T,dims...;scratch=false,mips=1,layers=1,samples=1,cube=false)` | Persistent by default, explicit scratch otherwise; native format/extent/usage facts validated before allocation. Cube layers are six-face groups. Attachments require compatible extents/samples. |
| `Image(target,T,win;scratch=false,mips=1,layers=1,samples=1,cube=false)` | Window-size-tracked image with the same explicit lifetime and format rules. Resize refreshes its generation/capacity; persistence does not promise preserved pixels across window resizing. |
| `view(image,ranges...;mip=1,layers=:,aspect=nothing)` | Julia one-based mip/layer/range indices. Infer a single unambiguous format aspect; combined depth/stencil requires an explicit typed aspect when selecting one component. Core produces a typed subresource view with shared root ownership; native index conversion is internal. |
| `sample(image,sampler,coords;lod=0f0,gradients=nothing)` | Typed KI sampling for eligible 1-D/2-D/3-D/array/cube views. Normalized coordinates, no implicit vertical flip; required dimensions/format/filter facts select eligible construction. |
| `KI.imageload`, `KI.imagestore!`, `KI.imageatomic!` | Target KI intrinsic families for typed storage-image views; enabled format/type/order/scope facts required. These additions are implementation gates, not claims of current KI support. |
| `Sampler(;filter=:linear,address=:clamp,anisotropy=1)` | Immutable value interned in the core weak sampler cache. Integer sampling requires nearest; anisotropy needs gradients or qualified fragment derivatives. |
| `GraphicsPipeline(;vertex,fragment,geometry=nothing,blend=Opaque(),cull=CullBack(),topology=TriangleList(),depth=DepthLess(),stencil=StencilOff(),...)`, `MeshPipeline(;mesh,fragment,object=nothing,blend=Opaque(),cull=CullBack(),depth=DepthLess(),stencil=StencilOff(),...)` | Immutable stage/fixed-state values; one core pipeline cache. Preserve blend/depth/stencil/per-attachment state and typed stage IO (§§7–8). |
| `draw!(target,attachments,pipeline,args,count;instances=1,indices=nothing)` | One draw; count/instances may be deferred geometry. Indices use zero-based UInt32 wire indices; vertex/instance builtins are one-based. |
| `clear!(target,attachment,value)` | Clear operation; normally a pass load, or a predicate-controlled clear draw in a conditional region. Disabled regions preserve attachments. |
| `when!(target,attachments,flag,pipeline,args,count;...)` | Convenience one conditional draw. Use when!(group,flag) for a complete clear-and-draw region. |
| `draw!(target,attachments,pipeline,args,commands::MantleArray{DrawCommand};count=launchrange(commands),indices=nothing)` | Bounded indirect command stream. DrawCommand carries vertex/index count, instance count, first index/vertex, signed base vertex and base instance. Core validates layout/count/limits and derives resources. |
| `dispatch!(target,execution_set::ExecutionSet,commands::MantleArray{DispatchCommand};count=launchrange(commands))` | Bounded generated compute command region. Execution set declares all eligible programs/resource layouts; no dynamically hidden resources. Native generated-command and core predication candidates share one semantic representation. |
| `attribute(a,i)` | Shader attribute access with the existing uniform/per-element convention: a[min(i,length(a))]. Empty attributes cannot be read; zero draws invoke no shader. Prepared extents allow one pipeline for uniform and per-element data; retain the existing numerical/performance gate. |
| `copyto!(image_or_view,source)`, `call!(target,copyto!,image_or_view,source)` | Ordinary eager copy or explicitly recorded copy/transfer/supported resolve, sharing one definition and returning the identical destination. No target is inferred from the image. Color MSAA -> matching single sample resolves explicitly; depth/stencil needs an explicitly qualified mode (§8), otherwise reject. |
| `Array(image)`, `call!(target,Array,image)` | Blocking eager conversion or explicitly recorded snapshot/future, using the same copy/hold protocol as arrays. Multisampled storage requires explicit resolve first. |

Shader sampled arguments are typed SampledImage{T,N}; storage views have typed
access/format, not an untyped descriptor index. RGBA normalized formats sample
Float32, sRGB RGB decode to linear and alpha stays linear. Unsupported native
paths select a qualified shared KI/core implementation or reject before record.
Mip generation/feedback/storage atomics require individual native qualification;
no implicit host bridge or fabricated derivative.

DrawCommand is an isbits semantic record of UInt32 vertex/index count,
instance count, first index/vertex and base instance plus Int32 base vertex.
Indexedness comes from the draw's explicit indices binding, not a backend enum.
ExecutionSet(dev,programs;resources) retains the complete allowed program tuple,
compatible argument layouts and reachable resource set. DispatchCommand is an isbits record with a bounded UInt32 program/argument-set
index and three UInt32 workgroup counts; core owns their wire
layout and validation. These planned value types do not expose native handles.

Core forms passes/pieces. Only dead scratch attachments can discard; persistent
images/windows store. An inserted draw's new layout/storage/feedback requirements
invalidate the fast piece path. Scene rectangles are shared data, with native
viewport/scissor candidates where eligible; general GPU state is not erased to
satisfy one consumer's fast path. §8 defines fixed raster state and resolves.

### 2.7 Windows and readbacks

| Call | Contract |
|---|---|
| `Window(dev,w,h;title="Mantle",vsync=true,capacity=(w,h))` | One presenting Graph; surface-specific format/color/usage negotiation in §4 of execution-contracts. Linear shader output, sRGB presentation path, FIFO/vsync default. |
| `resize!(win,w,h)` | Event marks requested extent; next presenting run refreshes generation/recordings. Beyond capacity recompiles; never invalidates other persistent roots. |
| `call!(presenting_graph,Array,win)` | Declare a snapshot of the acquired image after that frame's final writer and before present, returning the common one-shot future. Only the window's presenting Graph or an ordinary Group in it may declare it; GPU-control regions reject. Reuse the output with Readback(previous). Skipped/minimized/closed executions without an accepted copy reject affected requests, never return an earlier frame. A copy already accepted keeps its token/result under the normal partial-run failure rules; capture does not prove presentation completion. |

A Window is a presentation target, not an already-readable persistent array.
Array(win) rejects rather than blocking for unsubmitted future work or running a
graph implicitly. A screenshot uses the declared output, run!(presenting_graph),
then fetch; that fetch never requires the caller to produce another frame. Source
usage/format and any eligible offscreen capture path are checked before mutation;
no consumer backend branch or unconditional full-frame readback is introduced.
This supersedes the old Array(win) next-present proposal in the migration ledger.

Present enqueue is separate from rendering submission. GPU tokens alone do not
retire acquired/presented images. See execution-contracts §§2–4.

### 2.8 Memory and explicit device communication

`reserved(dev)`, `mapped(dev)`, `budget(dev)` and class/heap/opaque/imported
breakdowns distinguish physical, virtual and external bytes (§12).
Empty-cache reclamation is automatic core policy; applications do not call an
allocator-maintenance API. The private cache-trimming helper is listed in §3.

`devicecopy!(dst,src)` remains synchronous explicit snapshot/copy convenience.
`call!(target,copyto!,dst,src)` declares an asynchronous transfer when devices differ;
only this transport operation admits foreign-device roots. Its effects, snapshot
and completion are explicit. Native peer/external candidates or the shared host
bridge preserve them; a host bridge exposes a host cut and costs. Ordinary
mixed-device arithmetic is still an error.

`collective!(target,communicator,op,send,recv)` declares communication with explicit
participants, buffer uses and completion. Collective/P2P library adapters are
capability-qualified closed recordings; participation/failure semantics are §16.
No claim of universal recordability follows from the name.

### 2.9 Vendor views and imports

`withview(f,CuArray | MtlArray | ROCArray,a,stream)` lends a zero-copy native
view through an explicit canonical stream. Core applies pending changes/restores,
orders prior uses, takes an allocation hold and records completion even if f
throws. No Mantle lock spans vendor code. Concurrent Mantle use while lent rejects;
free! only drops the creator hold, preserving lease/graph holds. Untracked view
escape is invalid. If constructing the vendor header needs GPU-written dimensions,
use a coherent metadata checkpoint/lease outside locks before building that header;
zero-copy payload does not imply zero metadata synchronization. The lease preserves
that exact storage/shape version. Do not fabricate host dimensions from capacity or
perform this eager boundary during graph recording/device control. Arbitrary vendor
callbacks conservatively write as well as read; readonly resources need a qualified
read-only library recipe, not an unchecked callback bypass. The borrowed vendor
header cannot resize/rebind ownership outside the declared lease contract.

`MantleArray(dev,foreign;copy=false,stream)` performs the reverse import.
Native allocation identity (including generation/context) joins payload aliases in
the device-owned weak root registry; raw pointer equality alone is insufficient.
The registry canonicalizes allocation ownership/byte hazards, not arbitrary array
headers. Preserve each independent foreign header's element type, dimensions,
offset/strides and import bounds. Ordinary views/reshape/reinterpret aliases retain
their specified geometry; a shared Mantle logical header's size changes propagate
to users of that header, not to unrelated foreign headers. Import/export never
silently patches a CUDA.jl/other vendor header's shape. A new withview lease obtains
the current qualified Mantle descriptor. Each new foreign import captures the supplied header
geometry; reimporting an old vendor header does not discover another Mantle
wrapper's GPU-written size. Reuse that MantleArray directly across graphs when
shared logical size is intended. All overlapping data accesses, including
through different element types, still order against the same physical root; a
readonly/lease restriction cannot be escaped by importing another alias.
Retain the allocation owner, not only a mutable vendor array header. The stream
establishes initial readiness and lease/completion handoff. Caller must not
reallocate or mutate the foreign allocation outside the handoff protocol.
Imports cannot resize beyond their bound, relocate, arena-alias or evict.
Mantle drops its hold after all tracked uses; it never calls the foreign
allocator's free function. Cross-driver import additionally requires compatible
external memory/semaphore capabilities; otherwise use explicit copy.

## 3. Core types

Internal; listed because the verbs in section 4 receive or return them.

| type | what it is | lives in | protected by |
|---|---|---|---|
| `Device` | abstract; one object per hardware device and driver | the registry | registry lock (opening) |
| `DeviceRegistry`, `REGISTRY` | loaded device types, opened devices, the Vulkan instance | process | its lock |
| `Graph` | nodes as declared, named resources, rewrites, profile and priority settings, plan; the flags `recompile`, `windowchanged`, `freed` | user | `Base.Semaphore(1)` (not reentrant; owner task checked); the flags are atomics (section 5) |

| `Plan` | compiled state: stages, placement, argument block, tables, recordings, `lastrun`; the flags `stale`, `unmapped`, `restore`, `running`; `compiles`, the number of compiles of the graph so far, carried from each plan to the one that replaces it (for debugging and tests) | graph | graph lock; the flags are atomics (section 5) |
| `CompileState` | the plan under construction; a copy of the nodes | compile | graph lock |
| `Node` subtypes | `KernelNode`, `TraceNode`, `ClearNode`, `DrawNode`, `Group`, `HostNode`, `HeadNode`, `PrepareNode`, `CopyNode`, `LoopNode` (nested structured control, retained until core chooses native or portable encoding), `RenderNode` (a pass, formed by compile), `BuildNode`, `ForeignNode`, `RegionNode` (retained selectable-operation recipe or fused region, with stable output identities), `VideoDecodeNode`, `PresentNode` | plan (its own copies) | graph lock |
| `CompiledNode` | queue, predecessors, barrier, command, cost, slots | node | graph lock |
| `Origin` | the Julia call that launched a node | node | immutable |
| `Kernel`, `DeviceKernel`, `KernelInfo` | one kernel: its function, the access per argument (declaration), its code (compile), and per device the native handle and `KernelInfo` (first use) | attached to the kernel's `CodeInstance` (`GPUCompiler.cached_results`), one per compiler configuration | the kernel's own lock; code/access immutable once filled; idle native entries may be dropped and recreated under §13 of execution-contracts.md |
| `Bound{R,E}` | a plan's argument slots for a resource/use epoch | plan | node uses |
| `MantleArray{T,N,D}` | owning concrete Device; shared resource state; optional view offset/dims/strides; no graph/group destination | user | — |
| `ArrayState` | storage, dims, length source, capacity, cell, explicit lifetime, external allocation owner, sync state, holds, dependents, readers; the freed mark is on the sync state only | shared by an array and its views | its sync state's lock |
| `Cell`, `NoCell` | an array's address and dims in the device's cell arena; `NoCell` for a fixed explicit scratch whose descriptor is compile-time placement | array state | cells written by applied changes, heads/prepares or compiler-inferred device resize! effects |
| `Region`, `Reserve`, `Committed`, `Evicted`, `Imported`, `Released`, `Unplaced`, `ArenaSlice` | device storage: pool region, sparse reserve, doubling region, host eviction copy, released; explicit scratch: unplaced metadata, then a plan-owned arena slice; Imported retains foreign allocation identity | array state | the array's sync state (host view); graph lock for placement; pool lock for blocks |
| `Arena` | one arena part: range, mapped bytes, tenants, sync state, holder | pool | its sync state (mapping, tenants); its condition (holder) |
| `SyncState`, `Access` | per shared resource: the accesses of what is not a run (eager calls, host reads, vendor views, eviction copies, applied change steps: last write and reads since), the last tokens of plans retired while naming it, pending changes, transfer lease, lent and freed marks. Runs are not recorded here (their last-run tokens, found through the plans naming the resource) | every shared resource | its `ReentrantLock` (never held while `acquire!` runs) |
| `namers` (on `SyncState`), plan inbox, plan sync lock | per resource, the plans naming it with their use from compile (an immutable snapshot, replaced at registration and retirement; composite members reach the plans naming the composite through their readers); per plan, the resources written by other runs, changed or accessed eagerly since its last look (a lock-free list), and the lock that orders its submissions against the plans it shares resources with | the resource; the plan | the resource's sync state lock (`namers`); the plan's sync lock (the inbox is lock-free) |
| `Token` | `(channel, value)`: a value a channel signals. Core's `passed(t::Token)` and `waitfor(t::Token)` call the device-type verbs `passed(ch, value)` and `waitfor(ch, value, timeout)` (section 4.2) | accesses, `lastrun`, retirements | immutable |
| `Change` subtypes | Pending requests: `Store`, `Resize`, `BuildAccel`, each with a device-wide sequence number. A Store owns its upload region and destination mapping into the parent (a byte span when contiguous). `CellWrite`, `BindPages`, `MoveStorage`, `MoveAccel` and `Restore` are prepared application steps, never pending requests. | sync state | sync state lock |
| `SubmitChannel`, `ChannelState` | abstract native endpoint and canonical shared queue state: timeline, lock, next value, failure domain and outstanding submissions. Aliases share the identical ChannelState. | device / endpoint aliases | ChannelState.lock for enqueue; waits outside locks |
| `Recording{N}` | one recorded part of a segment, written once. A segment is a list of parts: recordings and `BuildRecording`s | plan | — |
| `Submission{T}` | a token and the command objects to recycle once it passed | channel | channel lock |
| `Builder` | a device type's recording in progress; a value that never escapes the call that made it (`record!`, an eager call, a submission's change prefix, the re-recording of a `BuildRecording` or a window-sized segment) | stack | — |
| `Deps` | explicit predecessor command IDs and the derived barrier, including internal commands of one recipe | compiled node | immutable |
| `CompiledDispatch`, `CompiledTrace`, `CompiledCopy`, `CompiledBuild`, `CompiledRender`, `CompiledDraw`, `CompiledLibrary`, `CompiledHead` | what compile decided for one command; `CompiledCopy(NoCopy())` is a barrier-only command (including the transition to presentation) | compiled node | immutable |
| `TLAS{D}`, `BLAS`, `InstanceRange`, `RangeEntry` (`mask`, `active`), `MemberSet` | resizing-and-raytracing.jl (acceleration-structure sketch); per-native-binding `builtcount`, the count of its last Build, which that binding's Update uses | user | the structure's sync state |
| `BuildRecording` | a plan's TLAS or BLAS build or refit in its own command buffer, on every device type (Vulkan records the count, Metal the storage). A change that alters what it recorded, or a Build that sets a new `builtcount`, marks it; the run re-records the marked ones after the change prefix (nothing is detected at run) | segment parts | the structure's sync state, held by the run |
| `Window`, swapchain generation | size, pending size, current generation with holds; the swapchain images acquired and not yet presented (the window is their one owner) | user | the window's lock |
| `GPURef` | a value box: one slot in device memory with inferred shader read/write effects and scheduled host stores | user | the slot's sync state |
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

### Core-only helper entries

These have one shared implementation and are not consumer APIs or backend hooks.

| Helper | Internal contract |
|---|---|
| location(args...) | Resolve/validate the common owning Device for eager operation frontends, including callable/keyword GPU resources. Never infer a Graph/Group. Unwrapped foreign device arrays/pointers require explicit import; known host-copy sources use the staged-copy definition. |
| queuepolicy!(dev,policy) | Test-only DefaultQueues/SpreadQueues placement override with identical numerical/ordering semantics. Applications use Graph priority and resource/control declarations; this is not a public queue-routing API. |
| trim!(dev) | Private core cache maintenance, neither exported nor declared public. Release eligible empty caches after passed retirements, without waiting for live uses or unmapping live array pages. Core pressure handling and memory-accounting fixtures may invoke it; applications do not. |
| recipe(f,args), candidates(recipe) | Code-owned Julia methods connect supported signatures to typed recipes and shared candidate emission functions. materialize selects/invokes them with enabled facts and enclosing requirements for eager/recorded execution. These are core/library-definition hooks, not per-backend selection policy or a public registry. |

## 4. Device-type verbs

[Execution contracts](execution-contracts.md) are part of this API: §2 fixes
prepare/claim/accept/publication and terminal failure; §3 completes factories,
object destruction, foreign streams and video; §4 defines presentation retirement.
Sections 6–16 complete capability, SBT, timestamp and memory-class verbs and their
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
| `caps(dev)` | core, at compile and at declaration | An immutable record of facts the device type reports; core decides from them (required schema in execution-contracts.md §6). Fields: `sparseresidency`; `accelbyaddress`; the device's limits; the largest single allocation; the sparse page size and the device's sparse address space (core's `reservesize` and `sparsethreshold`); the driver's job timeout and whether the device drives a display (core's submission budget); the roofline inputs of `costs!` and `schedule!`: throughput per element type, memory bandwidth, whether a transfer queue exists, the cost of a wait between queues (internals.jl core cost sketch: `caps(dev).throughput[T]`, `.bandwidth`, `.transfer`, `.crossqueuecost`); the inline store limit and alignment (`.inlinestore`, `.inlinealign`: 65536 and 4 on Vulkan, `vkCmdUpdateBuffer`); the capability flags for geometry stages, graphics, mesh pipelines, procedural traversal, ray-tracing pipelines and tessellation (Appendix A, `supports_*`). |
| `config(dev, f)` | `kernel!`, before `compilekernel` | Immutable compiler/feature/ABI/group/math/specialization configuration of kernel f; execution-contracts.md §6. |
| `countsource(dev)` | core, at record and run | `CountRecorded()` (Vulkan) or `CountFromMemory()` (Metal): where a TLAS or BLAS build gets its count. |
| `queuekinds(dev)` | core's queue policy | The queue kinds the device has (graphics+compute, compute, transfer, sparse binding, video decode, present), with the number of queues of each and their priority classes. |
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
| `waitfor(ch, value, timeout) -> Bool` | core's `waitfor(::Token)` (host reads, host nodes, pressure) | Blocks until `value` passed on `ch` (`true`) or `timeout` seconds elapsed (`false`; `Inf` waits without limit). A GC-safe call (`@ccall gc_safe = true`). Never called with a sync-state, channel, pool or window lock held. |
| `recycle!(ch, objects)` | the release engine, once a submission's token passed | Returns an eager call's command objects to the device's free list. On Metal the free list holds allocator and command-buffer pairs (plan, Metal, "Channels"). |
| `bindpages!(ch, token, binds, waits)` | core's `sparsesubmit!`, under the channel lock, the token claimed | A queue operation binding pages (Vulkan and Metal through the device type's one sparse-binding helper, only APIs that actually support queue-ordered buffer mapping), waiting on `waits`, signalling `token`. Prepared before claim; enqueue errors use the same terminal rule as `submitnative!`. |
| `unbindpages!(ch, token, unbinds, waits)` | as `bindpages!` (for core-managed idle arena pages) | The same, unbinding. |

A Token names canonical ChannelState plus a value; endpoint aliases share identity.
Core passed(Token) checks native completion and device health. Core
waitfor(Token;timeout=Inf) calls the native wait outside locks; false with a finite
deadline throws WaitTimeout and preserves all pending holds/tokens. It never
infers device loss from elapsed time or multiplies a watchdog interval into a
completion proof. Native device-loss/enqueue errors retain terminal quarantine.
ChannelState.next is core data, never a second backend counter.

### 4.3 Memory

| verb | called by, when | contract |
|---|---|---|
| `rawalloc(dev, kind, bytes, constraint)` | the pool's `tryacquire!`, under the pool lock | One block of `kind` (`Buffers`, `Unified`, `Readback`, `Images`, `Accel`) meeting `constraint` (alignment, usage). A driver call, not a GPU wait; runs no finalizer. The only call site of `vkAllocateMemory` (and of `cuMemCreate`/`hipMemCreate` for blocks). |
| `rawfree(dev, block)` | the release engine | Frees a block whose last use passed. |
| `createreserve(dev, bytes)` | core's growth decision, before any lock | A `Reserve`: sparse buffer, placement-sparse buffer (added to the device's residency set), or reserved address range; no pages mapped. `bytes` is core's `reservesize`. |
| `imagerequirements(dev, spec)`, `placeimage(dev, region, spec)` | core image creation | Query native size/alignment/memory-type/dedicated constraints, then bind an image to a core-owned region. spec and lifetime: execution-contracts.md §3. |
| `placeaccel(dev,path,backing)` | AS materialization | Create/bind the selected native representation using core-supplied compatible committed buffer/heap backing or an admitted opaque-allocation ticket. No backend allocation policy. Array sparse reserves are not implicitly AS backing. |
| `accelsizes(dev,path,spec)` | AS materialization | Requirements for the full geometry/layout/count-bound/build-flags signature: native storage bytes/alignment/backing class, separate Build/Update scratch requirements and qualified copy/compact/update constraints. Core caches and plans them. |
| `emitcopy!(builder,node,cmd::CompiledAccelCopy,deps)` | Explicit AS maintenance | Encode core-selected Clone/Compact from frozen source/destination generations with complete dependencies. Capability-qualified only; no allocation, sizing, policy or host wait during encode. |
| `emitaccelquery!(builder,node,cmd::CompiledAccelSizeQuery,deps)`, `accelqueryresult(dev,slot)` | Core compaction metadata job | Encode a compact-size query into a core-held native result slot; retrieve only after its tracked token/visibility. Retrieval does not enqueue or wait. Slot allocation/retirement and the checkpoint belong to core, not a backend maintenance loop. |
| `acceladdress(dev, as)` | the TLAS code (cell write) | `vkGetAccelerationStructureDeviceAddressKHR` or the Metal resource ID. |
| `objectbytes(dev, native)`, `destroy!(dev, native)` | accounting / release engine | Known native bytes (unknown tracked separately) and destruction after GPU and any presentation retirement. Full creation/destruction contract: execution-contracts.md §3. |

### 4.4 Compile

| verb | called by, when | contract |
|---|---|---|
| `compilekernel(dev, kernel, config) -> (native, compiled::Bool)` | `kernel!` at declaration (eager) and at compile, with no lock held other than the graph lock (compile runs under it) and the kernel's own | Fills the kernel's code if it is missing and its entry for `dev`; the `Kernel` is the only store, no device-type cache. `compiled` says whether this call compiled (core's per-device counter adds it up). |
| `kernelinfo(dev, native) -> KernelInfo` | `kernel!` | Native register/shared-memory facts, group limits and occupancy where available, plus instruction/memory-operation counts from compiled IR. Unavailable native queries yield explicitly marked estimates or bounds, not invented measurements (execution-contracts.md §6). Stored in the device entry; per-argument access is in the `Kernel`. |
| `compiledraw(dev, pipeline, argtypes, formats)` | core's pipeline cache on a render-pipeline miss | Creates the native pipeline for the supplied stages, device-side argument types, fixed state and formats. Cache lookup, invalidation and publication are core's, over device-owned storage. |
| `compiletrace(dev, pipeline, argtypes)` | core's pipeline cache on a trace-pipeline miss | Creates a native ray-tracing pipeline and its shader binding table layout. The same core cache machinery handles its shader/argument key. |
| `compilebindings(dev, spec)` | `bindings!` | Descriptor objects for the frozen binding layout core decided; spec contains no mutable Mantle plan/resource state. |

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
| `Builder(dev, kind)` | Command objects of the queue kind (Vulkan: its queue family) from the device's free list; any queue of that kind can run them. A value that never escapes the call that made it: `record!`, an eager call, a submission's change prefix, or the re-recording of a `BuildRecording` or a window-sized segment. |
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
| External allocation identity, closed library recordings, native regions/generated commands, transport and communicator facts | [§§3/16: composable native leaves](execution-contracts.md#16-composable-regions-resource-effects-and-review-corrections); foreignallocation, recordlibrary, recordregion, compileexecutionset, emitregion!, emitgenerated!, transfercaps, emittransfer!, communicatorfacts and recordcommunication |

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
| a cell | persistent/imported array metadata, or plan-owned metadata for explicit resizable scratch (fixed scratch may use NoCell) | at the last hold; arena backing remains plan-owned | written by applied changes/heads/prepares and inferred device resize! metadata writers |
| graph nodes, plan | declaration, compile | `free!(g)` / finalizer | graph semaphore; one declaring task |
| a `GPURef`'s slot | `GPURef(dev, v)` | its last hold | the slot's sync state; written by scheduled host stores or declared kernel writes with inferred effects |
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
| `recompile` | graph | `markstale!` (a dependency changed, explicit scratch beyond its placed bound or a persistent candidate/shape dependence), a window resized beyond its capacity, a failed compile | `recompile!` |
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
part id → window lock → plans' sync locks by (device identity, plan id) (a stage: its own,
those of the plans sharing what it writes and of the plans naming its inbox
resources; anything else: those of the plans naming the resources it
touches) → sync states by
(device identity, resource id) (an array and its readers together, through `lockexpanded`) →
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
`run!`) only with the graph lock and any run arena-part holds held, never with
a sync-state, channel, pool or window lock.

## 6. Invariants and the tests that check them

Cases named below are target acceptance gates, not assertions of implementation.
Old tests pinning changed signatures/limitations migrate as recorded in the ledger.

| Invariant | Required acceptance |
|---|---|
| One native boundary/core policy owner | Vocabulary/arity, driver-call, globals/probe, Aqua/JET and source-deletion checks. |
| Target, ownership and lifetime are independent | Same device-owned array referenced by sibling Groups/multiple graphs; persistent contents survive replay/recompile/free; explicit scratch escape rejection; operation targets forwarded through broadcast/KA/library recipes. |
| One resource per allocation | Root/view/import alias hazards, readonly checks, allocation-generation ownership and proven-span qualification. |
| One size query and inferred mutation effects | Base.length/size and device resize! through captures, loops and library kernels; KL-01–16 migrated/extended, checked producer scatter and live scratch blocks. |
| Atomic declaration | Invalid target/shape/control/generated prepare leaves nodes/holds unchanged. |
| One command DAG | Multi-command prefix/head/build/library edges, with a DAG-backend missing-edge negative control. |
| Safe scratch reuse | Two unordered readers on different queues; reuse waits for both, even when the last-listed reader finishes first. |
| Replay and plan replacement order | Every root follows previous tokens, including disjoint replacement plans; accepted-prefix tests retain partial work. |
| Geometry reads use a matching build | Same-segment deform -> query/trace, build -> deform, and failure before a later build; transitive dirty versions survive. |
| Resource lifetime follows actual uses | Holds/free/GC, retired plans/native tables, imported leases and snapshots; memory returns after wait/sweep/trim. |
| Native loss differs from deadline | Long legal cross-queue dependency chain yields WaitTimeout without poisoning; actual disposable-context loss quarantines. |
| Control composes | Nested Newton/Krylov/line-search and model loops, metadata refresh, conditional clear/draw and proper collective tails. |
| Candidate growth re-emits the operation | Change candidate command count/workspace while keeping output roots; reference matches fresh declaration. |
| Steady recorded replay stays cheap | No unreported compile/record/device allocation; all temporary prefix/eager encoding is counted; compile/plan-byte scaling is measured separately. |
| Image/presentation correctness | Actual sampled/storage texels, subresources/resolve, conditional shadow reset, swapchain resize and independent present retirement. |
| Native acceleration remains qualified | Modern launch/control/indirect/library/peer/collective features each need facts + compiler + native execution + measured consumer benefit. |
| Named use cases run real programs | DNN/ray/raster/editor/FEM/particles/LLMs and representative training/communication gates; GPUArrays alone is insufficient. |

## 7. Corner cases

- Views cannot be freed/resized separately; aliases retain root hazards. Deleting
  a Group removes/retire its nodes/holds, not the identity/device of operand arrays.
- A persistent array survives Graph teardown while held and can be passed directly
  to another graph. Explicit scratch cannot escape; reject before node publication.
- Only the explicit transport operation admits different device domains. Raw
  foreign arrays/pointers require import before becoming resources.
- Sibling Groups/unrelated graphs may reference the same device-owned array.
  Explicit Group work never silently declares into its parent or executes eagerly.
- A kernel/functor capture is an argument resource; invalid captured device/owner
  fails before node or hold publication. Bool-array flags require exactly one element.
- Device resize! cannot allocate, grow physical storage or synchronize a grid.
  Checked invalid dimensions reject; producer writes use bounded descriptors.
- Internal prepare/copy/build nodes participate in repeated regions. hostcall! is
  rejected only where it is enclosed by device control, not by ordinary Group use.
- free!(a) drops the creator hold; existing graph/composite/lease holders remain
  valid. free!(g) from its hostcall! throws before marking freed.
- hostcall! graph/arena re-entry and out-of-order hold acquisition reject before
  waiting. Arbitrary user task cycles are not proved deadlock-free by this rule.
- A stage may see changes arriving after a preceding hostcall!. Capacity/specialization
  changes it cannot follow stay deferred; other submitters wait outside all locks.
- Minimized windows skip the run and wake screenshot waiters with the documented
  error. Native present OOM retry preserves a completed frame and does not replay it.
- Imported storage cannot grow past its bound or be relocated/evicted/pool-freed.
  Foreign mutation/reallocation outside explicit handoff violates the import boundary.
- A host wait deadline is WaitTimeout, not DeviceLost. Pending objects remain live.
- RNG advancement follows executed regions and explicit shared-state ordering;
  graph recompilation never reseeds it.
- Dynamic snapshots capture payload and metadata together and retain submitted
  results after Graph teardown; no public Plan/record! lifecycle is required.

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

Native conditional-region mechanisms are selected only through enabled facts.
Portable launch/draw control uses indirect counts or entry predication; conditional
clear semantics are a core operation recipe (§16), never a backend policy fork.
`vkDeviceWaitIdle` belongs only to waitnativeidle!(ch) during teardown/debugging
(execution-contracts.md §6); ordinary waitidle(dev) waits a token snapshot.

## 9. Allowlists

- **Mutable globals:** `REGISTRY` (the device registry). Nothing else, in
  core or in a device type.
- **Probes** (`hasmethod`, `applicable`, `hasfield`): none.
- **`@static Sys.is*`:** only in `Mantle.jl`'s include block.
- **`isa`/`typeof` on device or node types:** none.

## 10. Implementation and qualification gates

CPU-only native qualification can load Lavapipe_jll before registry/instance
creation to expose a software ICD. This is test-environment/provider setup, not
an application API requirement; explicit software devices remain available to CI.


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
| `devicesized` | goes: deferred geometry resolved from the standard size model |
| `deviceslice` | goes: `devicearg` resolves views in one place |
| `deviceview` | goes: as deviceslice |
| `download` | goes: eager Array/copyto! or explicit call!(target,Array,a) nodes |
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
| `emitloophead!`, `emitloop!` | go as backend algorithms: core selects structured-region encoding before portable expansion; native emitregion! only encodes the selected immutable recipe |
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
| `makerecording` | goes: `Builder(dev, kind)` |
| `materialize!` | goes: explicit scratch placed by place! |
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
| `passbegin!`, `passend!` | Graph(profile=true), timings(g), shared diagnostics (execution-contracts.md §10) |
| `passed` | verb: `passed(ch, value)`; core's `passed(::Token)` calls it |
| `passedas` | goes: argument layout is core's (bindings!) |
| `patchable` | goes with the pushed patch mechanism |
| `pool` | core: `pool(dev)` |
| `present_frame!` | goes: present is the separately retired `presentnative!` operation in run! |
| `Profiler` | Graph(profile=true), timings(g), shared diagnostics (execution-contracts.md §10) |
| `rawalloc` | verb |
| `rawfree` | verb (its separate AS path goes) |
| `readback_framebuffer` | goes: eager Array(img) or explicit call!(target,Array,img) |
| `readback_window` | goes: call!(presenting_graph,Array,win), then run! and fetch; no implicit future-frame wait |
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
| `screenshot` | recorded window/image Array snapshot, run! and fetch |
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
| `waitfor` | verb: `waitfor(ch, value, timeout) -> Bool`, GC-safe; core's `waitfor(::Token)` calls it with an explicit host deadline; expiry is WaitTimeout, not inferred device loss |
| `waitfor!` | `waitfor!(::Stamp)` goes |
| `waitidle` | core token snapshot wait; native teardown idle is specified in execution-contracts.md §6 |
| `Window` | user: `Window(dev, w, h)` |
| `withpredicate` | folds into builder verbs: core supplies the indirect count/range or capacity entry predicate for `when!` |
| `workgroupsize` | goes: KernelInfo's group size |

## Graph readback futures

```julia
g = Graph(dev)
a = MantleArray(g, Float32, 0; capacity=1024)
app_produce!(g, a)                 # application declares payload + device resize!
result = call!(g, Array, a)        # snapshot here; no CPU array exists yet
run!(g)
value = fetch(result)             # owning Array; fetch submits no work
next_result = Readback(result)    # next request for this exact recorded output
run!(g)
free!(g)
next_value = fetch(next_result)   # submitted snapshot retains storage
latest = Array(a)                 # ordinary blocking eager conversion
```

An array has one device and may be used by many graphs. Array(a) converts accepted
work/pending ordered resource changes eagerly; it does not execute or choose a
graph. call!(target,Array,a), target Graph/Group, always declares a new snapshot
position, before or after compilation. Readback(previous_recorded_future) only
requests another execution of that existing output; it does not declare another
copy node. Multiple positions for one source stay unambiguous. No public eager
Readback(a) mode or second graph snapshot entry is needed.

Snapshot declarations in device-controlled regions reject; put them after the
region. An ordinary Group may contain one. Supported CPU-result operations state
their future result contract explicitly; a ! suffix never supplies that contract.
The canonical Array recipe is the minimum CPU-result case here, not a promise to
record arbitrary CPU code. An actual replay-time CPU callback uses hostcall!.

For requested executions the baseline records capacity-bounded payload and size
metadata in one ordered snapshot. Unrequested outputs omit/gate payload transfer
through the shared region/segment path. fetch returns the captured logical extent;
no host scalar download chooses the transfer size first. Capacity growth refreshes
output placement via plan replacement, preserving Graph/output identity and old
submitted snapshots. Eager and recorded forms share the same copy/hold/visibility
protocol; neither array adaptation nor another readback runtime is involved.

A recorded future isready is false before its execution and becomes true on
completion or failure.
wait returns the future; repeated fetch returns the same owning CPU array.
Compatible recompilation preserves pending requests by stable graph output ID;
submitted futures retain their old plan generation. Graph teardown or output
deletion rejects unsubmitted requests and preserves submitted ones. Failed
runs reject requests whose copy was not accepted instead of leaving waiters blocked.
A copy already accepted retains its token/storage and completes or fails by that
work's outcome; a later unrelated stage cannot erase its acceptance. The run takes
a fixed pending-request set before preparing/claiming that attempt. Requests made
afterward belong to a later run, never patch the in-flight request set. Before replay
overwrites a reusable staging slot, preserve its outstanding prior result, possibly
waiting outside record locks. Host visibility/invalidation follows §6 of the
execution companion. Readback() without arguments remains a memory-kind request.

### Warmup, repeated requests and retention

A request binds to the next run attempt for its output, including warmup.
For a graph with a declared output, the following explicitly warms up the final
command sequence and then requests each wanted result:

```julia
request = call!(g, Array, a)
run!(g)                           # first warmup consumes the initial request
run!(g)                           # no new request: no output payload transfer
for frame in 1:10
    request = Readback(request)    # same output, fresh request before this run
    run!(g)
    pixels = fetch(request)       # snapshot of this run, never "latest"
end
```

The initial warmup transfers one requested snapshot. If avoiding that transfer is
more useful than warming the output commands, declare the output after warmup;
adding its node may require one plan rebuild. Warmup advances persistent state;
resetting simulation/history is an explicit application decision. A retained older
future keeps its original result even after later runs and requests.

Readback(previous) extracts the stable output descriptor; it must not retain the
previous future, CPU payload or submitted generation/token. The Graph tracks
pending requests and the output descriptor, not a permanent list of completed
futures. After completion, only live future/user references retain each captured
result. With dropped prior futures, collectible result storage is bounded by live
requests/results plus the bounded staging cache; replay count alone cannot grow
retained history. Keeping many snapshots explicitly requires their storage. These
are lifecycle requirements, not a claim that current runtime code passes them.
