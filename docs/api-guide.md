# Mantle API: target design and implementation

Mantle provides one Julia GPU API for compute, recorded submissions, GPU control
flow, rasterization, ray tracing and resource management across hardware families.
Applications express work and dependencies; shared core selects implementations
from enabled device/compiler capabilities. Backend-specific application branches
and duplicated backend algorithms are forbidden.

This is the consolidated **target API**, including the new RT/memory contracts.
It is not an implemented/qualified release. All code is Julia pseudocode;
`app_*` functions and supplied data/programs are application code. Examples are
independent fragments. Implementation helpers are private sketches, not extra
public API. Exact signatures: [api.md](api.md); protocols:
[execution-contracts.md](execution-contracts.md); execution/status:
[recording-plan.md](recording-plan.md), [refactor-ledger.md](refactor-ledger.md).

## 1. Devices, arrays and lifetime

An array owns one Device/domain and may be referenced by many graphs/Groups.
Groups contain operations, not arrays or devices. Lifetime is an explicit resource
property, independent of its construction target.

```julia
dev = Device()                          # devices() enumerates available devices
g = Graph(dev)
a = MantleArray(dev, initial_data)       # persistent, initialized once
b = MantleArray(g, Float32, 1024)        # also persistent; no recording context
work = MantleArray(g, Float32, 1024; scratch=true)
v = view(a, 1:16)                        # shares storage; retains view geometry
```

Persistent contents survive replay, graph release and plan replacement while held.
Scratch is undefined until initialized on each relevant run/path; scratch-with-data
initializes at every owning region invocation. A helper may return a scratch
handle/view for further same-graph declarations, but data cannot escape its run.
Zero-size scratch is legal. A scoped constructor `do` form requests scratch and
drops the creator hold in `finally`; graph holds survive the host scope.

`similar`/`KA.allocate` default to persistent device allocations. `adapt(dev,x)`
adapts data/captures; same-domain Mantle arrays preserve identity. Graph/Group are
not adaptation targets; there is no public `Device(graph/group)` conversion.
`KA.get_backend(a)` returns its owning Mantle Device.

Persistent data construction captures its source once, including ordering a GPU
source snapshot before later writes. `readonly=true` and `streamed=true` explicitly
retain immutable sources; writes/resize reject. Streaming plans upload/residency
intervals through core, with transfer costs accounted. `unified=true` requests an
eligible memory class; it does not remove synchronization requirements.

`free!(a)` drops its creator hold and rejects new uses of that handle/aliases;
already declared graph/composite uses remain valid. Graph release drops graph
holds. Last-use tokens govern actual reclamation.

## 2. Eager work, declarations and replay

Ordinary Julia array calls remain eager. A graph names where a supported ordinary
operation is recorded. Raw Mantle node builders name their target directly.

```julia
mul!(C, A, B)                            # ordinary eager operation
cpu_C = Array(C)                         # blocking coherent accepted-state copy

g = Graph(dev; profile=true)
step = Group(g, "step")
call!(step, mul!, C, A, B)                # records; returns this definition's C
call!(step, broadcast!, tanh, C, C)
dispatch!(step, app_kernel!, (C, params), launchrange(C))
run!(g)                                 # compile/bake if needed, then submit
run!(g)                                 # replay; no implicit persistent-state reset
```

`call!` lowers a **known** definition; it is not arbitrary Julia capture. Unknown
functions reject before publication. Each definition specifies returns, effects,
shape/alias rules and workspace. No return heuristic follows from `!`.
The initial catalog includes `mul!`, `broadcast!`, `copyto!`, `fill!`, mapped
`sum!`, explicit-RNG `rand!` and recorded `Array` conversion. Other families
(FFT, convolution, attention, scans/sort, sparse/operator methods) require qualified
shared definitions; merely naming a family does not make every overload supported.

A domain builder is a normal construction-time function, called directly:

```julia
physics = Group(g, "physics")
PrintSim.step!(physics, world, state, dt)  # emits nodes; not call!(..., step!, ...)
# Same principle for DNN emitters, renderer and decoder graph builders.
```

One known-operation declaration validates/publishes atomically. A multi-call Julia
builder is not implicitly a transaction; it prevalidates/cleans up its contribution
on failure. `delete!(g,group)` and `sort!(group;...)` edit containers. A failed
compile never permits execution of stale incompatible bindings.

Actual CPU work during replay is a separate, explicit boundary:

```julia
hostcall!(g, app_inspect, (diagnostic,); writes=())
```

Host arguments are staged and declared writes copied back. This runs outside
record locks, can synchronize, and rejects inside GPU-controlled regions. `call!` never falls back to this CPU callback entry.

## 3. GPU control, parameters and reductions

```julia
active = GPURef(dev, Int32(1))
solve = Group(g, "solve")
dispatch!(solve, app_reset_active!, (active,), 1)
iterations = Group(solve, "iterations")
r2 = MantleArray(iterations, Float32, 1; scratch=true)
app_declare_solver_iteration!(iterations, state, residual)
call!(iterations, sum!, abs2, r2, residual)
dispatch!(iterations, app_check_convergence!, (active, r2, tolerance), 1)
repeat!(iterations, 100; while_nonzero=active)
```

`sum!` returns its ordinary device destination; the next kernel reads `r2[1]`.
Default initialization overwrites it, including the empty sum identity.
`init=false` reads/accumulates its initialized previous contents. No CPU future
or scalar download feeds GPU convergence.

`repeat!` is bounded and nestable, resetting child control at the specified
parent invocation. `iteration(group)` is a typed GPU-loop-index placeholder within
its lexical descendants. `when!(group,flag)` samples/latches its condition once at
entry: changing the flag inside does not disable later nodes of that invocation.
Single-launch conditional/repeat forms are conveniences over the same regions.

A host `GPURef` assignment schedules a value snapshot/store; it does not patch an
in-flight recording. Arrays remain explicit arguments. A parameter is safely
variable only if **all** derived setup tracks it: changing PrintSim's `dt` alone
cannot refresh a timestep-dependent coarse factorization.

## 4. GPU lengths, views and growth

`length`/`size` are the sole logical-size queries. Kernels read prepared descriptors;
host queries of persistent GPU-written dimensions synchronize and return ordinary
snapshots. They never acquire graph provenance. `launchrange(a)` defers launch
geometry without downloading a size.

```julia
active = MantleArray(g, Float32, 0; capacity=4096)
app_compact!(g, active, particles, app_predicate) # application graph builder
dispatch!(g, app_consume!, (active,), launchrange(active))

function app_publish_count!(a, count)
    if KI.get_global_id().x == 1
        resize!(a, count[1])              # device metadata write; no allocation
    end
end

function app_subrange!(out, parent, bounds)
    first, count = bounds[1], bounds[2]
    local_view = view(parent, first:first+count-1)
    app_consume_view!(out, local_view)
end
```

Device `resize!` stays within prepared backed capacity, changes the publishing
invocation's local root descriptor and supplies later dependent dispatches. Other
invocations retain entry snapshots; it is not a grid barrier. Views cannot resize.
The application compaction builder emits flags → scan → checked scatter → count
publication, including zero; this is not an additional Mantle operation.

Host `resize!` can grow storage; it preserves the ordered old logical prefix,
including a GPU-written extent, not stale host-requested dimensions. New elements
need initialization. `copy!(a,data)` atomically discards/resizes/stores; equal-size
view copies preserve mapping. Shrinking retains capacity for later reuse; explicit
array-backing trimming is outside the initial public API.
Capacity is a usable backing bound, not virtual reservation or resident-byte stats;
deferred/evicted storage must be backed before publishing a GPU descriptor.

## 5. Inputs, snapshots and output lifetime

```julia
input[:] = next_input_cpu                # explicit ordered per-run input store
request = call!(g, Array, output)          # declare position + arm next run
run!(g)
value = fetch(request)                   # submits no work
request = Readback(request)              # rearm that output, no new node
run!(g)
next_value = fetch(request)
```

Recorded host-source `copyto!` captures host bytes at declaration; replay does not
watch the original host array. Device-source copies read their graph-position
source. `Array(a)` eagerly snapshots accepted state/pending ordered changes; it
never runs a graph. Graph-position snapshots capture matching payload and logical
size. GPU-controlled snapshot nodes reject; snapshot after the region instead.

Requests bind to a fixed next-run-attempt set, including warmup. Each wanted later
run needs a fresh request. Repeated `fetch` returns that request's original CPU
array. Unrequested outputs transfer no payload. New requests retain output identity,
not a chain of old results; memory follows live requests/results and bounded staging.
Compatible recompiles preserve pending output IDs; submitted snapshots survive
graph teardown. Skips/failure before copy acceptance reject requests; an accepted
copy retains its token/storage despite a later unrelated failure.

## 6. Ray geometry, queries and functions

Geometry/instances are persistent logical resources; hit functions are separate.

```julia
geometry = [TriangleGeometry(vertices, faces), ProceduralGeometry(bounds)]
blas = BLAS(dev, geometry)                # single-geometry convenience forms exist
tlas = TLAS(dev)
instances = push!(tlas, blas, transforms; hitgroup=(1, 2))

tlas[instances] = new_transforms
visible!(tlas, instances, true)
# Also: hitgroup!(tlas, instances, ids), delete!(tlas, instances).
```

Inside ordinary device code, `rayquery(tlas,o,d,args;...)` returns the closest
accepted hit; `anyhit=true` permits early accepted-hit termination. Default
Float32 rays are not silently narrowed from other precision. Hit IDs are logical
and one-based. A miss has `hit=false`; direction is not implicitly normalized.

The same pure alpha/intersection functions work across all implementations:

```julia
struct SurfaceAttributes
    uv::NTuple{2,Float32}
    normal::NTuple{3,Float32}              # object space in this example
end
app_alpha_ok(hit, ray, scene) = app_alpha(scene, hit, ray) >= 0.5f0

function app_intersect(report, candidate, scene)
    # candidate exposes logical IDs, worldray/objectray and instance transforms.
    for (t, uv, normal) in app_roots(candidate.objectray, candidate, scene)
        candidate.objectray.tmin <= t <= candidate.objectray.tmax || continue
        report(t, SurfaceAttributes(uv, normal)) || return nothing
    end
    nothing
end

functions = HitFunctionTable((
    TriangleHitGroup(any_hit=app_alpha_ok),
    ProceduralHitGroup(app_intersect;
        any_hit=app_alpha_ok, attributes=SurfaceAttributes),
))
hit = rayquery(tlas, origin, direction, (scene,); hitgroups=functions)
blocked = rayquery(tlas, origin, direction, (scene,);
    hitgroups=functions, anyhit=true, tmax=light_distance).hit
```

One group ID selects a row, a per-geometry host tuple/vector selects rows by
geometry, or a device matrix `[geometry,instance]` selects dynamically. Table rows
may themselves be tuples with one group per ray type. No native offsets/pointers
are public. Mixed logical BLAS lowering preserves geometry/instance identity.

`intersection(report,candidate,args...)` streams zero/many roots. Filters run before
commit, so rejection of a near root preserves a valid far root. `report` returns
whether reporting should continue, not acceptance: false only after an early-
accept query finishes. Filters/capture are pure; invocation order/count is not
specified. Object rays are already transformed once and must not be normalized.

Each procedural group declares a concrete isbits attribute schema. Queries return
no custom attributes by default. To retain attributes, request a concrete Q and
pure `capture(hit,ray,args...) -> Q`; mixed groups normalize through capture.
`hitattributes(hit)` checks validity. No `zero(Q)`, heap box or scene-sized Union
is needed for a miss.

Programmable shading is optional; current Hikari can retain its wavefront queries:

```julia
shaders = HitFunctionTable((
    TriangleHitGroup(app_mesh_hit; any_hit=app_alpha_ok),
    ProceduralHitGroup(app_intersect, app_sphere_hit;
        any_hit=app_alpha_ok, attributes=SurfaceAttributes),
))
program = RayTracingPipeline(raygen=app_raygen, hitgroups=shaders,
    miss=(app_miss,), payload=NTuple{3,Float32}, maxdepth=2)
trace!(g, program, tlas, (scene,), (width, height))
# In app_raygen(scene): traceray(o, d, initial_payload; raytype=1).
```

| Callback | Contract |
|---|---|
| `raygen(args...)` | Emit rays; return ignored. |
| `intersection(report,candidate,args...)` | Report roots; return nothing. |
| `any_hit(hit,ray,args...)` | Pure Bool filter. |
| `closest_hit(hit,ray,payload,args...)` | Return declared payload P. |
| `miss(ray,payload,args...)` | Return P. |

The convenience `RayTracingPipeline` constructor normalizes a single ray type.
TLAS has no public ray-type stride: core owns any native binding views. Native
stages/SBTs, Metal function constructions or query + typed dispatch are eligible
implementations, never different Hikari algorithms. Resource discovery covers only
entry-point-reachable stages, including captures; no hidden shader globals.

## 7. Acceleration-structure memory

Geometry edits trigger the required derived work before the first query/trace,
even in the same segment. `build!(as)` schedules a full build; explicit
`build!(target,as)`/`refit!(target,as)` declare work. Refit is used only when creation
flags, topology/activity, native bounds and update rules permit it.

| Event | Shared core policy |
|---|---|
| Input growth/topology/count change | Query full native build signature; rebuild/reallocate where required. Spare source capacity alone does not authorize refit. |
| Native replacement/BLAS relocation | Publish readiness; repair parent TLAS references; preserve old child/member generations for their users. |
| Build/update scratch | Per-operation arena/temp workspace; alias only after all users finish. No permanent max-sized scratch per BLAS. |
| Shrink | Keep capacity by default. `compact!(as)` explicitly requests native compaction/smaller rebuild, with possible size checkpoint. |
| Pressure/release | Retire safe cached bindings; reconstruct from retained geometry. Never restore opaque AS bytes by ordinary memcpy. |

AS native byte size differs from geometry capacity. Account storage, native opaque
objects, scratch, tables, overlapping generations and bounded caches once. Admit
replacement/compaction **peak** memory before dropping old storage. Compaction can
change future refit eligibility. Failures preserve old usable generations before
acceptance; accepted prefixes retain their tokens. Each native binding tracks its
own accepted versions/builtcount. `free!(as)` drops the creator hold; graph/member/
generation holds govern actual retirement and residency release.

## 8. Images, raster and GPU-generated work

```julia
frame = Graph(dev)
pipeline = GraphicsPipeline(vertex=app_vertex, fragment=app_fragment,
    depth=DepthLess(), stencil=StencilOff())
color = Image(frame, RGBA{Float32}, width, height; scratch=true)
depth = Image(frame, Float32, width, height; scratch=true)
clear!(frame, color, (0f0, 0f0, 0f0, 1f0))
clear!(frame, depth, 1f0)
draw!(frame, (color, depth), pipeline, draw_args, vertex_count)

texture = Image(dev, RGBA{Float32}, 128, 128, 64; mips=4)
mip = view(texture; mip=2)                # Julia indices, native conversion internal
sampler = Sampler(filter=:linear, address=:clamp)
# Inside device code: sample(mip, sampler, coordinates).
```

Images are persistent by default; scratch is explicit. Subresource views include
mip/layer/ranges/aspect and share ownership. Sampling, storage loads/stores/atomics,
format/filter rules and KI intrinsics are individually capability/compiler-gated.
Shader vertex/instance indices are one-based. Draw index buffers are zero-based
UInt32 wire data; ordinary one-based BLAS face indices are converted by core.
`Sampler` is an immutable value. `GraphicsPipeline`/`MeshPipeline` describe stages,
blend/depth/stencil/topology and typed IO. Unsupported raw features diagnose;
qualified shared candidates must preserve requested semantics.

Plain image `copyto!` is eager; `call!(g,copyto!,dst,src)` records the same operation.
MSAA color resolve is explicit; depth/stencil needs a qualified mode. Snapshotting
multisampled storage requires resolution first. Conditional shadow updates group
**clear and redraw** together; a skipped region preserves existing pixels.

```julia
commands = MantleArray(frame, DrawCommand, 0; capacity=10000)
app_declare_culling!(frame, commands)       # writes records + publishes logical size
draw!(frame, attachments, pipeline, args, commands;
    count=launchrange(commands), indices=indices)

set = ExecutionSet(dev, programs; resources=allowed_resources)
dispatch!(frame, set, compute_commands; count=launchrange(compute_commands))
```

DrawCommand/DispatchCommand are typed semantic records; core owns wire layouts and
bounds checks. Execution sets bound every possible program/argument/resource set;
GPU commands cannot introduce hidden resources. Cooperative/matrix/cluster and
shared-memory requirements feed common eligibility; no invented vendor flags or
per-backend consumer implementation is required.

## 9. Windows and frame capture

```julia
window = Window(dev, width, height)
frame = Graph(dev)
draw!(frame, window, display_pipeline, display_args, 6)
request = call!(frame, Array, window)       # after last writer, before present
run!(frame)
pixels = fetch(request)
```

A window has one presenting graph. `Array(window)` rejects instead of waiting for
an unsubmitted future frame. Skipped/minimized/closed attempts without an accepted
copy reject requests. Accepted copies follow normal prefix semantics; pixels do
not prove presentation completion. WSI retirement is independent of GPU tokens.
`resize!(window,w,h)` changes requested extent; generations/recordings refresh as
needed. Persistent window-sized images do not promise preserved pixels on resize.
Picking captures matching ID/depth/color and immutable frame extent/pixel scale/
plot map. A small ROI uses a view/copy or extraction kernel, not forced full-frame
readback. The Makie adapter performs its documented CPU layout conversion once.

## 10. Foreign storage and multiple devices

```julia
# In an interop adapter, given a compatible Mantle driver/context and stream:
a = MantleArray(compatible_dev, foreign_array; copy=false, stream=foreign_stream)
# withview(CuArray, a, stream) do native_a ... end  # explicit vendor-library lease

transfer = Graph(destination_dev)
call!(transfer, copyto!, destination, source)
collective!(transfer, communicator, +, send, receive)
```

Import retains the actual external allocation, generation, readiness and owner;
never infer identity from pointer equality alone or pool-free foreign memory.
Canonical byte hazards/readonly/lease restrictions are shared, while independent
logical headers keep their own shapes/units. Reuse a MantleArray to share its GPU
size; reimporting a vendor header captures that header's geometry. Exporting a
vendor view may require a coherent size checkpoint; zero-copy is not zero-sync.
An arbitrary callback is conservatively read/write; readonly requires qualified
effects. No lock spans vendor code; stream completion closes the lease.

Only explicit transport/collectives permit different domains; ordinary mixed-
device arithmetic rejects. Peer/native or visible staged-host candidates share
one snapshot/order/ownership protocol. `devicecopy!` is synchronous convenience.
Collectives require matching participants/sequences and consistent failure handling.
Video/DPB and GL/external-image adapters use the same declared resources and explicit
readiness/completion leases; their existing integration gates remain necessary,
not extra decoder APIs invented by this guide.

## 11. RNG, diagnostics and application boundaries

```julia
rng = RNG(dev, 1234)
call!(g, rand!, rng, noise)                # explicit stream advances per execution
Random.seed!(rng, 1234)                   # explicit reset
submissions(g); timings(g)                # graph/queue/host-cut/sample information
naivebytes(g); peakbytes(g)               # scratch before/after aliasing
reserved(dev); mapped(dev); budget(dev)
# tune!(dev, mul!, C, A, B; check=app_reference)  # private trial state, not live data
```

Skipped work does not advance RNG; tuning uses private state/copies and independent
correctness checks. No tuning/probing occurs in baked replay. Statistics distinguish
resident/reserved/virtual/imported/opaque and pending-retirement bytes.
Core reclaims eligible empty allocator caches automatically; cache trimming is a
private implementation/test helper, absent from the supported consumer API.

DNN/LLM/solver/rendering libraries own domain strategies. Sparse composites declare
layout, index width/base, duplicates, symmetry and coherent live counts. Qualified
`mul!`/`ldiv!` definitions or direct builders support matrix-free/preconditioner
paths. LLM page tables/scales/cache leases are ordinary declared resources, not a
backend allocator; contiguous cache growth is not proof of paged attention. AD
operates on numerical definitions/forward-backward emitters, not `run!` bookkeeping;
saved cross-run activations are persistent, same-run ones may be scratch.

## 12. Implementation sketch: one engine, thin native encoding

Private pseudocode; these are the common implementation boundaries:

```julia
struct MantleArray{T,N,D<:Device} <: AbstractGPUArray{T,N}
    device::D
    state::ArrayState                   # logical header + canonical allocation owner
    view::Union{Nothing,ArrayView{N}}
end

# Dense same-element-type matrix multiplication is the expanded example.
# Other shapes/operator types require their own qualified recipe methods.
struct MatMulRecipe end
recipe(::typeof(mul!), ::Tuple{<:MantleArray{T,2},
                              <:MantleArray{T,2},
                              <:MantleArray{T,2}}) where {T} = MatMulRecipe()

LinearAlgebra.mul!(C::MantleArray, A::MantleArray, B::MantleArray) =
    _mul!(location(C, A, B), C, A, B)
call!(target::Union{Graph,Group}, ::typeof(mul!), C::MantleArray, A, B) =
    _mul!(target, C, A, B)

function _mul!(sink, C, A, B)
    validateuses(sink, (C, A, B))
    validate_mul!(C, A, B)                 # shapes/alias/type policy; no GPU-size download
    operation!(sink, mul!, (C, A, B), (C,))
    C                                    # this definition explicitly returns C
end

function operation!(target::Union{Graph,Group}, f, args, outputs)
    node = RegionNode(recipe(f, args), args, outputs, Origin(f, args))
    declare!(target, node)                 # retain intent; no kernel launch here
end
function operation!(dev::Device, f, args, outputs)
    node = RegionNode(recipe(f, args), args, outputs, Origin(f, args))
    executeeager!(materialize(eagercontext(dev), node))
end

# Minimal Candidate layout for this illustration; production metadata also
# carries version, cost and workspace. These are code-owned values, not a registry.
struct Candidate{F,R}
    emit::F
    requires::R
end

# This method connects MatMulRecipe to its implementations. Candidate metadata
# also includes version/cost/workspace; only one candidate is expanded here.
candidates(::MatMulRecipe) = (
    Candidate(emit_generic_matmul!, generic_matmul_requirements),
)
# Tiled/tensor/native-library candidates are additional code-owned entries using
# the SAME selector; no candidates(dev,f) policy table in each backend.

function materialize(context, node::RegionNode)
    impl = select(candidates(node.recipe), enabledfacts(context),
                  node.args, requirements(context))
    # select filters requirements/precision/layout/alias/workspace/control first,
    # then uses common cost/tuning evidence. Unknown/illegal recipes reject.
    impl.emit(context, node.args...)
    context                              # whole emitted DAG, not last emitter return
end

function emit_generic_matmul!(context, C, A, B)
    emit_kernel!(context, generic_matmul_kernel!, (C, A, B), launchrange(C))
end
# emit_kernel! writes into the private compiled command DAG; it neither appends
# declarations to the user's Graph nor immediately submits a kernel.

function generic_matmul_kernel!(C, A, B)
    id = KI.get_global_id()
    i, j = id.x, id.y
    i <= size(C, 1) && j <= size(C, 2) || return
    value = zero(eltype(C))
    for k in 1:size(A, 2)
        value += A[i, k] * B[k, j]
    end
    C[i, j] = value                       # empty inner dimension writes zero
    nothing
end
# Arithmetic illustration only: reuse/refactor the existing shared GPUArrays/KA
# kernel body in production rather than keep another generic GEMM copy.
# Requirement predicates/shape preparation/dispatch mechanics are omitted helpers;
# they are shared core code, not per-device numerical implementations.

function compile(g)
    ir = snapshotspecification(g)
    materialize_regions!(ir)             # common capability/cost selection
    normalizecontrols!(ir)               # preserve structure until encoding chosen
    lower_command_recipes!(ir)
    ensurebuilds!(ir)                    # geometry versions -> derived work
    prepare_metadata!(ir)                # resize writers -> later descriptors
    dependencies!(ir)                    # canonical byte/header effects, one DAG
    schedule!(ir)
    liveness!(ir); place!(ir); sharing!(ir)
    revalidate_schedule!(ir); barriers!(ir); submissions!(ir)
    record!(ir)                          # native encoding; no policy/graph walking
end
```

The explicit hookups above are ordinary Julia methods: recipe(mul!,argument_tuple)
selects MatMulRecipe; candidates(MatMulRecipe()) names emission functions;
materialize invokes the selected emitter; that emitter names the GPU kernel.
The dense example is an illustrative subset, not the complete production candidate
catalog. Additional scalar/shape/operator variants are not captured automatically.
For a graph, materialize_regions! calls this same materialize with final enclosing
control requirements. For eager work it is called immediately. Replay runs only
baked commands; growth/invalidation reruns selection/emission, not the frontend.

One definition/recipe serves eager and recorded execution. Recipes retain stable
output roots and may rebuild candidate commands/workspace on invalidation. Reuse
upstream GPUArrays/KA/other bodies rather than copy them. Code-owned Julia dispatch,
not a mutable registration service, extends known operations. Types encode device,
elements/rank and bounded ABI/program distinctions; graph sizes, resource IDs,
instance counts and dynamic extents remain values. Typed barriers stop cold
heterogeneous storage before hot packing/encoding/numerical loops.

The core DAG includes every internal prepare/copy/build. Aliasing waits for **all**
maximal old users, not the last node in one topological order. Captures and indirect
resources enter the same hazards/holds. Native methods report immutable facts,
allocate/bind supplied objects and encode chosen commands; no duplicate scheduler,
allocator, graph walker, tuning or traversal policy per backend. For example:

```julia
# Native provider implementation, after core has sized/placed/frozen the command:
emitbuild!(builder, node, cmd::CompiledBuild, deps) =
    _encode_native_build!(builder, cmd, deps)
# It must not select Build/Update, allocate scratch, walk nodes or wait here.
```

The same boundary covers native allocation/size facts, submission, image/AS copies,
query/table operations and presentation. Required verbs exist for enabled paths;
unsupported capability families need no dummy methods.

`run!` returns Submitted/Skipped after accepting/skipping work; submission is not
completion. It prepares, claims, submits and publishes each accepted prefix. It updates
resource tokens/dirty versions before unlocking, even if a later stage fails.
Same-graph runs remain ordered across replacement; different graphs order shared
roots while independent work may overlap. `priority` and `share_scratch` are
independent choices. Unchanged baked replay walks no declaration nodes and does
no per-node inference/selection. Host waits occur outside record locks;
WaitTimeout preserves live work, native terminal failures quarantine unsafe state.

## Tradeoffs and acceptance

- **Explicit recording:** no generic capture or implicit Group adaptation. More
  visible declarations buy predictable execution, replay and resource discovery.
- **Persistent default:** avoids accidental data loss; libraries must classify
  scratch to gain arena reuse. Sharing scratch can serialize graphs; independent
  arenas spend more memory for overlap.
- **GPU sizes:** avoid host roundtrips but require capacity bounds, checked producers
  and prepared metadata. Capacity-sized work/transfers can waste bandwidth; measure
  qualified alternatives rather than conceal synchronization.
- **RT generality:** filters/attributes/native tables can increase registers,
  stack, compilation and memory. Plain queries must pay no unused table/attribute/
  program-binding costs; keep compact wavefront records and eligible opaque paths.
- **Native replacement:** safe overlap needs old+new+workspace peak budget. Compaction
  has checkpoint/temporary-memory costs and may make later refits rebuild.
- **Capabilities:** common semantics, not identical native features or cost. A
  qualified shared implementation may replace a native one; unsupported raw
  semantics diagnose. Runtime wins an unavoidable measured compile/runtime tradeoff.

Existing five batches/eight MWE families govern implementation. Check small exact
references and affected paths at coherent boundaries; qualify actual consumers and
Metal/Vulkan separately. Compare warmed matched completed work with interleaved
A/A noise controls, compilation/cache and memory/codegen statistics. A repeatable
regression blocks its profile; scalar/documentation checks are not native or
no-regression evidence. Implementation/performance obligations remain in the ledger.
