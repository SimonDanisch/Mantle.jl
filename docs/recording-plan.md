# Mantle: one runtime model for every device

Mantle is the Julia GPU API across hardware families: compute, submission graphs,
indirect work, raster, ray tracing and resource management use one core model.
Native methods report facts and encode commands. Duplicating a scheduler,
allocator, hazard analysis, graph walker, lifetime algorithm or operation body
between backends is forbidden. Shared numerical bodies may live in Mantle or an
upstream/provider library; there must be one owner for each body. Reuse/move that
body and adapt its emission boundary rather than keep copied Mantle, GPUArrays and
consumer implementations. Distinct qualified algorithms remain valid candidates
under the single shared selector.

This document owns architecture. [api.md](api.md) owns public signatures;
[execution-contracts.md](execution-contracts.md) owns execution/ownership protocols.
For the full API in one document, read [api-guide.md](api-guide.md).
The Julia files are design sketches, not runnable implementations.
[refactor-ledger.md](refactor-ledger.md) owns phased implementation/deletion status.
Existing integrations, numerical decisions, probes and source inventories remain
indexed in [the evidence index](../experiments/review-2026-10-20/evidence-index.md).
Specified requirements are never replacement implementation passes.
User instructions take precedence over this target specification. Existing code
and historical evidence establish behavior and native facts, not authority to
restore a superseded architecture. The implementation protocol below applies
across sessions and compaction.

## Review status

This is a target design, not a completed semantic review or an implementation
qualification. Document/AST checks establish consistency and syntax only. The
open obligations F115–F122 retain their existing batch/consumer gates; they must
not be reported as resolved because examples parse or raw kernels are expressive.

For each API change, trace an actual consumer call path and state what happens
at construction, during GPU replay and on the host after completion, including
returns, storage lifetime and synchronization. Check that the existing interfaces
cannot already express it before adding a wrapper, type or extension protocol.
Review against user decisions as well as source: existing code is evidence, not
permission to restore a superseded model. Apply this within the existing five
batches/eight MWEs; it does not create another recursive validation cycle.

## Rules

1. Core owns policy; native verbs encode immutable commands supplied by core.
2. One public array family, one root resource identity, one operation body.
3. Execution target, storage ownership and storage lifetime are separate facts.
4. No ambient current graph, group, queue, device or RNG; arguments carry targets.
5. Shared changes are scheduled; nothing patches an in-flight recording.
6. Every accepted prefix publishes tokens, change commits and derived-resource
   dirtiness before releasing bookkeeping locks. Later failure cannot erase it.
7. Runs of one graph remain ordered across plan replacement.
8. Native acceleration extends the shared representation. Core chooses a qualified
   construction or diagnoses an unsupported operation before recording.
9. Unsupported raw arithmetic/atomic/fence semantics are never silently weakened.
10. Preserve working consumers and compare equivalent completed work.
11. An array belongs to one Device/domain; many nodes in many graphs may reference
    it. Graph/Group membership belongs only to operation nodes. Every recorded
    ordinary operation uses call!(Graph/Group,f,args...); raw graph primitives
    explicitly name their target. Standard no-target array calls are eager,
    including when an operand was constructed or already named in a graph.

## The model

### Devices

One physical device through one driver has one Device and one core Runtime/pool.
The registry is the only mutable process-level owner. Device() selects the platform
hardware default; explicit software devices remain available to CPU-only CI.
CUDA/HIP extensions add devices, not independent runtime models. Two drivers for
one GPU are distinct ownership/synchronization domains.

Vulkan desktop and Apple-silicon Metal 4 are the current graphics qualification
paths; CUDA/HIP add compute. Other platforms join through the same contracts.
Desktop evidence does not establish mobile/browser/older-Metal support, and a
compute-only API does not acquire native graphics capabilities.

The mechanisms below are native encoding facts, not separate runtime policies.

| Mechanism | Vulkan | Metal 4 | CUDA/HIP |
|---|---|---|---|
| Recording | Family-compatible command buffers | Compute/render ICBs, command buffer per segment | Explicit native graph DAG/executable where enabled |
| Argument descriptors | Prepared flat slots | Prepared argument memory | Same slot-pointer ABI, qualified per compiler |
| Device control | Indirect counts or eligible native region | Execution ranges/eligible region | Native control when qualified, otherwise shared bounded/predicated lowering |
| Mapping | Enabled queue-ordered sparse buffers | Independently qualified placement-sparse buffers | Committed initially; host VMM alone is not queue mapping |
| Libraries | Core/generated kernels, eligible native candidates | MLX kernels; event-chained foreign MPSGraph where eligible | Kernel/direct-populate/closed-recording/foreign candidates by actual requirements |
| Graphics | Enabled Vulkan stages/WSI | Enabled MTL4 stages/drawable protocol | Compute-only unless an independently enabled graphics bridge supplies it |

Metal render state/ICB residency and MPSGraph handoff cost remain P9.A1 gates.
Bindings never route through task-local vendor queues. CUDA/HIP native graph
parameter changes target idle/new executables, not an in-flight recording.

### Arrays

MantleArray{T,N,D} is the one array family; D is the owning Mantle Device type.
Ordinary views share its logical parent/device and carry offsets, dimensions and strides;
conjugating adjoints retain their conjugation wrapper. An array stores no recording
Graph/Group destination. Node uses hold the resource and identify their own target.

- MantleArray(dev, T, dims...) and MantleArray(g, T, dims...) are persistent.
  Physical placement may be deferred; storage and initialized contents survive
  replay, node edits, originating-graph release and plan replacement while held.
- scratch=true explicitly requests run-local workspace. Graph scratch is arena
  placed and aliased only with complete GPU ordering; eager scratch retires by
  last-use tokens. Ordinary constructors, similar and KA.allocate do not silently
  change lifetime. Numerical helpers request scratch explicitly.
- Generic library allocations stay persistent unless explicitly classified as
  scratch. Correctness does not depend on guessing which returned arrays escape.
- Another graph references the same persistent array directly in its explicitly
  targeted operations, without copying or adapting it. Scratch cannot escape its
  owning run; copy to a persistent result or an owner-local recorded snapshot.
- adapt(dev,x) is device/data adaptation. A Graph or Group is not an adaptation
  target, device, or KA backend. Core resolves the owning physical device internally
  for validation/materialization; this does not add public Device(Graph/Group)
  constructor/query overloads or change where an operation is recorded.
- Persistent construction data initializes once at its ordered construction point;
  deferred GPU copies preserve that source version. Replay reset is an explicit
  fill!, copyto! or kernel node. Recompile never resets persistent state.
  Explicit scratch=true data construction initializes each execution of its owning
  region (including repeated iterations) through the shared recorded-copy path.
- Imported native allocations use the same array/state with a retained external
  allocation owner, canonical alias identity and readiness/completion edges.
  Mantle never relocates, evicts or pool-frees them.

Images have the same persistent default and explicit scratch=true option.
Window-size tracking does not imply scratch lifetime.

### Lengths and resizing

Base.length and Base.size are the only logical-size queries. Kernels read a
prepared dispatch descriptor. On the host, querying GPU-written size synchronizes
and reads metadata; an ordinary returned Int is a snapshot. It does not acquire
hidden graph provenance. Graph recipes use prepared descriptors/retained shape
requirements. launchrange(a) supplies deferred geometry without a scalar download.
For a coherent payload/size result use call!(g,Array,a) at a graph position, or
Array(a) after eager/accepted work; length/size then query the CPU array.

Base.resize! changes dimensions. On the host it schedules resize/growth. In a
kernel it publishes dimensions within existing capacity and allocates nothing.
The compiler infers this metadata-write effect and core inserts preparation
before later consumers. No public writable size-cell handle exists.

A single device invocation publishes a given update unless a specified collective
protocol elects it. That invocation's local descriptor sees its new dimensions;
other invocations retain entry snapshots until a dependent dispatch is prepared.
resize! is not grid-wide synchronization. Views cannot be resized. Cells are
internal and ordinary numerical kernels cannot read them.

Capacity is the usable allocation bound, excluding unbacked virtual address space.
For deferred/evicted storage this is a retained backing requirement, not evidence of
resident memory; core backs it before publishing any GPU descriptor. Device-visible
capacity never exceeds the backing held for that use. The bound is initialized
even without an explicit reserve. It plans maximum scratch, never substitutes for
live length. Resident/reserved bytes are separate accounting facts.
For an application builder app_compact!(target, active, particles, predicate),
capacity N covers N inputs. Flags,
scan, checked scatter and final resize!(active,count) have explicit dependencies.
Scatter uses an internal capacity-bounded descriptor over the SAME root; consumers
get the prepared logical extent. The count publisher executes for empty input.
No new public compaction operation, array/view abstraction or automatic growth/retry
is required. This target-aware builder composes declarations directly, without
call! registration. Array shrink retains capacity; a public array-trimming API is
deferred pending a concrete consumer. Cache trimming remains private core policy.

Host growth preserves the old logical prefix using ordered GPU metadata when
needed; it never freezes a copy length from stale requested dimensions. It
prepares committed doubling storage or
qualified queue-ordered sparse mapping, copies, publishes metadata and retires
old storage by tokens. Imported storage cannot grow beyond its imported bound.
Plans rebuild only invalidated shape/candidate/placement dependencies. Host stores
against GPU-written logical bounds use a shared checked-store recipe; trim/restore
use coherent applied metadata, never stale host requested dimensions (§16.2).

### Operations

The shared Mantle operation layer defines array/numerical operations, in the same
role that GPUArrays provides common array methods. Standard eager methods keep
their ordinary Julia signatures: mul!(C,A,B), broadcast!(f,C,args...), Array(a).
There is no extra public device argument and no Graph/Group overload of mul!.
Reuse existing GPUArrays/KA/shared Julia algorithms where their interfaces permit
this emission model; do not fork eager and recorded copies of an algorithm.

Recorded ordinary-function calls use call!(target,f,args...;kwargs...), where
target is a Graph or Group. For example call!(group,mul!,C,A,B) declares that
operation in the Group; call!(g,Array,a) declares a snapshot at that graph position.
A plain mul!/broadcast/KA call remains eager regardless of array construction or
which graphs already use its operands. Direct Mantle graph primitives such as
dispatch!, draw! and trace! remain explicit node-building APIs.

The initial concrete catalog and GPU-resident sum! destination contract are in
api.md's Minimum recorded array-operation contracts section. A one-element
MantleArray result is consumed by later device kernels; no CPU future or new scalar
type is needed. In-place reductions use prepared live extents and execute their
empty-result publisher rather than branching on host size while recording.

This is supported-operation lowering, not arbitrary Julia capture. Each known
operation has one shared definition/recipe and thin eager/recording frontends,
implemented through ordinary Julia dispatch and code-owned metadata, not another
mutable global registry. Unsupported recorded calls reject before publication;
they never execute eagerly or become CPU callbacks as a fallback. A composed
library needs explicit recorded-call emission or a qualified closed-library recipe.
It cannot infer a target from its arrays, KA backend or ambient state.

Returns come from that definition, not the function's spelling or a ! suffix.
For mul!, the shared body returns the original C. Allocating GPU operations can
return stable device-owned outputs; a CPU result unavailable until execution,
such as Array(a), has a typed future under the common recorded-output protocol.
Neither every mutating call returns its first argument nor every call returns a
future. Each supported family states its declaration-time return contract.
Returning C makes its identity available, not its future contents: call! submits
nothing. Array(C) before run!(g) sees prior accepted state, not the pending call.
Use a recorded Array conversion when a particular graph position is required. GPU
control rejects CPU-result snapshot nodes; use device-output forms inside those
regions and snapshot after the region. Metadata-only returns need no fabricated
GPU command. This contract does not permit an eager computation during recording.

One shared core owner performs capability-based candidate selection for BOTH
frontends. It intersects enabled device and compiler facts with element/layout/
precision semantics, shape/capacity/alias bounds, workspace and execution-region
requirements. Eligible candidates may be shared Julia kernels, intrinsic-based
variants, or qualified native library recipes. Performance evidence selects among
eligible implementations; a feature bit alone is neither legality nor speed proof.
No backend chooses algorithms, workspaces, fallbacks or schedules independently.
Backends report native facts and encode chosen commands/intrinsics; reuse KI facts.

Selectable operations retain RegionNode recipes: call/code identity, arguments,
stable output identities, result contract, requirements and effects. Graph compile
materializes/selects commands and scratch using the final region requirements;
eager execution materializes the SAME recipe immediately. Replay uses the baked
choice, not Julia operator dispatch, probing or tuning. Growth/invalidation may
reselect the entire command/workspace sequence while preserving public outputs.
Raw kernel dispatches remain KernelNodes.

Selection keys include code, types, bounded shapes, strides/layout, alignment,
aliasing, target controls and compiler/feature/ABI versions. GPU live sizes use
shape/capacity classes, never scalar readback for selection. Recipe-declared shape
dependencies are validated; ordinary host integers are fixed snapshots, not values
with recoverable graph provenance. tune! uses private data/RNG/recurrent state,
independent references and completed timing; it never runs on declaration/replay.

Closed native-library capture is eligible only for a complete recipe with explicit
resources/workspace, compatible controls and qualified recording/handoff behavior.
Foreign/native host stages disclose their costs; they are not arbitrary Julia
capture or a second graph runtime. Actual CPU callbacks use hostcall! explicitly
and execute during replay with the existing staged-data/host-cut protocol.

### Kernels: Mantle.Kernel

CodeInstance-owned Kernel metadata holds inferred access and compiled code keyed
by compiler configuration. DeviceKernel holds native entry and honestly queried/
estimated KernelInfo. Shared core cache publication also owns raster/ray pipelines.

Discovery covers callable fields and every shader stage. Unknown effects are
ReadWrite; unsupported untracked pointers fail. Inference includes resize! metadata,
RNG, asynchronous descriptor resources and atomic/fence scope/order. Reuse KI's
tables and intrinsics, with no duplicate capability caches or per-intrinsic bridges.

Explicit workgroups are preserved. The bounded default targets 256 invocations:
1-D 256, 2-D 16x16, 3-D 8x8x4, reduced to legal limits. Matrix/collective kernels
supply legality/preferences. Cooperative/cluster and dynamic shared-memory
requirements enter candidate eligibility and compile keys.

### Composition boundaries

Classify a function by its actual body and contract, never by its name or ! suffix.

| Kind | Construction/host behavior | During replay |
|---|---|---|
| Target-aware graph builder, e.g. PrintSim.step!(group,...) or DNN emitgraph!(group,...) | Direct Julia call composes declarations into its explicit target. No call! wrapper/registration is required. | The emitted nodes execute; the builder body is not replayed. |
| Supported ordinary array operation, e.g. mul!(C,A,B) | The plain call is eager; call!(group,mul!,C,A,B) declares its known GPU operation. | Its selected GPU commands execute. |
| Host controller, e.g. Bonsai step!(session,token) | Updates inputs and runs a plan; it is ordinary host orchestration. Do not wrap the controller in call!. | Only the submitted plan's work is replayed. |
| Explicit hostcall! | Declares a host boundary with its staging/ordering contract. | Its CPU callback executes at that boundary. |

These are semantic roles, not new public wrapper types or a function-name registry.
Current consumer Graph-only helpers still need explicit Group support; showing a
target form here does not claim that port is implemented. Source traces and open
obligations are recorded in the existing use-case review.

### Groups and control flow

Target-aware graph-building helpers are called directly during construction:
PrintSim.step!(group,world,state,dt) emits its timestep into the supplied Group,
using call! for supported array operations and dispatch! for kernels inside it.
Such a helper needs no call! wrapper or operation registration; its Julia body
builds nodes once, and run! replays their GPU work. It is not a CPU callback during
replay. A recorded ordinary operation and a graph-building helper are distinct
interfaces; no target is inferred from arrays in either case.

Group(parent,name) is a structured region at a stable parent position. Groups
nest and contain operation nodes. Explicit call!(group,f,args...) and raw
node-building primitives add work to that Group; target-aware helpers compose
these declarations. Array operations use shared Mantle operation definitions. Arrays do not have
a Group membership, and returned aliases never carry a recording destination.
A requested Group target never silently falls back to its parent.

repeat!(group,maxiters;while_nonzero=count) and when!(group,flag) attach controls
to a region. A condition is sampled once at region entry and latched for its body; convenience single-kernel/draw forms remain. Loops nest. Metadata
prepares, eligible libraries, copies, render work and derived builds participate
in the same region. hostcall! is allowed in ordinary regions but rejected in a
device-controlled region before mutation. Host policy can orchestrate runs.

Core retains structured control until choosing a qualified native construction
or the portable expansion. Bounded fallback iterations have distinct command/
argument slots, shared logical scratch where legal, resettable control and edges
from ALL body tails to the next head. Stopped loops remain stopped until their
enclosing invocation resets them. Native loops preserve nested reset, effects,
termination and submission-budget semantics.

Without indirect dispatch, capacity predication gates whole inactive groups.
Partial groups retain collective participation with operation-specific neutral
lanes; divergent early return before a workgroup barrier is invalid.
Compile time, plan/slot bytes and post-termination command cost are measured.

### The graph and its plan

run!(g) compiles/records when needed and submits; Plan is internal.
Edits preserve Graph identity and persistent contents. Replacement plans inherit
previous-run tokens even for disjoint old/new resource sets. Old plans retire
by actual last-use tokens.

Compatible draw Groups remain locally replaceable pass pieces. Insert/delete/
sort compiles the affected piece and rewrites the flat pass list, O(pieces),
subject to RM-06. New layouts, shader writes, feedback or placement/queue changes
rebuild the plan. Core owns the draw walk and use registration/retirement.

### Compile

| Stage | Result |
|---|---|
| snapshot/validate | Immutable declaration tree, complete atomic validation and holds. |
| materialize regions | Candidate recipes, stable outputs, scratch/shape dependences. |
| normalize controls / formpasses! | Nested control tree and render regions; no mandatory early unrolling. |
| optional rewrites | Shared semantic fusion/region matching, with effects revalidated. |
| choose controls / lower recipes | Native encoding or portable expansion; command IDs and internal edges. |
| ensurebuilds! / metadata preparation | AS versions/builds-before-reads and descriptor preparation at each write epoch. |
| dependencies! / costs! / schedule! | Access frontiers, estimates, legal partitions and queue lanes. |
| hostcuts! / presentspan! | Host transfer/callback stages and window span, with edges updated. |
| liveness! / renderstores! / place! | Final DAG scratch lifetimes and observable attachment stores. |
| sharing! | ALL prior-use-frontier -> next-use-frontier alias edges; validate schedule again. |
| barriers! / submissions! | Visibility/layout/family transitions and cross-lane waits from the same effects/edges. |
| summary! / bindings! | External range uses, inbox registration, slots/tables and frozen commands. |

Any command-list rewrite rebuilds affected edges, costs, lifetimes and summaries
before record. A native emitter never infers sequencing from a barrier mask.

#### Submission split: estimated

Roofline/IR/calibrated costs are estimates, not proven lower bounds or universal
watchdog protection. Core cuts between operations/iterations. Candidate tiles
must preserve effects and reductions need a combine. Plain launches split only
with split=true and a valid workgroup-range contract; cooperative/cluster work
cannot be cut unless its declared semantics allow it.

LaunchTooLong requires a validated unsatisfied constraint or independently
justified bound. A calibrated heuristic alone does not prove impossibility.
Diagnostics retain estimate/provenance/risk. Confirmed native loss reports recent
submissions and quarantines the domain. Host deadline expiry is WaitTimeout:
work remains pending and held, with no fabricated completion or device loss.

### Queues

Compile assigns family/kind and logical lanes. A run maps independent lanes to
eligible native endpoints, including multiple endpoints of the SAME kind.
Aliases share one ChannelState. Core selects placement from waits/load; test
SpreadQueues changes placement only. A recording is family/kind compatible.

priority is urgency. Graph(...;share_scratch=true), the default, permits shared
scratch arenas and serialized tenants. share_scratch=false gives independent
scratch placement so equal-priority graphs can overlap when resources permit.
Persistent resources never arena-alias, and runs of one Graph remain ordered.
Priority promises no preemption or overlap.

### Ordering across graphs and queues

Views retain canonical root identity. Uses may refine it with proven byte spans/
image subresources; unknown overlap uses the whole root. Size/address metadata
and plan descriptor-slot epochs are separate accesses, with metadata changes
ordering all affected consumers even for disjoint data.

One core hazard algebra serves frontiers, DAGs, barriers and inbox conflicts.
Alias reuse follows ALL maximal preceding uses before ANY minimal new use,
or an equivalent join/entry. Topological position is not completion.
Unknown composite membership may retain AnyWrite, with its cost measured.

Namer/inbox ordering retains cheap unchanged runs: inspect modified/inbox resources
and written outputs, not every named read. Immutable member/namer snapshots choose
ordered locks and are validated under them. Whole-last-run waits are a conservative
baseline; precise range summaries can refine them without a second sync owner.

Lock order: graph -> arena holds by ID -> window -> plans by ID -> roots by ID ->
ChannelState. Pool/cache/table locks are leaves. Allocation/pressure, GPU waits and
callbacks occur outside plan/resource/channel locks; graph locks/arena holds may
span host stages. Re-entry/order violations throw before waiting.
Finalizers enqueue idempotent drops only. Detailed proof: execution-contracts §§1–2.

### Record

Core walks the final command DAG. Heads, build preparations, prefixes and library
recipes have explicit internal edges. Builder/emit/finish get frozen commands and
complete predecessor/barrier information. Record compiles nothing, allocates no
device storage, submits nothing and reads no live resource state. Prepare command/
packet capacity before claim.

### Run

Prepare -> claim -> native accept -> publish is the common path.
Preclaim failure can retry. Postclaim enqueue/publication failure poisons and
quarantines; no counter rollback. Publish tokens, change commits and transitive
dirtiness per accepted segment, including partial runs. Submission is not completion.

hostcall! uses staged host-visible arguments and retains re-entry rules.
Readbacks capture a graph position/run token and preserve outstanding snapshots
before staging reuse. Submitted futures remain fetchable after Graph teardown.
Graph-owned pending requests survive compatible plan replacement by output ID.
call!(target,Array,a) declares a position; Readback(previous_recorded_future)
requests another execution of that exact output. Array(a) is blocking eager
conversion; no separate eager Readback(a) operation is exposed. Unrequested outputs transfer no payload, with control/submission costs measured.
Dynamic results capture payload and size from the same execution.

### Eager work: operations on a device

Eager and graph entries materialize the same recipe/body. Eager dispatches submit
before return; host uploads into existing arrays remain scheduled stores.
Use the same command DAG, pool and retirement. No ambient batching or second
numerical implementation. Array(a) explicitly snapshots a device-bound persistent
resource. A recorded snapshot names its target explicitly as call!(g,Array,a),
including owner-local scratch snapshots. Host conversions never run a graph.

### Acceleration structures and textures

One core BLAS/TLAS owns ranges/members, count policy, dirty versions, scratch and
build/refit/retirement. Native verbs supply sizes, records, handles and commands.

Insert a build/refit between geometry writes and dependent AS reads even within
one segment. Replay can record that derived build without host rediscovery.
Cross-graph writes pend transitive dirtiness before publication. Only an accepted
build for the matching geometry version consumes it; built-somewhere-in-the-plan
never suppresses later dirtiness. A failing later stage cannot erase an update.

Host totals build exactly; device totals use fixed capacity, valid mask-zero TLAS
records and degenerate BLAS padding. Updates keep builtcount and native activity/
topology constraints; index writes require Build. Native exact-count optimizations
need independent legality/performance qualification.

Image views refine subresources without separate ownership. KI owns typed sampling,
storage and atomic intrinsics; core validates format/footprint/usage and derives
layouts. Descriptor tables, sampler interning and retirement retain §§7/8/11.

### Graphics

Core forms render passes and pieces. A conditional render region includes clear
and draws: active rebuilds the shadow map, inactive preserves it. Without native
region control, a shared core clear candidate uses predicate-controlled clear
draws with Keep loads, followed by conditional draws. Disabled draws write nothing.
Measure bandwidth/tile costs; consumers/backend policy do not implement resets.

Typed indirect-command arrays carry counts and offsets plus GPU command count.
Core owns layout, bounds, declared resource/execution sets and hazards. Native
multidraw/ICB/device-generated commands encode that same region. No GPU command
can introduce an untracked resource or arbitrary undeclared shader.

### Windows and present

One graph presents to a window; many windows may share a device. Minimized runs
skip and leave edits pending. Acquire happens outside record locks. Generations
retain images, GPU tokens AND independent present completion.
A screenshot is call!(presenting_graph,Array,win), then run! and fetch through the
common output protocol. Array(win) rejects; it never waits for unsubmitted future
frames or drives a graph implicitly. Frame pixel/ID/map/extent agreement and ROI
reads use the consumer contracts in execution-contracts §16.12.
Resize refreshes image/recordings within capacity or recompiles beyond it.
Persistent simulation/film state is not reset by that recompile.

presentnative! is separate from submitnative!, after all frame stages are accepted.
A not-enqueued presentation OOM retries only the completed frame before edits.
Legacy Vulkan shutdown retains its documented proof limitation; maintenance1 is
the strict retirement path. Normal resize never idles the device.

### Memory

One core pool owns blocks, regions, dedicated allocations, scratch arenas and
retirement/pressure. Imported allocations retain external owners and count
separately. Opaque native objects report unknown bytes honestly.

Pressure only in acquire!: drop idle native caches, unmap idle scratch pages,
trim empty blocks, collect, wait for retiring storage, evict eligible idle roots,
then fail. No record locks span waits. Persistent contents are never discarded.
Read-only retained sources obey their existing immutability contract; writable
eviction uses bounded pageable HostCopyJob. Direct restore and the debug-first/
no-speculative-staging ruling remain execution-contracts §12/F74.
QueueMapping and host VMM stay distinct; CUDA/HIP initially use committed arenas.

Streamed weights declare resident intervals/upload nodes and source holds; alias
and control analysis determines eligible reuse. Paging/bandwidth remain real costs.

### What a device type implements

api.md §4 and its linked execution-contracts sections define the exclusive native
boundary. No backend decides scheduling, growth, lifetime, hazards or AS dirtiness.
A library/region adapter encodes its supplied closed recipe, not the Mantle graph.
Shared numerical algorithms remain core recipes with native intrinsic lowering;
genuinely specialized candidates have explicit requirements and measured wins.

## One call, one graph

One semantic call declares one graph with reusable structured regions.
Persistent outputs connect calls without copies. hostcall!, native library regions
and streamed weights do not require consumer-owned graph partitioning.
GPU-controlled autoregressive decode is one region; host sampling may orchestrate
one run per host decision. Model load/quantization is its own call.

Explicit RNG resources carry seed/counter state through replay/nesting.
Recompilation preserves state; reseeding is explicit; tuning uses private state.
Scheduling does not select an ambient RNG.

Explicit asynchronous copy/communication nodes compose per-device graphs.
devicecopy! remains synchronous convenience; call!(target,copyto!,dst,src) is the
recordable asynchronous transport form. Qualified peer/external/collective candidates use
the same holds/effects/tokens. A portable bridge preserves source snapshot and
destination publication and discloses host cuts and transfer costs. Mixed-device
arithmetic remains an error; no implicit cross-driver scheduler is required.

## Compile and runtime performance

Runtime performance is a requirement, not an optional cleanup after the refactor.
First look for a representation that improves both compile and runtime costs:
shared inference/caches, compact indexed plans, retained structure and bounded
specialization. If measured alternatives leave an unavoidable tradeoff, prefer
runtime performance. Record the alternative, cold/warm compilation cost and
completed-work runtime evidence; do not choose slower execution merely because
its Julia implementation compiles faster.

Types describe stable representation and code/ABI requirements. Graph identity,
node count, runtime extent, resource identity, loop trip count and recorded values
are data, not type parameters. Preserve concrete device types, kernel argument
layouts and selected callable types at compilation/encoding boundaries. Never
encode an entire graph or loop body as a nested tuple type. Ordinary typed kernel
argument tuples remain appropriate. A value legitimately needed for generated
code may specialize only with a bounded, measured policy and a correct generic
path. Do not erase hot numerical argument types to save compilation.

Core owns compact command/resource tables and the single traversal. Typed payload
collections and function barriers isolate heterogeneous declaration state from
repeated execution. Abstract containers are allowed at cold boundaries, but cannot
force repeated inference, boxing or dynamic interpretation into a baked replay.
Do not replace dispatch with Symbol kinds or duplicate tables/algorithms per backend.
Details and measurement obligations are in execution-contracts.md §16.11.

## How we build it

### Authority and checkpoint

At the start of a batch and after compaction, read the target rules, the active
batch and its ledger IDs before reading old implementation code. Keep a small
checkpoint under the batch's experiments directory: target document hashes and
accepted decisions, base revision/worktree, symbol -> new owner/deletion map,
changed files, completed checks and exact next action. Each check records command,
input/profile, revision or file hashes, device/compiler facts, session/run ID,
result and log path. Read that checkpoint before launching anything. Completed
checks remain evidence for their inputs; a context reset never justifies rerunning
them. Changed contracts invalidate only the checks whose assumptions changed.

Old source is evidence for numerical behavior, ABI, lifetime obligations and real
consumer needs. Resolve conflicts by explicitly mapping the old behavior to the
current contract. Do not silently import its allocation defaults, backend policies,
legacy entry points or graph structure. A newly proven native limitation follows
the debug-first procedure below; it does not automatically revise the plan.

### Work batches and logical phases

Preserve frozen baselines, numerical rulings and per-file deletion maps.
Reconcile named case expectations before foundation work. Implement the final-API
vertical slice in an isolated checkout/load path: Device, MantleArray, Graph,
dispatch!, run!, actual allocation/holds/retirement and snapshots. No forwarding
into legacy graph/array implementations. Minimal fixtures use the final API;
private test instrumentation is not another runtime path.

Name one owner for every replaced algorithm. Consolidate verified independent
duplicates first; foundational replacements remove named old copies when a
backend joins. Separate runtime, numerical kernels, ABI, comments and application
algorithms in size reports. A line-count target permits no hidden duplication,
required-fallback deletion or moving code just to conceal it.

Phases retain stable ledger IDs and identify coverage/ownership, not mandatory
validation sessions. Work may cross phase boundaries without declaring an earlier
phase done. A batch implements its connected types, call sites, native boundary
and fixture before starting Julia; editing each file does not trigger a test run.
Static review/parse checks may run between edits. Prefer several such coherent
batches over one unvalidated rewrite of all backends and consumers: an initial
native slice detects ABI/ownership mistakes before they spread. Beyond that slice,
make connected changes together and reuse its evidence. Do not require incremental
legacy adapters to keep old tests green during construction.

| Batch | Implementation scope | First applicable fixed MWEs |
|---|---|---|
| B1 | Phases 1–3 plus minimum phase-4 descriptors: final types, declarations, pool/holds, command DAG, one production native encoder, replay/snapshot/failure handling; distinguish canonical payload identity from logical metadata (F123). | MWE-01 lifecycle; MWE-03 queues/prefix; MWE-08 straight-line |
| B2 | Remaining phase 4 and phase 6 operations/control: GPU sizes, growth/restore, selection recipes, nested Groups, broadcasts/KA, RNG; phase-7 dynamic snapshots. | MWE-02, MWE-04; MWE-01 GPU snapshots; MWE-08 control |
| B3 | Phase 5 and remaining phase 7: AS, images/indirect raster, windows/pieces/present; phase-8 native import/stream prerequisites. | MWE-05, MWE-06; MWE-03 AS prefix; MWE-07 import |
| B4 | Remaining phases 8–10: device/reset ownership, additional native encoders, qualified library/modern launch/control and transfer/communication paths. | Same MWEs under newly enabled capability profiles; affected existing native fixtures |
| B5 | Phases 11–12: actual consumer migrations, performance comparisons and final applicable suites. | Consumer gates below, then full qualification |

The logical phase scopes remain:

| Phase | Required slice |
|---|---|
| 0 | Documentation/case migration, frozen baseline and nonempty gates. |
| 1 | Final types, operation recipes, structured controls, command DAG/frontiers and selection rebuild. |
| 2 | One pool/holds/release engine; explicit persistent/scratch/external storage and alias ordering. |
| 3 | Canonical endpoints/lanes, inboxes, submitted prefixes, hostcall!, WaitTimeout and quarantine. |
| 4 | Slots/Cells, inferred device resize!, size/capacity, scheduled growth/stores and eviction. |
| 5 | Shared AS versions/builds-before-reads, image views/tables/samplers and recording refresh. |
| 6 | Explicit Graph/Group operation targets, broadcast/KA/library composition, selection, explicit RNG and indirect/image APIs. |
| 7 | Windows, conditional passes, local pieces, dynamic snapshots and independent present retirement. |
| 8 | Device ownership, reset generations and external allocation/stream integration. |
| 9 | Metal on the shared runtime; native ICB/control/image qualification. |
| 10 | CUDA/HIP on the shared runtime; modern launch/library/control and transfer/communication qualification. |
| 11 | Actual consumer migrations and numerical/memory/latency comparisons below. |
| 12 | Full applicable suites and a result-grounded State section. |

A joined path needs its relevant nonempty native checks before switchover. Phase
gates may cite one batch's fixture, benchmark and separate diff audit for several
phases; coverage must be explicit. Neither a phase boundary nor the word "green"
requests a full suite, application startup or new cold Julia session. Broader gates
stay open while independent implementation proceeds. Phase 0's authored/runtime
cases remain distinct from its documentation checks. Never disable true caps to
skip tests. Intentionally changed APIs need an old -> new expectation preserving
the invariant and negative control before implementation validation.

### Fixed MWE catalog

These are eight fixed fixture specifications, not existing passing tests. Implement
each once when its batch can exercise the final contract. Keep a single driver
and small shared assertion helpers; no second simulation/model/renderer framework.
Use deterministic integers or exact reference results where possible. Optional
capability profiles reuse the same fixture with a native counter/encoding assertion;
they do not multiply the base catalog or hide an unsupported native gate as a pass.
Graph-position assertions declare result = call!(g,Array,a) before their run,
then fetch(result) afterward. Array(a) is the ordinary blocking eager conversion;
it never selects a graph or captures unsubmitted nodes. Replays request Readback(previous_future) before
the next execution. Do not rebind an array merely to inspect it through an eager
Array conversion; the future identifies the intended graph position and execution.
The detailed ledger/native/consumer cases still qualify features beyond this smoke
set. Running a smoke fixture is not universal correctness evidence.

The profile names in the batch table identify prerequisite subsets, not new tests:
B1 cannot qualify GPU snapshots, loops or AS before B2/B3 supplies them. MWE-07's
transport/collective profile first runs in B4. A result records the exact subset;
only all implemented applicable profiles passing can close the whole fixture's
coverage on a target. One-lane devices can pass ordering but cannot establish native
queue overlap. Native fault interception must be observed, never silently skipped.

| ID | Small fixed input and reference result | Key failure detected / ledger |
|---|---|---|
| MWE-01 | 17 Int32 values 1:17 initialized once; three in-place +1 replays give 4:20, then grow to 33. Check the unchanged prefix 4:20 before any further update; initialize the new tail explicitly. Add a read-only checksum node to force plan replacement; the next +1 replay has prefix 5:21. Uninitialized bytes have no expected value. Pass the same leaf directly to operations in graph two; release graph one and retain contents. A helper may return a scratch handle/view for more declarations in the same graph; copying its data out requires a persistent destination or an owner-local recorded Array conversion. Cross-graph use or eager host reads of scratch reject. Fetch prior and zero/nonzero snapshots after later runs. A pending request survives compatible recompile; two snapshots around a write retain distinct positions before/after compilation. Owner-local scratch snapshots survive arena reuse; unrequested outputs copy zero payload bytes. Scratch-with-data initializes on every run and every repeated owner-region entry. | Implicit resets/lifetime, stale snapshots, early release; F81/94/96. |
| MWE-02 | Capacity 257, source integers 1:257, odd-element compact into logical length zero; test input lengths 0, 1, 255, 256, 257. Poison tails. Compare count/list/sum to host filter. Nest two outer iterations with at most three inner iterations, active counts 2 then 1, early stops and a condition whose source is changed inside its body; check reset/latching. An isolated invalid resize above capacity leaves prior dimensions unchanged and performs no out-of-bounds write. After compaction at 257 inputs, host-grow to capacity(active)+2 to force backing growth and verify the preserved 129-element odd prefix (the host's earlier requested length was zero); initialize the new tail explicitly. | Size publication/prepare, empty identities, partial-group collectives, bounded producers and lexical controls; F78/85. |
| MWE-03 | Two readers of a 17-element scratch root on separate logical lanes, delay the earlier-listed reader, then alias/reuse. Compare reader checksums. FaultDevice injects a known-not-enqueued later submission after one accepted writer; inspect accepted tokens/holds/AS dirtiness. A short host wait times out on a controlled unresolved dependency, then completes normally. | Missing DAG/frontier edges, lost accepted prefix, timeout poisoning; F76/77/79/87/95. A private missing-edge variant must fail the reference check; it is never a production option. |
| MWE-04 | Two deterministic test-only eligible candidates for one operation, threshold 32 elements, different command/workspace counts; grow 17 -> 65, preserve output identity and compare with fresh declaration. Recorded call!(group,broadcast!,...) -> call!(group,mul!,...) -> call!(group,sum!,identity,reduction,C) with exact small integer-valued Float32 reference result. Assert the mul! definition returns the identical destination, its KA backend stays the owning Device, and the same operands can be used by a sibling Group/second graph without adaptation. An unqualified operation is eager, not secretly grouped. Reset a qualified counter RNG and compare replay/conditional consumption with its reference stream. | Frozen candidate lists, target loss, argument boxing/state reset; F80/84/86/91. Test candidates implement the shared capability/selection protocol for eager and recorded calls, not a second selector. Unsupported recorded functions reject before any eager execution; an ! suffix never determines result handling. A test-only known definition with an ! name and a different declared return checks this, without guessing aliases or forcing every result into a future. The B4 library profile uses the same numeric input with an eligible real library candidate and observed native encoding. |
| MWE-05 | One triangle, one ray: deform its depth from 1 to 2, then trace in the same graph; compare hit distance with analytic geometry. Build -> later deform -> separate trace must also report 2. Inject failure before the required build; next trace still updates. Empty/masked instance case checks legal capacity padding. | Derived AS versions/build ordering/conditional validity; F77/85. Native and portable paths qualify separately. |
| MWE-06 | 16x16 depth/color image, one occluder and two indirect commands. Render, remove occluder with dirty flag, then skip a clean update. Exact selected pixels/depth prove reset versus preservation. GPU command producer publishes 0, 1, 2 commands with distinct mesh offsets. Write/read one mip/layer view using Julia one-based indices; neighbors stay poisoned. A depth view infers its depth aspect; window-sized images retain explicit lifetime. | Conditional clear, typed command bounds, image subresource hazards; F83/88/90. Headless offscreen smoke; actual WSI/present keeps its existing fixture with an explicit recorded window Array output, skipped-run request failure, prior-frame retention and resize-generation checks. |
| MWE-07 | Import one externally owned 17-element allocation with two aliasing wrappers and an explicit stream handoff; write through one, consume through the other, drop creator while pending, then verify ordered values and delayed release. When transfer/communication is enabled, exchange 17 Int32 values between two device graphs; two-participant sum has exact host reference result and injected participant failure must not publish success. | Hidden copies/alias roots/owner release, transport and completion; F82/93. Profiles require actual external API/devices; absence leaves their gate open. Different drivers for one physical GPU cannot be treated as a same-domain zero-copy alias. Also import independently shaped/reinterpreted headers of one allocation: overlapping byte writes order, while size metadata/typed capacity remain independent; ordinary aliases of one logical header still observe its resize. F123. |
| MWE-08 | Same typed kernel/argument layout with 100 and 1 000 command nodes and fresh resource identities; the 10 000-node scale is B5 qualification only. Record cold/warm compilation, method-instance growth, edges and peak host memory. Use one shared 17-element root for the straight-line chain; its user-kernel hazard chain has N-1 edges (internal preparation edges are counted separately). Replay the unchanged 100-node baked graph 1 000 times without per-node host work. One loop with maximum 1 024 stops after one iteration; report native/expanded plan bytes and GPU work. | Graph-size/value-driven specialization, quadratic frontiers, recompilation/boxing and fallback cost; F76/85/97. Compare baseline completed work and A/A noise; do not assert a universal wall-time threshold. |

MWE-01/02/03 cover simulation/particle state and solver control; MWE-04 covers
DNN/LLM-style composition; MWE-05/06 cover ray tracing and raster/Makie; MWE-07
covers foreign arrays and multi-device ownership; MWE-08 covers representation
cost across all workloads. Small fixtures do not replace actual FEM, particle,
DNN, LLM, training, ray-tracing or Makie consumer gates below.

### Validation schedule and Julia cost

1. Between edits: static dependency/ownership review and targeted parse/schema
   checks. Do not load large consumer packages or run JET/Aqua/full suites per file.
2. At a coherent batch boundary: run only its implemented MWEs plus existing tests
   directly affected by that batch. Keep a persistent Julia worker per environment,
   use Revise where valid, reuse compiled kernels/fixtures and collect background
   run IDs/logs before launching another job. Changes invalidating native/module
   state require a fresh worker; correctness takes precedence over cache reuse.
3. Before a path joins/swaps over: run its capability-specific native/fault tests,
   the relevant MWE profiles and representative completed-work benchmark against
   the frozen baseline. Use separate workers/load paths for baseline/replacement.
   Steady-state measurements reuse warm sessions; cold-start/precompile comparisons
   use fresh processes only for that measurement. Interleave and record A/A noise.
   One separate diff audit covers the batch's ledger rows; no extra agent is required.
4. At B5: run each required actual consumer, then full applicable suites once per
   joined hardware/API qualification target. Broad stress tests are scheduled here. A new native capability's required
   legality/failure tests must pass before that capability is enabled, even earlier;
   reuse their recorded results here rather than run them twice.
   After a fix rerun the failing/affected subset; repeat a full suite only if shared
   changes invalidate its previous coverage. Do not restart an hours-long suite
   just because another phase completed or context was compacted.

Tests record correctness, compile/host-submit/GPU-completion time, allocations,
resident/workspace/plan bytes and native recording/submit counters separately where
relevant. A fast enqueue is not fast completed work. Keep numerical references and
measurement inputs fixed before comparing implementations. Existing large benchmark
sizes remain qualification cases, not extra smoke tests on every batch.

### Failure handling and stopping rules

A fixture failure first means debug the implementation against the frozen reference result.
Classify it as fixture/API, semantics/ordering/lifetime, native legality or measured
performance. Inspect the existing log and reduce the failing fixture's inputs;
reuse the same ID/reference result. Fix connected code, then rerun the affected case. Do not
invent a new abstraction, alternative backend policy, new fixture family or weaker
assertion in response to a red result. An added focused regression must name the
proven defect; it does not become a new mandatory global validation cycle.

A design revision needs a reproducible contradiction between specified requirements
or an independently established native limitation after the debug-first procedure.
Record the evidence, narrow proposed change, affected contracts/ledger/fixtures and
alternatives first; reconcile the documents before implementing that change. A code
pattern or single failed attempt is insufficient. Preserve the original failed case
and expected invariant. If a decision needs the user, leave only that slice blocked
with a concrete reason and continue independent authorized work.

Every retry must test a new hypothesis or changed implementation. With no new
evidence, stop that experiment, record the precise unresolved condition and work
on an independent slice; repeated identical runs are not progress. Mark a batch
verified only when its named assertions and affected regressions pass; unavailable
hardware or unimplemented native features stay open. Finish planning when target
documents, fixed fixture reference results, batch dependencies and failure rules agree;
implementation qualification is future execution of that plan, not a reason to
recursively redesign it or start the whole runtime refactor during a planning task.

### Portable ray functions

execution-contracts §7 and api.md define the RT contract: one logical hit-function
table, per-geometry/instance selectors, streaming procedural reports and typed
attributes. Core owns filtering, accepted-result tracking, table packing/native
binding views and capability/cost selection. Metal/Vulkan encode chosen operations;
Hikari has no backend branch. Query-only entries do not require shading handlers
and retain the current wavefront/cut-set design. TLAS geometry is independent of
ray-type record layouts. No fixed zero geometry stride, single-root return or
universal two-float attribute assumption survives this target revision.

execution-contracts §16.5 defines AS memory growth/maintenance: full-signature
native sizing, operation-local scratch, parent repair after child relocation,
versioned in-flight retirement and explicit compact!(as). Shrink does not implicitly
compact; pressure may discard safe reconstructible bindings but never treats opaque
AS bytes as an array HostCopy. These costs share core budgeting/accounting.

The existing MWE-05 profiles cover two geometries × two ray types, typed attributes
larger than barycentrics, rejection of a near procedural root with acceptance of
its far root, reversed report order, and nonuniformly scaled object rays. Check
query and programmable execution against the same reference on actual Metal and
Vulkan, plus software where eligible. Miss/invalid IDs/attribute access and resource
reads confined to a filter/intersection must be checked. Native table/binding
bytes, selector rewrite cost, steady-state reuse and opaque fast-path latency are
reported; no additional fixture family or per-edit full suite is introduced.
The RT regression acceptance rules in execution-contracts §16.11 gate F124–F126:
no table/attribute/native-view overhead for consumers that do not request it,
retain eligible opaque native constructions and compact wavefront records, and
compare warmed matched workloads against the refreshed pushed baseline. A
repeatable regression outside measured A/A noise blocks the affected profile;
API flexibility and passing scalar models are not performance evidence.

## Use-case qualification

| Use case | Actual consumer gate |
|---|---|
| DNNs | DNNKernels/JuliaVision models, precision/quantized layouts/views, repeated broadcast+matmul+reduction, candidate growth/workspace. |
| Ray tracing | Current Hikari query/wavefront typed material queues, alpha/media/lights, empty/odd/even bounces, deform -> trace and partial-prefix dirtiness; future programmable hit groups have a separate capability profile, not a deleted consumer path to restore. |
| FEM | PrintSim assembly or matrix-free operator, independently checked solve, preconditioned Krylov/Newton/line search nesting, FP64 eligibility/convergence. |
| Particles | AScanner/PrintSim retained birth/death/contact timesteps, compaction/sparse conversion/SpMV, zero/partial groups and GPU sizes. |
| Raster/Makie | Picking/FXAA/atlas/text/volume, integer/depth/stencil/MSAA, moved/removed shadow occluders, pieces and GPU indirect commands. |
| LLMs | Bonsai packed KV/scales, prefill/decode/sampling, two sessions sharing weights, paged access, native-vs-expanded plan cost. |
| Multi-device | Per-device graphs with explicit transfers/collectives, participation/failure protocol; synchronous-copy evidence alone is insufficient. |
| Training/AD | Representative backward/optimizer alias/state/RNG gate; AD algorithms stay above Mantle and inference is not training evidence. |

The source-grounded [use-case review](../experiments/review-2026-10-29/use-case-review.txt)
tracks remaining contract/migration gaps as F115–F122. In particular, raw kernels
being expressive is not proof of a clean common scalar/sparse/operator interface;
contiguous cache growth is not paged-attention evidence, and inference is not AD
integration. Resolve each named obligation before closing its consumer gate. This
does not add another API, backend policy, MWE family or per-phase full suite.

Retain editor decode/DPB/GL fences, Paperhawk and all previously indexed gates.
GPUArrays tests alone do not establish consumer coverage. Modern features require
enabled facts, compiler support, actual native execution and measured benefit.

## Forbidden moves

No backend-specific consumer workaround, second policy owner, dummy required verb,
runtime feature probing, untracked memory, hidden library allocation or in-flight
native-graph/descriptor mutation. Do not weaken numerical or ordering semantics
to make a phase green. Debug failed native experiments against specifications,
bindings and an independent reference before proposing a fallback; §12/F74's
direct-restore evidence threshold remains in force.

### Debug before changing the design

A failed native probe is a debugging task. Minimize it, inspect validation/native
arguments, check the specification and bindings, and compare with a working
independent reference. Investigate alignment, feature enablement, memory classes,
visibility, ordering and lifetime before attributing failure to the platform.
Fix the implementation and rerun the original case.

Only an independently established limitation surviving that investigation can
support a design-change proposal. Record exact source/destination allocations,
root cause, target, reproducer, negative controls, alternatives/fixes tried and
the required consumer workload. A staged restore additionally needs overwhelming
evidence that direct-path fixes and supported allocation choices cannot meet it,
with correctness, memory and performance measurements. One extension flag, failed
import, uninvestigated error or historical report is insufficient. Never weaken
a contract or add a second algorithm just to make a broken probe pass.

## Review reconciliation

The 2026-10-29 findings/dispositions are in the ledger and
experiments/review-2026-10-29/plan-reconciliation.json. Corrected text and open
implementation/native qualification are separate. These docs implement no runtime.

## Out of scope

Codec conformance/parsers and AD/solver/model algorithms belong to their packages.
Their GPU resource ownership, execution composition and qualification are in scope.
