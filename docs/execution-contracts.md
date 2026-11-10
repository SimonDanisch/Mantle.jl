# Execution, failure and retirement contracts

This is normative companion material to recording-plan.md and api.md §§4–5.
It fixes the submission boundary, the proof obligations behind the lock model,
and the native facts a backend must supply. api.md owns public signatures and
indexes the native boundary; this document supplies the linked native signature
extensions and detailed execution protocols. The pseudocode must agree
with both, including failure edges, even where helper definitions are omitted.
The ledger records implementation/qualification status; a contract here is a
target requirement, not evidence that the replacement already implements it.
Host-only models in `test/design/` check the abstract rules and the actual fault
wrapper. They do not validate a driver, GPU visibility or scheduling performance.

## 1. Synchronization and lifetime argument

An access retains canonical root allocation identity, with a proven byte span or
image subresource when known and whole-root overlap otherwise, including a view, a cell,
argument memory, indirect commands, build scratch, descriptors and composite
members. The argument depends on sound resource discovery and access inference;
unknown code is conservatively ReadWrite, never silently omitted. Metadata and
descriptor-slot epochs are separate uses. One frontier/hazard algorithm derives
internal edges, alias reuse, native barriers and external conflicts (§16). Atomic-only
Unordered uses may overlap each other; they still conflict with ordinary reads
and writes. Queue execution order alone does not imply memory visibility: core
also derives the required native memory dependencies and ownership transitions.

The unit that becomes visible to competing host submitters is a **submitted
prefix**, not an entire successful run. Every accepted segment must publish its
token and reader-update requests before releasing its resource and plan locks.

| Invariant | Linearization / argument | Adversarial gate |
|---|---|---|
| Conflicting accesses are ordered | The submitter locks naming plans, root resources and transitive readers in id order. It validates their immutable membership snapshots, computes waits, submits, and publishes before unlocking. The next conflicting submitter either preceded this critical section or sees its token. A new namer must lock and validate the same records. | THR/AP areas; host model's reader publication and snapshot negative controls. |
| Channel values reflect actual issue order | All aliases of one physical queue share one ChannelState: lock, next value, native timeline and device failure domain. Claim and native enqueue are serialized. Token equality/hash use this state's identity, never a wrapper's identity. | Host wrapper tests and channel model; native validation under two host tasks. |
| A next run follows all of the previous run | Every root segment in stage 1 waits on the previous lastrun snapshot, including roots on other queues. Recompile transfers that snapshot before unregistering the old plan. Update lastrun after each accepted segment, including partial runs. | recompile/run areas; partial-prefix model. |
| Changes are applied exactly once | Prepare from a versioned immutable snapshot; allocate before resource locks; validate versions and capacity after locking. Missing storage returns Again before claim. Remove only the submitted prefix of pending requests after acceptance. Later requests remain pending. | applying/stores areas; host snapshot model. |
| Consumers cannot miss reader updates | A segment writing geometry publishes pending BLAS/TLAS updates before releasing expanded locks, even if a later segment throws. Conservative updates after a partial run are allowed. | partial-prefix model; phase-3 fault and geometry race tests. |
| Replacement cannot release live storage | A hold is acquired before publication; zero holds cannot become nonzero. Last-hold release makes the resource Released under its expanded locks and captures eager tokens, all naming plans' tokens and retired-plan tokens. Replacement commits retire old storage against the accepted copy/build token, which already waits for earlier uses. | holds/eviction/mutation areas; lifetime model. |
| Plan unregister does not erase uses | Before removing a weak namer, copy its latest tokens to each resource's retired-token set under naming-plan/resource locks. Prune passed tokens, retaining latest per ChannelState. | holds/recompile areas. |
| A host read sees one version | Direct unified reads wait outside locks, then validate an idle storage/shape/view/version snapshot and protect memcpy from submissions. Staged reads validate staging capacity against the applied state under locks, retain the copy's shape, and retire staging with its token. | READ-07; host capacity model. |
| Host failure preserves a submitted prefix | Cleanup distinguishes prepared-but-unused objects from accepted objects. Accepted objects and uploads retire against their actual tokens; no rollback or immediate free. The next run follows the retained prefix. | RUN-07/AP-05 recoverable preparation faults, RUN-18 partial acceptance; partial-prefix model. |

Deadlock argument: graph → arena holds by id → window → plans by id → resources
by id → ChannelState. Table/cache/pool bookkeeping locks are leaves. No GPU wait,
pressure, driver compilation or user callback occurs under plan/resource/channel
locks. An arena re-entry violating id order throws before waiting. Snapshot retry
must release the complete lock set before expanding it. This proves absence of
cycles for these locks, assuming callbacks obey the contract; it is not a
fairness or starvation proof. A changing workload can force retries.

Keep weak graph/plan references in resource namers and reader backreferences.
Plans strongly own their resources through holds; resources must not root their
plans. Removing weak references would make unreachable graph/resource cycles
retain GPU allocations until a different collector owns them. Finalizers enqueue
idempotent hold drops only; the release engine performs locked bookkeeping.

The existing ordered locks remain the target. A per-device bookkeeping lock is
a useful reference implementation for stress tests, but is not a second
production policy or a prerequisite benchmark project. It cannot be held across
GPU waits or callbacks either. Compare throughput/latency only once both run real
work; the host model establishes no contention or speed claim.

## 2. Submission is prepare, claim, accept, publish

`preparesubmit(ch, parts, waits, signals) -> packet` runs before claim. It can
allocate CPU/native submission descriptors and throw. Its output owns everything
needed by the native enqueue. It neither submits nor changes resource state.
Call it after recording and before the channel critical section; do not run
pressure from it. Device allocations must already have been acquired outside
resource locks. Packets are reused when the recording and wait capacity allow;
steady-state allocation gates include this step.

Under ChannelState.lock: check device health, claim the next value, record
provisional accesses, and call `submitnative!(ch, value, packet)`. Return means
accepted, not completed. Publish eager notifications, plan last-token updates,
change commits and transitive reader updates before releasing enclosing record
locks. Publication storage must have been reserved before claim; no Julia
allocation, arbitrary callback or additional fallible driver operation is
permitted there. Bookkeeping faults after acceptance are terminal too.

A native enqueue can fail: Vulkan explicitly allows allocation failures, and
asynchronous errors can be reported by CUDA/HIP calls. A no-fail driver promise
is impossible. Any native enqueue exception after claim makes the device's
failure domain terminal; subsequent submit/wait/read operations throw the stored
failure without waiting for an unsignalled value. `passed` never lies by treating
failed tokens as complete. The release engine first checks the shared failure domain. When poisoned it
quarantines every native object/region from every Runtime sharing that domain,
including retire!(..., ()) from a finally block that never received its token.
It bypasses normal token polling and recycling until native device teardown
establishes safety. release! of unused prepared storage follows the same rule. Recoverable injected OOM belongs before claim, not here.
A poisoned device is replaced as an entire native context, never by resetting
just the counter. This also handles a driver accepting work before reporting an
error. Other independent devices remain usable.

Presentation is a separate operation: rendering is accepted and committed
first. Only then, outside plan/resource locks, does `presentnative!(ch, request)`
enqueue presentation under the physical queue's serialization lock. WSI calls
may block in the driver; they cannot satisfy a universal nonblocking promise.
No other native queue operation may race this call, including through a wrapper.
The graph lock serializes its window. No presentation result can undo rendering
or make its token disappear. The native present call is deferred until every GPU stage of the run has been
accepted and published. The per-image PresentNode only records the final image
layout/ownership transition. Later segments/host stages may finish before WSI;
none may touch that transitioned image. This keeps presentation retry from
skipping work after the rendering segment and needs no saved run continuation.

Swapchain creation is also a transaction: native creation using oldSwapchain
can retire it even if creation fails. A native creation exception records
`oldretired::Bool`; a fault injected before the call has false. Core preserves
the pending extent, clears a retired current generation, and retains its GPU/
present completion records before retrying. It never acquires from a retired
old generation after an unsuccessful replacement.

`presentnative!` returns `PresentOutcome{R}(status::PresentStatus,
enqueued::Bool, retirement::R)`. PresentStatus values are PresentSuccess,
PresentSuboptimal, PresentOutOfDate, PresentSurfaceLost, PresentHostOOM and
PresentDeviceOOM. Device loss/ambiguous enqueue throws and poisons the domain;
it is never an ordinary retry status. A not-enqueued OOM result produces the
core exception PresentationRetry(status) after preserving the rendered request.
A borrowed FaultDevice may inject that result without calling native present,
because it claims no native enqueue or completion.


- Success/Suboptimal: enqueued; retain the completion record, mark resize on
  Suboptimal, consume the acquired image.
- OutOfDate/SurfaceLost: native rules determine enqueued; Vulkan still enqueues
  the semaphore wait. Consume the image, retain its completion record, mark
  recreate/closed. SurfaceLost is an actionable window error after cleanup.
- Host/device OOM before enqueue: not enqueued, keep the already-rendered image
  and its signalled render semaphore for a presentation retry. Do not render or
  signal that binary semaphore a second time. The render token stays valid.
- Device failure or ambiguous enqueue: poison the failure domain and quarantine
  the request. Never keep such an image for normal reuse.

The next run! first retries a kept rendered request, before resize, recompile
or applying stores. Successful retry returns Submitted without new GPU work:
all stages of that logical frame were already accepted. New edits stay pending
for the following fresh frame. A retry does not increment sample counters or
replay the frame. free! can abandon the request using acquired-image release
and token-safe semaphore retirement, never a fake present completion.

If a later GPU preparation/host stage fails before WSI was attempted, the run
is a partial-prefix failure, not a presentation-only retry. Retire the rendered
semaphore with its actual GPU token, replace it before another signal, and keep
or release the acquired image under §4. Do not mark that incomplete run as a
fully rendered pending frame. No rendering semaphore may be signalled twice
without a consumed wait or replacement. WIN-30 checks the complete-frame OOM
retry and call counts. A retry of presentation itself does not re-run kernels. A failed host node also leaves
all previous accepted stages tracked. Reader updates must therefore be published
per accepted segment rather than once at the successful end of a stage.

## 3. Native boundary: factories, recording and destruction

The following completes api.md §4's vocabulary. Types are internal, not user
choices. A capability-gated family needs methods only when its path is enabled.
Every native object has one owner and one destruction method; policy stays in
core. Optional native acceleration cannot require users to pick a backend.

| Exact verb / fact | Input/output and ownership |
|---|---|
| `openchannels(dev, kinds)` | Create native queue endpoints and canonical ChannelStates. Core chooses kinds from queue facts. Multiple kinds returning the same physical queue return the same state. Construction is fallible initialization. |
| `channelstate(ch)` | Canonical core synchronization state; aliases and test wrappers return the identical object. |
| `preparesubmit(ch, parts, waits, signals)` | Prepared packet above; bindings, graph-exec parameter updates and descriptor arrays prepared before claim. Native graph updates must target an idle or new executable; never mutate an in-flight executable. |
| `submitnative!(ch, value, packet)` | Sole native work-enqueue site. No present call, compile, pool acquisition or host wait. |
| `recycle!(ch, objects)` | After tokens pass, return command buffers/allocators, graph executables and packets to that endpoint's free lists; do not destroy objects still held by a recording. CUDA/HIP graph instantiation belongs to finish/prepare, never enqueue. |
| `destroy!(dev, native)` | Destroy one already-retired native object by dispatch; includes images, reserves, pipelines, bindings, native graph executables, windows/surfaces, swapchains and decoder sessions. Never walk a Mantle graph or free composite members. |
| `objectbytes(dev, native) -> Int` | Explicitly known driver-object bytes, or zero when the API supplies no size; report unknown counts separately. Zero does not claim the native allocation is free. Pool accounting distinguishes placed storage from opaque objects. |
| `placeimage(dev, region, spec)` | Native image and view bound to region. spec includes format, extent, mip/layer/sample counts, usage and sharing families. Query `imagerequirements(dev, spec)` before acquiring a region. |
| `imagerequirements(dev, spec)` | Bytes, alignment, compatible memory types and required/preferred dedicated allocation. Required dedicated allocations are pool-owned blocks with one tenant; core cannot prohibit driver-mandated allocations. |
| `openwindow(dev, width, height, options)` | Native window + surface/layer, main-thread native work. Core owns Window and schedules main-thread creation/destruction. |
| `surfacecaps(w)` | Formats/colour spaces, present modes, extent limits, image counts, usage bits, present families, retirement mode. These are surface-specific facts, not just device flags. |
| `presentnative!(ch, request)` | Sole present enqueue; structured result above, prepared semaphores/fence/drawable, serialized against all native operations on this queue. |
| `presentpassed(record)` | Nonblocking retirement proof; neither a rendering-token query nor a display-timing promise. |
| `releaseacquired!(sc, indices)` | Maintenance1 release, or native drawable release; only after any acquire signal is consumed and image GPU uses have completed. Absent native support, keep acquired images with their generation until its legal destruction. |
| `foreignchannel(dev, stream)` | Endpoint for the explicit foreign stream. Same native stream returns the same ChannelState. No private counter for an already wrapped stream. Core owns its lease/retirement. |
| `streamwaits!(ch, tokens)` / `signal!(ch, value)` | Encode foreign-stream waits before the callback and completion after it, using the same channel serialization and terminal-error rules. No Mantle lock is held while user/vendor code runs. |
| `vendorview(dev, region, T, dims, strides)` | Borrowed native array view; owns no storage. Core validates device identity, acquires the resource lease and delays its hold drop until the foreign completion token. |
| `videorequirements(dev, spec)` | Supported profile/format, DPB/reorder limits, native session memory and bitstream alignment. Unsupported profile selects an eligible CPU decode path outside a GPU recording, not a late failure in emit. |
| `createdecoder(dev, spec, regions)` / `emitdecode!(b, node, cmd, deps)` | Native session bound to core-owned regions, and one already-parsed decode command. Session/DPB slots are resources; parser, seek policy and codec conformance stay in the video package. |
| `foreignallocation(dev, foreign)` | Retained allocation identity/context/generation, bound extent, strides and owner hold; core canonicalizes aliases and validates readiness/leases. Never adopt a foreign allocator as another Mantle pool. |
| `recordlibrary(dev, recipe, bindings, workspace, stream)` | Qualified closed native library recording; core supplied explicit effects/control bounds, private stream lease and pool workspace. Capture is local to this recipe, never consumer code or the surrounding Mantle plan. |
| `recordregion(dev, spec, emit)` | Native region builder setup/finish around core's emission callback. Core walks the child command DAG; the backend never walks Mantle nodes or chooses expansion. Returns a held native region recording. |
| `compileexecutionset(dev, spec)` | Native program/execution-set object from core's frozen compatible programs/layout/resource set; core owns selection, cache publication and retirement. |
| `emitregion!(b, node, cmd, deps)`, `emitgenerated!(b, node, cmd, deps)` | Execute an already-recorded native region or bounded generated-command stream from frozen handles/buffers/sets and complete edges. No host walk of a Mantle command tree. |
| `transfercaps(dev, peer, spec)`, `emittransfer!(b, node, cmd, deps)` | Native bridge facts/encoding for an explicitly selected transfer; core owns source snapshot, destination publication, holds and any host-cut bridge. |
| `communicatorfacts(dev, native)`, `recordcommunication(dev, recipe, bindings)` | Native participant/library facts and one eligible closed communication recording; core validates participation, effects, ownership and failure protocol. |

`Builder`, piece builders and finish may allocate host/native command objects
from prepared capacity, but no device storage. Core must prepare all encoder,
recording, prefix and submission capacities before claim. Returning Again is
allowed only before claim. Commands and `deps` must be immutable during record;
resource-read probes check that record never consults live storage/shape fields.

Queue facts include native family/index and supported command kinds. Memory
barriers derived by core carry producer/consumer stages, access classes, layouts
and source/destination families. Different Vulkan families require release and
acquire barriers plus a semaphore edge, unless the resource was created with
concurrent sharing for exactly those families. Waits alone are insufficient.

### Buffer mapping capabilities

`caps(dev).sparseresidency` means **queue-ordered buffer page mapping**, not
merely native virtual-address support. The native facts also include
`hostvmm::Bool`, page granularity, mapping mode (`QueueMapping` or
`CommittedMapping`) and enabled native features. Vulkan sparse queues and an
actually supported Metal sparse-buffer path may supply QueueMapping. CUDA
`cuMemMap/cuMemSetAccess` and HIP `hipMemMap/hipMemSetAccess` have no stream
argument: host VMM alone does not enable QueueMapping. Initially use committed
pool/arena parts for them. VMM may prepare a new unused committed part; never
unmap or remap addresses reachable by in-flight commands. A later optimized
host-VMM protocol needs its own idle-range reservation proof before enabling
sparse arenas. Do not manufacture a token to turn a host mapping into a queue
operation. This retains functionality and removes an unimplementable fast-path
promise. See [CUDA VMM](https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__VA.html)
and [HIP VMM](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/memory_management/virtual_memory.html).

### Execution features and bounded submissions

Kernels receive prepared plan slots on all backends. Source Cells remain internal;
Base.length/size read a dispatch descriptor, and inferred device resize! publishes
size (§16). Normal consumer code does not receive another metadata handle.

Retain nested loop/condition regions until core selects an eligible native region
encoding or the portable bounded expansion. Native facts describe legal node kinds,
nesting, update rules, geometry and budget behavior; API identity alone does not
select a policy. Native executables are immutable in flight. Core owns the fallback,
including control reset, stop latching and explicit command/argument slots.

Without indirect launch, capacity predication gates whole inactive workgroups.
Partial groups preserve collective participation with neutral lanes; no arbitrary
early exit before a barrier. Bounded expansion connects all body exits before the
next head, even when a kernel ignores iteration. Nested child controls reset on
each enclosing invocation. A when!-region samples/latches its condition at entry;
body writes to that source cannot retroactively change the current region predicate. Internal metadata prepares share the region's control.
Expanded size is maximum iterations times body size; native-loop alternatives
must be selected before this information is erased. Compile/plan/termination cost
is a required measurement, not a universal constant-size promise.

Multi-command heads, changes, build preparations and library recipes are lowered
to core command IDs and explicit internal predecessor edges BEFORE recording.
An empty node-predecessor list plus a memory barrier does not create native DAG
order. DAG backends translate these edges; linear backends encode the corresponding
barriers. Cross-recording dependencies become submission edges/waits.

A submission budget and calibrated roofline/IR costs are estimates. Partition only
at legal operation/iteration boundaries or with the caller's split=true workgroup
contract. Reduction partitions require a combine, and cooperative/cluster work
must retain its declared geometry/residency constraints. LaunchTooLong requires
a validated unsatisfied execution constraint or independently justified bound;
a heuristic estimate alone cannot prove that a launch is impossible.

waitfor(Token;timeout=Inf) separates a host deadline from watchdog facts. A finite
deadline expiring throws WaitTimeout, leaves work/holds pending and does not poison
the device. A semaphore wait can include an arbitrarily long chain of otherwise
legal submissions on other queues. Actual native device loss still makes the
failure domain terminal and quarantines objects. Diagnostics retain recent
submissions, estimated costs and their provenance. No arbitrary-kernel watchdog
guarantee follows. See the native distinction between
[VK_TIMEOUT and device loss](https://docs.vulkan.org/refpages/latest/refpages/source/vkWaitSemaphores.html).

## 4. Presentation retirement

A generation owns images, per-image rendered semaphores, native image views and
its completion records. Plan recordings referencing its images have separate
GPU last-use tokens. It is destroyable only after its window/run/kept-image
holds are gone, all GPU-use tokens passed, all acquire signals were consumed,
and all presentation-retirement records are ready. A completion record owns its
native fence/drawable and a generation hold until then. Never put these objects
on a rendering-token-only release list.

Vulkan preferred mode uses KHR or EXT swapchain maintenance1 present fences.
A fence proves payload release, not that the image was displayed on screen.
OutOfDate/SurfaceLost that enqueue the semaphore wait still have a retirement
record. Poll fences without record locks. Reset/recycle a fence only after it
signalled and its record was retired. See the
[present fence contract](https://docs.vulkan.org/refpages/latest/refpages/source/VkSwapchainPresentFenceInfoKHR.html)
and [present result rules](https://docs.vulkan.org/refpages/latest/refpages/source/vkQueuePresentKHR.html).

The legacy fallback uses one render semaphore per swapchain image. Associate
its previous present with the next acquisition of that same image; **completion
of the acquire wait**, not the host return of acquire, proves reuse. On resize,
chain older retired generations to the first present in the successor whose
retirement is proved by acquisition. Consecutive resizes splice the chains;
minimized windows keep them pending. This is the
[Khronos recreation fallback](https://docs.vulkan.org/samples/latest/samples/api/swapchain_recreation/README.html).
Coalesce pending resize events before creating a generation. Rapid successful
recreations can still accumulate resources until acquisition progresses; report
this separately from a pool leak and do not free them early under pressure.

There is an unavoidable qualification limit: unextended Vulkan supplies no
specification-level proof for final shutdown without another acquisition.
The practical legacy shutdown uses queue/device idle, as described by the
[Khronos guide](https://docs.vulkan.org/guide/latest/swapchain_semaphore_reuse.html),
but that is not a formal presentation-completion guarantee. Keep this fallback
for platform coverage; strict lifetime qualification requires maintenance1.
The design must not claim both unconditional legacy support and a fully proved
shutdown. Adding a CPU display bridge is separate engineering work, not an
invented native guarantee. No device-wide wait is introduced for normal resize.

Metal 4 orders drawable acquisition with waitForDrawable, submits all image
commands, then signalDrawable before present. Retain the drawable until GPU
uses and the native presentation lifecycle release it; completion callbacks only
enqueue retirement notifications. A timeline event covering rendering does not
replace drawable ownership. See
[Metal 4 drawable signalling](https://developer.apple.com/documentation/metal/mtl4commandqueue/signaldrawable(_:)).

Core selects surface format/colour space and usage before pipeline compilation:
linear shader output, sRGB presentation by default, FIFO/vsync on by default.
Prefer an sRGB attachment with native conversion. If storage writes or readback
usage are absent, use a pool-owned intermediate image and a supported final
render/copy conversion. Never request unsupported swapchain usages. Other colour
spaces require an explicit conversion path, not relabelling bytes. Minimized
windows submit nothing; screenshot waiters wake on skip/close with the documented
error. One presenting graph per Window, many independent windows per device.

## 5. FaultDevice is a native-boundary instrument

FaultDevice owns a fresh core Runtime (pool, counters, tables and channel views)
over the same underlying driver device. A wrapper channel stores its test owner
and inner endpoint and returns the inner endpoint's canonical ChannelState;
this preserves physical queue serialization and token numbering. Builder wrappers
intercept each emit/finish and forward to the inner native builder. No global
fault map, changes to a real backend, or fictitious independent queue timeline.
`device` and diagnostic/pool queries observe the wrapper Runtime. Native handles
created by inner methods remain attributed to that Runtime by core.

Recoverable failures are injected in compile, raw allocation, record or
preparesubmit, before claim. Native submit Throw is allowed only on an isolated
fake device/failure domain; poisoning a borrowed live queue would invalidate the
suite's base device. Block/count interception is allowed on borrowed endpoints.
Answer overrides cannot pretend an enqueue succeeded without submitting or
providing a fake completion source. A test scheduling a fault must assert its
call count increased and the fault actually fired; an untouched schedule is a
test failure. Wrapper tests run without the replacement API or a GPU.

The phase gates remain separate:

1. Host: actual fault-wrapper forwarding, shared ChannelState identity, fault
   hit counts, model interleavings with deliberately broken negative controls.
2. Foundation: nonempty phase cases on an implemented replacement device; lock
   probes, mutable-state reads during record, partial-prefix failure, staging
   growth, retirement/GC and descriptor reuse with native validation enabled.
3. Each native backend: capability-to-verb conformance, queue-family transfers,
   device failure on a disposable context, versioned graph execution, and
   allocation/latency/memory gates. CPU-only Lavapipe does not stand in for a
   discrete GPU, Metal, CUDA, HIP or a display.
4. Consumers: the concrete migrations and acceptance cases in
   `experiments/review-2026-10-20/consumer-coverage.md`, followed by real package
   correctness and performance runs against the frozen baseline.

A parser pass, a fake native endpoint or an abstract state proof cannot close
steps 2–4. Ledger rows distinguish a specified contract from implemented evidence.

## 6. Capability schema and qualification matrix

`caps(dev)` is an immutable named record with this required schema. Unknown
measurements are represented as estimates with provenance, not invented native
facts. The suite overrides named records by one shared helper; core does not keep a second copy of
KernelInterface's capability tables.

| Fields | Type / meaning |
|---|---|
| `api` | Native API identifier for diagnostics/fixtures (:vulkan/:metal/:cuda/:hip); production dispatch still uses endpoint/device types, never a Symbol switch. |
| `kernel` | Existing KI.DeviceCaps: matrix/subgroup shape and occupancy-budget facts. It does not contain general scalar or atomic support; those are separate fields below. Core supplies KI query projections (matrix_shapes, wggranularity, supports(::MatrixShape), bestshape), plus scalar/atomic predicates from the additional tables. |
| `graphics`, `raytracing`, `rayquery`, `mesh`, `geometry`, `tessellation`, `display`, `videodecode`, `videorecordable`, `descriptorindexing`, `accelbyaddress`, `ser`, `software` | Bool, enabled and usable native paths. Software traversal remains available without raytracing/rayquery. display additionally needs a usable window-system fixture. |
| `sparseresidency`, `hostvmm`, `mappingmode` | Bool, Bool, QueueMapping/CommittedMapping, with the meanings in §3. Native sparse textures alone do not imply sparse buffers. |
| `maxdispatch`, `maximage`, `maxlayers`, `maxinstances` | Positive integer limits: 3-tuples of dispatch counts/image extents, image layers, instance count. Workgroup dimensions/product/shared-memory limits project from kernel facts; per-format image/sample constraints come from imagerequirements, not a duplicate global table. |
| `sparsepage`, `sparseaddressspace`, `maxalloc` | Positive Int granularities/limits; non-sparse devices still report ordinary allocation granularity. Checked multiplication/rounding; reject overflow before allocation. |
| `inlinestore`, `inlinealign` | Nonnegative Int byte limit, positive Int alignment. Zero limit selects upload-copy. |
| `throughput`, `bandwidth`, `crossqueuecost` | Immutable type→operations/second record, positive bytes/second, nonnegative seconds. Add `costsource` = :measured/:driver/:estimate; zero throughput means ineligible, never division by zero. |
| `transfer`, `displayattached` | Bool: useful transfer queue, watchdog/display risk. display means usable WSI and is independent of displayattached; headless use does not turn off a physical watchdog. queuekinds/openchannels carry family/index and command kinds. |
| `jobtimeout` | Seconds or Inf when unknown; this is watchdog risk, not a host-wait deadline or proof of unlimited execution. Core budgets conservatively; calibrated estimates do not prove an execution lower bound (§3). |
| `execution` | Immutable enabled facts for indirect dispatch/trace/count, nested native control, generated-command stages/tokens/execution sets, device graph launch, cooperative/cluster limits and dynamic shared-memory constraints. Core selects constructions; no backend policy table duplicates KI facts. |
| `libraries`, `bridges` | Eligible closed-recording/library workspace/control facts and peer/external/host-bridge transport facts, including visible host-cut requirements. Unsupported paths do not acquire dummy verbs. |
| `devicekey`, `compilerkey`, `enabledfeatures` | Immutable native identity/architecture, compiler+ABI version key, sorted enabled feature tuple. Compilation requires equality of relevant keys; device type identity alone is insufficient. |

Core owns dimension and byte-span validation before allocation, placement or
wrapping: reject negative dimensions and checked-product overflow, and test
subranges without overflowing offset + length. Copies check logical array ranges
as well as backing byte spans before host access or enqueue. Native verbs retain their
API-specific alignment/usage requirements rather than copying this policy.

Scalar numeric support is `scalars::Tuple` of supported scalar types;
`atomicfeatures::Tuple` carries operation/type/scope/ordering combinations using
the compiler's existing atomic operation/scope/order vocabulary. Fences have
explicit scope/order support too. A Bool supports_atomics cannot stand in for
this table. Lowering must preserve requested semantics or fail before enqueue;
CUDA/HIP/Vulkan/Metal names alone do not prove device-scope acquire/release.
Only reorder hints may become no-ops. No silent demotion of the atomic orderings
Vulkan supports. Sequential consistency is the one exception: `Atomix.@atomic`, and
with it portable KernelAbstractions code, requests it by default, and Vulkan does not
support SPIR-V SequentiallyConsistent semantics. Lava lowers it to the strongest
ordering Vulkan has: acquire/release for RMW, CAS success and fences, acquire for
loads and CAS failure, release for stores. That does not provide the requested total
order, so a kernel relying on one across locations (Dekker-style mutual exclusion) is
not correct on Vulkan, and compilation cannot detect that. It stays omitted from
atomicfeatures. CAS success/failure orderings remain separate, and fences need actual
lowering. F33's reproduced Lava demotion of supported orderings is fixed: atomic
loads/stores/fences with a supported ordering retain it, and Atomix's requested order
reaches the emitter.
Device scope must retain Device scope across queue families and enable its
native feature, rather than being narrowed to QueueFamily. Historical evidence
and the repair's compiler/native checks are indexed in the review's
native-evidence.md (F33). The native qualification covers same-Device-scope
message passing and release sequences; it does not establish arbitrary availability/visibility chains with different
scopes or all scalar/atomic types.

The published record and KI's existing vector-backed shape tables are read-only
by contract. Copy input tables once at device initialization to avoid aliasing a
backend's mutable construction buffers; retain one published KI table, not a
second Mantle matrix table. No per-query deep copying. Native enabled features
and table contents participate in the stable compiler key. Mutating caps in
place is not supported; tests create replacement records.

`config(dev,f)` is immutable and includes compilerkey, enabledfeatures, argument
ABI version, chosen group dimensions, math mode and kernel-specific specialization
options. The CodeInstance/code identity remains part of cache keys. KernelInfo
marks estimated register/occupancy/IR cost facts as estimates when native queries
are unavailable; no fabricated precision. Candidate requirements read these same
facts. Native library plans must specify non-owning workspace regions, allowed
stream/capture modes and all read/write effects before selection; otherwise
select a portable core candidate. Library-internal allocation is not hidden in
"zero allocations" diagnostics.

| Path | Required native qualification, in addition to common suite |
|---|---|
| Vulkan desktop | Synchronization validation; aliased queue endpoint stress; cross-family releases/acquires; dynamic rendering + actual sampled texels; sparse buffer qualification only when enabled; both maintenance1 and legacy present retirement fixtures. |
| Metal 4 | Slot ABI and barriers, ICB/resource residency, supported sparse-buffer path independently of sparse textures, drawable wait/signal lifecycle, allocator reuse after event completion. |
| CUDA | Recorded kernel/copy DAG and graph-executable versioning; slot-pointer capacity/predication baseline; explicit-stream views; committed arenas first; optional library candidates with caller-owned workspaces. Native graphics remains false. |
| HIP | Same portable DAG/slot/committed-arena contract; do not infer CUDA conditional-node availability; qualify each actual HIP graph/library entry point. Native graphics remains false. |

This is a conformance checklist, not evidence that these backends have been
implemented or that all native accelerators have portable equivalents. Native
feature availability and package integration must be measured on each target.

`waitidle(dev)` is a core snapshot wait: capture outstanding tokens, release all
locks, wait on those tokens. It does not close submission admission and does not
prove presentation retirement. Device teardown first closes admission for the
entire failure domain and drains host enqueue operations, then calls native
`waitnativeidle!(ch)` outside all Mantle locks before destroying children and the
native context. A borrowed FaultDevice cannot tear down its inner domain. Failed
contexts use driver-specific teardown; never wait on a claimed unsignalled value.

`nativeowner(ch, record)` defaults to identity and lets wrappers retain ownership
on native completion records. It changes no queue state or production policy.


Mapped host memory also has native visibility requirements. `flushhost!(dev,
region, range)` makes CPU-written upload bytes visible before enqueue;
`invalidatehost!(dev, region, range)` precedes CPU reads after writer completion.
A coherent region supplies no-op methods; noncoherent methods round to native
atom-size requirements within the allocation. Region facts retain mapping,
coherence, alignment and host/device access flags. Direct unified memcpy cannot
skip invalidation merely because the CPU can address the memory.

For retained external-API bridges, `externalcaps(dev, spec, handlekind)` reports
memory/semaphore compatibility, dedicated constraints and import ownership.
`exportmemory(dev, region, handlekind)` / `importmemory(dev, descriptor)` and
`exportsemaphore(dev, native, handlekind)` / `importsemaphore(dev, descriptor)`
create explicitly owned native handles. Descriptors include size, offset, format,
layout, device identity and ownership-transfer rules; core validates them before
acquisition. An fd consumed by a native import is duplicated for each importer;
a Win32 handle follows its native close rules. Imported memory is never placed
in the ordinary relocatable arena. Cross-API ownership transitions and fence
wait/signal operations are represented as resource accesses; one common pool or
physical GPU does not supply those edges. Missing compatible native handles
select an explicit copy bridge or the Mantle Window path. No zero-copy promise
is made across arbitrary drivers/OSes. GL context switching remains the host
UI package's responsibility. These verbs are required only for an enabled
external-memory family, not every device backend.

Video capability facts distinguish `videodecode` from `videorecordable`.
An asynchronous decoder that cannot be recorded returns resource-owned frames
with ready tokens through the external-channel boundary; it must not masquerade
as emitdecode!. Native session callbacks publish readiness and DPB ownership
through the release engine. Graph-recordable video candidates require a real
native command encoding path. Codecs/parsers remain in the video package.

## 7. Shader arguments, hits and ray programs

Shader intrinsics live in KernelInterface, below Mantle and its compilers; Mantle
owns resources and pipeline descriptions. Do not add one Mantle→Lava bridge per
builtin. `vertex_index()` and `instance_index()` are one-based. Draw index buffers
are explicitly zero-based UInt32 wire indices; the vertex builtin adds one.
GeometryBasics face indices used by BLAS remain Julia one-based and core converts
native records. This is an ABI distinction, not permission for backend-dependent
shader indexing. `KI.rt_launch_id_x()` retains its existing zero-based contract;
new portable raygen code uses `KI.get_global_id()` for a one-based launch index.
Do not silently change the existing KI builtin while migrating its callers.

### Portable queries and hits

rayquery(tlas,origin,direction,args::Tuple=(); tmin=0f0,tmax=Inf32,
mask=0xff,accept=AcceptAll(),intersect=BoxIntersection(),hitgroups=nothing,
raytype=1,anyhit=false,attributes=Nothing,capture=nothing) returns RayHit{Q}, Q=attributes.
This is device code inside an ordinary kernel on Vulkan, Metal and software.
The default finds the closest accepted hit; anyhit=true requests any accepted
intersection and permits early termination. Both invoke no shading handler. A
bounded hit-collection/transparent-shadow consumer keeps its existing explicit
algorithm; it is not implemented by order-dependent side effects in a filter.
accept(hit,ray,args...) is a pure Bool predicate; intersect(report,candidate,args...)
is the procedural candidate generator. If hitgroups is supplied, its selected
intersection/filter functions replace the shorthand intersect; supplying both a
nondefault intersect and a table rejects. The global accept predicate additionally
filters table hits. No callable mutates shared resources, RNG or traversal state;
reporting candidates is the sole permitted traversal effect. Invocation count/order
are unspecified. Hikari may instead keep its existing explicit wavefront/query
algorithm; this interface does not force shading callbacks into its trace kernel.

origin + t*direction defines distance; never normalize a transformed direction.
Validate finite origins/directions, nonzero direction and legal t interval. Masks
are eight bits and an instance participates iff instance-mask & ray-mask != 0.
No default facing cull. A required filter must see every candidate that could win,
including triangles marked opaque in storage: core selects an eligible nonopaque/
forced-filter construction or declines the native candidate; opacity never skips
alpha tests. Surface-transparency accumulation/stochastic material sampling remains
an explicit consumer algorithm, not an order-dependent any-hit side effect.

The initial catalog uses Float32 rays: other precision requires a qualified
extension, never silent narrowing to native Float32. RayHit{Q} keeps hit::Bool,
distance::Float32, UInt32 instanceid/customindex/primitiveid/geometryid/kind,
barycentrics::NTuple{2,Float32} and frontface::Bool. Index fields are one-based on a hit; miss has
zero IDs/barycentrics, frontface=false and distance=tmax. Triangle barycentrics are
(u,v), with weights (1-u-v,u,v). Equal-distance ties may pick either equivalent
accepted primitive. customindex is the stable logical range slot; instanceid is
the logical instance, independent of native split/packing. rangeoffset retains
its within-range mapping. Native index conversions and actual width limits are
checked internally. Preserve logical geometry IDs if a native lowering splits a
logical structure; no scene-wide geometry concatenation is required.

hitattributes(hit) returns the hit's concrete Q and checks hit=true; reading
attributes on a miss signals an error rather than manufacturing zero(Q).
Nothing is the default query result attribute type and consumes no attribute
storage. Private optional attribute storage keeps validity separately from packed
isbits bytes, so miss construction does not need zero(Q), an invalid Q value,
a heap box or a scene-sized Union. Decode Q only after the validity check.
For attributes=Q other than Nothing, capture(hit,ray,args...) -> Q explicitly
projects each accepted candidate's typed attributes; omitting capture means use
its attributes unchanged and requires that type to be Q for every reachable hit.
A mixed table can normalize different group attributes through a typed capture,
including deriving a triangle result from barycentrics. The shared compiler
retains only the current accepted result Q; it never boxes heterogeneous payloads
or introduces a Union-sized return tied to scene geometry count. capture is pure
and may be called whenever the current closest result improves. Candidate and
returned attributes are separate contracts; rejecting a candidate never changes
the retained result. Scalar attribute types/layouts are checked at compile time. Without a table,
attributes=Q also declares the shorthand procedural reporter schema. With a table,
each group declares its own schema A independently of the query result Q. Triangles
provide barycentric attributes. Every filter/capture receives the immutable world
ray explicitly; closest_hit also receives that ray, not a hidden stage global.

### Logical hit functions and geometry selection

TriangleHitGroup(closest_hit=nothing;any_hit=nothing) and
ProceduralHitGroup(intersection,closest_hit=nothing;any_hit=nothing,attributes=Nothing)
are immutable typed function descriptions. Triangle candidate attributes are the
two barycentrics; a procedural group's attributes=A declares a concrete isbits
schema with no GPU resource handles/pointers. closest_hit=nothing is legal for
query-only tables. A program that invokes shading requires the selected handler.

HitFunctionTable(groups) is the common logical table, not a native handle.
A flat tuple of groups is the one-ray-type form; a tuple of equal-length row tuples
is group-major, ray-type-minor. Reject ragged/empty rows. Callable types represent
the finite program set; geometry/instance counts, IDs and selectors are runtime
values, never type parameters. The general program constructor is
RayTracingPipeline(;raygen,hitgroups::HitFunctionTable,miss::Tuple,
payload=Nothing,maxdepth=1), with one miss function per ray type and concrete
isbits payload P. The closest_hit/miss/any_hit convenience form uses the same explicit callback
signatures and normalizes to this table; both forms together reject. maxdepth bounds nested traceray calls.

TriangleGeometry(vertices,faces) and ProceduralGeometry(bounds) are typed source
descriptions. BLAS(target,geometries::AbstractVector;scratch=false) retains a logical
ordered geometry list; the existing single-geometry constructors normalize to it.
Core handles compatible native grouping/splitting once, preserving logical IDs,
or selects its shared traversal construction. Native restrictions cannot force
backend-specific scene representations. The geometry list is an indexed value
collection with typed emission barriers, not a tuple type encoding scene size.

push!(tlas,blas,transforms;hitgroup=1) and hitgroup!(tlas,range,selector) accept:
- One integer group ID, broadcast over all geometries/instances of the range.
- A host tuple/vector with one ID per logical BLAS geometry, shared by its instances.
- A device integer matrix with rows=geometry and columns=instance within the range.

The host selector is snapshotted; the device selector is a declared resource read
at execution. One-based IDs choose table rows; raytype chooses columns. Group kind,
shape/index bounds and active counts are validated before unsafe native indexing;
GPU-invalid selections record failure and cannot become arbitrary function pointers.
The finite table declares the complete allowed program/resource set. GPU-written
selectors participate in prepare/barrier/version dependencies, not host downloads.

TLAS has no public raytypes setting: logical scene data does not depend on a
pipeline's record stride. Native binding views may depend on a table layout/ray-type
count. Shared core caches/holds them by scene version and layout; geometry storage
is shared and their build/update/retirement costs are measured. Updating one view
cannot clear the dirty version of another view. A stable compatible binding reuses
its recording; switching layout cannot patch a live native record. These are
internal representations, not extra public TLAS/device families.

For example, an eligible Vulkan SBT layout uses a slab per instance/geometry-group
selection: record = instance_base + (geometryid-1)*R + (raytype-1). Record contents
select the logical group for that geometry; identical slabs can share storage.
The old fixed geometry stride zero is not the general layout. Check actual native
offset/stride/attribute/recursion limits, alignment and all byte arithmetic.
GPU-changing mappings use bounded prepared records or an eligible shared dispatch
construction, with memory/traffic costs exposed. Core owns packing/deduplication,
not a backend material manager. Metal intersection/visible-function constructions
and shared typed dispatch use the same logical selection; neither API's table
layout is the public contract. Native providers report enabled compiler/device
facts and create/encode the chosen objects. No attribute or program truncation.

### Streaming procedural intersections and callback ABI

The one-returned-intersection sketch is replaced by a streaming reporter:
intersection(report,candidate,args...) -> nothing. RayCandidate exposes logical
IDs plus worldray and objectray (origin,direction,tmin,tmax,mask), and the instance
transforms needed to convert geometry results. objectray already applies the
inverse instance transform exactly once; its direction is not normalized, so t
has the same meaning in both spaces, including nonuniform scale. Report distances
in that parameterization; report attributes use the group's declared coordinate
convention (not a silently transformed normal). Triangle/procedural attribute
normal transformations remain explicit shader math.

report(t,attributes) -> Bool offers one candidate; report(t) uses Nothing
attributes. The result means continue reporting: it is false only when an
anyhit=true query has accepted a hit and is finished, otherwise true. It is not
an acceptance/final-winner result. A callback can use report(t,a) || return nothing
to avoid unnecessary root work. Ignoring the continuation flag remains correct;
core ignores later reports after termination. Closest-hit reporting remains true
even after an accepted root, since a later root may still be closer.
A procedural function may call it repeatedly, in any order, with finite in-interval
roots and the declared attribute schema. Core validates each report, runs the
selected any_hit and global accept filters before commit, and retains the closest
accepted t/result, or the first accepted result for anyhit=true. A rejected near sphere root must leave its far root eligible.
Attributes are invocation-local values: they cannot retain report/candidate or
hide GPU resources. Infinite procedural loops retain ordinary shader watchdog
obligations; no per-hit allocation, root-count type specialization or host callback.

All programmable paths use these explicit callback signatures:
- raygen(args...); its return is ignored.
- intersection(report,candidate,args...) -> nothing.
- any_hit(hit,ray,args...) -> Bool; it is a pure filter, never payload mutation.
- closest_hit(hit,ray,payload,args...) -> P.
- miss(ray,payload,args...) -> P.

The same intersection and alpha-filter function bodies serve inline queries,
program execution and software traversal. Resources come from explicit args and
callable captures. Query-only execution does not invoke closest_hit/miss. Existing
implicit rayhit/raypayload/currentray convenience sketches are superseded by these
explicit values; raw native KI stage builtins remain qualified low-level interfaces.
traceray(origin,direction,payload;raytype=1,tmin=0f0,tmax=Inf32,mask=0xff) -> P inside
a ray program uses its bound TLAS/table/args and invokes the selected handler.
The same anyhit=false default/optional early-accept mode is available on traceray;
its selected hit handler receives the accepted hit (not necessarily closest when
anyhit=true). Omitting payload is legal only for P=Nothing. No user-managed native payload buffer
or hit object is required. Metal may lower the logical program to query plus typed
function dispatch; Vulkan may use native stages when eligible. Recursion/native
restrictions select a semantics-preserving candidate or diagnose unsupported raw
features; they never require a consumer backend branch. SER remains an optional
semantics-preserving optimization, not different shading code.

On both inline-query paths, generated intersections may not carry user attributes
in native committed-hit storage. Shared core retains its typed best attributes
beside the native query and verifies the committed identity before returning them;
a later triangle cannot accidentally receive a previous box's attributes. Native
Vulkan report/any-hit and Metal query/commit or intersection-function constructions
encode the same filter-before-commit semantics. Their attribute packing is an
adapter/compiler ABI matter under common layout rules. Native attribute limits
filter eligibility; a larger A may use shared query-local state when legal rather
than truncate. Native-only raw features can reject when no qualified candidate
exists. The default BoxIntersection streams the finite in-interval entry/exit surface
roots of the box, deduplicating a tangent; it does not mistake a clipped tmin
inside a box for a surface. Attributes are Nothing. AABB bounds are
minimum/maximum Float32 triples; updating bounds is a refit,
changing geometry kind/count is a build. Inactive padding is valid geometry/masks,
never NaNs. These choices are design rulings informed by the native mechanisms:
[Vulkan traversal and SBT](https://docs.vulkan.org/spec/latest/chapters/raytracing.html)
and [Metal intersection queries and filters](https://developer.apple.com/videos/play/wwdc2021/10149/).

Core computes resource uses from entry-point-reachable functions: a full program
includes raygen, hit groups and miss; a query-only entry includes only its
intersection/filter/capture functions. A buffer touched only in closest-hit still receives
holds, readonly validation, placement and barriers. Native ray-pipeline support
is optional; portable ray queries and ordinary shade-kernel candidates remain
available when ray pipelines or procedural native stages are absent. Arbitrary
native-only shader intrinsics are not promised software emulation. SER is a
semantics-preserving hint: a no-op is correct; silently skipping an any-hit alpha
test is not. Any future native-program candidate must match requested alpha/media/
light semantics; an opaque-only result is not a reference for alpha-textured scenes.

The refreshed baseline is Hikari 407f9b2, Makie 42e6355fb and Mantle 3e9943f.
Hikari/RayMakie now use ray-query wavefront tracing; their old RT pipeline and
_VP_RT_PIPELINES cache are historical evidence, not code to preserve or delete
again. Mantle's pushed core query_record/query_transparent owns the traversal loop
over thin Vulkan/Metal query verbs. Preserve current typed lobe/material queues,
cut-set specialization and compile/cache improvements. A future programmable hit
function path remains a separate capability profile; do not force current Hikari
back to the deleted pipeline or claim it already qualifies that path.

The pushed Metal query_begin specialization uses a single Metal.intersect call
when no candidate processing is required. The target plan preserves that native
construction but moves its eligibility/cost choice to the shared selector (§9).
Core checks the complete query requirements: geometry kinds, closest/occlusion
mode, opacity, filters, procedural reports and requested result attributes. A
missing procedural payload alone must not bypass required alpha/filter work.
Metal reports enabled native/compiler facts and encodes the selected intrinsic;
Vulkan supplies its applicable native query operations. Shared core owns filter,
report/commit, closest-hit/result and fallback policy once. Native call differences
are adapter mechanics, not permission for a second traversal/renderer algorithm.
A pure capture may execute after a native complete result if that result supplies
all necessary fields; capture alone need not force a candidate loop. Required
candidate-time filtering/reporting must never be bypassed. An eligible
complete-result call can bypass the candidate loop without duplicating
that loop or adding consumer backend branches. Preserve the optimization and
qualify its completed-work benefit; do not treat the source's timing comment as
new measurement evidence.

For kernel closures/functors, resource discovery includes `(kernel,args...)`:
callable fields containing arrays or GPURefs are explicit ABI arguments too.
Location validation, holds, access inference and argument packing must traverse
them. A kernel capturing an array from a second device is rejected before graph
mutation. Untracked mutable global GPU handles/pointers are rejected; there is
no implicit current device or resource hidden outside the discovered arguments.

## 8. Raster state without duplicated policy

Keep stage outputs declared once and derive the next stage's input interface
from them. Named outputs interpolate by perspective unless Flat{T}; integer
outputs must be flat. A fragment gets `(inputs,args...)`, where args is the same
draw tuple as the vertex stage. One result per colour attachment is required;
integer attachments are written unblended. Depth/stencil are typed attachment
state, not an extra colour result. Image storage sample count determines the
native pipeline sample count; do not duplicate it in two user declarations.

Preserve the existing `PerAttachment(modes...)` and `NoWrite()` state used for
picking/FXAA metadata. Opaque replaces. AlphaBlend uses src.a*src.rgb +
(1-src.a)*dst.rgb and alpha=src.a+(1-src.a)*dst.a. Premultiplied uses
src.rgb+(1-src.a)*dst.rgb with the same alpha equation. Additive adds channels.
Colour blending happens in linear space; integer attachments reject blending.
General `Blend(rgb,alpha;write=(true,true,true,true))` uses per-channel
`BlendChannel(srcfactor,dstfactor,op)` with standard factors Zero/One,
SrcColour/DstColour/SrcAlpha/DstAlpha and their complements, and operations
Add/Subtract/ReverseSubtract/Min/Max. Singleton modes lower to these values in
core, leaving one native state encoder per API. No second blend algorithm in a
backend. Destination reads are part of the attachment's access contract.

`Image(...; samples=1)` requests an exact supported sample count through
imagerequirements; unsupported format/count fails before allocation/declaration.
All attachments in one pass must match extent and sample count. Plain copyto!(dst,src)
is eager; call!(target,copyto!,dst,src) declares the recorded form. Both use the
same shared copy/resolve definition and return the identical destination. A
multisampled colour source with matching single-sample destination selects a
resolve construction (internally CompiledCopy(ResolveImage(...))), never a byte
memcpy or implicit host transfer. An image's construction target does not choose
the execution mode or recording destination.
The resolve consumes the source's GPU uses and produces a normal image use;
compile/liveness derive barriers and aliasing. Sampling or Array on a
multisampled image requires an explicit resolve first. Sample count enters the
pipeline cache key. Depth/stencil resolves require an explicitly supported
resolve mode; do not quietly average depth or stencil.

`Rasterizer(...; stencil=StencilOff())` enables stencil with
`Stencil(front,back)` values. Each `StencilFace(compare,fail,depthfail,pass;
reference=0,readmask=typemax(UInt32),writemask=typemax(UInt32))` uses CompareOp
and StencilOp values; native enums remain backend details. CompareOp covers
Never/Less/Equal/LessEqual/Greater/NotEqual/GreaterEqual/Always. StencilOp covers
Keep/Zero/Replace/IncrementClamp/DecrementClamp/Invert/IncrementWrap/DecrementWrap.
Enabled stencil requires a DepthStencil attachment; format facts specify valid
read/write mask bits. Reference and masks are pipeline values, not GPURef fields
that a native command could magically read. Pipeline changes obey normal cache
and piece replacement rules. The existing Float32 depth attachment remains a
depth-only format. These additions share render, dependency and recording paths;
they do not create a separate pass or backend policy model.

## 9. Selection, tuning and default launch geometry

The shared Mantle operation layer owns the ONLY candidate eligibility/selection
policy for both ordinary eager methods and recorded call! operations. A backend
reports enabled native facts and encodes the selected commands/intrinsics; it does
not choose algorithms/workspace/fallbacks or keep an independent operator policy.
Intersect hardware facts with actual compiler support, semantic precision/ordering,
argument shape/layout/alias constraints, workspace and graph/control eligibility.
Reuse KI tables. Capability presence alone is not a measured performance win.
Eager calls materialize immediately; graph compile materializes the retained recipe
using final region requirements. Replay does not probe, tune or reselect. Only
invalidation/explicit tuning affects a later compile. Per-function Julia frontends
are shared across hardware, not a matrix of duplicated per-backend overloads.
Reuse existing common GPUArrays/KA/Julia algorithms where possible; refactor their
emission boundary rather than copying eager and recorded numerical bodies.

The hookup is ordinary Julia dispatch, expanded in api-guide.md/internals.jl:
recipe(::typeof(mul!),argument_tuple) returns a typed MatMulRecipe;
candidates(::MatMulRecipe) lists shared emission functions and requirement metadata.
materialize(context,RegionNode) uses this list with enabled facts/final region
requirements, then calls the selected emit function to create private DAG commands.
Eager operation! calls it immediately; graph materialize_regions! calls the same
function at compile. No user Graph mutation or eager mul! call occurs during recipe
emission. Replay does not call the frontend or select again. Native providers
supply facts/encoders, not a candidates(dev,f) algorithm-selection table.

Core stores `Candidate{F,R}` values with the implementation callable, requirement
predicate, implementation version, cost model, workspace description and optional
partitioner/region compiler. All candidates describe the same operator semantics,
including aliasing, numeric accumulation type, epilogues and output ownership.
An `implements` attribute alone cannot excuse different alpha tests, overflow,
rounding or memory ordering. Filter by requires and validfor(args) before reading
a measured score. Optional library candidates additionally require recordability,
caller-owned workspaces and the enclosing region's actual control/recording
requirements. A native library subgraph is eligible when its complete node kinds,
parameter bounds and effects satisfy them. Unsupported candidates decline before
compilation; vendor origin alone does not exclude a valid recording.

SelectionKey is an immutable value of (operator identity/code, argument element
types, logical shapes or explicit bounded shape classes, strides/layout,
base-alignment classes, alias relationships, target execution kind,
compiler/feature/ABI key, candidate-set version). Addresses and objectid of arrays
are not keys. Plan capacity/shape dependencies guarantee continued eligibility;
a move preserving those classes needs no reselection. Growth outside the bound or a changed code/config/candidate version invalidates
the retained RegionNode recipe and marks its plan for recompile. Materialization
reselects and rebuilds the complete command sequence/workspace atomically while
preserving public output roots; recompiling an old kernel list is insufficient. GPU-written live lengths are represented by their capacity
class; selection never reads them back to the host. Measured exact-shape scores
cannot win a different class merely because element types match.

`tune!(dev,f,args...; check=referencecheck(f))` makes private copies preserving
alias/view relationships, copies recurrent state per trial, acquires all scratch
through the core pool, verifies against an independent reference candidate/check,
then measures completed work after warmup. No default reference/check means
tuning rejects the request before timing; arbitrary user functions cannot be
certified by a generic tolerance. Numeric tolerances and accumulation policy are
operation-specific. Failed/incorrect candidates never enter the table. Publish
only if code/config versions still match under the table lock; compilation,
measurement and waits are outside that lock. A graph currently using an older
candidate remains valid; tuning affects future compile/eager selection, never
mutates an in-flight plan. Tuning stores no user buffers and initially persists
nothing. Core provides this harness once, not a different tuner per backend.

The initial default workgroup is a bounded core heuristic, not an unresolved
backend knob: target 256 invocations, capped by the kernel/device product and
per-dimension limits; 1-D starts (target,1,1), 2-D starts (16,16,1), 3-D starts
(8,8,4). Reduce dimensions by powers of two until legal, preserve contiguous
first-dimension lanes where possible, and use occupancy/shared-memory facts when
known. Zero range submits no dispatch. Collective/matrix kernels contribute a
validity predicate and preferred group and are not forced into a generic shape.
Explicit caller group sizes are never rounded or replaced: illegal ones fail
before compilation/enqueue with the offending limit. Tuning can replace heuristic
choices only within kernel-declared legality. F32 is a performance qualification
gate for these defaults, not permission to discard caller geometry.

## 10. Profiling and counters

`Graph(dev;profile=false)` is the default and creates no timestamp objects.
profile=true records normal and instrumented variants at compile. There is one
in-flight instrumented sample per plan; while it is pending, subsequent runs use
the normal variant. Completed samples are harvested by a nonblocking
`readtimestamps!(dev,objects,destination)` into preallocated host storage before
those objects are reused. No profiling wait or per-run recording/allocation.
Normal runs never overwrite an unharvested sample. This is sampling, not a claim
that every submitted frame has a measurement.

Timestamp layout has a begin/end pair per compiled operation (a render pass is
one operation, not one new pair per inserted draw), grouped by channel/segment.
Native `timestampinfo(dev,kind)` reports availability, period, valid bits and
clock comparability; `preparetimestamps(dev,layout)` creates counted query/event
objects before record; `emittimestamp!(builder,location)` encodes a marker.
Compilers may produce two execution variants; their code/pipeline objects share
the same caches and placement. CUDA/HIP event objects are not reused while their
instrumented executable is in flight. Native query/event results are read only
after its actual token passed. Unsupported queue timing reports unavailable,
never a fabricated zero-duration GPU sample.

Core keeps a bounded host ring of 64 completed samples per graph, overwriting
oldest completed samples and reporting dropped/skipped sample counts. `timings(g)`
returns an allocated immutable snapshot of harvested samples and counters, without
waiting; querying it may harvest an already-completed sample. Each row includes
run id, node id/origin, stage/segment/channel, interval seconds and availability.
Intervals on unrelated clocks are not subtracted, and overlapping node intervals
are not summed into a fictitious frame duration. GPU wall latency, host submit
latency and estimated cost are distinct fields. Timestamp interval wrap uses the
reported valid-bit width; an ambiguous interval is unavailable. Calibration may
influence a later compile, never reorder the active plan mid-run.

Native timestamp objects of a replaced plan retire against its tokens; completed
samples may be copied to the graph's host ring, but profiling does not keep an
otherwise unreachable graph alive. Any new device memory required by timing is
pool-owned/countable. Diagnostics.records counts persistent plan recordings, including pieces/build
replacements and profile variants; prefixrecords and eagerrecords count temporary
change-prefix/eager encoding separately. Performance reports include all three,
so zero plan re-records cannot conceal per-frame prefix work. There is no duplicate
Plan rerecord counter. Plan.compiles retains its graph-wide compile count as already
specified. The former Profiler/passbegin!/passend!/gpupasstime!/collect!/timestamps surfaces
converge on Graph(profile=true), timings(g) and the shared diagnostics, not a
second recording/profiling subsystem.

## 11. Texture restoration, samplers and device reset

Restore of an evicted image prepares a fresh native image and an unused table
entry before locking. The applying submission restores texels/layout and writes
its new index/extent to the cell; subsequent sampling waits for that token. Old
indices/native views retire against every old use token, not the new image's
lifetime. A cached native image handle cannot silently survive eviction. The
non-indexed fallback follows the same resource/version protocol for its buffer
representation. Restoring alone never recompiles a shader.

A Sampler is a pure immutable value. First compiled use interns a core
SamplerEntry resource (native object and table index); bindings/recordings take
holds on that entry. The value-to-entry cache is weak and cannot keep every
historical slider value alive. The last compiled holder retires the index and
native sampler with its GPU tokens; a concurrent first user either takes a
nonzero hold or creates a new entry under the ordinary cache-publication rule.
No 0→1 resurrection. Images do not own sampler entries merely because both were
used by one shader.

Anisotropic sampling needs a footprint. `sample(...;gradients=(duvdx,duvdy))`
provides it explicitly in any shader stage; a fragment's implicit derivative
form may supply it natively. Explicit fixed-LOD sampling without gradients allows
anisotropy=1 only. An unsupported native anisotropy request selects the core
footprint sampler when available, otherwise fails during candidate selection;
it cannot invent derivatives from coordinates or silently change the filter.
api.md §2.6 follows this same footprint requirement.

`reset_device!(dev)` is explicit debugging teardown, not an in-place resurrection.
Core closes admission for the whole native failure domain, drains outstanding
host enqueues and foreign leases, waits for accepted native work outside all
Mantle locks, releases presentation according to §4, destroys children/context,
and removes that exact object from the registry. A later Device()/devices()
creates a fresh device/context generation. Every old array, graph, pipeline and
borrowed view remains invalid and reports DeviceClosed/DeviceLost before native
use. free! of old handles is idempotent and never calls a destroyed driver twice.
Finalizer inboxes of a closed pool discard CPU bookkeeping safely. A reset from
that device's own hostcall! or foreign lease throws before closing admission;
waiting for itself would deadlock. Registry mutation does not hold the registry
lock across waits/driver destruction. Borrowed test wrappers cannot reset their
inner native device. Legacy WSI's final-shutdown limitation remains explicit.


Suite queue fixtures use `channelinfo(dev)` from core Runtime: entries expose
canonical state identity plus compute/graphics/transfer eligibility. A list of
queue-kind labels can alias one native endpoint and does not prove multiple
queues. :multicompute requires two distinct compute-capable states;
:transferqueue requires a transfer endpoint distinct from a compute endpoint.
This is a diagnostic view of existing facts, not another backend verb. Queue
count alone does not guarantee overlap/preemption; latency cases qualify that
scheduler/hardware behavior against their baseline rather than redefine API
correctness. Explicit driverpairs supplied to runsuite identify verified
CUDA/Vulkan pairs; equal display names do not identify one physical adapter.

## 12. Memory classes, pageable eviction and explicit transfers

Restoration uses a qualified direct transfer: a native import/registration of
retained host pages, or a CPU copy into the final CPU-visible device allocation
(the same construction already allowed for initial array upload). Core selects
from validated source/memory-class facts, not by catching arbitrary failures.
Core-owned HostCopy has suitable backing/alignment. Neither construction creates
a second array-sized upload allocation. Do not substitute permanent remote host
access for device-local storage without the operation's placement contract.

Prepare source access and destination storage before record locks. A native GPU
copy retains both through its accepted copy token and orders the cell write and
later uses after that copy. A CPU copy initializes only a private, unpublished
destination; finish host visibility/flush before publication, then validate its
snapshot and extent under record locks. Its source is no longer needed by GPU
work after the CPU copy, while the prepared destination remains held through
acceptance. Preclaim failure/retry releases unused leases/destinations without
consuming Evicted state or pending changes. Accepted work retains the native
objects it uses until retirement. Import lifetime must not leave idle eviction
copies permanently pinned. F74/P4.A31 tracks replacement integration, source and
capacity coverage, and performance qualification. Local primitive evidence is
in experiments/direct-host-2026-10-21; it does not close the replacement gate.

No staged restore fallback is prescribed or justified by the current evidence.
A failed direct transfer is first treated as an implementation bug: follow the
plan's [debug-before-design-change rule](recording-plan.md#debug-before-changing-the-design),
fix the implementation and repeat the original test. Only overwhelming evidence
of a genuine target/source limitation, after that investigation and exhaustion
of direct-path fixes, can justify a fallback for an actual required workload.
Extension availability is not successful-transfer evidence; a failed import or
historical claim is not proof of an unavoidable limitation. An unqualified source
path remains an explicit qualification gap, never a silent staged workaround or
a reason to drop a required consumer case. Bounded eviction below is a separate
protocol; it does not mandate staging for restore.

Budget bytes are not interchangeable across native memory heaps. Core receives
`memoryclasses(dev)` facts (native class/heap key, host visibility/coherence,
device-local flag, alignment/granularity and applicable usage constraints).
Core chooses a compatible class for Buffers/Unified/Readback/Images/Accel;
rawalloc receives that chosen class in its constraint, rather than choosing a
second allocation policy in the backend. `budget(dev,heap)` reports ADDITIONAL available
bytes, excluding current allocations, for that heap and its published provenance says driver/OS/estimate.
`budget(dev)` is core's aggregate over distinct eligible device-local heaps;
shared UMA memory counts once. Host readback/pinned allocations on a different
heap have their own bound and are not charged as discrete VRAM. Metal's working
set is an estimate/recommendation, not an exact measurement of all other apps.
Normalize Vulkan heapBudget-heapUsage, CUDA/HIP free bytes, and Metal's
recommended working set minus current native allocations (an estimate). Compare
only the next block size to this headroom; used+next <= free double-counts this
process's allocations. FaultDevice BudgetLimit dynamically returns
max(0, ceiling-reserved) rather than a constant free-byte answer.

A budget check is advisory: allocation failure remains possible and recoverable
before claim. Pool statistics report heap/class and opaque-object bytes, rather
than claiming an allocation-free backend when a driver owns unmeasured objects.

A writable eviction copy must become pageable CPU storage. The portable baseline
uses one core chunk-copy algorithm with at most 8 MiB of reusable Readback staging
per device, prepared/reserved at initialization before pressure. Its host
HostCopy contains bytes plus original applied extent/format; a device-written
length cell is copied in the same snapshot before publishing the host copy. Allocate that host
copy and acquire the staging lease before freezing a resource. The staging
lease is released without any record lock. Snapshots are captured under expanded
locks before host allocation and validated again when freezing; no unlocked
storage/shape read is assumed stable. There is no pinned Readback buffer
as large as each evicted tensor, no hidden driver staging allocator and no GPU
wait under resource locks.

Core represents an in-progress copy by `SyncState.transfer::Union{Nothing,
HostCopyJob}`. Under expanded locks, verify eviction eligibility and take a
resource hold, snapshot storage/applied extent/versions, and publish the job.
A new native user/register of a handle-dependent plan sees the transfer and
retries after its completion outside all record locks. Scheduled stores/resizes
may still append pending changes; they do not mutate the frozen physical source.
Only the job's private read submissions may use that source until completion.
For each chunk, submit an ordinary tracked GPU copy, release locks, wait for its
token, invalidate noncoherent staging, and memcpy into pageable HostCopy. Copying
is synchronous within the already-blocking pressure step; no new device worker
or backend eviction algorithm is required. CPU/GPU storage overlap is explicit.

After all chunks, under the same validated expanded locks, replace storage with
Evicted(host,nothing), retire old storage against every old use token, mark all
current namers for restore, clear transfer and publish completion. Pending
changes remain ordered for the next restore/apply. Old concurrent readers can
finish; old writers were waited before the first copy. On a preclaim/CPU failure,
keep the old physical storage, clear the job and wake waiters with failure. On
ambiguous native failure, quarantine storage/job holds for device teardown.
Finalizers/free cannot release the source while the job owns its hold. The job
never needs a graph lock: a graph waiting with its own graph/arena holds cannot
block the copy from completing. This is the extra linearization rule required
for a bounded staging fallback; a comment saying "memcpy after the token" is not
a complete pageable-copy protocol.

Read-only arrays retain a source that the caller promises to keep immutable for
the resource lifetime (including graph holds after creator free!). They need no
copy out. `readonly=true` cannot make an ordinary Julia Array immutable; mutating
that borrowed source while retained violates the source contract. A writable
unified allocation copied into pageable HostCopy still occupies system RAM until
the OS reclaims/compresses/swaps it. Do not promise that two 90%-of-budget graphs
always fit a UMA system without sufficient host backing. Native external-host
imports may optimize the same copy job when qualified; that is not a different
lifetime/commit protocol.

Empty native blocks can be cached within a bounded core warm-cache policy;
The private core helper `trim!(dev)` releases empty caches after passed retirements
without waiting for live uses. It is neither exported nor declared public, and is
not an application maintenance step. It never unmaps an array's live pages.
Explicit array-backing trimming is deferred, outside the initial public API;
logical shrinking retains capacity.
Memory/leak fixtures wait/sweep/trim before comparing physical reservation;
normal run latency is not forced to pay rawfree/rawalloc for every reusable block.
Under pressure, empty caches are trimmed before evicting live resources.

`devicecopy!(dst,src)` is the explicit cross-device operation for device arrays,
including views. The portable baseline takes one coherent source-root snapshot
through the shared snapshot/copy/hold protocol, then schedules that snapshot into
dst through the ordinary store path
and waits for its destination application token. It returns dst after completion,
without idling unrelated work. It can allocate host memory proportional to the
copy and fails before modifying dst if the snapshot/allocation fails. This explicit
transport operation snapshots a persistent root after accepted prior uses; it
never executes any graph that names the source. Different
sizes/types follow copyto!'s validation/conversion rules; no silent relocation
of later operations. Explicit call!(target,copyto!,dst,src) transport recipes may name a foreign source and
publish a destination in the target domain; ordinary mixed-device arithmetic
remains an error. A persistent source already named by a graph can be retained by transport;
scratch cannot escape its run. Qualified peer/external-memory candidates may replace staging
while preserving source snapshot, destination publication and resource holds;
no zero-copy or nonblocking claim is made for arbitrary driver pairs. This
baseline deliberately reuses existing read/store logic rather than inventing a
cross-driver graph scheduler. Ordinary execution remains device-local; explicit transport/communication regions
connect per-device graphs with declared readiness/completion (§16).

Streamed weights retain their explicit lifetime/interval contract; ordinary
Graph construction is persistent, and only scratch=true requests arena lifetime.
Streamed weights keep `streamed=true` as the keyword. Their retained host source
is immutable by the same contract. Streamed weights are read-only: inferred
writes/resize reject through the shared readonly validation before publication.
Uploading an old host source must never silently reset GPU-mutated state; mutable
state requires a separate persistent copy. Each graph declares a run-local resident
interval plus upload-copy nodes before the weight's uses. Core places/aliases
that interval and derives transfer edges, so several weight windows can overlap
compute without a consumer-written partition or backend-specific loader. The
bytes/upload bandwidth are charged to the estimate and performance report.
A repeated use in an overlapping live interval shares one upload; a later
nonoverlapping interval may reupload. Source paging and transfer time are real
costs; the option does not promise faster-than-resident execution. A consumer's
module/call graph still owns its semantic operator order.


Reservation statistics have fixed units: reserved is physical bytes in
Mantle-owned native blocks (including caches); mapped is the subset currently
bound to sparse virtual ranges; virtualreserved is reserved GPU address space,
not physical bytes. Never add mapped to reserved to estimate physical usage.
Opaque/external handles and host-copy bytes are reported separately with known/
unknown provenance; imported aliases are not counted as a second owning block.

## 13. Cache ownership and pressure

A native cache hit must return a live held object, not a raw handle whose last
owner can retire it concurrently. The same core cache-publication machinery
serves graphics/ray pipelines and samplers: per-key serialization, no compilation
or device allocation while the map lock is held, and explicit object holds for
plans/bindings/accepted eager work. Keys contain code/config/layout/fixed state;
GPU resource addresses and captured array objects are never cache-key roots.
Captured shader data belongs in argument slots and resource holds.

One bounded device-owned idle-object policy may retain native objects for warm
reuse after the last active plan/eager use. It is shared policy, not one LRU per
backend. Initial limits are 64 idle objects, 64 MiB of known native bytes and
8 idle objects of unknown size, whichever binds first. This is a measurable
cache bound, not a claim that an unknown-size native pipeline costs zero.
Pressure first drops idle cache holds and trims empty blocks, before evicting
live array data. Active plan/eager/token holds keep objects alive independently.
Invalidation removes the cache hold; destruction still waits for last GPU uses.
CodeInstance owns compute code/metadata and its per-device entry; pressure can
drop an idle native entry without erasing Julia code or introducing another
kernel dictionary. A later native rebuild uses the same owner/key and reports
its cost. Ordinary eager reuse within the bound remains warm.

Weak owner links prevent an idle list from retaining every dead CodeInstance or
consumer program forever. Native object finalizers enqueue drops through the
existing release engine. A ready weak entry pointing to an object whose hold
count reached zero is a miss; it cannot resurrect the old handle even before
the final driver destroy. Release checks do not acquire cache locks while a
resource/channel lock is held. The compile/record separation is unchanged.

Native raw allocation reports `RawMemoryExhausted(heap,bytes)` for recoverable
GPU allocation exhaustion before claim. Core tryacquire catches this typed
condition, runs pressure outside pool locks, and retries boundedly. Base
OutOfMemoryError (CPU bookkeeping allocation) and other driver errors propagate;
this is typed error handling, not a feature-detection try/catch probe. All
postclaim enqueue failures still use terminal quarantine, regardless of error
name. Cache creation/pressure also happens before claim.

## 14. Enforce recording purity and cell isolation

`checkrecord!(dev,true)` enables a device-owned debug guard. record! enters it
for the recording task and leaves it in finally. All semantic resource reads
used by compile/run pass through typed snapshot accessors; the guard rejects
reading storage, dimensions, pending changes, membership, cells or live native
handles while recording. Direct getfield on those mutable representations is
restricted to their owning accessors; the source/typed-IR audit enforces this,
including generated helpers. A backend emitter receives immutable commands,
slot addresses and deps, never a live ArrayState to inspect. Read counters alone
cannot prove purity if arbitrary getfield bypasses remain.

A negative control `withrecordguard(dev) do; readresourcesnapshot(a); end` must
throw RecordStateRead. A failed guard must unwind before a subsequent valid
record. `cellaccesses(g)` exposes compiled memory-origin facts (node id/role,
source-cell reads/writes), derived from the actual selected code's taint/IR
analysis and checked against resource uses. It distinguishes source Cells from
ordinary argument slots and GPURef data slots. Only head/prepare, TLAS record/offset preparation and compiler-inferred device
resize! metadata writers may access source Cells;
ordinary launch/raster/shade kernels read prepared slots. Cell pointers cannot escape through an opaque native argument or an untracked raw-pointer cast.
The public launch API does not expose source-cell addresses; only the compiler-owned resize! intrinsic lowering receives an internal write capability.
A backend must qualify the check against its compiler output rather than merely
stamp every ordinary kernel with "no cell reads". THR-04 and KL-10 define the
executable acceptance cases; debug checking is off on performance runs.

## Graph-declared readback snapshots

Array(array) is the standard blocking conversion of accepted device work/ordered
pending resource changes; it never chooses or runs a graph. call!(target,Array,array),
target Graph/Group, declares a recorded position and returns a one-shot future,
regardless of prior compilation. Both use one shared snapshot/copy/hold/visibility
recipe. No public eager Readback(array) operation is added; fetch starts no work. Readback(previous_future) requests the next execution of an existing
output ID; source identity never selects a declaration position. The copy has an
explicit source-read and
destination-write dependency, so source liveness reaches the copy and later
source writes follow it. Readback destinations do not alias explicit scratch storage.
Vulkan's recorded copy includes a transfer-write to host-read visibility barrier;
coherent host-cached staging is read only after the captured execution token.
The shared-memory implementation uses a normal graph copy kernel.

Graph owns stable output IDs and pending requests. Each plan generation owns its
output bindings and reusable staging slots. Compatible plan replacement rebinds
pending requests by output ID; it does not cancel them merely because the old
plan retires. Already submitted futures keep the old generation/token/storage.
Deleting an output or freeing the Graph rejects its unsubmitted requests.
Each execution with requests publishes a snapshot carrying that run's token and
a hold on the slot's resource. With no request for an output, do not transfer its
payload or materialize an old CPU result. Core lowers its known request state using
an eligible conditional-copy construction or an optional precompiled snapshot
segment, retaining source ordering when bypassed. This is the same region/segment
machinery, not a backend output scheduler. Do not cache 2^N recordings for N outputs.
Any remaining control/submission overhead is reported separately from payload bytes.
`fetch` waits for the captured token and makes an owning CPU array; no
submission is made. Before replay overwrites a previous outstanding snapshot,
core materializes it. Internal plan retirement remains nonblocking and drops its
slot holds, while submitted futures keep their storage until collection. Pending
requests belong to the live Graph/output, not to the retired plan. At the start of a run attempt, bind its
pending-request set before preparation/claim. A skip or failure before a copy is
accepted rejects its bound request; do not leave it waiting for an unspecified
future run. Requests created later remain pending for the next attempt. Already
accepted copies retain their published token/storage and complete or fail according
to that work; later unrelated stage/present failure cannot erase acceptance. A moved native source binding is recompiled/recorded before replay;
a changed logical size is captured with payload using capacity-bounded storage.
Exceeding the recording/placement bound rebuilds the affected plan/output slot,
preserving Graph identity and already submitted snapshots. The public lifecycle
is run!(g)/free!(g); Plan and record! remain internal.

The first warmup execution consumes an armed output request like any other run.
Each wanted later execution requires Readback(previous_future) before run!;
fetch remains tied to the captured execution, never a latest-value query. Warmup
can mutate persistent state and has no implicit reset. See the canonical sequence
in api.md's Warmup, repeated requests and retention section.

A new request copies the stable Graph/output descriptor, not the previous future
or its captured payload/token/generation. Completed futures are removed from the
Graph's pending set; the Graph must not retain a completed-result history. Preserve
outstanding snapshots before staging reuse, but release obsolete plan/slot holds
once their tokens and live result references permit. Result retention is bounded
by live requests/results plus a bounded staging cache, subject to ordinary Julia
collection; holding N snapshots can legitimately require N captured payloads.
Native staging/CPU memory bounds still require the existing F117 qualification.

## 15. AScanner qualification of the existing GPU-size model

The 2026-10-29 AScanner/Bosgame review adds consumer migration and qualification
coverage to an already settled design. It does not introduce a new array-shape abstraction or general overflow policy.
Public spelling now follows Base.length/size and Base.resize!; Cells/preparation
remain the same internal size model. Explicit storage lifetime follows api.md.
Source fingerprints and inspected paths are recorded in
`experiments/review-2026-10-29/ascanner-gpu-extents.json`. The consumer already has
working retained-frame and CUDA qualification evidence; the observations below
are source-level migration gaps, not newly executed failures.

### Settled representation and compaction

An array's capacity is its allocation bound. Its logical size is GPU-resident
resource metadata: the Cell/compiler-inferred device resize! model already specified in the plan,
conceptually `a.size::GPURef`. Head/prepare nodes propagate that size into
consumer argument slots and launch commands. Ordinary numerical kernels use
the prepared descriptors (§14); this is not a second unrelated user-managed
counter that must be threaded through every array operation.

A stable application builder `app_compact!(target, active, particles, predicate)`
emits declarations to evaluate flags, scan them,
scatters selected values and publishes the resulting count with
`resize!(active, selected_count)`. Reserve output capacity for the maximum input count: a
correct selection cannot produce more outputs than inputs. Scratch is declared
explicitly. A final count writer executes even for an empty input and publishes
zero; numerical work and the count write are ordered before consumers.
No host count readback or automatic growth/retry is required for this operation.
The builder is illustrative application/library composition, not a public Mantle
compact! overload or a recorded-call registration requirement.

`length`/`size` inside subsequent kernels observe the logical GPU size. Composed
operations plan scratch and maximum work from capacity, then consume that live
size on the GPU. Changing it within the allocation bound does not require shader
compilation or plan rebuilding. Host Base.length/size read/wait for GPU metadata.
Those host integers are explicit snapshots, not deferred dependencies. Graph
recipes consume prepared size metadata; callers use launchrange(a) for changing
geometry. call!(target,Array,a) captures a graph position; Array(a) captures
accepted work eagerly. CPU results use ordinary host length/size.

A view used in a kernel can be constructed there from its parent and GPU bounds.
The parent and bound resources remain dependencies. Composed operations over
such a subrange must carry those bounds to their internal kernels; a local
view does not convert an integer-only host planning helper into a dynamic one.
This review does not require a new public `dynamicview` constructor or array type.

### Concrete consumer migration

In the inspected `MatrixConverter.ge2sym!` source, `select_flagged!` returns a
GPURef, the next `loose_resize!` method accepts only `Integer`, and later arguments
call `Int32(m)`. That is a consumer conversion gap: the resulting logical size
must remain GPU state and downstream operations must use it. Adding a hidden
scalar download would defeat the retained computation. This source path was not
executed by the review.

The working contact path uses full-size views, separate GPU counts and inactive
sentinels. Preserve its validated results while migrating appropriate operations
to the canonical array-size metadata. The `cub.jl` reduction/scan/sort helpers
currently accept host integer lengths; their dynamic-input paths need the same
migration. P6.A24 and P11.G94 track this work rather than a new core design.

### Additional consumer qualification

These are **specified, unimplemented** acceptance cases for the existing
P4.A3/A10/A11/A12 contract and its composed operations. They are not native test
passes. Use independent expected values and poisoned inactive storage; retain
KL-01–10's underlying size/bounds/growth/readback invariants, migrating their
old explicit size-handle calls to inferred device resize! effects. KL-05's
out-of-capacity publication becomes a checked rejection, not a permitted invalid
logical descriptor; no automatic growth/retry is introduced.

| Gate | Required behavior |
|---|---|
| KL-11 | One recorded compaction replays through counts 0, 1, G-1, G, G+1, capacity, 0 and active again. Kernel-visible size and consumer work agree; count publication handles empty input, tails remain outside the logical result, and compile/record counters stay unchanged within bounds. |
| KL-12 | Compaction → scan → stable key/value sort → run reduction/unique → gather/copy/reduction composes without host count reads or integer conversion. Empty identities, ties, maximum-valued keys and poisoned tails have exact expected results. |
| KL-13 | A kernel constructs views from GPU-produced offsets/counts; composed consumers receive those same bounds. Parallel COO rows, columns and values use a consistent logical count. No extra public view/array abstraction is introduced. |
| KL-14 | A repeated/conditional body publishes an extent more than once before different consumers. Cover active, empty, skipped, zero-trip and reopened executions. Prepared descriptors refresh after each producer, not only at the run's head; partial workgroups preserve collective participation. |
| KL-15 | Producer/consumer graphs share an array and its logical-size state. Plan replacement, retained handles and host snapshots preserve matching payload/size and their existing synchronization/lifetime contracts. |
| KL-16 | A CUDA.jl allocation imported into a CUDA graph uses its existing storage with retained ownership and explicit stream ordering. GPU-size changes reach downstream descriptors without host readback or mutation of an in-flight native graph. |

P4.G4/P6.G4 require the relevant cases on every joined backend, including native
indirect launch and capacity predication. Extent dependencies include aliases,
callable captures and library-internal launches; extend the existing KL-10
compiler-origin audit to these consumers. A predicated launch must preserve
workgroup barriers and reduction identities for its partial tail.

Phase 11 must exercise the actual sparse-conversion and retained contact paths,
not only toy kernels. Keep the same-GPU interleaved performance comparison at
capacities 32/128/512 and a large sparse-live case. Bosgame's saved report already
shows slower larger frames despite one native graph launch. Distinguish live
work from capacity/padding work, and record scratch peak and complete frame
latency. Fixing host-count assumptions is not itself a measured speedup.

## 16. Composable regions, resource effects and review corrections

This section supplies execution protocols for api.md's reconciled entry points.
It replaces the old early-expansion/kernel-only/lifetime-by-location restrictions;
it does not create another runtime. Existing atomic, WSI, pressure and direct-restore
evidence/rulings remain in force. Implementation and native qualification are open.

### 16.1 Explicit targets, ownership and lifetime

An array owns exactly one Mantle Device/domain. Arbitrarily many nodes in sibling
Groups or unrelated graphs may reference the same root. Each node declares its
uses and retains holds; group membership is not an array/device property. Neither
adapt(group,a) nor adapt(graph,a) is a recording API. A recording-device wrapper,
array retargeting or target-bound output alias is forbidden.

Ordinary eager methods retain Julia signatures: mul!(C,A,B), broadcast!(f,C,...),
Array(a). A recorded ordinary operation uses call!(target,f,args...;kwargs...),
target Graph/Group. It has a thin Julia-dispatch frontend to a known shared Mantle
operation definition/recipe. There is no public mul!(dev,...)/mul!(group,...) pair.
Core resolves target ownership internally for validation/materialization; there
is no public Device(Graph/Group) constructor/query overload. KA.get_backend(a)
returns the owning Mantle Device. Direct kernel launches still use dispatch!(target,...).

Known lowering is not generic capture. A composed library either explicitly emits
recorded calls/raw dispatches or supplies a qualified closed-library recipe. Dotted
syntax remains eager; arrays/KA backends do not carry context. An unsupported f
rejects before declaration publication, never executes eagerly as a fallback and
never silently becomes a CPU callback. Native capture is a capability-qualified
candidate construction, not the universal API contract.

Target-aware library graph builders are called directly with a Graph/Group, e.g.
PrintSim.step!(group,world,state,dt). They compose declarations during construction
and require no call! wrapper/registration or replay-time CPU invocation.
Conversely, a consumer host controller that updates inputs, submits runs or waits
remains ordinary host orchestration; call! must not capture or defer that controller.
The function's actual body/contract determines which case applies, not its name.

Known recorded host-to-device copy definitions own the host source snapshot at
declaration and replay that captured content. Updating the original host array is
not an in-flight/baked-argument patch. Per-run host input changes are explicit
ordered stores or declaration replacement. A device-to-device copy reads its source
at its graph position instead, with ordinary data dependencies. These are common
copy semantics; an unknown function still never becomes a CPU callback.

The minimum recorded catalog is in api.md. A reduction with an ordinary device
array destination, e.g. call!(group,sum!,abs2,r,A), returns that identical array;
a later device kernel reads r[1]. This is not a CPU-result future. Its recipe
tracks input/capture reads, destination writes and destination reads for init=false.
GPU-written live lengths are descriptor inputs, not host-side isempty/size branches.
Empty sums still publish their identity; init=false preserves initialized prior
contents on empty input. This contract uses existing sum!/array semantics rather
than introducing a new public scalar result handle or guessing all reductions'
return contracts. Existing domain-specific device-output kernels remain valid.

The shared operation definition states its returns. mul! returns the original C
because its body does so. A ! suffix says nothing about which argument/value is
returned; no generic mutation/alias heuristic supplies a return contract. Stable
device-array outputs/metadata aliases may be returned when that definition can
declare them. A CPU value available only after execution has an explicit typed
future contract under the common output protocol; Array is the required snapshot
case. This is not an every-call-is-a-future rule. Each additional operation family
must specify and test its result contract. In device control, use device-output
forms (e.g. in-place reduction to device storage); CPU snapshot nodes reject.

An actual hostcall! means Julia CPU work executes during replay, on staged host
arguments with explicit writes and a host cut. call!(g,mul!,...) instead lowers a
supported GPU operation; it does not replay the original Julia mul! each run.
Qualified native library/event handoffs retain their disclosed stage/cost protocol.
A library helper forwards the explicit target at every recorded child call and
workspace allocation. This is one portable operation implementation, not a second
operator algorithm per backend.

For GPU-sized operands, explicit graph operation providers retain shape dependencies
before generic host axis instantiation and prepare/validate descriptors on device.
Host length/size synchronize for GPU-written metadata. Ordinary returned integers
are snapshots; arbitrary arithmetic over them does not retain hidden graph
provenance. Use launchrange(a) for changing launch geometry. Retained recipes track
their own declared shape/eligibility dependencies; raw integer arguments stay fixed.

Persistent roots have creator/naming/composite/lease holds independent of which
graph constructed them. Graph release drops graph holds only. Persistent creation
data initializes once at its ordered construction point; explicit nodes reset it
on replay. Deferred GPU-source copies must retain/order that source version through
the shared snapshot/copy protocol before later writes, not read a later first-use
version. Host data uses an owned snapshot unless readonly/streamed immutable-source
retention was explicit. scratch=true requests
run-local storage independently of the owning device. Scratch data construction
emits initialization for each execution of its explicit owning region; type/dims
construction leaves bytes undefined until written. Scratch metadata initializes
per run/path. A raw Write effect is not proof of full initialization; high-level
recipes declare coverage and raw authors retain their ordinary obligations.

A scratch handle cannot escape its owning run, be immediately host-read, or be
held by a persistent composite. Other regions in that same graph may use it when
initialization/control/liveness permit. call!(owner_target,Array,a) copies before
its lifetime ends and holds independent staging. Eager Array(a) cannot read
run-local graph scratch. readonly/streamed retention and foreign import require persistent ownership;
reject conflicting lifetime options atomically.

Declaration validates all generated nodes/controls/effects/captures/holds together,
reserves publication and commits once. Failure leaves nodes/holds unchanged. An
inferred resize writer and its prepares cannot be published partially.

### 16.2 Size effects and producer descriptors

Device resize! is a metadata-write intrinsic. Inference supplies its ABI-internal
write capability; ordinary device arrays expose no Cell pointer or separate public
size handle. It validates dimensions/capacity, changes that invocation's descriptor
and publishes metadata without changing address or allocating. Aliases of the same logical root header in
that invocation observe the local size update; independent headers/ordinary views retain their
own extents. Compiler alias/effect lowering must qualify this without adding
source-Cell reads to ordinary length queries. Other invocations
retain entry descriptors; no grid-wide barrier is implied. Subsequent dependent
dispatches receive prepared dimensions, including aliases/captures/library kernels.
Static views retain their own extents; device-local views from GPU bounds are normal
views with explicit parent/bound dependencies.

Compaction uses one canonical root and size. Checked internal producer descriptors
cover available capacity while its previous logical length may be zero. Flags ->
scan -> scatter -> final resize publisher are explicit command dependencies, and
consumer descriptors follow both payload and metadata. Empty input still executes
a single count publisher. Scratch partials track their actual live block count,
not their allocation capacity. This is the implementation protocol for the settled
size model, not another public array type or capacity-overflow policy.

The authoritative applied dimensions live in the Cell once a GPU writer exists;
host requested dimensions describe only ordered pending host changes. Never restore
or validate a store using stale requested dimensions as if they were the GPU result.
Host stores with unknown logical bounds use a prepared descriptor and checked store
kernel, not an unconditional inline/copy command. Invalid ranges record an error
without writing. copy!(a,data) can establish known dimensions through its atomic
discard-resize/store sequence. For rank > 1, host resize!(a,n) requires known prefix
axes; otherwise provide explicit dims rather than implicitly downloading size.

Capacity is the usable allocation bound, including backed page rounding rather
than the larger unbacked virtual reserve. Unplaced/evicted resources retain that
bound as an allocation/restore requirement; the field is not a resident-byte count
and capacity(a) does not allocate or execute work. Placement/restore must establish
backing of at least that bound before publishing address/capacity descriptors. If
it cannot, fail preparation rather than expose an unbacked producer range. Native
storage and its prepared capacity version remain held for all submitted users.
Device resize and producer bounds use that prepared backed capacity. Explicit trim
captures/validates one logical-size version, may require a visible metadata checkpoint
before host-native page binding, and reduces capacity only at accepted unmap commit.
Later host growth may remap the existing virtual range. Resizes/uses cannot race a
checkpoint's version lease. Candidate output-capacity requirements are revalidated;
an explicit trim cannot leave a checked producer writing unmapped pages.

Host growth after a GPU size writer also retains an ordered preserved-prefix
extent. Prepare the old descriptor and copy min(old logical bytes, new logical
bytes) before publishing the new dimensions/address. Do not freeze a keepbytes
integer from stale requested dimensions when queuing Resize. Known extents use a
native copy; GPU extents use the shared bounded copy recipe or a qualified native
equivalent without a host scalar roundtrip. New elements beyond the old logical
extent remain uninitialized until explicitly written, even when their backing
already existed. This uses the same shape/metadata preparation and copy protocol.

Eviction HostCopy retains the coherent GPU dimensions as well as bytes. Restore
publishes that saved shape and the new address/capacity, then applies later pending
host resizes in sequence. It never overwrites saved GPU dimensions with the stale
host requested shape. These use the existing snapshot/transfer-lease protocol;
no independent size counter or backend restore algorithm is introduced.

### 16.3 Regions, recipes and native control

RegionNode retains a semantic operation recipe, arguments, code/eligibility key,
effects and stable outputs. Compile materializes it with fresh plan-owned scratch;
eager execution uses the same recipe. An invalid shape domain re-emits the recipe,
not just the old kernels. Retained recipe shape dependencies are validated at their origin. Ordinary host
shape integers are fixed snapshots, never silently assumed to remain dynamic.

Nested Group/Loop/condition trees remain structured until core selects an enabled
encoding. Native candidates report supported node families/nesting/update limits;
portable expansion is one shared core algorithm. Control resets follow lexical
invocation, not just top-level run. Internal prepare/build/library work participates
in region predicates and exit frontiers. hostcall! inside GPU control is diagnosed;
ordinary Group host callbacks retain the staged host protocol. Library eligibility is
actual recordability, effects, caller workspace and control support, not a blanket
kernel-only rule. Private-stream closed capture can supply a library subgraph.

References: [CUDA conditional/device graph constraints](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html),
[cuBLAS graph support](https://docs.nvidia.com/cuda/cublas/index.html#cuda-graphs-support).
These establish available native mechanisms, not universal eligibility or a HIP
equivalent. Measure expanded/native compile time, executable/slot bytes, stop cost
and full workload time; a native region that cannot meet budget semantics declines.

### 16.4 One command DAG and complete alias frontiers

Each emitted command, including within a head, change prefix, build preparation or
closed recipe, has identity, declared effects and explicit predecessors. A builder
does not reconstruct order from a barrier or drop an edge merely because it points
outside its local native recording; core represents that edge as a segment wait.

Maintain maximal writer/reader frontiers per proven byte span/subresource and
descriptor epoch, retaining the union needed by both conditional outcomes. Unknown
uses fall back to the root. A linear chain therefore retains a frontier, not O(N²)
historical conflicts. Canonical root ownership is unchanged. Composite AnyWrite
reservations remain conservative where membership is dynamic, with cost reported.

For alias reuse, every maximal old user must precede every minimal new user, or
an equivalent join/entry. Interval position in a topological list is insufficient:
two independent readers can finish in either order. sharing! adds those edges and
revalidates queue waits/scheduling. Every command rewrite/split subsequently rebuilds
affected hazards, costs, lifetimes and summaries before recording.

### 16.5 Derived acceleration versions

Core models geometry writes as versions and inserts a matching build/refit before
the first dependent AS read in a graph, including a write and trace in one segment.
A fixed recorded build can execute every replay without host rediscovery. Native
representation-only builds may execute in a disabled region only if logical query
results are preserved and the count/activity/topology constraints remain valid;
otherwise choose an eligible kernel/native region construction before recording.

External writes publish transitive BLAS/TLAS dirty versions per accepted segment.
Only an accepted build consuming that version removes it. A build elsewhere in the
graph, a later geometry write or a future unsubmitted build cannot justify clearing
it. Failed host/preparation stages therefore retain the updates needed by another
graph. Member/backreference snapshots and accepted tokens use the existing ordered
locks and retirement rules; no per-backend dirty/version owner is added.

#### Acceleration storage, growth and maintenance

Logical BLAS/TLAS lifetime, source-array backing, native AS bytes and operation
workspace are different resources under one core allocator/retirement policy.
Retain geometry/transform inputs for future builds/refits and shader reads; a
native build having copied some inputs does not revoke logical resource holds.
Native binding generations additionally retain the exact child BLAS objects,
instance mapping, descriptors and table data they may traverse. Logical removal
alone cannot release a child still named by an old native TLAS/recording.

| Change | Required action |
|---|---|
| Vertex/AABB coordinates or instance transforms change | Update/refit only if creation flags, topology/activity, source/destination size and native rules permit; otherwise Build. |
| Primitive/geometry/instance counts, index topology, geometry kinds or deleted membership change | Revalidate the exact native build signature; use Build unless update legality is proved. Backing spare alone never authorizes Update. |
| Logical instance count changes within a prequalified padded bound | Update can reuse builtcount only under the existing active/mask-zero/valid-geometry rules; otherwise Build. A zero logical count still has a valid native construction or skips traversal safely. |
| Host input storage grows or a BLAS native address/object changes | Refresh prepared descriptors/instance records and invalidate every parent native TLAS binding that embeds the old child. Propagate transitively; bounds unchanged is insufficient. |
| Required native bytes exceed the current destination | Acquire a replacement from core before publication, build/copy by a legal native operation, then publish its generation/dependencies. Preserve logical handles/ranges. |
| Logical shrink/deletion | Rebuild when needed, but retain usable capacity by default. No automatic shrink/compaction every frame. Removed members survive through old generation leases/uses. |

Counts/capacities bound inputs and prepared native records; AS storage size is
queried from the full geometry/layout/count-bound/build-flags signature. It is not
sizeof(input) or a linear guess from instance count. Native facts include storage
bytes/alignment/backing class, separate build/update scratch bytes/alignment and
legal update/copy/compaction modes. Core caches requirements by that complete
signature. One common bounded growth policy can reserve count headroom to amortize
reallocation; its padding must remain native-legal and its byte cost is budgeted.
Do not equate array growth/reserve with reallocating an opaque AS by memcpy.

An eligible in-place update/build waits on every old reader/writer of that native
storage. Under the normal queue/resource dependency graph it needs no device-wide
idle or host wait under locks. An eligible out-of-place build may overlap old
readers when the shared budget/scheduler selects that construction. Preclaim
allocation/preparation failure preserves the old usable generation and dirty
versions. An accepted maintenance/build prefix publishes tokens/holds even if a
later phase fails. Only the matching accepted binding build consumes its covered
versions; ambiguous native failures use existing quarantine rules. Publish no new
address/member snapshot without its readiness token. A parent may keep using its
old child generation until an ordered parent replacement is ready; never retarget
its embedded reference in flight. Idle recordings holding an old generation must
be invalidated/rebound before that generation's hold is dropped.

Build/refit scratch is operation workspace, not permanently reserved per BLAS or
per native binding. A binding stores requirements; core places scratch from the
same run arena/eager temporary pool as other workspace. Alias it only after all
build/update users finish; simultaneous builds need disjoint live workspace or
explicit serialization. Use update-scratch requirements for an Update, build
requirements for Build, and a proven maximum when one prepared slot serves both.
Do not retain max(build,update) for every long-lived scene object by default.

Buffer/heap-backed and opaque native AS objects are separate native allocation
facts, not separate memory policies. Buffer-backed AS uses compatible committed
storage; an array's sparse virtual reserve is not automatically legal AS backing.
For opaque/heap-native allocation, core admits/account budgets before creation and
retains/retire its native allocation ticket. Report known sizes and unknown driver
overhead separately; do not double-count a heap/region and its placed object.
Native wrappers create/bind/destroy supplied objects but own no allocator/cache/
eviction strategy. Residency registrations leave at retirement, not through an
uncollectable strong-reference/finalizer cycle.

compact!(as) is explicit scheduled maintenance of the same logical BLAS/TLAS.
It does not change geometry counts, instance IDs, capacity promises or bindings'
semantics. Core chooses a qualified native compact-copy or equivalent smaller
rebuild; an already tight representation may remain unchanged. A native size-result
query, if needed, is tracked work followed by a visible metadata/host checkpoint
outside locks; it is never submitted during pure compile/requirements/recording.
Charge old plus destination bytes, workspace and unreclaimed generations at peak.
Do not free the source first to make a compaction fit. Preserve old storage on
preparation failure. Publish the replacement with a readiness token, repair parent
references/descriptors by the common version protocol, and retire the source after
all old users. A compacted destination is refittable only when its actual flags,
provenance and destination/scratch sizes satisfy native Update rules; otherwise a
later edit rebuilds/replaces it. Compaction is not promised to preserve cheap refit.
No native compact path may silently memcpy an opaque representation.

Pressure does not initiate compaction or a surprise size-download checkpoint.
Release bounded idle binding/recording caches first. Reconstructible AS bindings
can be discarded only after active uses/leases end and idle plans are invalidated;
retain source/version metadata and rebuild before reuse. Native AS is not an array
HostCopy blob: eviction/restore uses reconstruction or an explicitly qualified
native serialization scheme. Report old/new overlap, scratch, pinned children and
all live cached binding views. free!(as) drops its creator hold; graph/member/native
generation holds keep it usable for already declared work. Last-hold release
retires native objects, tables and residency after all tokens, not at host return.

Qualify the existing MWE-05 lifecycle profile with growth beyond a native bound,
in-bound deformation, topology rebuild, shrinking/deletion, one BLAS in two TLASes,
relocation/compaction while an old trace is pending, failed replacement and a second
layout binding still dirty. MWE-03 supplies controlled accepted-prefix faults;
MWE-08/B5 supplies allocation/steady-state memory and frame cost. Check peak and
post-retirement bytes, not just final object count. This is one core policy for
Metal/Vulkan/software paths; native legality comes from the actual providers.
Native size/refit/compaction facts are grounded in the
[Vulkan AS contract](https://docs.vulkan.org/spec/latest/chapters/accelstructures.html)
and [Metal AS sizes](https://developer.apple.com/documentation/metal/mtlaccelerationstructuresizes).

### 16.6 Imported allocations and explicit device communication

Import validates native context/device/generation, byte bound and strides, retains
the allocation owner itself and canonicalizes payload aliases in a weak per-device root
registry. Stream readiness and later lease/completion edges are tracked. External
mutation/reallocation outside this handoff is a boundary violation; retaining only
a mutable vendor header does not protect its old allocation. Imported storage is
nonrelocatable/non-evictable, cannot grow beyond its bound, and is never pool-freed.

Canonical payload identity and logical header identity are separate. Independent
foreign arrays/views/reinterprets may cover the same allocation with different
shapes, offsets, strides or element types. They share payload byte hazards, holds,
physical ownership and restrictions, not a dimensions cell solely because the
allocation key matches. Ordinary Mantle views retain their logical parent state;
independent logical headers keep independent size metadata. Resize effects update
aliases of that logical root, not all headers that overlap the same bytes. External
vendor header dimensions are not silently patched. A new foreign import captures
its supplied header geometry, while sharing canonical byte hazards; reuse the
existing MantleArray to share its GPU-written logical size across graphs. A withview
export requiring exact host dimensions takes a coherent metadata/version checkpoint
outside locks and retains its matching storage lease; zero-copy payload is not a
zero-synchronization promise. Capacity is not a substitute for those dimensions.
An arbitrary external callback is conservatively read/write; a qualified closed
library recipe may state read-only effects. readonly aliases cannot be exposed
through an unchecked writable callback. Data uses key the canonical
allocation/byte span; metadata uses key the relevant header/epoch. One core
frontier/retirement engine handles both; there is no separate import scheduler.
Element/byte conversions and address spans use checked arithmetic and alignment;
never treat a capacity counted in one header's T as another header's capacity.
The captured native allocation/version must back every published view span. A
readonly or outstanding lease restriction belongs to the aliased payload owner;
a second imported header cannot bypass it.

Explicit asynchronous transport alone may name roots in different domains. Order
plan/root lock IDs by (device identity, local ID), and never acquire a second Graph
lock while holding these locks. Core transport owns source snapshot, retained
buffers, any host-cut bridge and destination publication. A peer path can avoid the
host copy; unsupported peers use the shared bridge with visible costs. No implicit
mixed-device arithmetic or universal zero-copy claim is introduced.

Communication recipes declare communicator participants, input/output/root ranges,
readiness/completion and collective sequence. Each rank must enqueue matching
participation; cancellation/failure becomes communicator failure, never a locally
successful token while peers wait forever. Native qualifications include launch
ordering and library deadlock restrictions; no unconditional nonblocking promise.
See [NCCL graph constraints](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/cudagraph.html).

### 16.7 Conditional render reset and image subresources

Condition the whole clear+draw region. The active path resets and redraws, the
inactive path preserves the old image. A native region candidate or core conditional
clear-draw candidate implements this. The latter uses Keep loads, replacement color
writes, Always depth/write and the exact stencil/reset/sample/rectangle rules needed
for the selected attachment. Gate the clear and following draws by the same GPU
predicate. Eligibility must prove the clear conversion/format semantics match the
ordinary clear; unsupported combinations decline before record. Measure tile/bandwidth
cost instead of claiming all native pass-level work disappears.

Typed image views share root ownership and refine mip/layer/aspect ranges. Public
mip/layer/range indices are Julia one-based; sample lod=0 is a continuous level
offset relative to the view's base mip, not a Julia array index. Infer the aspect
from an unambiguous format rather than defaulting a depth image to ColorAspect.
Float32 retains the existing depth-only attachment convention (§8); RGBA color
volume examples must not silently construct a 3-D depth image. Window-sized Image
construction has the same explicit lifetime as fixed-size Image construction. Native
requirements validate formats, samples, storage reads/writes/atomics and feedback;
sampling/storage intrinsics belong in KI, not Mantle-to-each-compiler shims.
3-D/array/cube/mip paths, storage atomics and feedback have individual qualification
gates; existing 2-D sampling cannot close them. Unsupported raw semantics reject;
eligible high-level sampling can choose a core implementation with measured costs.

### 16.8 Indirect streams and modern launch requirements

DrawCommand is a typed semantic command, not user-authored native bytes. Core maps
counts, first index/vertex, signed base vertex and base instance into native wire
layouts and checks declared bounds. A GPU producer publishes live command count
through normal array size metadata. ExecutionSet bounds all selectable programs,
layouts and reachable resources; generated work cannot hide arbitrary pointers.
Validate/fuse command checks before native execution where producer safety is not
proved. Native multidraw, Metal ICB and generated-command candidates share those
effects, dependency, lifetime and counter semantics.

References: [Vulkan generated commands](https://docs.vulkan.org/features/latest/features/proposals/VK_EXT_device_generated_commands.html),
[Metal GPU ICB encoding](https://developer.apple.com/documentation/metal/encoding-indirect-command-buffers-on-the-gpu).

Reuse KI matrix/tensor facts. Runtime requirements additionally describe cooperative/
cluster launch geometry/residency, dynamic shared memory, asynchronous descriptor
resources and device-launch/region legality. config/compile keys and effect inference
include these facts. No native API name implies availability and no arbitrary
workgroup split may invalidate cooperation. F34's native PTX/Vulkan evidence carries
forward; production integration remains a candidate/qualification gate.

### 16.9 Explicit stochastic state

RNG owns a core counter/seed resource. call!(target,rand!,rng,a) declares reads/
writes at that target; ordinary rand!(rng,a) is eager. Destination arrays carry no region. Replay and executed loop invocations
advance it; disabled work does not. Recompile preserves it; seed! schedules explicit
reset. Private tuning copies state without consuming user randomness. Promise exact
counter stream behavior only for the qualified algorithm/version/type; floating
distribution transforms may have their documented cross-device numerical tolerance.
Shared RNG instances order through their effects; independent streams are explicit.
A logical random-fill operation reserves its declared counter interval once,
indexed by logical elements rather than padded threads/queue scheduling. A zero
or disabled operation consumes no interval. Pure local/counter RNG values inside
user kernels remain ordinary values; existing Hikari sampling need not be replaced
by global atomic RNG state. Preserve its baseline numerical contract.

### 16.10 Qualification and scope

Every MR-01–24 resolution maps to an open ledger item and independent acceptance
case in plan-reconciliation.json. Required cases include slow parallel alias readers,
same-segment deformation/trace, failure before a pending build, nested GPU size
publication, candidate-domain growth, deadline-with-live-dependencies, conditional
shadow reset, imported-alias lifetime, ordinary Group broadcast/library composition,
RNG replay and actual FEM/particle/LLM programs. Existing consumer evidence remains
baseline evidence; this specification does not close replacement/native gates.


### 16.11 Representation and performance qualification

The architecture's Compile and runtime performance rule applies to declaration,
compilation, native encoding and replay, measured separately. Array rank/element
layout, device/compiler configuration and kernel callable/argument ABI justify
concrete specialization. Graph/node/resource identities, raw dynamic sizes, maximum
loop iterations and runtime binding indices do not. A selected kernel may use a
value-specialized variant only when its validity domain/cache bound and measured
benefit are explicit; misses have a correct generic implementation or trigger
ordinary supported compilation, never incorrect reuse.

Keep graph topology and region bodies in indexed runtime structures. RegionNode's
typed recipe/argument payload is not a tuple encoding its child graph. Core owns
the same traversal over typed payload tables for every device; a function barrier
may dispatch once to a concrete recipe/native encoder. Do not trade that boundary
for per-node Julia specialization on topology, giant union inference or Symbol/isa
switches. Any heterogeneous cold declaration container must stop before hot
argument packing, numerical loops or unchanged baked submission. Walked encodings
still use the single core typed command walk; no per-backend interpreter is allowed.

An unchanged baked run consumes already prepared immutable recordings/bindings
without visiting its internal command nodes. Boundary work can still scale with
modified inputs, written-root/inbox summaries, native submissions, host cuts,
present pieces or snapshot transfers. Measure those counts separately; this is
not an unconditional constant-time promise for a graph with growing external
outputs or submission boundaries. Avoid
unnecessary host allocation, scalar download, resource scan, inference, recording
or tuning on replay. Kernel/device code retains concrete layouts and no dynamic
host-language dispatch. Native handle refresh follows invalidation edges only.

MWE-08 and the batch benchmark record cold load/first compilation, warm graph
compilation/recording, method-instance/cache growth, plan/workspace/peak host bytes,
steady host submission and GPU completed time. Compare equivalent numerical work,
same precision/layout and enabled capabilities, with warmup and interleaved A/A
noise. Near-linear frontier storage on chains does not promise a linear general
DAG: report interval fragmentation, fanout and conservative composite edges too.
Use actual consumer latency/throughput to settle competing representations; a toy
microbenchmark alone cannot justify a runtime regression. Search shared caching,
typed function barriers, bounded variants and structured native regions before
accepting a compile/runtime tradeoff. If unavoidable, runtime performance wins.
The fixed catalog and batch schedule in recording-plan.md govern when to collect
this evidence; the eight specifications and this document are not runtime passes.


#### RT regression acceptance

F124–F126 cannot close on expressibility/correctness alone. Compare against fetched
Mantle 3e9943f / Hikari 407f9b2 / Makie 42e6355fb (or a recorded newer matched
baseline), on the same actual device/compiler with identical completed work,
precision, sample counts and alpha/media semantics. An incorrect opaque shortcut
is never a filtered-render baseline. Source timing comments are not measurements.

The cheap path is a structural requirement before timing:
- Queries without hit tables allocate/upload/bind no function table or SBT. Query-
  only/wavefront consumers create no program-layout-specific TLAS binding views.
- Nothing result attributes occupy no result storage. Do not retain max-sized
  attributes for every possible group or expand compact Hikari hit records into
  full surfaces. Candidate context/attribute packing must be eliminated when unused;
  inspect generated IR/native resource statistics rather than assume inlining.
- Preserve eligible opaque closest/occlusion complete-result native constructions
  under the shared selector. Selection/requirements analysis is compile-time work;
  no device-family tests, table interpretation or host callbacks per ray/replay.
- Stable scene/program/selector versions reuse bindings/recordings. Group/type
  changes affect only relevant entry-point code/resources; keep current lobe/cut-set
  specialization and record method-instance/cache growth explicitly.
- Native program tables deduplicate identical geometry/instance selection slabs.
  Do not unconditionally allocate instance-count × geometry-count × ray-type-count
  records. Dynamic mappings charge actual bounded storage/update traffic and can
  select a qualified shared query/dispatch construction when native expansion costs
  more. Native binding-view count/bytes are bounded/accounted by core caches.

Use MWE-05 capability profiles and MWE-08 compile/resource measurements at their
existing batch boundaries, then actual Hikari/RayMakie B5 gates. Cover opaque
closest/occlusion, alpha camera/transparent shadow, procedural reports/attributes
and animated refit/selector updates. Record cold/warm compile and cache footprint,
GPU registers/private stack/spills where exposed, table/TLAS/scratch bytes, host
allocation/submission work, steady-state latency and complete frame throughput.
Do not start a full suite after each edit or replace end-to-end gates with a toy
kernel score. Costs of newly requested features are reported separately; callers
using existing semantics must not pay unused generality costs.

Interleave warmed A/B runs with A/A noise controls, using the same inputs/seeds and
completed GPU work. A repeatable regression outside that measured noise blocks
closing the affected profile; investigate codegen/packing/selection and improve the
shared implementation. Do not hide it in an aggregate, lower quality, remove alpha
checks or restore a parallel backend renderer. If a tradeoff remains, record its
workload/device impact and alternatives; it is not a no-regression pass. Unavailable
hardware, unresolved measurements and new unqualified native paths remain open.
These are acceptance requirements, not proof that the target API has no overhead.

### 16.12 Consumer composition contracts

These boundaries complete the interface requirements for the existing F115–F122
gates; they do not claim implementation, measured performance or passing consumer
ports. Domain algorithms/data models stay in their libraries and use one common
Mantle operation/resource/control layer. api.md owns the entry signatures.

**Operator and sparse composition.** Ordinary mul!/ldiv! recording requires a
qualified code-owned definition for the operator types, or a domain builder emits
the equivalent kernels explicitly. Never invoke an arbitrary eager Julia method
and assume its work was captured. The extension follows the same typed recipe,
return/effect/shape and shared-candidate protocol as core operations.

A supported sparse composite must state scalar/block shape, one-based logical
row/column convention, index width, duplicates/coalescing rule, symmetry/triangle
and diagonal interpretation, sortedness/pattern identity, live entry count and
root/view mapping. These are layout/semantic facts for shared algorithms and
candidate eligibility, not native-vendor API choices in consumer code. Existing
Julia/library composite types may hold MantleArrays; no second device-array family
or universal sparse format is required. Conversion is an explicit shared recipe
with real workspace/traffic cost. A native candidate requiring unavailable exact
host counts cannot silently read GPU lengths back to become eligible.

For AScanner's block COO, values/row/column arrays use the same logical triplet
count and block indexing convention. Compaction/conversion computes that count
once; one publishing invocation resizes all three after checked scatter, including
zero input. Every following consumer depends on all three payload/size writes.
It must not zip independently changing lengths or call an Integer-only host resize
on a GPU counter. Index-width overflow, invalid indices and insufficient output
capacity are checked before invalid memory access. Duplicate/symmetric operations
must match the declared mathematical operator, including not doubling diagonals.

Symbolic-analysis reuse depends on the declared pattern/shape version; numerical
factor/preconditioner reuse depends on value/configuration dependencies too. Shared
resource/derived-work ordering supplies those dependencies; each backend must not
invent its own cache/invalidation policy. A GPU-written changing pattern does not
become an exact host-known shape through a cached integer. Matrix-free operators
need no assembly/conversion merely to fit the graph interface.

**DNN allocation/escape map.** The source-level migration classifications are:

| Existing emitter role | Target ownership/lifetime |
|---|---|
| make for nonescaping intermediates, including zero-size outputs | MantleArray(target,...;scratch=true); unused/zero-byte scratch is legal and allocates no native payload. Do not preserve the old zero-size persistent-buffer workaround. |
| make for escaping output IDs, saved cross-run activations or session results | Persistent output/caller-supplied destination; preserve its identity and hold across plan replacement. |
| scratch and activationbuffer prepared representations | Explicit scratch with root/version-aware liveness across all consumers; a returned owner-graph handle/view remains legal. |
| Bound input/output resources | Reuse the exact supplied root/lifetime; do not promote/demote, copy or adapt to Group. Reject an actual scratch escape. |
| Declared views/tuple projections | Ordinary metadata aliases retaining the classified parent; a view does not turn scratch into persistence. |
| residentweights/imported arrays | Persistent existing allocation or explicit device import/copy. Inference may opt into readonly/streamed source retention; mutable training weights must remain writable persistent data. |
| MemoryBank key/value/history/sensory state | Persistent session resources, with GPU live bounds where applicable; never frame scratch. |
| Noise state | Explicit persistent qualified RNG resource, reset only by an explicit seed/reset operation. |

DNNKernels make/scratch/emitgraph!, activationprep activationbuffer and memory
MemoryBank/recordmemread are the concrete source seams. Constructor substitution
must follow these roles, not replace every old resource with persistent arrays or
infer lifetime by escape guessing at runtime. Keep exporter/model liveness hints
as semantic input, not a replacement allocator/hazard policy. Emission failure
cleans up resources/contributions owned by that emitter, never caller-bound roots.
The existing F116 module-composition/memory/native checks qualify the actual port.

**Simulation control and parameters.** PrintSim's Graph-only launch/recording
classification must support Group by dispatch; otherwise it can accidentally take
the immediate convergence-readback branch during construction. Each nested solver
loop resets its predicate/state at the specified enclosing invocation; CG and line
search are sibling regions where the algorithm says so, not a fabricated product
of their bounds. A parameter stored in GPURef is a per-run input only when all
consumers, including derived setup, support that dependency. Current FEM/coarse
data are timestep-dependent: a dt change requires affected setup recomputation/
rebuild or explicit rejection, preserving retained physical state as appropriate.
No generic scalar update can repair a stale factorization. CPU diagnostics use
explicit phase/controller host boundaries; do not hide them in a GPU-controlled
region. Native versus expanded nested solvers require realistic compile bytes,
slot/workspace memory, stop/predicate overhead and completed-work measurements;
one native launch or one tiny loop does not establish a fast simulation path.

**Frames, picking and snapshots.** A window capture is an explicitly recorded
Array output in its presenting graph, after the final window writer and before
present; it copies that acquired generation while held. Array(win) does not wait
for an unsubmitted future frame. A requested run that cannot acquire/render/present
its frame before accepting the copy rejects that request. If the copy was already
accepted, preserve its token/staging and follow the ordinary prefix/failure contract;
its pixels describe the rendered image, not proof of scanout/presentation completion. An eligible native copy or shared
capture construction must satisfy surface usage/format requirements before
recording. No new backend output scheduler or consumer surface workaround exists.

A renderer captures its color/ID/depth outputs after their matching producers and
retains the frame's immutable extent, pixel scale, format and plot-ID map alongside
the requested results. Cache rebuild/rebind cannot attach a new plot map to old
pixels. Subsequent frame writes follow snapshot reads. Frame-owned outputs and
frozen metadata must remain held until capture finishes; arbitrary external
writes are not an atomic multi-array snapshot promise. Image snapshot axes use
Julia source order and the same coordinate convention as device image access;
native pitch/channel/format conversions belong to the common copy/layout protocol.
The Makie adapter owns its documented CPU layout conversion once. A picking ROI
uses a source view/region copy or a small device extraction kernel into an ordinary
array, followed by the same recorded Array output; it does not require full-frame
readback or a new Mantle picking API. Qualify actual WSI resize/retirement separately
from headless color/ID agreement.

**LLM cache layout.** Bonsai's current kv_page rounds contiguous capacity; that is
the initial contiguous decode/prefill gate, not evidence of paged attention. A
logical paged cache belongs to the LLM library: ordinary device arrays hold cache
pages, block tables, positions and quantization scales, and its target-aware
builder/qualified attention definition declares those resources. Logical paging
does not require sparse hardware residency. Native attention candidates must
support that layout/precision or decline to the common kernel candidate. Explicit
page-table/scale version ordering and page leases prevent reuse while any old
reader can access the page; aliases follow the same root/span hazards. GPU-selected
membership must be covered conservatively by the declared resource set. Record
any lost concurrency from coarse root spans, page traffic and per-token overhead;
packed contiguous fixtures cannot close the shuffled-page/continuous-batching
profile. No Mantle-wide KV-cache class or per-backend cache manager is introduced.

**External AD and saved state.** Differentiate numerical definitions or an external
AD library's model before emitting forward/backward operation builders. No promise
is made that differentiating run!, mutable node bookkeeping, native handles or a
ReadbackFuture differentiates the GPU program. A library's ordinary-function rules
can emit qualified operation recipes through the same target-aware boundary.
Forward activations needed by backward in the same run may be explicit scratch;
values crossing runs/graphs must be persistent or copied into persistent storage.
Effects/holds preserve the saved version until its backward reader; an optimizer
or another forward pass must not overwrite it early. Gradient arrays, optimizer
state and RNG use the normal resource/version model, with no backend AD policy.
The training gate needs an actual external AD integration and independent gradient
reference; handwritten backward kernels or inference/optimizer-only runs alone
are insufficient evidence. This is an extension boundary, not a new public AD API.
