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

An access means an access to a root allocation, including a view, a cell,
argument memory, indirect commands, build scratch, descriptors and composite
members. The argument depends on sound resource discovery and access inference;
unknown code is conservatively ReadWrite, never silently omitted. Atomic-only
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

Freeze a portable CUDA/HIP baseline: kernels receive an immutable pointer to
plan argument slots, as the Vulkan/Metal plan does. Dynamic array descriptors,
GPURef values and logical extents are loaded from slots by the device wrapper.
A device without indirect launch launches the compiled capacity and predicates
work before calling the body. `when!` is an entry predicate; bounded repeat is
core-unrolled into an execution DAG, cut between iterations. GPU state is
re-read per iteration; empty bodies cannot skip collective barriers incorrectly.
Expansion precedes dependency analysis: a head writes plan-owned loop control,
every body node reads it, and the next head therefore follows all prior tails.
Each iteration has distinct command/argument slots; data/scratch resources stay
shared where declared. Compiled shader code is reused. Compile/record memory
grows with maximum iterations × body size; termination still incurs the
remaining predicated commands on this baseline. No constant-size native-loop
or early-termination performance claim follows. No host readback is inserted.
Native indirect/conditional nodes are candidates
when supported and equivalent. Tests must compare results for zero, one and
capacity counts; capacity-launch overhead is a performance qualification cost.

This does not permit host mutation of an in-flight graph executable or assume
HIP supports CUDA conditional nodes. Compile once against capacity; exceeding it
recompiles. Device slot loads are part of the wrapper, not backend-specific
consumer branches. The compile key includes this ABI and enabled feature set.

A budget is an estimate, not a hard execution-time guarantee. Core splits
repeat iterations and independently tileable candidates between submissions.
A candidate exposes `partition` only when tiles preserve its declared effects
and ordering. Reductions require an explicit combine, never arbitrary grid
cuts. Unknown-time or indivisible nodes cannot be guaranteed watchdog-safe;
report the node and choose a qualified alternative when available. No library
name proves bounded execution. Qualify heavy DNN/LLM workloads on display GPUs;
never promise universal timeout avoidance for arbitrary user kernels.

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
| `jobtimeout` | Seconds or Inf when no bound is known; an unknown watchdog bound is not proof of unlimited execution. Core applies its documented conservative submission estimate. |
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
Only reorder hints may become no-ops. No silent demotion of atomic loads/stores.
Sequential consistency also cannot be replaced by acquire/release: the latter
does not provide the requested total order. Vulkan does not support SPIR-V
SequentiallyConsistent semantics; absent a separately proved lowering, reject
that raw ordering before enqueue and omit it from atomicfeatures. Portable
algorithms may explicitly select a correct acquire/release implementation, but
must not weaken an arbitrary user kernel. CAS success/failure orderings remain
separate, and fences need actual lowering. F33's reproduced Lava demotion is
fixed: supported atomic loads/stores/fences retain their ordering, unsupported
sequential consistency fails compilation, and Atomix's requested order is
preserved. Default Atomix RMW requests sequential consistency too; changing a
caller to acquire/release or monotonic requires an algorithm-level justification.
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

`rayquery(tlas, origin, direction; tmin = 0f0, tmax = Inf32,
mask = 0xff, accept = AcceptAll(), intersect = BoxIntersection()) -> RayHit{Float32}` is a portable device
operation. It returns the closest accepted hit. Direction is not normalized:
origin + t*direction defines distance t. Invalid intervals/nonfinite origins or
zero direction are errors in checking builds, never native undefined behavior.
The ray mask uses eight bits; an instance is eligible iff its mask & ray mask
is nonzero. No facing cull by default. An opaque hit and an accepted nonopaque
hit have the same result contract. Pure `accept(hit)` may reject a candidate;
traversal order and callback invocation count are not specified, so accept must
not mutate queues, random-number state or other shared resources.

RayHit fields are `hit::Bool`, `distance::Float32`, `instanceid::UInt32`,
`customindex::UInt32`, `primitiveid::UInt32`, `geometryid::UInt32`,
`barycentrics::NTuple{2,Float32}`, `frontface::Bool`, `kind::UInt32`.
Indices are one-based when hit=true. On miss the indices and barycentrics are
zero, frontface=false and distance=tmax. A triangle has kind=0; its weights in
face-vertex order are (1-u-v,u,v). Equal-distance overlapping hits may select
any tied primitive: require equivalent accepted geometry, not identical native
traversal order. Software and hardware candidates obey the same mask/interval/
filter and transform semantics within an explicitly tested numeric tolerance.

customindex is Mantle's stable range slot. rangeoffset(tlas,slot) is the count
of instances preceding that range, so instanceid-rangeoffset is one-based within
the range. Native custom indices encode slot-1 and core decodes on shader access;
the same conversion applies in native RT callbacks. Range slots cannot exceed
native custom-index representation; check the actual native limits, never a
constant inferred from one workstation. No scene-wide geometry concatenation.

TLAS(dev;raytypes=1) fixes the ray-type count for its native instance-record
layout. Pipelines sharing it must use that count; core validates this before
enqueue. A different ray-type layout requires a different TLAS. Default one
covers existing consumers and keeps native record offsets independent of which
pipeline last used the TLAS. Check (groups*raytypes) against native offset limits.

For a kernel-written instance total, every portable build path uses a fixed
capacity count and valid mask-zero padding, including native count-from-memory.
Its count cell carries capacity, while logical offsets/lengths remain separate;
all Updates keep builtcount. A native exact-count optimization must prove legal
updates or choose builds without host readback, and qualify against this baseline.

A pipeline is an immutable, concretely parameterized Julia value:

- `RayTracingPipeline(;raygen, closest_hit, miss, any_hit=nothing,
  payload=Nothing, maxdepth=1)` is the convenient one-ray-type form. A tuple of
  closest_hit callables denotes multiple logical hit groups. Each shader receives
  the same explicit argument tuple; remove the old chit_miss_take_args switch.
- `TriangleHitGroup(closest_hit; any_hit=nothing)` and
  `ProceduralHitGroup(intersection, closest_hit; any_hit=nothing)` are values.
  The general constructor accepts a rectangular hit-group × ray-type tuple of
  these values, one miss callable per ray type and a concrete isbits payload
  type. Old payload symbols such as :f32_7 become NTuple{7,Float32}.
- `push!(tlas, blas, transforms; hitgroup=1)` assigns a logical one-based group
  index to a range. `hitgroup!(tlas,range,index)` pends a record update; it does
  not expose an SBT byte offset. Every tracing pipeline must have compatible
  geometry kinds and that group index. Invalid indices fail before enqueue.
  Two pipelines sharing one TLAS must agree on these logical group meanings.
- Core lays out records in group-major, ray-type-minor order. Native instance
  record offset is (hitgroup-1)*raytypes, per-ray offset is raytype-1, and
  geometry stride is zero: all geometry in that instance uses its logical
  group and exposes geometryid to the callback. Require a matching procedural/
  triangle group; mixed-kind native BLAS needs separate compatible instances.
- Core computes aligned SBT regions from native handle size, handle alignment,
  base alignment and maximum stride, obtains handles through
  `shaderhandles(dev,native)` and uploads through ordinary pool-owned resources.
  No shader arguments are duplicated into SBT records: callbacks read the same
  plan slots. The SBT and pipeline are held by recordings and retired by tokens.
  Metal function tables expose equivalent logical groups; a software candidate
  dispatches the callable tuple. The backend reports native requirements and
  creates native objects, never builds a second material/range policy.

`traceray(origin,direction;raytype=1,tmin=0f0,tmax=Inf32,mask=0xff)` is the
portable callback-invoking operation inside a raygen/hit program. Core lowers
logical raytype to record/miss indices and the KI trace/reorder/execute sequence;
SER is selected by capability without consumer extension branches. Native operand
limits are checked: Vulkan record offsets are limited to 16 ray-type choices,
not an arbitrary UInt32 offset. Recursion is bounded by program maxdepth and the
selected native/software implementation. Existing raw KI rt_* operand builtins
remain available for qualified shader implementations; their numeric native
conventions are not the portable traceray interface. The program's typed payload
is compiler-owned stage IO; no argument resource or native pointer may be hidden
in it outside the declared resource set.

The trace's accel argument binds the program's TLAS; it is not another element
of the explicit shader args tuple. `traceray(origin,direction,payload;...) -> P`
returns the pipeline's concrete payload type P after the selected closest-hit
or miss callable returns P. Omitting the payload is legal only for payload=Nothing
and supplies nothing. Raygen's return is ignored. `raypayload()` reads the current
stage's incoming P, `rayhit()` supplies the decoded RayHit in closest/any-hit,
and `currentray()` supplies the immutable origin/direction/interval/mask in hit,
miss and intersection stages. These are compiler stage IO, not mutable global
state. Hit/miss callables still receive only the same explicit args tuple and
return P; they may access only discovered resources. The Nothing callbacks in
HK-10 therefore return nothing. Any-hit is a pure Bool filter of the current hit;
intersection returns nothing or `RayIntersection(t, attributes::NTuple{2,Float32})`.
Out-of-interval/nonfinite reports are rejected. The portable filter never mutates
payload or relies on candidate invocation order; qualified native-only any-hit
controls remain below this interface. This fixes return/IO semantics as well as
SBT layout, without a consumer-managed payload buffer or hit-object protocol.

The SBT formula/alignments are constrained by the
[Vulkan ray-tracing contract](https://docs.vulkan.org/spec/latest/chapters/raytracing.html).
Native facts add `sbtproperties(dev) -> (handlesize,handlealign,basealign,maxstride,maxrecordoffset,maxrecursion)`
and `shaderhandles(dev,native)`. These are capability-gated verbs.

Procedural BLAS uses `BLAS(dev,bounds)` with a concrete AABB element format
(minimum,maximum Float32 triples). Its intersection callable reports finite t
and up to two Float32 attributes within the supplied ray interval, or no hit;
core/native lowering owns candidate/report/accept traversal. rayquery applies its explicit intersect callable to AABB candidates; the default
BoxIntersection treats the boxes themselves as primitives. A procedural pipeline
uses its group's intersection callable, so a sphere query and sphere pipeline
must supply the same geometric operation. Bounds updates are refits,
geometry-kind/count changes are builds. NaN padding is not a portable
inactive-state mechanism. A pure callable is shared by software traversal and
native intersection stages; it does not manage native hit-object or SBT state.

Core computes the union of resource uses across raygen, every hit group, miss,
intersection and any-hit. A buffer touched only in closest-hit still receives
holds, readonly validation, placement and barriers. Native ray-pipeline support
is optional; portable ray queries and ordinary shade-kernel candidates remain
available when ray pipelines or procedural native stages are absent. Arbitrary
native-only shader intrinsics are not promised software emulation. SER is a
semantics-preserving hint: a no-op is correct; silently skipping an any-hit alpha
test is not. Select the native Hikari pipeline only when its alpha/media/light
semantics match the requested operation. Its current opaque-only RT shortcut is
not an acceptable reference for alpha-textured scenes.

Hikari has both typed shading queues and typed closest-hit groups today
(Hikari/src/integrators/volpath/rt-pipeline.jl:530–575). A new material type may
specialize either or both. Core's ordinary pipeline cache keys include all group
callable code identities, payload and layout; delete the consumer's mutable
_VP_RT_PIPELINES cache during migration. Do not force all shading into one
monolithic kernel or erase the existing RT specialization path.

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
All attachments in one pass must match extent and sample count. copyto! from a
multisampled colour image to a matching single-sample image declares a resolve
(CompiledCopy(ResolveImage(...))), never a shader memcpy or a host transfer.
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

Core stores `Candidate{F,R}` values with the implementation callable, requirement
predicate, implementation version, cost model, workspace description and optional
partitioner/region compiler. All candidates describe the same operator semantics,
including aliasing, numeric accumulation type, epilogues and output ownership.
An `implements` attribute alone cannot excuse different alpha tests, overflow,
rounding or memory ordering. Filter by requires and validfor(args) before reading
a measured score. Optional library candidates additionally require recordability,
caller-owned workspaces and the current target kind (a repeated Group permits
kernel candidates only). Unsupported candidates are declined before compilation.

SelectionKey is an immutable value of (operator identity/code, argument element
types, logical shapes or explicit bounded shape classes, strides/layout,
base-alignment classes, alias relationships, target execution kind,
compiler/feature/ABI key, candidate-set version). Addresses and objectid of arrays
are not keys. Plan capacity/shape dependencies guarantee continued eligibility;
a move preserving those classes needs no reselection. Growth outside the bound
or a changed code/config/candidate version invalidates the entry and marks the
plan for recompile. GPU-written live lengths are represented by their capacity
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
that device's own HostCall or foreign lease throws before closing admission;
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
`trim!(dev)` releases empty caches after passed retirements without waiting for
live uses. It never unmaps an array's live pages (trim!(array) is distinct).
Memory/leak fixtures wait/sweep/trim before comparing physical reservation;
normal run latency is not forced to pay rawfree/rawalloc for every reusable block.
Under pressure, empty caches are trimmed before evicting live resources.

`devicecopy!(dst,src)` is the explicit cross-device operation for device arrays,
including views. The portable baseline takes one consistent Array(src) snapshot,
then schedules that complete snapshot into dst through the ordinary store path
and waits for its destination application token. It returns dst after completion,
without idling unrelated work. It can allocate host memory proportional to the
copy and fails before modifying dst if the snapshot/allocation fails. Different
sizes/types follow copyto!'s validation/conversion rules; no silent relocation
of later operations. Graph-array arguments and implicit mixed-device arithmetic
remain errors. Qualified peer/external-memory candidates may replace staging
while preserving source snapshot, destination publication and resource holds;
no zero-copy or nonblocking claim is made for arbitrary driver pairs. This
baseline deliberately reuses existing read/store logic rather than inventing a
cross-driver graph scheduler. Graph composition remains within one device domain.

Streamed weights keep `streamed=true` as the keyword. Their retained host source
is immutable by the same contract, and each graph declares a transient resident
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
ordinary argument slots and GPURef data slots. Only head/prepare, TLAS record/
offset preparation and explicit lengthof writers may access source Cells;
ordinary launch/raster/shade kernels read prepared slots. Cell pointers cannot
escape through an opaque native argument or an untracked raw-pointer cast.
The public launch API does not expose source-cell addresses except lengthof.
A backend must qualify the check against its compiler output rather than merely
stamp every ordinary kernel with "no cell reads". THR-04 and KL-10 define the
executable acceptance cases; debug checking is off on performance runs.

## Graph-declared readback snapshots

`Readback(array)` and `Readback(graph, array)` declare a snapshot at their graph
position and return a one-shot future. The copy has an explicit source-read and
destination-write dependency, so source liveness reaches the copy and later
source writes follow it. Readback destinations do not alias transient storage.
Vulkan's recorded copy includes a transfer-write to host-read visibility barrier;
coherent host-cached staging is read only after the captured execution token.
The shared-memory implementation uses a normal graph copy kernel.

Each output has a reusable staging slot owned by its live plan. Each execution
with requests publishes a snapshot carrying that run's token and a hold on the
slot's Buffer. `fetch` waits for that token and makes an owning CPU array; no
submission is made. Before replay overwrites a previous outstanding snapshot,
core materializes it. `free!(plan)` remains nonblocking: it rejects unsubmitted
requests and drops its slot hold, while submitted futures keep their storage
until collection. Run failures reject requests rather than leaving waiters
blocked. A moved native source binding is recompiled/recorded before replay;
a changed logical shape is rejected by this fixed-shape implementation.
