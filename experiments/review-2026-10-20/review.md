# Mantle refactor document review — 2026-10-20

Review base: `a81bf41`, branch `sd/lava-refactor`. This continues the saved
`notes-plan.md`; that file and the seven source audits are working evidence,
not additional target-state documents. The target design is the requested docs, the ledger and the linked normative
execution-contracts.md companion. No runtime implementation was changed.

The architecture has useful foundations: explicit device ownership, one
operation body for eager/graph execution, compile-owned decisions, recorded
submissions and resource-based ordering. The reviewed execution boundaries now have explicit contracts, ownership and
adversarial gates. The corrections below close concrete contradictions;
native/compiler implementation and consumer qualification remain phase gates.
This describes the checks for the isolated phase-1 slice. It does not establish
that the full target documents are synchronized or that replacement native paths work.

## Corrected contracts and specification defects

| Finding | Evidence in the corrected files | Consequence and correction |
|---|---|---|
| Early phase gates silently skipped the new API | [docs/recording-plan.md:1723](/sim/Programmieren/VideoEdit/dev/Mantle/docs/recording-plan.md:1723); [test/suite/helpers.jl:55](/sim/Programmieren/VideoEdit/dev/Mantle/test/suite/helpers.jl:55) | Removed the phase-6 floor. Phase 1 must supply a working slice of the final public entry points and the suite's declared type/verb surface. Freeze its dependency order before implementation (P1.A23/F36). |
| View representation omitted strides and disagreed with resizing pseudocode | [docs/internals.jl:87](/sim/Programmieren/VideoEdit/dev/Mantle/docs/internals.jl:87); [docs/resizing-and-raytracing.jl:313](/sim/Programmieren/VideoEdit/dev/Mantle/docs/resizing-and-raytracing.jl:313) | One view descriptor retains root offset, dims and strides. Stores carry a contiguous span or a scatter mapping, with packed upload size and actual-element coalescing. Complex adjoints preserve conjugation (F37). |
| Dependency construction could add self-edges | [docs/internals.jl:647](/sim/Programmieren/VideoEdit/dev/Mantle/docs/internals.jl:647) | Merge alias uses and compare every use against earlier nodes before publishing any use of the current node. This includes AnyWrite/composite accesses and ordinary in-place operations (F38). |
| Render targets could be discarded before a host read | [docs/internals.jl:779](/sim/Programmieren/VideoEdit/dev/Mantle/docs/internals.jl:779) | Only dead graph transients may discard. Persistent images/windows store their result. Store selection runs after liveness. A clear after a draw starts a new pass (F39). |
| A fixed pass barrier did not reserve dependencies for new draws on other queues | [docs/internals.jl:637](/sim/Programmieren/VideoEdit/dev/Mantle/docs/internals.jl:637) | Reserve pass dependencies on prior writers and before later writers, retaining cross-queue edges. Fast insertion requires read-only non-attachment access and compatible layouts/placement/queue; otherwise recompile (F40). |
| Plan replacement lost graph-wide previous-run tokens | [docs/internals.jl:1291](/sim/Programmieren/VideoEdit/dev/Mantle/docs/internals.jl:1291) | Carry them into the replacement before retirement, even for disjoint old/new resources. Per-resource conflicts are insufficient to enforce the stronger graph-wide non-overlap rule (F41). |
| Preparing changes read mutable pending vectors without locks | [docs/resizing-and-raytracing.jl:375](/sim/Programmieren/VideoEdit/dev/Mantle/docs/resizing-and-raytracing.jl:375) | Copy pending/storage snapshots under short record locks, allocate outside, validate before applying (F43). An unlocked Vector is not an immutable snapshot. |
| Automatic BLAS/TLAS updates were published after unlocking submitted writes | [docs/internals.jl:1517](/sim/Programmieren/VideoEdit/dev/Mantle/docs/internals.jl:1517) | Lock the written resources and transitive readers together; pend their updates and notifications before unlocking. Otherwise a concurrent trace can slip between the geometry write and its update request (F44). |
| Unified Array(a) waited, then copied without excluding the next writer | [docs/internals.jl:2543](/sim/Programmieren/VideoEdit/dev/Mantle/docs/internals.jl:2543) | Revalidate an idle storage/shape/writer snapshot and protect the CPU memcpy under the corresponding bookkeeping locks. All GPU waits and host output allocation remain outside them. Direct readback serializes submissions during its CPU copy (F55). |
| Staged readback could use a newer host-requested shape after waiting | [docs/internals.jl:2581](/sim/Programmieren/VideoEdit/dev/Mantle/docs/internals.jl:2581) | Use the shape belonging to the submitted copy, or its copied device-length cell; never a later size(a). Capacity revalidation remains an implementation obligation (F55). |
| Group propagation through model operations was assumed rather than specified | [docs/api.md:149](/sim/Programmieren/VideoEdit/dev/Mantle/docs/api.md:149) | Leading-Group Mantle operations share the same body and propagate the group to every internal launch. Ordinary broadcast/generic methods still target the arrays' graph. No ambient group, and no claim that arbitrary unchanged model code becomes repeatable. |
| Native piece recording invited a backend draw walk | [docs/api.md:423](/sim/Programmieren/VideoEdit/dev/Mantle/docs/api.md:423) | recordpiece takes a core-owned emission callback; the driver only creates/finishes the native piece builder. No backend plan/draw algorithm (F53). |
| Capability selection omitted kernel feature requirements | [docs/api.md:159](/sim/Programmieren/VideoEdit/dev/Mantle/docs/api.md:159) | Filter portable/native candidates by declared requirements, including matrix operations and atomic scope/ordering. Raw unsupported intrinsics fail explicitly; high-level operations choose eligible candidates (F42). |
| Tuning had no ownership or execution-time contract | [docs/api.md:179](/sim/Programmieren/VideoEdit/dev/Mantle/docs/api.md:179) | Device-owned table; explicit tuning on private mutable data, correctness checks, versioned/configuration-aware keys, no tuning in declaration/run!, no initial persistence. Concrete representation/harness are fixed in execution-contracts.md §9; implementation remains F8. |
| Suite definitions leaked between areas/device passes | [test/suite/runsuite.jl:62](/sim/Programmieren/VideoEdit/dev/Mantle/test/suite/runsuite.jl:62) | Each device/area pass gets a fresh module. Repeated helper names, constants and types no longer replace definitions used by later tests or the next run. |
| Displaying a graph array was required to throw | [test/suite/areas/declaration.jl:111](/sim/Programmieren/VideoEdit/dev/Mantle/test/suite/areas/declaration.jl:111) | Show metadata without reading contents, length cells, submitting or waiting. DECL-04 now checks that behavior. |

A smoke check also found that @case escaped its label into @testset in a form
Test rejects. The macro now binds its label before interpolating it. Actual
phase/capability/long filtering was checked without loading Mantle: four
assertions passed with three intentional skips, including the executed-case
counter used to reject empty full phase gates.

The resolved KernelInterface dependency defines KI.@launch, while suite
call sites and examples called the absent KI.@kernel. Those call sites now use
the actual launch macro (F56); no alias was added to the dependency.

The state representation also could not express its own fixed graph
transients: the storage union admitted only device storage and the cell field
required Cell. It now includes Unplaced/ArenaSlice and NoCell; only a plan
owns arena backing, while graph aliases keep their device storage (F58).
THR-04 now covers counters and a negative live-state-read guard; KL-10 checks
compiler-origin Cell accesses. Native guard/bypass/compiler qualification remains. Pipeline cache policy is also
core-owned (F57): native compilation verbs create objects on a miss rather
than duplicating cache algorithms in each backend.

The compile-step table, run loop, record verbs, window verbs, pending-request
vocabulary, GraphDevice device parameter, workgroup forwarding, visibility
operation and test-file references were also reconciled. Examples now
initialize the recurrent accumulator and reserve enough capacity for their
claimed 300,000-point update. Texture indices are stable per native storage
generation, not across a resize of the logical Image.

Two existing performance claims needed a more precise reading. The pass list
is flat and is rebuilt on structural edits ([docs/recording-plan.md:1259](/sim/Programmieren/VideoEdit/dev/Mantle/docs/recording-plan.md:1259)); that is linear in
piece count, even though compilation of the affected piece is local. The
existing RM-06 gate is a finite latency bound at 100/1,000/5,000 plots
(2x the 100-plot median + 1 ms), not an asymptotic constant-time proof. Its
assertions were preserved. Likewise, reserving dependencies for future pieces
can reduce compute/render overlap: this must be measured, not described as
free (F52).

## Remaining implementation and qualification gates

The follow-up has fixed the missing contracts in
[execution-contracts.md](../../docs/execution-contracts.md). It supplies the
linearization/lifetime argument, native factories and capability schema,
prepare/claim/accept failure boundary, foreign/video/external-memory ownership,
and presentation retirement. It retains weak backreferences and the ordered
lock design. Native submit faults are now intercepted by the actual test wrapper;
recoverable faults occur before claim, and partial acceptance has RUN-18 coverage.

Existing consumer integrations, native probes and deletion audits are preserved;
[evidence-index.md](evidence-index.md) corrects the earlier overly broad missing-work
claims and maps each existing result to the actual remaining refactor delta.

The work still requiring the replacement implementation is explicit:

1. Run the nonempty phase-1 vertical slice against the separate frozen baseline;
   no legacy graph/array adapter in the replacement (F36).
2. Stress the ordered-lock and retirement implementation with actual concurrent
   queues, resource growth, plan replacement, finalizers, foreign leases and
   partial/native failures (F41/F43/F44/F49/F55/F61). The bounded host model is
   evidence for the abstract argument, not native execution correctness.
3. Implement and qualify the native boundary and shared KI-capability projections
   per device; measure actual candidate/library workspace behavior, slot ABI,
   capacity predication, sparse-buffer availability and conservative render
   dependencies (F42/F46/F52/F54/F59). Driver names in a table are not evidence.
4. Qualify maintenance1 and legacy presentation separately. Legacy final shutdown
   lacks a specification-level completion proof; practical idle-based teardown
   remains an explicitly weaker fallback (F47). Arbitrary indivisible kernels
   likewise have no universal watchdog guarantee. CUDA/HIP host VMM alone is
   not queue sparse binding.
5. Execute the concrete consumer migrations and acceptance matrix in
   [consumer-coverage.md](consumer-coverage.md): real RayMakie attachments,
   picking/FXAA/shadows/atlas, Hikari paired/material queues, DNN module/view
   lowering, MatAnyone state transitions, quantized Bonsai KV growth and editor
   decode/GL handoff. The source traces and sketches exist; performance and
   numerical integration results do not (F48/F62/F63).

Hardware/software traversal equivalence, cross-workgroup memory semantics and
more advanced shader interfaces still require compiler/runtime qualification.
The desktop targets do not establish mobile/web/Intel-Mac coverage or automatic
cross-driver zero-copy. CPU-only Lavapipe does not satisfy multiple-device,
discrete-GPU, other native-backend or display fixtures (F60).

## Why the tree is large, and what must not return

Measured current inventory: 110 Julia files / 53,527 physical lines in src,
and one file / 1,478 lines in ext. These are physical lines including
comments/blank lines; they do not establish a 55,005-line minimum or a
10,000-line implementation estimate.

The seven saved audits identify concrete inflation mechanisms. Several were
spot-checked in this continuation: the backend walks ([src/graph/kalaunch.jl:770](/sim/Programmieren/VideoEdit/dev/Mantle/src/graph/kalaunch.jl:770), plus the
Vulkan/Metal graph files), global kernel tables ([src/array/gemm.jl:615](/sim/Programmieren/VideoEdit/dev/Mantle/src/array/gemm.jl:615)), and pipeline caches
([src/metal/graphics.jl:599](/sim/Programmieren/VideoEdit/dev/Mantle/src/metal/graphics.jl:599), plus MESH_CACHE). The plan now names the patterns and their owners:

| Pattern | Target owner / removal rule |
|---|---|
| Core/Vulkan/Metal plan walks and submission/lifetime policy | One compile/record/run path in core; native command encoding only in device methods. |
| Eager/declared GEMM/GEMV/FFT entry-body pairs | One target-parameterized operation body; kernel families remain measured candidates. |
| Vulkan/Metal TLAS range, member, transform and lifecycle bookkeeping | Core TLAS/BLAS; native record conversion and build commands are driver facts. |
| Separate upload/download/readback implementations | Core change/read paths, one backend memory/copy ABI. |
| Global dictionaries, per-variant runtime eval caches and duplicated policy state | CodeInstance-owned compute kernels, device-owned pipeline/selection tables, one representation per concept. |
| Experimental numerical variants and benchmark history in runtime files | Keep needed/tested fallbacks; archive experiments/results outside the runtime. Do not delete a fallback merely because today's main workload does not select it. |
| Collision/application algorithms bundled with the GPU runtime | Make ownership and consumer migration explicit before moving them; do not hide line count in another package. |

Audit reductions overlap and are estimates. They must not be summed into a
claimed final line count. A compact implementation is plausible; **10k total
lines with all listed kernels/backends/features is not established**. Measure
runtime policy, portable kernels, driver ABI, application algorithms and
comments separately. Using more dispatch is useful where type information is
stable; replacing every loop, weak reference or explicit check is not a
simplicity criterion.

## Verification and completion boundary

The new API is not implemented, so behavioral execution cannot validate these
contracts. The suite and Julia design files were parsed with Julia's parser;
case IDs and referenced suite paths were checked independently. All 585 case
macros were expanded with Test and KernelInterface, and all 42 area files
were expanded and lowered. These checks do not run their GPU bodies. No performance or backend-feasibility result is claimed from parsing. F36–F73 track the new
implementation/proof obligations; their implementation status remains open/doing until qualifying evidence
exists. Existing audit estimates and raw notes have not been promoted to
runtime facts.


## Follow-up: execution contracts and consumer evidence

The direct follow-up specifies the remaining execution boundaries in
[execution-contracts.md](../../docs/execution-contracts.md) and traces real
consumer integration paths in [consumer-coverage.md](consumer-coverage.md).
Earlier items described as unresolved above must now be read as native
implementation/qualification obligations where these contracts fix the choice.

Concrete corrections: publish each accepted segment before later work can fail;
keep native failure domains terminal; separate WSI from work enqueue and record
locks; retain independent present retirement holds; stop treating CUDA/HIP host
VMM as queue sparse mapping; choose slot-pointer/capacity-predicate baselines;
validate staged-copy capacity before mapping/claim; keep a typed sampled-image
ABI and actually sample atlas texels in RM-15. Driver-required dedicated memory
is pool-owned rather than prohibited. Preserve real decoder/DPB and GL handoff
ownership instead of deleting synchronization on an assumed common queue.

The suite instrument now wraps native channels/builders/completion records and
owns a fresh core Runtime. Host tests execute that actual wrapper against a fake
native ABI, check forwarding and fault hit counts, and verify canonical channel
state identity across wrappers. Recoverable RUN-07/AP-05 faults now occur in
preparesubmit; RUN-18 checks the accepted prefix after second-segment preparation
fails. DNN-12 checks module composition inside a repeated Group with persistent
state. These target runtime cases still await the replacement implementation.

Host evidence is recorded in test/design/runtests.jl, including bounded
interleavings and negative controls for reader publication, staging growth and
presentation retirement. This is an argument under stated access/ownership
assumptions, not a machine-checked proof of a GPU implementation or its compiler.
The legacy Vulkan shutdown limitation and arbitrary indivisible-kernel watchdog
limit are explicit restrictions, not solved by renaming a wait or token.

## Final reconciliation of the reviewed boundaries

The following were substantive design defects, not wording changes:

| Finding | Fixed contract and acceptance evidence |
|---|---|
| The review incorrectly described Hikari materials as never being hit groups | Actual rt-pipeline.jl:530–575 constructs per-material closest-hit groups. Preserve that path alongside typed shade queues; core owns SBT layout/cache. Its opaque-only alpha shortcut is ineligible for alpha-textured reference rendering. execution-contracts.md §7; HK-10. |
| Atomic capability facts did not establish compiler correctness | Historical source-hashed reproduction stripped acquire/release in Lava and weakened sequential consistency. F33 is now repaired: compiler regressions and four-device native visibility checks are in ../atomic-ordering-2026-10-21/README.md. Unsupported sequential consistency now fails compilation. F42 retains complete target eligibility; this is not a universal platform claim. See native-evidence.md. |
| General shader callables could conceal GPU resources | Discover the callable and every RT stage as well as explicit args. Captures enter location checks, holds, inference and slots; cache keys do not root arrays. DECL-12/13 and HK-10. |
| Budget/free-byte semantics double-counted allocations | Normalize additional per-heap headroom and compare next native block only; mapped is a subset of reserved. FaultDevice BudgetLimit changes with allocated bytes. execution-contracts.md §12 and host wrapper assertions. |
| Pageable eviction had no safe bounded-copy protocol | Reserve at most 8MiB staging, capture/validate one applied snapshot, freeze physical storage through a transfer lease, copy chunks outside locks, retain queued changes and copied device length. Host negative controls catch torn copies/lost stores; native pressure races remain. |
| Undersized staged reads could retry the same destination forever | OwnWork preclaim validation returns NeedDestination to the outer allocator. Snapshot growth is a retry, never an out-of-bounds copy. |
| TLAS count-from-memory contradicted legal refits | Kernel-written totals use capacity build counts on every path, including Metal. Valid mask-zero padding preserves builtcount; exact-count optimization requires separate legality evidence. |
| A present-only retry could skip later GPU stages or lose a pending resize | Native WSI is deferred until every stage was accepted. A complete-frame OOM request retries before edits/recompile; pre-WSI partial failures retire their signal semaphore and use normal partial-prefix semantics. WIN-30 checks counters, native call hits, no GPU resubmit and pending resize. |
| Sampler and pipeline cache lifetimes were unbounded/ambiguous | Weak sampler interning, active/cache/token holds and one bounded core idle policy; no per-backend cache algorithm. Fixed-LOD sampling cannot invent an anisotropic footprint. §§11/13. |
| Profiling could overwrite queries or conceal encoding work | One pending instrumented sample, normal variant while pending, nonblocking harvest and bounded completed ring. Persistent records, prefixrecords and eagerrecords have separate counters. §10. |
| Lavapipe and queue labels were invalid capability fixtures | Local native facts advertise sparse/RT even on Lavapipe. Force disabled features, inspect distinct canonical queue states, and supply verified driver pairs. Shared helpers reject misspelled capability overrides. DEV-10/12 and host fixture checks. |

A final pseudocode pass also removed the opaque backend emitloop! algorithm:
core lowerloops! expands iterations before dependency/cost/placement analysis,
with control edges for every body node. Its command/slot memory scales with the
maximum iteration count; compiled code is reused. TraceNode now explicitly
retains accel and captured resources contribute Bound slot dependencies. These
were representation omissions, not native performance results (F72/F73).

These choices use the existing core pool, holds, resource ordering, candidate
selection and record/submit path. They do not authorize a second lifecycle,
consumer-owned pipeline cache or backend-specific scheduler.

Final host validation: 78 actual fault-wrapper assertions, 34 bounded-model
assertions (with broken-protocol negative controls) and 19 shared-helper assertions
pass: 131 total. [validate-spec.jl](validate-spec.jl) reproducibly parses 52
suite/design/doc Julia files and expands/lowers 42 areas with 585 unique cases.
The atomic source probe separately reproduces a known defect; it is not counted
as passing target behavior. git diff --check passes. The new GPU API remains
unimplemented; no native performance or real consumer correctness pass is claimed.

## Correction and native feasibility follow-up — 2026-10-21

The chat assessment that real consumer migrations, a deletion map and broad
feasibility evidence were missing was inaccurate. Existing ports, baselines,
302-dispatch slot benchmarks, capacity/TLAS/interop experiments and the seven
per-file source audits already supply substantial evidence. They are indexed in
[evidence-index.md](evidence-index.md). Refactor-specific acceptance is a delta
against that work, not a restart. Existing numerical rulings carry forward.

The platform goal remains intact. Concrete compiler/backend gaps need fixes;
they do not justify a blanket conclusion that the cross-platform/acceleration
architecture is unsupported. The index states the actual evidence and limits.

A new bounded native foundation probe now executes on RX7900XTX, RTX4000Ada and
Lavapipe using Vulkan.jl/Lava and no old Mantle graph/array implementation:
132 assertions, zero captured validation errors with synchronization validation.
See [scope and results](../feasibility-2026-10-21/README.md). This changes the
previous host-only evidence boundary, while leaving full phase 1 and native
Metal/CUDA/HIP replacement qualification open. No production runtime was changed.

[source-disposition.tsv](source-disposition.tsv) consolidates the already completed
per-file audits into 111 source rows, target owners and phases. It is an index,
not a fresh duplication study or a promised final line count.

## Review-process and document-health correction — 2026-10-21

The failure to reconcile already completed ports, measurements and per-file
maps was the reviewer's error. Target-state documents intentionally describe
the replacement; absence of an experiment's full history there is not evidence
that the experiment or baseline capability is absent. The earlier broad
missing-work/readiness conclusions exceeded the evidence checked.

There is also real document debt: target contracts are distributed, several
fixed names retained open markers, the Plan rerecord proposal contradicted the
new canonical Diagnostics counters, and ledger statuses do not always distinguish
a settled design decision from verified replacement implementation. The obvious
counter/name mismatches are corrected, and the plan/API/ledger entrances now
identify authority, evidence and status scope. This is navigation/reconciliation,
not another architectural change.

The recent multi-file contract patch still needs a complete consistency and
simplicity review against existing accepted decisions and measurements before
freezing it. Host assertions, syntax/lowering checks and the bounded native
probe prove their stated scopes; they do not establish that the whole design
is complete. Earlier claims of completion/readiness should be read with this
explicit limitation. New policy defaults remain subject to their measurement
gates; adding a rule to a normative document does not establish its performance.


## Document consistency cleanup — 2026-10-21

This pass reconciles the seven target documents, including the normative
execution companion. It preserves existing consumer ports, measurements and
source-disposition audits. It does not change production runtime code or claim
that the replacement is implementation-ready.

- The plan identifies document ownership; api.md §4.9 indexes the additional
  required native signatures/protocols without creating another contract copy.
- Settled launch defaults, rewrite names, graph priority, streamed weights,
  introspection names and TLAS address/descriptor selection no longer appear as
  undecided names. Application helpers remain illustrative.
- GraphDevice is the KA backend for graph arrays; Graph is not. Enumeration
  includes Metal. Views carry strides. Fixed graph transients use NoCell;
  device arrays and resizable transients have cells. Graph pseudocode retains
  its profile and priority settings.
- Host-known TLAS counts are exact; device-written totals use the fixed-capacity,
  mask-0 baseline on every device type. The exact-count-in-larger-storage
  performance experiment remains a separate, unmeasured phase-5 gate.
- CUDA/HIP phase prose and ledger now match the slot-pointer/capacity-predication,
  core-unrolled-repeat and initially committed-arena contracts. Metal state
  coverage remains a native qualification gate; a per-run host draw walk cannot
  satisfy it.
- Kernel code sharing requires matching compiler/feature/ABI configurations.
  CodeInstance ownership, idle native-cache eviction, reset generations and
  estimated KernelInfo facts now agree across the documents. Pressure ordering
  includes the specified idle-cache purge.
- Ledger F2/F35 distinguish settled design from open replacement execution gates;
  historical documentation completions identify their commits. Current review
  changes remain uncommitted. Desktop qualification fixtures no longer read as
  a retraction of the cross-platform goal.

One real protocol obligation remains explicit: bounded staging for large restores
(F74/P4.A31). The existing Restore sketch prepares one upload for the copied
extent. The bounded eviction HostCopyJob model does not establish a bounded
restore protocol. api.md §10, the plan, execution-contracts.md §12 and the
pseudocode now identify the same scope and remaining obligations. Native and
consumer qualification gates remain open where their evidence is still required.

Validation: 131 host assertions passed; 52 Julia files parsed; all 42 suite areas
expanded/lowered with 585 unique case IDs. No target GPU bodies were executed by
these checks. Local links/anchors, explicit suite paths, case references, ledger
IDs and whitespace were also checked; see document-cleanup-result.txt. The
previous bounded native probe was not rerun because its code did not change.


## User ruling: debug before design changes — 2026-10-21

This supersedes the earlier F74 wording requiring bounded restore staging.
The user requires overwhelming evidence before introducing such a fallback,
and specifically requires exhaustive debugging of the direct implementation
before a failure can justify a design change. A broken implementation is not
architecture evidence.

The plan now gives the canonical investigation/evidence rule for design changes
generally. api.md §10, execution-contracts.md §12 and F74 specify direct-host
restore qualification, with no prescribed staged restore path. Both restore
pseudocode call sites use the prepared native source view over retained host
bytes; they no longer require a second full-sized upload allocation. Source
leases, destination retention, preclaim cleanup and ordered cell publication
remain obligations to qualify, not claimed runtime passes.

Existing importhost/importablehost and DNNKernels usage provide references to
debug against. Local extension enumeration and the historical RADV source-type
claim are not enough to justify a fallback. The evidence index records that
scope explicitly. No production backend or speculative fallback was added.


## Direct-host qualification and implementation debugging — 2026-10-21

The local native probe now passes 519 assertions with zero validation errors;
see ../direct-host-2026-10-21/README.md for its exact scope and source hashes.
It executes the existing native import functions without loading Mantle, verifies
actual copy bytes and accepted-work source lifetime, and exercises owned/cache
sources at 4/64 MiB. Source-dependent import rejections remain visible. A CPU
copy into the final CPU-visible device-local allocation passes for protected
file sources on all four local Vulkan devices; no staged restore was introduced.

The investigation first fixed Vulkan.jl's uninitialized output-header bug in the
generator and generated bindings. The regression has a wrong-header control;
seven unrelated generator-suite failures are reproduced with the original
initializer and remain explicit. Raw native argument/memory-type/protection
controls and an AMD kernel ioctl trace are retained. They narrow failures but do
not establish universal platform restrictions or authorize a fallback.

The target contracts now include the already-supported direct final-allocation
initialization construction with its private-destination/visibility/publication
obligations. F74 still tracks replacement integration and remaining coverage.
Copy timing variation and alignment controls are recorded as observations, not
state-of-the-art performance proof or grounds for a design change. Earlier
notes saying no production runtime code changed predate the Vulkan.jl binding
fix; Mantle src/ext remain unchanged.
