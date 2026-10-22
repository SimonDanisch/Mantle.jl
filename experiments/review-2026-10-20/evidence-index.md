# Existing evidence and remaining refactor deltas — 2026-10-21

This corrects the review's overly broad claims that consumer migrations,
feasibility evidence and a deletion map were missing. Those artifacts already
exist. The replacement target has not been implemented; this does not erase
work on the current implementation or require the consumers to be ported from
scratch again. Existing decisions and acceptance baselines carry forward unless
an explicit target change requires a different result.

## Existing consumer integrations

| Evidence | What already exists | Actual remaining delta |
|---|---|---|
| [DNNKernels migration](../../../JuliaVision/plans/2026-08-07-dnnkernels-on-mantle.md), [declared-graph baseline](../../../JuliaVision/plans/2026-09-15-declare-dont-capture-baseline.md), current emit.jl/launch.jl/plan.jl | Real Mantle graph emission, arena/view planning, SAM 2 correctness/memory/timing baselines; capture/allocator work has already been done. | Adapt changed target signatures/ownership and explicit Group propagation; preserve exporter/kernel decisions and compare with existing baselines. |
| [ROCm model measurements](../../../JuliaVision/plans/2026-09-21-rocm-baseline.md) | Actual HIP/Mantle model results and operator-level comparisons; identifies attention/convolution gaps specifically. | Preserve existing integrations and measure changed operator selection/workspaces; do not call the entire backend/model stack unproven. |
| [Consumer source map](consumer-coverage.md): Hikari graph.jl/per_material_queues.jl/rt-pipeline.jl | Declared integrators, paired bounce rounds, typed shading queues AND actual typed closest-hit programs. | Map to new core resources/Group/program contracts and preserve numerical/image behavior; fix the specifically documented opaque-alpha shortcut rather than adopt it as reference. |
| [Consumer source map](consumer-coverage.md): RayMakie overlay_rendering.jl/screen.jl | Real frame planning, raster/traced composition, picking, FXAA, shadows and atlas updates. | Replace old bindings/signature walks and handle invalidation with the new slots/pieces/AS lifecycle while retaining the existing features/tests. |
| [Consumer source map](consumer-coverage.md): GPUFiltering, editor/decoder/GL bridge | GPUFiltering already depends directly on Mantle and has graph launches. The editor has a real decode/preview/interop path. | Port changed API boundaries and preserve ownership/fences until their replacement is working. This is continuation of existing ports. |
| [Benchmark harness](../../../../benchmarks/jobs.jl) and benchmarks/makie,dnn,editor | Fresh-session benchmark entry points for the actual consumer APIs and baseline/fix comparisons. | Run the replacement through the existing adapters/harness; evidence availability must be checked before requesting new workloads. |

## Existing feasibility and performance measurements

| Path / mechanism | Existing artifact and scope | Remaining delta |
|---|---|---|
| Recorded slots/head versus patching/re-recording/walking | [AMD 302-dispatch plan](../../../../experiments/header_indirection/plan_radv.txt), [NVIDIA plan](../../../../experiments/header_indirection/plan_nvidia.txt); checked full/halved lengths, host and GPU timings. | These already establish useful whole-plan feasibility. New lifetime/concurrency/failure rules need their own checks; the entire slot architecture is not awaiting first evidence. |
| Capacity launch / inactive work | [capacity_radv.txt](../../../../experiments/header_indirection/capacity_radv.txt) measures 1x/2x/4x capacity. At n=1048576, listed copy times rise from about 7.68 to 10.39 to 17.10 microseconds. | Do not present capacity predication as unmeasured or universally free. This Vulkan result does not measure CUDA/HIP graph integration; qualify that native delta. |
| Recorded versus eager execution | [eager-vs-graph results](../../../../experiments/eager-vs-graph/results.txt): actual SAM 2 encoder, interleaved runs, numerical equality, AMD/NVIDIA measurements. | Preserve recorded/eager regression criteria and investigate the measured overhead; no new request to establish whether graphs help. |
| Exact-count/padded AS builds, sparse binding and traversal | [tlas_sparse experiment](../../../../experiments/tlas_sparse/bench.jl), [RTX3070 trace results](../../../../experiments/tlas_sparse/m4_legionboi.txt) and companion Windows/AMD output files. | Use the specific build/refit/traversal result for the relevant policy. Do not infer a new Metal count-from-memory policy's cost from Vulkan-only results. |
| Cross-context CUDA/HIP calls and Vulkan-native PTX | [Interop notes](../../../../experiments/interop/NOTES.md): working shared-memory round trips, measured CUDA handoff costs, HIP memory success/semaphore limits, working cp.async PTX inside Vulkan command buffers. | Per-operation context switching is a measured cost. Coarse region/native candidate selection is core's responsibility. F34 tracks production integration, not an absence of feasibility evidence. |
| Portable backend behavior and acceleration | [Current root-fix status](../../../../experiments/portable-tests/ROOTFIX.md), [findings](../../../../experiments/portable-tests/FINDINGS.md), [Metal allocation probe](../../../../experiments/metal-run-allocfree/tools/mtl4_probe.jl) | Existing Metal cooperative-matrix/KI, recorded range, window and numerical fixes must carry forward. Older per-area reports explicitly list then-unrun cases; current ROOTFIX/source/results supersede those historical findings. Do not count a source conversion as a pass or a fixed historical gap as still open. |
| Numerical defaults | [ROOTFIX decisions](../../../../experiments/portable-tests/ROOTFIX.md): minNum/maxNum, GPUArrays scalar-indexing policy, shared-buffer use tracking, Metal matrix operations. | Preserve existing numerical decisions; define only genuinely changed/unspecified operation contracts. The numerical specification does not start from zero. |

“Expensive fallback paths” was an imprecise label. The mechanisms needing an
explicit evidence check are capacity predication, bounded loop expansion,
committed rather than sparse arenas, bounded pageable eviction, software
traversal/sampling and cross-context vendor calls. Some are ordinary chosen
baselines, some are capability fallbacks, and several are already measured above.
The task is to identify uncovered changes and their workload-dependent costs,
not claim that all these mechanisms are missing or necessarily slow.

## Host transfer evidence and the debug-first ruling

Mantle already has [native host import](../../src/vulkan/runtime/external.jl),
[importablehost allocation and usage notes](../../src/runtime/api.jl), and a
[DNNKernels cache consumer](../../../JuliaVision/DNNKernels/src/quantcache.jl).
Carry these implementations and measurements forward as references. A query in
this review session found VK_EXT_external_memory_host with 4096-byte alignment
on the RX 7900 XTX, RTX 4000 Ada, Ryzen 7900X integrated GPU and llvmpipe. This is
capability enumeration only, not a successful direct-copy/lifetime test.

The [direct-host qualification](../direct-host-2026-10-21/README.md) now executes
Mantle's exact native import function, checks real bytes and lifetime, and adds
raw native diagnostic controls. The local run passes 519 assertions with zero
validation errors. Source-dependent import rejections are recorded explicitly;
CPU copies into the final CPU-visible device-local allocation also pass for
protected file sources on all four tested devices. These are native primitive
results, not replacement-runtime or full consumer qualification.

The investigation found and fixed a Vulkan.jl output-header initialization bug
before accepting results. Historical reports and single failed probes still
cannot justify a design change. F74 requires direct-restore integration and
qualification; no staged restore algorithm is justified by this evidence.
The canonical debug-first rule remains in the plan and execution-contracts.md §12.

## Deletion and ownership maps already exist

The [first implementation batch](consolidation-first-batch.md) now removes six
duplicated steps, including shared TLAS mesh preparation and vertex expansion.
It records 42 net source lines removed, 224 passing targeted assertions and
the outstanding Metal native qualification. The larger ownership/deletion
gates remain open in [consolidation-order.md](consolidation-order.md).

The [memory policy batch](consolidation-memory.md) removes Vulkan's independent
collection budget and pooled-allocation retry, shares quiescence and collection
coordination in core, and records 9,110 passing targeted assertions, including
AMD/NVIDIA native validation. It removes another 73 physical source lines.
Unified token retirement and the other ownership groups remain open.

The [TLAS/cache follow-up](consolidation-tlas-cache.md) shares batch triangle
tables, removes duplicated cache invalidation and records 240 passing targeted
assertions. Its host table benchmark reduces temporary allocation from 1.30 MB
to 388 KB. Remaining cache retention and TLAS mutation owners are explicitly
distinguished from the algorithms already consolidated.

The [pipeline-publication batch](consolidation-publication.md) gives native
raster/mesh/ray pipeline creation one core publication algorithm and removes
Metal's process-global raster/mesh cache owners. It records 888 passing targeted
assertions, including actual concurrent native creation and AMD/NVIDIA rendering
and tracing. The 52-line net source increase and remaining idle/hold/compute
ownership work are explicit.

The [kernel-preparation batch](consolidation-kernel-preparation.md) shares KI
preparation between eager array launches and declared dispatches, and centralizes
the repeated KA constructor/capture policy. It records 1,011 passing targeted
assertions, including GEMV, FFT/STFT and allocation-free recorded runs on
AMD/NVIDIA. The full Builder/emission replacement remains open.

The [TLAS mutation batch](consolidation-tlas-mutation.md) replaces both backend
batch structs and registration/mutation algorithms with a shared core owner,
removes duplicate metadata state, and records native and host qualification.
The [per-file source inventory](source-size-2026-10-21.tsv) separates
implementation lines from docstrings, comments and blanks. At `d367d09`,
55,051 physical lines contained 23,424 implementation lines; neither that
measurement nor these completed batches establish a final 10–20k size.

The [2026-10-21 similarity experiment](similarity-2026-10-21/README.md) provides
local semantic code-to-code and responsibility-based search across 1,987
methods/regions from all 111 source files. It compares Qwen and Jina embeddings
against lexical matching, retains ranked reports, and records false positives
and reviewed candidate contracts. It supplements the audits; it does not prove
complete duplication coverage or approve automatic consolidation.

The seven source audits have per-file inventory/disposition tables, duplication
pairs, named ownership violations and minimal-form estimates:

- [Array/geometry](audit-array-geometry.md)
- [Graph](audit-graph.md)
- [Memory/sync/runtime](audit-memory-sync-runtime.md)
- [Graphics/RT/core/HIP](audit-graphics-rt-core-rocm.md)
- [Metal](audit-metal.md)
- [Vulkan graphics/RT](audit-vulkan-graphics-rt.md)
- [Vulkan runtime](audit-vulkan-runtime.md)

The ledger's P*.D*/P*.R* rows are the phased symbol deletion/rename map; api.md
Appendix A gives vocabulary fates. [source-disposition.tsv](source-disposition.tsv)
now provides one index covering all 111 Julia files / 55,005 physical lines in
src+ext, with target owner, phase, disposition and the existing detailed audit.
This consolidates the research; it does not replace it with another survey.
Collision-code destination ownership remains explicit, and kernel deletion still
requires coverage of its supported inputs. Target-size estimates overlap, so no
summed estimate or moving-code exercise establishes a final implementation size.

## New native check and platform claim correction

The [2026-10-21 native probe](../feasibility-2026-10-21/README.md) now passes 132
assertions across AMD/NVIDIA/Lavapipe and capacity/indirect variants, with
synchronization validation and no captured errors. It executes the cell-to-slot,
move, repeat-submit and basic creator/graph-hold mechanisms without importing old
Mantle. The README states the exact scope; asynchronous storage reuse and full
phase-1 behavior are not claimed.

Cross-platform graphics/compute, shared resources and automatic use of native
acceleration/fallbacks remain target requirements. The review found concrete
implementation/semantic problems; it did not establish an architectural
impossibility or justify dropping this goal. Existing backend integrations and
measurements count. The remaining work is feature-specific: preserve supported
paths, implement changed native boundaries, and qualify their semantics/performance.
“Universal platform promise unsupported” was too broad and is withdrawn as a
review conclusion. This does not turn unexecuted replacement paths into passes.

## Atomic ordering repair (2026-10-21)

[Repair, permanent compiler regressions and native visibility qualification](../atomic-ordering-2026-10-21/README.md) close F33's demonstrated compiler defect. Historical demotion evidence remains a negative control; target eligibility and broader qualification remain F42.

## Implementation consolidation before the foundation

[Current-code consolidation order](consolidation-order.md) turns the existing
audits into an initial verified cleanup batch and ties larger consolidations to
one owner and explicit deletion gates. P0.A10 tracks implementation; the audit
and disposition map do not mean those deletions are already complete.

[Shared video sessions](consolidation-video.md) records native AMD/NVIDIA byte comparisons, mapped-input growth, cleanup, allocation measurements and remaining video limitations.

Graph backend, packing, lifetime, submission and queue comments/docs now state current contracts concisely. JuliaSyntax comparison confirms identical implementation trees excluding docstrings; the cleanup removes 884 prose lines, including stale Metal and hold-frame claims.

[Shared trace packing/setup](consolidation-trace.md) records eager/recorded ownership, native helper deletion and 677 host/AMD/NVIDIA assertions.

Boundary guards exposed missing registry entries for the existing BatchedTLAS
native primitives and mipmapped query. Their resource-specific applicability is
now explicit; no runtime fallback was added. The Vulkan-intrinsic AABB fixture
moved under test/vulkan. Boundary checks pass (one existing pure-method triage
remains broken); host batch mutations and native AABB/mipmap checks pass on both GPUs.
The guard's historical prose is condensed; its checking syntax is unchanged.

The memory report also records shared Vulkan arena-buffer construction and 200 native assertions across AMD/NVIDIA; required coherent/local flags, cached placement, BDA flags and block ownership remain explicit.

[Shared GEMM generation](consolidation-gemm-generator.md) records 171 removed implementation lines across two batches, native before/after output and shader comparisons, 2,486 suite assertions and interleaved timing preservation.

Core entry-point and GEMM prose now describe current dispatch, precision,
staging and ownership contracts. Removed switch descriptions, resolved-bug
narratives and benchmark tables remain in git history (`df21128`). JuliaSyntax
confirms identical definitions; Project.toml parses to identical data. Three
unused top-level prose strings are deleted and a fourth is attached as method
documentation. The inventory's resulting 40-line implementation decrease is
prose reclassification, not algorithm deletion. Source decreases 53,085 → 51,636
physical lines; all 115 source files still parse.

[Shared geometry/strides](consolidation-geometry.md) removes 49 implementation
lines, corrects stale geometry contracts and qualifies contact-slot ordering
with native contention/tail cases (629 targeted assertions).

[Shared draw preparation/packing](consolidation-draw.md) removes 50 implementation
lines and fixes manual draw argument retention. Native lifetime, rendering,
publication and exact attachment-format checks pass on AMD/NVIDIA.

[Shared typed host upload](consolidation-host-upload.md) removes transfer-sized
wrapper copies, with ordinary/mapped allocation measurements and 102 native
assertions. Adaptation comments now consistently require explicit leaf retention.

[Shared acceleration projection](consolidation-accel-projection.md) deletes both
backend field walks while preserving their distinct scene conversions; 126 host
and native assertions pass. Native Metal qualification remains outstanding.

[Shared dense GPU strides](consolidation-dense-strides.md) removes backend
metadata copies and fixes the absent ROCm method. 3,987 shared GEMM and 77 ROCm
backend assertions pass; the required existing SDK source is pinned explicitly.

[Shared pending stores](consolidation-stores.md) removes range-store and barrier
encoding copies, fixing Vulkan overlap ordering and early large-store submission.
656 Vulkan regression, 32 encoding and 74 ROCm assertions pass; fresh loading and
paired zero-allocation store/drain checks also pass.

[Redundant native argument packers](consolidation-argument-packers.md) deletes
two methods already covered by generic primitive/aggregate packing. 24 paired
byte checks and 3,492 native packing/move/allocation assertions pass.

The [kernel preparation follow-up](consolidation-kernel-preparation.md#context-abi-follow-up--2026-10-22)
deletes Metal's repeated KA context setup through the existing argtype hook.
284 host/Vulkan and 102 ROCm/host assertions pass; native Metal remains unqualified.
