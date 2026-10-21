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
