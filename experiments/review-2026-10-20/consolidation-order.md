# Implementation consolidation

P0.A10 remains **doing**. The seven source audits and `source-disposition.tsv`
map the codebase; they do not establish that every duplicate has been removed.
Each replacement must delete its old owners and preserve callers, synchronization,
error behavior and native requirements. Measure implementation and prose separately.

## Implemented

| Shared owner | Deleted duplication and qualification |
|---|---|
| [Setup helpers](consolidation-first-batch.md) | Memory-type search, borrowed Metal views, attachment/depth setup, fragment input types. Metal execution remains unqualified here. |
| [Memory collection](consolidation-memory.md) | Vulkan's independent collection budget, GC/reclaim coordination and pooled-allocation retry; native allocation still reports device requirements. Arena classes share buffer creation/binding/wrapping while retaining native memory requirements. |
| [TLAS metadata / cache invalidation](consolidation-tlas-cache.md) | Triangle-table preparation and complete compute-cache invalidation. Preserve native content reuse during replacement. |
| [Compilation publication](consolidation-publication.md) | Core publication for native graphics/mesh/ray pipelines; device-owned Metal caches replace process globals. |
| [Kernel preparation](consolidation-kernel-preparation.md) | Eager and declared KI preparation, KA construction and capture validation. |
| [GEMM generation](consolidation-gemm-generator.md) | One wide/narrow/ablation body and one tile-product generator for all eight staged families; distinct measured schedules remain. Native output, shader and timing checks preserve behavior. |
| [Trace packing/setup](consolidation-trace.md) | Eager/recorded argument packing, direct/indirect native bindings/timestamps, shader modules and TLAS descriptor writes. |
| [TLAS mutations](consolidation-tlas-mutation.md) | Shared batch state, geometry registration, push/update/delete/visibility/range algorithms; backend metadata copies and unused offset state deleted. |
| [Video sessions](consolidation-video.md) | Shared decoder state, session resources, Annex-B splitting, upload packing and batch decoding. Codec parsing and native picture/reference structures remain distinct. |

## Remaining owners and deletion gates

| Group | Current ownership → required replacement |
|---|---|
| Lifetime | Pool regions, channel native-object lists and `vk_free!` enqueue/drain bridges → P2.A5/P2.D13 token retirement. Replace all callers together; preserve recording holds, cross-queue last use, finalizer-safe enqueue and exactly-once destruction. Gate: hold/lifetime, queue, memory-safety and recorded-pool-trim regressions. |
| Compilation caches | Vulkan native pipelines, linked kernels and launch plans remain separate layers → P1.A1 / execution-contracts.md §13: compute ownership at Kernel/CodeInstance, device-specific entries and bounded idle retention. Preserve compiler/layout/workgroup keys, shader invalidation, warm reuse and in-flight holds. Native disk-cache serialization stays native. |
| Execution | ArrayLaunch already selects kernels once; eager and recorded emitters remain separate → api.md §§2.4/4.5 Builder. Replace compilation, access tracking, pending-change application and emission together. Metal Immediate also serves front-of-run work; delete it only when that caller is replaced. Gate: eager/graph numerics, recorded replay and warm allocations. |
| TLAS | Shared BatchedTLAS mutations still feed native pending-change/build/refit owners → TLAS/MemberSet/InstanceRange/RangeEntry. Preserve Vulkan's deferred updates and Metal's immediate submissions until the replacement owns both. Native packing, residency and driver calls stay native. |
| Pressure | Shared collection controller → P2.A10 acquisition policy, including P4.A13 idle cache eviction. Preserve trim, soft-cap and heap-pressure behavior. |
| Video | Shared legacy sessions still use blocking submits; H.265 supports COINCIDE only → target video/token contracts. Native availability and codec limitations remain explicit; no new CPU fallback is introduced. |

## Review boundaries

GEMM/GEMV/FFT kernels already live in core; distinct measured tilings are not
copies. Vulkan format mappings retain usage distinctions (Float32 depth versus
R32 sampling). Generic refusal methods and native implementations are dispatch
layers. Native encoding differences alone do not justify copied policy.

The [semantic search](similarity-2026-10-21/README.md) discovers structural
candidates; scores do not prove equivalence. Rebuild its source index after edits.
The size target is not yet proven; report actual net source changes per batch.
