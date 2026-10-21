# Implementation consolidation before the foundation

Checked against Mantle `45021c6`, including the recently merged Metal fixes.
The initial `src/` plus `ext/` inventory contained 111 Julia files and 55,115 physical lines.
These are inventory counts, not an estimate of removable code. The existing
seven audits and `source-disposition.tsv` remain the broad ownership map.

## First consolidation batch

These bodies were verified before the first implementation batch. Locations in
the table refer to that baseline; the [batch report](consolidation-first-batch.md)
records the changes and their qualification, including the untested Metal path.

| Duplicated implementation | Current locations | Consolidation and preserved behavior |
|---|---|---|
| Vulkan memory-type search | `src/vulkan/runtime/memory.jl:1346` and `:1359` | Keep one search. The throwing entry point calls the optional selector and retains its existing error on no match. Preserve type-bit eligibility, required flags, first-match ordering and zero-based result. |
| Borrowed Metal array construction | `src/metal/memory.jl:33` and `src/metal/ka.jl:111` | Have `deviceview` use the existing `deviceslice` constructor. Preserve the non-owning DataRef, shared Managed buffer, byte offset, dimensions and remaining capacity. Verify on Metal, including the recorded indirect-slot caller. |
| Metal attachment/depth setup in raster and mesh pipeline compilation | `src/metal/graphics.jl:665` and `src/metal/mesh.jl:350` | Share the attachment/blend/depth setup and depth-state construction. Keep stage compilation, descriptor types and stage-specific cache keys distinct. Different native descriptors do not justify copying identical attachment policy. |
| Fragment input type construction | `src/graphics/mesh.jl:76` and `src/graphics/pipeline.jl:89` | Share the identical type construction using the existing distinct `fragmentinputs` dispatch. Preserve the stage-specific choice of outputs. |
| TLAS mesh preparation and vertex expansion | `src/vulkan/raytracing/hwtlas.jl:312` and `src/metal/hwtlas.jl:374` | Now shared in `src/raytracing/geometry.jl`; preserve face order, degenerate filtering, metadata, normals, UVs and packed Float32 vertices. Native buffer creation and AS build calls remain in the backends. |

Use existing tests for each affected path and add checks only for uncovered
behavior. Report actual net source lines removed after each batch, with the
tested devices and any untested platform explicit. Do not sum overlapping audit
estimates or count moving a file as deleting its implementation.

## Larger consolidation batches

The audits also identify duplicated algorithms and ownership state, whose
implementations differ in syntax. These require caller and invariant checks,
not just matching text:

1. Memory pressure, allocation/release and retirement: core pool policy versus
   Vulkan `maybe_collect` and backend ownership bookkeeping. Give each decision
   one core owner; retain native allocation requirements, memory types and calls.
2. Kernel/launch/pipeline caches and eager versus recorded execution: one compile
   and operation path, with execution context supplied by dispatch. Remove old
   stores/walkers when the common implementation becomes usable.
3. TLAS member/range/update/lifetime management: one portable algorithm with
   device-specific record packing and native build/update commands.
4. Video session/DPB/decode bookkeeping shared by H.264 and H.265: retain codec
   parsing and codec-specific native structures; consolidate only the common
   session, allocation and command-lifetime logic.

For these batches, map every old implementation and caller to the single owner
before writing replacement code. A foundational replacement slice must name
the copies it supersedes and its deletion gate. Copies can remain temporarily
in the separately runnable baseline; replacement core/backends may not reproduce
them. Small independent consolidations happen first. Architecture-dependent
consolidations are the first replacement work, with paired deletion obligations.

### Active paths and replacement gates

These checks supplement the full file/caller dispositions in the existing
audits. They do not mark P0.A10 complete or assert that every caller has been
migrated.

| Group | Active ownership and callers checked | Existing target owner and deletion gate |
|---|---|---|
| Memory pressure | [The memory batch](consolidation-memory.md) removes Vulkan's independent collection budget, full-GC/reclaim coordination and extra pooled-allocation retry. Both native allocation entry points use core's shared controller; trim, soft-cap and heap-pressure triggers share cost/full history and one atomic claim. Native heap facts and channel selection remain native. | P2.A10 still replaces the complete pressure sequence, including cache eviction, in the target acquisition path. Current policy consolidation is qualified against existing pool-trim, recorded-pool protection, allocation and queue-lifetime baselines on AMD/NVIDIA. Unified token retirement remains a separate obligation below. |
| Lifetime | Regions retire through `memory/pool.jl`; native objects retire through `graph/lifetime.jl` channel lists; `vk_free!` enqueues native buffers and `quiesce_before_reclaim!` bridges channel draining and pool reclaim. These are distinct object domains, not identical functions. | P2.A5/P2.D13: one token-based core release engine. Replace all enqueue/drain/reclaim callers together, retaining recording holds, cross-queue last uses, finalizer-safe enqueue and exactly-once native destruction. Backend `rawfree` remains the actual driver destructor. Existing `test_hold_lifetime`, `test_batch_queue_lifetime`, memory-safety and recorded-pool-trim cases are regression gates. |
| Compilation caches | Vulkan `DeviceCaches.pipelines`/`pipeline_order`, `linked`, and `launchplans` are separate layers. Recorded `compile_dispatch` and eager `ka_backend.jl` both reach `launch_plan`; the latter reaches `get_compiled_kernel_and_pipeline` and `get_compute_pipeline`. `clear_kernel_cache!` now clears all three compute layers and the driver-cache probe delegates to it. Native graphics/mesh/ray creation uses core `CompilationCache`; Metal raster/mesh publication now shares a device-owned cache rather than process globals. Metal stage code already follows `CodeInstance`. The publication batch qualifies concurrent creation and removes captured shader values from graphics keys; idle retention and compute-entry ownership remain open. | P1.A1 and execution-contracts.md §13: compute ownership at `Kernel`/`CodeInstance`, one native entry per device, and shared core publication/idle-object policy for graphics/ray pipelines. Replace the stores and invalidation callers together. Preserve compiler configuration, workgroup/ray-query/layout keys, shader edit invalidation, warm reuse, in-flight holds and dispatch allocation baselines. Driver disk-cache serialization is a native primitive, not a second core ownership policy. |
| Eager/recorded execution | `array/launch.jl` already chooses kernels once through `ArrayLaunch`; `runlaunches!` and `dispatchlaunches!` consume those choices differently. Eager KI preparation now delegates to the same core `kikernel` as declared preparation; KA construction and its repeated declared capture policy are shared too (consolidation-kernel-preparation.md). The legacy `Immediate` emitter also handles front-of-run work on Metal: `openrun`/`closerun!` use it around recorded submissions. Deleting it because Metal has recordings would break that caller. | The existing one-dispatch/Builder contract in api.md §§2.4/4.5: share compilation, resource accesses, pending changes and emission; location selects declaration or eager submission. Remove `Immediate` and old front-of-run walkers only after their pending-change work has an equivalent Builder path. Preserve `test_recorded_run_semantics`, recorded-plan replay, eager/graph numerical baselines and zero warm-path allocation expectations. |
| TLAS state | [The mutation batch](consolidation-tlas-mutation.md) replaces the Vulkan/Metal batch structs and geometry registration, push, transform validation, deletion, visibility and flat-range algorithms with core `InstanceBatch`/`BatchedTLAS` methods. Triangle metadata belongs to batches; the backend metadata lists and unused Vulkan offset table are deleted. The intermediate subtype excludes low-level built `MetalTLAS`, whose fields serve a different contract. Vulkan still queues updates until `sync!`; Metal still submits them immediately. Native packing, build/refit, residency and retirement remain backend-specific. | Existing `TLAS{P}`/`MemberSet`/`InstanceRange`/`RangeEntry` and pending-change application remain the target owners. Replace the remaining pending-change/build/refit ownership without reproducing the shared mutation algorithms. Native qualification and untested Metal execution are recorded per batch; this is not completion of the target TLAS replacement. |

## Avoid false duplication claims

- GEMM, GEMV and FFT kernel families already live in shared `src/array/` files.
  Inspect their selection/declaration/submission wrappers; do not call different
  measured kernels or tilings copies simply because they implement multiplication.
- `ArrayLaunch` already shares eager/graph selection. Its later replacement
  must remove remaining execution layers, not create another selection system.
- The two Vulkan format functions have overlapping mappings but different
  meanings: `Float32` is a depth attachment in `vkformat`, and an R32 color/sample
  value in `julia_to_vk_format`. A shared table must retain usage distinctions.
- Generic video refusal methods and Vulkan implementations are dispatch layers,
  not duplicated decoder bodies. Native API encoding differences stay native.

The first batch is implemented and its tested scope is recorded separately.
The [TLAS/cache follow-up](consolidation-tlas-cache.md) also shares triangle-table
preparation and gives compute-cache invalidation one owner. It records real
native-pipeline content reuse, so the target cache replacement must preserve
deduplication while removing the remaining independent retention policies.
The larger source-disposition map is not evidence that the larger consolidations
have already been implemented, nor does
it prove that the complete project can fit in 10,000 lines.

The [semantic similarity experiment](similarity-2026-10-21/README.md) now adds
a reproducible discovery channel across 1,987 methods/regions, comparing two
embedding models with lexical retrieval. It includes ranked cross-file results
and manually checked distinctions; candidate scores are not deletion approval.
