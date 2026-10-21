# Implementation consolidation before the foundation

Checked against Mantle `45021c6`, including the recently merged Metal fixes.
Current `src/` plus `ext/` contains 111 Julia files and 55,115 physical lines.
These are inventory counts, not an estimate of removable code. The existing
seven audits and `source-disposition.tsv` remain the broad ownership map.

## First consolidation batch

These are current implementation bodies, verified in source, rather than
similar names or historical findings. No implementation deletion is claimed yet.

| Duplicated implementation | Current locations | Consolidation and preserved behavior |
|---|---|---|
| Vulkan memory-type search | `src/vulkan/runtime/memory.jl:1346` and `:1359` | Keep one search. The throwing entry point calls the optional selector and retains its existing error on no match. Preserve type-bit eligibility, required flags, first-match ordering and zero-based result. |
| Borrowed Metal array construction | `src/metal/memory.jl:33` and `src/metal/ka.jl:111` | Have `deviceview` use the existing `deviceslice` constructor. Preserve the non-owning DataRef, shared Managed buffer, byte offset, dimensions and remaining capacity. Verify on Metal, including the recorded indirect-slot caller. |
| Metal attachment/depth setup in raster and mesh pipeline compilation | `src/metal/graphics.jl:665` and `src/metal/mesh.jl:350` | Share the attachment/blend/depth setup and depth-state construction. Keep stage compilation, descriptor types and stage-specific cache keys distinct. Different native descriptors do not justify copying identical attachment policy. |

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

This pass supplies a concrete first batch. The larger source-disposition map is
not evidence that those consolidations have already been implemented, nor does
it prove that the complete project can fit in 10,000 lines.
