# Shared incremental TLAS registration and mutations

Baseline: `66e0bc2` after integrating concurrent native fixes.
The first qualification ran from `d367d09`. This batch replaces the duplicated
algorithms and state in
Vulkan's and Metal's incremental TLAS implementations. It does not implement the
target replacement runtime or finish the larger consolidation task.

## Owners and deleted copies

| Previous copies | Shared owner / native remainder |
|---|---|
| `InstanceBatch` and `MetalInstanceBatch` structs | Core `InstanceBatch{Tri,Geometry,Records}`. Native aliases select an object reference for Vulkan or a table index for Metal, plus the native record array. Batch count, handle, metadata, mask visibility and SBT offset have one representation. |
| `hwtlas_add_geometry!` / `add_geometry!`, `addgeometry!`, backend `addbatch!`, all backend `push!` variants | `raytracing/mutation.jl`: identity-based geometry registration, mesh preparation/bounds, record/count validation and registration. Native methods only build geometry, upload arrays or select a geometry reference. |
| Backend transform/count/handle validation, single-transform convenience, flat-range lookup | Shared core mutation methods; invalid ranges/counts fail before mutation. Native Mat3x4 arrays retain identity; they do not acquire a host conversion or upload. |
| Backend delete, visibility, instance-buffer, refit marking and count/bounds methods | One set of methods on `BatchedTLAS`. Low-level built `MetalTLAS` remains outside this interface. |
| Both `blas_triangles` lists and Vulkan's `blas_offsets` table | Deleted. Hit metadata already belongs to batches. The offset table had no reader; `HardwareAccel` still receives its distinct, live batch-offset tables from `trianglemetadata`. |
| Vulkan nested batch × BLAS compaction and parallel-list rebuild | Identity set plus in-place filtering. Build-time native TLAS keepalive lists are copied already, so old structures retain their own BLAS owners. Dropped BLAS retirement still follows the existing native path. |

Public calls retain their native array identity, per-instance ids/masks, batch
ordering, handles and return values. Metal submits transform writes immediately;
Vulkan retains its last-write-wins queue until `sync!`. Neither submission policy
is silently changed to match the other. Negative batch counts now refuse before
creating geometry, like oversized counts.

Caller search covered `src`, `ext`, `test`, Raycore, Hikari, RayMakie and
JuliaVision. No consumer referenced the removed private registration helpers or
the deleted backend metadata fields. Existing consumer-facing `dirty`,
`instances`, `blas_list` and adapted-form access remain available.

## Qualification

The full integrated run on `89ce3bd` plus this batch passed **558 assertions**
in a fresh Julia session:

- 56 core mutation assertions with native-object and integer-index geometry
  representations; 30 existing geometry/triangle-metadata assertions.
- 342 hardware-TLAS assertions across RX 7900 XTX and RTX 4000 Ada, with Vulkan validation
  and synchronization validation enabled. Existing push, delete, metadata,
  record identity, mesh mutation, refit, queue, device ownership, held leaves,
  trace and lifetime cases pass. The million-grain case builds two million
  instances and performs 1,000 refits on each GPU, then verifies the final hits.
- 78 BVH-publication/hit-agreement assertions across CPU/AMD/NVIDIA, including
  the software/hardware comparison on both GPUs after the caller fix below.
- Eight permanent Vulkan mutation assertions: warm GPU-array transform updates
  allocate zero host bytes and preserve ids/masks on both GPUs.
- 42 texture assertions from the incoming mipmap/upload changes, on both GPUs.
- Two final Vulkan validation drains, both empty.

After integrating `66e0bc2`, which adds procedural-buffer state to native BLAS
objects, a second fresh-session run passes **436 focused assertions**: core
contracts and metadata, native registration/mutation/refit/lifetime/trace,
zero-allocation updates, and the incoming device-AABB build/refit cases on both
GPUs. Both final validation drains are empty. These runs overlap; their counts
are not added as distinct coverage.

The 100-iteration mesh replacement checks retain the same measured live-region
and pool-block counts. These are existing workload checks, not a measured
before/after speedup from this consolidation. All 114 current source files parse.
Native Metal execution is unavailable on this Linux host; parsing and the host
index-representation tests do not qualify it. The replacement API suite still
cannot run against this legacy implementation.

The wider comparison initially failed on both GPUs while compiling Raycore's
software BVH. The unchanged Mantle `2b09378` worktree reproduces the same failure:
Atomix's default sequentially consistent atomic ordering is now correctly
refused by Lava. Raycore's sibling-arrival counters require release/acquire
publication, not a total order across counters. Raycore `9062e56` explicitly
requests acquire/release at both RMWs; Metal's required device fences remain.
Its existing 4,032-triangle publication test passes on CPU and both GPUs, and
both software/hardware hit comparisons pass. Lava's ordering refusal is not
weakened. This is a caller repair, not a fallback or target design change.

## Size accounting

The parser-based `tools/similarity/inventory.jl` measures every Julia source file
in `src` and `ext`. Mixed code/comment lines count as implementation; docstrings
are identified by syntax; embedded native shader strings remain implementation.
The [per-file inventory](source-size-2026-10-21.tsv) compares the same metric at
the baseline and after this batch.

| Metric | Baseline | After | Reduction |
|---|---:|---:|---:|
| Physical source lines | 55,233 | 54,826 | 407 |
| Lines containing implementation | 23,510 | 23,346 | 164 |

The other 243 removed lines are documentation, comments and blanks attached to
deleted copies. This is not a claim that 407 lines of implementation disappeared.
No code was compressed or documentation moved to manufacture the reduction.
The baseline also contained 13,646 docstring-only lines, 11,620 comment-only
lines and 6,457 blanks. A synthetic parser check distinguishes docstrings,
mixed comments, blanks and `#` inside native shader source.

The proposed 10–20k final size is not established by this inventory. Report both
physical and implementation counts; do not use their difference to claim the
requested architectural simplification has happened.

## Still open

The remaining independent TLAS pending-change, build/refit and native lifetime
owners still need the planned MemberSet/range/change machinery. The shared
mutation bodies are the source to carry forward, not duplicate anew per backend.
The other major ownership groups—cache idle/active holds and compute entries,
unified token retirement, Builder execution and decoder session/DPB policy—also
remain open. P0.A10 stays `doing`.
