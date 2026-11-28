# Shared pipeline publication — 2026-10-21

Compared with `2b09378`. Native raster, mesh and ray-pipeline publication now
uses core's `CompilationCache`. It replaces Vulkan's independent lookup/build/
store sequences and Metal's separately locked lookup and store. Metal's two
process-global pipeline dictionaries and their locks are removed; raster and
mesh entries share the owning `MetalDevice`'s cache.

One caller reserves a missing key under the map lock, then creates its native
object with no map lock held. Waiters receive the same result or failure through
a level-triggered Event with acquire/release ordering. Different keys can build
concurrently. Failed entries are removed for later retry. Clearing detaches an
in-flight publication: its existing callers can finish, but completion or failure
cannot replace/delete a newer entry. Recursive creation of the same key fails
instead of waiting on itself. Neither producer Tasks nor factory closures remain
cache roots after publication. Warm aggregate hits allocate zero bytes.

Pipeline fixed state is now described in core, separately from compiled stage
identity. Metal keys no longer retain shader-description values/captured data.
Its device is passed through native stage/pipeline creation, including delayed
geometry lowering, and its stage lookup includes the native device. Vulkan
graphics keys retain compiled code identities and compare the full key rather
than treating a UInt64 hash as equality. Identity keys retain their code objects,
so an expired objectid cannot become a different shader's identity. A forced
hash-collision test verifies distinct entries.

This implements shared publication and removes Metal's global pipeline owners.
It does not implement the target's bounded idle-object policy, explicit native
hold acquisition, CodeInstance per-device compute-entry replacement, or complete
thread safety of compiler metadata and submission channels. Compute content
deduplication remains intact. Those remaining obligations are still P1.A1 and
execution-contracts.md §13; do not add a new per-backend eviction policy here.

## Qualification

- 34 host assertions in `test_compilation_cache.jl`: concurrent same-key
  publication, unrelated-key progress, forced identity/hash collision, failure
  fan-out and retry, clear during successful/failed creation, recursive entry,
  zero-allocation immutable aggregate reuse, Nothing results and enabled GC
  finalizers during factories. These run in the ordinary suite.
- 12 native assertions across AMD RX 7900 XTX and NVIDIA RTX 4000 Ada: 16
  concurrent callers receive one native pipeline, cold creation after clear,
  warm reuse, explicit second-device isolation, rendered pixels and validation.
- 842 passing native regression assertions across the same GPUs, with Vulkan
  validation and synchronization validation: graphics pipelines, mesh/geometry
  draws, mesh graph rendering, render graphs and ray pipelines. Two existing
  geometry-lowering-only assertions are skipped on the native geometry paths.
- All 113 source/extension files parse with JuliaSyntax; 1,994 chunks extracted.

Total: 888 passing targeted assertions, two existing skips. Metal source and the
portable publication engine are checked, but native Metal execution is not
qualified: the available Mac worker cannot be used while remote Julia is off.
This limitation is not evidence for a fallback or an API change.

## Baseline defect found during qualification

The broad graphics run initially failed with VUID-vkCmdPushConstants-offset-01795:
an 8-byte address was pushed into a layout with no push-constant range. An
isolated checkout of unchanged `2b09378`, with the same dependencies, device and
validation settings, reproduced the same failure (33 graphics assertions passed,
one errored). The source was `/tmp/mantle-cache-baseline-2b09378`, loaded through
`/tmp/mantle-cache-baseline-env`; the working checkout was not reset.

A recorded graph assigns an argument-memory offset even to a zero-argument
shader. Regular/indexed native draw emission duplicated the constant-write
logic and interpreted every nonzero address as a request to push it. They now
share one encoder which first checks whether the layout declares a range. The
existing multi-attachment graph test now checks validation immediately after
that draw, before a later pipeline creation can report its error. The final
graphics/ray runs pass with no captured validation messages.

## Size

Physical source is 113 files / 55,053 lines: 52 net lines added, including the
shared implementation and its documentation. Four Metal global cache/lock
definitions and repeated backend publication/constant-emission bodies are
removed. This is a correctness and ownership consolidation, not a claim of a
large source-size reduction or a measured frame-speed improvement.
