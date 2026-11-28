# Shared AS dependencies and submission ownership — 2026-10-22

Baseline: `3898fc2`. Acceleration builds/refits, shared-scratch reuse and indirect
launches use the existing allocation-free memorybarrier! encoder. Producer and
consumer masks remain distinct; duplicate SDK wrapper/array construction and
the second raw indirect-barrier implementation are deleted.

AS commands now name their storage, inputs and scratch through core's existing
hold/stamp engine. This fixes an observed NVIDIA cross-queue READ_RACING_WRITE:
the secondary-queue AS build previously missed the default-queue instance upload.
Completed builds drain their core holds before explicit scratch release. Existing
channel dependencies supply ordering; no device-wide synchronization is added.
The legacy build API still waits for its completion token.

The pooled Raycore importer also uses owning BLAS arrays rather than CPU-indexing
borrowed device views, consumes canonical 4×3 affine storage, preserves masks and
hidden flags, and forwards the chosen build channel. Its old mask path reproduces
eight failures: mask-1 rays incorrectly hit mask-2 instances. These are repairs to
the existing host-mediated importer, not a new fallback or the target migration.

Qualification on AMD/NVIDIA Vulkan, with validation and synchronization validation:

- 206 refit, tracing, lifetime, cross-queue and nonblocking-sync assertions pass,
  including the existing million-instance/1,000-frame refit loops.
- 86 pooled-build assertions cover both builders, direct/indirect RT, nonuniform
  transforms, custom IDs, masks, hidden instances and a secondary channel. Fresh
  loading repeats all 86 with the published GPUArrays dependency.
- [Barrier probe](as-barrier-probe.jl) / [allocations](as-barrier-allocations.tsv):
  six checks pass. Recording 100 warm AS dependencies allocates 115,200 → 0 Julia
  bytes on each GPU. Driver memory and GPU throughput are not measured.
- The indirect queue fixture uses explicit monotonic slot reservation; later
  dispatch barriers publish payloads. Its original implicit seq_cst request was
  correctly rejected by the compiler. Both native chains pass.
- The existing orderstore! callback is registered in the backend vocabulary.
  Twelve boundary assertions pass; the existing pure-method heuristic triage
  remains marked broken.

All 115 source files parse; prose edits preserve definition trees. Source falls
49,597 → 49,457 physical and 22,752 → 22,717 implementation lines: 29 code lines
removed plus six lines of formerly unattached context prose reclassified as docs.
The full Builder/token retirement replacement remains open.

## Unused builder deletion

Against `b44d80f`, remove the private build_blas_pooled and
build_blas_from_primitives implementations and unused as_queue/as_vkctx accessors.
Workspace-wide caller checks find no production consumers, exports, target API
or executable-spec references. The new pooled-builder test was the only caller;
it now exercises ordinary build_blas as the control for the live HardwareAccel
importer, which retains pooled allocation and shared-scratch reuse.

All 86 dependency/transform/ID/mask/visibility assertions pass on AMD/NVIDIA,
both in the existing session and a fresh process with synchronization validation.
All 115 source files parse. Source falls 49,457 → 49,264 physical and
22,717 → 22,592 implementation lines: 125 code lines deleted, with no algorithm
moved elsewhere.

## Shared geometry/build encoding and direct inputs

Against `f62b33d`, host TLAS construction delegates to the device-record builder;
refits share native build-command encoding. Typed instance geometry replaces the
remaining Symbol/keyword branches. The pooled importer uses core trianglevertices;
its duplicate expansion and unused primitive-list bookkeeping are deleted.

Build/refit uploads consume original host arrays. GPU vertex inputs retain a byte
view of the same allocation: the old updateable-build path reproduced a compiler
failure by trying to convert vertex triples into UInt8 values. No new fallback.

432 native/host regression assertions pass, including offset canaries, moved
ray-hit distances, fixed topology, million-instance refits, lifetime, mesh updates
and tracing. Fresh loading repeats 104 build/refit assertions on AMD/NVIDIA.
The [paired probe](as-input-probe.jl) passes 30 native-byte and six allocation/
readback/validation checks. For 589,824 bytes of vertices, warm full refit calls
allocate 599,224 → 9,328 Julia bytes on each GPU ([results](as-input-allocations.tsv));
native allocation and throughput are not measured.

All 115 source files parse. This batch removes 352 physical and 140
implementation lines against its baseline; concurrent pool-cleanup changes are
excluded. Prose cleanup preserves definition trees. No algorithm moves into
another repository.
