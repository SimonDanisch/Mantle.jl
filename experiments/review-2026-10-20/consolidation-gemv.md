# Generated GEMV kernels — 2026-10-22

Baseline: `a367c3b`. Two typed generated callables replace the K/N-contiguous
kernel dictionaries, runtime eval, pre-generation lists and shared-memory ID
counter. Shared IDs derive from the family and configuration; literal unrolling,
reduction order and separate contiguous-axis algorithms remain. N-contiguous
selection returns one tuple of launches instead of an allocated vector.

Configuration now rejects zero unroll, invalid reduction trees and oversized
blocks. Defaults fit unusual positive workgroup limits: limit 384 previously
selected an invalid 12-group reduction. Dimension checks cover W; empty outputs
emit no dispatch. Eager and recorded numerical behavior remains shared.

Qualification:

- Existing baseline GEMV suites pass on AMD/NVIDIA Vulkan before edits.
- 88 paired old/new output-byte and validation checks cover 43 configurations
  on both Vulkan GPUs, including partial subgroups and nonstandard unrolling.
- 509 numerical, replay, configuration and cache-key assertions pass across the
  three native paths. Existing performance floors pass; both Vulkan validation
  and synchronization drains are empty.
- Six negative-control/configuration and 23 source-portability checks pass.
- The consumer fixture reproduces closed-module eval failure with the old GEMV
  constructors. The new fixture precompiles; fresh loading passes 169 native
  assertions, one isbits check and one empty validation drain. Probe-only setup
  errors (missing DISPLAY and an incorrect wait helper) were corrected.

[Paired allocation probe](array-kernel-probe.jl) and
[results](array-kernel-allocations.tsv) pass 24 checks. Over 100 warm calls,
K-contiguous calls save about 169–176 host bytes/call; N-contiguous calls save
32. These are Julia allocations, not VRAM or GPU-throughput measurements.
Native Metal remains unqualified.

All 115 source files parse; prose edits preserve definitions. Production source
falls 50,102 → 49,796 physical lines and 22,822 → 22,820 implementation lines.
Most removed lines are stale prose; the structural removal is the mutable
catalog/ID ownership, not hundreds of arithmetic lines.
