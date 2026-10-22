# Shared dense GPU strides — 2026-10-22

Baseline: `459d03b`. GPUArraysCore's AbstractGPUArray inherits DenseArray.
Core now describes dense vectors/matrices and matrix reshapes once; Vulkan and
Metal copies are deleted. Native array indexing continues to own view offsets.
Complex adjoints still require conjugation and are refused by stride projection.
This also fixes ROCm gemmlaunch!, whose missing stride method previously returned
nothing even after densification. A direct scalar product regression pins it.

2,658 GEMM assertions pass across AMD/NVIDIA Vulkan, including all staged families.
Both Vulkan validation/synchronization drains are empty. The initial extra 1,329
“ROCm” assertions used Vulkan; [explicit ROCm qualification](consolidation-gemm-catalogs.md)
supersedes that count and passes 1,341 native GEMM checks.
The staged-count fixture now uses core's portable KI compile/launch protocol;
constructing an uncompiled KI.Kernel directly was Vulkan-specific.

ROCm's 77 backend assertions pass with the existing split-capture SDK source;
[test/rocm/README.md](../../test/rocm/README.md) pins the dependency and limits.
The registered package's missing capture API was resolved by locating that source,
without changing recording semantics. No native Metal qualification is claimed.

All 115 source files parse. Source falls 50,824 → 50,805 physical and
22,877 → 22,872 implementation lines (five implementation lines removed).
