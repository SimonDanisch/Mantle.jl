# Generated GEMM configurations — 2026-10-22

Baseline: `65a8bf0`. BlockedGEMM and StagedGEMM replace nine mutable kernel
dictionaries and their population loops. Immutable tiling tuples and one staging
eligibility predicate replace copied catalog membership rules. Julia specializes
on family/configuration; the eight measured schedules retain their arithmetic,
barriers, index widths and register/shared-memory layouts. Four unused tuning
constants are deleted. Native compilation caches remain separate, open work.

DNNKernels and its tuning tools now query eligibility and request the callable
directly (JuliaVision `904d4ce`). A planning-only convolution test no longer allocates multi-gigabyte
operands. It forces gathering while checking chunk geometry: its previous
automatic-selection assumption fails on 96 cores (684 groups versus 768 needed).
The occupancy rule and production convolution algorithms are unchanged.

Qualification:

- 180 old/new outputs and eight eligibility sets match on AMD/NVIDIA Vulkan
  (188 assertions). All 180 paired shader modules have identical live
  instructions/declarations under compare-gemm-shaders.py.
- 2,496 staged/epilogue/widening assertions, 147 batching/subgroup/portability
  assertions, 360 native convolution assertions and 42 recordability assertions
  pass across both Vulkan GPUs. Validation/synchronization drains are empty.
- Consumer precompilation passes; fresh loading passes 1,248 native assertions
  plus isbits/immutable-configuration checks.
- [Explicit ROCm qualification](qualify-rocm.jl) passes 2,019 assertions on
  RX 7900 XTX/gfx1100: 1,341 GEMM, 337 FFT, 169 GEMV/cache-key, 77 backend,
  25 host preparation and 70 store/replay checks. One graphics case is skipped
  through the backend's existing graphics capability query. Native Metal is
  unqualified.

[Interleaved GPU medians](gemm-catalog-timings.tsv), 31 pairs after five warmups,
differ by less than 0.4%. This preserves the measured schedules; it is not a
speedup or a ranking between families. An earlier concurrent qualification run
was discarded before the isolated comparison.

The manual ROCm harness previously used platform-default Vulkan after selecting
a HIP device. Its mislabeled GEMM/GEMV/FFT/store counts are superseded by the
explicit, backend-asserting script above; the native 77-check backend suite was
correctly selected throughout. README instructions and affected reports are fixed.

All 115 source files parse. Production source falls 49,796 → 49,691 physical
and 22,820 → 22,776 implementation lines (44 implementation lines removed).
