# Shared pending stores — 2026-10-22

Baseline: `ec0caa9`. Buffer and TransientBuffer share linear-range validation,
conversion, pending-list locking and consumption. Transient authorization and
persistent length advancement remain distinct. Callback failure retains the list
and dirty flag; a concurrent setter waits for consumption and remains pending.
GPURef's seqlock protocol is unchanged.

Qualification exposed two Vulkan bugs in the old paths:

- Overlapping pending ranges emitted transfer writes without an intervening
  dependency. The old drain reproduces WRITE_AFTER_WRITE under synchronization
  validation. Core now detects overlap; Vulkan emits a transfer-write dependency
  only when required. Sorted disjoint stores skip predecessor scans.
- Persistent stores beyond the inline limit called upload! immediately, escaping
  the run's submission. The baseline emits a submission while encoding; the
  replacement emits none. Persistent/transient stores now share inline update
  and submission-owned scratch-copy emission. Empty stores emit nothing.

Head, compute and transfer barriers share one native memory-barrier encoder,
retaining their different masks and the head-barrier counter.

656 AMD/NVIDIA Vulkan regression assertions, 32 native encoding assertions and
70 explicitly selected ROCm assertions pass. The earlier 74-check “ROCm” label
used Vulkan; [qualification](consolidation-gemm-catalogs.md) corrects the harness.
Coverage includes overlapping/mixed small and >64 KiB
stores, canaries, empty/unaligned stores, discarded encoding, callback failure,
concurrent setters, replay and allocation-free runs. Validation/synchronization
drains are empty. A fresh process passes 46 store checks. Paired 10,000-cycle
host store/drain checks allocate zero bytes before and after for both resources.

All 115 source files parse. Source falls 50,805 → 50,708 physical and
22,872 → 22,844 implementation lines: 16 code lines removed plus 12 lines of
previously unattached prose reclassified as documentation. Prose cleanup
preserves definitions. This consolidates existing stores; the target's pending
Change/Builder migration remains open. Native Metal execution is unqualified.
