# Shared staged GEMM generator — 2026-10-22

Baseline: `b3f991a`. Wide-index, narrow-index and ablation kernels now come from
one quoted body. Index width and the diagnostic staging guard are generated
before compilation; names, signatures, tables, shared-memory layout, barriers,
unrolling and arithmetic remain. Prefetch, double-buffer and vec4 schedules
retain their separate bodies. The stale claim that wide indexing remains broken
is removed; the native regressions exercise it directly.

The existing staged GEMM, bias/epilogue and fp16-accumulation suites pass **2,430
assertions** across RX 7900 XTX and RTX 4000 Ada. A paired probe captures 66
native baseline cases, then checks every output byte and all three table key
sets (69 assertions). Twenty additional checks confirm the intentionally wrong
:nostage diagnostic still reuses only its first tile. A fresh Julia session
loads the current source and independently repeats all 66 CPU-reference cases.
Validation/synchronization validation are enabled; final drains are empty.

[Probe](gemm-generator-probe.jl) preserves old native recordings across Revise
so [timings](gemm-generator-timings.tsv) compare old/new programs interleaved in
one process on the same arrays. Medians of 15 pairs after five warmups differ
by less than 0.3% for the measured product: runtime preservation, not a speedup.

All 66 generated modules have matching live instructions and declaration
multisets after debug/ID canonicalization and removal of unused OpAccessChain
results. Raw differences include declaration order and unused shared-pointer
calculations. The [comparison](compare-gemm-shaders.py) only analyzes artifacts;
it does not rewrite the shaders used by tests or timings.

All 115 source files parse. Source shrinks 53,379 → 53,165 physical lines and
23,222 → 23,111 implementation lines: 111 implementation lines removed, with
the remainder from prose cleanup. Native Metal/ROCm execution is not qualified here.

## Shared tile products

Baseline: `e0e1c29`. All eight staged families now expand the same tile-product
quote, parameterized by packing width and double-buffer offsets. Queue storage,
staging, barriers and epilogues stay in their respective schedules. Plain prefetch
and gather retain their measured, distinct register queues.

The paired probe's `baseline(contexts;tables=STAGED_TABLES)` covers 180 native
cases, including all table entries and 1/3/9-block loops. All output bytes and
eight table key sets match (188 assertions); all 180 shader modules have matching
live instructions/declarations under the comparison above. The staged, epilogue
and widening suites pass 2,486 assertions, including new vec4 k-term coverage.
A fresh Julia session also loads all eight families and passes all 180 CPU-reference
cases. Validation and synchronization validation drains are empty on both GPUs.

[Interleaved medians](staged-mma-timings.tsv), using the same protocol above,
change by less than 0.7%; this preserves runtime rather than establishing a
speedup. Families were measured separately: these numbers do not rank schedules.
Source shrinks 53,165 → 53,085 physical and 23,111 → 23,051 implementation
lines: another 60 implementation lines removed, without a runtime abstraction.
