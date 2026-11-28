# Shared memory extents

Baseline: `3c134d9`. Core checkedbytes/checkspan replace repeated unchecked
size/span arithmetic across pool arrays, transients, host views and native
allocation/view wrappers. Explicit-rank DeviceArray construction uses the same
checked constructor. Scratch alignment slack also uses checked addition.

A pre-change probe reproduces an overflowing dimension product reaching the
allocator as zero bytes, and a negative-offset host view escaping its declared
span. Invalid dimensions/types now throw before native access; trydevicearray
returns nothing for representational overflow. Valid zero extents remain valid.

Qualification: 38 host fault assertions, 8,982 pool assertions, 142 AMD/NVIDIA
refusal/transfer assertions, 52 native pool checks, 16 warm replay/allocation
checks and 80 explicitly selected ROCm backend assertions pass. Fresh loading
repeats all 9,162 host-pool, fault and native transfer/refusal checks. Native Metal remains
unqualified; its wrappers use the same tested pure checks.

The boundary guard failed while reflecting an imported UnionAll/Union alias.
It now checks union members recursively instead of calling parentmodule on a
union; imported/core aliases and backend-containing unions have regressions.
Fifteen boundary assertions pass with one existing pure-method triage broken.

All 115 sources parse. Prose cleanup preserves definitions. This batch removes
134 physical lines and adds 18 implementation lines against its baseline,
excluding concurrent host-view upload changes. Shared validation repairs missing
checks rather than claiming an algorithm-size reduction.

## Copy-range follow-up

Against `5cdedb7`, a native negative control writes beyond a three-element view
into its parent’s fifth element. Shared copybounds now rejects bad source and
destination ranges before pointer arithmetic or submission. Negative counts throw;
zero-length Base.copyto! remains a no-op. Valid integer offsets normalize to Int.
The byte transfer also checks its backing storage span.

Array and word-view copies use the existing typed upload worker, including
partial source ranges. Core owns contiguous-host-layout detection. The pointer
and materialization paths are no longer repeated by those array wrappers.

[Paired measurements](copy-bounds-probe.jl) / [results](copy-bounds-allocations.tsv)
preserve warm host allocation exactly on AMD/NVIDIA: 592/624 bytes for ordinary
Array/word-view copies and 32/64 bytes for mapped copies. An initial heterogeneous
tuple loop boxed arguments; specialized range checks remove that regression.
These numbers exclude native allocation and do not measure GPU throughput.

636 copy/build/lifetime, cross-queue and warm-replay assertions pass on both GPUs,
including fresh loading and the concurrent whole-word-view copy entry point.
The paired allocation probe passes all 20 checks.
The older memory-safety reduction fixture’s implicit seq_cst was correctly
rejected; it now explicitly uses monotonic updates, consumed after synchronization.
The compiler refusal is unchanged. All 115 sources parse. This batch removes
39 physical lines and adds 12 implementation lines, excluding the concurrent
whole-word-view wrapper. The added code repairs missing validation.
