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
