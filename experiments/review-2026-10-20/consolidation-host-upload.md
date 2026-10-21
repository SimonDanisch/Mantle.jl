# Shared typed host upload — 2026-10-22

Baseline: `f80dc6f`. Array and byte uploads delegate to one typed pointer transfer.
Dense host Arrays retain their original storage; other layouts materialize once
in logical order. The array wrapper preserves its destination through the call;
the shared worker preserves host source storage. Boxed/union elements are rejected.

The [paired allocation probe](host-upload-probe.jl) uses the same 4 MiB source and
destination for old/new calls. [Results](host-upload-allocations.tsv): roughly
8 MiB of Julia host allocation becomes 624–640 bytes on ordinary unmapped buffers
and 32 bytes on mapped buffers, on both RX 7900 XTX and RTX 4000 Ada. These numbers
exclude native allocations and do not measure GPU throughput. Native mapping,
Readback scratch, submissions and synchronization remain in copy_buffer!; this
removes wrapper copies and does not qualify replacement-runtime restores/F74.

102 native assertions cover both allocation classes, logical order of views and
permutations, primitive/complex/Bool/static-array inputs, exact NaN/signed-zero
bits, offset/canary preservation, byte/typed entry points, empty inputs and bounded
Julia allocation. Twenty paired-probe assertions also pass. Validation and
synchronization validation are enabled; final message drains are empty.

Six source files now describe pure adaptation and explicit holdleaves! retention;
stale claims that Adapt pins allocations are removed. Prose edits preserve their
JuliaSyntax definition trees. All 115 source files parse. Source decreases
50,995 → 50,892 physical and 22,893 → 22,889 implementation lines. Four lines of
implementation disappear; the main gain is removal of transfer-sized host copies.
