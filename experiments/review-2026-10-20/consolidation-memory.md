# Shared memory collection policy — 2026-10-21

Compared with `0e669061`. Core now owns collection admission, measured cost,
full-collection history, re-entry/concurrency exclusion, quiescence and the
single retry after a genuine native allocation refusal. Vulkan supplies heap
facts, the allocating channel, native allocation/destruction and final OOM
diagnostics. Its separate EWMA, collection timestamps, productivity bonus and
reclamation guard are deleted. Heap-pressure urgency boosts remain; all entry
points use core's measured-cost budget.

The pressure trigger retains the 75% threshold (50% when blocking), and budget
boosts above 90% and 95%. A pressure collection includes a full collection only
when the pool also exceeds its soft cap and the shared full interval permits
it. Trimming shares the full-collection history, charges its collection work,
and cannot trigger a second collection through explicit reclamation. Queue
waiting is excluded from GC cost. Explicit reclamation bypasses pacing but
shares the same claim and history. Exceptions release the claim and charge
collection work already performed.

Recovery flushes submission holds before GC, then drains native retirement,
reclaims regions and trims empty blocks. A caller-supplied recovery scope keeps
this on the allocating channel. Recorded pools remain protected. This does not
replace the distinct region and native-object retirement engines; their token
ownership replacement remains P2.A5/P2.D13.

## Qualification

- `test_pool.jl`: 8,979 assertions, including budget-limited allocation,
  exactly one retry, no retry when waiting is forbidden, flush-before-GC owner
  release, allocating-channel scope and trimming without another collection.
- `test_collection_policy.jl`: 29 assertions covering shared pacing, independent
  devices, full-history reuse, nested/concurrent entry, exception cleanup and
  zero allocation on paced rejection.
- AMD RX 7900 XTX and NVIDIA RTX 4000 Ada, each with Vulkan validation and
  synchronization validation: 50 assertions from pool trim policy, recorded-pool
  protection, oversized-request refusal, explicit trim with work in flight,
  batch-queue lifetime and injected heap-pressure observations. One additional
  validation-message assertion per GPU; no captured messages. Total: 102.
- All 112 source/extension files parse with JuliaSyntax; 1,994 chunks extracted.

Total: 9,110 passing targeted assertions. This is not the full legacy suite,
physical GPU exhaustion qualification or execution of the unimplemented target
specification. Metal native execution is unavailable on this Linux host.

The lifetime fixture initially retained interpreter temporaries at top level.
Running recovery inside compiled functions and observing the owner through a
WeakRef makes its lifetime explicit. A negative control using the old
GC-before-flush order retains the entire 4 KiB block; the corrected order
reclaims it and succeeds on the bounded retry.

The existing 400 GiB probe initially failed with an actual Vulkan buffer-size
validation error, not native OOM. Core's already documented `maxalloc` limit
was not enforced during pool growth. Growth now checks it before native
allocation and clamps the default block size to the device limit. The probe
now explicitly tests oversized-request refusal and subsequent legal work;
genuine refusal/retry is tested with the budget-limited device. No claim is
made about dedicated native allocations or a need for a new fallback.

A warmed microbenchmark (median of 11 runs of 200,000 calls) measured paced
rejection at 26.1 ns for the original rate check and 39.2 ns for the shared
controller. The latter includes atomic coordination and allocates no memory.
This is an admission-check measurement, not a workload speedup claim.

## Size and remaining work

Physical source inventory after that batch was 112 files / 55,000 lines, 73 fewer than its
batch's baseline (including comments and blank lines). Backend collection and
retry policy is removed rather than copied into another backend. Tests and
this report are outside that count. Cache ownership, eager/recorded execution,
TLAS mutation state and the unified token retirement engine remain open in
[the ownership/deletion map](consolidation-order.md).

## Shared arena-buffer allocation

Baseline `bc6a542`: one native allocation/binding/wrapping body replaces three
in vulkan/graph.jl. Core still owns blocks and reclamation. Native classes retain:

| Class | Memory | Mapping / address |
|---|---|---|
| Buffers | Required local; existing device_memory budget check | Unmapped, BDA enabled |
| Unified | Required coherent host access; prefer local | Mapped once, BDA enabled, indirect usage |
| Readback | Required coherent host access; prefer cached | Mapped once, no BDA |

200 native assertions pass across AMD/NVIDIA: pool reuse/trim, cached downloads,
record/replay, shared-arena growth, device retirement, zero-byte allocations and
empty validation drains. AMD exercises the 4 GB allocation spill; that case skips
NVIDIA because its allocation limit is not below half capacity. Source shrinks
53,478 → 53,379 physical and 23,266 → 23,222 implementation lines; all 115 files parse.
