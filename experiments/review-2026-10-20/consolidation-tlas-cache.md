# TLAS metadata and compute-cache invalidation — 2026-10-21

Compared with `2acc59f`. Both native TLAS rebuilds now use core's
`trianglemetadata` to concatenate shading triangles and generate the hit table.
Vulkan requests the additional raw offsets its `HardwareAccel` needs; Metal
does not allocate those arrays. Outputs are sized once rather than grown by
repeated append/push. Registration order, hidden batches, zero-instance batches,
deletion/reindexing and the `NOTRIANGLES` sentinel retain their existing meaning.
This shares another TLAS bookkeeping algorithm; dirty/pending state and native
build/refit ownership are still separate and remain replacement obligations.

Compute-cache invalidation now has one owner. `clear_kernel_cache!` clears
native content lookup/order, linked kernels and launch plans together;
`no_pipeline_compilation` delegates to it. Previously clearing the two upper
layers silently reused the old native pipeline, contrary to the clear verb's
documented behavior. CodeInstance-owned SPIR-V and the driver's pipeline cache
survive. Queued work retains the native objects it already uses.

Do not delete the native content lookup simply because linked kernels also
refer to pipelines. In the AMD qualification context, six linked kernels had
four distinct native content keys: two keys were shared by distinct compiled
objects. NVIDIA's four linked kernels had four keys. That is real reuse.
The remaining strong retention and FIFO policy must migrate to the shared
publication/idle-object owner in execution-contracts.md §13, preserving this
reuse, rather than replacing it with one native creation per compiled object.

## Qualification

- 30 host assertions in `test_raytracing_geometry.jl`: the existing 15 geometry
  cases plus 15 metadata cases for empty, hidden, triangle-free, zero-instance
  and deleted batches, exact raw/hit offsets and optional-array omission.
- 47 assertions per GPU on AMD RX 7900 XTX and NVIDIA RTX 4000 Ada, with Vulkan
  validation and synchronization validation: batch triangle tracing, deletion,
  instance-buffer access, refit and a final validation-message check.
- 48 assertions per GPU: existing kernel invalidation, caching/allocation and
  recorded-run semantics, the new single-device cache-clear cases, driver-cache
  probe execution and validation-message check. Another 20 assertions exercise
  the new cache-clear cases with explicit second-device isolation. The tests
  verify native rebuild without SPIR-V recompilation, warm reuse and queued
  output across clear/GC. A WeakRef observes the actual native pipeline owner;
  the immutable aggregate wrapper can be reboxed and is not a lifetime probe.
- All 112 source/extension files parse; 1,992 method/region chunks extracted.
  The local semantic index was rebuilt, and the eager/recorded responsibility
  query was rerun against these sources.

Total: 240 passing targeted assertions. The driver-cache probe is not a claim
that every driver honors compile-refusal instrumentation. Metal native execution
is unavailable locally; the shared metadata algorithm is host-qualified and its
native Vulkan use is exercised on both GPUs.

A warmed host microbenchmark with 1,000 batches, 32 integer metadata values and
16 instances per batch compared the original Vulkan append/push loop with the
shared implementation (median of 11 rounds of 100 calls). Outputs compare equal.
Temporary allocation fell from 1,296,216 to 388,368 bytes; preparation time fell
from 165.3 to 60.0 microseconds. These measurements isolate host table preparation,
not the whole TLAS build or frame.

Physical source inventory is 112 files / 55,001 lines: this batch adds one net
line, including documentation and comments in source. It removes repeated
algorithms/invalidation bodies and reduces temporary allocations; it is not a
large source-size reduction. The memory batch's 73-line reduction is separate.

Eager/recorded execution still needs the existing common Builder replacement.
Current Vulkan `preparekernel` already shares low-level adaptation/packing;
`launch_plan` serves eager KA calls and declared compilation. Core `ArrayLaunch`
already shares algorithm choice. Metal's `Immediate` still applies front-of-run
stores/patches before recorded replay, so removing it requires replacing those
callers too. Do not introduce another algorithm-selection or launch-cache layer
as a temporary substitute for the documented owner.
