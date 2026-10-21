# Native facts and compiler evidence — 2026-10-20

This separates read-only native queries, source reproduction and specification
requirements from GPU execution. The replacement Mantle API has not run. No
Metal/CUDA/HIP native path, presentation frame or performance claim was tested
by the probes below.

## Local Vulkan queries

Commands: `vulkaninfo --summary` and `vulkaninfo`. Selected fields, queue families,
UUIDs, driver versions and the full report's SHA256 are retained in
[native-facts.json](native-facts.json). The raw report was written to
`/tmp/mantle-review-vulkaninfo.txt`; that temporary file is not a durable artifact.
Instance version is 1.4.363. Facts describe this installation, not every driver
of the same vendor. Advertised support also needs enabled features and compiler
support before Mantle may publish it as usable.

| Adapter | Driver | SBT handle / handle alignment / base alignment | Maximum stride |
|---|---|---|---|
| RX 7900 XTX | RADV, Mesa 26.2.4 | 32 / 16 / 32 bytes | 16384 bytes |
| RTX 4000 Ada | NVIDIA 595.104.02 | 32 / 32 / 64 bytes | 4096 bytes |
| Ryzen 7900X integrated GPU | RADV, Mesa 26.2.4 | 32 / 16 / 32 bytes | 16384 bytes |
| llvmpipe CPU | Mesa 26.2.4, LLVM 23.1.3 | 32 / 16 / 32 bytes | 16384 bytes |

All four report sparseBinding, sparseResidencyBuffer, rayQuery,
rayTracingPipeline, swapchainMaintenance1 and vulkanMemoryModelDeviceScope true;
accelerationStructureIndirectBuild is false. This disproves using this Lavapipe
fixture as inherently lacking sparse buffers or RT. DEV-10 now explicitly
overrides those capabilities. Queue counts are facts, not evidence of concurrent
execution or preemption. Suite fixtures inspect canonical distinct queue states;
verified CUDA/Vulkan pairs are supplied explicitly, never matched by name.

## Compiler defect F33: reproduction and repair

[probe-atomic-demotion.jl](probe-atomic-demotion.jl) extracts and executes the
actual demote_atomic_accesses! definition from Lava/src/compiler/target.jl in a
private Julia module. Julia 1.12.7 / LLVM.jl 10.1.0, no Lava import or GPU. The
[source-hashed result](atomic-demotion-result.txt) shows an acquire atomic load
and release atomic store becoming ordinary instructions. This is a reproduced
compiler transformation defect, not a GPU visibility experiment or a regression
test that blesses the defect.

A second source finding is Lava/src/compiler/spirv/emit.jl:8282–8330: its RMW
mapping groups sequentially consistent with acquire/release and calls them
equivalent. They are not generally equivalent: LLVM's sequential consistency
requires a total order that acquire/release alone does not establish.
[Vulkan's memory model](https://docs.vulkan.org/spec/latest/appendices/memorymodel.html)
forbids SequentiallyConsistent semantics; the
[LLVM ordering contract](https://llvm.org/docs/LangRef.html#ordering) still
applies to the input. Reject unsupported raw orderings before enqueue unless a
separately proved lowering exists. The CAS failure path does read failure_ordering
separately; this review does not claim that it ignores it. Fence lowering remains
a qualification gate. Native device-scope support alone does not repair a compiler.

The defect above is historical and retained as the negative control. The
[repair and qualification](../atomic-ordering-2026-10-21/README.md) remove the
pass, preserve supported atomic loads/stores/fences and independent CAS
orderings, reject unsupported sequential consistency at compilation, and remove
Atomix overrides that discarded the caller's ordering. Device scope is emitted
as Device; Mantle queries/enables its native feature. The current compiler
suite and four-device GPU visibility results are recorded there. The old probe
can replay the archived source with `MANTLE_ATOMIC_BASELINE`; it must not be
used to claim that the repaired compiler still demotes atomics.

F33's original compiler defect is repaired. F42 still owns complete per-target
operation/type/scope/order eligibility. These tests do not qualify arbitrary
mixed-scope chains or other native GPU APIs. Default Atomix RMW requests
sequential consistency, so it now fails rather than receiving a weaker
implementation; explicit caller orderings require algorithm-level review.

## Requirements checked against primary specifications

- Presentation retirement is independent of rendering completion. Maintenance1
  [present fences](https://docs.vulkan.org/refpages/latest/refpages/source/VkSwapchainPresentFenceInfoKHR.html)
  prove payload release, not displayed completion. The
  [recreation sample](https://docs.vulkan.org/samples/latest/samples/api/swapchain_recreation/README.html)
  supplies acquisition-history retirement; the
  [semaphore reuse guide](https://docs.vulkan.org/guide/latest/swapchain_semaphore_reuse.html)
  explains the remaining legacy shutdown limitation. These are requirements,
  not locally executed WSI qualification.
- CUDA [virtual-memory APIs](https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__VA.html)
  and HIP [VMM](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/memory_management/virtual_memory.html)
  map through host operations without a stream parameter. They do not prove
  queue-ordered sparse binding. Committed arenas are the initial baseline.
- Metal 4 does expose
  [sparse buffer mapping operations](https://developer.apple.com/documentation/metal/mtl4updatesparsebuffermappingoperation),
  explained in Apple's [Metal 4 session](https://developer.apple.com/videos/play/wwdc2025/254/).
  It needs hardware, SDK/binding, residency and queue-lifecycle qualification;
  sparse texture support alone is insufficient. No local Metal result is claimed.
- Vulkan [ray-tracing layout](https://docs.vulkan.org/spec/latest/chapters/raytracing.html)
  determines SBT addressing and alignment. The
  [SPIR-V RT operands](https://github.khronos.org/SPIRV-Registry/extensions/KHR/SPV_KHR_ray_tracing.html)
  use only four low bits of the trace SBT offset, so the fixed-layout portable
  Vulkan path cannot promise arbitrary ray-type offsets. TLAS group layout and
  pipeline compatibility are explicit in execution-contracts.md §7.
- Vulkan [heap budget/usage](https://docs.vulkan.org/spec/latest/chapters/memory.html),
  CUDA [free-memory queries](https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__MEM.html)
  and Metal's [recommended working set](https://developer.apple.com/documentation/metal/mtldevice/recommendedmaxworkingsetsize)
  have different native meanings. Core normalizes additional headroom per heap;
  it must not add already allocated bytes to the next request and compare that
  total against free bytes. Metal headroom remains an estimate.

Native execution, numerical/image equivalence and interleaved performance gates
remain in the ledger; none is discharged by these links or read-only queries.
