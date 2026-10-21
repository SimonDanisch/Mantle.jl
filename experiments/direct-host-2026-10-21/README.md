# Direct host transfer qualification — 2026-10-21

Direct transfer works for the owned host-copy and existing cache-loading source
constructions on all four local Vulkan devices. No staged restore implementation
was added. These are native primitive tests, not execution of the replacement
Mantle runtime.

## What ran

[probe.jl](probe.jl) evaluates the **actual existing** `importhost(bq,ptr,nbytes)`,
`find_memory_type` and `importablehost` function definitions in a small structural
harness. It does not load Mantle or rewrite their native import algorithm.
The legacy array wrapper/finalizer and complete DNNKernels API are not executed.
`cache_read` reproduces the cache's file-read-into-importable-host-buffer recipe;
it is evidence for that transfer segment, not a full model-load benchmark.

[run.jl](run.jl) executes the matrix on the RX 7900 XTX, RTX 4000 Ada, Ryzen 7900X
integrated GPU and llvmpipe. [results.txt](results.txt) records **519 assertions
passed**, zero Khronos/synchronization validation errors including final GC, and
source hashes. [results.tsv](results.tsv) retains every observation and timing.
Owned-copy/cache sources were tested at 4 MiB and 64 MiB. Logical copy sizes are
12 bytes smaller than the imported backing, exercising a non-page-sized extent.

Checks include every payload byte, repeated CPU-source changes and replay of one
recording, source contents, and accepted-work lifetime. A submission deliberately
waits on a separate unsignalled timeline: retirement must refuse, and the source
must survive creator-scope release/GC. After signalling/completion, native import
retirement and source collection are checked. No native object is deliberately
freed while in use. The lease is experimental scaffolding; replacement resource
locks, retries, pending stores, cell publication and cross-graph races remain
separate phase-4 gates.

## Exact native import observations

| Source, 4 MiB | RX 7900 XTX | RTX 4000 Ada | Ryzen iGPU | llvmpipe |
|---|---|---|---|---|
| Private anonymous/importable host memory | copied | copied | copied | copied |
| File read into that existing cache-style source | copied | copied | copied | copied |
| Private writable file mapping | rejected | copied | rejected | copied |
| Shared writable file mapping | rejected | copied | rejected | copied |
| Private protected read-only file mapping | rejected | rejected | rejected | copied |
| Shared protected read-only file mapping | rejected | rejected | rejected | copied |

There are 22 successful exact-import observations out of 32, including the eight
larger owned/cache cases. The ten rejections are retained as failures of those
source/import combinations, not counted as successful transfers or hidden skips.

The diagnostic controls in [diagnostics.jl](diagnostics.jl) bypass Vulkan.jl's
high-level allocation wrapper, initialize the raw C headers, query compatible
memory types, remove storage/BDA use and allocation flags, and retry at one page.
All compatible types are exercised. Owned private mapping-protection/COW-touch
controls leave contents unchanged. RADV still returns INVALID_EXTERNAL_HANDLE;
NVIDIA's protected-file mappings still return OUT_OF_DEVICE_MEMORY. The latter
error is not sufficient evidence of a capacity problem or a general limitation.

[The ioctl trace](ioctl-diagnostic.txt) narrows RADV's rejection to kernel
AMDGPU_GEM_USERPTR returning EPERM, with the anonymous control succeeding.
It does not decode the userptr flags; do not infer that every internal driver
decision has been proved. NVIDIA's precise internal rejection remains unresolved.
Neither result justifies a staging fallback or weakening a required consumer case.

The same run also verifies a second **existing direct-upload construction**:
CPU memcpy into the private final allocation with DEVICE_LOCAL, HOST_VISIBLE and
HOST_COHERENT properties, followed by GPU readback. All four devices pass for a
4 MiB protected file source; file/source bytes remain unchanged. There is no
source import or intermediate upload allocation. This does not prove that every
device has arbitrarily large CPU-visible device-local allocations, or that this
construction is always faster. Selection must preserve placement, budgets and
visibility and use qualified facts.

## Debugged implementation defects

The first byte-correct copy produced a validation error because Vulkan.jl passed
an uninitialized `VkMemoryHostPointerPropertiesEXT` header. Another diagnostic
query exposed the same problem in `VkExternalBufferProperties`. Both require an
initialized input header even though their payload is output. The generator now
initializes sType-bearing singular outputs, as it already did for arrays, and
the checked-in bindings are updated for Linux, Windows, macOS and BSD.

[The C-call regression test](../../../Vulkan/test/host_pointer.jl) passes four
assertions, including a poisoned-header negative control, without requiring a
GPU. [The generator change](../../../Vulkan/generator/src/wrap/initialization.jl)
also has a code-generation regression. Full generator tests have seven existing
failures: the original initializer plus the new regression produces 191 passes /
8 failures; the fix produces 192 / 7. Logs are
[baseline](generator-baseline-with-regression.txt) and [fixed](generator-fixed.txt).
Those unrelated failures remain; the full generator suite is not claimed green.
The earlier diagnostic with the bad header is retained separately and is not the
accepted validation run.

## Memory and timing scope

No explicit intermediate upload allocation exists in the successful direct
constructions. The experiment does allocate its host source, final destination,
verification readback and CPU reference data. It does not claim zero total
allocation, zero driver bookkeeping or a measured OS page-reclamation bound.

GPU timestamps bracket only the imported-source upload. Wall times include
upload, verification readback and completion; first import timings can include
JIT cost. They are not full consumer latency or isolated peak-bandwidth results.
The [alignment diagnostic](alignment.jl), [sequential observations](alignment.txt)
and [interleaved control](alignment-interleaved.txt) compare exact-length copies,
aligned-capacity copies and an exact bulk/tail region split without changing the
production algorithm. The NVIDIA interleaved medians were approximately 10.47,
12.41 and 10.48 ms for roughly 64 MiB; some samples were much slower.
[GPU monitoring](alignment-gpu-state.csv) recorded P8/P0 and PCIe generation
changes on an x4 link. It does not establish a per-sample causal explanation.
No transfer-policy threshold or new fallback is justified from these timings.

Run via the workspace Julia tool:

```julia
include("/sim/Programmieren/VideoEdit/dev/Mantle/experiments/direct-host-2026-10-21/run.jl")
DirectHostRun.run()
```

Metal, CUDA/HIP native APIs and other operating systems were not executed. F74
remains open for replacement integration and the remaining qualification scope.
