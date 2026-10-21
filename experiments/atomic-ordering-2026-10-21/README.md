# F33: atomic ordering repair

The reproduced compiler defect is fixed in `dev/Lava`. This is a compiler
correctness repair, with one required native feature-enablement change in
Mantle. It does not implement the replacement Mantle runtime.

## What changed

- Removed `demote_atomic_accesses!`. LLVM atomic loads and stores reach dedicated
  `OpAtomicLoad` and `OpAtomicStore` emission; fences reach `OpMemoryBarrier`.
  Atomic accesses never go through ordinary load/store decomposition.
- Acquire/release semantics include availability/visibility and the buffer,
  workgroup and image memory classes. A synchronization flag can protect memory
  in another class; selecting only the flag's class was insufficient.
- Sequential consistency fails compilation, including RMW, CAS success/failure,
  loads, stores, fences and default Atomix RMW. Acquire/release is not a valid
  general replacement for the requested total order.
- CAS success and failure retain separate semantics. Where LLVM permits acquire
  failure with relaxed/release success, SPIR-V requires strengthening the success
  path to acquire/acquire-release. The failure path is never weakened.
- Device scope remains Device, with `VulkanMemoryModelDeviceScope` declared.
  Mantle queries and enables the corresponding native feature; QueueFamily is
  insufficient when work can execute on different queue families. Unknown and
  system scopes are rejected. Ordered singlethread requests are rejected because
  Vulkan Invocation scope requires relaxed semantics. Volatile is retained.
- Removed the duplicated RMW IR and overrides that ignored Atomix's order.
  UnsafeAtomics emits the requested order/scope. The Float64 add/sub
  specialization uses integer CAS and preserves the caller's order, scope,
  volatility and alignment. Native Float32 min/max retain the existing
  minNum/maxNum policy and forward order/scope through UnsafeAtomics. Floating
  atomic loads/stores preserve their bits through integer atomic instructions.
  The min/max emitter also now declares its required MinMax capability,
  rather than the unrelated Add capability. Base's array RMW overrides pass
  their `order` on.
  The compatible dependency floors are Atomix 1.5 and UnsafeAtomics 0.4, matching
  the APIs used and the workspace's already installed versions.
- Native CAS qualification exposed a separate pointer-type inference defect:
  byte-offset GEP inference considered ordinary loads/stores but not RMW/CAS.
  Both minimized failures were reproduced before adding their missing cases.
  This is fixed in `spirv/types.jl`, with permanent regressions.

## Verification

[The complete Lava suite](lava-suite.txt) passes **970 checks**, with **two
existing skipped reference-oracle checks**. It includes 119 permanent F33 checks
in `Lava/test/spirv/test_atomic_ordering.jl`: validated SPIR-V, numeric scope and
semantics operands, 32/64-bit buffer/workgroup accesses, CAS success/failure,
fences, volatility, invalid requests, floating atomic accesses and Julia/Atomix entry points. Existing
instruction-emission fixtures now explicitly request monotonic ordering for
independent counters; these fixture changes do not migrate consumer algorithms.

[Native results](native-results.txt) pass **1,717 assertions**, **512 dispatches**
and **zero validation errors**, including synchronization validation and final
collection. The final run starts without Mantle loaded. Flags and payloads use a
separate **DEVICE_LOCAL** allocation on all four devices. The host-visible fixture
holds arguments and readback. Transfer barriers surround each dispatch; there
is no API barrier between the communicating invocations inside the dispatch.

| Device | Load/store | Fences | CAS | RMW release sequence |
|---|---:|---:|---:|---:|
| RX 7900 XTX / RADV | pass | pass | pass | pass |
| RTX 4000 Ada / NVIDIA | pass | pass | pass | pass |
| Ryzen 7900X iGPU / RADV | pass | pass | pass | pass |
| llvmpipe | pass | pass | pass | pass |

Each mode runs 32 times with 257 payloads and changing data. The first three
modes communicate between distinct producer/consumer workgroups, check all eight
payload words after observing publication, and use bounded polling so that the
test assumes no cross-workgroup scheduling guarantee. All 8,224 messages per
mode/device were observed in this run. The release-sequence test lets the last
arriving RMW inspect every workgroup's payload without spinning. The source and
executed SPIR-V are retained; `native-results.txt` hashes every relevant source.

[Production integration](runtime-integration.txt) additionally constructs a real
Mantle `VkContext` on RADV, with validation and synchronization validation, and
creates a Device-scope shader module/pipeline on that context with no validation
messages. This verifies native feature enablement, not the whole legacy runtime
or its resource registry/lifecycle. The existing GLFW dependency needed a working
X display for this separate check. The standalone context was retired afterward.

## Reproduction and controls

Run through Julia in the workspace environment:

```julia
include("dev/Lava/test/runtests.jl")
include("dev/Mantle/experiments/atomic-ordering-2026-10-21/native.jl")
AtomicVisibility.run()
```

The native fixture reuses the previous fixed-pool/submission fixture via
`include_string`, enabling `shaderInt8` in addition for emitted Bool temporaries.
It does not load old Mantle or adapt its array/graph implementation.

`baseline-target.jl` preserves the original source, whose SHA256 matches the
historical `atomic-demotion-result.txt`. Replay that actual demotion with:

```julia
withenv("MANTLE_ATOMIC_BASELINE" =>
    abspath("dev/Mantle/experiments/atomic-ordering-2026-10-21/baseline-target.jl")) do
    include("dev/Mantle/experiments/review-2026-10-20/probe-atomic-demotion.jl")
end
```

`byte-offset-before.txt` records both minimized inference failures before the
repair. `fixture-first-run.txt` records the initial test-fixture errors (missing
shaderInt8 and a zero-size Val mistakenly serialized as an argument).
`byte-offset-first-run.txt` records the actual compiler failure after fixing
those fixture bugs. Neither is evidence of a hardware limitation.
`minmax-test-policy-mismatch.txt` retains an additional diagnostic whose host
Julia NaN expectations were wrong for this project: the existing
`/sim/Programmieren/VideoEdit/experiments/portable-tests/ROOTFIX.md` explicitly
rules minNum/maxNum on all backends. The implementation keeps that policy and
the native Float32 operations; no numerical-policy change was made.
`native-host-visible.txt` retains the earlier successful host-visible run; the
accepted final device-local run is `native-results.txt`.

## Compatibility and limits

Existing default Atomix RMW now fails compilation because it requests sequential
consistency. Callers must choose an explicit supported ordering **after reviewing
the algorithm**; no automatic or blanket consumer rewrite is justified. No
sequential-consistency emulation, fallback or API weakening was introduced.

The compiler tests cover the emitted supported operations, but the native tests
qualify Int32 same-Device-scope visibility on the four local Vulkan devices.
They do not qualify every type, arbitrary combinations of scopes, other OS/API
backends, or performance. The queried optional availability/visibility-chain
feature is false on the two RADV devices and true on NVIDIA/llvmpipe; that fact
must not be treated as evidence that arbitrary mixed scopes preserve LLVM's
contract. Complete target eligibility remains F42, including enabling and checking
`VK_EXT_shader_atomic_float2` before creating pipelines for native Float32
min/max. Their emitted code is tested here; native min/max is not part of the
four-device visibility qualification. No replacement-runtime test
or universal hardware-support claim follows from these results.
