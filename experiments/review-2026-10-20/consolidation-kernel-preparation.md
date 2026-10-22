# Shared eager/declared kernel preparation — 2026-10-21

Compared with `e0b442b`. Array-library eager KI launches now call the existing
core `kikernel` used by declared dispatch preparation. Compiler-backend selection,
converted argument types and callable conversion have one implementation. The
support predicate checks the converted callable and returns false for a backend
without `argconvert`, retaining the named unsupported-kernel error.

This fixes a protocol inconsistency: eager launch already converted the callable
before asking KI to compile it, while declared preparation passed the host
callable directly. A valid backend whose conversion changes the callable type
could therefore work eagerly but be rejected by the declared helper. The host
contract fixture uses distinct reported/compiler backend types and a converted
callable/argument representation, checks computed values, and covers explicit
and default workgroups.

KA array launches now use the existing core `kernelfor` constructor too, retaining
`invokelatest` for generated kernels and resolving the ArrayLaunch group sentinel
once. Three identical declared-KA capture checks in Vulkan, Metal and ROCm move
into one core overload of that constructor. Their policy and diagnostic remain
unchanged; backend-specific compilation, argument packing and emission stay with
their existing callers.

## Qualification

- 13 host assertions in `test_kernel_preparation.jl`, included in the ordinary
  suite: converted ABI and compiler-backend selection, results through eager and
  deferred invocation, named refusal, CPU KA default/explicit groups and launch
  order. This qualifies the preparation helper, not arbitrary captured-callable
  support throughout every native graph emitter.
- 998 native assertions across AMD RX 7900 XTX and NVIDIA RTX 4000 Ada, with
  Vulkan validation and synchronization validation: `test_kernelinterface.jl`,
  `test_declared_kernel.jl`, `test_gemv.jl`, `test_fft.jl` (including real FFT and
  STFT), `test_run_allocates_nothing.jl` and per-device validation checks.
  Numerical results and zero-allocation recorded runs pass; no validation
  messages. Existing timing output is not a before/after speedup measurement.
- All 113 source/extension files parse; 1,995 chunks extracted.

Total: 1,011 passing targeted assertions. Native Metal and ROCm execution is not
qualified by these Vulkan runs. The shared conversion/constructor contracts have
host coverage; the moved native capture-policy bodies remain textually identical.

As a negative control, the same host fixture was run against the untouched
`2b09378` checkout/environment used for the publication baseline. Its three KA
constructor checks passed, but the converted-callable support assertion failed
and declared preparation then threw the unsupported-kernel error. The final
implementation passes all thirteen checks.

Physical source is 113 files / 55,051 lines, two fewer than this batch's baseline.
No new selection cache, operation representation or emitter is introduced.

The complete one-dispatch/Builder replacement remains open. `ArrayLaunch` still
chooses algorithms once; Vulkan's existing preparation/launch plans still share
lower-level work; Metal's `Immediate` still handles pending front-of-run work.
Those callers must migrate with the pending-change and resource-lifetime owners
before deleting the old execution layers. This batch removes repeated preparation
and policy, not those remaining ownership domains.

## Context ABI follow-up — 2026-10-22

Compared with `69f2ff4`, core's ordinary KA signature converts the iteration
context through the existing argtype hook. Metal's duplicate setup is deleted;
its mtlconvert expression now comes from that hook. Vulkan retains its IterPlan
override. A host fixture with a different device-context type fails against the
old unconverted signature and passes for default/explicit/2-D workgroups.

284 host/Vulkan access and preparation assertions pass, with empty validation
and synchronization drains. ROCm's 77 backend assertions plus 25 host preparation
checks pass using the pinned SDK source in test/rocm/README.md. Native Metal
execution remains unqualified; source preserves its converted signature.

Metal KA comments now state current access, library eligibility and submission
contracts. Prose edits preserve definitions. All 115 source files parse; source
falls 50,570 → 50,338 physical and 22,824 → 22,816 implementation lines (eight
implementation lines removed, no prose reclassification).
