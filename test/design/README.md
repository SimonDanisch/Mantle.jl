# Host design checks

From a Julia REPL with Test available:

```julia
include("test/design/runtests.jl")
```

No GPU or Mantle import is needed. faultdevice.jl declares a minimal native ABI
in a private module, then includes **the actual test/suite/faultdevice.jl**.
Missing forwarding, lost keyword arguments, separate channel state, an unfired
scheduled error or a builder callback bypass fails the checks. Runtime ownership
here tests the wrapper contract, not a nonexistent Mantle Runtime implementation.

execution-model.jl enumerates bounded interleavings. Correct protocols must
satisfy the visibility/lifetime oracle, and deliberately broken reader-update,
readback-capacity and render-only-retirement protocols must have counterexamples.
The pageable-copy model additionally checks source-version consistency and
queued-store preservation, with torn-copy/lost-store negative controls.
It does not model GPU cache visibility, compiler access inference or OS drivers.

suite-helpers.jl extracts the actual shared capability/fixture helpers into a
private fake device module and checks overrides, canonical queue identity,
CPU-only defaults and explicitly verified driver pairs.

Review validation on Julia 1.12.7 (2026-10-20): 78 wrapper, 34 model and 19 shared-helper assertions pass (131 total).
All 42 suite areas macroexpand/lower, containing 585 unique cases; 52
suite/design/doc Julia files parse. Reproduce the latter with
experiments/review-2026-10-20/validate-spec.jl. git diff --check passes. None of these
checks executes the replacement GPU API, which has not been implemented.

A separately warmed, typed measurement of 200 unscheduled due calls allocated
zero bytes. The fault schedule is copy-on-write on injection; interception does
not copy the schedule vector on each call. This isolates instrumentation overhead
and establishes no native submission/consumer performance result.
