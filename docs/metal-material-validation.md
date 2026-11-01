# Unified materials on Metal

Validated on 2026-11-01, Apple M5 (Mac17,3), 32 GiB RAM, Julia 1.12.7.
The isolated acceptance environment uses GPUCompiler 2.16; the normal Mac
project uses GPUCompiler 2.17. The normal project also passes the complete
2,308-assertion material suite, a canonical Glass render against the typed
differential reference and all 18 loop-policy checks.

## Correctness and reuse

- 28,377 model, class, composition and canonical-material assertions across
  73 configurations pass.
- The complete six-file GPU/reference suite passes 2,308 assertions, including
  all 321 Cycles reference assertions, white-furnace tests, hardware/software
  image comparisons, native compositions and subsurface selection.
- RayMakie passes 93 material/geometry update assertions and 24 image/text/frame
  update assertions. Nested Glass fields update the canonical Principled
  material in place, retaining its slot and render-object handle.
- Forty native pipeline/cache assertions pass. Diffuse, Sheen and Glossy use
  the same pipeline object. Principled, its Glass preset, Add and Mix use another
  shared pipeline object. Five warm renders per graph incur zero compiler misses.
  Different input programs can still compile their own values kernel.

This establishes within-session native pipeline reuse. It does not establish
persistent cache reuse for Metal objects containing symbolic Julia relocations;
that remains a separate item in the compilation reuse plan.

## Native compilation failure and fix

The Thin Wall subsurface class, `SurfaceProgramNodes{16,true,0x44fe,false}`,
initially crashed Apple's compiler service. Its entry had 5,309 blocks and
86,404 LLVM instructions; the crash trace reached `llvm::report_bad_alloc_error`
inside native machine-function compilation.

A compile-only reproducer keeps executable LLVM instructions unchanged and
adds `llvm.loop.unroll.disable` to uncontrolled natural loops. Marking all 158
backedges compiles in 5.302 seconds. An AIR roundtrip without these hints and a
variant marking only the six largest loops each reach their 90-second deadline.
Those are bounded observations, not measurements of their eventual compile time.

Metal now applies that metadata policy after AIR lowering to functions with at
least 4,096 blocks. Small functions are untouched; explicit source unroll hints
and unrelated metadata survive. The material storage and evaluator are unchanged.
Eighteen tests cover hook dispatch, unchanged instructions, valid LLVM,
idempotence and metadata preservation. The formerly failing render and the full
reference suite pass with existing tolerances.

## Runtime remains open

Paired wall-clock measurements use the shared-surface differential benchmark:
512 by 512 pixels, eight samples, software intersections, five warmups, 51 pairs,
alternating order and medians. Both implementations render pixel-identical images.
The typed implementation exists only as the benchmark reference.

| Material | Typed reference | Shared graph | Overhead |
| --- | ---: | ---: | ---: |
| Principled, coat/metal/sheen | 39.180 ms | 57.816 ms | +47.57% |
| Diffuse | 25.840 ms | 31.330 ms | +21.25% |
| Sheen | 26.035 ms | 30.318 ms | +16.45% |
| Glossy | 28.572 ms | 33.067 ms | +15.73% |
| Glass preset | 52.325 ms | 86.950 ms | +66.17% |
| Metallic | 29.729 ms | 36.851 ms | +23.96% |

These M5 results do not satisfy the accepted approximately four-percent target.
Vulkan measurements cannot substitute for this measurement.
Repeating Principled and Glass in the normal project with GPUCompiler 2.17 gives
38.819 to 55.893 ms (+43.98%) and 51.764 to 83.252 ms (+60.83%), respectively.
The pinned compiler version alone therefore does not account for the regression.

Reducing the shading workgroup cap from 64 to 32 retains image equality but
leaves approximately 44% Principled and 62% Glass overhead in 31 paired trials.
Increasing that cap alone cannot exceed the caller's requested group size.
No workgroup policy change was adopted. Metal's integrated profiler reports
host waits but misses the main passes replayed through indirect command buffers,
so its tiny reported GPU total cannot explain the frame cost. A register-spill
or local-storage explanation has not been demonstrated.

A further isolated experiment replaces dynamic tuple indexing in mixture,
sampling and weight summation with constant field accesses selected by `ifelse`.
It preserves pixel equality but increases the ratio to the typed reference to
2.230 for Principled and 1.990 for Glass (51 pairs). The original methods are
restored in a `finally` block. This rules out that particular scalar-selection
replacement as an improvement; it does not by itself identify spills or native
register pressure as the cause.

The final AIR for the shared eight-slot Principled class has 4,287 blocks and
63,842 instructions, so the loop-retention policy also applies to that class.
An isolated threshold-5,000 variant leaves this shader unmarked while retaining
the fix for the 5,309-block subsurface shader. Both render correctly, including
the formerly failing Thin Wall cases. With no other validation running, 51
paired measurements still give 38.964 to 57.949 ms (+48.72%) for Principled and
52.389 to 86.793 ms (+65.67%) for Glass. Removing the loop policy from this class
therefore does not remove the observed regression. The production threshold
remains 4,096; the experimental source was restored and its session cleared.

## Mac working-tree reconciliation

Original edits are preserved in per-repository `recovery/mac-metal-20261101`
commits and `mac-local-edits-integrated-20261101` stashes. Current implementations
supersede the older tracked Mantle and Makie edits; historical design notes live
under `docs/archive/mac-20261101`. Validated merges are installed in the main
Mac checkouts, not only in the isolated environment. The normal project's
manifest was resolved against those sources.

The additional cooperative tensor test depends on `MtlCooperativeTensor`,
`Int2b` and `coordinate`, absent from both the current branch and upstream Metal.
It reports undefined imports on the M5 and later aborts in LLVM's inliner.
The incomplete prototype is preserved under Metal's `docs/archive/mac-20261101`
and in the original recovery commit; it is not an active, passing regression test.
The separate GPUCompiler Vulkan prototype remains unchanged and is not used by
this Metal validation.
