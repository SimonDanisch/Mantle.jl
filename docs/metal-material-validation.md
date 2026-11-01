# Unified materials on Metal

Validated on 2026-11-01, Apple M5 (Mac17,3), 32 GiB RAM, Julia 1.12.7.
The isolated acceptance environment uses GPUCompiler 2.16; the normal Mac
project uses GPUCompiler 2.17. Both pass the complete 2,308-assertion material
suite and 31 compiler checks with Metal `d9969275` and Hikari `499e350e`.
The latter commit adds diagnostic benchmarks; it does not change rendering.

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
least 4,096 blocks or 32,768 instructions, before the AIR LLVM-version downgrade.
Small functions are untouched; explicit source unroll hints and unrelated
metadata survive. The material storage and evaluator are unchanged. Twenty-six
tests cover hook dispatch, both size thresholds and their boundaries, unchanged
instructions, valid LLVM, idempotence and metadata preservation.

The instruction threshold addresses a second, denser shader with 3,922 blocks.
Its 49,766 instructions expand to 67,879 during AIR downgrade. A threshold of
65,536 at the earlier pass therefore missed it. The unmarked shader reaches a
90-second compile-only deadline; adding loop metadata without changing executable
instructions compiles in 3.714 seconds. With the corrected threshold, both full
material suites pass, including the previously failing furnace cases. Large
artifacts were preflighted through an independent compile-only Metal executable
with a 90-second alarm before allowing renderer compilation. That deadline
limits the client, not the separate Apple compiler service's memory usage.

Float16/Float32 min/max now remain LLVM minnum/maxnum intrinsics until final
lowering. This preserves Metal's existing NaN behavior while allowing LLVM to
prove bounds such as `max(x, 0)`: a noinline clamped-square-root helper remains
separate rather than carrying a domain-error exit into its caller. Five focused
checks cover optimized LLVM, final AIR and GPU output, and 21 existing numeric
edge-case checks also pass. This does not remove all exceptional exits from
Hikari's weighted evaluation; other square-root and bounds paths still cause
late inlining. It is not a complete solution for composable GPU functions.

## Runtime remains open

Paired wall-clock measurements use the shared-surface differential benchmark:
512 by 512 pixels, eight samples, software intersections, five warmups, 51 pairs,
alternating order and medians. Both implementations render pixel-identical images.
The typed implementation exists only as the benchmark reference. These are the
final measurements with the fixes above in the normal GPUCompiler 2.17 project;
all six images have maximum component error zero.

| Material | Typed reference | Shared graph | Overhead |
| --- | ---: | ---: | ---: |
| Principled, coat/metal/sheen | 38.983 ms | 58.535 ms | +50.15% |
| Diffuse | 25.798 ms | 30.249 ms | +17.25% |
| Sheen | 25.992 ms | 30.092 ms | +15.77% |
| Glossy | 28.481 ms | 32.283 ms | +13.35% |
| Glass preset | 52.208 ms | 88.651 ms | +69.80% |
| Metallic | 29.564 ms | 36.024 ms | +21.85% |

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
retains the 4,096-block threshold plus the dense-function instruction fallback
described above; the experimental source was restored.

## Reproducible performance reductions

Hikari's `benchmarks/shared_surface/metal_stages.jl` and `metal_prebuilt.jl`
run standalone evaluation/sampling kernels, compare their output and collect
actual GPU timestamps through `Metal.@profile`. They are explicit diagnostics,
not additional renderer paths. The stage-zero output observes only some setup
fields, so dead-code elimination can omit other setup work. Independently
compiled stage timings cannot be subtracted to obtain a full-render breakdown.

The prebuilt experiment removes GPU material setup and RGB lookup. At 65,536
lanes and 31 alternating pairs, shared/typed wall-time ratios are 2.189 for
Principled evaluation and 3.008 for Glass evaluation; adding sampling gives
1.109 and 1.275 respectively. GPU timestamps also show a residual cost. Setup
alone therefore cannot account for the regression. The packaged diagnostics
pass all 20 output comparisons; their later validation ran alongside package
precompilation and is not an idle-device performance measurement.

Additional experiments retain image equality but provide no demonstrated gain:

- Compiling the native pipeline with `maxthreads=64`, independently of launch
  size, leaves approximately +48% Principled and +65% Glass overhead. Actual
  recorded pipelines otherwise have no explicit native thread maximum; setting
  the launch cap alone is a different experiment.
- Requesting a noinline weighted evaluator does not retain that boundary in AIR:
  exceptional exits cause GPUCompiler to inline it again. Its unchanged runtime
  does not measure the benefit of a surviving function boundary.
- Propagating inbounds through weighted evaluation leaves +48.03% Principled
  and +68.49% Glass overhead. Deferring padding initialization gives +50.68%
  and +77.93%; reserving fixed logical slots while skipping inactive setup gives
  +49.34% and +71.72%. All use 51 alternating full-frame pairs. These layouts
  have not been adopted.
- Passing a runtime-uniform model tag separately in a one-lobe standalone kernel
  gives 4–7% higher wall times for Diffuse/Sheen/Glossy. Those measurements have
  no GPU timestamps, so they do not establish the isolated device-time effect.

Temporary method overrides are restored in `finally` blocks. These results
exclude specific proposed changes; they do not establish a register-spill,
aliasing or local-memory explanation. The remaining task is to find a measured
device-code improvement while retaining the one production interpreter and
shared storage, not to enable the typed differential reference in production.

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
