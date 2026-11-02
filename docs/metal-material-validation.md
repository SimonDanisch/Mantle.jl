# Unified materials on Metal

Validated on 2026-11-01, Apple M5 (Mac17,3), 32 GiB RAM, Julia 1.12.7.
The initial isolated acceptance environment uses GPUCompiler 2.16; the normal Mac
project uses GPUCompiler 2.17. Both passed the complete 2,308-assertion material
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

This establishes within-session native pipeline reuse. A later cache fix,
Metal `49ae96bf`, removes references reachable only from dead helpers before
deciding whether the object can be cached. Ten checks cover byte-identical AIR
and continued exclusion of live references. The real Principled fixture passes
an additional two-session test on GPUCompiler 2.16: 89 hits, zero misses and an
identical image, with a 19.56-second cached render call (r670/r671). Fresh
`Pkg.test` children with current Hikari `2774e44`, the published Metal source
and explicit `--check-bounds=auto` subsequently give 85 hits, zero misses and
the same image; the cached render call takes 17.67 seconds (r710/r711). The
fixtures differ in tessellation and path depth, so their times are not an A/B
startup improvement. This does
not skip host compilation or establish persistence for live references. That
broader case remains a research probe in the compilation reuse plan.

The current-source six-file GPU repeat passes all 857 assertions, including all
17 Cycles reference batches (321 assertions), shading, native compositions,
runtime storage, material-type changes and white-furnace checks. This is r711
with Hikari `2774e44`, Metal `49ae96bf`, GPUCompiler 2.16 and explicit normal
`--check-bounds=auto`. Its separate seven-assertion cache probe also passes.
The separate GPUCompiler 2.17 environment repeats the same six files with
857/857 passing assertions and also passes all ten cache regressions (r718,
explicit normal `--check-bounds=auto`). Both runs use fresh `Pkg.test` children
and the published source, without session-only renderer overrides.

## Forced bounds-check compilation

A fresh `Pkg.test` adds `--check-bounds=yes` by default. Earlier direct
acceptance runs used Julia's normal `auto` setting. This difference exposes
a separate native compile problem for the first software-traced Diffuse shader,
`SurfaceProgramNodes{1,true,14,false}`. The forced-check AIR entry has 2,278
blocks and 27,864 instructions (644,848 bytes); the auto entry has 1,294 blocks
and 22,426 instructions (470,880 bytes). In the standalone Objective-C
`Metal/research/native_pipeline_compile.m` probe, auto compiles in 0.964 seconds,
while forced checking does not finish within 45 seconds. The initial test was
stopped after 9m41s while waiting for Apple's compiler; this is not a passing
default-configuration Metal test run.

The cache-cleanup fix is excluded as the trigger: fresh captures with it enabled
and disabled have byte-identical AIR and metallib. Keeping 48 natural loop
latches, limiting native workgroups to 64, removing 3,618 alias annotations,
stripping debug information, and compiling without indirect-command support
each still exceed the 45-second limit. The strict input contains no undef or
poison values. These bounded controls did not resolve the problem.

The follow-up identifies 476 identical BoundsError-reporting tails in the strict
shader, compared with three in auto mode. Sharing 475 of those complete blocks
reduces native compilation to 1.321 seconds. The production matcher also shares
other identical exception tails: 509 candidates become three blocks, leaving
1,772 entry blocks, and native compilation takes 1.315 seconds (r742). All
1,171 conditional branches remain. The matcher compares the state extraction,
called functions and calling conventions, atomic arguments and exit target;
it rejects escaping values, successor PHIs, non-void calls, address-taken blocks,
operand bundles, call-site attributes and metadata other than source locations.
It does not remove checks or change the existing loop and alias policies.

Once compilation succeeds, forced checking exposes a second issue. A tagged
error-store diagnostic identifies `gc_pool_alloc` for `surface_setup_slots`'s
local MVector, rather than an invalid material index (r771). Passing the mutable
array to the noinline bounds-error wrapper keeps it alive as a device heap
allocation. Inlining that small wrapper removes the escape; the reporting helper
it calls remains out of line. The exact failing 16-sample, three-bounce Diffuse
fixture then renders successfully (r778).

With both fixes, the six current material/reference files pass **857/857** with
forced checks and GPUCompiler 2.17 (r792), including all 17 Cycles batches. The
normal-auto repeat also passes 857/857 (r762). The focused tests retain real
out-of-bounds failures at debug levels 0–2, verify a local eight-record MVector
with runtime indices, and check device recovery after a caught failure. A
full-render debug-level-2 diagnostic before the wrapper fix was stopped in
Apple's compiler; fast compilation with complete device stacks is not established.

The final focused compiler tests pass 91/91 with both GPUCompiler 2.16 and 2.17
(r794/r795). They include 62 structural checks, 25 real bounds-error/recovery
checks and four checks for the escaping-record reproducer. GPUCompiler 2.16 also
passes all 455 shader-material assertions with forced bounds checks (r794);
2.17 repeats all ten persistent-cache regressions (r795). Renderer acceptance
uses Hikari `2774e44`; these compiler fixes do not change its material API.

The published compiler `2153de32` and Hikari `7fc8bfd` also pass a new two-child
cache test on GPUCompiler 2.16 (r810/r812). Population records 41 hits/44 misses;
the second process records **85 hits/zero misses**, an identical image SHA-256
(`b53913b476f6a7a90ff3fc9984a9f295d8fc60a99a211e3a25afb682af8263eb`), and an
18.55-second cached render call. This includes host compilation and does not
establish fast startup or persistence for live Julia references. The newly
included ConstTexture initialization commit also passes 121 description/constant
texture checks on Metal and 52 constant-texture checks on RTX (r807).

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

## Hardware ray tracing follow-up

A separate M5 measurement on 2026-11-01 enables `hw_accel=true` in the
same full-frame differential benchmark: 512² pixels, eight samples, five warmups
and 51 alternating pairs. The fixture's accelerator is checked to be an
`HWTLAS` (Metal's `BatchedTLAS` with `MetalBLAS`). All 36 renderer assertions
pass and all six image comparisons have maximum component error zero.
The measured Hikari checkout is `8a5cb6b1000e2c6267e7860a05eca17cc2ea250e`;
this measurement is recorded as run `r619`.

| Material | Typed reference | Shared graph | Overhead |
| --- | ---: | ---: | ---: |
| Principled, coat/metal/sheen | 37.708 ms | 56.447 ms | +49.69% |
| Glass preset, roughness 0.2 | 48.293 ms | 82.253 ms | +70.32% |
| Diffuse | 24.146 ms | 28.035 ms | +16.11% |
| Sheen | 24.152 ms | 27.776 ms | +15.00% |
| Glossy | 26.919 ms | 29.854 ms | +10.90% |
| Metallic | 28.207 ms | 33.483 ms | +18.70% |

Hardware intersections therefore do not remove the shared-material runtime
regression. The earlier small Diffuse/Sheen/Glossy prototype's hardware results
cannot establish performance parity for the current wider production storage.
These timings establish the cost in this fixture, not its native compiler cause.

A hardware-RT control repeats Principled and Glass at the normal shader debug
level 1 and at level 0, using separate compiler configurations and the same
51-pair protocol. Principled changes from 37.690/56.678 ms typed/shared (+50.38%)
to 37.517/51.954 ms (+38.48%). Glass changes from 48.463/81.935 ms (+69.07%) to
46.546/77.222 ms (+65.90%). All 24 image checks pass with zero component error.
Shader diagnostics thus contribute to this configuration's cost but do not
explain most of the regression. The diagnostic configurations are restored;
production retains its normal exception reporting.

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
  initially gives 4–7% higher wall times for Diffuse/Sheen/Glossy. A follow-up with
  30 GPU timestamps per implementation/model confirms device-time increases of
  11.11%, 20.20% and 14.00%, respectively. Wall time alone had understated this
  result, including a Glossy repeat with a small apparent wall-time improvement.
- Transposing the same stored fields into a structure of arrays changes device
  time by only 0.4–1.8% in the prebuilt tests. An explicit local mutable array
  improves evaluation alone by 1.5–4.8%, but evaluation plus sampling becomes
  5.1–8.5% slower. Both preserve output; neither explains or removes the large
  rendering overhead.

Temporary method overrides are restored in `finally` blocks. These results
exclude specific proposed changes; they do not establish a register-spill,
aliasing or local-memory explanation. The remaining task is to find a measured
device-code improvement while retaining the one production interpreter and
shared storage, not to enable the typed differential reference in production.

Hikari's additional `metal_single_lobe.jl` diagnostic decodes a single active
simple term once before evaluation and sampling. It passes eight output/reuse
checks; Sheen and Glossy incur zero compiler misses after Diffuse. An isolated
full-render experiment reduces Diffuse/Sheen/Glossy overhead to 6.04%, 5.75% and
4.78%, with zero image error (36 assertions). This is a partial result, not a
production change or validation of a larger-graph implementation.

Two attempts to generalize the fast case expose a Diffuse image failure. Both
GPUCompiler versions reproduce it; the ordinary shared representation remains
correct. Complete BSDF-operation tests over 65,536 lanes, including tilted
normals, pass eight checks. Full-render captures also preserve weights,
coefficients, wavelengths, normals and outgoing directions, yet the direct
lighting's evaluated BSDF differs. Replaying 6,138 captured input sets on the
CPU matches the typed GPU evaluation within 2.98e-8; the experimental shader
differs by up to 35.82 in that spectral component. Shader dispatch and generated
code are still being isolated. No specific compiler pass, spill mechanism or
native driver defect has been established, and these failing variants are not
integrated. Forty pipeline/warm-cache checks still pass in the operation-level
experiment; reuse alone does not establish rendering correctness.

GPU replay also reproduces the checked spectral component bit-for-bit for
typed, counted and single-record tuple carriers (six checks). An AIR roundtrip
control and retaining every uncontrolled loop both leave the full-render error
unchanged at 0.34359404. Capturing the evaluated BSDF immediately before and
after shadow traversal in the same shader shows no change in its first three
components. These reductions do not support blaming the loop-retention policy
or a value overwritten across that traversal. The failure still requires the
larger shading context; no experimental carrier change is enabled in production.

A further native compiler control removes alias assumptions from the failing
single-record experiment before Apple's compilation. Removing 4,278 scoped-alias
and TBAA annotations leaves the image error exactly 0.34359404. A second control
also removes 172 parameter/call `noalias`, `nocapture`, `readonly` and `writeonly`
attributes, with the same error and identical changed-component counts. The
166- and 167-assertion diagnostic runs validate the IR transformations, not the
incorrect image; a separate six-assertion render confirms restoration of the
correct production implementation. These controls do not support blaming those
alias annotations for the experimental failure. No alias policy is changed.

A fresh normal-auto repeat with the checked-storage compiler fix and
GPUCompiler 2.17 retains exact images in the 512²/eight-sample hardware fixture
(12 assertions, r799). Principled measures 37.533 ms typed versus 54.215 ms
shared (+44.45%); Glass measures 48.974 versus 78.982 ms (+61.28%). This confirms
that the compile/escape fixes do not remove the runtime regression. Different
compiler versions and sessions must not be used to attribute the change from
the earlier +49.69%/+70.32% result to this fix alone.

## Capacity and Metallic follow-up

The new `Hikari/benchmarks/shared_surface/metal_active_capacity.jl` diagnostic
compares the typed BSDF, the production eight-slot span and the same CPU-prepared
span trimmed to its active count. Setup is absent from the measured kernel.
All 16 output checks pass over 65,536 lanes. Each result uses 51 alternating
wall-time triples and 30 device timestamps per implementation (r774):

| Material / operation | Eight slots | Active slots | Active / eight |
| --- | ---: | ---: | ---: |
| Principled evaluation, five active | 670.6 µs | 325.5 µs | 0.485 |
| Principled evaluation + sampling | 663.8 µs | 346.3 µs | 0.522 |
| Glass evaluation, one active | 655.2 µs | 132.9 µs | 0.203 |
| Glass evaluation + sampling | 654.3 µs | 198.7 µs | 0.304 |

This establishes a substantial capacity-dependent device cost. Trimming changes
the argument type and generated code as well as the byte count (868 to 544 or
112 logical bytes); it does not establish spilling or memory bandwidth as the
native mechanism, nor does it measure a full-frame speedup. Value-dependent
capacity is not integrated: changing a numeric weight must retain the warm
kernel when another term becomes active.

A same-input control keeps the eight-slot argument unchanged and creates only
a shorter local span inside the kernel. All 24 output checks pass (r808).
Principled device times change from 671.2 to 269.7 µs for evaluation and from
656.6 to 312.2 µs with sampling; Glass changes from 650.2 to 132.0 µs and from
645.1 to 198.2 µs. This rules out a smaller incoming argument block as the sole
explanation. The generated local representation/traversal remains the measured
variable; register spilling is still unproven.

A separate access-pattern control replaces each dynamic whole-tuple lookup
in evaluation/sampling with a balanced choice between two-record subtuples.
The data, accumulation order, capacity and material signatures remain unchanged.
Apple rejects the resulting full Glass shader before dispatch (r837). The saved
914,176-byte AIR passes LLVM verification (r842); its 914,456-byte library
reproduces `AGXMetalG17G Code=3:
Internal compiler error` in the standalone Objective-C compiler probe, without
Julia, in 3.54 seconds (r840). Disabling indirect-command support or limiting
the pipeline to 64 threads also fails in about 3.1 seconds (r841). Restoring the
ordinary traversal passes all six full-image assertions (r838). This identifies
a native compilation limitation of this additional experiment; it does not
explain the production runtime cost or establish the cause of the different
count-one image failure below. The access-pattern change is not integrated.

A full-render fast branch for one active term still fails the Diffuse image
even when the BSDF carrier type is unchanged (r738; reproduced on the published
source in r809). The same change passes the full Diffuse/Glass images on RTX
(r811); no Metal backend or native-driver cause has been established. Renaming the original
traversal into an identical helper alone preserves the image exactly (r779).
These controls further localize the failing experimental change; they do not
establish a native compiler defect or justify adopting the branch.

Restricting that branch to capacities greater than one avoids changing the
already one-slot Diffuse shader and passes both full images. Glass changes only
from 79.513 to 77.970 ms shared/control, against 48.864 ms typed (r814, 51
alternating triples). This small full-render gain does not recover the much
larger standalone improvement. These controls do not establish local indexing
as the sole source of the full-render regression; no traversal change is
integrated.

On RTX 4000 Ada, removing the thin-film evaluation branch only in a no-film
Metallic diagnostic leaves 25.032 versus 25.062 ms shared/control, compared with
23.249 ms typed (r752). A narrower one-slot capability mask gives 24.494 ms
versus 24.779 ms shared and 22.437 ms typed (r783, 51 alternating triples,
maximum image error 7.83e-8). All three pipelines still use 255 registers.
Neither control removes the Metallic overhead, and neither is integrated.
A singleton-decoder control produces distinct code (835,880 bytes
versus 843,392 shared) but measures 24.380 versus 24.392 ms shared, against
22.626 ms typed (r815). All still use 255 registers. A separate generated-callee edit MWE invalidates correctly (r793).
A closed two-case Metallic/Film setup diagnostic also retains the images but
reduces SPIR-V from 843,392 to 641,640 bytes without reducing frame time:
24.443 ms shared versus 24.469 ms control, against 22.418 ms typed (r816).
Removing that cold setup code does not by itself remove the regression.

The driver inventory distinguishes SPIR-V size from machine-code size (r817).
Typed/shared native binaries are 201,344/487,424 bytes. Both report 255 registers,
16 bytes local memory and zero stack, while shared memory is 22,784/25,344 bytes.
The shared module's extra function is the noinline thin-film reflectance helper;
the driver exposes no textual internal representation. These statistics do not
establish a spill or occupancy cause.

## Thin-film function boundary

Removing F82 thin-film code from both setup and evaluation in a no-film-only
RTX control reduces the shared overhead from 7.99% to 3.00% (r818). Removing
only the spectral evaluation branch had not helped. This control disables a
feature and is not a production change.

Keeping all film formulas but inlining the helper in Julia reduces a repeat
from 24.313 to 23.681 ms against 22.413 ms typed: +8.48% becomes +5.66% (r820).
SPIR-V grows from 843,820 to 950,764 bytes. The corresponding M5 hardware control
retains exact images but gives no gain: Principled 53.897/54.123 ms and Glass
78.577/78.662 ms shared/inline (r823). A global source-inline change is therefore
not integrated.

A later RTX control changes only the thin-film function's SPIR-V function-control
word, after LLVM and SPIR-V optimization. The before/after executable instructions,
arguments and CFG are otherwise unchanged; SPIR-V validation and full images pass.
This preserves the separately emitted helper and its compilation-cache boundary.
Typed/shared/native-None/native-Inline frame medians are 22.467/24.446/23.576/
23.494 ms (r825, 51 alternating quadruples). Allowing native inlining reduces the
shared overhead from 8.81% to 4.57% without removing thin-film support. Both hints
produce 617,472-byte native binaries versus 487,424 bytes shared; register,
local-memory and shared-memory counts remain unchanged. This establishes a cost
of preserving that native call boundary, not a register-count or spill cause.
A production annotation/policy separating reusable compiler objects from native
inlining is not added by these diagnostic overrides.

The simpler source change removes only the helper's forced `@noinline`, allowing
normal compiler decisions without forcing source inlining. With every film formula
retained, RTX medians are 22.503/24.532/23.553 ms typed/forced-call/ordinary-function:
the overhead falls from 9.02% to 4.66% (r827, 51 alternating triples). This emits
951,400 SPIR-V bytes and still uses 255 registers. The M5 counterpart retains
exact images and measures 54.102/53.847 ms for Principled and 78.906/78.380 ms
for Glass, forced-call/ordinary-function (r829). It does not resolve the large
Metal regression. Native-hint controls also retain active Metallic and Glass
films, with all 16 assertions passing (r830).

Hikari `230eba1` now leaves `_thin_film_reflectance` unannotated. This changes neither
the formulas nor the material classes and adds no renderer mode. Unlike the
native-only hint experiment, it can change the emitted helper/object boundary;
the larger SPIR-V is an explicit tradeoff. A fresh M5 test child passes all
857 assertions with `--check-bounds=yes` (r832). The same six files pass
857/857 on RTX with forced checking (r831, 30m23.5s test body); the M5 test body
takes 10m20.6s. These are complete test durations, not shader compile timings.
A separate two-session
GPUCompiler 2.16 check retains the exact image hash and gets 85 hits with zero
misses after restarting Julia (r834/r835). Its 20.67-second cached render still
includes host compilation and was run alongside validation; it is not a startup
speedup measurement. The full Lavapipe workflow at the
preceding Hikari `7fc8bfd` head passes all jobs, including documentation and
every pbrt partition ([run 41318194128](https://github.com/JuliaGraphics/Hikari.jl/actions/runs/41318194128)).

Fresh normal-auto source measurements on M5 hardware ray tracing still show
37.498/54.146 ms typed/shared for layered Principled (+44.40%) and
48.634/78.765 ms for Glass (+61.95%), with exact images and 12 passing assertions
(r836, 512², eight samples, five warmups, 51 alternating pairs). The Glass
fixture's first draw after warming Principled's class takes 0.757 seconds;
this is within-session class reuse, not cold-start latency.

The first fresh RTX source repeat measures 22.487/24.094 ms typed/shared,
or +7.15% (r846), so the earlier +4.66% MWE result is not a reliable final
production number. A controlled repeat in that same fresh environment records
the loaded source pipeline before temporarily restoring the forced call boundary.
Over 101 alternating triples, typed/current/forced-call medians are
22.538/24.110/24.582 ms (r848). The current source improves frame time by 1.92%
against the forced-call control, reducing overhead from +9.07% to **+6.97%**.
Both images pass; current/forced-call SPIR-V is 950,972/843,820 bytes with
one/two functions, and all three report 255 registers. The remaining Metallic
regression is therefore still open. This repeat confirms a smaller benefit
than the initial AST-override MWE; it does not establish the cause of that
code-generation/performance variation.

The six-file validation above tests the published Hikari `230eba1` source
without renderer overrides. Its new full hosted Lavapipe workflow is running
([run 41327763962](https://github.com/JuliaGraphics/Hikari.jl/actions/runs/41327763962));
the green preceding-head workflow is not evidence that this newer head has
already completed hosted CI.

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

Captured-hit replay also exposed an independent Metal upload bug: a CPU
`reshape(reinterpret(...))` reached scalar iteration over a GPU destination.
Mantle now materializes host array wrappers before transfer, preserving native
Metal-array copy paths. Seventeen checks pass with both compiler versions,
covering strided views, transpose/permutation, reinterpret/reshape, empty arrays,
index views and independent device copies. This host-transfer fix does not
change BSDF storage or shader selection.

## Capacity-bound follow-up (2026-11-02)

On M5 hardware ray tracing, Julia 1.12.7 / GPUCompiler 2.17, 512² pixels,
eight samples, five warmups and 51 alternating frame triples:

| Material | Retained typed reference | Former shared dispatch | Bounded dispatches |
| --- | ---: | ---: | ---: |
| Glass, roughness 0.2 | 48.938 ms | 80.681 ms | 54.675 ms |
| Layered Principled | 37.647 ms | 55.067 ms | 45.031 ms |

Both image comparisons are bit-identical in this measurement. Frame time drops
32.2% and 18.2%; the remaining differences from the typed reference are +11.7%
and +19.6%. This does not eliminate every runtime regression. General fallback
cases also pay for rejected dispatches: partial transmission 81.766 → 85.171 ms,
thin wall 76.147 → 77.856 ms, computed transmission 75.206 → 78.302 ms.

A conservative host proof is stored in the unused upper output-width bits of
the first closure instruction. A single solid Principled node without subsurface
can use one record for full glass without overlying layers, five for opaque
layers, or its existing general capacity. The eligible class has three fixed
kernels, independent of graph arrangement or parameter values. Larger compositions
and subsurface classes retain their existing single dispatch. No per-pixel
buffer is added. In-place program copies update the proof and retain recorded
plans and buffer layouts. Glass and explicitly configured Principled produce
identical programs. The proven zero/one controls expose dead layer setup to the
compiler while using the same scattering formulas.

`Hikari/benchmarks/shared_surface/capacity_bounds.jl` records the typed reference,
former shared dispatch and production dispatches independently, then compares
images and interleaves timings. Its temporary control method is deleted in
`finally`; there is no public alternate rendering mode. Raw frame samples and
experiment records are under `out/metal_methodical_20261102` in the development
workspace. GPU register/spill explanations are not established by these timings.

Final-source checks: 455 Metal shader/material assertions; 55 new capacity,
program and update assertions on Metal and Metal with forced bounds checks.
The latest test adds three assertions comparing the first post-update frame
with a repeat; all 58 pass on Lavapipe_jll, a fresh Radeon RX 7900 XTX session,
and the canonical Mac checkout. The Radeon also passed 300
successive update frames, 30 forced re-recording transitions, and synchronization
validation with zero messages. These tests compare against the retained typed
formulas and verify that updates retain the same plans and buffer layouts.

Two first-update anomalies appeared in the heavily edited Radeon session and
could not be reproduced in the fresh/stress/re-recording/validation checks.
Their cause is not established; passing subsequent checks is not a synchronization
fix. Keep the first-frame acceptance test, rather than hiding it with warmups.

A fresh Metal process reuses all 146 kernel lookups with zero cache misses and
identical image hashes. Glass construction takes 4.903 s and its first render
34.295 s (17.583 s in Julia JIT accounting); the next Principled scene constructs
in 0.037 s and renders in 0.087 s. Cache reuse does not make first-render startup
cheap; substantial host compilation remains.

Profiling also exposed a separate diagnostic bug: unlabeled recorded Metal
submissions were absent from `Metal.@profile`. Mantle now labels them only while
profiling. The regression test fails before the fix (zero replay intervals) and
passes six checks afterwards. It measures complete recorded submissions, not
individual compute passes inside an indirect command buffer.

Repeated-frame controls are bit-identical for the retained typed reference,
production dispatches and the full-mask control. The rejected lazy primitive
lookup produces different pixels between identical renders (five frames,
256²/eight samples); this is not explained by deterministic Float32 rounding.
That experiment is not applied. Its native/compiler cause remains unestablished.

### Counted traversal and pass-grouping controls

Acceptance remains incomplete: a subsequent Radeon in-place update check caught
a repeat-frame mismatch. A capture then found differences in 19 of 36 updates,
already in spectral `pixel_L`, with all sample counters equal to eight and plans
retained. Static controls and later update controls passed after temporary
recompilation; that is not a demonstrated fix. The failing log, images and
spectral buffers are preserved in `out/metal_methodical_20261102`.

The follow-up measures the bounded implementation above, rather than the former
eight-record baseline. At 512²/eight samples, five warmups and 31 alternating
triples on M5 hardware ray tracing:

| Material | Bounded baseline | Scalar normal comparison + one-record traversal |
| --- | ---: | ---: |
| Glass | 54.640 ms | 51.153 ms |
| Layered Principled | 45.232 ms | 44.886 ms |
| Rough coloured Diffuse | 27.930 ms | 28.043 ms |

All images are bit-identical. Paired bootstrap 95% intervals for the runtime
ratio are 0.931–0.941, 0.989–0.995 and 0.999–1.015 respectively. The small Diffuse
change is inconclusive; these measurements do not establish universal parity
with the retained typed renderer. The source adds constant-index traversal
only for one-record storage. Larger storage classes keep the counted loop.
An empty span ignores even a positive backing record; a record without sample
weight contributes nothing.

The normal comparison matters for correctness: removing the one-record loop
alone changes rough Diffuse's full-render image on Metal, with maximum component
error 0.324, despite matching CPU and standalone Metal scattering kernels. It
also fails before shadow traversal and with software ray tracing. The same
renderer variants agree on the Radeon. Replacing the generic vector comparison
in `bump_shadowing` with three scalar comparisons removes the observed failure.
The exact native compiler defect is not isolated; this is not evidence that
generic vector equality is incorrect in every kernel. A full-render differential
regression now exercises Diffuse, Sheen, Glossy and film Glass, with software and
hardware scene settings, against the retained typed formulas.

Other measured controls were retained as diagnostics:

- Unrolling five records expands the Principled entry from 50,835 to 82,100 LLVM
  instructions, 3,062 to 5,534 blocks and 105 to 201 allocas. Frame time rises
  45.397 → 52.404 ms. These counts do not measure hardware registers or spills.
- Each capacity dispatch currently occupies a separate Mantle pass. Combining
  all three reduces Glass 55.267 → 53.492 ms, but increases Principled
  45.834 → 46.918 ms. Combining only the one-record and general dispatches while
  retaining the five-record pass reduces Glass 55.202 → 54.625 ms, Principled
  45.591 → 45.095 ms and partial transmission 85.806 → 84.625 ms. Images match;
  neither grouping experiment changes production pass construction.
- A 65,536-lane setup/evaluation/sampling kernel takes 0.350 ms for bounded Glass
  versus 0.333 ms typed, and 0.980 ms for bounded Principled versus 0.539 ms typed.
  Separate stage compilations permit different dead-code elimination, so their
  timings cannot be subtracted to derive a renderer breakdown.

The historical MWE measurements above used temporary methods restored in
`finally`. The corresponding source changes and tests were subsequently applied
to the canonical checkouts. Following the user's correction,
the source changes and tests are now in the existing Mac `dev/Hikari` and
`dev/Mantle` repositories. Validation runs from the project root and verifies
the actual loaded package paths. The 56 direct/count, 58 capacity/update,
51 full-render parity and six profiler assertions pass there (run r154).
The extra Desktop and Mac worktrees have been removed; their history, patches
and unique compatibility backport are preserved without additional checkouts
under `out/canonical_repository_reconciliation_20261102`.

Local Radeon and Lavapipe_jll runs each pass the 455 shader/material assertions,
the 51 composition/one-record full-render assertions and 56 direct/counted-storage
assertions. Lavapipe also passes all 58 capacity/update assertions. Radeon
capacity/update results must be read with the failure above; later passing
checks do not replace it. A CPU comparison of original and scalar normal
shadowing produces 28,090 identical results, including tilted, flipped,
signed-zero and NaN inputs.

A subsequent fresh Radeon process, loading the unchanged files rather than
temporary scattering methods, passes the 58 capacity/update assertions and
108 further captured frames with zero repeat differences. This narrows the
failure to conditions not reproduced by that fresh run; its cause remains open.

The Radeon timing control also separates the capacity optimization from the
remaining shared-storage cost (512²/eight samples, 51 alternating triples):

| Material | Typed reference | Former eight-record dispatch | Current bounded dispatches |
| --- | ---: | ---: | ---: |
| Glass | 5.645 ms | 8.936 ms | 6.535 ms |
| Layered Principled | 4.721 ms | 6.935 ms | 6.892 ms |

The bounded dispatch improves Glass on this device too, but the shared renderer
still has a substantial difference from the typed reference. It does not meet a
claim of universal runtime parity.

A separate MWE scans program headers into three count integers before shading,
allowing absent capacity classes to launch zero workgroups without rebuilding
plans after parameter edits. Glass changes 6.704 → 6.301 ms and Principled
7.093 → 6.990 ms, but partial transmission increases 8.764 → 9.278 ms. All three
images match. The extra classification pass and its dependency remain in every
bounce, so skipping rejected shader launches is not a universal win. That MWE
is saved under `out/metal_methodical_capacity_count_mwe_20261102.jl`; it is not
part of the production renderer.

Canonical-checkout acceptance reruns the 455 shader/material assertions in
addition to the 171 focused assertions above. All 13 changed Hikari file hashes
match Desktop; active packages resolve to the existing Mac `dev/` repositories
with managed GPUCompiler 2.17. The seven linked Mac worktrees, two plain source
copies and six Desktop worktrees are retired. Five Mac and fourteen Desktop
validation environments now reference canonical sources; the two obsolete
GPUCompiler 2.16 environments and their unique backport are archived explicitly.

The canonical-source timing rerun uses the same 512²/eight-sample hardware-ray
tracing fixture, five warmups and 51 alternating triples. Glass measures
45.128 / 79.772 / 51.365 ms and layered Principled 36.944 / 54.988 / 44.605 ms
for typed reference / former shared dispatch / current dispatches. Every image
is bit-identical, with the expected one / one / three shade signatures. Current
dispatches improve the former shared path by 35.6% and 18.9%, while remaining
13.8% and 20.7% slower than the retained typed reference. Raw samples are saved
in `out/canonical_repository_reconciliation_20261102/metal-canonical-runtime.json`.

The same canonical project also passes all 171 focused assertions with
`--check-bounds=yes` (r164). That fresh run takes 10m09s, including 140 seconds
of package-cache rebuilds; the direct-shader test takes 4m44s. These cold costs
remain distinct from the warmed frame timings above.

The final source audit also found older plain validation snapshots outside Git's
worktree registry. Those snapshots are archived with verified file hashes and
their source directories removed: 17 on Desktop and eight on Mac. The archives
and environment backups remain under `out/canonical_repository_reconciliation_20261102`;
validation uses the existing `dev/` repositories.
