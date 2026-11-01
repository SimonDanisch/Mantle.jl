# Material compilation reuse

Status: production integration validated on Vulkan (2026-10-31) and Metal
(2026-11-01); Metal runtime overhead remains above the accepted target.
The user accepted approximately 4% runtime overhead; eliminating it is no longer
a prerequisite for integration. The current source selects shared Cycles storage
by default and removes the old three-model storage implementation. Glass now returns the canonical Principled configuration. Full-suite Vulkan reference validation
and current-source timing pass. The accepted roughly 4% overhead applies to
the layered Principled fixture; some other models have larger
measured overhead, recorded below. The broader compiler work below
is recorded separately from this integration.

## Contracts

- Public material construction and nested field updates retain one canonical
  material graph and shading path. `Glass(...)` is a convenience constructor for
  a Principled configuration, not a second glass implementation.
- Graph combinations within an existing kernel capability and storage class reuse
  native BSDF pipelines. New capabilities can compile a bounded additional class.
- Correct sampling, PDFs, wavelength termination and rendered images are required
  alongside runtime, memory and compilation measurements on Vulkan and Metal.
- Do not persist host addresses or invalidate correctness to obtain cache hits.

## Metal shaders with symbolic Julia references

Metal's current persistent cache excludes modules with any relocation. Real Hikari
shaders carry Julia symbol references such as `convert`, `sqrt` and `trunc` from
error paths; these therefore miss after restarting Julia. Warm session reuse works.

Separate the portable shader code from the current session's bindings:

1. Normalize references into deterministic, indexed runtime loads. Complete
   pruning and finalize the slot order before deriving the cache key. Establish
   exactly which part of Metal/GPUCompiler lowering owns this boundary.
2. Key the cache by canonical shader IR, the ordered portable slot descriptions,
   relocation ABI/schema, compiler identity and target/configuration. Canonical
   symbol renaming must preserve the correspondence between IR and slot metadata.
3. Store compiled AIR/metallib plus its entry point, logging requirements and
   portable slot manifest. Never store resolved process addresses.
4. On a hit, validate the manifest against the current compilation and resolve
   bindings in the current Julia process. Keep resolved objects rooted and bind
   the resulting table for the lifetime of the kernel's use.
5. Initially admit only well-defined symbol targets demonstrated by the renderer.
   Keep arbitrary objects and unsupported target kinds excluded until their
   identity, invalidation and lifetime contracts have tests.

The final slot order must match the indices baked into the cached code. Reusing
the pre-lowering relocation list without proving that correspondence is unsafe.

Acceptance checks:

- A small symbol-reference kernel and real Hikari shade kernels execute correctly
  after a Julia restart with zero new Metal shader compilations.
- Reordered manifests, altered symbols, callee edits, target changes and compiler
  changes cannot reuse incompatible artifacts.
- Relocation-free kernels retain their existing behavior. Unsupported references
  remain a deliberate miss, with a diagnostic reason.
- Record cache lookup and binding time independently from Julia inference/LLVM
  generation. This change alone does not skip LLVM generation before lookup.

## Structural specialization outside the BSDF kernel

Hikari's BSDF queue can use a bounded runtime storage class while other passes
still specialize on exact material/graph types. Identify those misses with an
update trace before changing their interfaces. The current source still records
a queue/dispatch per concrete source material type; sharing a native pipeline
does not merge those queues. Many structurally distinct materials therefore
need separate launch-overhead measurements.

Replace exact graph structure in kernel argument types with stable buffers and
runtime descriptors: program instructions, material slots, input offsets and
resource indices. Keep the operations and buffer ABI in a bounded capability
class. Preserve required specialization where it actually changes generated code
or resource access, rather than placing every operation in every kernel.

Maintain Mantle's access analysis and synchronization contracts: stable argument
types must not hide which buffers or images a pass reads or writes. Numeric edits
should update data; structural edits within a class should update instructions
and offsets. Changes outside a class can select a new class.

Acceptance checks:

1. Warm a capability class, then construct unseen node combinations and orders.
2. Record inference, LLVM, shader and native-pipeline cache hits/misses separately
   for each pass; verify actual native pipeline reuse where available.
3. Verify rendered images and repeated updates, including buffer growth, class
   transitions, images, derivatives and alpha tests.
4. Compare full-frame runtime and memory against the retained reference. Report
   first-use startup, restart startup and warm updates separately.
5. Exercise the same update through RayMakie's ComputeGraph/nested material fields.

## Scope still to complete

The shared Cycles storage is integrated. Current-source checks pass for all 17
Cycles reference batches, nested material/image updates and native pipeline
reuse. The runtime tradeoff has been measured on the final source.
The canonical Glass constructor returns Principled. There is no
production switch between the former small storage and the integrated storage.

Native pbrt component/layered estimators retain schema/type specialization.
Current Metal material and update checks pass; the larger M5 runtime regression
remains open, as detailed in [Metal validation](metal-material-validation.md).
Neither broader renderer-pass specialization nor Metal symbolic-reference
caching is solved by the Cycles storage integration below.

## Shared BSDF storage: immediate implementation

The storage holds the BSDF parameters already calculated for one surface hit:
weights, local frames, roughness, Fresnel parameters and model-specific data.
It is local to the shading invocation. It is not a new buffer for every pixel,
and sampling must not evaluate the material graph a second time.

Use one counted local layout for the supported Cycles models: Principled, its
glass preset, metallic Fresnel models, refraction, translucency, Toon and subsurface.
The former separate small layout has been removed. Each record has a model tag,
fifteen Float32 parameters and the existing weighted frame: 108 logical bytes per slot.
This reduces the original 124-byte prototype by
restoring known unit Schlick tints from the accepted Principled model contract.
It keeps coloured glass transmission and both physical-conductor spectra.
Logical size alone does not establish GPU register or spill-memory usage.

The implementation sequence is:

1. Make `Glass(...)` return the same Principled description, program and BSDF
   implementation as explicit Principled transmission. Delete the separate glass
   implementations and route nested updates through Principled fields.
2. Pack and restore every supported lobe without changing its evaluation,
   sampling, PDFs, scattering restrictions, dispersion or subsurface walk.
3. Interpret the existing program into local storage once. The setup signature
   uses a capability mask and bounded capacity, not the exact ordered BSDF-node
   tuple. Packing the result of the old typed setup is only a storage experiment;
   it does not demonstrate compilation reuse.
4. Exercise the setup and evaluation on the GPU in isolation, then in the full
   renderer. Bound compilation work before enabling the larger storage by default.
5. Warm a class and test previously unseen mixtures and node orders. Measure
   native pipeline reuse, first-use compilation, warm frame time, registers and
   spill memory. Repeat on Metal before claiming cross-backend parity.
6. Integrate the validated classes and regression tests into Hikari. Reject
   unregistered Cycles nodes explicitly; retain the native pbrt estimator adapter
   for its distinct sampling semantics, and document remaining specialization.

Measured integration progress as of 2026-10-31:

| Part | Evidence | Remaining acceptance work |
| --- | --- | --- |
| Canonical Glass | Separate implementations removed; constructor/material/shader and all 93 current nested RayMakie checks pass on Vulkan and Metal | Broader compiler work below |
| Shared record and program | 28,377 final model/class/direct/composition/canonical checks across 73 configurations; all 17 Cycles reference batches pass 321 assertions on both backends | Metal runtime cost |
| Rendered parity | 19 final checks cover hardware/software tracing, computed inputs, bump normals, eight-term graphs and updates on both backends | Wider scene coverage |
| Mixed native graphs and energy | 1,968 final-source checks pass, including native CPU/GPU composition on software/hardware tracing, furnace, subsurface selection and retained emission | Native schema specialization remains |
| Native pipeline reuse | Vulkan's 28 checks and Metal's 40 checks confirm shared native pipeline identity within tested capability classes, with pixel-identical reference images | Other capability classes and persistence |
| Runtime | Vulkan Principled: 26.367 ms typed versus 27.414 ms shared (+3.97%); Glass +0.95%. M5 Principled +47.57%, Glass +66.17% | Metallic overhead, Metal runtime regression and hardware-tracing timings |
| First use after a warm class | Final-source subsequent unseen graphs take 0.274–0.279 s including host setup/rendering; initial loaded-session candidate draw took 33.56 s | Cold/restart startup measured separately |
| Compiler correctness | 158 focused compiler checks plus six interleaved-access regressions pass | General outlined-evaluation device fault remains unresolved |

Final runtime measurements use 51 alternating frame pairs after five warmups
on an initially idle NVIDIA RTX 4000 Ada. These are final production measurements
after the capacity fix below. Medians in milliseconds:

| Material | Typed reference | Shared storage | Difference |
| --- | ---: | ---: | ---: |
| Layered Principled | 26.367 ms | 27.414 ms | +3.97% |
| Diffuse | 21.662 ms | 21.911 ms | +1.15% |
| Sheen | 21.853 ms | 22.248 ms | +1.81% |
| Glossy | 22.565 ms | 22.881 ms | +1.40% |
| Glass | 36.964 ms | 37.314 ms | +0.95% |
| Metallic | 22.890 ms | 24.322 ms | +6.25% |

All six full-image comparisons pass (36 assertions). The largest component
error is 3.83e-6 for Glass. These measurements compare the same current
material semantics against retained typed formulas, including Principled for
the canonical Glass preset. They do not claim old coloured Blender Glass
semantics. The driver did not return register/scratch statistics; inferring a
register or spill cause from these timings would be speculation.

The capacity experiment kept the same interpreter, record and evaluation
formulas while reducing capacity from four to one for one-lobe graphs. Paired
51-frame comparisons measured Diffuse 21.211 → 21.962 ms (+3.54%), Sheen
21.477 → 22.256 ms (+3.62%), Glossy 22.113 → 22.873 ms (+3.44%), and Metallic
22.485 → 24.409 ms (+8.56%). All image comparisons pass. Capacity therefore
explains most of the simple-model overhead in this fixture; the precise native
register/local-memory mechanism has not been established.

This capacity choice is now integrated for both backend classifiers: zero for
non-scattering graphs, one for one-lobe graphs, and powers of two starting at
four for compositions. It does not add another evaluator or a renderer switch.
Different single-node Diffuse/Sheen/Glossy graphs share the one-slot class;
two-to-four-lobe compositions share the four-slot class. Crossing that capacity
boundary can compile a new kernel. The tradeoff adds one bounded capacity class,
not a class for every ordered node tuple. Principled/Glass's existing eight-slot
class and its demonstrated composition reuse remain unchanged. The fresh
73-configuration model/class/direct/composition/canonical suite passes 28,377
checks after this change. A fresh GPU run with that capacity choice passes all
340 assertions: the 321 Cycles reference checks and 19 hardware/software
image/update checks (13m26.5s inside the testset). The final production repeat measured +1.15–1.81% for Diffuse/Sheen/Glossy
and +6.25% for Metallic. Native pipeline identity confirms that the three
simple models share one pipeline. The two paired runs place Metallic's
remaining overhead at roughly 6–9%; this remains an explicit optimization
question. The 4% result does not describe every model.

The common counted
layout and inline evaluator are the production path. The inline boundary avoids
a separate outlined-evaluation device fault; it does not fix that compiler bug.

The full GPU reference run reached batch 15 after completing batches 1–14.
An earlier run was OOM-killed; a fresh retry grew beyond 20 GB RSS at the
microfacet-distribution batch and was interrupted. A signal-triggered Julia stack
sample placed the active work in NVIDIA's native compiler under
`vkCreateComputePipelines`; LLVM passes and SPIR-V emission had already finished.

A shared reflection-only variant still failed native compilation under a 4-GiB
data-segment limit. Sharing GGX/Beckmann geometry as well reduced the captured
shader from 2,104,936 to 1,921,576 bytes. The latter compiled in an independent
Vulkan process under that same limit and a 60-second deadline (18.4 seconds
including device setup). The exact same shader bytes then rendered successfully
in Hikari (21.8-second first draw). All 28,116 differential checks over 73
configurations, plus the five Sheen shadowing checks, passed for this variant.
The integration now shares both distribution and Fresnel geometry. This evidence
does not identify NVIDIA's internal algorithm or establish that byte size alone
explains its previous memory growth. The final-source full Cycles reference run
now passes all 321 assertions across 17 batches. It finished in 16m53s including
package loading (15m12.5s inside the testset), under the validation resource
limits below. This is a full reference run, not a partial batch result.

Alternative local-array storage rendered correctly but measured 680.00 ms
versus 31.57 ms typed (21.54 times slower). Deferred padding initialization
passed 5,376 CPU checks and image comparison but measured +6.95%. Both remain
diagnostic experiments; neither is a selectable production rendering path.

### Compiler constraints found by the experiment

GPUCompiler 2.16 deliberately inlines functions containing unreachable error
paths so that aborting a helper also aborts its kernel invocation. `@noinline`
does not override this correctness requirement. Dispatch over a host-validated
closed model set must not introduce an unnecessary throwing fallback at every
stored lobe. Unregistered Cycles node models are rejected before kernel creation; native
pbrt models retain their documented estimator adapter.

Lava also inlines helpers when pointer arguments have incompatible logical
SPIR-V types. Passing a lobe by address from within a larger packed array hits
this constraint. The integrated evaluator passes its tag and parameter values
individually instead. Whole, consistently typed local arrays used by pure
numeric helpers can retain their container type across a function boundary;
subobject addresses and pointer-bearing resource descriptors need separate
handling. An initial broader exception failed SPIR-V validation in the reference
renderer and has been narrowed. Qualified numeric helper arguments now receive
owned local copies before logical SPIR-V lowering. Wide byte loads crossing
fields and ABI padding have focused regression tests. The integrated Principled
renderer and the full distribution reference coverage now pass.

The generic full-render attempts reached approximately 16 MB of LLVM IR and an
observed system OOM event. Do not keep repeating those runs without checking the
surviving function boundaries and generated size first. Historical storage experiments are retained in
`Hikari/benchmarks/shared_surface/` for reproducible investigation. They are not
loaded by the normal package or benchmark loader.
The working inline variant retains a bounded capability/capacity signature;
its shared evaluation code is inlined into each capability kernel. Pipeline reuse
across graphs is verified; per-function reuse across distinct capability classes
is not established by that result.

The latest function-boundary reduction reads exact per-lane captured records on
the GPU, without any GPU material setup or RGB lookup. The outlined reflection
evaluator passes coat and loses the device on specular; replacing its private
input pointers by scalar arguments passes the captured specular case. The complete program still loses the device with scalar inputs. The captured
specular case passes on software Vulkan (llvmpipe); identical emitted shader
bytes across devices have not yet been established. A packed three-lane UInt64
vector return also passes the captured RTX case in a verified current compiler
world, but does not fix complete program evaluation. These are diagnostics, not
accepted implementations. Generated-function redefinition experiments must use
a fresh kernel and the current compiler world before their outcome is trusted.

## Integration and open optimization work

The accepted implementation is the 108-byte counted immutable tuple with inline
model evaluation. It is now included by Hikari and selected for all supported
Cycles nodes, including single nodes, Principled and its Glass preset, metallic
Fresnel variants, refraction, translucency, Toon and subsurface. Capacity grows
by powers of two instead of falling back at 32 lobes. The separate small
Diffuse/Sheen/Glossy implementation is removed. The fresh class/model suite
passes 28,181 checks across 73 configurations, and all Cycles reference batches pass.
A missing diffuse bump-shadowing trait for stored Sheen was found during review
and corrected with a nontrivial tilted-normal regression.

The remaining optimizations are follow-up work, with distinct evidence gaps:

- Mutable local-array storage: image parity holds, but paired runtime was 21.54
  times the typed reference. The cause is unresolved. Inspect native local
  memory/spills, array accesses, aliasing constraints and compiler optimization
  before reconsidering it; attributing it to spills now would be speculation.
- Deferred padding initialization: 5,376 CPU checks and image parity pass;
  measured overhead was 6.95%. Per-stage timing and generated-code comparisons
  are needed before explaining why fewer initial stores did not help.
- Shared microfacet geometry: the reflection-only experiment did not improve
  frame time, but extending sharing across GGX/Beckmann fixes the observed native
  compile failure in the isolated Glass/Beckmann scene. This is now part of the
  integrated implementation. Measure its final runtime rather than inferring a
  runtime win from smaller code.
- Outlined evaluation: the retained small reproducers and logs localize failures,
  but neither scalar inputs nor packed vector returns fixes the complete program.
  Splitting evaluation into a float4 response and scalar PDF passes the captured
  specular term but still loses the device after runtime setup. Outlining only the
  Beckmann visible-normal sampler gives pixel-identical results in a 128-lane
  inline/outlined comparison but did not resolve the large native compile in a
  bounded observation. Neither experiment is integrated. The production path
  inlines evaluation; per-function reuse across distinct classes is not proved.

Native pbrt component and layered random-walk estimators remain in the same graph
infrastructure but retain schema/type specialization. Their common storage and
structural compilation costs are not declared solved by the Cycles integration.
The Mac working changes have now been reconciled with the current sources and
installed in its main checkouts. Metal's complete material reference suite and
nested RayMakie updates pass. Native pipeline reuse is verified, but paired M5
rendering shows materially larger runtime overhead than Vulkan. See
[Metal validation](metal-material-validation.md) for measurements, the native
compiler fix and the remaining performance and persistence limitations.

The subsequent Metal acceptance uses optimizable LLVM minnum/maxnum intrinsics
and retains uncontrolled loops in functions with at least 4,096 blocks or
32,768 instructions before AIR downgrade. The dense-shader fallback matters:
49,766 instructions at that point expand to 67,879 later. Both GPUCompiler 2.16
and 2.17 pass all 2,308 material assertions plus 31 compiler checks. This resolves
the observed native compile failures, not the runtime goal. The final M5 paired
measurements still show +50.15% for layered Principled and +69.80% for its Glass
preset. Standalone prebuilt-BSDF diagnostics retain evaluation overhead without
material setup, so optimizing setup alone is insufficient. Weighted evaluation
still loses requested noinline boundaries when exceptional exits require late
inlining; the smaller bounded-square-root example now retains its boundary.

Native compile diagnostics used an independent Vulkan executable with time and
data-segment limits. Disabling native optimization did not finish within 60
seconds; marking all previously uncontrolled loops `DontUnroll` did not finish
within 90 seconds. These are bounded observations, not proof that those flags
can never help. Repeating unbounded native compiles was rejected by automatic
approval review; subsequent validation preflights large shaders in a bounded
process before the renderer is allowed to compile them.

A further cold-compile profile localized a 112.77-second pass to repeated
`flatten_type_with_offsets` calls in `decompose_typepun_gep_loads!`. A separate
bounded run reached its 20-minute deadline in the same family of work (31.97
billion cumulative Julia allocations; this is not resident memory). The new
local layout cache lives for one pass invocation and one DataLayout. In an
8,192-field/16-load reproducer it preserves byte-identical LLVM output, reduces
allocated bytes from 57,853,232 to 6,958,024 and takes 5.56 ms versus 36.87 ms.
158 related checks pass, including reuse of one LLVM struct type across modules
with different DataLayouts. The same full-render workgroup/typepun block fell
from 112.77 to 34.23 seconds, and the bounded full reference suite now completes.
A second profile identified repeated inspection of already checked accesses.
Remembering those accesses within each function/pass avoids repeating that work
while retaining iterator restarts after instruction erasure. Six additional tests
cover interleaved unchanged and rewritten loads/stores and pass idempotence.
Both changes preserve byte-identical LLVM output against the earlier pass in
the focused comparison fixtures. They do not establish perfect cold compilation.

Validation now limits the dedicated Julia worker's data segment to 8 GiB and
its systemd scope lifetime to 20 minutes, in addition to the per-native-compile
limits. The user service does not delegate the cgroup memory controller;
`MemoryMax` alone would not enforce a RAM limit here. The kernel's verified
`RLIMIT_DATA` is used instead. These limits belong to validation, not production
renderer behavior.

## Native Metal binary-function probe

Metal's `research/binary_function_reuse.jl` now demonstrates the native separate
compilation boundary on the M5. It creates two visible MSL helpers with
`MTLFunctionOptionCompileToBinary` once, passes those same function objects in
`MTLLinkedFunctions.binaryFunctions` to two different compute pipelines, and
dispatches both through visible function tables. One pipeline selects a helper
per lane; the other composes the two helpers in either order. All four checks
pass with zero numerical error across 256 lanes (run r641, 2026-11-01).

This follows Apple's [separate compilation API](https://developer.apple.com/videos/play/wwdc2020/10013/).
The probe uses MSL, not Julia-generated helper AIR. It establishes neither
renderer speed nor persistent Julia helper caching. Connecting Julia helpers
requires a compatible visible-function ABI and metadata, the device-side table
call, and correct lifetime and invalidation handling. Those pieces still need
small correctness and reuse checks before a renderer experiment. Binary calls
also give up cross-function optimization, so they require a runtime comparison;
the current Metal material regression remains unresolved.

The companion `research/julia_binary_function_reuse.jl` extends the same probe
to Julia-generated `Float32(Float32)` helpers. GPUCompiler emits their LLVM,
the probe attaches scalar `air.visible` input/output metadata, supplies the
module AIR/Metal version metadata omitted for non-kernel jobs, and packs each
helper as `PROGRAM_VISIBLE`. Native binary compilation and both existing MSL
consumers pass eight checks with zero error (run r653). This proves the scalar
Julia producer boundary. It does not yet cover Julia consumers, aggregate BSDF
arguments, kernel state, relocations or persistent helper invalidation. The
earlier "MSL only" limitation applies to the first probe, not this extension.
