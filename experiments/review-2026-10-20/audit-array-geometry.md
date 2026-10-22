# Audit: `src/array/` and `src/geometry/`

Historical snapshot: counts, locations and deletion hypotheses below predate
consolidation. Follow [current ownership and deletion gates](consolidation-order.md).

Scope: every line of `src/array/{gemm,fft,gemv,gemm_cm2,gemm_tensile,indexing,launch}.jl`
and `src/geometry/{epa,gjk,narrow_phase,transform,convex_shape,types}.jl` (6716 lines),
plus `src/vulkan/array/gemm.jl` (211) as the GEMM entry point they serve. Paths are
relative to `/sim/Programmieren/VideoEdit/dev/Mantle` unless stated. Line counts of
code/comment/docstring per file are from a script run over the files (comment = lines
starting with `#`; docstring = lines inside `"""` blocks); bucket splits are estimates
from reading and are labelled so.

Target, as the docs state it: an operation is one Julia method on `MantleArray` whose
body calls `dispatch!(location(a), …)`; selection is a call into the device's selection
table keyed on function, element types, shapes, layout, alignment (`docs/api.md:103-120`,
`docs/recording-plan.md:234-290`); device types contribute candidates and verbs only
(`api.md:241-357`); `ArrayLaunch`/`runlaunches!`/`dispatchlaunches!`/`kilaunch!` and the
`native_*`/`staged_gemm_tile`/`librarygemm` hooks are deleted (`recording-plan.md:1962-1966`,
`api.md:679,687-690,725`).

## 1. Inventory

Buckets: (a) core operation/kernel per `api.md` 2.4 and the plan's "Operations";
(b) backend verb; (c) nowhere (dead, superseded, abandoned variant, test/debug only,
duplicate, history prose); (d) a different package. The four sum to the file's lines;
code/comment columns are measured, buckets are estimates.

| file | lines | code | comment+doc | what it does | (a) | (b) | (c) | (d) |
|---|---|---|---|---|---|---|---|---|
| `src/array/gemm.jl` | 2926 | 1343 | 1583 | fp16 cooperative-matrix GEMM in 8 kernel families × 7 tilings, an fp32 staged GEMM, a per-element GEMM, a split-K GEMV, their tiling tables, flags, selection chain and eager/declared entry pairs | 1300 | 0 | 1626 | 0 |
| `src/array/gemm_tensile.jl` | 264 | 138 | 126 | the shipping fp16 GEMM: rocBLAS's RDNA3 kernel structure ported (128×128×16, 2 LDS buffers, 128-bit loads), bias+epilogue in the store | 224 | 0 | 40 | 0 |
| `src/array/gemm_cm2.jl` | 371 | 151 | 220 | two coopmat2 (tensor-addressed) GEMM kernels, "not routed to" (`gemm_cm2.jl:4-5`), reached only by `test/test_gemm_cm2.jl` and `JuliaVision/tools/gemm_cm2_*.jl` | 0 | 0 | 371 | 0 |
| `src/array/gemv.jl` | 598 | 218 | 380 | two `@eval`-generated GEMV kernel families (K-contiguous llama.cpp port; M-contiguous with bias+epilogue), their configs, eager and declared entry points, load-time pregeneration | 200 | 0 | 398 | 0 |
| `src/array/fft.jl` | 1018 | 546 | 472 | batched 1-D FFT: a radix-8 power-of-two kernel, a mixed-radix code generator, real FFT via half-length trick, eager and declared twins of each, STFT and Hann window | 640 | 0 | 296 | 82 |
| `src/array/launch.jl` | 120 | 46 | 74 | `ArrayLaunch` value + `runlaunches!`/`kilaunch!`/`dispatchlaunches!`: the eager-vs-declared split the plan deletes (`recording-plan.md:1962-1964`) | 0 | 0 | 120 | 0 |
| `src/array/indexing.jl` | 141 | 76 | 65 | `FastDiv32`, `cart32`, `splitidx`, `launchgroup`/`staticgroup` (kernel index helpers), and `apair` (a convolution gather hook DNNKernels extends) | 100 | 0 | 8 | 33 |
| `src/geometry/transform.jl` | 134 | 46 | 88 | `Mat3x4f` instance transform, 4×4→3×4 conversion, quaternion→`NTuple{9}` and `build_4x3` hand math | 50 | 0 | 0 | 84 |
| `src/geometry/types.jl` | 52 | 28 | 24 | `TrianglesGeometry`/`AABBsGeometry` holding raw `UInt64` device addresses and a `VkFormat` as `UInt32`; `AABB` | 0 | 36 | 12 | 4 |
| `src/geometry/convex_shape.jl` | 55 | 13 | 42 | `ConvexShape`, `UnitCube`, `support` | 0 | 0 | 0 | 55 |
| `src/geometry/gjk.jl` | 378 | 217 | 161 | GJK overlap test over `NTuple{12,Float32}` transforms, "CPU reference" (`gjk.jl:1`) | 0 | 0 | 0 | 378 |
| `src/geometry/epa.jl` | 475 | 275 | 200 | EPA penetration depth with 7 `MVector` scratch arrays, "CPU reference" (`epa.jl:1`) | 0 | 0 | 0 | 475 |
| `src/geometry/narrow_phase.jl` | 184 | 66 | 118 | two kernels composing GJK+EPA per pair; `ContactRecord`, `NO_CONTACT` sentinel | 0 | 0 | 0 | 184 |
| **total** | **6716** | **3163** | **3553** | | **2514** | **36** | **2871** | **1295** |

Notes on the table:

- 53% of the lines are comments or docstrings (3553 of 6716). `gemm.jl` is 54%
  (1583/2926), `gemv.jl` 64%, `gemm_cm2.jl` 59%. 112 of those lines contain a
  throughput or "measured" figure, 12 carry a date. See pattern 12.
- `gemm.jl` (c) = 1626: the six kernel families nothing selects by default (517 source
  lines, listed in section 2), their docstrings and flag constants (≈430: `gemm.jl:182-297`,
  `299-313`, `357-400`, `702-805`, `808-831`, `848-871`, `874-921`, `961-1003`, `1005-1015`),
  the selection chain over them (`2531-2550`), and ≈660 lines of history and benchmark prose
  outside those (e.g. `1-47`, `53-80`, `1947-1998`, `2031-2075`, `2104-2130`, `2148-2177`,
  `2612-2667`, `2698-2752`).
- `gemm_cm2.jl` is entirely (c): the file says so (`gemm_cm2.jl:3-7`), and the only
  references outside its tests/tools are comments in `src/graph/access.jl:681,1068`,
  `src/graph/build.jl:58`, `src/vulkan/access.jl:118`.
- `gemv.jl` (c) = 398: the K-contiguous family (`gemv.jl:139-227`, `420-439`, `441-470`,
  `544-550`) has no caller outside `test/test_gemv.jl` (every DNNKernels call is the
  `Transpose` form, `JuliaVision/DNNKernels/src/kernels/extern/matmul.jl:284`,
  `src/emit.jl:4643`); pregeneration `555-598`; `SHARED_ID` `127-137`; prose `1-124`,
  `336-412`.
- `fft.jl` (d) = 82: `stft`/`stft_frames_kernel!`/`hannwindow` (`fft.jl:937-1018`) are a
  DSP front end, consumed by `JuliaVision/DNNKernels/src/kernels/extern/spectral.jl` and
  `WhisperRunner`; neither `api.md` nor the plan names them.
- Geometry (d): `gjk`, `epa`, `narrow_phase_*`, `ConvexShape`, `UnitCube` have **no caller
  in `/sim/Programmieren/VideoEdit/dev` outside Mantle's own tests** (grep over every `.jl`
  and `.md`; hits are `test/test_gjk.jl`, `test_epa.jl`, `test_narrow_phase_*.jl`,
  `test_kernel_tails.jl`, `test_convex_shape.jl`, `test_contact_record.jl`,
  `test_double_indirect.jl`, and one comment each in `Lava/src/compiler/compilation.jl:2626`
  and `ext/MantleROCmExt.jl:447`). The code is for a granular-physics project: "grain",
  "XPBD solver (P5)", "P4.4…P4.7" (`narrow_phase.jl:8-12,23-25`, `convex_shape.jl:33-34`,
  `epa.jl:13,240-257`, `gjk.jl:5`). `Raycore/src/collision.jl` exists (not read here) and
  is where collision code already lives outside Mantle. The export comment
  `src/Mantle.jl:390-393` ("`support` … the single most-used name Hikari and RayMakie take
  from this package") is false: Hikari and RayMakie call `Mantle.supports_rt_pipeline` /
  `supports_procedural_traversal` (`Hikari/src/fem.jl:762`, `integrators/volpath/hw-rt.jl:110`),
  never `support(`. Only `UnitCube` exists as a shape.
- `transform.jl` (d) = 84: `quat_to_rot3x3`, `build_4x3`, `build_4x3_pervec`
  (`transform.jl:59-106`) are consumed by `Raycore/src/instanced-bvh.jl` (28 references),
  `Makie/RayMakie/src/plots/meshscatter.jl` (12) and `src/raytracing/instances.jl` (2):
  shared geometry math over `NTuple{9}`/`NTuple{12}` that belongs beside GeometryBasics'
  `Mat`/`Quaternion`, not in a runtime. `Mat3x4f` + `mat4_to_vk_transform` (`41-56`,
  `109-134`) stay as the TLAS records kernel's conversion (api 2.6 takes `Mat4f`,
  `api.md:135`).
- `types.jl` (b) = 36: `TrianglesGeometry` stores `vertex_format::UInt32` ("Maps to
  `VK_GEOMETRY_TYPE_TRIANGLES_KHR`", `types.jl:13-23`) and raw addresses: a Vulkan build
  description in core. The target `BLAS(dev, vertices, faces)` (`api.md:133`) takes arrays;
  `accelsizes`/`placeaccel` (`api.md:294-295`) are the verbs. `AABB` (`49-52`) is
  GeometryBasics' `Rect3f`.

## 2. Duplication

### 2.1 Every GEMM/GEMV variant

Kernel function, where, what it covers, whether the public API reaches it.
"Public API" = `LinearAlgebra.mul!` on a `LavaArray` (`src/vulkan/array/gemm.jl:48-103`)
or the exported-by-name entry points DNNKernels calls (`coopmat_gemm!`,
`coopmat_gemm_dispatch!`, `scalar_gemm_dispatch!`, `gemv!`, `gemv_dispatch!`).

| # | kernel | file:line | source lines | types / tile | reached by | status |
|---|---|---|---|---|---|---|
| 1 | `coopmat_gemm_tensile_kernel!` | `gemm_tensile.jl:182-234` (body spliced from `131-165`) | 106 | fp16×fp16→fp16/fp32; 128×128×16, 4 subgroups, 2 LDS buffers; bias+epilogue | `coopmat_gemm_launches` first (`gemm.jl:2497-2499`) when `tensile_gemm_ok` (`gemm_tensile.jl:251-259`: M,N %128, K %32, 16-byte aligned, nbatch=1, splitk=1) | shipping default |
| 2 | `coopmat_gemm_staged_kernel_{1..7}_prefetch!` | `gemm.jl:1290-1406` | 117 ×7 | fp16→fp16/fp32, 7 tilings (`147-174`), vec2 shared, `Int32` indices; bias+epilogue | `narrow && prefetch` (`2546-2547`); shapes tensile declines but a tiling divides | shipping |
| 3 | `coopmat_gemm_staged_kernel_{1..9}_gather!` | `gemm.jl:1783-1875` | 93 ×9 | as 2, A read through `apair` loader; 7 tilings + `CONV_GATHER_TILINGS` (`1730-1733`) | `aload !== nothing` (`2516-2529`); DNNKernels conv (`conv_coopmat.jl:1102,1116`) | shipping (conv) |
| 4 | `coopmat_gemm_kernel_{1,2,4}!` | `gemm.jl:617-700` | 84 | fp16→fp32 register block, no shared; batched (`nbatch`) and split-K via subgroup index | fallback when no tiling divides, `nbatch>1` or `splitk>1` (`2562-2575`); attention (`attention.jl:799-801`), conv split-K | shipping |
| 5 | `…_v2n!` ×7 | `gemm.jl:1217-1284` | 68 ×7 | as 2 without prefetch | only `prefetch=false` (`2548-2550`) or the `doublebuf` fallback chain (`2539-2541`) | not selected by default; `prefetch` is never passed by a consumer |
| 6 | `…_v2!` ×7 | `gemm.jl:1149-1212` | 64 ×7 | as 5 with `Int` indices | only `!gemm_fits32` (≥2³¹ elements) or `narrow_ok=false` (`2552-2553`); "nothing in this repo reaches it" (`2509-2514`) | unreached; its docstring records it computed every element wrong, then "no longer reproduces" (`848-871`) |
| 7 | `coopmat_gemm_staged_kernel_{1..7}!` (scalar shared) | `gemm.jl:1062-1131` | 70 ×7 | as 2 with `Float16` shared | only `vec2=false` (`2531-2532`); every tiling has a vec2 twin (checked: all seven satisfy `1142-1143`) | dead |
| 8 | `…_db!` ×7 | `gemm.jl:1424-1519` | 95 ×7 | double-buffered | only `doublebuf=true`; default `false` (`367`); "MEASURED, and it does not pay" (`874-921`) | dead |
| 9 | `…_v4!` ×7 | `gemm.jl:1536-1605` | 70 ×7 | 4-wide shared | only `vec4=true`; default `false` (`371`); measured −4% (`961-1003`) | dead |
| 10 | `…_abl!` ×7 | `gemm.jl:1626-1693` | 68 ×7 | wrong answers on purpose (`1005-1015`) | never in `coopmat_gemm_launches` (`GEMM_STAGED_ABL_KERNELS` appears only at `1015`, `1693`); `JuliaVision/tools/gemm_ablate.jl` | debug only |
| 11 | `gemm_cm2!` | `gemm_cm2.jl:134-222` | 89 | fp16→fp16/fp32 workgroup-scope matrices, device-derived tiling (`111-122`) | `coopmat_gemm_cm2!` (`339-371`), no caller but tests/tools | dead |
| 12 | `gemm_cm2_sg!` | `gemm_cm2.jl:252-301` | 50 | subgroup-scope tensor loads | `coopmat_gemm_cm2_sg!` (`309-327`), tests/tools | dead |
| 13 | `scalar_gemm_staged_kernel!` | `gemm.jl:2178-2280` | 103 | fp32 out, fp16/fp32 in; 64×64×32; α, β, strides, masked | `staged_gemm_ok` (`2689-2696`, `2877-2889`) | shipping (fp32) |
| 14 | `strided_gemm_kernel!` | `gemm.jl:1935-2029` | 95 (52 comment) | any `T<:Number`, strides, α, β | last fallback (`2898-2901`); fp64, complex, small, ragged, and every fp16→fp16 `mul!` (see 2.2) | shipping |
| 15 | `splitk_reduce_kernel!` | `gemm.jl:2394-2407` | 14 | sums partial planes | with #4 when `splitk>1` (`2581-2583`) | shipping |
| 16 | `gemv_splitk_kernel!` + `gemv_reduce_kernel!` | `gemm.jl:2784-2836` | 53 | GEMV, fp32 accum, split K | `mul!` with `N==1` (`2864-2876`) | shipping (third GEMV) |
| 17 | `gemvk_{NROWS}_{BLOCK}_{SUB}` | `gemv.jl:151-227` | 77 (generator) | `C[n] = A·B[:,n]`, Float32 only, subgroup reduce | `gemv!(C,A,B)` (`441-470`); no consumer | unused outside `test/test_gemv.jl` |
| 18 | `gemvn_{TM}_{BLOCK}_{UNROLL}` | `gemv.jl:268-333` | 66 (generator) | `C[m] = epi(bias[m] + W[m,:]·x)`, Float32, tree reduce | `gemv!(C,x,::Transpose)` (`472-491`), `gemv_dispatch!` (`526-542`); DNNKernels `matmul.jl:284`, `emit.jl:4643` | shipping (second GEMV) |
| 19 | `MPSGraphGemm` | `src/metal/ka.jl:227-235` | — | Metal's `mul!` is MPS; core's kernels are declined by `dev.tile == 16` (`src/metal/metal.jl:102-106`) | `native_gemm_dispatch!` | the second implementation the plan's rule 6 forbids |
| 20 | `GPUArrays.generic_matmatmul!` | upstream | — | non-`Number` eltypes | outside the `T<:Number` bound (`vulkan/array/gemm.jl:37-46`) | upstream |

Kernel source in `gemm.jl` that no default path selects: rows 5-10, 1217-1284 + 1149-1212
+ 1062-1131 + 1409-1521 + 1523-1607 + 1609-1695 ≈ 517 lines of source (generated ×7 each),
plus their docstrings/flags ≈ 430, plus `gemm_cm2.jl` 371, plus the K-contiguous GEMV ≈ 150.

### 2.2 Dispatch paths and the decision made in three places

Entry points into the family: `LinearAlgebra.mul!` (2 methods, `vulkan/array/gemm.jl:48-77,97-103`),
3 `Diagonal` disambiguations copied from GPUArrays (`124-191`, 68 lines, "mirrors its
implementation rather than `invoke`-ing it", `116-117`), `coopmat_gemm!` (`gemm.jl:2588-2592`),
`coopmat_gemm_dispatch!` (`2605-2609`), `gemmlaunch!` (`2908-2914`), `scalar_gemm_dispatch!`
(`2921-2926`), `gemv!` ×2, `gemv_dispatch!`, `gemv` ×2, Metal `native_gemm_dispatch!`.
Eleven names for one operation.

The fp16-tensor-core decision is made three times with three different predicates:

1. `mul!` (`vulkan/array/gemm.jl:69-72`): `T === Float32 && fp16 operands && isone(α) && iszero(β)
   && A isa LavaArray && B isa LavaArray && %16 && coopmatgemm(C)`. An fp16 **destination**
   never passes, so `mul!(C::LavaArray{Float16}, A16, B16)` runs `strided_gemm_kernel!`
   (row 14; ~0.5 TFLOP/s, `gemm.jl:2037`) while `coopmat_gemm!` on the same arrays runs the
   Tensile kernel (88 TFLOP/s, `gemm_tensile.jl:26-30`). A transposed operand (`PermutedDimsArray`)
   is refused by the `isa LavaArray` test and lands on the scalar path too (`gemm.jl:1891-1895`).
2. `coopmat_gemm_launches` (`gemm.jl:2454-2575`): a 122-line chain of `tensile → staged_gemm_tiling
   → gather → vec2/vec4/narrow/doublebuf/prefetch Dict lookups → block kernel → split-K`.
3. DNNKernels' own planner (`JuliaVision/DNNKernels/src/kernelplans.jl:42`, `mm_coopmat_plan`)
   reads `GEMM_TILE` (93 references in 3 files), `gemm_tiling` (21), `GEMM_TILINGS` (23),
   `coopmat_gemm_shape` (9), `gemm_padn` (7), `staged_gemm_tile` (8), `SGEMM_BM/BN/MINTILES/MAXWASTE`
   and decides, before calling Mantle, which Mantle kernel will run. The consumer has a copy
   of the selection.

How the plan collapses this: one `mul!(C::MantleArray, A, B, α, β)` (`api.md:72`) whose body
asks the device's selection table (`api.md:117-120`); each kernel above is a candidate value
with declared attributes (element types, divisibility, alignment, implements bias/epilogue,
batch, split-K); the table orders candidates by measurement or core's cost estimate. The
`isa LavaArray` and `%16` tests become the candidate's `divides` predicate; α/β become an
attribute (`implements αβ`) rather than a gate; the eager/declared pair collapses because
`dispatch!(location(C), …)` runs now or declares (`recording-plan.md:236-240`). DNNKernels'
planner goes (plan phase 11: "DNNKernels' GEMM … as `mul!`", `recording-plan.md:2064-2066`).

### 2.3 Three GEMVs

`gemm.jl:2784-2836` (split-K, fp32 planes, `N==1` of `mul!`), `gemv.jl:151-227` (K-contiguous,
llama.cpp `mul_mat_vec.comp`), `gemv.jl:268-333` (M-contiguous with bias+epilogue). The
header `gemv.jl:69-84` argues GEMV must not be `mul!` because the coopmat tile wastes 15/16;
that is a shape in the selection key, not a reason for a second public name. One GEMV kernel
over strides (the M-contiguous one already takes `W[m + M*(k-1)]`, `gemv.jl:312`) serves all
three layouts; the K-contiguous one has no consumer.

### 2.4 fft.jl

| operation | eager | declared | duplicated logic |
|---|---|---|---|
| complex FFT, power of two | `fft!` `fft.jl:354-385` | inside `fftany_dispatch!` `800-834` | `fftplan`, limit check, `fftgroup`, launch shape: `364-373` vs `814-822` |
| complex FFT, mixed radix | `fftmixed!` `713-745` | inside `fftany_dispatch!` `824-832` | plan, limit check, launch: `723-736` vs `824-831` |
| length dispatch | `fftany!` `747-766` | `fftany_dispatch!` `811-813,823-826` | `count_ones(N) == 1 && N >= 8` twice |
| real forward | `rfft!` `446-476` | `rfft_dispatch!` `836-863` | **different algorithms**: eager packs by `reinterpret` (`466-468`, free), declared launches `rfft_pack_kernel!` (`856-857`) and needs two caller-supplied scratches |
| real inverse | none | `irfft_dispatch!` `865-889` | the eager API cannot compute an inverse real FFT |
| allocating | `fft` `387-393`, `rfft` `478-489` | — | `rfft(; skew)` and `rfft!(; skew)` accept `skew` and never use it (`446-447`, `458-459` call `fftany!` without it) |

Two kernels for one transform: `fft_kernel!` (`159-268`, radix 8 with a leading 2/4) and the
`fftmixed_kernel` generator (`572-710`); `492-516` says the split is deliberate ("tuned vs
untuned"). The power-of-two kernel is the radix sequence `(LEAD, 8, 8, …)` with grouping
(`NB`) and a twiddle recurrence; folding it into the generator is possible (guess: −110 lines)
but not required by the plan. Required by the plan: one `fft!` method; the twins
(`800-889`, 90 lines) and the duplicated planning (≈40) go.

Lines recoverable (estimate): GEMM/GEMV ≈ 1760 of 4159 (`gemm.jl` dead families and their
prose ≈ 970, `gemm_cm2.jl` 371, K-contiguous GEMV 150, pregeneration 44, `launch.jl` 120,
eager/declared twins ≈ 110) plus the 68 copied `Diagonal` methods in `vulkan/array/gemm.jl`;
FFT ≈ 300 of 1018 (twins 90, duplicated planning 40, `FFT_SKEW` and dead `skew` kwargs 30,
pregeneration 44, STFT/Hann moved out 82, header prose).

## 3. Patterns that inflate the code

Costs are line estimates over the 13 files unless stated. No `try`/`catch` exists in these
files (0 `catch`); no `hasproperty`/`isdefined`/`hasmethod`/`applicable` call in code (the three
grep hits are comments, `gemm.jl`, `launch.jl:56-59`); no `isa` in code; no `_`-prefixed
function. Those owner rules are not violated here. The others are.

1. **Near-copies of one kernel per staging width, index type and schedule, then per tiling by
   `@eval`.** Rows 2, 5-10 of section 2.1 are one kernel: `1149-1212` (`v2`) and `1217-1284`
   (`v2n`) differ only in `Int32(…)` wrapping (182 `Int32(` in `gemm.jl`); `1536-1605` (`v4`)
   differs from `v2` by `NTuple{4}` vs `NTuple{2}` and `÷4` vs `÷2`; `1424-1519` (`db`) adds a
   buffer index; `1290-1406` (`prefetch`) vs `1783-1875` (`gather`) differ by `apair` and an
   `@nexprs` unroll (`1754-1780` explains why two kernels rather than a flag: Lava's scalar
   replacement). Each is then `@eval`ed seven times over `GEMM_TILINGS` because `@nexprs`
   wants a literal (`gemm.jl:38-43`). The three block kernels (`617-700`) are the same body
   `@eval`ed per `BLK`. The two GEMV families are `@eval` generators (`gemv.jl:112-123`), as is
   the mixed-radix FFT (`572-710`). The plain-Julia form is one kernel with the tiling, width,
   index type, schedule and loader as type parameters (a `@generated` body where `@nexprs`
   needs literals; untested under Lava, labelled a guess). Cost: ≈ 630 lines of kernel source
   in `gemm.jl:1052-1695` collapse to one ≈ 150-line kernel; the `@eval` loops force
   `invokelatest` at 11 sites (`launch.jl:61-65,115`, `gemv.jl:167`, `fft.jl:62+680`), a
   `SHARED_ID` counter (`gemv.jl:127-137`), and pregeneration lists because `@eval` on the
   call path breaks precompilation (`gemv.jl:555-581`, `fft.jl:891-934`): ≈ 120 more lines.

2. **Capability branching by hand, `nothing` as "not supported".** `gemm_tiling` (`402-413`),
   `staged_gemm_tiling` (`430-443`), `gather_gemm_tiling` (`421-428`), `gemm_cm2_tiling`
   (`gemm_cm2.jl:111-122`), `coopmat_gemm_cm2!` (`339-348`, "returns `nothing` … so a caller can
   fall back"), `gemmstrides(::AbstractArray) = nothing` (`2315`) then checked at `2855-2858`,
   `2909-2910`; `fftplan_mixed` (`538`); `KI.supports_tensor_addressing(backend) || return nothing`
   (`gemm_cm2.jl:314,346`); `coopmatgemm(C)` as a `Bool` gate (`vulkan/array/gemm.jl:72`,
   `src/memory/array.jl:157`) plus the `staged_gemm_tile()` hook comparing `dev.tile == 16`
   (`src/runtime/backendhooks.jl:31-50`). Element-type checks as runtime `===` chains:
   `staged_gemm_ok` (`2689-2696`), `gemv_split` (`2780-2782`), `tensile_gemm_ok`
   (`gemm_tensile.jl:252-253`), `eltype(C) === Float16` (`gemm_cm2.jl:215,293`),
   `F !== typeof(identity)` (`gemm.jl:606`), a Symbol kind `ABL !== :nostage` (`1649`). The plan
   forbids all of these by name (`recording-plan.md:26-28`, `1636-1640`; `api.md:9-21`). Cost:
   ≈ 180 lines of predicates that become candidate attributes and dispatch.

3. **`=== nothing` defaults resolved in the body.** 39 occurrences. `bias === nothing ? bias :
   pointer(bias)` is repeated in ten kernels (`gemm.jl:655,1078,1165,1233,1306,1440,1552,1643,1799`,
   `gemm_tensile.jl:204`) although `accinit` already dispatches on `Nothing` (`554-573`);
   `Union{Nothing,Int}` keywords with `x === nothing ? default : x` in `gemv.jl:152-153,161-162,
   209-211,219-221`, `fft.jl:356-357,371`; `forced === nothing || return …` (`408`);
   `partials === nothing ? splitscratch(…) : partials` (`2555-2556`, `2868`);
   `something(tiling, gemm_cm2_tiling(dev))` (`gemm_cm2.jl:347`). Cost ≈ 60 lines; the
   replacement is a method per case or a typed default.

4. **Globals: 12 `Dict{…,Any}` kernel tables and a `Ref` counter.** `GEMM_BLOCK_KERNELS`
   (`615`), `GEMM_STAGED_KERNELS` (`806`), `_V2_` (`833`), `_V2N_` (`872`), `_DB_` (`921`),
   `_PREFETCH_` (`959`), `_V4_` (`1003`), `_ABL_` (`1015`), `_GATHER_` (`1034`), `GEMV_KERNELS`
   (`gemv.jl:126`), `GEMVN_KERNELS` (`230`), `FFT_MIXED_KERNELS` (`fft.jl:570`); `SHARED_ID`
   (`gemv.jl:136`, mutated by `nextsharedid!`). Every lookup is `Dict{…,Any}` → type-unstable →
   `invokelatest`. Plus ≈ 70 scalar constants, among them device-specific numbers as
   process-wide facts: `GEMV_TARGET_THREADS = 1 << 16` from "40 CUs × 4 SIMDs" (`2741-2748`),
   `GEMV_NOSPLIT_ROWS` (`2735`), `GEMM_WORKGROUP = 64 # 2 subgroups` assuming 32 lanes
   (`50`, `516-527`), `fftgroup`'s "one warp = 32" (`fft.jl:343`), `coopmat_gemm_shape(…; cores = 48)`
   (`2356`), the 19 `SGEMM_*` constants (`2087-2145`). Owner in the target: the per-device
   selection table owns candidates and their measured order; a tiling is a type, so no table
   of functions exists; thresholds derive from `caps(dev)` (`api.md:256`); `SHARED_ID` is a
   KernelInterface/Lava defect (`gemv.jl:127-135`) to fix there. Cost ≈ 150 lines, and the
   plan's "no mutable globals" check (`recording-plan.md:1660`) fails on all 13.

5. **Feature flags for abandoned approaches, threaded through as keywords.**
   `GEMM_STAGED_DEFAULT` (`360`), `GEMM_DOUBLEBUF_DEFAULT` (`367`), `GEMM_VEC4_DEFAULT` (`371`),
   `GEMM_NARROW_DEFAULT` (`400`), `GEMM_TENSILE_DEFAULT` (`gemm_tensile.jl:57`), `FFT_SKEW`
   (`fft.jl:293`), `BROADCAST_FASTDIV_DEFAULT` (`indexing.jl:35`), `GEMM_TILING_SWEEP_DONE = true`
   (`297`, referenced nowhere; its 115-line docstring `182-296` is the only reason it exists);
   two orphan docstrings for `Ref`s that no longer exist (`299-306`, `308-312`, `808-831`).
   `coopmat_gemm_launches` takes 15 keywords (`2454-2490`); DNNKernels passes `bias`, `epilogue`,
   `tiling`, `nbatch`, `partials`, `reduce`, `blk_split`, `aload`, and `staged` twice (benchmarks);
   `vec2`, `narrow_ok`, `prefetch`, `doublebuf`, `vec4`, `tensile` are passed by nothing
   (`matmul.jl:98-106`: "no shipping path passes it"). `gemv_ncontig_launches` has `tm`/`block`/
   `unroll` overrides (`493-498`), `fft!` has `skew`/`group` (`356-357`). Cost ≈ 130 lines of
   flags, docstrings and keyword plumbing; the plan's alternative is a benchmark that passes a
   candidate, not a flag.

6. **Eager/declared twins over an `ArrayLaunch` indirection (a second implementation of every
   operation).** `launch.jl` (120) + `coopmat_gemm!`/`coopmat_gemm_dispatch!` (`2588-2609`),
   `gemmlaunch!`/`scalar_gemm_dispatch!` (`2908-2926`), `gemv!`/`gemv_dispatch!` (`472-491`,
   `526-542`), `fft!`…`fftany_dispatch!`/`rfft_dispatch!`/`irfft_dispatch!` (`800-889`). The
   declared forms re-validate sizes, re-plan and in `rfft` use a different algorithm (2.4). The
   `runlaunches!` "MIGRATION SHIM" branch (`launch.jl:51-67`) and the `group == 0` sentinel
   (`33-36`, `61-62`, `116`) exist only because of the split. Plan rule 6 and `recording-plan.md:
   1962-1964` delete the whole mechanism. Cost ≈ 230 lines.

7. **Forwarding layers.** `splitscratch(C, …) → splitscratch(backend, C, …)` (`2427-2436`) with a
   Vulkan override (`vulkan/array/gemm.jl:79-93`) that keeps a grow-only cache on `ctx.caches`
   because the pool could reclaim a live buffer (`2412-2425`), a memory-model bug worked around
   in an operation; `gemv_split(C,A,B,M,K) → gemv_split(M,K)` (`2760-2782`); `fftany! → fft!|fftmixed!`;
   `gemv → gemv!`, `fft → fft!`, `rfft → rfft!`; `tensile_gemm_launch` (`gemm_tensile.jl:262-264`).
   Cost ≈ 60 lines.

8. **Extents baked into the kernel type.** `Val(M), Val(N), Val(K)` on every coopmat kernel
   (`640-642`, `1064`, `2178-2180`, `gemm_tensile.jl:183`), `Val(K)` on `strided_gemm_kernel!`
   (`1935`), five `Val`s on `fft_kernel!` (`159-161`). `2158-2169` states the cost: "a SPIR-V
   module per `(M, N, K, FAST)`" and a precompile workload that must enumerate shapes. This
   conflicts with the plan's `launchrange(a)`/no-baked-length rule (`recording-plan.md:283-289`,
   `api.md:66`): an indirect launch cannot change a `Val`. Section 4 lists it as a gap the
   docs must decide; the code cost is the per-shape compile, not lines.

9. **Pointers and `llvmcall` in core.** `gemmload16` via `Core.Intrinsics.pointerref` and
   `Core.LLVMPtr` (`gemm_tensile.jl:107-109`), `stage16!` via `Base.llvmcall` shufflevector
   (`117-129`), `UInt64(pointer(B))` ×8 (`gemm_cm2.jl:179-219,281-297`), `pointer(bias)`/
   `pointer(A)` into `AcceleratedMatrix(ptr, off, ld)` (`gemm.jl:655,669-673,612`). The
   coopmat API is pointer-based (KernelInterface's), and `gemm_tensile.jl:37-39` gives a measured
   reason for byte offsets; still 16 unsafe sites in core where a `view`/strided device array
   argument is the Julian form. Cost: small in lines, large in what a reader must trust.

10. **Hand-rolled vector/matrix math.** `transform_point`/`transform_dir`/`inv_transform_dir`
    over `NTuple{12,Float32}` (`gjk.jl:31-69`, 39 lines) are `Mat{3,4,Float32} * Vec3f` and
    `transpose(Mat3f) * Vec3f`; `Vec3f(T_A[4], T_A[8], T_A[12])` column extraction (`gjk.jl:322-323`,
    `epa.jl:96-97`); `dot3` (`gjk.jl:106`); `quat_to_rot3x3(::Vec4f) -> NTuple{9}` (`transform.jl:70-80`)
    and `build_4x3`/`build_4x3_pervec` returning `NTuple{12}` (`88-106`) while `Mat3x4f` is an
    `SMatrix{4,3}` (`41`): two representations of one transform in one file;
    `mat4_to_vk_transform` is twelve `Float32(m[i,j])` (`129-133`). `fft.jl:238-246` is complex
    multiplication spelled on real pairs (deliberate, `72-77`). Cost ≈ 100 lines; violates the
    owner's Vec/Mat rule throughout `src/geometry/`.

11. **Manual loops that are `findfirst`/`argmin`/`sum`.** `find_closest_face` (`epa.jl:173-183`),
    reverse-edge search (`376-382`), dead-slot scan (`436-441`), partial-sum loops
    (`gemv.jl:216-218`, `gemm.jl:2401-2404`, `2826-2829`). Inside kernels these are the norm;
    cost ≈ 30 lines.

12. **Comments that narrate history and benchmarks instead of the contract.** 3553 of 6716
    lines; 112 lines carry a throughput/"measured" figure; full result tables at `gemm.jl:27-36,
    66-69, 122-126, 159-162, 191-197, 217-234, 259-263, 277-281, 379-382, 2616-2624, 2645-2653,
    2709-2715, 2741-2747`, `gemv.jl:46-53, 67-71, 79-83, 97-101`, `fft.jl:49-56, 278-281, 328-332`,
    `gemm_tensile.jl:26-30`, `gemm_cm2.jl:13-23`; "FIXED: … the paragraphs above are kept because
    they are the measurement" (`gemm.jl:1995-1998`, 52 lines on a bug that no longer exists);
    "the first version of this note missed that" (`745`); a docstring that inverts an earlier
    docstring (`242-247`); 12 dated entries. The tools cited exist (`JuliaVision/tools/gemm_ablate.jl`,
    `gemm_tiling_sweep.jl`, `gemm_cm2_ab.jl`, `gemm_cm2_why.jl`, `gemm_doublebuf_ab.jl`,
    `gemm_vs_cublas.jl`, `bench_gemm_padn.jl`, `bench_sam2.jl`); `mwe_tensor_gemm_nonsquare.jl`
    (`gemm_cm2.jl:69`) does not. The plan's comment rule (`recording-plan.md:1672-1681`) keeps
    history "only where it still constrains the code" and says a measurement names where it was
    measured: the tables belong in the tools' results. Cost ≈ 2000 lines (guess: ≈ 1500 of
    the 3553 state no contract).

13. **Functions over 100 lines.** `epa` `epa.jl:258-475` (218), `fftmixed_kernel` `fft.jl:572-710`
    (139), `coopmat_gemm_launches` `gemm.jl:2454-2575` (122), the prefetch kernel `1290-1406`
    (117), `fft_kernel!` `159-268` (110), the Tensile `let`/`@eval` block `gemm_tensile.jl:131-236`
    (106), `scalar_gemm_staged_kernel!` `2178-2280` (103). The kernels are long because the
    schedule is spelled out; `coopmat_gemm_launches` and `epa` are long because selection and
    polytope bookkeeping are inline.

14. **Keyword explosions.** 15 on `coopmat_gemm_launches` (`2454-2490`), 5 on
    `gemv_ncontig_launches`, 3 on `fft!`, `coopmat_gemm_cm2!`, `fftany_dispatch!`; `max_iters`/`eps`
    keywords on GPU-callable `gjk`/`epa` (`gjk.jl:319`, `epa.jl:261`).

15. **Code copied from upstream that could be a call.** The three `Diagonal` `mul!` methods
    (`vulkan/array/gemm.jl:124-191`) are GPUArrays' bodies; `strided_gemm_kernel!` (`1935-2029`)
    is `GPUArrays.generic_matmatmul!` with a widened accumulator (`1879-1898`), i.e. a 10-line
    upstream change; `hannwindow` (`fft.jl:1005-1018`) is `DSP.hanning`; `densify` (`2335-2339`)
    exists again as DNNKernels' own `densify` (`JuliaVision/DNNKernels/src/kernels/extern/attention.jl:302-325`);
    `launchgroup` (`indexing.jl:70-83`) restates KernelAbstractions' default group heuristic.
    The ported kernels (llama.cpp `mul_mm.comp`/`mul_mat_vec.comp`, VkFFT, rocBLAS Tensile,
    Bullet GJK/EPA) are legitimately "ours" under the plan's candidate list
    (`recording-plan.md:271-273`). Cost ≈ 150 lines.

16. **A consumer hook declared in the runtime for precompilation reasons.** `apair`
    (`indexing.jl:109-141`) is DNNKernels' convolution gather, declared in Mantle because defining
    it beside the Vulkan kernel "made `DNNKernels` fail to PRECOMPILE on any platform without
    Vulkan" (`131-138`). The gather kernels (`1737-1877`) exist for it. The plan names `conv!`
    (`recording-plan.md:2065`) and nothing about an operand loader.

## 4. What the design docs miss

Operations the code has and the docs name or do not:

| capability | code | consumer | `api.md` / plan |
|---|---|---|---|
| GEMM `C = αAB + βC` | `mul!` `vulkan/array/gemm.jl:48-77`; α, β only on the scalar path, coopmat requires `α==1, β==0` (`70`) | `mul!` from Makie/generic code | `mul!` listed (`api.md:72`); α/β not mentioned |
| fp16→fp32 and fp16→fp16 accumulate-in-fp32 | `accstore!` converts (`575-613`), `gemmaccum` (`1897-1898`) | every autocast model (`1883-1884`) | key has "element types" (`api.md:118`); no statement that accumulation type is a candidate attribute |
| bias + unary epilogue fused in the store | `accinit`/`accstore!` (`554-613`), `gemvn` (`gemv.jl:324-327`), Tensile (`gemm_tensile.jl:205-206,231-232`) | `bias =` 64 uses, `epilogue =` 18 uses in DNNKernels | only via the opt-in `FuseEpilogues` rewriter matching a candidate that "implements exactly this Julia function" (`recording-plan.md:505-511`). Today's kernels implement **any** singleton unary `f` and a row bias; the docs must say so or the fusion the models rely on is lost (rewriters are off by default) |
| batched GEMM (`nbatch`, contiguous planes) | block kernel `gemm.jl:647-663` | attention `attention.jl:799-801` (`nbatch = NB`) | absent |
| split-K with caller-owned partials and no reduce | `coopmat_gemm_shape` (`2356-2382`), `partials`/`reduce` (`2555-2583`) | conv (`conv_coopmat.jl:1102,1116`), `matmul.jl:455` | absent; "a library's workspace is a transient" (`recording-plan.md:282-284`) covers scratch, not the choice of split |
| strided batches | none (planes are `bat*M*K`, `661-663`) | — | absent |
| transposed/permuted operands | `gemmstrides` for the scalar path only (`2283-2325`); coopmat path refuses (`vulkan/array/gemm.jl:70`) | 27 of 29 MatAnyone matmuls fall to scalar for this reason (`1891-1895`) | `devicearg` resolves views (`api.md:339`), key includes "layout" (`118`); no candidate covers transposed fp16 |
| padding N to a block | `gemm_padn` (`500-514`) | DNNKernels pads in its planner (7 uses) | absent: who pads, who unpads (the `.unpad` dispatch in `emit.jl:4866-4871` is the consumer's) |
| gathered A operand (implicit im2col) | `apair` + gather kernels | conv | absent (`conv!` named, `recording-plan.md:2065`) |
| GEMV (three kernels) | section 2.3 | `gemv!` `matmul.jl:284`, `emit.jl:4643` | absent; a shape in the `mul!` key |
| quantized GEMV | DNNKernels' own `q8gemv!`/`kqgemv!` (`quant.jl:929`, `q4k.jl:485`) | — | absent |
| FFT | `fft!` out-of-place, power-of-two + mixed radix | `fftany!` 1, `fft!` 1 | `fft` listed (`api.md:72`); no in-place, no `dims`, no non-power-of-two statement |
| real FFT, inverse real FFT | `rfft!`, `rfft_dispatch!`, `irfft_dispatch!` | `rfft_dispatch!` 1, `irfft_dispatch!` 1 | absent |
| STFT, Hann window | `fft.jl:937-1018` | `Mantle.stft` 3, `hannwindow` 1 | absent; not a runtime's |
| collision (GJK/EPA/narrow phase) | `src/geometry/` 1092 lines | none | absent |

What "the core picks the implementation" needs that the code shows is missing:

- **Capability detection.** Today: `KernelInterface.DeviceCaps` (`KernelInterface/src/caps.jl:54-65`:
  `coopmat`, `tile`, `subgroup`, `coopmatsubgroup`, `sharedbudget`, `workgrouplimit`, `cores`,
  `warps`, `wggran`, `shapes`) read through `coopmatgemm(x)`, `workgrouplimit(x)`,
  `sharedbudget(x)`, `subgroupwidth(x)` (`src/memory/array.jl:90-157`), plus `coopmat_gemm_available`
  needing a 32-lane subgroup or pinning (`vulkan/array/gemm.jl:194-211`), `basealignment`
  (`src/Mantle.jl:513-535`), `KI.supports_tensor_addressing`, and the build-level
  `staged_gemm_tile()` hook (`backendhooks.jl:31-50`). `api.md:256` lists `caps(dev)` fields as
  "the device's limits … throughput per element type … bandwidth"; it does not name the
  cooperative-matrix shape table, subgroup width, shared-memory budget, workgroup limit,
  workgroup-scope granularities or the subgroup-pinning fact, all of which every candidate's
  `divides`/`fits` predicate reads. Either `DeviceCaps` is kept and named, or those fields are
  added.
- **Tile choice.** Today: first-divides-wins over a hand-ordered list (`gemm_tiling` `402-413`)
  with one hand-coded veto (`gemm_aliasing` `355`, "suspected mechanism … not confirmed",
  `344-348`), a fill heuristic (`coopmat_gemm_shape` `2356-2382`: `4*cores` subgroups), and
  a waste bound (`staged_gemm_ok` `2689-2696`). The plan says "measurement fills the table per
  device; an unmeasured entry uses core's cost estimate" (`recording-plan.md:278-280`) and lists
  the selection table's locking and "when measurement runs" as open (`api.md:551`). Missing: the
  cost estimate's inputs (`caps(dev).throughput[T]`, `bandwidth`, `api.md:256` give a roofline,
  but a tiling choice needs occupancy: registers and shared bytes per candidate, which
  `KernelInfo` has, `api.md:306`), where measurements persist (per process? per device file?),
  and what triggers one.
- **Shapes as compile-time constants.** The fastest kernels take `Val(M), Val(N), Val(K)`
  (pattern 8; `2158-2169` measured 2.7 → 4.9 TFLOP/s from folding them). The plan's indirect
  `launchrange(a)` after a `resize!` (`api.md:66`) and "no length is baked into a node"
  (`recording-plan.md:283-289`) cannot feed a `Val`. The docs must state that a GEMM candidate is
  specialised on its shape (the key includes "shapes", `api.md:118`, which implies it) and that a
  resize of an operand recompiles that node, or require shape-agnostic kernels and accept the
  measured cost.
- **How the choice is tested.** Today: `test/test_gemm_staged.jl` compares kernels at ten K per
  tiling (`gemm.jl:135-138`), `test_gemm_staged_scalar.jl` pins the staged/per-element boundary
  (`2633-2634`), `test_gemm_tensile.jl`, `test_gemm_cm2.jl`, `test_gemv.jl`, `test_fft.jl`. No test
  asserts *which* candidate a shape selects on a given `caps`, although `DeviceCaps(c; kw...)`
  exists for exactly that (`caps.jl:84-96`). The plan's green list for phase 6 (`recording-plan.md:
  1968-1969`) names no selection test.
- **Consumer coupling.** DNNKernels references 40 names from these files (counts in the grep:
  `GEMM_TILE` 93, `launchgroup` 37, `splitidx` 35, `GEMM_TILINGS` 23, `gemm_tiling` 21, `cart32` 21,
  `FastDiv32` 20, `coopmat_gemm!` 17, `accstore!` 13, `broadcastextents` 12, `planewise_worth`
  reading `SGEMM_*` 11, `coopmat_gemm_shape` 9, `CONV_GATHER_TILINGS` 9, `staged_gemm_tile` 8,
  `gemm_padn` 7, …). Phase 11 (`recording-plan.md:2064-2066`) says DNNKernels' GEMM becomes `mul!`;
  it does not say what replaces the index helpers (`indexing.jl`, a KernelInterface concern) or
  the planner's reads of the tiling table.
- **Metal.** `mul!` on Metal is MPS (`src/metal/ka.jl:146`, `160-235`) and core's kernels are
  declined by `dev.tile == 16` (`src/metal/metal.jl:102-106`): the second implementation rule 6
  forbids, and the plan's Metal section lists MLX kernels first (`recording-plan.md:1577-1586`).
  The fp16 staged kernel is written for a 16×16×16 subgroup tile and 32 lanes; the docs do not
  say which candidate covers `simdgroup_matrix` 8×8.

## 5. Minimal form

Estimate for the same functionality in the target design, my files only (guesses, ±30%):

| piece | lines | basis |
|---|---|---|
| one staged cooperative-matrix GEMM kernel, parameters `Tiling{STM,STN,WM,WN,BK,PAD}`, staging width `W`, index type `I`, schedule `Prefetch|Plain`, loader `L`, bias/epilogue args | 150 | the prefetch kernel is 117 (`gemm.jl:1290-1406`); parameters add ≈ 30 |
| Tensile-structure kernel as a second candidate (byte-offset addressing, 2 LDS buffers) — or the generic kernel at `Tiling{4,4,2,2,16,24}` with `W = 8` and a `DoubleBuffer` schedule (`gemm_tensile.jl:13-21` describes exactly that structure; whether ACO keeps the addressing win without the hand-formed offsets, `37-39`, is untested) | 110 | `gemm_tensile.jl:93-236` minus prose |
| register-block kernel for batch/split-K, or batch and split as launch dimensions of the staged kernel | 0-60 | `617-700` |
| fp32 staged kernel | 90 | `2178-2280` |
| per-element fallback with accumulator widening (or an upstream GPUArrays change) | 40 | `2000-2029` |
| one GEMV kernel over strides with bias/epilogue; split-K as a launch dimension | 60 | `gemv.jl:297-330` |
| `mul!` method + candidate values + initial table rows + divisibility/fill/waste predicates + padding | 120 | `402-514`, `2356-2382`, `2689-2696`, `vulkan/array/gemm.jl:48-77` |
| `accinit`/`accstore!` | 25 | `554-613` |
| index helpers (`FastDiv32`, `cart32`, `splitidx`) | 60 | `indexing.jl:8-68` |
| FFT: power-of-two kernel 100, mixed-radix generator 120, plan 40, real/inverse-real 80, one `fft!`/`rfft!`/`irfft!` method each 40 | 380 | `fft.jl` |
| contract docstrings | 250 | |
| TLAS instance transform conversion (in the records kernel) | 20 | `transform.jl:41-56,121-134` |
| GJK/EPA/narrow phase | 0 in Mantle | moves to the physics package (Raycore has `collision.jl`) with `Mat3x4` math; ≈ 700 lines there |
| **total** | **≈ 1300-1400** | against 6716 today |

Concrete structure for the GEMM family:

```julia
# candidates are values; the device's table holds an ordered list per (function, key)
struct CoopmatGemm{Tiling,W,I,Schedule,Loader} end         # one kernel, gemm_staged!
struct ScalarGemm{Tiling} end                                # scalar_gemm_staged_kernel!
struct ElementGemm end                                       # per-element, any Number
implements(::CoopmatGemm) = (bias = true, epilogue = Any, αβ = false, batch = true, splitk = true)
divides(c::CoopmatGemm{T}, M, N, K, align) = M % bm(T) == 0 && N % bn(T) == 0 && K % bk(T) == 0 && align >= 16*W
cost(c, dev, M, N, K) = …                                    # from caps(dev) + KernelInfo (registers, shared)

function LinearAlgebra.mul!(C::MantleArray, A::AbstractVecOrMat, B::AbstractVecOrMat, α, β)
    loc = location(C, A, B)                                  # device or graph (api 2.4)
    key = (mul!, eltype.((C, A, B)), size(C)…, strides via devicearg, basealignment.((A, B)))
    cand = select(device(loc), key)                          # table row, else cost estimate
    launch!(cand, loc, C, A, B, α, β)                        # dispatch!(loc, gemm_staged!, (…), ndrange; group)
end
```

Batched and split-K products are the same method with a 3-D `C` and a `splitk` the candidate
chooses; `bias`/`epilogue` arrive through `FuseEpilogues` or, if the docs keep the keyword,
through one `mul!(C, A, B, α, β; bias, epilogue)` whose candidates declare `implements`.
No tiling table of functions, no flags, no eager/declared pair, no `nothing`-returning
probes, no `@eval`, no `invokelatest`.

## Owner-rule violations, by rule

- `_name` functions: none.
- Globals: 12 `Dict{…,Any}` tables and one `Ref` (pattern 4); ≈ 70 scalar `const`s, several
  encoding one device (`gemm.jl:50,2735,2748,2752,2356`, `fft.jl:343`). State lives in no
  context object; the plan's allowlist check fails.
- `try`/`catch` without rethrow: none.
- Name variants instead of dispatch: `coopmat_gemm!`/`coopmat_gemm_dispatch!`/`gemmlaunch!`/
  `scalar_gemm_dispatch!`/`gemv!`/`gemv_dispatch!`/`fft!`/`fftany!`/`fftmixed!`/`fftany_dispatch!`/
  `rfft!`/`rfft_dispatch!` (section 2.2, 2.4); seven kernel families named by suffix (`_v2`,
  `_v2n`, `_db`, `_v4`, `_abl`, `_prefetch`, `_gather`) instead of one kernel with parameters.
- GeometryBasics Vec/Mat: violated throughout `src/geometry/` (pattern 10).
- Pointers/unsafe: 16 sites (pattern 9).
- Defensive code: 39 `=== nothing`, `nothing`-returning probes, sentinel results
  (`NO_CONTACT` by `depth == 0f0`, `narrow_phase.jl:42-53`; `idx <= 0` "polytope corrupted"
  paths, `epa.jl:318-323,462-464`; an unused `gjk_simplex` argument kept "for forward
  compatibility", `epa.jl:263-268`).
