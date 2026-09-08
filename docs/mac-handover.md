# Handover: bringing the backend-independence refactor up on Metal

For an agent starting from an older copy of `VulkanDev` on a Mac. Everything
below was done on Linux against an NVIDIA RTX 4000 Ada (2026-09-07/08). The
Metal side has not been run since the refactor; that is what this session is for.

## 1. Get the code

`VulkanDev` itself is not a git repository; the packages under `dev/` are, and
the root `Project.toml`/`Manifest.toml` point at them by relative path
(`dev/Mantle`, `dev/Hikari`, `dev/Lava`, `dev/Makie/RayMakie`, `dev/Raycore`,
…). Julia 1.12.7. Fetch and check out these branches, in place:

| repo | branch | commit |
|---|---|---|
| `dev/Mantle` | `sd/lava-refactor` | 1954417 |
| `dev/Hikari` | `sd/lava-refactor` | b8332d2 |
| `dev/Makie` (holds RayMakie) | `sd/lava-refactor` | e491a30 |
| `dev/Lava` | `sd/lava-refactor` | cd0df70 |
| `dev/Raycore` | `sd/with-texture` | d737d8f (unchanged) |

    for r in dev/Mantle dev/Hikari dev/Makie dev/Lava; do git -C $r fetch origin && git -C $r checkout sd/lava-refactor && git -C $r pull --ff-only; done

Do not touch the root env with `Pkg`; the Manifest already resolves these
paths. Run Julia from the `VulkanDev` directory with that env, through the
`bt_julia_eval` tool if it is available (persistent session, Revise picks up
edits; restart only for struct changes or when an extension's import list
changed).

## 2. What changed, in one paragraph

Mantle's core now owns every sequence: `record!`, `run!`, `beforeframe!`,
`execute!`, the pass walk, the fused prepare of device-sized dispatches, host
stores and address patches, the Barriers phase, the submission sweep, trace
pinning, argument-memory layout. A backend implements only the names in
`Mantle.BACKEND_VOCABULARY` (`src/graph/backend.jl`); the Vulkan extension is
checked against it by `test/vulkan/test_backend_vocabulary.jl`. Plans are
recorded once, at build (`Mantle.record!(Mantle.Plan(g))`), and `run!` never
records; a windowed plan cannot be recorded and is walked per frame. Per-run
values are `GPURef`s, never `Ref`s. The full story, step by step with what each
step deleted and what verified it, is `docs/backend-independence.md`; the
earlier submission refactor is `docs/submission-refactor.md`.

Metal today: `openrecording` answers `nothing`, so a Metal plan is walked per
run through the core `Immediate` path — every dispatch is a Metal.jl kernel
launch made by the compile phase, and the recording primitives
(`emitkernel!`, `storebytes!`, `emitinline!`, `indirectslot`,
`closerecording!`) are never reached. Metal answers 52 vocabulary names and
inherits core methods for 37; the 33 Vulkan-only names are recording
primitives, batch-queue graphics verbs RayMakie's overlay/window path uses
(Metal opted out through `supports_batch_queue`), the RT-pipeline entry
points Metal does not claim, coopmat, and Vulkan's sync lowering. Two Metal
methods were written blind and never run: `abandonframe!` (drops the
drawable, `src/metal/window.jl`) and `syncbackend` (`src/metal/ka.jl`).

## 3. Run these, in this order, and fix what they say

1. `using Mantle, Metal` — the extension must load. Then
   `include("dev/Mantle/test/test_ext_imports_are_declared.jl")`: it reads
   both extensions' source and checks every `import Mantle: name` names
   something Mantle declares. A name that fails here is a step-7 or step-8
   rename I could not see from Linux.
2. `include("dev/Mantle/test/runtests.jl")` once. The `Metal backend`
   testset runs the nine files in `test/metal/` only when Metal is
   functional; the Vulkan section is skipped without a Vulkan loader.
3. Hikari: `test/test_plan_invalidation.jl`, `test/test_sample_is_one_run.jl`,
   `test/test_volpath_graph.jl`. First recorded-plan renders end to end. On
   Metal they run interpreted, so "one submission per sample" may not hold as
   written on that backend — if so, the assertion needs a backend-aware
   spelling in Mantle's vocabulary, not a skip.
4. RayMakie: `test/test_render_allocates_nothing.jl`, then its
   `runtests.jl`. The graphics files are skipped on a backend without a batch
   queue, by design.
5. Only then a real scene: `RayDemo/benchmark/run_benchmarks.jl` with
   `backend = Mantle.defaultbackend()`.

## 4. Rules that were paid for

- **No vendor-conditional code, no vendor-conditional tests.** A driver
  difference is fixed in the emitter or the walk for everyone. Example from
  this refactor: `vkCmdTraceRaysIndirect` is not subject to conditional
  rendering, so the fused prepare of a `repeat!` iteration is emitted before
  the predicate scope on every backend (`emitpass!` in
  `src/graph/kalaunch.jl`).
- **The vocabulary is the line.** A backend defines only vocabulary names,
  never shadows a core name, and imports every core name it calls
  (`import Mantle: name` in the extension). Converting an existing local
  function into an import needs a Julia restart.
- **Never allocate GPU objects per run**, never free from high-level code;
  Mantle pins and retires on the timeline. `test_run_allocates_nothing.jl`
  pins zero bytes per run on Vulkan; the Metal equivalent is worth writing.
- **Every bug gets a regression test that fails before the fix.** Run it
  before and after.
- **Never dismiss a failure as pre-existing without an MWE**, and never hand
  over a fix you have not run.
- Never commit or push without asking; short English messages; no
  Co-Authored-By.

## 5. Diagnosis methods that found the bugs the layers missed

- **Sentinel and read back.** A device that dies on the first render of a
  fresh session but not after another scene has run is reading memory nothing
  wrote. Fill the suspect region (indirect slots: `plan.args.indirect[k]`)
  with a sentinel before the run and read it back after; the slot still
  holding the sentinel is the one nobody wrote (`test_discarded_iteration_prepare.jl`).
- **Save both images and look.** Two renders with identical pixel counts
  after a scene edit were identical images: the rebuilt plan traced the old
  TLAS snapshot because `invalidate!` kept the cached adaptation
  (`test_plan_invalidation.jl`). Counts hide this; images do not.
- **Check identities, not values**: `vp.adapted.accel === tlas.static_tlas`,
  `plans.sample.recording === before`, `Mantle.outstanding(bq)` tags.
- **A second queue.** Nothing in Mantle's suite drives two queues at once;
  RayMakie's `test_overlay_compositing.jl` does, and it is what found the
  sweep asking the wrong timeline. Run it after any queue change.
- Validation layers (core, sync) saw none of the above; GPU-assisted
  validation crashed inside `vkQueueSubmit2` on the failing workload.

## 6. Open items, none of them Metal's

- Lava codegen tests failing on NVIDIA only: `test_static_workgroup.jl`,
  `test_int32_cartesian_miscompile.jl`, `test_shared_index_division.jl`.
- `AcceleratedKernels`, `DNNKernels` and `JET` are not in the root env, so a
  handful of Mantle and Hikari test files cannot load.
- Two Hikari pbrt references were rendered at 512 spp against a 256 spp run.
- A `compute!` dispatch naming an external `LavaArray` neither pins nor stamps
  it at submit; only the trace pass pins. Vulkan-side lifetime gap.
- Two lifetime mechanisms for "return memory once the device is done": the
  pool's retire/reclaim and the Vulkan backend's finalizer-driven deferred
  frees (`drain!`). Folding the second into the first would delete both
  drains and the last-write stamping only they read.
