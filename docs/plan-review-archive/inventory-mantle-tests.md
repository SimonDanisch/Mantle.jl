I've inventoried the existing tests. Almost all of them are written against the API the redesign deletes, so they can serve as scenario references but not be carried over. The reusable parts are the fake devices, the backend probing, the source-scan checks and the hang guard. Nothing exists yet for graph mutation, host nodes, hostwritten stores, eviction, "any thread" or minimized windows. I only read and grepped; nothing was edited or run.

I read Mantle at `5c2d4c6` and Lava at `5ab8755`, both on `sd/lava-refactor`. The new docs (api.md, recording-plan.md, internals.jl, resizing-and-raytracing.jl, refactor-ledger.md) are untracked in git. Paths below are relative to `/sim/Programmieren/VideoEdit/dev/Mantle`.

## 1. Test files by area

| Area | Files | Testsets | Device | Broken / skipped |
|---|---|---|---|---|
| Placement and sync lowering, inline in `test/runtests.jl:81-452` | — | ~25 | placement on CPU (needs the minimalloc checkout); sync needs Vulkan | `@test_broken` gaps placement (`:225`) |
| CPU only: `test_pool`, `test_pipeline_stages`, `test_lowering`, `test_access` (type half) | 4 | 19 / 12 / 8 / 2 | none | — |
| Source scans: `test_mantle_owns_it`, `test_core_names_no_backend` | 2 | 10 | none | 5 `@test_broken` guards that are meant to flip as phases land |
| Portable graph tests, run once per backend: `compile_golden`, `devicerange`, `run_ordering`, `declared_kernel`, `kernel_tails`, `declared_call`, `argument_packing` | 7 | 19 | GPU | — |
| Graphics, skip themselves on missing capability: `isubd_mesh`, `mesh_pipeline_graph`, `discard`, `headless_rebindable_draw`, `texture_upload`, `many_stage_args`, `tuple_varyings` | 7 | ~20 | GPU | capability skips |
| `test_arena_recording` | 1 | 15 | Vulkan only | — |
| Windows: `test_window` (2122 lines), `test_window_portable` | 2 | 44 | GPU and display, run in a child process | — |
| Metal (`test/metal/`) | 12 | 110 | Metal | — |
| ROCm (`test/rocm/test_rocm.jl`) | 1 | 20 | AMDGPU | — |
| Vulkan (`test/vulkan/`): compiler/codegen, GEMM/coopmat, HWTLAS/ray query, recording, holds, pool, caches, two devices | 204 | ~800 | Vulkan; 2 test second-device placement, need lavapipe | 35 files never included (below); `int32_cartesian` 3 broken, `shared_index_division` 2 |
| Lava (`dev/Lava/test`): SPIR-V emission (27 files under `spirv/`), IR passes, compile overhead | 37 | — | mostly no device; `run_lavapipe.sh` | — |

- **Never included:** 24 `mwe_*` repro scripts, plus `test_ci`, `test_coopmat_components`, `test_geometry_types`, `test_hwtlas_refit_stress`, `test_kernelinterface`, `test_math_intrinsics`, `test_rayquery_device_probe`, `test_real_kernels`, `test_target_features_push`, `test_workgroup_struct_accesschain`. Six `mwe_*` loops are included (`runtests.jl:1480-1502`).
- **Probably stale (my guess, not run):** `runtests.jl:590-598` still runs three files with `KernelAbstractions.CPU()` as the backend, but `5c2d4c6` deleted the host backend and `test_host.jl`.

## 2. Reusable infrastructure

- **Fake devices, `test/test_pool.jl`:**
  - `FakeDev` (`:21`) records every `rawalloc` size.
  - `BitDev` (`:411`) merges constraints as bitmasks.
  - `NilDev` (`:447`) and `PickyDev` (`:460`) cover the trivial and strict compatibility answers.
  - `TimelineDev` (`:484`) has a settable `fence`/`passed` timeline with an `asked` counter, which fits tests of tokens and the "memory returns" invariant.
  - `retire_from_finalizer!` (`:536`) is a finalizer-race fixture.
  - These need updating: `bufferusage` and `fence` are on the delete list, and the redesign's `passed(ch, value)` takes a channel.
- **Backend gating:**
  - `test/backend_probe.jl:61` `backend_loadable` loads a package once, quietly, and memoizes the result.
  - `_VULKAN_OK` (`runtests.jl:18`) and `_METAL_OK` (`:29`) decide availability before anything runs.
  - `foreachbackend` (`:526`) runs a file once per backend in a fresh module, through `Main.MANTLE_TEST_BACKEND`.
  - Everything sits in one outer `@testset` (`:56`) so a failure does not stop the run.
- **Hang guard:** the window files run in a child process with a 600 s deadline and get SIGKILL on timeout (`runtests.jl:630-675`). The child is started with `julia --project=$(active_project)`, a deliberate exception to "one persistent session". Smaller patterns: a polled `deadline` (`vulkan/test_argument_memory_isolation.jl:79`), `verify_gpu_av(timeout=30)` (`vulkan/test_gpuav_clean.jl:109`), and a subprocess exit-code check (`vulkan/test_twodevice_shutdown.jl`).
- **Global state hygiene:** `FROZEN_*` is restored in a `finally` (`runtests.jl:1427-1435`). There is an ordering trap: `test_arena_recording` must be the first file that builds a plan, because it counts tenants of a cached shared arena (`runtests.jl:548-552`).
- **Fixtures:**
  - `test_compile_golden.jl` has `buildprobe` (`:47`), `compileresult` (`:64`), `compilectx` (`:172`) and pinned compile output.
  - `test_arena_recording.jl` has `chainplan` (`:27`) and `batchrecorded` (`:236`).
  - `test_window.jl` has `advect_plan` (`:233`), a random-DAG corpus (`:710`, `:734`) and the RPS schedule benchmark (`:794`).
  - `test_kernel_tails.jl:29-34` uses padded sentinels to catch out-of-bounds writes.
  - `test_run_ordering.jl:51` is two plans with no wait between them, checking every element on both sides. Its header records a manual negative control (the ordering call deleted).
- **Source and AST scanners** (starting points for checks P0.T16–T19): `test_mantle_owns_it.jl:26-98` (`codeonly`, `sourcefiles`, `definednames`) plus `DRIVER_TOKENS` (`:277`), and `test_core_names_no_backend.jl:30` (`free_names`).
- **Second device:** lavapipe beside the real GPU via `Mantle.device("lavapipe")` (`vulkan/test_device_identity.jl:27-34`, `vulkan/twodevice_probe.jl`).
- **Validation:** `DebugConfig(validation=true)` in 9 files. `bench/validate.jl` runs sync validation but is kept out of the suite (about 212 ms per frame).
- **SPIR-V helpers:** `dev/Lava/test/spirv_test_utils.jl` is included by Mantle Vulkan tests.
- **How suites run:** `include(".../test/runtests.jl")` through bt_julia_eval in the main env; never `Pkg.test`, which forces `--check-bounds=yes`. The output overflows, so capture it to a file and grep.

## 3. Benchmarks

- None of the 15 files in `bench/` uses `tools/measure.jl`:
  - **Per-frame `time()`, median and p95:** `baseline_lava.jl:47-62`, `chain.jl:80-104`, `targets.jl:81-104`.
  - **Own interleaved, warmed `compare`:** `dnn.jl:134-140`, `independent.jl:13,63`.
  - **`time_ns` plus `waitidle`, 20 warm-up runs, latency and throughput:** `rocm_vs_vulkan.jl:120-129`.
  - **fps counter:** `two_scatters.jl:129-139`.
  - **Correctness, not timing:** `fuzz.jl`, `stress.jl`, `validate.jl`.
  - None gates on clock plateau or clock dips.
- **`tools/measure.jl`:**
  - `plateau` (`:316`) waits until the clock settles under load.
  - `bench` (`:514`) gates samples on a 0.90 clock floor, records GC time, reports median and spread.
  - `compare` (`:593`/`:615`) interleaves the arms and returns `trustworthy`.
  - `ratio` (`:430`) refuses Results from different processes.
  - Telemetry comes from nvidia-smi or amdgpu sysfs (`:86-147`).
- **Copies:** none under `experiments/*/tools`; the only tool there is `plan-review/tools/count.py`, the ledger's word-bounded name counter.

## 4. Coverage holes against the new API

| Area | Today | Missing |
|---|---|---|
| compile / record / run | 39 files, all through `Plan`/`record!`/`run!(plan)` | `run!(g)` compiling on first call; "recorded once" (`run_allocates_nothing` is the closest) |
| Graph mutation | none (`delete!` appears only in HWTLAS files) | everything: `insert!`/`delete!(g,h)`, hold release, P1.A17 |
| Render passes | `test_window.jl` (load ops, depth, several targets, viewport, indexed, GPU draw count), Metal render graph | eager `render!(dev,…)` |
| `resize!` | arrays (`recorded_move_patch`, `partitioned_recording`); window `test_window.jl:1041-1207` | queued semantics; recompile beyond capacity |
| Stores | `test_window.jl:1668` | `hostwritten`, `x[:]=data`, `DimensionMismatch` |
| GPURef | 10 files, value box only | array box rebind `r[]=b` (today's `rebind!` is deleted) |
| TLAS/BLAS | about 40 Vulkan files, all through `build_accel!`/`refit_tlas!`/HWTLAS | TLAS as a graph node, capacity, count changes |
| Windows | Vulkan in a child process, portable file | minimized, OUT_OF_DATE, SUBOPTIMAL, image return after a throw |
| Multi-queue | `crossqueue_sync`, `sweep_per_queue`, all via `batchqueue` | core-chosen queue count |
| Graphs sharing arrays | `run_ordering`, `arena_recording:51,459` | per-segment sync states, `ordering = Unordered()` negative control |
| Eviction / pressure | `pool_alloc_oom`, `pool_trim`, `no_pool_trim_while_recorded` | eviction, `OutOfDeviceMemory` after the pressure steps |
| Threads | `phase4_singlethread` asserts `WrongThread`, the opposite of "any thread" | `test_any_thread`, interleaved recording |
| Eager calls | ~100 files of eager `LavaArray` work | `test_one_path`, pending changes run in front |
| Host nodes | none | everything |
| Error paths | `@test_throws` in 40 files | api.md section 7 corner cases |
| Device loss | only `mwe_*` loops | device usable after a throw; recovery |

## 5. Uses of deleted names per file

**How counted:** 270 names (Appendix A "goes" rows plus identifiers from ledger delete items). Comment lines are stripped and count.py's word-bounded match is used. I excluded partial deletions (`Device(::KA.Backend)`, `reclaim!(::SubmitChannel)`, `caps`, `drain!`, `release!`, `acquire!`), field names, and `use`/`draw!`/`blit!`.

**Caveat:** `alignment` and `workgroupsize` are inflated by keyword arguments and KA's own `workgroupsize`.

**"Verbs" column:** my own derivation, not from Appendix A or the ledger. It counts lines that call `Plan(`, `record!(`, `update!(` or `waitfor!(`, which api.md section 2 drops.

| File | Hits | Distinct | Verbs | Top names |
|---|---|---|---|---|
| test_window | 125 | 10 | 35 | Buffer 60, Transient 45, Surface 8 |
| test_mantle_owns_it | 58 | 51 | 0 | BACKEND_VOCABULARY 6 |
| test_arena_recording | 39 | 8 | 12 | Buffer 27, Transient 6 |
| test_pool | 28 | 10 | 2 | DeviceArray 6, rawfree 5, download 4 |
| runtests | 24 | 11 | 0 | alignment 6, eachbackend 4, needs_transition 3 |
| test_declared_call / test_declared_kernel / test_devicerange | 18 / 17 / 18 | 4–5 | 5 / 7 / 3 | Buffer, Transient, DeviceRange |
| test_argument_packing / test_texture_upload / test_kernel_tails / test_compile_golden | 16 / 14 / 13 / 11 | 3–7 | ≤4 | passedas, devicearray, Buffer |
| Other 15 top-level files | ≤10 each | — | ≤5 | defaultbackend, todevice, Transient |
| metal/render_graph · graphics · device · record | 87 / 62 / 63 / 35 | — | 18 / 1 / 16 / 28 | Transient, Framebuffer, LegacyQueue, Buffer |
| other 8 metal files | 2–23 | — | ≤5 | MetalAPI, build_accel!, Metal.synchronize |
| rocm/test_rocm | 67 | 14 | 28 | Buffer 24, ROCmAPI 19 |
| vulkan: math_intrinsics · atomics_and_dispatch · spirv_pattern · struct_broadcast | 84 / 77 / 66 / 66 | 3–5 | 0 | LavaArray, flush!, defaultbackend |
| vulkan: repeat · caching_and_allocations · struct_alignment · ci · hold_lifetime | 41 / 39 / 37 / 35 / 31 | — | 11 / 0 / 0 / 0 / 0 | Buffer, flush!, batchqueue, oneshot! |
| vulkan: 175 files in total | 1672 | — | 89 | 13 files with 0 hits |
| vulkan/mwe_* (29 files) | 94 | — | 0 | LavaArray, LavaBackend |
| bench/ · examples/ · Lava/test | 231 / 13 / 20 | — | 30 / 0 / 0 | Buffer 90, Transient 44 / devicearray / FROZEN_* |

**Totals across `test/`:** 2616 hits of 157 distinct names. The biggest are `LavaArray` 598, `Buffer` 355, `defaultbackend` 205, `flush!` 185, `LavaBackend` 151, `Transient` 134, `batchqueue` 88, `VulkanAPI` 64 and `build_accel!` 58.
