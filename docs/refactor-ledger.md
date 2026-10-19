# Refactor ledger

Every add and delete item of every phase of `recording-plan.md`, and every
item found while working, with its status and evidence. Every session starts
by reading the plan and this ledger.

- **Status:** `open`, `doing`, `done`, `blocked`, `n/a`. `done` needs evidence:
  the test that proves it and the commit. `blocked` needs the blocker with
  evidence (the forbidden-moves alternative: stop and report). `n/a` needs the
  reason.
- **Baseline:** occurrences at Mantle 5c2d4c6 in `src/`, `ext/` and `test/`,
  counted with `git grep -c -E '(^|[^A-Za-z0-9_])NAME($|[^A-Za-z0-9_!])'`.
  Counts include comments and tests that pin the old behaviour; a delete item
  is done when the name is gone from the device types that joined, except
  where the plan keeps it for a later phase (noted on the item).
- **Ids** are stable: `P<phase>.<kind><n>`, kinds A add, D delete, R rename,
  W rewrite a test, G green, T test, B bug, C comments, F found while working.
  New items are appended to their phase with the next free number. An item
  that moves to another phase or merges into another keeps its row, with
  status `n/a` and the id it went to; the new row says "(was …)".
- **Marks:** eviction and readers are ruled (decisions.md B3.9, B3.10) and
  carry those ids; "(not in the plan: confirm)"
  for an item the plan does not list; "S<n>" for a fix of the review round
  of 2026-10-02 (decisions.md, "Review round 2026-10-02", and the plan text
  that carries it).
- **A phase is done** when every item is `done` or `n/a`, its green tests
  pass on every joined device type, one real use ran (the editor opened and
  played, a RayMakie frame, a model call), an independent review of its diff
  against the plan, `api.md` and this ledger found no forbidden move (or its
  findings are fixed), and the baseline re-measured interleaved shows no
  regression beyond noise. These gates are the last items of every phase.

## Phase 0. Docs, comments, tests, bugs

### Docs

| id | item | baseline at 5c2d4c6 | status | evidence |
|---|---|---|---|---|
| P0.A1 | `docs/recording-plan.md` consistent with the rulings of 2026-09-30 and 2026-10-02 | written | doing | decisions.md B4.1-B4.10 (2026-10-14) folded in; the contradiction review of 2026-10-14 read all six docs against decisions.md, its open questions remain |
| P0.A2 | `docs/internals.jl`, `docs/examples.jl`, `docs/resizing-and-raytracing.jl` consistent with the plan | written | doing | brought in line with decisions.md B4 after the contradiction review of 2026-10-14; the questions that review raised remain |
| P0.A3 | `docs/api.md`: every type and verb with its contract, ownership, invariants, corner cases, driver entry points, allowlists, vocabulary fates | written 2026-10-02 | doing | decisions.md B4.1-B4.10 (2026-10-14) folded in; the contradiction review of 2026-10-14 read all six docs against decisions.md, its open questions remain |
| P0.A4 | `docs/refactor-ledger.md` (this file) | written 2026-10-02 | open | |
| P0.D1 | delete `docs/design.md`, `docs/submission-refactor.md`, `docs/backend-independence.md`, `docs/mantle-owns-it.md`, `docs/dispatch-api.md`; repoint the 9 references in `src/` and `test/` to the plan; move the two contents that only they hold (`vulkan/runtime/command.jl:404`'s measurement, `test/runtests.jl:73`'s clone note) into a comment or the plan (not in the plan: confirm) | 5 files tracked | open | |
| P0.D2 | move `docs/recording-plan-review.md` to `experiments/plan-review/` (not in the plan: confirm) | untracked | open | |
| P0.A5 | the docs committed (Simon's approval) | untracked | done | 034ae50 (the plan, api.md, internals, the suite) |

### Comments

Every comment and docstring checked against the code and the plan (plan,
"Comment rules"); comments only, no code change in the same commit.

| id | item | baseline at 5c2d4c6 | status | evidence |
|---|---|---|---|---|
| P0.C1 | comments in src/Mantle.jl and src/phases.jl | - | open | |
| P0.C2 | comments in src/graph | - | open | |
| P0.C3 | comments in src/memory | - | open | |
| P0.C4 | comments in src/sync | - | open | |
| P0.C5 | comments in src/array | - | open | |
| P0.C6 | comments in src/runtime | - | open | |
| P0.C7 | comments in src/graphics | - | open | |
| P0.C8 | comments in src/raytracing | - | open | |
| P0.C9 | comments in src/vulkan | - | open | |
| P0.C10 | comments in src/metal | - | open | |
| P0.C11 | comments in ext | - | open | |
| P0.C12 | comments in test | - | open | |

### Tests, red where the code does not satisfy them yet

| id | item | baseline at 5c2d4c6 | status | evidence |
|---|---|---|---|---|
| P0.T1 | write `test_interleaved_recording.jl` as the plan describes it (Phase 0) | - | open | |
| P0.T2 | write `test_builder_is_a_value.jl` as the plan describes it (Phase 0) | - | open | |
| P0.T3 | write `test_record_is_pure.jl` as the plan describes it (Phase 0) | - | open | |
| P0.T4 | write `test_recorded_once.jl` as the plan describes it (Phase 0) | - | open | |
| P0.T5 | write `test_window_resize.jl` as the plan describes it (Phase 0): beyond the capacity the graph recompiles once, at its next run (S1; it no longer throws) | - | open | |
| P0.T6 | write `test_run_is_submit.jl` as the plan describes it (Phase 0) | - | open | |
| P0.T7 | write `test_resize_is_queued.jl` as the plan describes it (Phase 0) | - | open | |
| P0.T8 | write `test_device_is_an_argument.jl` as the plan describes it (Phase 0) | - | open | |
| P0.T9 | write `test_any_thread.jl` as the plan describes it (Phase 0) | - | open | |
| P0.T10 | write `test_lifetime.jl` as the plan describes it (Phase 0) | - | open | |
| P0.T11 | write `test_crossplan_order.jl (with its negative control)` as the plan describes it (Phase 0) | - | open | |
| P0.T12 | write `test_memory_returns.jl` as the plan describes it (Phase 0) | - | open | |
| P0.T13 | write `test_host_node.jl` as the plan describes it (Phase 0) | - | open | |
| P0.T14 | write `test_one_path.jl` as the plan describes it (Phase 0) | - | open | |
| P0.T15 | write `test_hostwritten.jl` as the plan describes it (Phase 0) | - | n/a | went to P0.T28: hostwritten graph arrays are gone (decisions.md B4.9) |
| P0.T16 | check: vocabulary frozen to `api.md` section 4, over `src/vulkan`, `src/metal`, `ext/`, with arity (replaces `test/vulkan/test_backend_vocabulary.jl`); starts `@test_broken` with the violations listed | - | open | |
| P0.T17 | check: driver calls in one place (`api.md` section 8); starts `@test_broken` with the violations listed | - | open | |
| P0.T18 | check: no mutable globals outside `api.md` section 9; starts `@test_broken` with the violations listed | - | open | |
| P0.T19 | check: no probes outside `api.md` section 9; starts `@test_broken` with the violations listed | - | open | |
| P0.T20 | check: Aqua and JET on core; starts `@test_broken` with the violations listed | - | open | |
| P0.T21 | check: two devices open for the whole suite; starts `@test_broken` with the violations listed | - | open | |

#### Added 2026-10-02 (review round 5)

Tests the plan's phase 0 list names since the review round.

| id | item | baseline at 5c2d4c6 | status | evidence |
|---|---|---|---|---|
| P0.T22 | write `test_runs_never_overlap.jl` as the plan describes it (Phase 0; rule 9), with its negative control | - | open | |
| P0.T23 | write `test_run_host_time.jl` as the plan describes it (Phase 0): 100, 2 500, 25 000 and 50 000 named records, array `GPURef`s included, against the phase 3 target (open decision 19) | - | open | |
| P0.T24 | write `test_eviction.jl` as the plan describes it (Phase 0) (B3.9); was P2.T1 | - | open | |
| P0.T25 | write `test_composite_fuzz.jl` as the plan describes it (Phase 0); was P5.T1 | - | open | |
| P0.T26 | write RayMakie's compile-counter session as the plan describes it (Phase 0); green as P11.G2 | - | open | |
| P0.T27 | check: launch kernels never read cells (only head, prepare, the TLAS offsets and records kernels and `lengthof` writers have a `CellRead` use), over every plan the suite compiles (api.md section 6, derived; not in the plan: confirm, G16) | - | open | |
| P0.T28 | write `test_gpuref_slot.jl` (was P0.T15): `r[] = v` before the first run reaches it; a store during a run reaches the next submission that touches the slot; two plans and a piece baking one slot read the stored value (B4.9) | - | open | |
| P0.T29 | write `test_view_barriers.jl` as the plan describes it (Phase 0): a write through an array and a read through a view, `reshape` or storage of it in one graph get a barrier; negative control: an object-keyed resource table (today's `Graph.ids`) | - | open | found porting GPUFiltering and Paperhawk, 2026-10-15: on Mantle 94c4258 a `Buffer`, `storage(b)` and a reshape are three resources and the plan has no barrier between the write and the read |

### Bugs to fix first, each with a regression test that fails before the fix

| id | item | baseline at 5c2d4c6 | status | evidence |
|---|---|---|---|---|
| P0.B1 | `flush_stall_report` reads fields core's `Submission` does not have (`vulkan/runtime/command.jl`) | locations at 90e582f in the plan | open | |
| P0.B2 | `gemm_cm2!`/`gemm_cm2_sg!` called as kernel factories (`vulkan/array/gemm_cm2.jl`) | locations at 90e582f in the plan | open | |
| P0.B3 | a conditional region can drop a barrier (`graph/build.jl` docstring, `sync/transition.jl` rule) | locations at 90e582f in the plan | open | |
| P0.B4 | Metal waits call `Metal.synchronize()` (task's queue, not the device's): `waitfor`/`waitidle`, `upload_texture_data!`, `readback_framebuffer`, `metal/hwtlas.jl`, `metal/record.jl` | locations at 90e582f in the plan | open | |
| P0.B5 | walked Metal plans keep compile-time arrays with `patchable = true` and no move listener (`metal/record.jl`) | locations at 90e582f in the plan | open | |
| P0.B6 | readback and copy leave images in TRANSFER_SRC before a render pass that loads them (`vulkan/graphics/`) | locations at 90e582f in the plan | open | |
| P0.B7 | validation checks read `VK_CONTEXT_REF[]` inside functions that hold a context | locations at 90e582f in the plan | open | |
| P0.B8 | arena regrowth dereferences tenant weak references without a `nothing` check (`memory/pool.jl`) | locations at 90e582f in the plan | open | |
| P0.B9 | `append!` on a `LavaArray` view ignores the view's offset (`vulkan/array/gpuarrays.jl`) | locations at 90e582f in the plan | open | |
| P0.B10 | unlocked `live_buffers` delete from a finalizer | locations at 90e582f in the plan | open | |
| P0.B11 | H.265 bitstream alignment | locations at 90e582f in the plan | open | |
| P0.B12 | the two disagreeing texture format tables | locations at 90e582f in the plan | open | |
| P0.B13 | `Ref` without `GC.@preserve` and UserID vs `instance_id` in `metal/trace.jl` | locations at 90e582f in the plan | open | |
| P0.B14 | device features enabled without querying them | locations at 90e582f in the plan | open | |
| P0.B15 | regression test for `destroy_pool!` calling `empty!` on an `Inbox` (fixed in 2c7dc14) | locations at 90e582f in the plan | open | |

### Baseline

| id | item | baseline at 5c2d4c6 | status | evidence |
|---|---|---|---|---|
| P0.A6 | whole-plan benchmark (`experiments/header_indirection/plan_*.txt` shape) on the joined devices | - | open | |
| P0.A7 | each suite's model calls timed (JuliaVision runners, Hikari, RayMakie, VideoEditor) | - | open | |
| P0.A8 | the new-API suite `test/suite/` (cases from experiments/plan-review/catalogue-core.md and catalogue-resize-rt.md, each with its catalogue id and ledger phase; later phases skipped) and its test hooks in Mantle: `diagnostics`/`resetdiagnostics!` (plan, pipeline and kernel compiles, records, submissions per channel, allocations, host waits), `reserved`, `mapped`, `liveuploads`, `livecells`, `pendingof`, `steplog!`/`steplog`, `checklocks!`; `Window` and `Swapchain` parameterized by their device's type and `inner(x)` (so the suite's `FaultDevice` intercepts window verbs); `caps(dev).graphics`; `FaultDevice` in the suite | - | open | |
| P0.A9 | before/after benchmarks through the consumers' unchanged APIs (`/sim/Programmieren/VideoEdit/benchmarks/`: Makie on GLMakie, RayMakie raster and traced; SAM 2, QwenImage, TRELLIS; VideoEditor; RayDemo `run_benchmarks.jl`); baseline environment `VideoEdit-baseline` kept on each worker; the comparison alternates both environments in fresh processes, with an A/A pair as the noise floor | - | open | |

| id | item | baseline at 5c2d4c6 | status | evidence |
|---|---|---|---|---|
| P0.G90 | gate: the phase's green tests pass on every joined device type | - | open | |
| P0.G91 | gate: one real use ran (editor opened and played, a RayMakie frame, a model call) | - | open | |
| P0.G92 | gate: independent review of the phase's diff against plan, api.md, ledger: no forbidden move | - | open | |
| P0.G93 | gate: baseline re-measured interleaved, no regression beyond noise | - | open | |

## Phase 1. One compile path

| id | item | baseline at 5c2d4c6 | status | evidence |
|---|---|---|---|---|
| P1.A1 | `Kernel`/`DeviceKernel`/`KernelInfo`: one `Mantle.Kernel` per kernel at its `CodeInstance` (`GPUCompiler.cached_results`), filled in stages under its own lock: access at declaration, code at compile, per device the native handle and `KernelInfo` (B5.11) | - | open | |
| P1.A2 | `compilekernel(dev, kernel, config)`/`kernelinfo` on every joined device type, filling the `Kernel`; no device-type kernel cache (B5.11) | - | open | |
| P1.A3 | the per-device compile counter | - | open | |
| P1.A4 | node types with their origin | - | open | |
| P1.A5 | `compile!(s, node)` | - | open | |
| P1.A6 | the dependency graph kept and reduced, one hazard function shared with barriers | - | open | |
| P1.A7 | barrier merging for conditional kernels | - | open | |
| P1.A8 | `when!(g, flag) do … end` on a device-side flag (Simon, 2026-10-02), as count × flag (S13): the launches in the block are indirect, and whoever writes their indirect commands multiplies the count by the flag (the head for a host-written flag, a prepare node after the kernel for a kernel-written one; both nodes are phase 4's, P4.A3), like `repeat!`'s loop head; no conditional rendering, the same on Metal; the flag is never a handle dependence | - | n/a | went to P1.A18: one launch per `when!`, `Bool` flag (decisions.md B4.1, decisions.md B4.2) |
| P1.A9 | `repeat!` bodies limited to launches, hazards derived between iterations | - | n/a | went to P1.A19: launch groups as values (decisions.md B4.2) |
| P1.A10 | the estimated submission split (`costs!`, `submissions!`; the budget is core's, from `caps(dev)` facts: P1.A15) | - | open | |
| P1.A11 | placement stored in the plan | - | open | |
| P1.A12 | `sharing!` | - | open | |
| P1.A13 | the per-segment access summary | - | open | |
| P1.R1 | the compile phases become step functions on a `CompileState`; `run!(phase, ctx)`, the phase marker types and `PHASES` go | 3 hits of `PHASES` | open | |

| id | item | baseline at 5c2d4c6 | status | evidence |
|---|---|---|---|---|
| P1.D1 | delete `compile_dispatch` | 32 hits, 10 files | open | |
| P1.D2 | delete `buildskernel` | 16 hits, 8 files | open | |
| P1.D3 | delete `kisupported` | 9 hits, 5 files | open | |
| P1.D4 | delete `get_or_build_iter_plan` | 7 hits, 4 files | open | |
| P1.D5 | delete the host-read `when!`: `hostcond` (`Pass.hostcond`) and `RecordingParts.conds`; the name `when!` is reused (Simon, 2026-10-02). Includes the former P1.D19 | 8 hits of `hostcond`, 3 files; 33 hits of `when!` (name reused) | open | |
| P1.D6 | delete `batchconds` (`RecordingParts.batchconds`, the host-read `when!`'s batching). Includes the former P1.D19 | 4 hits, 1 files | open | |
| P1.D7 | delete `measure!` | 4 hits, 2 files | open | |
| P1.D8 | delete `measuring` | 21 hits, 11 files | open | |
| P1.D9 | delete `partitionranges` | 16 hits, 6 files | open | |
| P1.D10 | delete `submissionbatches` | 5 hits, 2 files | open | |
| P1.D11 | delete `recordparts!` | 9 hits, 4 files | open | |
| P1.D12 | delete `recordplan!` | 6 hits, 4 files | open | |
| P1.D13 | delete `submitrecording!` | 28 hits, 6 files | open | |
| P1.D14 | delete `passcost` | 16 hits, 5 files | open | |
| P1.D15 | delete `stamped` | 31 hits, 16 files | open | |
| P1.D16 | delete `emitloophead!` | 10 hits, 4 files | open | |
| P1.D17 | delete `Pass.kind::Symbol` | - | open | |
| P1.D18 | delete the `Plan` fields `partition`, `stale` (as the re-record trigger) and `profiler` as the split's input | - | open | |
| P1.D19 | (merged into P1.D5 and P1.D6, review round 5) | - | n/a | merged |
| P1.D20 | delete the default query pool on every Vulkan plan and `timestamps` as a device hook for the split | - | open | |

| id | item | baseline at 5c2d4c6 | status | evidence |
|---|---|---|---|---|
| P1.W1 | rewrite `test/vulkan/test_partitioned_recording.jl` | exists | open | |
| P1.W2 | rewrite `test/vulkan/test_run_allocates_nothing.jl` | exists | open | |
| P1.W3 | rewrite `test/vulkan/test_when.jl` | exists | open | |
| P1.G1 | green: `test_record_is_pure.jl` for compiles | - | open | |
| P1.G2 | green: `test_recorded_once.jl` (SAM 2 encoder and decoder recorded once each; today twice, `census_radv.txt`) | - | open | |

### Added 2026-10-02 (review round 5)

| id | item | baseline at 5c2d4c6 | status | evidence |
|---|---|---|---|---|
| P1.A14 | `caps(dev)` facts for `costs!` and `schedule!`: throughput per element type, memory bandwidth, transfer queue presence, cross-queue cost; the kernel compile configuration `config(dev, f)` (api.md 4.1, G3) | - | open | |
| P1.A15 | the submission budget as core's function of `caps(dev)` facts (job timeout, whether the device drives a display); `submissionbudget` is no longer a device-type hook (S15) | 8 hits of `submissionbudget`, 5 files | open | |
| P1.A16 | `submissions(g)`: segments and cuts, including those host nodes cause (plan, "The graph and its plan") | n/a (no definition at 5c2d4c6; the word occurs 49 times in prose) | open | |
| P1.D21 | delete `supportspredicate` (no conditional execution: S13) | 13 hits, 8 files | open | |
| P1.D22 | delete conditional rendering (`vkCmdBeginConditionalRenderingEXT`, today for `repeat!`'s per-iteration flags, `vulkan/graph.jl:158, 1327`; the plan: "Mantle uses it for nothing") | 2 hits of `vkCmdBeginConditionalRenderingEXT`, 2 files; 1 of `cmd_begin_conditional_rendering_ext` | open | |
| P1.D23 | delete `callgroup` (launch geometry is one core calculation from `KernelInfo`; api.md Appendix A) | 21 hits, 10 files | open | |
| P1.D24 | delete `materialize!` (transients placed by `place!`; api.md Appendix A) | 16 hits, 9 files | open | |
| P1.D25 | delete `passbarriers` (`barriers!` is core's; api.md Appendix A) | 11 hits, 8 files | open | |
| P1.A17 | graphs stay across structure changes (B3.16): declaring into a compiled graph and `delete!(g, h)` of an `insert!(g) do … end` handle (names open) mark the plan; `run!(g)` recompiles from the graph's nodes, retires the old plan, drops the holds of resources no node names after its last run; declarations take the graph lock (nested ones reuse it); holds counted per top-level node; test: add and remove a node block on a running graph, the graph's device arrays keep their contents, no hold leaks | - | open | |
| P1.A18 | `when!(g or group, flag, kernel, args, ndrange)` (was P1.A8): one launch; `flag` a `GPURef{Bool}` or one-element `Bool` array, enforced by the method's argument types; indirect, count × flag (B4.1, B4.2) | - | open | |
| P1.A19 | `Group(graph, name)` (was P1.A9): a value, declared into with `dispatch!(group, …)` anywhere, attached with `repeat!(group, n; while_nonzero)`; launches only, by which methods exist for `Group`; not a location; no mode state on the graph; hazards between iterations derived from the group's accesses (B4.2, B4.4) | - | open | |
| P1.A20 | `Plan.compiles`: how often the plan was compiled, beside `stale`, for debugging and tests (B4.9); proposed with it: a count of re-records | - | open | |
| P1.A21 | a group size the caller gives (KA's `kernel(backend, workgroupsize)`, `dispatch!(…; group)`) is kept as given; `config(dev, f)` picks one only where none is given (plan, Kernels; the 2-D default is F32) | - | open | GPUFiltering launches 2-D image kernels with (16, 16) because the default for a 2-D range is one row of 64 |
| P1.A22 | access per argument stored in the `Kernel` at declaration (B5.10, B5.11); `AccessCache`/`accesscache` go | 2 (`accesscache` in api.md Appendix A) | open | |
| P1.D26 | delete `workgroupsize` as a vocabulary hook (`KernelInfo`'s group size; api.md Appendix A) | 110 hits (KA's `workgroupsize` included), 31 files | open | |
| P1.D27 | delete the `do`-block body forms of `repeat!` and `when!` (`repeat!(f, g, n…)`, `when!(f, g, …)`); `repeat!(g, kernel, args, ndrange; max, while_nonzero)` stays (B4.2) | 40 hits of `repeat!(…) do`/`repeat!(g…`, 11 files at f6e359a | open | |

| id | item | baseline at 5c2d4c6 | status | evidence |
|---|---|---|---|---|
| P1.G90 | gate: the phase's green tests pass on every joined device type | - | open | |
| P1.G91 | gate: one real use ran (editor opened and played, a RayMakie frame, a model call) | - | open | |
| P1.G92 | gate: independent review of the phase's diff against plan, api.md, ledger: no forbidden move | - | open | |
| P1.G93 | gate: baseline re-measured interleaved, no regression beyond noise | - | open | |

## Phase 2. Memory

| id | item | baseline at 5c2d4c6 | status | evidence |
|---|---|---|---|---|
| P2.A1 | sync states per shared resource | - | open | |
| P2.A2 | pool kinds for every allocation | - | open | |
| P2.A3 | `placeimage`/`placeaccel` | - | open | |
| P2.A4 | counted driver objects (verbs: G10) | - | open | |
| P2.A5 | the release engine: last-use tokens, release at every `acquire!` and `run!` | - | open | |
| P2.A6 | holds and `free!` | - | open | |
| P2.A7 | graph finalizers | - | open | |
| P2.A8 | tenancy by id | - | open | |
| P2.A9 | sparse arenas with queued mapping (Vulkan; `bindpages!`/`unbindpages!` through `sparsesubmit!`) | - | open | |
| P2.A10 | the pressure policy in `acquire!` with `makeroom!` folded in: unmap idle arena pages, trim empty blocks, collect garbage, wait for retiring storage, then throw `OutOfDeviceMemory` (S18; the eviction step is P4.A13). Pressure never unmaps pages beyond an array's size and never compacts acceleration-structure storage (S2) | - | open | |
| P2.A11 | one meaning of capacity (`budget(dev)`) | - | open | |

| id | item | baseline at 5c2d4c6 | status | evidence |
|---|---|---|---|---|
| P2.D1 | delete `vk_alloc` | 20 hits, 9 files | open | |
| P2.D2 | delete `MemoryPolicy` | 11 hits, 3 files | open | |
| P2.D3 | delete `reclaim!(::SubmitChannel)` only; `reclaim!(p::Pool, dev)` (`memory/pool.jl:765`) is the pool's release step | 87 hits of `reclaim!` (all methods), 18 files | open | |
| P2.D4 | delete `gemm_split_scratch` | 5 hits, 3 files | open | |
| P2.D5 | delete `gemm_split_retired` | 2 hits, 2 files | open | |
| P2.D6 | delete `reduce_scratch` | 9 hits, 4 files | open | |
| P2.D7 | delete `remap!` | 15 hits, 7 files | open | |
| P2.D8 | delete `remappable` | 9 hits, 5 files | open | |
| P2.D9 | delete `movable` | 9 hits, 6 files | open | |
| P2.D10 | delete `headroom` | 6 hits, 3 files | open | |
| P2.D11 | delete `arenaroom` | 9 hits, 4 files | open | |
| P2.D12 | delete `makeroom!` | 3 hits, 2 files | open | |
| P2.D13 | delete the channel's `pending`/`retiring`/`taken` and `retire!(::SubmitChannel)` | - | open | |
| P2.D14 | delete `rawfree` for acceleration structures as a separate path | - | open | |
| P2.D15 | delete `Plan.replaced` | - | open | |

| id | item | baseline at 5c2d4c6 | status | evidence |
|---|---|---|---|---|
| P2.G1 | green: `test_lifetime.jl` | - | open | |
| P2.G2 | green: `test_memory_returns.jl` | - | open | |

### Added 2026-10-02 (graphics, eviction, composites)

| id | item | baseline at 5c2d4c6 | status | evidence |
|---|---|---|---|---|
| P2.A12 | (moved to P4.A13: eviction needs phase 4's dependents and pending changes) | - | n/a | moved |
| P2.A13 | (moved to P5.A12: composites come with the TLAS and BLAS) | - | n/a | moved |
| P2.A14 | tables with free lists (material, medium, light, texture) | - | open | |
| P2.A15 | (removed: pressure never compacts acceleration-structure storage, S2; the plan's phase 2 no longer lists it) | - | n/a | contradicts S2 |
| P2.T1 | (moved: written as P0.T24, green as P4.G3) | - | n/a | moved |

### Added 2026-10-02 (review round 5)

| id | item | baseline at 5c2d4c6 | status | evidence |
|---|---|---|---|---|
| P2.A16 | the last hold under the array's record lock (S23): storage swapped for `Released()`, the old storage retired with the last tokens; dropping pending changes, the eviction list and the cell went to P4.A35 (they come in phase 4) | n/a (`Released` 0 hits) | open | |
| P2.R1 | the vocabulary's `capacity(dev)` (the device budget hook, `vulkan/graph.jl:107`; Metal's `recommendedMaxWorkingSetSize`; `AMDGPU.free()`) becomes `budget(dev)` (api.md Appendix A) | 92 hits of `capacity` (prose and `capacity = n` included), 32 files | open | |
| P2.R2 | `maxalloc(dev)` becomes a `caps(dev)` field, the largest single allocation (api.md 4.1, G3) | 22 hits, 10 files | open | |
| P2.D16 | delete `alignment` as a vocabulary hook (part of `rawalloc`'s constraint; api.md Appendix A) | 100 hits (the word in prose included), 38 files | open | |
| P2.D17 | delete `bufferusage` (the memory kind's usage constraint; api.md Appendix A) | 15 hits, 9 files | open | |
| P2.D18 | delete `extrausage` (as `bufferusage`) | 8 hits, 5 files | open | |
| P2.D19 | delete `imageusage` (the `Images` kind's usage constraint) | 9 hits, 5 files | open | |

| id | item | baseline at 5c2d4c6 | status | evidence |
|---|---|---|---|---|
| P2.G90 | gate: the phase's green tests pass on every joined device type | - | open | |
| P2.G91 | gate: one real use ran (editor opened and played, a RayMakie frame, a model call) | - | open | |
| P2.G92 | gate: independent review of the phase's diff against plan, api.md, ledger: no forbidden move | - | open | |
| P2.G93 | gate: baseline re-measured interleaved, no regression beyond noise | - | open | |

## Phase 3. Channels, recording and ordering

| id | item | baseline at 5c2d4c6 | status | evidence |
|---|---|---|---|---|
| P3.A1 | core `SubmitChannel`/`Recording`/`Submission` | - | open | |
| P3.A2 | `Builder` and the emit verbs on core's walk (signatures: api.md 4.5) | - | open | |
| P3.A3 | queue kinds and core's queue policy | - | open | |
| P3.A4 | tokens per segment and channel, claim and submit in one step | - | open | |
| P3.A5 | waits derived for the whole run before its accesses are recorded | - | open | |
| P3.A6 | the lock order with `finally` | - | open | |
| P3.A7 | GC-safe host waits | - | open | |
| P3.A8 | the graph lock as a non-reentrant `Base.Semaphore(1)` | - | open | |
| P3.A9 | host nodes with `hostcuts!` and host-visible placement of their arguments | - | open | |
| P3.A10 | arena parts held by a run until its last stage is submitted (`holdparts!`, Simon 2026-10-02) | - | open | |
| P3.A11 | one head per stage | - | open | |

| id | item | baseline at 5c2d4c6 | status | evidence |
|---|---|---|---|---|
| P3.D1 | delete `Closed` | 41 hits, 12 files | open | |
| P3.D2 | delete `OneShot` and the Vulkan `OneShot`/`Recording` structs (core's `Recording`/`Submission` replace them). Includes the former P3.D51 | 48 hits of `OneShot`, 13 files | open | |
| P3.D3 | delete `ownthread` | 15 hits, 5 files | open | |
| P3.D4 | delete `WrongThread` | 8 hits, 3 files | open | |
| P3.D5 | delete `openoneshot` | 5 hits, 3 files | open | |
| P3.D6 | delete `oneshot` | 22 hits, 10 files | open | |
| P3.D7 | delete `oneshot!` | 38 hits, 21 files | open | |
| P3.D8 | delete `handover!` | 14 hits, 8 files | open | |
| P3.D9 | delete `sealed!` | 5 hits, 2 files | open | |
| P3.D10 | delete `maywait` | 5 hits, 2 files | open | |
| P3.D11 | delete `submissionholds!` | 6 hits, 3 files | open | |
| P3.D12 | delete `hold!` | 81 hits, 24 files | open | |
| P3.D13 | delete `holdleaves!` | 60 hits, 17 files | open | |
| P3.D14 | delete `submitlist!` | 4 hits, 2 files | open | |
| P3.D15 | delete `Stamp` | 19 hits, 9 files | open | |
| P3.D16 | delete `stampof` | 36 hits, 16 files | open | |
| P3.D17 | delete `claim!` | 2 hits, 1 files | open | |
| P3.D18 | delete `unclaim!` | 2 hits, 1 files | open | |
| P3.D19 | delete `crosswaits!` | 16 hits, 5 files | open | |
| P3.D20 | delete `stamp!` | 10 hits, 4 files | open | |
| P3.D21 | delete `syncbuf!` | 17 hits, 8 files | open | |
| P3.D22 | delete `indirectbarrier!` | 3 hits, 3 files | open | |
| P3.D23 | delete `openrecording` | 32 hits, 9 files | open | |
| P3.D24 | delete `Outstanding` | 14 hits, 6 files | open | |
| P3.D25 | delete `takeholds!` | 9 hits, 3 files | open | |
| P3.D26 | delete `emptyframe!` | 2 hits, 1 files | open | |
| P3.D27 | delete `unhold!` | 11 hits, 4 files | open | |
| P3.D28 | delete `pushholds!` | 2 hits, 1 files | open | |
| P3.D29 | delete `dropholds!` | 2 hits, 1 files | open | |
| P3.D30 | delete `Emitter` (the Vulkan one; core's `Builder` replaces it). Includes the former P3.D51 | 83 hits, 13 files | open | |
| P3.D31 | delete `emit_dispatch!` | 13 hits, 5 files | open | |
| P3.D32 | delete `emit_dispatch_indirect!` | 5 hits, 3 files | open | |
| P3.D33 | delete `emit_barrier!` | 8 hits, 2 files | open | |
| P3.D34 | delete `emit_draw!` | 6 hits, 1 files | open | |
| P3.D35 | delete `emit_trace!` | 8 hits, 5 files | open | |
| P3.D36 | delete `emit_trace_indirect!` | 4 hits, 3 files | open | |
| P3.D37 | delete `emithead!` | 16 hits, 6 files | open | |
| P3.D38 | delete `emitkernel!` | 17 hits, 8 files | open | |
| P3.D39 | delete `emitinline!` | 9 hits, 4 files | open | |
| P3.D40 | delete `emitbarriers!` | 13 hits, 8 files | open | |
| P3.D41 | delete `emitupdate!` | 11 hits, 4 files | open | |
| P3.D42 | delete `withpredicate` | 11 hits, 6 files | open | |
| P3.D43 | delete `openrun` (Metal's uses stay until phase 9) | 24 hits, 8 files | open | |
| P3.D44 | delete `closerun!` (Metal's uses stay until phase 9) | 23 hits, 8 files | open | |
| P3.D45 | delete `submitrun!` (Metal's uses stay until phase 9) | 8 hits, 3 files | open | |
| P3.D46 | delete `batchqueue` (Metal's uses stay until phase 9) | 107 hits, 47 files | open | |
| P3.D47 | delete `allocate_batch_queue!` (Metal's uses stay until phase 9) | 31 hits, 17 files | open | |
| P3.D48 | delete `supports_batch_queue` (Metal's uses stay until phase 9) | 15 hits, 9 files | open | |
| P3.D49 | delete `release_batch_queue!` (Metal's uses stay until phase 9) | 26 hits, 9 files | open | |
| P3.D50 | delete the channel's hold stack (`SubmitChannel.holds`), `spare`, `thread` | - | open | |
| P3.D51 | (merged into P3.D2 and P3.D30, review round 5) | - | n/a | merged |
| P3.D52 | delete `LavaBackend`'s upload queue and `external.jl`'s direct use of `compute_queue` | 7 hits of `compute_queue` | open | |
| P3.D53 | delete the head barrier | - | open | |
| P3.D54 | delete `drain!(::SubmitChannel)` (core's `drain!(pool.inbox)` stays, internals.jl) and `flush!` as user-facing verbs; `acquire!(::SubmitChannel)`, `release!(::SubmitChannel, r)`, `waitfor!(::Stamp)` | 60 hits of `drain!` (all methods), 16 files | open | |

| id | item | baseline at 5c2d4c6 | status | evidence |
|---|---|---|---|---|
| P3.W1 | rewrite `test/vulkan/test_hold_lifetime.jl` | exists | open | |
| P3.W2 | rewrite `test/vulkan/test_batch_queue_lifetime.jl` | exists | open | |
| P3.W3 | rewrite `test/vulkan/test_closed_command_buffers.jl` | exists | open | |
| P3.W4 | rewrite `test/vulkan/test_hold_trace.jl` | exists | open | |
| P3.W5 | rewrite `test/vulkan/test_holdleaves_stops_at_tlas.jl` | exists | open | |
| P3.G1 | green: `test_interleaved_recording.jl`, `test_builder_is_a_value.jl`, `test_run_is_submit.jl` (headless), `test_crossplan_order.jl`, `test_any_thread.jl`, `test_host_node.jl` without its resize part | - | open | |

### Added 2026-10-02 (review round 5)

| id | item | baseline at 5c2d4c6 | status | evidence |
|---|---|---|---|---|
| P3.A12 | host waits only under the graph lock (S6): no sync-state, channel, pool or window lock is held while the host waits; the graph lock may be (shared arena parts, a host node's inputs, the swapchain acquire, `acquire!`'s pressure steps); no store waits for a run, value `GPURef`s included (B4.9) | - | open | |
| P3.A13 | segments as lists of parts: recordings written once and `BuildRecording`s (S8; plan phase 3) | - | open | |
| P3.A14 | `lockgraph(g)` throws when the calling task holds an arena part of `g`'s plan; `holdparts!` throws when the task would wait for a part while holding one with a higher id (S25) | - | open | |
| P3.A15 | `Token = (channel, value)`; core's `passed(::Token)`/`waitfor(::Token)` over the device-type verbs `passed(ch, value)`/`waitfor(ch, value)`; `fence(ch)` becomes a read of `SubmitChannel.next` (api.md 4.2) | 112 hits of `fence` (the word in prose included), 30 files | open | |
| P3.A16 | measure the real `run!` against the model of "Measurements this plan rests on" (target: within 3x; open decision 19) | - | open | |
| P3.G2 | green: `test_runs_never_overlap.jl`; `test_run_host_time.jl`: the same host time per run for every number of named resources (open decision 19, ruled B3.15; was P11.G3) | - | open | |
| P3.D55 | delete the queue-slot fields (`QueueSlots` and `VulkanQueue.slots`, `vulkan/runtime/device.jl:154, 200`; plan, Queues) | 4 hits of `QueueSlots`, 1 file | open | |
| P3.D56 | delete `RecordingParts.batches` (plan, Core types) | 21 hits of `RecordingParts`, 9 files | open | |
| P3.D57 | delete `abandonframe!`, `abandonrecording!`, `abandonrun!` (the recording lifecycle is `Builder`/`finish` and the free list; api.md Appendix A) | 9 + 9 + 12 hits, 5 + 5 + 4 files | open | |
| P3.D58 | delete `makerecording`, `finishrecording!`, `resetrecording!`, `destroyrecording!`, `recorder` (`Builder(dev, ch)`, `finish(b)`, the free list, retirement by token; api.md Appendix A) | 6 + 12 + 7 + 5 + 41 hits (`recorder` includes prose), 4 + 4 + 4 + 4 + 15 files | open | |
| P3.D59 | delete `awaitwrites` (eager calls wait per sync state; api.md Appendix A) | 15 hits, 6 files | open | |
| P3.D60 | delete `emitpreparebarrier!` (prepare is a node; api.md Appendix A) | 7 hits, 4 files | open | |

| id | item | baseline at 5c2d4c6 | status | evidence |
|---|---|---|---|---|

### Added 2026-10-05 (rulings B3.13, B3.14)

| id | item | baseline at 5c2d4c6 | status | evidence |
|---|---|---|---|---|
| P3.A20 | `AccessRecord` → `SyncState`: per shared resource only the accesses of what is not a run (eager calls, host reads, vendor views, eviction copies, sparse binding, applied change steps), the last tokens of retired plans, pending changes, lent/freed (B3.14) | - | open | |
| P3.A21 | between graphs through the resource (B3.15): `namers` per `SyncState` (plans with their use, replaced at `register!`/`unregister!`; composite members reach them through readers); a stage writing a shared resource waits for the other namers' complete last runs and fills the readers' inboxes; a stage waits for the writers' last runs of its inbox resources; waits kept for the run's later stages; `register!` seeds the plan's inbox, only re-locks (never recompiles) when plans change meanwhile; `unregister!` adds its last-run tokens to `retired` (composite members included); `delete!`/rebind/box release drop the reader after the applying submission passed | - | open | |
| P3.A22 | inbox per plan: pending a change or recording a non-run access pushes the resource into the inboxes of the plans naming it (and naming composites built from it); a stage locks only its inbox resources (expanded to the members their pending builds read), its arena parts (stage 1), and the sync locks of itself, the plans sharing what it writes and the plans naming its inbox resources; no per-resource walk at run (B3.14, B3.15) | - | open | |
| P3.A23 | plan sync lock in the lock order (after the window lock, before sync states); eager calls, evictions and storage release take the sync locks of the plans naming what they touch (`lockresources`) | - | open | |
| P3.A24 | "old or new, never mixed" holds per STAGE: a change arriving between a run's stages goes into the next submission (B4.9) | - | open | |
| P3.G90 | gate: the phase's green tests pass on every joined device type | - | open | |
| P3.G91 | gate: one real use ran (editor opened and played, a RayMakie frame, a model call) | - | open | |
| P3.G92 | gate: independent review of the phase's diff against plan, api.md, ledger: no forbidden move | - | open | |
| P3.G93 | gate: baseline re-measured interleaved, no regression beyond noise | - | open | |

## Phase 4. Arguments and sizes

| id | item | baseline at 5c2d4c6 | status | evidence |
|---|---|---|---|---|
| P4.A1 | Lava's entry wrapper: fields in the argument slots, block marked `NonWritable` | - | open | |
| P4.A2 | the cell arena; each array's cell, shared by its views | - | open | |
| P4.A3 | head and prepare nodes with the plan's slots as a resource | - | open | |
| P4.A4 | one entry per plan, host writes wait for the previous run | - | n/a | went to P4.A33: no per-plan entry, value `GPURef`s are slots (decisions.md B4.9) |
| P4.A5 | `GPURef` with its lock | - | n/a | went to P4.A33: a value `GPURef` is a slot with a sync state, no lock of its own (decisions.md B4.9); the array box's sync state is P6's |
| P4.A6 | hostwritten data through `x[:] = data`, kept on the host while no current plan exists | - | n/a | went to P4.A33: hostwritten graph arrays are gone (decisions.md B4.9) |
| P4.A7 | array state with storage kinds (`Region`, `Reserve`, `Committed`, `Evicted`, `Released`) and the first `resize!` (P4.A14) | - | open | |
| P4.A8 | pending changes on sync states with a device-wide sequence (`change!`, `pendone!`, `submitwith!`, `lockexpanded`) | - | open | |
| P4.A9 | dependents with registration after compile and marking at the change | - | open | |
| P4.A10 | `LengthOf` and `SizedBy` resolved at compile | - | open | |
| P4.A11 | indirect launches only over resizable arrays | - | open | |
| P4.A12 | `lengthof(a)` for lengths a kernel writes | - | open | |

| id | item | baseline at 5c2d4c6 | status | evidence |
|---|---|---|---|---|
| P4.D1 | delete `emitpatches!` | 2 hits, 1 files | open | |
| P4.D2 | delete `patchtab` | 10 hits, 9 files | open | |
| P4.D3 | delete `pending_patches` | 10 hits, 5 files | open | |
| P4.D4 | delete `closerecording!` | 21 hits, 7 files | open | |
| P4.D5 | delete `movelisteners` | 8 hits, 1 files | open | |
| P4.D6 | delete `notify_move!` | 30 hits, 13 files | open | |
| P4.D7 | delete `listen_moves!` | 2 hits, 2 files | open | |
| P4.D8 | delete `unlisten_moves!` | 3 hits, 2 files | open | |
| P4.D9 | delete the pointer-to-inline-copy argument layout (Lava's entry wrapper and Mantle's packer) | - | open | |
| P4.D10 | delete `graph/packing.jl`'s patch half and Vulkan's `Recording.patches` | - | open | |
| P4.D11 | delete `DeviceRange(count; max)`; the one-argument `DeviceRange(a)` is dropped too, `launchrange(a)` is the one name (S14) | 64 hits of `DeviceRange` (both forms), 18 files | open | |

| id | item | baseline at 5c2d4c6 | status | evidence |
|---|---|---|---|---|
| P4.W1 | rewrite `test/vulkan/test_recorded_move_patch.jl` (the Metal one in phase 9) | exists | open | |
| P4.G1 | green: `test_resize_is_queued.jl`, `test_host_node.jl` in full, `test_gpuref_slot.jl` (was `test_hostwritten.jl`, B4.9) | - | open | |
| P4.G2 | green: the whole-plan benchmark within its noise of `plan_*.txt` | - | open | |

### Added 2026-10-02 (review round 5)

| id | item | baseline at 5c2d4c6 | status | evidence |
|---|---|---|---|---|
| P4.A13 | eviction of idle resources under pressure (B3.9, input mode B4.6; was P2.A12): `Evicted` storage, `Restore` change, plans marked through dependents (every naming plan registers). Evictable (S22): no composite holds it, no handle dependent, no plan that names it running, not lent, nothing pending; accesses in flight allowed (the copy is ordered after them); `acquire!` waits for eviction copies outside every lock; the same on every device type, unified-memory ones included (B5.4); acceleration structures not evicted; `restore!` of an already-evicted array with no prepared storage marks the plan for restore, then `Again` | - | open | |
| P4.A14 | the first `resize!` (S5): in place when the new size fits the current storage (only a cell write); beyond it a large array on a device with sparse residency moves into a reserve, any other array into a pool region of twice the need (each later doubling a move) | - | open | |
| P4.A15 | `trim!(a)` (name open, S2): unmaps a reserve's pages beyond the size, pending; pressure never does | 23 hits of `trim!` (today's `trim!(pool, dev)`), 12 files | open | |
| P4.A16 | inline stores (S4; was P6.A13): the call copies the bytes into an upload region (B3.18); the applying submission writes them inline with `emitstore!(b, dst, upload)` up to the `caps(dev)` limit (Vulkan `vkCmdUpdateBuffer`, 64 KiB at 4-byte offsets and sizes), and a larger store is a copy from the upload region; no staging is allocated at apply. Today's core `emitstore!` (`graph/kalaunch.jl:816-823`, through `emitinline!`) gives the builder verb its name | 11 hits of `emitstore!`, 3 files | open | |
| P4.A17 | a value `GPURef` store marks the plans whose entry holds it (`entrychanged`); only marked plans write the entry and wait (S30) | n/a (0 hits) | n/a | went to P4.A33: no entry to mark (decisions.md B4.9) |
| P4.A18 | later segments of a stage that touch records whose changes were taken wait on the first segment's token (S26) | - | open | |
| P4.A19 | `lockexpanded` expands composite reads transitively (TLAS → BLASes and range arrays, BLAS → geometry, array `GPURef` → target); the run's waits and recorded accesses use the expanded uses (S21; each composite's method in P5.A18 and P6.A8) | - | open | |
| P4.A20 | `copy!(a, data)` with a new length does not copy the old contents (S31) | - | open | |
| P4.A21 | `reservesize` and `sparsethreshold` as core functions of `caps(dev)` fields (sparse page size, sparse address space), not device verbs (S15) | n/a (0 hits each) | open | |
| P4.A22 | `devicesize(a)` (name open): the length a kernel wrote, read back (plan, Operations) | n/a (0 hits) | n/a | went to P4.A34: `length(a)` is the real length (decisions.md B4.10) |
| P4.G3 | green: `test_eviction.jl` (plan phase 4; B3.9; was P2.T1) | - | open | |
| P4.D12 | delete `argbytes`, `argidentity`, `argsize` (core's `bindings!` lays out the argument block; api.md Appendix A) | 19 + 7 + 22 hits, 5 + 5 + 8 files | open | |
| P4.D13 | delete `passedas` (argument layout is core's; api.md Appendix A) | 21 hits, 5 files | open | |
| P4.D14 | delete `indirectindex`, `indirectslot` (launch commands are slots core lays out; api.md Appendix A) | 13 + 7 hits, 7 + 3 files | open | |
| P4.D15 | delete `devicesized` (`LengthOf` resolved at compile; api.md Appendix A) | 15 hits, 7 files | open | |
| P4.D16 | delete `patchable` (with the pushed patch mechanism; api.md Appendix A) | 16 hits, 6 files | open | |
| P4.D17 | delete `storebytes!` and `upload!` (stores are pending changes; api.md Appendix A) | 10 + 44 hits, 4 + 20 files | open | |
| P4.D18 | delete `hostwritten` graph arrays (B4.9) | 64 hits, 14 files at f6e359a | open | |
| P4.D19 | delete `persistent = true` on graph arrays (B4.8) | 1 hit, 1 file at f6e359a | open | |

### Added 2026-10-05 (ruling B3.13: changes are scheduled)

| id | item | baseline at 5c2d4c6 | status | evidence |
|---|---|---|---|---|
| P4.A30 | `change!(apply, uses)`: the call only enqueues (requested size for `length(a)`; a store's data copied into an upload region acquired at the call, B3.18, the one allocation a call makes); no other allocation, no retry, never a wait for a run at the call (B3.13) | - | open | |
| P4.A31 | the applying submission allocates before its locks (`prepare!`: storage or pages for a Resize, a structure's capacity, restore storage) and applies in sequence order (`submitwith!` with `stepsof`); Again if what it prepared no longer fits; every submitter on this path (`submitapplying`, run stages) | - | open | |
| P4.A32 | `resizable` flag set by the first `resize!` call or by `capacity = n` at creation (B4.10) (compile launches indirectly from then on); handle dependents get their parts marked for re-recording when a move is applied (open decision 17 (c), B3.11; nothing recompiles, nothing waits) | - | open | |
| P4.A33 | a value `GPURef` is one slot in device memory whose address plans and pieces bake; `r[] = v` a scheduled store (was P4.A4, P4.A6, P4.A17; B4.9) | - | open | |
| P4.A34 | `length(a)`/`size(a)` the real length; for a kernel-written length a wait and a read of the cell, a throw during the declaration of the graph that writes it; `resize!` beyond the capacity allowed, `capacity` like `sizehint!` (was P4.A22; B4.10) | - | open | |
| P4.A35 | the last hold's phase-4 half (was P2.A16): pending changes dropped (their upload regions retired), the array off the eviction list (B3.9), the cell released | - | open | |
| P4.T1 | benchmarks behind B3.11 before P4.A32: re-record cost at 100, 1 000 and 5 000 draws, and pulled indices against indexed draws, on NVIDIA, AMD and the M5 (`experiments/draw_bench/`) | - | open | |
| P4.G90 | gate: the phase's green tests pass on every joined device type | - | open | |
| P4.G91 | gate: one real use ran (editor opened and played, a RayMakie frame, a model call) | - | open | |
| P4.G92 | gate: independent review of the phase's diff against plan, api.md, ledger: no forbidden move | - | open | |
| P4.G93 | gate: baseline re-measured interleaved, no regression beyond noise | - | open | |

## Phase 5. Acceleration structures and textures

| id | item | baseline at 5c2d4c6 | status | evidence |
|---|---|---|---|---|
| P5.A1 | builds and refits as operations with `TraceRead`/`TraceBuild` declarations | - | open | |
| P5.A2 | the core `TLAS` (hardware and software) with instance ranges and the device-side range table | - | open | |
| P5.A3 | storage for a doubling capacity, builds and refits with the exact count; `BuildRecording` on Vulkan, count from a cell on Metal (Simon, 2026-10-02) | - | open | |
| P5.A4 | builds with all their uses | - | open | |
| P5.A5 | the texture table | - | open | |
| P5.A6 | (renumbered as P5.R1: a rename) | - | n/a | renumbered |
| P5.A7 | measure: a build with count N into storage sized for C costs the same as exact (expected, not measured; `experiments/tlas_sparse/`) | - | open | |

| id | item | baseline at 5c2d4c6 | status | evidence |
|---|---|---|---|---|
| P5.D1 | delete `AccelBuildContext` | 9 hits, 5 files | open | |
| P5.D2 | delete `MetalAccelBuildContext` | 3 hits, 1 files | open | |
| P5.D3 | delete `build_accel!` | 99 hits, 28 files | open | |
| P5.D4 | delete `refit_tlas!` | 29 hits, 11 files | open | |
| P5.D5 | delete `find_tlas_in_args` | 10 hits, 3 files | open | |
| P5.D6 | delete both HWTLAS implementations' bookkeeping (`vulkan/raytracing/hwtlas.jl`, `metal/hwtlas.jl`) | - | open | |
| P5.D7 | delete `hasfield(:hwtlas)` | 35 hits of `hasfield` | open | |
| P5.D8 | delete `Raycore.sync!` inside `adapt` | 89 hits of `Raycore.sync!` | open | |
| P5.D9 | delete `bind_textures`' per-call descriptor pool | 18 hits of `bind_textures` | open | |

| id | item | baseline at 5c2d4c6 | status | evidence |
|---|---|---|---|---|
| P5.G1 | green: phase 3's tests with a ray-traced graph whose TLAS is rebuilt every run | - | open | |

### Added 2026-10-02 (graphics, eviction, composites)

| id | item | baseline at 5c2d4c6 | status | evidence |
|---|---|---|---|---|
| P5.A8 | BLAS storage sized from its geometry with a capacity that doubles (S3): a new topology a Build into it (a `MoveAccel` beyond the capacity, the new address through its cell), moved vertices with the same topology an Update; TLASes over a BLAS pend an Update when its address or contents change; counts by the TLAS rules (`BuildRecording`, `BuildAccel`) | - | open | |
| P5.A9 | readers (B3.10): structure updates pended by host changes and by runs that write their arrays (`pendupdate!`) | - | open | |
| P5.A10 | per-range data (geometry arrays, material index) through the range table; no scene-wide concatenation | - | open | |
| P5.A11 | BLASes shared by geometry arrays; a `mask` per range entry for visibility (hide = mask 0, a refit; `active = false` only for a deleted range, a Build; S19) | - | open | |
| P5.D10 | delete the host triangle concatenation: `tri_gpu` | 32 hits, 9 files | open | |
| P5.D11 | delete `off_gpu` | 34 hits, 10 files | open | |
| P5.D12 | delete `triangle_data` | 19 hits, 6 files | open | |
| P5.D13 | delete `HardwareAccel` | 32 hits, 5 files | open | |
| P5.T1 | (moved: written as P0.T25, green as P5.G2) | - | n/a | moved |

### Added 2026-10-02 (review round 5)

| id | item | baseline at 5c2d4c6 | status | evidence |
|---|---|---|---|---|
| P5.R1 | rename the plan-internal `refit!` (`graph/build.jl`), which collides with the TLAS `refit!` (was P5.A6) | - | open | |
| P5.A12 | composites hold and never allocate (was P2.A13): TLAS, BLAS, tables, render objects built from arrays, array `GPURef`s and accel storage; only holds and change commits retire; `TLAS` and `BLAS` finalizers push their hold drops through the pool's inbox (S23; `ArrayBox` in P6.A15, render objects in P11.A16) | - | open | |
| P5.A13 | `BuildRecording`s marked by the change that alters what they recorded; the run re-records only marked ones; nothing compared at run (S20) | - | open | |
| P5.A14 | build coalescing per structure: any Build → Build, else Update; `builtcount` = the count of the last Build, used by every Update (S28) | n/a (`builtcount` 0 hits) | open | |
| P5.A15 | kernel-written counts (S29, to be confirmed): a TLAS total built over the capacity with mask-0 records past the total (Vulkan); a BLAS length built over the geometry capacity with degenerate primitives past the length, so later changes are Updates | - | open | |
| P5.A16 | a change that would make a running plan stale is applied after the run's end: the submitter that would apply it waits (`WaitFor`), never the call (B3.13); whether and for which changes this exception stays is open (review of 2026-10-14), throwing from the plan's own host function; no `except`; the records kernel writes mask 0 past the device total (S27, revised in the verification round) | - | open | |
| P5.A17 | reader snapshots (B3.10; S7): an array's readers are an immutable snapshot replaced under its lock and counted per reader; changes to an array with readers lock it with its readers and pend the readers' builds; the submission that applies them prepares reader capacity before its locks (B3.13); a run pends `pendupdate!(reader, array)` after each stage | - | open | |
| P5.A18 | `lockexpanded` methods for the TLAS (BLASes, range arrays) and the BLAS (geometry); the mechanism is P4.A19 (S21) | - | open | |
| P5.A19 | a TLAS total past the device's `maxInstanceCount` throws at the call that raises it (`push!`, a source's `resize!`/`copy!`, a rebind); capacity capped at it (B4.8) | - | open | |
| P5.A20 | `resize!` of a texture keeps the texels where the old and new sizes overlap (B4.8) | - | open | |
| P5.G2 | green: `test_composite_fuzz.jl` (plan phase 5; was P5.T1) | - | open | |
| P5.D14 | delete `AdaptedAccel` (with Metal's HWTLAS bookkeeping; api.md Appendix A) | 82 hits, 16 files | open | |
| P5.D15 | delete `remakeimage!` (a new size is a new image under a new index; api.md Appendix A) | 9 hits, 6 files | open | |
| P5.D16 | Raycore: delete the TLAS finalizer that finalizes arrays a recording and `StaticTLAS` still use (`instanced-bvh.jl:372, 400-411`), `blas_array`, and host readbacks and `synchronize()` between build steps (`resizing-and-raytracing.jl` D.1) | - (Raycore `sd/with-texture`, tree 9accd25) | open | |

| id | item | baseline at 5c2d4c6 | status | evidence |
|---|---|---|---|---|
| P5.G90 | gate: the phase's green tests pass on every joined device type | - | open | |
| P5.G91 | gate: one real use ran (editor opened and played, a RayMakie frame, a model call) | - | open | |
| P5.G92 | gate: independent review of the phase's diff against plan, api.md, ledger: no forbidden move | - | open | |
| P5.G93 | gate: baseline re-measured interleaved, no regression beyond noise | - | open | |

## Phase 6. One array type and operations

| id | item | baseline at 5c2d4c6 | status | evidence |
|---|---|---|---|---|
| P6.A1 | `MantleArray <: AbstractGPUArray` on devices and graph devices; `GraphDevice` | - | open | |
| P6.A2 | core's KA backend and GPUArrays interface (`storage`, `copy`, `derive`, `mapreducedim!`) | - | open | |
| P6.A3 | operations as methods calling `dispatch!(location(a), …)` | - | open | |
| P6.A4 | the selection table (the Vulkan GEMM/GEMV selection out of `vulkan/array/gemm.jl`, `basealignment` in the key) | - | open | |
| P6.A5 | optional rewriters | - | open | |
| P6.A6 | `devicearg` | - | open | |
| P6.A7 | scoped vendor views: core's `withview` and the foreign channel (G11; each device type's constructor with its phase, P9.A9, P10.A7); `withview` takes a hold, applies pending changes and restores first (the stream waits on that submission) and drops the hold at the end with the stream's token (S24) | - | open | |

| id | item | baseline at 5c2d4c6 | status | evidence |
|---|---|---|---|---|
| P6.D1 | delete `LavaArray` | 1021 hits, 145 files | open | |
| P6.D2 | delete `DeviceArray` | 64 hits, 11 files | open | |
| P6.D3 | delete `Buffer` | 525 hits, 83 files | open | |
| P6.D4 | delete `Transient` | 177 hits, 37 files | open | |
| P6.D5 | delete `TransientBuffer` | 53 hits, 12 files | open | |
| P6.D6 | delete `ka_launch_indirect!` | 5 hits, 3 files | open | |
| P6.D7 | delete `ArrayLaunch` | 13 hits, 3 files | open | |
| P6.D8 | delete `runlaunches!` | 7 hits, 4 files | open | |
| P6.D9 | delete `dispatchlaunches!` | 5 hits, 3 files | open | |
| P6.D10 | delete `kilaunch!` | 13 hits, 5 files | open | |
| P6.D11 | delete `gate!` | 5 hits, 1 files | open | |
| P6.D12 | delete `Mantle.Call` | 10 hits, 4 files | open | |
| P6.D13 | delete `librarygemm` | 9 hits, 4 files | open | |
| P6.D14 | delete `native_gemm_dispatch!` | 7 hits, 2 files | open | |
| P6.D15 | delete `native_attention_dispatch!` | 4 hits, 2 files | open | |
| P6.D16 | delete `native_batched_gemm_dispatch!` | 4 hits, 2 files | open | |
| P6.D17 | delete `native_conv2d_dispatch!` | 4 hits, 2 files | open | |
| P6.D18 | delete `native_gemm_available` | 3 hits, 2 files | open | |
| P6.D19 | delete `runscalls` | 24 hits, 11 files | open | |
| P6.D20 | delete `staged_gemm_tile` | 10 hits, 5 files | open | |
| P6.D21 | delete `begin_render_pass!` | 17 hits, 7 files | open | |
| P6.D22 | delete `end_render_pass!` | 13 hits, 7 files | open | |
| P6.D23 | delete `record_draw!` | 34 hits, 12 files | open | |
| P6.D24 | delete `begin_pass!` | 27 hits, 12 files | open | |
| P6.D25 | delete `draw_in_pass!` | 14 hits, 9 files | open | |
| P6.D26 | delete `draw_indexed_in_pass!` | 11 hits, 8 files | open | |
| P6.D27 | delete `draw_indirect_in_pass!` | 9 hits, 7 files | open | |
| P6.D28 | delete `end_pass!` | 15 hits, 9 files | open | |
| P6.D29 | delete `set_viewport!` | 16 hits, 11 files | open | |
| P6.D30 | delete `trace_rays!` | 20 hits, 11 files | open | |
| P6.D31 | delete `trace_rays_indirect!` | 14 hits, 9 files | open | |
| P6.D32 | delete `trace_closest_hits!` | 40 hits, 13 files | open | |
| P6.D33 | delete `trace_closest_hits_indirect!` | 13 hits, 9 files | open | |
| P6.D34 | delete `trace_closest_hits_anyhit!` | 9 hits, 5 files | open | |
| P6.D35 | delete `trace_closest_hits_anyhit_indirect!` | 7 hits, 5 files | open | |
| P6.D36 | delete `vulkan/array/gpuarrays.jl`'s duplicated paths and the eager `mapreduce.jl`/`accumulate.jl` implementations | - | open | |
| P6.D37 | delete the Vulkan and Metal KernelInterface host copies | - | open | |
| P6.D38 | delete the pass-level `dispatch!(p, …)`, `pass!`/`blit!`/`draw!` on a device | - | open | |
| P6.D39 | delete the `SGEMM_*` constants; DNNKernels' direct calls of `coopmat_gemm_dispatch!` | 9 hits of `coopmat_gemm_dispatch!` in JuliaVision | open | |

| id | item | baseline at 5c2d4c6 | status | evidence |
|---|---|---|---|---|
| P6.G1 | green: `test_one_path.jl` | - | open | |
| P6.G2 | green: GPUArrays' TestSuite on every joined device type | - | open | |
| P6.G3 | green: `test_view_barriers.jl` | - | open | |

### Added 2026-10-02 (graphics, eviction, composites)

| id | item | baseline at 5c2d4c6 | status | evidence |
|---|---|---|---|---|
| P6.A8 | array `GPURef` (rebind pending, holds, indirect launches, readers): device-only, the target a root device array (a view throws), rebinding to the current target a no-op (S16); `r[] = b` a pending cell write of the box, its reader TLASes pend an Update, a Build if the length changed (S19); its `lockexpanded` method (P4.A19) | - | open | |
| P6.A9 | `attribute(a, i) = a[min(i, length(a))]` (1-based; the benchmark's 0-based form `a[min(i, n-1)]`, S17): one argument type for a value and per-element attributes | - | open | |
| P6.A10 | graphics and ray-tracing pipelines from the device type's cache under one key (`compiledraw`, `compiletrace`); `npipelines` counts pipelines on every device type | - | open | |
| P6.A11 | images as resources (`Image`, stores, copies, `Array(img)`), one format table per device type (`imageformat`), the sampler table (`samplerentry!`) | - | open | |
| P6.A12 | `render!`/`draw!` as nodes; index arrays through `bindindices!`; `when!` draws, one per call with a `Bool` flag (B4.1) (indirect, the count multiplied by the flag, S13); the scene rectangle as one array per scene applied in the vertex stage (S32) | - | open | |
| P6.A16 | render passes as lists of pieces (B3.17): `render!` returns the pass; `insert!(p) do … end` / `delete!(p, h)` applied by the next run (`updatepieces!`: compile and record the piece, its argument region and cell-copy entries, its registration; the list part re-recorded); the head's cell-copy table a resizable array launched indirectly; a fixed barrier at every pass start; device-type verbs `recordpiece`, `emitpieces!` (Vulkan secondaries with dynamic rendering, Metal ICB per piece); gate: insert/delete time per plot at 100, 1 000, 5 000 plots does not grow with the count, a visibility change records nothing, a large mesh's draw time equal to a pass with only that mesh | - | open | |
| P6.A17 | KernelAbstractions on graph arrays declares: `KA.get_backend` is the graph, `KA.allocate` a transient; host reads of contents throw at their line; launch sizes recorded per node, a resize behind one throws at the next compile (B4.3) | - | open | |
| P6.A18 | `adapt(Mantle.Device(graph), x)`: zero-copy aliases the graph names, with a hold; `adapt(Mantle.Device(), x)` uploads (B4.4) | - | open | |
| P6.A19 | `MantleArray{T,N,L}`: a view is the same type (no role parameter); read-only, `Readback` memory and resizable are runtime properties of the resource (B4.5) | - | open | |
| P6.A20 | `readonly = true`: a reference to the host array for restores; writes throw at the call from the access `dispatch!` derives at declaration (B4.6) | - | open | |
| P6.A21 | bounds-checked kernels unless `@inbounds`; verify that Lava reports a device-side bounds error rather than faulting (B4.7) | - | open | |
| P6.A22 | `sort!(pass; by, lt, rev)` reorders a pass's pieces (B4.8) | - | open | |
| P6.A23 | resource identity: a resource is the array's state, shared by its views, reshapes and storage, never the object; an argument that reaches device memory but is not a `MantleArray` (a vendor array, an array escaping `withview`, a raw backend array) throws at declaration (plan, Arrays; api.md section 7). Replaces `Graph.ids` keyed by object and `resourceleaves!` taking every device-array leaf | - | open | |
| P6.A13 | (moved to P4.A16: the plan puts inline stores in phase 4) | - | n/a | moved |
| P6.A14 | tessellation as stage types | - | open | |
| P6.D40 | delete `pass!` (the hand-recorded path) | 22 hits, 9 files | open | |
| P6.D41 | delete `PassRecorder` | 10 hits, 2 files | open | |
| P6.D42 | delete `viewport!` | 10 hits, 3 files | open | |
| P6.D43 | delete `blit!` as an eager call (becomes `render!(dev, …)`) | 14 hits, 9 files | open | |
| P6.D44 | delete `SampledTexture` | 16 hits, 7 files | open | |
| P6.D45 | (moved to P7.D16: `Surface` goes in phase 7 with the window code that replaces it) | - | n/a | moved |
| P6.D46 | (moved to P7.D17) | - | n/a | moved |
| P6.D47 | (moved to P7.D18) | - | n/a | moved |
| P6.D48 | (moved to P7.D19) | - | n/a | moved |
| P6.D49 | (moved to P7.D20) | - | n/a | moved |
| P6.D50 | delete `transition_image!` | 17 hits, 9 files | open | |
| P6.D51 | delete `readback_framebuffer` | 35 hits, 8 files | open | |
| P6.D52 | delete `readback_target` | 21 hits, 4 files | open | |
| P6.D53 | delete `Attr` with a stored stride | 13 hits, 4 files | open | |
| P6.T1 | test: value ↔ per-element switch, rebind and visibility on one screen: one pipeline, zero recompiles, pixels match a fresh render (not in the plan: confirm) | - | open | |

### Added 2026-10-02 (review round 5)

| id | item | baseline at 5c2d4c6 | status | evidence |
|---|---|---|---|---|
| P6.A15 | `ArrayBox` (the array `GPURef`) finalizer pushing its hold drop through the pool's inbox (S23) | - | open | |
| P6.R1 | `compile_draw` becomes `compiledraw(dev, pipeline, argtypes, formats)` (api.md Appendix A) | 35 hits, 13 files | open | |
| P6.R2 | `makeimage` becomes `placeimage` behind `Image(dev, T, dims...)` (api.md Appendix A) | 7 hits, 5 files | open | |
| P6.D54 | delete the eager `mul!`/`norm` code in `vulkan/array/` (their kernels become candidates; plan, Eager work, "Goes") | - | open | |
| P6.D55 | delete `use` (plan, Operations, "Deleted"): at 5c2d4c6 only the export at `Mantle.jl:452` and comments remain | - (the word `use` is prose) | open | |
| P6.D56 | delete `argtype`, `devicebuffertype` (`devicearg(dev, a)` is the one conversion; api.md Appendix A) | 10 + 11 hits, 6 + 5 files | open | |
| P6.D57 | delete `deviceslice`, `deviceview` (`devicearg` resolves views in one place) | 7 + 28 hits, 5 + 13 files | open | |
| P6.D58 | delete `devicearray` (`MantleArray(dev, data)`) and `isdevicearray` (dispatch on `MantleArray`) | 41 + 11 hits, 11 + 6 files | open | |
| P6.D59 | delete `download` (host reads are `Array(a)` and copy nodes) | 58 hits (the word in prose included), 22 files | open | |
| P6.D60 | delete `indexbuffer` (indices are a `MantleArray` bound by `bindindices!`; open decision 17) | 20 hits, 9 files | open | |
| P6.D61 | delete `Framebuffer`, `colorimage`, `depthimage` (render targets are `Image`s and windows) | 50 + 9 + 9 hits, 16 + 6 + 6 files | open | |
| P6.D62 | delete `blittarget`, `copy_target!` (image copies are `copyto!` operations) | 14 + 7 hits, 7 + 4 files | open | |
| P6.D63 | delete `setviewport!` (the scene rectangle is a per-draw uniform) | 10 hits, 7 files | open | |

| id | item | baseline at 5c2d4c6 | status | evidence |
|---|---|---|---|---|
| P6.G90 | gate: the phase's green tests pass on every joined device type | - | open | |
| P6.G91 | gate: one real use ran (editor opened and played, a RayMakie frame, a model call) | - | open | |
| P6.G92 | gate: independent review of the phase's diff against plan, api.md, ledger: no forbidden move | - | open | |
| P6.G93 | gate: baseline re-measured interleaved, no regression beyond noise | - | open | |

## Phase 7. Windows and present

| id | item | baseline at 5c2d4c6 | status | evidence |
|---|---|---|---|---|
| P7.A1 | `Window` as a resource | - | open | |
| P7.A2 | swapchain generations with holds | - | open | |
| P7.A3 | acquire before the run's locks with binary semaphores and OUT_OF_DATE handling | - | open | |
| P7.A4 | per-image present segments | - | open | |
| P7.A5 | present in `submitnative!` | - | open | |
| P7.A6 | `resize!(win, …)` as a mark the next run applies | - | open | |
| P7.A7 | window capacity and window-sized arrays and images placed at it (a sparse arena part where the device has sparse residency, mapped per generation for its size); a resize beyond the capacity recompiles the graph at its next run (S1, S9) | - | open | |

| id | item | baseline at 5c2d4c6 | status | evidence |
|---|---|---|---|---|
| P7.D1 | delete `device_wait_idle` | 9 hits, 6 files | open | |
| P7.D2 | delete `submit_and_present!` | 10 hits, 4 files | open | |
| P7.D3 | delete `DrawBinding` | 23 hits, 7 files | open | |
| P7.D4 | delete `boundbindings` | 4 hits, 3 files | open | |
| P7.D5 | delete `boundcount` | 4 hits, 3 files | open | |
| P7.D6 | delete `boundindices` | 4 hits, 3 files | open | |
| P7.D7 | delete `boundinstances` | 4 hits, 3 files | open | |
| P7.D8 | delete `recordsplans` | 17 hits, 11 files | open | |
| P7.D9 | delete `Immediate` | 37 hits, 7 files | open | |
| P7.D10 | delete `recordable` | 24 hits, 9 files | open | |
| P7.D11 | delete `rebind!` | 11 hits, 6 files | open | |
| P7.D12 | delete the swapchain rebuild inside acquire and present | - | open | |
| P7.D13 | delete the walked path in `execute!` (8d95632's walked headless plans: RayMakie's composited frame becomes a recorded graph) | 14 hits of `execute!` | open | |

| id | item | baseline at 5c2d4c6 | status | evidence |
|---|---|---|---|---|
| P7.W1 | rewrite `test/test_headless_rebindable_draw.jl` | exists | open | |
| P7.W2 | rewrite `test/vulkan/test_recordable_plans.jl` | exists | open | |
| P7.G1 | green: `test_window_resize.jl`, `test_run_is_submit.jl` for a windowed graph with overlays | - | open | |

### Added 2026-10-02 (graphics, eviction, composites)

| id | item | baseline at 5c2d4c6 | status | evidence |
|---|---|---|---|---|
| P7.A8 | window-sized segments recorded per swapchain generation | - | open | |
| P7.A9 | screenshots: `Array(win)` through a second recording of the present segment | - | open | |
| P7.A10 | (removed: the default colour format and the `vsync` default are open, api.md G9; F9) | - | n/a | open question, not an item |
| P7.D14 | delete `readback_window` | 20 hits, 10 files | open | |

### Added 2026-10-02 (review round 5)

| id | item | baseline at 5c2d4c6 | status | evidence |
|---|---|---|---|---|
| P7.A11 | window-sized images recreated per swapchain generation at the actual size on the same arena memory (S9; the segments drawing into them are P7.A8) | - | open | |
| P7.A12 | the swapchain image has one owner, the window: an image of a run that returns `Again` or throws goes back to it (`keepimage!`); only real images are kept (S33) | n/a (`keepimage!` 0 hits) | open | |
| P7.A13 | the window open verb (name open, G13; the plan's "windows: open, acquire, present") | - | open | |
| P7.A14 | windows (B4.8): any node of the presenting graph writes the window and the first graph declaring a write presents, a second throws; the binding ends at `free!`, collection or the render node's removal; eager `render!` into a window throws; `Array(win)` throws when minimized, unpresented or from the graph's own host function, and wakes a waiting call with the error; `run!` returns `Submitted()`/`Skipped()` | - | open | |
| P7.R1 | `acquire_next_image!` becomes `acquirenext(sc)` (api.md Appendix A) | 32 hits, 12 files | open | |
| P7.R2 | `screenshot` becomes `Array(win)` (name open) | 17 hits, 8 files | open | |
| P7.D15 | RayMakie: delete `DrawBinding` use (was P11.D14, numbered P11.D13 before the renumbering; the plan deletes `DrawBinding` in phase 7, "RayMakie uses it") | 6 hits, 2 files (RayMakie e6e1aa9b0) | open | |
| P7.D16 | delete `Surface` (was P6.D45) | 18 hits, 8 files | open | |
| P7.D17 | delete `target_view` (was P6.D46) | 24 hits, 11 files | open | |
| P7.D18 | delete `target_image` (was P6.D47) | 15 hits, 7 files | open | |
| P7.D19 | delete `target_extent` (was P6.D48) | 36 hits, 13 files | open | |
| P7.D20 | delete `target_format` (was P6.D49; the format comes from `imageformat(dev, T)`) | 18 hits, 9 files | open | |
| P7.D21 | delete `beginframe!` (acquire and present are part of `run!`; api.md Appendix A) | 14 hits, 9 files | open | |
| P7.D22 | delete `present_frame!` (present is part of `submitnative!`) | 23 hits, 12 files | open | |
| P7.D23 | delete `currentimage` (the run holds the acquired image; the window owns it) | 7 hits, 5 files | open | |
| P7.D24 | delete `use_bindings!` (draw-binding code; bindings compiled with the plan; its last caller is the walked draw in `execute!`, P7.D13) | 16 hits, 10 files | open | |

| id | item | baseline at 5c2d4c6 | status | evidence |
|---|---|---|---|---|
| P7.G90 | gate: the phase's green tests pass on every joined device type | - | open | |
| P7.G91 | gate: one real use ran (editor opened and played, a RayMakie frame, a model call) | - | open | |
| P7.G92 | gate: independent review of the phase's diff against plan, api.md, ledger: no forbidden move | - | open | |
| P7.G93 | gate: baseline re-measured interleaved, no regression beyond noise | - | open | |

## Phase 8. Devices are arguments

| id | item | baseline at 5c2d4c6 | status | evidence |
|---|---|---|---|---|
| P8.A1 | every function takes its device or channel; every hook takes the device and dispatches on its type | 302 hits of `vk_context`, 99 of `default_bq` | open | |
| P8.A2 | the device registry returns one object per hardware device | - | open | |
| P8.A3 | `Lavapipe_jll` becomes an extension | - | open | |

| id | item | baseline at 5c2d4c6 | status | evidence |
|---|---|---|---|---|
| P8.D1 | delete `DEVICES` | 4 hits, 3 files | open | |
| P8.D2 | delete `lavadevice` | 21 hits, 8 files | open | |
| P8.D3 | delete `VK_CONTEXT_REF` | 24 hits, 7 files | open | |
| P8.D4 | delete `METAL_DEVICE` | 6 hits, 4 files | open | |
| P8.D5 | delete `ROCM_DEVICE` | 4 hits, 1 files | open | |
| P8.D6 | delete `adopt!` | 12 hits, 1 files | open | |
| P8.D7 | delete `adoptqueue!` | 16 hits, 4 files | open | |
| P8.D8 | delete `LIVE_CONTEXTS` | 4 hits, 1 files | open | |
| P8.D9 | delete `LAYER_SETTING_STORAGE` | 3 hits, 1 files | open | |
| P8.D10 | delete `PIPELINE_NO_COMPILE` | 16 hits, 3 files | open | |
| P8.D11 | delete `WORKGROUP_FALLBACK` | 15 hits, 2 files | done | the fault was NVIDIA's 64-bit division; Lava's `lower_wide_divisions!` removes it (2026-10-12); Mantle 28a58dc, `test_static_workgroup.jl` |
| P8.D12 | delete `KERNEL_RECORDERS` | 5 hits, 1 files | open | |
| P8.D13 | delete `BACKEND_PROBES` | 7 hits, 1 files | open | |
| P8.D14 | delete `COMMITTED` | 10 hits, 2 files | open | |
| P8.D15 | delete `COLLECT` | 3 hits, 1 files | open | |
| P8.D16 | delete `Metal.device` | 33 hits, 10 files | open | |
| P8.D17 | delete `Metal.synchronize` | 30 hits, 8 files | open | |
| P8.D18 | delete `initbackend!` | 11 hits, 7 files | open | |
| P8.D19 | delete `Backend` | 37 hits, 21 files | open | |
| P8.D20 | delete `VulkanAPI` | 160 hits, 41 files | open | |
| P8.D21 | delete `MetalAPI` | 46 hits, 23 files | open | |
| P8.D22 | delete `ROCmAPI` | 32 hits, 4 files | open | |
| P8.D23 | delete `WebGPUAPI` | 7 hits, 4 files | open | |
| P8.D24 | delete `syncbackend` | 9 hits, 6 files | open | |
| P8.D25 | delete `LavaBackend` | 265 hits, 107 files | open | |
| P8.D26 | delete `kibackend` | 11 hits, 5 files | open | |
| P8.D27 | delete `todevice` | 41 hits, 21 files | open | |
| P8.D28 | delete `defaultbackend` | 218 hits, 90 files | open | |
| P8.D29 | delete `availablebackends` | 6 hits, 4 files | open | |
| P8.D30 | delete `eachbackend` | 18 hits, 9 files | open | |
| P8.D31 | delete `register_backend!` | 8 hits, 5 files | open | |
| P8.D32 | delete `BACKEND_VOCABULARY` | 17 hits, 7 files | open | |
| P8.D33 | delete `HostDevice` and `src/host/` | 0 hits (removed in 5c2d4c6): n/a once confirmed | open | |
| P8.D34 | delete `FROZEN_*` as process globals; lock or remove `iterplans`/`launchplans` | 6 + 12 hits | open | |
| P8.D35 | delete graphics constructors keyed on a KA backend; hooks without a device argument other than `staged_gemm_tile()` (P6.D20) and `initbackend!()` (P8.D18) | - | open | |
| P8.D36 | delete `Device(::KA.Backend)`, `caps(::KA.Backend)`, `Device(api; select)`, `backend(dev)` | - | open | |

| id | item | baseline at 5c2d4c6 | status | evidence |
|---|---|---|---|---|
| P8.G1 | green: `test_device_is_an_argument.jl`, `test_any_thread.jl` | - | open | |

### Added 2026-10-02 (review round 5)

| id | item | baseline at 5c2d4c6 | status | evidence |
|---|---|---|---|---|
| P8.D37 | delete `HostAPI` (plan, Devices, "Deleted") | 0 hits: n/a once confirmed | open | |
| P8.D38 | delete `defaultdevice!` (`Device()` and the registry; api.md Appendix A) | 18 hits, 9 files | open | |
| P8.D39 | delete `use_frozen_kernels` (the `Mantle.Kernel` at its `CodeInstance`, B5.11; api.md Appendix A) | 11 hits, 8 files | open | |

| id | item | baseline at 5c2d4c6 | status | evidence |
|---|---|---|---|---|
| P8.G90 | gate: the phase's green tests pass on every joined device type | - | open | |
| P8.G91 | gate: one real use ran (editor opened and played, a RayMakie frame, a model call) | - | open | |
| P8.G92 | gate: independent review of the phase's diff against plan, api.md, ledger: no forbidden move | - | open | |
| P8.G93 | gate: baseline re-measured interleaved, no regression beyond noise | - | open | |

## Phase 9. Metal on the core model

| id | item | baseline at 5c2d4c6 | status | evidence |
|---|---|---|---|---|
| P9.A1 | open items 6 (cost of one event-chained `MPSGraphExecutable` call) and 7 (per-draw state an MTL4 indirect render command holds) answered first | - | open | |
| P9.A2 | channels, compiled form and run (one command buffer per segment) | - | open | |
| P9.A3 | placement-sparse arenas | - | open | |
| P9.A4 | MLX's kernels | - | open | |
| P9.A5 | the device-wide residency set on every queue plus a set per plan for its ICBs and pipelines | - | open | |
| P9.A6 | the argument layout measured on the M5 | - | open | |

| id | item | baseline at 5c2d4c6 | status | evidence |
|---|---|---|---|---|
| P9.D1 | delete `LegacyQueue` | 40 hits, 3 files | open | |
| P9.D2 | delete `LegacySubmission` | 6 hits, 1 files | open | |
| P9.D3 | delete `MTL4Submission` | 6 hits, 1 files | open | |
| P9.D4 | delete `opensubmit!` | 20 hits, 4 files | open | |
| P9.D5 | delete `closesubmit!` | 11 hits, 4 files | open | |
| P9.D6 | delete `suspendsubmit!` | 5 hits, 2 files | open | |
| P9.D7 | delete `replaywithcalls!` | 7 hits, 3 files | open | |
| P9.D8 | delete `canreplay` | 11 hits, 3 files | open | |
| P9.D9 | delete `ownflush!` | 8 hits, 2 files | open | |
| P9.D10 | delete `ensureresident!` | 6 hits, 2 files | open | |
| P9.D11 | delete `make_persistently_resident!` | 7 hits, 2 files | open | |
| P9.D12 | delete `walkedplan` | 2 hits, 1 files | open | |
| P9.D13 | delete `planshape` | 3 hits, 1 files | open | |
| P9.D14 | delete `MetalRecordedDispatch` | 15 hits, 4 files | open | |
| P9.D15 | delete the slot ring, `encodes` and per-run encoding | - | open | |
| P9.D16 | delete every `MPSGraph*` callable that MLX's kernels match | - | open | |
| P9.D17 | delete Metal's `openrun`/`closerun!`, `batchqueue`, `DeviceArray`, `TransientBuffer`, `Buffer`, `LavaArray` (kept until this phase) | - | open | |

| id | item | baseline at 5c2d4c6 | status | evidence |
|---|---|---|---|---|
| P9.W1 | rewrite `test/metal/test_recorded_move_metal.jl` | exists | open | |
| P9.G1 | green: phase 0's tests on the MacBook worker; SAM 2.1's frame no slower than today; RayDemo's crown matching today's image | - | open | |

### Added 2026-10-02 (review round 5)

| id | item | baseline at 5c2d4c6 | status | evidence |
|---|---|---|---|---|
| P9.A7 | stores on Metal: inline limit 0 in `caps(dev)`, every store a copy from its upload region (S4, B3.18) | - | open | |
| P9.A8 | one residency helper used by `rawalloc`, `rawfree`, `createreserve`, `bindpages!`, `unbindpages!` and the per-plan sets (api.md section 8) | - | open | |
| P9.A9 | `MtlArray` vendor view constructor (G11; core's `withview` is P6.A7) | - | open | |
| P9.D18 | delete Mantle's use of Metal.jl's `BatchedCommandQueue` (`metal/device.jl:82, 93`; plan, Metal, "Channels") | 10 hits, 2 files | open | |

| id | item | baseline at 5c2d4c6 | status | evidence |
|---|---|---|---|---|
| P9.G90 | gate: the phase's green tests pass on every joined device type | - | open | |
| P9.G91 | gate: one real use ran (editor opened and played, a RayMakie frame, a model call) | - | open | |
| P9.G92 | gate: independent review of the phase's diff against plan, api.md, ledger: no forbidden move | - | open | |
| P9.G93 | gate: baseline re-measured interleaved, no regression beyond noise | - | open | |

## Phase 10. CUDA and HIP

| id | item | baseline at 5c2d4c6 | status | evidence |
|---|---|---|---|---|
| P10.A1 | CUDA device type new; ROCm rebuilt on the core model: graphs built node by node | - | open | |
| P10.A2 | argument layout decided by measurement (kernel parameters by value with node-parameter updates, or a pointer into the block) | - | open | |
| P10.A3 | counters written by the stream as tokens; streams as queue kinds; VMM arenas | - | open | |
| P10.A4 | library candidates (CUTLASS, cuDNN, Composable Kernel; cuBLAS/rocBLAS/hipBLASLt as foreign nodes) | - | open | |
| P10.A5 | validation on CUDA before building: changed kernel node, cuDNN into our graph, CUTLASS via NVRTC, a WHILE node, a VMM arena under a live graph, two streams with a wait (HIP: the same minus conditional node and cuDNN) | - | open | |

| id | item | baseline at 5c2d4c6 | status | evidence |
|---|---|---|---|---|
| P10.D1 | delete `checkresolved` | 5 hits, 2 files | open | |
| P10.D2 | delete `ROCmCompiledDispatch` | 12 hits, 2 files | open | |
| P10.D3 | delete `needs_transition` | 21 hits, 10 files | open | |
| P10.D4 | delete stream capture | - | open | |
| P10.D5 | delete the retirement-counter timeline with full-sync waits | - | open | |
| P10.D6 | delete AMDGPU `Managed` tracking of Mantle memory; ROCm's `TransientBuffer`, `DeviceArray` (`ext/MantleROCmExt.jl`) | - | open | |

| id | item | baseline at 5c2d4c6 | status | evidence |
|---|---|---|---|---|
| P10.G1 | green: phase 0's tests on the workstation (CUDA) and the Bosgame (HIP) | - | open | |

### Added 2026-10-02 (review round 5)

| id | item | baseline at 5c2d4c6 | status | evidence |
|---|---|---|---|---|
| P10.A6 | stores on CUDA and HIP: inline limit 0 in `caps(dev)`, every store a copy from its upload region (S4, B3.18) | - | open | |
| P10.A7 | `CuArray` and `ROCArray` vendor view constructors (G11; core's `withview` is P6.A7) | - | open | |

| id | item | baseline at 5c2d4c6 | status | evidence |
|---|---|---|---|---|
| P10.G90 | gate: the phase's green tests pass on every joined device type | - | open | |
| P10.G91 | gate: one real use ran (editor opened and played, a RayMakie frame, a model call) | - | open | |
| P10.G92 | gate: independent review of the phase's diff against plan, api.md, ledger: no forbidden move | - | open | |
| P10.G93 | gate: baseline re-measured interleaved, no regression beyond noise | - | open | |

## Phase 11. Consumers

| id | item | baseline at 5c2d4c6 | status | evidence |
|---|---|---|---|---|
| P11.D1 | VideoEditor: `SCENELOCK` goes (as the plan says for it) | 2 hits, 1 files (VideoEditor ab160d9) | open | |
| P11.D2 | VideoEditor: `mantlethread` goes (as the plan says for it) | 3 hits, 1 files (VideoEditor ab160d9) | open | |
| P11.D3 | VideoEditor: `renderthread`'s Mantle branch goes (the plan removes only that branch) | 8 hits of `renderthread`, 3 files (VideoEditor ab160d9) | open | |
| P11.D4 | VideoEditor: `runowned`'s `WrongThread` discovery goes (the plan removes only that) | 14 hits of `runowned`, 3 files (VideoEditor ab160d9) | open | |
| P11.D5 | VideoEditor: `onthread` for Mantle renderers goes (the plan removes only those uses) | 10 hits of `onthread`, 3 files (VideoEditor ab160d9) | open | |
| P11.D6 | Hikari: `fill_aux_buffers!` and `postprocess!` become methods, not deleted (plan, phase 11); what goes is their `waitidle` (P11.D16); the add side is P11.A2 | 9 hits of `fill_aux_buffers!`, 6 files (Hikari c5ca85a) | open | |
| P11.D7 | Hikari: `notify_scene_changed` and `invalidate!` dropping plans go (scene changes never free the integrator graph). Includes P11.D15 (numbered P11.D14 before the renumbering) | 18 + 8 hits (Hikari c5ca85a), 3 files for `notify_scene_changed` | open | |
| P11.D8 | JuliaVision: `replay!` goes (as the plan says for it) | 52 hits, 29 files (JuliaVision 2f0c3ed) | open | |
| P11.D9 | JuliaVision: `planfor` goes (as the plan says for it) | 47 hits, 25 files (JuliaVision 2f0c3ed) | open | |
| P11.D10 | JuliaVision: `recordedplan` goes (as the plan says for it) | 14 hits, 7 files (JuliaVision 2f0c3ed) | open | |
| P11.D11 | JuliaVision: helper work run as its own plan or through `runonce!` becomes part of the call's graph (the plan lists `runonce!` as a pattern, "Where JuliaVision is today", not as a named delete) | 61 hits of `runonce!`, 18 files (JuliaVision 2f0c3ed) | open | |
| P11.D12 | VideoEditor: the `repeat!(g, 1; while_nonzero = gate)` idiom becomes one `when!` per kernel with a `Bool` gate (B4.1) | 9 hits of `while_nonzero` | open | |
| P11.A1 | VideoEditor: the engine's device passed everywhere | - | open | |
| P11.A2 | Hikari: `fill_aux_buffers!` and `postprocess!` become methods; builds and refits are operations; one `VolPath` rendered by one task at a time through its graph's lock | - | open | |
| P11.A3 | RayMakie: frame graphs recorded; overlay compositing through the one path | - | open | |
| P11.A4 | JuliaVision: one graph per call; DNNKernels' GEMM, attention and convolution as `mul!`, `conv!`, `attention!` on graph arrays; `call`, `replay!`, `planfor`, `recordedplan`'s hooks go | - | open | |
| P11.A5 | comments contradicting Mantle corrected in the commit that makes them wrong | - | open | |
| P11.G1 | green: each consumer suite with two devices open; each runner's per-call assertion (one run per call, no bytes copied outside the graph, host waits only for outputs and at host nodes) | - | open | |

### Added 2026-10-02 (RayMakie on the core model)

| id | item | baseline | status | evidence |
|---|---|---|---|---|
| P11.A6 | RayMakie: frame graph recorded once; adding or removing a plot recompiles nothing (a piece, B3.17), a pipeline change is a piece delete and insert; recompiled only for kernels declared per plot, a new pass, a new Hikari material type (open decision 18) | - | open | |
| P11.A7 | RayMakie: attributes as array `GPURef`s; `update!` as stores, resizes and rebinds; zero elements keeps the render object | - | open | |
| P11.A8 | RayMakie: a dirty list put by ComputePipeline edges instead of scene-tree walks | `poll_all_plots` 10 hits, `for_each_atomic_plot` 11 hits | open | |
| P11.A9 | RayMakie: the screen as a reader of its plots' arrays (redraw marks from runs that write them) | - | open | |
| P11.A10 | RayMakie: shadow and AO passes only when their inputs changed (under `when!`, plan phase 11) | - | open | |
| P11.A11 | RayMakie: grid lines from the camera on the GPU; lines' indices stored; `visible` for traced plots; the missing dependencies (image camera, stroke and glow colours, colormaps) | - | open | |
| P11.D13 | RayMakie: delete `frame_signature` (numbered P11.D12 before the review round, a duplicate id) | 6 hits, 2 files (RayMakie e6e1aa9b0) | open | |
| P11.D14 | (moved to P7.D15: the plan deletes `DrawBinding` in phase 7) | - | n/a | moved |
| P11.D15 | (merged into P11.D7, review round 5) | - | n/a | merged |
| P11.D16 | Hikari and RayMakie: no `waitidle` inside a frame | 7 hits in Hikari, 5 in RayMakie | open | |
| P11.G2 | green: RayMakie's compile-counter session (written as P0.T26; attribute changes, value ↔ per-element, rebinds, visibility, camera, layout, resizes within capacity, TLAS moves and counts): zero compiles (plan phase 11) | - | open | |
| P11.G3 | (moved to P3.G2: `test_run_host_time.jl` is green in phase 3; flat in the number of named resources, open decision 19, ruled B3.15) | - | n/a | moved |

### Added 2026-10-02 (review round 5)

| id | item | baseline at 5c2d4c6 | status | evidence |
|---|---|---|---|---|
| P11.A12 | RayMakie: plots are pieces of the frame graph's pass (`insert!(p) do … end`/`delete!(p, h)`, names open, B3.17); Hikari's integrator declared into the frame graph once, outside the plots' pieces; film and sample count are device arrays the screen holds (survive plot changes and recompiles), work queues window-sized graph arrays (S32 corrected by B3.16 and B3.17) | - | open | |
| P11.A13 | RayMakie: one scene-rectangle array per scene, shared by its plots (S32) | - | open | |
| P11.A14 | RayMakie: glyph UVs in texels, normalized in the shader, so atlas growth rewrites no UV (S32) | - | open | |
| P11.A15 | RayMakie: per-frame values (the camera matrix) as one-element arrays written by stores, inline from their upload region up to the `caps(dev)` limit (S4, B3.18) | - | open | |
| P11.A16 | RayMakie: render objects' finalizers push their hold drops through the pool's inbox (S23) | - | open | |
| P11.A17 | JuliaVision: load is a call: load-time transposes, `hoistconstants` and quantization become one load graph per model (plan, "What removes every split", item 10) | 37 hits of `hoistconstants`, 11 files (JuliaVision 2f0c3ed) | open | |
| P11.D17 | GPUFiltering: its internal `synchronize` and readbacks go (plan, "What removes every split", item 2) | 22 hits of `synchronize`, 12 files (JuliaVision 2f0c3ed, `GPUFiltering/`) | doing | the `synchronize` calls are gone (JuliaVision c2f859b; 3 hits left at 1018c51, all comments); readbacks remain in `goodfeatures` (`features.jl:86`) and `fitaffine`/`fithomography` (`flow.jl:117`, `:333`, fits on the CPU over the flow field), which need device reductions |
| P11.D18 | GPUFiltering: `launch!(device, kernel, args, ndrange)` and the explicit device argument go once a KA launch over graph arrays declares a node (P6.A17); ops with temporaries keep two methods of one name, one allocating them (`MantleArray(location(out), …)`) and one taking them | 31 hits of `launch!(`, 12 files (JuliaVision 1018c51, `GPUFiltering/src`) | open | GPUFiltering depends on Mantle directly since 1018c51 |
| P11.D19 | Paperhawk (github.com/SimonDanisch/Paperhawk.jl): `DeviceArrays` (`src/gpu.jl`) goes | not checked out here: count at its port | open | |
| P11.D20 | `Mantle.storage(...)` calls go in Paperhawk, PPOCRRunner and UVDocRunner (`MantleArray` is the array; P6.D58 for `devicearray`) | 10 hits, 2 files in PPOCRRunner/UVDocRunner (JuliaVision 1018c51); Paperhawk: count at its port | open | |
| P11.A18 | Paperhawk: one graph per call; its per-step plans (detector frame, spans, edges, thumbnail; page warp, cleanup, rotate, pack; preview tiles) become one graph each, and host readbacks after `run!` become `Readback` arrays or copy nodes (plan, Consumers) | - | open | |

| id | item | baseline at 5c2d4c6 | status | evidence |
|---|---|---|---|---|
| P11.G90 | gate: the phase's green tests pass on every joined device type | - | open | |
| P11.G91 | gate: one real use ran (editor opened and played, a RayMakie frame, a model call) | - | open | |
| P11.G92 | gate: independent review of the phase's diff against plan, api.md, ledger: no forbidden move | - | open | |
| P11.G93 | gate: baseline re-measured interleaved, no regression beyond noise | - | open | |

## Phase 12. Full suites, once

| id | item | baseline at 5c2d4c6 | status | evidence |
|---|---|---|---|---|
| P12.G1 | Mantle, Hikari, RayMakie, VideoEditor, JuliaVision and Paperhawk on Linux (Vulkan, CUDA, HIP); Paperhawk's suite is `Pkg.test()` in its root, with a display and the GPU | - | open | |
| P12.G2 | Mantle, Hikari, RayMakie on the Mac | - | open | |
| P12.A1 | a "State" section written from the results | - | open | |

## Found while working

Items not in a phase's list when the ledger was written. Each names the
phase that needs it resolved.

| id | item | baseline at 5c2d4c6 | status | evidence |
|---|---|---|---|---|
| F1 | G1 graphics resources and render state need a plan section (`api.md` section 10) | before phase 6 | n/a | the plan's "Graphics" section (2026-10-02); what stays open is F16 |
| F2 | G2 per-argument access of kernels and shaders: plan line or ruling (proposal: part of `KernelInfo`) | before phase 1 | done | ruled 2026-10-19: decisions.md B5.10 (declaration-time inference, not `KernelInfo`) and B5.11 (stored in the `Kernel`); api.md section 10 G2 and Appendix A updated |
| F3 | G3 device facts as fields of `caps(dev)`: field names for today's `supports_*`, `maxalloc`, the roofline inputs (throughput per element type, bandwidth, transfer queue, cross-queue cost) and `config(dev, f)` | before phases 1 and 8 | open | |
| F4 | G4 profiling (open decision 13) | before phase 1 (the split may not use it) | open | |
| F5 | G5 device wait-idle verb | before phase 3 | open | |
| F6 | G6 `recycle!` on Vulkan, CUDA and HIP (the recording lifecycle and Metal's free list are specified) | before phase 3 | open | |
| F7 | G7 copies between devices (open decision 7) | before phase 6 | open | |
| F8 | G8 the selection table: locking and when measurement runs (key, per-device table and the measured-else-estimate rule are specified) | before phase 6 | open | |
| F9 | G9 window API: GLFW hints, the default colour space and format, the `vsync` default (was P7.A10) | before phase 7 | open | |
| F10 | G10 counted driver objects: verb names | before phase 2 | open | |
| F11 | G11 vendor views: core's `withview` and the foreign channel in phase 6 (the plan's phase list); each device type's constructor with its phase (9, 10) | before phase 6 | open | |
| F12 | 71 vocabulary fates marked D in `api.md` Appendix A (after the review round's relabels: 91 P, 71 D, 27 G of 189): confirm in review | phase 0 | open | |
| F13 | open decisions 2-13 of the plan: ruled at the start of the phase that needs each | per phase | open | |
| F14 | open decision 15: per-frame draw order changes (transparency sorting); a changed order is `sort!(pass; by, lt, rev)`, a rewrite of the piece list (decisions.md B4.8) | before phase 7 | open | |
| F15 | open decision 16: rule eviction (decisions.md D9) and readers after runs (D10); the array `GPURef`, `attribute` and the per-draw scene rectangle were ruled in B3.7 | before phases 4, 5 | n/a | ruled: eviction automatic (B3.9), readers (B3.10), read-only inputs (B4.6) |
| F16 | G1 rest: SBT layout for procedural hit groups (Hikari's material types are shade-kernel specializations, not hit groups), stencil, multisampling, blend modes | before phase 6 | open | |
| F17 | bugs the 2026-10-02 survey found, to triage into the bug list: Metal `npipelines` counts draws; a resource-valued draw count has no Metal method; `trace!` has no Metal compile path; Vulkan walked plans rewrite argument memory while a frame may be in flight; the Vulkan pipeline cache key holds only whether a descriptor layout exists and misses `tess_eval`; Vulkan `vsync` lost on resize; a hand-recorded window frame may lose its contents | phase 0 | open | |

### Added 2026-10-02 (review round 5)

| id | item | baseline at 5c2d4c6 | status | evidence |
|---|---|---|---|---|
| F18 | G12 device teardown for debugging (`reset_device!`, declared and never implemented) | before phase 8 | open | |
| F19 | G13 the window open verb: name and signature | before phase 7 | open | |
| F20 | G14 texture and sampler table locking (`api.md` gives each a leaf lock, derived); table indices retired by token or not; a restored texture's new index | before phase 5 | open | |
| F21 | G15 contracts still marked "(contract: phase N)": `placeimage`'s arguments, `recycle!` | before phases 2 and 3 | open | |
| F22 | G16 launch kernels never read cells: a test (P0.T27 proposes a check) | before phase 4 | open | |
| F23 | open decision 17: index buffers and other handle dependences (decisions.md D11) | before phase 6 | n/a | ruled 2026-10-05 (B3.11): (c) re-record; its benchmarks are P4.T1 |
| F24 | open decision 18: RayMakie updates that change a shader (decisions.md D12) | before phase 11 | n/a | ruled 2026-10-05 (B3.12): value changes never compile (GPU flag, `when!`); type changes are delete! + insert! (a piece, B3.17) |
| F25 | open decision 19: run host time, the lock loop measured against the phase 3 target or a per-plan inbox (decisions.md D13) | phase 3 | n/a | ruled 2026-10-05 (B3.14, B3.15): per-plan inbox, flat host time; tested by P3.G2 |
| F26 | the review round's choices applied and open to veto (decisions.md, "Review round 2026-10-02"): S1, S5, S16, S22, S25, S29, S31, S32 | phase 0 | open | S13 settled by B4.1 |
| F27 | `docs/internals.jl`, `docs/examples.jl`, `docs/resizing-and-raytracing.jl` still describe `when!` blocks, `repeat!` bodies, the per-plan entry, hostwritten and persistent graph arrays; bring them in line with decisions.md B4 (P0.A2) | phase 0 | doing | fixed with P0.A2 |
| F28 | the baseline column counts names at 5c2d4c6, 133 commits before the frozen head f6e359a; recount (rows added 2026-10-14 already count at f6e359a) | phase 0 | open | |
| F29 | Lava's eager launch costs about 24 µs per dispatch (SAM 2 encoder on the RTX 4000 Ada, `experiments/eager-vs-graph/`); eager stays a first-class path, so find where it goes | phase 1 | open | |
| F30 | `insert!(g) do … end` and `render!(…) do p … end` keep `do`-block bodies; B4.2 removed them for launch groups only | before phase 6 | open | |
| F31 | Lava d688952's RADV/ACO workaround (`foldedoffset.jl`, negative address constants kept off split loads) runs on every device: it costs ~9% of Qwen-Image's VAE decode on the RTX 4000 Ada (1.138 s without, 1.246-1.254 s with) and ~1% on the 7900; decide whether it becomes a per-device compile setting through `config(dev, f)` (F3) | phase 1 | open | A/B 2026-10-15, `experiments/bugfix/env-newest/BASELINE.md` |
| F32 | the default group size `config(dev, f)` picks for 2-D and 3-D ranges: today one row of 64; measure a 2-D tile (16 x 16, as GPUFiltering passes) against it on the image kernels before choosing (P1.A21 keeps a caller's size either way) | phase 1 | open | not measured |
| F33 | Lava drops user atomic orderings: `demote_atomic_accesses!` (`Lava/src/compiler/target.jl`) makes every acquire/release load and store a plain one, and `fence` does not compile (`Unsupported LLVM instruction: FenceInst`); measured 2026-10-19 on the 7900 (`experiments/kernelforge/probe_lava_ordering.jl`). No kernel in the stack relies on it yet; all four desktop Vulkan devices report `vulkanMemoryModelDeviceScope` | phase 1 | open | fix: demote only `unordered`, emit `OpAtomicLoad`/`OpAtomicStore` and `OpMemoryBarrier`, a cross-workgroup message-passing test |
| F34 | NVIDIA kernels in PTX inside Vulkan command buffers (`VK_NV_cuda_kernel_launch`, provisional): `cp.async` ran correctly on Vulkan device addresses on the RTX 4000 Ada (`experiments/interop/`); a separate CUDA context costs 80-175 µs per hand-off, HIP imports Vulkan memory but no semaphores (ROCm 7.2.1). Whether a PTX kernel node goes into the plan is open (Simon, 2026-10-16) | phase 10 | open | |
| F35 | B5.1 (operations take the graph, groups only for repeated launches) leaves model-level loops open: examples.jl 19 (diffusion, `transformer(step, …)`) and 20 (decode until a stop token, `decodestep!(dec, …)`) pass a group into model functions | phase 0 | done | not a gap (Simon, 2026-10-19): B4.2 already answers it; what is repeated is what `dispatch!(group, …)` gets, so model code passes the group to its dispatches and a call without the group lands in the graph (decisions.md B5.1) |
