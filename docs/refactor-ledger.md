# Refactor ledger

Every add and delete item of every phase of `recording-plan.md`, and every
item found while working, with its status and evidence. Every session starts
by reading the plan and this ledger.

Separate four facts when updating a row: the target contract, what the current
baseline already implements, existing evidence, and what has been verified on
the replacement. An open replacement item does not mean the baseline feature,
consumer migration or benchmark is absent. Check the
[evidence index](../experiments/review-2026-10-20/evidence-index.md) before claiming
missing work. Found-while-working rows that settle a design choice must identify
that scope; a settled choice does not close its phase's implementation gates.

- **Status:** `open`, `doing`, `done`, `blocked`, `n/a`. `done` needs evidence:
  the test that proves a runtime change and the commit; a documentation-only
  item needs its committed artifact. Historical documentation rows retain their
  original completion evidence and do not close replacement runtime gates. `blocked` needs the blocker with
  evidence (the forbidden-moves alternative: stop and report). `n/a` needs the
  reason.
- **Failure evidence:** a failed implementation/probe first requires debugging
  and a fix. A fallback or design change needs the plan's debug-first root-cause
  evidence; a failing run alone cannot settle that decision.
- **Baseline:** occurrences at Mantle 5c2d4c6 in `src/`, `ext/` and `test/`,
  counted with `git grep -c -E '(^|[^A-Za-z0-9_])NAME($|[^A-Za-z0-9_!])'`.
  Counts include comments and tests that pin the old behaviour; a delete item
  is done when the name is gone from the device types that joined, except
  where the plan keeps it for a later phase (noted on the item).
- **Ids** are stable: `P<phase>.<kind><n>`, kinds A add, D delete, R rename,
  W rewrite a test, G green, T test, B bug, C comments, F found while working.
  New items are appended to their phase with the next free number. An item
  that moves to another phase or merges into another is removed here and
  added there; an item the design no longer has is removed.
- **Marks:** "(not in the plan: confirm)" for an item the plan does not list.
- **A phase is done** when every item is `done` or `n/a`, its green tests
  pass on every joined device type, its completed slice ran against the frozen
  baseline (phases 1–10), or the full migrated consumers ran (phases 11–12),
  an independent review of its diff
  against the plan, `api.md` and this ledger found no forbidden move (or its
  findings are fixed), and the comparable workload re-measured interleaved shows no
  regression beyond noise. Full consumer gates apply at phases 11–12. These gates are the last items of every phase.

## Phase 0. Docs, comments, tests, bugs

### Docs

| id | item | baseline at 5c2d4c6 | status | evidence |
|---|---|---|---|---|
| P0.A1 | `docs/recording-plan.md`: the target state only | written | doing | fixed choices and native qualification gates in plan/execution-contracts.md; review committed in 45021c6; subsequent consolidation-order changes remain in progress |
| P0.A2 | `docs/internals.jl`, `docs/examples.jl`, `docs/resizing-and-raytracing.jl` consistent with the plan | written | doing | |
| P0.A3 | `docs/api.md`: every type and verb with its contract, ownership, invariants, corner cases, driver entry points, allowlists, vocabulary fates | written | doing | |
| P0.A4 | `docs/refactor-ledger.md` (this file) | written | doing | document consistency pass committed in 45021c6; replacement gates and implementation consolidation remain open |
| P0.D1 | delete `docs/design.md`, `docs/submission-refactor.md`, `docs/backend-independence.md`, `docs/mantle-owns-it.md`, `docs/dispatch-api.md`; repoint the 9 references in `src/` and `test/` to the plan; move the two contents that only they hold (`vulkan/runtime/command.jl:404`'s measurement, `test/runtests.jl:73`'s clone note) into a comment or the plan (not in the plan: confirm) | 5 files tracked | done | a81bf41; deleted; their references in src/ and test/ point to recording-plan.md and api.md |
| P0.D2 | remove the review material (`docs/recording-plan-review.md`, `experiments/plan-review/`) | - | done | a81bf41 historical removal; execution-contracts.md is now a normative companion, new review evidence lives under experiments/review-2026-10-20 |
| P0.A5 | the docs committed | - | done | 034ae50 (the plan, api.md, internals, the suite) |

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
| P0.T1 | interleaved recording (THR-02): complete coverage in `test/suite/areas/threads.jl` as the plan describes it (Phase 0) | - | open | |
| P0.T2 | builder ownership (THR-03): complete coverage in `test/suite/areas/threads.jl` as the plan describes it (Phase 0) | - | open | |
| P0.T3 | pure recording (THR-04 counters plus rejecting live-state guard; qualify bypass audit): complete coverage in `test/suite/areas/threads.jl` as the plan describes it (Phase 0) | - | open | |
| P0.T4 | compile/record once (GR-01): complete coverage in `test/suite/areas/graphs.jl` as the plan describes it (Phase 0) | - | open | |
| P0.T5 | window resize/generation retirement: complete coverage in `test/suite/areas/windows.jl` as the plan describes it (Phase 0): beyond the capacity the graph recompiles once, at its next run (it no longer throws) | - | open | |
| P0.T6 | steady-state submit only: complete coverage in `test/suite/areas/run.jl` as the plan describes it (Phase 0) | - | open | |
| P0.T7 | queued resize and stage consistency: complete coverage in `test/suite/areas/resize.jl` as the plan describes it (Phase 0) | - | open | |
| P0.T8 | explicit devices and registry: complete coverage in `test/suite/areas/devices.jl` as the plan describes it (Phase 0) | - | open | |
| P0.T9 | thread-independent lifecycle (THR-01): complete coverage in `test/suite/areas/threads.jl` as the plan describes it (Phase 0) | - | open | |
| P0.T10 | in-flight finalization/holds: complete coverage in `test/suite/areas/holds.jl` as the plan describes it (Phase 0) | - | open | |
| P0.T11 | cross-plan order and negative control: complete coverage in `test/suite/areas/ordering.jl` (with its negative control) as the plan describes it (Phase 0) | - | open | |
| P0.T12 | memory return after free: complete coverage in `test/suite/areas/free.jl` as the plan describes it (Phase 0) | - | open | |
| P0.T13 | host cuts/re-entry/failure: complete coverage in `test/suite/areas/host.jl` as the plan describes it (Phase 0) | - | open | |
| P0.T14 | eager/graph shared path: complete coverage in `test/suite/areas/eager.jl` as the plan describes it (Phase 0) | - | open | |
| P0.T16 | check: vocabulary frozen to `api.md` section 4, over `src/vulkan`, `src/metal`, `ext/`, with arity (replaces `test/test_backend_vocabulary.jl`); starts `@test_broken` with the violations listed | - | open | |
| P0.T17 | check: driver calls in one place (`api.md` section 8); starts `@test_broken` with the violations listed | - | open | |
| P0.T18 | check: no mutable globals outside `api.md` section 9; starts `@test_broken` with the violations listed | - | open | |
| P0.T19 | check: no probes outside `api.md` section 9; starts `@test_broken` with the violations listed | - | open | |
| P0.T20 | check: Aqua and JET on core; starts `@test_broken` with the violations listed | - | open | |
| P0.T21 | check: two devices open for the whole suite; starts `@test_broken` with the violations listed | - | open | |


| id | item | baseline at 5c2d4c6 | status | evidence |
|---|---|---|---|---|
| P0.T22 | nonoverlapping runs and negative control: complete coverage in `test/suite/areas/run.jl` as the plan describes it (Phase 0; rule 9), with its negative control | - | open | |
| P0.T23 | run host-time scaling: complete coverage in `test/suite/areas/run.jl` as the plan describes it (Phase 0): 100, 2 500, 25 000 and 50 000 named records, against the phase 3 target | - | open | |
| P0.T24 | eviction and restore: complete coverage in `test/suite/areas/eviction.jl` as the plan describes it (Phase 0) | - | open | |
| P0.T25 | composite fuzz: complete coverage in `test/suite/areas/fuzz.jl` as the plan describes it (Phase 0) | - | open | |
| P0.T26 | write RayMakie's compile-counter session as the plan describes it (Phase 0); green as P11.G2 | - | open | |
| P0.T27 | check: launch kernels never read cells (only head, prepare, the TLAS offsets and records kernels and `lengthof` writers have a `CellRead` use), over every plan the suite compiles (api.md section 6, derived; not in the plan: confirm, G16) | - | open | |
| P0.T28 | GPURef slots: complete coverage in `test/suite/areas/gpuref.jl`: `r[] = v` before the first run reaches it; a store during a run reaches the next submission that touches the slot; two plans and a group of draws baking one slot read the stored value | - | open | |
| P0.T29 | view alias barriers and negative control: complete coverage in `test/suite/areas/views.jl` as the plan describes it (Phase 0): a write through an array and a read through a view, `reshape` or storage of it in one graph get a barrier; negative control: an object-keyed resource table (today's `Graph.ids`) | - | open | found porting GPUFiltering and Paperhawk,: on Mantle 94c4258 a `Buffer`, `storage(b)` and a reshape are three resources and the plan has no barrier between the write and the read |

### Bugs to fix first, each with a regression test that fails before the fix

| id | item | baseline at 5c2d4c6 | status | evidence |
|---|---|---|---|---|
| P0.B1 | `flush_stall_report` reads fields core's `Submission` does not have (`vulkan/runtime/command.jl`) | - | open | |
| P0.B2 | `gemm_cm2!`/`gemm_cm2_sg!` called as kernel factories (`vulkan/array/gemm_cm2.jl`) | - | open | |
| P0.B3 | a conditional region can drop a barrier (`graph/build.jl` docstring, `sync/transition.jl` rule) | - | open | |
| P0.B4 | Metal waits call `Metal.synchronize` (task's queue, not the device's): `waitfor`/`waitidle`, `upload_texture_data!`, `readback_framebuffer`, `metal/hwtlas.jl`, `metal/record.jl` | - | open | |
| P0.B5 | walked Metal plans keep compile-time arrays with `patchable = true` and no move listener (`metal/record.jl`) | - | open | |
| P0.B6 | readback and copy leave images in TRANSFER_SRC before a render pass that loads them (`vulkan/graphics/`) | - | open | |
| P0.B7 | validation checks read `VK_CONTEXT_REF[]` inside functions that hold a context | - | open | |
| P0.B8 | arena regrowth dereferences tenant weak references without a `nothing` check (`memory/pool.jl`) | - | open | |
| P0.B9 | `append!` on a `LavaArray` view ignores the view's offset (`vulkan/array/gpuarrays.jl`) | - | open | |
| P0.B10 | unlocked `live_buffers` delete from a finalizer | - | open | |
| P0.B11 | H.265 bitstream alignment | - | open | |
| P0.B12 | the two disagreeing texture format tables | - | open | |
| P0.B13 | `Ref` without `GC.@preserve` and UserID vs `instance_id` in `metal/trace.jl` | - | open | |
| P0.B14 | device features enabled without querying them | - | open | |
| P0.B15 | regression test for `destroy_pool!` calling `empty!` on an `Inbox` (fixed in 2c7dc14) | - | open | |

### Baseline

| id | item | baseline at 5c2d4c6 | status | evidence |
|---|---|---|---|---|
| P0.A6 | whole-plan benchmark (`experiments/header_indirection/plan_*.txt` shape) on the joined devices | - | open | |
| P0.A7 | each suite's model calls timed (JuliaVision runners, Hikari, RayMakie, VideoEditor) | - | open | |
| P0.A8 | the new-API suite `test/suite/` (each case with its ledger phase; later phases skipped) and its test hooks in Mantle: `diagnostics`/`resetdiagnostics!` (plan, pipeline and kernel compiles, records, submissions per channel, allocations, host waits), `reserved`, `mapped`, `liveuploads`, `livecells`, `pendingof`, `steplog!`/`steplog`, `checklocks!`; `Window` and `Swapchain` parameterized by their device's type and `inner(x)` (so the suite's `FaultDevice` intercepts window verbs); `caps(dev).graphics`; `FaultDevice` in the suite | - | open | |
| P0.A9 | before/after benchmarks through the consumers' unchanged APIs (`/sim/Programmieren/VideoEdit/benchmarks/`: Makie on GLMakie, RayMakie raster and traced; SAM 2, QwenImage, TRELLIS; VideoEditor; RayDemo `run_benchmarks.jl`); baseline environment `VideoEdit-baseline` kept on each worker; the comparison alternates both environments in fresh processes, with an A/A pair as the noise floor | - | open | existing benchmarks/jobs.jl and consumer acceptance baselines already provide the harness/workloads; run replacement delta against frozen baseline; evidence-index.md |

| id | item | baseline at 5c2d4c6 | status | evidence |
|---|---|---|---|---|
| P0.A10 | consolidate verified duplicate implementation bodies before the foundation; map larger duplicated algorithms/state and every caller to one core owner and a deletion gate; each replacement slice supersedes named old copies rather than reproducing them | seven source audits and source-disposition.tsv exist | doing | consolidation-order.md and evidence-index.md record owners, deletions, size and qualification. Open: cache retention/holds and CodeInstance compute entries, token retirement, Builder execution and TLAS pending-change/build ownership. Native Metal/ROCm qualification remains explicit per batch. |

| id | item | baseline at 5c2d4c6 | status | evidence |
|---|---|---|---|---|
| P0.G90 | gate: the phase's green tests pass on every joined device type | - | open | |
| P0.G91 | gate: one real use ran (editor opened and played, a RayMakie frame, a model call) | - | open | |
| P0.G92 | gate: independent review of the phase's diff against plan, api.md, ledger: no forbidden move | - | open | |
| P0.G93 | gate: baseline re-measured interleaved, no regression beyond noise | - | open | |

## Phase 1. One compile path

| id | item | baseline at 5c2d4c6 | status | evidence |
|---|---|---|---|---|
| P1.A23 | foundational vertical slice in the isolated replacement checkout with final Device/Graph/MantleArray/GraphDevice{D}, fixed regions, one channel, recordings, dispatch!, run!, host reads and suite hooks; declare the full type/verb surface needed to load faultdevice.jl; no adapters to old arrays/graphs | prerequisite moved forward from phases 2/3/6/8 | open | plan, Order of work fixes the dependency order; enumerate and run the phase-1 case gate |
| P1.A1 | `Kernel`/`DeviceKernel`/`KernelInfo`: one `Mantle.Kernel` per kernel at its `CodeInstance` (`GPUCompiler.cached_results`), filled in stages under its own lock: access at declaration, code at compile, per device the native handle and `KernelInfo` | - | open | |
| P1.A2 | `compilekernel(dev, kernel, config)`/`kernelinfo` on every joined device type, filling the `Kernel`; no device-type kernel cache | - | open | |
| P1.A3 | the per-device compile counter | - | open | |
| P1.A4 | node types with their origin | - | open | |
| P1.A5 | `compile!(s, node)` | - | open | |
| P1.A6 | the dependency graph kept and reduced, one hazard function shared with barriers | - | open | |
| P1.A7 | barrier merging for conditional kernels | - | open | |
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
| P1.D5 | delete the host-read `when!`: `hostcond` (`Pass.hostcond`) and `RecordingParts.conds`; the name `when!` is reused . Includes the former P1.D19 | 8 hits of `hostcond`, 3 files; 33 hits of `when!` (name reused) | open | |
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
| P1.D20 | delete the default query pool on every Vulkan plan and `timestamps` as a device hook for the split | - | open | |

| id | item | baseline at 5c2d4c6 | status | evidence |
|---|---|---|---|---|
| P1.W1 | rewrite `test/test_partitioned_recording.jl` | exists | open | |
| P1.W2 | rewrite `test/test_run_allocates_nothing.jl` | exists | open | |
| P1.W3 | rewrite `test/test_when.jl` | exists | open | |
| P1.G1 | green: pure-record compile counters (P0.T3); GR-01 compile/record once | - | open | |
| P1.G2 | green: `test/suite/areas/graphs.jl` (SAM 2 encoder and decoder recorded once each; today twice, `census_radv.txt`) | - | open | |

| id | item | baseline at 5c2d4c6 | status | evidence |
|---|---|---|---|---|
| P1.A14 | `caps(dev)` facts for `costs!` and `schedule!`: throughput per element type, memory bandwidth, transfer queue presence, cross-queue cost; the kernel compile configuration `config(dev, f)` (api.md 4.1, G3) | - | open | |
| P1.A15 | the submission budget as core's function of `caps(dev)` facts (job timeout, whether the device drives a display); `submissionbudget` is no longer a device-type hook | 8 hits of `submissionbudget`, 5 files | open | |
| P1.A16 | `submissions(g)`: segments and cuts, including those host nodes cause (plan, "The graph and its plan") | n/a (no definition at 5c2d4c6; the word occurs 49 times in prose) | open | |
| P1.D21 | delete `supportspredicate` (no conditional execution) | 13 hits, 8 files | open | |
| P1.D22 | delete conditional rendering (`vkCmdBeginConditionalRenderingEXT`, today for `repeat!`'s per-iteration flags, `vulkan/graph.jl:158, 1327`; the plan: "Mantle uses it for nothing") | 2 hits of `vkCmdBeginConditionalRenderingEXT`, 2 files; 1 of `cmd_begin_conditional_rendering_ext` | open | |
| P1.D23 | delete `callgroup` (launch geometry is one core calculation from `KernelInfo`; api.md Appendix A) | 21 hits, 10 files | open | |
| P1.D24 | delete `materialize!` (transients placed by `place!`; api.md Appendix A) | 16 hits, 9 files | open | |
| P1.D25 | delete `passbarriers` (`barriers!` is core's; api.md Appendix A) | 11 hits, 8 files | open | |
| P1.A17 | graphs stay across structure changes: declaring into a compiled graph and `delete!(g, group)` mark the plan, except groups of draws (P6.A16); `run!(g)` recompiles from the graph's nodes, retires the old plan, drops the holds of resources no node names after its last run; declarations take the graph lock (nested ones reuse it); holds counted per node; test: add and delete a group of launches on a running graph, the graph's device arrays keep their contents, no hold leaks | - | open | |
| P1.A18 | `when!(g or group, flag, kernel, args, ndrange)`: one launch; `flag` a `GPURef{Bool}` or one-element `Bool` array, enforced by the method's argument types; indirect, count × flag | - | open | |
| P1.A19 | `Group(g or group, name)`: a value and a place in its parent's order, declared into with `dispatch!`, `trace!`, `clear!`, `draw!`, `when!` and `Group` anywhere; GPU work only, by which methods exist for `Group`; `repeat!(group, n; while_nonzero)` over launches only (a draw, a clear or a group in a repeated group throws); `delete!(g, group)`; `sort!(group; by, lt, rev)`; not an owner; no mode state on the graph; compile flattens groups (a repeated one into a `LoopNode`); hazards between iterations derived from the group's accesses | - | open | |
| P1.A20 | `Plan.compiles` carries graph-wide compile count across replacement; native recording counts live in shared Diagnostics.records/prefixrecords/eagerrecords, with no duplicate Plan rerecord counter (execution-contracts.md §10) | - | open | |
| P1.A21 | a group size the caller gives (KA's `kernel(backend, workgroupsize)`, `dispatch!(…; group)`) is kept as given; `config(dev, f)` picks one only where none is given (plan, Kernels; the 2-D default is F32) | - | open | GPUFiltering launches 2-D image kernels with (16, 16) because the default for a 2-D range is one row of 64 |
| P1.A22 | access per argument stored in the `Kernel` at declaration; `AccessCache`/`accesscache` go | 2 (`accesscache` in api.md Appendix A) | open | |
| P1.D26 | delete `workgroupsize` as a vocabulary hook (`KernelInfo`'s group size; api.md Appendix A) | 110 hits (KA's `workgroupsize` included), 31 files | open | |
| P1.D27 | delete the `do`-block body forms of `repeat!` and `when!` (`repeat!(f, g, n…)`, `when!(f, g, …)`); `repeat!(g, kernel, args, ndrange; max, while_nonzero)` stays | 40 hits of `repeat!(…) do`/`repeat!(g…`, 11 files at f6e359a | open | |

| id | item | baseline at 5c2d4c6 | status | evidence |
|---|---|---|---|---|
| P1.G90 | gate: the phase's green tests pass on every joined device type | - | open | |
| P1.G91 | gate: a representative program exercises this completed slice through the final API against the separate frozen baseline | - | open | |
| P1.G92 | gate: independent review of the phase's diff against plan, api.md, ledger: no forbidden move | - | open | |
| P1.G93 | gate: comparable completed-slice baseline re-measured interleaved, no regression beyond noise | - | open | |

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
| P2.A10 | the pressure policy in `acquire!` with `makeroom!` folded in: drop idle native cache entries (§13), unmap idle arena pages, trim empty blocks, collect garbage, wait for retiring storage, then throw `OutOfDeviceMemory` (the eviction step is P4.A13). Pressure never unmaps pages beyond an array's size and never compacts acceleration-structure storage | - | open | |
| P2.A11 | one meaning of capacity (`budget(dev)`) | - | open | |

| id | item | baseline at 5c2d4c6 | status | evidence |
|---|---|---|---|---|
| P2.D1 | delete `vk_alloc` | 20 hits, 9 files | open | |
| P2.D2 | delete `MemoryPolicy` | 11 hits, 3 files | open | |
| P2.D3 | delete `reclaim!(::SubmitChannel)` only; `reclaim!(p::Pool, dev)` (`memory/pool.jl:765`) is the pool's release step | 87 hits of `reclaim!` (all methods), 18 files | open | |
| P2.D4 | delete `gemm_split_scratch` | 5 hits, 3 files | done | core-owned independent partials; overlapping programs/replay qualified |
| P2.D5 | delete `gemm_split_retired` | 2 hits, 2 files | done | no retained growth history; native region retirement qualified |
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
| P2.G1 | green: `test/suite/areas/holds.jl` | - | open | |
| P2.G2 | green: `test/suite/areas/free.jl` | - | open | |

| id | item | baseline at 5c2d4c6 | status | evidence |
|---|---|---|---|---|
| P2.A14 | tables with free lists (material, medium, light, texture) | - | open | |

| id | item | baseline at 5c2d4c6 | status | evidence |
|---|---|---|---|---|
| P2.A16 | the last hold under the array's record lock: storage swapped for `Released`, the old storage retired with the last tokens; dropping pending changes, the eviction list and the cell are P4.A35 (phase 4) | n/a (`Released` 0 hits) | open | |
| P2.R1 | the vocabulary's `capacity(dev)` (the device budget hook, `vulkan/graph.jl:107`; Metal's `recommendedMaxWorkingSetSize`; `AMDGPU.free`) becomes `budget(dev)` (api.md Appendix A) | 92 hits of `capacity` (prose and `capacity = n` included), 32 files | open | |
| P2.R2 | `maxalloc(dev)` becomes a `caps(dev)` field, the largest single allocation (api.md 4.1, G3) | 22 hits, 10 files | open | |
| P2.D16 | delete `alignment` as a vocabulary hook (part of `rawalloc`'s constraint; api.md Appendix A) | 100 hits (the word in prose included), 38 files | open | |
| P2.D17 | delete `bufferusage` (the memory kind's usage constraint; api.md Appendix A) | 15 hits, 9 files | open | |
| P2.D18 | delete `extrausage` (as `bufferusage`) | 8 hits, 5 files | open | |
| P2.D19 | delete `imageusage` (the `Images` kind's usage constraint) | 9 hits, 5 files | open | |

| id | item | baseline at 5c2d4c6 | status | evidence |
|---|---|---|---|---|
| P2.G90 | gate: the phase's green tests pass on every joined device type | - | open | |
| P2.G91 | gate: a representative program exercises this completed slice through the final API against the separate frozen baseline | - | open | |
| P2.G92 | gate: independent review of the phase's diff against plan, api.md, ledger: no forbidden move | - | open | |
| P2.G93 | gate: comparable completed-slice baseline re-measured interleaved, no regression beyond noise | - | open | |

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
| P3.A10 | arena parts held by a run until its last stage is submitted (`holdparts!`) | - | open | |
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
| P3.D52 | delete `LavaBackend`'s upload queue and `external.jl`'s direct use of `compute_queue` | 7 hits of `compute_queue` | open | |
| P3.D53 | delete the head barrier | - | open | |
| P3.D54 | delete `drain!(::SubmitChannel)` (core's `drain!(pool.inbox)` stays, internals.jl) and `flush!` as user-facing verbs; `acquire!(::SubmitChannel)`, `release!(::SubmitChannel, r)`, `waitfor!(::Stamp)` | 60 hits of `drain!` (all methods), 16 files | open | |

| id | item | baseline at 5c2d4c6 | status | evidence |
|---|---|---|---|---|
| P3.W1 | rewrite `test/test_hold_lifetime.jl` | exists | open | |
| P3.W2 | rewrite `test/test_batch_queue_lifetime.jl` | exists | open | |
| P3.W3 | rewrite `test/vulkan/test_closed_command_buffers.jl` | exists | open | |
| P3.W4 | rewrite `test/vulkan/test_hold_trace.jl` | exists | open | |
| P3.W5 | rewrite `test/test_holdleaves_stops_at_tlas.jl` | exists | open | |
| P3.G1 | green: threads.jl THR-01/02/03 plus P0.T3 pure recording; run.jl (headless), ordering.jl, host.jl without its resize part | - | open | |

| id | item | baseline at 5c2d4c6 | status | evidence |
|---|---|---|---|---|
| P3.A12 | host waits only under the graph lock: no sync-state, channel, pool or window lock is held while the host waits; the graph lock may be (shared arena parts, a host node's inputs, the swapchain acquire, `acquire!`'s pressure steps); no store waits for a run, value `GPURef`s included | - | open | |
| P3.A13 | segments as lists of parts: recordings written once and `BuildRecording`s (plan phase 3) | - | open | |
| P3.A14 | `lockgraph(g)` throws when the calling task holds an arena part of `g`'s plan; `holdparts!` throws when the task would wait for a part while holding one with a higher id | - | open | |
| P3.A15 | `Token = (channel, value)`; core's `passed(::Token)`/`waitfor(::Token)` over the device-type verbs `passed(ch, value)`/`waitfor(ch, value)`; `fence(ch)` becomes a read of `SubmitChannel.next` (api.md 4.2) | 112 hits of `fence` (the word in prose included), 30 files | open | |
| P3.A16 | measure the real `run!` against the model of "Measurements this plan rests on" (target: within 3x; plan, submission-cost measurement gate) | - | open | |
| P3.G2 | green: `test/suite/areas/run.jl`; `test/suite/areas/run.jl`: the same host time per run for every number of named resources | - | open | |
| P3.D55 | delete the queue-slot fields (`QueueSlots` and `VulkanQueue.slots`, `vulkan/runtime/device.jl:154, 200`; plan, Queues) | 4 hits of `QueueSlots`, 1 file | open | |
| P3.D56 | delete `RecordingParts.batches` (plan, Core types) | 21 hits of `RecordingParts`, 9 files | open | |
| P3.D57 | delete `abandonframe!`, `abandonrecording!`, `abandonrun!` (the recording lifecycle is `Builder`/`finish` and the free list; api.md Appendix A) | 9 + 9 + 12 hits, 5 + 5 + 4 files | open | |
| P3.D58 | delete `makerecording`, `finishrecording!`, `resetrecording!`, `destroyrecording!`, `recorder` (`Builder(dev, ch)`, `finish(b)`, the free list, retirement by token; api.md Appendix A) | 6 + 12 + 7 + 5 + 41 hits (`recorder` includes prose), 4 + 4 + 4 + 4 + 15 files | open | |
| P3.D59 | delete `awaitwrites` (eager calls wait per sync state; api.md Appendix A) | 15 hits, 6 files | open | |
| P3.D60 | delete `emitpreparebarrier!` (prepare is a node; api.md Appendix A) | 7 hits, 4 files | open | |

| id | item | baseline at 5c2d4c6 | status | evidence |
|---|---|---|---|---|

| id | item | baseline at 5c2d4c6 | status | evidence |
|---|---|---|---|---|
| P3.A20 | `AccessRecord` → `SyncState`: per shared resource only the accesses of what is not a run (eager calls, host reads, vendor views, eviction copies, sparse binding, applied change steps), the last tokens of retired plans, pending changes, lent/freed | - | open | |
| P3.A21 | between graphs through the resource: `namers` per `SyncState` (plans with their use, replaced at `register!`/`unregister!`; composite members reach them through readers); a stage writing a shared resource waits for the other namers' complete last runs and fills the readers' inboxes; a stage waits for the writers' last runs of its inbox resources; waits kept for the run's later stages; `register!` seeds the plan's inbox, only re-locks (never recompiles) when plans change meanwhile; `unregister!` adds its last-run tokens to `retired` (composite members included); `delete!` drops the reader after the applying submission passed | - | open | |
| P3.A22 | inbox per plan: pending a change or recording a non-run access pushes the resource into the inboxes of the plans naming it (and naming composites built from it); a stage locks only its inbox resources (expanded to the members their pending builds read), its arena parts (stage 1), and the sync locks of itself, the plans sharing what it writes and the plans naming its inbox resources; no per-resource walk at run | - | open | |
| P3.A23 | plan sync lock in the lock order (after the window lock, before sync states); eager calls, evictions and storage release take the sync locks of the plans naming what they touch (`lockresources`) | - | open | |
| P3.A24 | "old or new, never mixed" holds per STAGE: a change arriving between a run's stages goes into the next submission | - | open | |
| P3.G90 | gate: the phase's green tests pass on every joined device type | - | open | |
| P3.G91 | gate: a representative program exercises this completed slice through the final API against the separate frozen baseline | - | open | |
| P3.G92 | gate: independent review of the phase's diff against plan, api.md, ledger: no forbidden move | - | open | |
| P3.G93 | gate: comparable completed-slice baseline re-measured interleaved, no regression beyond noise | - | open | |

## Phase 4. Arguments and sizes

| id | item | baseline at 5c2d4c6 | status | evidence |
|---|---|---|---|---|
| P4.A1 | Lava's entry wrapper: fields in the argument slots, block marked `NonWritable` | - | open | |
| P4.A2 | cell arena; device-array and resizable-transient cells shared by views; fixed graph transients use NoCell and compile-time placement | - | open | |
| P4.A3 | head and prepare nodes with the plan's slots as a resource | - | open | |
| P4.A7 | array state with storage kinds (`Region`, `Reserve`, `Committed`, `Evicted`, `Released`, `Unplaced`, `ArenaSlice`) and the first `resize!` (P4.A14) | - | open | |
| P4.A8 | pending changes on sync states with a device-wide sequence (`change!`, `pendone!`, `submitwith!`, `lockexpanded`) | - | open | |
| P4.A9 | dependents with registration after compile and marking at the change | - | open | |
| P4.A10 | `LengthOf` and `SizedBy` resolved at compile | - | open | |
| P4.A11 | dynamic launch extents for resizable arrays, plus explicit loop/when control; indirect where supported, capacity predication otherwise | - | open | |
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
| P4.D11 | delete `DeviceRange(count; max)`; the one-argument `DeviceRange(a)` is dropped too, `launchrange(a)` is the one name | 64 hits of `DeviceRange` (both forms), 18 files | open | |

| id | item | baseline at 5c2d4c6 | status | evidence |
|---|---|---|---|---|
| P4.W1 | rewrite `test/test_recorded_move_patch.jl` (the Metal one in phase 9) | exists | open | |
| P4.G1 | green: `test/suite/areas/resize.jl`, `test/suite/areas/host.jl` in full, `test/suite/areas/gpuref.jl` | - | open | |
| P4.G2 | green: the whole-plan benchmark within its noise of `plan_*.txt` | - | open | |

| id | item | baseline at 5c2d4c6 | status | evidence |
|---|---|---|---|---|
| P4.A13 | eviction of idle resources under pressure: `Evicted` storage, `Restore` change, plans marked through dependents (every naming plan registers). Evictable: no composite holds it, no handle dependent, no plan that names it running, not lent, nothing pending; accesses in flight allowed (the copy is ordered after them); the copy goes to pageable host memory (`HostCopy`), so the OS may compress or swap it; restores transfer directly from prepared host imports/registrations (F74/P4.A31; no speculative staged restore); `acquire!` waits for eviction copies outside every lock; the same on every device type, unified-memory ones included; acceleration structures not evicted; `restore!` of an already-evicted array with no prepared storage marks the plan for restore, then `Again` | - | open | |
| P4.A14 | the first `resize!`: in place when the new size fits the current storage (only a cell write); beyond it a large array on a device with sparse residency moves into a reserve, any other array into a pool region of twice the need (each later doubling a move) | - | open | |
| P4.A15 | `trim!(a)`: unmaps a reserve's pages beyond the size, pending; pressure never does | 23 hits of `trim!` (today's `trim!(pool, dev)`), 12 files | open | |
| P4.A16 | inline stores: the call copies the bytes into an upload region; the applying submission writes them inline with `emitstore!(b, dst, upload)` up to the `caps(dev)` limit (Vulkan `vkCmdUpdateBuffer`, 64 KiB at 4-byte offsets and sizes), and a larger store is a copy from the upload region; no staging is allocated at apply. Today's core `emitstore!` (`graph/kalaunch.jl:816-823`, through `emitinline!`) gives the builder verb its name | 11 hits of `emitstore!`, 3 files | open | |
| P4.A18 | later segments of a stage that touch records whose changes were taken wait on the first segment's token | - | open | |
| P4.A19 | `lockexpanded` expands composite reads transitively (TLAS → BLASes and range arrays, BLAS → geometry); the run's waits and recorded accesses use the expanded uses (each composite's method in P5.A18) | - | open | |
| P4.A20 | `copy!(a, data)` with a new length does not copy the old contents | - | open | |
| P4.A21 | `reservesize` and `sparsethreshold` as core functions of `caps(dev)` fields (sparse page size, sparse address space), not device verbs | n/a (0 hits each) | open | |
| P4.G3 | green: `test/suite/areas/eviction.jl` (plan phase 4) | - | open | |
| P4.D12 | delete `argbytes`, `argidentity`, `argsize` (core's `bindings!` lays out the argument block; api.md Appendix A) | 19 + 7 + 22 hits, 5 + 5 + 8 files | open | |
| P4.D13 | delete `passedas` (argument layout is core's; api.md Appendix A) | 21 hits, 5 files | open | |
| P4.D14 | delete `indirectindex`, `indirectslot` (launch commands are slots core lays out; api.md Appendix A) | 13 + 7 hits, 7 + 3 files | open | |
| P4.D15 | delete `devicesized` (`LengthOf` resolved at compile; api.md Appendix A) | 15 hits, 7 files | open | |
| P4.D16 | delete `patchable` (with the pushed patch mechanism; api.md Appendix A) | 16 hits, 6 files | open | |
| P4.D17 | delete `storebytes!` and `upload!` (stores are pending changes; api.md Appendix A) | 10 + 44 hits, 4 + 20 files | open | |
| P4.D18 | delete `hostwritten` graph arrays | 64 hits, 14 files at f6e359a | open | |
| P4.D19 | delete `persistent = true` on graph arrays | 1 hit, 1 file at f6e359a | open | |

| id | item | baseline at 5c2d4c6 | status | evidence |
|---|---|---|---|---|
| P4.A30 | `change!(apply, uses)`: the call only enqueues (requested size for `length(a)`; a store's data copied into an upload region acquired at the call, the one allocation a call makes); no other allocation, no retry, never a wait for a run at the call | - | open | |
| P4.A31 | the applying submission allocates before its locks (`prepare!`: storage or pages for a Resize, a structure's capacity, restore storage) and applies in sequence order (`submitwith!` with `stepsof`); Again if what it prepared no longer fits; every submitter on this path (`submitapplying`, run stages) | - | open | |
| P4.A32 | `resizable` flag set by the first `resize!` call or by `capacity = n` at creation (compile launches indirectly from then on); handle dependents get their parts marked for re-recording when a move is applied (execution-contracts.md §§3/7; compatible moves require no shader compile or host wait) | - | open | |
| P4.A33 | a `GPURef` is one slot in device memory whose address plans and groups of draws bake; `r[] = v` a scheduled store | - | open | |
| P4.A34 | `length(a)`/`size(a)` the real length; for a kernel-written length a wait and a read of the cell, a throw during the declaration of the graph that writes it; `resize!` beyond the capacity allowed, `capacity` like `sizehint!` | - | open | |
| P4.A35 | the last hold's phase-4 half: pending changes dropped (their upload regions retired), the array off the eviction list, the cell released | - | open | |
| P4.T1 | benchmarks before P4.A32: re-record cost at 100, 1 000 and 5 000 draws, and pulled indices against indexed draws, on NVIDIA, AMD and the M5 (`experiments/draw_bench/`) | - | open | |
| P4.T2 | measure on Linux (NVIDIA, RADV) and the M5 (plan decision 14 / execution-contracts.md §12): whether `Readback` memory and Metal shared buffers are pinned; whether a device type can write a page-aligned host allocation directly (Vulkan `VK_EXT_external_memory_host`, Metal no-copy buffers) or needs a staging copy; how much physical memory evicting a writable array frees once the OS compresses or swaps the copy | - | open | |
| P4.G90 | gate: the phase's green tests pass on every joined device type | - | open | |
| P4.G91 | gate: a representative program exercises this completed slice through the final API against the separate frozen baseline | - | open | |
| P4.G92 | gate: independent review of the phase's diff against plan, api.md, ledger: no forbidden move | - | open | |
| P4.G93 | gate: comparable completed-slice baseline re-measured interleaved, no regression beyond noise | - | open | |

## Phase 5. Acceleration structures and textures

| id | item | baseline at 5c2d4c6 | status | evidence |
|---|---|---|---|---|
| P5.A1 | builds and refits as operations with `TraceRead`/`TraceBuild` declarations | - | open | |
| P5.A2 | the core `TLAS` (hardware and software) with instance ranges and the device-side range table | - | open | |
| P5.A3 | storage for a doubling capacity; host-known totals build/refit exactly, device-written totals use the fixed capacity with mask-0 padding; `BuildRecording` on Vulkan, count from a cell on Metal | - | open | |
| P5.A4 | builds with all their uses | - | open | |
| P5.A5 | the texture table | - | open | |
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

| id | item | baseline at 5c2d4c6 | status | evidence |
|---|---|---|---|---|
| P5.A8 | BLAS storage sized from its geometry with a capacity that doubles: a new topology a Build into it (a `MoveAccel` beyond the capacity, the new address through its cell), moved vertices with the same topology an Update; TLASes over a BLAS pend an Update when its address or contents change; counts by the TLAS rules (`BuildRecording`, `BuildAccel`) | - | open | |
| P5.A9 | readers: structure updates pended by host changes and by runs that write their arrays (`pendupdate!`) | - | open | |
| P5.A10 | per-range data (geometry arrays, material index) through the range table; no scene-wide concatenation | - | open | |
| P5.A11 | BLASes shared by geometry arrays; a `mask` per range entry for visibility (hide = mask 0, a refit; `active = false` only for a deleted range, a Build) | - | open | |
| P5.D10 | delete the host triangle concatenation: `tri_gpu` | 32 hits, 9 files | open | |
| P5.D11 | delete `off_gpu` | 34 hits, 10 files | open | |
| P5.D12 | delete `triangle_data` | 19 hits, 6 files | open | |
| P5.D13 | delete `HardwareAccel` | 32 hits, 5 files | open | |

| id | item | baseline at 5c2d4c6 | status | evidence |
|---|---|---|---|---|
| P5.R1 | rename the plan-internal `refit!` (`graph/build.jl`), which collides with the TLAS `refit!` | - | open | |
| P5.A12 | composites hold and never allocate: TLAS, BLAS, tables, render objects built from arrays and accel storage; only holds and change commits retire; `TLAS` and `BLAS` finalizers push their hold drops through the pool's inbox (render objects in P11.A16) | - | open | |
| P5.A13 | `BuildRecording`s marked by the change that alters what they recorded; the run re-records only marked ones; nothing compared at run | - | open | |
| P5.A14 | build coalescing per structure: any Build → Build, else Update; `builtcount` = the count of the last Build, used by every Update | n/a (`builtcount` 0 hits) | open | |
| P5.A15 | kernel-written counts: a TLAS total built over capacity with mask-0 records and valid BLAS references on every device type; a BLAS length built over geometry capacity with degenerate primitives past the length, so later changes are Updates | - | open | fixed baseline in api.md 2.6 and execution-contracts.md §7; replacement/native qualification remains |
| P5.A16 | a change a running plan cannot follow is left pending by later stages; other submitters return `WaitFor` outside their locks until the run has submitted its final stage; never wait at the mutating call, throw from a host function to avoid a wait cycle; no `except` | - | open | fixed rule in api.md §7 and resizing-and-raytracing.jl B.3; replacement race/deadlock tests remain |
| P5.A17 | reader snapshots: an array's readers are an immutable snapshot replaced under its lock and counted per reader; changes to an array with readers lock it with its readers and pend the readers' builds; the submission that applies them prepares reader capacity before its locks; a run pends `pendupdate_locked!(reader, array)` after each accepted segment, including partial-stage failure | - | open | |
| P5.A18 | `lockexpanded` methods for the TLAS (BLASes, range arrays) and the BLAS (geometry); the mechanism is P4.A19 | - | open | |
| P5.A19 | a TLAS total past the device's `maxInstanceCount` throws at the call that raises it (`push!`, a source's `resize!`/`copy!`); capacity capped at it | - | open | |
| P5.A20 | `resize!` of a texture keeps the texels where the old and new sizes overlap | - | open | |
| P5.G2 | green: `test/suite/areas/fuzz.jl` (plan phase 5) | - | open | |
| P5.D14 | delete `AdaptedAccel` (with Metal's HWTLAS bookkeeping; api.md Appendix A) | 82 hits, 16 files | open | |
| P5.D15 | delete `remakeimage!` (a new size is a new image under a new index; api.md Appendix A) | 9 hits, 6 files | open | |
| P5.D16 | Raycore: delete the TLAS finalizer that finalizes arrays a recording and `StaticTLAS` still use (`instanced-bvh.jl:372, 400-411`), `blas_array`, and host readbacks and `synchronize` between build steps (`resizing-and-raytracing.jl` D.1) | - (Raycore `sd/with-texture`, tree 9accd25) | open | |

| id | item | baseline at 5c2d4c6 | status | evidence |
|---|---|---|---|---|
| P5.G90 | gate: the phase's green tests pass on every joined device type | - | open | |
| P5.G91 | gate: a representative program exercises this completed slice through the final API against the separate frozen baseline | - | open | |
| P5.G92 | gate: independent review of the phase's diff against plan, api.md, ledger: no forbidden move | - | open | |
| P5.G93 | gate: comparable completed-slice baseline re-measured interleaved, no regression beyond noise | - | open | |

## Phase 6. One array type and operations

| id | item | baseline at 5c2d4c6 | status | evidence |
|---|---|---|---|---|
| P6.A1 | complete the `MantleArray <: AbstractGPUArray` interface on devices and `GraphDevice{D}`; the types and minimal working entry points are P1.A23 | - | open | |
| P6.A2 | core's KA backend and GPUArrays interface (`storage`, `copy`, `derive`, `mapreducedim!`) | - | open | |
| P6.A3 | operations as methods calling `dispatch!(location(a), …)` | - | open | |
| P6.A4 | the selection table (the Vulkan GEMM/GEMV selection out of `vulkan/array/gemm.jl`, `basealignment` in the key) | - | open | |
| P6.A5 | optional rewriters | - | open | |
| P6.A6 | `devicearg` | - | open | |
| P6.A7 | scoped vendor views: core's `withview` and the foreign channel (G11; each device type's constructor with its phase, P9.A9, P10.A7); `withview` takes a hold, applies pending changes and restores first (the stream waits on that submission) and drops the hold at the end with the stream's token | - | open | |

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
| P6.G1 | green: `test/suite/areas/eager.jl` | - | open | |
| P6.G2 | green: GPUArrays' TestSuite on every joined device type | - | open | |
| P6.G3 | green: `test/suite/areas/views.jl` | - | open | |

| id | item | baseline at 5c2d4c6 | status | evidence |
|---|---|---|---|---|
| P6.A9 | `attribute(a, i) = a[min(i, length(a))]` (1-based; the benchmark's 0-based form `a[min(i, n-1)]`): one argument type for a value and per-element attributes | - | open | |
| P6.A10 | graphics and ray-tracing pipelines from the device type's cache under one key (`compiledraw`, `compiletrace`); `npipelines` counts pipelines on every device type | - | open | |
| P6.A11 | images as resources (`Image`, stores, copies, `Array(img)`), one format table per device type (`imageformat`), the sampler table (`samplerentry!`) | - | open | |
| P6.A12 | `clear!`/`draw!`/`when!` on a graph or group as nodes into a target (window, image, or tuple); compile forms the render passes (`formpasses!`: clears and draws into the same targets with no reader between them; load operations from clears, store operations from liveness); index arrays through `bindindices!`; `when!` draws, one per call with a `Bool` flag (indirect, the count multiplied by the flag); eager `draw!(dev, …)` as its own pass; the scene rectangle as one array per scene applied in the vertex stage | - | open | |
| P6.A16 | groups of draws as pieces: a group whose nodes are draws into one pass is compiled and recorded on its own; a new one at a place inside a pass with its targets, draws declared into one, its `delete!` and a `sort!` of such groups are applied by the next run (`applychanges!`/`updatepieces!`: compile and record the piece, its argument region and cell-copy entries, its registration; the list part re-recorded), any other change recompiles; the head's cell-copy table a resizable array launched indirectly; a fixed barrier at every pass start; device-type verbs `recordpiece`, `emitpieces!` (Vulkan secondaries with dynamic rendering, Metal ICB per piece); gate: insert/delete latency at 100, 1 000, 5 000 plots satisfies RM-06 (2x the 100-plot median + 1 ms); the flat pass-list rewrite is linear and batched once per run, a visibility change records nothing, a large mesh's draw time equal to a pass with only that mesh | - | open | |
| P6.A17 | KernelAbstractions on graph arrays declares: `KA.get_backend(a)` is its GraphDevice, `KA.allocate` gives a transient; host reads of contents throw at their line; launch sizes recorded per node, a resize behind one throws at the next compile | - | open | |
| P6.A18 | `adapt(Mantle.Device(graph), x)`: zero-copy aliases the graph names, with a hold; `adapt(Mantle.Device, x)` uploads | - | open | |
| P6.A19 | `MantleArray{T,N,L}`: a view is the same type (no role parameter); read-only, unified memory (`unified = true`) and resizable are runtime properties of the resource | - | open | |
| P6.A20 | `readonly = true`: a reference to the host array for restores; writes throw at the call from the access `dispatch!` derives at declaration | - | open | |
| P6.A21 | bounds-checked kernels unless `@inbounds`; verify that Lava reports a device-side bounds error rather than faulting | - | open | |
| P6.A22 | `sort!(group; by, lt, rev)` reorders the groups in a group; for groups of draws a rewrite of the pass's list | - | open | |
| P6.A23 | resource identity: a resource is the array's state, shared by its views, reshapes and storage, never the object; an argument that reaches device memory but is not a `MantleArray` (a vendor array, an array escaping `withview`, a raw backend array) throws at declaration (plan, Arrays; api.md section 7). Replaces `Graph.ids` keyed by object and `resourceleaves!` taking every device-array leaf | - | open | |
| P6.A14 | tessellation as stage types | - | open | |
| P6.D40 | delete `pass!` (the hand-recorded path) | 22 hits, 9 files | open | |
| P6.D41 | delete `PassRecorder` | 10 hits, 2 files | open | |
| P6.D42 | delete `viewport!` | 10 hits, 3 files | open | |
| P6.D43 | delete `blit!` as an eager call (becomes an eager `draw!(dev, …)`) | 14 hits, 9 files | open | |
| P6.D44 | delete `SampledTexture` | 16 hits, 7 files | open | |
| P6.D50 | delete `transition_image!` | 17 hits, 9 files | open | |
| P6.D51 | delete `readback_framebuffer` | 35 hits, 8 files | open | |
| P6.D52 | delete `readback_target` | 21 hits, 4 files | open | |
| P6.D53 | delete `Attr` with a stored stride | 13 hits, 4 files | open | |
| P6.T1 | test: value ↔ per-element switch, another array object for an attribute (a new group of draws) and visibility on one screen: one pipeline, zero recompiles, pixels match a fresh render (not in the plan: confirm) | - | open | |

| id | item | baseline at 5c2d4c6 | status | evidence |
|---|---|---|---|---|
| P6.R1 | `compile_draw` becomes `compiledraw(dev, pipeline, argtypes, formats)` (api.md Appendix A) | 35 hits, 13 files | open | |
| P6.R2 | `makeimage` becomes `placeimage` behind `Image(dev, T, dims...)` (api.md Appendix A) | 7 hits, 5 files | open | |
| P6.D54 | delete the eager `mul!`/`norm` code in `vulkan/array/` (their kernels become candidates; plan, Eager work, "Goes") | - | open | |
| P6.D55 | delete `use` (plan, Operations, "Deleted"): at 5c2d4c6 only the export at `Mantle.jl:452` and comments remain | - (the word `use` is prose) | open | |
| P6.D56 | delete `argtype`, `devicebuffertype` (`devicearg(dev, a)` is the one conversion; api.md Appendix A) | 10 + 11 hits, 6 + 5 files | open | |
| P6.D57 | delete `deviceslice`, `deviceview` (`devicearg` resolves views in one place) | 7 + 28 hits, 5 + 13 files | open | |
| P6.D58 | delete `devicearray` (`MantleArray(dev, data)`) and `isdevicearray` (dispatch on `MantleArray`) | 41 + 11 hits, 11 + 6 files | open | |
| P6.D59 | delete `download` (host reads are `Array(a)` and copy nodes) | 58 hits (the word in prose included), 22 files | open | |
| P6.D60 | delete `indexbuffer` (indices are a `MantleArray` bound by `bindindices!`; zero-based wire indices, execution-contracts.md §7) | 20 hits, 9 files | open | |
| P6.D61 | delete `Framebuffer`, `colorimage`, `depthimage` (render targets are `Image`s and windows) | 50 + 9 + 9 hits, 16 + 6 + 6 files | open | |
| P6.D62 | delete `blittarget`, `copy_target!` (image copies are `copyto!` operations) | 14 + 7 hits, 7 + 4 files | open | |
| P6.D63 | delete `setviewport!` (the scene rectangle is a per-draw uniform) | 10 hits, 7 files | open | |

| id | item | baseline at 5c2d4c6 | status | evidence |
|---|---|---|---|---|
| P6.G90 | gate: the phase's green tests pass on every joined device type | - | open | |
| P6.G91 | gate: a representative program exercises this completed slice through the final API against the separate frozen baseline | - | open | |
| P6.G92 | gate: independent review of the phase's diff against plan, api.md, ledger: no forbidden move | - | open | |
| P6.G93 | gate: comparable completed-slice baseline re-measured interleaved, no regression beyond noise | - | open | |

## Phase 7. Windows and present

| id | item | baseline at 5c2d4c6 | status | evidence |
|---|---|---|---|---|
| P7.A1 | `Window` as a resource | - | open | |
| P7.A2 | swapchain generations with holds | - | open | |
| P7.A3 | acquire before the run's locks with binary semaphores and OUT_OF_DATE handling | - | open | |
| P7.A4 | per-image present segments | - | open | |
| P7.A5 | separate `presentnative!` after accepted rendering publication, outside record locks; independent retirement | - | open | |
| P7.A6 | `resize!(win, …)` as a mark the next run applies | - | open | |
| P7.A7 | window capacity and window-sized arrays and images placed at it (a sparse arena part where the device has sparse residency, mapped per generation for its size); a resize beyond the capacity recompiles the graph at its next run | - | open | |

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
| P7.W2 | rewrite `test/test_recordable_plans.jl` | exists | open | |
| P7.G1 | green: `test/suite/areas/windows.jl`, `test/suite/areas/run.jl` for a windowed graph with overlays | - | open | |

| id | item | baseline at 5c2d4c6 | status | evidence |
|---|---|---|---|---|
| P7.A8 | window-sized segments recorded per swapchain generation | - | open | |
| P7.A9 | screenshots: `Array(win)` through a second recording of the present segment | - | open | |
| P7.D14 | delete `readback_window` | 20 hits, 10 files | open | |

| id | item | baseline at 5c2d4c6 | status | evidence |
|---|---|---|---|---|
| P7.A11 | window-sized images recreated per swapchain generation at the actual size on the same arena memory (the segments drawing into them are P7.A8) | - | open | |
| P7.A12 | the swapchain image has one owner, the window: an image of a run that returns `Again` or throws goes back to it (`keepimage!`); only real images are kept | n/a (`keepimage!` 0 hits) | open | |
| P7.A13 | the window open verb openwindow(dev,width,height,options) (G13; the plan's "windows: open, acquire, present") | - | open | |
| P7.A14 | windows: any node of the presenting graph writes the window and the first graph declaring a write presents, a second throws; the binding ends at `free!`, collection or the removal of the nodes writing the window; an eager draw into a window throws; `Array(win)` throws when minimized, unpresented or from the graph's own host function, and wakes a waiting call with the error; `run!` returns `Submitted`/`Skipped` | - | open | |
| P7.R1 | `acquire_next_image!` becomes `acquirenext(sc)` (api.md Appendix A) | 32 hits, 12 files | open | |
| P7.R2 | `screenshot` becomes `Array(win)` | 17 hits, 8 files | open | |
| P7.D15 | RayMakie: delete `DrawBinding` use | 6 hits, 2 files (RayMakie e6e1aa9b0) | open | |
| P7.D16 | delete `Surface` | 18 hits, 8 files | open | |
| P7.D17 | delete `target_view` | 24 hits, 11 files | open | |
| P7.D18 | delete `target_image` | 15 hits, 7 files | open | |
| P7.D19 | delete `target_extent` | 36 hits, 13 files | open | |
| P7.D20 | delete `target_format` (was P6.D49; the format comes from `imageformat(dev, T)`) | 18 hits, 9 files | open | |
| P7.D21 | delete `beginframe!` (acquire and present are part of `run!`; api.md Appendix A) | 14 hits, 9 files | open | |
| P7.D22 | delete `present_frame!` (present uses separately retired `presentnative!` in run!) | 23 hits, 12 files | open | |
| P7.D23 | delete `currentimage` (the run holds the acquired image; the window owns it) | 7 hits, 5 files | open | |
| P7.D24 | delete `use_bindings!` (draw-binding code; bindings compiled with the plan; its last caller is the walked draw in `execute!`, P7.D13) | 16 hits, 10 files | open | |

| id | item | baseline at 5c2d4c6 | status | evidence |
|---|---|---|---|---|
| P7.G90 | gate: the phase's green tests pass on every joined device type | - | open | |
| P7.G91 | gate: a representative program exercises this completed slice through the final API against the separate frozen baseline | - | open | |
| P7.G92 | gate: independent review of the phase's diff against plan, api.md, ledger: no forbidden move | - | open | |
| P7.G93 | gate: comparable completed-slice baseline re-measured interleaved, no regression beyond noise | - | open | |

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
| P8.D11 | delete `WORKGROUP_FALLBACK` | 15 hits, 2 files | done | the fault was NVIDIA's 64-bit division; Lava's `lower_wide_divisions!` removes it; Mantle 28a58dc, `test_static_workgroup.jl` |
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
| P8.D35 | delete graphics constructors keyed on a KA backend; hooks without a device argument other than `staged_gemm_tile` (P6.D20) and `initbackend!` (P8.D18) | - | open | |
| P8.D36 | delete `Device(::KA.Backend)`, `caps(::KA.Backend)`, `Device(api; select)`, `backend(dev)` | - | open | |

| id | item | baseline at 5c2d4c6 | status | evidence |
|---|---|---|---|---|
| P8.G1 | green: `test/suite/areas/devices.jl`, `test/suite/areas/threads.jl` | - | open | |

| id | item | baseline at 5c2d4c6 | status | evidence |
|---|---|---|---|---|
| P8.D37 | delete `HostAPI` (plan, Devices, "Deleted") | 0 hits: n/a once confirmed | open | |
| P8.D38 | delete `defaultdevice!` (`Device` and the registry; api.md Appendix A) | 18 hits, 9 files | open | |
| P8.D39 | delete `use_frozen_kernels` (the `Mantle.Kernel` at its `CodeInstance`; api.md Appendix A) | 11 hits, 8 files | open | |

| id | item | baseline at 5c2d4c6 | status | evidence |
|---|---|---|---|---|
| P8.G90 | gate: the phase's green tests pass on every joined device type | - | open | |
| P8.G91 | gate: a representative program exercises this completed slice through the final API against the separate frozen baseline | - | open | |
| P8.G92 | gate: independent review of the phase's diff against plan, api.md, ledger: no forbidden move | - | open | |
| P8.G93 | gate: comparable completed-slice baseline re-measured interleaved, no regression beyond noise | - | open | |

## Phase 9. Metal on the core model

| id | item | baseline at 5c2d4c6 | status | evidence |
|---|---|---|---|---|
| P9.A1 | Metal qualification gates answered first: event-chained `MPSGraphExecutable` cost and MTL4 indirect render-command state coverage without a per-run host draw walk | - | open | plan, Metal qualification gates; native measurements remain |
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

| id | item | baseline at 5c2d4c6 | status | evidence |
|---|---|---|---|---|
| P9.A7 | stores on Metal: inline limit 0 in `caps(dev)`, every store a copy from its upload region | - | open | |
| P9.A8 | one residency helper used by `rawalloc`, `rawfree`, `createreserve`, `bindpages!`, `unbindpages!` and the per-plan sets (api.md section 8) | - | open | |
| P9.A9 | `MtlArray` vendor view constructor (G11; core's `withview` is P6.A7) | - | open | |
| P9.D18 | delete Mantle's use of Metal.jl's `BatchedCommandQueue` (`metal/device.jl:82, 93`; plan, Metal, "Channels") | 10 hits, 2 files | open | |

| id | item | baseline at 5c2d4c6 | status | evidence |
|---|---|---|---|---|
| P9.G90 | gate: the phase's green tests pass on every joined device type | - | open | |
| P9.G91 | gate: a representative program exercises this completed slice through the final API against the separate frozen baseline | - | open | |
| P9.G92 | gate: independent review of the phase's diff against plan, api.md, ledger: no forbidden move | - | open | |
| P9.G93 | gate: comparable completed-slice baseline re-measured interleaved, no regression beyond noise | - | open | |

## Phase 10. CUDA and HIP

| id | item | baseline at 5c2d4c6 | status | evidence |
|---|---|---|---|---|
| P10.A1 | CUDA device type new; ROCm rebuilt on the core model: graphs built node by node | - | open | |
| P10.A2 | slot-pointer argument ABI and capacity-predication baseline; qualify native layout and descriptor loads, measure optional alternatives separately | - | open | |
| P10.A3 | counters written by the stream as tokens; streams as queue kinds; committed arenas first, host VMM is not queue-ordered sparse mapping | - | open | |
| P10.A4 | library candidates (CUTLASS, cuDNN, Composable Kernel; cuBLAS/rocBLAS/hipBLASLt as foreign nodes) | - | open | |
| P10.A5 | native baseline qualification before joining: fixed-capacity slot-pointer kernels/copies, core-unrolled bounded repeats, graph-executable versioning, committed arena reuse and two streams with a wait. Qualify cuDNN/CUTLASS, native conditionals and host-VMM optimizations independently; HIP uses its actual supported entry points | - | open | |

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

| id | item | baseline at 5c2d4c6 | status | evidence |
|---|---|---|---|---|
| P10.A6 | stores on CUDA and HIP: inline limit 0 in `caps(dev)`, every store a copy from its upload region | - | open | |
| P10.A7 | `CuArray` and `ROCArray` vendor view constructors (G11; core's `withview` is P6.A7) | - | open | |

| id | item | baseline at 5c2d4c6 | status | evidence |
|---|---|---|---|---|
| P10.G90 | gate: the phase's green tests pass on every joined device type | - | open | |
| P10.G91 | gate: a representative program exercises this completed slice through the final API against the separate frozen baseline | - | open | |
| P10.G92 | gate: independent review of the phase's diff against plan, api.md, ledger: no forbidden move | - | open | |
| P10.G93 | gate: comparable completed-slice baseline re-measured interleaved, no regression beyond noise | - | open | |

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
| P11.D12 | VideoEditor: the `repeat!(g, 1; while_nonzero = gate)` idiom becomes one `when!` per kernel with a `Bool` gate | 9 hits of `while_nonzero` | open | |
| P11.A1 | VideoEditor: the engine's device passed everywhere | - | open | |
| P11.A2 | Hikari: `fill_aux_buffers!` and `postprocess!` become methods; builds and refits are operations; one `VolPath` rendered by one task at a time through its graph's lock | - | open | |
| P11.A3 | RayMakie: frame graphs recorded; overlay compositing through the one path | - | open | |
| P11.A4 | JuliaVision: one graph per call; DNNKernels' GEMM, attention and convolution as `mul!`, `conv!`, `attention!` on graph arrays; `call`, `replay!`, `planfor`, `recordedplan`'s hooks go | - | open | |
| P11.A5 | comments contradicting Mantle corrected in the commit that makes them wrong | - | open | |
| P11.G1 | green: each consumer suite with two devices open; each runner's per-call assertion (one run per call, no bytes copied outside the graph, host waits only for outputs and at host nodes) | - | open | |

| id | item | baseline | status | evidence |
|---|---|---|---|---|
| P11.A6 | RayMakie: frame graph recorded once; adding or removing a plot recompiles nothing (a group of draws), a pipeline change is a group delete and a new group; recompiled only for kernels declared per plot, a new pass, a new Hikari material specialization (execution-contracts.md §7) | - | open | |
| P11.A7 | RayMakie: attributes as `MantleArray`s mutated in place; `update!` as stores and resizes, another array object as a new group of draws; zero elements keeps the render object | - | open | |
| P11.A8 | RayMakie: a dirty list put by ComputePipeline edges instead of scene-tree walks | `poll_all_plots` 10 hits, `for_each_atomic_plot` 11 hits | open | |
| P11.A9 | RayMakie: the screen as a reader of its plots' arrays (redraw marks from runs that write them) | - | open | |
| P11.A10 | RayMakie: shadow and AO passes only when their inputs changed (under `when!`, plan phase 11) | - | open | |
| P11.A11 | RayMakie: grid lines from the camera on the GPU; lines' indices stored; `visible` for traced plots; the missing dependencies (image camera, stroke and glow colours, colormaps) | - | open | |
| P11.D13 | RayMakie: delete `frame_signature` | 6 hits, 2 files (RayMakie e6e1aa9b0) | open | |
| P11.D16 | Hikari and RayMakie: no `waitidle` inside a frame | 7 hits in Hikari, 5 in RayMakie | open | |
| P11.G2 | green: RayMakie's compile-counter session (written as P0.T26; attribute changes, value ↔ per-element, another array for an attribute, visibility, camera, layout, resizes within capacity, TLAS moves and counts): zero compiles (plan phase 11) | - | open | |

| id | item | baseline at 5c2d4c6 | status | evidence |
|---|---|---|---|---|
| P11.A12 | RayMakie: one group per screen for its plots, starting with the clears, and one group of draws per plot in it (`Group(screen.plots, name)`, `delete!(g, group)`); Hikari's integrator declared into the frame graph once, outside the plots' groups; film and sample count are device arrays the screen holds (survive plot changes and recompiles), work queues window-sized graph arrays | - | open | |
| P11.A13 | RayMakie: one scene-rectangle array per scene, shared by its plots | - | open | |
| P11.A14 | RayMakie: glyph UVs in texels, normalized in the shader, so atlas growth rewrites no UV | - | open | |
| P11.A15 | RayMakie: per-frame values (the camera matrix) as one-element arrays written by stores, inline from their upload region up to the `caps(dev)` limit | - | open | |
| P11.A16 | RayMakie: render objects' finalizers push their hold drops through the pool's inbox | - | open | |
| P11.A17 | JuliaVision: load is a call: load-time transposes, `hoistconstants` and quantization become one load graph per model (plan, "What removes every split", item 10) | 37 hits of `hoistconstants`, 11 files (JuliaVision 2f0c3ed) | open | |
| P11.D17 | GPUFiltering: its internal `synchronize` and readbacks go (plan, "What removes every split", item 2) | 22 hits of `synchronize`, 12 files (JuliaVision 2f0c3ed, `GPUFiltering/`) | doing | the `synchronize` calls are gone (JuliaVision c2f859b; 3 hits left at 1018c51, all comments); readbacks remain in `goodfeatures` (`features.jl:86`) and `fitaffine`/`fithomography` (`flow.jl:117`, `:333`, fits on the CPU over the flow field), which need device reductions |
| P11.D18 | GPUFiltering: `launch!(device, kernel, args, ndrange)` and the explicit device argument go once a KA launch over graph arrays declares a node (P6.A17); ops with temporaries keep two methods of one name, one allocating them (`MantleArray(location(out), …)`) and one taking them | 31 hits of `launch!(`, 12 files (JuliaVision 1018c51, `GPUFiltering/src`) | open | GPUFiltering depends on Mantle directly since 1018c51 |
| P11.D19 | Paperhawk (github.com/SimonDanisch/Paperhawk.jl): `DeviceArrays` (`src/gpu.jl`) goes | not checked out here: count at its port | open | |
| P11.D20 | `Mantle.storage(...)` calls go in Paperhawk, PPOCRRunner and UVDocRunner (`MantleArray` is the array; P6.D58 for `devicearray`) | 10 hits, 2 files in PPOCRRunner/UVDocRunner (JuliaVision 1018c51); Paperhawk: count at its port | open | |
| P11.A18 | Paperhawk: one graph per call; its per-step plans (detector frame, spans, edges, thumbnail; page warp, cleanup, rotate, pack; preview tiles) become one graph each, and host readbacks after `run!` become device arrays the host reads after the run, or copy nodes (plan, Consumers) | - | open | |

| id | item | baseline at 5c2d4c6 | status | evidence |
|---|---|---|---|---|
| P11.G90 | gate: the phase's green tests pass on every joined device type | - | open | |
| P11.G91 | gate: migrated editor opened and played, RayMakie raster/traced frames rendered, and real model calls ran | - | open | |
| P11.G92 | gate: independent review of the phase's diff against plan, api.md, ledger: no forbidden move | - | open | |
| P11.G93 | gate: baseline re-measured interleaved, no regression beyond noise | - | open | |

## Phase 12. Full suites, once

| id | item | baseline at 5c2d4c6 | status | evidence |
|---|---|---|---|---|
| P12.G1 | Mantle, Hikari, RayMakie, VideoEditor, JuliaVision and Paperhawk on Linux (Vulkan, CUDA, HIP); Paperhawk's suite is `Pkg.test` in its root, with a display and the GPU | - | open | |
| P12.G2 | Mantle, Hikari, RayMakie on the Mac | - | open | |
| P12.A1 | a "State" section written from the results | - | open | |

## Found while working

Items not in a phase's list when the ledger was written. Each names the
phase that needs it resolved.

| id | item | baseline at 5c2d4c6 | status | evidence |
|---|---|---|---|---|
| F2 | per-argument access of kernels and shaders | before phase 1 | doing | design settled: declaration-time inference stored in the `Kernel` (api.md §§3/4.4 and Appendix A); replacement inference/soundness gates remain open |
| F3 | device capability schema, KI projections and actual native/compiler support; memory headroom and watchdog facts stay distinct | before phases 1 and 8 | doing | execution-contracts.md §§6/12 fixes schema and units; DEV-09 projections, native-facts.json and native-evidence.md; implement/qualify providers |
| F4 | bounded nonblocking profiling and canonical recording counters | before phase 1 (the split may not use it) | doing | execution-contracts.md §10 fixes dual variants, one pending sample, ring64, timestamps and counters; native timer qualification remains |
| F5 | core token-snapshot waitidle and admission-closed native teardown | before phase 3 | doing | execution-contracts.md §6: waitnativeidle!(ch); no present-completion inference; implement teardown |
| F6 | recycle native command/graph/packet objects only after last-use tokens | before phase 3 | doing | execution-contracts.md §3 fixes endpoint ownership and finish/prepare/enqueue separation; native qualification remains |
| F7 | explicit copies between devices with synchronous snapshot/read-store baseline | before phase 6 | doing | execution-contracts.md §12 fixes devicecopy!; graphs stay in one device domain; qualified peer paths optional |
| F8 | SelectionKey/Candidate and independent-reference tuning harness | before phase 6 | doing | execution-contracts.md §9 fixes representation, validity, aliasing, private state and timing/publication; implement shared harness |
| F9 | window API: GLFW hints, the default colour space and format, the `vsync` default | before phase 7 | doing | execution-contracts.md §§3/4/6 fixes core schema, colour/vsync and native open/surface contracts; native implementation/GLFW platform hints still require qualification |
| F10 | driver-object byte counts, unknown accounting and destruction signatures | before phase 2 | doing | execution-contracts.md §3: objectbytes/destroy!; §13 bounded idle caches; native implementations remain |
| F11 | scoped vendor views and explicit canonical foreign stream | before phase 6 | doing | api.md 2.9; execution-contracts.md §§3/11 fixes leases/failure/reset; qualify CUDA/HIP paths |
| F12 | vocabulary fates in `api.md` Appendix A reconciled with source disposition | phase 0 | doing | existing seven source audits and source-disposition.tsv supply ownership/deletion evidence; per-phase implementation deletions remain open |
| F13 | fixed decisions 1–14 and native measurement gates | per phase | doing | plan Fixed decisions and remaining measurements; execution-contracts.md §§3–13; choices fixed, qualification remains |
| F14 | per-frame draw-order changes use sort! and flat pass-list rewrite | before phase 7 | doing | plan Graphics/RayMakie; O(pieces), finite RM-06 gate; measure actual frame cost |
| F16 | typed RT groups, core SBT layout, procedural geometry, stencil/MSAA/blend | before phase 6 | doing | execution-contracts.md §§7/8; actual Hikari per-material closest-hit path preserved (earlier no-hit-groups assertion corrected); HK-10/TL-43/PIECE-24 await runtime |
| F17 | bugs the survey found, to triage into the bug list: Metal `npipelines` counts draws; a resource-valued draw count has no Metal method; `trace!` has no Metal compile path; Vulkan walked plans rewrite argument memory while a frame may be in flight; the Vulkan pipeline cache key holds only whether a descriptor layout exists and misses `tess_eval`; Vulkan `vsync` lost on resize; a hand-recorded window frame may lose its contents | phase 0 | open | |

| id | item | baseline at 5c2d4c6 | status | evidence |
|---|---|---|---|---|
| F18 | closed-generation reset_device! debugging teardown | before phase 8 | doing | execution-contracts.md §11; old handles stay invalid, close admission and drain outside locks; implement/qualify |
| F19 | window open signature and surface-specific choices | before phase 7 | doing | execution-contracts.md §§3/4/6: openwindow(dev,width,height,options), surfacecaps; native main-thread/WSI qualification remains |
| F20 | texture restore indices and independent sampler-entry retirement | before phase 5 | doing | execution-contracts.md §§11/13 fixes fresh generation indices, weak interning and token holds; native descriptor reuse gates remain |
| F21 | complete placeimage/recycle! contracts | before phases 2 and 3 | doing | execution-contracts.md §3: imagerequirements/placeimage spec, dedicated constraints and endpoint recycling; implement native paths |
| F22 | record-state guard and compiler-origin source-cell isolation | before phase 4 | doing | execution-contracts.md §14; THR-04 negative guard control and KL-10 real origin audit gates; qualify compiler output |
| F28 | the baseline column counts names at 5c2d4c6, 133 commits before the frozen head f6e359a; recount (rows added already count at f6e359a) | phase 0 | open | |
| F29 | Lava's eager launch costs about 24 µs per dispatch (SAM 2 encoder on the RTX 4000 Ada, `experiments/eager-vs-graph/`); eager stays a first-class path, so find where it goes | phase 1 | open | |
| F31 | Lava d688952's RADV/ACO workaround (`foldedoffset.jl`, negative address constants kept off split loads) runs on every device: it costs ~9% of Qwen-Image's VAE decode on the RTX 4000 Ada (1.138 s without, 1.246-1.254 s with) and ~1% on the 7900; decide whether it becomes a per-device compile setting through `config(dev, f)` (F3) | phase 1 | open | A/B, `experiments/bugfix/env-newest/BASELINE.md` |
| F32 | bounded default workgroups preserve caller geometry and kernel legality | phase 1 | doing | execution-contracts.md §9 fixes target256/2-D16x16/3-D8x8x4 bounded heuristic; image/collective performance qualification remains |
| F33 | preserve atomic load/store/fence semantics and reject unsupported sequential consistency before enqueue | phase 1 | done | Lava demotion removed; atomic load/store/fence emission and independent CAS orderings verified; order-dropping Atomix overrides removed, Device scope preserved and its native feature queried/enabled. experiments/atomic-ordering-2026-10-21 contains permanent compiler regressions, complete Lava suite results and four-device native visibility qualification. Unsupported seq_cst now fails compilation, including default Atomix RMW; callers must justify explicit ordering. Scope/type/feature eligibility remains F42; this is not universal hardware qualification |
| F34 | NVIDIA kernels in PTX inside Vulkan command buffers (`VK_NV_cuda_kernel_launch`, provisional): `cp.async` ran correctly on Vulkan device addresses on the RTX 4000 Ada (`experiments/interop/`); a separate CUDA context costs 80-175 µs per hand-off, HIP imports Vulkan memory but no semaphores (ROCm 7.2.1). Whether a PTX kernel node goes into the plan is open | phase 10 | open | |
| F35 | explicit Group propagation in model-level loops: examples.jl 19 (diffusion, `transformer(step, …)`) and 20 (decode until a stop token, `decodestep!(dec, …)`) pass a group into model functions | phases 0/6/11 | doing | design settled in api.md 2.4: leading-Group operations share one body; model code propagates the group. Examples express the target; replacement/model execution gates remain open |


## Document review follow-up (2026-10-20)

These rows track implementation and proof obligations, not completed runtime
fixes. The design corrections are in the target docs, execution-contracts.md and the suite; the new API
has not been executed. Detailed rationale: `experiments/review-2026-10-20/review.md`.

| id | item | needed by | status | evidence |
|---|---|---|---|---|
| F36 | implement the phase-1 foundation in the isolated replacement checkout against the separate frozen baseline; no legacy adapter path or phase-6 floor | before phase 1 | doing | bounded native probe experiments/feasibility-2026-10-21 passes 132 assertions on AMD/NVIDIA/Lavapipe without old Mantle import; fixed pool/slots/length/move/holds exercised; full phase-1 API/cache/reclamation/locks remains |
| F37 | preserve dims, offset and strides of nested views; conjugating adjoints preserve conjugation; scatter stores carry actual destination mapping and coalesce by element coverage | phases 4/6 | open | internals.jl types; resizing-and-raytracing.jl B.3/B.5; views.jl and stores.jl ST-10/ST-11/ST-21 |
| F38 | merge alias uses per node, and publish uses only after checking predecessors; no self-edge for in-place/view aliases or AnyWrite | phase 1 | open | internals.jl dependencies!; views.jl, repeat.jl and composite cases |
| F39 | persistent images/windows always stored; only dead graph transients discarded; a clear after a draw starts another pass | phase 6 | open | api.md 2.6b; pieces.jl PIECE-01/20; PIECE-23 adds draw-clear-draw coverage |
| F40 | piece insertion reserves cross-queue dependencies, checks shader writes/image layouts/placement, and recompiles if incompatible; register new uses and retire removed uses | phase 6 | open | api.md 2.6b; PIECE-15/17; extend with forced second queue and sampled-image/storage-write cases |
| F41 | carry the graph's previous-run tokens across plan replacement, including disjoint old/new resources | phase 3 | open | internals.jl recompile!; extend mutation.jl/run.jl with disjoint-plan replacement |
| F42 | immutable KI/scalar/atomic facts and candidate/compiler eligibility | before phase 1 | doing | execution-contracts.md §6; DEV-09, host caps tests; native facts must intersect compiler support; F33 compiler repair is complete. Native Float32 min/max still needs VK_EXT_shader_atomic_float2 enablement/eligibility qualification, distinct from the validated compiler emission |
| F43 | capture mutable pending/storage snapshots under resource locks and validate before apply; no unlocked iteration over a mutating Vector | phase 4 | open | resizing-and-raytracing.jl B.4; changes_during_runs.jl/threads.jl |
| F44 | pend transitive BLAS/TLAS reader updates before publishing each accepted segment’s writes, including a partial-stage failure; lock expanded reader resources and validate snapshots first | phases 3/5 | open | internals.jl lockstage/submitstage!/pendreaders_locked!; add a gated concurrent writer/trace case |
| F45 | test infrastructure: each device/area pass has isolated helpers; valid @case/@testset macro labels, no phase floor; graph-array show reads metadata only | phase 0 | doing | runsuite.jl/helpers.jl and DECL-04 changed; parsing verified, execution awaits the API |
| F46 | complete native vocabulary without backend policy duplication | before phases 2/3 | doing | execution-contracts.md §§2–14 adds timestamp, SBT, memory-class, external/foreign/video verbs and typed failures; native implementations remain |
| F47 | freeze presentation completion/retirement, queue ownership and swapchain capability requirements; rendering tokens alone cannot release a generation | before phase 7 | doing | execution-contracts.md §4 fixes independent completion, surface/family rules and acquisition-history fallback; legacy shutdown lacks a formal native proof |
| F48 | actual consumer paths and resource/callable coverage | before phase 11 | doing | existing consumer ports/baselines preserved; evidence-index.md and consumer-coverage.md map actual refactor deltas; DNN-12/RM-15/HK-10/TL-43/DECL-12/13; replacement integration checks remain |
| F49 | prove the ordered-lock model; use a simpler per-device lock as an optional stress-test reference, not a second production policy; retain weak references until ownership/finalization proof allows a change | before phase 3 | doing | execution-contracts.md §1 gives linearization/lifetime arguments and retains weak ownership; host execution model has negative controls; native stress/qualification remain |
| F50 | maintain a source inventory separating runtime code, portable kernels, driver ABI code, comments and application algorithms; measured deletion savings, no hard 10k-line gate | each phase | doing | seven existing per-file audits plus experiments/review-2026-10-20/source-disposition.tsv cover 111 src/ext files and 55,005 physical lines; owners/phases indexed; estimates overlap; no hard 10k gate |
| F51 | source debt: delete duplicate plan walks in graph/kalaunch.jl, vulkan/graph.jl and metal/record.jl; unify HWTLAS member bookkeeping; eliminate paired eager/declared operation bodies and mutable kernel/pipeline globals | phases 1/3/5/6/9 | open | source audit files and review.md give names and ownership |
| F52 | quantify the flat pass-list rewrite and conservative render dependencies on large scenes; meet the existing finite RM-06 latency bound without claiming asymptotically constant insertion or free cross-queue overlap | phase 6 | open | plan Graphics; updatepieces! rebuilds compiledpieces; RM-06 unchanged |
| F53 | core owns the piece draw loop; recordpiece(dev, node, region, emit) only sets up the native piece builder, calls the callback and finishes it | phases 3/6/9 | open | api.md 4.5; core owns the draw walk; the native verb only brackets emission and returns its builder |
| F54 | submission budgets need a policy for a single over-budget kernel/library node or indivisible present span; select/tile an eligible candidate in core, never require consumer graph partitioning | before phases 1/6 | doing | execution-contracts.md §3 fixes core-unrolled loops/partition eligibility and honest indivisible-node limits; native candidates and performance qualification remain |
| F55 | unified host reads validate an idle snapshot and protect CPU memcpy against subsequent submissions; staged reads retain the submission's shape and recheck readback capacity | phase 3/4 | doing | internals.jl Array validates projected staging capacity before bind/claim and retains copy shape; READ-07 + host growth-interleaving model; native race test remains |
| F56 | use the resolved KernelInterface launch macro (`KI.@launch`); keep it distinct from KernelAbstractions kernel-definition macros | phase 0 | doing | environment Manifest resolves KernelAbstractions/lib/KernelInterface; EAGER-01/HOLD-12, views.jl and examples corrected; no compatibility macro added |
| F57 | shared bounded native-cache publication, invalidation and holds | phases 1/6 | doing | execution-contracts.md §13 fixes idle64/64MiB/unknown8 bounds, code identities and no captured-array keys; implement/measure |
| F58 | represent graph-transient state as Unplaced/NoCell then ArenaSlice; prepare a Cell at the first resizable compile; only plan tenancy owns arena backing; graph aliases of device arrays retain device storage | phases 1/4 | open | resizing-and-raytracing.jl B.1/B.2; fixed-cell field/region-only union contradicted the transient contract |
| F59 | finish FaultDevice channel/builder interception and pool ownership before trusting native fault cases; injected submitnative! must actually be reached, and shared native queues must share synchronization | before phase 3 | doing | actual channel/builder/present wrappers and fresh Runtime contract added; test/design/faultdevice.jl executes interception without GPU; replacement Runtime integration remains |
| F60 | fixture correctness, empty-gate rejection and verified driver pairs | phase 0 | doing | shared helpers and host tests; DEV-01/03/04/05/08/10/12, DECL-03 corrected; canonical multicompute/transferqueue fixtures; native-facts disproves Lavapipe no-RT/no-sparse assumption |
| F61 | publish accepted-prefix tokens/reader updates per segment; preparation errors retry, postclaim native errors poison/quarantine; presentation is outside record locks | phases 3/7 | open | execution-contracts.md §§1/2; internals.jl submitstage!/submit!/finishpresent!; RUN-07/AP-05 now fault preparation and assert interception |
| F62 | typed sampled-image device ABI, real sampling intrinsic and descriptor-indexing fallback; verify actual atlas content updates, not just index changes | phases 5/6 | open | api.md §2.6; RM-15 now samples before/after atlas growth and upload; textures.jl uses internal textureindex projection |
| F63 | consumer decoder/DPB lifetime and external GL ownership/fence handoff; portable API must not erase foreign synchronization before replacement exists | phase 11 | open | consumer-coverage.md editor paths and execution-contracts.md native resource contract |
| F64 | callable captures and all ray-program stages enter discovery, holds, access inference and ABI packing | phases 1/5/6 | doing | execution-contracts.md §7; DECL-12/13 and HK-10 cover hidden-only uses and foreign-device rejection; runtime/compiler qualification remains |
| F65 | normalize budget headroom per heap; never double count mapped/reserved or used/free | phase 2 | doing | execution-contracts.md §12; corrected tryacquire and dynamic FaultDevice BudgetLimit; host 400/700/1200 allocation cases; native pressure qualification remains |
| F66 | bounded pageable eviction freezes one source version, blocks native users, preserves queued changes and copies device length | phases 2/4 | doing | execution-contracts.md §12; HostCopyJob/transfer pseudocode; host interleaving model has torn-copy/lost-store negative controls; native copy/pressure race remains |
| F67 | staged-read capacity failure escapes to host destination reallocation, never retries identical undersized storage | phases 3/4 | doing | internals.jl OwnWork/NeedDestination validation before claim; host growth negative control; actual Array race remains |
| F68 | sampled-image footprint, alpha blend alpha equation and sampler lifecycle are explicit | phases 5/6 | doing | execution-contracts.md §§8/11; PIECE-24 checks alpha .625; no invented derivatives or image-owned sampler lifetime; native qualification remains |
| F69 | native query evidence is distinct from enabled compiler paths and GPU execution | phases 0/1/8 | doing | native-facts.json and native-evidence.md; all local Vulkan adapters including Lavapipe advertise RT/sparse; F33 records a reproduced compiler-side gap and its scoped repair |
| F70 | kernel-written TLAS totals use fixed-capacity build counts even on count-from-memory paths | phase 5 | doing | api.md and resizing pseudocode now agree with plan; valid mask-zero padding keeps Update count fixed; exact-count optimization needs separate legality proof |
| F71 | defer native present until all stages are accepted; retry before resize/recompile without replaying kernels or skipping later stages | phase 7 | doing | execution-contracts.md §2; corrected run!/finishpresent! pseudocode; WIN-30 injects not-enqueued OOM, checks native calls/counter/no GPU submit on retry and preserves pending resize; native execution remains |
| F72 | core lowers repeated groups before hazards/costs/placement; every body node participates in loop-control edges, even if it ignores iteration | phase 1 | doing | internals.jl lowerloops! and execution-contracts.md §3; remove backend emitloop! and opaque loop split; code reused but commands/slots grow with maximum iterations; repeat/split runtime gates remain |
| F73 | TraceNode retains its explicit acceleration structure; callable captures also contribute Bound slot dependencies | phases 1/5 | doing | internals.jl TraceNode.accel and kernel/program uses now agree with trace!/capture contracts; HK-10/DECL-12/13 and compiler-origin gates await runtime |
| F74 | qualify direct-host restores and enforce debugging before any fallback/design change | before closing phase 4 | doing | direct restore is specified in api.md §10 / execution-contracts.md §12; internals and resize steps use prepared direct copies or CPU initialization of an unpublished CPU-visible final allocation. User ruling: no staged restore without overwhelming evidence after debugging/exhausting direct-path fixes; existing importhost/importablehost and DNNKernels usage are reference evidence. experiments/direct-host-2026-10-21: 519 native assertions, zero validation errors; exact imports work for owned copies/cache sources on all four local Vulkan devices, mapped-final-destination copies cover protected file sources at tested extents. Vulkan.jl output-header bug fixed with regression controls. Replacement integration, capacity/performance and other OS/API qualification remain; no staged fallback justified |
