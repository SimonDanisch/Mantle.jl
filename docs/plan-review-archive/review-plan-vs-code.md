I checked the plan against Mantle HEAD 90e582f; no files were modified. Mantle has no local changes in `src/`, `ext/` or `test/`, so the working tree is HEAD. The line-by-line sweep and the JuliaVision census came from three parallel read-only checks. I re-checked their contested claims myself and dropped one that was wrong: that `claim!`/`unclaim!` are dead code. They are called from `hold!` and `unhold!` (`graph/lifetime.jl:261,375`).

Paths are relative to `/sim/Programmieren/VideoEdit/dev/Mantle/src` unless they start with `ext/`, `test/` or a repo name. JuliaVision is `/sim/Programmieren/VideoEdit/dev/JuliaVision`, HEAD 8063736 (29 commits after the census commit 4551984).

## Task 1 — Wrong or stale references

**Systematic problems**
- **Mixed reference points.** The plan says its references are to 7e64cd5, but several are HEAD positions (`phases.jl:724`, `types.jl:649`). Two point at the uncommitted tree that later became 692a4e1, so they match no commit (`kalaunch.jl:1068`, `build.jl:1454`).
- **Wrong path.** `graph/phases.jl` does not exist at either commit. The file is `phases.jl`.

| plan line | claim | reality | plan should say |
|---|---|---|---|
| 3-5, 514 | `measure!` and the re-record in `run!` are uncommitted | Committed in 692a4e1 (09-28). `measure!` is at `graph/kalaunch.jl:1241-1257`, called from `run!` at 677; the re-record is at 681 (`pl.stale && record!(pl)`). Default budget is `submissionbudget(::Device) = 0.1` (`graph/backend.jl:643`) | "committed in 692a4e1" |
| 11 | references are to 7e64cd5 | Mixed, as above | One commit (HEAD), re-checked |
| 79-80 | ROCm `openrecording` takes 2 args | Fixed in af50bfc (`ext/MantleROCmExt.jl:885`); `test/rocm/test_rocm.jl` pins the signature | Remove |
| 81-82 | `destroy_pool!` calls `empty!` on an `Inbox` | Fixed in 2c7dc14 (`takeall!`, `vulkan/runtime/memory.jl:1026-1027`). No regression test was added | Remove, or add only the test |
| 83-84 | `flush_stall_report` bug at `command.jl:1147-1150` | Still broken, now `vulkan/runtime/command.jl:1165-1168` | Update lines |
| 64 | `graph/build.jl:1382`; `vulkan/graph.jl:1802-1823` | `build.jl:1457`; `vulkan/graph.jl:1868-1889` | Update |
| 89 | `when!` at `graph/build.jl:342-354` | `build.jl:353-368`. The false docstring "Barriers are unaffected" is at 348 | Update |
| 91 | Metal `Metal.synchronize()` in `waitfor`/`waitidle`/`openrun` | Only `waitfor(::LegacyQueue)` (`metal/device.jl:448`); `openrun` is now `metal/record.jl:1197` | Update |
| 99-100 | `device.jl:1735`, `command.jl:1257,1330`, `raytracing/pipeline.jl:86-133` | `device.jl:1744`, `command.jl:1275,1348`, `pipeline.jl:85-133` | Update |
| 222 | `gemm.jl:2189-2214` | `vulkan/array/gemm.jl:2267-2292` | Update |
| 204-210 | GPUArrays 11.4 requires `storage`, `copy`, `mapreducedim!`, `default_rng` | The project Manifest has GPUArrays 11.5.14. It requires `storage`, `copy`, `derive` and `mapreducedim!`; `default_rng` is only a stub now (`deprecated.jl:15-18`) | 11.5.14 list |
| 195 vs 304 | Graph arrays are `AbstractArray`s / are not | The plan contradicts itself | Pick one |
| 320 | `graph/types.jl:180` `Pass.kind` | `types.jl:196` | Update |
| 323-329 | `repeat!` copies its body per iteration | Already one body since 692a4e1 (`build.jl:370-449`); see Task 3 | Rewrite as existing |
| 332 | `DeviceRange` without `max` costs a sync per launch | True only on the walked `Launch` path. Vulkan's recorded path dispatches indirectly with no host read | Qualify |
| 401-403 | `compile_dispatch` at `vulkan/graph.jl:958-990`; "four launch-extent calculations" | 958-994. There are more than four extent sites, including Metal `record.jl:354` and `378-389`, ROCm `ext:631`, `678`, `714-726`, and `host.jl:304` | Cite the sites |
| 428-430 | `ki_launch_extents` goes | Moved into core by 692a4e1 (`graph/kalaunch.jl:242`); also used by `host/host.jl:342` | Name the core location |
| 453, 488 | `graph/phases.jl:382-388,461` | `phases.jl:481-488` and `618`; `materialize!` writes placement into the graph at `build.jl:1388-1394` | Update |
| 461 | `PassPlan` at `graph/types.jl:649` | 649 is HEAD; at 7e64cd5 it was 600 | Consistent commit |
| 484 | Dag at `graph/phases.jl:198-213` | Rewritten by 692a4e1 (`overlapgroup`) at `phases.jl:211-243`. Still not stored in `Plan` | Update |
| 485-486 | "selection is at Pipelines, `phases.jl:724`"; `compile!(ctx, phases)` | Wrong. `phases.jl` had 686 lines at 7e64cd5. GEMM selection happens at declaration (`gemm.jl:2561-2565` via `dispatchlaunches!`; `metal/ka.jl:162-175`). `compile!` is at `phases.jl:866`, `PHASES` at 724 | Correct the claim |
| 515-516 | `kalaunch.jl:1068`, `build.jl:1454` | `partitionranges` at `graph/kalaunch.jl:1177-1208` (Inf cost at 1187); profiler for every plan at `build.jl:1526-1530` | Update |
| 545-548 | The head barrier (`vulkan/graph.jl:1262-1269`) is the only cross-plan order | Now 1269-1276, plus `emitloophead!` at 1287-1293. Vulkan `submit!` also waits across queues through `crosswaits!`/`stamp!` (`command.jl:568-570,669-671`) | Correct |
| 543, 569 | Arenas per queue; tenancy is a count | Arenas are per pool and kind (`memory/pool.jl:358`); a kind can span several `ArenaPart`s (34166c3). `tenant!` keeps a list of `WeakRef`s (`pool.jl:980-1012`) | Correct |
| 583-615 | Core types "keep existing names" | Same names, other fields: `SubmitChannel{Q,R,T,P}` (`lifetime.jl:108`), `Submission{R}(recording, holds)` (674), Vulkan `mutable struct Recording <: Closed` (`vulkan/runtime/device.jl:98`). `fence`/`passed`/`waitfor` take a device today (`memory/pool.jl:673-714`), not a channel. `recycle!` returns to the channel's `free` list | Say the names are reused with new fields |
| 634-647 | New builder verbs; "today's `emit_*` fold into these" | `emitdispatch!`, `emitcopy!`, `emitdraw!`, `beginrender!`, `endrender!`, `emitbarriers!`, `emitupdate!` and `emitloophead!` already exist. Core's walk also exists (`emitplan!`/`emitpasses!`, `graph/kalaunch.jl:872,899`). No `submitnative!` exists; the existing verbs are `submit!`, `submitlist!` and `submitrecording!` | Build on the existing walk and name the signature change |
| 767, 771 | `raytracing/api.jl:59-171`; `runtime/dispatch.jl:232-303` | `api.jl:61-74,134-171`; `dispatch.jl:235-306` | Update |
| 790-792 | `vulkan/graphics/window.jl:211,311-438` | 214, 426 (441 is `close`) | Update |
| 800, 1396 | `Mantle.refit!` for acceleration structures | Name taken: `refit!(pl::Plan)` (`build.jl:753`) and transient `refit!` (730, 867) | Rename one of them |
| 817 | `hwtlas.jl:243` | Call is on 244 | 243-244 |
| 880-885 | Pressure policy only in Vulkan; `trim!` has no core caller; `headroom` counts an arena's region as reusable | Core `makeroom!` (`memory/pool.jl:837-842`) runs GC, `flush!`, waiting `reclaim!` and `trim!`, called from Place (`phases.jl:579`). `headroom` counts the region as spent; `arenaroom` (`pool.jl:484-497`) counts it as reusable | Rewrite |
| 881, 887, 922 | memory.jl `1079-1298`; `pool.jl:590-619`; `Mantle.jl:532-567` | `1083-1302`; `605-634`; `562-597` | Update |
| 928-929 | Zero-argument hooks | Only `staged_gemm_tile()` and `initbackend!()` take no arguments; `use_frozen_kernels(version)` and `intrinsic_usage(::Symbol)` take one | "hooks that take no device" |
| 1143-1156 | `render_icb.jl`, `sparse_arena.jl`, `gemm_select.jl` | Not in any repo on this machine | Give a path |
| 1186-1188 | Vocabulary check did not cover `ext/` | `test/test_mantle_owns_it.jl` checks `ext/` by name, on every machine; nothing checks arity | Correct |
| 1311, 424 | "core's per-device counter" | Does not exist; `kernelcompiles` reads process-global counters (`vulkan/graph.jl:168`, `metal/metal.jl:108`) | Mark as new |
| 1436-1437 | "80 default-queue sites, 44 `vk_context()` calls" | 44 non-comment lines, of which 30 are calls (13 docstrings, 1 definition); `default_bq` appears on 54 lines. No single grep gives 80 | Give the grep |
| 1549 | 19 commits to 58a05a9 | Correct; HEAD is 23 commits past 7e64cd5 | Re-sync to 90e582f |

The commit hashes 76b85f6, 1954417, ec5b20a, 2dc58eb, 1e26aeb, 692a4e1 and 58a05a9 exist and their subjects match what the plan says. The remaining references are correct and unchanged.

## Task 2 — Bugs

**The plan's list and review §1**

| # | bug | status at HEAD |
|---|---|---|
| 1 | ROCm `openrecording` arity | **Fixed**, af50bfc. At 7e64cd5 `run!` did throw (`recordsplans(::ROCmDevice) = true` plus `execute!`), so af50bfc's message ("ran unrecorded") is inaccurate |
| 2 | `destroy_pool!` `empty!` on an `Inbox` | **Fixed**, 2c7dc14; no test |
| 3 | `flush_stall_report` | Present, `command.jl:1165-1168` |
| 4 | `gemm_cm2!`/`gemm_cm2_sg!` used as kernel factories | Present, `gemm_cm2.jl:324,362`; unchanged since 70ca5cc |
| 5 | `when!` can drop a barrier | Present, `build.jl:353-368`, `sync/transition.jl:98-99`. The docstring's "harmless" claim is still there |
| 6 | Metal `synchronize()` with no device | Present, `metal/device.jl:448,1141-1142`, `record.jl:1197` |
| 7 | Walked Metal plans stay `patchable` | Present, `metal/record.jl:153-156,345,368`; no Metal `patchable` method |
| 8 | TRANSFER_SRC vs LOAD pass | Present, `framebuffer.jl:254-257`, `pipeline.jl:449-452` |
| 9 | `VK_CONTEXT_REF[]` read in validation checks | Present, at the Task 1 line numbers |
| 10 | Weak reference without a `nothing` check | Present, `memory/pool.jl:956-958` (the first loop at 941 checks; the `remap!` loop does not) |
| 11 | Unlocked `live_buffers` delete | Present, `vulkan/runtime/memory.jl:675` vs 635 |
| 12 | `append!` ignores the offset | Present, `vulkan/array/gpuarrays.jl:695` |
| 13 | H.265 aligns to 256 | Present, `video_h265.jl:605,635` |
| 14 | Two format tables disagree | Present, `textures.jl:218` (moved by de92f9b) vs `lowering.jl:239` |
| 15 | `Ref` without `GC.@preserve`; UserID vs `instance_id` | Present, `metal/trace.jl:162-163,90` |
| 16 | Features enabled without querying | Present, `vulkan/runtime/device.jl:1684` |

**Review §2 (the uncommitted split, now 692a4e1)**
- Unmeasured pass costs `Inf`: present (`kalaunch.jl:1135,1187`).
- `measure!` and re-record: present, and tests pin it (`test/vulkan/test_partitioned_recording.jl`, `test_run_allocates_nothing.jl`).
- `when!` passes never measured: now measured only if they ran before the re-record, which then drops the timestamps. Unmeasured regions stay one submission per pass, and every conditional piece is always its own submission.
- Query pool on every Vulkan plan: present, by default (`build.jl:1529`).
- `Plan()` throws on `timestampPeriod == 0`: **fixed** by `timestamps(::LavaDevice)` (`vulkan/graph.jl:1500`).
- Metal and ROCm never split: present; only `LavaDevice` has `timestamps`.

**Review §3 (structural)**

Still present except:
- `trim!` now has a core caller, and Place's own-region refusal is fixed through `arenaroom` (58a05a9); `headroom` itself is unchanged.
- Placement is half in the plan (`pl.parts`, `pl.offsets`) and still written into the graph's declarations (`phases.jl:481-488`, `build.jl:1388-1394`).
- The loop now allocates an index `Buffer` besides the predicate (`build.jl:381-382`).
- More probe uses: `gate!` (`build.jl:494-495`, `kisupported`) and `runlaunches!` (`buildskernel`).
- A new eager path: `kilaunch!`.
- `ki_launch_extents` now has one core copy.

**Review §4 (contradictions in the previous plan)**

Mostly resolved by the rewrite. Still open:
- #9: phase 3's delete list lacks `oneshot!` and `hold!`, although the Core types section deletes them. Vulkan's non-bang `oneshot` (`command.jl:409`), `drain!`, `flush!` and `waitfor!` have no fate.
- #10: the name collisions remain (see Task 1).

**Review §5** referred to the previous plan text and does not apply.

## Task 3 — Commits 7e64cd5..HEAD the plan does not account for

| commit | change | direction |
|---|---|---|
| 692a4e1 | `repeat!` declared and compiled once: `Pass.loop::Loop(count, index)`, device index buffer, `loop_start!`/`loop_advance!`, loop-carried liveness (`phases.jl:494-513`), scheduler keeps a body contiguous (`loopdeps`), Vulkan submits body pieces `count` times (`submissionbatches`) | **Toward** the plan's Loops section. Differences: a full barrier per iteration (`emitloophead!`) instead of derived hazards; images, `when!` inside loops and nesting are refused; a `Pass.loop` field, not a `LoopNode` |
| 692a4e1 | Measured budget: profiler on every Vulkan plan, `passcost`, `partition`, `stamped`, `measure!`, re-record in `run!` (SAM 2 is recorded twice), new vocabulary `timestamps`/`submissionbudget`/`emitloophead!`, `submitlist!`, `RecordingParts.batches` | **Away** from decision B7 and plan L512-516. Tests now assert the re-record. Gains: `maxpasses` removed, and `submissionbudget(dev)` is a device hook the estimated split can keep |
| 692a4e1 | `ki_launch_extents` moved to core; HostDevice gains a KernelInterface kernel executor (`host.jl:269-360`); Metal `gatesopen` deleted | Partly toward (one launch-geometry calculation, though the name is on the delete list); **away** on HostDevice, which the plan deletes |
| 8d95632 | Unrecorded headless Vulkan plans walked every run; `emitdraw!` reads `DrawBinding` count, indices, instances and textures per run; new `test/test_headless_rebindable_draw.jl` pins `rebind!` and `!recordable` | **Away.** It adds a user (RayMakie's composited frame) of the walked path and `rebind!` that phase 6 deletes. Plan L711-712 does not cover this case |
| 58a05a9 | `arenaroom`; `makeroom!` (GC, `flush!`, waiting `reclaim!`, `trim!`) called from Place | Toward "memory goes back"; **away** from "pressure only in `acquire!`, an allocation never flushes a queue". Also uses the `supports_batch_queue` probe |
| 34166c3 | `ArenaPart`: arenas split past `maxalloc` (4 GB on RADV) | Not in the plan. Sparse arenas (L862-868) must answer the same one-allocation size limit |
| dd34a07 | `gate!` falls back to a host `Call` where `kisupported` is false | **Away**: a probe plus the callable host call the plan deletes (L284-287) |
| 9353e79, d290665 | `kilaunch!` (`array/launch.jl:88`) with `kibackend(todevice(backend))`; `ArrayLaunch` extended to gemv and fft | `ArrayLaunch` with `runlaunches!`/`dispatchlaunches!` is already "one lowering, eager or declared", the concept the plan calls `lower`. The plan must name it, not invent it. It relies on `kibackend`/`todevice`, which are deleted |
| b713df9, d5393dd | Vocabulary 179 → 189 names, including `submissionholds!` (which the plan deletes) and `basealignment` | **Away**: the growth the plan's L1182-1185 describes as the failure mode |
| aaf7dfd | `basealignment(x)` operand alignment | Not in the plan; belongs in the selection-table key and `devicearg` |
| 17e0f52 | Executable properties per context; `pipeline_exec_ir` | Toward `kernelinfo` (Vulkan statistics); the request flag is still a global |
| f14df5b | Pass timestamps at `BOTTOM_OF_PIPE` | Affects what the calibration's "measured node times" mean (barrier included) |
| 48e494b, af50bfc | ROCm stream capture hardened; `openrecording` fixed | Investment in capture, which phase 9 deletes |
| 6d497f6, 2c5447c | More Vulkan GEMM and convolution selection in `gemm.jl` | Grows what must move into the selection table |
| 90e582f | `GLFW.DefaultWindowHints()` added to both Vulkan and Metal `window.jl` | Small duplication (rule 2) |
| 3ede88f, de92f9b, 2c7dc14 | Kernel tail guards, texture upload check, `destroy_pool!` | Bug fixes |

No commit touched `GPURef`, argument memory, `graph/packing.jl`, the patch table, `DeviceRange` semantics, `when!` barriers or queues. Nothing implements decisions.md B11, B14 or C1-C4.

## Task 4 — Census and delete lists

**JuliaVision census vs HEAD 8063736**

| plan claim | at 4551984 | at HEAD |
|---|---|---|
| "twelve ported runners" | 11 | 12 (Trellis2 added) |
| Bonsai quote at `BonsaiRunner.jl:800-812` | at 807-811; a reason, not a measurement | Gone |
| `emitgraph` at `emit.jl:103-170` | correct | Body is `emitinto!` (193-261); `emitgraph!(g, …)` (134-169) already composes into a caller's graph, and Trellis2 uses it |
| modules glued on the host: 6 | correct | 7; QwenImage now 10 graphs, Trellis2 5-7 |
| eager work between graphs | examples correct | `euler_step!` replaced by `DNNKernels.eulerstep!`/`cfg`; Trellis2 does CFG and Euler inside `repeat!` |
| `replay!` copies: 7 runners | correct | 8; copy loop at `driver.jl:1064-1066`. Whisper copies ≈80 MB in and 18 MB back per token |
| `record_maxpasses`: 5 runners, 7 graphs | correct | 0 (JuliaVision b18788a, Mantle 692a4e1) |
| host loop with fixed count | 4 | 5 |
| host compute mid-call | 6 | ≥7 |
| helper plan or `runonce!` | 4 | 5 (Trellis2 has 44 `runonce!` lines and its own blocking `runonce!`) |
| autoregressive readback per step | 3 | 3 |
| output length only the device knows | 1 | 2, plus device-decided loop exits (a pattern the table lacks) |
| Horizon "device argmax is slower" at 364-368 | correct | 362-366 |

**Consumer-side names at HEAD**

| name | call sites |
|---|---|
| `call` | 26 runner lines + 9 in DNNKernels |
| `replay!` | 4 |
| `planfor` | 6 |
| `recordedplan` hooks | 1 user (Hunyuan3D) |
| `hoistconstants` | 2 |
| GPUFiltering `synchronize` | 7 |
| GPUFiltering readbacks | 4 |
| `coopmat_gemm_dispatch!` | 6, now also called directly by BonsaiRunner.jl:655 and Trellis2 decoder.jl:439 |
| `native_*_dispatch!` / `native_gemm_available` | 2-5 each |
| `librarygemm` | 1 |
| `runscalls` | 3 |
| `staged_gemm_tile` | 1 |
| `SGEMM_*` | 4 |
| `GEMM_TILE` | 55 lines |

JuliaVision passes no `maxpasses`. Six tool/dead-code sites call names Mantle removed (`record_into`, `recordingpass`, `RecordingSequence`); no runner path is broken.

**Mantle delete lists at HEAD.** Every name still exists except `record_maxpasses`/`maxpasses` (gone) and `gatesopen` (gone). `measure!` now exists. Sample counts (lines matching the word in `src/`+`ext/`):

| name | lines |
|---|---|
| `LavaBackend` | 117 |
| `VulkanAPI` | 91 |
| `hold!` | 66 |
| `holdleaves!` | 50 |
| `OneShot` | 41 |
| `build_accel!` | 40 |
| `HostDevice` | 39 |
| `Closed` | 39 |
| `PassPlan` | 27 |
| `stampof` | 22 |
| `VK_CONTEXT_REF` | 22 |
| `LegacyQueue` | 21 |
| `batchqueue` | 19 |
| `runscalls` | 19 |
| `Immediate` | 37 |
| `recordable` | 17 |
| `rebind!` | 9 |

## Task 5 — Phases given HEAD

- **Phase 0**
  - `api.md` and `refactor-ledger.md` do not exist; `docs/internals.jl` and `docs/examples.jl` exist but the plan never mentions them.
  - None of the 11 phase-0 tests exist.
  - The comment pass has not happened: the `when!` docstring and the MTL4 stall comment (`metal/device.jl:139-186`) are still wrong.
  - Existing tests pin behaviour the plan deletes: `test/vulkan/test_partitioned_recording.jl`, `test/vulkan/test_run_allocates_nothing.jl`, `test/test_headless_rebindable_draw.jl`, `test/vulkan/test_hold_lifetime.jl:47-52`. Deleting or rewriting them is a "forbidden move" unless the plan lists them per phase.
  - Bugs: 2 of 9 listed fixed.
- **Phase 1**
  - Already done: the `maxpasses` removal, one `ki_launch_extents`, loops as one body (not listed in any phase).
  - Assumption broken: the measured split is committed and pinned by tests. Phase 1 must delete `passcost`, `partition`, `stamped`, the `timestamps` hook and the default profiler, and should keep `submissionbudget(dev)`.
  - The Dag is still discarded.
- **Phase 2** must reconcile `makeroom!` (in Place, flushes a queue), `arenaroom` and `ArenaPart`/`maxalloc` with sparse arenas and "pressure only in `acquire!`".
- **Phase 3** has more to delete (`submitlist!`, batch submission through `seal!`/`hold!`/`handover!`) and name collisions to resolve.
- **Phase 4:** `refit!` collision.
- **Phase 5:** `ArrayLaunch`/`kilaunch!` must be named as the existing `lower` seed; the GPUArrays interface changed.
- **Phase 6** gained a new dependency (8d95632, RayMakie's composited frame) that needs a replacement before `rebind!`/walked `execute!` can go.
- **Phase 7** counts are unsourced.
- **Phase 8:** `gatesopen` is already gone.
- **Phase 10:** JuliaVision already did `record_maxpasses` and part of composition (`emitgraph!`); Trellis2 is a new consumer (direct Vulkan GEMM call, 44 `runonce!` lines, `repeat!` diffusion).

## Passages to rewrite before anyone implements from this plan

1. L3-11: status and reference commit; re-sync to 90e582f.
2. L75-105: Bugs to fix first (drop 2 fixed items, update lines, note missing regression tests).
3. L186-240 and L304-305: arrays (the `AbstractArray` contradiction, GPUArrays 11.5.14 interface, add `basealignment`).
4. L242-305: Operations. Name `ArrayLaunch`/`runlaunches!`/`dispatchlaunches!`/`kilaunch!` as today's `lower`; deal with `gate!`'s `Call` fallback.
5. L321-329: Loops, as implemented in 692a4e1; decide full barrier vs derived hazard; `Pass.loop` vs `LoopNode`.
6. L482-495: the compile-step table (paths; "selection is at Pipelines" is wrong).
7. L497-516: submission split (what 692a4e1 committed, what goes, what stays).
8. L541-577: ordering (`crosswaits!`/`stamp!`, arenas per pool, `ArenaPart`, `tenant!` as a `WeakRef` list).
9. L579-658: Core types and Record (reused names, existing `emitplan!`/`emitpasses!` walk and `emit*` verbs, `submitnative!` vs `submit!`/`submitlist!`/`submitrecording!`).
10. L698-719: Run and patches vs 8d95632's `DrawBinding` walked path.
11. L798-819: acceleration-structure `refit!` naming.
12. L821-892: Memory (`makeroom!`, `arenaroom`, `trim!` caller, `maxalloc`/`ArenaPart`, line numbers).
13. L921-931: hooks wording.
14. L972-1042: JuliaVision census, redone at 8063736.
15. L1180-1291: vocabulary history (179 → 189), the `ext/` coverage claim, tests that pin to-be-deleted behaviour.
16. L1293-1505: phase delete lists and counts (drop the done items; add `makeroom!`, `submitlist!`, `RecordingParts.batches`, `emitloophead!`, `timestamps`, the host KernelInterface executor, `gate!` fallback, `test_headless_rebindable_draw.jl`; fix phase 7 counts).
17. L1549-1553: Open decision 3 (23 commits, not 19).