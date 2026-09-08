# Backend independence: the plan, and what holds it in place

The line: `src/graph/` owns every sequence, layout and decision; `src/vulkan/`
and `src/metal/` contain verbs on their own objects and nothing else. The test
for any function about to be written under a backend is "would the other
backend have to write this too?" If yes, it is core with a hook.

This is held mechanically, not by memory:

* `Mantle.BACKEND_VOCABULARY` (`src/graph/backend.jl`) lists every core function
  a backend may add a method to. Names a step deletes are marked with the step.
* `test/vulkan/test_backend_vocabulary.jl` fails when a backend extends a core
  name outside the vocabulary, defines a local function that shadows a core
  one (a missed `import Mantle:`), or calls a core name it never imported.
* A step is DONE when its names are removed from the vocabulary and that test
  passes, the named tests below pass, and the backend diff has been read and
  every remaining function in it is a verb on a backend object. Not a grep.

Each step names what it deletes before what it adds. Nothing is kept "for now".

## 1. One `Pipelines` phase — DONE 2026-09-07

Core has `run!(::Pipelines, c)` building callables; Vulkan has its own that
also runs the argument cursor and the indirect-slot counter. One core phase
lays out argument slots and indirect slots and calls `compile_dispatch(dev, d,
argoff, ind)` and `compile_draw`. The interpreted answer is a callable with
`argsize` zero, which it already is.

Deletes: `run!(::Pipelines, ::Compile{LavaDevice})`, the `identity.(compiled)`
narrowing, `_bakedraws`' "zero is the honest value" special case.
Vocabulary: `run!` loses its Pipelines method (Barriers stays until step 6).
Tests: `test_plan_indirect_ownership.jl`, `test_devicerange.jl`, `test_host.jl`.
Done: core `run!(::Pipelines)` in `graph/kalaunch.jl` with `checkrenderarea`,
`argsize`, the interpreted `compiledraw`/`compile_dispatch`; Vulkan answers
`compiledraw`, `passbarriers`, `compile_dispatch(::Compile{LavaDevice}, …)`.
Verified: vocabulary, indirect ownership, recorded semantics, allocation, move
patch, traced usages, devicerange, host, the whole windowed file.

## 2. `ArgMemory` layout in core — DONE 2026-09-07

The offsets, the indirect stride and the 256-byte minimum are arithmetic over
what step 1 laid out. The backend answers `argbytes(dev, n) -> (region,
address, ptr)`; the interpreted answer stays `nothing`.

Deletes: the Vulkan `ArgMemory(dev, passes)` constructor.
Vocabulary: `ArgMemory`, `makeargmemory` out; `argbytes` in.
Tests: `test_run_allocates_nothing.jl`, `test_recorded_move_patch.jl`.
Done: core `makeargmemory` over `argbytes`/`indirectslot`; `indirectindex` beside
`argsize`. Verified: vocabulary, allocation, move patch, indirect ownership,
recorded semantics, devicerange, host.

## 3. Prepare and indirect in core, predicates as counts — DONE 2026-09-07

`multi_prepare_indirect_kernel`, which dispatches are device-sized, the fused
prepare and its barrier are a plain kernel plus bookkeeping. Core owns
`emitprepares!`; the backend answers `emitkernel!`, `emitpreparebarrier!` and
`workgroupsize`. A trace's ray count is written by the same prepare (a
"workgroup" of one), so the trace's own prepare in the backend went too.

The `repeat!` gate is folded into the prepare as a count multiplier: a
discarded iteration writes zero groups for every device-sized dispatch of the
pass, on every backend. `supportspredicate` therefore narrowed to "can this
backend discard FIXED-size gated work — a draw, a host-sized dispatch": `true`
in core (an interpreted backend reads the flag between passes), Vulkan answers
with its conditional-rendering flag and keeps the scope as an optimisation,
and `repeat!` refuses a gated pass with fixed-size work only on a backend that
answers `false`.

Deleted: the Vulkan `emitprepares!`, `multiprepare!`, the kernel, the trace's
own prepare in `tracelaunch!`, the blanket refusal in `repeat!`.
Vocabulary: `emitprepares!` out; `emitkernel!`, `emitpreparebarrier!`,
`workgroupsize` in; `supportspredicate` stays with the narrower meaning.
Tests: `test_repeat.jl` (device trip count, the host reading the flag, and the
fold with conditional rendering switched off), `test_devicerange.jl`,
`test_host.jl`, Hikari `test_sample_is_one_run.jl`.

## 4. The run sequence in core — DONE 2026-09-07

Vulkan's `execute!` is the front (stores, patches), the recording, one
submission, the token, and a windowed branch. Core `execute!` becomes:
`e = openrun(dev, pl)`; the update passes through `emitupdate!`;
`emitpatches!` from the core patch table; `replay!(e, pl.recording)`;
`closerun!(e)` returning the token. The windowed frame is the same sequence
with the walk in place of `replay!` and `presentready!` at the end.
`beforeframe!` splits the same way: core collects and polls, the backend
answers `pollwindow!` and `collect!`.

Deletes: both Vulkan `execute!` branches, `emitupdates!`, the Vulkan
`beforeframe!`, `timings(::Plan{LavaDevice})`.
Vocabulary: `execute!`, `beforeframe!`, `timings` out; `openrun`, `replay!`,
`closerun!`, `pollwindow!`, `presentready!` in.
Tests: `test_recorded_run_semantics.jl`, `test_closed_command_buffers.jl`,
`test_window.jl`, RayMakie `test_window_frame.jl`.
Done: core `execute!` (front, recording, windowed walk), `beforeframe!`,
`emitupdates!`, `emitpatches!`, `timings`; hooks `recordsplans`, `openrun`,
`closerun!`, `abandonrun!`, `emitinline!`, `beginframe!`, `collect!(prof, dev)`;
`openoneshot` split out of `oneshot`. Verified: vocabulary, recorded semantics,
closed command buffers, allocation, waitidle, move patch, repeat, devicerange,
host, the whole windowed file. Seen twice and not reproduced: `M.Window`
spinning at 100 % CPU right after a killed worker's windows went away; the
same call succeeded minutes later and the compositor answered throughout.

## 5. One store path — DONE 2026-09-07

Inline-or-staged is a decision by size and alignment: core makes it, the
backend answers `emitinline!` (bytes inside the command buffer) and
`emitcopy!` (from a staging region). Metal answers both with a blit.

Deletes: `updatestore`, the three `inplace!` methods, `emitstore!`.
Vocabulary: `inplace!` out; `emitinline!` in (`emitcopy!` exists).
Tests: `test_recorded_run_semantics.jl` ("argument memory is byte-identical"),
`test_window.jl` ("a store writes at the reserved position").
Done: core `StoreEmitter` and `emitstore!` (a `GPURef` from its cell, a
`Buffer` range from its vector) over one primitive `storebytes!(e, store, off,
ptr, n)`; Vulkan carries it inline within `vkCmdUpdateBuffer`'s limits and
stages past them through `upload!`. The device-source variant was unreachable
(a buffer's pending list holds host vectors only) and the old staged fallback
called a two-argument `upload!` that did not exist. Deleted `inplace!` ×3,
`updatestore`, the backend `emitstore!`/`StoreEmitter`. Verified: vocabulary,
recorded semantics, allocation, move patch, host, the whole windowed file.

## 6. Barriers: core phase, backend lowering — DONE 2026-09-07

`run!(::Barriers, ::Compile{LavaDevice})` was 170 lines; the state tracking,
seeding from the end of the schedule, segment handovers and the per-pass
`transition!` walk are policy, and were already written over `sync/transition.jl`
with the backend only as a `needs_transition` answer. Core now runs the phase in
`phases.jl` over `syncbackend(dev)`; the mask lowering stays where it was, in the
backend's `passbarriers` (step 3), which is asked once per pass at emit time.
Metal's `run!(::Barriers) = c` and the whole Vulkan method are gone.

What did NOT happen, and why: the plan named `lowerbarrier` and `subsumes` as
new hooks. Reading the method showed the coalescing state (`settled`,
`last_barrier`, `covered_*`, `widen`, `subsumed`) was dead — computed, never
read, since the per-resource hazard set replaced it — so there is nothing to
lower or subsume in the phase. Deleted rather than hoisted.

Deleted: both backend `Barriers` methods, the dead coalescing state,
`EMPTY_HANDOVER` in the backend.
Vocabulary: `run!` out entirely; `syncbackend` in.
Verified: vocabulary, `test_barrier_skip.jl`, `test_traced_usages.jl`,
`test_recorded_run_semantics.jl`, `test_host.jl` (the host answers
`HostAPI()` and derives nothing), the whole windowed file including "the
emitted set is the per-resource hazard set", "lowered to mask tuples",
"derived barriers agree with the backend on random DAGs".

## 7. One record of what is in flight — DONE 2026-09-07

`Outstanding{T,P}` carries the backend payload (Vulkan: the `Submission`, with
the one-shots it owns and the recordings it pinned); core `sweep!` hands each
passed entry to `recycle!(queue, payload)` before dropping it. `bq.in_flight`
and `sweep_retired!` are gone, and so is the second sweep in `submit!`.
`submitted!(q, token, payload; tag)` is the one call that records a
submission; `recycle!(q, ::Nothing)` is the method a backend whose
submissions own nothing gets for free.

What stays in the backend: `drain!(bq)` — core `sweep!` plus the two
deferred-free drains (buffers and acceleration structures the GC released
while a submission still named them), because those lists are Vulkan's manual
destruction, not a second record of what is in flight. The stall report,
`waitfor`'s diagnostics and `show_memory_report` read `outstanding`.

Deleted: `in_flight`, `sweep_retired!`, the second sweep in `submit!`.
Vocabulary: `recycle!` in (declared in `graph/submission.jl`, imported by the
extension again — it was one of the seven names the import audit struck).

Found the day after, by RayMakie's `test_overlay_compositing.jl` (blank
overlay, then a lost device): core `sweep!` asked `passed(dev, token)`, and
the Vulkan device answers that from its DEFAULT queue's counter, so RayMakie's
graphics queue — a second `VulkanBatchQueue` with its own timeline — had its
submissions swept the moment they were made and their command buffers begun
again mid-flight. The old per-queue `sweep_retired!` had compared against its
own `query_timeline(bq)`; moving the sweep into core inherited the
device-shaped question. `sweep!(queue)` asks `passed(queue, token)` now, and
`test_sweep_per_queue.jl` gates a second queue's submission on the default
timeline so the misreading is deterministic. Lesson, recorded in memory: a
warm-session check after ANY queue change includes RayMakie's overlay test,
because nothing in Mantle's own suite drives two queues at once.
Verified: vocabulary, `test_ext_imports_are_declared.jl`,
`test_closed_command_buffers.jl` (now pins that `in_flight` is gone and that
`outstanding` carries the `Submission`), `test_waitidle_submits.jl`,
`test_run_allocates_nothing.jl` (still zero bytes per run),
`test_recorded_run_semantics.jl`, `test_phase1_pin.jl`,
`test_free_during_recording.jl`, `test_argument_memory_isolation.jl`,
`test_caching_and_allocations.jl`, `test_pinned_buffer_lifetime.jl`,
`test_pin_leaves_stops_at_tlas.jl`, `test_gpu_memory_safety.jl`,
`test_arena_recording.jl`. Two stale tests fixed on the way: `test_phase1_pin`
read a `last_write` field that had been split into `last_write_bq`/`_val`, and
`test_arena_recording` ran plans it had never recorded (step 4's rule).

## 8. Trace pinning — DONE 2026-09-07

"The acceleration structures a trace reads" is core's `pintrace!(owner, tlas)`
in `raytracing/api.jl`: `pin!(owner, tlas)`, then `pin!(owner, blas)` for each
of `blases(tlas)`. The backend answers `pin!` for its two level types (handle
plus storage) and `blases` for its top level. Six copies went, not three: the
recorded walk's `pintrace!`, the loops in `trace_rays!`, `trace_rays_indirect!`
and `bindtlas!`, and two top-level-only pairs inside `emit_trace!` and
`emit_trace_indirect!` that nobody had counted.

What the test found: the TLAS a `VulkanTLAS` builds (`build_tlas` from an
instance buffer) carried `LavaBLAS[]`, with a comment saying the VulkanTLAS
pinned the BLASes "at the higher level". It did not — every trace path pinned
`tlas.blases`, which was that empty list — so a BLAS `sync!` dropped could be
destroyed under a trace still walking it. `build_tlas` now takes `blases`,
`rebuild_hw_tlas_from_batch!` passes the compacted `blas_list`, and
`test_pintrace.jl` pins that both levels of a two-mesh TLAS are held through
either owner.

Deletes: `pintrace!` in the backend, the four loops, the two partial pairs.
Vocabulary: `pin!`, `blases` in.
Verified: vocabulary, `test_pintrace.jl` (new), `test_pin_leaves_stops_at_tlas.jl`,
`test_phase1_pin.jl`, `test_closesthit_via_rayquery.jl`,
`test_hwadapted_via_rayquery.jl`, `test_hwtlas_uaf_safety.jl`,
`test_hwtlas_mesh_update.jl`, `test_ext_imports_are_declared.jl`. Hikari's
`test_sample_is_one_run.jl` runs in the end-to-end session below.

## Found on the way: a resize noticed at acquire — FIXED 2026-09-07

`test_window.jl` lost the device in a resize testset one run in three, on the
RTX, before and after every step above. `out/resize_race_mwe.jl` reproduces it
in one or two resize cycles and named the path: `GLFW.GetFramebufferSize` asks
X directly, so a resize can land between `beginframe!`'s swapchain sync and
the acquire, and the acquire's own sync then rebuilt the swapchain UNDER a
frame whose attachments `refit!` had already sized for the old one — a
1200x900 colour target beside an 800x600 depth target, which is
VUID-VkRenderingInfo-pNext-06079 and a lost device two frames later. RADV
draws it wrong instead of faulting, which is why the resize testsets ever
passed there.

The order was the bug, and it was core's: `run!` refit before `openrun`
acquired. `beforeframe!` now acquires right after `beginframe!`, ahead of
`refit!`, so whatever the acquire notices is what the frame is refit for;
`openrun` opens the buffer and nothing else, on every backend. A frame that
fails after its image was acquired hands it back through the new
`abandonframe!` hook (Vulkan: an untouched present through the same
submit-and-present as a drawn frame, so the slot's fence and semaphores stay
paired; Metal: drop the drawable; interpreted: nothing), so the next frame
can acquire — a `checkextents` error no longer leaves the window holding an
image.

Vocabulary: `abandonframe!` in. Test: "a resize the acquire notices is refit
before the frame is recorded" in `test_window.jl`, forced rather than raced
(the window is resized between the two steps `run!` takes), with a structural
first assertion that gates the behavioural rest because the behaviour it
guards is a lost device. Fails before the fix, passes after; the MWE ran 12
cycles (36 resizes) clean afterwards where it faulted within two.

## Found on the way: a discarded bounce traced garbage — FIXED 2026-09-08

RayMakie's `test_transform_update_hwtlas.jl` lost the RTX on its first render,
every time, in a fresh session; the same box scene rendered after another
scene had run, and rendered identically at depth 1, 2 and 4. A sentinel in the
indirect slots (`test_discarded_iteration_prepare.jl`) named it: the fused
prepare of a `repeat!` iteration was emitted INSIDE the predicate scope, so
conditional rendering discarded it with the iteration, and the slot kept
whatever the memory held. A dispatch inside the scope is discarded too, so
nothing noticed — but `vkCmdTraceRaysIndirectKHR` is not subject to
conditional rendering (VK_KHR_ray_tracing_pipeline leaves the trace commands
out), so the trace of a discarded bounce ran with a ray count read from an
unwritten slot: garbage from freed BLAS scratch in a fresh session (lost
device), a stale small count from the previous plan's argument memory
otherwise (a quiet wrong render). Hikari's rays all leave a convex box after
the first bounce, so the loop discards from the second; black_hole discards
~90 rounds, which is where its "DEVICE_LOST twice in long sessions" came
from. RADV runs trace commands as compute, which conditional rendering does
discard, so it never showed.

The fix is in the walk, for every backend: `emitpass!` emits the fused prepare
before it opens the predicate scope, so the fold — zero rays or groups for a
discarded iteration — is what the trace reads. The `Predicated` usage lowers
to the compute stage as well, because that fold reads the flag in a compute
kernel and the barrier from the gate's write had only covered the
conditional-rendering read.

## Found on the way, in the consumers' suites (2026-09-08)

Running RayMakie's and Hikari's whole suites (not the named files) after the
steps found three more, all fixed:

* Hikari's denoiser built its plan with `Ref`s for the three sigmas — read
  once at `record!`, which Mantle refuses since the store API — so every
  `denoise!` errored. They are `GPURef`s owned by the plan's `DeviceMemory`
  now; the kernel reads one-element device arrays. `denoise.jl` green (24).
* `LavaBackend() == LavaBackend(ctx)` was false: the default `==` compared the
  unresolved queue fields (`nothing` against a queue), so an array the default
  backend allocated did not belong to it as far as RayMakie's meshscatter tests
  could tell. Equality resolves the queues (`test_backend_equality.jl`).
* RayMakie's `test_materials_scene.jl` activates the CPU backend and leaves it
  active; the graphics files after it made CPU screens (blank overlays,
  `HWTLAS(::CPU)`). `runtests.jl` puts the GPU back before them.

* RayMakie's "instance count change visibly drops lit pixels" rendered NINE
  spheres after the scene was down to one — the images were identical, and
  not because of accumulation. Hikari's `invalidate!` dropped the plans but
  kept the cached ADAPTED scene, and a software TLAS is adapted as a snapshot
  of its node and instance arrays; the plans rebuilt after the edit were built
  from the old snapshot, and in Hikari's own flow the accel's dirty flag was
  never even consumed (nothing re-adapted, so nothing synced). `invalidate!`
  drops the adaptation and the cached initial-medium key with the plans;
  `test_plan_invalidation.jl` now asserts the flag is consumed and the plans'
  accel is the current snapshot (failed before, passes after). RayMakie's
  whole suite is green with it.

Still open there, none of them Mantle's: two pbrt references rendered at 512
spp against a 256 spp run (regenerate or set `HIKARI_PBRT_SPP`), and JET not
in the env (two Hikari files).

## What Metal is afterwards

`openrecording` keeps answering `nothing`: a Metal command buffer is single-use,
and re-encoding per run costs 0.3–2 % (measured 2026-09-07 on the RTX, RayDemo
A/B), so no indirect command buffers and no MSL codegen. Metal implements
`openrun`/`closerun!` on a command buffer that signals its shared event,
`passed` and `waitfor` on `signaledValue`, `emitkernel!` and `emitindirect!`
through Metal.jl's binding, `emitinline!` and `emitcopy!` as blits,
`syncbackend`/`needs_transition` and `passbarriers`, timestamps for `profiled!`,
`pin!` for its two acceleration-structure levels and `blases` for the top one,
`abandonframe!` (drop the drawable — written, unrun), nothing for `recycle!`
(its submissions own nothing, so the payload is `nothing`), plus the render and
present verbs it has. Under twenty short methods; Hikari runs on it through
the same core walk with nothing changed in Hikari. None of it can be run on
this machine — the Metal methods are verified on a Mac or not at all.

## Order and verification

Steps 1–8 in that order; each keeps Vulkan and the host backend green in the
warm session on its named tests plus `test_backend_vocabulary.jl`. The full
suite runs once, at the end.

**Full suite, 2026-09-07, RTX 4000 Ada, `out/mantle_full_suite_2026_09_07.log`:**
36161 passed, 23 failed, 127 errored, 2 broken, 26 min. Everything the refactor
touched was green; what failed and what became of it:

* Tests pinning the OLD lazy-record contract (`run!` recorded on first use):
  `test_record_after_run.jl` (55 errors), `test_record_does_not_execute.jl`,
  `test_recordable_plans.jl`, `test_arena_recording.jl` — rewritten to the
  step-4 contract (record at build, an unrecorded plan is refused) and green.
* Stale from earlier steps, fixed and green: `test_compile_golden.jl` (the
  `updates` pass is first in every plan now), `test_phase3_lifecycle.jl`
  (`last_write` split), `test_phase6_graphics.jl` (`vk_draw!` takes an emitter),
  `test_handwritten_rt.jl` and `test_blas_refit.jl` (`rt_dispatch!` → one-shot +
  `pintrace!` + `emit_trace!`), `test_window.jl` in its own process
  (`VulkanWindow`, `copy_framebuffer!` and `bench/chain.jl`'s framebuffer were
  names only a warm session had), `runtests.jl` included a deleted
  `test_compile_overhead.jl`.
* A real bug the GPUArrays testsuite found (60 errors): the 0-norm of every
  `LavaArray` called Mantle's `count` (the extension imports it for `Buffer`)
  instead of `Base.count`. Fixed in `array/gpuarrays.jl`.
* NOT fixed, environment: `AcceleratedKernels` and `DNNKernels` are not in the
  project env (5 testsets); `two devices in one process` needs lavapipe, which
  `VK_DRIVER_FILES` pinned to the NVIDIA ICD hides.
* End to end after the sweep fix (2026-09-08, fresh full session, RTX):
  Hikari `test_mantle_device.jl` (its five plans now `record!`ed),
  `test_volpath_graph.jl`, `test_volpath_per_iter_lifecycle.jl`,
  `test_trace_pass_modelled.jl`, `test_plan_invalidation.jl`,
  `test_sample_is_one_run.jl`; RayMakie `test_overlay_compositing.jl` (20),
  `test_window_frame.jl`, `test_render_allocates_nothing.jl` (measuring after
  a GC — see the file). Per sample, baked: killeroo 21.4–21.8 ms, black_hole
  30.5 ms, the same as before steps 6–8.
* NOT fixed, Lava codegen on NVIDIA, untouched by this refactor:
  `test_int32_cartesian_miscompile.jl` (3), `test_shared_index_division.jl` (3),
  `test_static_workgroup.jl` (12 — "the failure law no longer fires" fires:
  coverage 0.125 of 1.0). Their headers attribute the fault to NVIDIA's
  compiler; whether the `WORKGROUP_FALLBACK` repair regressed is a Lava
  question, to be bisected on its own.
