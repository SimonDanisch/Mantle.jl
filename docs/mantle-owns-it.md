# Mantle owns it: delete first, then build

The line is already written down, in three places, and it is not being followed.

* `graph/backend.jl:3-11` — "Mantle owns the graph … Anything longer than this
  list means graph logic leaked back into a backend."
* `metal/device.jl:1-8` — "Everything above it — which block, what offset, when
  to grow, when to coalesce, when to release — is `Mantle.Pool`'s. A backend
  that finds itself making placement decisions here has taken work that is not
  its own."
* `graph/submission.jl:12-18` — "Five records of one fact is five chances to
  read the wrong one … there was no single place that knew what was
  outstanding."

Every finding below is one of those three sentences being broken. The rule they
share, stated once so the rest of this document can refer to it:

> **A backend implements verbs on its own driver objects. Every decision — what
> lives where, how long, in what order, under what name — is Mantle's, and
> Mantle's alone. A function in a backend that contains no driver call is in the
> wrong package.**

## Why this refactor deletes before it builds

The previous attempts at this failed the same way each time, and the failure is
mine rather than the code's: **a finished implementation is a reference, and I
reproduce its shape.** Three instances from one session, 2026-09-08:

* `hostview` existed four times — the host backend, the Metal backend, the pool
  fixture, and a `reinterpret` variant that could not carry a padded struct. I
  had read three of them before writing the fourth.
* Asked whether `pin!` belongs in a portable vocabulary, I read Vulkan's 35 call
  sites, concluded "this is Vulkan's private lifetime management, which is
  fine", and had to be told twice that lifetime management is exactly what
  Mantle must own.
* Writing up that same finding I said "the backend answers `recycle!` — Vulkan
  destroys or returns to the pool", fifteen minutes after quoting the sentence
  that forbids a backend deciding anything about a pool.

In each case the existing implementation was in front of me and I described it
instead of the design. So the order below is not a preference:

**Phase 1 deletes every instance of every problematic pattern, in one commit,
and leaves the tree broken.** Not deprecated, not `@warn`-ed, not moved to a
`legacy/` directory — deleted, so there is nothing to read and nothing to copy.
It is expected and correct that Mantle, Hikari and RayMakie do not load at the
end of Phase 1. Nothing is rebuilt until it can be rebuilt in the right place.

If, during Phase 2 onwards, the answer to "how should this work" is reached by
reading what used to be there, the deletion was not complete. Fix the deletion.

## Phase 0 — the guards, before anything is deleted

After Phase 1 everything is red. A red suite cannot tell "not built yet" from
"the pattern came back", so the guards go in first, while the tree is green,
and each one FAILS on today's code. That failure is the proof the guard works;
they are committed failing, with the finding they pin named in the test.

**0.1 No vendor name in the portable vocabulary.**
`BACKEND_VOCABULARY` contains `:vkformat`. Assert that no entry matches
`^(vk|mtl|lava|VK_|MTL)` — the same rule CLAUDE.md states for code paths.

**0.2 Every vocabulary name is answered by core or by every backend.**
Today 32 of 123 are answered only by Vulkan, with no core default. That is the
signature of a hook that is either not portable (belongs in the backend) or not
implemented (belongs on a list). Assert: for each name, EITHER a method exists
outside both backend directories, OR every loaded backend has one. A name that
is neither must be listed in a literal `NOT_PORTABLE` set in the test, with a
comment — so the decision is written down once rather than rediscovered.

**0.3 No function without a driver call in a backend directory.**
The mechanical form of the rule. For each method defined under `src/vulkan/` or
`src/metal/`, assert its body names at least one symbol from the driver's
namespace (`VK.`/`vk_`, `MTL.`/`Metal.`), or is listed in an explicit
`PURE_BOOKKEEPING_ALLOWED` set. `recycle!` is the finding that motivates it: one
implementation, no `vkDestroy`, no `vkFree`, no driver call at all.

**0.4 Core names no backend.**
`src/` outside `src/vulkan/` and `src/metal/` must not contain `Lava`, `Vulkan`,
`Metal`, `MTL` or `vk_` in code position (comments and docstrings excluded, as
in `test_ext_imports_are_declared.jl`'s `free_names`). Extend that file's
scanner, which was fixed on 2026-09-08 to stop reporting keyword parameters,
positional parameters and `using M: x` module paths as free references.

**0.5 Downstream names no backend at all.**
The rule is `using Metal, Mantle` and not one backend name after it, in source
AND tests. Assert over `Hikari/`, `RayMakie/` and `Raycore/`. Raycore passes
today with zero violations and is the model; Hikari has 2 source files and 35
test sites, RayMakie 11 source files.

**0.6 One device per process.**
`caps(::MetalBackend)` builds a second `MetalDevice` with its own `Pool`,
`MTLCommandQueue` and `MTLSharedEvent`, through a second cache (`_DEVICE` in
`caps.jl`) beside `METAL_DEVICE` in `device.jl`. Assert
`Device(MetalAPI()) === <whatever any public entry point returns>`.

**0.7 The suite itself is backend-partitioned, and that is why this list is
long.** `runtests.jl` is two sections, `if _VULKAN_OK … end` and
`if _METAL_OK … end`, over two file lists: 188 files under `test/vulkan/`, 9
under `test/metal/`. That much is honest — 185 of the 188 genuinely test Vulkan
internals, and only three are misfiled (`hwtlas_helpers.jl`,
`narrow_phase_helpers.jl`, and `test_backend_vocabulary.jl`, which counts double
because it is the guard that would have caught the drift and has never run on a
machine without a Vulkan driver).

The problem is the layer above. `test/*.jl` looks shared and is not: it
hardcodes `VulkanAPI()` 57 times against `MetalAPI()` once —
`test_window.jl` 35 times in 2,021 lines, `test_arena_recording.jl` 10,
`test_compile_golden.jl` 4, `test_devicerange.jl` 3. Portable behaviour —
windows, surfaces, resize, presentation, arena recording, the compile golden,
device ranges — is therefore tested against exactly one backend, which is the
same violation as Hikari's 35 `MVE.LavaBackend()` sites and has the same cause.

It is why the clip-space flip and the window extent were found by a person
looking at a window rather than by `test_window.jl`.

*Guard:* assert that no file in `test/*.jl` names a backend API marker; the
portable ones take a device from a `for backend in availablebackends()` loop.
*Cleared by:* a shared, parameterised set, before phase 2 — because phase 2
cannot be checked on one backend, and building it on one backend is how the
current state was reached.

## Phase 1 — delete, all at once, and let it break

One commit. The tree does not load afterwards. Each item says what goes; none
says what replaces it, because that is Phase 2's job and knowing the replacement
while deleting is how the shape gets copied.

**1.1 Lifetime.** `pin!`, `blases`, `pintrace!`; every one of the 35 call sites;
the `pinned` and `pinned_refs` fields on `Closed`/`OneShot`/`Submission`;
`release_pinned_refs!`; `unpin_buffer!`. From `BACKEND_VOCABULARY`: `pin!`,
`blases`.

**1.2 Object pools in the backend.** `recycle!` and its two methods;
`bq.free_oneshots`, `bq.free_submissions`, and the four
`isempty(free_*) ? new : pop!` sites. From `BACKEND_VOCABULARY`: `recycle!`.

**1.3 Duplicated acceleration-structure bookkeeping.** `_register_batch!` in
both backends; the `instance_batches`, `handle_to_batch_idx` and
`next_handle_id` fields of `VulkanTLAS` and `MetalHWTLAS`; the handle allocation
in both. `build_blas` keeps its two bodies for now — they ARE driver calls —
but loses the shared name.

**1.4 The shader vocabulary in the wrong package.** `graphics/builtins.jl` and
`SHADER_BUILTINS` out of Mantle; the two bridge loops
(`vulkan/graphics/api.jl:508-512`, `metal/graphics.jl:962-966`) that exist only
because the declaration sits where Lava cannot reach it; Lava's own
`gfx_intrinsics.jl` definitions of the same names; the 13 `lava_rt_*`
declarations. Nothing is added to KernelInterface in this phase.

**1.5 Downstream backend names.** Hikari's `rt-pipeline.jl` import block and
`precompile_statements.jl` (138 sites, generated, names `LavaArray`,
`LavaBackend`, `LavaDevice`, `MVE`); RayMakie's `import Lava` blocks and the
`LavaRenderObject` type; the 35 `MVE.LavaBackend()` sites in Hikari's tests and
the `MVE === nothing && error(...)` gate in its `runtests.jl`. The files
`lava_scatter.jl` and `lava_lines.jl` are renamed in this phase; their contents
are Phase 2's.

**1.6 The second Metal device.** `_DEVICE` and `_device_for` in `caps.jl`.

**1.7 The dead and the contradictory.** `vkformat` out of the vocabulary.
`allocate_batch_queue!`'s error text, which tells the caller to check
`supports_graphics` and take the compute path, on a backend where
`supports_graphics` is now `true`. `fence(::MetalDevice)`'s "Read, never forced"
docstring, which describes a function that increments a counter. The
"ColorTypes is not a dependency of Mantle" claim in three places — ColorTypes is
in `[deps]`; what is missing is one `using`.

**1.8 One of the two barrier stories.** `metal/ka.jl:15-21` and
`metal/device.jl:167-169` give opposite reasons for the same behaviour, and the
first rests on "every pooled allocation here is `Shared`", which
`device.jl:99` and `images.jl:72` contradict. Delete both comments. Which one
was right is Phase 2.5's question and must be answered by measurement, not by
whichever paragraph survived.

## Phase 2 — rebuild, in dependency order

Each step: what it adds, where, and the test that says it is done. A step is
not done because it runs; it is done when its test fails without it.

**2.1 KernelInterface owns the device vocabulary. — DONE 2026-09-08**
`lib/KernelInterface/src/graphics.jl` and `raytracing.jl`, beside the existing
`device.jl` which already declares `get_global_id`, `barrier` and
`sub_group_reduce_add`. Both Lava and Mantle depend on KernelInterface; Lava
does not depend on Mantle, which is the whole reason the bridge existed.
Lava overrides directly. Mantle re-exports so `using Mantle` stays enough.
Names carry no vendor: `lava_rt_trace_ray` becomes `rt_trace_ray`.
A backend that cannot answer one (Metal has no geometry shader stage) says so
in a capability query — see 2.6 — rather than by a missing method discovered at
compile time.
*Done when:* a shader source naming only KernelInterface builtins compiles on
both backends, and `test_ext_imports_are_declared.jl` reports no bridge.

**2.2 Mantle owns lifetime.**
`Outstanding(token, payload, tag)`, `submitted!`, `sweep!` and `passed` already
exist and already carry a payload — its docstring even says the payload holds
"the closed command buffers the submission carried, with their pins and
scratch". What is missing is that anything uses it. Core keeps, per submission,
the list of objects it must hold; `sweep!` releases them when `passed` says so.
The backend's only remaining part is the primitive that actually destroys a
driver object, in the shape `rawfree` already has.
*Done when:* no backend file contains a free list or a pin list, and
`test_gpu_memory_safety.jl` plus `test_pinned_buffer_lifetime.jl` pass against
the core mechanism.

**2.3 Mantle owns object reuse.**
One-shots and submissions are pooled objects. `Pool` is Mantle's and its
discipline — which block, when to grow, when to release — is the same question.
Either they go through `Pool`, or through something core-owned with the same
contract; not through a `Vector` on a backend queue.
*Done when:* `test_pool.jl`'s fixture, which drives 8,927 assertions against a
device that allocates nothing, also covers object reuse.

**2.4 Acceleration structures are graph resources.**
Today an AS travels as a kernel argument (`rawargs`/`devargs`); `use(p, tlas)`
does not exist, so the graph never sees the TLAS→BLAS edge, and `pintrace!`
exists to patch the hole from outside. Declare them: `use(p, tlas)` implies the
BLASes it instances. Then liveness covers them, `Raycore.sync!` cannot swap one
out from under a pass the graph knows reads it, and 1.1's deletion needs no
replacement at all.
*Done when:* `test_pintrace.jl`'s property — both levels held through either
owner — is asserted through the graph, with no `pin!` in the tree.

**2.5 The instances a TLAS holds. — DONE 2026-09-08**
`InstanceBatches{B}` in `raytracing/batches.jl` owns the order, the handles,
the lookup and the reindex on delete; a backend supplies only the batch type.
Metal's `_register_batch!` is one call to `register!`. Handles are monotone and
never reused, which neither deleted copy said out loud.
*Verified:* `test_hwtlas_metal.jl` green, whole suite 9335 passed 0 failed.

**2.6 One barrier story, chosen by measurement. — DONE 2026-09-08**
`needs_transition` for `MetalAPI` falls through to the generic `true`;
`passbarriers` is core's empty default; so the plan carries no transitions and
correctness rests on `MTLHazardTrackingModeTracked`. Host and WebGPU each state
their answer with a method and a reason, and `sync/backend.jl:49-52` says a
backend whose lowering is empty should answer `false`. Metal must do one of two
things and say which: answer `false` explicitly, or implement the lowering and
make the heap `Untracked`. Measure both — a tracked heap serialises.
*Measured*, a chained graph (render A, copy A, render B, copy B) at 2048x2048,
median of five runs of forty frames:

    Tracked      0.364 ms/frame     4194304 pixels correct
    Untracked    0.332 ms/frame     4194304 pixels correct

Both correct, and hazard tracking costs about 9% rather than the order of
magnitude "a tracked heap serialises" assumed. The reason ordering holds is
neither comment's: it is COMMIT ORDER. Since phase 1.2 each render and copy pass
opens its own command buffer and commits it before returning, all on one queue,
and Metal runs a queue's buffers in commit order.

So `needs_transition(::MetalAPI, …)` answers `false` explicitly, with that
reason — and the heap stays `Tracked` on purpose. Untracked is safe only BECAUSE
of one buffer per pass; if phase 2.3's object reuse ever batches two passes into
one buffer again, commit order stops separating them and the 9% becomes a race.
The flip is one line to make once 2.3 is settled.

`test_compile_golden.jl`'s `all(r.barriered)` reads the answer now instead of
assuming one, which is what made it portable rather than what silenced it.

**2.7 Capabilities are queries, not exceptions.**
`compile_pipeline` throws on geometry shaders and on tessellation;
`GraphicsPipeline` has both fields; `DeviceCaps` has no answer and
`supports_graphics` is one Bool. A caller cannot ask. Add the queries, and let
`GraphicsPipeline` construction fail against a device that says no.
*Done when:* RayMakie can ask whether to take its geometry-shader path instead
of finding out at compile time.

**2.8 Portable window and format.**
Add `using ColorTypes: RGBA, BGRA` to Mantle — the dependency is already
declared and paid for — then `mtlformat` matches on types instead of on
`nameof(T)`, and Metal answers `Mantle.Window(backend, w, h)`, which is in the
vocabulary and which Lava already answers. The Retina sizing currently in
`bench/showcase_metal.jl` belongs in that constructor.
*Done when:* `bench/showcase.jl` is ONE file that picks a backend, and
`showcase_metal.jl` is gone.

**2.9 Downstream imports Mantle only.**
Hikari and RayMakie drop `Lava` from `[deps]`. RayMakie's shaders move from
Lava's location-based `gfx_output`/`gfx_input`/`set_position!` to Mantle's
declarative varyings — `varyings = (albedo = Vec4f, …)` and a vertex stage that
returns `(position = …, …)` — which both backends already support and which
`bench/showcase.jl` exercises. That removes seven of the names outright rather
than porting them.
*Done when:* 0.5 passes, and Hikari's suite runs on Metal.

**2.10 The array algorithms.**
`array/fft.jl` (893 lines) and `array/gemv.jl` (485) are core;
`vulkan/array/gemm.jl` (2335) is not. One of those placements is wrong. Decide
which, with the same question: does it name a driver type?

## What "not going back" means mechanically

Phase 0's guards stay green for the rest of the project. In addition, three
habits, written here because each was violated today:

1. **"Defined in both backends" means "probably implemented twice", not
   "interface method".** `_register_batch!`, `build_blas` and `matrixshapes`
   were reported as vocabulary violations on a name match; all three are
   private helpers with different signatures. The duplication was real; the
   diagnosis was not. Read both bodies before naming the finding.

2. **A comment is evidence, not proof.** Three separate places assert that
   ColorTypes is not a Mantle dependency; it is in `[deps]`. Two comments in
   one backend give opposite reasons for the same behaviour. Check the claim
   against the code before building on it.

3. **When the question is "where does this belong", do not open the file that
   already implements it.** Answer from the rule at the top of this document,
   then open the file to see how far away it is.

## What 0.7 found the moment the portable tests ran on a second backend

`test_compile_golden.jl` and `test_devicerange.jl` had never executed against
anything but Vulkan. Running them per backend cost nothing to arrange and
immediately produced six failures, none of which is breakage — each is a
portability gap that was there all along and had no way to be seen. Three are
fixed; three remain and each maps to a phase.

1. **`test_compile_golden.jl:95`, `all(r.barriered)`.** Nothing is barriered on
   Metal, because `passbarriers` is core's empty default there. This is the
   assertion that phase 2.6 has to make true or make backend-aware, and it is
   the first hard evidence for which of the two deleted comments was right.

2. **`test_devicerange.jl:145`, the tail — FIXED, and it was the test.** 4096
   where it wanted 832, and both are right: a recording backend reads the count
   on the device and launches whole workgroups over it, while a
   KernelAbstractions backend has no indirect dispatch and launches the CEILING
   on purpose — resolving the range by reading the count on the host means
   synchronising before every such dispatch, measured at 320 synchronises and
   0.585 s of a 0.602 s frame on an M5. `test_host.jl` states that contract and
   the equivalence it rests on. The assertion asks `recordsplans(dev)` now,
   which is the same distinction under a name the vocabulary already has.

3. **The window subprocess on Metal.** It now runs (the DISPLAY guard was an
   X11 question, and the file skipped itself on macOS), and it fails — 26 of
   its assertions still reach for `MVE`, which phase 2.8 removes.

4-6. **Three pool tests — FIXED.** Not a regression: with 0.6's second device
   cache gone there is genuinely ONE Metal device per process, so the portable
   tests that now run before them left the pool in a state their assertions
   could not see. All three were statements about running FIRST rather than
   about the pool — "a new block was allocated", "the acquire lands at offset
   0", "trim! leaves nothing reserved". They take a private `Pool` over the same
   real device now and hold in any order.

Two remain, and neither is papered over:

* **1 stays red on purpose.** Making it backend-aware would encode "Metal emits
  no barriers" as correct, which is exactly the unsettled claim phase 1.8
  deleted. It is the flag for 2.6 and it should stay visible until 2.6 measures.
* **3 is the rest of 2.8**: 26 assertions in `test_window.jl` still reach for
  `MVE` to drive a window without a graph, and `Window(backend, w, h)` is what
  replaces them.

Four of the six were fixed by making the test say what it meant. That ratio is
worth remembering: most of what a second backend "breaks" is a test that was
describing one backend and calling it a property.
