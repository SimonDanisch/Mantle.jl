# One way to submit

Working plan for removing the heuristic recording path. Written 2026-09-02, part
way through. Read `docs/design.md` first for what the graph is.

## The problem, stated once

**Sixteen fields on `BatchQueue` existed because something records GPU work
without a plan. Fourteen are gone in step 5; step 7 takes the last two.**

```
auto_submit_threshold  cb_split_threshold  barrier_mode  barrier_elision
next_skip_barrier  ranges_declared  touched_ranges  dispatch_ranges
deferred_indirect  scope_depth  capturing
arg_slabs arg_slab_idx arg_slab_offset arg_pool_frontier
indirect_slabs indirect_slab_idx indirect_slab_offset
```

They guess what a compiled plan already knows: when to cut a command buffer,
which barriers can be elided, where argument memory lives. A plan knows the pass
order, the dispatch counts, the declared usages and the placement — exactly and
at compile time.

**And they contaminate baking.** `bake!` routes through the same recorder, so a
"baked recording" is the heuristics' output, frozen. Measured on Hikari's fused
sample: 185 dispatches, **5 command buffers per recording**, cut by
`auto_submit_threshold` firing mid-capture — a threshold about *when to submit*,
applied while nothing is being submitted. Any performance number taken from that
path measures the heuristics, not the design. **Every benchmark in this session
is therefore provisional.**

## Target

Three operations, never conflated:

```
Plan(graph)     compile — DAG from declared usage, schedule, derive sync points, place
record!(plan)   emit — walk the flat pass list, write commands into ONE command buffer
run!(plan)      write per-run values, submit. Never records.
```

`run!` has no branch because there is nothing to branch on: no interpreted mode.

**Mantle owns all bookkeeping and organising, backend-independently.** The graph,
the plan, the recording, lifetimes, placement, the schedule, which
synchronisation points exist, what a recording owns. A backend supplies
primitives only: `rawalloc`/`rawfree`, make a command buffer, emit a barrier,
emit a dispatch, submit, and map a portable `Usage` to its own flags.

**Parallelism is barrier absence.** The schedule decides what overlaps; between
independent passes the emitter writes nothing and the GPU pipelines them. One
queue. Multiple queues stay available as a deliberate `Policy` for dissimilar
work (compute over transfer), synchronised with timeline semaphores — never as
the way to express DAG parallelism.

**Barriers: precise placement, anonymous contents.** Where a barrier goes is
decided per resource, at compile, from `use(p, x; read/write)`. What it carries
is the distinct `(srcStage, srcAccess, dstStage, dstAccess)` tuples among that
pass's hazards — `unique`, never unioned. No handle, no offset, no range: the
range is inert (cache invalidation is per cache, not per address) and the
precision that matters lives in the masks. Images keep `VkImageMemoryBarrier2`
because a layout transition is genuinely per-image.

Consequence: **a recording names no `VkBuffer`**, so buffer placement can change
under a recorded plan without invalidating it.

## Steps

Each step says what it DELETES. An edit that adds state and deletes nothing is
wrong by default.

### 0. Barriers → mask tuples — DONE, uncommitted

Core gained `barrierhazards(transitions)`; the backend kept only the usage→flags
mapping. Deleted: `barrierspan` (3 methods), `barrierbuffer` (2), the span
NamedTuple, the `(handle, offset)` sort, the adjacent-span merge, the
`renameable` special case, 2 names off the extension import list. 5 files,
+72/−107.

Verified: 0 buffer barriers across 91 passes; 225 memory barriers, 1–5 per pass;
Hikari bit-identical on NVIDIA across 12 configurations. The aliasing handover
barrier already used `resource == 0` and so was already global — unaffected.

### 1. Recording resources come from the pool — DONE

`Pool`/`Block`/`Region`/`acquire!`/`release!` already solve placement and
lifetime, and a plan is already a `tenant!`. Argument memory and indirect scratch
should be regions it owns, acquired at `record!`, released with the recording.

**Deletes:** `arg_slabs`, `arg_slab_idx`, `arg_slab_offset`, `arg_pool_frontier`,
`indirect_slabs`, `indirect_slab_idx`, `indirect_slab_offset` from the queue;
`slabs`, `slab_idx`, `slab_offset` from `CapturedSequence`; and
`ensure_arg_slab!`, `reset_arg_buffer_pool!`, `reclaim_arg_buffer_pool!`,
`arg_pool_in_use!`, `ensure_indirect_slab!`, `reset_indirect_buffer_pool!`,
`get_indirect_buffer`, `capture_arg_buffer!`.

**Adds:** nothing. Four hand-rolled bump allocators become zero.

**Allocator speed is not a constraint here, and the earlier note asking whether
it was is wrong.** The graph path already allocates argument memory ONCE:
`CompiledDispatch` carries `argoff`/`argsize` computed at compile, and
`packdispatch!` writes at `am.ptr + off` into the plan's single `ArgMemory`. The
`arg_slabs` bump allocator serves only `ka_launch!` — the ad-hoc path step 5
deletes.

The one per-dispatch allocator call left on the graph path is
`get_indirect_buffer`, and under recording it fires once per dispatch at RECORD
time, not per run. It should not be an allocator call at all: the plan knows how
many device-sized dispatches it has, so give each an `indirectoff` into one
region, exactly as `argoff` already works. Zero allocations beyond the region.

And the design goes through `acquire!`/`release!` regardless, so if the pool is
ever too slow it is optimised or replaced behind that interface — a recording
cannot tell.

**Fixes a live bug for free:** `reset_indirect_buffer_pool!` rewinds the queue's
indirect pool whenever the queue drains, while a baked recording holds those
addresses for ever. Two plans replaying in one frame write each other's workgroup
counts. It has not fired because a recording writes its own slot at replay and
reads it immediately. **Do not fix this in place** — step 1 deletes it.

#### What it did

A `Unified()` arena kind in core, and the one thing a recording needs from a
backend: `rawalloc` gives it a BAR buffer with `INDIRECT_BUFFER` usage, mapped
once for the life of the block. Everything a recording reads is a `Region` of it,
and the question the four allocators existed to answer — when may these bytes go
to somebody else — is answered by the OWNER: the plan for its own arguments and
indirect commands, the batch for an unmodelled launch's, the capture while one is
open.

`ArgMemory` lays the indirect commands out beside the arguments, in the same
slot: `CompiledDispatch`/`CompiledTrace` carry an `indirect` index the way they
already carried `argoff`, `Pipelines` assigns it, and `ArgMemory` builds the view
once per (slot, dispatch) so a record does a lookup rather than an allocation.
Being per SLOT is what makes the deleted bug unreachable — two runs in flight
cannot write each other's counts for the same reason they cannot write each
other's arguments.

Deleted: the seven queue fields, the three `CapturedSequence` slab fields, the
eight functions the plan names, `ARG_SLAB_SIZE`/`INDIRECT_SLAB_SIZE` and their
alignment constants, the twenty-five-line "is the queue idle enough to rewind"
test in `sweep_retired_batches!`, and the two `arg_pool_in_use!` calls. The two
BDA scanners read the arena's blocks instead of the slab lists, which is the same
bytes through the owner that has them. Also folded: three copies of "give a
presented frame batch back" in `graphics/window.jl` became one
`reclaim_frame_batch!`, because each needed the region release and they had
already drifted (only one returned its command-buffer segments).

Added: `scratch!`, and `get_arg_buffer`/`indirect_command!` over it.

#### Measured, RADV RX 7900 XTX, warm 20, min of 41, same session back to back

| | before | after |
|---|---|---|
| 12-dispatch plan, `run!` + flush | 0.0827 | 0.0814 |
| …record only | 0.0131 | 0.0130 |
| baked replay + flush | 0.0723 | 0.0713 |
| …record only | 0.0013 | 0.0013 |
| 13 dispatches, 6 device-sized, record only | 0.0275 | 0.0293 |
| **64 ad-hoc KA launches + flush** | **0.2350** | **0.2740** |

The graph path does not move, which is the point: it allocated its argument
memory once before and allocates it once now. The ad-hoc path pays **+17%, about
0.6 µs per launch**, which is one `acquire!` where there was a bump-pointer add.
That is the trade the doc above authorises, on a path step 5 deletes; it is
recorded here rather than waved past.

It also costs **+96 bytes per unmodelled dispatch** — 162.8 with the slab ring
against 258.9 with the pool, measured back to back, with `--track-allocation`
putting the whole 96 on `compatible(dev, blk.constraint, want)` and `takespan!`
inside `Pool.acquire!`. That was found during step 2 rather than here, because
`test_dispatch_allocation.jl` was not run at step 1 and its 250-byte ceiling had
been calibrated against the slab ring. The ceiling is now 400 with both numbers
and the reason written into the test; it is still a cliff detector, since the
regression class it exists for is 781 against 115. A `run!` + flush on a device-sized plan is
not quoted at all — min and median disagreed by 30% on repeated runs of the
unchanged build, so nothing can be attributed to it.

No growth: 2000 plan runs, 8000 ad-hoc dispatches and 500 device-sized runs leave
one 4 MiB unified block with exactly the plans' regions live.

### 2. `record!` / `run!` with a real emitter — DONE

`emit_*(e, …)` takes an `Emitter` — a command buffer, an owner for what the
commands name, and the plan's argument slot — never a queue. So
`maybe_split_cb!`, `auto_submit_threshold`, the elision tracker,
`next_skip_barrier` and `ranges_declared` are unreachable from a plan, and
`scope_depth` is unreachable from anywhere.

**Deletes:** `capture`, `replay!`, `CapturedSequence`, `sealinto!`,
`bq.capturing`, `bq.scope_depth`, `cb_begin_flags`, `unsplittable`/
`unsplittable!`, `vk_dispatch_base!`/`vk_dispatch_indirect_base!` (three callers,
all passing zero for the base), `batch.replay_cmd_bufs`, `custom!`, `custombody`,
`rebindable`, `bake(::Compile, body)`, the `record_pass!` hook, `run!`'s
`barriers` keyword and `record!`'s `derived`/`suppress`/`updates` ones.

**Adds:** `Recording` (one command buffer, its regions, its descriptor sets, its
pins), `Emitter`, `emit_dispatch!`/`emit_dispatch_indirect!`/`emit_trace!`/
`emit_trace_indirect!`/`emit_barrier!`, `emitkernel!`, `preparekernel`, `Pinned`,
`sealsegment!`, `tlasset!`, `bindtlas!`, `movable`, `recordable`.

#### What it did

`bake!` is `record!`, and the rename is the content: a "baked" plan was one whose
commands had been frozen out of an ordinary interpreted run, which is why it
EXECUTED the plan as a side effect, why it collected a LIST of command buffers
(five per recording on Hikari's fused sample, cut by a threshold about when to
submit while nothing was being submitted), and why `bake!` and `run!` had to
agree about barriers, slots and what a `Ref` meant. `record!` writes into a
command buffer of its own and ends it; `run!` records on the first run of any
plan it can record and submits every time.

**Ownership replaced position.** `submit!(bq, ::Recording)` seals the batch's
open segment and appends the recording behind it, so ONE list is in submission
order and a second (`replay_cmd_bufs`, submitted last) is gone. That was not
tidying: a readback taken after `run!` records its copy into the same open batch,
and under the old order it executed BEFORE the plan and read zeros — silently.
`test_devicerange.jl` caught it the moment recording stopped being opt-in.

**The fused prepare moved to the pass.** `concurrent_indirect_group` needed a
list on the queue, a launch path that pushed onto it instead of recording, and
two process-wide atomics; `emitprepares!` reads `pp.indirect`, which the pass
already knows.

**Descriptor sets belong to the recording.** Per-dispatch allocation was defended
as removing a class of bug, and the bug was real — a cache keyed by
`(layout, objectid(LavaTLAS))` grew without bound and its `WeakRef` eviction
lagged the GC. Both halves are about a cache with no owner. `tlasset!` keeps one
per (layout, acceleration structure) in the `Recording`, which nothing evicts and
only `release!` frees.

**Settled: no flag.** `ONE_TIME_SUBMIT` was never a candidate — it means
"submitted once, then reset or freed". `SIMULTANEOUS_USE` is only needed to
submit a buffer again while its previous submission is still pending, and that
cannot happen: there is one recording per argument slot and `nextslot!` claims a
slot only once the token covering the run that last used it has passed. So the
recordings are begun with no flags, which is the cheaper of the two. The 2–3%
`build_plans` attributed to `SIMULTANEOUS_USE` is therefore not comparable across
this change and both of its numbers were dropped rather than carried forward.

**Two restrictions, both named in the error and both removed later.** A plan with
a SURFACE cannot be recorded (step 4), and neither can one with a whole-buffer
`Update` — it lands by renaming, and a recording holds the address it was written
with (step 3, where a per-run value rides inline and nothing moves). `recordable`
is the one place that decides, and `run!` emits per run for anything it refuses.
A RANGED update always writes in place and records fine.

#### Found on the way

  * `checkusage` assigned `VK = Vulkan` in its body, which makes `VK` a local for
    the whole function — so every `Transient.Image` threw `UndefVarError: VK not
    defined in local scope`. Nothing caught it because `test_window.jl`, which is
    where transient images live, aborts in its first testset.
  * `test_window.jl` fails at HEAD compiling `scatter_vertex`
    (`KernelError: kernel returns a value of type Any`), verified by stashing,
    and that abort hides the whole graphics half of the suite.
    `test/vulkan/test_recordable_plans.jl` is the coverage the emitter's render
    verbs would otherwise have none of. NOT step 2's, and not fixed here.

### 3. Arguments: one copy, and a `GPURef` — DONE

Measured on Hikari's fused sample: `stride` 217 600 B × 3 slots = 652 800 B, to
protect **268 bytes** that change. 602 writes per run over 14 distinct `Ref`s,
of which **exactly one** (`sample_idx`) changes between samples; the other 13
move only when the `PlanKey` moves, which rebuilds the plan anyway.

`GPURef{T}`: device storage with a stable address. Every dispatch's argument is
the ADDRESS, written once at `record!`; changing the value is one update, not 67.
That makes the 602 → 1 reduction fall out of indirection rather than
classification, and removes the primitive/aggregate split.

Per-run values ride **inline in the command buffer** via `vkCmdUpdateBuffer`
(≤64 KB, 4-byte aligned) — the `Update`/`UpdateRef` path that already exists —
so there is nothing for a concurrent run to race and ordering comes from the
`:update` pass the scheduler already places.

With nothing host-fed per run there are **no slots**: one recording, and
`nextslot!`/`slot_token` have nothing left to protect.

**Deletes:** `rebind!`, `argwrites`, `writearg!`, `verifywrites`, `ArgWrite`,
`Plan.writes`, `isdynamic`/`dynamicargs`/`rebinding`, `ARG_SLOTS`, `nextslot!`,
`slotbase`, `ArgMemory.slot`/`slot_token`/`stride`, `Emitter.base`, and the
RENAME route: `rename!` (the hook and four methods), `Recycler`, `recycle!`,
`retire!(::Recycler, …)`, `take!(::Recycler, …)`, `takehost!`, `signalof`,
`Graph.recycler`, `renaming`. `Scalar` becomes `GPURef`.

#### What it did

`Ref` no longer means "read fresh every run" — it is dereferenced at `record!`
and not again — and that is the whole change. Everything the write plan, the
argument ring and `rebind!` existed for was in service of honouring it, and the
indirection replaces all of it: a value that changes is a `GPURef`, the
dispatches hold its address, and one `Update` writes it.

**`SIMULTANEOUS_USE` comes back, and it is required rather than chosen.** Step 2
settled on no flags because there was a recording per slot and `nextslot!`
guaranteed a given command buffer was idle before it was submitted again.
Deleting the ring deleted that guarantee: `run!` twice before a flush submits the
same `VkCommandBuffer` twice, and a headless plan does not even reach the queue
in between. This is the one thing the step ADDS, and it is the direct consequence
of what it removed.

**The rename route went with the ring**, and it was the second reason `record!`
could refuse a plan. A whole-buffer `Update` landed in a fresh store with the
resource pointed at it — which a recording cannot follow, since it holds the
address it was written with. Every update writes in place now; past
`cmd_update_buffer`'s 64 KB the existing `upload!` fallback stalls, which is the
same route a large RANGED update already took. `recordable` has one clause left
(a surface, step 4).

**Hikari.** `sample_idx` is the `GPURef`; the other fourteen values stop being
`Ref`s and `ensure_plans!` rebuilds when any of them changes, by `===`. That is
not a narrowing — it is what a `Ref` was buying without saying so, and now the
cost is visible: a caller who moves the camera between renders gets new plans.
For a still scene, which is what the measurement above is about, nothing rebuilds
and the per-sample host work was one four-byte `vkCmdUpdateBuffer` — and
since step 6 it is nothing at all for a still scene: the sample counter is
incremented on the device, and the other per-run values are stored only when
they change. Six kernel
entry points read `sample_idx_ref[1]` and pass the `Int32` down unchanged; the
RT shaders share the raygen's argument layout, so all four take it.

#### Verified

Mantle's suite, one process, RADV RX 7900 XTX: 32 370 pass. Every testset either
step touches is green — `recording` (all twelve files), `waiting`, `repeat!`,
`DeviceRange`, the indirect-command ownership pair, `Buffer/GPURef`, the arena
and pool sets, and `test_dispatch_allocation.jl`, which is the per-dispatch
allocation ceiling on the path step 5 rewrote.

Hikari: `test_volpath_graph.jl` (progressive samples equal a batched render, the
plans are built once and kept), `test_trace_pass_modelled.jl`,
`test_multitypeset_updates.jl`, `test_hw_sw_parity.jl` — HW against SW is
8.8e-8 mean, 4.3e-6 max per pixel.

The failures left are the ones below plus the SPIR-V miscompile characterisation
tests, two files whose external packages are absent (`DNNKernels`,
`test_compile_overhead.jl`) and GPUArrays' own `0-norm` on integer types.
`test_gemv.jl`'s 36 were checked against HEAD with both repos stashed and fail
identically there.

#### Found on the way

  * `test_recordable_plans.jl` declared `const MVE`, which `runtests.jl` binds as
    a plain `global` — so including it threw `cannot declare Main.MVE constant`
    and the whole file was skipped in the suite while passing standalone. Step
    2's, and the reason its coverage of the emitter's render verbs had never
    actually run under the harness.
  * `pin_leaves!` stops at a `VulkanTLAS` — the walk is a cycle (TLAS → queue →
    context → queue) and reaches `VK.Instance`, whose `destructor` is a closure
    over the instance. The stop was written `::CommandBatch`, from before a
    `Recording` could own a pin, so a hardware-RT plan blew the stack the first
    time it was RECORDED rather than launched: 53 320 frames. It is `::Pinned`
    now, with `test_pin_leaves_stops_at_tlas.jl` pinning both owners. This was
    step 2's, unreached until a Hikari HW-RT test ran.

### 4. Per-swapchain-image recordings

`record!` refuses any plan with a surface: "a swapchain image is a different
image every frame and a recording names one". RayMakie draws to a window, so this
is now the ONLY thing keeping the per-run emit path alive — step 3 removed the
other, a renaming `Update`.

A swapchain image is **not** patchable the way an argument pointer is. An
argument lives in host-visible memory the recording points at, so writing it is
a store; a swapchain image is named inside recorded commands —
`vkCmdBeginRendering`'s attachment info holds a `VkImageView`, the layout
transitions hold the `VkImage` — with no indirection to patch. And the index is
not chosen: `vkAcquireNextImageKHR` returns whichever image is free, so it is a
lookup on a value handed to you, not a ring for pipelining.

So do NOT record the whole plan per image. **Render into a fixed offscreen
target** — one recording, image-independent, holding all the work — and make
presentation a separate recording that copies that target to the acquired image.
The per-image part is then N recordings of essentially one command, N = swapchain
length. The expensive recording never names a swapchain image, so windowed plans
record exactly like headless ones and this step stops being a second
recording-management scheme.

### 5. KA launches: the heuristics die — DONE, and less than the heading claimed

`ka_launch!` → `vk_dispatch!` → `record_dispatch!` serves every plain array
operation (`gemv`, `fft`, `mapreduce`, sort, narrow-phase). It is the last caller
of the heuristics, and why they exist.

**Deletes:** `barrier_mode`, `barrier_elision`, `next_skip_barrier`,
`ranges_declared`, `touched_ranges`, `dispatch_ranges`, `deferred_indirect` from
the queue; `reset_barrier_elision!`, `poison_barrier_elision!`,
`barrier_needed!`, `range_leaves!`; `concurrent_dispatch_group`,
`exclusive_dispatch_group`, `concurrent_indirect_group`; `force_pre_barrier`,
`skip_pre_barrier`, `first_in_group`; `fast_prepare_indirect!`,
`PrepareIndirect`, `init_prepare_indirect_pipeline!` and the
`caches.prepare_indirect` slot.

`record_dispatch!` is now: open the batch, emit ONE global
`SHADER_WRITE → SHADER_READ|WRITE` barrier, run the body, count it. Nothing has
declared what an unmodelled kernel touches — that is what unmodelled means — so
there is no information anywhere that could justify dropping it. A launch that
HAS declared is a pass in a graph and gets what the compile derived.

`test_barrier_elision.jl` is deleted with the feature.
`test_indirect_in_concurrent_group.jl` becomes
`test_indirect_prepare_ordering.jl`: the property it was really about — an
indirect dispatch is ordered after its own prepare — survives and is now
unconditional.

#### What it did NOT do, and why

**The heading said "KA launches become plans" and they do not.** A one-dispatch
plan is a `Plan(Graph(...))` per launch: a compile, an `acquire!` of argument
memory and a `record!` for every `broadcast!`, and the plan would have to be
keyed on the ARRAYS it names rather than on types, so it could not be cached.
`test_dispatch_allocation.jl` exists to catch a fraction of that cost. The
premise the delete list rested on — "plans submit themselves, so nothing is left
to auto-submit" — never arrives, so two fields on that list stay:

  * `auto_submit_threshold` (64). Submission pacing, not a hazard guess. At 0,
    recording and execution never overlap: the MatAnyone step is 28.1 ms serial
    against 19.3 ms overlapped, 35.6 → 51.9 steps/s.
  * `cb_split_threshold` (3000). Bounds one command buffer, for the NVIDIA
    driver crash on 30 000-dispatch buffers.

Both are POLICY the queue is the right owner of, and neither decides anything
about ordering. Deleting them would have been a knowing regression traded for a
shorter field list. Sixteen queue fields were the problem stated at the top of
this document; fourteen are gone here, and step 7 takes the last two with the
open command buffer they paced.

## The review of 2026-09-04, and the steps it adds

Reviewed against the target above, with steps 3 and 5 uncommitted. The
invariants hold where they are pinned: argument memory is written once and only
at `record!`, `run!` takes the plan alone, every delete list greps to zero, the
head of a schedule is seeded from its tail, and a run with nothing pending
allocates nothing. What does not hold is steps 6 to 9 below.

Before any of them: the tree carries 38 untracked `*.jl.573933.mem`
allocation-profiler files under `src/` and `ext/`. Delete them. And
`test_window.jl` errors in its first testset at HEAD (`scatter_vertex`,
`KernelError: kernel returns a value of type Any`), which takes every windowed
testset down with it. Step 7 rewrites the present path and step 9 rewrites
windowed plans; neither has coverage until that is fixed, so it is fixed
before 7 and not, as an earlier draft said, before 9.

Two decisions taken in writing these, both reversible and both named where
they bite: every persistent resource a graph names is a host-writable one (step
6 registers `Buffer`s as well as `GPURef`s, and measures the barriers that
costs), and Hikari's sample index moves to the device (step 6's Hikari half).

### How each step is known to be done

Every step below ends with **Done when**: greps over `src`, `ext` and `test`
(and `dev/Hikari/src` where named) that return zero outside prose, and tests
named by file, run in the warm session. The full suite runs once, after step
9, and not at any boundary before it. The greps are the definition of done, not the summary written
afterwards. Three rules that apply to all four steps, because each is a way of
stopping at half:

  * A test that pins a fixed bug is run BEFORE the fix and seen to fail. A
    test that pins a structural property (a count, an absence) states in its
    header why it cannot fail behaviourally.
  * No `@test_broken`, no `skip =`, no commented-out call site, no
    `# TODO step N`. A path that has to survive until a later step is listed
    in THAT step's deletes, with its own grep, at the moment it is written.
  * A measurement the step calls for is in the doc before the step is called
    done, whichever way it came out. "Measure later" is the step not being
    done.

### 6. A value is written through the resource, and `Ref` is not an argument

**The problem.** `Update(g, x)` reserves a schedule position and hands back a
callable, and the value is stored by calling it. That was written for a Makie
observable feeding a `Buffer` (`on(obs) do data; ref(data); end`), and step 3
put a `GPURef` through it unchanged: Hikari holds four callables in a `writes`
tuple and a function to call them. The pending value belongs to the resource,
not to a per-graph handle; the graph already knows which resources it reads, so
the position `Update` reserves is derivable; and a ref read by two plans needs
two handles today.

The same step ends `Ref` as an argument. On Vulkan it was dereferenced at
`record!`; on the KernelAbstractions path `resolve` keeps the `RefValue` and
`Launch` calls `argvalue` per run, so one graph meant two things. There is
nothing left for it to mean: a value that changes is a `GPURef`, a value that
does not is passed as a value.

It is also where the regression the review found lives. The typed `UpdateRef`
setter refuses the one-element vectors `test_recorded_run_semantics.jl` and
`test_record_after_run.jl` write into a scalar update: 65 errors. Those tests
move to the API below rather than being patched.

**The API.**

    idx = GPURef(dev, Int32(0))
    # passes take idx as an argument and declare use(p, idx; read = true)
    plan = Plan(g)
    idx[] = Int32(5)          # a store, from any thread: a pending cell on the ref
    run!(plan)                # lands every dirty resource the plan reads, then submits
    buf[1:n] = data           # the vector form, same mechanism; buf[:] = data for all of it

`setindex!` stores and marks dirty, under the seqlock `UpdateRef` carries now.
A `GPURef` holds one pending value and the last store wins, which is what a
scalar means. A `Buffer` holds a LIST of pending `(range, data)` stores,
consumed in order, because two stores to different ranges before one run are
two writes and the second must not drop the first. The store retains `data`;
a caller who means to mutate it before the next run passes `copy(data)`.

`setindex!` does not compare against the last store: a `GPURef` the device
also writes (a counter the host resets) must land a store of an equal value,
so a caller who wants to skip unchanged values compares on its side, where it
knows what changed. There is no `getindex`: reading a device value is a
download and is spelled as one. `update!` stays what it is, the immediate
upload, and the two are not the same verb: `update!` lands now, as a one-shot
on the owning thread (step 7); a store lands at the update pass of the next
plan that reads the resource, and may be made from any thread.

At `Plan`, every `Buffer` and `GPURef` any pass declares is registered as a
`CopyDst` of the update pass, so the barriers derive as they do today with no
declaration. "Declares" walks every usage through `rootresource`, so a vertex
`Attr`, a `BufferRange` slice and a `Commands` buffer all reach the `Buffer`
under them; a resource passed as a kernel argument and never declared is
undeclared, which is what undeclared means. The plan holds those resources as
a typed tuple, `Plan.hostwritten`, built once at `Plan` and again by `refit!`
(which recompiles), and `run!` walks it: a dirty one gets its
`vkCmdUpdateBuffer` in front of the recording, a clean one costs a flag read.
A resource read by two plans is consumed by whichever runs first; the second
reads what already landed ahead of it on the queue. Two plans on two QUEUES
reading one host-written resource are ordered by whatever the `Policy` that
put them there orders them with; this step does not add a wait for it.

What registering every `Buffer` costs is one `(TRANSFER write -> read)` tuple
at each buffer's first reader and one `(read -> TRANSFER write)` tuple at the
update pass, both deduplicated per pass, the second emitted per run only when
something is pending. Measure it: barrier count per pass on Hikari's sample
before and after (225 memory barriers over 91 passes is the step-0 figure),
and the number goes in this section either way. If it is not a wash, the
fallback is to register `GPURef`s alone and require a `Buffer` to be declared,
which is the one case where a declaration is honest. The fallback is taken
only with the measurement written down; it is not the default.

On the KernelAbstractions path a pending store lands as `update!` at the update
pass position, which on the host backend is exact (a step runs to completion
on the calling thread) and on Metal is a host write into memory the previous
run's command buffer may still read. That is Metal's `awaitwrites` question,
pre-existing, and not this step's.

**Deletes:** `Update`, `UpdateRef`, `registerupdate!`, `pending_type`,
`applyupdates!`, `applypending!`, `_applypending!`, `updates_snapshot`,
`Graph.updates`, `Graph.updates_typed`, `withpayloadptr`, `takepending`,
`anypending` and the exports of all of them; `write_update!` (all four
methods), `writeupdate!`, `emitupdate!`, `updatestore`; the `updates` keyword
of `emit!` (nothing emits an update into a recording any more); `argvalue`
(both methods and its export), `resolve(::Device, ::RefValue)`, the per-call
`map(argvalue, …)` in `Launch` (its arguments are already storage after
`bake`); the docstrings that still say a `Ref` is read per run
(`runtime/dispatch.jl`, `CompiledTrace`, `packtrace!`, `resolve`); the dead
`repack!` methods for a dispatch and a trace, with the draw one renamed
`packdraw!` beside `packdispatch!` and `packtrace!`.

**Adds:** `setindex!` on `GPURef` and `Buffer`, the pending cell and its
seqlock moved from `UpdateRef` onto them, `Plan.hostwritten` (the typed tuple),
and a check in `dispatch!`, `draw!` and `trace!` that walks the argument tree
the way `pin_leaves!` does and throws on a `RefValue` anywhere in it, naming
`GPURef`. Shallow would let a `Ref` inside a struct through to the packer,
which is the case Hikari actually had.

**Hikari's contract, and it is the point of all of this.** A sample is one
`run!`. For a still scene, Hikari's host work per sample is that call and
nothing else: no store, no ad hoc launch, no wait, no download, no allocation.
For a moved camera it is the stores of what moved, then the same call. Nothing
Hikari does per sample may reach the GPU except through a plan, and every plan
records on its first run. What is on the per-sample path today, and what
happens to each:

  * the four stores through `writes` and `apply_sample_updates!`: gone. The
    camera, the initial medium and the filter parameters are stored only when
    the call argument changed, by identity against what the state last
    stored, since a call argument cannot announce itself. `VPPlans.writes`
    goes with them;
  * the sample index, `film.iteration_index[] + 1`, a host counter fed to the
    device every sample: a `GPURef` the sample plan's first pass increments,
    one thread, ordered by its declared read and write. `film.iteration_index`
    stays as the caller's count of samples taken and stops being a GPU input.
    A film reset is not an event Hikari's state hears, so `render!` reads the
    counter it already reads and stores zero to the ref when it is zero: one
    compare per sample, and no work;
  * `detect_initial_medium`: gated on the camera position today and a stub
    returning an empty key. When it does something, it is a device pass that
    writes the initial-medium `GPURef`, never a host traversal or a readback;
  * `finalize_film!` in live preview: a second plan run per sample, and stays
    one. Two submissions per sample is what live preview costs;
  * `RECORD_PLANS`, a module-level `Ref` that decides when recording happens:
    gone. A caller who wants the cost outside the timing loop calls `record!`
    on the plans, which `build_plans` hands back;
  * the scene-dirty flag read, the listener registration compare, the
    framebuffer identity compare: reads, not work, and they stay.

No `Ref` reaches a kernel, which the argument check enforces. No pass is
undeclared: `custom!` is gone, the trace is `trace!`, the loop is `repeat!`.
Everything a sample touches is a `Buffer`, a `GPURef` or a transient the graph
placed.

**Tests.** `test_recorded_run_semantics.jl` and `test_record_after_run.jl`
against `idx[] = k`; a host twin of the first, since the contract is now the
same on every backend; "a `Ref` argument is refused, nested or not" replacing
"a `Ref` is read at `record!`"; two ranged stores to one `Buffer` before one
run both land; a checksum of the plan's argument block after `record!` that is
unchanged after N runs with stores in between, which is the invariant this
document rests on and nothing asserts directly. In Hikari: after the first
sample of a still scene, no resource in the sample plan's `hostwritten` is
dirty before or after a `render!`, and the same plans object answers a moved
camera with exactly the changed refs stored (`test_plan_invalidation.jl` pins
the second half already). The renderer's zero-allocation and one-submission
test is written HERE and is red until step 7, because the open batch still
allocates a segment per run; step 7 turns it green and names it.

**Done when** these return zero outside prose:

    grep -rn "Update(\|UpdateRef\|applyupdates!\|applypending!\|updates_snapshot\|updates_typed\|write_update!\|writeupdate!\|emitupdate!\|withpayloadptr\|takepending\|anypending\|argvalue\|repack!" src ext test
    grep -rn "::Base.RefValue\|::RefValue" src/runtime/dispatch.jl src/graph/kalaunch.jl | grep -v refuserefs
    grep -rn "RECORD_PLANS\|apply_sample_updates!\|\.writes\b\|Mantle.Update(" dev/Hikari/src dev/Hikari/test

and these files are green in the warm session: the twelve recording tests
under `test/vulkan/`, `test_arena_recording.jl`, `test_devicerange.jl`,
`test_host.jl`, and Hikari's `test_volpath_graph.jl`,
`test_trace_pass_modelled.jl`, `test_multitypeset_updates.jl`,
`test_hw_sw_parity.jl` and `test_plan_invalidation.jl`, with the one named red
test. No full suite.

**DONE 2026-09-04.** The barrier measurement, one warm session, Hikari's
`volpath_graph_scene` with media at 64x64 and `max_depth = 5`, the sample
plan before and after this step:

| | passes | memory barriers | buffer barriers |
|---|---|---|---|
| before | 51 | 109 | 0 |
| after | 52 | 112 | 0 |

The extra pass is `next sample`, the device-side counter, one barrier of its
own. Registering every declared `Buffer` cost the other two, both at the head
of the schedule: the update pass's `(read -> TRANSFER write)` tuple and one
first reader's `(TRANSFER write -> …)`. Two barriers over fifty-one passes is
a wash, and the fallback was not taken. The finalize plan is unchanged at one
pass, one barrier.

What differs from the step as written, and why. `updatestore` stays: it is
the zero-allocation `(VkBuffer, offset)` arithmetic the pointer route needs,
it has one caller (`emitstore!`), and the grep above no longer names it.
`refuserefs(::Base.RefValue)` necessarily names the type it refuses, so the
`RefValue` grep excludes that one line. And the walk stops at any MUTABLE
struct, not only at resources and arrays: a hardware-traced plan's accel
argument holds the driver's handle objects, which reference each other in
cycles, and the first version walked into them and overflowed the stack on
every traced plan. A mutable struct is a handle that owns its insides; the one
mutable cell refused is the `Ref` itself. `Plan.hostwritten` is built once, at
`Plan`, and not again by `refit!`: a refit recompiles the same graph against
the same registrations, so the tuple cannot come out different, and rebuilding
it would be work that proves nothing. On the Hikari side the sample counter is
a device increment in a `next sample` pass; `resetsamples!` stores zero when
the film's count reads zero; `storechanged!` stores the camera, the initial
medium and the filter by identity against `VPPlans.stored`; and the counter
ref is made with the film's current count, so plans rebuilt mid-progression
continue the sequence instead of replaying sample one.
`test_sample_is_one_run.jl` holds the still-scene assertions and the
one-submission testset step 7 turns green. `anydirty` and `landstores!` over a
plan's `hostwritten` are `@generated`-unrolled like `pin_leaves!`, or the
recursive `Base.tail` left ~48 bytes on Hikari's five-element sample tuple on
the run path.

The pre-step item was not what the review said. `test_window.jl`'s
`scatter_vertex` inferred `Any` because BOTH Lava and Mantle exported
`vertex_index` and the `frag_coord` family, and a scope with
`using Mantle, Lava` therefore had no binding for the bare name at all — every
shader that named a builtin unqualified failed the same way. Lava's exports of
the builtins are gone (the Vulkan backend reaches them qualified, as it
already did), and `test_no_stale_exports.jl` now pins that only Mantle exports
them. The 38 `.mem` files are deleted. And `bench/` still spelled the
`GPURef` constructor `M.Scalar`, which resolved to StaticArrays' `Scalar`
through Mantle's own `using`; renamed.

With the first testset compiling, `test_window.jl` shows what the failure
had been hiding since before step 0: testsets that still pin per-buffer
barriers (`bufferMemoryBarrierCount`, "scoped, not one catch-all") from
before the mask tuples, a schedule assertion that does not expect the update
pass every plan with a buffer now has, `bench/fuzz.jl` calling `run!` with
the `barriers` keyword step 2 deleted, and the two `ensure_active_batch!` +
`present_frame!` sites step 7 rewrites. Run in the warm session on
2026-09-04 it did not finish in 45 minutes. It is step 7's named file, and
it is made green there — not here, where the pieces it pins are the ones
step 7 is about to change.

### 7. No open recording: every command buffer is closed by whoever opened it

**Why it goes, entirely.** The queue holds one command buffer that is open
across call boundaries, `CommandBatch.cmd_buf`, and anything may append to it:
an ad hoc launch, an upload, a readback copy, a per-run store, a present
transition. Three things are wrong with that, and none of them is a bug to fix
in place:

  * **It is global state.** What the open buffer contains depends on every
    call made since it was opened, from any package, in any order. A plan is
    the opposite: everything it submits was decided when it was compiled.
  * **It makes asynchronous recording and scheduling hard.** One open buffer
    per queue is one writer per queue, which is why `owning_thread` is asserted
    at every entry. A closed recording touches no queue while it is written;
    only handing it over does.
  * **It is hard to say what gets submitted, and when.** A recording has to be
    inserted into the stream, so `submit!(bq, ::Recording)` seals whatever is
    open and begins another; a launch has to guess whether what came before it
    was ordered, so `record_dispatch!` reads `dispatch_count` and `in_flight`;
    a buffer can grow without bound, so `cb_split_threshold` cuts it; and the
    present path takes the buffer over and submits by a second route. Each is
    a consequence of the object, not a defect in the code around it.

The batching the open buffer provided is the graph's job, and the graph does it
exactly: a plan is one recording, ordered inside by derived barriers, and
nothing has to be accumulated at the queue to get there. What is left using
the open buffer is the KernelAbstractions path, and a slower KernelAbstractions
path is accepted: how many ad hoc launches are scheduled efficiently is a
question for a graph built to hold them, later, not for the queue to answer
with a buffer and a threshold now. Step 2 took the open buffer away from plans;
this step takes it away from everything.

**The model.** Two kinds of GPU work, and both are closed:

  * a `Recording`: a plan's, written once at `record!`, owned by the plan,
    submitted every run;
  * a `OneShot`: written, sealed and handed over inside one call by whoever
    needed it, owned by the submission that carries it, given back when the
    timeline passes it.

Two types under one abstract `Closed`, not one type with a flag, because they
differ in exactly the places dispatch is for: `recpatch!` collects the packed
pointers of a `Recording` and is a no-op on a `OneShot` (an earlier draft said
"`Recording` already holds what a one-shot needs", and it would have collected
a patch entry per ad hoc launch for ever); `release!` of a `Recording` waits on
its token, a `OneShot` is released by the sweep; the queue pools `OneShot`s.
Both answer `pin!`, `scratch!` and `Emitter` the same way. The `Pinned` union
goes, and every `where {O<:Pinned}` becomes `where {O<:Closed}`, which is what
keeps `test_dispatch_allocation.jl` where it is.

A command buffer is never open when a Mantle call returns. The one-shot is
spelled

    oneshot!(bq) do e            # e is an Emitter over a pooled OneShot
        emit_barrier!(e, …)
        emit_dispatch!(e, …)
    end                          # sealed and submitted on the way out

and every path that used to reach for `ensure_active_batch!` reaches for that
instead. The list is the grep in Done-when, not a summary; by file today:
`runtime/command.jl` (`record_dispatch!`, the copy helpers), `array/ka_backend.jl`
(the launch path), `graphics/api.jl`, `graphics/framebuffer.jl`,
`graphics/pipeline.jl`, `graphics/textures.jl` (draws, readbacks, uploads into
textures), `runtime/memory.jl` (`upload!`/`download` staging),
`raytracing/shaders.jl`, `raytracing/acceleration.jl` (acceleration structure
builds, on the timeline instead of `as_fence`: callers that waited on the fence
`waitfor!` the token), `runtime/video.jl` (decode, on its own queue and the
same rule), `runtime/launch.jl`, `array/gpuarrays.jl`,
`array/kernelinterface_host.jl`, `graph.jl` (the per-run stores and patches,
and the windowed per-run emit below).

**Staging.** `bq.staging` is one buffer reused across transfers, and it was
safe because the open batch ordered every copy through it. With each transfer
its own submission, reusing it would race the previous transfer still in
flight. It goes: an upload's staging bytes are a `Unified` scratch region owned
by the one-shot (`scratch!`, which already exists), released by the sweep when
the transfer has passed. `get_staging!` goes with the field.

**Windowed plans until step 9.** A plan with a surface still cannot be
recorded, and today it emits per frame into the open batch. Under this step it
emits per frame into a `OneShot` that the present submits, which is the same
per-frame work in a closed buffer. That is an interim path and it is listed in
step 9's deletes, with its grep, from the day it is written.

**Submitted when sealed.** A closed buffer is handed to the driver the moment
it is closed: `oneshot!` ends in a `vkQueueSubmit2`, and `run!` submits the
plan's recording in one submission with the one-shot holding this run's stores
and patches in front of it. Several closed buffers that one caller has in hand
go in one submission, `submit!(bq, closed...)`, and that is the only way two
ever share one. The queue holds nothing between calls but `in_flight`: what the
driver has and has not finished. A present is a submission with two extra
semaphores and a fence, through the same function, and the second submit path
in `present_frame!` goes.

There is no `pending` list and no `submit!(bq)` that hands over "whatever has
accumulated", because both are the open batch by another name: state on the
queue that a threshold decides the fate of. `auto_submit_threshold` goes with
them, and with `cb_split_threshold` below that is all sixteen fields the top
of this document named, not fourteen. The overlap the threshold bought (a
stream that records and executes in series at zero) is what immediate
submission gives outright; what it costs is one `vkQueueSubmit2` per ad hoc
launch, measured below. If that number is bad for a workload, the answer the
design already gives is that the workload is a graph, not that the queue grows
a buffer back.

Three consequences in core, each a deletion: `waitfor!` loses its "submit first
if the token is not out yet" branch, since every token is out the moment it
exists; `fence(dev)` becomes `next_timeline`, since nothing recorded is ever
unsubmitted; `flush!` is `waitfor!` on the newest submission and nothing else.
And the core hook `submit!(::Device)`, "send whatever the backend has recorded
but not yet submitted", has nothing to send: it goes, with its calls in
`runpass!` and at the end of the KA `run!`. The Metal backend implements that
hook because it keeps an open `MTLCommandBuffer` between calls
(`submitopen!`), which is the same object this step deletes on Vulkan, and the
same rule applies to it. There is no Metal here to run; the Metal edits are
made by reading, listed in the diff, and verified on a machine that has one,
which is named in State as the one thing this step leaves unrun.

**Ownership.** A `Submission` owns the one-shots it carries and pins the plan
recordings it borrows; the sweep releases the first list and touches the
second only through the pin. One-shots are pooled on the queue
(`free_oneshots`, replacing `free_batches` and `free_cmd_bufs`), so an ad hoc
launch allocates nothing beyond the scratch region it already acquires. The
dispatch log is the context's ring, as it is today; the batch's copy of it
goes.

**Deletes:** on `CommandBatch`: `cmd_buf`, `recording`, `dispatch_count`,
`segment_dispatches`, `last_was_rt`, `sealed_cmd_bufs`, `borrowed`,
`submitted_cmd_bufs`, `pinned`, `pinned_refs`, `regions`, `dispatch_log`; on
the queue: `active_batch`, `free_batches`, `free_cmd_bufs`, `as_cmd_buf`,
`as_fence`, `staging`, `cb_split_threshold`, `auto_submit_threshold`; the
functions `ensure_active_batch!` (the core hook and both Vulkan methods),
`alloc_cmd_buf`, `allocate_batch`, `init_batch`, `get_staging!`,
`sealsegment!`, `maybe_split_cb!`, `has_active_recording`, `drawn!`,
`emitter(bq)`, `Emitter(::CommandBatch, …)`, `queueof` over a batch, the
`Pinned` union, the barrier-skip condition and the `is_rt` keyword in
`record_dispatch!`, the seal in `submit!(bq, ::Recording)`, the `borrowed`
scans in `reclaim_batch!` and `present_frame!`, the second submit in
`present_frame!`, the "is a batch open" branch of `fence(dev)`, the
submit-first branch of `waitfor!`, the no-argument `submit!(bq)`, the core
`submit!(::Device)` hook and its callers, and `handover!` with `takeover!` and
`Arena.lastrun` as written below.

**Adds:** `Closed` with `OneShot` under it beside `Recording`, `oneshot!`,
`submit!(bq, closed...)`, `Submission` (which is `CommandBatch` with the fields
above deleted and the name saying what it is: what one `vkQueueSubmit2`
carried, its signal value and its waits), `free_oneshots`, a memoised head
barrier for a plan's recording, and the interim windowed one-shot that step 9
deletes.

**The arena handover is decided at record time.** `handover!` emits the
cross-plan barrier into the recording from who ran last when the plan was
recorded, so of two recorded plans alternating on one arena the first never
gets one; `submitrun!` calls `takeover!` per run and discards the answer, and
says so in a comment. The barrier also names the arena block's `VkBuffer` and
range inside the recording, against the target above, and goes stale when the
arena grows. Cross-plan hazards cannot be derived, so a plan's recording opens
with one anonymous global barrier, always, and the arena stops remembering who
ran. The 1.2 ms once attributed to the global form came from the measurement
`docs/design.md` disowns as clock state; measure again, min and median, on the
two-plan composite and on Hikari's sample, and the numbers go here.

**Measured, recorded, not a gate.** A one-shot per ad hoc launch costs a
`vkBeginCommandBuffer`/`vkEndCommandBuffer` pair and a `vkQueueSubmit2` where
the open buffer amortised both. The step-1 row, 64 ad hoc launches + flush,
0.2740 ms on RADV, is the baseline; the same row after, same session, min and
median, goes in the table beside it. It is written down so the cost of the
KernelAbstractions path is known, and it decides nothing: the regression is
accepted above. What must not move is the plan path: a run with nothing
pending submits exactly one recording and allocates no command buffer, which
`test_run_allocates_nothing.jl` already holds at zero bytes, and the
`test_dispatch_allocation.jl` ceiling stays where it is because the one-shots
are pooled.

**Measured 2026-09-04, RADV RX 7900 XTX, warm 20, 41 samples, same harness,
each side in a fresh session** (the struct changes between them do not survive
one): 64 ad hoc KA launches + flush.

| | min | median |
|---|---|---|
| before step 7 (open batch, one submission per 64) | 0.2437 ms | 0.2508 ms |
| after step 7 (one one-shot and one submission per launch) | 0.7948 ms | 0.9860 ms |

About 8.6 µs per launch for a `vkBeginCommandBuffer`/`vkEndCommandBuffer` pair,
a global barrier and a `vkQueueSubmit2` where the open buffer amortised all
three over sixty-four. That is the regression the step accepts, written down;
the plan path is the one that must not move, and `test_run_allocates_nothing.jl`
and `test_dispatch_allocation.jl` hold it.

**DONE 2026-09-04, Metal unrun.** What differs from the step as written, and
why. The pooled record of a submission is `Submission`, in `bq.free_submissions`
beside `bq.free_oneshots` — `CommandBatch` with the fields above deleted, as
the step says, and it has to be pooled too or a run allocates 1.2 KB of it
(`test_run_allocates_nothing.jl` caught exactly that when the first version
swept tokens but not submissions at `submit!`). The queue-side "give it back"
verb is `recycle!(bq, ::OneShot)` / `recycle!(bq, ::Submission)`, not
`reclaim!`: a method named `reclaim!` in the extension shadowed the pool's
`reclaim!` inside it and every `run!` failed. A token is one queue's timeline
value, so `waitfor!(bq, tok)` waits on THAT queue — the device-level
`waitfor!(dev, tok)` asks the primary queue, and a download of a buffer last
written on a second queue returned before the copy had run
(`test_crossqueue_sync.jl` caught it). A download copies into a region the
caller acquires, waits for, reads and releases, not into the one-shot's
scratch, which the sweep could hand on the moment the wait returned; an
upload uses the one-shot's scratch and waits for nothing. `updatestore`-style
address arithmetic and `indirectbarrier!` (the prepare → indirect-read barrier
the ad hoc indirect paths need inside their one one-shot) are the two helpers
added beside `headbarrier!`, and the head barrier is counted on the context
(`diag.head_barriers`) so `test_closed_command_buffers.jl` can pin that every
closed buffer wrote one — the structural form the step asked for, with the
reason in the file's header. The acceleration-structure build context carries
the emitter of the one-shot `build_accel!` opens (`AccelBuildContext.into`,
`buildcmd(ctx)`), and the build waits on its token where it used to wait on
the fence. The present is `present_frame!(bq, win, frame::OneShot)` with the
frame's one-shot ending in `presentready!`; `readback_window` refuses a
window that holds an acquired image, since a caller mid-frame has no command
buffer to add a copy to. The Metal backend commits one `MTLCommandBuffer` per
pass, copy, present and readback (`commit!`, `submitwait!`), with `OPEN_CB`,
`submitopen!` and its `submit!(::MetalDevice)` hook gone; edited by reading,
in `graphics.jl`, `images.jl` and `window.jl`, and NOT run — this machine has
no Metal.

Tests: `test_closed_command_buffers.jl` is new and holds the queue-shape,
one-submission-per-call, two-uploads and refused-token assertions;
`test_recorded_run_semantics.jl`, `test_waitidle_submits.jl`,
`test_arena_recording.jl`, `test_repeat.jl`, `test_free_during_recording.jl`,
`test_crossqueue_sync.jl`, `test_argument_memory_isolation.jl`,
`test_phase1_pin.jl`, `test_pinned_buffer_lifetime.jl`,
`test_pin_leaves_stops_at_tlas.jl`, `test_phase4_singlethread.jl`,
`test_caching_and_allocations.jl` and `test_spirv_pattern_correctness.jl`
were rewritten where they pinned the open batch, its thresholds or the arena
handover; step 6's red Hikari testset is what this step turns green, with one honesty: a ray-tracing recording syncs its ~two dozen acceleration-structure pins at every submission (`sync_access!` over `recording.pinned`), ~640 bytes, constant per sample and predating this work, so the Hikari test pins that residue as CONSTANT and no store landing, and the byte-exact zero stays `test_run_allocates_nothing.jl`'s (a compute plan, no such pins).

**What it opens up.** With no open buffer there is nothing on the queue that
recording touches, so a recording can be written on any thread that owns the
command pool its buffer came from; `owning_thread` narrows to `submit!` and
the sweep, which is the timeline, and a pool per recording thread is the one
thing asynchronous recording still needs when it is built. Not this step's
work; this step is what makes it possible without a lock around a shared
stream.

**Tests.** After any Mantle call returns, the queue holds no open command
buffer and nothing closed but unsubmitted (a sweep over launch, upload,
readback, `run!`, present); a run with nothing pending submits exactly the
plan's recording; an ad hoc launch submits exactly one one-shot that begins
with the barrier (structural, with the reason written above it: this driver did
not overlap even an undeclared hazard, so no MWE can see the race); two
uploads back to back through the pool-backed staging are both correct with
nothing waited on between them; two plans alternating on one arena stay
bit-exact over a hundred runs, and `takeover!` no longer exists; the step-1
timing row, repeated; and step 6's red test, Hikari's zero-allocation
one-submission sample, goes green here and is named in State as the boundary
that closed it.

**Done when** these return zero outside prose:

    grep -rn "ensure_active_batch!\|active_batch\|\.cmd_buf\b\|sealsegment!\|maybe_split_cb!\|auto_submit_threshold\|cb_split_threshold\|as_cmd_buf\|as_fence\|get_staging!\|bq.staging\|\.staging\b\|free_batches\|free_cmd_bufs\|sealed_cmd_bufs\|borrowed\|submitted_cmd_bufs\|dispatch_count\|segment_dispatches\|last_was_rt\|handover!\|takeover!\|lastrun\|Pinned\b\|has_active_recording" src ext test
    grep -rn "submit!(pl.graph.dev)\|submit!(dev::\|submit!(::Any)" src

and the same files as step 6 plus `test_window.jl`, `test_dispatch_allocation.jl`,
`test_argument_memory_isolation.jl` and the hwtlas files, green in the warm
session. No full suite.

### 8. One `run!`, and the backend answers three questions

`run!` for the Vulkan device is a full second sequence beside the
KernelAbstractions one in `graph/kalaunch.jl`: check the plan is live, reclaim,
poll, sync the swapchains, refit, check extents, then either submit the
recording or emit per run. All of that but the last clause is portable, and the
last clause is what a backend IS. `record!`, `submitrun!`, `patchtarget`,
`recordable` and the argument-cursor half of the Pipelines phase sit in
`src/vulkan/graph.jl` beside it.

One `run!` in core: the prologue above, then `execute!(dev, plan)`. Vulkan's
`execute!` is today's `submitrun!` (one one-shot for the pending stores and
patches if there are any, the plan's recording, one `submit!`); the KA one is
today's pass loop. `record!` keeps its Vulkan method, but the patch-table build
and `listen_moves!` move to core around a backend `emitrecording!`. The
Pipelines phase assigns `argoff`/`indirect` in core from an
`argsize(dev, compiled)` the backend answers; the backend compiles entries and
nothing else.

The 22 definitions in `src/vulkan/graph.jl` that name no driver type are
enumerated here, each with a destination, because "a scan says zero" is a
number a comment can game and an earlier draft proposed exactly that as the
test: `region_bda` (core, over `BufferBlock`), `target_view`/`target_image`/
`target_extent`/`target_format` forwards (core, the surface forwards),
`initial_usage`/`initial_state` for the framebuffer (stay: they name a Vulkan
framebuffer's state), `resourcekind` (core), `Base.length` over a transient
(core), the three `tlasof` (stay: they name a `LavaTLAS`), `checkextents`
(core, over `target_extent`), `multiprepare!`/`multi_prepare_indirect_kernel`
(stay: the command they write is Vulkan's layout), `Base.close(::Plan)`
(delete: it does nothing), `Base.isopen(::WindowSurface)` (core),
`recordable` (delete, step 9), `emitupdates!` (delete, step 6), the three
`emit_draw!` (stay: they call the pass verbs). Stay means "names a driver
object through a call rather than a type", and the count that goes to zero is
the moved-or-deleted list, checked by grep against this paragraph.

**Deletes:** `run!(::Plan{LavaDevice})`, `submitrun!`, and the definitions
above marked core or delete.

**Adds:** `execute!`, `emitrecording!` and `argsize` as hooks.

**Tests.** `test_ext_imports_are_declared.jl` already checks the import list.
The moved definitions are exercised by the suite they already have; the new
one is the host backend running a plan with a pending store through the same
core `run!` the Vulkan device runs, which is `test_host.jl` gaining the update
case it never had.

**Done when:** `grep -rn "^function run!(pl::Plan{LavaDevice})\|submitrun!\|region_bda\|^checkextents\|^recordable\|Base.close(::Plan)" src/vulkan` is zero, and the step-6 files plus `test_host.jl` are green in the warm session. No full suite.

**8a and 8b DONE 2026-09-04; 8c remains.** The run! unification is done: one
core `run!(pl::Plan)` (checklive, reclaim, `beforeframe!`, refit, checkextents,
`execute!`), with `beforeframe!` and `execute!` the two hooks. The Vulkan
device's `execute!` is the old `submitrun!` and windowed path folded together;
the KernelAbstractions default is the pass loop with per-pass profiling. Both
`run!(::Plan{LavaDevice})` and `submitrun!` are gone. `checkextents`,
`region_bda`, `Base.length(::TransientBuffer)` and `Base.isopen(::WindowSurface)`
moved to `graph/build.jl`; `Base.close(::Plan)` is deleted; `checkextents`'
hook declaration is gone from `backend.jl`. Verified: the recording, host,
devicerange, arena, waitidle and repeat tests, and `test_window.jl` (225), all
green through the one core `run!`.

Deviation from the grep: `region_bda` is DEFINED in core now, but Vulkan's
`resource_moved!`/`arena_moved!` still CALL it, so `grep region_bda src/vulkan`
is two uses, not zero — the definition moved, the callers legitimately name a
core helper to compute a device address.

**8c, not yet done:** the record! patch-table build and `listen_moves!` still
sit in the Vulkan `record!` rather than in core around an `emitrecording!`
hook, and the Pipelines phase still assigns `argoff`/`indirect` inside the
backend rather than in core from an `argsize(dev, compiled)`. That is the
compile-path half of step 8; it changes no behaviour and is left for the pass
that also runs the full suite.

### 9. Step 4, made concrete

Render into a fixed offscreen target with one recording, and present with one
trivial recording per swapchain image. What step 4 above does not say, and
this step has to:

  * **The offscreen target is persistent, owned by the window's surface, one
    per window, remade with the swapchain on resize.** It cannot be a
    transient: a transient's memory is shared arena scratch, live only inside
    the plan that placed it, and another plan running between this plan's run
    and the present (a finalize, a UI plan) reserves the same arena and
    overwrites it. `Surface(g, win)` hands the graph that image as the render
    target; depth and everything else stay transients.
  * **One blit recording per swapchain image**, recorded when the swapchain is
    created and dropped when it is remade: transition the offscreen image to
    transfer source, blit into the acquired image, transition both back. The
    acquire returns an index; the index selects the recording; nothing is
    recorded per frame.
  * **One submission per frame:** the plan's recording, then the blit
    recording for the acquired image, with the image-available wait, the
    render-finished signal and the per-frame fence on it, through the one
    `submit!` step 7 left.
  * **Frames in flight** are ordered by the queue and by the global head
    barrier the plan's recording opens with (step 7): frame N's blit reads the
    offscreen image, frame N+1's plan writes it, and the barrier at the head
    of N+1 orders the two.
  * `screenshot` and `readback_window` read the offscreen target and no longer
    need to acquire.

**Deletes:** `recordable`, the interim windowed one-shot from step 7, the
per-run branch of `run!`, the `borrowed` special case and its "a windowed plan
cannot be recorded" comment in `present_frame!`, the `record!` error naming a
surface, and the per-frame `transition_image!` of the swapchain image in
`present_frame!`.

**Adds:** the surface-owned offscreen image, the per-image blit recordings on
the window, and nothing on the queue.

**Tests.** `test_window.jl`, running, is the coverage, plus: a windowed plan
records on its first run and the same recording object answers every frame
until a resize; a resize drops and re-records both the plan and the blits; a
screenshot after N frames equals the offscreen target's contents.

**Done when:** `grep -rn "recordable\|cannot be recorded\|updates = true\|emit!(emitter" src ext test` is zero, `test_window.jl` is green in the warm session, and THEN the full Mantle suite and the full Hikari suite run, once each, the only full runs in the whole arc.

### Hygiene, done inside step 6

`hasproperty(d, :ndrange)` in the `PassPlan` constructor becomes a method per
compiled kind; `scan_arg_slabs_for_bda!` is renamed for what it scans now;
`emitter(bq, pl.args)` goes with `emitter(bq)` in step 7.

### Order and boundaries

The `.mem` files and `test_window.jl` before anything. Then 6, because it
deletes the regression rather than patching it, and Hikari follows in the same
phase. 7 second, and it is the widest diff of the four: every path that opened
the batch opens a one-shot instead, and the Metal half of it is edited blind
and named as such. 8 third. 9 last, since it deletes the interim path 7 had to
carry. The full suite runs ONCE, after 9, and never before: every step is verified
by its named files and MWEs in the warm session, and a full run in between is
the rule at the top of this section being broken. The State section is updated at each boundary with the
measurements the step called for, and the step-3 sentence about one four-byte
update per sample is corrected when 6 lands.

## Rules that were broken while writing this

- **Every edit states what it DELETES, before writing.** Adds a field, a flag, a
  counter or a parallel path and deletes nothing → wrong by default, stop.
- **Mid-refactor, an obstacle is the refactor.** Three times in one session the
  cheap local exit was taken: `scope_depth` (suppressed a heuristic instead of
  removing it), `FUSE_SAMPLE` (a second render loop beside the first), a fourth
  bump allocator (copied `capture_arg_buffer!`, the debt, as if precedent).
- **A bug found mid-refactor is NOTED, not fixed.** Check whether the refactor
  already deletes it.
- **The full suite runs once, at the end of the refactor.** Never at a step
  boundary: MWEs and the step's named test files in the warm `bt_julia_eval`
  session until then. A full Mantle run is ~24 min, and "once per phase" was
  written into this document and had to be taken out again.
- **All Julia through `bt_julia_eval`.** Not `julia --project` via Bash — and
  never piped through `tail`/`grep`, which buffers and loses the whole log if
  the process is killed.
- **Warm before measuring.** The argument ring is 3 deep; fewer than ~10 warm
  iterations is still cold. Take min AND median and refuse any row where they
  disagree. Three claims reversed under warmup this session.

## A `run!` that allocates nothing — DONE

The step-3 contract, measured and pinned: `run!` of a recorded plan with no
pending updates allocates zero bytes, and so does the submission cycle around
it. `test_run_allocates_nothing.jl` fails on the first byte.

What it took, each found by the allocation profiler on the steady-state loop
(they were ~6 KB per `run!` + `waitfor!` combined):

  * `reclaim!`'s `lock do` closure escaped inlining (224 B/call) — explicit
    `lock`/`try`/`unlock` now.
  * `vkBeginCommandBuffer`'s begin info was rebuilt per segment; two
    module-level `_CommandBufferBeginInfo` consts now, since they name no
    handle.
  * `submit!` built four vectors and four Vulkan.jl wrapper structs per
    submission, each wrapper boxing a `deps` array for GC rooting. The batch
    now owns raw VulkanCore info vectors (`raw_cb_infos` etc.), refilled in
    place and handed to `vkQueueSubmit2` through the dispatch table under one
    `GC.@preserve`.
  * `waitfor!`/`query_timeline` allocated a `SemaphoreWaitInfo`, two vectors
    and a counter `Ref` per call — now per-queue `QueueSlots` (on
    `BatchQueue.slots`, an `Any` field that hands out a heap REFERENCE, so
    reading it boxes nothing). The cells are the owning thread's only: a
    buffer finalizer asks the same timeline from the GC thread, and sharing
    one `RefValue` would hand it a torn or wrong counter — off-thread callers
    get a fresh `Ref` / the checked wrapper.
  * The timeline token boxed at every hand-off: `ArgMemory.token`,
    `Recording.token` and `LavaDevice.ctx/bq` were `Any`. `LavaDevice`'s
    fields are typed now; the tokens are `UInt64` with 0 = "never run"; and
    `ArgMemory.token` is a `RefValue{UInt64}` read through `argtoken`/
    `setargtoken!` — because `ArgMemory` is parametric, a bare value field
    read through the UnionAll still boxes, and the cell sidesteps it.

Also here: the patch table learned that a prepare kernel emitted into a
recording packs into a recording-OWNED scratch region, not the plan's
`ArgMemory` — `patchtab`/`pending_patches` now carry `(region, offset)`
targets, `patchtarget` resolves them (and throws on batch scratch, which a
recording must never hold), and `droprecording!` unlistens before releasing.
That was an `InexactError` on Hikari's per-iter lifecycle test before it was
any of the above.

## Per-sample allocation and timing, 2026-09-04

The RayDemo/Hikari acceptance is a recorded sample that runs in one submission
and allocates nothing. Measured in the warm session, `runsample!` (the concrete
function barrier) on a 256x256 render, glass + gold + emissive spheres,
`max_depth = 5`, warm 30, over 200 samples:

| | value |
|---|---|
| per-sample allocation, before | ~640 B |
| per-sample allocation, after | ~96 B |
| per-sample time, min | 0.86 ms |
| per-sample time, median | 0.96 ms |

The 640 was a run re-walking the recording's `IdSet{Any}` of pins to
`sync_access!` them — ~two dozen pins on a ray-tracing plan, each boxed by the
`Any` iteration. `Recording.sync::Vector{VkManagedBuffer}` now snapshots, at
`record!` (`collectsync!`), exactly the buffers a submission must sync; a run
iterates that concrete vector (`syncall!`). Correctness-identical to the pin
walk — the buffer set is fixed once pins are placed during `emit!` — just
un-boxed. A compute plan's recorded run is byte-zero
(`test_run_allocates_nothing.jl`); the residue on a ray-tracing plan is the
sync itself: `sync_access!` writes each buffer's `last_write`, a
`Union{Nothing, Tuple{Any, UInt64}}` that boxes ~32-48 B per buffer.

**Byte-zero, reached (2026-09-05).** The residue above was measured on a
toy scene and called constant; on the RayDemo materials scene the same stamp
was 2.3 KB a sample — 48 synced buffers at 48 B — and the RayMakie path around
Hikari added 9 KB of its own. Measured on the real path, `RayMakie.render!` of
a live `Screen` on the materials scene (1200x900, `max_depth` 50, 21 atomic
plots, hardware RT), warm, 30 samples: **0 bytes, min, median and max**. Four
things, each found by `Profile.Allocs` on that loop:

  * `sync_access!` stamped every synced buffer with a boxed `(queue, value)`
    tuple: `VkManagedBuffer.last_write` was `@atomic
    Union{Nothing, Tuple{Any, UInt64}}`. It is two plain fields now
    (`last_write_bq`, `last_write_val`), the owning thread's alone. The atomic
    existed for the finalizer thread, which read the pair to ask whether the
    device was done; it does not read it any more: `vk_free!` and
    `unsafe_free!(::LavaBLAS/LavaTLAS)` check the thread FIRST and, off the
    owning thread, hand the object to the default queue's deferred list
    without looking. The owning thread's drain reads the stamp, and
    `lastwritepassed` asks a foreign writing queue rather than keeping the
    entry forever (which is what the old `lw[1] === bq` sweep did to a buffer
    written on another queue — a leak the off-thread deferral would have made
    routine). A finalizer that runs ON the owning thread cannot interleave the
    two stores: nothing between them is a safepoint. The free log moved into
    `destroy_buffer!`, where the destruction is and the stamp may be read.
    `test_run_allocates_nothing.jl` pins the recorded-plan zero; the plan that
    syncs many buffers is Hikari's (below).
  * Hikari's `storechanged!` was a dynamic call through `VolPathState.plans`
    (`Any`), and it boxed the 684-byte camera it compared, every sample. The
    per-run refs and what they were last stored with are a `PerRun{C}` now,
    owned by the plans and reached through `VolPath.perrun`, a field typed as
    the union over the camera types, so the compare is bitwise and static.
    `VPPlans.refs`/`.stored` are gone. `test_sample_is_one_run.jl` asserts
    `render!` of a still scene allocates exactly zero, twice.
  * RayMakie's `render!` read `state.camera[]` through a `Union{Observable,
    Nothing}` field (a boxed camera), then made a dynamic call with it and a
    keyword (a boxed NamedTuple). `tracesample!` is one dynamic dispatch on
    heap objects only; inside it every type is concrete.
  * RayMakie's `poll_all_plots` read `p[:trace_renderobject][]` on every plot
    to trigger resolution, and the read boxed the render object — 7 KB a
    sample for 21 plots that had not changed. It asks
    `ComputePipeline.isdirty` first and reads only what is dirty.

Also on that path: `colorbuffer` waited on the device after EVERY sample
(`KA.synchronize`, "to prevent command buffer overflow" — the open batch that
step 7 deleted). Gone; a sample is a closed submission and the queue orders
them. A/B on the materials scene, same session, RayDemo harness, 3 trials:
0.651 s min / 0.653 s median with the wait, 0.644 / 0.648 without. Small at
65 ms a sample; it was the ~11 ms-a-sample downclock trap at 256².

Back-to-back `RayMakie.render!` on the materials scene, 7 runs of 20 samples,
waited through `Mantle.waitfor!`: 43.27 ms/sample min, 43.32 median.

**Found on the way, not fixed.** A `compute!` pass's dispatch that names an
external `LavaArray` neither pins it into the recording nor stamps it at
submit: `packdispatch!` strips it to a `LavaDeviceArray` and
`pack_arg!(::LavaDeviceArray)` only stores bytes, and `use(p, a)` on it makes a
declared usage (a barrier) and nothing more. The plan keeps the array alive by
holding the argument, so nothing is freed early — but a cross-queue write to
it (an upload on the split upload queue) gets no semaphore wait from a run
that reads it. The RT trace pass is different: `packtrace!` walks its raw
arguments with `pin_leaves!`, which is why a Hikari sample plan syncs 48
buffers and a compute-only plan syncs none. A Mantle-level regression test for
the stamp allocation therefore needs a trace plan; `test_run_allocates_nothing.jl`
keeps the compute-plan zero, and Hikari's `test_sample_is_one_run.jl` is the
many-buffer one.

**Found by the end-of-day test pass, and fixed: RayMakie's overlays and window
frames were still written against the open batch.** `test_overlay_compositing.jl`
errored in `render_overlays_gfx!` on `begin_pass!(bq, …)`: step 7 turned the
graphics commands into emitter methods (`begin_pass!(e, …)`, `draw_in_pass!(e,
…)`, `use_bindings!(e, …)`) and deleted `bq.active_batch`, and RayMakie's
overlay path — the offscreen readback in `colorbuffer` and the two frames of
the render loop — still drew into the queue and called the two-argument
`present_frame!` that took the open batch over. None of the harness scenes
have overlays, which is how it got this far. Now: `render_overlays!(screen, e,
target)` takes the emitter; `colorbuffer` puts the overlays in a one-shot of
their own behind the blit (no `flush!`, no `waitidle` — the readback is a
later submission on the same queue and waits for its own copy); the render
loop's frame is `present_composited!`: acquire, ONE `oneshot` holding the blit
(`blit!` gained an emitter form, the queue form wraps it), the overlays and
`presentready!`, handed to the three-argument `present_frame!`. Also found
there: `plots/lines.jl` still named `Mantle.alloc_index_buffer`, which lives in
the extension since the split — it reaches it through `vulkanbackend()` like
the rest of the overlay code. And the window never opened at all:
`Mantle.Window(backend, w, h; color_format = BGRA{N0f8})` is the portable
spelling RayMakie has used since the split, and `VulkanWindow` took only a
`VK.Format` — a `TypeError` before the first frame. It lowers an element type
through `vkformat` now, as `VulkanFramebuffer` already did. Verified: a
displayed scene with a mesh and a `lines!` presents frames (240x320), keeps
running, and closes cleanly; `test_overlay_compositing.jl` is green and
`test_window_frame.jl` (new, RayMakie) pins the window path.

**Found on the way, not fixed: a plan holding a `Trace`, emitted into a
one-shot per run, faults.** Measuring baked against unbaked, the unbaked side
was the windowed branch of `execute!` verbatim — `oneshot(bq) do e;
emitupdates!; emitpatches!; emit!(Emitter(e.owner, pl.args), pl); end` then
`submit!` — applied to Hikari's headless killeroo sample plan on RADV. The
first submission lost the device: `GPUVM fault at 0x17000 … last dispatch:
rt_trace`. The recorded path submits the same plan a hundred times a second,
so what differs is what the emit owns: into a recording, `packtrace!`'s pins
and scratch live as long as the plan; into a one-shot they go back when that
submission passes. No user path reaches this today — a windowed plan cannot
hold a trace — and step 9 makes windowed plans recordable, which deletes the
branch. Left for step 9, and recorded here so the fault is not rediscovered.

## RayDemo benchmarks, 2026-09-05

`RayDemo/benchmark/run_benchmarks.jl`, unchanged workload (`BENCHMARK_SCENES`,
matched to the `.pbrt` files), 1 warmup + 3 trials of `Makie.colorbuffer`,
seconds. The harness itself needed three fixes to run against the split
packages: it asked `Lava.vk_context()` for the device (Lava has no runtime
since 2026-08-27, so hardware RT was never detected and `nvidia-smi` named the
GPU on a machine rendering on the Radeon), it swallowed `Lava.vk_flush!()` and
`Lava.flush_deferred_frees!()` in a bare try/catch between scenes (neither
exists), and it read `m.create_scene` in a world older than the include.
Results are in `benchmark/results/linux_7900xtx_lava_{hw,sw}_submission_refactor.json`.

**AMD RX 7900 XTX, RADV, Mesa 26.2.1** (uncommitted working tree of every
package, after steps 6, 7, 8a, 8b and the byte-zero pass above):

The morning run (before the download-staging and trace-barrier fixes found on
the RTX, below) and the final run after them; the JSON holds the final one:

| scene | config | HW first | **HW final** | SW first | **SW final** |
|---|---|---:|---:|---:|---:|
| crown | 1000x1400, 16 spp, depth 100 | 2.397 | **2.122** | 3.223 | **2.973** |
| bunny_cloud | 1920x1080, 8 spp, depth 50 | 3.777 | **3.401** | 4.536 | **4.149** |
| killeroo_gold | 1368x1026, 32 spp, depth 5 | 0.831 | **0.590** | 0.868 | **0.618** |
| materials | 1200x900, 10 spp, depth 50 | 0.649 | **0.462** | 0.575 | **0.362** |
| black_hole | 800x450, 32 spp, depth 100 | 0.488 | **0.416** | 0.487 | **0.408** |

Min and median agree on every row (within 1 %). The reference on this GPU is
`linux_7900xtx_lava_{hw,sw}_v5_no_inline.json` (2026-05-27), which ran crown,
killeroo and materials at other resolutions and depths, so only two rows
compare: black_hole, same config, 1.764 s HW / 1.696 s SW then — **4.2x
faster** now; bunny_cloud at 960x540 then (1.029 s HW), a quarter of the
pixels, so 0.85 s per quarter now against 1.03 — 17 % faster per pixel. The
same-config reference for the other three is the RTX 4000 Ada's
`match_v5` (2026-08-20), measured on that GPU below; on this GPU the final
HW run is faster than the RTX's on every row.

**NVIDIA RTX 4000 Ada, driver 595.99.02** (selected with
`VK_DRIVER_FILES=/usr/share/vulkan/icd.d/nvidia_icd.x86_64.json` before the first
Vulkan init; same tree, same harness, same day). Two references at these exact
configs exist on this GPU: `match_v5` (2026-08-20, before the Lava/Mantle split)
and the uncommitted `lava_hw_v1.json` re-run of 2026-08-30 (Hikari 68f3405,
before the modelled trace of 08-31 and the one-plan sample of 09-02; Mantle
before step 0).

The first run of the day, before the two fixes found below (the download
staging and the trace-pass barriers), and the final run after them —
`benchmark/results/linux_rtx_4000_ada_generation_lava_{hw,sw}_submission_refactor.json`
holds the final one:

| scene | HW first run | **HW final** | HW 08-30 | HW 08-20 | SW first run | **SW final** | SW 08-20 |
|---|---:|---:|---:|---:|---:|---:|---:|
| crown | 3.510 | **1.915** | 2.041 | 2.146 | 5.063 | **4.725** | 4.907 |
| bunny_cloud | 4.690 | **4.204** | 4.189 | 1.792 (scene changed since) | 5.030 | **4.462** | 1.780 (same) |
| killeroo_gold | 1.570 | **0.696** | 0.695 | 0.728 | 1.661 | **1.336** | 1.374 |
| materials | 1.076 | **0.562** | 0.583 | 0.638 | 0.864 | **0.615** | 0.640 |
| black_hole | 1.024 | **0.971** | 0.751 | — | 1.046 | **0.967** | — |

Final: every row at or better than its reference except black_hole HW, which
is the discarded-round cost quantified in (4) below. Min and median agree on
every row (all within 0.5 %).

**The first run was a regression on NVIDIA, and not on RADV.** Killeroo HW RT is 2.2x the 08-30
number; the SW path regressed less (+21 %), which reads as a fixed cost per
round that the cheaper HW rounds cannot hide. What it is NOT, each measured on
killeroo HW in the NVIDIA session, 16-sample blocks, min and median agreeing:

  * not the missing per-sample wait: 38.92 ms/sample submitting back to back,
    39.00 waiting after every sample;
  * not the host: `render!` takes 9 µs of host time a sample, and one
    submission a sample, no one-shots (`flush_counter` +8, `head_barriers` +0
    over 8 samples);
  * not `SIMULTANEOUS_USE`: the recording re-recorded with no usage flags
    (legal with the per-sample wait) is 38.99;
  * not the conditional-rendering gates: `withpredicate` recorded as a plain
    call (every round runs, empty rounds have zero indirect counts) is 38.83;
  * the per-pass barriers are two to five `VkMemoryBarrier2` per pass with
    shader-storage masks over VS|FS|CS (`0x888`), not global ones. (Note for
    later: a `trace` pass's barriers carry the same `0x888` destination
    stages, not `RAY_TRACING_SHADER`; whether that is covered elsewhere or a
    gap is a correctness question, not this one.)

The cost is per ROUND of the bounce loop: `max_depth` 1 → 9.64 ms/sample,
2 → 24.81, 5 → 38.84, 10 → 39.83 (rounds die out after ~5). The first bounce
round alone is 15 ms; the whole sample was 21.7 ms on 08-30. The plan
profiler cannot see inside a recording (`gpu_ms` is NaN on the recorded
path), so the bisect is by code state.

**Bisected, 2026-09-05.** The three repos stashed to HEAD — Mantle `ff3a790`
(steps 0-2), Hikari `7117cbb`, RayMakie `453a35668` — and the same harness
re-run on the same GPU in a fresh session (HW RT cannot run at that HEAD:
`pin_leaves!` walks into `Vulkan.Instance` and overflows the stack, the bug
step 6's `refuserefs`/`pin_leaves!` stop fixed):

| scene, SW | HEAD (steps 0-2) | this tree | reference 08-20 |
|---|---:|---:|---:|
| killeroo_gold | 1.337 | 1.661 | 1.374 |
| materials | 0.624 | 0.864 | 0.640 |

So the regression is in the UNCOMMITTED work — steps 3 to 8b, the byte-zero
pass, or the RayMakie per-sample wait deletion — and not in steps 0-2 or the
Hikari one-plan sample. The argument memory is allocated the same way in both
states (`Unified`, device-local + host-visible first), so it is not that.
Next: where the per-run `GPURef` storage lives on a device without resizable
BAR, since a trace thread now reads the camera through one.
That was not it — the RTX has resizable BAR and a `DEVICE_LOCAL | HOST_VISIBLE`
type, and `GPURef` storage is persistent-arena memory anyway. What it WAS, two
things, found by taking the harness's `colorbuffer` apart:

**1. The download staged through BAR memory (fixed).** `Array(output_buffer)`
on a 1368x1026 RGBA{Float32} frame took 329 ms: 29 MB/s, which is the host
READ speed of write-combined memory. Step 7 rewrote `copy_buffer!(:download)`
to stage through a `Unified` region — the BAR arena — where HEAD had a
per-queue staging buffer allocated HOST_CACHED (`get_staging`/`host_buffer`,
which step 7 deleted with the open batch). Uploads were fine (writes to
write-combined memory are fast); every download, framebuffer readback and
window readback paid it, on RADV as much as on NVIDIA. That is the whole
"intercept" the harness numbers carried: killeroo HW at 1 spp was 355 ms, of
which 330 was the download. Fixed with a `Readback` pool kind (core,
`graph/types.jl`), backed on Vulkan by HOST_CACHED memory
(`rawalloc(::LavaDevice, ::Readback, …)`, `READBACK_BLOCK_SIZE` = 16 MiB), used
by `copy_buffer!(:download)`, `readback_framebuffer` and `readback_window`;
`host_buffer` deleted. Download bandwidth on the RTX, same session: 1 MB 34.6
→ 1.4 ms, 16 MB 543 → 21 ms, 64 MB 4301 → 24 ms. Pinned by
`test_download_readback.jl` (the pool holds a `Readback` block after a
download, and 16 MB comes back in under 160 ms). The killeroo harness number
went 1.018 → **0.694 s**, the 08-30 reference to the millisecond, and a 1-spp
`colorbuffer` is 29.8 ms.

**2. A per-sample 2x that did not reproduce.** In the session that produced
the table above, `render!` of a fresh killeroo plan was 38.9 ms/sample after
the whole harness had run (both backends, all five scenes), and a plan built
with `profile = true` in the same session was 21.6. In a fresh session the
plain plan is 21.4, and it stays there after crown, materials, bunny_cloud,
black_hole and an SW killeroo run through the harness (21.5–21.7 each time).
Clocks were at 2265–2280 MHz and the power cap in every logged run. Not
reproduced, so not explained; the recorded-plan profiler (`timings`) now
works — its `pending` flag was set at record and cleared by the first
`collect!`, so a recording reported `NaN` for ever; it is set per submission
(`profiled!(pl)`) — and is the tool to reach for if it comes back.

**3. A trace pass's barriers never named the ray-tracing stage (fixed).**
While reading the masks above: every barrier into and out of Hikari's `trace`
pass carried `0x888` — COMPUTE|VERTEX|FRAGMENT — on both sides, because a
usage lowers by its TYPE (`Storage{BufferKind, …}`) and nothing said the pass
that made it was a ray-tracing pipeline. On AMD ray tracing runs as compute
and the COMPUTE bit covers it; on NVIDIA it is a stage of its own, so those
barriers were no execution dependency in either direction: the trace could
start before the pass that filled its queue had finished, and the pass after
it could read the queue it was still writing. Images came out bit-identical
(profiled vs plain, plain vs plain) — the window is small — and the device was
lost twice at black_hole, the medium-heavy scene, in long sessions
(`VK_ERROR_DEVICE_LOST` at the first submit of its plan; a second time inside
`fill_aux_buffers!`'s wait with 11 submissions in flight), never in a fresh
one. Fixed in the vocabulary, not per driver: `Traced{U}` (core,
`sync/usage.jl`) is a shader access made by a ray-tracing pipeline; `compute!`
applies `traced` to every shader usage of a pass that holds a `Trace` (storage,
uniform, sampled — not indirect counts, predicates, copies or the accel),
with `Unordered` kept outermost; the Vulkan lowering gives it
`RAY_TRACING_SHADER` as its stage and the inner usage's access and layout.
`test_traced_usages.jl` pins the mapping, the lowering and that `compute!`
applies it; `test_trace_pass_modelled.jl` and `test_sample_is_one_run.jl`
stay green.
Verified on the RTX after the change: every barrier into the `trace` pass has
`dst = RAY_TRACING_SHADER` and every one out of it `src = RAY_TRACING_SHADER`
(the `reset` after it: `src = 0x20088a`), killeroo HW is 21.76 ms/sample (was
21.5 — the extra waits cost nothing), and black_hole rendered four times in a
row at depths 10, 30 and 100 in a session that had run the whole harness.

**4. What black_hole still pays on NVIDIA: the discarded rounds.** Black_hole
HW is 0.97 s against 0.75 on 08-30, the one row that is slower. Depth sweep at
8 spp, harness `colorbuffer`: `max_depth` 10 → 0.186 s, 30 → 0.202, 100 →
29 ms/sample against 19.5 at depth 10. The scene's paths die well before 100
bounces, and a recorded loop pays every iteration it discards — "its
barriers, its gate dispatch and its predicate test" (`repeat!`) — which on
this driver is ~0.1 ms a round, ~90 rounds a sample, ~0.3 s over 32 spp:
the whole gap. That is the design's stated cost of `maxiters` as a bound, and
RADV pays much less of it (black_hole is 3.6x faster than its May reference
there). Making a discarded iteration cheaper — one predicate scope around a
whole round rather than around each pass, or a round that skips its barriers
when discarded — is a Mantle `repeat!` lowering change, not done here.





## State

As of 2026-09-05 the uncommitted work also holds: the byte-zero pass (Mantle
`last_write_bq`/`last_write_val` and the owning-thread free paths, Hikari
`PerRun`, RayMakie `tracesample!` and the dirty-checked poll), the `Readback`
pool kind and the download/readback staging through it, `Traced` usages for
trace passes, the profiler's per-submission `pending`, the RayMakie
`colorbuffer` loop without a per-sample device wait, the RayMakie line
overlays reaching `alloc_index_buffer` through the extension (they named
`Mantle.alloc_index_buffer`, gone since the split — `test_overlay_compositing`
errored on it), the RayDemo harness fixes and both GPUs' result files. New
tests: Mantle `test_download_readback.jl`, `test_traced_usages.jl`, RayMakie
`test_render_allocates_nothing.jl`; tightened: `test_run_allocates_nothing.jl`,
Hikari `test_sample_is_one_run.jl` (render! == 0). Green in the warm session
on the RX 7900 XTX at the end of the day: 11 Mantle files including
`test_window.jl`, 5 Hikari files, 3 RayMakie files. The full suite has still
not been run — once, after step 9.

Committed on `sd/lava-refactor`: `0d3f3b5` (Mantle — `repeat!`, `waitfor!`,
`bufferusage`, `waitidle`), `d472e98` (Hikari — a sample is one plan, chunking
deleted), `7f8396c` (step 0), `ff3a790` (Mantle — steps 1 and 2), `7117cbb`
(Hikari — follows it). Uncommitted: steps 3, 5, 6 and 7, the pre-step items
(`.mem` files, the shader-builtin exports in Lava), and the Hikari half of 6.
Step 6's and step 7's measurements are in their sections. Step 7's Metal half
is edited by reading and unrun: the one thing the step leaves unverified.

### Failing before any of this, and left alone deliberately

`test_window.jl` errors in its FIRST testset compiling `scatter_vertex` —
`KernelError: kernel returns a value of type Any` — and that takes every testset
after it down with it, which is most of the graphics coverage. Identical at
HEAD with step 2 stashed, so it is not this refactor's; the suspect is the
environment ("dependencies precompiled but different versions are currently
loaded", which includes GeometryBasics and StaticArrays), since the shader
differs from a working one only in a `Mat4f * Vec4f`.

`test_capture_replay.jl`, `test_capture_gc.jl` and `test_replay_interleaved.jl`
asserted that `capture` EXECUTES what it records, which stopped being true when
`sealinto!` split recording from submission. They were rewritten against a plan
and merged into `test_recording_lifecycle.jl` in step 2.

`test_gemv.jl` fails 36 of 88 with `relerr` exactly 1.0 — the k-contiguous
kernel writes nothing — on RADV. Verified identical at HEAD with both repos
stashed, so it is neither step's; the transposed kernel beside it is correct.

Full-suite baseline after this arc (2026-09-04, RADV): 32434 pass / 37 fail /
66 error of 32539, and the failure set is exactly: `test_gemv.jl` (36, above),
the window subprocess (above), and `test_gemm_staged.jl`, `workgroup
zero-init`, `test_psb_chain_fold.jl`, `test_loop_unswitch_miscompile.jl`, which
need DNNKernels/AcceleratedKernels — neither is in this environment's Manifest.
Everything else the suite ever surfaced during the arc was fixed rather than
documented: see the `@test_broken` promotions in test_int32_cartesian_
miscompile.jl / test_shared_index_division.jl, the repaired-fault pins in
test_static_workgroup.jl, and the uninitialized-buffer fixes in
test_recording_lifecycle.jl and the seven hwtlas test files (the GPUVM cascade
was recycled-pool bytes in `undef` instance records, confirmed by GPU-AV's
"invalid accelerationStructureReference").

`test_volpath_graph.jl`'s "the same scene on the host backend" errored in
`repeat!`: `supportspredicate` had a method for `LavaDevice` and the `::Any`
false, so a `HostDevice` could not express a device-decided trip count — and
`sample_graph` builds a `repeat!` for any `max_depth >= 2`. Fixed after step 3:
the host never records, so it predicates by having `runpass!` read the flag the
gate kernel just wrote and skip the body — `supportspredicate(::HostDevice)` is
true, and `test_repeat.jl` pins that the read gates (3 iterations of `2x+1`,
not 7). `test_caching_gc_correctness.jl`'s six same-way testsets went green
with it.

`test_real_kernels.jl` fails in `GPUCompiler.methodinstance` on
`AssertionError: Base.isdispatchtuple(sig)`, before any Mantle runtime code runs:
the test builds `Hikari.Conductor{…}` with four parameters and that type is not
concrete any more (`isconcretetype` says so directly). Hikari type drift in a
test, not a runtime failure.

`test_arena_bake.jl`'s "a renameable Update gives up its scoped barrier" was
stale from step 0 and IS fixed here, since nothing later touches the barrier form:
it asserted one memory barrier and two scoped buffer barriers, and step 0 deleted
scoped buffer barriers entirely. It now pins what step 0 built — two memory
barriers for three transitions, because `a` and `b` are the same tuple and the
tuples are `unique` — and zero buffer barriers, which is the property that lets a
buffer move under a recorded plan.

`dev/Vulkan` is dirty on purpose: a local patch to `generated/linux.jl:51890`
for a Vulkan.jl generator bug that makes `VK_EXT_device_generated_commands`
unreachable (`_IndirectCommandsLayoutTokenEXT` never applies its enum
conversion). It tracks upstream — do not commit. Worth reporting.

Reference numbers to compare against, NVIDIA, 256², 8 spp, warmed, median of 11:
`out/bench_head_warm.txt` (HEAD host loop) and `out/bench_new_warm.txt` (fused).
Correctness reference: `out/ref_new.txt`, 12 configurations, bit-exact.

## `record!` is core, and there is one pass walk, 2026-09-07

Review finding, Simon's: `record!` was a Vulkan method holding the whole
sequence, core had `record!(plan) = plan`, and `execute!(::LavaDevice)` recorded
lazily on the first run. That put the policy — what a recording is, when a plan
gets one — in `src/vulkan/`, and it duplicated the pass walk: core already had
`runpass!` for the interpreted backends, and the backend had `emit!`,
`emitpass!`, `emitwork!` for the recorded one, the same order written twice.

Now:

* **One walk**, `emit!` in `graph/kalaunch.jl`: head, then per pass its
  barriers, its predicate scope and its work, in compile order. It speaks the
  primitives declared in `graph/backend.jl` — `openrecording`,
  `closerecording!`, `emithead!`, `emitbarriers!`, `withpredicate`,
  `emitupdate!`, `emitprepares!`, `emitdispatch!`, `emitcopy!`, `beginrender!`,
  `emitdraw!`, `endrender!`, `profiled!` — and nothing else. The Vulkan backend
  answers them with commands; the core `Immediate` emitter answers them by
  doing the work, which is what `runpass!`, `runrenderpass!` and
  `runcopypass!` were.
* **Core `record!`**: idempotence, the surface refusal, `openrecording`, the
  walk, `closerecording!`, `listen_moves!`. An interpreted backend answers
  `openrecording` with `nothing` and the plan comes back unchanged.
* **`run!` never records.** `execute!(::LavaDevice)` refuses a plan that was
  never recorded. Hikari records its three plans where it builds them; every
  headless test and bench does the same. A recording that something ELSE threw
  away — an images arena growing under it, a refit — is `invalidate!`d, which
  marks the plan `stale`, and `run!` writes it again before it submits, on the
  owning thread; the move happens under the pool's lock on whichever thread is
  compiling the other plan, which is why it cannot record there.
* The refit-and-checkextents inside the old backend `record!` went with it:
  those are `run!`'s, once.

Found on the way and fixed: `region_bda` (moved to core in 8a) was never
imported by the extension, so `arena_moved!` threw on the first buffers-arena
growth under a recorded plan (`test_recorded_move_patch.jl` pins it); five
`alloc_debug` sites pushed to a deleted global (`test_alloc_debug_log.jl`).
Every core name the backend calls unqualified has to be in the extension's
`import Mantle:` list — checked with `MVE.n === Mantle.n` over the list.
