# One way to submit

Working plan for removing the heuristic recording path. Written 2026-09-02, part
way through. Read `docs/design.md` first for what the graph is.

## The problem, stated once

**Sixteen fields on `BatchQueue` exist because something records GPU work without
a plan.**

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

### 3. Arguments: one copy, and a `GPURef`

Measured on Hikari's fused sample: `stride` 217 600 B × 3 slots = 652 800 B, to
protect **268 bytes** that change. 602 writes per run over 14 distinct `Ref`s,
of which **exactly one** (`sample_idx`) changes between samples; the other 13
move only when the `PlanKey` moves, which rebuilds the plan anyway.

Mantle already has the distinction — a `Ref` argument means "read fresh every
run", a plain value means "resolve once" — and Hikari over-uses `Ref`.

`GPURef{T}`: device storage with a stable address, assignment queues rather than
uploads. Because the address is stable, every dispatch's argument is a *pointer*
written once at `record!`; changing the value is one update, not 67. That makes
the 602 → 1 reduction fall out of indirection rather than classification, and
removes the primitive/aggregate split (today aggregates go through an inline
area and get a pointer; primitives are packed by value).

Per-run values ride **inline in the command buffer** via `vkCmdUpdateBuffer`
(≤64 KB, 4-byte aligned) — the `Update`/`UpdateRef` path that already exists for
`Buffer` contents — so there is nothing for a concurrent run to race and
ordering comes from the `:update` pass the scheduler already places.

With `sample_idx` device-derived there is nothing host-fed per run at all, so
there are **no slots**: one recording, and `nextslot!`/`slot_token`/`passed`/
`waitfor` have nothing left to protect. Anything a future caller genuinely feeds
per run gets two small copies — 268 bytes and an index flip, not a ring.

**Deletes:** `rebind!`, `argwrites`, `verifywrites`, `ArgWrite`, the write plan,
`ARG_SLOTS` (the depth is `length(slot_token)`, already data), and most of
`nextslot!`. A plan with no per-run values needs **one** slot and no wait. It
also deletes the RENAME route through `write_update!`, and with it the second
reason `recordable` can answer false: a per-run value that rides inline in the
command buffer moves no store, so nothing a recording names can move under it.

`Scalar` is `GPURef` plus a `stride`-0 draw-binding rule; it has one real caller
in three packages. Fold it in, keeping `Buffer{T}`/`Scalar{T}` erasing to the
same `Attr{T}` — that is load-bearing for pipeline identity.

**Hikari side:** `sample_idx` is `iteration_index + 1`, a counter the host
happens to own. Derive it on the device and the write plan is *empty* — same
move `repeat!` makes for the loop counter.

### 4. Per-swapchain-image recordings

`record!` refuses any plan with a surface: "a swapchain image is a different
image every frame and a recording names one". RayMakie draws to a window, so this
is one of the two things keeping the per-run emit path alive (the other is a
renaming `Update`, which step 3 removes).

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

### 5. KA launches become plans — the ad-hoc path dies

`ka_launch!` → `vk_dispatch!` → `record_dispatch!` serves every plain array
operation (`gemv`, `fft`, `mapreduce`, sort, narrow-phase). It is the last
caller of the heuristics, and why they exist.

A single dispatch is a one-pass graph; the machinery exists. **Deletes:** the remaining queue fields, `record_dispatch!`'s decision-making,
`barrier_mode`, `barrier_elision`, `next_skip_barrier`, `ranges_declared`,
`touched_ranges`, `dispatch_ranges`, `auto_submit_threshold`,
`cb_split_threshold`, `maybe_split_cb!`, `concurrent_dispatch_group`,
`concurrent_indirect_group`, `deferred_indirect`, `lava_launch!` and
`fast_prepare_indirect!` — `emitkernel!` and `preparekernel` are already the
half of them a plan uses.

## Rules that were broken while writing this

- **Every edit states what it DELETES, before writing.** Adds a field, a flag, a
  counter or a parallel path and deletes nothing → wrong by default, stop.
- **Mid-refactor, an obstacle is the refactor.** Three times in one session the
  cheap local exit was taken: `scope_depth` (suppressed a heuristic instead of
  removing it), `FUSE_SAMPLE` (a second render loop beside the first), a fourth
  bump allocator (copied `capture_arg_buffer!`, the debt, as if precedent).
- **A bug found mid-refactor is NOTED, not fixed.** Check whether the refactor
  already deletes it.
- **No full suites mid-refactor.** MWEs in the warm `bt_julia_eval` session;
  suites at phase boundaries. A full Mantle run is ~24 min.
- **All Julia through `bt_julia_eval`.** Not `julia --project` via Bash — and
  never piped through `tail`/`grep`, which buffers and loses the whole log if
  the process is killed.
- **Warm before measuring.** The argument ring is 3 deep; fewer than ~10 warm
  iterations is still cold. Take min AND median and refuse any row where they
  disagree. Three claims reversed under warmup this session.

## State

Committed on `sd/lava-refactor`: `0d3f3b5` (Mantle — `repeat!`, `waitfor!`,
`bufferusage`, `waitidle`), `d472e98` (Hikari — a sample is one plan, chunking
deleted), `7f8396c` (step 0). Uncommitted: steps 1 and 2.

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
