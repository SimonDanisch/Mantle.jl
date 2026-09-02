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

### 1. Recording resources come from the pool

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

### 2. `record!` / `run!` with a real emitter

`emit_*(e, …)` takes a plan, a slot and a command buffer — never a queue. So
`maybe_split_cb!`, `auto_submit_threshold`, the elision tracker,
`next_skip_barrier`, `ranges_declared` and `scope_depth` become unreachable.

**Deletes:** `capture`, `sealinto!`, `bq.capturing`, `cb_begin_flags`'s branch,
`scope_depth`, `record_pass_work!`'s queue coupling, and the `:custom` pass kind
(unused — Hikari has none; Mantle's tests do).

**Also:** descriptor sets for HWTLAS are allocated per dispatch today, defended
as "removes the entire class of bug". That was a lifetime fix wearing a
performance hat; the recording owns them instead. See
`feedback_no_per_dispatch_descriptor_sets`.

**Settle: `SIMULTANEOUS_USE` or no flag.** Not `ONE_TIME_SUBMIT` — that means
"submitted once, then reset or freed" and is illegal for a buffer that is
replayed, so it was never a candidate. No flag permits resubmission but only
after the previous submission has completed; `SIMULTANEOUS_USE` permits it while
still pending. With step 3 there are no argument slots, so this reduces to: does
`run!` submit again before the previous submission finished? If it always waits,
no flag is correct and cheaper.

`build_plans` names `SIMULTANEOUS_USE` as the suspect for baking's 2–3%, but that
comparison moved two variables at once (interpreted-and-recorded-fresh against
baked-and-replayed) through the heuristic recorder. Re-measure after step 2, do
not inherit the suspicion.

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
`nextslot!`. A plan with no per-run values needs **one** slot and no wait.

`Scalar` is `GPURef` plus a `stride`-0 draw-binding rule; it has one real caller
in three packages. Fold it in, keeping `Buffer{T}`/`Scalar{T}` erasing to the
same `Attr{T}` — that is load-bearing for pipeline identity.

**Hikari side:** `sample_idx` is `iteration_index + 1`, a counter the host
happens to own. Derive it on the device and the write plan is *empty* — same
move `repeat!` makes for the loop counter.

### 4. Per-swapchain-image recordings

`bake!` refuses any plan with a surface: "a swapchain image is a different image
every frame and a recording names one". RayMakie draws to a window, so this is
what keeps the interpreted path alive.

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

A single dispatch is a one-pass graph; the machinery exists. **Deletes:** the
remaining queue fields, `record_dispatch!`'s decision-making, `barrier_mode`
(`:derived` becomes the only mode — `:backend`/`:both` are A/B scaffolding),
`concurrent_dispatch_group`, `concurrent_indirect_group`, `deferred_indirect`.

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
deleted). Uncommitted: step 0.

`dev/Vulkan` is dirty on purpose: a local patch to `generated/linux.jl:51890`
for a Vulkan.jl generator bug that makes `VK_EXT_device_generated_commands`
unreachable (`_IndirectCommandsLayoutTokenEXT` never applies its enum
conversion). It tracks upstream — do not commit. Worth reporting.

Reference numbers to compare against, NVIDIA, 256², 8 spp, warmed, median of 11:
`out/bench_head_warm.txt` (HEAD host loop) and `out/bench_new_warm.txt` (fused).
Correctness reference: `out/ref_new.txt`, 12 configurations, bit-exact.
