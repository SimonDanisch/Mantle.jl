# Memory, and moving VideoEditor onto Mantle

Written 2026-08-08 on `sd/host-backend`, because the earlier plans could not be
found: Mantle appears in no markdown anywhere under `/sim/Programmieren`, its git
history has never contained a `.md` on any branch, and the VulkanDev project's
memory has 40+ notes and none about it. This is a reconstruction from the code,
the docstrings and one design conversation — **not a recovered document**, so
treat every "decided" below as "proposed, correct me".

The design record that *did* survive is the docstrings, and they are good: the
RPS citations, the "two non-negative terms rather than their difference" note in
`Schedule`, the fuzzing history in `Aliasing`. What is missing is the memory
story, and the only trace of it in code is a parameter nothing sets.

---

## 1. The claim

One allocator across simulation, graphics, raytracing, DNN inference and video —
not one allocator per subsystem. Everything else Mantle does (the DAG, the
schedule, the barriers) exists to make that possible: liveness is what lets two
things share bytes, and a graph is what makes liveness derivable instead of
declared.

Two properties follow, and they are different problems:

**Under-subscribed — how close to optimal?** A ratio, not a byte count, and
machine-independent. Already ratcheted: `readproblem` against the minimalloc
benchmarks, plus a test that the gap to their capacity does not grow.

**Over-subscribed — what happens when it does not fit?** Unanswered. This is
where a graph beats a conventional allocator, because it knows the producer of
every value and therefore what is cheap to drop and recompute rather than keep.

## 2. What exists, and the hole in it

`Problem` takes a capacity. The CSV reader uses it, every benchmark is about it,
and the runtime passes infinity:

```julia
pl = place(Problem(a.items[idx], typemax(Int)))     # src/phases.jl, Place
```

So the entire bounded case is unexercised by anything that runs. `Usage` already
exports `evictable` and `aliasable`; nothing consumes them.

Open, in the order they need answering:

1. **What does `Place` do when the items do not fit?** Today: whatever `place`
   does, and nothing decides what to do about it.
2. **Where does the capacity come from?** Nothing queries the device.
3. **Given liveness, what is cheapest to spill or recompute?** The graph knows
   every value's producer. No conventional allocator has that, and it is the
   whole differentiator — not a smaller peak, a *graceful* one.

## 3. The device owns the pools

Scheduling and memory are different axes. A plan decides what runs and in what
order; the **device** decides where bytes live, across every plan it knows about.

Nothing about that should reach the caller:

```julia
plan = Plan(g)      # that is all. `g` knows its device; the device owns the pools.
```

An earlier draft of this file had `ArenaPool(dev)` passed in and plans declaring
`group = :models`. Both were wrong in the same way — they made the caller plumb
and promise the things Mantle exists to decide, against the standing rule that
memory management is centralized and high-level code never manages GPU resources.

**One pool per memory kind, held by the device, permanently.** The mechanism is
already in embryo: `arena(t)` returns `Buffers()` or `Images()` and `Place` runs
one placement per arena. That is a *derived* per-transient property — the
transient says what kind of memory it needs, nothing declares where it lands.
Extending `arena` is then how a new workload gets its memory right without new
API: device-local, host-visible/BAR, images keyed by `req.type_bits` (already why
`Images` intersects them), acceleration-structure scratch for RT.

`allocate` changes meaning with it: not "make an allocation" but "give me `bytes`
in the pool for this kind, growing it if needed". That also removes the
per-compile allocation — on Lava, `device_memory` plus `bind_image!` per
transient, every compile, where it should be per *growth*.

**Residency is observed, not declared.** `run!` makes a plan resident; a plan
that has not run holds a region that is a reclaim candidate. No promise for
anyone to get wrong, and it degrades on real behaviour rather than on an
annotation someone forgot to update. The concrete win is unchanged: SAM 2's
activations and the editor's per-frame transients cannot coexist — a click stalls
the preview by construction — so today that is gigabytes held twice.

**The part that cannot be hand-waved — and it is two different problems.**
"Rebinding" was the wrong word for it; buffers and images do not pay the same
price, and only one of them can move at all.

*Buffers move freely.* `materialize!(t::TransientBuffer, slab, offset)` builds a
`LavaArray` over the slab at an offset, and the `copy(slab.buf)` in it copies a
handle struct, not bytes. Moving one is constructing a Julia view. The driver is
not involved, and everything above applies unchanged.

*Images cannot be rebound.* `vkBindImageMemory` is once-only — an image that
already has memory bound may never be bound again. So moving an image transient
is: destroy the old `VkImage` (once it is out of flight), `image_2d` a new one,
re-query `image_requirements` (alignment and type bits may differ), `bind_image!`,
and `image_view`. Four driver object operations per image plus a deferred
destroy.

That is not hypothetical machinery: `refit!` already does exactly this for the
resize case, so the price is in the codebase today and can be measured without
building anything.

**But ask how often this happens before pricing it.** Plan construction: once.
Structural edit: a user action, seconds apart. Residency eviction: rare by
construction, since the plan being evicted is the one not running. In the steady
state — scrubbing, playing, matting — nothing moves and images stay bound.

So reclaim cost is NOT load-bearing for this design, and an earlier draft of this
section treated it as the main event. It is not. The one genuinely hot case is
already in the tree and has nothing to do with residency: `refit!(pl)` runs every
frame inside `run!`, and during a window-edge drag it fires for real — every image
destroyed and recreated plus a full recompile, at 60 Hz. THAT is what to measure.

Which also means an image pool keyed by `(extent, format, usage)` is probably
unnecessary complexity. Bind once at plan construction, pay on structural change.
The pool earns its keep only if the resize drag measures badly.

The strategies do still differ by kind:

* buffers — memory-level suballocation, offsets are free to move
* images — the unit of reuse is the IMAGE OBJECT, not the region: a pool keyed by
  `(extent, format, usage)` handing out a `VkImage` whose memory was bound once.
  Which is also why `Images` already intersects `req.type_bits` — the constraint
  was visible in the existing code.

## 3a. Where the pool work actually stands, and the one open hazard

Done, on `sd/mantle-dev` and not yet pushed:

* `src/memory/pool.jl` — `Pool`/`Block`/`Region`, first-fit with coalescing
  release, growth that ADDS a block rather than moving one. Backend primitives
  only, no policy. Tests run against a fake device that allocates nothing and
  counts calls — no backend at all, which is the assertion that no policy leaked.
* `src/memory/array.jl` — `DeviceArray{T,N}`, a typed handle on a `Region` that
  owns NOTHING. No finalizer, nothing for the GC thread to race.
* `src/memory/resources.jl` — `Buffer`/`Scalar` in core over `DeviceArray`.
  `LavaBuffer`/`LavaScalar` and the Host pair are gone.
* `Place` suballocates instead of allocating per compile. Measured: ten plans of
  524_288 B peak went from ten device allocations (5_242_880 B) to ONE block,
  plans 6-10 reaching the device zero times.
* **Two allocation verbs, and which one you get is decided by what you are
  allocating.** `acquire!` hands out a private slice; `reserve!` hands every
  tenant of an arena the SAME slice, so two plans on one device cost the larger
  rather than the sum. `Place` reserves, because a transient is scratch scoped to
  one run. `allocate` — every `Buffer` and `Scalar` — acquires, because a
  persistent resource holds data between runs and sharing its bytes would be
  silent corruption. There is deliberately no flag to get that wrong with.

  Sharing is what a device-owned arena is *for*: SAM 2's scratch and MatAnyone's
  never coexist, and two plans that each acquired a private region could not
  share however well either was placed. What makes the overlap safe is split by
  who knows what — ordering is one barrier at the head of a recording, emitted
  when `takeover!` says a DIFFERENT tenant wrote these bytes last, which only a
  backend can do because only it has a command stream; data lifetime is the
  caller's, and is the same rule that already governs two runs of one plan.

  Not `sharing`: that says the arena has more than one tenant, which is true of
  every run once two plans exist, so a plan run repeatedly — playing one clip,
  replaying a baked model — paid a memory barrier per frame to be ordered against
  itself. `takeover!` records and asks in one call, because asking without
  recording emits forever and recording without asking emits never.

  Both of those are corrections of *meaning*, not measured wins, and the attempt
  to measure them is worth recording as a warning. A composite benchmark read
  1.95 ms before the merge and 3.15 ms after, which looked like a 60% regression
  and sent me scoping the barrier to fix it. It was not one: the SAME benchmark
  on the SAME build, minutes apart, gives medians of 0.999 ms and 4.948 ms while
  its MINIMUM moves the other way (0.817 -> 0.614). The median is GPU clock
  state; only the minimum is stable, and by it nothing moved. Cross-session
  medians cannot attribute anything here — see the project's measurement notes,
  which say exactly this and which I did not follow.

  Growth carves a fresh region, remaps live tenants, then releases the old — in
  that order, so it never tramples the bytes it is copying out of. `remappable`
  is asked first, of the INCUMBENTS: a baked plan's recording holds the addresses
  its region has today, so growth under one is refused with a message naming the
  order that would have avoided it, rather than producing a replay that reads
  freed storage. Refcounting is the weak tenant list itself, not a number that
  can disagree with it; `free!` deregisters and the last tenant out gives the
  bytes back.
* **An arena reconciles what all its tenants need, not just the newest.**
  `constraintof` is per-plan — buffer usage bits UNION over that plan's
  transients, image memory-type bits INTERSECT — so an arena sized by one plan
  could hand the next a block that does not permit what it does. `reserve!`'s
  fast path therefore checks `compatible` as well as size, the arena carries the
  merged constraint of everything placed in it, and growth passes that to
  `acquire!` rather than deriving one from the newest plan alone.
  `mergeconstraints` is a backend primitive because the two directions are not
  symmetrical and core cannot guess; its default demands equality, which is the
  safe reading for a backend that has not said. An empty image intersection is an
  error naming that no allocation can serve them all, rather than a bigger arena
  that would not have helped.
* **What sharing makes possible to get wrong, and what stops it.** Overlapping
  plans is a correctness claim, so each way of breaking it is refused by name
  rather than left to produce a plausible answer:

  | you might | and instead of | you get |
  |---|---|---|
  | acquire a persistent resource from a shared arena | two live buffers on one byte range | no route to — `allocate` acquires, `Place` reserves, decided by what you allocate rather than by an argument |
  | grow an arena under a baked plan | a replay reading freed storage | `remappable` asked of the incumbents, refused with the ordering that avoids it |
  | run a plan after `free!` | recording against another plan's bytes | `checklive`, from the state that already exists — transients but no regions |
  | place more than the device has | `ERROR_OUT_OF_DEVICE_MEMORY` somewhere later | `checkcapacity` at the point that still knows the items, naming the largest |

  What is NOT guarded is data lifetime across a handover: a plan's transients are
  gone once another tenant runs. That is the same rule two runs of one plan
  already follow, it is what the sharing buys, and no barrier can make it false.
* **Sharing, measured on the models rather than on a synthetic pair.** SAM 2
  builds three plans and they share ONE 180 MiB arena; the sum of their peaks is
  196.21 MiB. Warming MatAnyone on top grew the pool by **zero** — 392.23 MiB
  before and after, arena unchanged — because the second model's transients fit
  the arena the first had already sized. That is the whole premise of one
  allocator seeing every workload, and it is the number that says whether it
  works.
* **What is still in both extensions, and why.** Of the names defined in each,
  the backend primitives belong there — `rawalloc`, `rawfree`, `upload!`,
  `download`, `deviceview`, `devicecopy!`, `materialize!`, `constraintof`,
  `compatible`, `mergeconstraints`, `capacity`, `maxalloc`, `Device`, `backend`,
  `storage`, `arena`, `alignment`, `nbytes`, `describe` — as do the four that say
  what a pass IS: `newpass`, `handle`, `dispatches`, `passes`. Most of the rest
  are one-line accessors answering for a struct that backend owns
  (`analysis(c) = c.analysis`).

  What is left is the graph data model, and — MEASURED rather than assumed — it
  is not worth hoisting. The two graphs declare the same five fields, but that is
  parallel *declaration*, not shared *work*: only 30 references touch them at all
  (Host 8, Lava 22), so a core `GraphCore` would trade ten declaration lines for
  thirty indirections. And the two functions that look duplicated are not:
  `updates_pass!` identifies its pass by name on the host and by kind on Lava
  (the host's pass has no kind, and it attaches a step where Lava dispatches on
  kind in `run!`), and `Transient.Buffer` registers into `transient_by_id` on the
  host where Lava does it in `touch!`.

  So `custom!` and `compute!` were the last things genuinely identical, and they
  came across once the four pass hooks existed. An earlier draft of this
  paragraph called the data model "the obvious next lift"; counting the
  references is what changed the answer, and it is recorded here so the next
  reader does not do it on that say-so.
* **What a second backend costs, measured.** Against `sd/mantle-dev`, the merge
  moved the allocator, placement, the capacity bound, `UpdateRef`, the `custom!`
  contract, the id table, the pass protocol and plan teardown into `src/`, while
  `ext/MantleLavaExt.jl` SHRANK — it gained `bake!` and lost more than that to
  core. `MantleHostExt.jl` is 363 lines for a whole second backend, which is the
  number that says whether the split is real: it implements primitives and no
  policy, and its own barrier method is "nothing to emit".

  Core 2619 lines against ext 3063. The number to watch is **ext:core** — how
  much backend code it takes to carry a line of shared code — and it went 2.33
  (before any lift) to 2.32 (`sd/mantle-dev`) to **1.17**. Stated that way round
  deliberately: the figure is above 1 because Vulkan is genuinely large (2700 of
  the 3063 lines), not because core is thin, and reading it upside down turns a
  halving of backend weight into a claim core is smaller than it is.
* The capacity bound is core's, not a backend's. `headroom` is
  `min(maxalloc, budget − reserved + largestfree)`; the last term keeps it tight,
  since a request an existing block can absorb reaches no device allocation and
  must not be refused by a budget it never spends. It is checked at the point
  that still knows the ITEMS, because "over budget by 40 MB" is answerable by
  dropping one buffer and unanswerable without knowing which. `maxalloc` is a
  primitive with a permissive default — NVIDIA answers "no limit", the APU
  answers 4 GB.
* Lava's buffer arena takes raw `device_memory` via `bind_buffer!`, so neither
  arena is carved out of Lava's own pool.
* `Device(Lava)` and `Device(LavaBackend())` resolve to ONE cached device per
  `VkContext` — a second would be a second pool over one VkDevice.
* `DNNKernels.scratchfor`'s slab comes from the pool. That slab is the largest
  thing a model allocates and it is scratch, so it is exactly what an allocator
  seeing every workload should reuse.

The primitive surface a backend supplies, and it is all of it: `rawalloc`,
`rawfree`, `constraintof`, `compatible`, `upload!`, `download`, `deviceview`,
`devicecopy!`, `bufferusage`, plus `device`/`pool`/`blocksize` on the context.

**One Mantle device per physical device.** A draft let the KA extension accept
any KA backend, reasoning that a KA workload wants scheduling and allocation
rather than render passes — true, and beside the point, because it would hand
DNNKernels a second `Device` with a second `Pool` beside the real one. Fidelity
is per PASS (`dispatch!` / `render!` / `custom!`) and already exists.

## 4. Moving the editor

Composition needs no new concept: build one graph from parts, so the placer sees
every clip's transients at once.

```julia
g = Graph(dev)
layers = [chain!(g, c, source!(g, c, frame), engine) for c in visibleclips(seq, frame)]
canvas = composite!(g, layers, seq)
ui!(g, editor, canvas)       # VkMakie
present!(g, window)
```

**One graph per cadence, three of them.** Per-frame (UI + clip chains +
compositor). Per-invocation (a model, built once and kept — 500–1500 passes, and
`lastuses` was 1.75 s of a 2 s compile at that scale before the `sliceindex`
fix). Per-clip analysis (a driver running one plan N times while streaming frames
in and results out).

The payoff of the first is not tidiness. Today a frame goes **GPU → host readback
→ texture upload → GLMakie draw**, every frame. With the UI in the same graph the
clip chain writes a transient the UI samples: the picture never leaves the
device, and the compositor's intermediates alias against the UI's scratch.

### Stages

| | | done when |
|---|---|---|
| 0 | Delete the 5 redundant node syncs (#96) | **landed** — suite unchanged |
| 1 | The clip chain as a Mantle graph | **landed, not behind a flag** — see below |
| 2 | Parameters as `Ref`s read at record time | **landed** — a keyframed σ costs a store, not a recompile |
| 3 | Result store + `PlaneOp` (#94, #95) | **landed** — see below |
| 4 | Compositor — `placelayer!`, `layermatrix`, `mattealpha!` | currently outside the graph entirely |
| 5 | VkMakie | the readback round-trip is gone |

**What landed for 1–2, and how it differs from the guess above.** `execute!`,
`FxGraph`, `BufferPool` and the per-node `eval_node!`s are *deleted*, not
flagged: a clip's chain is a Mantle graph, one `custom!` pass per node. Every
node is `custom!` rather than `dispatch!` because the bodies are multi-launch
or host-branching — a decode, a model call, a separable blur with a scratch
buffer — and `dispatch!` expresses exactly one kernel. That choice also settled
stage 2's mechanism: a `custom!` body reads its node's parameters from a `Ref`
at record time, so per-frame values need no `Scalar`+`Update` at all (that pair
is the mechanism for `dispatch!`/`draw!` *arguments*, and no node is one). The
engine caches one compiled plan per (node types in order, frame size), and the
graph is rebuilt only when the structure changes. The compositor's scratch
(`accum`/`warpbuf`/`cover`/`alphalayer`) is persistent `Mantle.Buffer`s on the
same pool — inside its own graph is stage 4. Plan teardown is `Mantle.free!`,
called by `emptyengine!`.

Verified: GPU old-vs-new is **bit-identical** on both source types (CPU-frame
upload and `GpuVideoStream` hardware decode); the CPU graph is bit-identical to
the same kernels applied by hand; a param change reuses the plan; aliasing
engages (opacity→blur→sharpen at 320×180: peak 3 frames, naive 5).

What the falsification step surfaced: how the decoded frame arrives (a `custom!`
source pass reading `FxState` — not `Update`, because the source object itself
changes per call), what `cropaway!` becomes while crop is not a node (it stays
one, applied to the output view after `run!`), and that `gaussianblur!`'s
separable two-pass form fits `custom!` exactly as claimed.

**What landed for 3.** The distinction the stage turns on is not matte-vs-restore
but what a per-frame analysis result *is*. A colour gain and a warp matrix are a
handful of numbers and ride along as kernel arguments; a matte's alpha and a
restoration's finished picture are whole images, and those are `PlaneOp`s —
`MatteOp`, `RestoreOp`, one `PlaneNode{O}`, one pass, four methods each
(`planeeltype`, `planeshape`, `planedata`, `applyplane!`) defined next to the
kernel that wants them. The two nodes, two kernels-with-their-own-lookup and two
device caches are one route now.

That route is the graph's: the plane is a pool-backed `Mantle.Buffer` written
through `Update` and declared `read` by the node's pass, so the copy is ordered
by the barrier the graph derives. What it replaced was a `KA.allocate` per frame
behind a module global — outside the pool, never freed, and re-allocating ~6 MB
every time a restored frame changed.

Three things had to move for that to be possible:

* **The decode left the source pass.** An `Update`'s write position is at the
  head of the schedule, and the plane is keyed by the frame the source really
  *served* — which `frameat!` decides while it runs. So `decodesource` is called
  from `runchain!` before `run!`; the pass converts what it produced. The
  decode's own submits also stop landing in the middle of the plan's recording.
* **The plane's SHAPE joined the plan signature.** Values flow through `Ref`s,
  but a shape sizes a graph resource, so a matte analysed at another resolution
  gets its own plan rather than a buffer of the wrong size.
* **Invalidation became a stamp on the result's identity.** `(clip, frame)`
  cannot see a re-propagation — a new `MatteTrack` under an unchanged clip and
  frame — which is why eight explicit `freematteplanes!` calls existed. They are
  gone: the slot remembers *which* track's bytes it holds.

Five storage schemes became two mechanisms and one field. `MATTEPLANES`,
`RESTOREPLANES`, `CanvasScratch` and `alphalayers` are one `BufferStore` (a
per-frame plane and a sized scratch differ only in whether an `Update` is bound
to it); `RESTORECACHES`, a module global keyed by clip id, is a `Clip` field like
the three tracks beside it — which deletes `sharerestore!` and the bug it
existed for, since `split!` mints the right half's id fresh and nothing followed.
The compositor's coverage now comes from the chain's own matte binding rather
than a third independent lookup, so "is this layer keyed" has one answer.

Verified on the host backend and on Lava: the plane node's shape matches the
analysis; the graph's picture matches the host-side `applymatte!`/`applyrestore!`
(exactly on the host, within 1/255 on Lava — 92 of 57_600 pixels, Float32
rounding on the soft edge, the same parity the WYSIWYG check tolerates at 0.01);
a strength change reuses the plan; an unchanged frame queues no upload; a new
track under the same clip and frame does; a frame outside the analysed range
renders the picture untouched; the compositor shows the track below where the
matte removed the background; `emptyengine!` returns every slot.

**Measured, and it is not a speed change.** A/B on the composite path (two
layers, top matted, 120 distinct 1080p frames, matte 480x270), three repeats per
tree in one session each:

| | old | new |
|---|---|---|
| medians | 1.943 / 2.044 / 2.088 ms | 1.949 / 1.923 / 1.958 ms |
| within-tree spread | 7.5% | 1.8% |
| host alloc per frame | 274.8 KiB | 271.6 KiB |
| Mantle pool reserved | 128 MiB | 128 MiB |
| Lava-tracked device | 76.0 MiB | 12.1 MiB |

The distributions overlap — the old tree's fastest run beats the new tree's
slowest — so the 4.6% on median-of-medians is smaller than the old tree's own
spread and is not claimed. The real number is the last row: the per-frame
`KA.allocate` made Lava's own allocator open a 64 MiB block the pool could
neither see nor reuse, and the plane now sits in the persistent block the
compositor scratch had already reserved. That is specific to compositing — in a
`render`-only workload it is a wash (64 + 76 against 128 + 12.1), because there
the pool opens its persistent block for the plane alone.

Host allocation did not move, though `planedata` hands over a view where the old
path built `collect(@view …)`: Lava's `rename!` collects a non-`Vector` anyway,
so the copy moved rather than went. And `Update`'s recorded copy was expected to
beat the old stalling `copyto!`; at a 130 KB plane against a 2 ms frame it does
not show, and is not claimed either.

That last check is also what found a bug the rewrite did not introduce:
`restore_kernel!` built its `RGB{N0f8}` with the *checked* constructor, whose
error path allocates a string, so Lava rejected the kernel — the restoration
could never have run on the GPU tier, and nothing had ever asked it to.

## 5. Measured vs assumed

**Measured** (this branch, unless noted):

* Phase costs, Host, 6-node 1080p graph: Dag 0.002 ms, Schedule 0.014, Liveness
  0.003, Place **4.644**, Aliasing 0.004, Barriers 0.0006, Pipelines 0.008. Place
  is 150× everything else, and it is the `zeros()` — not the placer.
* `bench/chain.jl` on Lava after the phase lift: 22.89 MB peak vs 48.64 naive,
  23.69 ms median over 120 frames, picture correct.
* The five lifted phases contain **zero** `Lava.` references between them.

**Assumed, and load-bearing:**

* That reclaiming an IMAGE region is affordable. It is a destroy + recreate +
  rebind + new view per image, not a rebind, and nothing has measured it. `refit!`
  already performs exactly that sequence, so it can be timed today. Buffers are
  not in question — moving one is a Julia view.

* That the Host numbers say anything about Lava. They do not — Lava's `allocate`
  is `device_memory` and its `materialize!` is `bind_image!` + `image_view` per
  transient. **The Host `Place` number must not be used to justify anything.**
* That Mantle's aliasing beats `BufferPool`. Only shown to alias *well*, never
  *better* than a pool that mutates in place aggressively. Stage 1 settles it.
* That a per-frame plan can be rebuilt on a timeline drag. Needs Lava `Place` and
  `Pipelines` at ~30 and ~1000 passes.

**Deliberately not designed for:** the host backend. It is a fallback and a test
harness. No architectural decision should be justified by a CPU measurement.

---

## Porting Hikari onto Mantle — what the code says, 2026-08-19

Written while adding `DeviceRange`. Four findings, in the order they constrain
the work.

### 1. Indirect compute dispatch existed nowhere. It does now.

`dispatch!` took a host-side `ndrange`; indirect was a draw-only path
(`drawover` → `vk_draw_indirect_in_pass!`) plus `Indirect` as a usage for state
tracking. Hikari is a wavefront tracer: every stage dispatches over "however
many rays survived", a number that only exists on the device. Reading it back
per stage per bounce is a flush and a fence each — the cost the design exists to
avoid — and `ndrange = capacity` is recorded in the project's notes as
catastrophic.

`DeviceRange(count)` takes a device-resident ELEMENT count and the backend
converts. See its docstring for the tail contract.

### 1a. A Hikari kernel already runs unchanged inside a graph

Proved before touching Hikari, which is the cheap half of the question:
`vp_accumulate_to_rgb_kernel!` — static ndrange, one writer, one reader, no
indirect — dispatched through `Mantle.dispatch!` on Mantle-owned buffers gives
output BIT-IDENTICAL to the same kernel launched straight through KA. So the
kernels port as they are; what does not port for free is their memory, which is
the next finding.

Script: `/sim/tmp/bench/mantle_slice.jl`. One wrinkle worth keeping: the
accumulators are flat `Float32` of length `3n`, not `RGBSpectrum` of length `n`,
because the kernel accumulates with Atomix and atomics need a scalar element
type. A port that "tidies" that into a struct array will fail to compile with a
method-lookup failure inside `Atomix.modify!`, which reads as a GPU-codegen
problem and is not one.

### 2. Memory ownership is one-way, and this decides the sequencing

`Mantle.deviceview` builds a `LavaArray` over a Mantle region with a NO-OP
releaser — Mantle stays the owner. There is no inverse: nothing adopts an
existing `LavaArray` into a graph.

What that does NOT mean, and an earlier draft of this section got it wrong: a
pass can still READ and WRITE a buffer Mantle does not own. `M.use(p, x)` takes
any resource, `resourceid` assigns it an id, and a plain KA-allocated
`LavaArray` passes through to the kernel unchanged — measured, not assumed
(`/sim/tmp/bench/mixed_ownership.jl`).

So the constraint is narrower than "move the memory first". What Mantle cannot
do is ADOPT a foreign buffer into its arena, which is what aliasing and liveness
need; what it can do is order passes around one. That makes the port
INCREMENTAL: move a kernel into a graph with its inputs still Hikari's, gain the
derived barriers immediately, and move ownership later only where aliasing is
worth having.

Which also reprioritises the first slice. The film accumulate is a poor one — two
passes, run at different frequencies (accumulate per sample, finalize once), so
there is barely an edge to derive. The bounce loop is the real target and is now
reachable without moving any memory: its ~20 dispatches per round have real
dependencies and are ordered today by hand-placed
`concurrent_dispatch_group`/`concurrent_indirect_group` calls, which is the
thing worth deleting.

### 3. The bounce loop does not need a graph-level loop node

A `Plan` is compiled once and `run!` repeatedly, so a plan per bounce, driven by
the host loop that already exists, is expressible today. What is NOT expressible
is the early exit, which reads a device counter and breaks — that stays a host
readback, which is what it already is (`EXIT_CHECK_INTERVAL`, every 8 rounds).
No new mechanism needed; just do not expect the loop to disappear into the DAG.

### 4. Ray tracing has vocabulary but no dispatch

`AccelKind`, `TraceRead`, `TraceBuild` exist and lower to real stages, so the
barrier half is there. There is no RT pass, no SBT, no raygen/chit/miss. Hikari
on hardware RT needs that built; Hikari on the software BVH does not, and the
software path is where a port should start for exactly this reason.

---

## The port, done — 2026-08-19

The bounce loop is a graph. Three of the four findings above held; the fourth
was wrong in a way worth keeping, and what it cost was three gaps in core rather
than anything in Hikari.

### What it took

**Three graphs, not one.** `setup` (clear the film, reset the queue, generate
camera rays), `round` (one bounce), `accumulate`, plus `finalize` as its own
plan because a batched caller runs it once at the end rather than per sample.
`round` is built TWICE, one per direction of the ray-queue ping-pong: a plan
resolves its arguments once, and the two directions are different arguments.

**Everything per-sample rides a `Ref`.** The sample index, the camera, the scene
re-adapted every call. That rule existed only in the Lava extension —
`devarg(::RefValue)` — so the host backend handed the `Ref` itself to the kernel
and every CPU render failed on the first dispatch. It is `Mantle.argvalue` in
core now, because it is a promise about the API rather than a conversion.

**A struct of device arrays is an ordinary argument.** `todevice` handled a
top-level buffer and passed everything else through, so a work queue — a payload
array plus an atomic counter — reached the kernel as the host struct. It goes
through Lava's own `Adapt` rules now. Which surfaced the ordering that matters:
a ray query needs the `HWTLAS` bound as a descriptor and adapting an
`HWAdaptedAccel` deliberately strips it, so the TLAS is looked for in the RAW
arguments. Asking the adapted ones finds nothing and the shading kernel compiles
with ray query disabled — which fails in the SPIR-V emitter, a long way from the
cause.

**N device-sized dispatches in one pass share one prepare and one barrier.** The
barrier before an indirect dispatch is forced (the command processor's read of
the count depends on the prepare that wrote it), so per dispatch it serialised a
pass whatever the schedule said. Twelve per-material shading kernels were twelve
barriers. This is the one hazard a graph cannot derive away, and the answer is
to make the pass the unit: the prepares are fused, the dispatches share the
barrier behind them.

### Finding 4 was wrong, and `custom!` is why

"Hikari on hardware RT needs an RT pass built" — it does not. The hardware trace
is `vkCmdTraceRaysIndirect` with an SBT, which is not a dispatch, so there is no
kernel and no ndrange for the graph to record. `custom!` is exactly that case:
the body records through the backend and the graph still orders it, because what
it touches is declared the same way either way. The compute and hardware forms
of the trace stage differ in the pass KIND and in nothing else — same
`trace_uses!`, same position in the round.

That does not make an RT pass pointless — an SBT the graph knows about could
carry `TraceRead`/`TraceBuild` properly — but it is no longer on the critical
path for a wavefront tracer.

### What it bought, measured

Correctness first, because a renderer that got faster and darker is not a port:
the 96x96 Cornell scene with media, four material types, an area light and a
point light is **bit-identical** to the pre-port tree on the software BVH, and
four progressive samples equal one four-sample render to the last digit. The
pbrt reference suite is **450/450 SW and 450/450 HW at 256 spp**.

Speed, minimum of five renders at 32 spp (the median on this machine is GPU
clock state — see the measurement notes above):

| scene | old SW | new SW | old HW | new HW |
|---|---|---|---|---|
| shadow_bumpgold_dome_over_velvet | 0.788 | **0.762** | 0.875 | **0.861** |
| mat_mix_light_point | 0.095 | **0.086** | 0.108 | **0.097** |
| medium_cloud_point | **0.352** | 0.355 | **0.362** | 0.365 |
| medium_null_interface_homog | 0.159 | **0.157** | **0.179** | 0.183 |

The one number outside noise is the multi-material scene, ~9 % SW and ~10 % HW,
which is where the derived independence and the fused prepares have something to
work with. Everything else is a wash, which is the right result: the point was
to stop asserting the ordering by hand, not to make the GPU do less.

### After the port: three things the graph could not say

Each was found by the port needing it, and each was a gap in Mantle rather than
in the renderer.

**`Mantle.argvalue`** — a `Ref` argument is read per run, on every backend. That
rule lived only in the Lava extension, so the host backend handed the `Ref`
itself to the kernel and every CPU render failed on the first dispatch. It is a
promise about the API, not a conversion, so it belongs in core.

**`rebind!` and `rebindable`** — `bake!` was only half an API. A baked plan does
not record, so the values it packed at capture are the ones it replays, silently
and for ever; `rebind!` writes new ones. And it cannot do that for a `custom!`
pass, whose body packs its own arguments as it runs — so `rebindable` says
whether it can, and `rebind!` throws rather than returning quietly. Baking
Hikari's hardware RT trace, which is exactly such a pass, made every sample after
the first replay the first one's paths while the parity test still passed.

**Lowering `Unordered`** — the type and `needs_transition`'s answer for it were
already here; no backend could lower one, so a graph that declared an unordered
access died in `build_pass_barrier`. Three forwarding methods. On Hikari's
per-pixel radiance, which six stages do nothing to but `atomic +=`, it takes a
chunk of eight rounds from 1788 buffer barriers to 412 — and does not move the
render time, on four scenes, paired and interleaved. That frame is not barrier
bound, the same answer the stage-mask note above reached.

### Recording the whole thing once: measured, and not taken

`bake!` replays a capture instead of re-recording, and the host saving is real —
a round records in 0.101 ms and replays in 0.0058 ms. `Lava.replay!` waits on a
semaphore for the previous replay, so replays serialise, and an ordering that
one recording expresses with an intra-submission barrier becomes a GPU
round-trip between submissions. Free for what capture was built for, one plan
replayed once per inference step; not free for a renderer replaying a chunk four
times a sample.

Paired and interleaved, unbaked against baked: at a chunk of 8 rounds baking
costs +5.2 % on `medium_null`, and at a chunk of 64 — the whole sample, hence ONE
replay — it wins 14.5 %. The whole-sample plan is not the way out, because it
gives up the early exit. So the bounce plans are not baked, and the reason is a
property of replay rather than of the renderer.

It did surface a real leak on the way: `Lava.capture` reserved its argument
slabs against the shared pool for ever. Fixed in Lava — a capture owns its slabs
now — and the fix stands whatever this renderer does with baking.

### What is still Hikari's

Memory. Every queue, accumulator and table is still a `KA.allocate`, and the
passes read them as foreign buffers — which finding 2 says is fine and is what
made the port incremental. Moving ownership is what would buy aliasing between,
say, the medium queues and the surface queues (they are live in disjoint halves
of a round), and it is the next thing worth doing rather than a loose end.
