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

Done and pushed:

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
