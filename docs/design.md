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
  release, growth that ADDS a block rather than moving one. Four backend
  primitives, no policy: `rawalloc`, `rawfree`, `constraintof`, `compatible`.
  15 tests against a fake device that allocates nothing and counts calls — it
  needs no backend at all, which is the assertion that no policy leaked out.
* Lava `bind_buffer!` + `unbound_buffer` + `buffer_requirements`, mirroring the
  image family. Verified on the RTX 4000: two 4096-byte buffers into one
  8192-byte `device_memory` at offsets 0 and 4096. Before this there was nowhere
  to bind a suballocated buffer — every `bind_buffer_memory` in Lava passed 0 —
  which is *why* the buffer arena took the `LavaArray` shortcut.

Not done: `Place` still calls `allocate` and gets a fresh slab, and a `LavaPlan`
still owns its memory instead of holding regions it releases.

**THE OPEN HAZARD, and it must be settled before wiring Lava's `rawalloc`.**
Whether a `LavaArray` built over an existing buffer OWNS that buffer is
unverified. Evidence points both ways:

* Lava's memory.jl says "a sub-allocation is returned to its block by a
  **finalizer**", so LavaArrays do finalize.
* But `VkManagedBuffer.pool_block` is documented "nothing = non-pooled", and
  Mantle's current `materialize!` already does
  `LavaArray{T,1}(copy(slab.buf), (n,); offset)` on every transient without
  double-freeing — so either that path is non-owning, or the copy carries a null
  `pool_block`.

Get this wrong and memory Mantle owns is returned to Lava's pool by the GC
thread. That is the exact shape of three bugs this codebase has already paid
for — the pool free-list finalizer race that became a SIGSEGV, the buffer
lifetime OOM, and the device loss in #13. So: an MWE with a negative control
first (allocate, wrap, drop, GC, assert the memory is still valid), and only then
the `rawalloc` wiring. Do not infer it from reading.

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
| 0 | Delete the 5 redundant node syncs (#96); drop `isneutral` from `graphof` | suite unchanged; makes any later A/B honest rather than flattering |
| 1 | `source → colour → blur` as a Mantle graph on Lava, behind a flag | **pixel-identical** to `execute!`; `peakbytes` vs the pool's high-water mark |
| 2 | Parameters as `Scalar` + `Update` instead of baked constants | a keyframed σ costs a store, not a recompile |
| 3 | Result store + `PlaneOp` (#94, #95) | matte/restore/stabilize are nodes, not special cases; five storage schemes become one |
| 4 | Compositor — `placelayer!`, `layermatrix`, `mattealpha!` | currently outside `execute!` entirely |
| 5 | VkMakie | the readback round-trip is gone |

Stage 1 is the falsification step and everything after it is porting. It will
surface what design cannot: how the decoded frame arrives (`Update` vs a
`custom!` decode pass), what `cropaway!` becomes while crop is not a node, and
whether `gaussianblur!`'s separable two-pass form fits `custom!` as cleanly as
claimed.

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
