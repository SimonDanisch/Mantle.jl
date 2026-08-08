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

**The part that cannot be hand-waved:** reclaiming a region means rebinding when
the plan comes back — `bind_image!` + `image_view` per transient. Reclaim is not
free, the policy has to know its price, and nobody has measured it. That is the
real content of this section, and it is a measurement rather than a decision.

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

* That reclaiming a plan's region is affordable. It costs `bind_image!` +
  `image_view` per transient on return, and nothing has measured that. It is the
  number the residency policy in §3 turns on.

* That the Host numbers say anything about Lava. They do not — Lava's `allocate`
  is `device_memory` and its `materialize!` is `bind_image!` + `image_view` per
  transient. **The Host `Place` number must not be used to justify anything.**
* That Mantle's aliasing beats `BufferPool`. Only shown to alias *well*, never
  *better* than a pool that mutates in place aggressively. Stage 1 settles it.
* That a per-frame plan can be rebuilt on a timeline drag. Needs Lava `Place` and
  `Pipelines` at ~30 and ~1000 passes.

**Deliberately not designed for:** the host backend. It is a fallback and a test
harness. No architectural decision should be justified by a CPU measurement.
