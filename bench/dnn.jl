# An exported ATen graph, run as a Mantle graph.
#
# DNNKernels already lays a network's transients into one slab (`planslab`) and
# lets Lava's automatic per-dispatch barrier order the work. Mantle does both of
# those from declared usage: it derives lifetimes from which pass touches what,
# places the transients with aliasing, and emits one scoped barrier per genuine
# hazard. So the two answer the same two questions, and this runs SAM 2 through
# Mantle's answers to compare them on a real network rather than on a synthetic
# graph.
#
# One `custom!` pass per op. `dispatch!` is not usable here: it takes one kernel,
# its arguments and an ndrange, while an ATen operator is arbitrary host code that
# picks a kernel, may launch several, and may take workspace scratch on the way.
# `custom!` is the declaration without the how — the pass says what it reads and
# writes, and the body is `runop!`.
#
# **The declarations are discovered by running the graph.** They could be derived
# from the graph JSON, and that derivation would be wrong in the direction that
# matters: `dest` is asked for the elements of a multi-output op under keys no
# buffer has, for the copy `contiguous` forces out of a permuted view, and for a
# dtype the reservation was not sized for. Every one of those is a write, and a
# write nobody declared is a barrier nobody emits. Running the graph once with a
# plan that records what it is asked for names all of them exactly.
#
# What stays outside Mantle's hands, and why it is still ordered:
#
#   * weights and graph inputs — written once by the host before any pass and
#     never again, so there is no hazard to derive.
#   * kernel workspace scratch (`Workspace`) — one buffer handed out by a bump
#     pointer and reset per op, so consecutive ops really do collide on it. It is
#     declared read-write by every pass that takes any, which is what the
#     collision is.
#   * launches *within* one op — `record_pass!` lifts the concurrent group around
#     a custom body for exactly this reason; see `Mantle.exclusive_dispatch_group`.
#
# ── measured, SAM 2.1 large, RADV STRIX_HALO ─────────────────────────────────
#
#                              encoder            decoder
#   ops / passes               549                149
#   buffers placed             791  (slab 890)    163  (slab 175)
#   peak                       180.0 MB           16.02 MB
#     the same by planslab     261.25             16.08
#   passes needing a barrier   548 / 549          140 / 149
#   transitions emitted        2123               449
#   DNNKernels + Lava          734.3 ms           52.8 ms
#   Mantle, derived barriers   704.2              52.0
#   Mantle, backend barriers   697.1              51.9
#   Mantle, baked and replayed 557.2              51.0
#   difference in the result   0                  0
#   the same, baked            0                  0
#
# Re-measured after the extension landed, and both halves of the table moved. The
# structural numbers moved because the *graph* did — 640 ops became 549 — so the
# previous timings described a different network and were not comparable to these
# whatever the conditions. The timings are ~2.5x the previous ones on top of that,
# in a session with a model, two plans and a benchmark loaded on a shared card;
# the arms are interleaved and warmed, so they compare to each other and to
# nothing else.
#
# **The memory is the win and the barriers are not.** Bit-exact both graphs —
# baked and unbaked — and the encoder's peak is 31% lower. Most of that is one
# thing: `planslab` reserves a slot for every element of a multi-output op that
# the exported graph declares a shape for, including elements no handler ever
# writes (the indices of `max_pool2d_with_indices`, the logsumexp of the flash
# attention). Declaring from what `dest` was actually asked for cannot reserve
# them, which is the second reason to discover rather than derive — and it is
# visible in the counts above, 890 slab slots against 791 placed. The rest is
# tighter liveness: an element of a tuple gets its own interval instead of the
# whole tuple's, and a `contiguous` copy is constrained against the buffer it
# reads rather than given that buffer's entire lifetime.
#
# The split between those two was measured on the previous export as 133 MB of a
# 261 MB gap. That attribution is NOT carried forward: the gap is now 81.25 MB on
# a graph with 91 fewer ops, and nobody has re-derived which part is which. The
# 99 unwritten slots are counted; how many bytes they are is not.
#
# The three timings are the same to within 1%, which is the third measurement in
# a row saying that scoping a barrier to a buffer does not make a frame faster on
# this hardware — see `bench/independent.jl`, where the same answer came back for
# a synthetic graph of independent chains and for the showcase. An inference graph
# is close to a linear chain, so there is almost no independence to find in the
# first place: 548 of 549 passes need a barrier. What the derivation buys is that
# a dependency the graph missed is not silently supplied by a barrier that
# happened to be wide enough. Which is also what made this port hard, and is the
# point: two under-declarations here came back as NaN and as a wrong tensor —
# `storageids!`'s two paragraphs on multi-output elements and on the copy's read
# of its parent — where under a global barrier both would have run correctly and
# said nothing.
#
# Absolute times here are ~2.4x the figures in `SAM2Runner`, on a device shared
# with a desktop session and in a process whose precompile workload was skipped.
# The arms are interleaved and warmed, so the comparison between them holds; the
# absolute numbers are not a claim about how fast SAM 2 is.
#
# ── where the implementation lives ───────────────────────────────────────────
#
# **In `DNNKernels/ext/DNNKernelsMantleExt.jl`, not here.** This file prototyped
# it — `MantlePlan`, `Discovery`, `storageids!`, `discover`, `build` — and then
# kept its own copy after the extension took it, which is a copy that drifts: the
# extension gained a declarations cache, `bakeplan!`, and a fix for discovery
# creating its transients in the graph the plan was being built into. The copy
# here had none of them and still carried that bug.
#
# So this is now what its name says: the benchmark. It measures the extension.
# Anything that reads or writes a graph belongs on the DNNKernels side, where the
# arrow points the right way — Mantle knows nothing about ATen graphs.

import Mantle
using Lava, DNNKernels, KernelAbstractions
using DNNKernels: fusableset, planslab

const M = Mantle
const D = DNNKernels
const KA = KernelAbstractions
# Was `Base.get_extension(DNNKernels, :DNNKernelsMantleExt)`. Mantle is a
# dependency of DNNKernels now, not a weak one, so the bridge is `src/mantle.jl`
# and there is no extension to fetch or to be missing.
const X = DNNKernels

# ── what the plan came out as ─────────────────────────────────────────────────

"""Passes that need a barrier, and how many transitions those barriers carry.

Here rather than in the extension because it reaches into `PassPlan.pre`, which
is Mantle's compile state and not something a runner should depend on. A bench
may; it is the thing being measured."""
barrierstats(plan) = (passes = length(plan.passes),
                      with_barrier = count(pp -> !isempty(pp.pre), plan.passes),
                      transitions = sum(pp -> length(pp.pre), plan.passes; init = 0))

# ── the benchmark ─────────────────────────────────────────────────────────────

"""
    compare(fs...; iters = 20) -> Vector

Median of `iters` rounds of each, interleaved and all warmed first. Interleaved
because a first timed loop runs measurably slow on this setup, so running one arm
to completion and then the other measures the order as much as the arms.
"""
function compare(fs...; iters::Int = 20)
    foreach(f -> f(), fs)
    ts = [Float64[] for _ in fs]
    for _ in 1:iters, (k, f) in enumerate(fs)
        t = time(); f(); push!(ts[k], time() - t)
    end
    [round(1000 * sort(t)[iters ÷ 2 + 1], digits = 3) for t in ts]
end

"""
    benchgraph(model, args...; name = "sam2_encoder", bake = false, …) -> NamedTuple

One graph of a `SAM2` under Mantle, against the same graph under DNNKernels' own
slab and Lava's automatic barrier.

Reports what each side spends and what each side derived: peak bytes placed,
barriers emitted, and whether the two produce the same numbers. The last one is
not a formality — a hazard nobody declared shows up as a wrong tensor, and fast
and wrong is the only outcome worth guarding against here.

`bake = true` adds a fourth arm: the plan recorded once and replayed, which is
where the host time goes rather than the GPU time. Its correctness is checked
against the same reference as the others, because a replay that has gone stale
returns a plausible answer rather than an error.
"""
function benchgraph(model, args...; iters::Int = 20, name::AbstractString = "sam2_encoder",
                    policy = M.Overlap(), alias::Bool = true, coalesce::Bool = true,
                    bake::Bool = false)
    m = model.model                        # the DNNKernels `Model` behind the SAM2
    graph = m.graphs[name]
    dims = model.dims
    dev = M.Device(M.VulkanAPI())
    backend = m.backend
    # The decoder's attentions are 23 tokens and want the padded cooperative-matrix
    # path; the encoder has six that would go along at 50% waste. `sam2.jl` measures
    # both, and this has to make the same choice or it is timing a different graph.
    clampattn = name == "sam2_decoder"

    inputs = Dict{String,Any}(zip(graph.inputs, args))
    gp = X.build(dev, graph, inputs, m.weights; dims, clampattn, policy, alias, coalesce)

    # Correctness first, and against the arm that is not being changed.
    want = D.call(m, name, args...; dims, clampattn)
    got = X.runplan(gp)
    KA.synchronize(backend)
    diffs = [maximum(abs.(Float32.(Array(a)) .- Float32.(Array(b)))) for (a, b) in zip(want, got)]

    dnn() = (D.call(m, name, args...; dims, clampattn); KA.synchronize(backend))
    derived() = (X.runplan(gp); KA.synchronize(backend))
    backendbar() = (X.runplan(gp; barriers = :backend); KA.synchronize(backend))
    dnn_ms, derived_ms, backend_ms = compare(dnn, derived, backendbar; iters)

    baked_ms, baked_diff = nothing, nothing
    if bake
        X.bakeplan!(gp)
        KA.synchronize(backend)
        gotb = X.runplan(gp)
        KA.synchronize(backend)
        baked_diff = maximum((maximum(abs.(Float32.(Array(a)) .- Float32.(Array(b))))
                              for (a, b) in zip(want, gotb)); init = 0f0)
        baked_ms = only(compare(derived; iters))
    end

    slab = planslab(graph, dims)
    (; name,
     ops = length(graph.ops),
     placed_mantle = length(gp.res),
     placed_slab = length(slab.offsets),
     peak_mantle_MB = round(M.peakbytes(gp.plan) / 2^20, digits = 2),
     peak_slab_MB = round(slab.bytes / 2^20, digits = 2),
     barrierstats(gp.plan)...,
     dnn_ms, derived_ms, backend_ms, baked_ms,
     maxdiff = maximum(diffs; init = 0f0),
     baked_maxdiff = baked_diff,
     unplaced = X.misses(gp),
     shortlived = count(x -> !occursin(r"\.\d+$", x[1]), X.shortlived(gp)))
end
