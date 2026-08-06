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
#     a custom body for exactly this reason; see `Lava.exclusive_dispatch_group`.
#
# ── measured, SAM 2.1 large, RADV STRIX_HALO ─────────────────────────────────
#
#                              encoder            decoder
#   ops / passes               640                149
#   buffers placed             882  (slab 981)    162  (slab 175)
#   peak                       180.0 MB           16.02 MB
#     the same by planslab     261.25             16.08
#   passes needing a barrier   639 / 640          139 / 149
#   transitions emitted        2396               446
#   DNNKernels + Lava          293.9 ms           15.78 ms
#   Mantle, derived barriers   292.4              15.75
#   Mantle, backend barriers   290.9              15.70
#   difference in the result   0                  0
#
# **The memory is the win and the barriers are not.** Bit-exact both graphs, and
# the encoder's peak is 31% lower — but 133 MB of that is one thing: `planslab`
# reserves a slot for every element of a multi-output op that the exported graph
# declares a shape for, and 102 of them are elements no handler ever writes (the
# indices of `max_pool2d_with_indices`, the logsumexp of the flash attention).
# Declaring from what `dest` was actually asked for cannot reserve them, which is
# the second reason to discover rather than derive. The rest is tighter liveness:
# an element of a tuple gets its own interval instead of the whole tuple's, and a
# `contiguous` copy is constrained against the buffer it reads rather than given
# that buffer's entire lifetime.
#
# The three timings are the same to within 1%, which is the third measurement in
# a row saying that scoping a barrier to a buffer does not make a frame faster on
# this hardware — see `bench/independent.jl`, where the same answer came back for
# a synthetic graph of independent chains and for the showcase. An inference graph
# is close to a linear chain, so there is almost no independence to find in the
# first place: 639 of 640 passes need a barrier. What the derivation buys is that
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

import Mantle
using Lava, DNNKernels, KernelAbstractions
using DNNKernels: Ctx, Diagnostics, Graph, Op, Slab, Workspace, alignup, coerce,
                  escaping, evalexpr, fusableset, place, planslab, reset!, slabview,
                  timeop!, value

const M = Mantle
const D = DNNKernels
const KA = KernelAbstractions

# ── the two plans `dest` can be given ─────────────────────────────────────────

"""
What `dest` hands out once Mantle has placed the graph.

One entry per buffer id: the array the placer gave that resource, which for a
transient is a window into the arena and for a persistent buffer is the buffer
itself. Filled in after `Plan`, because a transient has no storage before then.
"""
struct MantlePlan
    views::Dict{String,Any}
    bytes::Dict{String,Int}
end

MantlePlan() = MantlePlan(Dict{String,Any}(), Dict{String,Int}())

function D.place(p::MantlePlan, slab, id::AbstractString, ::Type{T}, dims) where {T}
    v = get(p.views, id, nothing)
    v === nothing && return nothing
    # The same rule the slab planner applies: the reservation is what the
    # discovery run asked for, and a larger request would overrun into whatever
    # Mantle aliased next door.
    prod(dims) * sizeof(T) <= p.bytes[id] || return nothing
    slabview(T, v, dims, 0)
end

"""
The plan the discovery run carries.

It serves `dest` out of DNNKernels' ordinary slab — so the run is the one that
would have happened anyway, with correct values for the next op to be discovered
against — and records every id it is asked for on the way. `writes` is emptied by
the caller before each op, so what is left in it afterwards is that op's writes.

Creating the Mantle resource here rather than later is what makes the record
exact: the id, its size and the decision transient-or-persistent are all settled
at the moment the request is made.
"""
struct Discovery
    mgraph::Any                     # the Mantle graph the transients belong to
    dev::Any
    esc::Set{String}
    res::Dict{String,Any}           # buffer id -> the Mantle resource holding it
    bytes::Dict{String,Int}
    writes::Vector{String}
    slab::Slab                      # DNNKernels' own plan, to serve the run
    store::Any
end

function D.place(d::Discovery, slab, id::AbstractString, ::Type{T}, dims) where {T}
    n = alignup(prod(dims) * sizeof(T))
    have = get(d.bytes, id, 0)
    if have == 0
        # An escaping buffer is read after the plan has run — by the next graph,
        # or by the caller reading an output — so it cannot be a transient whose
        # bytes Mantle is free to hand to something else at its last use.
        d.res[id] = id in d.esc ? M.Buffer(d.dev, UInt8, n) :
                                  M.Transient.Buffer(d.mgraph, UInt8, n)
        d.bytes[id] = n
    elseif n > have
        error("buffer $id was asked for $have bytes and then $n: the op sequence " *
              "is not static, so one discovery run cannot describe it")
    end
    push!(d.writes, id)
    place(d.slab, d.store, id, T, dims)
end

# ── what a read touches ───────────────────────────────────────────────────────

"""
    storageids!(out, graph, res, byout, id) -> out

The ids whose storage a read of `id` actually lands on.

A read does not always land where it is spelled. A value the fuser left lazy has
no storage at all — `emit` returned the `Broadcasted` and never called `dest` —
so reading it reads its operands, recursively. A view `makeview` resolves lazily
(a reshape, a slice, a `PermutedDimsArray`) reads its parent's storage.

Anything with an entry in `res` answers for itself and stops the walk, which is
what makes a view that *was* materialised come out as itself rather than as the
buffer it was copied from.

The one id that is not spelled anywhere in the graph is an element of a
multi-output op: `native_layer_norm` names a tuple with no single shape, so the
planner reserves `"native_layer_norm.0"` and the graph reaches it through a
`getitem` view of the tuple. Walking past that to the tuple, and then to the op
that produced it, declares a read of the op's *inputs* — which is a live read of
memory that has already been handed to something else, and the reason SAM 2's
first attempt came out as NaN.
"""
function storageids!(out, g::Graph, res, byout, id::AbstractString, depth::Int = 0)
    depth > 64 && return out
    haskey(res, id) && (push!(out, id); return out)
    b = get(g.buffers, id, nothing)
    b === nothing && return out
    if !isempty(b.of)
        if occursin("getitem", b.viewop)
            k = string(b.of, '.', Int(b.attrs["arg1"]))
            haskey(res, k) && (push!(out, k); return out)
        end
        return storageids!(out, g, res, byout, b.of, depth + 1)
    end
    o = get(byout, id, nothing)
    o === nothing && return out         # a weight, a graph input, a host constant
    for i in o.ins
        storageids!(out, g, res, byout, i, depth + 1)
    end
    out
end

# ── discovery ─────────────────────────────────────────────────────────────────

"""What one op turned out to touch."""
struct OpUse
    reads::Vector{String}
    writes::Vector{String}
    scratch::Bool
end

"""
    discover(graph, ctx, disc, byout) -> Vector{OpUse}

Run the graph once, recording per op what it read and what it wrote.

A separate phase from building the passes, rather than one interleaved with them,
because the workspace resource is one declaration whose *size* is only known once
the whole run has grown it. Interleaving would mean either declaring it before it
exists or reaching back into passes already built.
"""
function discover(graph::Graph, ctx::Ctx, disc::Discovery, byout)
    map(graph.ops) do op
        empty!(disc.writes)
        ctx.outid[] = op.out
        reset!(ctx.ws)
        ctx.values[op.out] = coerce(timeop!(ctx, op), graph.buffers[op.out])
        # Reads after the run, not before it. An op that reads a view of a
        # permuted parent makes the copy itself, on the way in — so at the moment
        # it started, the buffer it is about to read had no storage to name, and
        # the walk would have gone past it to the parent. 51 of SAM 2's encoder
        # buffers are that copy.
        reads = Set{String}()
        for inp in op.ins
            storageids!(reads, graph, disc.res, byout, inp)
        end
        # Making that copy is itself a read, of the buffer being copied *from*,
        # and the walk above cannot see it: it stops at the copy, which is the
        # right answer for every later op and the wrong one for this one. Left
        # out, the parent's interval ends at the op that produced it, the placer
        # hands its bytes on, and `permutedims!` copies whatever landed there.
        for id in disc.writes
            b = get(graph.buffers, id, nothing)
            (b === nothing || isempty(b.of)) && continue
            storageids!(reads, graph, disc.res, byout, b.of)
        end
        OpUse(sort!(collect(reads)), sort!(unique(disc.writes)), ctx.ws.used > 0)
    end
end

# ── building the graph ────────────────────────────────────────────────────────

"""What a built graph needs to be run and read."""
struct GraphPlan
    graph::Graph
    mgraph::Any
    plan::Any
    ctx::Ctx
    res::Dict{String,Any}
    esc::Set{String}
    uses::Vector{OpUse}
    viewids::Vector{String}
    scratch::Any                    # the workspace resource, or nothing
end

"""Seed the value table the way `execute!` does: weights, inputs, host constants."""
function seed!(ctx::Ctx, graph::Graph, inputs, weights)
    for id in graph.order
        b = graph.buffers[id]
        if b.kind === :weight
            haskey(weights, b.key) || error("missing weight $(b.key)")
            ctx.values[id] = weights[b.key]
        elseif b.kind === :external && haskey(inputs, id)
            ctx.values[id] = inputs[id]
        elseif b.kind === :host
            ctx.values[id] = evalexpr(String(b.attrs["expr"]), ctx.dims)
        end
    end
    ctx
end

"""
    build(dev, graph, inputs, weights; dims, …) -> GraphPlan

One Mantle pass per op, declared from a discovery run and compiled.

Two contexts. The discovery one runs the graph now, out of DNNKernels' own slab,
so that `dest` can be asked what each op wants; the record one runs it again
inside the plan, out of what Mantle placed. They share nothing but the graph and
the weights, because the discovery run leaves materialised views cached in its
value table and those point at slab storage the plan does not own.
"""
function build(dev, graph::Graph, inputs::AbstractDict, weights::AbstractDict;
               dims, clampattn::Bool = false, alias::Bool = true,
               coalesce::Bool = true, policy = M.Overlap())
    mg = M.Graph(dev)
    backend = M.backend(dev)
    lazy = fusableset(graph)
    byout = Dict(o.out => o for o in graph.ops)

    dslab = planslab(graph, dims)
    dstore = KA.allocate(backend, UInt8, max(dslab.bytes, 1))
    disc = Discovery(mg, dev, escaping(graph), Dict{String,Any}(), Dict{String,Int}(),
                     String[], dslab, dstore)
    ctxd = Ctx(Dict{String,Any}(), graph, dims, backend;
               slab = dstore, plan = disc, ws = Workspace(backend), lazy, clampattn)
    seed!(ctxd, graph, inputs, weights)
    uses = discover(graph, ctxd, disc, byout)

    mplan = MantlePlan()
    ctxr = Ctx(Dict{String,Any}(), graph, dims, backend;
               plan = mplan, ws = Workspace(backend), lazy, clampattn,
               diag = Diagnostics(; planmisses = Dict{String,Tuple{Int,Int}}()))
    seed!(ctxr, graph, inputs, weights)

    # One resource for the kernel workspace, pre-sized to the high-water mark the
    # discovery run reached so the record run's `scratch!` never grows it. A grow
    # would swap the buffer the declared barriers name for a different one.
    scratch = ctxd.ws.buf === nothing ? nothing : M.Buffer(dev, UInt8, length(ctxd.ws.buf))
    scratch === nothing || (ctxr.ws.buf = M.storage(scratch))

    for (op, u) in zip(graph.ops, uses)
        M.custom!(mg, op.aten) do p
            # One `use` per resource, not one per role: two of them on the same
            # resource in one pass is two entries in the usage sequence, and two
            # transitions derived where the pass has one state.
            for id in union(u.reads, u.writes)
                M.use(p, disc.res[id]; read = id in u.reads, write = id in u.writes)
            end
            u.scratch && M.use(p, scratch; read = true, write = true)
            return () -> runrecorded!(ctxr, graph, op)
        end
    end

    plan = M.Plan(mg; alias, coalesce, policy)
    pos = IdDict(p => i for (i, p) in enumerate(mg.passes))
    issorted([pos[pp.pass] for pp in plan.passes]) || error(
        "the scheduler reordered the passes. A custom body is host code reading a " *
        "value table its predecessors filled, so it cannot run out of declaration " *
        "order; use policy = Overlap(), where declaration order dominates the score.")

    for (id, r) in disc.res
        mplan.views[id] = M.storage(r)
        mplan.bytes[id] = disc.bytes[id]
    end
    GraphPlan(graph, mg, plan, ctxr, disc.res, disc.esc, uses,
              [id for (id, b) in graph.buffers if b.kind === :view], scratch)
end

"""One op inside its pass: the three lines `execute!` runs, against Mantle's storage."""
function runrecorded!(ctx::Ctx, graph::Graph, op::Op)
    ctx.outid[] = op.out
    reset!(ctx.ws)
    ctx.values[op.out] = coerce(timeop!(ctx, op), graph.buffers[op.out])
    nothing
end

"""
    runplan(gp) -> outputs

Run the plan and resolve the graph's declared outputs.

The cached views go first. `makeview` memoises what it resolves, which is right
within one run and wrong across two: a view that is merely lazy still points at
the parent's live bytes and would be fine, but one `contiguous` had to *copy*
holds the previous run's numbers and nothing would refresh it.
"""
function runplan(gp::GraphPlan; barriers::Symbol = :derived)
    for id in gp.viewids
        delete!(gp.ctx.values, id)
    end
    M.run!(gp.plan; barriers)
    Tuple(value(gp.ctx, o) for o in gp.graph.outputs)
end

# ── what the plan came out as ─────────────────────────────────────────────────

"""Passes that need a barrier, and how many transitions those barriers carry."""
barrierstats(plan) = (passes = length(plan.passes),
                      with_barrier = count(pp -> !isempty(pp.pre), plan.passes),
                      transitions = sum(pp -> length(pp.pre), plan.passes; init = 0))

"""Ids `dest` could not place from Mantle's plan — expected to be empty."""
misses(gp::GraphPlan) = D.planmisses(gp.ctx.diag)

"""
    unread(gp) -> Vector{String}

Placed transients no pass declares a read of.

A dead store is possible in principle; in a graph that has been through
`dropdead` it is usually a symptom. The reader exists and was attributed to some
*other* id, so this buffer's interval ends at its write, the placer hands its
bytes to the next thing, and the read lands on them.

Cheaper to ask than to diff 640 intermediates, which cannot be done at all once
the plan has run: the intermediates live in aliased memory and hold whatever took
it over. It is what named the 51 copies `contiguous` forces on SAM 2's encoder,
whose only reader is the op that makes them — declared as writes and not as
reads, because the walk ran before the op did.

What is left on this graph is the mean and the reciprocal standard deviation of
every `native_layer_norm`. Nothing reads those, and both planners reserve them.
"""
unread(gp::GraphPlan) =
    let r = Set{String}()
        for u in gp.uses, id in u.reads
            push!(r, id)
        end
        sort!([id for id in keys(gp.res) if !(id in r) && !(id in gp.esc)])
    end

"""
    shortlived(gp) -> Vector{(id, mantle, dnn)}

Placed buffers whose Mantle interval ends before DNNKernels' own planner says
they are dead.

The two derive lifetimes differently — `planslab` walks the graph and the fusion
chains, Mantle counts which pass touched what — so they are an independent check
on each other, and the direction that matters is one-sided: an interval that ends
too *early* is memory handed to something else while it is still being read.

Not every entry is a fault, in either of two ways. `lifetimes` gives a
multi-output op one interval for the whole tuple, so each element inherits the
longest element's, while Mantle gives each its own. And a `contiguous` copy is
given its *root's* lifetime by `planslab`, deliberately, so that it can never be
placed on top of the buffer it is a copy of; declaring the read that copy makes
says the same thing exactly, and leaves the copy free for the rest of the root's
life. On SAM 2's decoder that is the one entry left after filtering the tuples.

What it is for is the third case. It is how the `contiguous` copy's read of the
buffer it copies *from* was found — not declared at all, so the parent's interval
ended at the op that produced it and `permutedims!` copied whatever the placer
had put there since.
"""
function shortlived(gp::GraphPlan)
    ext = Base.get_extension(Mantle, :MantleLavaExt)
    lt = D.lifetimes(gp.graph, fusableset(gp.graph))
    out = Tuple{String,Tuple{Int,Int},Tuple{Int,Int}}[]
    for (id, r) in gp.res
        r isa ext.TransientBuffer || continue
        i = findlast('.', id)
        k = i !== nothing && haskey(lt, id[1:(i - 1)]) ? id[1:(i - 1)] :
            haskey(lt, id) ? id : D.rootbuffer(gp.graph, id)
        haskey(lt, k) || continue
        r.last < lt[k][2] && push!(out, (id, (r.first, r.last), lt[k]))
    end
    sort!(out, by = x -> x[2][2] - x[3][2])
end

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
    benchgraph(model, args...; name = "sam2_encoder", …) -> NamedTuple

One graph of a `SAM2` under Mantle, against the same graph under DNNKernels' own
slab and Lava's automatic barrier.

Reports what each side spends and what each side derived: peak bytes placed,
barriers emitted, and whether the two produce the same numbers. The last one is
not a formality — a hazard nobody declared shows up as a wrong tensor, and fast
and wrong is the only outcome worth guarding against here.
"""
function benchgraph(model, args...; iters::Int = 20, name::AbstractString = "sam2_encoder",
                    policy = M.Overlap(), alias::Bool = true, coalesce::Bool = true)
    m = model.model                        # the DNNKernels `Model` behind the SAM2
    graph = m.graphs[name]
    dims = model.dims
    dev = M.Device(Lava)
    backend = m.backend
    # The decoder's attentions are 23 tokens and want the padded cooperative-matrix
    # path; the encoder has six that would go along at 50% waste. `sam2.jl` measures
    # both, and this has to make the same choice or it is timing a different graph.
    clampattn = name == "sam2_decoder"

    inputs = Dict{String,Any}(zip(graph.inputs, args))
    gp = build(dev, graph, inputs, m.weights; dims, clampattn, policy, alias, coalesce)

    # Correctness first, and against the arm that is not being changed.
    want = D.call(m, name, args...; dims, clampattn)
    got = runplan(gp)
    KA.synchronize(backend)
    diffs = [maximum(abs.(Float32.(Array(a)) .- Float32.(Array(b)))) for (a, b) in zip(want, got)]

    dnn() = (D.call(m, name, args...; dims, clampattn); KA.synchronize(backend))
    derived() = (runplan(gp); KA.synchronize(backend))
    backendbar() = (runplan(gp; barriers = :backend); KA.synchronize(backend))
    dnn_ms, derived_ms, backend_ms = compare(dnn, derived, backendbar; iters)

    slab = planslab(graph, dims)
    (; name,
     ops = length(graph.ops),
     placed_mantle = length(gp.res),
     placed_slab = length(slab.offsets),
     peak_mantle_MB = round(M.peakbytes(gp.plan) / 2^20, digits = 2),
     peak_slab_MB = round(slab.bytes / 2^20, digits = 2),
     barrierstats(gp.plan)...,
     dnn_ms, derived_ms, backend_ms,
     maxdiff = maximum(diffs; init = 0f0),
     unplaced = misses(gp),
     shortlived = count(x -> !occursin(r"\.\d+$", x[1]), shortlived(gp)))
end
