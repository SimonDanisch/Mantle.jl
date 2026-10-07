# Fuzz and steady-state gates: GT-01..GT-03 (catalogue-resize-rt.md, table R)
# and THR-07, HOLD-10 (catalogue-core.md, tables 13 and 17). Tasks on several
# threads change arrays, graphs and composites while graphs run; every array
# has a host model and is compared exactly at the end; memory is back at its
# baseline after everything is freed. All cases are `:long`: gates run at the
# end of a phase.
#
# Results stay exact under any interleaving: each task changes only its own
# arrays, in its program order, and arrays shared between tasks only receive
# additions, which commute, so a shared counter ends at the sum of every
# addition whatever the order. Every graph also checks that a shared counter
# another graph writes is uniform (all elements equal): a read that overlapped
# a write would see a mix and count it in its `bad` array.
#
# Not covered here: window resizes (THR-07 lists them; they need a display and
# are in the windows area), plot updates (GT-01, phase 11, RayMakie's), and
# "validation clean" (GT-01), which needs the validation layers on in the
# suite's device configuration and no hook reports it.
#
# Worker tasks call no @test (Test keeps its test set in task-local storage);
# they only act and update models, and a throw reaches the case through
# `wait` under the watchdog.

using KernelAbstractions: @kernel, @index
using GeometryBasics: Mat4f, Point3f, TriangleFace

const FZ_COUNTERS = 4           # shared counters
const FZ_COUNTERLENGTH = 1 << 16
const FZ_OWNMAX = 4096          # largest length of a task's own array
const FZ_TIMEOUT = 3600         # seconds for a whole fuzz

@kernel function fz_add!(a, k)
    i = @index(Global)
    @inbounds a[i] += k
end

@kernel function fz_uniform!(bad, s)
    i = @index(Global)
    @inbounds bad[i] += s[i] == s[1] ? Int32(0) : Int32(1)
end

# One work item: names four arrays.
@kernel function fz_touch4!(a, b, c, d)
    @inbounds a[1] += Int32(1)
    @inbounds b[1] += Int32(1)
    @inbounds c[1] += Int32(1)
    @inbounds d[1] += Int32(1)
end

@kernel function fz_frametick!(out, cam)
    i = @index(Global)
    @inbounds out[i] = cam[1][1, 1] * Float32(i)
end

"""An instance transform moved by `v`."""
movedby(v) = Mat4f(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, v[1], v[2], v[3], 1)

"""A uniform scale by `s`."""
scaledby(s) = Mat4f(s, 0, 0, 0, 0, s, 0, 0, 0, 0, s, 0, 0, 0, 0, 1)

"""A BLAS of one triangle, with the arrays it holds."""
function fuzzblas(dev)
    vertices = MantleArray(dev, Point3f[(0, 0, 0), (1, 0, 0), (0, 1, 0)])
    faces = MantleArray(dev, [TriangleFace{UInt32}(1, 2, 3)])
    return BLAS(dev, vertices, faces), vertices, faces
end

"""Memory figures after a case equal those before it (pool bytes may only have gone down)."""
function atbaseline(before, after)
    @test after.uploads == before.uploads
    @test after.cells == before.cells
    @test after.reserved <= before.reserved
    @test after.mapped <= before.mapped
end

# ── The threaded fuzz (THR-07, GT-01) ──

"""What every task may touch: shared counters, graphs any task runs, and (with ray tracing) a TLAS."""
struct FuzzShared
    dev::Any
    counters::Vector{Any}
    sums::Vector{Threads.Atomic{Int}}     # every addition to each counter
    graphs::Vector{Any}                   # graph j adds 1000 to counter j and checks counter j + 1
    bads::Vector{Any}
    tlas::Any
    blas::Any
    geometry::Vector{Any}
    tlasgraphs::Vector{Any}               # graphs that refit or build the TLAS every run
end

function FuzzShared(dev; rt::Bool)
    counters = Any[MantleArray(dev, zeros(Int32, FZ_COUNTERLENGTH)) for _ in 1:FZ_COUNTERS]
    sums = [Threads.Atomic{Int}(0) for _ in 1:FZ_COUNTERS]
    graphs, bads = Any[], Any[]
    for j in 1:2
        g = Graph(dev)
        bad = MantleArray(dev, zeros(Int32, FZ_COUNTERLENGTH))
        dispatch!(g, fz_add!, (counters[j], Int32(1000)), FZ_COUNTERLENGTH)
        dispatch!(g, fz_uniform!, (bad, counters[j + 1]), FZ_COUNTERLENGTH)
        push!(graphs, g)
        push!(bads, bad)
    end
    rt || return FuzzShared(dev, counters, sums, graphs, bads, nothing, nothing, Any[], Any[])
    blas, vertices, faces = fuzzblas(dev)
    tlas = TLAS(dev)
    tlasgraphs = Any[Graph(dev) for _ in 1:3]
    refit!(tlasgraphs[1], tlas)
    build!(tlasgraphs[2], tlas)
    refit!(tlasgraphs[3], tlas)
    return FuzzShared(dev, counters, sums, graphs, bads, tlas, blas, Any[vertices, faces], tlasgraphs)
end

"""What one task owns, with the host model of each of its arrays."""
mutable struct FuzzWorld
    rng::Xoshiro
    own::Any                    # resized, stored into and added to
    ownmodel::Vector{Int32}
    a::Any                      # the two targets of `ref`
    b::Any
    amodel::Vector{Int32}
    bmodel::Vector{Int32}
    ref::Any                    # array GPURef: the task's graph adds 1 to its target every run
    ona::Bool                   # whether `ref` points at `a`
    bad::Any                    # the task's graph's count of non-uniform reads
    graph::Any
    writes::Int                 # the counter the task's graph adds to
    add::Int32                  # what it adds per run
    block::Any                  # a node block inserted into `graph`, or nothing
    extra::Int32                # what that block adds to `own` per run
    ranges::Vector{Any}         # (range, source) pairs pushed into the shared TLAS
end

function FuzzWorld(dev, shared, k)
    rng = Xoshiro(k)
    ownmodel = rand(rng, Int32(-100):Int32(100), 256)
    amodel, bmodel = rand(rng, Int32(-100):Int32(100), 300), rand(rng, Int32(-100):Int32(100), 700)
    own, a, b = MantleArray(dev, ownmodel), MantleArray(dev, amodel), MantleArray(dev, bmodel)
    ref = GPURef(dev, a)
    bad = MantleArray(dev, zeros(Int32, FZ_COUNTERLENGTH))
    writes, reads, add = mod1(k, FZ_COUNTERS), mod1(k + 1, FZ_COUNTERS), Int32(k)
    graph = Graph(dev)
    dispatch!(graph, fz_add!, (own, Int32(1)), launchrange(own))
    dispatch!(graph, fz_add!, (ref, Int32(1)), launchrange(ref))
    dispatch!(graph, fz_add!, (shared.counters[writes], add), FZ_COUNTERLENGTH)
    dispatch!(graph, fz_uniform!, (bad, shared.counters[reads]), FZ_COUNTERLENGTH)
    return FuzzWorld(rng, own, ownmodel, a, b, amodel, bmodel, ref, true, bad, graph, writes, add,
                     nothing, Int32(0), Any[])
end

targetmodel(w) = w.ona ? w.amodel : w.bmodel
newvalues(w, n) = rand(w.rng, Int32(-100):Int32(100), n)

# The operations. Each method acts on the device and updates the host model
# in the same program order.
struct OpRun end            # the task's own graph
struct OpRunShared end      # a graph any task runs
struct OpStore end          # a store into a range of its array
struct OpCopy end           # copy!: a new length, old contents dropped
struct OpGrow end           # resize! past the length, then a store of the tail
struct OpShrink end         # resize! below the length
struct OpEager end          # an eager broadcast on its array
struct OpEagerShared end    # an eager broadcast on a shared counter
struct OpRebind end         # its array GPURef to the other target
struct OpTempArray end      # a temporary array used eagerly, then freed or dropped
struct OpTempGraph end      # a temporary graph run once, then freed or dropped
struct OpCollect end        # a garbage collection
struct OpInsert end         # a node block inserted into its graph
struct OpDelete end         # that block deleted
struct OpPushRange end      # a TLAS range over a new source
struct OpDeleteRange end    # one of its ranges deleted
struct OpStoreRange end     # a store into a range source: an Update
struct OpResizeRange end    # copy! of a range source: a new count, a Build
struct OpRunTLAS end        # a graph that refits or builds the TLAS

function fuzzstep!(w, s, ::OpRun)
    run!(w.graph)
    w.ownmodel .+= Int32(1) + w.extra
    targetmodel(w) .+= Int32(1)
    Threads.atomic_add!(s.sums[w.writes], Int(w.add))
end
function fuzzstep!(w, s, ::OpRunShared)
    j = rand(w.rng, eachindex(s.graphs))
    run!(s.graphs[j])
    Threads.atomic_add!(s.sums[j], 1000)
end
function fuzzstep!(w, s, ::OpStore)
    n = length(w.ownmodel)
    n == 0 && return
    lo = rand(w.rng, 1:n)
    hi = rand(w.rng, lo:n)
    data = newvalues(w, hi - lo + 1)
    w.own[lo:hi] = data
    w.ownmodel[lo:hi] = data
end
function fuzzstep!(w, s, ::OpCopy)
    data = newvalues(w, rand(w.rng, 0:FZ_OWNMAX))
    copy!(w.own, data)
    w.ownmodel = copy(data)
end
function fuzzstep!(w, s, ::OpGrow)
    n = length(w.ownmodel)
    n >= FZ_OWNMAX && return fuzzstep!(w, s, OpShrink())
    m = rand(w.rng, n+1:FZ_OWNMAX)
    tail = newvalues(w, m - n)
    resize!(w.own, m)
    w.own[n+1:m] = tail
    append!(w.ownmodel, tail)
end
function fuzzstep!(w, s, ::OpShrink)
    m = rand(w.rng, 0:length(w.ownmodel))
    resize!(w.own, m)
    resize!(w.ownmodel, m)
end
function fuzzstep!(w, s, ::OpEager)
    w.own .+= Int32(2)
    w.ownmodel .+= Int32(2)
end
function fuzzstep!(w, s, ::OpEagerShared)
    j = rand(w.rng, eachindex(s.counters))
    s.counters[j] .+= Int32(3)
    Threads.atomic_add!(s.sums[j], 3)
end
function fuzzstep!(w, s, ::OpRebind)
    w.ona = !w.ona
    w.ref[] = w.ona ? w.a : w.b
end
function fuzzstep!(w, s, ::OpTempArray)
    t = MantleArray(s.dev, Int32, length(w.ownmodel))
    fill!(t, Int32(5))
    w.own .+= t
    w.ownmodel .+= Int32(5)
    rand(w.rng, Bool) && free!(t)        # else dropped: its finalizer releases it
end
function fuzzstep!(w, s, ::OpTempGraph)
    g = Graph(s.dev)
    dispatch!(g, fz_add!, (w.own, Int32(7)), launchrange(w.own))
    run!(g)
    w.ownmodel .+= Int32(7)
    rand(w.rng, Bool) && free!(g)        # else dropped while its run may be in flight
end
fuzzstep!(w, s, ::OpCollect) = GC.gc(false)
function fuzzstep!(w, s, ::OpInsert)
    w.block === nothing || return
    w.block = insert!(w.graph) do
        dispatch!(w.graph, fz_add!, (w.own, Int32(10)), launchrange(w.own))
    end
    w.extra = Int32(10)
end
function fuzzstep!(w, s, ::OpDelete)
    w.block === nothing && return
    delete!(w.graph, w.block)
    w.block = nothing
    w.extra = Int32(0)
end
function fuzzstep!(w, s, ::OpPushRange)
    src = MantleArray(s.dev, [movedby(rand(w.rng, Float32, 3)) for _ in 1:rand(w.rng, 1:8)])
    push!(w.ranges, (push!(s.tlas, s.blas, src), src))
end
function fuzzstep!(w, s, ::OpDeleteRange)
    isempty(w.ranges) && return
    r, src = popat!(w.ranges, rand(w.rng, eachindex(w.ranges)))
    delete!(s.tlas, r)
    rand(w.rng, Bool) && free!(src)      # the TLAS keeps its hold until the build that deactivated the range has run
end
function fuzzstep!(w, s, ::OpStoreRange)
    isempty(w.ranges) && return
    _, src = rand(w.rng, w.ranges)
    src[1:1] = [movedby(rand(w.rng, Float32, 3))]
end
function fuzzstep!(w, s, ::OpResizeRange)
    isempty(w.ranges) && return
    _, src = rand(w.rng, w.ranges)
    copy!(src, [movedby(rand(w.rng, Float32, 3)) for _ in 1:rand(w.rng, 1:16)])
end
fuzzstep!(w, s, ::OpRunTLAS) = run!(rand(w.rng, s.tlasgraphs))

const FZ_CHANGEOPS = (OpRun(), OpRun(), OpRun(), OpRunShared(), OpStore(), OpStore(), OpCopy(), OpGrow(),
                      OpShrink(), OpEager(), OpEagerShared(), OpRebind(), OpRebind(), OpTempArray(),
                      OpTempGraph(), OpCollect(), OpInsert(), OpDelete())
const FZ_TLASOPS = (OpPushRange(), OpPushRange(), OpDeleteRange(), OpStoreRange(), OpResizeRange(),
                    OpRunTLAS(), OpRunTLAS())

"""Every array against its host model, every reader's `bad` count zero."""
function checkmodels(worlds, shared)
    for w in worlds
        @test Array(w.own) == w.ownmodel
        @test Array(w.a) == w.amodel
        @test Array(w.b) == w.bmodel
        @test all(iszero, Array(w.bad))
    end
    for (c, total) in zip(shared.counters, shared.sums)
        @test all(==(total[]), Array(c))
    end
    for bad in shared.bads
        @test all(iszero, Array(bad))
    end
end

"""Drops every hold the fuzz took: ranges, graphs, boxes, arrays, the TLAS and its BLAS (free! is once per handle)."""
function freefuzz!(worlds, shared)
    for w in worlds
        for (r, src) in w.ranges
            delete!(shared.tlas, r)
            free!(src)
        end
        free!(w.graph)
        free!(w.ref)
        foreach(free!, (w.own, w.a, w.b, w.bad))
    end
    foreach(run!, shared.tlasgraphs)             # applies the last deletes' build
    foreach(free!, shared.tlasgraphs)
    shared.tlas === nothing || foreach(free!, (shared.tlas, shared.blas))
    foreach(free!, shared.geometry)
    foreach(free!, shared.graphs)
    foreach(free!, shared.counters)
    foreach(free!, shared.bads)
end

"""
`steps` random operations from `ops` on each of `ntasks` tasks, against the
shared counters and graphs (and with `rt` a shared TLAS), under a watchdog;
then every array is compared with its model and everything is freed.
"""
function fuzz!(dev, ntasks, steps, ops; rt)
    shared = FuzzShared(dev; rt)
    worlds = [FuzzWorld(dev, shared, k) for k in 1:ntasks]
    tasks = map(worlds) do w
        Threads.@spawn for _ in 1:steps
            fuzzstep!(w, shared, rand(w.rng, ops))
        end
    end
    withtimeout(() -> foreach(wait, tasks), FZ_TIMEOUT)
    Mantle.waitidle(dev)
    checkmodels(worlds, shared)
    freefuzz!(worlds, shared)
end

# ── The hold fuzz (HOLD-10) ──

"""An array of the hold fuzz: its handle, its host model, and whether its creator freed it."""
mutable struct HeldArray
    array::Any
    model::Vector{Int32}
    freed::Bool
end

"""The hold fuzz's state: arrays, the graph, its node blocks as (block, array index, addition)."""
mutable struct HoldState
    dev::Any
    rng::Xoshiro
    arrays::Vector{HeldArray}
    graph::Any
    blocks::Vector{Tuple{Any,Int,Int32}}
    mismatches::Int
end

struct OpNewArray end       # an array created from data
struct OpFree end           # an array's creator hold dropped; the graph's blocks keep using it
struct OpNewGraph end       # the graph freed or dropped, a new one in its place
struct OpRead end           # an array read back and compared with its model

liveindices(h) = findall(x -> !x.freed, h.arrays)

function holdstep!(h, ::OpNewArray)
    model = rand(h.rng, Int32(-100):Int32(100), rand(h.rng, 1:FZ_OWNMAX))
    push!(h.arrays, HeldArray(MantleArray(h.dev, model), model, false))
end
function holdstep!(h, ::OpInsert)
    live = liveindices(h)
    isempty(live) && return
    i = rand(h.rng, live)
    k = Int32(rand(h.rng, 1:9))
    x = h.arrays[i]
    block = insert!(h.graph) do
        dispatch!(h.graph, fz_add!, (x.array, k), length(x.model))
    end
    push!(h.blocks, (block, i, k))
end
function holdstep!(h, ::OpDelete)
    isempty(h.blocks) && return
    block, _, _ = popat!(h.blocks, rand(h.rng, eachindex(h.blocks)))
    delete!(h.graph, block)
end
function holdstep!(h, ::OpRun)
    run!(h.graph)
    for (_, i, k) in h.blocks
        h.arrays[i].model .+= k
    end
end
function holdstep!(h, ::OpFree)
    live = liveindices(h)
    isempty(live) && return
    x = h.arrays[rand(h.rng, live)]
    free!(x.array)
    x.freed = true
end
function holdstep!(h, ::OpNewGraph)
    rand(h.rng, Bool) && free!(h.graph)    # else dropped: its finalizer releases the plan and the holds
    h.graph = Graph(h.dev)
    empty!(h.blocks)
end
holdstep!(h, ::OpCollect) = GC.gc(false)
function holdstep!(h, ::OpRead)
    live = liveindices(h)
    isempty(live) && return
    x = h.arrays[rand(h.rng, live)]
    h.mismatches += Array(x.array) != x.model
end

const FZ_HOLDOPS = (OpNewArray(), OpNewArray(), OpInsert(), OpInsert(), OpDelete(), OpRun(), OpRun(),
                    OpFree(), OpNewGraph(), OpCollect(), OpRead())

"""`steps` random hold operations; then every live array is read once more. Returns the state."""
function holdfuzz!(dev, steps; seed)
    h = HoldState(dev, Xoshiro(seed), HeldArray[], Graph(dev), Tuple{Any,Int,Int32}[], 0)
    for _ in 1:steps
        holdstep!(h, rand(h.rng, FZ_HOLDOPS))
    end
    for i in liveindices(h)
        h.mismatches += Array(h.arrays[i].array) != h.arrays[i].model
    end
    return h
end

"""Drops every hold of the hold fuzz."""
function freeholds!(h)
    free!(h.graph)
    foreach(x -> free!(x.array), h.arrays)
end

# ── Host time of a run against the resources it names (GT-03) ──

"""A graph naming `n` one-element arrays, four per launch; with `viaref` every other one through an array GPURef."""
function namedgraph(dev, n; viaref)
    arrays = [MantleArray(dev, Int32, 1) for _ in 1:n]
    named = [viaref && isodd(k) ? GPURef(dev, a) : a for (k, a) in enumerate(arrays)]
    g = Graph(dev)
    for k in 1:4:n
        dispatch!(g, fz_touch4!, Tuple(named[k:k+3]), 1)
    end
    return g, arrays, named
end

"""The median host time of `run!(g)`, each run started on an idle device."""
function runhosttime(g, dev; reps = 30)
    run!(g)
    times = map(1:reps) do _
        Mantle.waitidle(dev)
        @elapsed run!(g)
    end
    return sort(times)[cld(reps, 2)]
end

# FZ-01 = THR-07 (with HOLD-10's operations mixed in): 8 tasks, each with its
# own graph, plus graphs any task runs, with crossing reads and writes
# through the shared counters; stores, copy!, resizes, eager calls, array
# GPURef rebinds, node blocks inserted and deleted, temporary arrays and
# graphs freed or dropped, collections; on a ray-tracing device also TLAS
# push!/delete!, source stores and resizes and refit/build runs. Under the
# watchdog and the lock checker (FI-8: lock order, no sync-state, channel,
# pool or window lock held in a host wait). Every array exact; memory back at
# its baseline. A short fuzz first warms kernels, the pool and the cell arena.
@case "FZ-01" 6 :long begin
    dev = testdevice()
    rt = has(dev, :rt)
    ops = rt ? (FZ_CHANGEOPS..., FZ_TLASOPS...) : FZ_CHANGEOPS
    fuzz!(dev, 2, 50, ops; rt)
    before = memory(dev)
    Mantle.checklocks!(dev, true)
    try
        fuzz!(dev, 8, 2_000, ops; rt)
    finally
        Mantle.checklocks!(dev, false)
    end
    atbaseline(before, memory(dev))
end

# FZ-02 = GT-01: 4 tasks, 10^5 operations on composites (TLAS push!, delete!,
# source stores and resizes, array GPURef rebinds) and their arrays, against
# three graphs that refit or build the TLAS and the tasks' own graphs. No
# deadlock, arrays exact, memory back at its baseline after free!.
@case "FZ-02" 6 :rt :long begin
    dev = testdevice()
    ops = (FZ_TLASOPS..., OpRebind(), OpStore(), OpGrow(), OpShrink(), OpRun())
    fuzz!(dev, 2, 50, ops; rt = true)
    before = memory(dev)
    fuzz!(dev, 4, 25_000, ops; rt = true)
    atbaseline(before, memory(dev))
end

# FZ-03 = GT-02: steady state of a frame with per-frame stores (a camera as a
# one-element array, a moved instance's source) and a TLAS refit every run
# (api.md 6, test_run_is_submit.jl). run! allocates, compiles and records
# nothing (the instance count never changes, so no AccelCommand is marked)
# and submits its one segment once.
@case "FZ-03" 5 :rt :long begin
    dev = testdevice()
    blas, vertices, faces = fuzzblas(dev)
    tlas = TLAS(dev)
    sources = [MantleArray(dev, [movedby(Float32[k, 0, 0])]) for k in 1:8]
    foreach(src -> push!(tlas, blas, src), sources)
    cam = MantleArray(dev, [scaledby(1f0)])
    out = MantleArray(dev, Float32, 1024)
    g = Graph(dev)
    refit!(g, tlas)
    dispatch!(g, fz_frametick!, (out, cam), length(out))
    foreach(_ -> run!(g), 1:3)
    stats = map(1:1000) do f
        sources[mod1(f, 8)][1:1] = [movedby(Float32[f, 0, 0])]
        cam[1:1] = [scaledby(Float32(f))]
        last(counted(() -> run!(g)))
    end
    @test all(d -> d.allocations == 0, stats)
    @test all(d -> d.plancompiles == 0 && d.pipelinecompiles == 0 && d.kernelcompiles == 0, stats)
    @test all(d -> d.records == 0, stats)
    @test all(d -> submitcount(d) == 1, stats)
    @test Array(out) == Float32.(1000 .* (1:1024))
    free!(g)
    foreach(free!, sources)
    foreach(free!, (tlas, blas, vertices, faces, cam, out))
end

# FZ-04 = GT-03: plans naming 100, 2 500, 25 000 and 50 000 resources, once
# directly and once with every other one through an array GPURef, run with
# nothing changed: the host time per run does not grow with the count (api.md
# 6, test_run_host_time.jl; open decision 19, ruled). That file gates against
# the phase 3 model; this case only checks the growth, with a 3x margin for
# noise, and records the times.
@case "FZ-04" 6 :long begin
    dev = testdevice()
    for viaref in (false, true)
        times = map((100, 2_500, 25_000, 50_000)) do n
            g, arrays, named = namedgraph(dev, n; viaref)
            t = runhosttime(g, dev)
            free!(g)
            foreach(free!, named)
            foreach(free!, arrays)
            t
        end
        @info "FZ-04: host time per run for 100, 2 500, 25 000, 50 000 named resources" viaref times
        @test last(times) <= 3 * first(times)
    end
end

# FZ-05 = HOLD-10: 10 000 random declarations (node blocks inserted and
# deleted), runs, frees of arrays a graph still uses, graphs freed or dropped,
# collections and reads, under the watchdog. Every live array exact; a
# declaration naming an array its creator freed throws and leaves the graph
# runnable; memory back at its baseline.
@case "FZ-05" 4 :long begin
    dev = testdevice()
    freeholds!(holdfuzz!(dev, 200; seed = 1))
    before = memory(dev)
    h = withtimeout(() -> holdfuzz!(dev, 10_000; seed = 2), FZ_TIMEOUT)
    @test h.mismatches == 0
    freed = filter(x -> x.freed, h.arrays)
    if !isempty(freed)
        x = first(freed)
        @test_throws ArgumentError fill!(x.array, Int32(0))
        @test_throws ArgumentError insert!(h.graph) do
            dispatch!(h.graph, fz_add!, (x.array, Int32(1)), length(x.model))
        end
        run!(h.graph)
    end
    freeholds!(h)
    atbaseline(before, memory(dev))
end
