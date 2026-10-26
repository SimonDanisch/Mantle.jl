# Holds, free! and finalizers. Contracts: api.md 2.2 (`free!`, views), 5
# (an array's holds, the last hold) and 7; recording-plan.md "Memory" (holds
# and the last hold); internals.jl section 10; resizing-and-raytracing.jl
# B.10; examples.jl 36 and 38.
#
# Use after free is caught with canaries allocated right after the release
# while work that writes the released memory is still on the GPU
# (slowwrite!): had the memory been reused early, that write lands in a
# canary. Leaks are caught with memory() back at its baseline.

const HO_LEN = 2^16

"""Spins `iters` steps, then writes `value` into x[1] (vendor work in flight)."""
function ho_spinwrite!(x, value, iters)
    KI.get_global_id().x == 1 || return nothing     # ndrange is rounded up to whole work-groups
    acc = Int32(0)
    for i in Int32(1):iters
        acc = acc * Int32(1103515245) + i
    end
    x[1] = value + (acc == Int32(-7) ? Int32(1) : Int32(0))
    return nothing
end

"""dst[i] = src[i] + k."""
function ho_add!(dst, src, k)
    i = KI.get_global_id().x
    i <= length(dst) || return nothing      # ndrange is rounded up to whole work-groups
    @inbounds dst[i] = src[i] + k
    return nothing
end

"""A graph that adds one to device array `a` into a transient and copies it to `out`: it names both."""
function plusonegraph(dev, a, out)
    g = Graph(dev)
    t = MantleArray(g, eltype(a), length(a))
    t .= a .+ one(eltype(a))
    copyto!(out, t)
    return g
end

"""The CUDA module; loaded wherever a CudaDevice exists."""
cudamodule() = Base.require(Base.PkgId(Base.UUID("052768ef-5323-5732-b1bb-66c8b64840ba"), "CUDA"))

"""Waits for `t` (a watchdog fails the test after `timeout` s) and returns what it threw, or `nothing`."""
function thrownby(t::Task; timeout = 30)
    ok = timedwait(() -> istaskdone(t), timeout) === :ok
    @test ok
    ok || error("task did not finish after $timeout s: likely a deadlock")
    return istaskfailed(t) ? t.result : nothing
end

"""
Creates an array with a pending store, an output array and a graph naming
both, and drops all three without free!. Returns the live cells and upload
regions just before they become unreachable.
"""
@noinline function ho_dropall(dev)
    a = MantleArray(dev, Int32, HO_LEN)
    a[1:4] = Int32[1, 2, 3, 4]
    plusonegraph(dev, a, MantleArray(dev, Int32, HO_LEN))
    return Mantle.livecells(dev), Mantle.liveuploads(dev)
end

"""Drops, without free!, an array of `n` Int32 whose last use (a write of `value` into its first element) is still on the GPU."""
@noinline ho_dropinflight(dev, n, value) = (slowwrite!(dev, MantleArray(dev, Int32, n), value; ms = 500); nothing)

# HOLD-10's operations on its state `s` (live arrays, graphs, groups).
ho_new!(s) = push!(s.arrays, MantleArray(s.dev, zeros(Int32, rand(s.rng, 1:4096))))
ho_free!(s) = isempty(s.arrays) || free!(popat!(s.arrays, rand(s.rng, eachindex(s.arrays))))
ho_drop!(s) = isempty(s.arrays) || popat!(s.arrays, rand(s.rng, eachindex(s.arrays)))   # left to the GC
ho_graph!(s) = push!(s.graphs, Graph(s.dev))
function ho_group!(s)
    (isempty(s.graphs) || isempty(s.arrays)) && return
    g, a = rand(s.rng, s.graphs), rand(s.rng, s.arrays)
    h = Group(g, "plusone")
    t = MantleArray(g, Int32, length(a))
    dispatch!(h, ho_add!, (t, a, Int32(1)), length(a))
    dispatch!(h, ho_add!, (a, t, Int32(0)), length(a))
    push!(s.groups, (g, h))
end
function ho_delete!(s)
    isempty(s.groups) && return
    g, h = popat!(s.groups, rand(s.rng, eachindex(s.groups)))
    delete!(g, h)
end
ho_run!(s) = isempty(s.groups) || run!(first(rand(s.rng, s.groups)))
function ho_freegraph!(s)
    isempty(s.graphs) && return
    g = popat!(s.graphs, rand(s.rng, eachindex(s.graphs)))
    filter!(b -> first(b) !== g, s.groups)
    free!(g)
end
ho_collect!(s) = GC.gc(false)
const HO_FUZZ = (ho_new!, ho_free!, ho_drop!, ho_graph!, ho_group!, ho_delete!, ho_run!, ho_freegraph!, ho_collect!)

# ── Holds, free!, finalizers (HOLD) ──

# api.md 2.2 (`free!(a)`: the creator's hold drops once, whatever the number of threads): free!(a) from 16 threads at once while a graph holds a: the creator's hold drops once, so the graph keeps a (exact runs, a canary allocated after is never written) and a is released once at free!(g).
@case "HOLD-01" 2 begin
    dev = testdevice()
    base = memory(dev)
    x = testdata(Int32, HO_LEN)
    a, o = MantleArray(dev, x), MantleArray(dev, Int32, HO_LEN)
    g = plusonegraph(dev, a, o)
    run!(g)
    go = Threads.Event()
    tasks = [Threads.@spawn (wait(go); free!(a)) for _ in 1:16]
    notify(go)
    withtimeout(() -> foreach(wait, tasks), 30)
    c = canary(dev, HO_LEN)
    run!(g)
    @test Array(o) == x .+ Int32(1)
    @test intact(c)
    free!(g); free!(o); free!(c)
    @test memory(dev) == base
end

# api.md 2.2 (`free!(a)`: later use through any handle throws), 7: free!(a) while g1 and g2 hold it: both keep running; an eager call, a store and a declaration through a throw.
@case "HOLD-02" 2 begin
    dev = testdevice()
    x = testdata(Int32, HO_LEN)
    a = MantleArray(dev, x)
    o1, o2 = MantleArray(dev, Int32, HO_LEN), MantleArray(dev, Int32, HO_LEN)
    g1, g2 = plusonegraph(dev, a, o1), plusonegraph(dev, a, o2)
    free!(a)
    @test_throws ArgumentError (a .+= Int32(1))
    @test_throws ArgumentError (a[1:1] = Int32[0])
    g3 = Graph(dev)
    t3 = MantleArray(g3, Int32, HO_LEN)
    @test_throws ArgumentError (t3 .= a)
    run!(g1); run!(g2)
    @test Array(o1) == x .+ Int32(1)
    @test Array(o2) == x .+ Int32(1)
end

# examples.jl 38: free!(feat), free!(e1), free!(d1) while both graphs' runs still write feat: feat's memory is released only after both last runs (canaries allocated at once stay intact), and memory comes back.
@case "HOLD-03" 2 begin
    dev = testdevice()
    base = memory(dev)
    feat = MantleArray(dev, Int32, HO_LEN)
    e1, d1 = Graph(dev), Graph(dev)
    slowwrite!(e1, feat, 7; ms = 300)
    slowwrite!(d1, feat, 9; ms = 300)
    run!(e1); run!(d1)                    # compiled
    run!(e1); run!(d1)                    # both on the GPU, writing feat late
    free!(feat); free!(e1); free!(d1)     # the last hold goes while they run
    cs = [canary(dev, HO_LEN) for _ in 1:4]
    Mantle.waitidle(dev)
    @test all(intact, cs)
    foreach(free!, cs)
    @test memory(dev) == base
end

# api.md 7 (resize! or free! of a view throws): free! and resize! of a view or a reshape throw; the parent stays usable.
@case "HOLD-04" 6 begin
    dev = testdevice()
    a = MantleArray(dev, Int32, 16)
    v, r = view(a, 1:8), reshape(a, 4, 4)
    @test_throws ArgumentError free!(v)
    @test_throws ArgumentError resize!(v, 4)
    @test_throws ArgumentError free!(r)
    @test_throws ArgumentError resize!(r, (2, 2))
    fill!(a, Int32(3))
    @test all(==(Int32(3)), Array(a))
end

# internals.jl section 10 (addhold!: a hold never goes from zero back to one): a declaration adding a hold races the last hold's drop (100 times): either the declaration throws ArgumentError, or it won and the graph keeps a (exact run, canary intact); a hold never goes from 0 back to 1.
@case "HOLD-05" 2 begin
    dev = testdevice()
    x = testdata(Int32, HO_LEN)
    for _ in 1:100
        a = MantleArray(dev, x)
        o = MantleArray(dev, Int32, HO_LEN)
        g = Graph(dev)
        t = MantleArray(g, Int32, HO_LEN)
        go = Threads.Event()
        dropper = Threads.@spawn (wait(go); free!(a))
        declarer = Threads.@spawn (wait(go); t .= a .+ Int32(1))
        notify(go)
        withtimeout(() -> wait(dropper), 30)
        e = thrownby(declarer)
        if e === nothing
            copyto!(o, t)
            c = canary(dev, HO_LEN)       # would take a's memory had the drop released it
            run!(g)
            @test Array(o) == x .+ Int32(1)
            @test intact(c)
            free!(c)
        else
            @test e isa ArgumentError
        end
        free!(g); free!(o)
    end
end

# api.md 5 (the last hold: pending changes dropped, upload regions retired): the last hold of an array with a pending resize and store: no submission, the changes and their upload region are dropped, nothing is written into the released memory (a canary), memory back at the baseline.
@case "HOLD-06" 4 begin
    dev = testdevice()
    base = memory(dev)
    a = MantleArray(dev, Int32, HO_LEN)
    resize!(a, 2HO_LEN)
    a[1:HO_LEN] = testdata(Int32, HO_LEN)
    _, diag = counted(() -> free!(a), dev)
    @test submitcount(diag) == 0
    c = canary(dev, 2HO_LEN)
    Mantle.waitidle(dev)
    @test intact(c)
    free!(c)
    @test memory(dev) == base
end

# recording-plan.md "Memory" (one release engine: finalizers push onto the inbox and nothing else), internals.jl section 10 (finalizers, settle!): arrays and a graph dropped without free!: the GC's finalizers only push to the inbox (cells and upload regions still live), and sweep! releases them.
@case "HOLD-07" 2 begin
    dev = testdevice()
    base = memory(dev)
    cells, uploads = ho_dropall(dev)
    GC.gc(true)
    @test Mantle.livecells(dev) == cells
    @test Mantle.liveuploads(dev) == uploads
    @test memory(dev) == base             # memory() sweeps
end

# examples.jl 36: an array dropped without free! while its last write is on the GPU, then GC: an array allocated at once does not get its memory before that write passed (a canary).
@case "HOLD-08" 2 begin
    dev = testdevice()
    ho_dropinflight(dev, HO_LEN, 7)
    GC.gc(true)
    c = canary(dev, HO_LEN)
    Mantle.waitidle(dev)
    @test intact(c)
    free!(c)
end

# api.md 2.2 (views share the parent's freed mark): a view of a freed parent shares its freed mark: eager calls, stores, host reads and declarations through it throw.
@case "HOLD-09" 6 begin
    dev = testdevice()
    a = MantleArray(dev, Int32, 16)
    v = view(a, 1:8)
    free!(a)
    @test_throws ArgumentError fill!(v, Int32(1))
    @test_throws ArgumentError (v[1:2] = Int32[1, 2])
    @test_throws ArgumentError Array(v)
    g = Graph(dev)
    t = MantleArray(g, Int32, 8)
    @test_throws ArgumentError (t .= v)
end

# api.md 5 (an array's holds), 2.3 (`Group`, `delete!(g, group)`): 10 000 random creations, free!s, drops, graphs, groups declared and deleted, runs, graph free!s and collections: no error, and memory back at the baseline after everything is freed.
@case "HOLD-10" 2 :long begin
    dev = testdevice()
    base = memory(dev)
    s = (dev = dev, rng = Xoshiro(10), arrays = Any[], graphs = Any[], groups = Any[])
    for _ in 1:10_000
        rand(s.rng, HO_FUZZ)(s)
    end
    foreach(free!, s.graphs)
    foreach(free!, s.arrays)
    empty!(s.graphs); empty!(s.arrays); empty!(s.groups)
    @test memory(dev) == base
end

# internals.jl section 10 (release!(g): g.unnamed): free!(g) after delete!(g, h) and no further run: the holds the deleted group left (g.unnamed) drop with the plan; memory back at the baseline.
@case "HOLD-11" 2 begin
    dev = testdevice()
    base = memory(dev)
    a, o = MantleArray(dev, testdata(Int32, HO_LEN)), MantleArray(dev, Int32, HO_LEN)
    g = Graph(dev)
    h = Group(g, "plusone")
    t = MantleArray(g, Int32, HO_LEN)
    dispatch!(h, ho_add!, (t, a, Int32(1)), HO_LEN)
    dispatch!(h, ho_add!, (o, t, Int32(0)), HO_LEN)
    run!(g)
    delete!(g, h)                         # no node names a or o; g's plan still does
    free!(a); free!(o)
    free!(g)
    @test memory(dev) == base
end

# api.md 2.2 (`free!`), 2.9 (the hold is dropped with the stream's token): free!(a) inside withview while a vendor kernel still writes a: a's memory goes back only after the stream's token (canaries allocated after the block stay intact); memory back at the baseline.
@case "HOLD-12" 10 :vendorview begin
    dev = testdevice()
    base = memory(dev)
    cuda = cudamodule()
    s = cuda.stream()
    a = MantleArray(dev, Int32, HO_LEN)
    withview(cuda.CuArray, a, s) do v
        KI.@launch KI.get_backend(v) ndrange = 1 ho_spinwrite!(v, Int32(7), Int32(calibratedspin(dev, 500)))
        free!(a)
    end
    cs = [canary(dev, HO_LEN) for _ in 1:4]
    cuda.synchronize(s)
    @test all(intact, cs)
    foreach(free!, cs)
    @test memory(dev) == base
end

# api.md 2.2 (`free!(a)`: graphs that hold a keep using it): free!(a) with a store pending while a graph holds a: the graph's next run applies the store.
@case "HOLD-13" 4 begin
    dev = testdevice()
    x = testdata(Int32, HO_LEN)
    a, o = MantleArray(dev, x), MantleArray(dev, Int32, HO_LEN)
    g = plusonegraph(dev, a, o)
    run!(g)
    a[1:4] = Int32[1, 2, 3, 4]
    free!(a)
    run!(g)
    x[1:4] = 1:4
    @test Array(o) == x .+ Int32(1)
end

# internals.jl section 11 (`Array`: a host read through a freed handle throws), api.md 2.2: a host read through a freed handle throws, as other calls do, while a graph still holds the array and keeps running.
@case "HOLD-14" 2 begin
    dev = testdevice()
    x = testdata(Int32, HO_LEN)
    a, o = MantleArray(dev, x), MantleArray(dev, Int32, HO_LEN)
    g = plusonegraph(dev, a, o)
    free!(a)
    @test_throws ArgumentError Array(a)
    @test_throws ArgumentError copyto!(zeros(Int32, HO_LEN), a)
    run!(g)
    @test Array(o) == x .+ Int32(1)
end

# internals.jl section 9 (storage is retired with the tokens of every access in flight, on any channel), recording-plan.md Queues: HOLD-03 with every submission on the next queue (spreadqueues): feat is written late on two queues when its last hold goes; canaries allocated at once stay intact, and memory comes back.
@case "HOLD-15" 3 :multicompute begin
    dev = testdevice()
    base = memory(dev)
    spreadqueues(dev) do
        feat = MantleArray(dev, Int32, HO_LEN)
        e1, d1 = Graph(dev), Graph(dev)
        slowwrite!(e1, feat, 7; ms = 300)
        slowwrite!(d1, feat, 9; ms = 300)
        run!(e1); run!(d1)                    # compiled
        run!(e1); run!(d1)                    # on two queues, writing feat late
        free!(feat); free!(e1); free!(d1)     # the last hold goes while they run
        cs = [canary(dev, HO_LEN) for _ in 1:4]
        Mantle.waitidle(dev)
        @test all(intact, cs)
        foreach(free!, cs)
    end
    @test memory(dev) == base
end
