# Holds, free! and finalizers (catalogue-core.md, table 17: HOLD). Contracts:
# api.md:79, 91, 374-375, 471, 482; recording-plan.md "Memory" (holds and the
# last hold, 1299-1321); internals.jl section 10 (2082-2179);
# resizing-and-raytracing.jl B.10 (1312-1323); examples.jl 36 and 38.
# Line numbers of 2026-10-05.
#
# Use after free is caught with canaries allocated right after the release
# while work that writes the released memory is still on the GPU
# (slowwrite!): had the memory been reused early, that write lands in a
# canary. Leaks are caught with memory() back at its baseline.

using KernelAbstractions: @kernel, @index, get_backend

const HO_LEN = 2^16

# Spins `iters` steps, then writes `value` into x[1] (vendor work in flight).
@kernel function ho_spinwrite!(x, value, iters)
    acc = Int32(0)
    for i in Int32(1):iters
        acc = acc * Int32(1103515245) + i
    end
    x[1] = value + (acc == Int32(-7) ? Int32(1) : Int32(0))
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

# HOLD-10's operations on its state `s` (live arrays, graphs, insert! blocks).
ho_new!(s) = push!(s.arrays, MantleArray(s.dev, zeros(Int32, rand(s.rng, 1:4096))))
ho_free!(s) = isempty(s.arrays) || free!(popat!(s.arrays, rand(s.rng, eachindex(s.arrays))))
ho_drop!(s) = isempty(s.arrays) || popat!(s.arrays, rand(s.rng, eachindex(s.arrays)))   # left to the GC
ho_graph!(s) = push!(s.graphs, Graph(s.dev))
function ho_insert!(s)
    (isempty(s.graphs) || isempty(s.arrays)) && return
    g, a = rand(s.rng, s.graphs), rand(s.rng, s.arrays)
    h = insert!(g) do
        t = MantleArray(g, Int32, length(a))
        t .= a .+ Int32(1)
        copyto!(a, t)
    end
    push!(s.blocks, (g, h))
end
function ho_delete!(s)
    isempty(s.blocks) && return
    g, h = popat!(s.blocks, rand(s.rng, eachindex(s.blocks)))
    delete!(g, h)
end
ho_run!(s) = isempty(s.blocks) || run!(first(rand(s.rng, s.blocks)))
function ho_freegraph!(s)
    isempty(s.graphs) && return
    g = popat!(s.graphs, rand(s.rng, eachindex(s.graphs)))
    filter!(b -> first(b) !== g, s.blocks)
    free!(g)
end
ho_collect!(s) = GC.gc(false)
const HO_FUZZ = (ho_new!, ho_free!, ho_drop!, ho_graph!, ho_insert!, ho_delete!, ho_run!, ho_freegraph!, ho_collect!)

# ── 17. Holds, free!, finalizers (HOLD) ──

# api.md:79: free!(a) from 16 threads at once while a graph holds a: the creator's hold drops once, so the graph keeps a (exact runs, a canary allocated after is never written) and a is released once at free!(g).
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

# api.md:79, 482: free!(a) while g1 and g2 hold it: both keep running; an eager call, a store and a declaration through a throw.
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

# examples.jl:331-336: free!(feat), free!(e1), free!(d1) while both graphs' runs still write feat: feat's memory is released only after both last runs (canaries allocated at once stay intact), and memory comes back.
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

# api.md:471: free! and resize! of a view or a reshape throw; the parent stays usable.
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

# internals.jl:2084-2092: a declaration adding a hold races the last hold's drop (100 times): either the declaration throws ArgumentError, or it won and the graph keeps a (exact run, canary intact); a hold never goes from 0 back to 1.
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

# api.md:375: the last hold of an array with a pending resize and store: no submission, the changes and their upload region are dropped, nothing is written into the released memory (a canary), memory back at the baseline.
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

# recording-plan.md:1299-1303, internals.jl:2168-2179: arrays and a graph dropped without free!: the GC's finalizers only push to the inbox (cells and upload regions still live), and sweep! releases them.
@case "HOLD-07" 2 begin
    dev = testdevice()
    base = memory(dev)
    cells, uploads = ho_dropall(dev)
    GC.gc(true)
    @test Mantle.livecells(dev) == cells
    @test Mantle.liveuploads(dev) == uploads
    @test memory(dev) == base             # memory() sweeps
end

# examples.jl:316-322: an array dropped without free! while its last write is on the GPU, then GC: an array allocated at once does not get its memory before that write passed (a canary).
@case "HOLD-08" 2 begin
    dev = testdevice()
    ho_dropinflight(dev, HO_LEN, 7)
    GC.gc(true)
    c = canary(dev, HO_LEN)
    Mantle.waitidle(dev)
    @test intact(c)
    free!(c)
end

# api.md:91: a view of a freed parent shares its freed mark: eager calls, stores, host reads and declarations through it throw.
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

# api.md:374: 10 000 random creations, free!s, drops, graphs, insert!/delete! blocks, runs, graph free!s and collections: no error, and memory back at the baseline after everything is freed.
@case "HOLD-10" 2 :long begin
    dev = testdevice()
    base = memory(dev)
    s = (dev = dev, rng = Xoshiro(10), arrays = Any[], graphs = Any[], blocks = Any[])
    for _ in 1:10_000
        rand(s.rng, HO_FUZZ)(s)
    end
    foreach(free!, s.graphs)
    foreach(free!, s.arrays)
    empty!(s.graphs); empty!(s.arrays); empty!(s.blocks)
    @test memory(dev) == base
end

# internals.jl:2156-2167: free!(g) after delete!(g, h) and no further run: the holds the deleted block left (g.unnamed) drop with the plan; memory back at the baseline.
@case "HOLD-11" 2 begin
    dev = testdevice()
    base = memory(dev)
    a, o = MantleArray(dev, testdata(Int32, HO_LEN)), MantleArray(dev, Int32, HO_LEN)
    g = Graph(dev)
    h = insert!(g) do
        t = MantleArray(g, Int32, HO_LEN)
        t .= a .+ Int32(1)
        copyto!(o, t)
    end
    run!(g)
    delete!(g, h)                         # no node names a or o; g's plan still does
    free!(a); free!(o)
    free!(g)
    @test memory(dev) == base
end

# api.md:79, 206: free!(a) inside withview while a vendor kernel still writes a: a's memory goes back only after the stream's token (canaries allocated after the block stay intact); memory back at the baseline.
@case "HOLD-12" 10 :vendorview begin
    dev = testdevice()
    base = memory(dev)
    cuda = cudamodule()
    s = cuda.stream()
    a = MantleArray(dev, Int32, HO_LEN)
    withview(cuda.CuArray, a, s) do v
        ho_spinwrite!(get_backend(v))(v, Int32(7), Int32(calibratedspin(dev, 500)); ndrange = 1)
        free!(a)
    end
    cs = [canary(dev, HO_LEN) for _ in 1:4]
    cuda.synchronize(s)
    @test all(intact, cs)
    foreach(free!, cs)
    @test memory(dev) == base
end

# api.md:79: free!(a) with a store pending while a graph holds a: the graph's next run applies the store.
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

# internals.jl:2229-2230 (ambiguity 22, settled there): a host read through a freed handle throws, as other calls do, while a graph still holds the array and keeps running.
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
