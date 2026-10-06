# Entry, value GPURefs, hostwritten arrays (catalogue-core.md, table 10:
# ENTRY). Contracts: api.md 2.5 and 5 (a plan's entry, a GPURef value);
# recording-plan.md "Run" (the entry, GPURef); internals.jl section 6
# (writeentry!) and section 7 (ValueBox, setindex! of a hostwritten array).
#
# Host waits are read from the diagnostics counter `hostwaits`. Where a case
# checks that a call waited, the graph's previous run is still in flight (a
# slow kernel from slowwrite!), so the wait is a real one.

import KernelInterface as KI
import GeometryBasics as GB

"""a[i] = v."""
function entry_set!(a, v)
    i = KI.get_global_id().x
    i <= length(a) || return
    @inbounds a[i] = v
    return
end

"""dst[i] = src[i] + k, over the shorter of the two."""
function entry_add!(dst, src, k)
    i = KI.get_global_id().x
    i <= min(length(dst), length(src)) || return
    @inbounds dst[i] = src[i] + k
    return
end

"""c[1] += 1, one thread."""
function entry_increment!(c)
    @inbounds c[1] += one(eltype(c))
    return
end

"""Appends `v` at log[c[1] + 1] and advances c (one thread)."""
function entry_append!(log, c, v)
    k = c[1] + Int32(1)
    k <= length(log) && (@inbounds log[k] = v)
    @inbounds c[1] = k
    return
end

"""torn[1] += 1 when the 16 words of the matrix `m` are not all equal (one thread)."""
function entry_torn!(torn, m)
    v = m[1]
    same = true
    for k in 2:16
        same &= m[k] == v
    end
    same || (@inbounds torn[1] += Int32(1))
    return
end

"""A full-screen triangle (vertex_index counts from one); ignores the draw's arguments."""
function entry_vertex(args...)
    v = vertex_index()
    x = v == Int32(2) ? 3f0 : -1f0
    y = v == Int32(3) ? 3f0 : -1f0
    return (position = GB.Vec4f(x, y, 0f0, 1f0),)
end
"""Red is the value argument."""
entry_valuefragment(inputs, level) = GB.Vec4f(level, 0f0, 0f0, 1f0)
"""Red is the first element of the array argument."""
entry_arrayfragment(inputs, a) = GB.Vec4f(a[1], 0f0, 0f0, 1f0)
entrypipeline(fragment) = Mantle.GraphicsPipeline(; vertex = Mantle.VertexShader(entry_vertex),
    fragment = Mantle.FragmentShader(fragment), topology = Mantle.TriangleList(),
    blend = Mantle.Opaque(), cull = Mantle.NoCull(), depth = Mantle.DepthOff())

"""A graph that fills `out` with the value of `r` beside a slow kernel (`ms`), so its runs stay in flight a while."""
function valuegraph(dev, out, r; ms = 300)
    g = Graph(dev)
    slowwrite!(g, MantleArray(dev, Int32, 1), 1; ms)
    dispatch!(g, entry_set!, (out, r), length(out))
    return g
end

"""A graph that copies the hostwritten `x` into `out` beside a slow kernel (`ms`); returns the graph and `x`."""
function hostwrittengraph(dev, out; ms = 300)
    g = Graph(dev)
    x = MantleArray(g, eltype(out), length(out); hostwritten = true)
    slowwrite!(g, MantleArray(dev, Int32, 1), 1; ms)
    dispatch!(g, entry_add!, (out, x, zero(eltype(out))), length(out))
    return g, x
end

"""Waits for `t` (a watchdog fails the test after `timeout` s) and returns what it threw, or `nothing`."""
function thrownby(t::Task; timeout = 30)
    ok = timedwait(() -> istaskdone(t), timeout) === :ok
    @test ok
    ok || error("task did not finish after $timeout s: likely a deadlock")
    return istaskfailed(t) ? t.result : nothing
end
"""What `f()` throws on another task, or `nothing`; a deadlock fails the test."""
thrown(f; timeout = 30) = thrownby(Threads.@spawn(f()); timeout)

# ── 10. Entry, value GPURefs, hostwritten (ENTRY) ──

# A:142-143, I:1339-1347: r = GPURef(dev, 1f0) taken by value; with g's run in flight, r[] = 2f0: the next run waits for the previous one (one host wait), writes the entry and uses 2f0.
@case "ENTRY-01" 4 begin
    dev = testdevice(); n = 64
    r = GPURef(dev, 1f0)
    out = MantleArray(dev, Float32, n)
    g = valuegraph(dev, out, r)
    run!(g)
    @test Array(out) == fill(1f0, n)
    run!(g)
    r[] = 2f0
    _, d = counted(() -> run!(g))
    @test d.hostwaits == 1
    @test Array(out) == fill(2f0, n)
end

# A:143: r unchanged: 10 runs wait for nothing on the host (no entry is written).
@case "ENTRY-02" 4 begin
    dev = testdevice(); n = 64
    r = GPURef(dev, 1f0)
    out = MantleArray(dev, Float32, n)
    g = valuegraph(dev, out, r; ms = 100)
    run!(g)
    Array(out)
    _, d = counted(() -> foreach(_ -> run!(g), 1:10))
    @test d.hostwaits == 0
    @test Array(out) == fill(1f0, n)
end

# P:896-899: T1 stores Mat4f values whose 16 words are equal (k, k, …, k) in a loop while T2 runs g 10 000 times; the kernel counts runs whose matrix words disagree: never torn.
@case "ENTRY-03" 4 :long begin
    dev = testdevice(); runs = 10_000
    r = GPURef(dev, GB.Mat4f(ntuple(_ -> 0f0, 16)...))
    torn = MantleArray(dev, Int32, 1)
    fill!(torn, Int32(0))
    g = Graph(dev)
    dispatch!(g, entry_torn!, (torn, r), 1)
    run!(g)
    stop = Threads.Atomic{Bool}(false)
    storer = Threads.@spawn begin
        k = 0f0
        while !stop[]
            k += 1f0
            r[] = GB.Mat4f(ntuple(_ -> k, 16)...)
        end
    end
    runner = Threads.@spawn foreach(_ -> run!(g), 1:runs)
    @test thrownby(runner; timeout = 1800) === nothing
    stop[] = true
    @test thrownby(storer) === nothing
    @test Array(torn) == Int32[0]
end

# I:1333-1337: T1 stores r[] = 1, 2, …, 2000 while T2 runs g 300 times, each run appending the value it got: every logged value is one that was stored, in store order; a run after both finished uses the newest, 2000.
@case "ENTRY-04" 4 begin
    dev = testdevice(); runs = 300; stores = 2000
    r = GPURef(dev, Int32(0))
    log = MantleArray(dev, Int32, runs + 2)
    c = MantleArray(dev, Int32, 1)
    fill!(log, Int32(-1))
    fill!(c, Int32(0))
    g = Graph(dev)
    dispatch!(g, entry_append!, (log, c, r), 1)
    run!(g)
    t1 = Threads.@spawn for k in 1:stores
        r[] = Int32(k)
    end
    t2 = Threads.@spawn foreach(_ -> run!(g), 1:runs)
    @test thrownby(t1) === nothing
    @test thrownby(t2) === nothing
    run!(g)
    seen = Array(log)
    @test all(v -> 0 <= v <= stores, seen)
    @test issorted(seen)
    @test seen[end] == stores
end

# D:507-508: r is in the entries of g1 and g2, not of g3; with all three runs in flight, r[] = 5: the next runs of g1 and g2 wait for their previous run and use 5; g3's next run neither waits nor writes its entry.
@case "ENTRY-05" 4 begin
    dev = testdevice(); n = 64
    r = GPURef(dev, Int32(1))
    outs = [MantleArray(dev, Int32, n) for _ in 1:3]
    g1 = valuegraph(dev, outs[1], r)
    g2 = valuegraph(dev, outs[2], r)
    g3 = valuegraph(dev, outs[3], Int32(3))
    graphs = (g1, g2, g3)
    foreach(run!, graphs)
    foreach(run!, graphs)
    r[] = Int32(5)
    waits = map(g -> counted(() -> run!(g))[2].hostwaits, graphs)
    @test waits == (1, 1, 0)
    @test Array(outs[1]) == fill(Int32(5), n)
    @test Array(outs[2]) == fill(Int32(5), n)
    @test Array(outs[3]) == fill(Int32(3), n)
end

# A:406: r[] = 7 before g's first run, then (r unchanged) an insert! that recompiles g: each new plan starts with r's current value in its entry.
@case "ENTRY-06" 4 begin
    dev = testdevice(); n = 64
    r = GPURef(dev, Int32(3))
    r[] = Int32(7)
    out = MantleArray(dev, Int32, n)
    out2 = MantleArray(dev, Int32, n)
    g = Graph(dev)
    dispatch!(g, entry_set!, (out, r), n)
    run!(g)
    @test Array(out) == fill(Int32(7), n)
    insert!(g) do
        dispatch!(g, entry_set!, (out2, r), n)
    end
    fill!(out, Int32(0))
    _, d = counted(() -> run!(g))
    @test d.plancompiles == 1
    @test Array(out) == fill(Int32(7), n)
    @test Array(out2) == fill(Int32(7), n)
end

# P:960-961: a GPURef argument of an eager call is packed at the call: r[] changed right after the call does not reach it.
@case "ENTRY-07" 4 begin
    dev = testdevice(); n = 64
    r = GPURef(dev, 1f0)
    out = MantleArray(dev, Float32, n)
    dispatch!(dev, entry_set!, (out, r), n)
    r[] = 2f0
    @test Array(out) == fill(1f0, n)
    dispatch!(dev, entry_set!, (out, r), n)
    @test Array(out) == fill(2f0, n)
end

# I:1364: 500 recompiles of g (an insert!/delete! of a block using r) and 500 graphs compiled and freed, all with r in their entries: r's list of plans does not grow. The list has no hook; r's size is read with Base.summarysize, the device and the weak references to plans excluded.
@case "ENTRY-08" 4 :long begin
    dev = testdevice(); n = 16
    r = GPURef(dev, Int32(1))
    out = MantleArray(dev, Int32, n)
    g = Graph(dev)
    dispatch!(g, entry_set!, (out, r), n)
    run!(g)
    size0 = Base.summarysize(r; exclude = Union{Mantle.Device,WeakRef})
    for _ in 1:500
        h = insert!(g) do
            dispatch!(g, entry_set!, (out, r), n)
        end
        run!(g)
        delete!(g, h)
        run!(g)
        gk = Graph(dev)
        dispatch!(gk, entry_set!, (out, r), n)
        run!(gk)
        free!(gk)
    end
    memory()
    r[] = Int32(2)
    run!(g)
    @test Array(out) == fill(Int32(2), n)
    @test Base.summarysize(r; exclude = Union{Mantle.Device,WeakRef}) <= size0 + 256
end

# A:144, P:1932-1933: x[:] = d before g's first run: the first run uses d.
@case "ENTRY-09" 4 begin
    dev = testdevice(); n = 64
    data = testdata(Int32, n)
    out = MantleArray(dev, Int32, n)
    g = Graph(dev)
    x = MantleArray(g, Int32, n; hostwritten = true)
    dispatch!(g, entry_add!, (out, x, Int32(0)), n)
    x[:] = data
    run!(g)
    @test Array(out) == data
end

# A:144, I:1384: x[:] = d while g's 400 ms run is in flight: the store waits for that run (one host wait); the run! after it waits for nothing.
@case "ENTRY-10" 4 begin
    dev = testdevice(); n = 64
    out = MantleArray(dev, Int32, n)
    g, x = hostwrittengraph(dev, out; ms = 400)
    x[:] = testdata(Int32, n; seed = 1)
    run!(g)
    Array(out)
    run!(g)
    data = testdata(Int32, n; seed = 2)
    _, dw = counted(() -> (x[:] = data))
    @test dw.hostwaits == 1
    _, dr = counted(() -> run!(g))
    @test dr.hostwaits == 0
    @test Array(out) == data
end

# I:1381-1382: with a recompile pending (an insert! after the last run) and g's run in flight, x[:] = d keeps d on the host (no host wait); the new plan's entry gets it.
@case "ENTRY-11" 4 begin
    dev = testdevice(); n = 64
    out = MantleArray(dev, Int32, n)
    out2 = MantleArray(dev, Int32, n)
    g, x = hostwrittengraph(dev, out; ms = 400)
    x[:] = testdata(Int32, n; seed = 1)
    run!(g)
    Array(out)
    run!(g)
    insert!(g) do
        dispatch!(g, entry_add!, (out2, x, Int32(1)), n)
    end
    data = testdata(Int32, n; seed = 2)
    _, d = counted(() -> (x[:] = data))
    @test d.hostwaits == 0
    run!(g)
    @test Array(out) == data
    @test Array(out2) == data .+ Int32(1)
end

# I:1045: x[:] = d1, run, insert! (a recompile), then a run with no new write: the new plan's entry starts from d1 (the old entry's hostwritten ranges are copied).
@case "ENTRY-12" 4 begin
    dev = testdevice(); n = 64
    data = testdata(Int32, n)
    out = MantleArray(dev, Int32, n)
    out2 = MantleArray(dev, Int32, n)
    g, x = hostwrittengraph(dev, out; ms = 1)
    x[:] = data
    run!(g)
    insert!(g) do
        dispatch!(g, entry_add!, (out2, x, Int32(1)), n)
    end
    fill!(out, Int32(0))
    _, d = counted(() -> run!(g))
    @test d.plancompiles == 1
    @test Array(out) == data
    @test Array(out2) == data .+ Int32(1)
end

# P:881-886: x[:] = d once, then three runs that each add 1 to x in place: each run starts again from d (the head copies the entry's data into the arena every run).
@case "ENTRY-13" 4 begin
    dev = testdevice(); n = 64
    data = testdata(Int32, n) .% Int32(10^6)
    out = MantleArray(dev, Int32, n)
    g = Graph(dev)
    x = MantleArray(g, Int32, n; hostwritten = true)
    dispatch!(g, entry_add!, (x, x, Int32(1)), n)
    dispatch!(g, entry_add!, (out, x, Int32(0)), n)
    x[:] = data
    for _ in 1:3
        run!(g)
        @test Array(out) == data .+ Int32(1)
    end
end

# I:1378 (ambiguity 18): x[:] on a graph array that is not hostwritten, a partial range x[1:3] =, and data of the wrong length each throw. Which exception is unspecified; for the length the store rule (A:82, DimensionMismatch) is the plausible one. A full write still works afterwards.
@case "ENTRY-14" 4 begin
    dev = testdevice(); n = 16
    out = MantleArray(dev, Int32, n)
    g = Graph(dev)
    x = MantleArray(g, Int32, n; hostwritten = true)
    y = MantleArray(g, Int32, n)
    dispatch!(g, entry_add!, (out, x, Int32(0)), n)
    @test_throws Exception y[:] = zeros(Int32, n)
    @test_throws Exception x[1:3] = zeros(Int32, 3)
    @test_throws DimensionMismatch x[:] = zeros(Int32, n - 1)
    x[:] = ones(Int32, n)
    run!(g)
    @test Array(out) == ones(Int32, n)
end

# I:1380: x[:] = d from g's own host function throws (it would take g's lock, which the run holds); run! throws with it.
@case "ENTRY-15" 4 begin
    dev = testdevice(); n = 16
    out = MantleArray(dev, Int32, n)
    g = Graph(dev)
    x = MantleArray(g, Int32, n; hostwritten = true)
    t = MantleArray(g, Int32, 1)
    dispatch!(g, entry_add!, (out, x, Int32(0)), n)
    dispatch!(g, entry_set!, (t, Int32(1)), 1)
    dispatch!(g, HostCall(_ -> (x[:] = zeros(Int32, n))), (t,); writes = ())
    x[:] = ones(Int32, n)
    @test thrown(() -> run!(g)) isa ArgumentError
end

# A:379, A:428: r[] = v from one task while another runs g (r in its entry) 500 times and a third makes and drops graphs and arrays and collects garbage (finalizers push hold drops): no deadlock, the lock order holds (checklocks!), and a run after them uses the newest value.
@case "ENTRY-16" 4 begin
    dev = testdevice(); n = 64; runs = 500
    Mantle.checklocks!(dev, true)
    r = GPURef(dev, Int32(0))
    out = MantleArray(dev, Int32, n)
    g = Graph(dev)
    dispatch!(g, entry_set!, (out, r), n)
    run!(g)
    stop = Threads.Atomic{Bool}(false)
    storer = Threads.@spawn begin
        k = Int32(0)
        while !stop[]
            r[] = (k += Int32(1))
        end
    end
    garbage = Threads.@spawn while !stop[]
        gk = Graph(dev)
        a = MantleArray(dev, Int32, 1024)
        dispatch!(gk, entry_set!, (a, r), 1024)
        run!(gk)
        GC.gc(false)
    end
    runner = Threads.@spawn foreach(_ -> run!(g), 1:runs)
    @test thrownby(runner; timeout = 120) === nothing
    stop[] = true
    @test thrownby(storer) === nothing
    @test thrownby(garbage; timeout = 60) === nothing
    r[] = Int32(-5)
    run!(g)
    @test Array(out) == fill(Int32(-5), n)
    Mantle.checklocks!(dev, false)
end

# I:1361-1372 (ambiguity 18): r[] = a value of another type. Most plausible: the box keeps its type: a value that converts (2.0 into a Float32 box) is stored converted; one that does not ("two") throws and leaves the value.
@case "ENTRY-17" 4 begin
    dev = testdevice(); n = 16
    r = GPURef(dev, 1f0)
    out = MantleArray(dev, Float32, n)
    g = Graph(dev)
    dispatch!(g, entry_set!, (out, r), n)
    run!(g)
    r[] = 2.0
    run!(g)
    @test Array(out) == fill(2f0, n)
    @test_throws MethodError r[] = "two"
    run!(g)
    @test Array(out) == fill(2f0, n)
end

# I:826-836 vs R:686 (ambiguity 21): a value GPURef read by one piece's draw and a hostwritten array read by another's, the pieces inserted after the graph compiled. Recommended: each piece has its own entry slots in its argument region, marked per piece: r[] = v and x[:] = data reach the next run's pixels, with no recompile.
# pending decision 10
@case "ENTRY-18" 6 begin
    dev = testdevice(); w = 8
    r = GPURef(dev, 0.25f0)
    g = Graph(dev)
    x = MantleArray(g, Float32, 1; hostwritten = true)
    img1 = Image(dev, RGBA{Float32}, w, w)
    img2 = Image(dev, RGBA{Float32}, w, w)
    p1 = render!(g, img1 => Clear((0f0, 0f0, 0f0, 1f0))) do p
    end
    p2 = render!(g, img2 => Clear((0f0, 0f0, 0f0, 1f0))) do p
    end
    run!(g)
    insert!(p1) do
        draw!(p1, entrypipeline(entry_valuefragment), (r,), 3)
    end
    insert!(p2) do
        draw!(p2, entrypipeline(entry_arrayfragment), (x,), 3)
    end
    x[:] = [0.5f0]
    _, d1 = counted(() -> run!(g))
    @test all(c -> c.r == 0.25f0, Array(img1))
    @test all(c -> c.r == 0.5f0, Array(img2))
    r[] = 0.75f0
    x[:] = [1f0]
    _, d2 = counted(() -> run!(g))
    @test all(c -> c.r == 0.75f0, Array(img1))
    @test all(c -> c.r == 1f0, Array(img2))
    @test d1.plancompiles == 0
    @test d2.plancompiles == 0
end

# A:130: a value GPURef as a when! flag, toggled before each of 1000 runs: the launch in the block ran on exactly the runs with flag 1; every toggling run waits for its previous run (1000 host waits); 100 runs with no store wait for nothing.
@case "ENTRY-19" 4 begin
    dev = testdevice()
    flag = GPURef(dev, Int32(1))
    c = MantleArray(dev, Int32, 1)
    fill!(c, Int32(0))
    g = Graph(dev)
    when!(g, flag) do
        dispatch!(g, entry_increment!, (c,), 1)
    end
    run!(g)
    _, dt = counted() do
        for i in 1:1000
            flag[] = Int32(isodd(i))
            run!(g)
        end
    end
    @test Array(c) == Int32[501]
    @test dt.hostwaits == 1000
    _, dn = counted(() -> foreach(_ -> run!(g), 1:100))
    @test dn.hostwaits == 0
    @test Array(c) == Int32[501]
end
