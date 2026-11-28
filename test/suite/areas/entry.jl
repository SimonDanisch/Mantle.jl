# GPURef slots and per-run host data (ENTRY). Contracts: api.md 2.3 (run!),
# 2.5 (GPURef, per-run host data) and 5 (a GPURef's slot); recording-plan.md
# "Run" (GPURefs are slots, stores into device arrays) and "Ordering across
# graphs and queues" (no store waits for a run); internals.jl section 7
# (GPURef, setindex!).
#
# A GPURef is one slot in device memory; `r[] = v` is a scheduled store into
# it. Data the host supplies per run is a device array the graph names,
# written by stores. Host waits are read from the diagnostics counter
# `hostwaits`. Where a case checks that a call does not wait, the graph's
# previous run is still in flight (a slow kernel from slowwrite!), so a wait
# would be a real one.

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

"""c[1] += 1, one work item."""
function entry_increment!(c)
    KI.get_global_id().x == 1 || return nothing   # ndrange is rounded up to whole work-groups
    @inbounds c[1] += one(eltype(c))
    return
end

"""Appends `v` at log[c[1] + 1] and advances c (one work item)."""
function entry_append!(log, c, v)
    KI.get_global_id().x == 1 || return nothing
    k = c[1] + Int32(1)
    k <= length(log) && (@inbounds log[k] = v)
    @inbounds c[1] = k
    return
end

"""torn[1] += 1 when the 16 words of the matrix `m` are not all equal (one work item)."""
function entry_torn!(torn, m)
    KI.get_global_id().x == 1 || return nothing
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

"""A graph that copies the device array `x` (written by stores) into `out` beside a slow kernel (`ms`); returns the graph and `x`."""
function storedgraph(dev, out; ms = 300)
    g = Graph(dev)
    x = MantleArray(dev, eltype(out), length(out))
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

# ── GPURef slots and per-run host data (ENTRY) ──

# api.md 2.3 (run!: a GPURef store never makes it wait), 2.5, internals.jl section 7: r = GPURef(dev, 1f0) read by g's kernel; with g's run in flight, r[] = 2f0 is a store applied by the next submission that touches r (g's next run, ordered after the previous run on the GPU): that run makes no host wait and uses 2f0.
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
    @test d.hostwaits == 0
    @test Array(out) == fill(2f0, n)
end

# api.md 2.5: r unchanged: 10 runs wait for nothing on the host (nothing is pending on r).
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

# plan, Run (GPURef(dev, v): the value is copied once at the call, so a multi-word value is never torn): T1 stores Mat4f values whose 16 words are equal (k, k, …, k) in a loop while T2 runs g 10 000 times; the kernel counts runs whose matrix words disagree: never torn.
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

# internals.jl section 7 (setindex! of a GPURef: a Store pended in sequence): T1 stores r[] = 1, 2, …, 2000 while T2 runs g 300 times, each run appending the value it got: every logged value is one that was stored, in store order; a run after both finished uses the newest, 2000.
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

# api.md 2.3 (run!), 2.5, internals.jl section 7 (the slot is a shared resource with namers): g1 and g2 read r, g3 does not; with all three runs in flight, r[] = 5: no next run waits on the host; g1 and g2 use 5, g3 is untouched.
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
    @test waits == (0, 0, 0)
    @test Array(outs[1]) == fill(Int32(5), n)
    @test Array(outs[2]) == fill(Int32(5), n)
    @test Array(outs[3]) == fill(Int32(3), n)
end

# api.md 2.3 (declaring into a compiled graph marks its plan), 2.5: r[] = 7 before g's first run, then (r unchanged) a group declared into g, which recompiles it: each plan bakes r's slot and reads 7.
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
    h = Group(g, "second")
    dispatch!(h, entry_set!, (out2, r), n)
    fill!(out, Int32(0))
    _, d = counted(() -> run!(g))
    @test d.plancompiles == 1
    @test Array(out) == fill(Int32(7), n)
    @test Array(out2) == fill(Int32(7), n)
end

# plan, Eager work; internals.jl section 11: a GPURef argument of an eager call is packed as its slot's address; r[] = 2f0 right after the call is a store applied by the next submission that touches r, after the launch's read: the launch uses 1f0, the next eager call 2f0.
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

# api.md 3 (namers), internals.jl section 8 (namers replaced at registration and retirement): 500 recompiles of g (a group using r declared, run, deleted) and 500 graphs compiled and freed, all reading r: r's namers do not grow. The namers have no hook; r's size is read with Base.summarysize, the device and the weak references to plans excluded.
@case "ENTRY-08" 4 :long begin
    dev = testdevice(); n = 16
    r = GPURef(dev, Int32(1))
    out = MantleArray(dev, Int32, n)
    g = Graph(dev)
    dispatch!(g, entry_set!, (out, r), n)
    run!(g)
    size0 = Base.summarysize(r; exclude = Union{Mantle.Device,WeakRef})
    for _ in 1:500
        h = Group(g, "block")
        dispatch!(h, entry_set!, (out, r), n)
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

# api.md 2.5 (per-run host data), plan, Run (stores into device arrays): x is a device array g names; x[:] = d before g's first run: the first run uses d.
@case "ENTRY-09" 4 begin
    dev = testdevice(); n = 64
    data = testdata(Int32, n)
    out = MantleArray(dev, Int32, n)
    g = Graph(dev)
    x = MantleArray(dev, Int32, n)
    dispatch!(g, entry_add!, (out, x, Int32(0)), n)
    x[:] = data
    run!(g)
    @test Array(out) == data
end

# api.md 2.5, plan, Ordering (no store waits for a run): x[:] = d while g's 400 ms run is in flight: the store waits for nothing on the host; the next submission that touches x (g's next run) applies it after that run's read on the GPU; the run! waits for nothing and uses d.
@case "ENTRY-10" 4 begin
    dev = testdevice(); n = 64
    out = MantleArray(dev, Int32, n)
    g, x = storedgraph(dev, out; ms = 400)
    x[:] = testdata(Int32, n; seed = 1)
    run!(g)
    Array(out)
    run!(g)
    data = testdata(Int32, n; seed = 2)
    _, dw = counted(() -> (x[:] = data))
    @test dw.hostwaits == 0
    _, dr = counted(() -> run!(g))
    @test dr.hostwaits == 0
    @test Array(out) == data
end

# api.md 2.3 (declaring marks the plan), 2.5: with a recompile pending (a group declared after the last run) and g's run in flight, x[:] = d waits for nothing; the recompiled plan's first run applies it, and both launches read d.
@case "ENTRY-11" 4 begin
    dev = testdevice(); n = 64
    out = MantleArray(dev, Int32, n)
    out2 = MantleArray(dev, Int32, n)
    g, x = storedgraph(dev, out; ms = 400)
    x[:] = testdata(Int32, n; seed = 1)
    run!(g)
    Array(out)
    run!(g)
    h = Group(g, "second")
    dispatch!(h, entry_add!, (out2, x, Int32(1)), n)
    data = testdata(Int32, n; seed = 2)
    _, d = counted(() -> (x[:] = data))
    @test d.hostwaits == 0
    run!(g)
    @test Array(out) == data
    @test Array(out2) == data .+ Int32(1)
end

# api.md 2.3 (what must survive runs and recompiles is a device array the graph names): x[:] = d1, run, a group declared (a recompile), then a run with no new store: the new plan reads d1 from x.
@case "ENTRY-12" 4 begin
    dev = testdevice(); n = 64
    data = testdata(Int32, n)
    out = MantleArray(dev, Int32, n)
    out2 = MantleArray(dev, Int32, n)
    g, x = storedgraph(dev, out; ms = 1)
    x[:] = data
    run!(g)
    h = Group(g, "second")
    dispatch!(h, entry_add!, (out2, x, Int32(1)), n)
    fill!(out, Int32(0))
    _, d = counted(() -> run!(g))
    @test d.plancompiles == 1
    @test Array(out) == data
    @test Array(out2) == data .+ Int32(1)
end

# plan, Run (a store is pending until the next submission that touches the array): x[:] = d once, then three runs that each add 1 to x in place: the store is applied once, by the first run, and x keeps its contents between runs, so run k sees d + k.
@case "ENTRY-13" 4 begin
    dev = testdevice(); n = 64
    data = testdata(Int32, n) .% Int32(10^6)
    out = MantleArray(dev, Int32, n)
    g = Graph(dev)
    x = MantleArray(dev, Int32, n)
    dispatch!(g, entry_add!, (x, x, Int32(1)), n)
    dispatch!(g, entry_add!, (out, x, Int32(0)), n)
    x[:] = data
    for k in 1:3
        run!(g)
        @test Array(out) == data .+ Int32(k)
    end
end

# internals.jl section 2 (setindex! of a graph array throws), api.md 2.2 (`a[r] = data`): a store into a graph array throws (which exception is unspecified); a partial store x[1:3] = into a device array is a store like any other; data of the wrong length throws DimensionMismatch. A full store afterwards replaces the pending partial one and reaches the run.
@case "ENTRY-14" 4 begin
    dev = testdevice(); n = 16
    out = MantleArray(dev, Int32, n)
    g = Graph(dev)
    x = MantleArray(dev, Int32, n)
    y = MantleArray(g, Int32, n)
    dispatch!(g, entry_add!, (out, x, Int32(0)), n)
    @test_throws Exception y[:] = zeros(Int32, n)
    x[1:3] = zeros(Int32, 3)
    @test_throws DimensionMismatch x[:] = zeros(Int32, n - 1)
    x[:] = ones(Int32, n)
    run!(g)
    @test Array(out) == ones(Int32, n)
end

# plan, Ordering (a change queued while a host function runs is ordered before the later stages), internals.jl section 6 (runhost!): x[:] = d from g's own host function takes no graph lock and does not throw: the store is pending, applied in front of a later stage or the next run, so the run after it reads d.
@case "ENTRY-15" 4 begin
    dev = testdevice(); n = 16
    out = MantleArray(dev, Int32, n)
    g = Graph(dev)
    x = MantleArray(dev, Int32, n)
    t = MantleArray(g, Int32, 1)
    dispatch!(g, entry_add!, (out, x, Int32(0)), n)
    dispatch!(g, entry_set!, (t, Int32(1)), 1)
    dispatch!(g, HostCall(_ -> (x[:] = zeros(Int32, n))), (t,); writes = ())
    x[:] = ones(Int32, n)
    @test thrown(() -> run!(g)) === nothing
    run!(g)
    @test Array(out) == zeros(Int32, n)
end

# api.md 5 (a GPURef's slot; lock order): r[] = v from one task while another runs g (which reads r) 500 times and a third makes and drops graphs and arrays and collects garbage (finalizers push hold drops): no deadlock, the lock order holds (checklocks!), and a run after them uses the newest value.
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

# internals.jl section 7 (setindex! of a GPURef converts to its type at the call): r[] = a value of another type: a value that converts (2.0 into a Float32 box) is stored converted; one that does not ("two") throws and leaves the value.
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

# api.md 2.6b (a group of draws; GPURefs it reads are slots it bakes, like a plan), 2.5: a value GPURef read by the draw of one group of draws and a device array written by stores read by another's, each group added after the graph compiled, inside the pass of its image (in the group that holds the pass's clear): r[] = v and x[:] = data reach the next run's pixels, with no recompile.
@case "ENTRY-18" 6 begin
    dev = testdevice(); w = 8
    r = GPURef(dev, 0.25f0)
    g = Graph(dev)
    x = MantleArray(dev, Float32, 1)
    img1 = Image(dev, RGBA{Float32}, w, w)
    img2 = Image(dev, RGBA{Float32}, w, w)
    s1 = Group(g, "pass 1")
    clear!(s1, img1, (0f0, 0f0, 0f0, 1f0))
    s2 = Group(g, "pass 2")
    clear!(s2, img2, (0f0, 0f0, 0f0, 1f0))
    run!(g)
    p1 = Group(s1, "value")
    draw!(p1, img1, entrypipeline(entry_valuefragment), (r,), 3)
    p2 = Group(s2, "array")
    draw!(p2, img2, entrypipeline(entry_arrayfragment), (x,), 3)
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

# api.md 2.4 (when!: the flag is a GPURef{Bool} or a one-element Bool array), 2.3 (a GPURef store never makes run! wait): a GPURef{Bool} as a when! flag, toggled before each of 1000 runs: the launch ran on exactly the runs with flag true; no run waits on the host, toggling or not.
@case "ENTRY-19" 4 begin
    dev = testdevice()
    flag = GPURef(dev, true)
    c = MantleArray(dev, Int32, 1)
    fill!(c, Int32(0))
    g = Graph(dev)
    when!(g, flag, entry_increment!, (c,), 1)
    run!(g)
    _, dt = counted() do
        for i in 1:1000
            flag[] = isodd(i)
            run!(g)
        end
    end
    @test Array(c) == Int32[501]
    @test dt.hostwaits == 0
    _, dn = counted(() -> foreach(_ -> run!(g), 1:100))
    @test dn.hostwaits == 0
    @test Array(c) == Int32[501]
end
