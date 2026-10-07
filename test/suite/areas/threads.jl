# Locks and threads (catalogue-core.md, table 13: THR). Any thread compiles,
# records, runs and frees; builders are values; host waits are GC-safe and
# never hold a sync-state, channel, pool or window lock; the lock order holds
# (api.md section 5, "Lock order"; recording-plan.md "Ordering across graphs
# and queues", "Threads"; internals.jl sections 6 and 8).
#
# Doc keys as in the catalogue: A api.md, P recording-plan.md, I internals.jl,
# D decisions.md. The lock tracker is the test hook Mantle.checklocks!
# (helpers.jl): with it on, a lock taken out of order, or a sync-state,
# channel, pool or window lock held in a host wait, throws.

using KernelAbstractions: @kernel, @index

@kernel function thr_copy!(dst, src)
    i = @index(Global)
    @inbounds dst[i] = src[i]
end

@kernel function thr_fill!(a, v)
    i = @index(Global)
    @inbounds a[i] = v
end

@kernel function thr_addone!(dst, src)
    i = @index(Global)
    @inbounds dst[i] = src[i] + Int32(1)
end

@kernel function thr_add!(dst, a, b)
    i = @index(Global)
    @inbounds dst[i] = a[i] + b[i]
end

@kernel function thr_accumulate!(acc, src)
    i = @index(Global)
    @inbounds acc[i] += src[i]
end

@kernel function thr_muladd!(x, k)           # x = 3x + k, wrapping: the result depends on the order of the calls
    i = @index(Global)
    @inbounds x[i] = x[i] * Int32(3) + k
end

@kernel function thr_broadcastfirst!(dst, val)   # dst[i] = val[1]
    i = @index(Global)
    @inbounds dst[i] = val[1]
end

@kernel function thr_setfirst!(a, v)
    @inbounds a[1] = v
end

"""A host function: dst = src + 1."""
thr_hostaddone(src, dst) = (dst .= src .+ Int32(1); nothing)

"""A host function that throws while `fail[]` is set, else dst = src + 1."""
thr_hostfail(fail, src, dst) = fail[] ? error("host function failed") : thr_hostaddone(src, dst)

"""
x[i] += 1 through a host node: stage 1 copies x into a transient, the host
function adds one, stage 2 copies the result back into x.
"""
function thr_hostbump(dev, x, n)
    g = Graph(dev)
    t, u = MantleArray(g, Int32, n), MantleArray(g, Int32, n)
    dispatch!(g, thr_copy!, (t, x), n)
    dispatch!(g, HostCall(thr_hostaddone), (t, u); writes = (u,))
    dispatch!(g, thr_copy!, (x, u), n)
    return g
end

"""
Two graphs with host nodes sharing x and their arena part, eager calls and
host reads, on two tasks; true if x counted every run of the first graph.
"""
function thr_workload(dev; rounds = 200, n = 1024)
    x, y, z = MantleArray(dev, zeros(Int32, n)), MantleArray(dev, Int32, n), MantleArray(dev, Int32, n)
    g1 = thr_hostbump(dev, x, n)
    g2 = Graph(dev); t = MantleArray(g2, Int32, n)
    dispatch!(g2, thr_copy!, (t, x), n)
    dispatch!(g2, thr_copy!, (y, t), n)
    one = Threads.@spawn for _ in 1:rounds
        run!(g1); dispatch!(dev, thr_addone!, (z, x), n)
    end
    two = Threads.@spawn for _ in 1:rounds
        run!(g2); Array(y)
    end
    withtimeout(() -> (wait(one); wait(two)), 300)
    exact = Array(x) == fill(Int32(rounds), n)
    free!(g1); free!(g2)
    return exact
end

"""
A task that declares a graph and makes eager calls with a yield after every
call, `iterations` rounds; true if every round's result was exact.
"""
function thr_interleaved(dev, k, iterations; n = 4096)
    val = MantleArray(dev, Int32[0]); yield()
    out = MantleArray(dev, Int32, n); yield()
    plusone = MantleArray(dev, Int32, n); yield()
    g = Graph(dev); yield()
    t = MantleArray(g, Int32, n); yield()
    dispatch!(g, thr_broadcastfirst!, (t, val), n); yield()
    dispatch!(g, thr_copy!, (out, t), n); yield()
    exact = true
    for i in 1:iterations
        v = Int32(100_000 * k + i)
        dispatch!(dev, thr_setfirst!, (val, v), 1); yield()
        run!(g); yield()
        dispatch!(dev, thr_addone!, (plusone, out), n); yield()
        exact &= Array(plusone) == fill(v + Int32(1), n); yield()
    end
    free!(g)
    return exact
end

"""Eager x = 3x + k for k in 1:steps, in program order."""
thr_chain!(dev, x, steps) = foreach(k -> dispatch!(dev, thr_muladd!, (x, Int32(k)), length(x)), 1:steps)

"""Whether `target` is reachable from `root` through fields, array and Memory elements."""
function thr_reaches(root, target)
    seen = Base.IdSet{Any}()
    stack = Any[root]
    while !isempty(stack)
        x = pop!(stack)
        x === target && return true
        (isbits(x) || x in seen) && continue
        push!(seen, x)
        append!(stack, thr_children(x))
    end
    return false
end
thr_children(x) = Any[getfield(x, i) for i in 1:nfields(x) if isdefined(x, i)]
thr_children(x::Union{Array{T},Memory{T}}) where {T} =
    isbitstype(T) ? Any[] : Any[x[i] for i in eachindex(x) if isassigned(x, i)]
thr_children(::Union{Module,Task,Type,Method,Core.MethodInstance,Core.CodeInstance}) = Any[]

"""Graphs on `d` that name `a`, each run once; nothing keeps them, so the next GC runs their finalizers."""
function thr_garbage(d, a, howmany)
    for _ in 1:howmany
        g = Graph(d); t = MantleArray(g, Int32, length(a))
        dispatch!(g, thr_copy!, (t, a), length(a))
        run!(g)
    end
end

"""Fuzz graph k: reads two of the arrays (crossing the other graphs' writes), a host node, writes a third."""
function thr_fuzzgraph(dev, arrays, k, n)
    src, other, dst = (arrays[mod1(k + j, length(arrays))] for j in 0:2)
    g = Graph(dev)
    t, u = MantleArray(g, Int32, n), MantleArray(g, Int32, n)
    dispatch!(g, thr_add!, (t, src, other), n)
    dispatch!(g, HostCall(thr_hostaddone), (t, u); writes = (u,))
    dispatch!(g, thr_copy!, (dst, u), n)
    return g
end

"""A fuzz task owning graph k: runs it, makes eager calls and stores, frees and rebuilds it."""
function thr_fuzzgraphtask(dev, arrays, k, iterations, n)
    rng = Xoshiro(k)
    g = thr_fuzzgraph(dev, arrays, k, n)
    for _ in 1:iterations
        op = rand(rng, 1:6)
        if op <= 3
            run!(g)
        elseif op == 4
            dispatch!(dev, thr_add!, (rand(rng, arrays), rand(rng, arrays), rand(rng, arrays)), n)
        elseif op == 5
            rand(rng, arrays)[1:4] = rand(rng, Int32(0):Int32(9), 4)
        else
            free!(g); g = thr_fuzzgraph(dev, arrays, k, n)
        end
    end
    free!(g)
    return true
end

"""A fuzz task with no graph: eager calls and stores on the shared arrays."""
function thr_fuzzchangetask(dev, arrays, k, iterations, n)
    rng = Xoshiro(k)
    for _ in 1:iterations
        if rand(rng, Bool)
            dispatch!(dev, thr_add!, (rand(rng, arrays), rand(rng, arrays), rand(rng, arrays)), n)
        else
            rand(rng, arrays)[1:4] = rand(rng, Int32(0):Int32(9), 4)
        end
    end
    return true
end

"""A fuzz task resizing a window between runs of the graph presenting to it."""
function thr_fuzzwindowtask(dev, win, iterations)
    rng = Xoshiro(99)
    g = Graph(dev)
    render!(g, win => Clear((0f0, 0f0, 0f0, 1f0))) do p end
    for _ in 1:iterations
        resize!(win, rand(rng, 64:640), rand(rng, 64:480))
        run!(g)
    end
    free!(g)
    return true
end

# ── 13. Locks and threads (THR) ──

# A:459, P:1910: declare on task A, first run (compile and record) on task B, run on task C, free on task D: exact, and the freed graph throws.
@case "THR-01" 3 begin
    dev = testdevice(); n = 1024
    out = MantleArray(dev, zeros(Int32, n))
    g = fetch(Threads.@spawn begin
        g = Graph(dev); t = MantleArray(g, Int32, n)
        dispatch!(g, thr_fill!, (t, Int32(3)), n)
        dispatch!(g, thr_copy!, (out, t), n)
        g
    end)
    fetch(Threads.@spawn run!(g))
    fetch(Threads.@spawn (dispatch!(dev, thr_fill!, (out, Int32(0)), n); run!(g)))
    @test Array(out) == fill(Int32(3), n)
    fetch(Threads.@spawn free!(g))
    @test_throws ArgumentError run!(g)
end

# P:1876-1878: test_interleaved_recording. Four tasks on one thread, then four on four threads, each declaring a graph and making eager calls with a yield after every call, 10^4 rounds in all: exact.
@case "THR-02" 3 :long begin
    dev = testdevice()
    for spawn in (f -> (@async f()), f -> (Threads.@spawn f()))
        tasks = [spawn(() -> thr_interleaved(dev, k, 2500)) for k in 1:4]
        @test all(withtimeout(() -> fetch.(tasks), 1200))
    end
end

# P:1879-1880: test_builder_is_a_value. Two builders on one channel, both open: no field of the device (its pool included) or of the channel references either.
@case "THR-03" 3 begin
    dev = testdevice()
    ch = Mantle.channel(dev, Mantle.Main())           # the channel eager calls use (plan, Eager work)
    b1, b2 = Mantle.Builder(dev, ch), Mantle.Builder(dev, ch)
    @test b1 !== b2
    @test !thr_reaches(dev, b1) && !thr_reaches(dev, b2)
    @test !thr_reaches(ch, b1) && !thr_reaches(ch, b2)
end

# P:738-740: task T1 waits for a 1.2 s GPU job (Array of its output); meanwhile GC.gc() on another task returns at once: the host wait is GC-safe.
@case "THR-05" 3 begin
    dev = testdevice()
    if Threads.nthreads() < 2
        @test_skip "needs two threads"
    else
        out = MantleArray(dev, Int32[0])
        slowwrite!(dev, out, 6; ms = 600); slowwrite!(dev, out, 7; ms = 600)
        waiter = Threads.@spawn Array(out)
        sleep(0.3)
        gctime = withtimeout(() -> @elapsed(GC.gc()), 30)
        @test !istaskdone(waiter)                     # the GC ran while T1 was in its wait
        @test gctime < 0.5
        @test fetch(waiter) == [7]
    end
end

# A:439, D:476-477: the lock tracker on over a workload of two graphs with host nodes sharing an array and an arena part, eager calls and host reads on two tasks: no violation, exact.
@case "THR-06" 3 begin
    dev = testdevice()
    Mantle.checklocks!(dev, true)
    exact = thr_workload(dev)
    Mantle.checklocks!(dev, false)
    @test exact
end

# A:421-433: fuzz, 8 tasks: six own a graph each (crossing reads and writes, host nodes, runs, frees and rebuilds, eager calls, stores), two make eager calls and stores, and from phase 7 one resizes a window between runs; lock tracker on: no deadlock, no violation.
@case "THR-07" 4 :long begin
    dev = testdevice(); n = 1024; iterations = 2000
    arrays = [MantleArray(dev, zeros(Int32, n)) for _ in 1:4]
    win = SUITE.phase[] >= 7 && has(dev, :window) ? Window(dev, 320, 240) : nothing
    Mantle.checklocks!(dev, true)
    tasks = [[Threads.@spawn(thr_fuzzgraphtask(dev, arrays, k, iterations, n)) for k in 1:6];
             [Threads.@spawn(thr_fuzzchangetask(dev, arrays, k, iterations, n)) for k in 7:8];
             win === nothing ? Task[] : [Threads.@spawn(thr_fuzzwindowtask(dev, win, iterations))]]
    @test all(withtimeout(() -> fetch.(tasks), 1800))
    Mantle.checklocks!(dev, false)
end

# P:980-981: two tasks each make 500 eager calls x = 3x + k on their own array through the one channel: each task's program order is kept.
@case "THR-08" 3 begin
    dev = testdevice(); n = 256; steps = 500
    xs = [MantleArray(dev, ones(Int32, n)) for _ in 1:2]
    tasks = [Threads.@spawn(thr_chain!(dev, x, steps)) for x in xs]
    withtimeout(() -> foreach(wait, tasks), 300)
    want = foldl((v, k) -> v * Int32(3) + Int32(k), 1:steps; init = Int32(1))
    @test all(x -> Array(x) == fill(want, n), xs)
end

# I:1019-1028: g's host function throws inside lockgraph: the exception propagates, the owner is cleared, so the same task declares into g and runs it again, and another task can run it.
@case "THR-09" 3 begin
    dev = testdevice(); n = 64
    out, again = MantleArray(dev, Int32, n), MantleArray(dev, Int32, n)
    fail = Ref(true)
    g = Graph(dev); t, u = MantleArray(g, Int32, n), MantleArray(g, Int32, n)
    dispatch!(g, thr_fill!, (t, Int32(4)), n)
    dispatch!(g, HostCall((src, dst) -> thr_hostfail(fail, src, dst)), (t, u); writes = (u,))
    dispatch!(g, thr_copy!, (out, u), n)
    @test_throws ErrorException run!(g)
    fail[] = false
    run!(g)
    @test Array(out) == fill(Int32(5), n)
    dispatch!(g, thr_copy!, (again, u), n)            # declaring works on the same task
    run!(g)
    @test Array(again) == fill(Int32(5), n)
    @test withtimeout(() -> fetch(Threads.@spawn (run!(g); true)), 30)
end

# I:244, I:237: a nested declaration, fill! of a graph array inside a repeat! body, reuses the graph lock: no self-deadlock; four iterations add four.
@case "THR-10" 1 begin
    dev = testdevice(); n = 64
    acc = MantleArray(dev, zeros(Int32, n))
    g = Graph(dev); t = MantleArray(g, Int32, n)
    withtimeout(30) do
        repeat!(g, 4) do i
            fill!(t, Int32(1))
            dispatch!(g, thr_accumulate!, (acc, t), n)
        end
    end
    run!(g)
    @test Array(acc) == fill(Int32(4), n)
end

# P:1301-1302, I:2100-2111: graph finalizers fire (GC.gc) while another task holds a's sync-state lock (its eager call blocked in submitnative! [FI]); the graphs named a, yet the finalizers only push to the inbox: no deadlock.
@case "THR-11" 3 begin
    d = FaultDevice(testdevice()); n = 1024
    a = MantleArray(d, zeros(Int32, n))
    thr_garbage(d, a, 8)
    Mantle.waitidle(d)
    gate = Channel{Nothing}(1)
    inject!(d, :submitnative!, Block(calls(d, :submitnative!) + 1, gate))
    writer = Threads.@spawn dispatch!(d, thr_fill!, (a, Int32(1)), n)
    sleep(0.3)
    @test !istaskdone(writer)                         # blocked, a's sync state locked
    withtimeout(() -> GC.gc(true), 30)
    @test !istaskdone(writer)
    put!(gate, nothing)
    withtimeout(() -> wait(writer), 30)
    @test Array(a) == ones(Int32, n)
    Mantle.sweep!(d)                                  # the releases the finalizers queued
end
