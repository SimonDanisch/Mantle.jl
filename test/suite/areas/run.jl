# Run semantics (catalogue-core.md, table 9: RUN). Contracts: api.md 2.3
# (run!), 4.2 (submitnative!), 5 and 7; recording-plan.md "Run" and
# "Ordering across graphs and queues"; internals.jl section 6 (run!(g),
# run!(pl), lockstage, submitstage!); resizing-and-raytracing.jl B.4
# (takepending, waitforruns).
#
# Queues and segments are core's decisions; cases that need two queues or
# many segments set the caps facts core decides from through FaultDevice
# [FI-10]: crossqueuecost 0 puts independent branches on a second compute
# queue, a bandwidth of 1 byte/s makes every kernel's estimate exceed the
# submission budget (one segment per kernel).

import KernelInterface as KI

"""a[i] = i."""
function run_iota!(a)
    i = KI.get_global_id().x
    i <= length(a) || return
    @inbounds a[i] = eltype(a)(i)
    return
end

"""dst[i] = src[i] + k, over the shorter of the two."""
function run_add!(dst, src, k)
    i = KI.get_global_id().x
    i <= min(length(dst), length(src)) || return
    @inbounds dst[i] = src[i] + k
    return
end

"""marks[i] = 1 for every i the launch over `a` reaches: the launch's extent."""
function run_mark!(marks, a)
    i = KI.get_global_id().x
    i <= length(a) || return
    @inbounds marks[i] = Int32(1)
    return
end

"""A graph of two kernels over a transient in the device-memory arena part: 1:n into the returned device array."""
function twokernels(dev, n)
    g = Graph(dev)
    t = MantleArray(g, Int32, n)
    out = MantleArray(dev, Int32, n)
    dispatch!(g, run_iota!, (t,), n)
    dispatch!(g, run_add!, (out, t, Int32(0)), n)
    return g, out
end

"""
`k` kernels in a chain over transients of `n` Int32: 1:n, then +1 per kernel
after the first; the last writes the returned device array, which then holds
(1:n) .+ (k - 1).
"""
function chaingraph(dev, n, k)
    g = Graph(dev)
    out = MantleArray(dev, Int32, n)
    prev = k == 1 ? out : MantleArray(g, Int32, n)
    dispatch!(g, run_iota!, (prev,), n)
    for j in 2:k
        next = j == k ? out : MantleArray(g, Int32, n)
        dispatch!(g, run_add!, (next, prev, Int32(1)), n)
        prev = next
    end
    return g, out
end

"""
A graph with one independent branch per counter: each reads its one-element
device counter into a transient and writes it back plus one, so every run
adds 1 to each counter.
"""
function countinggraph(dev, counters...; kw...)
    g = Graph(dev; kw...)
    for c in counters
        t = MantleArray(g, Int32, 1)
        dispatch!(g, run_add!, (t, c, Int32(1)), 1)
        dispatch!(g, run_add!, (c, t, Int32(0)), 1)
    end
    return g
end

"""`c` with the fields in `kw` replaced (caps(dev) is immutable)."""
withcaps(c; kw...) = typeof(c)((get(kw, f, getfield(c, f)) for f in fieldnames(typeof(c)))...)

"""A FaultDevice over the suite's device whose caps report the fields in `kw` instead [FI-10]."""
function capsdevice(; kw...)
    inner = testdevice()
    fd = FaultDevice(inner)
    inject!(fd, :caps, Answer(1, withcaps(caps(inner); kw...)))
    return fd
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

# submissions(g) lists stages (cut by host nodes), each with its segments (cut
# by budget and queue). api.md 2.3 leaves the shape open; this is the one the
# suite assumes.
segmentcount(g) = sum(s -> length(s.segments), Mantle.submissions(g))

"""The channels a run's submissions went to (diagnostics count them per channel)."""
channelsused(d) = count(>(0), values(d.submissions))

"""Julia bytes `k` steady-state runs of `g` allocate (in a function: no global is boxed into the count)."""
function runbytes(g, k)
    return @allocated for _ in 1:k
        run!(g)
    end
end

# ── 9. Run semantics (RUN) ──

# A:453, P:1895-1899: a graph that reads, increments and writes two device counters in two independent branches on two queues (crossqueuecost 0 [FI-10]), run 10 000 times from each of 2 threads with no host wait: both counters count every run. Negative control: the plan's test-only `ordering = Unordered()` keyword (test_crossplan_order.jl), assumed to drop the wait on the previous run too: the counts come out short.
# The counter is a device array, not a persistent graph array.
# pending decision 1
@case "RUN-01" 3 :multiqueue :long begin
    fd = capsdevice(; crossqueuecost = 0.0); runs = 10_000
    a = MantleArray(fd, Int32, 1)
    b = MantleArray(fd, Int32, 1)
    fill!(a, Int32(0))
    fill!(b, Int32(0))
    g = countinggraph(fd, a, b)
    run!(g)
    _, d = counted(fd) do
        ws = [Threads.@spawn(foreach(_ -> run!(g), 1:runs)) for _ in 1:2]
        foreach(w -> @test(thrownby(w; timeout = 1800) === nothing), ws)
    end
    @test channelsused(d) >= 2
    @test d.hostwaits == 0
    @test Array(a) == Int32[2runs + 1]
    @test Array(b) == Int32[2runs + 1]
    fill!(a, Int32(0))
    fill!(b, Int32(0))
    gu = countinggraph(fd, a, b; ordering = Mantle.Unordered())
    ws = [Threads.@spawn(foreach(_ -> run!(gu), 1:runs)) for _ in 1:2]
    foreach(w -> @test(thrownby(w; timeout = 1800) === nothing), ws)
    @test only(Array(a)) + only(Array(b)) < 4runs
end

# A:455, P:1891-1894: in steady state (after two runs), ten runs allocate nothing, compile nothing, record nothing, and submit once each (the graph is one segment).
@case "RUN-02" 3 begin
    dev = testdevice(); n = 1024
    g, out = twokernels(dev, n)
    run!(g)
    run!(g)
    _, d = counted(() -> foreach(_ -> run!(g), 1:10))
    @test d.allocations == 0
    @test d.plancompiles == 0
    @test d.pipelinecompiles == 0
    @test d.kernelcompiles == 0
    @test d.records == 0
    @test segmentcount(g) == 1
    @test submitcount(d) == 10
    @test Array(out) == Int32.(1:n)
end

# P:863-868: with nothing marked (no GPURef stored, nothing pending), two runs of a graph with a 400 ms kernel return while the GPU still works, with no host wait (the second does not wait for the first on the host).
@case "RUN-03" 3 begin
    dev = testdevice()
    g = Graph(dev)
    out = MantleArray(dev, Int32, 1)
    slowwrite!(g, out, 7; ms = 400)
    run!(g)
    Array(out)
    took = Ref(0.0)
    _, d = counted(() -> (took[] = @elapsed (run!(g); run!(g))))
    @test d.hostwaits == 0
    @test took[] < 0.3
    @test Array(out) == Int32[7]
end

# R:705-712: T2 makes the first resize! of a, which g launches over directly, while T1 calls run!(g): the run either finds its plan stale (Again) and recompiles inside run!, covering the new size, or runs the old plan and the next run recompiles; exact either way, one recompile in total. 20 races.
@case "RUN-04" 4 begin
    dev = testdevice(); n = 64
    for _ in 1:20
        a = MantleArray(dev, Int32, n)
        marks = MantleArray(dev, Int32, 2n)
        g = Graph(dev)
        dispatch!(g, run_mark!, (marks, a), launchrange(a))
        run!(g)
        fill!(marks, Int32(0))
        (raced, after), d = counted() do
            t2 = Threads.@spawn resize!(a, 2n)
            t1 = Threads.@spawn run!(g)
            @test thrownby(t2) === nothing
            @test thrownby(t1) === nothing
            seen = sum(Array(marks))
            fill!(marks, Int32(0))
            run!(g)
            (seen, sum(Array(marks)))
        end
        @test raced in (n, 2n)
        @test after == 2n
        @test d.plancompiles == 1
    end
end

# I:1015-1021: before one run!, the window is resized, the plan is made stale (first resize! of a, which it launches over directly) and its arena pages are unmapped (budget lowered [FI-9], an allocation): run! converges and returns Submitted after one recompile, exact.
@case "RUN-05" 7 :window :sparse begin
    fd = FaultDevice(testdevice()); n = 64
    win = Window(fd, 64, 64)
    g = Graph(fd)
    render!(g, win => Clear((0f0, 0f0, 0f0, 1f0))) do p
    end
    a = MantleArray(fd, Int32, n)
    marks = MantleArray(fd, Int32, 4n)
    dispatch!(g, run_mark!, (marks, a), launchrange(a))
    t = MantleArray(g, Int32, 2^22)
    out = MantleArray(fd, Int32, 16)
    dispatch!(g, run_iota!, (t,), 2^22)
    dispatch!(g, run_add!, (out, t, Int32(0)), 16)
    run!(g)
    m1 = memory(fd)
    resize!(win, 80, 60)
    resize!(a, 2n)
    inject!(fd, :budget, Answer(calls(fd, :budget) + 1, m1.reserved))
    ballast = MantleArray(fd, UInt8, 3 * 2^22)
    @test Mantle.mapped(fd) < m1.mapped
    free!(ballast)
    memory(fd)
    fill!(marks, Int32(0))
    fill!(out, Int32(0))
    r, d = counted(() -> run!(g), fd)
    @test r isa Mantle.Submitted
    @test d.plancompiles == 1
    @test sum(Array(marks)) == 2n
    @test Array(out) == Int32.(1:16)
end

# P:712-715: the next rawalloc throws [FI-2] while a run's prepare! allocates for a pending resize! of a beyond its storage: the error comes out of run! before any token is claimed (no hook shows tokens; later host waits all return); with rawalloc working again, the next run applies the resize and is exact, a's contents kept.
@case "RUN-06" 4 begin
    fd = FaultDevice(testdevice()); n = 64
    data = testdata(Int32, n)
    a = MantleArray(fd, Int32, n; capacity = 2n)
    copyto!(a, data)
    out = MantleArray(fd, Int32, n)
    g = Graph(fd)
    dispatch!(g, run_add!, (out, a, Int32(0)), n)
    run!(g)
    inject!(fd, :rawalloc, Throw(calls(fd, :rawalloc) + 1, OutOfMemoryError()))
    resize!(a, 2^26)
    e = thrown(() -> run!(g))
    @test e isa OutOfMemoryError || e isa Mantle.OutOfDeviceMemory
    @test withtimeout(() -> Array(out), 30) == data
    withtimeout(() -> run!(g), 60)
    @test Array(out) == data
    @test Array(view(a, 1:n)) == data
    @test length(a) == 2^26
end

# P:712-715, A:281: submitnative! throws at g's next submission [FI-6]: the error comes out of run!; every lock and the arena part are released: another task's eager call on g's output, a graph sharing the part, and g's next run all finish, exact.
@case "RUN-07" 3 begin
    fd = FaultDevice(testdevice()); n = 64
    g, out = twokernels(fd, n)
    run!(g)
    failsubmit!(fd, calls(fd, :submitnative!) + 1, ErrorException("device lost (injected)"))
    @test thrown(() -> run!(g)) isa ErrorException
    @test thrown(() -> fill!(out, Int32(0))) === nothing
    g2, out2 = twokernels(fd, n)
    @test thrown(() -> run!(g2)) === nothing
    @test thrown(() -> run!(g)) === nothing
    @test Array(out) == Int32.(1:n)
    @test Array(out2) == Int32.(1:n)
end

# P:928-930: a stale plan (first resize! of a, which g launches over directly) is recompiled and retried inside run!, never thrown. With a window (phase 7, a device with a display), a window resized from another task during 50 runs is applied and retried inside run!: every run returns Submitted. OUT_OF_DATE itself comes from the driver and has no FaultDevice verb (acquirenext takes the swapchain); the resize race takes the same retry path (the window changed, Again).
@case "RUN-08" 4 begin
    dev = testdevice(); n = 64
    a = MantleArray(dev, Int32, n)
    marks = MantleArray(dev, Int32, 4n)
    g = Graph(dev)
    dispatch!(g, run_mark!, (marks, a), launchrange(a))
    run!(g)
    fill!(marks, Int32(0))
    resize!(a, 2n)
    @test thrown(() -> run!(g)) === nothing
    @test sum(Array(marks)) == 2n
    if SUITE.phase[] >= 7 && has(dev, :window)
        win = Window(dev, 64, 64)
        gw = Graph(dev)
        render!(gw, win => Clear((0f0, 0f0, 0f0, 1f0))) do p
        end
        run!(gw)
        resizer = Threads.@spawn for k in 1:50
            resize!(win, 64 + k, 64 + k)
            sleep(0.01)
        end
        results = withtimeout(() -> [run!(gw) for _ in 1:50], 120)
        @test thrownby(resizer) === nothing
        @test all(r -> r isa Mantle.Submitted, results)
    end
end

# A:101: run!(g) from a task that holds an arena part of g's plan (the host function of h, which shares g's device part) throws.
@case "RUN-09" 3 begin
    dev = testdevice(); n = 64
    g, out = twokernels(dev, n)
    run!(g)
    h, hout = twokernels(dev, n)
    t = MantleArray(h, Int32, 1)
    dispatch!(h, run_iota!, (t,), 1)
    dispatch!(h, HostCall(_ -> run!(g)), (t,); writes = ())
    @test thrown(() -> run!(h)) isa ArgumentError
end

# P:1884-1885: an encoder and a decoder graph (SAM 2's shape: several kernels each, a device array between them), 20 runs each: every plan is recorded in its first run only. SAM 2 itself runs in the dnn_patterns area.
@case "RUN-10" 1 begin
    dev = testdevice(); n = 4096
    feats = MantleArray(dev, Int32, n)
    out = MantleArray(dev, Int32, n)
    enc = Graph(dev)
    t = MantleArray(enc, Int32, n)
    dispatch!(enc, run_iota!, (t,), n)
    dispatch!(enc, run_add!, (feats, t, Int32(1)), n)
    dec = Graph(dev)
    u = MantleArray(dec, Int32, n)
    dispatch!(dec, run_add!, (u, feats, Int32(1)), n)
    dispatch!(dec, run_add!, (out, u, Int32(1)), n)
    _, d1 = counted(() -> (run!(enc); run!(dec)))
    @test d1.records >= 2
    _, later = counted(() -> foreach(_ -> (run!(enc); run!(dec)), 2:20))
    @test later.records == 0
    @test Array(out) == Int32.(4:n+3)
end

# D:498-499: two independent branches on two queues (crossqueuecost 0 [FI-10]) both read w; a store into w before each run is applied in the first segment's change prefix, and the segment on the other queue waits for that segment's token: both branches see the new data in each of 100 runs.
@case "RUN-11" 4 :multiqueue begin
    fd = capsdevice(; crossqueuecost = 0.0); n = 4096
    w = MantleArray(fd, Int32, n)
    fill!(w, Int32(0))
    outa = MantleArray(fd, Int32, n)
    outb = MantleArray(fd, Int32, n)
    g = Graph(fd)
    ta = MantleArray(g, Int32, n)
    tb = MantleArray(g, Int32, n)
    dispatch!(g, run_add!, (ta, w, Int32(1)), n)
    dispatch!(g, run_add!, (outa, ta, Int32(0)), n)
    dispatch!(g, run_add!, (tb, w, Int32(2)), n)
    dispatch!(g, run_add!, (outb, tb, Int32(0)), n)
    run!(g)
    _, d = counted(() -> run!(g), fd)
    @test channelsused(d) >= 2
    wrong = 0
    for s in 1:100
        data = testdata(Int32, n; seed = s) .% Int32(10^6)
        w[1:n] = data
        run!(g)
        wrong += (Array(outa) != data .+ Int32(1)) + (Array(outb) != data .+ Int32(2))
    end
    @test wrong == 0
end

# P:645-651: two threads each store into w and run g (branches on two queues, crossqueuecost 0 [FI-10]) 200 times: per channel, the tokens of the applied change steps never go back in emission order (claim and submit in one step under the channel lock); results exact. "Validation clean" has no hook in this suite; the validation-layer runs check it.
@case "RUN-12" 4 :multiqueue begin
    fd = capsdevice(; crossqueuecost = 0.0); n = 1024
    data = Int32.(1:n)
    w = MantleArray(fd, Int32, n)
    fill!(w, Int32(0))
    outa = MantleArray(fd, Int32, n)
    outb = MantleArray(fd, Int32, n)
    g = Graph(fd)
    dispatch!(g, run_add!, (outa, w, Int32(1)), n)
    dispatch!(g, run_add!, (outb, w, Int32(2)), n)
    run!(g)
    Mantle.steplog!(fd, true)
    ws = [Threads.@spawn(foreach(_ -> (w[1:n] = data; run!(g)), 1:200)) for _ in 1:2]
    foreach(t -> @test(thrownby(t; timeout = 300) === nothing), ws)
    log = Mantle.steplog(fd)
    Mantle.steplog!(fd, false)
    bychannel = Dict{Any,Vector{Any}}()
    for (_, _, _, token) in log
        push!(get!(Vector{Any}, bychannel, token.channel), token.value)
    end
    @test !isempty(bychannel)
    @test all(issorted, values(bychannel))
    @test Array(outa) == data .+ Int32(1)
    @test Array(outb) == data .+ Int32(2)
end

# P:676-685: g is between its stages (blocked in its host function) and launches over a directly; T2 resizes a and fills it eagerly: the fill waits until g has submitted its last stage, then applies the resize; g's stage 2 used the old size; g's next run recompiles once and covers the new size.
@case "RUN-13" 5 begin
    dev = testdevice(); n = 64
    a = MantleArray(dev, Int32, n)
    marks = MantleArray(dev, Int32, 4n)
    fill!(marks, Int32(0))
    gate = Gate()
    g = Graph(dev)
    t = MantleArray(g, Int32, 1)
    dispatch!(g, run_iota!, (t,), 1)
    dispatch!(g, blockinghost(gate), (t,); writes = ())
    dispatch!(g, run_mark!, (marks, a), launchrange(a))
    t1 = Threads.@spawn run!(g)
    waitentered(gate)
    t2 = Threads.@spawn (resize!(a, 2n); fill!(a, Int32(5)))
    sleep(0.5)
    @test !istaskdone(t2)
    letgo!(gate)
    @test thrownby(t1) === nothing
    @test thrownby(t2) === nothing
    @test sum(Array(marks)) == n
    @test Array(a) == fill(Int32(5), 2n)
    fill!(marks, Int32(0))
    letgo!(gate)                     # the next run's host function passes the gate at once
    _, d = counted(() -> withtimeout(() -> run!(g), 30))
    @test d.plancompiles == 1
    @test sum(Array(marks)) == 2n
end

# P:679-680: as RUN-13 with a single-stage g (a 400 ms kernel beside the launch over a), in flight: g is never between stages, so the eager fill after resize!(a) makes no host wait (it waits for g's tokens on the GPU); g's run used the old size.
@case "RUN-14" 5 begin
    dev = testdevice(); n = 64
    a = MantleArray(dev, Int32, n)
    marks = MantleArray(dev, Int32, 4n)
    busy = MantleArray(dev, Int32, 1)
    g = Graph(dev)
    slowwrite!(g, busy, 1; ms = 400)
    dispatch!(g, run_mark!, (marks, a), launchrange(a))
    run!(g)
    Array(marks)
    fill!(marks, Int32(0))
    run!(g)
    resize!(a, 2n)
    _, d = counted(() -> fill!(a, Int32(5)))
    @test d.hostwaits == 0
    @test sum(Array(marks)) == n
    @test Array(a) == fill(Int32(5), 2n)
end

# P:684-685: g (two stages around a no-op host node, a launch over a directly) runs in a tight loop on T1 while T2 resizes a and fills it eagerly: T2 proceeds once g's current run has submitted its last stage, since g's next run recompiles to an indirect launch the change no longer stales. No fairness is promised; the bound this case holds T2 to is 30 s, and the time taken is logged.
@case "RUN-15" 5 begin
    dev = testdevice(); n = 64
    a = MantleArray(dev, Int32, n)
    marks = MantleArray(dev, Int32, 4n)
    g = Graph(dev)
    t = MantleArray(g, Int32, 1)
    dispatch!(g, run_iota!, (t,), 1)
    dispatch!(g, HostCall(_ -> nothing), (t,); writes = ())
    dispatch!(g, run_mark!, (marks, a), launchrange(a))
    run!(g)
    stop = Threads.Atomic{Bool}(false)
    loop = Threads.@spawn while !stop[]
        run!(g)
    end
    sleep(0.2)
    took = Ref(0.0)
    @test thrown(() -> (took[] = @elapsed (resize!(a, 2n); fill!(a, Int32(5)))); timeout = 30) === nothing
    stop[] = true
    @test thrownby(loop) === nothing
    @test Array(a) == fill(Int32(5), 2n)
    @info "RUN-15: resize! and fill! behind a tight run loop took $(round(took[]; digits = 3)) s"
end

# R:460 vs A:487 (ambiguity 9): an eager call inside an insert!(g) block (a declaration, not a run) whose change would stale g2, which is between its stages: it waits until g2 has submitted its last stage, then proceeds. The current internals agree with api.md: a declaration holding a graph lock is not inside a run (R:481-485).
@case "RUN-16" 5 begin
    dev = testdevice(); n = 64
    a = MantleArray(dev, Int32, n)
    marks = MantleArray(dev, Int32, 4n)
    fill!(marks, Int32(0))
    gate = Gate()
    g2 = Graph(dev)
    t = MantleArray(g2, Int32, 1)
    dispatch!(g2, run_iota!, (t,), 1)
    dispatch!(g2, blockinghost(gate), (t,); writes = ())
    dispatch!(g2, run_mark!, (marks, a), launchrange(a))
    g, out = twokernels(dev, n)
    run!(g)
    t2 = Threads.@spawn run!(g2)
    waitentered(gate)
    resize!(a, 2n)
    t1 = Threads.@spawn insert!(g) do
        fill!(a, Int32(5))
    end
    sleep(0.5)
    @test !istaskdone(t1)
    letgo!(gate)
    @test thrownby(t2) === nothing
    @test thrownby(t1) === nothing
    @test sum(Array(marks)) == n
    @test Array(a) == fill(Int32(5), 2n)
end

# P:712-715: no allocation between claiming tokens and submitting (submitnative!, bindpages!). No hook counts allocations inside a verb; the closest observable: with a bandwidth of 1 byte/s [FI-10] every kernel is its own segment, and 200 steady-state runs of an 8-segment graph allocate no more Julia memory than those of a 1-segment graph (the per-segment path, submitnative! included, allocates nothing).
@case "RUN-17" 3 begin
    fd = capsdevice(; bandwidth = 1.0); n = 256
    g1, out1 = chaingraph(fd, n, 1)
    g8, out8 = chaingraph(fd, n, 8)
    foreach(g -> (run!(g); run!(g)), (g1, g8))
    @test segmentcount(g8) == 8
    b1 = runbytes(g1, 200)
    b8 = runbytes(g8, 200)
    @test b8 <= b1 + 1024
    @test Array(out8) == Int32.(8:n+7)
end
