# Host nodes and stages (catalogue-core.md, table 7: HOST). Contracts:
# api.md 2.4 (HostCall), 5 and 7; recording-plan.md "Host nodes" and
# "Ordering across graphs and queues"; internals.jl section 4 (hostcuts!,
# submissions!) and section 6 (run!, lockgraph, holdparts!, runhost!).
#
# Arena parts in these cases: graphs on one device share the device-memory
# arena part by default (plan, Memory: separate parts only for graphs core
# expects to overlap, by priority). A transient no host node touches lives
# in that part; one a host node reads or writes lives in the host-visible
# part. `hostdevicepart!` gives a graph a tenancy in the device-memory part.

import KernelInterface as KI

"""a[i] = i."""
function host_iota!(a)
    i = KI.get_global_id().x
    i <= length(a) || return
    @inbounds a[i] = eltype(a)(i)
    return
end

"""dst[i] = src[i] + k, over the shorter of the two."""
function host_add!(dst, src, k)
    i = KI.get_global_id().x
    i <= min(length(dst), length(src)) || return
    @inbounds dst[i] = src[i] + k
    return
end

"""dst[i] = a[i] + b[i]."""
function host_sum!(dst, a, b)
    i = KI.get_global_id().x
    i <= length(dst) || return
    @inbounds dst[i] = a[i] + b[i]
    return
end

"""c[1] += 1, one thread."""
function host_increment!(c)
    @inbounds c[1] += one(eltype(c))
    return
end

"""marks[i] = 1 for every i the launch over `a` reaches: the launch's extent."""
function host_mark!(marks, a)
    i = KI.get_global_id().x
    i <= length(a) || return
    @inbounds marks[i] = Int32(1)
    return
end

"""Clears texel (i, j) of the window `w`; texel (1, 1) also copies src[1] into dst[1]."""
function host_paint!(w, src, dst)
    i = KI.get_global_id().x
    j = KI.get_global_id().y
    (i <= size(w, 1) && j <= size(w, 2)) || return
    @inbounds w[i, j] = zero(eltype(w))
    i == 1 && j == 1 && (@inbounds dst[1] = src[1])
    return
end

"""
Declares into `g` two kernels over a transient no host node touches, so the
plan holds the device-memory arena part. The second writes 1:n into the
returned device array.
"""
function hostdevicepart!(g, dev, n = 64)
    t = MantleArray(g, Int32, n)
    out = MantleArray(dev, Int32, n)
    dispatch!(g, host_iota!, (t,), n)
    dispatch!(g, host_add!, (out, t, Int32(0)), n)
    return out
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
stagecount(g) = length(Mantle.submissions(g))

# ── 7. Host nodes and stages (HOST) ──

# A:128, P:914-919: K1 writes src, HostCall(f) writes reverse(src) into idx, K2 copies idx into a device array: exact; f runs on run!'s task; 2 stages, one submission each.
@case "HOST-01" 3 begin
    dev = testdevice(); n = 256
    g = Graph(dev)
    src = MantleArray(g, Int32, n)
    idx = MantleArray(g, Int32, n)
    out = MantleArray(dev, Int32, n)
    caller = Ref{Any}(nothing)
    dispatch!(g, host_iota!, (src,), n)
    dispatch!(g, HostCall((s, x) -> (caller[] = current_task(); x .= reverse(s))), (src, idx); writes = (idx,))
    dispatch!(g, host_add!, (out, idx, Int32(0)), n)
    run!(g)
    @test Array(out) == Int32.(n:-1:1)
    @test caller[] === current_task()
    @test stagecount(g) == 2
    _, d = counted(() -> run!(g))
    @test submitcount(d) == 2
    @test Array(out) == Int32.(n:-1:1)
end

# P:500, I:584-588: K1, H1, a kernel that depends on neither host node, H2, K2: hostcuts! moves the kernel across and merges H1 and H2 into one cut (2 stages, one host wait per run); H1 runs before H2; exact.
@case "HOST-02" 3 begin
    dev = testdevice(); n = 128
    g = Graph(dev)
    a = MantleArray(g, Int32, n)
    b = MantleArray(g, Int32, n)
    c = MantleArray(g, Int32, n)
    out = MantleArray(dev, Int32, n)
    side = MantleArray(dev, Int32, n)
    order = Int[]
    dispatch!(g, host_iota!, (a,), n)
    dispatch!(g, HostCall((x, y) -> (push!(order, 1); y .= x .* Int32(2))), (a, b); writes = (b,))
    dispatch!(g, host_iota!, (side,), n)
    dispatch!(g, HostCall((x, y) -> (push!(order, 2); y .= x .+ Int32(1))), (b, c); writes = (c,))
    dispatch!(g, host_add!, (out, c, Int32(0)), n)
    run!(g)
    @test order == [1, 2]
    @test Array(out) == Int32.(2 .* (1:n) .+ 1)
    @test Array(side) == Int32.(1:n)
    @test stagecount(g) == 2
    _, d = counted(() -> run!(g))
    @test submitcount(d) == 2
    @test d.hostwaits == 1
end

# I:1328-1331: a slow kernel (about 300 ms) in stage 1 writes 42; the host function after it reads 42, in every run.
@case "HOST-03" 3 begin
    dev = testdevice()
    g = Graph(dev)
    flag = MantleArray(g, Int32, 1)
    seen = Ref{Int32}(0)
    slowwrite!(g, flag, 42; ms = 300)
    dispatch!(g, HostCall(v -> (seen[] = v[1])), (flag,); writes = ())
    for _ in 1:3
        seen[] = 0
        run!(g)
        @test seen[] == 42
    end
end

# A:128, P:920-922: the host function throws: the exception comes out of run!, stage 2 is not submitted, no arena part stays held (a graph sharing the device part runs), and the next run is exact.
@case "HOST-04" 3 begin
    dev = testdevice(); n = 64
    fail = Ref(false)
    g = Graph(dev)
    devout = hostdevicepart!(g, dev, n)
    t = MantleArray(g, Int32, n)
    out = MantleArray(dev, Int32, n)
    dispatch!(g, host_iota!, (t,), n)
    dispatch!(g, HostCall(x -> fail[] && error("host function failed")), (t,); writes = ())
    dispatch!(g, host_add!, (out, t, Int32(0)), n)
    run!(g)
    fill!(out, Int32(0))
    fail[] = true
    Mantle.resetdiagnostics!(dev)
    @test_throws ErrorException run!(g)
    @test submitcount(Mantle.diagnostics(dev)) == 1
    @test all(iszero, Array(out))
    g2 = Graph(dev)
    out2 = hostdevicepart!(g2, dev, n)
    withtimeout(() -> run!(g2), 30)
    @test Array(out2) == Int32.(1:n)
    fail[] = false
    withtimeout(() -> run!(g), 30)
    @test Array(out) == Int32.(1:n)
    @test Array(devout) == Int32.(1:n)
end

# A:473: the host function of g calls run!(g): it throws (the graph lock is not reentrant), no deadlock.
@case "HOST-05" 3 begin
    dev = testdevice()
    g = Graph(dev)
    t = MantleArray(g, Int32, 16)
    dispatch!(g, host_iota!, (t,), 16)
    dispatch!(g, HostCall(_ -> run!(g)), (t,); writes = ())
    @test thrown(() -> run!(g)) isa ArgumentError
end

# A:473, I:994: the host function of g runs g2, which shares g's device arena part (both compiled): lockgraph(g2) throws; g2 stays usable.
@case "HOST-06" 3 begin
    dev = testdevice(); n = 64
    g2 = Graph(dev)
    out2 = hostdevicepart!(g2, dev, n)
    run!(g2)
    g = Graph(dev)
    hostdevicepart!(g, dev, n)
    t = MantleArray(g, Int32, n)
    dispatch!(g, host_iota!, (t,), n)
    dispatch!(g, HostCall(_ -> run!(g2)), (t,); writes = ())
    @test thrown(() -> run!(g)) isa ArgumentError
    fill!(out2, Int32(0))
    withtimeout(() -> run!(g2), 30)
    @test Array(out2) == Int32.(1:n)
end

# P:694-695: the host function of g runs g3, which has no transients and so no arena part to share: allowed; both graphs exact.
@case "HOST-07" 3 begin
    dev = testdevice(); n = 64
    count = MantleArray(dev, Int32, 1)
    fill!(count, Int32(0))
    g3 = Graph(dev)
    dispatch!(g3, host_increment!, (count,), 1)
    run!(g3)
    g = Graph(dev)
    out = hostdevicepart!(g, dev, n)
    t = MantleArray(g, Int32, n)
    dispatch!(g, host_iota!, (t,), n)
    dispatch!(g, HostCall(_ -> run!(g3)), (t,); writes = ())
    @test thrown(() -> run!(g)) === nothing
    @test Array(out) == Int32.(1:n)
    @test Array(count) == Int32[2]
end

# A:437, I:1268-1289: the host function of g (holding the device part) runs g4, never compiled, which will use the same part: lockgraph passes (g4 has no plan yet), holdparts! would wait for a part of equal id the task holds itself and throws; afterwards no part is held.
# The catalogue's form (part 5 held, part 3 held by T2's run) needs part ids, which no hook shows; this is the equal-id case api.md 5 pins ("higher or equal", ambiguity 10).
@case "HOST-08" 3 begin
    dev = testdevice(); n = 64
    g4 = Graph(dev)
    out4 = hostdevicepart!(g4, dev, n)
    g = Graph(dev)
    hostdevicepart!(g, dev, n)
    t = MantleArray(g, Int32, n)
    dispatch!(g, host_iota!, (t,), n)
    dispatch!(g, HostCall(_ -> run!(g4)), (t,); writes = ())
    @test thrown(() -> run!(g)) isa ArgumentError
    g5 = Graph(dev)
    out5 = hostdevicepart!(g5, dev, n)
    withtimeout(() -> run!(g5), 30)
    withtimeout(() -> run!(g4), 30)
    @test Array(out5) == Int32.(1:n)
    @test Array(out4) == Int32.(1:n)
end

# P:662-664: the host function stores into the device array src (src[1:n] = 2t); stage 2 copies src: it sees the store.
@case "HOST-09" 4 begin
    dev = testdevice(); n = 128
    src = MantleArray(dev, Int32, n)
    fill!(src, Int32(0))
    out = MantleArray(dev, Int32, n)
    g = Graph(dev)
    t = MantleArray(g, Int32, n)
    dispatch!(g, host_iota!, (t,), n)
    dispatch!(g, HostCall(x -> (src[1:n] = x .* Int32(2))), (t,); writes = ())
    dispatch!(g, host_add!, (out, src, Int32(0)), n)
    run!(g)
    @test Array(out) == Int32.(2:2:2n)
    @test Array(src) == Int32.(2:2:2n)
end

# P:671-675, R:343-351: the host function resizes a, which stage 2 launches over directly: the call only enqueues (size(a) is the new one, a change pending), stage 2 launches over the old size, and the next run recompiles once and covers the new size.
@case "HOST-10" 4 begin
    dev = testdevice(); n = 64
    a = MantleArray(dev, Int32, n)
    marks = MantleArray(dev, Int32, 4n)
    fill!(marks, Int32(0))
    sz = Ref((0,)); pend = Ref(false)
    g = Graph(dev)
    t = MantleArray(g, Int32, 1)
    dispatch!(g, host_iota!, (t,), 1)
    dispatch!(g, HostCall(_ -> (resize!(a, 2n); sz[] = size(a); pend[] = !isempty(Mantle.pendingof(a)))), (t,); writes = ())
    dispatch!(g, host_mark!, (marks, a), launchrange(a))
    run!(g)
    @test sz[] == (2n,)
    @test pend[]
    @test sum(Array(marks)) == n
    fill!(marks, Int32(0))
    _, d = counted(() -> run!(g))
    @test d.plancompiles == 1
    @test sum(Array(marks)) == 2n
end

# D:500-505, R:449-451: the host function resizes a (which stage 2 launches over directly) and then makes an eager call on a: applying the resize would stale its own running plan, so the eager call throws (inside a host function), and run! with it; the next run recompiles and is exact.
@case "HOST-11" 5 begin
    dev = testdevice(); n = 64
    a = MantleArray(dev, Int32, n)
    marks = MantleArray(dev, Int32, 4n)
    fill!(marks, Int32(0))
    g = Graph(dev)
    t = MantleArray(g, Int32, 1)
    dispatch!(g, host_iota!, (t,), 1)
    dispatch!(g, HostCall(_ -> (resize!(a, 2n); fill!(a, Int32(7)))), (t,); writes = ())
    dispatch!(g, host_mark!, (marks, a), launchrange(a))
    @test thrown(() -> run!(g)) isa ArgumentError
    fill!(marks, Int32(0))
    withtimeout(() -> run!(g), 30)
    @test sum(Array(marks)) == 2n
end

# P:1921-1928: while g's host function blocks, T2 runs g2 (shares g's device part) and T3 resizes b, which g's stage 2 reads over launchrange(b): resize! returns at once, g2 waits until g has submitted its last stage, and all results are exact.
@case "HOST-12" 4 begin
    dev = testdevice(); n = 64
    data = testdata(Int32, n)
    b = MantleArray(dev, Int32, n; capacity = 4n)
    copyto!(b, data)
    out = MantleArray(dev, Int32, 4n)
    fill!(out, Int32(0))
    gate = Gate()
    g = Graph(dev)
    devout = hostdevicepart!(g, dev, n)
    t = MantleArray(g, Int32, 1)
    dispatch!(g, host_iota!, (t,), 1)
    dispatch!(g, blockinghost(gate), (t,); writes = ())
    dispatch!(g, host_add!, (out, b, Int32(0)), launchrange(b))
    g2 = Graph(dev)
    out2 = hostdevicepart!(g2, dev, n)
    run!(g2)
    fill!(out2, Int32(0))
    t1 = Threads.@spawn run!(g)
    waitentered(gate)
    t2 = Threads.@spawn run!(g2)
    withtimeout(() -> resize!(b, 2n), 10)
    sleep(0.5)
    @test !istaskdone(t2)
    letgo!(gate)
    @test thrownby(t1) === nothing
    @test thrownby(t2) === nothing
    @test Array(out)[1:n] == data
    @test Array(devout) == Int32.(1:n)
    @test Array(out2) == Int32.(1:n)
    @test length(b) == 2n
end

# P:1044-1045, I:660-667: K1 writes the window and c, a host node reads c, K2 writes the window from the host node's output, then the present: the host node sits between the first write of the swapchain image and its present and cannot be moved out, a compile error raised by the first run! (ambiguity 27). The window's device-side element type is open (G9): the kernel writes zero texels.
@case "HOST-13" 7 :window begin
    dev = testdevice()
    win = Window(dev, 64, 64)
    g = Graph(dev)
    s = MantleArray(g, Int32, 1)
    c = MantleArray(g, Int32, 1)
    d = MantleArray(g, Int32, 1)
    e = MantleArray(g, Int32, 1)
    dispatch!(g, host_iota!, (s,), 1)
    dispatch!(g, host_paint!, (win, s, c), (64, 64))
    dispatch!(g, HostCall((x, y) -> (y .= x)), (c, d); writes = (d,))
    dispatch!(g, host_paint!, (win, d, e), (64, 64))
    @test_throws ArgumentError run!(g)
end

# A:128: a HostCall inside a repeat! body and inside a when! block throws at declaration; neither leaves a node behind (the graph runs exact afterwards, the host functions never ran).
@case "HOST-14" 1 begin
    dev = testdevice(); n = 16
    g = Graph(dev)
    t = MantleArray(g, Int32, n)
    out = MantleArray(dev, Int32, n)
    flag = MantleArray(dev, Int32[1])
    calls = Ref(0)
    @test_throws ArgumentError repeat!(g, 3) do i
        dispatch!(g, HostCall(x -> (calls[] += 1)), (t,); writes = ())
    end
    @test_throws ArgumentError when!(g, flag) do
        dispatch!(g, HostCall(x -> (calls[] += 1)), (t,); writes = ())
    end
    dispatch!(g, host_iota!, (t,), n)
    dispatch!(g, host_add!, (out, t, Int32(0)), n)
    run!(g)
    @test Array(out) == Int32.(1:n)
    @test calls[] == 0
end

# I:648: a graph whose first node is a host node, and one whose last node is one: both run, and the head of the stage after the cut binds the device array w (exact, over two runs each).
@case "HOST-15" 3 begin
    dev = testdevice(); n = 64
    w = MantleArray(dev, Int32.(1:n))
    out = MantleArray(dev, Int32, n)
    gfirst = Graph(dev)
    x = MantleArray(gfirst, Int32, n)
    dispatch!(gfirst, HostCall(v -> (v .= Int32(10))), (x,); writes = (x,))
    dispatch!(gfirst, host_sum!, (out, x, w), n)
    glast = Graph(dev)
    y = MantleArray(glast, Int32, n)
    seen = Ref(Int32[])
    dispatch!(glast, host_add!, (y, w, Int32(1)), n)
    dispatch!(glast, HostCall(v -> (seen[] = copy(v))), (y,); writes = ())
    for _ in 1:2
        fill!(out, Int32(0))
        run!(gfirst)
        @test Array(out) == Int32.(11:n+10)
        seen[] = Int32[]
        run!(glast)
        @test seen[] == Int32.(2:n+1)
    end
end

# I:589-591: the host function reads a device array in normal memory (copied into host-visible memory before the cut) and one in Readback memory (read in place, no copy); it sees the data in both. The copy shows in peakbytes: its host-visible target is a transient, and the Readback graph has no transient at all.
@case "HOST-16" 3 begin
    dev = testdevice(); n = 1024
    plain = MantleArray(dev, Int32, n)
    rb = MantleArray(dev, Int32, n; memory = Mantle.Readback())
    seen = Dict{Symbol,Vector{Int32}}()
    gn = Graph(dev)
    dispatch!(gn, host_iota!, (plain,), n)
    dispatch!(gn, HostCall(v -> (seen[:plain] = copy(v))), (plain,); writes = ())
    gr = Graph(dev)
    dispatch!(gr, host_iota!, (rb,), n)
    dispatch!(gr, HostCall(v -> (seen[:readback] = copy(v))), (rb,); writes = ())
    run!(gn)
    run!(gr)
    @test seen[:plain] == Int32.(1:n)
    @test seen[:readback] == Int32.(1:n)
    @test Mantle.peakbytes(gn) >= sizeof(Int32) * n
    @test Mantle.peakbytes(gr) == 0
end

# P:1925-1928: g's host function runs g2 while T3's run of g2 holds g2's lock and waits for the device part g's run holds, and T2 waits for g2's lock: lockgraph throws for the host function at once; nobody deadlocks; g2's runs complete afterwards.
@case "HOST-17" 3 begin
    dev = testdevice(); n = 64
    g2 = Graph(dev)
    out2 = hostdevicepart!(g2, dev, n)
    run!(g2)
    fill!(out2, Int32(0))
    gate = Gate()
    g = Graph(dev)
    hostdevicepart!(g, dev, n)
    t = MantleArray(g, Int32, 1)
    dispatch!(g, host_iota!, (t,), 1)
    dispatch!(g, HostCall(_ -> (put!(gate.entered, nothing); take!(gate.release); run!(g2))), (t,); writes = ())
    t1 = Threads.@spawn run!(g)
    waitentered(gate)
    t3 = Threads.@spawn run!(g2)
    sleep(0.2)
    t2 = Threads.@spawn run!(g2)
    sleep(0.3)
    @test !istaskdone(t3)
    @test !istaskdone(t2)
    letgo!(gate)
    @test thrownby(t1) isa ArgumentError
    @test thrownby(t3) === nothing
    @test thrownby(t2) === nothing
    @test Array(out2) == Int32.(1:n)
end

# P:925-926, I:986-990: the host function starts a task that runs g and waits for it. The inner run! waits for g's lock, which the outer run holds: an unbounded wait would deadlock, undetected by Mantle. The host function waits 2 s only; the inner run completes after the outer one.
@case "HOST-18" 3 begin
    dev = testdevice(); n = 64
    inner = Ref{Any}(nothing)
    blocked = Ref(false)
    g = Graph(dev)
    out = hostdevicepart!(g, dev, n)
    t = MantleArray(g, Int32, 1)
    dispatch!(g, host_iota!, (t,), 1)
    startinner = function (_)
        inner[] === nothing || return
        inner[] = Threads.@spawn run!(g)
        blocked[] = timedwait(() -> istaskdone(inner[]), 2) === :timed_out
    end
    dispatch!(g, HostCall(startinner), (t,); writes = ())
    @test thrown(() -> run!(g)) === nothing
    @test blocked[]
    @test thrownby(inner[]) === nothing
    @test Array(out) == Int32.(1:n)
end

# P:1353-1355: on a device with sparse residency, the host function's transients (4 MiB each, many sparse pages) are one contiguous view each: unit stride, every element readable and writable.
@case "HOST-19" 3 :sparse begin
    dev = testdevice(); n = 2^20
    g = Graph(dev)
    x = MantleArray(g, Int32, n)
    y = MantleArray(g, Int32, n)
    out = MantleArray(dev, Int32, n)
    unitstride = Ref(false)
    dispatch!(g, host_iota!, (x,), n)
    dispatch!(g, HostCall((a, b) -> (unitstride[] = strides(a) == (1,) && strides(b) == (1,); b .= a .+ Int32(1))), (x, y); writes = (y,))
    dispatch!(g, host_add!, (out, y, Int32(0)), n)
    run!(g)
    @test unitstride[]
    @test Array(out) == Int32.(2:n+1)
end

# A:128: the host function writes u, which its `writes` does not list (ambiguity 25): undefined contents, not detected: run! neither throws nor refuses, and returns Submitted.
@case "HOST-20" 3 begin
    dev = testdevice(); n = 64
    g = Graph(dev)
    t = MantleArray(g, Int32, n)
    u = MantleArray(g, Int32, n)
    out = MantleArray(dev, Int32, n)
    dispatch!(g, host_iota!, (t,), n)
    dispatch!(g, HostCall((a, b) -> (b .= a)), (t, u); writes = ())
    dispatch!(g, host_add!, (out, u, Int32(0)), n)
    @test run!(g) isa Mantle.Submitted
    @test run!(g) isa Mantle.Submitted
end

# P:926-927: K1, H1, K2, H2, K3, each reading what the one before wrote: 3 stages, so 3 submissions and 2 host waits per run (one GPU-to-host round trip per cut); exact.
@case "HOST-21" 3 begin
    dev = testdevice(); n = 64
    g = Graph(dev)
    a, b, c, d = (MantleArray(g, Int32, n) for _ in 1:4)
    out = MantleArray(dev, Int32, n)
    dispatch!(g, host_iota!, (a,), n)
    dispatch!(g, HostCall((x, y) -> (y .= x .* Int32(2))), (a, b); writes = (b,))
    dispatch!(g, host_add!, (c, b, Int32(1)), n)
    dispatch!(g, HostCall((x, y) -> (y .= x .* Int32(3))), (c, d); writes = (d,))
    dispatch!(g, host_add!, (out, d, Int32(0)), n)
    run!(g)
    @test stagecount(g) == 3
    _, diag = counted(() -> run!(g))
    @test submitcount(diag) == 3
    @test diag.hostwaits == 2
    @test Array(out) == Int32.(3 .* (2 .* (1:n) .+ 1))
end
