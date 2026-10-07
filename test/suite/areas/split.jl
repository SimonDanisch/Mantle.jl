# Compile, split, queues (catalogue-core.md, table 22: SPLIT). Contracts:
# recording-plan.md "Submission split: estimated" and "Queues"; internals.jl
# section 4 (costs!, schedule!, queuefor, submissions!) and section 6
# (submitstage!: the change prefix in the first segment).
#
# The split and the queues are core's decisions from caps(dev) facts; cases
# set those facts through FaultDevice [FI-10]: a bandwidth of 1 byte/s makes
# every kernel's estimate exceed any submission budget (one segment per
# kernel), a crossqueuecost of 0 puts independent branches on a second
# compute queue.

import KernelInterface as KI

"""a[i] = i."""
function split_iota!(a)
    i = KI.get_global_id().x
    i <= length(a) || return
    @inbounds a[i] = eltype(a)(i)
    return
end

"""dst[i] = src[i] + k, over the shorter of the two."""
function split_add!(dst, src, k)
    i = KI.get_global_id().x
    i <= min(length(dst), length(src)) || return
    @inbounds dst[i] = src[i] + k
    return
end

"""dst[i] = a[i] + b[i]."""
function split_sum!(dst, a, b)
    i = KI.get_global_id().x
    i <= length(dst) || return
    @inbounds dst[i] = a[i] + b[i]
    return
end

"""Declares into `g` a chain of `k` kernels over transients of `n` Int32 (1:n, then +1 per kernel); returns the last transient."""
function splitchain!(g, n, k)
    prev = MantleArray(g, Int32, n)
    dispatch!(g, split_iota!, (prev,), n)
    for _ in 2:k
        next = MantleArray(g, Int32, n)
        dispatch!(g, split_add!, (next, prev, Int32(1)), n)
        prev = next
    end
    return prev
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

# submissions(g) lists stages (cut by host nodes), each with its segments (cut
# by budget and queue). api.md 2.3 leaves the shape open; this is the one the
# suite assumes.
segmentcount(g) = sum(s -> length(s.segments), Mantle.submissions(g))

"""The channels a run's submissions went to (diagnostics count them per channel)."""
channelsused(d) = count(>(0), values(d.submissions))

# ── 22. Compile, split, queues (SPLIT) ──

# P:537-548: with a bandwidth of 1 byte/s [FI-10] the estimated cost of a 7-kernel graph is many times the submission budget: it is cut into at least 3 segments at compile, and recorded once (5 later runs record nothing, compile nothing, and submit every segment once each).
@case "SPLIT-01" 1 begin
    fd = capsdevice(; bandwidth = 1.0); n = 1024
    g = Graph(fd)
    out = MantleArray(fd, Int32, n)
    tail = splitchain!(g, n, 6)
    dispatch!(g, split_add!, (out, tail, Int32(0)), n)
    run!(g)
    segments = segmentcount(g)
    @test segments >= 3
    _, d = counted(() -> foreach(_ -> run!(g), 1:5), fd)
    @test submitcount(d) == 5segments
    @test d.records == 0
    @test d.plancompiles == 0
    @test Array(out) == Int32.(6:n+5)
end

# P:567-568: the host function reads w, a device array no kernel of g writes; its copy into host-visible memory is independent of g's kernels and goes to the transfer queue: a run submits on at least two channels; the host function sees w. Assumes the suite's devices with several queue kinds have a transfer queue.
@case "SPLIT-02" 3 :multiqueue begin
    dev = testdevice(); n = 2^16
    data = testdata(Int32, n)
    w = MantleArray(dev, data)
    out = MantleArray(dev, Int32, n)
    seen = Ref(Int32[])
    g = Graph(dev)
    t = splitchain!(g, n, 2)
    dispatch!(g, split_add!, (out, t, Int32(0)), n)
    dispatch!(g, HostCall(v -> (seen[] = copy(v))), (w,); writes = ())
    run!(g)
    _, d = counted(() -> run!(g))
    @test channelsused(d) >= 2
    @test seen[] == data
    @test Array(out) == Int32.(2:n+1)
end

# I:578: two independent branches whose estimated overlap gain exceeds the cross-queue cost (reported as 0 [FI-10]): the second branch runs on a second compute queue (a run submits on two channels); both exact.
@case "SPLIT-03" 3 :multiqueue begin
    fd = capsdevice(; crossqueuecost = 0.0); n = 2^16
    g = Graph(fd)
    outa = MantleArray(fd, Int32, n)
    outb = MantleArray(fd, Int32, n)
    dispatch!(g, split_add!, (outa, splitchain!(g, n, 3), Int32(0)), n)
    dispatch!(g, split_add!, (outb, splitchain!(g, n, 4), Int32(0)), n)
    run!(g)
    _, d = counted(() -> run!(g), fd)
    @test channelsused(d) >= 2
    @test Array(outa) == Int32.(3:n+2)
    @test Array(outb) == Int32.(4:n+3)
end

# I:1218-1243: a store into w before each run; every kernel of a 6-segment chain (bandwidth 1 byte/s [FI-10]) reads w. The change prefix is applied once, in front of the first segment (every step the run applied carries one token), and the later segments read the stored data: exact, three runs with new data each.
@case "SPLIT-04" 4 begin
    fd = capsdevice(; bandwidth = 1.0); n = 256; k = 6
    w = MantleArray(fd, Int32, n)
    fill!(w, Int32(0))
    out = MantleArray(fd, Int32, n)
    g = Graph(fd)
    prev = MantleArray(g, Int32, n)
    dispatch!(g, split_iota!, (prev,), n)
    for _ in 2:k-1
        next = MantleArray(g, Int32, n)
        dispatch!(g, split_sum!, (next, prev, w), n)
        prev = next
    end
    dispatch!(g, split_sum!, (out, prev, w), n)
    run!(g)
    @test segmentcount(g) >= 3
    Mantle.steplog!(fd, true)
    for s in 1:3
        data = testdata(Int32, n; seed = s) .% Int32(1000)
        w[1:n] = data
        before = length(Mantle.steplog(fd))
        run!(g)
        steps = Mantle.steplog(fd)[before+1:end]
        @test !isempty(steps)
        @test length(unique(token for (_, _, _, token) in steps)) == 1
        @test Array(out) == Int32.(1:n) .+ Int32(k - 1) .* data
    end
    Mantle.steplog!(fd, false)
end

# P:2321-2323: two independent chains and a kernel joining them, one segment per kernel (bandwidth 1 byte/s) and the chains on two queues (crossqueuecost 0) [FI-10]: a segment waits only on segments of other queues, never on an earlier one of its own queue (queue order covers that). Reads the segments from submissions(g); that a segment shows its `channel` and `after` (the run's segments it waits on, internals.jl submitstage!) is assumed, the shape being open (P1.A16).
@case "SPLIT-05" 3 :multiqueue begin
    fd = capsdevice(; crossqueuecost = 0.0, bandwidth = 1.0); n = 1024
    g = Graph(fd)
    out = MantleArray(fd, Int32, n)
    dispatch!(g, split_sum!, (out, splitchain!(g, n, 3), splitchain!(g, n, 3)), n)
    run!(g)
    segs = [s for stage in Mantle.submissions(g) for s in stage.segments]
    @test length(unique(s.channel for s in segs)) >= 2
    @test any(s -> !isempty(s.after), segs)
    @test all(s -> all(p -> p.channel !== s.channel, s.after), segs)
    @test Array(out) == Int32.(2 .* (1:n) .+ 4)
end
