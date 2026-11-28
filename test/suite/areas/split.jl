# Compile, split, queues. Contracts:
# recording-plan.md "Submission split: estimated" and "Queues"; internals.jl
# section 4 (costs!, schedule!, queuefor, submissions!) and section 6
# (submitstage!: the change prefix in the first segment).
#
# The split and the queues are decided by core from caps(dev) facts; cases
# set those facts through FaultDevice: a bandwidth of 1 byte/s makes
# every kernel's estimate exceed any submission budget (one segment per
# kernel), a crossqueuecost of 0 puts independent branches on a second
# compute queue. Cases about cuts between nodes report no job timeout
# (`jobtimeout = Inf`): with one, such estimates are LaunchTooLong (SPLIT-06).


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

"""ids[i] = i, and sizes[i] the launch's global size as the kernel sees it."""
function split_grid!(ids, sizes)
    i = KI.get_global_id().x
    i <= length(ids) || return
    @inbounds ids[i] = Int32(i)
    @inbounds sizes[i] = Int32(KI.get_global_size().x)
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


"""A FaultDevice over the suite's device whose caps report the fields in `kw` instead."""
function capsdevice(; kw...)
    inner = testdevice()
    fd = FaultDevice(inner)
    inject!(fd, :caps, Answer(1, withcaps(caps(inner); kw...)))
    return fd
end

# ── Compile, split, queues (SPLIT) ──

# recording-plan.md "Submission split: estimated", internals.jl section 4 (submissions!): with a bandwidth of 1 byte/s the estimated cost of a 7-kernel graph is many times the submission budget: it is cut into at least 3 segments at compile, and recorded once (5 later runs record nothing, compile nothing, and submit every segment once each).
@case "SPLIT-01" 1 begin
    fd = capsdevice(; bandwidth = 1.0, jobtimeout = Inf); n = 1024
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

# recording-plan.md "Queues" (host-device copies go to a transfer queue where the copy is independent), internals.jl section 4 (queuefor): the host function reads w, a device array no kernel of g writes; its copy into host-visible memory is independent of g's kernels and goes to the transfer queue: a run submits on at least two channels; the host function sees w. Assumes the suite's devices with several queue kinds have a transfer queue.
@case "SPLIT-02" 3 :transferqueue begin
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

# internals.jl section 4 (queuefor: overlap gain against caps(dev).crossqueuecost), recording-plan.md "Queues": two independent branches whose estimated overlap gain exceeds the cross-queue cost (reported as 0): the second branch runs on a second compute queue (a run submits on two channels); both exact.
@case "SPLIT-03" 3 :multicompute begin
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

# internals.jl section 6 (submitstage!: the first segment carries the pending changes of the inbox resources): a store into w before each run; every kernel of a 6-segment chain (bandwidth 1 byte/s) reads w. The change prefix is applied once, in front of the first segment (every step the run applied carries one token), and the later segments read the stored data: exact, three runs with new data each.
@case "SPLIT-04" 4 begin
    fd = capsdevice(; bandwidth = 1.0, jobtimeout = Inf); n = 256; k = 6
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

# internals.jl section 4 (submissions!: a segment waits on the segments holding its predecessors on other queues): two independent chains and a kernel joining them, one segment per kernel (bandwidth 1 byte/s) and the chains on two queue kinds (crossqueuecost 0): a segment waits only on segments of other kinds, never on an earlier one of its own kind (a run picks one queue per kind, and queue order covers that). Reads the segments from submissions(g); that a segment shows its `kind` and `after` (the run's segments it waits on, internals.jl section 6, submitstage!) is assumed, the shape being open (api.md 2.3, `submissions(g)`).
@case "SPLIT-05" 3 :multicompute begin
    fd = capsdevice(; crossqueuecost = 0.0, bandwidth = 1.0, jobtimeout = Inf); n = 1024
    g = Graph(fd)
    out = MantleArray(fd, Int32, n)
    dispatch!(g, split_sum!, (out, splitchain!(g, n, 3), splitchain!(g, n, 3)), n)
    run!(g)
    segs = [s for stage in Mantle.submissions(g) for s in stage.segments]
    @test length(unique(s.kind for s in segs)) >= 2
    @test any(s -> !isempty(s.after), segs)
    @test all(s -> all(p -> p.kind !== s.kind, s.after), segs)
    @test Array(out) == Int32.(2 .* (1:n) .+ 4)
end

# recording-plan.md "Submission split: estimated" (a launch runs whole), internals.jl section 4 (costs!), api.md section 7: a launch whose estimate (bandwidth 1 byte/s) exceeds the job timeout (1 s) throws LaunchTooLong at compile, naming the launch; nothing is submitted. The same launch on the device throws before it submits. On a device that reports no job timeout the same graph runs and is exact.
@case "SPLIT-06" 1 begin
    fd = capsdevice(; bandwidth = 1.0, jobtimeout = 1.0); n = 1024
    a = MantleArray(fd, Int32, n)
    g = Graph(fd)
    dispatch!(g, split_iota!, (a,), n)
    _, d = counted(fd) do
        ex = (@test_throws Mantle.LaunchTooLong run!(g)).value
        @test occursin("split_iota!", sprint(showerror, ex))
        @test ex.timeout == 1.0 && ex.estimate > 1.0
        @test_throws Mantle.LaunchTooLong dispatch!(fd, split_iota!, (a,), n)
    end
    @test submitcount(d) == 0
    free = capsdevice(; bandwidth = 1.0, jobtimeout = Inf)
    b = MantleArray(free, Int32, n)
    h = Graph(free)
    dispatch!(h, split_iota!, (b,), n)
    run!(h)
    @test Array(b) == Int32.(1:n)
end

# recording-plan.md "Submission split: estimated" (opting in), internals.jl section 4 (costs!, cutranges!), api.md dispatch!: a launch over 2^16 items whose estimate (bandwidth 10 kB/s, about 52 s) exceeds the job timeout (10 s), declared with `split = true`, runs as consecutive workgroup ranges in several submissions; every work-item ran once (ids exact) and every range saw the launch's full global size. The same on the device: several submissions, exact.
@case "SPLIT-07" 1 begin
    fd = capsdevice(; bandwidth = 1.0e4, jobtimeout = 10.0); n = 2^16
    ids = MantleArray(fd, Int32, n); sizes = MantleArray(fd, Int32, n)
    g = Graph(fd)
    dispatch!(g, split_grid!, (ids, sizes), n; group = 64, split = true)
    run!(g)
    @test segmentcount(g) >= 2
    @test Array(ids) == Int32.(1:n)
    @test all(==(Int32(n)), Array(sizes))
    ids2 = MantleArray(fd, Int32, n); sizes2 = MantleArray(fd, Int32, n)
    _, d = counted(() -> dispatch!(fd, split_grid!, (ids2, sizes2), n; group = 64, split = true), fd)
    @test submitcount(d) >= 2
    @test Array(ids2) == Int32.(1:n)
    @test all(==(Int32(n)), Array(sizes2))
end

# recording-plan.md "Submission split: estimated" (run time), internals.jl waitfor(::Token), execution-contracts.md sections 3 and 5: on an isolated FaultDevice (its own failure domain) whose native wait never sees a value pass (`waitfor` answers false) and whose job timeout is 50 ms, a host read of a graph's output gives up and throws DeviceLost naming the graph's launches; the next run throws DeviceLost without submitting; the suite's device is untouched.
@case "SPLIT-08" 3 begin
    fd = FaultDevice(testdevice(); isolated = true)
    inject!(fd, :caps, Answer(1, withcaps(caps(testdevice()); jobtimeout = 0.05)))
    inject!(fd, :waitfor, Answer(1, false))
    n = 1024
    out = MantleArray(fd, Int32, n)
    g = Graph(fd)
    dispatch!(g, split_add!, (out, splitchain!(g, n, 2), Int32(0)), n)
    run!(g)
    ex = (@test_throws Mantle.DeviceLost Array(out)).value
    msg = sprint(showerror, ex)
    @test occursin("split_iota!", msg) && occursin("split_add!", msg)
    @test calls(fd, :waitfor) >= 1
    _, d = counted(() -> (@test_throws Mantle.DeviceLost run!(g)), fd)
    @test submitcount(d) == 0
    a = MantleArray(testdevice(), Int32, 4)
    dispatch!(testdevice(), split_iota!, (a,), 4)
    @test Array(a) == Int32[1, 2, 3, 4]
end
