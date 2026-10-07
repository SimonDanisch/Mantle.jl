# Fixed vs resizable arrays, staleness, register! (catalogue-resize-rt.md,
# table F).
#
# References: api.md; RR = docs/resizing-and-raytracing.jl; PLAN =
# docs/recording-plan.md (line numbers of 2026-10-05).
# Test hooks as in resize.jl: Mantle.pendingof(x) gives Change objects in
# sequence order; Mantle.steplog(dev) entries have `kind` (a Symbol),
# `target` (the root resource), `seq` and `token`.
# Whether a launch compiled direct or indirect has no test hook; it shows in
# its consequence: a direct launch over `a` makes the first resize! of `a`
# recompile the plan once (RR:664-667), an indirect one never does.
# The catalogue's FI-R (a pause between record! and register!) is a Block on
# the device verb `compilebindings`: compile has decided the layout (every
# node's extents and dependences), recording and register! follow.

using KernelAbstractions: @kernel, @index

@kernel function fx_copy!(dst, src)
    i = @index(Global)
    dst[i] = src[i]
end

@kernel function fx_mark!(marks, a)
    i = @index(Global)
    marks[i] = Int32(1)
end

@kernel function fx_readfirst!(out, a)
    out[1] = a[1]
end

"""The kinds of the changes pending on `x`, in sequence order."""
pendingkinds(x) = [nameof(typeof(c)) for c in Mantle.pendingof(x)]

"""`f()` and the prefix steps applied while it ran (the step log restarted around it)."""
function stepsduring(f, dev = testdevice())
    Mantle.steplog!(dev, false)
    Mantle.steplog!(dev, true)
    v = f()
    log = Mantle.steplog(dev)
    Mantle.steplog!(dev, false)
    return v, log
end

"""The kinds of the steps in `log` that changed `x`, in emission order."""
stepkinds(log, x) = [s.kind for s in log if s.target === x]

"""How many work items a launch marked in `marks`."""
launchcount(marks) = count(==(Int32(1)), Array(marks))

"""Zeroes `marks` and runs `g`; returns the counters of the run."""
markedrun!(g, marks, dev = testdevice()) = (fill!(marks, Int32(0)); last(counted(() -> run!(g), dev)))

"""A graph marking the items of a launch over launchrange(a); compiled and run once."""
function markgraph(dev, marks, a)
    g = Graph(dev)
    dispatch!(g, fx_mark!, (marks, a), launchrange(a))
    run!(g)
    return g
end

"""An eager call reading `a[1]` into `s`: a submission that touches `a`, so it applies what is pending on it."""
applypending!(s, a) = dispatch!(location(s, a), fx_readfirst!, (s, a), 1)

"""An element count whose growth by 16x stays below the sparse threshold (4 sparse pages; RR:202)."""
smallcount(dev) = max(16, caps(dev).sparsepage ÷ 64)

"""Blocks the next call of `verb` on `fd`; returns the call count before it and the channel that lets it go."""
function holdnext!(fd, verb)
    n = calls(fd, verb)
    release = Channel{Nothing}(1)
    inject!(fd, verb, Block(n + 1, release))
    return n, release
end

"""Whether `fd` entered `verb` more than `n` times within `timeout` seconds."""
entered(fd, verb, n; timeout = 30) = timedwait(() -> calls(fd, verb) > n, timeout) === :ok

# ── F. Fixed vs resizable arrays, staleness, register! ──

# FX-01 api.md:89, RR:775: a launch over launchrange of a fresh array and of one created from data compiles direct: the first resize! (a shrink) recompiles the plan once.
@case "FX-01" 4 begin
    dev = testdevice()
    n = 1000
    marks = MantleArray(dev, Int32, n)
    for a in (MantleArray(dev, Float32, n), MantleArray(dev, testdata(Float32, n)))
        g = markgraph(dev, marks, a)
        @test launchcount(marks) == n
        resize!(a, n - 1)
        d = markedrun!(g, marks, dev)
        @test d.plancompiles == 1
        @test launchcount(marks) == n - 1
        free!(g); free!(a)
    end
    free!(marks)
end

# FX-02 RR:670-677,775-776: resize! called and not yet applied, then a new graph compiles: its launch is indirect (later growth recompiles nothing) and its first run applies the Resize.
@case "FX-02" 4 begin
    dev = testdevice()
    n = 1000
    a = MantleArray(dev, testdata(Float32, n))
    marks = MantleArray(dev, Int32, 4n)
    fill!(marks, Int32(0))
    resize!(a, 2n)
    g = Graph(dev)
    dispatch!(g, fx_mark!, (marks, a), launchrange(a))
    _, log = stepsduring(() -> run!(g), dev)
    @test :CellWrite in stepkinds(log, a)
    @test isempty(Mantle.pendingof(a))
    @test launchcount(marks) == 2n
    resize!(a, 3n)
    d = markedrun!(g, marks, dev)
    @test d.plancompiles == 0
    @test launchcount(marks) == 3n
    free!(g); foreach(free!, (a, marks))
end

# FX-03 api.md:80, RR:346-349,664-667: a recorded plan launches directly over fixed `a`; resize!(a, n+1) submits and compiles nothing at the call; the next run! recompiles once and launches n+1 items, the one after compiles nothing.
@case "FX-03" 4 begin
    dev = testdevice()
    n = 1000
    a = MantleArray(dev, testdata(Float32, n))
    marks = MantleArray(dev, Int32, 2n)
    g = markgraph(dev, marks, a)
    _, d = counted(() -> resize!(a, n + 1), dev)
    @test submitcount(d) == 0
    @test d.plancompiles == 0
    d1 = markedrun!(g, marks, dev)
    @test d1.plancompiles == 1
    @test launchcount(marks) == n + 1
    @test markedrun!(g, marks, dev).plancompiles == 0
    free!(g); foreach(free!, (a, marks))
end

# FX-04 api.md:78, RR:779-782,800-802: a plan with a temporary SizedBy(identity, a) (2·length(a) at compile) over resizable `a`: resizes to 2n and to n÷2 recompile nothing and the results are exact.
@case "FX-04" 4 begin
    dev = testdevice()
    n = 1024
    a = MantleArray(dev, testdata(Float32, n))
    resize!(a, n)                                   # resizable: the launches below are indirect
    out = MantleArray(dev, Float32, 4n)
    g = Graph(dev)
    tmp = MantleArray(g, Float32, Mantle.SizedBy(identity, a))
    dispatch!(g, fx_copy!, (tmp, a), launchrange(a))
    dispatch!(g, fx_copy!, (out, tmp), launchrange(a))
    run!(g)
    for len in (2n, n ÷ 2)
        x = testdata(Float32, len; seed = len)
        copy!(a, x)
        _, d = counted(() -> run!(g), dev)
        @test d.plancompiles == 0
        @test Array(out)[1:len] == x
    end
    free!(g); foreach(free!, (a, out))
end

# FX-05 RR:779-782,800: the same plan, `a` grown to 2n+1: the temporary is outgrown, the plan recompiles once and the result is exact.
@case "FX-05" 4 begin
    dev = testdevice()
    n = 1024
    a = MantleArray(dev, testdata(Float32, n))
    resize!(a, n)
    out = MantleArray(dev, Float32, 4n)
    g = Graph(dev)
    tmp = MantleArray(g, Float32, Mantle.SizedBy(identity, a))
    dispatch!(g, fx_copy!, (tmp, a), launchrange(a))
    dispatch!(g, fx_copy!, (out, tmp), launchrange(a))
    run!(g)
    x = testdata(Float32, 2n + 1; seed = 2)
    copy!(a, x)
    _, d = counted(() -> run!(g), dev)
    @test d.plancompiles == 1
    @test Array(out)[1:2n+1] == x
    free!(g); foreach(free!, (a, out))
end

# FX-07 RR:684-691,721: an eager call moves `a` between compile and register! (held in compilebindings): register! returns Again, the plan compiles again, and the run is exact.
@case "FX-07" 4 begin
    fd = FaultDevice(testdevice())
    n = smallcount(testdevice())
    a, s = MantleArray(fd, testdata(Float32, n)), MantleArray(fd, Float32, 1)
    resize!(a, n); applypending!(s, a)            # resizable, storage for n
    out = MantleArray(fd, Float32, 8n)
    g = Graph(fd)
    dispatch!(g, fx_copy!, (out, a), launchrange(a))
    before, release = holdnext!(fd, :compilebindings)
    job = Threads.@spawn counted(() -> run!(g), fd)
    @test entered(fd, :compilebindings, before)
    y = testdata(Float32, 4n; seed = 2)
    copy!(a, y)
    applypending!(s, a)                            # moves `a`: 4n is past its storage
    put!(release, nothing)
    _, d = withtimeout(() -> fetch(job), 60)
    @test d.plancompiles == 2
    @test Array(out)[1:4n] == y
    free!(g); foreach(free!, (a, s, out))
end

# FX-08 RR:705-712,748-755: a resize! that changes the length a plan launches over directly lands between compile and register!: the plan compiles again before its first submission and launches n+1 items.
@case "FX-08" 4 begin
    fd = FaultDevice(testdevice())
    n = 1000
    a = MantleArray(fd, testdata(Float32, n))
    marks = MantleArray(fd, Int32, 2n)
    fill!(marks, Int32(0))
    g = Graph(fd)
    dispatch!(g, fx_mark!, (marks, a), launchrange(a))
    before, release = holdnext!(fd, :compilebindings)
    job = Threads.@spawn counted(() -> run!(g), fd)
    @test entered(fd, :compilebindings, before)
    resize!(a, n + 1)
    put!(release, nothing)
    _, d = withtimeout(() -> fetch(job), 60)
    @test d.plancompiles == 2
    @test launchcount(marks) == n + 1
    free!(g); foreach(free!, (a, marks))
end

# FX-09 RR:684-691,705-711 (unregister!; the dependents table has no test hook, its leak would show with the plans' memory): 1000 recompiles of a graph naming `a` leave memory and live cells as after the first compile.
@case "FX-09" 4 :long begin
    dev = testdevice()
    n = 1024
    a = MantleArray(dev, testdata(Float32, n))
    out1, out2 = MantleArray(dev, Float32, n), MantleArray(dev, Float32, n)
    g = Graph(dev)
    dispatch!(g, fx_copy!, (out1, a), n)
    run!(g)
    m = memory(dev)
    _, d = counted(dev) do
        for _ in 1:500
            h = insert!(g) do
                dispatch!(g, fx_copy!, (out2, a), n)
            end
            run!(g)
            delete!(g, h)
            run!(g)
        end
    end
    @test d.plancompiles == 1000
    m2 = memory(dev)
    @test m2.cells == m.cells
    @test m2.reserved == m.reserved
    @test m2.mapped == m.mapped
    @test Array(out2) == Array(a)
    free!(g); foreach(free!, (a, out1, out2))
end

# FX-10 api.md:180, RR:1285-1288: a launch over launchrange(r), r a GPURef of a fixed array, is indirect: rebinding to another length and the target's first resize! recompile nothing.
@case "FX-10" 6 begin
    dev = testdevice()
    a, b = MantleArray(dev, Float32, 10), MantleArray(dev, Float32, 1000)
    r = GPURef(dev, a)
    marks = MantleArray(dev, Int32, 1000)
    fill!(marks, Int32(0))
    g = markgraph(dev, marks, r)
    @test launchcount(marks) == 10
    r[] = b
    @test markedrun!(g, marks, dev).plancompiles == 0
    @test launchcount(marks) == 1000
    r[] = a
    resize!(a, 20)
    @test markedrun!(g, marks, dev).plancompiles == 0
    @test launchcount(marks) == 20
    free!(g); free!(r); foreach(free!, (a, b, marks))
end

# FX-11 api.md:130,178, RR:807-810: a launch over fixed `a` inside when!(g, flag) is indirect (count × flag): toggling the flag is a store, and neither it nor a resize! of `a` recompiles.
@case "FX-11" 4 begin
    dev = testdevice()
    n = 1000
    a = MantleArray(dev, testdata(Float32, n))
    flag = MantleArray(dev, Int32[1])
    marks = MantleArray(dev, Int32, n)
    fill!(marks, Int32(0))
    g = Graph(dev)
    when!(g, flag) do
        dispatch!(g, fx_mark!, (marks, a), launchrange(a))
    end
    run!(g)
    @test launchcount(marks) == n
    flag[1:1] = Int32[0]
    @test pendingkinds(flag) == [:Store]
    @test markedrun!(g, marks, dev).plancompiles == 0
    @test launchcount(marks) == 0
    flag[1:1] = Int32[1]
    resize!(a, n ÷ 2)
    @test markedrun!(g, marks, dev).plancompiles == 0
    @test launchcount(marks) == n ÷ 2
    free!(g); foreach(free!, (a, flag, marks))
end

# FX-12 RR:215,775-777: resized back to its original length, `a` stays resizable: a graph compiled then launches indirectly, so a later resize! recompiles nothing.
@case "FX-12" 4 begin
    dev = testdevice()
    n = 1000
    a, s = MantleArray(dev, testdata(Float32, n)), MantleArray(dev, Float32, 1)
    resize!(a, 2n)
    resize!(a, n)
    applypending!(s, a)
    marks = MantleArray(dev, Int32, 2n)
    g = markgraph(dev, marks, a)
    resize!(a, n + 1)
    @test markedrun!(g, marks, dev).plancompiles == 0
    @test launchcount(marks) == n + 1
    free!(g); foreach(free!, (a, s, marks))
end
