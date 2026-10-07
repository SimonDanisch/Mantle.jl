# Graph lifecycle (catalogue-core.md, table 2): compile and record once, free!,
# finalizers, introspection, rewrites, the graph lock.

using KernelAbstractions: @kernel, @index, @Const
using LinearAlgebra: mul!
import NNlib

@kernel function gr_add!(dst, @Const(src), v)
    i = @index(Global)
    @inbounds dst[i] = src[i] + v
end

@kernel function gr_mul!(dst, @Const(src), v)
    i = @index(Global)
    @inbounds dst[i] = src[i] * v
end

@kernel function gr_fill!(a, v)
    i = @index(Global)
    @inbounds a[i] = v
end

@kernel function gr_bump!(c)
    @inbounds c[1] += one(eltype(c))
end

@kernel function gr_ends!(out, @Const(t))
    @inbounds out[1] = t[1] + t[end]
end

# submissions(g) lists stages (cut by host nodes), each with its segments (cut
# by budget and queue). api.md 2.3 leaves the shape open; this is the one the
# suite assumes.
segmentcount(g) = sum(s -> length(s.segments), Mantle.submissions(g))
stagecount(g) = length(Mantle.submissions(g))

"""Three kernels through two transients: out = (src + 1) * 3 - 2. Returns the graph, out and the expected contents."""
function threekernels(dev, n; kw...)
    src, out = MantleArray(dev, Int32.(1:n)), MantleArray(dev, Int32, n)
    g = Graph(dev; kw...)
    t1, t2 = MantleArray(g, Int32, n), MantleArray(g, Int32, n)
    dispatch!(g, gr_add!, (t1, src, Int32(1)), n)
    dispatch!(g, gr_mul!, (t2, t1, Int32(3)), n)
    dispatch!(g, gr_add!, (out, t2, Int32(-2)), n)
    return g, out, (Int32.(1:n) .+ Int32(1)) .* Int32(3) .- Int32(2)
end

"""
A chain of kernels over two device arrays whose estimated cost (8n bytes per
kernel at `caps(dev).bandwidth`) is about `seconds`: beyond the submission
budget, so the split cuts it.
"""
function longchain(g, dev, seconds; n = 1 << 25)
    a, b = MantleArray(dev, Float32, n), MantleArray(dev, Float32, n)
    fill!(a, 0f0)
    for j in 1:ceil(Int, seconds * caps(dev).bandwidth / (8n))
        isodd(j) ? dispatch!(g, gr_add!, (b, a, 1f0), n) : dispatch!(g, gr_add!, (a, b, 1f0), n)
    end
    return g
end

"""Starts a run whose slow kernel is still in flight on return, then lets every reference to the graph and its transient go."""
@noinline function runanddrop(dev, out)
    g = Graph(dev)
    t = MantleArray(g, Int32, 1 << 24)
    dispatch!(g, gr_fill!, (t, Int32(3)), 1 << 24)
    slowwrite!(g, out, 7)
    run!(g)
    return nothing
end

"""
A graph whose host function frees the graph on its first call: that run
throws, the next one runs. Every reference goes when this returns.
"""
@noinline function hostfreesitself(dev, out)
    g = Graph(dev)
    calls = Ref(0)
    dispatch!(g, gr_fill!, (out, Int32(0)), 1)
    dispatch!(g, HostCall(_ -> ((calls[] += 1) == 1 && free!(g); nothing)), (); writes = ())
    dispatch!(g, gr_fill!, (out, Int32(9)), 1)
    @test_throws ArgumentError run!(g)
    run!(g)                                         # the throw came before g was marked freed
    return nothing
end

"""A transient of a graph nothing refers to."""
@noinline transientofdropped(dev) = MantleArray(Graph(dev), Float32, 64)

"""Device arrays dropped without free!: their memory comes back only when the GC runs their finalizers."""
@noinline function dropgarbage(dev, count, bytes)
    foreach(_ -> fill!(MantleArray(dev, UInt8, bytes), 0x01), 1:count)
    return nothing
end

# ── 2. Graph lifecycle (GR) ──

# A:101, P:433-434: three kernels, run! 20 times: compiled and recorded by the first run only.
@case "GR-01" 1 begin
    g, out, expected = threekernels(testdevice(), 4096)
    _, d1 = counted(() -> run!(g))
    @test d1.plancompiles == 1
    @test d1.records >= 1
    _, d19 = counted(() -> foreach(_ -> run!(g), 1:19))
    @test d19.plancompiles == 0
    @test d19.records == 0
    @test d19.kernelcompiles == 0
    @test submitcount(d19) >= 19
    @test Array(out) == expected
end

# P:888-889: run! of an empty graph, and of a graph whose head is empty (it names
# transients only), does not throw. The return value is unspecified (ambiguity 18).
@case "GR-02" 1 begin
    dev = testdevice()
    nothingdeclared = Graph(dev)
    @test (run!(nothingdeclared); run!(nothingdeclared); true)
    g = Graph(dev)
    t = MantleArray(g, Int32, 64)
    dispatch!(g, gr_fill!, (t, Int32(1)), 64)
    run!(g)
    _, d = counted(() -> run!(g))
    @test d.records == 0
end

# A:101, I:1010: run! after free!(g) throws.
@case "GR-03" 2 begin
    g, _, _ = threekernels(testdevice(), 64)
    run!(g)
    free!(g)
    @test_throws ArgumentError run!(g)
end

# A:102, I:2088-2090: free!(g) twice, and from 2 threads at once: g's hold on a
# device array it names is dropped once (a second drop would release b, which
# its creator still holds), and the memory goes back.
@case "GR-04" 2 begin
    dev = testdevice(); n = 1024
    b, out = MantleArray(dev, Int32.(1:n)), MantleArray(dev, Int32, n)
    baseline = memory()
    namer() = (named = Graph(dev); dispatch!(named, gr_add!, (out, b, Int32(1)), n); run!(named); named)
    g = namer()
    free!(g); free!(g)
    @test Array(b) == 1:n
    g2 = namer()
    withtimeout(() -> foreach(wait, [Threads.@spawn(free!(g2)) for _ in 1:2]), 30)
    @test Array(b) == 1:n
    @test Array(out) == 2:n+1
    @test memory().reserved <= baseline.reserved
end

# A:102, I:243: after free!(g), fill! on its array, MantleArray(g, …) and
# dispatch!(g, …) throw, and the device array passed adds no hold.
@case "GR-05" 2 begin
    dev = testdevice(); n = 16 << 20
    g = Graph(dev)
    x = MantleArray(g, Float32, 64)
    fill!(x, 1f0)
    run!(g)
    free!(g)
    baseline = memory()
    a = MantleArray(dev, Float32, n)
    @test_throws ArgumentError fill!(x, 2f0)
    @test_throws ArgumentError MantleArray(g, Float32, 64)
    @test_throws ArgumentError dispatch!(g, gr_fill!, (a, 1f0), n)
    free!(a)
    @test memory().reserved <= baseline.reserved
end

# P:1911-1913: every reference to g and its transient dropped while a run is in
# flight, then GC: the run completes correctly and the memory comes back.
@case "GR-06" 2 begin
    dev = testdevice()
    out = MantleArray(dev, Int32, 1)
    calibratedspin(dev, 200)
    baseline = memory()
    runanddrop(dev, out)
    GC.gc(true); GC.gc(true)
    @test Array(out) == [7]
    @test memory().reserved <= baseline.reserved
end

# A:102, I:2088: T1 runs g, held in its host function; T2 calls free!(g): T2
# waits for T1's last stage, the results are correct, a later run! throws.
@case "GR-07" 3 begin
    dev = testdevice(); n = 1024
    a, out = MantleArray(dev, Int32, n), MantleArray(dev, Int32, n)
    gate = Gate()
    g = Graph(dev)
    dispatch!(g, gr_fill!, (a, Int32(4)), n)
    dispatch!(g, blockinghost(gate), (a,); writes = ())
    dispatch!(g, gr_add!, (out, a, Int32(1)), n)
    runner = Threads.@spawn run!(g)
    waitentered(gate)
    freer = Threads.@spawn free!(g)
    sleep(0.5)
    @test !istaskdone(freer)
    letgo!(gate)
    withtimeout(() -> (wait(runner); wait(freer)), 30)
    @test Array(out) == fill(5, n)
    @test_throws ArgumentError run!(g)
end

# A:474, I:2084-2087: g's host function calls free!(g): run! throws, g still
# runs, and once nothing refers to g its finalizer gives the memory back.
@case "GR-08" 3 begin
    dev = testdevice()
    out = MantleArray(dev, Int32, 1)
    baseline = memory()
    hostfreesitself(dev, out)
    @test Array(out) == [9]
    GC.gc(true); GC.gc(true)
    @test memory().reserved <= baseline.reserved
end

# A:103: submissions(g) of a graph with a host cut and a budget cut lists both:
# two stages, and more segments than stages.
@case "GR-09" 1 begin
    dev = testdevice()
    g = longchain(Graph(dev), dev, 0.3)            # three times today's 0.1 s budget
    h1, h2, out = (MantleArray(dev, Float32, 1) for _ in 1:3)
    dispatch!(g, gr_fill!, (h1, 1f0), 1)
    dispatch!(g, HostCall(_ -> nothing), (h1,); writes = (h2,))
    dispatch!(g, gr_add!, (out, h2, 1f0), 1)
    run!(g)
    @test stagecount(g) == 2
    @test segmentcount(g) > stagecount(g)
end

# A:105: naivebytes and peakbytes of a chain of four transients: only adjacent
# ones live together, so the peak is about two of them.
@case "GR-10" 1 begin
    dev = testdevice(); n = 1 << 20; bytes = sizeof(Int32) * n
    src, out = MantleArray(dev, Int32.(1:n)), MantleArray(dev, Int32, n)
    g = Graph(dev)
    t = [MantleArray(g, Int32, n) for _ in 1:4]
    dispatch!(g, gr_add!, (t[1], src, Int32(1)), n)
    foreach(k -> dispatch!(g, gr_add!, (t[k+1], t[k], Int32(1)), n), 1:3)
    dispatch!(g, gr_add!, (out, t[4], Int32(1)), n)
    run!(g)
    @test Mantle.naivebytes(g) >= 4bytes
    @test 2bytes <= Mantle.peakbytes(g) < 3bytes
    @test Array(out) == (1:n) .+ 5
end

# I:58-61: a graph array kept, its graph dropped and collected, then fill!: a clean error.
@case "GR-11" 6 begin
    x = transientofdropped(testdevice())
    GC.gc(true); GC.gc(true)
    @test_throws Exception fill!(x, 1f0)
end

# A:100, E:105-117: mul! + bias + gelu with no rewrites and with FuseEpilogues():
# the results are identical. The node count (2 vs 1) is not visible through
# the suite's hooks.
@case "GR-12" 6 begin
    dev = testdevice()
    W = MantleArray(dev, reshape(testdata(Float32, 512 * 1024), 512, 1024))
    bias = MantleArray(dev, testdata(Float32, 512; seed = 2))
    x = testdata(Float32, 1024; seed = 3)
    results = map(((), (Mantle.FuseEpilogues(),))) do rewrites
        g = Graph(dev; rewrites)
        xd = MantleArray(g, Float32, 1024; hostwritten = true)
        yd = MantleArray(g, Float32, 512)
        out = MantleArray(dev, Float32, 512)
        mul!(yd, W, xd)
        yd .= NNlib.gelu.(yd .+ bias)
        copyto!(out, yd)
        xd[:] = x
        run!(g)
        Array(out)
    end
    @test results[1] == results[2]
end

# P:546-548: Graph(dev; profile = true), 20 runs: recorded once; after a
# structure change, recorded once again (profiling never re-records).
@case "GR-13" 1 begin
    dev = testdevice()
    g, out, expected = threekernels(dev, 4096; profile = true)
    run!(g)
    _, d = counted(() -> foreach(_ -> run!(g), 1:19))
    @test d.records == 0
    @test d.plancompiles == 0
    @test Array(out) == expected
    extra = MantleArray(dev, Int32, 1)
    dispatch!(g, gr_fill!, (extra, Int32(1)), 1)
    _, d = counted(() -> run!(g))
    @test d.plancompiles == 1
    _, d = counted(() -> foreach(_ -> run!(g), 1:5))
    @test d.records == 0
end

# A:217, P:729-731: a compile under the graph lock meets a budget [FI-9] that
# only finalizers can make room in: they run (the lock is a semaphore) and the
# arena allocation succeeds.
@case "GR-14" 3 begin
    fd = FaultDevice(testdevice()); n = 16 << 20
    out = MantleArray(fd, Float32, 1)
    Array(out)                                      # the readback staging exists before the budget is set
    dropgarbage(fd, 4, 64 << 20)
    inject!(fd, :budget, Answer(1, Mantle.reserved(fd)))
    g = Graph(fd)
    t = MantleArray(g, Float32, n)
    dispatch!(g, gr_fill!, (t, 2f0), n)
    dispatch!(g, gr_ends!, (out, t), 1)
    run!(g)
    @test Array(out) == [4f0]
end

# A:101: run!(g) from 8 threads, 1000 times each: all complete, serialized, no
# run overlaps another (a read-modify-write counter is exact).
@case "GR-15" 3 begin
    dev = testdevice()
    c = MantleArray(dev, Int32, 1)
    fill!(c, Int32(0))
    g = Graph(dev)
    dispatch!(g, gr_bump!, (c,), 1)
    run!(g)
    withtimeout(() -> foreach(wait, [Threads.@spawn(foreach(_ -> run!(g), 1:1000)) for _ in 1:8]), 600)
    @test Array(c) == [8001]
end
