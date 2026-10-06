# Applying changes: prepare!, submitwith!, Again (catalogue-resize-rt.md,
# table I).
#
# References: api.md; RR = docs/resizing-and-raytracing.jl; PLAN =
# docs/recording-plan.md; INT = docs/internals.jl (line numbers of 2026-10-05).
# Test hooks as in resize.jl: Mantle.pendingof(x) gives Change objects in
# sequence order; Mantle.steplog(dev) entries have `kind` (a Symbol),
# `target` (the root resource), `seq` and `token`.
# The catalogue's FI-P (a pause between prepare! and the locks) is a Block on
# the device verb `createreserve`: prepare! calls it before any lock, and
# outside the pool lock, for a growth into a reserve (so these cases need
# sparse residency). FI-A is a Throw on the same verb.

using KernelAbstractions: @kernel, @index
using GeometryBasics: Mat4f, Point3f, TriangleFace

@kernel function ap_copy!(dst, src)
    i = @index(Global)
    dst[i] = src[i]
end

@kernel function ap_readfirst!(out, a)
    out[1] = a[1]
end

@kernel function ap_copyfirst!(out, a, i)
    out[i] = a[1]
end

# Copies after reading `gate`, which slowwrite! writes: the copy waits for the spin.
@kernel function ap_copyafter!(dst, src, gate)
    i = @index(Global)
    if gate[1] != Int32(-7)
        dst[i] = src[i]
    end
end

"""The exception the fault injection throws."""
struct InjectedFault <: Exception end

const AP_STOREKINDS = (:InlineStore, :StagedStore)

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

"""An eager call reading `a[1]` into `s`: a submission that touches `a`, so it applies what is pending on it."""
applypending!(s, a) = dispatch!(location(s, a), ap_readfirst!, (s, a), 1)

"""The first `n` elements of `a`, read through a kernel (no host read of the whole array)."""
function firstof(a, n)
    out = MantleArray(location(a), eltype(a), n)
    dispatch!(location(out, a), ap_copy!, (out, a), n)
    r = Array(out)
    free!(out)
    return r
end

"""The Float32 elements of `pages` sparse pages."""
pagecount(dev, pages) = pages * caps(dev).sparsepage ÷ sizeof(Float32)

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

"""A graph that spins about `ms` ms, then copies `n` elements of `src` into `dst`: a run that reads `src` late."""
function slowcopygraph(dev, dst, src, n; ms = 300)
    gate = MantleArray(dev, Int32, 1)
    g = Graph(dev)
    slowwrite!(g, gate, 1; ms)
    dispatch!(g, ap_copyafter!, (dst, src, gate), n)
    return g
end

translation(x, y, z) = Mat4f(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, x, y, z, 1)

"""The CUDA module; loaded wherever a CudaDevice exists."""
cudamodule() = Base.require(Base.PkgId(Base.UUID("052768ef-5323-5732-b1bb-66c8b64840ba"), "CUDA"))

# ── I. Applying changes: prepare!, submitwith!, Again ──

# AP-01 RR:327-338, api.md:432-433, INT:1799-1814: storage for a growth past the storage is allocated in prepare! with no sync-state, channel or pool lock held: while that allocation is held, resize! of the array, a device wait and an eager call elsewhere complete.
@case "AP-01" 4 :sparse begin
    fd = FaultDevice(testdevice())
    Mantle.checklocks!(fd, true)
    n = pagecount(testdevice(), 4)
    x = testdata(Float32, n)
    a, s = MantleArray(fd, x), MantleArray(fd, Float32, 1)
    other = MantleArray(fd, Float32, 16)
    applypending!(s, other)                          # the eager path's own regions exist
    resize!(a, 2n)
    before, release = holdnext!(fd, :createreserve)
    job = Threads.@spawn applypending!(s, a)
    @test entered(fd, :createreserve, before)
    withtimeout(() -> resize!(a, 2n + 1), 30)        # a's sync-state lock is free
    withtimeout(() -> Mantle.waitidle(fd), 30)       # the channel lock is free
    withtimeout(() -> fill!(other, 1f0), 30)         # the pool lock and the channel are free
    put!(release, nothing)
    withtimeout(() -> wait(job), 60)
    @test length(a) == 2n + 1
    @test firstof(a, n) == x
    Mantle.checklocks!(fd, false)
    foreach(free!, (a, s, other))
end

# AP-02 RR:339-344,410-415: another thread resize!s `a` past what a submitter is preparing: the submitter returns Again before emitting anything, and its retry applies both changes in one move.
@case "AP-02" 4 :sparse begin
    fd = FaultDevice(testdevice())
    n = pagecount(testdevice(), 4)
    x = testdata(Float32, n)
    a, s = MantleArray(fd, x), MantleArray(fd, Float32, 1)
    applypending!(s, a)
    resize!(a, 2n)                                   # a reserve of 4·need: 8n elements
    before, release = holdnext!(fd, :createreserve)
    Mantle.steplog!(fd, false); Mantle.steplog!(fd, true)
    job = Threads.@spawn applypending!(s, a)
    @test entered(fd, :createreserve, before)
    withtimeout(() -> resize!(a, 16n), 30)           # past the reserve being prepared
    put!(release, nothing)
    withtimeout(() -> wait(job), 60)
    log = Mantle.steplog(fd)
    Mantle.steplog!(fd, false)
    @test count(==(:MoveStorage), stepkinds(log, a)) == 1
    @test calls(fd, :createreserve) == before + 2    # the held one, then the retry's
    @test isempty(Mantle.pendingof(a))
    @test length(a) == 16n
    @test firstof(a, n) == x
    foreach(free!, (a, s))
end

# AP-03 RR:339-344,431: another submitter applies the change while the first is preparing: the first returns Again, nothing is applied twice, and memory is back at its baseline after free!.
@case "AP-03" 4 :sparse begin
    fd = FaultDevice(testdevice())
    n = pagecount(testdevice(), 4)
    s1, s2 = MantleArray(fd, Float32, 1), MantleArray(fd, Float32, 1)
    base = memory(fd)
    x = testdata(Float32, n)
    a = MantleArray(fd, x)
    applypending!(s1, a)
    resize!(a, 2n)
    a[1:1] = [3f0]
    before, release = holdnext!(fd, :createreserve)
    Mantle.steplog!(fd, false); Mantle.steplog!(fd, true)
    job = Threads.@spawn applypending!(s1, a)
    @test entered(fd, :createreserve, before)
    withtimeout(() -> applypending!(s2, a), 60)      # prepares its own reserve and applies the changes
    put!(release, nothing)
    withtimeout(() -> wait(job), 60)
    log = Mantle.steplog(fd)
    Mantle.steplog!(fd, false)
    kinds = stepkinds(log, a)
    @test count(==(:MoveStorage), kinds) == 1
    @test count(in(AP_STOREKINDS), kinds) == 1
    @test Array(s1) == [3f0]
    @test firstof(a, n) == [3f0; x[2:n]]
    free!(a)
    after = memory(fd)
    @test after.mapped == base.mapped
    @test after.cells == base.cells
    @test after.uploads == base.uploads
    foreach(free!, (s1, s2))
end

# AP-04 RR:327-338, INT:1799-1814: the allocation in prepare! throws: the call throws, the pending changes are unchanged, and the next submitter applies them.
@case "AP-04" 4 :sparse begin
    fd = FaultDevice(testdevice())
    n = pagecount(testdevice(), 4)
    x = testdata(Float32, n)
    a, s = MantleArray(fd, x), MantleArray(fd, Float32, 1)
    applypending!(s, a)
    resize!(a, 2n)
    a[1:1] = [9f0]
    pending = pendingkinds(a)
    inject!(fd, :createreserve, Throw(calls(fd, :createreserve) + 1, InjectedFault()))
    @test_throws InjectedFault applypending!(s, a)
    @test pendingkinds(a) == pending
    applypending!(s, a)                              # the fault was for one call only
    @test isempty(Mantle.pendingof(a))
    @test firstof(a, n) == [9f0; x[2:n]]
    foreach(free!, (a, s))
end

# AP-05 RR:440-445 (bound pages are marked and not bound again), PLAN:712-715: the submission after the page bind throws (FI-B; injected at submitnative!, the first point after bindnow! a FaultDevice reaches): the next submitter maps no more than an unfaulted twin, and free! unmaps everything.
@case "AP-05" 4 :sparse begin
    fd = FaultDevice(testdevice())
    n = pagecount(testdevice(), 4)
    s = MantleArray(fd, Float32, 1)
    base = memory(fd).mapped
    twin = MantleArray(fd, testdata(Float32, n))
    resize!(twin, 2n)
    applypending!(s, twin)
    twinpages = memory(fd).mapped - base
    x = testdata(Float32, n; seed = 2)
    a = MantleArray(fd, x)
    resize!(a, 2n)
    failsubmit!(fd, calls(fd, :submitnative!) + 1, InjectedFault())
    @test_throws InjectedFault applypending!(s, a)
    applypending!(s, a)
    @test memory(fd).mapped - base == 2twinpages
    @test firstof(a, n) == x
    foreach(free!, (a, twin))
    @test memory(fd).mapped == base
    free!(s)
end

# AP-06 INT:1241, PLAN:466-470: a store applied in the prefix of a run's first segment is seen by every later part of the run, on whichever queue core puts it (the negative control, removing that wait, has no hook).
@case "AP-06" 4 :multiqueue begin
    dev = testdevice()
    n = 4096
    a = MantleArray(dev, testdata(Float32, n))
    outs = [MantleArray(dev, Float32, n) for _ in 1:4]
    gate = MantleArray(dev, Int32, 1)
    g = Graph(dev)
    slowwrite!(g, gate, 1; ms = 100)
    dispatch!(g, ap_copyafter!, (outs[1], a, gate), n)
    foreach(o -> dispatch!(g, ap_copy!, (o, a), n), outs[2:end])
    run!(g)
    y = testdata(Float32, n; seed = 2)
    a[1:n] = y
    run!(g)
    @test all(o -> Array(o) == y, outs)
    free!(g); foreach(free!, (a, gate)); foreach(free!, outs)
end

# AP-07 RR:395-397, PLAN:580-601: graph g reads `a` and is in flight; a store into `a` applied by an eager host read waits for g: g still reads the old values (no write-after-read hazard).
@case "AP-07" 4 :multiqueue begin
    dev = testdevice()
    n = 4096
    x, y = testdata(Float32, n), testdata(Float32, n; seed = 2)
    a, out = MantleArray(dev, x), MantleArray(dev, Float32, n)
    g = slowcopygraph(dev, out, a, n)
    run!(g); Array(out)                             # compiled and recorded
    run!(g)                                         # in flight: the copy of `a` comes after the spin
    a[1:n] = y
    @test Array(a) == y
    @test Array(out) == x
    free!(g); foreach(free!, (a, out))
end

# AP-08 RR:395-397,569-571: storage a move replaced while a run still reads it is not reused before that run's token (VAL: GPU-AV clean).
@case "AP-08" 4 begin
    dev = testdevice()
    n = smallcount(dev)
    x = testdata(Float32, n)
    a, out = MantleArray(dev, x), MantleArray(dev, Float32, n)
    g = slowcopygraph(dev, out, a, n)
    run!(g); Array(out)
    run!(g)                                         # in flight
    resize!(a, 4n)
    a[1:n] = fill(-1f0, n)
    @test firstof(a, n) == fill(-1f0, n)            # the move and the store are applied
    others = [fill!(MantleArray(dev, Float32, n), -2f0) for _ in 1:8]   # would land in the old storage if it were free
    @test Array(out) == x
    free!(g); foreach(free!, others); foreach(free!, (a, out))
end

# AP-09 PLAN:462-472, RR:402-433: 1000 arrays with a pending store each, one run naming them all: no extra submission, 1000 store steps, and the run's host time grows linearly with the changes.
@case "AP-09" 4 begin
    dev = testdevice()
    arrays = [MantleArray(dev, Float32, 16) for _ in 1:1000]
    out = MantleArray(dev, Float32, 1000)
    g = Graph(dev)
    foreach(((i, a),) -> dispatch!(g, ap_copyfirst!, (out, a, Int32(i)), 1), enumerate(arrays))
    run!(g)
    _, idle = counted(() -> run!(g), dev)
    foreach(((i, a),) -> (a[1:1] = [Float32(i)]), enumerate(arrays))
    (_, d), log = stepsduring(() -> counted(() -> run!(g), dev), dev)
    @test submitcount(d) == submitcount(idle)
    @test count(in(AP_STOREKINDS), [s.kind for s in log]) == 1000
    @test Array(out) == Float32.(1:1000)
    hosttime(k) = minimum(1:3) do _
        foreach(a -> (a[1:1] = [0f0]), arrays[1:k])
        t = @elapsed run!(g)
        Array(out)
        t
    end
    @test hosttime(1000) < 20 * hosttime(100) + 0.002
    free!(g); foreach(free!, arrays); free!(out)
end

# AP-10 api.md:200-207, PLAN:260-268: a resize and a store pending; withview(CuArray, a, stream): the view has the new length and data; the stream waited for the applying submission.
@case "AP-10" 6 :vendorview begin
    dev = testdevice()
    cuda = cudamodule()
    x, y = testdata(Float32, 100), testdata(Float32, 100; seed = 2)
    a = MantleArray(dev, x)
    resize!(a, 200)
    a[101:200] = y
    withview(cuda.CuArray, a, cuda.stream()) do v
        @test length(v) == 200
        @test Array(v) == [x; y]
    end
    free!(a)
end

# AP-11 RR:487-525 (lockexpanded starts over when a member snapshot changed): runs building `t` on one thread while another pushes ranges with new BLASes: no deadlock, every push! applied (the pause between expansion and locking has no hook; this drives the race; that builds see the new members needs a trace).
@case "AP-11" 5 :rt begin
    dev = testdevice()
    vertices = MantleArray(dev, Point3f[(0, 0, 0), (1, 0, 0), (0, 1, 0)])
    faces = MantleArray(dev, [TriangleFace{UInt32}(1, 2, 3)])
    t = TLAS(dev)
    push!(t, BLAS(dev, vertices, faces), translation(0, 0, 0))
    g = Graph(dev)
    build!(g, t)
    run!(g)
    stop = Threads.Atomic{Bool}(false)
    runner = Threads.@spawn while !stop[]
        run!(g)
    end
    blases = [BLAS(dev, vertices, faces) for _ in 1:200]
    foreach(((i, b),) -> push!(t, b, translation(i, 0, 0)), enumerate(blases))
    stop[] = true
    withtimeout(() -> wait(runner), 60)
    run!(g)
    @test isempty(Mantle.pendingof(t))
    free!(g); free!(t); foreach(free!, blases); foreach(free!, (vertices, faces))
end
