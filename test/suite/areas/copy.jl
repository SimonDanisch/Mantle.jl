# copy!, copyto! and views (catalogue-resize-rt.md, table D).
#
# References: api.md; RR = docs/resizing-and-raytracing.jl; PLAN =
# docs/recording-plan.md (line numbers of 2026-10-05).
# What this file assumes of the test hooks (helpers.jl):
#   Mantle.pendingof(x)  the Change objects (Store, Resize, ...) pending on x,
#                        in sequence order, with RR B.3's fields.
#   Mantle.steplog(dev)  entries with `kind` (the step's name as a Symbol),
#                        `target` (the root resource the user holds; a view's
#                        steps are its parent's), `seq` and `token`.

using KernelAbstractions: @kernel, @index

@kernel function cp_copy!(dst, src)
    i = @index(Global)
    dst[i] = src[i]
end

@kernel function cp_copymark!(dst, marks, src)
    i = @index(Global)
    dst[i] = src[i]
    marks[i] = Int32(1)
end

# Copies `src` and writes its device-side length into `len[1]`.
@kernel function cp_copylength!(dst, len, src)
    i = @index(Global)
    dst[i] = src[i]
    if i == 1
        len[1] = Int32(length(src))
    end
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

"""An element count whose growth by 16x stays below the sparse threshold (4 sparse pages; RR:202)."""
smallcount(dev) = max(16, caps(dev).sparsepage ÷ 64)

"""The length CP-04 gives the array holding value `v`: lengths and contents always come in these pairs."""
pairedlength(v) = 64 * (1 + Int(v) % 64)

# ── D. copy!, copyto! and views ──

# CP-01 api.md:83, RR:615: copyto!(a, x) with x shorter than a: a store into the prefix, no Resize; the tail unchanged.
@case "CP-01" 4 begin
    dev = testdevice()
    x0, y = testdata(Float32, 100), testdata(Float32, 40; seed = 2)
    a = MantleArray(dev, x0)
    copyto!(a, y)
    @test pendingkinds(a) == [:Store]
    @test length(a) == 100
    r = Array(a)
    @test r[1:40] == y
    @test r[41:end] == x0[41:end]
    free!(a)
end

# CP-02 api.md:83, RR:599,604,615: copyto!(a, x) with x longer than a throws BoundsError; nothing pends; the upload region goes back.
@case "CP-02" 4 begin
    dev = testdevice()
    a = MantleArray(dev, testdata(Float32, 100))
    uploads = memory(dev).uploads
    @test_throws BoundsError copyto!(a, testdata(Float32, 101))
    @test isempty(Mantle.pendingof(a))
    @test memory(dev).uploads == uploads
    free!(a)
end

# CP-03 api.md:84,489, RR:616-646: copy!(a, x) past the storage: length(a) == length(x) at return; a Resize keeping nothing and a Store with consecutive sequence numbers; afterwards a == x.
@case "CP-03" 4 begin
    dev = testdevice()
    a = MantleArray(dev, testdata(Float32, 100))
    x = testdata(Float32, 1000; seed = 2)
    copy!(a, x)
    @test length(a) == 1000
    @test pendingkinds(a) == [:Resize, :Store]
    grow, store = Mantle.pendingof(a)
    @test grow.keep == 0
    @test store.seq == grow.seq + 1
    r, log = stepsduring(() -> Array(a), dev)
    @test :MoveStorage in stepkinds(log, a)
    @test r == x
    free!(a)
end

# CP-04 RR:616-618: copy!(a, x_i) on one thread, a[1:1] = y on another, runs on a third: no run sees a new length with old contents (each run's length and data form one pair).
@case "CP-04" 4 begin
    dev = testdevice()
    a = MantleArray(dev, fill(0f0, pairedlength(0)))
    out, len = MantleArray(dev, Float32, pairedlength(63)), MantleArray(dev, Int32, 1)
    g = Graph(dev)
    dispatch!(g, cp_copylength!, (out, len, a), launchrange(a))
    run!(g)
    writer = Threads.@spawn for i in 1:500
        copy!(a, fill(Float32(i), pairedlength(i)))
    end
    poker = Threads.@spawn for _ in 1:500
        a[1:1] = [-1f0]
    end
    mismatches = 0
    for _ in 1:500
        run!(g)
        n = Int(only(Array(len)))
        r = Array(out)[1:n]
        mismatches += !(n == pairedlength(r[end]) && all(==(r[end]), r[2:n]))
    end
    withtimeout(() -> wait(writer), 60)
    withtimeout(() -> wait(poker), 60)
    @test mismatches == 0
    free!(g); foreach(free!, (a, out, len))
end

# CP-05 api.md:84, RR:647: copy!(view(a, 1:10), x) with 10 elements: a Store into the parent's 1:10, no Resize.
@case "CP-05" 4 begin
    dev = testdevice()
    x0, y = testdata(Float32, 100), testdata(Float32, 10; seed = 2)
    a = MantleArray(dev, x0)
    copy!(view(a, 1:10), y)
    @test pendingkinds(a) == [:Store]
    @test length(a) == 100
    r = Array(a)
    @test r[1:10] == y
    @test r[11:end] == x0[11:end]
    free!(a)
end

# CP-06 api.md:84, RR:647-648: copy!(view(a, 1:10), x) with 5 elements throws ArgumentError; nothing pends.
@case "CP-06" 4 begin
    dev = testdevice()
    a = MantleArray(dev, testdata(Float32, 100))
    @test_throws ArgumentError copy!(view(a, 1:10), testdata(Float32, 5))
    @test isempty(Mantle.pendingof(a))
    @test length(a) == 100
    free!(a)
end

# CP-07 RR:626-627, PLAN:59-63: copy!(a, x), then x .= 0, then a submission: `a` holds the original x, carried by one upload region taken at the call.
@case "CP-07" 4 begin
    dev = testdevice()
    a = MantleArray(dev, Float32, 16)
    x = testdata(Float32, 500)
    keep = copy(x)
    uploads = memory(dev).uploads
    copy!(a, x)
    @test memory(dev).uploads == uploads + 1
    x .= 0
    @test Array(a) == keep
    @test memory(dev).uploads == uploads
    free!(a)
end

# CP-08 RR:620-622,633: copy!(a, x) with a's length is a plain store: no Resize, no recompile of a plan launching directly over `a`, which stays fixed (its first resize! then recompiles once).
@case "CP-08" 4 begin
    dev = testdevice()
    n = 1000
    a = MantleArray(dev, testdata(Float32, n))
    out, marks = MantleArray(dev, Float32, 2n), MantleArray(dev, Int32, 2n)
    g = Graph(dev)
    dispatch!(g, cp_copymark!, (out, marks, a), launchrange(a))
    run!(g)
    y = testdata(Float32, n; seed = 2)
    copy!(a, y)
    @test pendingkinds(a) == [:Store]
    _, d = counted(() -> run!(g), dev)
    @test d.plancompiles == 0
    @test Array(out)[1:n] == y
    resize!(a, n + 1)
    fill!(marks, Int32(0))
    _, d2 = counted(() -> run!(g), dev)
    @test d2.plancompiles == 1
    @test launchcount(marks) == n + 1
    free!(g); foreach(free!, (a, out, marks))
end

# CP-09 api.md:79,91, PLAN:213-218: a view shares its parent's freed mark: after free!(a), a store through the view and a dispatch over it throw ArgumentError.
@case "CP-09" 4 begin
    dev = testdevice()
    a = MantleArray(dev, testdata(Float32, 100))
    v = view(a, 1:10)
    out = MantleArray(dev, Float32, 10)
    free!(a)
    @test_throws ArgumentError (v[1:1] = [1f0])
    @test_throws ArgumentError dispatch!(dev, cp_copy!, (out, v), 10)
    free!(out)
end

# CP-10 api.md:91, RR:234-237: a live view after its parent moved: a kernel reading the view reads the parent's new storage at the view's offset.
@case "CP-10" 4 begin
    dev = testdevice()
    n = smallcount(dev)
    a = MantleArray(dev, testdata(Float32, n))
    v = view(a, 11:20)
    resize!(a, 8n)
    y = testdata(Float32, 10; seed = 2)
    a[11:20] = y                                   # applied after the move: lands in the new storage
    out = MantleArray(dev, Float32, 10)
    _, log = stepsduring(() -> dispatch!(dev, cp_copy!, (out, v), 10), dev)
    @test :MoveStorage in stepkinds(log, a)
    @test Array(out) == y
    foreach(free!, (a, out))
end
