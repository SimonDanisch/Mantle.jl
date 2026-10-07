# Stores: a[r] = data, scheduled (catalogue-resize-rt.md, table C).
#
# References: api.md; RR = docs/resizing-and-raytracing.jl; PLAN =
# docs/recording-plan.md; INT = docs/internals.jl (line numbers of 2026-10-05).
# What this file assumes of the test hooks (helpers.jl):
#   Mantle.pendingof(x)  the Change objects (Store, Resize, Rebind, BuildAccel)
#                        pending on x, in sequence order, with RR B.3's fields
#                        (a BuildAccel's `mode` is Build() or Update()).
#   Mantle.steplog(dev)  entries with `kind` (the step's name as a Symbol:
#                        :InlineStore, :StagedStore, :MoveStorage, :CellWrite,
#                        :BindPages, :BuildAccel), `target` (the root resource
#                        the user holds), `seq` and `token`.
#   diagnostics `allocations` counts every acquire!, upload regions included.

using KernelAbstractions: @kernel, @index
using GeometryBasics: Mat4f, Point3f, TriangleFace

@kernel function st_copy!(dst, src)
    i = @index(Global)
    dst[i] = src[i]
end

# Copies after reading `gate`, which slowwrite! writes: the copy waits for the spin.
@kernel function st_copyafter!(dst, src, gate)
    i = @index(Global)
    if gate[1] != Int32(-7)
        dst[i] = src[i]
    end
end

const ST_STOREKINDS = (:InlineStore, :StagedStore)

"""The kinds of the changes pending on `x`, in sequence order."""
pendingkinds(x) = [nameof(typeof(c)) for c in Mantle.pendingof(x)]

"""The BuildAccel changes pending on a TLAS or BLAS (its range-table stores share its record)."""
pendingbuilds(t) = [c for c in Mantle.pendingof(t) if nameof(typeof(c)) === :BuildAccel]
buildmode(c) = nameof(typeof(c.mode))

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

"""Whether the last step of a kind in `firsts` comes before the first step of kind `then`."""
function emittedbefore(log, firsts, then)
    kinds = [s.kind for s in log]
    i, j = findlast(in(firsts), kinds), findfirst(==(then), kinds)
    return i !== nothing && j !== nothing && i < j
end

"""The step a store of `bytes` at byte `offset` becomes (api.md:82): inline within the device type's limit and alignment, else staged."""
storekind(dev, offset, bytes) = (c = caps(dev);
    bytes <= c.inlinestore && offset % c.inlinealign == 0 && bytes % c.inlinealign == 0 ? :InlineStore : :StagedStore)

translation(x, y, z) = Mat4f(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, x, y, z, 1)

"""A BLAS over two triangles, with its vertex and face arrays."""
function quadblas(dev)
    vertices = MantleArray(dev, Point3f[(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)])
    faces = MantleArray(dev, [TriangleFace{UInt32}(1, 2, 3), TriangleFace{UInt32}(2, 4, 3)])
    return BLAS(dev, vertices, faces), vertices, faces
end

"""A graph that builds `t` every run; its first run applies what was pended on `t` so far."""
function buildgraph(dev, t)
    g = Graph(dev)
    build!(g, t)
    run!(g)
    return g
end

# ── C. Stores ──

# ST-01 api.md:82, RR:590: a[1:3] = [1, 2] throws DimensionMismatch before an upload region is acquired; nothing pends.
@case "ST-01" 4 begin
    dev = testdevice()
    a = MantleArray(dev, testdata(Float32, 8))
    Mantle.resetdiagnostics!(dev)
    @test_throws DimensionMismatch (a[1:3] = Float32[1, 2])
    @test Mantle.diagnostics(dev).allocations == 0
    @test isempty(Mantle.pendingof(a))
    free!(a)
end

# ST-02 RR:599: length 5, resize!(a, 10), a[6:10] = x: bounds are checked against the requested size, so the store is accepted and applied.
@case "ST-02" 4 begin
    dev = testdevice()
    x0, y = testdata(Float32, 5), testdata(Float32, 5; seed = 2)
    a = MantleArray(dev, x0)
    resize!(a, 10)
    a[6:10] = y
    r = Array(a)
    @test r[1:5] == x0
    @test r[6:10] == y
    free!(a)
end

# ST-03 RR:599,604: length 5, resize!(a, 3), a[4:5] = x: BoundsError; the upload region goes back with no tokens; only the Resize pends.
@case "ST-03" 4 begin
    dev = testdevice()
    a = MantleArray(dev, testdata(Float32, 5))
    resize!(a, 3)
    uploads = memory(dev).uploads
    @test_throws BoundsError (a[4:5] = Float32[1, 2])
    @test memory(dev).uploads == uploads
    @test pendingkinds(a) == [:Resize]
    free!(a)
end

# ST-04 RR:593-594,604 (catalogue A5): a[1:1] = [1.5] into a MantleArray{Int32}: InexactError, and the live upload count is back where it was.
@case "ST-04" 4 begin
    dev = testdevice()
    a = MantleArray(dev, Int32[1, 2, 3, 4])
    uploads = memory(dev).uploads
    @test_throws InexactError (a[1:1] = [1.5])
    @test memory(dev).uploads == uploads
    @test isempty(Mantle.pendingof(a))
    free!(a)
end

# ST-05 RR:252-257, PLAN:59-63: a[1:n] = x, then x .= 0, then a submission: `a` holds the original x (copied at the call).
@case "ST-05" 4 begin
    dev = testdevice()
    n = 1000
    x = testdata(Float32, n)
    keep = copy(x)
    a = MantleArray(dev, Float32, n)
    a[1:n] = x
    x .= 0
    @test Array(a) == keep
    free!(a)
end

# ST-06 api.md:82, RR:575-580,608-611: a store of exactly 65536 bytes at offset 0: inline where the device type's limit allows it, else copied from its upload region; the same contents on every device.
@case "ST-06" 4 begin
    dev = testdevice()
    n = 65536 ÷ sizeof(Float32)
    x = testdata(Float32, n)
    a = MantleArray(dev, Float32, 2n)
    a[1:n] = x
    r, log = stepsduring(() -> Array(a), dev)
    @test stepkinds(log, a) == [storekind(dev, 0, 65536)]
    @test r[1:n] == x
    free!(a)
end

# ST-07 api.md:82, RR:256-257,610: a store of 65540 bytes is copied from its upload region (staged), which is retired with the applying submission's token.
@case "ST-07" 4 begin
    dev = testdevice()
    n = 65540 ÷ sizeof(Float32)
    x = testdata(Float32, n)
    a = MantleArray(dev, Float32, n)
    uploads = memory(dev).uploads
    a[1:n] = x
    @test memory(dev).uploads == uploads + 1
    r, log = stepsduring(() -> Array(a), dev)
    @test stepkinds(log, a) == [storekind(dev, 0, 65540)]
    @test r == x
    @test memory(dev).uploads == uploads
    free!(a)
end

# ST-08 RR:578-580,611: a[2:4] = 3 bytes into a UInt8 array: not 4-byte aligned, so staged; a[1] and a[5] untouched.
@case "ST-08" 4 begin
    dev = testdevice()
    a = MantleArray(dev, UInt8[1, 2, 3, 4, 5, 6, 7, 8])
    a[2:4] = UInt8[20, 30, 40]
    r, log = stepsduring(() -> Array(a), dev)
    @test stepkinds(log, a) == [storekind(dev, 1, 3)]
    @test r == UInt8[1, 20, 30, 40, 5, 6, 7, 8]
    free!(a)
end

# ST-09 api.md:82, RR:259-262: a[1:3] = x, then resize!(a, 10^6) (a move): the store is applied before the move, which carries it; a[1:3] == x.
@case "ST-09" 4 begin
    dev = testdevice()
    a = MantleArray(dev, testdata(Float32, 16))
    y = Float32[1, 2, 3]
    a[1:3] = y
    resize!(a, 10^6)
    r, log = stepsduring(() -> Array(a), dev)
    @test filter(in((ST_STOREKINDS..., :MoveStorage)), stepkinds(log, a)) == [storekind(dev, 0, 12), :MoveStorage]
    @test r[1:3] == y
    free!(a)
end

# ST-10 RR:583-586: v = view(m, 1:2:9), v[1:5] = x: nothing is submitted at the call; the applying submission writes the five strided elements and leaves the ones between untouched.
@case "ST-10" 4 begin
    dev = testdevice()
    m = MantleArray(dev, zeros(Float32, 10))
    v = view(m, 1:2:9)
    x = Float32[1, 2, 3, 4, 5]
    _, d = counted(() -> (v[1:5] = x), dev)
    @test submitcount(d) == 0
    @test pendingkinds(m) == [:Store]
    r = Array(m)
    @test r[1:2:9] == x
    @test r[2:2:10] == zeros(Float32, 5)
    free!(m)
end

# ST-11 api.md:82, RR:258: v = view(a, 101:200), v[1:10] = x: the Store's span is the parent bytes of elements 101:110 (400 bytes past a store at a[1]).
@case "ST-11" 4 begin
    dev = testdevice()
    x0, y = testdata(Float32, 300), testdata(Float32, 10; seed = 2)
    a = MantleArray(dev, x0)
    a[1:1] = [7f0]
    v = view(a, 101:200)
    v[1:10] = y
    s1, s2 = Mantle.pendingof(a)
    @test length(s2.span) == 10 * sizeof(Float32)
    @test first(s2.span) - first(s1.span) == 100 * sizeof(Float32)
    r = Array(a)
    @test r[101:110] == y
    @test r[2:100] == x0[2:100]
    @test r[111:end] == x0[111:end]
    free!(a)
end

# ST-12 api.md:155, RR:580-582,1115-1117: a store into a TLAS range's source pends the TLAS Update in the same call; the applying run emits the Store, then the Update.
@case "ST-12" 5 :rt begin
    dev = testdevice()
    blas, vertices, faces = quadblas(dev)
    xs = MantleArray(dev, [translation(0, 0, 0), translation(2, 0, 0)])
    t = TLAS(dev)
    push!(t, blas, xs)
    g = buildgraph(dev, t)
    @test isempty(pendingbuilds(t))
    xs[2:2] = [translation(3, 0, 0)]
    @test buildmode.(pendingbuilds(t)) == [:Update]
    _, log = stepsduring(() -> run!(g), dev)
    @test emittedbefore(log, ST_STOREKINDS, :BuildAccel)
    @test isempty(pendingbuilds(t))
    free!(g); foreach(free!, (t, blas, xs, vertices, faces))
end

# ST-13 api.md:150, RR:580-582,1116-1117: a store into a BLAS's index array pends a BLAS Build and a TLAS Update in the same call.
@case "ST-13" 5 :rt begin
    dev = testdevice()
    blas, vertices, faces = quadblas(dev)
    t = TLAS(dev)
    push!(t, blas, translation(0, 0, 0))
    g = buildgraph(dev, t)
    @test isempty(pendingbuilds(blas))
    faces[2:2] = [TriangleFace{UInt32}(1, 4, 3)]
    @test buildmode.(pendingbuilds(blas)) == [:Build]
    @test buildmode.(pendingbuilds(t)) == [:Update]
    free!(g); foreach(free!, (t, blas, vertices, faces))
end

# ST-14 api.md:79, RR:598,604: after free!(a) a store throws ArgumentError, directly and through a view; the upload region goes back; nothing pends.
@case "ST-14" 4 begin
    dev = testdevice()
    a = MantleArray(dev, testdata(Float32, 16))
    v = view(a, 1:8)
    free!(a)
    uploads = memory(dev).uploads
    @test_throws ArgumentError (a[1:1] = [1f0])
    @test_throws ArgumentError (v[1:1] = [1f0])
    @test memory(dev).uploads == uploads
    @test isempty(Mantle.pendingof(a))
end

# ST-15 api.md:83,93, RR:589-594 (catalogue A15: device sources, scalar and colon stores not yet specified): a[1:n] = b and copyto!(c, b) from a device array copy after b's pending store with no host wait; a[5] = v and a[:] = x store like ranges.
@case "ST-15" 4 begin
    dev = testdevice()
    n = 64
    xb, yb = testdata(Float32, n), testdata(Float32, 4; seed = 3)
    b = MantleArray(dev, xb)
    b[1:4] = yb
    a, c = MantleArray(dev, Float32, n), MantleArray(dev, Float32, n)
    _, d = counted(() -> (a[1:n] = b; copyto!(c, b)), dev)
    @test d.hostwaits == 0
    @test Array(a) == [yb; xb[5:end]]
    @test Array(c) == [yb; xb[5:end]]
    a[5] = 1f0
    @test Array(a)[5] == 1f0
    x = testdata(Float32, n; seed = 4)
    a[:] = x
    @test Array(a) == x
    foreach(free!, (a, b, c))
end

# ST-16 api.md:82, RR:256-257,292: 10^4 covering stores into a[1:16] with a run every 100: at most one upload region live between runs, none after the last run.
@case "ST-16" 4 begin
    dev = testdevice()
    a, out = MantleArray(dev, Float32, 16), MantleArray(dev, Float32, 16)
    g = Graph(dev)
    dispatch!(g, st_copy!, (out, a), 16)
    run!(g)
    uploads = memory(dev).uploads
    between = Int[]
    for i in 1:10^4
        a[1:16] = fill(Float32(i), 16)
        i % 1000 == 50 && push!(between, memory(dev).uploads - uploads)
        i % 100 == 0 && run!(g)
    end
    @test all(==(1), between)
    @test memory(dev).uploads == uploads
    @test Array(out) == fill(Float32(10^4), 16)
    free!(g); foreach(free!, (a, out))
end

# ST-17 PLAN:455,470-472, RR:575-580: a one-element camera array stored every frame while the GPU is frames behind: no host wait in the store or in run!, an inline store per frame, no allocation in run!.
@case "ST-17" 4 begin
    dev = testdevice()
    cam, out = MantleArray(dev, [translation(0, 0, 0)]), MantleArray(dev, Mat4f, 1)
    gate = MantleArray(dev, Int32, 1)
    g = Graph(dev)
    slowwrite!(g, gate, 1; ms = 50)
    dispatch!(g, st_copyafter!, (out, cam, gate), 1)
    run!(g); Array(out)                            # compiled, recorded, idle
    for i in 1:6
        _, ds = counted(() -> (cam[1:1] = [translation(i, 0, 0)]), dev)
        (_, d), log = stepsduring(() -> counted(() -> run!(g), dev), dev)
        @test ds.hostwaits == 0
        @test d.hostwaits == 0
        @test d.allocations == 0
        @test stepkinds(log, cam) == [storekind(dev, 0, sizeof(Mat4f))]
    end
    @test Array(out) == [translation(6, 0, 0)]
    free!(g); foreach(free!, (cam, out, gate))
end

# ST-18 RR:586-588: a store waits only inside acquiring its upload region (held here in rawalloc); a resize! on another thread completes meanwhile, and the store pends after it.
@case "ST-18" 4 begin
    fd = FaultDevice(testdevice())
    n = 2^24                                         # 64 MiB: a new upload block
    a = MantleArray(fd, Float32, 16)
    resize!(a, n)
    x = testdata(Float32, n)
    before = calls(fd, :rawalloc)
    release = Channel{Nothing}(1)
    inject!(fd, :rawalloc, Block(before + 1, release))
    store = Threads.@spawn (a[1:n] = x)
    @test timedwait(() -> calls(fd, :rawalloc) > before, 30) === :ok
    withtimeout(() -> resize!(a, n + 16), 30)
    put!(release, nothing)
    withtimeout(() -> wait(store), 60)
    @test pendingkinds(a) == [:Resize, :Store]
    @test length(a) == n + 16
    free!(a)
end

# ST-19 RR:589-606 (catalogue A24: zero-length stores not yet specified): a[1:0] = T[] pends nothing and acquires no upload region.
@case "ST-19" 4 begin
    dev = testdevice()
    a = MantleArray(dev, testdata(Float32, 8))
    uploads = memory(dev).uploads
    Mantle.resetdiagnostics!(dev)
    a[1:0] = Float32[]
    @test Mantle.diagnostics(dev).allocations == 0
    @test isempty(Mantle.pendingof(a))
    @test memory(dev).uploads == uploads
    free!(a)
end

# ST-20 api.md:85, INT:2229-2245: a store is pending; Array(a) applies it and copies in one submission and returns the new data.
@case "ST-20" 4 begin
    dev = testdevice()
    x, y = testdata(Float32, 64), testdata(Float32, 4; seed = 2)
    a = MantleArray(dev, x)
    a[1:4] = y
    (r, d), log = stepsduring(() -> counted(() -> Array(a), dev), dev)
    @test submitcount(d) == 1
    @test stepkinds(log, a) == [storekind(dev, 0, 16)]
    @test r == [y; x[5:end]]
    free!(a)
end
