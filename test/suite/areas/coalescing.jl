# Coalescing and sequence numbers (catalogue-resize-rt.md, table H).
#
# References: api.md; RR = docs/resizing-and-raytracing.jl; PLAN =
# docs/recording-plan.md (line numbers of 2026-10-05).
# What this file assumes of the test hooks (helpers.jl):
#   Mantle.pendingof(x)  the Change objects (Store, Resize, Rebind, BuildAccel)
#                        pending on x, in sequence order, with RR B.3's fields
#                        (a BuildAccel's `mode` is Build() or Update()).
#   Mantle.steplog(dev)  entries with `kind` (the step's name as a Symbol:
#                        :InlineStore, :StagedStore, :CellWrite, :BuildAccel,
#                        ...), `target` (the root resource the user holds),
#                        `seq` and `token`.
# "Released" is read from memory(dev).cells: every array owns one cell, given
# back at its last hold.

using KernelAbstractions: @kernel, @index
using GeometryBasics: Mat4f, Point3f, TriangleFace

@kernel function co_copy!(dst, src)
    i = @index(Global)
    dst[i] = src[i]
end

@kernel function co_sumfirst!(out, a1, a2, a3, a4, a5, a6, a7, a8)
    out[1] = a1[1] + a2[1] + a3[1] + a4[1] + a5[1] + a6[1] + a7[1] + a8[1]
end

const CO_STOREKINDS = (:InlineStore, :StagedStore)

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

translation(x, y, z) = Mat4f(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, x, y, z, 1)

"""A BLAS over one triangle; its geometry arrays stay with their creator."""
function triangleblas(dev)
    vertices = MantleArray(dev, Point3f[(0, 0, 0), (1, 0, 0), (0, 1, 0)])
    faces = MantleArray(dev, [TriangleFace{UInt32}(1, 2, 3)])
    return BLAS(dev, vertices, faces)
end

"""A graph that builds `t` every run; its first run applies what was pended on `t` so far."""
function buildgraph(dev, t)
    g = Graph(dev)
    build!(g, t)
    run!(g)
    return g
end

"""A graph that copies `n` elements of what `r` points at into `out`; compiled and run once."""
function readgraph(dev, out, r, n)
    g = Graph(dev)
    dispatch!(g, co_copy!, (out, r), n)
    run!(g)
    return g
end

# ── H. Coalescing and sequence numbers ──

# CO-01 api.md:82, RR:256-257,292: a[1:4] = x then a[1:8] = y: one Store (y); x's upload region is retired at the second call, with no tokens.
@case "CO-01" 4 begin
    dev = testdevice()
    a = MantleArray(dev, testdata(Float32, 16))
    x, y = testdata(Float32, 4; seed = 2), testdata(Float32, 8; seed = 3)
    uploads = memory(dev).uploads
    a[1:4] = x
    a[1:8] = y
    @test pendingkinds(a) == [:Store]
    @test memory(dev).uploads == uploads + 1
    r, log = stepsduring(() -> Array(a), dev)
    @test length(stepkinds(log, a)) == 1
    @test r[1:8] == y
    free!(a)
end

# CO-02 RR:292: a[1:4] = x then a[3:6] = y overlap without covering: two Stores in order, giving [x1, x2, y1..y4].
@case "CO-02" 4 begin
    dev = testdevice()
    a = MantleArray(dev, testdata(Float32, 16))
    x, y = testdata(Float32, 4; seed = 2), testdata(Float32, 4; seed = 3)
    a[1:4] = x
    a[3:6] = y
    @test pendingkinds(a) == [:Store, :Store]
    @test Array(a)[1:6] == [x[1:2]; y]
    free!(a)
end

# CO-03 RR:258,292: view(a, 3:6)[1:4] = y then a[1:8] = z: spans compare in parent bytes, so the second covers the first: one Store.
@case "CO-03" 4 begin
    dev = testdevice()
    a = MantleArray(dev, testdata(Float32, 16))
    y, z = testdata(Float32, 4; seed = 2), testdata(Float32, 8; seed = 3)
    view(a, 3:6)[1:4] = y
    a[1:8] = z
    @test pendingkinds(a) == [:Store]
    @test Array(a)[1:8] == z
    free!(a)
end

# CO-04 RR:293-294,544: resize!(a, 10), resize!(a, 5), resize!(a, 20) on 100 elements: one Resize to (20,) keeping 5 elements.
@case "CO-04" 4 begin
    dev = testdevice()
    x = testdata(Float32, 100)
    a = MantleArray(dev, x)
    resize!(a, 10)
    resize!(a, 5)
    resize!(a, 20)
    @test pendingkinds(a) == [:Resize]
    c = only(Mantle.pendingof(a))
    @test c.dims == (20,)
    @test c.keep == 5 * sizeof(Float32)
    @test Array(a)[1:5] == x[1:5]
    free!(a)
end

# CO-05 RR:293-294: resize!(a, 10), a[1:3] = x, resize!(a, 2): Resizes do not merge across the Store: three changes; the result is [x1, x2].
@case "CO-05" 4 begin
    dev = testdevice()
    a = MantleArray(dev, testdata(Float32, 100))
    x = Float32[1, 2, 3]
    resize!(a, 10)
    a[1:3] = x
    resize!(a, 2)
    @test pendingkinds(a) == [:Resize, :Store, :Resize]
    @test Array(a) == x[1:2]
    free!(a)
end

# CO-06 RR:294-296, PLAN:459-460: before any submission push! (Build), tlas[h] = m and visible!(t, h, false) (each a Store and an Update): one Build pending, emitted after both stores (that the build sees m and mask 0 needs a trace: tlas area).
@case "CO-06" 5 :rt begin
    dev = testdevice()
    blas = triangleblas(dev)
    t = TLAS(dev)
    h = push!(t, blas, translation(0, 0, 0))
    t[h] = translation(1, 0, 0)
    visible!(t, h, false)
    @test buildmode.(pendingbuilds(t)) == [:Build]
    g = Graph(dev)
    build!(g, t)
    _, log = stepsduring(() -> run!(g), dev)
    @test count(in(CO_STOREKINDS), [s.kind for s in log]) >= 2
    @test emittedbefore(log, CO_STOREKINDS, :BuildAccel)
    @test isempty(pendingbuilds(t))
    free!(g); foreach(free!, (t, blas))
end

# CO-07 RR:292-296: tlas[h] = m1 then tlas[h] = m2: one source Store (m2) and one Update.
@case "CO-07" 5 :rt begin
    dev = testdevice()
    blas = triangleblas(dev)
    t = TLAS(dev)
    h = push!(t, blas, translation(0, 0, 0))
    g = buildgraph(dev, t)
    t[h] = translation(1, 0, 0)
    t[h] = translation(2, 0, 0)
    @test buildmode.(pendingbuilds(t)) == [:Update]
    _, log = stepsduring(() -> run!(g), dev)
    @test count(in(CO_STOREKINDS), [s.kind for s in log]) == 1
    @test count(==(:BuildAccel), [s.kind for s in log]) == 1
    free!(g); foreach(free!, (t, blas))
end

# CO-08 RR:294-296,1250-1260, api.md:180: r[] = b then r[] = c: one Rebind (c), one cell write when applied; after it, freeing the creators releases a and b.
@case "CO-08" 6 begin
    dev = testdevice()
    a, b, c = (MantleArray(dev, testdata(Float32, 16; seed)) for seed in 1:3)
    out = MantleArray(dev, Float32, 16)
    r = GPURef(dev, a)
    g = readgraph(dev, out, r, 16)
    r[] = b
    r[] = c
    @test pendingkinds(r) == [:Rebind]
    @test only(Mantle.pendingof(r)).new === c
    _, log = stepsduring(() -> run!(g), dev)
    @test count(==(:CellWrite), stepkinds(log, r)) == 1
    @test Array(out) == Array(c)
    cells = memory(dev).cells
    free!(a); free!(b)
    @test memory(dev).cells == cells - 2
    free!(g); free!(r); foreach(free!, (c, out))
end

# CO-09 RR:294-296,1238-1260, api.md:180: r[] = b then r[] = a, `a` the applied target: one Rebind (a); after it b is released by free!(b), while `a` stays held by r past free!(a).
@case "CO-09" 6 begin
    dev = testdevice()
    xa = testdata(Float32, 16)
    a, b = MantleArray(dev, xa), MantleArray(dev, testdata(Float32, 16; seed = 2))
    out = MantleArray(dev, Float32, 16)
    r = GPURef(dev, a)
    g = readgraph(dev, out, r, 16)
    r[] = b
    r[] = a
    @test pendingkinds(r) == [:Rebind]
    @test only(Mantle.pendingof(r)).new === a
    run!(g)
    cells = memory(dev).cells
    free!(b)
    @test memory(dev).cells == cells - 1
    free!(a)
    run!(g)
    @test Array(out) == xa
    @test memory(dev).cells == cells - 1
    free!(g); free!(r); free!(out)
end

# CO-10 RR:294-296: a store into src1 (Update), copy! of src2 to a new length (Build), a store into src1 again: one merged Build with the last sequence number, emitted after the src1 store.
@case "CO-10" 5 :rt begin
    dev = testdevice()
    blas = triangleblas(dev)
    src1, src2 = MantleArray(dev, [translation(0, 0, 0)]), MantleArray(dev, [translation(2, 0, 0)])
    t = TLAS(dev)
    push!(t, blas, src1)
    push!(t, blas, src2)
    g = buildgraph(dev, t)
    src1[1:1] = [translation(1, 0, 0)]
    copy!(src2, [translation(2, 0, 0), translation(3, 0, 0)])
    src1[1:1] = [translation(4, 0, 0)]
    build = only(pendingbuilds(t))
    @test buildmode(build) == :Build
    @test build.seq > only(Mantle.pendingof(src1)).seq
    _, log = stepsduring(() -> run!(g), dev)
    @test emittedbefore(log, CO_STOREKINDS, :BuildAccel)
    free!(g); foreach(free!, (t, blas, src1, src2))
end

# CO-11 RR:296-300,1095-1102: delete!(t, r1), push!, delete!(t, r2) before any submission: both deferred hold drops are kept; r1's and r2's sources are released only after the applying run.
@case "CO-11" 5 :rt begin
    dev = testdevice()
    blas = triangleblas(dev)
    s1, s2 = MantleArray(dev, [translation(0, 0, 0)]), MantleArray(dev, [translation(1, 0, 0)])
    t = TLAS(dev)
    r1 = push!(t, blas, s1)
    r2 = push!(t, blas, s2)
    g = buildgraph(dev, t)
    delete!(t, r1)
    push!(t, blas, translation(2, 0, 0))
    delete!(t, r2)
    free!(s1); free!(s2)                            # only t holds them now
    cells = memory(dev).cells
    @test memory(dev).cells == cells                # nothing applied yet: still held
    run!(g)
    @test memory(dev).cells == cells - 2
    free!(g); foreach(free!, (t, blas))
end

# CO-12 RR:302 (nextseq!: a device-wide atomic), PLAN:59-71: 8 threads store 10^4 times each into their own array; the sequence numbers are unique, and one eager kernel over all of them emits the steps sorted by them.
@case "CO-12" 4 :long begin
    dev = testdevice()
    n = 10^4
    arrays = [MantleArray(dev, Float32, n) for _ in 1:8]
    tasks = [Threads.@spawn(foreach(j -> (a[j:j] = [Float32(j)]), 1:n)) for a in arrays]
    foreach(t -> withtimeout(() -> wait(t), 600), tasks)
    seqs = [c.seq for a in arrays for c in Mantle.pendingof(a)]
    @test length(seqs) == 8n
    @test allunique(seqs)
    out = MantleArray(dev, Float32, 1)
    _, log = stepsduring(() -> dispatch!(dev, co_sumfirst!, (out, arrays...), 1), dev)
    @test length(log) == 8n
    @test issorted([s.seq for s in log])
    @test Array(arrays[1]) == Float32.(1:n)
    foreach(free!, arrays); free!(out)
end
