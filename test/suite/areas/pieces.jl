# Render pass pieces (catalogue-core.md, table 14; decisions.md B3.17): a pass
# is a list of pieces, each compiled and recorded on its own; insert!(p) and
# delete!(p, h) are applied by the graph's next run, which compiles and records
# the piece and rewrites the pass's list, without a plan compile (api.md 2.6b;
# recording-plan.md, Graphics; internals.jl section 5; ledger P6.A16).
#
# Targets are read back with Array(img): a `w × h` matrix, first index x. The
# cases draw vertical stripes over the whole height, so only x positions are
# checked. vertex_index and instance_index count from one.

using KernelAbstractions: @kernel, @index, @Const
using GeometryBasics: Vec2f, Vec3f, Vec4f
using ColorTypes: RGBA
using FixedPointNumbers: N0f8

const BLACK = Vec3f(0, 0, 0)
const RED = Vec3f(1, 0, 0)
const GREEN = Vec3f(0, 1, 0)
const BLUE = Vec3f(0, 0, 1)
const BACKGROUND = Clear((0f0, 0f0, 0f0, 1f0))

opaque(c::Vec3f) = Vec4f(c[1], c[2], c[3], 1)

"""Stripe `k` of `n` vertical stripes over the whole target: (x0, y0, x1, y1) in normalized device coordinates."""
stripe(k, n) = Vec4f(-1 + 2(k - 1) / n, -1, -1 + 2k / n, 1)

"""The pixel in the middle of stripe `k` of `n` in a readback `A` (first index x), on the middle row."""
stripepixel(A, k, n) = A[clamp(round(Int, (k - 0.5) / n * size(A, 1) + 0.5), 1, size(A, 1)), cld(size(A, 2), 2)]

rgb(px) = Vec3f(px.r, px.g, px.b)

"""Whether pixel `px` has colour `c`, within 0.02 per channel (8-bit targets, blending)."""
iscolour(px, c::Vec3f; tol = 0.02) = maximum(abs.(rgb(px) .- c)) <= tol

medianof(xs) = sort(xs)[cld(length(xs), 2)]

# Rectangles, one per instance: rects[i] = (x0, y0, x1, y1) in normalized
# device coordinates; colours one for all rectangles or one per rectangle,
# read with attribute(a, i) (api.md 2.6b). Vertices 1-3 and 4-6 are the two
# triangles of a rectangle.
function rects_vertex(rects, colours)
    i = instance_index()
    r = Mantle.attribute(rects, i)
    v = vertex_index()
    right = v == 2 || v == 4 || v == 5
    top = v == 3 || v == 5 || v == 6
    position = Vec4f(right ? r[3] : r[1], top ? r[4] : r[2], 0, 1)
    return (; position, colour = Mantle.attribute(colours, i))
end
rects_fragment(inputs, args...) = inputs.colour

rectpipeline(blend = Mantle.Opaque()) =
    Rasterizer(; vertex = Mantle.VertexShader(rects_vertex; outputs = (colour = Vec4f,)),
               fragment = Mantle.FragmentShader(rects_fragment), topology = TriangleList(),
               blend, cull = Mantle.NoCull(), depth = Mantle.DepthOff())

"""Draws `rects` in `colours` into pass `p`: inside render!'s block or an insert!(p) block."""
drawrects!(p, rects, colours; blend = Mantle.Opaque()) =
    draw!(p, rectpipeline(blend), (rects, colours), 6; instances = launchrange(rects))

# A mesh: vertices[i] a 2D position, drawn indexed with 0-based indices, so
# vertex_index() is the index plus one; one colour.
function mesh_vertex(vertices, colours)
    q = vertices[vertex_index()]
    return (position = Vec4f(q[1], q[2], 0, 1), colour = Mantle.attribute(colours, 1))
end
meshpipeline() =
    Rasterizer(; vertex = Mantle.VertexShader(mesh_vertex; outputs = (colour = Vec4f,)),
               fragment = Mantle.FragmentShader(rects_fragment), topology = TriangleList(),
               blend = Mantle.Opaque(), cull = Mantle.NoCull(), depth = Mantle.DepthOff())

"""A grid of at least `ntriangles` triangles over the whole target: vertices and 0-based indices."""
function gridmesh(ntriangles)
    n = ceil(Int, sqrt(ntriangles / 2))
    xs = range(-1f0, 1f0; length = n + 1)
    vertices = [Vec2f(x, y) for y in xs for x in xs]
    at(i, j) = UInt32(j * (n + 1) + i)
    indices = UInt32[]
    sizehint!(indices, 6n^2)
    for j in 0:n-1, i in 0:n-1
        append!(indices, (at(i, j), at(i + 1, j), at(i, j + 1), at(i + 1, j), at(i + 1, j + 1), at(i, j + 1)))
    end
    return vertices, indices
end

"""
A graph with one pass into a new square target cleared to black; `f(p)`
declares the pass's first piece. Returns the graph, the pass and the target.
"""
function framegraph(f, dev; side = 64)
    g = Graph(dev)
    img = Image(dev, RGBA{N0f8}, side, side)
    p = render!(f, g, img => BACKGROUND)
    return (; g, p, img)
end
framegraph(dev; side = 64) = framegraph(p -> nothing, dev; side)

"""A piece of `p` drawing `rect` in `colour` from arrays of its own; returns the handle and the arrays."""
function piece!(p, dev, rect::Vec4f, colour::Vec4f; blend = Mantle.Opaque())
    rects = MantleArray(dev, [rect])
    colours = MantleArray(dev, [colour])
    h = insert!(() -> drawrects!(p, rects, colours; blend), p)
    return (; h, rects, colours)
end

"""A pass of `n` pieces drawing `rects` in `colours`, each under its own `when!` flag, compiled."""
function crowdedpass(dev, n, rects, colours)
    (; g, p) = framegraph(dev)
    flags = [MantleArray(dev, UInt32[1]) for _ in 1:n]
    for f in flags
        insert!(p) do
            when!(() -> drawrects!(p, rects, colours), p, f)
        end
    end
    run!(g)
    return (; g, p, flags)
end

"""Host time of inserting one piece and of deleting it, each with the run that applies it."""
function insertdelete(s, rects, colours, dev)
    Mantle.waitidle(dev)
    local h
    tinsert = @elapsed begin
        h = insert!(() -> drawrects!(s.p, rects, colours), s.p)
        run!(s.g)
    end
    Mantle.waitidle(dev)
    tdelete = @elapsed begin
        delete!(s.p, h)
        run!(s.g)
    end
    return tinsert, tdelete
end

"""Time of one frame of `g` on the GPU and the host: the run and a wait for the device."""
frametime(g, dev) = @elapsed (run!(g); Mantle.waitidle(dev))

@kernel function pc_setone!(a)
    @inbounds a[1] = one(eltype(a))
end

# Copies src[1] into dst[1] after about `iters` dependent steps: a write that
# lands late (the chain of helpers.jl's slowwrite!).
@kernel function pc_slowcopy!(dst, @Const(src), iters)
    acc = Int32(0)
    for i in Int32(1):iters
        acc = acc * Int32(1103515245) + i
    end
    @inbounds dst[1] = acc == Int32(-7) ? zero(eltype(dst)) : src[1]
end

# ── 14. Render pass pieces, B3.17 (PIECE) ──

# api.md:176, decisions.md:255-268. Insert a piece into a compiled pass, run: drawn; no plan compile; the piece and the list part recorded.
@case "PIECE-01" 6 :graphics begin   # pending decision 8
    dev = testdevice()
    (; g, p, img) = framegraph(dev)
    run!(g)
    piece!(p, dev, stripe(1, 2), opaque(RED))
    _, d = counted(() -> run!(g))
    @test d.plancompiles == 0
    @test d.records == 2                 # the piece and the pass's list part
    @test d.pipelinecompiles <= 1        # the piece's pipeline, counted apart from plan compiles
    A = Array(img)
    @test iscolour(stripepixel(A, 1, 2), RED)
    @test iscolour(stripepixel(A, 2, 2), BLACK)
    piece!(p, dev, stripe(2, 2), opaque(GREEN))   # the same pipeline: from the cache
    _, d = counted(() -> run!(g))
    @test d.plancompiles == 0
    @test d.pipelinecompiles == 0
    @test d.records == 2
    @test iscolour(stripepixel(Array(img), 2, 2), GREEN)
end

# api.md:176. delete!(p, h) and run: the piece is gone; no plan compile; only the list part is recorded, no other piece.
@case "PIECE-02" 6 :graphics begin
    dev = testdevice()
    (; g, p, img) = framegraph(dev)
    a = piece!(p, dev, stripe(1, 2), opaque(RED))
    piece!(p, dev, stripe(2, 2), opaque(GREEN))
    run!(g)
    delete!(p, a.h)
    _, d = counted(() -> run!(g))
    @test d.plancompiles == 0
    @test d.records == 1
    A = Array(img)
    @test iscolour(stripepixel(A, 1, 2), BLACK)
    @test iscolour(stripepixel(A, 2, 2), GREEN)
end

# internals.jl:881-883 (900-902 today). insert! changes nothing until the next run: the target read before run! is unchanged.
@case "PIECE-03" 6 :graphics begin
    dev = testdevice()
    (; g, p, img) = framegraph(dev)
    run!(g)
    before = Array(img)
    piece!(p, dev, stripe(1, 1), opaque(RED))
    @test Array(img) == before
    run!(g)
    @test iscolour(stripepixel(Array(img), 1, 1), RED)
end

# api.md:176, internals.jl:918-919 (940-941 today). The piece's pipeline fails to compile [FI-1]: run! throws; the piece is taken out, its holds dropped; the next run works.
@case "PIECE-04" 6 :graphics begin
    fd = FaultDevice(testdevice())
    (; g, p, img) = framegraph(fd)
    run!(g)
    cells = memory(fd).cells
    inject!(fd, :compiledraw, Throw(calls(fd, :compiledraw) + 1, ErrorException("injected pipeline failure")))
    a = piece!(p, fd, stripe(1, 2), opaque(RED))
    @test_throws ErrorException run!(g)
    _, d = counted(() -> run!(g), fd)
    @test d.plancompiles == 0
    @test iscolour(stripepixel(Array(img), 1, 2), BLACK)
    @test_throws ArgumentError delete!(p, a.h)    # taken out: no longer a piece of the pass
    free!(a.rects); free!(a.colours)
    @test memory(fd).cells == cells               # the graph holds neither array
    piece!(p, fd, stripe(2, 2), opaque(GREEN))
    run!(g)
    @test iscolour(stripepixel(Array(img), 2, 2), GREEN)
end

# api.md:176. An insert!(p) block with dispatch!, a HostCall, render! or build! throws and adds nothing: the next run compiles and records nothing.
@case "PIECE-05" 6 :graphics begin
    dev = testdevice()
    (; g, p, img) = framegraph(dev)
    x = MantleArray(dev, Int32, 1)
    other = Image(dev, RGBA{N0f8}, 8, 8)
    tlas = TLAS(dev; hardware = false)
    run!(g)
    blocks = (() -> dispatch!(g, pc_setone!, (x,), 1),
              () -> dispatch!(g, HostCall(_ -> nothing), (x,); writes = ()),
              () -> render!(q -> nothing, g, other => BACKGROUND),
              () -> build!(g, tlas))
    for b in blocks
        @test_throws ArgumentError insert!(b, p)
    end
    _, d = counted(() -> run!(g))
    @test d.plancompiles == 0
    @test d.records == 0
end

# internals.jl:887 (906 today). insert!(p) inside an insert!(p) block throws; the outer block adds nothing.
@case "PIECE-06" 6 :graphics begin
    dev = testdevice()
    (; g, p, img) = framegraph(dev)
    rects = MantleArray(dev, [stripe(1, 1)])
    colours = MantleArray(dev, [opaque(RED)])
    run!(g)
    @test_throws ArgumentError insert!(p) do
        insert!(() -> drawrects!(p, rects, colours), p)
    end
    _, d = counted(() -> run!(g))
    @test d.records == 0
    @test iscolour(stripepixel(Array(img), 1, 1), BLACK)
end

# api.md:178. A store into a piece's when! flag: the draw's count times the flag; nothing recorded, nothing compiled.
@case "PIECE-07" 6 :graphics begin
    dev = testdevice()
    (; g, p, img) = framegraph(dev)
    rects = MantleArray(dev, [stripe(1, 2)])
    colours = MantleArray(dev, [opaque(RED)])
    flag = MantleArray(dev, UInt32[1])
    insert!(p) do
        when!(() -> drawrects!(p, rects, colours), p, flag)
    end
    run!(g)
    @test iscolour(stripepixel(Array(img), 1, 2), RED)
    for (v, c) in ((0, BLACK), (1, RED))
        flag[1:1] = UInt32[v]
        _, d = counted(() -> run!(g))
        @test d.records == 0
        @test d.plancompiles == 0
        @test iscolour(stripepixel(Array(img), 1, 2), c)
    end
end

# api.md:465, recording-plan.md:1255-1258. Passes of 100, 1 000, 5 000 pieces, one piece inserted and deleted per frame: insert and delete time flat in the count; a visibility store records nothing.
@case "PIECE-08" 6 :graphics begin   # pending decision 8
    dev = testdevice()
    rects = MantleArray(dev, [Vec4f(-1, -1, -0.9, -0.9)])
    colours = MantleArray(dev, [opaque(RED)])
    passes = [crowdedpass(dev, n, rects, colours) for n in (100, 1000, 5000)]
    times = [Tuple{Float64,Float64}[] for _ in passes]
    for _ in 1:21, (s, t) in zip(passes, times)     # interleaved over the three passes
        push!(t, insertdelete(s, rects, colours, dev))
    end
    inserts = [medianof(first.(t)) for t in times]
    deletes = [medianof(last.(t)) for t in times]
    # Tolerance: at most twice the median at 100 pieces plus 1 ms (growth with
    # the count would be 10x and 50x).
    for i in 2:3
        @test inserts[i] <= 2inserts[1] + 1e-3
        @test deletes[i] <= 2deletes[1] + 1e-3
    end
    for s in passes
        s.flags[1][1:1] = UInt32[0]
        _, d = counted(() -> run!(s.g))
        @test d.records == 0
        @test d.plancompiles == 0
        free!(s.g)
    end
end

# ledger P6.A16 (L:603). A 4M-triangle mesh among 5 000 pieces draws in the time it takes alone: its time is the frame time with it visible minus hidden.
@case "PIECE-09" 6 :graphics begin
    dev = testdevice()
    v, i = gridmesh(4_000_000)
    vertices = MantleArray(dev, v)
    indices = MantleArray(dev, i)
    blue = MantleArray(dev, [opaque(BLUE)])
    flag = MantleArray(dev, UInt32[1])
    meshpiece!(p) = insert!(p) do
        when!(p, flag) do
            draw!(p, meshpipeline(), (vertices, blue), length(indices); indices)
        end
    end
    alone = framegraph(dev)
    meshpiece!(alone.p)
    small = MantleArray(dev, [Vec4f(-1, -1, -0.97, -0.97)])
    red = MantleArray(dev, [opaque(RED)])
    crowded = framegraph(dev)
    for k in 1:5000
        k == 2501 && meshpiece!(crowded.p)
        insert!(() -> drawrects!(crowded.p, small, red), crowded.p)
    end
    for _ in 1:3
        frametime(alone.g, dev); frametime(crowded.g, dev)
    end
    dalone, dcrowded = Float64[], Float64[]
    for _ in 1:9                                    # interleaved
        flag[1:1] = UInt32[1]
        va, vc = frametime(alone.g, dev), frametime(crowded.g, dev)
        flag[1:1] = UInt32[0]
        ha, hc = frametime(alone.g, dev), frametime(crowded.g, dev)
        push!(dalone, va - ha); push!(dcrowded, vc - hc)
    end
    ma, mc = medianof(dalone), medianof(dcrowded)
    @test ma > 0
    @test abs(mc - ma) <= 0.15ma                    # tolerance: 15% of the mesh's time alone
    free!(alone.g); free!(crowded.g)
end

# internals.jl:1031, 955 (1047-1048, 954 today). insert!(p), then insert!(g) before the run: one plan compile; the piece recorded once; registered once (its holds drop once).
@case "PIECE-10" 6 :graphics begin
    fd = FaultDevice(testdevice())
    initial = (rects = MantleArray(fd, [stripe(1, 2)]), colours = MantleArray(fd, [opaque(RED)]))
    (; g, p, img) = framegraph(fd) do p
        drawrects!(p, initial.rects, initial.colours)
    end
    x = MantleArray(fd, Int32, 1)
    run!(g)
    cells = memory(fd).cells
    a = piece!(p, fd, stripe(2, 2), opaque(GREEN))
    insert!(() -> dispatch!(g, pc_setone!, (x,), 1), g)
    pieces = calls(fd, :recordpiece)
    _, d = counted(() -> run!(g), fd)
    @test d.plancompiles == 1
    @test calls(fd, :recordpiece) - pieces == 2      # the first piece and the inserted one, once each
    @test iscolour(stripepixel(Array(img), 2, 2), GREEN)
    @test Array(x) == Int32[1]
    delete!(p, a.h)
    run!(g)
    free!(a.rects); free!(a.colours)
    @test memory(fd).cells == cells
end

# internals.jl:1031. A pass declared in an insert!(g) block gets a piece; delete!(g, block): the piece's holds drop.
@case "PIECE-11" 6 :graphics begin
    dev = testdevice()
    g = Graph(dev)
    img = Image(dev, RGBA{N0f8}, 64, 64)
    local p
    block = insert!(g) do
        p = render!(q -> nothing, g, img => BACKGROUND)
    end
    run!(g)
    cells = memory(dev).cells
    a = piece!(p, dev, stripe(1, 1), opaque(RED))
    run!(g)
    @test iscolour(stripepixel(Array(img), 1, 1), RED)
    delete!(g, block)
    run!(g)
    free!(a.rects); free!(a.colours)
    @test memory(dev).cells == cells
end

# internals.jl:902-906 (922-928 today; ambiguity 11). A second delete!(p, h) throws; the shared array's holds are not dropped twice.
@case "PIECE-12" 6 :graphics begin
    dev = testdevice()
    (; g, p, img) = framegraph(dev)
    green = MantleArray(dev, [opaque(GREEN)])
    left = MantleArray(dev, [stripe(1, 2)])
    right = MantleArray(dev, [stripe(2, 2)])
    h = insert!(() -> drawrects!(p, left, green), p)
    insert!(() -> drawrects!(p, right, green), p)
    run!(g)
    free!(green)                                     # only the graph's holds, one per draw, are left
    delete!(p, h)
    @test_throws ArgumentError delete!(p, h)
    run!(g)
    c = canary(dev)                                  # would take green's storage if it had been released
    run!(g)
    A = Array(img)
    @test iscolour(stripepixel(A, 1, 2), BLACK)
    @test iscolour(stripepixel(A, 2, 2), GREEN)
    @test intact(c)
end

# internals.jl:907 (unspecified in api.md). insert!(p) after the insert!(g) block holding p was deleted throws.
@case "PIECE-13" 6 :graphics begin
    dev = testdevice()
    g = Graph(dev)
    img = Image(dev, RGBA{N0f8}, 64, 64)
    rects = MantleArray(dev, [stripe(1, 1)])
    colours = MantleArray(dev, [opaque(RED)])
    local p
    block = insert!(g) do
        p = render!(q -> nothing, g, img => BACKGROUND)
    end
    run!(g)
    delete!(g, block)
    @test_throws ArgumentError insert!(() -> drawrects!(p, rects, colours), p)
    run!(g)
    @test_throws ArgumentError insert!(() -> drawrects!(p, rects, colours), p)
end

# internals.jl:826-830 (833-837 today). 10 000 insert/delete cycles of a piece over an array GPURef: the cell-copy table stays bounded, regions come back, memory at its baseline.
@case "PIECE-14" 6 :graphics :long begin
    dev = testdevice()
    (; g, p, img) = framegraph(dev)
    rects = GPURef(dev, MantleArray(dev, [stripe(1, 1)]))   # through a cell: the piece has cell-copy entries
    colours = MantleArray(dev, [opaque(RED)])
    cycle!() = (h = insert!(() -> drawrects!(p, rects, colours), p); run!(g); delete!(p, h); run!(g))
    foreach(_ -> cycle!(), 1:100)
    baseline = memory(dev)
    foreach(_ -> cycle!(), 1:10_000)
    @test memory(dev) == baseline
end

# internals.jl:835-836 (842-843 today). An inserted piece reads x, which graph S writes late; run!(S); run!(frame): the frame's first run after the insert waits for S.
@case "PIECE-15" 6 :graphics begin
    dev = testdevice()
    x = MantleArray(dev, [opaque(RED)])
    src = MantleArray(dev, [opaque(RED)])
    S = Graph(dev)
    dispatch!(S, pc_slowcopy!, (x, src, Int32(calibratedspin(dev, 200))), 1)
    run!(S)
    (; g, p, img) = framegraph(dev)
    run!(g)
    rects = MantleArray(dev, [stripe(1, 1)])
    insert!(() -> drawrects!(p, rects, x), p)
    src[1:1] = [opaque(GREEN)]
    run!(S)                                          # writes green into x about 200 ms from now
    run!(g)
    @test iscolour(stripepixel(Array(img), 1, 1), GREEN)
end

# internals.jl:842-844 (849-851 today). A window pass of 100 pieces; resize: each piece recorded once for the new generation; no plan compile.
@case "PIECE-16" 7 :graphics :window begin
    fd = FaultDevice(testdevice())
    win = Window(fd, 64, 64; capacity = (128, 128))
    rects = MantleArray(fd, [stripe(1, 1)])
    colours = MantleArray(fd, [opaque(GREEN)])
    g = Graph(fd)
    p = render!(g, win => BACKGROUND) do p
        drawrects!(p, rects, colours)
    end
    foreach(_ -> insert!(() -> drawrects!(p, rects, colours), p), 1:100)
    run!(g); run!(g)
    pieces = calls(fd, :recordpiece)
    resize!(win, 96, 80)
    _, d = counted(() -> run!(g), fd)
    @test d.plancompiles == 0
    @test calls(fd, :recordpiece) - pieces == 101
    run!(g)
    @test calls(fd, :recordpiece) - pieces == 101
end

# api.md:327. A kernel declared before the pass writes x late; a piece inserted after compile reads x: drawn with the kernel's value (the pass's fixed barrier).
@case "PIECE-17" 6 :graphics begin
    dev = testdevice()
    x = MantleArray(dev, [opaque(RED)])
    src = MantleArray(dev, [opaque(RED)])
    g = Graph(dev)
    dispatch!(g, pc_slowcopy!, (x, src, Int32(calibratedspin(dev, 200))), 1)
    img = Image(dev, RGBA{N0f8}, 64, 64)
    p = render!(q -> nothing, g, img => BACKGROUND)
    run!(g)
    rects = MantleArray(dev, [stripe(1, 1)])
    insert!(() -> drawrects!(p, rects, x), p)
    src[1:1] = [opaque(GREEN)]
    run!(g)
    @test iscolour(stripepixel(Array(img), 1, 1), GREEN)
end

# internals.jl:913-916 (936-937 today). Delete a piece whose last run is still on the GPU: its region and recording are retired after that run.
@case "PIECE-18" 6 :graphics begin
    dev = testdevice()
    g = Graph(dev)
    busy = MantleArray(dev, Int32, 1)
    slowwrite!(g, busy, 1)                           # each run stays on the GPU about 200 ms before the pass
    img = Image(dev, RGBA{N0f8}, 64, 64)
    p = render!(q -> nothing, g, img => BACKGROUND)
    run!(g)
    a = piece!(p, dev, stripe(1, 2), opaque(RED))
    run!(g)                                          # in flight
    delete!(p, a.h)
    run!(g)
    piece!(p, dev, stripe(2, 2), opaque(GREEN))      # new regions, where an early release would have put them
    c = canary(dev)
    withtimeout(() -> run!(g), 30)
    A = Array(img)
    @test iscolour(stripepixel(A, 1, 2), BLACK)
    @test iscolour(stripepixel(A, 2, 2), GREEN)
    @test intact(c)
end

# internals.jl:898 (918 today). insert!(p) before the first compile: the piece is in the first plan.
@case "PIECE-19" 6 :graphics begin
    dev = testdevice()
    (; g, p, img) = framegraph(dev)
    piece!(p, dev, stripe(1, 2), opaque(RED))
    _, d = counted(() -> run!(g))
    @test d.plancompiles == 1
    @test iscolour(stripepixel(Array(img), 1, 2), RED)
    _, d = counted(() -> run!(g))
    @test d.plancompiles == 0
    @test d.records == 0
end

# api.md:175 (unspecified). insert! into the pass of an eager render!(dev, …), which ran at the call: no run follows to apply it, so it throws.
@case "PIECE-20" 6 :graphics begin
    dev = testdevice()
    img = Image(dev, RGBA{N0f8}, 64, 64)
    rects = MantleArray(dev, [stripe(1, 2)])
    colours = MantleArray(dev, [opaque(RED)])
    p = render!(dev, img => BACKGROUND) do p
        drawrects!(p, rects, colours)
    end
    @test iscolour(stripepixel(Array(img), 1, 2), RED)
    more = MantleArray(dev, [stripe(2, 2)])
    @test_throws ArgumentError insert!(() -> drawrects!(p, more, colours), p)
end

# api.md:491. A piece whose draw has zero instances draws nothing and is kept: it draws once its array grows, with no compile or record.
@case "PIECE-21" 6 :graphics begin
    dev = testdevice()
    (; g, p, img) = framegraph(dev)
    rects = MantleArray(dev, Vec4f, 0)
    resize!(rects, 0)                                # resizable before compile: the draw is indirect
    colours = MantleArray(dev, [opaque(RED)])
    insert!(() -> drawrects!(p, rects, colours), p)
    run!(g)
    @test all(px -> iscolour(px, BLACK), Array(img))
    copy!(rects, [stripe(1, 2)])
    _, d = counted(() -> run!(g))
    @test d.plancompiles == 0
    @test d.records == 0
    @test iscolour(stripepixel(Array(img), 1, 2), RED)
end

# api.md:328. Pieces with different pipelines (opaque, additive, alpha) inserted alternately into two passes of different target sizes: each binds its own state.
@case "PIECE-22" 6 :graphics begin
    dev = testdevice()
    g = Graph(dev)
    big = Image(dev, RGBA{N0f8}, 64, 64)
    small = Image(dev, RGBA{N0f8}, 32, 16)
    pbig = render!(q -> nothing, g, big => BACKGROUND)
    psmall = render!(q -> nothing, g, small => BACKGROUND)
    run!(g)
    halfwhite = Vec4f(1, 1, 1, 0.5)
    piece!(pbig, dev, stripe(1, 4), opaque(RED))
    piece!(psmall, dev, stripe(2, 2), opaque(BLUE))
    piece!(pbig, dev, Vec4f(-1, -1, 1, 1), Vec4f(0, 0.5, 0, 1); blend = Mantle.Additive())
    piece!(psmall, dev, stripe(1, 2), halfwhite; blend = Mantle.AlphaBlend())
    piece!(pbig, dev, stripe(3, 4), opaque(BLUE))
    piece!(pbig, dev, stripe(4, 4), halfwhite; blend = Mantle.AlphaBlend())
    run!(g)
    A, B = Array(big), Array(small)
    @test iscolour(stripepixel(A, 1, 4), Vec3f(1, 0.5, 0))
    @test iscolour(stripepixel(A, 2, 4), Vec3f(0, 0.5, 0))
    @test iscolour(stripepixel(A, 3, 4), BLUE)
    @test iscolour(stripepixel(A, 4, 4), Vec3f(0.5, 0.75, 0.5))
    @test iscolour(stripepixel(B, 1, 2), Vec3f(0.5, 0.5, 0.5))
    @test iscolour(stripepixel(B, 2, 2), BLUE)
end
