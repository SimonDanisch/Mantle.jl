# Pieces: groups of draws (api.md 2.3, 2.6b; recording-plan.md, Graphics;
# internals.jl section 5, "Render passes: formed by compile" and "Pieces:
# groups of draws change without a recompile"). Compile forms the render
# passes from clears and draws; there is no pass object. A group whose nodes
# are draws into one pass is a piece of that pass, compiled and recorded on its
# own. A group of draws made inside the group that holds the pass's clear,
# draws declared into it, delete!(g, group) and sort! of such groups are
# applied by the graph's next run, which compiles and records the group's
# draws and rewrites the pass's list, without a plan compile.
#
# Targets are read back with Array(img): a `w × h` matrix, first index x. The
# cases draw vertical stripes over the whole height, so only x positions are
# checked. vertex_index and instance_index count from one.

using GeometryBasics: Vec2f, Vec3f, Vec4f
using ColorTypes: RGBA
using FixedPointNumbers: N0f8

const BLACK = Vec3f(0, 0, 0)
const RED = Vec3f(1, 0, 0)
const GREEN = Vec3f(0, 1, 0)
const BLUE = Vec3f(0, 0, 1)
const BACKGROUND = (0f0, 0f0, 0f0, 1f0)   # the clear value: clear!(l, target, BACKGROUND)

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

"""
    drawrects!(l, target, rects, colours; blend)
    drawrects!(l, target, flag, rects, colours; blend)

One draw of `rects` in `colours` into `target`, declared into `l` (a graph or a
group; on a device it runs now, as its own pass). With `flag`, a `when!` draw:
its count times the `Bool` flag.
"""
drawrects!(l, target, rects, colours; blend = Mantle.Opaque()) =
    draw!(l, target, rectpipeline(blend), (rects, colours), 6; instances = launchrange(rects))
drawrects!(l, target, flag, rects, colours; blend = Mantle.Opaque()) =
    when!(l, target, flag, rectpipeline(blend), (rects, colours), 6; instances = launchrange(rects))

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
A graph with one pass into a new square target cleared to black. `pass` is the
group that holds the pass's clear: groups of draws made in it are pieces of the
pass, and `f(pass, img)` declares draws into it directly (the pass's first
piece). Returns the graph, the group and the target.
"""
function framegraph(f, dev; side = 64)
    g = Graph(dev)
    img = Image(dev, RGBA{N0f8}, side, side)
    pass = Group(g, "pass")
    clear!(pass, img, BACKGROUND)                    # the load operation of the pass
    f(pass, img)
    return (; g, pass, img)
end
framegraph(dev; side = 64) = framegraph((pass, img) -> nothing, dev; side)

"""
A group of draws in `pass` (a piece of its pass) drawing `rect` in `colour` into
`target` from arrays of its own; returns the group and the arrays.
"""
function piece!(pass, target, dev, rect::Vec4f, colour::Vec4f; blend = Mantle.Opaque())
    rects = MantleArray(dev, [rect])
    colours = MantleArray(dev, [colour])
    group = Group(pass, "piece")
    drawrects!(group, target, rects, colours; blend)
    return (; group, rects, colours)
end

"""A pass of `n` groups of draws drawing `rects` in `colours`, each under its own `when!` flag, compiled."""
function crowdedpass(dev, n, rects, colours)
    (; g, pass, img) = framegraph(dev)
    flags = [MantleArray(dev, [true]) for _ in 1:n]
    foreach(f -> drawrects!(Group(pass, "piece"), img, f, rects, colours), flags)
    run!(g)
    return (; g, pass, img, flags)
end

"""Host time of making one group of draws and of deleting it, each with the run that applies it."""
function insertdelete(s, rects, colours, dev)
    Mantle.waitidle(dev)
    local group
    tinsert = @elapsed begin
        group = Group(s.pass, "piece")
        drawrects!(group, s.img, rects, colours)
        run!(s.g)
    end
    Mantle.waitidle(dev)
    tdelete = @elapsed begin
        delete!(s.g, group)
        run!(s.g)
    end
    return tinsert, tdelete
end

"""Time of one frame of `g` on the GPU and the host: the run and a wait for the device."""
frametime(g, dev) = @elapsed (run!(g); Mantle.waitidle(dev))

"""a[1] = 1, by one work-item (the ndrange of 1 is rounded up to a whole work-group)."""
function pc_setone!(a)
    KI.get_global_id().x == 1 || return nothing
    @inbounds a[1] = one(eltype(a))
    return nothing
end

# Copies src[1] into dst[1] after about `iters` dependent steps: a write that
# lands late (the chain of helpers.jl's slowwrite!). One work-item.
function pc_slowcopy!(dst, src, iters)
    KI.get_global_id().x == 1 || return nothing
    acc = Int32(0)
    for i in Int32(1):iters
        acc = acc * Int32(1103515245) + i
    end
    @inbounds dst[1] = acc == Int32(-7) ? zero(eltype(dst)) : src[1]
    return nothing
end

# ── Pieces: groups of draws (PIECE) ──

# api.md 2.6b (a group of draws). A group of draws made in a compiled pass, run: drawn; no plan compile; the piece and the list part recorded.
@case "PIECE-01" 6 :graphics begin
    dev = testdevice()
    (; g, pass, img) = framegraph(dev)
    run!(g)
    piece!(pass, img, dev, stripe(1, 2), opaque(RED))
    _, d = counted(() -> run!(g))
    @test d.plancompiles == 0
    @test d.records == 2                 # the piece and the pass's list part
    @test d.pipelinecompiles <= 1        # the piece's pipeline, counted apart from plan compiles
    A = Array(img)
    @test iscolour(stripepixel(A, 1, 2), RED)
    @test iscolour(stripepixel(A, 2, 2), BLACK)
    piece!(pass, img, dev, stripe(2, 2), opaque(GREEN))   # the same pipeline: from the cache
    _, d = counted(() -> run!(g))
    @test d.plancompiles == 0
    @test d.pipelinecompiles == 0
    @test d.records == 2
    @test iscolour(stripepixel(Array(img), 2, 2), GREEN)
end

# api.md 2.3 (delete!), 2.6b. delete!(g, group) of a group of draws and run: the piece is gone; no plan compile; only the list part is recorded, no other piece.
@case "PIECE-02" 6 :graphics begin
    dev = testdevice()
    (; g, pass, img) = framegraph(dev)
    a = piece!(pass, img, dev, stripe(1, 2), opaque(RED))
    piece!(pass, img, dev, stripe(2, 2), opaque(GREEN))
    run!(g)
    delete!(g, a.group)
    _, d = counted(() -> run!(g))
    @test d.plancompiles == 0
    @test d.records == 1
    A = Array(img)
    @test iscolour(stripepixel(A, 1, 2), BLACK)
    @test iscolour(stripepixel(A, 2, 2), GREEN)
end

# internals.jl section 5 (applychanges!, run by run!). A group of draws changes nothing until the next run: the target read before run! is unchanged.
@case "PIECE-03" 6 :graphics begin
    dev = testdevice()
    (; g, pass, img) = framegraph(dev)
    run!(g)
    before = Array(img)
    piece!(pass, img, dev, stripe(1, 1), opaque(RED))
    @test Array(img) == before
    run!(g)
    @test iscolour(stripepixel(Array(img), 1, 1), RED)
end

# api.md 2.6b; internals.jl section 5 (recordgroup!). The group's pipeline fails to compile: run! throws; the group is deleted, its holds dropped; the next run works.
@case "PIECE-04" 6 :graphics begin
    fd = FaultDevice(testdevice())
    (; g, pass, img) = framegraph(fd)
    run!(g)
    cells = memory(fd).cells
    inject!(fd, :compiledraw, Throw(calls(fd, :compiledraw) + 1, ErrorException("injected pipeline failure")))
    a = piece!(pass, img, fd, stripe(1, 2), opaque(RED))
    @test_throws ErrorException run!(g)
    _, d = counted(() -> run!(g), fd)
    @test d.plancompiles == 0
    @test iscolour(stripepixel(Array(img), 1, 2), BLACK)
    @test_throws ArgumentError delete!(g, a.group)   # deleted by the failed run already
    free!(a.rects); free!(a.colours)
    @test memory(fd).cells == cells               # the graph holds neither array
    piece!(pass, img, fd, stripe(2, 2), opaque(GREEN))
    run!(g)
    @test iscolour(stripepixel(Array(img), 2, 2), GREEN)
end

# api.md 2.3 (Group), 2.6b; internals.jl section 5 (piecechange). A host node or a build has no method for a group in the pass: a MethodError, nothing added, the next run compiles and records nothing. A group in the pass holding a launch, a clear of another target, or a draw into another target (a new pass) throws nothing: the next run compiles the plan.
@case "PIECE-05" 6 :graphics begin
    dev = testdevice()
    (; g, pass, img) = framegraph(dev)
    x = MantleArray(dev, Int32, 1)
    other1 = Image(dev, RGBA{N0f8}, 8, 8)
    other2 = Image(dev, RGBA{N0f8}, 8, 8)
    rects = MantleArray(dev, [stripe(1, 1)])
    colours = MantleArray(dev, [opaque(RED)])
    tlas = TLAS(dev; hardware = false)
    group = Group(pass, "piece")                     # empty, in the first plan
    run!(g)
    @test_throws MethodError dispatch!(group, HostCall(_ -> nothing), (x,); writes = ())
    @test_throws MethodError build!(group, tlas)
    _, d = counted(() -> run!(g))
    @test d.plancompiles == 0
    @test d.records == 0
    blocks = (gr -> dispatch!(gr, pc_setone!, (x,), 1),
              gr -> clear!(gr, other1, BACKGROUND),
              gr -> drawrects!(gr, other2, rects, colours))
    for b in blocks
        b(Group(pass, "piece"))
        _, d = counted(() -> run!(g))
        @test d.plancompiles == 1
    end
    @test Array(x) == Int32[1]
end

# internals.jl section 5 (formpasses! cuts pieces by holder; piecechange). A group made inside a group of draws after compile, with a draw: a piece of its own; the next run draws it with no plan compile and records its piece and the list part only.
@case "PIECE-06" 6 :graphics begin
    dev = testdevice()
    (; g, pass, img) = framegraph(dev)
    outer = piece!(pass, img, dev, stripe(1, 2), opaque(RED))
    rects = MantleArray(dev, [stripe(2, 2)])
    colours = MantleArray(dev, [opaque(GREEN)])
    run!(g)
    drawrects!(Group(outer.group, "inner"), img, rects, colours)
    _, d = counted(() -> run!(g))
    @test d.plancompiles == 0
    @test d.records == 2                             # the inner group's piece and the pass's list part
    A = Array(img)
    @test iscolour(stripepixel(A, 1, 2), RED)
    @test iscolour(stripepixel(A, 2, 2), GREEN)
end

# api.md 2.6b (when!). A store into a piece's when! flag: the draw's count times the flag; nothing recorded, nothing compiled.
@case "PIECE-07" 6 :graphics begin
    dev = testdevice()
    (; g, pass, img) = framegraph(dev)
    rects = MantleArray(dev, [stripe(1, 2)])
    colours = MantleArray(dev, [opaque(RED)])
    flag = MantleArray(dev, [true])
    drawrects!(Group(pass, "piece"), img, flag, rects, colours)
    run!(g)
    @test iscolour(stripepixel(Array(img), 1, 2), RED)
    for (v, c) in ((false, BLACK), (true, RED))
        flag[1:1] = [v]
        _, d = counted(() -> run!(g))
        @test d.records == 0
        @test d.plancompiles == 0
        @test iscolour(stripepixel(Array(img), 1, 2), c)
    end
end

# api.md 6; recording-plan.md, Graphics (the gate). Passes of 100, 1 000, 5 000 groups of draws, one made and deleted per frame: make and delete time flat in the count; a visibility store records nothing.
@case "PIECE-08" 6 :graphics begin
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
        s.flags[1][1:1] = [false]
        _, d = counted(() -> run!(s.g))
        @test d.records == 0
        @test d.plancompiles == 0
        free!(s.g)
    end
end

# recording-plan.md, Graphics; internals.jl section 5 (each draw keeps its own pipeline, index buffer and argument layout). A 4M-triangle mesh among 5 000 groups of draws draws in the time it takes alone: its time is the frame time with it visible minus hidden.
@case "PIECE-09" 6 :graphics begin
    dev = testdevice()
    v, i = gridmesh(4_000_000)
    vertices = MantleArray(dev, v)
    indices = MantleArray(dev, i)
    blue = MantleArray(dev, [opaque(BLUE)])
    flag = MantleArray(dev, [true])
    meshpiece!(pass, img) =
        when!(Group(pass, "mesh"), img, flag, meshpipeline(), (vertices, blue), length(indices); indices)
    alone = framegraph(dev)
    meshpiece!(alone.pass, alone.img)
    small = MantleArray(dev, [Vec4f(-1, -1, -0.97, -0.97)])
    red = MantleArray(dev, [opaque(RED)])
    crowded = framegraph(dev)
    for k in 1:5000
        k == 2501 && meshpiece!(crowded.pass, crowded.img)
        drawrects!(Group(crowded.pass, "piece"), crowded.img, small, red)
    end
    for _ in 1:3
        frametime(alone.g, dev); frametime(crowded.g, dev)
    end
    dalone, dcrowded = Float64[], Float64[]
    for _ in 1:9                                    # interleaved
        flag[1:1] = [true]
        va, vc = frametime(alone.g, dev), frametime(crowded.g, dev)
        flag[1:1] = [false]
        ha, hc = frametime(alone.g, dev), frametime(crowded.g, dev)
        push!(dalone, va - ha); push!(dcrowded, vc - hc)
    end
    ma, mc = medianof(dalone), medianof(dcrowded)
    @test ma > 0
    @test abs(mc - ma) <= 0.15ma                    # tolerance: 15% of the mesh's time alone
    free!(alone.g); free!(crowded.g)
end

# internals.jl section 5 (applychanges!: a recompile compiles every piece). A group of draws, then a group holding a launch, before the run: one plan compile; each piece recorded once; registered once (its holds drop once).
@case "PIECE-10" 6 :graphics begin
    fd = FaultDevice(testdevice())
    initial = (rects = MantleArray(fd, [stripe(1, 2)]), colours = MantleArray(fd, [opaque(RED)]))
    (; g, pass, img) = framegraph(fd) do pass, img
        drawrects!(pass, img, initial.rects, initial.colours)
    end
    x = MantleArray(fd, Int32, 1)
    run!(g)
    cells = memory(fd).cells
    a = piece!(pass, img, fd, stripe(2, 2), opaque(GREEN))
    dispatch!(Group(g, "launch"), pc_setone!, (x,), 1)
    pieces = calls(fd, :recordpiece)
    _, d = counted(() -> run!(g), fd)
    @test d.plancompiles == 1
    @test calls(fd, :recordpiece) - pieces == 2      # the first piece and the new group's, once each
    @test iscolour(stripepixel(Array(img), 2, 2), GREEN)
    @test Array(x) == Int32[1]
    delete!(g, a.group)
    run!(g)
    free!(a.rects); free!(a.colours)
    @test memory(fd).cells == cells
end

# api.md 2.3 (delete!); internals.jl section 5. A pass whose clear is declared in a group gets a group of draws; delete!(g, block): the block and its piece go, the piece's holds drop.
@case "PIECE-11" 6 :graphics begin
    dev = testdevice()
    g = Graph(dev)
    img = Image(dev, RGBA{N0f8}, 64, 64)
    block = Group(g, "block")
    clear!(block, img, BACKGROUND)
    run!(g)
    cells = memory(dev).cells
    a = piece!(block, img, dev, stripe(1, 1), opaque(RED))
    run!(g)
    @test iscolour(stripepixel(Array(img), 1, 1), RED)
    delete!(g, block)
    run!(g)
    free!(a.rects); free!(a.colours)
    @test memory(dev).cells == cells
end

# internals.jl section 2 (delete!). A second delete!(g, group) throws; the shared array's holds are not dropped twice.
@case "PIECE-12" 6 :graphics begin
    dev = testdevice()
    (; g, pass, img) = framegraph(dev)
    green = MantleArray(dev, [opaque(GREEN)])
    left = MantleArray(dev, [stripe(1, 2)])
    right = MantleArray(dev, [stripe(2, 2)])
    group = Group(pass, "left")
    drawrects!(group, img, left, green)
    drawrects!(Group(pass, "right"), img, right, green)
    run!(g)
    free!(green)                                     # only the graph's holds, one per draw, are left
    delete!(g, group)
    @test_throws ArgumentError delete!(g, group)
    run!(g)
    c = canary(dev)                                  # would take green's storage if it had been released
    run!(g)
    A = Array(img)
    @test iscolour(stripepixel(A, 1, 2), BLACK)
    @test iscolour(stripepixel(A, 2, 2), GREEN)
    @test intact(c)
end

# internals.jl section 2 (addnode!: declaring into a deleted group throws). A group made in a block that holds the pass's clear, after delete!(g, block), throws, before and after the run.
@case "PIECE-13" 6 :graphics begin
    dev = testdevice()
    g = Graph(dev)
    img = Image(dev, RGBA{N0f8}, 64, 64)
    block = Group(g, "block")
    clear!(block, img, BACKGROUND)
    run!(g)
    delete!(g, block)
    @test_throws ArgumentError Group(block, "piece")
    run!(g)
    @test_throws ArgumentError Group(block, "piece")
end

# internals.jl section 5 (a piece's cell-copy entries are blanked and reused). 10 000 make/delete cycles of a group of draws over a resizable array: the cell-copy table stays bounded, regions come back, memory at its baseline.
@case "PIECE-14" 6 :graphics :long begin
    dev = testdevice()
    (; g, pass, img) = framegraph(dev)
    rects = MantleArray(dev, [stripe(1, 1)])
    resize!(rects, 1)                                # resizable: read through its cell, so the piece has cell-copy entries
    colours = MantleArray(dev, [opaque(RED)])
    function cycle!()
        group = Group(pass, "piece")
        drawrects!(group, img, rects, colours)
        run!(g)
        delete!(g, group)
        run!(g)
    end
    foreach(_ -> cycle!(), 1:100)
    baseline = memory(dev)
    foreach(_ -> cycle!(), 1:10_000)
    @test memory(dev) == baseline
end

# internals.jl section 5 (registerpiece!: a piece's own registration). A group of draws made after compile reads x, which graph S writes late; run!(S); run!(frame): the frame's first run after the change waits for S.
@case "PIECE-15" 6 :graphics begin
    dev = testdevice()
    x = MantleArray(dev, [opaque(RED)])
    src = MantleArray(dev, [opaque(RED)])
    S = Graph(dev)
    dispatch!(S, pc_slowcopy!, (x, src, Int32(calibratedspin(dev, 200))), 1)
    run!(S)
    (; g, pass, img) = framegraph(dev)
    run!(g)
    rects = MantleArray(dev, [stripe(1, 1)])
    drawrects!(Group(pass, "piece"), img, rects, x)
    src[1:1] = [opaque(GREEN)]
    run!(S)                                          # writes green into x about 200 ms from now
    run!(g)
    @test iscolour(stripepixel(Array(img), 1, 1), GREEN)
end

# internals.jl section 5 (the pieces of a pass into a window are recorded per swapchain generation). A window pass of 100 groups of draws; resize: each piece recorded once for the new generation; no plan compile.
@case "PIECE-16" 7 :graphics :window begin
    fd = FaultDevice(testdevice())
    win = Window(fd, 64, 64; capacity = (128, 128))
    rects = MantleArray(fd, [stripe(1, 1)])
    colours = MantleArray(fd, [opaque(GREEN)])
    g = Graph(fd)
    pass = Group(g, "pass")
    clear!(pass, win, BACKGROUND)
    drawrects!(pass, win, rects, colours)
    foreach(_ -> drawrects!(Group(pass, "piece"), win, rects, colours), 1:100)
    run!(g); run!(g)
    pieces = calls(fd, :recordpiece)
    resize!(win, 96, 80)
    _, d = counted(() -> run!(g), fd)
    @test d.plancompiles == 0
    @test calls(fd, :recordpiece) - pieces == 101
    run!(g)
    @test calls(fd, :recordpiece) - pieces == 101
end

# api.md 4.5; plan, Graphics (every pass starts with a fixed barrier). A kernel declared before the pass writes x late; a group of draws made after compile reads x: drawn with the kernel's value.
@case "PIECE-17" 6 :graphics begin
    dev = testdevice()
    x = MantleArray(dev, [opaque(RED)])
    src = MantleArray(dev, [opaque(RED)])
    g = Graph(dev)
    dispatch!(g, pc_slowcopy!, (x, src, Int32(calibratedspin(dev, 200))), 1)
    img = Image(dev, RGBA{N0f8}, 64, 64)
    pass = Group(g, "pass")
    clear!(pass, img, BACKGROUND)
    run!(g)
    rects = MantleArray(dev, [stripe(1, 1)])
    drawrects!(Group(pass, "piece"), img, rects, x)
    src[1:1] = [opaque(GREEN)]
    run!(g)
    @test iscolour(stripepixel(Array(img), 1, 1), GREEN)
end

# internals.jl section 5 (deleted!: retired with the plan's last tokens). Delete a group of draws whose last run is still on the GPU: its region and recording are retired after that run.
@case "PIECE-18" 6 :graphics begin
    dev = testdevice()
    g = Graph(dev)
    busy = MantleArray(dev, Int32, 1)
    slowwrite!(g, busy, 1)                           # each run stays on the GPU about 200 ms before the pass
    img = Image(dev, RGBA{N0f8}, 64, 64)
    pass = Group(g, "pass")
    clear!(pass, img, BACKGROUND)
    run!(g)
    a = piece!(pass, img, dev, stripe(1, 2), opaque(RED))
    run!(g)                                          # in flight
    delete!(g, a.group)
    run!(g)
    piece!(pass, img, dev, stripe(2, 2), opaque(GREEN))   # new regions, where an early release would have put them
    c = canary(dev)
    withtimeout(() -> run!(g), 30)
    A = Array(img)
    @test iscolour(stripepixel(A, 1, 2), BLACK)
    @test iscolour(stripepixel(A, 2, 2), GREEN)
    @test intact(c)
end

# api.md 2.3 (run!: the first call compiles). A group of draws made before the first compile: the piece is in the first plan.
@case "PIECE-19" 6 :graphics begin
    dev = testdevice()
    (; g, pass, img) = framegraph(dev)
    piece!(pass, img, dev, stripe(1, 2), opaque(RED))
    _, d = counted(() -> run!(g))
    @test d.plancompiles == 1
    @test iscolour(stripepixel(Array(img), 1, 2), RED)
    _, d = counted(() -> run!(g))
    @test d.plancompiles == 0
    @test d.records == 0
end

# api.md 2.3 (Group), 2.6b (clear!, draw! on a device). On a device a clear and a draw run at the call, each as its own pass; a group has no method for a device, so nothing is added to an eager pass (MethodError). A later eager draw is a pass of its own and keeps the target's contents.
@case "PIECE-20" 6 :graphics begin
    dev = testdevice()
    img = Image(dev, RGBA{N0f8}, 64, 64)
    rects = MantleArray(dev, [stripe(1, 2)])
    colours = MantleArray(dev, [opaque(RED)])
    clear!(dev, img, BACKGROUND)
    drawrects!(dev, img, rects, colours)
    @test iscolour(stripepixel(Array(img), 1, 2), RED)
    @test_throws MethodError Group(dev, "piece")
    more = MantleArray(dev, [stripe(2, 2)])
    drawrects!(dev, img, more, colours)              # no clear before it: the contents are kept
    A = Array(img)
    @test iscolour(stripepixel(A, 1, 2), RED)
    @test iscolour(stripepixel(A, 2, 2), RED)
end

# api.md 7 (a plot with zero elements). A group whose draw has zero instances draws nothing and is kept: it draws once its array grows, with no compile or record.
@case "PIECE-21" 6 :graphics begin
    dev = testdevice()
    (; g, pass, img) = framegraph(dev)
    rects = MantleArray(dev, Vec4f, 0)
    resize!(rects, 0)                                # resizable before compile: the draw is indirect
    colours = MantleArray(dev, [opaque(RED)])
    drawrects!(Group(pass, "piece"), img, rects, colours)
    run!(g)
    @test all(px -> iscolour(px, BLACK), Array(img))
    copy!(rects, [stripe(1, 2)])
    _, d = counted(() -> run!(g))
    @test d.plancompiles == 0
    @test d.records == 0
    @test iscolour(stripepixel(Array(img), 1, 2), RED)
end

# api.md 4.5 (recordpiece: each piece binds its own pipelines and sets viewport and scissor). Groups of draws with different pipelines (opaque, additive, alpha) made alternately in two passes of different target sizes: each binds its own state.
@case "PIECE-22" 6 :graphics begin
    dev = testdevice()
    g = Graph(dev)
    big = Image(dev, RGBA{N0f8}, 64, 64)
    small = Image(dev, RGBA{N0f8}, 32, 16)
    pbig = Group(g, "big")
    clear!(pbig, big, BACKGROUND)
    psmall = Group(g, "small")
    clear!(psmall, small, BACKGROUND)
    run!(g)
    halfwhite = Vec4f(1, 1, 1, 0.5)
    piece!(pbig, big, dev, stripe(1, 4), opaque(RED))
    piece!(psmall, small, dev, stripe(2, 2), opaque(BLUE))
    piece!(pbig, big, dev, Vec4f(-1, -1, 1, 1), Vec4f(0, 0.5, 0, 1); blend = Mantle.Additive())
    piece!(psmall, small, dev, stripe(1, 2), halfwhite; blend = Mantle.AlphaBlend())
    piece!(pbig, big, dev, stripe(3, 4), opaque(BLUE))
    piece!(pbig, big, dev, stripe(4, 4), halfwhite; blend = Mantle.AlphaBlend())
    run!(g)
    A, B = Array(big), Array(small)
    @test iscolour(stripepixel(A, 1, 4), Vec3f(1, 0.5, 0))
    @test iscolour(stripepixel(A, 2, 4), Vec3f(0, 0.5, 0))
    @test iscolour(stripepixel(A, 3, 4), BLUE)
    @test iscolour(stripepixel(A, 4, 4), Vec3f(0.5, 0.75, 0.5))
    @test iscolour(stripepixel(B, 1, 2), Vec3f(0.5, 0.5, 0.5))
    @test iscolour(stripepixel(B, 2, 2), BLUE)
end
