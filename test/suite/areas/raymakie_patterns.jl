# RayMakie patterns (catalogue-resize-rt.md, table Q; resizing-and-raytracing.jl
# D.3; recording-plan.md, "RayMakie on the core model"), written against
# Mantle's API only, in D.3's shapes: one frame graph per screen whose pass
# holds one piece per plot (insertplot!, deleteplot!); a plot's attributes as
# array GPURefs with the plot's own arrays behind them (setattribute!);
# visibility as a store into the when! flag around its draw; the scene's
# rectangle as one array per scene, applied in the vertex stage. The cases
# check Mantle's side of each contract, so their phase is Mantle's (pieces and
# array GPURefs, P6.A16 and P6.A8), not RayMakie's (phase 11).
#
# Targets are read back with Array(img): a `w × h` matrix, first index x; plots
# are vertical stripes, so only x positions are checked. vertex_index and
# instance_index count from one.
#
# What this file assumes of the test hooks (as coalescing.jl and stores.jl):
# Mantle.pendingof(x) holds Change objects (Store, Resize, Rebind, BuildAccel
# with `mode` Build() or Update()); Mantle.steplog(dev) entries have `kind`
# (the step's name as a Symbol) and `target`. "Released" is read from
# memory(dev).cells: every array owns one cell, given back at its last hold.

using KernelAbstractions: @kernel, @index, @Const
using GeometryBasics: Vec2f, Vec3f, Vec4f, Point3f, TriangleFace
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

testpositions(n; seed = 1) = testdata(Point3f, n; seed)

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

"""The kinds of the changes pending on `x`, in sequence order."""
pendingkinds(x) = [nameof(typeof(c)) for c in Mantle.pendingof(x)]

"""The BuildAccel changes pending on a TLAS or BLAS (its range-table stores share its record)."""
pendingbuilds(t) = [c for c in Mantle.pendingof(t) if nameof(typeof(c)) === :BuildAccel]
buildmode(c) = nameof(typeof(c.mode))

# A plot of rectangles (a scatter reduced to what the cases check):
# rects[i] = (x0, y0, x1, y1) in the scene's coordinates [-1, 1]², colours one
# for all or one per rectangle (attribute(a, i), api.md 2.6b), scene[1] the
# scene's rectangle in the target, applied here (a layout change is one store).
function plot_vertex(rects, colours, scene)
    i = instance_index()
    r = Mantle.attribute(rects, i)
    s = scene[1]
    v = vertex_index()
    right = v == 2 || v == 4 || v == 5
    top = v == 3 || v == 5 || v == 6
    x = right ? r[3] : r[1]
    y = top ? r[4] : r[2]
    position = Vec4f(s[1] + (x + 1) / 2 * (s[3] - s[1]), s[2] + (y + 1) / 2 * (s[4] - s[2]), 0, 1)
    return (; position, colour = Mantle.attribute(colours, i))
end
rects_fragment(inputs, args...) = inputs.colour

# A text plot's vertex stage: also takes the glyph atlas (its texture index).
# The device-side call that samples a table texture is not named in api.md,
# so the shader takes the atlas and does not sample it.
glyph_vertex(rects, colours, scene, atlas) = plot_vertex(rects, colours, scene)

pipelineof(vertex; blend = Mantle.Opaque()) =
    Rasterizer(; vertex = Mantle.VertexShader(vertex; outputs = (colour = Vec4f,)),
               fragment = Mantle.FragmentShader(rects_fragment), topology = TriangleList(),
               blend, cull = Mantle.NoCull(), depth = Mantle.DepthOff())
plotpipeline(blend = Mantle.Opaque()) = pipelineof(plot_vertex; blend)
glyphpipeline() = pipelineof(glyph_vertex)

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

"""The four corners of each of `n` stripes: bottom left, bottom right, top left, top right."""
stripevertices(n) = [Vec2f(x, y) for k in 1:n for y in (-1, 1) for x in (stripe(k, n)[1], stripe(k, n)[3])]

"""0-based indices of the two triangles of each stripe in `ks`."""
stripeindices(ks) = UInt32[4(k - 1) + o for k in ks for o in (0, 1, 2, 1, 3, 2)]

"""
A plot as RayMakie keeps it (D.3, RenderObject): attributes as array GPURefs,
the plot's own arrays behind them (host data lands there), the when! flag
around its draw.
"""
struct RenderObject
    attributes::NamedTuple
    owned::NamedTuple
    visible::MantleArray
end
ownarray(dev, a::MantleArray) = a
ownarray(dev, data) = MantleArray(dev, data)
function RenderObject(dev, rects, colours)
    owned = (rects = ownarray(dev, rects), colours = ownarray(dev, colours))
    return RenderObject(map(a -> GPURef(dev, a), owned), owned, MantleArray(dev, UInt32[1]))
end

# update!(plot; name = data), once per changed attribute (D.3, setattribute!).
# A device array: a rebind, no copy. Host data (one value is a one-element
# vector): copy! into the plot's own array, and a rebind back to it if the
# attribute pointed elsewhere.
setattribute!(r::RenderObject, name, data::MantleArray) = (r.attributes[name][] = data; r)
function setattribute!(r::RenderObject, name, data::AbstractVector)
    own = r.owned[name]
    copy!(own, data)
    r.attributes[name][] === own || (r.attributes[name][] = own)
    return r
end
setvisible!(r::RenderObject, v::Bool) = (r.visible[1:1] = UInt32[v]; r)

"""
A screen (D.3, declareframe!): a frame graph with `before(g)` declared first,
then one pass into a target whose pieces are the plots, and the scene's
rectangle (the whole target).
"""
function newscreen(dev; side = 64, before = g -> nothing)
    g = Graph(dev)
    before(g)
    img = Image(dev, RGBA{N0f8}, side, side)
    scene = MantleArray(dev, [Vec4f(-1, -1, 1, 1)])
    pass = render!(p -> nothing, g, img => BACKGROUND)
    return (; graph = g, pass, img, scene)
end

"""The plot as a piece of the screen's pass, its draw under its flag (D.3, insertplot!); returns the piece."""
insertplot!(s, r::RenderObject; pipeline = plotpipeline()) =
    insert!(s.pass) do
        when!(s.pass, r.visible) do
            draw!(s.pass, pipeline, (r.attributes.rects, r.attributes.colours, s.scene), 6;
                  instances = launchrange(r.attributes.rects))
        end
    end
deleteplot!(s, piece) = delete!(s.pass, piece)

"""A plot inserted into `s` whose render object is garbage once this returns; returns the piece."""
plotandforget!(s, dev) = insertplot!(s, RenderObject(dev, [stripe(1, 1)], [opaque(RED)]))

"""A screen with `n` plots over the shared arrays `rects` and `colours`, compiled."""
function crowdedscreen(dev, n, rects, colours)
    s = newscreen(dev)
    foreach(_ -> insertplot!(s, RenderObject(dev, rects, colours)), 1:n)
    run!(s.graph)
    return s
end

"""Host time of inserting plot `r` and of deleting it, each with the frame that applies it."""
function plotinsertdelete(s, r, dev)
    Mantle.waitidle(dev)
    local h
    tinsert = @elapsed begin
        h = insertplot!(s, r)
        run!(s.graph)
    end
    Mantle.waitidle(dev)
    tdelete = @elapsed begin
        deleteplot!(s, h)
        run!(s.graph)
    end
    return tinsert, tdelete
end

"""A marker mesh's BLAS: one triangle (D.3: one BLAS per marker mesh, shared by the ranges that use it)."""
markerblas(dev) = BLAS(dev, MantleArray(dev, [Point3f(0, 0, 0), Point3f(0.1, 0, 0), Point3f(0, 0.1, 0)]),
                       MantleArray(dev, [TriangleFace{UInt32}(1, 2, 3)]))

"""A graph whose kernel takes the TLAS (ray queries, RR A): its runs apply the TLAS's pending builds."""
function touching(dev, tlas)
    g = Graph(dev)
    dispatch!(g, rm_touch!, (MantleArray(dev, Int32, 1), tlas), 1)
    return g
end

@kernel function rm_touch!(out, @Const(tlas))
    @inbounds out[1] = Int32(1)
end

@kernel function rm_copy!(dst, @Const(src))
    i = @index(Global)
    @inbounds dst[i] = src[i]
end

# Copies src[1] into dst[1] after about `iters` dependent steps: a write that
# lands late (the chain of helpers.jl's slowwrite!).
@kernel function rm_slowcopy!(dst, @Const(src), iters)
    acc = Int32(0)
    for i in Int32(1):iters
        acc = acc * Int32(1103515245) + i
    end
    @inbounds dst[1] = acc == Int32(-7) ? zero(eltype(dst)) : src[1]
end

# ── Q. RayMakie (D.3) (RM) ──

# RR:1441-1442 (setattribute!), decisions.md:157-170. update!(plot; positions = host data) of the same length: a store into the owned array; no rebind, compile, record or host wait.
@case "RM-01" 6 :graphics begin
    dev = testdevice()
    s = newscreen(dev)
    r = RenderObject(dev, [stripe(1, 4), stripe(3, 4)], [opaque(RED)])
    insertplot!(s, r)
    run!(s.graph)
    _, d = counted(() -> setattribute!(r, :rects, [stripe(2, 4), stripe(4, 4)]))
    @test d.hostwaits == 0
    @test submitcount(d) == 0
    @test r.attributes.rects[] === r.owned.rects
    @test isempty(pendingkinds(r.attributes.rects))   # no rebind
    @test :Store in pendingkinds(r.owned.rects)
    _, d = counted(() -> run!(s.graph))
    @test d.plancompiles == 0
    @test d.pipelinecompiles == 0
    @test d.records == 0
    @test d.hostwaits == 0
    A = Array(s.img)
    @test all(k -> iscolour(stripepixel(A, k, 4), RED), (2, 4))
    @test all(k -> iscolour(stripepixel(A, k, 4), BLACK), (1, 3))
end

# RR:1441-1442. The same with a new length: a resize and a store; the draw's instance count from the cell; no compile or record.
@case "RM-02" 6 :graphics begin
    dev = testdevice()
    s = newscreen(dev)
    r = RenderObject(dev, [stripe(2, 8), stripe(4, 8)], [opaque(RED)])
    insertplot!(s, r)
    run!(s.graph)
    setattribute!(r, :rects, [stripe(k, 8) for k in 1:2:8])   # 4 elements, past the owned array's storage
    @test pendingkinds(r.owned.rects) == [:Resize, :Store]
    @test length(r.owned.rects) == 4
    _, d = counted(() -> run!(s.graph))
    @test d.plancompiles == 0
    @test d.records == 0
    A = Array(s.img)
    @test all(k -> iscolour(stripepixel(A, k, 8), RED), 1:2:8)
    @test all(k -> iscolour(stripepixel(A, k, 8), BLACK), 2:2:8)
end

# RR:1445-1446 (one value), recording-plan.md:1199-1210. Colour per element, then one value, then per element again: lengths n, 1, n; one pipeline, no compile.
@case "RM-03" 6 :graphics begin
    dev = testdevice()
    s = newscreen(dev)
    r = RenderObject(dev, [stripe(k, 4) for k in 1:4], opaque.([RED, GREEN, BLUE, RED]))
    insertplot!(s, r)
    run!(s.graph)
    for colours in ([GREEN], [BLUE, RED, GREEN, BLUE])
        setattribute!(r, :colours, opaque.(colours))
        @test length(r.owned.colours) == length(colours)
        _, d = counted(() -> run!(s.graph))
        @test d.plancompiles == 0
        @test d.pipelinecompiles == 0
        A = Array(s.img)
        @test all(k -> iscolour(stripepixel(A, k, 4), colours[min(k, end)]), 1:4)
    end
end

# RR:1441-1444. positions = a device array: a rebind only; then positions = host data: a store into the owned array and a rebind back to it.
@case "RM-04" 6 :graphics begin
    dev = testdevice()
    s = newscreen(dev)
    r = RenderObject(dev, [stripe(1, 4)], [opaque(RED)])
    insertplot!(s, r)
    run!(s.graph)
    b = MantleArray(dev, [stripe(2, 4)])
    setattribute!(r, :rects, b)
    @test r.attributes.rects[] === b
    @test pendingkinds(r.attributes.rects) == [:Rebind]
    @test isempty(pendingkinds(r.owned.rects))       # no copy
    _, d = counted(() -> run!(s.graph))
    @test d.plancompiles == 0
    @test d.records == 0
    A = Array(s.img)
    @test iscolour(stripepixel(A, 2, 4), RED)
    @test iscolour(stripepixel(A, 1, 4), BLACK)
    setattribute!(r, :rects, [stripe(3, 4)])
    @test r.attributes.rects[] === r.owned.rects
    @test :Store in pendingkinds(r.owned.rects)
    @test pendingkinds(r.attributes.rects) == [:Rebind]
    _, d = counted(() -> run!(s.graph))
    @test d.plancompiles == 0
    A = Array(s.img)
    @test iscolour(stripepixel(A, 3, 4), RED)
    @test iscolour(stripepixel(A, 2, 4), BLACK)
end

# RR:1447 (setvisible!), decisions.md:255-268. visible false, then true: a flag store each; nothing recorded; instance count 0, then n.
@case "RM-05" 6 :graphics begin
    dev = testdevice()
    s = newscreen(dev)
    r = RenderObject(dev, [stripe(1, 2)], [opaque(RED)])
    insertplot!(s, r)
    run!(s.graph)
    for (v, c) in ((false, BLACK), (true, RED))
        setvisible!(r, v)
        _, d = counted(() -> run!(s.graph))
        @test d.records == 0
        @test d.plancompiles == 0
        @test iscolour(stripepixel(Array(s.img), 1, 2), c)
    end
end

# RR:1449-1465, recording-plan.md:1255-1258. Insert and delete one plot in screens of 100, 1 000, 5 000 plots: no plan compile; one piece compiled and recorded; insert time flat in the count.
@case "RM-06" 6 :graphics begin   # pending decision 8
    dev = testdevice()
    rects = MantleArray(dev, [Vec4f(-1, -1, -0.9, -0.9)])
    colours = MantleArray(dev, [opaque(RED)])
    screens = [crowdedscreen(dev, n, rects, colours) for n in (100, 1000, 5000)]
    r = RenderObject(dev, [stripe(1, 1)], [opaque(GREEN)])
    for s in screens
        h, d = counted(() -> (piece = insertplot!(s, r); run!(s.graph); piece))
        @test d.plancompiles == 0
        @test d.pipelinecompiles == 0
        @test d.records == 2                         # the plot's piece and the pass's list part
        @test iscolour(stripepixel(Array(s.img), 1, 1), GREEN)
        deleteplot!(s, h)
        run!(s.graph)
    end
    times = [Tuple{Float64,Float64}[] for _ in screens]
    for _ in 1:21, (s, t) in zip(screens, times)     # interleaved over the three screens
        push!(t, plotinsertdelete(s, r, dev))
    end
    inserts = [medianof(first.(t)) for t in times]
    deletes = [medianof(last.(t)) for t in times]
    # Tolerance: at most twice the median at 100 plots plus 1 ms (growth with
    # the count would be 10x and 50x).
    for i in 2:3
        @test inserts[i] <= 2inserts[1] + 1e-3
        @test deletes[i] <= 2deletes[1] + 1e-3
    end
    foreach(s -> free!(s.graph), screens)
end

# RR:1478-1485, recording-plan.md:1647-1667. A plot over a simulation's array, run!(sim) every frame: the frame waits for the simulation's write, the next write waits for the frame's draw; no host wait. (The redraw mark is RayMakie's reader, not checked here.)
@case "RM-07" 6 :graphics begin
    dev = testdevice()
    simrects = MantleArray(dev, [stripe(1, 8)])
    src = MantleArray(dev, [stripe(1, 8)])
    sim = Graph(dev)
    dispatch!(sim, rm_slowcopy!, (simrects, src, Int32(calibratedspin(dev, 200))), 1)
    busy = MantleArray(dev, Int32, 1)
    s = newscreen(dev; before = g -> slowwrite!(g, busy, 1; ms = 400))   # the frame draws about 400 ms after it starts
    r = RenderObject(dev, [stripe(1, 8)], [opaque(GREEN)])
    setattribute!(r, :rects, simrects)               # mesh!(sim's arrays): the GPURef points at them
    insertplot!(s, r)
    run!(sim); run!(s.graph)
    Mantle.waitidle(dev)
    for k in 2:4
        src[1:1] = [stripe(k, 8)]
        run!(sim)                                    # writes stripe k about 200 ms from now
        _, d = counted(() -> run!(s.graph))
        @test d.hostwaits == 0
        src[1:1] = [stripe(k + 4, 8)]
        run!(sim)                                    # without waiting for the frame it would write before the draw
        A = Array(s.img)
        @test iscolour(stripepixel(A, k, 8), GREEN)
        @test iscolour(stripepixel(A, k + 4, 8), BLACK)
        Mantle.waitidle(dev)
    end
end

# RR:1484-1485, api.md:493. The simulation's index array resized past its storage: the piece that bound it by handle is recorded again; no plan compile.
@case "RM-08" 6 :graphics begin
    dev = testdevice()
    vertices = MantleArray(dev, stripevertices(4))
    faces = MantleArray(dev, stripeindices(1:2))
    resize!(faces, length(faces))                    # resizable before any compile: launches over it are indirect
    allfaces = MantleArray(dev, stripeindices(1:4))
    sim = Graph(dev)
    dispatch!(sim, rm_copy!, (faces, allfaces), launchrange(faces))
    s = newscreen(dev)
    green = MantleArray(dev, [opaque(GREEN)])
    insert!(s.pass) do
        draw!(s.pass, meshpipeline(), (vertices, green), launchrange(faces); indices = faces)
    end
    run!(sim); run!(s.graph)
    A = Array(s.img)
    @test iscolour(stripepixel(A, 2, 4), GREEN)
    @test iscolour(stripepixel(A, 3, 4), BLACK)
    resize!(faces, length(allfaces))                 # moved by the next submission that touches it
    run!(sim)
    _, d = counted(() -> run!(s.graph))
    @test d.plancompiles == 0
    @test d.records >= 1
    _, d = counted(() -> run!(s.graph))
    @test d.records == 0
    A = Array(s.img)
    @test all(k -> iscolour(stripepixel(A, k, 4), GREEN), 1:4)
end

# RR:1506-1511. Ray-traced meshscatter, particles moved with the same count: a TLAS Update only; the marker's BLAS untouched.
@case "RM-09" 6 :graphics :rt begin
    dev = testdevice()
    tlas = TLAS(dev)
    marker = markerblas(dev)
    positions = GPURef(dev, MantleArray(dev, testpositions(100)))
    push!(tlas, marker, positions; scale = 1f0)
    g = touching(dev, tlas)
    run!(g)
    positions[][1:100] = testpositions(100; seed = 2)
    @test buildmode.(pendingbuilds(tlas)) == [:Update]
    @test isempty(pendingbuilds(marker))
    _, log = stepsduring(() -> run!(g), dev)
    @test count(==(:BuildAccel), stepkinds(log, tlas)) == 1
    @test isempty(stepkinds(log, marker))
end

# RR:1508-1511. The particle count changes: a TLAS Build (with the exact count, not observable through the API); the same BLAS, not rebuilt.
@case "RM-10" 6 :graphics :rt begin
    dev = testdevice()
    tlas = TLAS(dev)
    marker = markerblas(dev)
    positions = GPURef(dev, MantleArray(dev, testpositions(100)))
    push!(tlas, marker, positions; scale = 1f0)
    g = touching(dev, tlas)
    run!(g)
    copy!(positions[], testpositions(150; seed = 2))
    @test buildmode.(pendingbuilds(tlas)) == [:Build]
    @test isempty(pendingbuilds(marker))
    _, log = stepsduring(() -> run!(g), dev)
    @test count(==(:BuildAccel), stepkinds(log, tlas)) == 1
    @test isempty(stepkinds(log, marker))
end

# RR:1512-1514, RR:132 (visible!). Rebind to an array of the same length: Update; of another length: Build; hide the range: Update (mask 0).
@case "RM-11" 6 :graphics :rt begin
    dev = testdevice()
    tlas = TLAS(dev)
    positions = GPURef(dev, MantleArray(dev, testpositions(100)))
    h = push!(tlas, markerblas(dev), positions; scale = 1f0)
    g = touching(dev, tlas)
    run!(g)
    changes = ((() -> positions[] = MantleArray(dev, testpositions(100; seed = 2))) => :Update,
               (() -> positions[] = MantleArray(dev, testpositions(120; seed = 3))) => :Build,
               (() -> visible!(tlas, h, false)) => :Update)
    for (change, mode) in changes
        change()
        @test buildmode.(pendingbuilds(tlas)) == [mode]
        run!(g)
        @test isempty(pendingbuilds(tlas))
    end
end

# RR:1498-1505. Two meshscatters share a marker; delete one: the BLAS and its geometry survive; released once the last range is deleted and its build applied.
@case "RM-12" 6 :graphics :rt begin
    dev = testdevice()
    tlas = TLAS(dev)
    g = touching(dev, tlas)
    run!(g)
    cells = memory(dev).cells
    vertices = MantleArray(dev, [Point3f(0, 0, 0), Point3f(0.1, 0, 0), Point3f(0, 0.1, 0)])
    faces = MantleArray(dev, [TriangleFace{UInt32}(1, 2, 3)])
    marker = BLAS(dev, vertices, faces)
    h1 = push!(tlas, marker, GPURef(dev, MantleArray(dev, testpositions(10))); scale = 1f0)
    h2 = push!(tlas, marker, GPURef(dev, MantleArray(dev, testpositions(20; seed = 2))); scale = 1f0)
    free!(vertices); free!(faces); free!(marker)     # only the two ranges hold the marker now
    run!(g)
    delete!(tlas, h1)
    run!(g)
    @test memory(dev).cells >= cells + 3             # the marker's two geometry arrays and h2's positions
    delete!(tlas, h2)
    run!(g)
    @test memory(dev).cells == cells
end

# RR:1436, api.md:181. A layout change: one store into the scene's rectangle moves every plot of the scene; nothing recorded or compiled.
@case "RM-13" 6 :graphics begin
    dev = testdevice()
    s = newscreen(dev)
    insertplot!(s, RenderObject(dev, [stripe(1, 2)], [opaque(RED)]))
    insertplot!(s, RenderObject(dev, [stripe(2, 2)], [opaque(GREEN)]))
    run!(s.graph)
    s.scene[1:1] = [Vec4f(-1, -1, 0, 1)]             # the scene now covers the left half of the target
    _, d = counted(() -> run!(s.graph))
    @test d.records == 0
    @test d.plancompiles == 0
    A = Array(s.img)
    @test iscolour(stripepixel(A, 1, 4), RED)
    @test iscolour(stripepixel(A, 2, 4), GREEN)
    @test iscolour(stripepixel(A, 3, 4), BLACK)
    @test iscolour(stripepixel(A, 4, 4), BLACK)
end

# api.md:491. A plot with zero elements keeps its render object and draws nothing; it grows to n: drawn, no compile or record.
@case "RM-14" 6 :graphics begin
    dev = testdevice()
    s = newscreen(dev)
    r = RenderObject(dev, Vec4f[], [opaque(RED)])
    insertplot!(s, r)
    run!(s.graph)
    @test all(px -> iscolour(px, BLACK), Array(s.img))
    setattribute!(r, :rects, [stripe(1, 2)])
    _, d = counted(() -> run!(s.graph))
    @test d.plancompiles == 0
    @test d.records == 0
    @test iscolour(stripepixel(Array(s.img), 1, 2), RED)
end

# RR:1487-1491, recording-plan.md:1692-1699. The glyph atlas grows: no text or scatter piece recorded again or compiled; the frame and the glyph texels unchanged.
@case "RM-15" 6 :graphics begin   # pending decision 9
    dev = testdevice()
    s = newscreen(dev)
    glyphs = [RGBA{N0f8}(i / 255, j / 255, 0, 1) for i in 1:16, j in 1:16]
    atlas = Texture2D(dev, RGBA{N0f8}, 16, 16)
    copyto!(atlas, glyphs)
    text = RenderObject(dev, [stripe(1, 2)], [opaque(RED)])
    insert!(s.pass) do
        draw!(s.pass, glyphpipeline(), (text.attributes.rects, text.attributes.colours, s.scene, atlas), 6;
              instances = launchrange(text.attributes.rects))
    end
    insertplot!(s, RenderObject(dev, [stripe(2, 2)], [opaque(GREEN)]))
    run!(s.graph)
    before = Array(s.img)
    resize!(atlas, 32, 32)                           # a new image under a new index
    _, d = counted(() -> run!(s.graph))
    @test d.plancompiles == 0
    @test d.pipelinecompiles == 0
    @test d.records == 0
    @test Array(s.img) == before
    @test Array(atlas)[1:16, 1:16] == glyphs
end

# decisions.md:205-207 (B3.12). Toggling transparency: two draws in the piece, opaque and blended, each under a flag; a toggle is two flag stores, no compile or record.
@case "RM-16" 6 :graphics begin
    dev = testdevice()
    s = newscreen(dev)
    r = RenderObject(dev, [stripe(1, 2)], [Vec4f(1, 1, 1, 0.5)])
    opaqueflag = MantleArray(dev, UInt32[1])
    blendflag = MantleArray(dev, UInt32[0])
    args = (r.attributes.rects, r.attributes.colours, s.scene)
    insert!(s.pass) do
        when!(s.pass, opaqueflag) do
            draw!(s.pass, plotpipeline(Mantle.Opaque()), args, 6; instances = launchrange(r.attributes.rects))
        end
        when!(s.pass, blendflag) do
            draw!(s.pass, plotpipeline(Mantle.AlphaBlend()), args, 6; instances = launchrange(r.attributes.rects))
        end
    end
    run!(s.graph)
    @test iscolour(stripepixel(Array(s.img), 1, 2), Vec3f(1, 1, 1))
    for (o, b, c) in ((0, 1, Vec3f(0.5, 0.5, 0.5)), (1, 0, Vec3f(1, 1, 1)))
        opaqueflag[1:1] = UInt32[o]
        blendflag[1:1] = UInt32[b]
        _, d = counted(() -> run!(s.graph))
        @test d.plancompiles == 0
        @test d.pipelinecompiles == 0
        @test d.records == 0
        @test iscolour(stripepixel(Array(s.img), 1, 2), c)
    end
end

# RR:1429-1430, RR:1322-1323. Delete a plot and drop every reference to it: the holds drop through the finalizers; memory back at its baseline.
@case "RM-17" 6 :graphics begin
    dev = testdevice()
    s = newscreen(dev)
    run!(s.graph)
    h = plotandforget!(s, dev)                       # one cycle first: the plan's tables grow once
    run!(s.graph)
    deleteplot!(s, h)
    run!(s.graph)
    h = nothing
    baseline = memory(dev)
    h = plotandforget!(s, dev)
    run!(s.graph)
    @test iscolour(stripepixel(Array(s.img), 1, 1), RED)
    deleteplot!(s, h)
    run!(s.graph)
    h = nothing
    @test memory(dev) == baseline
end
