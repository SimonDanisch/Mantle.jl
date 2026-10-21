# RayMakie patterns (resizing-and-raytracing.jl D.3; recording-plan.md,
# "RayMakie on the core model" and Graphics), written against Mantle's API
# only, in D.3's shapes: one frame graph per screen whose plots' group starts
# with the pass's clear and holds one group of draws per plot (insertplot!,
# deleteplot!); a plot's attributes as plain device arrays, mutated in place,
# with the plot's own arrays for host data, and another array object for an
# attribute as the plot's group deleted and a new group of draws over it
# (setattribute!, redeclare!); visibility as a store into the when! flag of its
# draw; the scene's rectangle as one array per scene, applied in the vertex
# stage. The cases check Mantle's side of each contract, so their phase is
# Mantle's (groups of draws), not RayMakie's (phase 11).
#
# Targets are read back with Array(img): a `w × h` matrix, first index x; plots
# are vertical stripes, so only x positions are checked. vertex_index and
# instance_index count from one.
#
# What this file assumes of the test hooks (as coalescing.jl and stores.jl):
# Mantle.pendingof(x) holds Change objects (Store, Resize, BuildAccel with
# `mode` Build() or Update()); Mantle.steplog(dev) entries have `kind`
# (the step's name as a Symbol) and `target`. "Released" is read from
# memory(dev).cells: every array owns one cell, given back at its last hold.

using GeometryBasics: Vec2f, Vec3f, Vec4f, Point3f, TriangleFace
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

# The glyph path really samples the atlas. Index-only tests cannot catch stale
# bindings, missing upload visibility or sampling the wrong native allocation.
glyph_vertex(rects, colours, scene, atlas, sampler) = plot_vertex(rects, colours, scene)
function glyph_fragment(inputs, rects, colours, scene, atlas, sampler)
    uv = (0.5f0 / size(atlas, 1), 0.5f0 / size(atlas, 2))
    px = Mantle.sample(atlas, sampler, uv)
    return Vec4f(px.r, px.g, px.b, px.alpha)
end

pipelineof(vertex; blend = Mantle.Opaque()) =
    Rasterizer(; vertex = Mantle.VertexShader(vertex; outputs = (colour = Vec4f,)),
               fragment = Mantle.FragmentShader(rects_fragment), topology = TriangleList(),
               blend, cull = Mantle.NoCull(), depth = Mantle.DepthOff())
plotpipeline(blend = Mantle.Opaque()) = pipelineof(plot_vertex; blend)
glyphpipeline() = Rasterizer(;
    vertex = Mantle.VertexShader(glyph_vertex; outputs = (colour = Vec4f,)),
    fragment = Mantle.FragmentShader(glyph_fragment), topology = TriangleList(),
    blend = Mantle.Opaque(), cull = Mantle.NoCull(), depth = Mantle.DepthOff())

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
A plot as RayMakie keeps it (D.3, RenderObject): its group of draws in the
frame graph while inserted; its attributes, the device arrays its draw reads;
the plot's own arrays, where host data lands; the when! flag of its draw.
"""
mutable struct RenderObject
    group::Union{Nothing,Group}
    attributes::NamedTuple
    const owned::NamedTuple
    const visible::MantleArray
end
"""
The plot's own array for host data `data`: resizable from the start
(`capacity`), so its draw launches over it indirectly and a new length is a
resize and a store, no recompile (api.md 2.2).
"""
function ownarray(dev, data::AbstractArray)
    a = MantleArray(dev, eltype(data), length(data); capacity = max(1, length(data)))
    copyto!(a, data)
    return a
end
ownarray(dev, a::MantleArray) = a
function RenderObject(dev, rects, colours)
    owned = (rects = ownarray(dev, rects), colours = ownarray(dev, colours))
    return RenderObject(nothing, owned, owned, MantleArray(dev, [true]))
end

# update!(plot; name = data), once per changed attribute (D.3, setattribute!).
# Host data (one value is a one-element vector): copy! into the plot's own
# array (a store, and a resize if the length changed), and the plot's draws
# declared again over it if the attribute is another array. A device array:
# no copy, the plot's draws declared again over it.
function setattribute!(s, r::RenderObject, name, data::AbstractArray)
    own = r.owned[name]
    copy!(own, data)
    r.attributes[name] === own || redeclare!(s, r, name, own)
    return r
end
setattribute!(s, r::RenderObject, name, data::MantleArray) =
    (r.attributes[name] === data || redeclare!(s, r, name, data); r)
"""
Another array object for attribute `name` (D.3, redeclare!): the plot's group
deleted and a new group of draws over `a`, which the next run records on its
own; nothing recompiles. A plot not inserted yet only takes the array.
"""
function redeclare!(s, r::RenderObject, name, a::MantleArray)
    r.attributes = Base.setindex(r.attributes, a, name)
    r.group === nothing || (deleteplot!(s, r); insertplot!(s, r))
    return r
end
setvisible!(r::RenderObject, v::Bool) = (r.visible[1:1] = [v]; r)

"""
A screen (D.3, declareframe!): a frame graph with `before(g)` declared first,
then the plots' group, which starts with the clear of the target (so the render
pass exists before the first plot) and holds one group of draws per plot, and
the scene's rectangle (the whole target).
"""
function newscreen(dev; side = 64, before = g -> nothing)
    g = Graph(dev)
    before(g)
    img = Image(dev, RGBA{N0f8}, side, side)
    scene = MantleArray(dev, [Vec4f(-1, -1, 1, 1)])
    plots = Group(g, "plots")                        # the plots' place in the frame
    clear!(plots, img, BACKGROUND)                   # the pass's load operation
    return (; graph = g, plots, img, scene)
end

"""
The plot as a group of draws in the screen's plots' group, appended (the order
is the draw order): one when! draw over its rectangles (D.3, insertplot!).
Returns the group.
"""
function insertplot!(s, r::RenderObject)
    r.group = Group(s.plots, "plot")
    when!(r.group, s.img, r.visible, plotpipeline(), (r.attributes.rects, r.attributes.colours, s.scene), 6;
          instances = launchrange(r.attributes.rects))
    return r.group
end
"""The plot's group deleted from the frame graph (D.3, deleteplot!)."""
deleteplot!(s, group::Group) = delete!(s.graph, group)
deleteplot!(s, r::RenderObject) = (deleteplot!(s, r.group); r.group = nothing; r)

"""A plot inserted into `s` whose render object is garbage once this returns; returns its group."""
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
    tinsert = @elapsed begin
        insertplot!(s, r)
        run!(s.graph)
    end
    Mantle.waitidle(dev)
    tdelete = @elapsed begin
        deleteplot!(s, r)
        run!(s.graph)
    end
    return tinsert, tdelete
end

"""A marker mesh's BLAS: one triangle (D.3: one BLAS per marker mesh, shared by the ranges that use it)."""
markerblas(dev) = BLAS(dev, MantleArray(dev, [Point3f(0, 0, 0), Point3f(0.1, 0, 0), Point3f(0, 0.1, 0)]),
                       MantleArray(dev, [TriangleFace{UInt32}(1, 2, 3)]))

"""A graph whose kernel takes the TLAS (ray queries, resizing-and-raytracing.jl A): its runs apply the TLAS's pending builds."""
function touching(dev, tlas)
    g = Graph(dev)
    dispatch!(g, rm_touch!, (MantleArray(dev, Int32, 1), tlas), 1)
    return g
end

"""out[1] = 1, by one work-item (the ndrange of 1 is rounded up to a whole work-group); takes the TLAS."""
function rm_touch!(out, tlas)
    KI.get_global_id().x == 1 || return nothing
    @inbounds out[1] = Int32(1)
    return nothing
end

"""dst[i] = src[i] over dst."""
function rm_copy!(dst, src)
    i = KI.get_global_id().x
    i <= length(dst) || return nothing
    @inbounds dst[i] = src[i]
    return nothing
end

# Copies src[1] into dst[1] after about `iters` dependent steps: a write that
# lands late (the chain of helpers.jl's slowwrite!). One work-item.
function rm_slowcopy!(dst, src, iters)
    KI.get_global_id().x == 1 || return nothing
    acc = Int32(0)
    for i in Int32(1):iters
        acc = acc * Int32(1103515245) + i
    end
    @inbounds dst[1] = acc == Int32(-7) ? zero(eltype(dst)) : src[1]
    return nothing
end

# ── RayMakie (RM) ──

# resizing-and-raytracing.jl D.3 (setattribute!). update!(plot; positions = host data) of the same length: a store into the owned array; no new group of draws, no compile, record or host wait.
@case "RM-01" 6 :graphics begin
    dev = testdevice()
    s = newscreen(dev)
    r = RenderObject(dev, [stripe(1, 4), stripe(3, 4)], [opaque(RED)])
    group = insertplot!(s, r)
    run!(s.graph)
    _, d = counted(() -> setattribute!(s, r, :rects, [stripe(2, 4), stripe(4, 4)]))
    @test d.hostwaits == 0
    @test submitcount(d) == 0
    @test r.attributes.rects === r.owned.rects
    @test r.group === group                          # the same group: nothing declared again
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

# resizing-and-raytracing.jl D.3 (setattribute!). The same with a new length: a resize and a store; the draw's instance count from the cell; no compile or record.
@case "RM-02" 6 :graphics begin
    dev = testdevice()
    s = newscreen(dev)
    r = RenderObject(dev, [stripe(2, 8), stripe(4, 8)], [opaque(RED)])
    insertplot!(s, r)
    run!(s.graph)
    setattribute!(s, r, :rects, [stripe(k, 8) for k in 1:2:8])   # 4 elements, past the owned array's capacity
    @test pendingkinds(r.owned.rects) == [:Resize, :Store]
    @test length(r.owned.rects) == 4
    _, d = counted(() -> run!(s.graph))
    @test d.plancompiles == 0
    @test d.records == 0
    A = Array(s.img)
    @test all(k -> iscolour(stripepixel(A, k, 8), RED), 1:2:8)
    @test all(k -> iscolour(stripepixel(A, k, 8), BLACK), 2:2:8)
end

# resizing-and-raytracing.jl D.3; recording-plan.md, Graphics (one argument type for a value and a per-element attribute). Colour per element, then one value, then per element again: lengths n, 1, n; one pipeline, no compile.
@case "RM-03" 6 :graphics begin
    dev = testdevice()
    s = newscreen(dev)
    r = RenderObject(dev, [stripe(k, 4) for k in 1:4], opaque.([RED, GREEN, BLUE, RED]))
    insertplot!(s, r)
    run!(s.graph)
    for colours in ([GREEN], [BLUE, RED, GREEN, BLUE])
        setattribute!(s, r, :colours, opaque.(colours))
        @test length(r.owned.colours) == length(colours)
        _, d = counted(() -> run!(s.graph))
        @test d.plancompiles == 0
        @test d.pipelinecompiles == 0
        A = Array(s.img)
        @test all(k -> iscolour(stripepixel(A, k, 4), colours[min(k, end)]), 1:4)
    end
end

# resizing-and-raytracing.jl D.3 (redeclare!). positions = a device array: the plot's group deleted and a new group of draws over it, no copy, no plan compile, one piece recorded; then positions = host data: a store into the owned array and a new group of draws over it again.
@case "RM-04" 6 :graphics begin
    dev = testdevice()
    s = newscreen(dev)
    r = RenderObject(dev, [stripe(1, 4)], [opaque(RED)])
    group = insertplot!(s, r)
    run!(s.graph)
    b = MantleArray(dev, [stripe(2, 4)])
    setattribute!(s, r, :rects, b)
    @test r.attributes.rects === b
    @test r.group !== group                          # a new group of draws over b
    @test isempty(pendingkinds(r.owned.rects))       # no copy
    _, d = counted(() -> run!(s.graph))
    @test d.plancompiles == 0
    @test d.records == 2                             # the new group's piece and the pass's list part
    A = Array(s.img)
    @test iscolour(stripepixel(A, 2, 4), RED)
    @test iscolour(stripepixel(A, 1, 4), BLACK)
    group = r.group
    setattribute!(s, r, :rects, [stripe(3, 4)])
    @test r.attributes.rects === r.owned.rects
    @test :Store in pendingkinds(r.owned.rects)
    @test r.group !== group                          # a new group of draws over the owned array
    _, d = counted(() -> run!(s.graph))
    @test d.plancompiles == 0
    A = Array(s.img)
    @test iscolour(stripepixel(A, 3, 4), RED)
    @test iscolour(stripepixel(A, 2, 4), BLACK)
end

# resizing-and-raytracing.jl D.3 (setvisible!); api.md 2.6b (when!). visible false, then true: a flag store each; nothing recorded; instance count 0, then n.
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

# resizing-and-raytracing.jl D.3 (insertplot!, deleteplot!); recording-plan.md, Graphics (the gate). Insert and delete one plot in screens of 100, 1 000, 5 000 plots: no plan compile; one piece compiled and recorded; insert/delete latency within the finite bound below.
@case "RM-06" 6 :graphics begin
    dev = testdevice()
    rects = MantleArray(dev, [Vec4f(-1, -1, -0.9, -0.9)])
    colours = MantleArray(dev, [opaque(RED)])
    screens = [crowdedscreen(dev, n, rects, colours) for n in (100, 1000, 5000)]
    r = RenderObject(dev, [stripe(1, 1)], [opaque(GREEN)])
    for s in screens
        _, d = counted(() -> (insertplot!(s, r); run!(s.graph)))
        @test d.plancompiles == 0
        @test d.pipelinecompiles == 0
        @test d.records == 2                         # the plot's piece and the pass's list part
        @test iscolour(stripepixel(Array(s.img), 1, 1), GREEN)
        deleteplot!(s, r)
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

# resizing-and-raytracing.jl D.3; recording-plan.md, RayMakie on the core model (a simulation drawn without copies). A plot over a simulation's array, run!(sim) every frame: the frame waits for the simulation's write, the next write waits for the frame's draw; no host wait. (The redraw mark is RayMakie's reader, not checked here.)
@case "RM-07" 6 :graphics begin
    dev = testdevice()
    simrects = MantleArray(dev, [stripe(1, 8)])
    src = MantleArray(dev, [stripe(1, 8)])
    sim = Graph(dev)
    dispatch!(sim, rm_slowcopy!, (simrects, src, Int32(calibratedspin(dev, 200))), 1)
    busy = MantleArray(dev, Int32, 1)
    s = newscreen(dev; before = g -> slowwrite!(g, busy, 1; ms = 400))   # the frame draws about 400 ms after it starts
    r = RenderObject(dev, [stripe(1, 8)], [opaque(GREEN)])
    setattribute!(s, r, :rects, simrects)            # mesh!(sim's arrays): the plot's draw reads them
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

# resizing-and-raytracing.jl D.3; api.md 7 (an index array that moves). The simulation's index array resized past its storage: the piece that bound it by handle is recorded again; no plan compile.
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
    draw!(Group(s.plots, "mesh"), s.img, meshpipeline(), (vertices, green), launchrange(faces); indices = faces)
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

# resizing-and-raytracing.jl D.3 (meshscatter). Ray-traced meshscatter, particles moved with the same count: a TLAS Update only; the marker's BLAS untouched.
@case "RM-09" 6 :graphics :rt begin
    dev = testdevice()
    tlas = TLAS(dev)
    marker = markerblas(dev)
    positions = MantleArray(dev, testpositions(100))
    push!(tlas, marker, positions; scale = 1f0)
    g = touching(dev, tlas)
    run!(g)
    positions[1:100] = testpositions(100; seed = 2)
    @test buildmode.(pendingbuilds(tlas)) == [:Update]
    @test isempty(pendingbuilds(marker))
    _, log = stepsduring(() -> run!(g), dev)
    @test count(==(:BuildAccel), stepkinds(log, tlas)) == 1
    @test isempty(stepkinds(log, marker))
end

# resizing-and-raytracing.jl D.3 (meshscatter). The particle count changes: a TLAS Build (with the exact count, not observable through the API); the same BLAS, not rebuilt.
@case "RM-10" 6 :graphics :rt begin
    dev = testdevice()
    tlas = TLAS(dev)
    marker = markerblas(dev)
    positions = MantleArray(dev, testpositions(100))
    push!(tlas, marker, positions; scale = 1f0)
    g = touching(dev, tlas)
    run!(g)
    copy!(positions, testpositions(150; seed = 2))
    @test buildmode.(pendingbuilds(tlas)) == [:Build]
    @test isempty(pendingbuilds(marker))
    _, log = stepsduring(() -> run!(g), dev)
    @test count(==(:BuildAccel), stepkinds(log, tlas)) == 1
    @test isempty(stepkinds(log, marker))
end

# resizing-and-raytracing.jl D.3 (meshscatter: another device array), A (visible!). Another positions array, of the same length or of another: the range deleted and pushed again over it, a Build; the range hidden: an Update (mask 0).
@case "RM-11" 6 :graphics :rt begin
    dev = testdevice()
    tlas = TLAS(dev)
    marker = markerblas(dev)
    h = push!(tlas, marker, MantleArray(dev, testpositions(100)); scale = 1f0)
    g = touching(dev, tlas)
    run!(g)
    for (n, seed) in ((100, 2), (120, 3))
        delete!(tlas, h)
        h = push!(tlas, marker, MantleArray(dev, testpositions(n; seed)); scale = 1f0)
        @test buildmode.(pendingbuilds(tlas)) == [:Build]
        run!(g)
        @test isempty(pendingbuilds(tlas))
    end
    visible!(tlas, h, false)
    @test buildmode.(pendingbuilds(tlas)) == [:Update]
    run!(g)
    @test isempty(pendingbuilds(tlas))
end

# resizing-and-raytracing.jl D.3 (meshscatter: one BLAS per marker mesh, held by every range that uses it). Two meshscatters share a marker; delete one: the BLAS and its geometry survive; released once the last range is deleted and its build applied.
@case "RM-12" 6 :graphics :rt begin
    dev = testdevice()
    tlas = TLAS(dev)
    g = touching(dev, tlas)
    run!(g)
    cells = memory(dev).cells
    vertices = MantleArray(dev, [Point3f(0, 0, 0), Point3f(0.1, 0, 0), Point3f(0, 0.1, 0)])
    faces = MantleArray(dev, [TriangleFace{UInt32}(1, 2, 3)])
    marker = BLAS(dev, vertices, faces)
    h1 = push!(tlas, marker, MantleArray(dev, testpositions(10)); scale = 1f0)
    h2 = push!(tlas, marker, MantleArray(dev, testpositions(20; seed = 2)); scale = 1f0)
    free!(vertices); free!(faces); free!(marker)     # only the two ranges hold the marker now
    run!(g)
    delete!(tlas, h1)
    run!(g)
    @test memory(dev).cells >= cells + 3             # the marker's two geometry arrays and h2's positions
    delete!(tlas, h2)
    run!(g)
    @test memory(dev).cells == cells
end

# resizing-and-raytracing.jl D.3 (RenderObject.rect); api.md 2.6b (scene rectangle). A layout change: one store into the scene's rectangle moves every plot of the scene; nothing recorded or compiled.
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

# api.md 7 (a plot with zero elements). A plot with zero elements keeps its render object and draws nothing; it grows to n: drawn, no compile or record.
@case "RM-14" 6 :graphics begin
    dev = testdevice()
    s = newscreen(dev)
    r = RenderObject(dev, Vec4f[], [opaque(RED)])
    insertplot!(s, r)
    run!(s.graph)
    @test all(px -> iscolour(px, BLACK), Array(s.img))
    setattribute!(s, r, :rects, [stripe(1, 2)])
    _, d = counted(() -> run!(s.graph))
    @test d.plancompiles == 0
    @test d.records == 0
    @test iscolour(stripepixel(Array(s.img), 1, 2), RED)
end

# resizing-and-raytracing.jl D.3 (glyph atlas); api.md 2.6 (an Image a shader samples). The glyph atlas grows: no text or scatter piece recorded again or compiled; the frame and the glyph texels unchanged.
@case "RM-15" 6 :graphics begin
    dev = testdevice()
    s = newscreen(dev)
    glyphs = [RGBA{N0f8}(i / 255, j / 255, 0, 1) for i in 1:16, j in 1:16]
    atlas = Image(dev, RGBA{N0f8}, 16, 16)
    copyto!(atlas, glyphs)
    text = RenderObject(dev, [stripe(1, 2)], [opaque(RED)])
    draw!(Group(s.plots, "text"), s.img, glyphpipeline(), (text.attributes.rects, text.attributes.colours, s.scene, atlas,
           Sampler(; filter = :nearest)), 6;
          instances = launchrange(text.attributes.rects))
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
    # A same-size upload must reach the already-recorded fragment shader.
    copyto!(atlas, fill(RGBA{N0f8}(0, 0, 1, 1), 32, 32))
    _, uploaded = counted(() -> run!(s.graph))
    @test uploaded.plancompiles == 0 && uploaded.pipelinecompiles == 0
    @test iscolour(stripepixel(Array(s.img), 1, 2), BLUE)
    @test iscolour(stripepixel(Array(s.img), 2, 2), GREEN)
end

# recording-plan.md, Graphics; resizing-and-raytracing.jl D.3 (one recorded when! draw per variant). Toggling transparency: two draws in the plot's group, opaque and blended, each under a flag; a toggle is two flag stores, no compile or record.
@case "RM-16" 6 :graphics begin
    dev = testdevice()
    s = newscreen(dev)
    r = RenderObject(dev, [stripe(1, 2)], [Vec4f(1, 1, 1, 0.5)])
    opaqueflag = MantleArray(dev, [true])
    blendflag = MantleArray(dev, [false])
    args = (r.attributes.rects, r.attributes.colours, s.scene)
    plot = Group(s.plots, "plot")
    when!(plot, s.img, opaqueflag, plotpipeline(Mantle.Opaque()), args, 6; instances = launchrange(r.attributes.rects))
    when!(plot, s.img, blendflag, plotpipeline(Mantle.AlphaBlend()), args, 6; instances = launchrange(r.attributes.rects))
    run!(s.graph)
    @test iscolour(stripepixel(Array(s.img), 1, 2), Vec3f(1, 1, 1))
    for (o, b, c) in ((false, true, Vec3f(0.5, 0.5, 0.5)), (true, false, Vec3f(1, 1, 1)))
        opaqueflag[1:1] = [o]
        blendflag[1:1] = [b]
        _, d = counted(() -> run!(s.graph))
        @test d.plancompiles == 0
        @test d.pipelinecompiles == 0
        @test d.records == 0
        @test iscolour(stripepixel(Array(s.img), 1, 2), c)
    end
end

# resizing-and-raytracing.jl D.3 (a RenderObject's finalizer drops its arrays through the pool's inbox), B.10. Delete a plot and drop every reference to it: the holds drop through the finalizers; memory back at its baseline.
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
