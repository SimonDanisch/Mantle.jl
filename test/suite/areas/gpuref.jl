# GPURef of an array (catalogue-resize-rt.md, table N).
#
# References: API = docs/api.md; RR = docs/resizing-and-raytracing.jl (line
# numbers of 2026-10-05), B.8b; DEC = experiments/plan-review/decisions.md.
#
# What this file assumes where the docs name nothing:
#   Mantle.pendingof(x): a GPURef's pending rebind is a `Rebind`; a
#       BuildAccel's `mode` is Build() or Update(). Mantle.steplog(dev):
#       entries with `kind` (a Rebind applies as :CellWrite) and `target` (the
#       resource the user holds).
#   Mantle.rayquery(t, origin, direction) -> (; hit, instanceid, customindex),
#       the device-side ray query (name open), as in tlas.jl.
#   An index array bound by handle is drawn through a GPURef with
#   draw!(p, pipeline, args, launchrange(r); indices = r) (RR:757-765).
# "Released" is read from memory(dev).cells: an array or a GPURef gives its
# cell back at its last hold. Validation-layer and GPU-AV expectations are
# comments marked VAL.

using KernelAbstractions: @kernel, @index
using GeometryBasics: Point3f, Vec3f, Vec4f, Vec, Mat4f, TriangleFace
using ColorTypes: RGBA
import KernelInterface as KI

# ── Ray-tracing helpers, the same in tlas.jl, blas.jl, gpuref.jl and hikari_patterns.jl ──

const TRIANGLEVERTICES = (Point3f(-0.4, -0.4, 0), Point3f(0.4, -0.4, 0), Point3f(0, 0.4, 0), Point3f(0, -0.2, 0))
const TRIANGLEFACE = TriangleFace{UInt32}(1, 2, 3)   # covers the origin
const MOVEDFACE = TriangleFace{UInt32}(1, 2, 4)      # misses the origin: a new topology
const MOVEDVERTEX = Point3f(0, -0.3, 0)              # as vertex 3, TRIANGLEFACE misses the origin: same topology
const NOINSTANCE = typemax(UInt32)
const InstanceHit = Vec{2,UInt32}                    # (instanceCustomIndex, index within the range)
const NOHIT = InstanceHit(NOINSTANCE, NOINSTANCE)

"""The vertex and face arrays of one triangle around the origin in the plane z = 0."""
trianglearrays(dev) = (MantleArray(dev, collect(TRIANGLEVERTICES)), MantleArray(dev, [TRIANGLEFACE]))

"""A BLAS of one triangle that is the only holder of its arrays."""
function ownedblas(dev)
    v, f = trianglearrays(dev)
    b = BLAS(dev, v, f)
    free!(v); free!(f)
    return b
end

translation(x, y, z) = Mat4f(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, x, y, z, 1)

"""`n` instance positions on a grid of spacing 1 in the plane z = 0, 256 to a row, from grid index `from` on."""
instancegrid(n; from = 0) = [Point3f(k % 256, k ÷ 256, 0) for k in from:from+n-1]

rayorigins(dev, positions) = MantleArray(dev, [p - Vec3f(0, 0, 1) for p in positions])

"""An open gate: kernels read it, so a case can order them after a slowwrite! into a gate of its own."""
readygate(dev) = MantleArray(dev, Int32[0])

"""Ray i from origins[i] along +z: (instanceCustomIndex, InstanceId - rangeoffset) of the closest hit, or NOHIT."""
@kernel function rt_hits!(out, t, origins, gate)
    i = @index(Global)
    q = Mantle.rayquery(t, origins[i], Vec3f(0, 0, 1))
    out[i] = q.hit & (gate[1] != Int32(-1)) ?
             InstanceHit(q.customindex, q.instanceid - Mantle.rangeoffset(t, q.customindex)) : NOHIT
end

"""The hits of rays through `positions`, by one eager launch (which applies what is pending on `t`)."""
function queryhits(dev, t, positions)
    out = MantleArray(dev, InstanceHit, length(positions))
    dispatch!(dev, rt_hits!, (out, t, rayorigins(dev, positions), readygate(dev)), length(positions))
    return Array(out)
end

"""Declares the ray query through `positions` into `g`; returns the device array it writes."""
function declarehits!(g, dev, t, positions; gate = readygate(dev))
    out = MantleArray(dev, InstanceHit, length(positions))
    dispatch!(g, rt_hits!, (out, t, rayorigins(dev, positions), gate), length(positions))
    return out
end

"""Whether `hits` are the instances 1:n of one range, in order."""
isonerange(hits) = !isempty(hits) && first(hits)[1] != NOINSTANCE &&
                   all(h -> h[1] == first(hits)[1], hits) && [h[2] for h in hits] == 1:length(hits)
allmissed(hits) = all(==(NOHIT), hits)

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

"""The number of build steps (Build or Update) on `x` in `log`."""
accelbuilds(log, x) = count(==(:BuildAccel), stepkinds(log, x))

# ── This file's helpers ──

"""out[i] = src[i] over src's length."""
@kernel function ab_copy!(out, src)
    i = @index(Global)
    out[i] = src[i]
end

"""out[i] = src[i]; reading `gate` orders it after a slowwrite! into it."""
@kernel function ab_gatedcopy!(out, src, gate)
    i = @index(Global)
    out[i] = src[i] + (gate[1] == Int32(-1) ? 1f0 : 0f0)
end

"""out[i] = attribute(a, i): one value for every i, or one per element (API:179)."""
@kernel function ab_attribute!(out, a)
    i = @index(Global)
    out[i] = Mantle.attribute(a, i)
end

"""positions[i] = (i - 1, y, 0): a row of instances at height y."""
@kernel function ab_row!(positions, y)
    i = @index(Global)
    positions[i] = Point3f(i - 1, y, 0)
end

"""A copy of `r`'s target through `r`, by one eager launch over launchrange(r)."""
function readthrough(dev, r)
    out = MantleArray(dev, eltype(r[]), length(r[]))
    dispatch!(dev, ab_copy!, (out, r), launchrange(r))
    return Array(out)
end

const PIXELS = 8

"""Vertex k (one-based) as a one-pixel point at pixel k of a PIXELS × 1 target."""
function ab_pointvertex(args...)
    k = KI.vertex_index()
    KI.set_point_size!(1f0)
    return (position = Vec4f(-1f0 + (2f0 * k - 1f0) / PIXELS, 0f0, 0f0, 1f0),)
end
ab_pointfragment(inputs, args...) = Vec4f(1f0, 1f0, 1f0, 1f0)

pointspipeline() = Mantle.GraphicsPipeline(; vertex = Mantle.VertexShader(ab_pointvertex),
    fragment = Mantle.FragmentShader(ab_pointfragment), topology = Mantle.PointList(),
    blend = Mantle.Opaque(), cull = Mantle.NoCull(), depth = Mantle.DepthOff())

"""A graph drawing, as points into `img`, the vertices `indices` names (an index array or its GPURef, bound by handle)."""
function pointsgraph(dev, img, indices)
    g = Graph(dev)
    render!(g, img => Clear((0f0, 0f0, 0f0, 0f0))) do p
        draw!(p, pointspipeline(), (), launchrange(indices); indices)
    end
    return g
end

"""The pixels a fresh graph draws for `indices`: what a rebound graph must match."""
function drawn(dev, indices)
    img = Image(dev, RGBA{Float32}, PIXELS, 1)
    run!(pointsgraph(dev, img, indices))
    return Array(img)
end

# ── N. GPURef of an array ──

# RR:1200-1220, RR:1269: GPURef(dev, a) pends the cell write of its target like a rebind; the first submission touching r applies it. A kernel over r.
@case "AB-01" 6 begin
    dev = testdevice()
    data = testdata(Float32, 100)
    a = MantleArray(dev, data)
    r = GPURef(dev, a)
    @test pendingkinds(r) == [:Rebind]
    copied, log = stepsduring(() -> readthrough(dev, r), dev)
    @test count(==(:CellWrite), stepkinds(log, r)) == 1
    @test copied == data
    @test isempty(Mantle.pendingof(r))
end

# RR:1223-1227, RR:1214-1218: the target is a root array; a view throws and the box's cell is retired, with no finalizer. GPURef(dev, view).
@case "AB-02" 6 begin
    dev = testdevice()
    a = MantleArray(dev, Float32, 100)
    base = memory(dev)
    @test_throws ArgumentError GPURef(dev, view(a, 1:10))
    @test memory(dev).cells == base.cells
end

# API:180, API:488, RR:1222 (A16): GPURef of a transient throws instead of making a value box. GPURef(dev, MantleArray(g, T, n)).
@case "AB-03" 6 begin
    dev = testdevice()
    g = Graph(dev)
    @test_throws ArgumentError GPURef(dev, MantleArray(g, Float32, 10))
end

# RR:1195-1197, RR:1209: a freed target throws inside the constructor's change; the cell is retired. GPURef of a freed array.
@case "AB-04" 6 begin
    dev = testdevice()
    a = MantleArray(dev, Float32, 10)
    free!(a)
    base = memory(dev)
    @test_throws ArgumentError GPURef(dev, a)
    @test memory(dev).cells == base.cells
end

# API:180, RR:1229-1233, RR:1256-1260: r[] = b reports b at once; runs submitted before read a, which is released after the applying token. A run reading r in flight; r[] = b; free!(a); the next run.
@case "AB-05" 6 begin
    dev = testdevice()
    adata, bdata = testdata(Float32, 64; seed = 1), testdata(Float32, 64; seed = 2)
    a = MantleArray(dev, adata); b = MantleArray(dev, bdata)
    r = GPURef(dev, a)
    gate = MantleArray(dev, Int32[0]); out = MantleArray(dev, Float32, 64)
    g = Graph(dev)
    slowwrite!(g, gate, 7)
    dispatch!(g, ab_gatedcopy!, (out, r, gate), launchrange(r))
    run!(g)
    run!(g)                                                # in flight for about 200 ms
    r[] = b
    @test r[] === b
    free!(a)
    held = memory(dev)                                     # the Rebind is pending: r still holds a
    @test Array(out) == adata                              # the in-flight run read a
    run!(g)
    @test Array(out) == bdata
    @test memory(dev).cells == held.cells - 1              # a, released after the applying run's token
end

# RR:1238-1241: rebinding to the current target is a no-op. r[] = r[].
@case "AB-06" 6 begin
    dev = testdevice()
    r = GPURef(dev, MantleArray(dev, testdata(Float32, 16)))
    readthrough(dev, r)
    r[] = r[]
    @test isempty(Mantle.pendingof(r))
    _, log = stepsduring(() -> readthrough(dev, r), dev)
    @test isempty(stepkinds(log, r))
end

# API:488, RR:1240-1247: r[] = b throws for another element type or dimension count (no method), a view, a transient, a freed array, and after free!(r); nothing pends, no hold leaks. (A lent array needs a vendor view: not covered here.)
@case "AB-07" 6 begin
    dev = testdevice()
    a = MantleArray(dev, Float32, 16)
    r = GPURef(dev, a)
    readthrough(dev, r)
    base = memory(dev)
    other = MantleArray(dev, Int32, 16)
    square = MantleArray(dev, Float32, 4, 4)
    parent = MantleArray(dev, Float32, 32)
    g = Graph(dev)
    freed = MantleArray(dev, Float32, 16); free!(freed)
    @test_throws MethodError (r[] = other)
    @test_throws MethodError (r[] = square)
    @test_throws ArgumentError (r[] = view(parent, 1:16))
    @test_throws MethodError (r[] = MantleArray(g, Float32, 16))
    @test_throws ArgumentError (r[] = freed)
    @test r[] === a && isempty(Mantle.pendingof(r))
    free!(other); free!(square); free!(parent); free!(g)
    @test memory(dev).cells == base.cells
    s = MantleArray(dev, Float32, 16)
    free!(r)
    @test_throws ArgumentError (r[] = s)
end

# RR:1238-1241, RR:1256-1260: concurrent rebinds leave r on one of the targets, which alone stays held; nothing leaks. Two threads rebind 10^4 times while runs read r.
@case "AB-08" 6 :long begin
    dev = testdevice()
    base = memory(dev)
    bdata, cdata = testdata(Float32, 64; seed = 1), testdata(Float32, 64; seed = 2)
    b = MantleArray(dev, bdata); c = MantleArray(dev, cdata)
    r = GPURef(dev, b)
    out = MantleArray(dev, Float32, 64)
    g = Graph(dev)
    dispatch!(g, ab_copy!, (out, r), launchrange(r))
    run!(g)
    withtimeout(600) do
        tb = Threads.@spawn for _ in 1:10^4; r[] = b; end
        tc = Threads.@spawn for _ in 1:10^4; r[] = c; end
        for _ in 1:10^3; run!(g); end
        wait(tb); wait(tc)
    end
    @test r[] === b || r[] === c
    run!(g)
    @test Array(out) == (r[] === b ? bdata : cdata)
    setup = memory(dev)
    free!(b); free!(c)
    @test memory(dev).cells == setup.cells - 1             # the other one; r's target stays held
    free!(g); free!(r); free!(out)
    @test memory(dev).cells == base.cells
end

# RR:1285-1288: launches over an array GPURef are indirect and follow the target's length; nothing recompiles. Targets of 10 and 10^6 elements.
@case "AB-09" 6 begin
    dev = testdevice()
    small = MantleArray(dev, testdata(Float32, 10)); large = MantleArray(dev, testdata(Float32, 10^6; seed = 2))
    r = GPURef(dev, small)
    out = MantleArray(dev, Float32, 10^6)
    g = Graph(dev)
    dispatch!(g, ab_copy!, (out, r), launchrange(r))
    fill!(out, -1f0); run!(g)
    @test count(!=(-1f0), Array(out)) == 10
    for (target, n) in ((large, 10^6), (small, 10))
        r[] = target
        fill!(out, -1f0)
        _, d = counted(() -> run!(g))
        @test d.plancompiles == 0
        @test count(!=(-1f0), Array(out)) == n
    end
end

# API:180, RR:1234-1235, RR:1251-1252: a TLAS with a range over r gets an Update on a rebind, a Build if the length changed. Rebind to the same and to another length.
@case "AB-10" 6 :rt begin
    dev = testdevice()
    t = TLAS(dev); b = ownedblas(dev)
    r = GPURef(dev, MantleArray(dev, instancegrid(8)))
    push!(t, b, r)
    queryhits(dev, t, instancegrid(1))
    r[] = MantleArray(dev, instancegrid(8; from = 8))
    @test buildmode.(pendingbuilds(t)) == [:Update]
    hits = queryhits(dev, t, instancegrid(16))
    @test allmissed(hits[1:8]) && isonerange(hits[9:16])
    r[] = MantleArray(dev, instancegrid(12; from = 16))
    @test buildmode.(pendingbuilds(t)) == [:Build]
    hits = queryhits(dev, t, instancegrid(28))
    @test allmissed(hits[1:16]) && isonerange(hits[17:28])
end

# API:180, API:389, RR:1111-1117: r is a reader of its target and forwards to its TLASes: a store pends an Update, a resize a Build. A store into r[], then a resize of r[].
@case "AB-11" 6 :rt begin
    dev = testdevice()
    t = TLAS(dev); b = ownedblas(dev)
    r = GPURef(dev, MantleArray(dev, instancegrid(8)))
    push!(t, b, r)
    queryhits(dev, t, instancegrid(1))
    r[][1:1] = [Point3f(0, 5, 0)]
    @test buildmode.(pendingbuilds(t)) == [:Update]
    hits = queryhits(dev, t, vcat([Point3f(0, 5, 0)], instancegrid(1)))
    @test isonerange(hits[1:1]) && hits[2] == NOHIT
    resize!(r[], 12)
    @test buildmode.(pendingbuilds(t)) == [:Build]
    r[][9:12] = instancegrid(4; from = 8)
    @test isonerange(queryhits(dev, t, vcat([Point3f(0, 5, 0)], instancegrid(11; from = 1))))
end

# RR:1256-1260: once the applying submission has passed, the old target no longer forwards to r's TLASes. Store into the old target after the rebind.
@case "AB-12" 6 :rt begin
    dev = testdevice()
    t = TLAS(dev); b = ownedblas(dev)
    a = MantleArray(dev, instancegrid(8))
    r = GPURef(dev, a)
    push!(t, b, r)
    queryhits(dev, t, instancegrid(1))
    r[] = MantleArray(dev, instancegrid(8; from = 8))
    queryhits(dev, t, instancegrid(1))                     # applies the rebind
    memory(dev)                                            # its token passed: a's deferred reader drop ran
    a[1:1] = [Point3f(0, 5, 0)]
    @test isempty(pendingbuilds(t))
    _, log = stepsduring(() -> queryhits(dev, t, instancegrid(1)), dev)
    @test isempty(stepkinds(log, t))
end

# RR:1178-1180, API:408-416: a run writing through r writes r's applied target; the update goes to the TLASes over that target. A kernel writes through r; r[] = c between runs.
@case "AB-13" 6 :rt begin
    dev = testdevice()
    b = ownedblas(dev)
    a = MantleArray(dev, instancegrid(8)); c = MantleArray(dev, instancegrid(8; from = 8))
    ta = TLAS(dev); push!(ta, b, a)
    tc = TLAS(dev); push!(tc, b, c)
    queryhits(dev, ta, instancegrid(1)); queryhits(dev, tc, instancegrid(1))
    r = GPURef(dev, a)
    g = Graph(dev)
    dispatch!(g, ab_row!, (r, 5f0), launchrange(r))
    run!(g)
    @test buildmode.(pendingbuilds(ta)) == [:Update] && isempty(pendingbuilds(tc))
    queryhits(dev, ta, instancegrid(1))
    r[] = c
    run!(g)
    @test buildmode.(pendingbuilds(tc)) == [:Update] && isempty(pendingbuilds(ta))
    @test Array(c) == [Point3f(k, 5, 0) for k in 0:7]
end

# API:177, RR:757-765, DEC B3.17 (open decision 17 (c)): indices bound by handle through r: a rebind marks the parts that recorded the handle; they are recorded again, nothing recompiles. Draw through r; rebind.
@case "AB-14" 6 :graphics begin
    dev = testdevice()
    a = MantleArray(dev, UInt32[0, 1]); b = MantleArray(dev, UInt32[4, 5, 6])
    expecteda, expectedb = drawn(dev, a), drawn(dev, b)
    @test expecteda != expectedb
    r = GPURef(dev, a)
    img = Image(dev, RGBA{Float32}, PIXELS, 1)
    g = pointsgraph(dev, img, r)
    run!(g)
    @test Array(img) == expecteda
    r[] = b
    _, d = counted(() -> run!(g))
    @test d.plancompiles == 0 && d.records >= 1
    @test Array(img) == expectedb
end

# RR:757-762, RR:644-651: a rebind between compile and register! makes register! return Again; the graph compiles again. FI-R: the compile held in compilebindings while r is rebound.
@case "AB-15" 6 :graphics begin
    dev = FaultDevice(testdevice())
    a = MantleArray(dev, UInt32[0, 1]); b = MantleArray(dev, UInt32[4, 5, 6])
    expectedb = drawn(dev, b)
    r = GPURef(dev, a)
    img = Image(dev, RGBA{Float32}, PIXELS, 1)
    g = pointsgraph(dev, img, r)
    compiling = Channel{Nothing}(1)
    at = calls(dev, :compilebindings) + 1
    inject!(dev, :compilebindings, Block(at, compiling))
    running = Threads.@spawn counted(() -> run!(g), dev)
    @test timedwait(() -> calls(dev, :compilebindings) >= at, 30) === :ok
    r[] = b
    put!(compiling, nothing)
    _, d = withtimeout(() -> fetch(running), 60)
    @test d.plancompiles == 2
    @test Array(img) == expectedb
end

# RR:764-765: plans that bind an array directly are not marked by a rebind of some GPURef that also points at it. A draw binds a; r points at a; r[] = b.
@case "AB-16" 6 :graphics begin
    dev = testdevice()
    a = MantleArray(dev, UInt32[0, 1]); b = MantleArray(dev, UInt32[4, 5, 6])
    expecteda = drawn(dev, a)
    img = Image(dev, RGBA{Float32}, PIXELS, 1)
    g = pointsgraph(dev, img, a)
    run!(g)
    r = GPURef(dev, a)
    r[] = b
    readthrough(dev, r)                                    # applies the rebind
    _, d = counted(() -> run!(g))
    @test d.records == 0 && d.plancompiles == 0
    @test Array(img) == expecteda
end

# RR:762-766 (A17): the old target is never freed under a plan that recorded its buffer: a plan keeps its first handle target until it retires. Draw through r; r[] = b; free!(a).
@case "AB-17" 6 :graphics begin
    dev = testdevice()
    a = MantleArray(dev, zeros(UInt32, 1 << 20))           # 4 MiB, kept while the plan lives
    b = MantleArray(dev, UInt32[1, 2])
    r = GPURef(dev, a)
    img = Image(dev, RGBA{Float32}, PIXELS, 1)
    g = pointsgraph(dev, img, r)
    run!(g)
    r[] = b
    run!(g)                                                # the rebind applied; the parts recorded again with b
    before = memory(dev)
    free!(a)
    @test memory(dev).cells == before.cells                # still held by the plan
    free!(g)                                               # the plan retires
    @test memory(dev).cells < before.cells
end

# RR:762-764: rebinds and frees under a handle-bound plan never leave a draw reading a released index array. A chain a → b → c, each old target freed.
@case "AB-18" 6 :graphics begin
    dev = testdevice()
    targets = [MantleArray(dev, UInt32[0, 1]), MantleArray(dev, UInt32[2, 3, 4]), MantleArray(dev, UInt32[5, 6, 7])]
    expected = [drawn(dev, x) for x in targets]
    r = GPURef(dev, targets[1])
    img = Image(dev, RGBA{Float32}, PIXELS, 1)
    g = pointsgraph(dev, img, r)
    run!(g)
    @test Array(img) == expected[1]
    for k in 2:3
        r[] = targets[k]
        free!(targets[k-1])
        run!(g)
        @test Array(img) == expected[k]
    end
    # VAL: GPU-AV clean (no draw reads a freed index buffer)
end

# RR:1271-1282: at r's last hold its pending cell write is dropped; both targets' holds drop with r's last tokens. free!(r) with a Rebind pending.
@case "AB-19" 6 begin
    dev = testdevice()
    base = memory(dev)
    a = MantleArray(dev, testdata(Float32, 16)); b = MantleArray(dev, testdata(Float32, 16; seed = 2))
    r = GPURef(dev, a)
    readthrough(dev, r)
    r[] = b
    _, d = counted(() -> (free!(r); memory(dev)))
    @test submitcount(d) == 0                              # the Rebind is never applied
    free!(a); free!(b)
    @test memory(dev).cells == base.cells
end

# RR:1200-1221: a constructor that throws releases what the box took once and registers no finalizer. GPURef of a freed array, repeated, then a collection.
@case "AB-20" 6 begin
    dev = testdevice()
    a = MantleArray(dev, Float32, 16)
    free!(a)
    base = memory(dev)
    for _ in 1:10
        @test_throws ArgumentError GPURef(dev, a)
    end
    @test memory(dev).cells == base.cells                  # memory() collects: a finalizer would release twice
    # The throw comes from the freed target (holdmember!): change! calls no device verb a FaultDevice could fail.
end

# API:179, RR:142-147: attribute(a, i) = a[min(i, length(a))]: one argument type for one value and one per element; switching compiles nothing. Length 1, then n, then 1.
@case "AB-21" 6 begin
    dev = testdevice()
    n = 64
    color = GPURef(dev, MantleArray(dev, [2f0]))
    out = MantleArray(dev, Float32, n)
    g = Graph(dev)
    dispatch!(g, ab_attribute!, (out, color), n)
    run!(g)
    @test all(==(2f0), Array(out))
    data = testdata(Float32, n)
    _, d = counted(() -> (copy!(color[], data); run!(g)))
    @test d.plancompiles == 0 && d.kernelcompiles == 0 && d.pipelinecompiles == 0
    @test Array(out) == data
    copy!(color[], [3f0]); run!(g)
    @test all(==(3f0), Array(out))
end

# RR:145, API:84: copy!(color[], data) resizes and stores the target; the box is not rebound. One value, then one per element.
@case "AB-22" 6 begin
    dev = testdevice()
    n = 64
    color = GPURef(dev, MantleArray(dev, [2f0]))
    out = MantleArray(dev, Float32, n)
    dispatch!(dev, ab_attribute!, (out, color), n)
    target = color[]
    data = testdata(Float32, n)
    copy!(color[], data)
    @test color[] === target
    @test isempty(Mantle.pendingof(color))
    @test pendingkinds(target) == [:Resize, :Store]
    dispatch!(dev, ab_attribute!, (out, color), n)
    @test Array(out) == data
end

# API:179, API:491 (A21, contract open): a colour target emptied while the positions keep n elements. The run completes and writes every output.
@case "AB-23" 6 begin
    dev = testdevice()
    n = 64
    color = GPURef(dev, MantleArray(dev, [2f0]))
    out = MantleArray(dev, Float32, n)
    g = Graph(dev)
    dispatch!(g, ab_attribute!, (out, color), n)
    run!(g)
    resize!(color[], 0)
    fill!(out, -1f0)
    run!(g)
    @test length(Array(out)) == n
    # VAL: GPU-AV clean: attribute(color, i) reads no element of the empty array (A21 decides the value)
end
