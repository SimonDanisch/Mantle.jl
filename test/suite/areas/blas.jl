# BLAS (catalogue-resize-rt.md, table M).
#
# References: API = docs/api.md; RR = docs/resizing-and-raytracing.jl (line
# numbers of 2026-10-05); VK-AS = vkdocs chapters/accelstructures.adoc;
# VK-BC/VK-DC = vkdocs chapters/commonvalidity/
# build_acceleration_structure_{common,device_common}.adoc.
#
# A BLAS is checked through a TLAS with one instance of it at the origin and
# one ray per triangle (rt_hits!, as in tlas.jl): the k-th triangle of a row
# (rowvertices, rowfaces) stands around grid position k - 1, so the host knows
# which rays hit. Build steps come from the step log, compiles and re-records
# from `counted`.
#
# What this file assumes where the docs name nothing (as tlas.jl):
#   Mantle.rayquery(t, origin, direction) -> (; hit, instanceid, customindex),
#       the device-side ray query (name open), indices one-based;
#   Mantle.pendingof(x): a BuildAccel's `mode` is Build() or Update();
#   Mantle.steplog(dev): entries with `kind` (:MoveStorage, :BindPages,
#       :MoveAccel, :BuildAccel, ...) and `target` (the resource the user holds);
#   a new BLAS pends its first Build (API:150, "a new topology: a Build").
# Validation-layer and GPU-AV expectations are comments marked VAL.

using KernelAbstractions: @kernel, @index
using GeometryBasics: Point3f, Vec3f, Vec, Mat4f, TriangleFace

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

"""The vertices of `n` triangles shaped like TRIANGLEVERTICES[1:3], the k-th around grid position k - 1."""
rowvertices(n) = [p + Vec3f(q...) for q in instancegrid(n) for p in TRIANGLEVERTICES[1:3]]
rowfaces(n) = [TriangleFace{UInt32}(3k + 1, 3k + 2, 3k + 3) for k in 0:n-1]

"""A BLAS over a row of `n` triangles, and its vertex and face arrays."""
function rowblas(dev, n)
    v = MantleArray(dev, rowvertices(n)); f = MantleArray(dev, rowfaces(n))
    return BLAS(dev, v, f), v, f
end

"""A TLAS with one instance of `b` at the origin."""
function oneinstance(dev, b)
    t = TLAS(dev)
    push!(t, b, translation(0, 0, 0))
    return t
end

"""Which of the rays through the first `n` grid positions hit something."""
hitflags(hits) = [h != NOHIT for h in hits]
hitmask(dev, t, n) = hitflags(queryhits(dev, t, instancegrid(n)))

"""The vertices of the first `n` row triangles, and 3n as their length (a length a kernel writes)."""
@kernel function bl_rowvertices!(v, len, n)
    i = @index(Global)
    if i <= 3n
        k, c = divrem(i - 1, 3)
        corner = c == 0 ? Point3f(-0.4f0, -0.4f0, 0f0) : c == 1 ? Point3f(0.4f0, -0.4f0, 0f0) : Point3f(0f0, 0.4f0, 0f0)
        @inbounds v[i] = corner + Vec3f(k % 256, k ÷ 256, 0)
        i == 1 && (len[] = 3n)
    end
end

"""a[i] = x."""
@kernel function bl_set!(a, i, x)
    a[i] = x
end

# ── M. BLAS ──

# API:150, RR:1152-1158: moved vertices with the same topology are a BLAS Update and an Update of the TLASes over it. A store into the vertices, same count.
@case "BL-01" 5 :rt begin
    dev = testdevice()
    v, f = trianglearrays(dev); b = BLAS(dev, v, f)
    t = oneinstance(dev, b)
    @test hitmask(dev, t, 1) == [true]
    v[3:3] = [MOVEDVERTEX]
    @test buildmode.(pendingbuilds(b)) == [:Update]
    @test buildmode.(pendingbuilds(t)) == [:Update]
    mask, log = stepsduring(() -> hitmask(dev, t, 1), dev)
    @test accelbuilds(log, b) == 1 && :MoveAccel ∉ stepkinds(log, b)
    @test accelbuilds(log, t) == 1
    @test mask == [false]
end

# RR:974-975, RR:1153-1155, VK-BC:259-268 (VUID-03768): new indices are a Build into the BLAS's own storage, and a TLAS Update. A store into the faces.
@case "BL-02" 5 :rt begin
    dev = testdevice()
    v, f = trianglearrays(dev); b = BLAS(dev, v, f)
    t = oneinstance(dev, b)
    @test hitmask(dev, t, 1) == [true]
    f[1:1] = [MOVEDFACE]
    @test buildmode.(pendingbuilds(b)) == [:Build]
    @test buildmode.(pendingbuilds(t)) == [:Update]
    mask, log = stepsduring(() -> hitmask(dev, t, 1), dev)
    @test accelbuilds(log, b) == 1 && :MoveAccel ∉ stepkinds(log, b)
    @test accelbuilds(log, t) == 1
    @test mask == [false]
    # VAL: VUID-03768 (an Update keeps the indices of the last build) never fires: this is a Build
end

# RR:1149-1153, VK-BC:164-171 (VUID-03764): a new topology within the capacity is a Build in place, no MoveAccel. Shrink 8 triangles to 4, then back to 8.
@case "BL-03" 5 :rt begin
    dev = testdevice()
    b, v, f = rowblas(dev, 8)
    t = oneinstance(dev, b)
    @test all(hitmask(dev, t, 8))
    resize!(v, 12); resize!(f, 4)
    @test buildmode.(pendingbuilds(b)) == [:Build]
    mask, log = stepsduring(() -> hitmask(dev, t, 8), dev)
    @test accelbuilds(log, b) == 1 && :MoveAccel ∉ stepkinds(log, b)
    @test buildmode.(pendingbuilds(t)) == [] && accelbuilds(log, t) == 1
    @test mask == [trues(4); falses(4)]
    resize!(v, 24); resize!(f, 8)
    v[13:24] = rowvertices(8)[13:24]; f[5:8] = rowfaces(8)[5:8]
    mask, log = stepsduring(() -> hitmask(dev, t, 8), dev)
    @test accelbuilds(log, b) == 1 && :MoveAccel ∉ stepkinds(log, b)
    @test all(mask)
    # VAL: VUID-03764 clean
end

# RR:1151-1158, VK-AS:159-162: beyond the capacity a MoveAccel, the new address through the BLAS's cell; the TLAS gets an Update, not a Build. Grow 8 triangles to 1024.
@case "BL-04" 5 :rt begin
    dev = testdevice()
    b, v, f = rowblas(dev, 8)
    t = oneinstance(dev, b)
    queryhits(dev, t, instancegrid(1))
    copy!(v, rowvertices(1024)); copy!(f, rowfaces(1024))
    @test buildmode.(pendingbuilds(t)) == [:Update]
    mask, log = stepsduring(() -> hitmask(dev, t, 1024), dev)
    k = stepkinds(log, b)
    @test :MoveAccel in k && findfirst(==(:MoveAccel), k) < findlast(==(:BuildAccel), k)
    @test accelbuilds(log, t) == 1
    @test all(mask)
    # VAL: clean (an Update may change which BLAS an instance references)
end

# RR:1141-1143, RR:1155-1158: the old BLAS storage is retired after the traces in flight and the TLAS Update. BL-04 with a trace in flight.
@case "BL-05" 5 :rt begin
    dev = testdevice()
    b, v, f = rowblas(dev, 8)
    t = oneinstance(dev, b)
    gate = MantleArray(dev, Int32[0])
    ga = Graph(dev); slowwrite!(ga, gate, 7)
    outa = declarehits!(ga, dev, t, instancegrid(1024); gate)
    gb = Graph(dev)
    outb = declarehits!(gb, dev, t, instancegrid(1024))
    run!(ga); run!(gb)
    run!(ga)                                               # a trace in flight for about 200 ms
    copy!(v, rowvertices(1024)); copy!(f, rowfaces(1024))
    run!(gb)                                               # moves the BLAS while ga's trace runs
    @test hitflags(Array(outa)) == [trues(8); falses(1016)]
    @test all(hitflags(Array(outb)))
    # VAL: GPU-AV clean (the old BLAS is never read after its release)
end

# RR:434-439: a TLAS Update pended by a BLAS move is applied by the next submission touching the TLAS, before its trace. The move applied through another TLAS; then t traced.
@case "BL-06" 5 :rt begin
    dev = testdevice()
    b, v, f = rowblas(dev, 8)
    t = oneinstance(dev, b); t2 = oneinstance(dev, b)
    queryhits(dev, t, instancegrid(1)); queryhits(dev, t2, instancegrid(1))
    copy!(v, rowvertices(1024)); copy!(f, rowfaces(1024))
    _, log = stepsduring(() -> queryhits(dev, t2, instancegrid(1)), dev)
    @test :MoveAccel in stepkinds(log, b) && isempty(stepkinds(log, t))
    @test buildmode.(pendingbuilds(t)) == [:Update]
    mask, log = stepsduring(() -> hitmask(dev, t, 1024), dev)
    @test accelbuilds(log, t) == 1 && all(mask)
    # VAL: GPU-AV clean (no query reads the old BLAS address)
end

# API:150, RR:968-975, VK-AS:261-262: geometry whose length a kernel writes is built over the capacity with zero-area primitives past the length; a length change is an Update. The kernel writes 4, 10, 2 triangles.
@case "BL-07" 5 :rt begin
    dev = testdevice()
    v = MantleArray(dev, Point3f, 0; capacity = 48)
    f = MantleArray(dev, rowfaces(16))
    b = BLAS(dev, v, f)
    t = oneinstance(dev, b)
    n = GPURef(dev, Int32(4))
    g = Graph(dev)
    dispatch!(g, bl_rowvertices!, (v, lengthof(v), n), 48)
    for (k, mode) in ((4, :Build), (10, :Update), (2, :Update))
        n[] = Int32(k)
        run!(g)                                            # writes the vertices: pends the BLAS's update
        @test buildmode.(pendingbuilds(b)) == [mode]
        mask, log = stepsduring(() -> hitmask(dev, t, 16), dev)
        @test accelbuilds(log, b) == 1
        @test mask == [i <= k for i in 1:16]
    end
    # VAL: clean; the primitives past the length are zero-area (a NaN vertex would make them inactive
    # and the Update invalid); maxVertex is the capacity's in every build
end

# RR:973-975, API:150: a run that writes a BLAS's vertices pends an Update; one that writes its faces pends a Build.
@case "BL-08" 5 :rt begin
    dev = testdevice()
    v, f = trianglearrays(dev); b = BLAS(dev, v, f)
    t = oneinstance(dev, b)
    queryhits(dev, t, instancegrid(1))
    gv = Graph(dev)
    dispatch!(gv, bl_set!, (v, 3, MOVEDVERTEX), 1)
    run!(gv)
    @test buildmode.(pendingbuilds(b)) == [:Update]
    @test hitmask(dev, t, 1) == [false]
    gf = Graph(dev)
    dispatch!(gf, bl_set!, (f, 1, MOVEDFACE), 1)
    run!(gf)
    @test buildmode.(pendingbuilds(b)) == [:Build]
end

# API:150, RR:1158-1159: one BLAS serves every range over it and is built once (blas!(scene, mesh), RR:1502, is RayMakie's lookup over this). Two ranges in one TLAS and one in another.
@case "BL-09" 5 :rt begin
    dev = testdevice()
    v, f = trianglearrays(dev); b = BLAS(dev, v, f)
    t = TLAS(dev); t2 = TLAS(dev)
    push!(t, b, MantleArray(dev, instancegrid(4)))
    push!(t, b, MantleArray(dev, instancegrid(4; from = 4)))
    push!(t2, b, MantleArray(dev, instancegrid(4)))
    (hits, hits2), log = stepsduring(dev) do
        queryhits(dev, t, instancegrid(8)), queryhits(dev, t2, instancegrid(4))
    end
    @test accelbuilds(log, b) == 1
    @test isonerange(hits[1:4]) && isonerange(hits[5:8]) && hits[1][1] != hits[5][1] && isonerange(hits2)
end

# RR:1312-1323: at the last hold the pending changes are dropped and nothing is submitted. free!(blas) with its first Build pending and no other holder.
@case "BL-10" 5 :rt begin
    dev = testdevice()
    v, f = trianglearrays(dev); b = BLAS(dev, v, f)
    free!(v); free!(f)
    @test buildmode.(pendingbuilds(b)) == [:Build]
    before = memory(dev)
    _, d = counted(() -> (free!(b); memory(dev)))
    @test submitcount(d) == 0
    @test memory(dev).cells < before.cells
end

# RR:932-933, API:158: a refit of a structure never built is a Build. refit!(g, blas) on a new BLAS, traced through a TLAS.
@case "BL-11" 5 :rt begin
    dev = testdevice()
    v, f = trianglearrays(dev); b = BLAS(dev, v, f)
    t = TLAS(dev)
    push!(t, b, MantleArray(dev, instancegrid(4)))
    g = Graph(dev)
    refit!(g, b)
    out = declarehits!(g, dev, t, instancegrid(4))
    run!(g)
    @test isonerange(Array(out))
    @test isempty(pendingbuilds(b))
end

# RR:177-206, VK-DC:28-35 (VUID-03673): geometry arrays keep their build-input usage after moves into Committed and Reserve storage. Grow to 64 triangles, then 2^14.
@case "BL-12" 5 :rt :sparse begin
    dev = testdevice()
    b, v, f = rowblas(dev, 8)
    t = oneinstance(dev, b)
    queryhits(dev, t, instancegrid(1))
    for n in (64, 1 << 14)                                 # a move into Committed storage; past the sparse threshold, a reserve
        copy!(v, rowvertices(n)); copy!(f, rowfaces(n))
        mask, log = stepsduring(() -> hitmask(dev, t, n), dev)
        @test :MoveStorage in stepkinds(log, v)
        n == 64 || @test :BindPages in stepkinds(log, v)
        @test all(mask)
    end
    # VAL: VUID-03673 (build input buffers carry ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY usage) clean
end
