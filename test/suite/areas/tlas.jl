# TLAS (catalogue-resize-rt.md, table L).
#
# References: API = docs/api.md; RR = docs/resizing-and-raytracing.jl (line
# numbers of 2026-10-05); VK-BC/VK-DC = vkdocs chapters/commonvalidity/
# build_acceleration_structure_{common,device_common}.adoc.
#
# Ray tracing is checked with one ray-query kernel (rt_hits!): one ray per
# instance, along +z through the instance's position, writing
# (instanceCustomIndex, index within its range) of the closest hit, or NOHIT.
# Instances stand on a grid of spacing 1 (instancegrid), each a small triangle
# around its position, so the host computes which ray hits which instance.
#
# What this file assumes where the docs name nothing:
#   Mantle.rayquery(t, origin, direction)  the device-side ray query on a TLAS
#       argument (hardware and software), returning (; hit, instanceid,
#       customindex); api.md 2.6 names the built-ins instanceCustomIndex and
#       InstanceId and the device function rangeoffset(tlas, slot), not the
#       query (name open). Indices are one-based, as Mantle hands every index
#       to a shader.
#   Mantle.pendingof(x)  Change objects; a BuildAccel's `mode` is Build() or
#       Update(). Mantle.steplog(dev): entries with `kind` (:InlineStore,
#       :StagedStore, :CellWrite, :MoveStorage, :BindPages, :MoveAccel,
#       :BuildAccel), `target` (the resource the user holds), `seq`, `token`.
#   t.capacity, t.builtcount  the fields api.md section 3 and RR B.8 list.
#   caps(dev).maxinstancecount  the device's maxInstanceCount (G3: names open).
# "Released" is read from memory(dev).cells: an array, BLAS or TLAS gives its
# cell back at its last hold. Validation-layer and GPU-AV expectations are
# comments marked VAL: the suite has no hook that reads validation messages.

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

const TL_STOREKINDS = (:InlineStore, :StagedStore)

"""`c` with the fields in `kw` replaced (caps(dev) is immutable)."""
withcaps(c; kw...) = typeof(c)((get(kw, f, getfield(c, f)) for f in fieldnames(typeof(c)))...)

"""The first `n` grid positions into `alive`, and `n` as its length (a length a kernel writes)."""
@kernel function tl_spawn!(alive, len, n)
    i = @index(Global)
    if i <= n
        @inbounds alive[i] = Point3f((i - 1) % 256, (i - 1) ÷ 256, 0)
        i == 1 && (len[] = n)
    end
end

"""positions[i] = (i - 1, y, 0): a row of instances at height y."""
@kernel function tl_row!(positions, y)
    i = @index(Global)
    positions[i] = Point3f(i - 1, y, 0)
end

"""Rays through rows of 8 instances (x = 0:7) at the given heights."""
rowpixels(heights) = [Point3f(x, y, 0) for y in heights for x in 0:7]

"""Raygen of TL-01: one invocation adds 1."""
tl_raygen(counter) = (@inbounds counter[1] += one(eltype(counter)); nothing)
tl_noop(counter) = nothing

# ── L. TLAS ──

# API:151, RR:124-125: a new TLAS pends a zero-instance Build. An eager ray query and an eager trace over a new TLAS.
@case "TL-01" 5 :rt begin
    dev = testdevice()
    t = TLAS(dev)
    @test buildmode.(pendingbuilds(t)) == [:Build]
    hits, log = stepsduring(() -> queryhits(dev, t, instancegrid(16)), dev)
    @test accelbuilds(log, t) == 1
    @test t.builtcount == 0
    @test allmissed(hits)
    counter = MantleArray(dev, Int32[0])
    trace!(dev, RayTracingPipeline(; raygen = tl_raygen, closest_hit = tl_noop, miss = tl_noop), t, (counter,), 1)
    @test Array(counter) == [1]
    # VAL: clean (the query and the trace read a built, empty structure)
end

# RR:124-125, RR:735-738 (A1): a pending zero-instance Build reaches a plan registered after it. A new TLAS; a graph tracing it compiles and runs, nothing else touches it.
@case "TL-02" 5 :rt begin
    dev = testdevice()
    t = TLAS(dev)
    g = Graph(dev)
    out = declarehits!(g, dev, t, instancegrid(8))
    _, log = stepsduring(() -> run!(g), dev)
    @test accelbuilds(log, t) == 1
    @test isempty(pendingbuilds(t))
    @test allmissed(Array(out))
    # VAL: clean (the trace never reads an unbuilt structure)
end

# RR:1119-1139, API:151: capacity starts at 64 and doubles; outgrowing it, the applier emits MoveAccel before the Build and marks the AccelCommands. 64 instances, then a 65th.
@case "TL-03" 5 :rt begin
    dev = testdevice()
    t = TLAS(dev); b = ownedblas(dev)
    push!(t, b, MantleArray(dev, instancegrid(64)))
    g = Graph(dev)
    refit!(g, t)
    out = declarehits!(g, dev, t, instancegrid(65))
    _, log = stepsduring(() -> run!(g), dev)
    @test :MoveAccel ∉ stepkinds(log, t) && accelbuilds(log, t) == 1
    @test t.capacity == 64
    p = instancegrid(1; from = 64)[1]
    push!(t, b, translation(p...))
    (_, log), d = counted(() -> stepsduring(() -> run!(g), dev))
    k = stepkinds(log, t)
    @test :MoveAccel in k && findfirst(==(:MoveAccel), k) < findlast(==(:BuildAccel), k)
    @test t.capacity == 128
    @test d.plancompiles == 0
    @test d.records >= 1                                   # the refit's AccelCommand, marked by the move
    hits = Array(out)
    @test isonerange(hits[1:64]) && isonerange(hits[65:65]) && hits[65][1] != hits[1][1]
    # VAL: GPU-AV clean (the old AS and scratch are retired with the applying token)
end

# RR:1119-1123: the capacity never halves; the build carries the exact count. Grow to 10^5 instances, then delete! every range.
@case "TL-04" 5 :rt begin
    dev = testdevice()
    t = TLAS(dev); b = ownedblas(dev)
    ranges = [push!(t, b, MantleArray(dev, instancegrid(10_000; from = 10_000k))) for k in 0:9]
    g = Graph(dev)
    out = declarehits!(g, dev, t, instancegrid(100_000))
    run!(g)
    @test all(isonerange, Iterators.partition(Array(out), 10_000))
    @test t.capacity == 131072
    foreach(r -> delete!(t, r), ranges)
    _, log = stepsduring(() -> run!(g), dev)
    @test allmissed(Array(out))
    @test :MoveAccel ∉ stepkinds(log, t)                   # storage kept
    @test t.capacity == 131072 && t.builtcount == 0
    kept = memory(dev)
    free!(g); free!(t)
    @test memory(dev).cells < kept.cells                   # released with free!(t)
end

# RR:22, RR:953-958, DEC B3.2: a build's count is the host total, never the capacity. 3 instances, then 65, then a refit.
@case "TL-05" 5 :rt begin
    dev = testdevice()
    t = TLAS(dev); b = ownedblas(dev)
    src = MantleArray(dev, instancegrid(3))
    push!(t, b, src)
    queryhits(dev, t, instancegrid(1))
    @test t.builtcount == 3 && t.capacity == 64
    push!(t, b, MantleArray(dev, instancegrid(62; from = 3)))
    @test isonerange(queryhits(dev, t, instancegrid(3)))
    @test t.builtcount == 65 && t.capacity == 128
    src[1:1] = [Point3f(0, 5, 0)]                          # a refit carries builtcount
    queryhits(dev, t, instancegrid(1))
    @test t.builtcount == 65
end

# RR:954-956 (Metal reads the count from countcell): builtcount is the host total on Metal too. 3 instances, then 65.
@case "TL-06" 9 :metal :rt begin
    dev = testdevice()
    t = TLAS(dev); b = ownedblas(dev)
    push!(t, b, MantleArray(dev, instancegrid(3)))
    @test isonerange(queryhits(dev, t, instancegrid(3)))
    @test t.builtcount == 3
    push!(t, b, MantleArray(dev, instancegrid(62; from = 3)))
    hits = queryhits(dev, t, instancegrid(65))
    @test isonerange(hits[1:3]) && isonerange(hits[4:65])
    @test t.builtcount == 65
    # countcell is internal: its value is not read through the suite's hooks
end

# API:155, RR:131: tlas[h] = m is a store into the range's source and an Update carrying builtcount. One instance moved.
@case "TL-07" 5 :rt begin
    dev = testdevice()
    t = TLAS(dev); b = ownedblas(dev)
    h = push!(t, b, translation(0, 0, 0))
    @test isonerange(queryhits(dev, t, [Point3f(0, 0, 0)]))
    t[h] = translation(5, 0, 0)
    @test buildmode.(pendingbuilds(t)) == [:Update]
    hits, log = stepsduring(() -> queryhits(dev, t, [Point3f(0, 0, 0), Point3f(5, 0, 0)]), dev)
    i, j = findfirst(s -> s.kind in TL_STOREKINDS, log), findfirst(s -> s.kind === :BuildAccel, log)
    @test i !== nothing && j !== nothing && i < j
    @test accelbuilds(log, t) == 1
    @test hits[1] == NOHIT && isonerange(hits[2:2])
    @test t.builtcount == 1
end

# API:166, RR:132, RR:874: hiding a range is mask 0 in its records, a refit. Three ranges; visible!(t, r2, false), then true.
@case "TL-08" 5 :rt begin
    dev = testdevice()
    t = TLAS(dev); b = ownedblas(dev)
    rs = [push!(t, b, MantleArray(dev, instancegrid(4; from = 4k))) for k in 0:2]
    before = queryhits(dev, t, instancegrid(12))
    visible!(t, rs[2], false)
    @test buildmode.(pendingbuilds(t)) == [:Update]
    hidden = queryhits(dev, t, instancegrid(12))
    @test allmissed(hidden[5:8])
    @test hidden[[1:4; 9:12]] == before[[1:4; 9:12]]       # slots and InstanceIds of the others unchanged
    visible!(t, rs[2], true)
    @test buildmode.(pendingbuilds(t)) == [:Update]
    @test queryhits(dev, t, instancegrid(12)) == before
end

# API:156, RR:1085-1102: delete! deactivates the range's entry and pends a Build; its holds drop after the applying token. Three ranges; delete!(t, r1).
@case "TL-09" 5 :rt begin
    dev = testdevice()
    t = TLAS(dev)
    b1 = ownedblas(dev); src1 = MantleArray(dev, instancegrid(4))
    r1 = push!(t, b1, src1)
    push!(t, ownedblas(dev), MantleArray(dev, instancegrid(4; from = 4)))
    push!(t, ownedblas(dev), MantleArray(dev, instancegrid(4; from = 8)))
    queryhits(dev, t, instancegrid(12))
    delete!(t, r1); free!(b1); free!(src1)
    @test buildmode.(pendingbuilds(t)) == [:Build]
    pending = memory(dev)                                  # the Build is not applied: t still holds b1 and src1
    hits, log = stepsduring(() -> queryhits(dev, t, instancegrid(12)), dev)
    @test accelbuilds(log, t) == 1
    @test allmissed(hits[1:4]) && isonerange(hits[5:8]) && isonerange(hits[9:12])
    @test memory(dev).cells < pending.cells                # released after the Build's token
end

# RR:832, RR:865-868, API:162-164: a deleted range's slot is reused; slots of other ranges are stable. Delete r1 of three, then push! a new range.
@case "TL-10" 5 :rt begin
    dev = testdevice()
    t = TLAS(dev); b = ownedblas(dev)
    rs = [push!(t, b, MantleArray(dev, instancegrid(4; from = 4k))) for k in 0:2]
    before = queryhits(dev, t, instancegrid(12))
    delete!(t, rs[1])
    queryhits(dev, t, instancegrid(1))
    push!(t, b, MantleArray(dev, instancegrid(6; from = 12)))
    after = queryhits(dev, t, instancegrid(18))
    @test allmissed(after[1:4])
    @test after[5:12] == before[5:12] && isonerange(after[5:8]) && isonerange(after[9:12])
    @test isonerange(after[13:18]) && after[13][1] == before[1][1]   # instanceCustomIndex is r1's slot
end

# RR:978-988: build!(g, t) is an AccelCommand recorded once; nothing marks it while the count stays. 100 runs, constant count.
@case "TL-11" 5 :rt begin
    dev = testdevice()
    t = TLAS(dev); b = ownedblas(dev)
    push!(t, b, MantleArray(dev, instancegrid(16)))
    g = Graph(dev)
    build!(g, t)
    out = declarehits!(g, dev, t, instancegrid(16))
    run!(g)
    _, d = counted(() -> foreach(_ -> run!(g), 1:100))
    @test d.records == 0 && d.plancompiles == 0
    @test t.builtcount == 16 && isonerange(Array(out))
    # One Build per run is the AccelCommand's own; the step log shows prefix steps only.
end

# RR:897-903, RR:1037-1045, PLAN:1090-1093: a push! pends a prefix Build; the marked refit is recorded again after it with the new builtcount. refit!(g, t); push! between runs.
@case "TL-12" 5 :rt begin
    dev = testdevice()
    t = TLAS(dev); b = ownedblas(dev)
    push!(t, b, MantleArray(dev, instancegrid(8)))
    g = Graph(dev)
    refit!(g, t)
    out = declarehits!(g, dev, t, instancegrid(12))
    run!(g)
    push!(t, b, MantleArray(dev, instancegrid(4; from = 8)))
    (_, log), d = counted(() -> stepsduring(() -> run!(g), dev))
    @test accelbuilds(log, t) == 1 && :MoveAccel ∉ stepkinds(log, t)
    @test d.records == 1 && d.plancompiles == 0            # only the refit, recorded again
    @test t.builtcount == 12
    hits = Array(out)
    @test isonerange(hits[1:8]) && isonerange(hits[9:12])
    # VAL: VK-BC VUID-03758, VUID-03769 (primitiveCount of an Update), VUID-03663, VUID-03664 clean
end

# TL-13 (TL-12 with the mark! in buildcount! removed, a validation error expected) is not
# written: the suite has no hook that removes a mark! (FI-W) and none that reads validation messages.

# RR:932-933, RR:1018-1024, API:158: a refit of a structure never built is a Build; later refits are Updates. refit!(g, t) compiled while builtcount is -1.
@case "TL-14" 5 :rt begin
    dev = testdevice()
    t = TLAS(dev); b = ownedblas(dev)
    h = push!(t, b, translation(0, 0, 0))
    g = Graph(dev)
    refit!(g, t)
    out = declarehits!(g, dev, t, [Point3f(0, 0, 0), Point3f(3, 0, 0)])
    @test t.builtcount == -1
    run!(g)
    @test t.builtcount == 1
    hits = Array(out)
    @test isonerange(hits[1:1]) && hits[2] == NOHIT
    t[h] = translation(3, 0, 0)
    _, d = counted(() -> run!(g))
    @test d.records == 0                                   # builtcount unchanged: nothing marked
    hits = Array(out)
    @test hits[1] == NOHIT && isonerange(hits[2:2])
end

# RR:984-988, RR:1039: a Build that sets a new builtcount marks every command of the structure. Graph A builds t with a changing count; graph B refits t.
@case "TL-15" 5 :rt begin
    dev = testdevice()
    t = TLAS(dev); b = ownedblas(dev)
    push!(t, b, MantleArray(dev, instancegrid(4)))
    ga = Graph(dev); build!(ga, t)
    gb = Graph(dev); refit!(gb, t)
    out = declarehits!(gb, dev, t, instancegrid(8))
    run!(ga); run!(gb)
    for k in 4:7
        run!(ga)
        _, d = counted(() -> run!(gb))
        @test d.records == 0                               # no count change: B records nothing
        p = instancegrid(1; from = k)[1]
        push!(t, b, translation(p...))
        run!(ga)                                           # applies the Build: a new builtcount
        _, d = counted(() -> run!(gb))
        @test d.records == 1
        _, d = counted(() -> run!(gb))
        @test d.records == 0
    end
    hits = Array(out)
    @test isonerange(hits[1:4]) && all(k -> isonerange(hits[k:k]), 5:8)
end

# VK-BC:113-120 (VUID-03667): every build sets ALLOW_UPDATE, so a refit after any of them is valid. Refits after a prefix Build, an AccelCommand Build and a capacity growth.
@case "TL-16" 5 :rt begin
    dev = testdevice()
    t = TLAS(dev); b = ownedblas(dev)
    h = push!(t, b, translation(0, 0, 0))
    queryhits(dev, t, instancegrid(1))                     # built by a prefix
    t[h] = translation(1, 0, 0)
    @test buildmode.(pendingbuilds(t)) == [:Update]
    hits, log = stepsduring(() -> queryhits(dev, t, [Point3f(1, 0, 0)]), dev)
    @test accelbuilds(log, t) == 1 && isonerange(hits)
    g = Graph(dev); build!(g, t); run!(g)                  # built by an AccelCommand
    t[h] = translation(2, 0, 0)
    @test buildmode.(pendingbuilds(t)) == [:Update]
    @test isonerange(queryhits(dev, t, [Point3f(2, 0, 0)]))
    push!(t, b, MantleArray(dev, instancegrid(100; from = 10)))
    _, log = stepsduring(() -> queryhits(dev, t, instancegrid(1)), dev)
    @test :MoveAccel in stepkinds(log, t)                  # built after a capacity growth
    t[h] = translation(3, 0, 0)
    @test buildmode.(pendingbuilds(t)) == [:Update]
    @test isonerange(queryhits(dev, t, [Point3f(3, 0, 0)]))
    # VAL: VUID-03667 (the source of an Update was built with ALLOW_UPDATE) clean after each
end

# VK-DC:7-27 (VUID-12258, VUID-12259): scratch fits both the build and the update. Build and Update with exactly the capacity (64).
@case "TL-17" 5 :rt begin
    dev = testdevice()
    t = TLAS(dev); b = ownedblas(dev)
    src = MantleArray(dev, instancegrid(64))
    push!(t, b, src)
    @test isonerange(queryhits(dev, t, instancegrid(64)))
    @test t.capacity == 64
    src[1:1] = [Point3f(0, 5, 0)]
    @test buildmode.(pendingbuilds(t)) == [:Update]
    hits = queryhits(dev, t, vcat([Point3f(0, 5, 0)], instancegrid(64)))
    @test hits[2] == NOHIT && isonerange(hits[[1; 3:65]])  # instance 1 moved; 2:64 in place
    # VAL: VUID-12258, VUID-12259 (scratch size) clean
end

# VK-BC:278-283 (VUID-03801), RR:1058, A12: a total past maxInstanceCount throws at push!, never a build or a sizes query past it. maxInstanceCount 1000; 1000 instances, then 500 more.
@case "TL-18" 5 :rt begin   # pending decision 12
    dev = FaultDevice(testdevice())
    inject!(dev, :caps, Answer(1, withcaps(caps(dev.inner); maxinstancecount = 1000)))
    t = TLAS(dev); b = ownedblas(dev)
    push!(t, b, MantleArray(dev, instancegrid(1000)))
    @test isonerange(queryhits(dev, t, instancegrid(1000)))
    @test t.capacity <= 1000
    sizes = calls(dev, :accelsizes)
    @test_throws ArgumentError push!(t, b, MantleArray(dev, instancegrid(500; from = 1000)))
    @test isempty(Mantle.pendingof(t))
    @test calls(dev, :accelsizes) == sizes
    @test isonerange(queryhits(dev, t, instancegrid(1000))) && t.builtcount == 1000
end

# RR:38-39, RR:207, RR:1124-1127: acceleration-structure storage is never a reserve (VUID-03615). t grows past the sparse threshold.
@case "TL-19" 5 :rt :sparse begin
    dev = testdevice()
    t = TLAS(dev); b = ownedblas(dev)
    _, log = stepsduring(dev) do
        for k in 0:3
            push!(t, b, MantleArray(dev, instancegrid(1 << 15; from = k << 15)))
            queryhits(dev, t, instancegrid(1))
        end
    end
    @test t.capacity == 1 << 17
    @test :BindPages ∉ stepkinds(log, t)                   # the records array may bind pages; the AS never does
    @test count(==(:MoveAccel), stepkinds(log, t)) == 3    # 64 → 32768 → 65536 → 131072
end

# RR:954-968, API:151, API:479: a total a kernel writes is built over the capacity with mask-0 records past it; refits stay legal. The kernel writes 10, 50, 3; refit! only.
@case "TL-20" 5 :rt begin
    dev = testdevice()
    t = TLAS(dev); b = ownedblas(dev)
    alive = MantleArray(dev, Point3f, 0; capacity = 256)
    push!(t, b, alive)
    n = GPURef(dev, Int32(10))
    g = Graph(dev)
    dispatch!(g, tl_spawn!, (alive, lengthof(alive), n), 256)
    refit!(g, t)
    out = declarehits!(g, dev, t, instancegrid(256))
    for k in (10, 50, 3)
        n[] = Int32(k)
        run!(g)
        hits = Array(out)
        @test isonerange(hits[1:k]) && allmissed(hits[k+1:end])
        @test t.builtcount == t.capacity
    end
    # VAL: the records past the total reference a valid BLAS; each refit is valid (VUID-03758)
end

# RR:840-842, RR:958: deleting the kernel-written range returns to a host total: a Build with the exact count. TL-20's TLAS plus a host range; delete! the kernel-written one.
@case "TL-21" 5 :rt begin
    dev = testdevice()
    t = TLAS(dev); b = ownedblas(dev)
    alive = MantleArray(dev, Point3f, 0; capacity = 256)
    kernelwritten = push!(t, b, alive)
    push!(t, b, MantleArray(dev, instancegrid(5; from = 300)))
    n = GPURef(dev, Int32(10))
    g = Graph(dev)
    dispatch!(g, tl_spawn!, (alive, lengthof(alive), n), 256)
    refit!(g, t)
    run!(g)
    @test t.builtcount == t.capacity
    delete!(t, kernelwritten)
    hits, log = stepsduring(() -> queryhits(dev, t, instancegrid(5; from = 300)), dev)
    @test accelbuilds(log, t) == 1
    @test t.builtcount == 5 && isonerange(hits)
end

# RR:540-549, RR:1111-1117: resize! of a source pends a Build on its TLAS in the same call; the capacity is the applier's. A source of 10 resized to 100.
@case "TL-22" 5 :rt begin
    dev = testdevice()
    t = TLAS(dev); b = ownedblas(dev)
    src = MantleArray(dev, instancegrid(10))
    push!(t, b, src)
    queryhits(dev, t, instancegrid(1))
    resize!(src, 100)
    @test buildmode.(pendingbuilds(t)) == [:Build]
    @test t.capacity == 64                                 # the call allocates nothing
    src[11:100] = instancegrid(90; from = 10)
    @test isonerange(queryhits(dev, t, instancegrid(100)))
    @test t.capacity == 128 && t.builtcount == 100
end

# RR:1234-1235, RR:1251-1252, API:180: a rebind of a range's array GPURef pends an Update, a Build if the length changed. Rebind to the same length, then to another.
@case "TL-23" 6 :rt begin
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

# RR:1055-1056, RR:1189-1197: push! with a freed member throws; holds taken so far drop; no slot is used. A freed BLAS, a freed source, a freed GPURef.
@case "TL-24" 6 :rt begin
    dev = testdevice()
    b = ownedblas(dev); src = MantleArray(dev, instancegrid(4))
    fresh = TLAS(dev)
    push!(fresh, b, src)
    firstslot = queryhits(dev, fresh, instancegrid(1))[1][1]
    t = TLAS(dev)
    pending = length(Mantle.pendingof(t))
    base = memory(dev)
    dead = ownedblas(dev); free!(dead)
    held = MantleArray(dev, instancegrid(4))
    @test_throws ArgumentError push!(t, dead, held)
    gone = MantleArray(dev, instancegrid(4)); free!(gone)
    @test_throws ArgumentError push!(t, b, gone)
    target = MantleArray(dev, instancegrid(4)); r = GPURef(dev, target); free!(r)
    @test_throws ArgumentError push!(t, b, r)
    @test length(Mantle.pendingof(t)) == pending
    free!(held); free!(target)
    @test memory(dev).cells == base.cells                  # no hold taken by a failed push! survives it
    push!(t, b, src)
    @test queryhits(dev, t, instancegrid(1))[1][1] == firstslot
end

# RR:1060-1071, RR:1095-1102: a range's holds and reader entries go with its delete!. push!(t, blas, pos; scale = pos), then delete!.
@case "TL-25" 5 :rt begin
    dev = testdevice()
    t = TLAS(dev)
    warm = push!(t, ownedblas(dev), MantleArray(dev, instancegrid(4)))   # the range table has its first slot
    delete!(t, warm)
    queryhits(dev, t, instancegrid(1))
    base = memory(dev)
    pos = MantleArray(dev, instancegrid(4)); b = ownedblas(dev)
    r = push!(t, b, pos; scale = pos)
    queryhits(dev, t, instancegrid(1))
    delete!(t, r)
    queryhits(dev, t, instancegrid(1))
    free!(pos); free!(b)
    @test memory(dev).cells == base.cells
end

# RR:818-823, RR:1223-1227 (A22, contract open): a range entry names a cell, which carries no view offset; the suite takes GPURef's rule, a view throws. push! with a view as source.
@case "TL-26" 5 :rt begin
    dev = testdevice()
    t = TLAS(dev); b = ownedblas(dev)
    src = MantleArray(dev, instancegrid(8))
    pending = length(Mantle.pendingof(t))
    @test_throws ArgumentError push!(t, b, view(src, 1:4))
    @test length(Mantle.pendingof(t)) == pending
end

# RR:1090 (A23): delete! of a range not in t throws (the error type is the suite's choice); t unchanged. delete! twice; delete! a range of another TLAS.
@case "TL-27" 5 :rt begin
    dev = testdevice()
    t = TLAS(dev); t2 = TLAS(dev); b = ownedblas(dev)
    r = push!(t, b, MantleArray(dev, instancegrid(4)))
    push!(t, b, MantleArray(dev, instancegrid(4; from = 4)))
    other = push!(t2, b, MantleArray(dev, instancegrid(4)))
    delete!(t, r)
    before = queryhits(dev, t, instancegrid(8))
    @test_throws ArgumentError delete!(t, r)
    @test_throws ArgumentError delete!(t, other)
    @test isempty(pendingbuilds(t))
    @test queryhits(dev, t, instancegrid(8)) == before
end

# RR:1095-1102, API:156: members of a deleted range are released after the applying Build's token, never under a trace in flight. delete!, free!(blas), free!(src) while a trace runs.
@case "TL-28" 5 :rt begin
    dev = testdevice()
    t = TLAS(dev)
    b1 = ownedblas(dev); src1 = MantleArray(dev, instancegrid(4))
    r1 = push!(t, b1, src1)
    push!(t, ownedblas(dev), MantleArray(dev, instancegrid(4; from = 4)))
    gate = MantleArray(dev, Int32[0])
    g = Graph(dev)
    slowwrite!(g, gate, 7)
    out = declarehits!(g, dev, t, instancegrid(8); gate)
    run!(g)                                                # compiled; the TLAS built
    run!(g)                                                # in flight for about 200 ms
    delete!(t, r1); free!(b1); free!(src1)
    inflight = Array(out)
    @test isonerange(inflight[1:4]) && isonerange(inflight[5:8])
    pending = memory(dev)                                  # the Build is not applied: t still holds b1 and src1
    run!(g)
    hits = Array(out)
    @test allmissed(hits[1:4]) && isonerange(hits[5:8])
    @test memory(dev).cells < pending.cells
    # VAL: GPU-AV clean: the in-flight query never reads the released BLAS or source
end

# RR:1124-1127, RR:1141-1143: the old AS is retired with the token of the submission that replaced it, after the traces in flight. A trace in flight; push! past the capacity; another graph's run.
@case "TL-29" 5 :rt :multiqueue begin
    dev = testdevice()
    t = TLAS(dev); b = ownedblas(dev)
    push!(t, b, MantleArray(dev, instancegrid(64)))
    gate = MantleArray(dev, Int32[0])
    ga = Graph(dev); slowwrite!(ga, gate, 7)
    outa = declarehits!(ga, dev, t, instancegrid(64); gate)
    gb = Graph(dev)
    outb = declarehits!(gb, dev, t, instancegrid(128))
    run!(ga); run!(gb)
    run!(ga)                                               # a trace in flight
    push!(t, b, MantleArray(dev, instancegrid(64; from = 64)))
    run!(gb)                                               # applies the growth: MoveAccel
    @test isonerange(Array(outa))
    hits = Array(outb)
    @test isonerange(hits[1:64]) && isonerange(hits[65:128])
    @test t.capacity == 128
    # VAL: GPU-AV clean. Which queue each graph runs on is core's choice; on a device with
    # one queue the case is the single-queue form of the same contract.
end

# RR:302-309, RR:1111-1117: push! and a source's resize from two threads; the build count is the sum of the current lengths, no deadlock. 10^4 iterations with runs.
@case "TL-30" 5 :rt :long begin
    dev = testdevice()
    t = TLAS(dev); b = ownedblas(dev)
    a = MantleArray(dev, instancegrid(8))
    push!(t, b, MantleArray(dev, instancegrid(8; from = 512)))
    g = Graph(dev)
    declarehits!(g, dev, t, instancegrid(1))
    withtimeout(600) do
        pusher = Threads.@spawn for _ in 1:10^4
            r = push!(t, b, a)
            run!(g)
            delete!(t, r)
        end
        resizer = Threads.@spawn for i in 1:10^4
            copy!(a, instancegrid(1 + i % 64))             # a resize and its store in one call
        end
        wait(pusher); wait(resizer)
    end
    push!(t, b, a)
    queryhits(dev, t, instancegrid(1))
    @test t.builtcount == 8 + length(a)
end

# RR:434-439, RR:1155-1158: a TLAS Update pended by a BLAS change waits for the BLAS write applied by another submission. The BLAS rebuilt through another TLAS; then a trace of t.
@case "TL-31" 5 :rt begin
    dev = testdevice()
    v, f = trianglearrays(dev); b = BLAS(dev, v, f)
    t = TLAS(dev); t2 = TLAS(dev)
    push!(t, b, MantleArray(dev, instancegrid(4)))
    push!(t2, b, MantleArray(dev, instancegrid(4)))
    queryhits(dev, t, instancegrid(4)); queryhits(dev, t2, instancegrid(4))
    f[1:1] = [MOVEDFACE]                                   # a BLAS Build; an Update on t and on t2
    _, log = stepsduring(() -> queryhits(dev, t2, instancegrid(4)), dev)
    @test accelbuilds(log, b) == 1 && isempty(stepkinds(log, t))
    @test buildmode.(pendingbuilds(t)) == [:Update]
    @test allmissed(queryhits(dev, t, instancegrid(4)))    # t's Update traced the rebuilt BLAS
    # VAL: synchronization validation clean (t's Update waits for the BLAS write)
end

# RR:1052-1054, RR:1119-1129: new slots grow the range table by Resize steps; nothing recompiles or waits. 1000 one-instance ranges pushed between runs.
@case "TL-32" 5 :rt begin
    dev = testdevice()
    t = TLAS(dev); b = ownedblas(dev)
    g = Graph(dev)
    refit!(g, t)
    out = declarehits!(g, dev, t, instancegrid(1000))
    run!(g)
    foreach(p -> push!(t, b, translation(p...)), instancegrid(1000))
    _, d = counted(() -> run!(g))
    @test d.plancompiles == 0 && d.hostwaits == 0
    hits = Array(out)
    @test all(h -> h[1] != NOINSTANCE && h[2] == 1, hits) && allunique(h[1] for h in hits)
end

# RR:865-868, RR:878-886: the records kernel reads a source through its cell. A range source moved by a resize past its storage.
@case "TL-33" 5 :rt begin
    dev = testdevice()
    t = TLAS(dev); b = ownedblas(dev)
    src = MantleArray(dev, instancegrid(8))
    push!(t, b, src)
    queryhits(dev, t, instancegrid(1))
    copy!(src, instancegrid(4096))
    hits, log = stepsduring(() -> queryhits(dev, t, instancegrid(4096)), dev)
    @test :MoveStorage in stepkinds(log, src)
    @test isonerange(hits)
end

# RR:1403-1405, API:455: a frame with nothing pending does no acceleration-structure step and re-records nothing. A graph with refit!(g, t), run with no change.
@case "TL-34" 5 :rt begin
    dev = testdevice()
    t = TLAS(dev); b = ownedblas(dev)
    push!(t, b, MantleArray(dev, instancegrid(16)))
    g = Graph(dev)
    refit!(g, t)
    declarehits!(g, dev, t, instancegrid(16))
    run!(g); run!(g)
    (_, log), d = counted(() -> stepsduring(() -> run!(g), dev))
    @test isempty(stepkinds(log, t)) && d.records == 0
end

# API:408-416, RR:1394-1396: a run that writes a source pends its TLAS's Update; a graph tracing t applies it after that write. Graph S writes the source each run; graph F traces.
@case "TL-35" 5 :rt begin
    dev = testdevice()
    t = TLAS(dev); b = ownedblas(dev)
    src = MantleArray(dev, rowpixels(0:0))
    push!(t, b, src)
    height = GPURef(dev, 0f0)
    s = Graph(dev)
    dispatch!(s, tl_row!, (src, height), launchrange(src))
    f = Graph(dev)
    out = declarehits!(f, dev, t, rowpixels(0:2:10))
    for k in 1:5
        height[] = Float32(2k)
        run!(s)
        @test buildmode.(pendingbuilds(t)) == [:Update]
        run!(f)
        hits = Array(out)
        lit = 8k+1:8k+8
        @test isonerange(hits[lit]) && allmissed(hits[setdiff(1:48, lit)])
    end
end

# API:411-412 (pendupdate! after a stage), RR:992-993: a run that writes a source and refits t leaves no Update pending. One graph writes the source and has refit!(g, t).
@case "TL-36" 5 :rt begin
    dev = testdevice()
    t = TLAS(dev); b = ownedblas(dev)
    src = MantleArray(dev, rowpixels(0:0))
    push!(t, b, src)
    height = GPURef(dev, 0f0)
    s = Graph(dev)
    dispatch!(s, tl_row!, (src, height), launchrange(src))
    refit!(s, t)
    out = declarehits!(s, dev, t, rowpixels(0:2:6))
    for k in 1:3
        height[] = Float32(2k)
        run!(s)
        @test isempty(pendingbuilds(t))
        hits = Array(out)
        lit = 8k+1:8k+8
        @test isonerange(hits[lit]) && allmissed(hits[setdiff(1:32, lit)])
    end
end

# RR:1381-1384, API:151: traversal of an empty software TLAS returns a miss. A software TLAS with no instances, queried.
@case "TL-37" 5 begin
    dev = testdevice()
    t = TLAS(dev; hardware = false)
    @test allmissed(queryhits(dev, t, instancegrid(64)))
    # VAL: GPU-AV clean: no node is read out of bounds
end

# RR:1155-1158, API:150: TLASes over a BLAS pend an Update when its contents change. Two TLASes over one BLAS; the BLAS rebuilt.
@case "TL-38" 5 :rt begin
    dev = testdevice()
    v, f = trianglearrays(dev); b = BLAS(dev, v, f)
    t1 = TLAS(dev); t2 = TLAS(dev)
    push!(t1, b, MantleArray(dev, instancegrid(4)))
    push!(t2, b, MantleArray(dev, instancegrid(4)))
    queryhits(dev, t1, instancegrid(1)); queryhits(dev, t2, instancegrid(1))
    f[1:1] = [MOVEDFACE]
    @test buildmode.(pendingbuilds(b)) == [:Build]
    @test buildmode.(pendingbuilds(t1)) == [:Update] && buildmode.(pendingbuilds(t2)) == [:Update]
    @test allmissed(queryhits(dev, t1, instancegrid(4))) && allmissed(queryhits(dev, t2, instancegrid(4)))
end

# RR:1119-1123, RR:897-899: capacity doubles until the total fits, in one step; pending builds coalesce into one. 300 instances before the first submission.
@case "TL-39" 5 :rt begin
    dev = testdevice()
    t = TLAS(dev); b = ownedblas(dev)
    push!(t, b, MantleArray(dev, instancegrid(300)))
    hits, log = stepsduring(() -> queryhits(dev, t, instancegrid(300)), dev)
    @test count(==(:MoveAccel), stepkinds(log, t)) == 1
    @test accelbuilds(log, t) == 1
    @test t.capacity == 512 && isonerange(hits)
end

# RR:1018-1034, RR:1135: a MoveAccel marks the AccelCommands; a plan's refit in the same submission is recorded after the prefix with the new handle. refit!(g, t); growth past capacity.
@case "TL-40" 5 :rt begin
    dev = testdevice()
    t = TLAS(dev); b = ownedblas(dev)
    push!(t, b, MantleArray(dev, instancegrid(32)))
    g = Graph(dev)
    refit!(g, t)
    out = declarehits!(g, dev, t, instancegrid(132))
    run!(g)
    push!(t, b, MantleArray(dev, instancegrid(100; from = 32)))
    (_, log), d = counted(() -> stepsduring(() -> run!(g), dev))
    @test :MoveAccel in stepkinds(log, t)
    @test d.records >= 1 && d.plancompiles == 0
    hits = Array(out)
    @test isonerange(hits[1:32]) && isonerange(hits[33:132])
    # VAL: clean (the refit records the new AS handle, accelstructures.adoc:764-769)
end

# RR:327-344, RR:889-895: capacity prepared before the locks no longer fits: Again, prepared again. A push! past the prepared capacity lands between prepare! and the locks (FI-P: placeaccel blocked).
@case "TL-41" 5 :rt begin
    dev = FaultDevice(testdevice())
    t = TLAS(dev); b = ownedblas(dev)
    push!(t, b, MantleArray(dev, instancegrid(64)))
    queryhits(dev, t, instancegrid(1))
    push!(t, b, MantleArray(dev, instancegrid(1; from = 64)))   # 65: needs 128
    gap = Channel{Nothing}(1)
    at = calls(dev, :placeaccel) + 1
    inject!(dev, :placeaccel, Block(at, gap))
    applying = Threads.@spawn queryhits(dev, t, instancegrid(265))
    @test timedwait(() -> calls(dev, :placeaccel) >= at, 30) === :ok
    push!(t, b, MantleArray(dev, instancegrid(200; from = 65)))  # 265: the prepared 128 no longer fits
    put!(gap, nothing)
    hits = withtimeout(() -> fetch(applying), 60)
    @test calls(dev, :placeaccel) > at                     # prepared again, for 512
    @test t.capacity == 512 && t.builtcount == 265
    @test isonerange(hits[1:64]) && isonerange(hits[65:65]) && isonerange(hits[66:265])
end

# RR:1294-1296: a BLAS reached only through a TLAS is resident (Metal's device-wide residency set). The BLAS's creator handle freed; t traced.
@case "TL-42" 9 :metal :rt begin
    dev = testdevice()
    t = TLAS(dev); b = ownedblas(dev)
    push!(t, b, MantleArray(dev, instancegrid(4)))
    free!(b)
    GC.gc()
    @test isonerange(queryhits(dev, t, instancegrid(4)))
end
