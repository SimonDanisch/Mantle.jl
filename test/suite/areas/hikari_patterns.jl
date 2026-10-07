# Hikari's patterns (catalogue-resize-rt.md, table P), written against
# Mantle's API only, in the shapes of resizing-and-raytracing.jl D.2.
#
# References: API = docs/api.md; RR = docs/resizing-and-raytracing.jl (line
# numbers of 2026-10-05).
#
# The scene (hikariscene) is D.2's: one TLAS, a material table, and the
# screen's film, accumulation and sample count, which are device arrays so
# they survive runs and recompiles. The integrator (integrate!) is declared
# once into the frame graph and reduced to its shape: one ray per pixel
# against the TLAS (a ray query, as in tlas.jl), the material's value on a
# hit, 0 on a miss. Pixels stand on the instance grid of tlas.jl, so the host
# computes the expected film.
#
# What this file assumes where the docs name nothing (as tlas.jl):
#   Mantle.rayquery(t, origin, direction) -> (; hit, instanceid, customindex),
#       the device-side ray query on hardware and software TLASes (name open);
#   Mantle.pendingof(x), Mantle.steplog(dev) as in tlas.jl (build steps are
#       :BuildAccel, capacity steps :MoveAccel).
# Validation-layer expectations are comments marked VAL.

using KernelAbstractions: @kernel, @index
using GeometryBasics: Point3f, Vec3f, Mat4f, TriangleFace

# ── Ray-tracing helpers, a subset of those in tlas.jl, blas.jl and gpuref.jl ──

const TRIANGLEVERTICES = (Point3f(-0.4, -0.4, 0), Point3f(0.4, -0.4, 0), Point3f(0, 0.4, 0), Point3f(0, -0.2, 0))
const TRIANGLEFACE = TriangleFace{UInt32}(1, 2, 3)   # covers the origin

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

# ── This file's helpers ──

"""Rays through rows of 8 instances (x = 0:7) at the given heights."""
rowpixels(heights) = [Point3f(x, y, 0) for y in heights for x in 0:7]

"""The film of `rows` pixel rows of 8 with row `k` (zero-based) lit at `value`."""
litrow(k, rows; value = 1f0) = [y == k ? value : 0f0 for y in 0:rows-1 for x in 0:7]

"""D.2's scene: a TLAS, a material table, and the film, accumulation and sample count the screen holds."""
hikariscene(dev, pixels; tlas = TLAS(dev)) =
    (; tlas, materials = MantleArray(dev, [1f0]), film = MantleArray(dev, zeros(Float32, length(pixels))),
       accum = MantleArray(dev, zeros(Float32, length(pixels))), samples = MantleArray(dev, Int32[0]),
       origins = rayorigins(dev, pixels))

"""The factor of a material type: Val(M) stands for one."""
brightness(::Val{M}) where {M} = Float32(M)

"""
One sample per pixel: film[i] = brightness(material) * materials[1] on a hit,
0 on a miss; accum adds it; thread 1 counts the sample.
"""
@kernel function hk_shade!(film, accum, samples, t, origins, materials, material, gate)
    i = @index(Global)
    q = Mantle.rayquery(t, origins[i], Vec3f(0, 0, 1))
    c = q.hit & (gate[1] != Int32(-1)) ? brightness(material) * materials[1] : 0f0
    film[i] = c
    accum[i] += c
    i == 1 && (samples[1] += Int32(1))
end

"""Declares the integrator of scene `s` into `g` (D.2's declare!, reduced to one launch)."""
integrate!(g, dev, s; material = Val(1), gate = readygate(dev)) =
    dispatch!(g, hk_shade!, (s.film, s.accum, s.samples, s.tlas, s.origins, s.materials, material, gate),
              length(s.film))

"""The first `n` grid positions into `alive`, and `n` as its length (a particle count a kernel writes)."""
@kernel function hk_spawn!(alive, len, n)
    i = @index(Global)
    if i <= n
        @inbounds alive[i] = Point3f((i - 1) % 256, (i - 1) ÷ 256, 0)
        i == 1 && (len[] = n)
    end
end

"""positions[i] = (i - 1, y, 0): a simulation step moving a row of instances to height y."""
@kernel function hk_row!(positions, y)
    i = @index(Global)
    positions[i] = Point3f(i - 1, y, 0)
end

"""One window pixel (x, y): film[x + w(y - 1)] = 1 where its ray hits, else 0; w is the queue's width."""
@kernel function hk_windowshade!(film, queue, t)
    x, y = @index(Global, NTuple)
    w = size(queue, 1)
    q = Mantle.rayquery(t, Point3f(x - 1, y - 1, -1), Vec3f(0, 0, 1))
    i = x + w * (y - 1)
    i <= length(film) && (film[i] = q.hit ? 1f0 : 0f0)
end

"""The work queue's items and length, the loop flag and the bounce count, set for a new frame."""
@kernel function hk_seed!(queue, len, active, bounces, n)
    i = @index(Global)
    if i <= n
        @inbounds queue[i] = UInt32(i)
        if i == 1
            len[] = n
            active[1] = Int32(1)
            bounces[1] = Int32(0)
        end
    end
end

"""
D.2's shade!, reduced: a ray per queue item; thread 1 halves the queue's
length, sets active = (length > 0) and counts the bounce.
"""
@kernel function hk_bounce!(queue, len, active, bounces, t)
    i = @index(Global)
    n = length(queue)
    q = Mantle.rayquery(t, Point3f(queue[i] % 8, 0, -1), Vec3f(0, 0, 1))
    queue[i] = q.hit ? queue[i] : UInt32(0)
    if i == 1
        len[] = n ÷ 2
        active[1] = Int32(n ÷ 2 > 0)
        bounces[1] += Int32(1)
    end
end

"""Ray-generation shader of HK-08's trace: the queue is read by the shade launch."""
hk_raygen(queue) = nothing
hk_noop(queue) = nothing

# ── P. Hikari (D.2) ──

# RR:1394-1405, DEC B3.16: scene changes are pending TLAS changes and stores; the integrator is declared once and compiles nothing for them. Between frames: push! a mesh, delete! an object, store a material value.
@case "HK-01" 5 :rt begin
    dev = testdevice()
    s = hikariscene(dev, instancegrid(12))
    b = ownedblas(dev)
    push!(s.tlas, b, MantleArray(dev, instancegrid(4)))
    gone = push!(s.tlas, b, MantleArray(dev, instancegrid(4; from = 4)))
    g = Graph(dev)
    integrate!(g, dev, s)
    run!(g)
    @test Array(s.film) == [fill(1f0, 8); zeros(Float32, 4)]
    push!(s.tlas, b, MantleArray(dev, instancegrid(4; from = 8)))
    delete!(s.tlas, gone)
    s.materials[1:1] = [2f0]
    _, d = counted(() -> run!(g))
    @test d.plancompiles == 0 && d.kernelcompiles == 0
    @test Array(s.film) == [fill(2f0, 4); zeros(Float32, 4); fill(2f0, 4)]
end

# RR:1403-1405: a frame with no change does no acceleration-structure work. 100 frames with no change.
@case "HK-02" 5 :rt begin
    dev = testdevice()
    s = hikariscene(dev, instancegrid(16))
    push!(s.tlas, ownedblas(dev), MantleArray(dev, instancegrid(16)))
    g = Graph(dev)
    integrate!(g, dev, s)
    run!(g)
    (_, log), d = counted(() -> stepsduring(() -> foreach(_ -> run!(g), 1:100), dev))
    @test isempty(stepkinds(log, s.tlas))
    @test d.records == 0 && d.plancompiles == 0
    @test Array(s.samples) == [101]
end

# RR:1398-1400, API:107-113: film and sample count are device arrays the screen holds; a recompile (a new material type) keeps their contents. Three frames, the integrator replaced, one frame.
@case "HK-03" 5 :rt begin
    dev = testdevice()
    s = hikariscene(dev, instancegrid(8))
    push!(s.tlas, ownedblas(dev), MantleArray(dev, instancegrid(4)))
    g = Graph(dev)
    h = insert!(g) do
        integrate!(g, dev, s)
    end
    foreach(_ -> run!(g), 1:3)
    accum = Array(s.accum)
    @test Array(s.samples) == [3] && accum == [fill(3f0, 4); zeros(Float32, 4)]
    delete!(g, h)
    insert!(g) do
        integrate!(g, dev, s; material = Val(2))           # a new material type: a new argument type
    end
    _, d = counted(() -> run!(g))
    @test d.plancompiles == 1
    @test Array(s.samples) == [4]
    @test Array(s.accum) == accum .+ [fill(2f0, 4); zeros(Float32, 4)]
end

# RR:1400-1403, API:191 (A20, film size open): a window resize within the capacity records the window's segments again; nothing is declared or compiled. The film is made resizable before the plan compiles, so its size is never recorded.
@case "HK-04" 7 :rt :window :graphics begin
    dev = testdevice()
    win = Window(dev, 64, 32; capacity = (128, 64))
    t = TLAS(dev)
    push!(t, ownedblas(dev), translation(0, 0, 0))
    film = MantleArray(dev, Float32, 64 * 32)
    resize!(film, 64 * 32)
    g = Graph(dev)
    queue = MantleArray(g, UInt32, win)                    # a window-sized work queue (D.2)
    dispatch!(g, hk_windowshade!, (film, queue, t), launchrange(queue))
    render!(g, win => Clear((0f0, 0f0, 0f0, 1f0))) do p
    end
    run!(g)
    resize!(win, 96, 48)
    resize!(film, 96 * 48)
    _, d = counted(() -> run!(g))
    @test d.plancompiles == 0 && d.records >= 1
    pixels = Array(film)
    @test length(pixels) == 96 * 48 && pixels[1] == 1f0 && count(==(1f0), pixels) == 1
end

# RR:1405-1406, API:411-412: transforms a kernel of the frame graph writes are refit by its refit! node: one Update a frame, none pended. The graph moves a row and refits.
@case "HK-05" 5 :rt begin
    dev = testdevice()
    s = hikariscene(dev, rowpixels(0:2:6))
    src = MantleArray(dev, rowpixels(0:0))
    push!(s.tlas, ownedblas(dev), src)
    height = GPURef(dev, 0f0)
    g = Graph(dev)
    dispatch!(g, hk_row!, (src, height), launchrange(src))
    refit!(g, s.tlas)
    integrate!(g, dev, s)
    run!(g)
    for k in 1:3
        height[] = Float32(2k)
        _, log = stepsduring(() -> run!(g), dev)
        @test isempty(stepkinds(log, s.tlas))              # no prefix Update: the refit node is the frame's one
        @test isempty(pendingbuilds(s.tlas))
        @test Array(s.film) == litrow(k, 4)
    end
end

# RR:1406-1408, RR:954-968: a particle count a kernel writes is built over the capacity once; then refits only. The count changes over frames.
@case "HK-06" 5 :rt begin
    dev = testdevice()
    s = hikariscene(dev, instancegrid(256))
    alive = MantleArray(dev, Point3f, 0; capacity = 256)
    push!(s.tlas, ownedblas(dev), alive)
    n = GPURef(dev, Int32(10))
    g = Graph(dev)
    dispatch!(g, hk_spawn!, (alive, lengthof(alive), n), 256)
    refit!(g, s.tlas)
    integrate!(g, dev, s)
    _, log = stepsduring(() -> run!(g), dev)
    @test :BuildAccel in stepkinds(log, s.tlas)            # the one Build, over the capacity
    for k in (40, 7, 200)
        n[] = Int32(k)
        (_, log), d = counted(() -> stepsduring(() -> run!(g), dev))
        @test isempty(stepkinds(log, s.tlas)) && d.records == 0
        @test s.tlas.builtcount == s.tlas.capacity
        @test Array(s.film) == [i <= k ? 1f0 : 0f0 for i in 1:256]
    end
end

# RR:1394-1396, API:408-416: a simulation run writing positions pends the TLAS's Update; the frame applies it after that write, and the simulation's next write waits for the frame. The frame's refit reads the positions after a slow kernel.
@case "HK-07" 5 :rt begin
    dev = testdevice()
    s = hikariscene(dev, rowpixels(0:2:8))
    src = MantleArray(dev, rowpixels(0:0))
    push!(s.tlas, ownedblas(dev), src)
    height = GPURef(dev, 0f0)
    sim = Graph(dev)
    dispatch!(sim, hk_row!, (src, height), launchrange(src))
    gate = MantleArray(dev, Int32[0])
    frame = Graph(dev)
    slowwrite!(frame, gate, 7)
    refit!(frame, s.tlas)                                  # after the slow kernel: reads src late in the frame
    integrate!(frame, dev, s; gate)
    run!(sim); run!(frame)
    for k in 1:3
        height[] = Float32(2k)
        run!(sim)
        @test buildmode.(pendingbuilds(s.tlas)) == [:Update]
        run!(frame)                                        # in flight for about 200 ms
        height[] = Float32(2k + 1)
        run!(sim)                                          # must wait on the GPU for the frame
        @test Array(s.film) == litrow(k, 5)                # the frame traced step k's positions
    end
end

# RR:1409-1416, API:129-131: a loop around a trace whose queue length a kernel writes ends on the GPU when the length is 0; the remaining traces have count 0. repeat! with while_nonzero over trace! and shade.
@case "HK-08" 5 :rt begin
    dev = testdevice()
    t = TLAS(dev)
    push!(t, ownedblas(dev), MantleArray(dev, instancegrid(8)))
    pipeline = RayTracingPipeline(; raygen = hk_raygen, closest_hit = hk_noop, miss = hk_noop)
    g = Graph(dev)
    queue = MantleArray(g, UInt32, 0; capacity = 64)       # a work queue whose length a kernel writes
    active = MantleArray(dev, Int32, 1); bounces = MantleArray(dev, Int32, 1)
    dispatch!(g, hk_seed!, (queue, lengthof(queue), active, bounces, Int32(64)), 64)
    repeat!(g, 10; while_nonzero = active) do i
        trace!(g, pipeline, t, (queue,), launchrange(queue))
        dispatch!(g, hk_bounce!, (queue, lengthof(queue), active, bounces, t), launchrange(queue))
    end
    for _ in 1:2
        run!(g)
        @test Array(bounces) == [7]                        # 64, 32, 16, 8, 4, 2, 1 items; then 0 and the loop ends
    end
    # VAL: clean; the traces and launches after the loop ended have count 0
end

# RR:124-125, RR:1381-1384: an empty scene renders: every ray misses, on the hardware and the software TLAS.
@case "HK-09" 5 begin
    dev = testdevice()
    for tlas in (TLAS(dev), TLAS(dev; hardware = false))
        s = hikariscene(dev, instancegrid(64); tlas)
        g = Graph(dev)
        integrate!(g, dev, s)
        run!(g)
        @test all(iszero, Array(s.film)) && Array(s.samples) == [1]
    end
    # VAL: clean; software traversal reads no node of the empty structure
end
