# Windows and present (catalogue-core.md, table 15; api.md 2.7; recording-plan.md,
# "Windows and present"; internals.jl section 7; decisions.md B3.8; ledger
# phase 7).
#
# resize!(win, w, h) is called the way the window's resize event calls it; the
# cases assume the surface takes the requested extent (the native window's
# size is the window open verb's, G13). A zero extent is what the event reports
# for a minimized window.
#
# Array(win) returns once a run presenting to the window has copied its image,
# so windowimage runs the graph until it does. The readback is a `w × h`
# matrix, first index x, of a colour type with fields r, g, b (the format is
# open, G9); the cases use colour components 0 and 1, which an sRGB swapchain
# stores unchanged.
#
# The cases that count or fault window verbs (acquirenext, createswapchain,
# surfacesize) go through FaultDevice. api.md 4.7 gives those verbs a window or
# a swapchain as first argument, not the device: FaultDevice's forwarding has
# to reach them for these cases to count anything.

using KernelAbstractions: @kernel, @index, @Const
using GeometryBasics: Vec3f, Vec4f, Point3f, Mat4f, TriangleFace
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

# Rectangles, one per instance: rects[i] = (x0, y0, x1, y1) in normalized
# device coordinates; colours one for all rectangles or one per rectangle,
# read with attribute(a, i) (api.md 2.6b). vertex_index and instance_index
# count from one; vertices 1-3 and 4-6 are the two triangles.
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

"""
A graph presenting to `win`: `before(g)` declares its work before the window
pass, which draws stripe `k` of `n` in `colour`. Returns the graph, the pass and
the pass's arrays (stores into them change the next frame).
"""
function presenting(win, dev; k = 1, n = 1, colour = GREEN, before = g -> nothing)
    g = Graph(dev)
    before(g)
    rects = MantleArray(dev, [stripe(k, n)])
    colours = MantleArray(dev, [opaque(colour)])
    p = render!(p -> drawrects!(p, rects, colours), g, win => BACKGROUND)
    return (; g, p, rects, colours)
end

"""A graph that presents to `win` once and is then garbage."""
presentonce(win, dev) = (run!(presenting(win, dev).g); nothing)

"""A kernel, then a host node calling `f`: a stage boundary before what is declared next."""
function hostnode!(g, f)
    t = MantleArray(g, Int32, 1)
    dispatch!(g, win_setone!, (t,), 1)
    dispatch!(g, HostCall(f), (t,); writes = ())
end

"""Copy nodes of `g` reading the device image `img` into the device array `out`."""
function readimage!(g, img, out)
    t = MantleArray(g, eltype(out), size(out)...)
    copyto!(t, img)
    copyto!(out, t)
end

"""
`Array(win)` (a screenshot) on another task while this task runs `g` until it
returns: the image of a frame presented after the call.
"""
function windowimage(g, win; timeout = 30)
    shot = Threads.@spawn Array(win)
    deadline = time() + timeout
    while !istaskdone(shot) && time() < deadline
        run!(g)
    end
    istaskdone(shot) || error("Array(win) did not return within $timeout s")
    return fetch(shot)
end

"""The exception `f()` throws on another task within `timeout` seconds; `nothing` if it returned or still runs."""
function thrownwithin(f, timeout)
    t = Threads.@spawn f()
    timedwait(() -> istaskdone(t), timeout)
    return istaskfailed(t) ? t.exception : nothing
end

"""Runs `f(dev)` where the contract is Vulkan's (swapchain images, result codes); skipped elsewhere."""
onvulkan(f, dev::Mantle.LavaDevice) = f(dev)
onvulkan(f, dev) = @test_skip "Vulkan only"

"""What acquirenext returns for an out-of-date swapchain: the result, no image."""
outofdate(::Mantle.LavaDevice) = (Mantle.VK.ERROR_OUT_OF_DATE_KHR, 0, nothing)

"""A FaultDevice answer for the `at`-th call only (an `Answer` answers every call from `at` on)."""
# AnswerOnce and totalcalls are in faultdevice.jl.

"""A graph drawing into a window-sized image and into the window, whose capacity is `capacity`."""
function layeredwindow(dev, capacity)
    win = Window(dev, 64, 64; capacity)
    g = Graph(dev)
    layer = Image(g, RGBA{N0f8}, win)
    rects = MantleArray(dev, [stripe(1, 1)])
    colours = MantleArray(dev, [opaque(GREEN)])
    render!(p -> drawrects!(p, rects, colours), g, layer => BACKGROUND)
    render!(p -> drawrects!(p, rects, colours), g, win => BACKGROUND)
    return (; g, win)
end

@kernel function win_setone!(a)
    @inbounds a[1] = one(eltype(a))
end

@kernel function win_lengthout!(out, @Const(a))
    @inbounds out[1] = Int32(length(a))
end

@kernel function win_clear!(w)
    I = @index(Global, Cartesian)
    @inbounds w[I] = zero(eltype(w))
end

# Copies src[1] into dst[1] after about `iters` dependent steps: a write that
# lands late (the chain of helpers.jl's slowwrite!).
@kernel function win_slowcopy!(dst, @Const(src), iters)
    acc = Int32(0)
    for i in Int32(1):iters
        acc = acc * Int32(1103515245) + i
    end
    @inbounds dst[1] = acc == Int32(-7) ? zero(eltype(dst)) : src[1]
end

# Reads the depth image and writes a transform scaled by it.
@kernel function win_depthscale!(transforms, @Const(depth))
    @inbounds d = depth[1, 1]
    @inbounds transforms[1] = Mat4f(d, 0, 0, 0, 0, d, 0, 0, 0, 0, d, 0, 0, 0, 0, 1)
end

# ── 15. Windows and present (WIN) ──

# api.md:190, decisions.md:185-192. g1 presents to win; g2 declaring render! into win throws at declaration; g2 is unchanged.
@case "WIN-01" 7 :graphics :window begin
    dev = testdevice()
    win = Window(dev, 64, 64)
    s = presenting(win, dev)
    run!(s.g)
    g2 = Graph(dev)
    x = MantleArray(dev, Int32, 1)
    dispatch!(g2, win_setone!, (x,), 1)
    run!(g2)
    @test_throws ArgumentError render!(p -> drawrects!(p, s.rects, s.colours), g2, win => BACKGROUND)
    _, d = counted(() -> run!(g2))
    @test d.plancompiles == 0
    @test Array(x) == Int32[1]
    @test iscolour(stripepixel(windowimage(s.g, win), 1, 1), GREEN)
end

# recording-plan.md:1022-1023 (ambiguity 13). One graph declaring render! into a second window throws at declaration.
@case "WIN-02" 7 :graphics :window begin   # pending decision 4
    dev = testdevice()
    win1, win2 = Window(dev, 64, 64), Window(dev, 64, 64)
    s = presenting(win1, dev)
    @test_throws ArgumentError render!(p -> drawrects!(p, s.rects, s.colours), s.g, win2 => BACKGROUND)
    @test iscolour(stripepixel(windowimage(s.g, win1), 1, 1), GREEN)
    s2 = presenting(win2, dev; colour = BLUE)       # win2 is still free
    @test iscolour(stripepixel(windowimage(s2.g, win2), 1, 1), BLUE)
end

# examples.jl:490 vs decisions.md:185 (ambiguity 13). A kernel of the presenting graph may write the window; one of another graph throws at declaration.
@case "WIN-03" 7 :graphics :window begin   # pending decision 4
    dev = testdevice()
    win = Window(dev, 64, 64)
    s = presenting(win, dev)
    dispatch!(s.g, win_clear!, (win,), (64, 64))    # after the pass: clears what it drew
    g2 = Graph(dev)
    @test_throws ArgumentError dispatch!(g2, win_clear!, (win,), (64, 64))
    @test iscolour(stripepixel(windowimage(s.g, win), 1, 1), BLACK)
end

# api.md:191, recording-plan.md:1031-1059. resize! on the event thread while a run is on the GPU; the next run: the new size, no device-wide wait, no plan compile.
@case "WIN-04" 7 :graphics :window begin
    dev = testdevice()
    win = Window(dev, 64, 64; capacity = (128, 128))
    busy = MantleArray(dev, Int32, 1)
    s = presenting(win, dev; before = g -> slowwrite!(g, busy, 1; ms = 500))
    run!(s.g); run!(s.g)
    Mantle.waitidle(dev)
    run!(s.g)                                        # on the GPU for about 500 ms
    withtimeout(() -> resize!(win, 96, 80), 5)       # the event, on another thread
    local d
    t = @elapsed begin
        _, d = counted(() -> run!(s.g))
    end
    @test t < 0.25                                   # returned while the earlier run is on the GPU
    @test d.plancompiles == 0
    A = windowimage(s.g, win)
    @test size(A) == (96, 80)
    @test iscolour(stripepixel(A, 1, 1), GREEN)
end

# api.md:191. resize!(win) makes no driver call, submission, allocation, record or compile on the event thread.
@case "WIN-05" 7 :graphics :window begin
    fd = FaultDevice(testdevice())
    win = Window(fd, 64, 64; capacity = (128, 128))
    s = presenting(win, fd)
    run!(s.g); run!(s.g)
    before = totalcalls(fd)
    _, d = counted(() -> withtimeout(() -> resize!(win, 80, 72), 5), fd)
    @test totalcalls(fd) == before
    @test submitcount(d) == 0
    @test d.allocations == 0
    @test d.records == 0
    @test d.plancompiles == 0
end

# api.md:191, decisions.md:442-444. A resize beyond the capacity never throws; the next run compiles the plan once.
@case "WIN-06" 7 :graphics :window begin
    dev = testdevice()
    win = Window(dev, 64, 64; capacity = (128, 128))
    out = MantleArray(dev, Int32, 1)
    s = presenting(win, dev; before = g -> dispatch!(g, win_lengthout!, (out, MantleArray(g, Float32, win)), 1))
    run!(s.g)
    resize!(win, 160, 144)
    _, d = counted(() -> run!(s.g))
    @test d.plancompiles == 1
    _, d = counted(() -> run!(s.g))
    @test d.plancompiles == 0
    @test Array(out) == Int32[160 * 144]
    A = windowimage(s.g, win)
    @test size(A) == (160, 144)
    @test iscolour(stripepixel(A, 1, 1), GREEN)
end

# api.md:483. Minimized: run! returns Skipped(), submits nothing, the store stays pending; after the restore it is applied.
@case "WIN-07" 7 :graphics :window begin   # pending decision 4
    dev = testdevice()
    win = Window(dev, 64, 64)
    s = presenting(win, dev)
    run!(s.g); run!(s.g)
    resize!(win, 0, 0)
    s.colours[1:1] = [opaque(BLUE)]
    r, d = counted(() -> run!(s.g))
    @test r isa Mantle.Skipped
    @test submitcount(d) == 0
    @test !isempty(Mantle.pendingof(s.colours))
    resize!(win, 64, 64)
    @test run!(s.g) isa Mantle.Submitted
    @test isempty(Mantle.pendingof(s.colours))
    @test iscolour(stripepixel(windowimage(s.g, win), 1, 1), BLUE)
end

# api.md:483. OUT_OF_DATE [FI-3]: a resize to surfacesize, a new generation, retried and presented by the same run!.
@case "WIN-08" 7 :graphics :window begin
    onvulkan(testdevice()) do dev
        fd = FaultDevice(dev)
        win = Window(fd, 64, 64)
        s = presenting(win, fd)
        run!(s.g); run!(s.g)
        acquires, sizes, chains = calls(fd, :acquirenext), calls(fd, :surfacesize), calls(fd, :createswapchain)
        inject!(fd, :acquirenext, AnswerOnce(acquires + 1, outofdate(dev)))
        @test run!(s.g) isa Mantle.Submitted
        @test calls(fd, :acquirenext) == acquires + 2
        @test calls(fd, :surfacesize) == sizes + 1
        @test calls(fd, :createswapchain) == chains + 1
        @test iscolour(stripepixel(windowimage(s.g, win), 1, 1), GREEN)
    end
end

# api.md:483. SUBOPTIMAL [FI-3]: presented, the window marked, the next run makes a new generation.
@case "WIN-09" 7 :graphics :window begin
    # Not testable with faultdevice.jl: an Answer replaces the acquirenext call,
    # so it cannot return SUBOPTIMAL with an image the driver really acquired,
    # and a made-up image cannot be presented. Needs a fault that forwards the
    # call and rewrites its result code.
    @test_skip "WIN-09 needs a forwarding fault"
end

# internals.jl:1475-1486 (1519-1556 today) [FI-7]. A resize while a run is held in its acquire: the run returns, the next run has the new size, no plan compile.
@case "WIN-10" 7 :graphics :window begin
    fd = FaultDevice(testdevice())
    win = Window(fd, 64, 64; capacity = (128, 128))
    s = presenting(win, fd)
    run!(s.g); run!(s.g)
    gate = Channel{Nothing}(1)
    acquires = calls(fd, :acquirenext)
    inject!(fd, :acquirenext, Block(acquires + 1, gate))
    t = Threads.@spawn run!(s.g)
    @test timedwait(() -> calls(fd, :acquirenext) > acquires, 10) === :ok   # inside its acquire
    withtimeout(() -> resize!(win, 96, 80), 5)
    put!(gate, nothing)
    @test withtimeout(() -> fetch(t), 30) isa Mantle.Submitted
    _, d = counted(() -> run!(s.g), fd)
    @test d.plancompiles == 0
    @test size(windowimage(s.g, win)) == (96, 80)
end

# api.md:192, 485. A host function throws after the acquire, before the window pass: the image goes back to the window; the next run presents it without acquiring.
@case "WIN-11" 7 :graphics :window begin
    fd = FaultDevice(testdevice())
    win = Window(fd, 64, 64)
    fail = Ref(false)
    s = presenting(win, fd; before = g -> hostnode!(g, _ -> (fail[] && error("injected host failure"); nothing)))
    run!(s.g); run!(s.g)
    acquires = calls(fd, :acquirenext)
    fail[] = true
    @test_throws ErrorException run!(s.g)
    @test calls(fd, :acquirenext) == acquires + 1
    fail[] = false
    @test run!(s.g) isa Mantle.Submitted
    @test calls(fd, :acquirenext) == acquires + 1   # the kept image
    run!(s.g)
    @test calls(fd, :acquirenext) == acquires + 2
end

# internals.jl:1461 (1479-1481, 1515 today). An image kept after a throw, then a resize: the kept image goes with its generation; the next run acquires from the new one.
@case "WIN-12" 7 :graphics :window begin
    fd = FaultDevice(testdevice())
    win = Window(fd, 64, 64; capacity = (128, 128))
    fail = Ref(false)
    s = presenting(win, fd; before = g -> hostnode!(g, _ -> (fail[] && error("injected host failure"); nothing)))
    run!(s.g); run!(s.g)
    acquires = calls(fd, :acquirenext)
    fail[] = true
    @test_throws ErrorException run!(s.g)
    fail[] = false
    resize!(win, 96, 80)
    @test run!(s.g) isa Mantle.Submitted
    @test calls(fd, :acquirenext) == acquires + 2
    @test size(windowimage(s.g, win)) == (96, 80)
end

# recording-plan.md:1023-1029. 1 000 runs, every other one meeting OUT_OF_DATE (Again): every real acquire is presented; memory at its baseline.
@case "WIN-13" 7 :graphics :window :long begin
    onvulkan(testdevice()) do dev
        fd = FaultDevice(dev)
        win = Window(fd, 64, 64; vsync = false)
        s = presenting(win, fd)
        foreach(_ -> run!(s.g), 1:10)
        baseline = memory(fd)
        acquires = calls(fd, :acquirenext)
        results = map(1:1000) do i
            isodd(i) && inject!(fd, :acquirenext, AnswerOnce(calls(fd, :acquirenext) + 1, outofdate(dev)))
            run!(s.g)
        end
        @test all(r -> r isa Mantle.Submitted, results)
        @test calls(fd, :acquirenext) - acquires == 1000 + 500   # one real acquire per run, plus the 500 answered
        @test memory(fd) == baseline
    end
end

# api.md:184, recording-plan.md:1273-1278. Array(win) from another thread during frames: the image of one frame presented after the call.
@case "WIN-14" 7 :graphics :window begin
    dev = testdevice()
    win = Window(dev, 128, 64)
    s = presenting(win, dev; n = 32)
    for k in 1:10
        s.rects[1:1] = [stripe(k, 32)]
        run!(s.g)
    end
    shot = Threads.@spawn Array(win)
    k = 10
    deadline = time() + 30
    while !istaskdone(shot) && time() < deadline     # frame k draws stripe k
        k = min(k + 1, 32)
        s.rects[1:1] = [stripe(k, 32)]
        run!(s.g)
    end
    istaskdone(shot) || error("Array(win) did not return")
    A = fetch(shot)
    drawn = filter(j -> iscolour(stripepixel(A, j, 32), GREEN), 1:32)
    @test length(drawn) == 1
    @test all(>(10), drawn)
end

# api.md:184 (ambiguity 14). Array(win) while minimized, or after free! of the presenting graph, throws.
@case "WIN-15" 7 :graphics :window begin   # pending decision 4
    dev = testdevice()
    win = Window(dev, 64, 64)
    s = presenting(win, dev)
    run!(s.g)
    resize!(win, 0, 0)
    run!(s.g)
    @test thrownwithin(() -> Array(win), 10) isa ArgumentError
    other = Window(dev, 64, 64)
    s2 = presenting(other, dev)
    run!(s2.g)
    free!(s2.g)
    @test thrownwithin(() -> Array(other), 10) isa ArgumentError
end

# api.md:184 (ambiguity 14). Array(win) from a host function of the presenting graph throws instead of waiting for its own run.
@case "WIN-16" 7 :graphics :window begin   # pending decision 4
    dev = testdevice()
    win = Window(dev, 64, 64)
    s = presenting(win, dev; before = g -> hostnode!(g, _ -> (Array(win); nothing)))
    @test thrownwithin(() -> run!(s.g), 30) isa ArgumentError
end

# api.md:75. MantleArray(g, T, win) across resizes: its device length follows the window; no plan compile within the capacity.
@case "WIN-17" 7 :graphics :window begin
    dev = testdevice()
    win = Window(dev, 64, 64; capacity = (128, 128))
    out = MantleArray(dev, Int32, 1)
    s = presenting(win, dev; before = g -> dispatch!(g, win_lengthout!, (out, MantleArray(g, Float32, win)), 1))
    run!(s.g)
    @test Array(out) == Int32[64 * 64]
    for (w, h) in ((96, 80), (32, 128))
        resize!(win, w, h)
        _, d = counted(() -> run!(s.g))
        @test d.plancompiles == 0
        @test Array(out) == Int32[w * h]
    end
end

# recording-plan.md:1050-1054. A small window with a large capacity on a sparse device: the window part maps pages for the size, not the capacity.
@case "WIN-18" 7 :graphics :sparse :window begin
    dev = testdevice()
    capacity = (2048, 2048)
    capbytes = prod(capacity) * 4
    m0 = memory(dev).mapped
    s = layeredwindow(dev, capacity)
    run!(s.g)
    @test memory(dev).mapped - m0 < capbytes ÷ 4
    resize!(s.win, 1024, 1024)
    run!(s.g)
    grown = memory(dev).mapped - m0
    @test grown >= 1024 * 1024 * 4
    @test grown < capbytes
end

# recording-plan.md:1053-1054. Without sparse residency the capacity is mapped, and the window works.
@case "WIN-19" 7 :graphics :window begin
    dev = testdevice()
    if has(dev, :sparse)
        @test_skip "sparse device: WIN-18"
    else
        capacity = (1024, 1024)
        m0 = memory(dev).mapped
        s = layeredwindow(dev, capacity)
        run!(s.g)
        @test memory(dev).mapped - m0 >= prod(capacity) * 4
        @test iscolour(stripepixel(windowimage(s.g, s.win), 1, 1), GREEN)
    end
end

# recording-plan.md:993-1001. Steady state: nothing recorded per frame (per-image segments once per generation); one submission per frame, the pass writing the swapchain image (no copy).
@case "WIN-20" 7 :graphics :window begin
    onvulkan(testdevice()) do dev
        win = Window(dev, 64, 64)
        s = presenting(win, dev)
        foreach(_ -> run!(s.g), 1:8)                 # every swapchain image acquired once
        _, d = counted(() -> foreach(_ -> run!(s.g), 1:20))
        @test d.records == 0
        @test submitcount(d) == 20
    end
end

# internals.jl:1453-1455 (1483, 1508 today). createswapchain throws [FI-5]: run! throws, the window mark is restored; the next run makes the generation.
@case "WIN-21" 7 :graphics :window begin
    fd = FaultDevice(testdevice())
    win = Window(fd, 64, 64; capacity = (128, 128))
    s = presenting(win, fd)
    run!(s.g)
    chains = calls(fd, :createswapchain)
    inject!(fd, :createswapchain, Throw(chains + 1, ErrorException("injected swapchain failure")))
    resize!(win, 96, 80)
    @test_throws ErrorException run!(s.g)
    @test run!(s.g) isa Mantle.Submitted
    @test calls(fd, :createswapchain) == chains + 2
    @test size(windowimage(s.g, win)) == (96, 80)
end

# recording-plan.md:1019. g1 renders image I late; the presenting g2 reads I: g2 waits for g1 (namers).
@case "WIN-22" 7 :graphics :window begin
    dev = testdevice()
    shared = Image(dev, RGBA{N0f8}, 32, 32)
    x = MantleArray(dev, [opaque(RED)])
    src = MantleArray(dev, [opaque(RED)])
    full = MantleArray(dev, [stripe(1, 1)])
    g1 = Graph(dev)
    dispatch!(g1, win_slowcopy!, (x, src, Int32(calibratedspin(dev, 200))), 1)
    render!(p -> drawrects!(p, full, x), g1, shared => BACKGROUND)
    win = Window(dev, 64, 64)
    out = MantleArray(dev, RGBA{N0f8}, 32, 32)
    s2 = presenting(win, dev; before = g -> readimage!(g, shared, out))
    run!(g1); run!(s2.g)
    src[1:1] = [opaque(BLUE)]
    run!(g1)                                         # draws blue into shared about 200 ms from now
    run!(s2.g)
    @test all(px -> iscolour(px, BLUE), Array(out))
end

# recording-plan.md:1061. A window opened on the main thread; its graph runs on a worker thread.
@case "WIN-23" 7 :graphics :window begin
    dev = testdevice()
    win = Window(dev, 64, 64)                        # this task: the main thread
    s = presenting(win, dev)
    if Threads.nthreads() < 2
        @test_skip "needs a worker thread"
    else
        shots = Vector{Any}(nothing, Threads.nthreads())
        Threads.@threads :static for i in 1:Threads.nthreads()
            Threads.threadid() == 2 && (shots[i] = windowimage(s.g, win))
        end
        @test iscolour(stripepixel(something(shots...), 1, 1), GREEN)
    end
end

# internals.jl:1407-1413 (ambiguity 13). The presenting graph freed, or collected: another graph may then render into the window.
@case "WIN-24" 7 :graphics :window begin   # pending decision 4
    dev = testdevice()
    win = Window(dev, 64, 64)
    s = presenting(win, dev)
    run!(s.g)
    free!(s.g)
    s2 = presenting(win, dev; colour = BLUE)
    @test iscolour(stripepixel(windowimage(s2.g, win), 1, 1), BLUE)
    other = Window(dev, 64, 64)
    presentonce(other, dev)
    GC.gc(true); GC.gc(true)                         # the window holds its graph weakly
    s3 = presenting(other, dev; colour = BLUE)
    @test iscolour(stripepixel(windowimage(s3.g, other), 1, 1), BLUE)
end

# recording-plan.md:1046-1049 (open decision 10). 20 resizes, each retiring a swapchain: memory back at its baseline (validation runs with the suite's layers).
@case "WIN-25" 7 :graphics :window begin
    onvulkan(testdevice()) do dev
        win = Window(dev, 64, 64; capacity = (128, 128))
        s = presenting(win, dev)
        run!(s.g); run!(s.g)
        baseline = memory(dev)
        for i in 1:20
            resize!(win, 64 + 2i, 64 + i)
            run!(s.g); run!(s.g)
        end
        resize!(win, 64, 64)
        run!(s.g); run!(s.g)
        @test memory(dev) == baseline
    end
end

# internals.jl:1409-1414 (1463-1470 today). 100 resize events between two runs: one new generation, at the last size.
@case "WIN-26" 7 :graphics :window begin
    fd = FaultDevice(testdevice())
    win = Window(fd, 64, 64; capacity = (256, 256))
    s = presenting(win, fd)
    run!(s.g)
    chains = calls(fd, :createswapchain)
    foreach(i -> resize!(win, 64 + i, 64 + i ÷ 2), 1:100)
    _, d = counted(() -> run!(s.g), fd)
    @test calls(fd, :createswapchain) == chains + 1
    @test d.plancompiles == 0
    @test size(windowimage(s.g, win)) == (164, 114)
end

# api.md:190. Window-sized arrays are placed at the capacity: explicit (resizes up to it compile nothing, past it once) and default (the largest monitor's size).
@case "WIN-27" 7 :graphics :window begin
    dev = testdevice()
    out = MantleArray(dev, Int32, 1)
    sized(win) = g -> dispatch!(g, win_lengthout!, (out, MantleArray(g, Float32, win)), 1)
    win = Window(dev, 64, 64; capacity = (128, 96))
    s = presenting(win, dev; before = sized(win))
    run!(s.g)
    resize!(win, 128, 96)
    _, d = counted(() -> run!(s.g))
    @test d.plancompiles == 0
    @test Array(out) == Int32[128 * 96]
    resize!(win, 129, 96)
    _, d = counted(() -> run!(s.g))
    @test d.plancompiles == 1
    other = Window(dev, 64, 64)
    s2 = presenting(other, dev; before = sized(other))
    run!(s2.g)
    resize!(other, 640, 480)                         # within any monitor
    _, d = counted(() -> run!(s2.g))
    @test d.plancompiles == 0
    @test Array(out) == Int32[640 * 480]
end

# recording-plan.md:1002-1003. Metal: a new drawable per frame; each frame shows what it drew.
@case "WIN-28" 7 :graphics :metal :window begin
    dev = testdevice()
    win = Window(dev, 128, 64)
    s = presenting(win, dev; n = 32)
    for k in (3, 17, 30)
        s.rects[1:1] = [stripe(k, 32)]
        @test iscolour(stripepixel(windowimage(s.g, win), k, 32), GREEN)
    end
end

# internals.jl:688-698 (684-705 today; ambiguity 27). A build between two window passes: independent of the first, it moves before the present span; depending on it, run! throws.
@case "WIN-29" 7 :graphics :rt :window begin
    dev = testdevice()
    blas = BLAS(dev, MantleArray(dev, [Point3f(0, 0, 0), Point3f(1, 0, 0), Point3f(0, 1, 0)]),
                MantleArray(dev, [TriangleFace{UInt32}(1, 2, 3)]))
    green = MantleArray(dev, [opaque(GREEN)])
    left, right = MantleArray(dev, [stripe(1, 2)]), MantleArray(dev, [stripe(2, 2)])
    tlas = TLAS(dev)
    push!(tlas, blas, MantleArray(dev, [one(Mat4f)]))
    win = Window(dev, 64, 64)
    g = Graph(dev)
    render!(p -> drawrects!(p, left, green), g, win => BACKGROUND)
    build!(g, tlas)
    render!(p -> drawrects!(p, right, green), g, win => Keep)
    @test run!(g) isa Mantle.Submitted
    A = windowimage(g, win)
    @test iscolour(stripepixel(A, 1, 2), GREEN)
    @test iscolour(stripepixel(A, 2, 2), GREEN)
    # The build reads transforms a kernel writes from the depth the first window pass wrote.
    transforms = MantleArray(dev, [one(Mat4f)])
    tlas2 = TLAS(dev)
    push!(tlas2, blas, transforms)
    win2 = Window(dev, 64, 64)
    g2 = Graph(dev)
    depth = Image(g2, Float32, win2)
    render!(p -> drawrects!(p, left, green), g2, win2 => BACKGROUND, depth => Clear(1f0))
    dispatch!(g2, win_depthscale!, (transforms, depth), 1)
    build!(g2, tlas2)
    render!(p -> drawrects!(p, right, green), g2, win2 => Keep)
    @test_throws ArgumentError run!(g2)              # a compile error, raised by the first run!
end
