# Needs a display and a GPU. Skipped when the worker has no session, which is the
# default for bt_julia_eval: see docs, DISPLAY/XAUTHORITY are not inherited.
#
# ── This file BLOCKS on some X setups, and the guard below cannot see it ──
#
# Measured 2026-08-27 on DISPLAY=:1 with a live X server: `GLFW.Init()` succeeds,
# the guard therefore passes, and the process then sits at 100% CPU indefinitely
# inside window creation. `timeout -s TERM` after 600 s gives:
#
#     _glfwCreateWindowX11   libglfw.so
#     glfwCreateWindow       libglfw.so
#     CreateWindow           GLFW/src/glfw3.jl:600
#     VulkanWindow           Mantle/src/vulkan/graphics/window.jl:78
#     Window                 Mantle/src/vulkan/graph.jl:140
#
# Nothing above that frame is Mantle's, and no test output is produced before it,
# so a suite that reaches this file never finishes. `DISPLAY` being set is not
# the same question as "a window can be created here" — an X server with no
# window manager answers the first and hangs the second — and there is no way to
# ask the second without risking the hang.
#
# Deliberately NOT auto-skipped. A skip that hides a hang is how a test file rots
# unnoticed, and this one had already spent the runtime move unable to run at all
# (it names `Rasterizer`, whose export was left behind in `Lava.jl`, so it
# errored on its first pipeline). Run it directly, on a session with a window
# manager, and give it a timeout:
#
#     timeout 900 julia --project -e 'using Mantle, Test; include("test/test_window.jl")'
using Mantle, Test
# Here rather than in the branch below, which is one top-level expression: the
# `@kernel` in it is expanded before any of it runs, so a `using` inside cannot
# be what binds the macro.
using KernelAbstractions
# Same reason as above: a `@kernel` body using `Atomix.@atomic` is expanded with
# the rest of the branch, so a `using` inside it cannot be what binds the name.
using Atomix
const M = Mantle

# The probe has to be at top level: `@eval using GLFW` followed by `GLFW.Init()`
# inside one function body cannot see the new binding (world age), so it always
# reported "no display" even when there was one.
# Validation is deliberately NOT enabled here. Synchronization validation costs
# 212 ms/frame against 0.16 ms without it, a factor of 1300, which turns this
# file from seconds into tens of minutes. The barrier check lives in
# bench/validate.jl and is run on purpose.

if isempty(get(ENV, "DISPLAY", ""))
    @info "DISPLAY unset; skipping window tests"
else
    using GLFW, Lava, GeometryBasics, LinearAlgebra, ColorTypes
    GLFW.Init() || error("DISPLAY=$(ENV["DISPLAY"]) is set but GLFW.Init failed")
    include(joinpath(@__DIR__, "..", "bench", "two_scatters.jl"))

    # Reads along rows and writes down columns, which is the shape of anything
    # that shades a g-buffer, and the one the workgroup size matters for.
    @kernel function transpose_scale!(dst, @Const(src), w::Int32, h::Int32, a::Float32)
        ix, iy = @index(Global, NTuple)
        @inbounds dst[(ix - Int32(1)) * h + iy] = src[(iy - Int32(1)) * w + ix] * a
    end

    # Two of these chained is the smallest thing that can tell whether the
    # launches inside one pass are ordered against each other.
    @kernel function bump!(dst, @Const(src), k::Float32)
        i = @index(Global)
        @inbounds dst[i] = src[i] + k
    end

    # A fullscreen pass: the triangle comes from the vertex index, so the vertex
    # stage takes no arguments and the fragment stage is the one with a buffer.
    function fullscreen_vertex()
        v = vertex_index()
        p = v == Int32(1) ? Vec4f(-1, -1, 0, 1) :
            v == Int32(2) ? Vec4f(3, -1, 0, 1) : Vec4f(-1, 3, 0, 1)
        (position = p,)
    end
    function ramp_fragment(inputs, src, w::Int32, h::Int32)
        x = unsafe_trunc(Int32, frag_coord_x())
        y = unsafe_trunc(Int32, frag_coord_y())
        @inbounds v = src[y * w + x + Int32(1)]
        Vec4f(v, 0.25f0, 1f0 - v, 1f0)
    end
    RAMP = Rasterizer(vertex = fullscreen_vertex, fragment = ramp_fragment,
                      varyings = NamedTuple(), topology = TriangleList(),
                      blend = Opaque(), cull = NoCull(), depth = DepthOff())

    # Quads stacked up the image, six vertices each, so how many are drawn is
    # readable off the picture as how much of it is lit.
    function band_vertex(nb::Int32)
        v = vertex_index() - Int32(1)
        band = v ÷ Int32(6)
        c = v - band * Int32(6)
        k = c == Int32(0) ? Int32(0) : c == Int32(1) ? Int32(1) : c == Int32(2) ? Int32(2) :
            c == Int32(3) ? Int32(0) : c == Int32(4) ? Int32(2) : Int32(3)
        fx = (k == Int32(1) || k == Int32(2)) ? 1f0 : 0f0
        fy = (k == Int32(2) || k == Int32(3)) ? 1f0 : 0f0
        y0 = -1f0 + 2f0 * Float32(band) / Float32(nb)
        y1 = -1f0 + 2f0 * Float32(band + Int32(1)) / Float32(nb)
        (position = Vec4f(-1f0 + 2f0 * fx, y0 + (y1 - y0) * fy, 0.5f0, 1f0),
         color = Vec4f(1, 1, 1, 1))
    end
    band_fragment(inputs) = inputs.color
    BANDS = Rasterizer(vertex = band_vertex, fragment = band_fragment,
                       varyings = (color = Vec4f,), topology = TriangleList(),
                       blend = Opaque(), cull = NoCull(), depth = DepthOff())

    @kernel function set_draw!(cmds, k::UInt32)
        @inbounds cmds[1] = DrawIndirectCommand(UInt32(6) * k, UInt32(1), UInt32(0), UInt32(0))
    end

    @kernel function frombytes!(dst, @Const(src))
        i = @index(Global)
        @inbounds dst[i] = Float32(src[i] & 0x000000ff)
    end
    @kernel function scaleby!(dst, @Const(src), a::Float32)
        i = @index(Global)
        @inbounds dst[i] = src[i] * a
    end

    """The memory barrier a pass emits, as (src stage, dst stage), or nothing."""
    function passmasks(pp)
        pp.barrier === nothing && return nothing
        v = pp.barrier.vks
        v.memoryBarrierCount == 0 && return nothing
        m = unsafe_load(v.pMemoryBarriers, 1)
        (Mantle.VK.PipelineStageFlag2(m.srcStageMask),
         Mantle.VK.PipelineStageFlag2(m.dstStageMask))
    end

    # A quad at a given depth, and a fragment that writes no attachment, which is
    # what a depth-only pass has to be able to say.
    function depthonly_vertex(z::Float32)
        v = vertex_index() - Int32(1)
        k = v == Int32(0) ? Int32(0) : v == Int32(1) ? Int32(1) : v == Int32(2) ? Int32(2) :
            v == Int32(3) ? Int32(0) : v == Int32(4) ? Int32(2) : Int32(3)
        fx = (k == Int32(1) || k == Int32(2)) ? 1f0 : 0f0
        fy = (k == Int32(2) || k == Int32(3)) ? 1f0 : 0f0
        # the left half only, so the right half keeps the clear value
        (position = Vec4f(-1f0 + fx, -1f0 + 2f0 * fy, z, 1f0),)
    end
    depthonly_fragment(inputs) = nothing
    DEPTHONLY = Rasterizer(vertex = depthonly_vertex, fragment = depthonly_fragment,
                           varyings = NamedTuple(), topology = TriangleList(),
                           blend = Opaque(), cull = NoCull(), depth = DepthLess())

    @testset "two scatters share one pipeline" begin
        dev = M.Device(M.VulkanAPI())
        win = M.Window(W, H; title = "mantle test")
        s = build(dev, win, 50_000, 20_000)
        a, b = s.a, s.b

        # A has a vector colour and a scalar size; B has the opposite. If the
        # binding types differed they would compile two shaders.
        @test typeof(a) !== typeof(b)
        @test M.npipelines(s.plan) == 1

        for _ in 1:30
            M.run!(s.plan)
        end
        KernelAbstractions.synchronize(M.backend(dev))
        img = M.screenshot(win)
        close(win)

        bg = img[1, 1]
        @test count(p -> p != bg, img) > length(img) ÷ 100   # something was drawn
    end

    @testset "the advect pass actually moves the points" begin
        # The camera orbiting is not the simulation running. An earlier version of
        # this demo had no compute pass at all: the cloud rotated, looked alive,
        # and every position was the one it was uploaded with.
        dev = M.Device(M.VulkanAPI())
        win = M.Window(W, H; title = "advect")
        s = Base.invokelatest(build, dev, win, 20_000, 10_000)

        before = copy(Array(s.a.positions))
        for _ in 1:30
            s.dt[] = 1f0 / 60
            M.run!(s.plan)
        end
        KernelAbstractions.synchronize(M.backend(dev))
        after = Array(s.a.positions)
        close(win)

        moved = [norm(after[i] - before[i]) for i in eachindex(before)]
        @test count(<(1e-6), moved) == 0            # none of them stood still
        @test sum(moved) / length(moved) > 0.05     # and they went somewhere
        @test maximum(norm, after) <= 1.6f0 + 1e-4  # the shell reflection holds

        # The render pass reads what the compute pass wrote, so it must wait for
        # it. Nothing declares that: `use(..., write)` is Storage and `Attribute`
        # is Vertices, and the barrier falls out of the pair.
        adv, scat = s.plan.passes
        @test any(t -> Mantle.writes(t.from), scat.pre)

        # And the first pass of a frame is not free either. A plan is replayed, so
        # "first" is only first *within* a frame: the previous frame's draw read
        # these positions and this frame's advect overwrites them. This assertion
        # used to read `isempty(adv.pre)`, which was the old belief that a frame
        # starts from nothing — true only while every frame was flushed.
        @test !isempty(adv.pre)
        @test any(t -> Vertices in t.waits, adv.pre)
    end

    # A compute-only graph, for the two things that are only visible without a
    # surface: nothing presents, so nothing submits on its own.
    function advect_plan(dev, n)
        g = M.Graph(dev)
        pos = M.Buffer(dev, cloud(n))
        vel = M.Buffer(dev, drift(n))
        M.compute!(g, "advect") do p
            x = M.use(p, pos; read = true, write = true)
            v = M.use(p, vel; read = true, write = true)
            M.dispatch!(p, advect!, (x, v, Ref(1f0 / 60), 1.4f0), n)
        end
        (plan = M.Plan(g), pos = pos)
    end

    @testset "a headless plan runs past its argument slots" begin
        # A plan with a surface submits every frame inside `present_frame!`. One
        # without submits when something asks it to, so `ARG_SLOTS` frames all
        # land in the same recording batch — and the frame that first reuses a
        # slot then waited for a timeline value the queue had never been given.
        # `vkWaitSemaphores` is a foreign call, so that wait cannot be
        # interrupted, profiled, or garbage collected: the process is simply
        # gone. Found by running a compute-only graph 30 times.
        ext = Mantle
        dev = M.Device(M.VulkanAPI())
        s = Base.invokelatest(advect_plan, dev, 20_000)
        before = copy(Array(M.storage(s.pos)))
        for _ in 1:(3 * ext.ARG_SLOTS)
            M.run!(s.plan)
        end
        KernelAbstractions.synchronize(M.backend(dev))

        # Nothing a later frame could wait for is still sitting unsubmitted.
        @test all(<=(dev.bq.next_timeline), s.plan.args.signal)
        after = Array(M.storage(s.pos))
        moved = [norm(after[i] - before[i]) for i in eachindex(before)]
        @test count(<(1e-6), moved) == 0
        @test maximum(norm, after) <= 1.4f0 + 1e-4
    end

    @testset "a frame never compiles a kernel" begin
        # Recording has to be atomic against the Julia scheduler, because a frame
        # holds an acquired swapchain image from `acquire_next_image!` until
        # `present_frame!`. Compiling breaks that: it shells out to `spirv-opt`,
        # and waiting on a subprocess is a task switch.
        #
        # It used to compile once per kernel after *any* method definition
        # anywhere in the session, because the dispatch went through
        # KernelAbstractions per frame and its launch plan is keyed on
        # `Base.get_world_counter()`. A plan now resolves its dispatches when it
        # is built, the same as its draws.
        #
        # The assertion is on yielding rather than on time: a compile that took
        # no measurable time would still be a compile inside a frame, and a
        # machine under load would still not yield here.
        dev = M.Device(M.VulkanAPI())
        s = Base.invokelatest(advect_plan, dev, 4_096)
        M.run!(s.plan)                       # the frame that may compile
        KernelAbstractions.synchronize(M.backend(dev))

        @eval $(gensym("bump"))() = 1         # what every eval in a session does
        stop = Ref(false)
        ticks = Ref(0)
        canary = @async while !stop[]
            ticks[] += 1
            yield()
        end
        yield()

        # `invokelatest`, because a testset is one top-level expression: the
        # global counter moved above, but this task's world age would not until
        # the expression ended, and the compiler cache reads the task's. Without
        # it the frame runs in the world it was compiled in and the test cannot
        # fail.
        before = ticks[]
        Base.invokelatest(M.run!, s.plan)
        yielded = ticks[] - before
        stop[] = true
        wait(canary)
        @test yielded == 0
    end

    @testset "transients are placed, not allocated separately" begin
        # A chain where stage k reads t[k] and writes t[k+1]: only adjacent
        # transients are ever live together, so every other one can share bytes.
        # This is the allocator from the headless tests, driving real memory.
        include(joinpath(@__DIR__, "..", "bench", "chain.jl"))
        dev = M.Device(M.VulkanAPI())
        # A Lava window, not a Mantle one: this test drives acquire/blit/present
        # by hand because the chain ends in a compute pass and Mantle has no blit.
        # Anything that renders through a plan uses `M.Window`.
        win = VulkanWindow(W, H; title = "chain", vsync = false)
        s = Base.invokelatest(build_chain, dev, win, 20_000)

        peak = M.peakbytes(s.plan)
        naive = Mantle.naivebytes(s.plan)
        @test peak < naive                       # aliasing happened at all
        @test peak <= naive ÷ 2 + 2^20           # about two buffers, not five

        for _ in 1:20
            acquire_next_image!(win)
            M.run!(s.plan)
            copy_framebuffer!(M.storage(s.raw), s.fb)
            blit!(dev.bq, WindowTarget(win), M.storage(s.out))
            present_frame!(dev.bq, win)
        end
        KernelAbstractions.synchronize(M.backend(dev))
        img = readback_window(win)
        close(win)
        bg = img[1, 1]
        @test count(p -> p != bg, img) > length(img) ÷ 100
    end

    @testset "no barrier is dropped because another resource's covers it" begin
        # Eight independent chains interleaved: stage 1 of every chain is a first
        # touch and needs nothing, and one barrier before stage 2 covers all of
        # them. Emitting per pass without checking coverage gives one barrier per
        # chain per stage, which is correct but pointless.
        #
        # Measured with aliasing *off*. With it on, every pass begins a transient
        # that took over another's bytes, and the handover is appended after the
        # coverage test precisely because nothing can cover it — so it is 48 of
        # 48 whether coalescing runs or not, and the number measures nothing.
        # This used to be asserted with aliasing on and passed by a margin of one
        # or two; that margin was the aliasing barrier naming every stage and so
        # making the next few look covered, not coalescing doing its job. Once
        # the handover was derived from what the old tenant actually did, the
        # margin went to zero and the real figure came out: a third.
        include(joinpath(@__DIR__, "..", "bench", "independent.jl"))
        dev = M.Device(M.VulkanAPI())
        emitted(pl) = count(pp -> !isempty(pp.pre), pl.passes)
        tight = Base.invokelatest(build_interleaved, dev, 1 << 12, 8, 6; alias = false)
        loose = Base.invokelatest(build_interleaved, dev, 1 << 12, 8, 6;
                                  alias = false, coalesce = false)
        @test emitted(loose.plan) == length(loose.plan.passes)   # one per pass
        # This used to assert that a third of them went away, which was true and
        # was the bug: a barrier scoped to one buffer orders that buffer and
        # nothing else, so "an earlier barrier already covers this" is only ever
        # sound about the *same* resource — and the per-resource walk emits only
        # where a resource's own state changes, so there is nothing left to cover.
        # Interleaved chains are the case that shows it: eight chains over
        # disjoint buffers, where the removals were all cross-chain.
        @test emitted(tight.plan) == emitted(loose.plan)

        s = Base.invokelatest(build_interleaved, dev, 1 << 12, 8, 6)
        @test emitted(s.plan) == length(s.plan.passes)

        # Both strategies must produce the same answer, or the elision is wrong.
        # SEEDED before each run, and that is not incidental. Every chain's first
        # buffer is a transient no pass writes — it is only ever a `src` — and
        # with aliasing on it shares bytes with four transients that ARE written
        # (measured: bufs[1][1] at [16384, 32768), four writers on exactly those
        # bytes). So a run leaves its own output in the next run's input, and two
        # runs of this plan disagree under ANY barrier strategy: measured
        # 6.015 -> 12.066 -> 17.137 with the strategy held fixed. Comparing two
        # unseeded runs therefore tested nothing about barriers, which is what it
        # was there to do.
        #
        # Seeding also keeps the elision under test: stage 1's read of the seed
        # and the aliased write that later takes those bytes are exactly the
        # hazard a dropped barrier would let race.
        seedchains!() = for row in s.bufs
            copyto!(M.storage(row[1]), zeros(Float32, length(row[1])))
        end

        seedchains!()
        M.run!(s.plan; barriers = :backend)
        KernelAbstractions.synchronize(M.backend(dev))
        reference = Array(M.storage(s.bufs[1][end]))

        seedchains!()
        M.run!(s.plan; barriers = :derived)
        KernelAbstractions.synchronize(M.backend(dev))
        @test Array(M.storage(s.bufs[1][end])) == reference
    end

    @kernel function clear_and_tally!(c, tally)
        c[1] = UInt32(0)
        Atomix.@atomic tally[1] += UInt32(1)
    end
    @kernel function bump_counter!(c)
        i = @index(Global)
        Atomix.@atomic c[1] += UInt32(1)
    end

    @testset "a frame that splits its command buffer still runs every pass" begin
        # `maybe_split_cb!` ends the current command buffer mid-frame and starts a
        # fresh one, leaving the first half in `sealed_cmd_bufs`. `present_frame!`
        # submitted only `batch.cmd_buf`, so for a windowed plan everything before
        # the split was recorded, never submitted, and silently did not run — no
        # validation error, because nothing about it is invalid.
        #
        # It showed up once every `cb_split_threshold` dispatches: a frame that did
        # nothing. For this graph — clear a counter in the first pass, accumulate
        # into it later — a lost clear means the counter never restarts, and the
        # showcase's cull then indexed past the end of its visible list into the
        # buffers behind it.
        #
        # The threshold is dropped so the split happens every frame instead of
        # every few thousand; `tally` is monotonic on purpose, because the counter
        # itself cannot detect this. A skipped frame leaves a cleared-then-refilled
        # counter at exactly its previous correct value.
        dev = M.Device(M.VulkanAPI())
        dev.bq.cb_split_threshold = 3          # every frame, several times over
        win = M.Window(64, 64; title = "split", vsync = false)
        g = M.Graph(dev)
        cnt, tally = M.Buffer(dev, UInt32[0]), M.Buffer(dev, UInt32[0])
        M.compute!(g, "clear") do p
            M.dispatch!(p, clear_and_tally!, (M.use(p, cnt; write = true),
                                              M.use(p, tally; read = true, write = true)), 1)
        end
        M.compute!(g, "accumulate") do p
            M.dispatch!(p, bump_counter!, (M.use(p, cnt; read = true, write = true),), 512)
        end
        screen = M.Surface(g, win)
        M.render!(g, "present", screen => M.Clear((0f0, 0f0, 0f0, 1f0))) do p
        end
        plan = M.Plan(g)
        frames = 40
        for _ in 1:frames
            M.run!(plan)
            Mantle.flush!(dev.bq, dev.ctx.device)
        end
        @test Int(Array(tally)[1]) == frames        # every frame ran its first pass
        @test Int(Array(cnt)[1]) == 512             # and the counter restarted each time
        close(win)
    end

    @kernel function fill_span!(dst, base::Int32, v::Float32)
        i = @index(Global)
        @inbounds dst[base + i] = v
    end

    @testset "disjoint slices of one buffer are not ordered against each other" begin
        # Ported from Vulkan-ValidationLayers,
        # tests/unit/sync_val_positive.cpp: PositiveSyncVal.BufferCopyNonOverlappedRegions —
        # two writes into non-overlapping regions of one buffer, asserted there to
        # raise no hazard. Tracking a buffer whole says it does, and the barrier
        # between them orders memory neither pass touches.
        #
        # `range` hands the slice its own resource id, so the existing
        # per-resource walk answers this without knowing anything about ranges.
        # The count is not the test — each pass still has a real loop-carried WAW
        # against the previous frame, so both spellings emit two transitions. What
        # differs is *whose*: one resource means the second waits on the first.
        dev = M.Device(M.VulkanAPI())
        n = 256
        build(ranged) = begin
            g = M.Graph(dev)
            b = M.Buffer(dev, zeros(Float32, n))
            M.compute!(g, "lower") do p
                w = ranged ? M.use(p, b; write = true, range = 1:128) :
                             M.use(p, b; write = true)
                M.dispatch!(p, fill_span!, (w, Int32(0), 1f0), 128)
            end
            M.compute!(g, "upper") do p
                w = ranged ? M.use(p, b; write = true, range = 129:256) :
                             M.use(p, b; write = true)
                M.dispatch!(p, fill_span!, (w, Int32(128), 2f0), 128)
            end
            (; b, plan = M.Plan(g))
        end
        touched(pl) = unique(t.resource for pp in pl.passes for t in pp.pre)

        whole = build(false)
        @test length(touched(whole.plan)) == 1      # one resource: ordered, needlessly

        sliced = build(true)
        @test length(touched(sliced.plan)) == 2     # two: independent

        # And it still computes the right thing, which is the point of not
        # ordering them: both halves land.
        M.run!(sliced.plan)
        KernelAbstractions.synchronize(M.backend(dev))
        got = Array(sliced.b)
        @test all(==(1f0), got[1:128]) && all(==(2f0), got[129:256])

        # Same file, PositiveSyncVal.WriteAndReadNonOverlappedUniformBufferRegions:
        # a write to one region and a *read* of another. Write-after-write is not
        # the only pair that matters — the wait set is built differently for a
        # read (the one previous state) than for a write (every outstanding
        # reader), so RAW and WAR go down a different path than the case above.
        #
        # Counting resources does not say it here: a slice this graph only ever
        # reads is seeded from its own last use, so it needs no transition at all
        # and never appears in the emitted set. What distinguishes the two
        # spellings is whether the *reader* waits — whole-buffer gives it a
        # WriteOnly -> ReadOnly against the writer, and disjoint slices give it
        # nothing to wait for.
        readerwaits(ranged) = begin
            g = M.Graph(dev)
            b = M.Buffer(dev, zeros(Float32, n))
            M.compute!(g, "write low") do p
                w = ranged ? M.use(p, b; write = true, range = 1:128) :
                             M.use(p, b; write = true)
                M.dispatch!(p, fill_span!, (w, Int32(0), 3f0), 128)
            end
            M.compute!(g, "read high") do p
                r = ranged ? M.use(p, b; read = true, range = 129:256) :
                             M.use(p, b; read = true)
                M.dispatch!(p, fill_span!, (r, Int32(128), 4f0), 128)
            end
            length(M.Plan(g).passes[2].pre)
        end
        @test readerwaits(false) == 1      # whole buffer: RAW against the writer
        @test readerwaits(true) == 0       # disjoint slices: nothing to wait for

        # The control, and the failure mode that matters: naming a range must not
        # make everything independent. Identical ranges are the same resource and
        # stay ordered — if this ever reads 2, ranges have stopped meaning
        # anything and every test above passes for the wrong reason.
        gs = M.Graph(dev)
        bs = M.Buffer(dev, zeros(Float32, n))
        for nm in ("first", "second")
            M.compute!(gs, nm) do p
                M.dispatch!(p, fill_span!, (M.use(p, bs; write = true, range = 1:128),
                                            Int32(0), 5f0), 128)
            end
        end
        @test length(touched(M.Plan(gs))) == 1

        # Partial overlap. `1:128` and `64:200` share `64:128`, so the two passes
        # do have a hazard and must stay ordered — while a third pass on `201:256`
        # touches none of it and must not be dragged in. This is the case the
        # atomic-segment partition exists for: the cuts fall at 1, 64, 129, 201.
        gp = M.Graph(dev)
        bp = M.Buffer(dev, zeros(Float32, n))
        M.compute!(gp, "low") do p
            M.dispatch!(p, fill_span!, (M.use(p, bp; write = true, range = 1:128),
                                        Int32(0), 6f0), 128)
        end
        M.compute!(gp, "mid") do p
            M.dispatch!(p, fill_span!, (M.use(p, bp; write = true, range = 64:200),
                                        Int32(63), 7f0), 137)
        end
        M.compute!(gp, "tail") do p
            M.dispatch!(p, fill_span!, (M.use(p, bp; write = true, range = 201:256),
                                        Int32(200), 8f0), 56)
        end
        pp = M.Plan(gp)
        # `mid` overlaps `low`, so it waits; `tail` overlaps neither and does not.
        @test !isempty(pp.passes[2].pre)
        lowseg = Set(t.resource for t in pp.passes[1].pre)
        midseg = Set(t.resource for t in pp.passes[2].pre)
        tailseg = Set(t.resource for t in pp.passes[3].pre)
        @test !isempty(intersect(lowseg, midseg))    # they share the 64:128 segment
        @test isempty(intersect(lowseg, tailseg))    # and share nothing with the tail

        # Slicing a buffer partitions it, and a later whole-buffer usage then
        # stands for every segment. Those are contiguous and ask for the same
        # thing, so they lower to one barrier rather than one per segment —
        # otherwise naming a range anywhere makes every whole use of that buffer
        # cost a barrier per cut.
        gm = M.Graph(dev)
        bm = M.Buffer(dev, zeros(Float32, n))
        for (k, r) in enumerate((1:64, 65:128, 129:192, 193:256))
            M.compute!(gm, "part $k") do p
                M.dispatch!(p, fill_span!, (M.use(p, bm; write = true, range = r),
                                            Int32(first(r) - 1), Float32(k)), length(r))
            end
        end
        M.compute!(gm, "whole") do p
            M.dispatch!(p, fill_span!, (M.use(p, bm; read = true, write = true),
                                        Int32(0), 9f0), n)
        end
        pm = M.Plan(gm)
        whole = pm.passes[5]
        @test length(whole.pre) == 4                                   # four segments
        @test Int(whole.barrier.vks.bufferMemoryBarrierCount) == 1      # merged to one

        # Out of bounds is still a mistake worth naming.
        g2 = M.Graph(dev)
        b2 = M.Buffer(dev, zeros(Float32, n))
        @test_throws ArgumentError M.compute!(g2, "past the end") do p
            M.use(p, b2; write = true, range = 200:400)
        end

        # A sliced *transient* still gets its handover. The handover is derived
        # per transient and looked its usage up by that transient's own id, so a
        # pass naming only a slice answered `nothing` and the caller skipped the
        # barrier — losing the one hazard no per-resource sequence can see,
        # silently. `usage_of` counts a slice as naming its parent, and the wait
        # set is gathered across every id the old tenant is tracked under.
        g3 = M.Graph(dev)
        a3 = M.Transient.Buffer(g3, Float32, n)
        b3 = M.Transient.Buffer(g3, Float32, n)
        M.compute!(g3, "fill a") do p
            M.dispatch!(p, fill_span!, (M.use(p, a3; write = true), Int32(0), 1f0), n)
        end
        M.compute!(g3, "read a") do p
            M.dispatch!(p, fill_span!, (M.use(p, a3; read = true), Int32(0), 1f0), n)
        end
        M.compute!(g3, "slice of b") do p
            M.dispatch!(p, fill_span!, (M.use(p, b3; write = true, range = 1:128),
                                        Int32(0), 2f0), 128)
        end
        out3 = M.Buffer(dev, zeros(Float32, n))
        M.compute!(g3, "drain") do p
            M.dispatch!(p, fill_span!, (M.use(p, out3; write = true),
                                        Int32(0), 3f0), n)
            M.use(p, b3; read = true)
        end
        p3 = M.Plan(g3; alias = true)
        # `b` takes over `a`'s bytes, and the pass that first writes it names only
        # a slice. The handover is the transition with no resource of its own.
        @test any(t -> t.resource == 0, vcat((pp.pre for pp in p3.passes)...))
    end

    @testset "the emitted set is the per-resource hazard set, exactly" begin
        # What a *scoped* barrier set has to be, written out by hand rather than
        # counted. A barrier on one buffer cannot stand in for one on another, so
        # nothing here is droppable: the required set is one entry per genuine
        # state change on each resource's own usage sequence, and coalescing has
        # nothing left to remove.
        #
        # Two chains of three stages over disjoint buffers, interleaved. Per
        # chain, with `b[k]` the buffer stage k reads:
        #   b[1]  read at s1 only          — the replayed state is already
        #                                    ReadOnly, so no transition at all
        #   b[2]  written s1, read s2      — WAR at s1, RAW at s2
        #   b[3]  written s2, read s3      — WAR at s2, RAW at s3
        #   b[4]  written s3, never read   — WAW at s3 against the last frame
        # Five per chain, ten in total, landing 1/2/2 across the three passes.
        #
        # Asserted as two subset checks rather than one equality, because the two
        # directions are different bugs and want different messages: a missing
        # entry is a race, a spurious one is serialisation nobody asked for. The
        # pair is also what makes the test unsatisfiable by a degenerate answer —
        # emitting nothing fails the first, one unconditional barrier per pass
        # fails the second.
        include(joinpath(@__DIR__, "..", "bench", "independent.jl"))
        dev = M.Device(M.VulkanAPI())
        SR = M.Storage{M.BufferKind,M.ReadOnly}
        SW = M.Storage{M.BufferKind,M.WriteOnly}
        chains, stages = 2, 3
        s = Base.invokelatest(build_interleaved, dev, 1 << 10, chains, stages; alias = false)

        required = Set()
        for c in 1:chains, k in 1:stages
            k > 1 && push!(required, ("c$c s$k", s.g.ids[s.bufs[c][k]], SW, SR, Set([SW])))
            from = k == stages ? SW : SR      # the tail buffer is written and never read
            push!(required, ("c$c s$k", s.g.ids[s.bufs[c][k + 1]], from, SW, Set([from])))
        end
        @test length(required) == 10

        emitted(pl) = Set((pp.pass.name, t.resource, t.from, t.to, Set(t.waits))
                          for pp in pl.passes for t in pp.pre)

        # The per-resource walk itself, with no coalescing on top, is the oracle's
        # own check: if this disagrees the hand derivation above is wrong, not the
        # elision.
        raw = Base.invokelatest(build_interleaved, dev, 1 << 10, chains, stages;
                                alias = false, coalesce = false)
        @test emitted(raw.plan) == required

        got = emitted(s.plan)
        # Too few. Coalescing drops five of the ten, among them chain 2's RAW on
        # its own `b[2]` — written at `c2 s1`, read at `c2 s2`, nothing touching
        # that buffer in between. It is dropped because a barrier emitted for
        # chain 1's buffers is global and happens to stand in the way. That is
        # not a local barrier set; scope the barriers and it is a race. Same
        # shape as the showcase losing `sun cull`'s barrier.
        @test issubset(required, got)
        # Too many. Nothing spurious is emitted today, and must not start being.
        @test issubset(got, required)
    end

    @testset "the hazard set is lowered to scoped barriers, not one catch-all" begin
        # Everything above asserts the *derived* set. Nothing in it looks at what
        # is handed to Vulkan, and the two can disagree: a pass could carry ten
        # correct transitions and still lower them to one `VkMemoryBarrier2` with
        # the masks ORed together, which orders all memory and is exactly the
        # thing being removed. The derivation would look perfect and the barrier
        # would still be a global one.
        #
        # So this reads the emitted `VkDependencyInfo`: one buffer barrier per
        # transition, scoped to that buffer's own range, and no global memory
        # barrier at all. A handover names two resources and no single buffer, so
        # it stays global — hence aliasing off here, and its own test elsewhere.
        include(joinpath(@__DIR__, "..", "bench", "independent.jl"))
        dev = M.Device(M.VulkanAPI())
        s = Base.invokelatest(build_interleaved, dev, 1 << 10, 2, 3; alias = false)
        nbuf = nmem = ntrans = 0
        for pp in s.plan.passes
            ntrans += length(pp.pre)
            pp.barrier === nothing && continue
            nbuf += Int(pp.barrier.vks.bufferMemoryBarrierCount)
            nmem += Int(pp.barrier.vks.memoryBarrierCount)
        end
        @test ntrans == 10                # the hand-derived set, above
        @test nbuf == ntrans              # each one lowered, scoped to its buffer
        @test nmem == 0                   # and nothing ordering all of memory
    end

    @testset "the local hazard set holds over a corpus, not one topology" begin
        # One hand-derived topology says the shape is right; it does not say the
        # derivation is right in general. This drives the same equality from an
        # oracle built out of the rules — `needs_transition` plus the wait-set
        # rule — over random read/write/readwrite usages, where WAR, WAW, RAW and
        # read-after-read all occur without any liveness constraint to arrange.
        #
        # Persistent buffers and no aliasing on purpose: an alias handover is a
        # hazard between two resources that never mention each other, so it comes
        # from the placer and not from any usage sequence. It has its own test.
        include(joinpath(@__DIR__, "..", "bench", "stress.jl"))
        dev = M.Device(M.VulkanAPI())
        nbufs, npasses, seeds = 5, 12, 1:20

        # Swept over how often a usage names a range, because slicing is the case
        # the hand-written tests above cover by example only. The generator draws
        # both bounds from one small set, so identical, disjoint *and* partially
        # overlapping slices of a buffer all occur — and the oracle derives the
        # atomic partition itself rather than reading it off the compiler. Both
        # sides are keyed by `(parent, span)`, so agreeing on ids is not what
        # makes them agree.
        walk_is_local = 0; spurious = 0; missing_total = 0; required_total = 0
        cases = 0
        for sl in (0.0, 0.5), s in seeds
            mk(co) = usage_graph(dev, MersenneTwister(s),
                                 [M.Buffer(dev, fill(Float32(i), 256)) for i in 1:nbufs],
                                 npasses; coalesce = co, slicing = sl)
            raw = mk(false)
            oracle = local_hazards(raw.g, raw.plan)
            # The compiler's uncoalesced walk must BE the local set. If this
            # fails the oracle and the walk disagree and neither number below
            # means anything, so it is checked first and per case.
            normalized(raw.g, raw.plan) == oracle && (walk_is_local += 1)
            tight = mk(true)
            got = normalized(tight.g, tight.plan)
            required_total += length(oracle)
            missing_total += length(setdiff(oracle, got))
            spurious += length(setdiff(got, oracle))
            cases += 1
        end
        @test walk_is_local == cases
        # Too many: never emit a barrier the hazard set does not call for.
        @test spurious == 0
        # Too few: every local hazard must have its own barrier. Coalescing used
        # to drop about a fifth of them at this size and a third at thirty passes,
        # each justified only by the emitted barrier being global.
        @test missing_total == 0
        @test required_total > 0
    end

    @testset "derived barriers agree with the backend on random DAGs" begin
        # Elision is where races come from, and a race that happens to give the
        # right answer proves nothing. Three separate bugs were found here:
        #   - aliased transients need a barrier at the reuse boundary, which
        #     per-resource tracking cannot see (RPS calls it ResourceAliasingInfo)
        #   - a covering barrier only covers a later dependency if its masks are
        #     at least as wide; a write-only alias barrier does not make a
        #     subsequent read visible
        #   - comparing every transient is meaningless once they are aliased
        include(joinpath(@__DIR__, "..", "bench", "fuzz.jl"))
        r = Base.invokelatest(fuzz, 1:20)
        @test r.disagreements == 0
        # Against the same graphs compiled with coalescing off, which is the
        # question. `possible_total` was the old comparison and it is `passes - 1`:
        # a plan is replayed, so the first pass needs a barrier against the
        # previous frame and the real ceiling is one higher. It held only by a
        # margin of one, and narrowing a stage mask — which reduces how much a
        # covering barrier subsumes — was enough to cross it.
        # Equal, not fewer: a scoped barrier cannot subsume another resource's, so
        # the emitted set is the per-resource hazard set and compiling with
        # coalescing off changes nothing. It was `<` while barriers were global.
        @test r.emitted_total == r.uncoalesced_total
        @test r.emitted_total > 0
    end

    @kernel function sched_write!(dst, v::Float32)
        i = @index(Global)
        @inbounds dst[i] = v
    end
    @kernel function sched_accum!(dst, @Const(src))
        i = @index(Global)
        @inbounds dst[i] += src[i]
    end

    @testset "the schedule matches RPS on its own benchmark" begin
        # Ported from AMD's Render Pipeline Shaders,
        # tests/console/test_scheduler.rpsl :: memory_saving, with the expected
        # orders from test_scheduler.cpp. Six independent producer/consumer pairs
        # over six transients, every consumer writing one output.
        #
        # RPS asserts the whole node sequence, which is the part worth taking: our
        # own scheduler test pins two pass names, and two names are satisfied by
        # schedules that are wrong from the third onwards. Their two expectations
        # map exactly onto our two policies —
        #   default / performance   PushExpectedRange(0, 12, 1)   -> 0..11
        #   RPS_SCHEDULE_PREFER_MEMORY_SAVING_BIT
        #                           PushExpectedRange(i, i+7, 6)  -> (0,6) (1,7) ...
        # — which is `Overlap` and `Compact`, and Mantle produces both.
        dev = M.Device(M.VulkanAPI())
        n = 256
        mk(pol) = begin
            g = M.Graph(dev)
            out = M.Buffer(dev, zeros(Float32, n))
            ts = [M.Transient.Buffer(g, Float32, n) for _ in 1:6]
            for i in 1:6
                M.compute!(g, "draw$(i-1)") do p
                    M.dispatch!(p, sched_write!, (M.use(p, ts[i]; write = true), Float32(i)), n)
                end
            end
            for i in 1:6
                M.compute!(g, "blt$(i-1)") do p
                    M.dispatch!(p, sched_accum!, (M.use(p, out; read = true, write = true),
                                                  M.use(p, ts[i]; read = true)), n)
                end
            end
            M.Plan(g; policy = pol)
        end
        order(pl) = [pp.pass.name for pp in pl.passes]

        fast, small = mk(M.Overlap()), mk(M.Compact())
        @test order(fast) == vcat(["draw$i" for i in 0:5], ["blt$i" for i in 0:5])
        @test order(small) == vcat([["draw$i", "blt$i"] for i in 0:5]...)
        # And the reason the second order exists: one transient alive at a time
        # instead of six.
        @test M.peakbytes(small) * 6 == M.peakbytes(fast)

        # Their other case, `program_order` — draws 0-5, blts 6-11, then twelve
        # alternating draw/blt over the same transients — is deliberately *not*
        # asserted by sequence. RPS expects the second round of draws hoisted
        # (12,14..22 then 13,15..23); we keep declaration order, and measured,
        # the two cost exactly the same: 24 barriers, 36 transitions, 6144 peak,
        # for their order and ours alike. Grouping independent work paid when a
        # barrier stopped everything; per-resource barriers already scope to what
        # each pass touches, so the choice is between equals.
        #
        # Which is what their own harness says to do about it — `unorderedEqual`,
        # for graphs where several orders are legal. So: assert the schedule is a
        # linear extension of the DAG, and that it costs no more than theirs.
        # `hoist` picks between the two declaration orders for passes 12-23.
        program_order(hoist) = begin
            g = M.Graph(dev)
            out = M.Buffer(dev, zeros(Float32, n))
            ts = [M.Transient.Buffer(g, Float32, n) for _ in 1:6]
            dr = (id, i) -> M.compute!(g, "$id") do p
                M.dispatch!(p, sched_write!, (M.use(p, ts[i]; write = true), Float32(id)), n)
            end
            bl = (id, i) -> M.compute!(g, "$id") do p
                M.dispatch!(p, sched_accum!, (M.use(p, out; read = true, write = true),
                                              M.use(p, ts[i]; read = true)), n)
            end
            for i in 1:6; dr(i - 1, i); end
            for i in 1:6; bl(i + 5, i); end
            if hoist                                    # RPS: every draw, then every blt
                for i in 1:6; dr(10 + 2i, i); end
                for i in 1:6; bl(11 + 2i, i); end
            else                                        # as declared: alternating
                for i in 1:6; dr(10 + 2i, i); bl(11 + 2i, i); end
            end
            M.Plan(g; policy = M.Overlap())
        end
        forced, ours = program_order(true), program_order(false)
        cost(pl) = (count(pp -> !isempty(pp.pre), pl.passes),
                    sum(length(pp.pre) for pp in pl.passes),
                    M.peakbytes(pl))
        @test cost(ours) == cost(forced)

        # A linear extension of the DAG: a pass declared earlier that shares a
        # written resource has to still come first. Rebuilt from the usages here,
        # rather than read off the compiler, so a scheduler that reordered past a
        # dependency would fail this and not merely disagree with RPS.
        decl = ours.graph.passes
        pos = Dict(pp.pass.name => k for (k, pp) in enumerate(ours.passes))
        ok = true
        for j in eachindex(decl), i in 1:(j - 1)
            dep = any(((idi, Ui),) -> any(((idj, Uj),) ->
                          idi == idj && (M.writes(Ui) || M.writes(Uj)), decl[j].usages),
                      decl[i].usages)
            dep && pos[decl[i].name] > pos[decl[j].name] && (ok = false)
        end
        @test ok
    end

    @testset "scheduling reorders passes and cuts peak memory" begin
        # Six independent chains declared interleaved. Declaration order keeps
        # every chain's buffers alive at once; the memory-saving policy discovers
        # chain-major order by itself and the peak drops accordingly.
        dev = M.Device(M.VulkanAPI())
        mk(pol) = begin
            g = M.Graph(dev)
            bufs = [[M.Transient.Buffer(g, Float32, 1 << 12) for _ in 1:5] for _ in 1:6]
            for s in 1:4, c in 1:6
                M.compute!(g, "c$c s$s") do p
                    M.dispatch!(p, stir!, (M.use(p, bufs[c][s + 1]; write = true),
                                           M.use(p, bufs[c][s]; read = true), 1.0f0), 1 << 12)
                end
            end
            M.Plan(g; policy = pol)
        end
        loose, tight = mk(M.Overlap()), mk(M.Compact())
        @test M.peakbytes(tight) < M.peakbytes(loose)
        @test [pp.pass.name for pp in loose.passes][1:2] == ["c1 s1", "c2 s1"]   # interleaved
        @test [pp.pass.name for pp in tight.passes][1:2] == ["c1 s1", "c1 s2"]   # chain-major
    end

    @testset "reordering preserves results" begin
        # A schedule may only permute passes within the DAG, so both policies must
        # compute what the CPU replay says. Checked absolutely rather than by
        # comparing the two: agreement alone would also be satisfied by both being
        # wrong the same way.
        include(joinpath(@__DIR__, "..", "bench", "fuzz.jl"))
        dev = M.Device(M.VulkanAPI())
        for pol in (M.Overlap(), M.Compact())
            r = Base.invokelatest(fuzz, 1:12; passes = 12, policy = pol)
            @test r.wrong == 0
            @test r.disagreements == 0
        end
    end

    @testset "aliasing trades memory against barriers" begin
        # RPS ships DEFAULT_MEMORY and DEFAULT_PERFORMANCE because sharing bytes
        # forces ordering between the passes that use them. Our design document
        # claims both wins at once; this is the measurement that says otherwise.
        dev = M.Device(M.VulkanAPI())
        mk(al) = begin
            g = M.Graph(dev)
            bufs = [M.Transient.Buffer(g, Float32, 1 << 12) for _ in 1:12]
            for i in 2:12
                M.compute!(g, "p$i") do p
                    M.dispatch!(p, stir!, (M.use(p, bufs[i]; write = true),
                                           M.use(p, bufs[i - 1]; read = true), 1.0f0), 1 << 12)
                end
            end
            M.Plan(g; alias = al)
        end
        on, off = mk(true), mk(false)
        @test M.peakbytes(on) < M.peakbytes(off)
        @test count(pp -> !isempty(pp.pre), on.passes) >=
              count(pp -> !isempty(pp.pre), off.passes)
    end

    @testset "a second draw does not erase the first" begin
        # One begin/end per pass with the clear as pass configuration. Issuing a
        # fresh draw! per item transitions from UNDEFINED each time and discards
        # everything before it, which shows up as `both` being darker than either.
        lit(which) = begin
            dev = M.Device(M.VulkanAPI())
            win = M.Window(W, H; title = "t")
            sa, sb = cloud(50_000), cloud(20_000)
            a = Scatter(M.Buffer(dev, sa), M.Buffer(dev, tint.(sa)), M.Scalar(dev, 2.0f0))
            b = Scatter(M.Buffer(dev, sb), M.Scalar(dev, Vec4f(0.10, 0.04, 0.01, 1)),
                        M.Buffer(dev, 1.0f0 .+ 3.0f0 .* rand(Float32, 20_000)))
            mvp = Ref(camera(0.7f0))
            g = M.Graph(dev); screen = M.Surface(g, win)
            picked = which === :a ? (a,) : which === :b ? (b,) : (a, b)
            M.render!(g, "s", screen => M.Clear((0.02f0, 0.02f0, 0.04f0, 1f0))) do p
                for s in picked
                    M.draw!(p, SCATTER, bind(p, s, mvp), s.positions)
                end
            end
            plan = M.Plan(g)
            for _ in 1:20; Mantle.GLFW.PollEvents(); M.run!(plan); end
            KernelAbstractions.synchronize(M.backend(dev))
            img = M.screenshot(win); close(win)
            sum(Float64(p[1]) + p[2] + p[3] for p in img)
        end
        la, lb, lo = lit(:a), lit(:b), lit(:both)
        @test lo > la
        @test lo > lb
    end

    @testset "transient render targets are placed and aliased" begin
        # Two 1000x750 targets whose lifetimes do not overlap. The placer should
        # give the second the first's bytes, and the result must not change:
        # target A is copied out before B is rendered, so sharing is legal.
        include(joinpath(@__DIR__, "..", "bench", "targets.jl"))
        dev = M.Device(M.VulkanAPI())
        # The same points for both: `cloud` is random, so two graphs built from
        # separate calls differ for a reason that has nothing to do with aliasing.
        pts = Base.invokelatest(cloud, 20_000)
        tight = Base.invokelatest(build_targets, dev, 20_000; points = pts)
        loose = Base.invokelatest(build_targets, dev, 20_000; points = pts, alias = false)

        E = Mantle
        imgbytes(p) = only(length(s) for s in p.slabs
                           if any(t -> t isa E.TransientImage && t.memory === M.memoryof(s),
                                  p.graph.transients))
        one = E.nbytes(first(t for t in tight.plan.graph.transients if t isa E.TransientImage))

        # Exactly one target's worth when they share, and two when they do not.
        # Not `tight * 2 == loose`: these images want 64 kB alignment and one
        # target is not a multiple of it, so the unaliased arena carries padding
        # between them that the aliased arena never pays.
        @test imgbytes(tight.plan) == one
        @test 2 * one <= imgbytes(loose.plan) < 2 * one + E.alignment(
            first(t for t in loose.plan.graph.transients if t isa E.TransientImage))

        got = map((tight, loose)) do s
            M.run!(s.plan)
            KernelAbstractions.synchronize(M.backend(dev))
            Array(M.storage(s.out))
        end
        @test got[1] == got[2]                 # aliasing changed nothing
        # Against the background, not an absolute threshold. The tint is scaled
        # by 0.06 and the combine averages two 8-bit images, so a lit pixel is
        # around 0.05 and a fixed `> 0.5` would pass on a blank frame only by
        # counting alpha, which is 1 everywhere.
        green = [p[2] for p in got[1]]
        @test count(>(minimum(green) + 0.02f0), green) > 1000
    end

    # A resize is asynchronous: the compositor applies it when it applies it, and
    # a frame drawn before that lands in a swapchain that is about to be replaced,
    # so the next readback finds an image nothing has drawn into. Render until the
    # swapchain matches what was asked for, then draw the frames the assertion
    # reads. Without this the tests below fail one time in several, and the size
    # assertion still passes — it is only the pixels that are missing.
    function resize_and_settle!(dev, plan, win, w, h)
        GLFW.SetWindowSize(win.win.handle, w, h)
        for _ in 1:200
            M.run!(plan)
            size(win) == (w, h) && break
        end
        for _ in 1:6; M.run!(plan); end
        KernelAbstractions.synchronize(M.backend(dev))
        return size(win)
    end

    @testset "the swapchain follows a window resize" begin
        # A resize is not reliably reported: OUT_OF_DATE is what a driver *may*
        # return, and SUBOPTIMAL is a success code that never reaches an error
        # branch. On this platform neither arrives — the window shrank and the
        # swapchain kept presenting at the old size, which the compositor scales,
        # so it looks blurry and reads back at the wrong resolution. What notices
        # is comparing the framebuffer size against what the swapchain was built
        # for, every frame.
        dev = M.Device(M.VulkanAPI())
        win = M.Window(W, H; title = "resize")
        s = Base.invokelatest(build, dev, win, 20_000, 10_000)
        for _ in 1:10; M.run!(s.plan); end
        KernelAbstractions.synchronize(M.backend(dev))
        @test size(M.screenshot(win)) == (W, H)

        for (w, h) in ((640, 480), (1280, 900), (300, 200))
            resize_and_settle!(dev, s.plan, win, w, h)
            img = M.screenshot(win)
            @test size(img) == (w, h)          # the swapchain went with it
            bg = img[1, 1]
            @test count(p -> p != bg, img) > length(img) ÷ 100   # and still drew
        end
        close(win)
    end

    @testset "a transient given a target follows it through a resize" begin
        # `Transient.Image(g, Float32, screen)` takes its size from the surface
        # rather than from a tuple, so a window resize carries it along: the image
        # is recreated and everything is placed again, which is a recompile
        # because a different size is different offsets, different aliasing and
        # therefore different barriers.
        zfrag(inputs) = inputs.color
        zpipe = Rasterizer(vertex = scatter_vertex, fragment = zfrag,
                           varyings = (color = Vec4f,), topology = PointList(),
                           blend = Opaque(), cull = NoCull(), depth = DepthLess())
        dev = M.Device(M.VulkanAPI())
        win = M.Window(800, 600; title = "tracking depth")
        g = M.Graph(dev)
        screen = M.Surface(g, win)
        z = M.Transient.Image(g, Float32, screen)
        pts = cloud(20_000)
        sc = Scatter(M.Buffer(dev, pts), M.Buffer(dev, tint.(pts)), M.Scalar(dev, 4f0))
        mvp = Ref(camera(0f0))
        M.render!(g, "depth", screen => M.Clear((0.02f0, 0.02f0, 0.04f0, 1f0)),
                  z => M.Clear(1f0)) do p
            M.draw!(p, zpipe, bind(p, sc, mvp), sc.positions)
        end
        plan = M.Plan(g)
        for _ in 1:5; M.run!(plan); end
        KernelAbstractions.synchronize(M.backend(dev))
        @test size(z) == (800, 600)
        small = M.peakbytes(plan)

        for (w, h) in ((1200, 900), (400, 300))
            resize_and_settle!(dev, plan, win, w, h)
            @test size(z) == (w, h)                  # the depth target went along
            # One screenshot for both assertions: a readback acquires an image,
            # and right after a resize the one the presentation engine hands back
            # may be a new image nothing has drawn into yet.
            img = M.screenshot(win)
            @test size(img) == (w, h)
            bg = img[1, 1]
            @test count(p -> p != bg, img) > length(img) ÷ 100   # and still drew
        end
        # Placed again at each size, so the arena tracks it rather than staying
        # at whatever the first frame needed.
        @test M.peakbytes(plan) < small              # 400x300 needs less than 800x600
        close(win)
    end

    @testset "an attachment that stops covering the render area is an error" begin
        # The swapchain follows a resize; a transient sized when the graph was
        # built does not. Growing the window past it makes every frame violate
        # VUID-VkRenderingInfo-pNext-06079, which RADV draws anyway — so it is a
        # wrong picture unless something says so.
        zfrag(inputs) = inputs.color
        zpipe = Rasterizer(vertex = scatter_vertex, fragment = zfrag,
                           varyings = (color = Vec4f,), topology = PointList(),
                           blend = Opaque(), cull = NoCull(), depth = DepthLess())
        dev = M.Device(M.VulkanAPI())
        win = M.Window(800, 600; title = "resize mismatch")
        g = M.Graph(dev)
        screen = M.Surface(g, win)
        z = M.Transient.Image(g, Float32, size(win))
        pts = cloud(2_000)
        sc = Scatter(M.Buffer(dev, pts), M.Buffer(dev, tint.(pts)), M.Scalar(dev, 3f0))
        mvp = Ref(camera(0f0))
        M.render!(g, "depth", screen => M.Clear((0f0, 0f0, 0f0, 1f0)), z => M.Clear(1f0)) do p
            M.draw!(p, zpipe, bind(p, sc, mvp), sc.positions)
        end
        plan = M.Plan(g)
        M.run!(plan)                              # same size: fine
        KernelAbstractions.synchronize(M.backend(dev))

        GLFW.SetWindowSize(win.win.handle, 1200, 900)
        sleep(0.2)
        @test_throws ErrorException M.run!(plan)  # bigger window, same depth target
        close(win)
    end

    @testset "a readback returns the image it acquired" begin
        # `readback_window` acquires an image when no frame is in flight. Leaving
        # it outstanding is invisible once and a deadlock in a loop: a swapchain
        # has two or three images, so the fourth readback waits forever inside
        # vkAcquireNextImageKHR for one that is never handed back. Found by
        # measuring two graphs against one window, which hung on the second.
        #
        # The assertion is on the state rather than on not hanging, so a
        # regression fails on the first iteration instead of stopping the suite.
        win = VulkanWindow(256, 256; title = "readback pairing", vsync = false)
        for _ in 1:6
            readback_window(win)
            @test !win.acquired
        end
        close(win)
    end

    @testset "a second frame started under an open one is an error" begin
        # Recording is not atomic against the Julia scheduler: compiling a kernel
        # mid-frame yields, and the launch plan is keyed on the world counter, so
        # one method definition anywhere in the session is enough to make the next
        # frame compile. A second task rendering the same window then arrives at
        # `acquire_next_image!` between the acquire and the present, and waits on
        # the fence the missing present would have signalled. That wait is a
        # blocking foreign call, so the task holding the frame is never scheduled
        # again, GC stops, and the process cannot be interrupted or profiled.
        #
        # Found by measuring a running demo from the REPL, which wedged the whole
        # session. A regression here hangs the suite rather than failing it —
        # that is exactly what the check is for.
        win = VulkanWindow(256, 256; title = "one frame at a time", vsync = false)
        acquire_next_image!(win)
        @test win.acquired
        @test win.acquirer === current_task()

        err = try; acquire_next_image!(win); catch e; e; end
        @test err isa ErrorException
        @test occursin("the same task", err.msg)

        other = fetch(@async(try; acquire_next_image!(win); nothing; catch e; e; end))
        @test other isa ErrorException
        @test occursin("another task", other.msg)

        # The refused acquires changed nothing, so the frame that owns the image
        # can still finish, and the window is usable afterwards.
        bq = win.ctx.default_bq
        Mantle.ensure_active_batch!(bq)
        present_frame!(bq, win)
        @test !win.acquired
        @test win.acquirer === nothing
        acquire_next_image!(win)
        Mantle.ensure_active_batch!(bq)
        present_frame!(bq, win)
        @test !win.acquired
        close(win)
    end

    @testset "closing a window destroys what it owns" begin
        # `close` used to destroy only the GLFW window, leaving the swapchain,
        # its views and the surface to finalizers. Julia does not run finalizers
        # at exit, so every process ended with
        # VUID-vkDestroyInstance-instance-00629 reporting a leaked VkSurfaceKHR,
        # and every resize leaked a swapchain's worth of image views.
        #
        # The surface needs destroying by name rather than by `finalize`: Lava
        # wraps it around the pointer GLFW returns without going through
        # Vulkan.jl's `init_handle!`, so it has no destructor and no finalizer.
        win = VulkanWindow(64, 64; title = "close test", vsync = false)
        @test !isempty(win.views)
        @test win.swapchain !== nothing
        @test win.surface.destructor isa UndefInitializer   # why finalize cannot work

        close(win)
        @test isempty(win.views)
        @test isempty(win.image_available)
        @test win.swapchain === nothing
        close(win)                                          # idempotent
    end

    @testset "each load op reaches its Vulkan attachment op" begin
        # Discard is the point of the exercise: inferring the op from whether a
        # clear colour was given can only ever pick CLEAR or LOAD, so a pass that
        # covers every pixel used to pay for a load it discards.
        E = Mantle
        @test E.loadop(M.Keep) == Mantle.VK.ATTACHMENT_LOAD_OP_LOAD
        @test E.loadop(M.Discard) == Mantle.VK.ATTACHMENT_LOAD_OP_DONT_CARE
        @test E.loadop(M.Clear((0f0, 0f0, 0f0, 1f0))) == Mantle.VK.ATTACHMENT_LOAD_OP_CLEAR
        @test E.clearvalue(M.Keep) === nothing
        @test E.clearvalue(M.Clear(Vec4f(0.1, 0.2, 0.3, 1))) == (0.1f0, 0.2f0, 0.3f0, 1f0)

        # And all three actually render. The target is a transient, so its state
        # before the pass is Undefined and the barrier into it comes from there.
        dev = M.Device(M.VulkanAPI())
        pts = cloud(20_000)
        for (op, want) in ((M.Clear((0f0, 0f0, 0f0, 1f0)), M.ColorAttachment{true}),
                           (M.Discard, M.ColorAttachment{true}),
                           (M.Keep, M.ColorAttachment{false}))
            g = M.Graph(dev)
            img = M.Transient.Image(g, BGRA{ColorTypes.FixedPointNumbers.N0f8}, (256, 256))
            raw = M.Transient.Buffer(g, UInt8, 256 * 256 * 4)
            mvp = Ref(camera(0f0))
            pos = M.Buffer(dev, pts); col = M.Buffer(dev, tint.(pts)); siz = M.Scalar(dev, 3f0)
            M.render!(g, "p", img => op) do p
                M.draw!(p, SCATTER, (M.Attribute(p, pos), M.Attribute(p, col),
                                     M.Attribute(p, siz), mvp, Int32(1), Int32(0)), pos)
            end
            M.copy!(g, "read", raw, img)
            plan = M.Plan(g)
            @test last(first(plan.graph.passes[1].usages)) === want
            M.run!(plan)
            KernelAbstractions.synchronize(M.backend(dev))
            px = reshape(Array(M.storage(raw)), 4, :)
            lit = count(j -> Int(px[1, j]) + Int(px[2, j]) + Int(px[3, j]) > 40, axes(px, 2))
            @test lit > 1000
        end
    end

    @testset "a depth attachment decides which fragment wins" begin
        # Two points at the same place, one near and one far. Without depth the
        # later draw wins, which is why the order is the control: with a depth
        # attachment the near one has to win in *both* orders.
        opaque_frag(inputs) = inputs.color
        ZPIPE = Rasterizer(vertex = scatter_vertex, fragment = opaque_frag,
                           varyings = (color = Vec4f,), topology = PointList(),
                           blend = Opaque(), cull = NoCull(), depth = DepthLess())
        FLAT = Rasterizer(vertex = scatter_vertex, fragment = opaque_frag,
                          varyings = (color = Vec4f,), topology = PointList(),
                          blend = Opaque(), cull = NoCull(), depth = DepthOff())

        N = 64
        dev = M.Device(M.VulkanAPI())
        ident = Mat4f(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1)
        mvp = Ref(ident)
        # z is NDC depth here, so 0.3 is nearer than 0.7.
        near = Scatter(M.Buffer(dev, [Vec3f(0, 0, 0.3)]),
                       M.Scalar(dev, Vec4f(0, 0, 1, 1)), M.Scalar(dev, 40f0))
        far = Scatter(M.Buffer(dev, [Vec3f(0, 0, 0.7)]),
                      M.Scalar(dev, Vec4f(1, 0, 0, 1)), M.Scalar(dev, 40f0))

        function shot(order, pipe; withdepth = true)
            g = M.Graph(dev)
            img = M.Transient.Image(g, BGRA{ColorTypes.FixedPointNumbers.N0f8}, (N, N))
            raw = M.Transient.Buffer(g, UInt8, N * N * 4)
            z = withdepth ? M.Transient.Image(g, Float32, (N, N)) : nothing
            att = withdepth ? (img => M.Clear((0f0, 0f0, 0f0, 1f0)), z => M.Clear(1f0)) :
                              (img => M.Clear((0f0, 0f0, 0f0, 1f0)),)
            M.render!(g, "scene", att...) do p
                for s in order
                    M.draw!(p, pipe, bind(p, s, mvp), s.positions)
                end
            end
            M.copy!(g, "read", raw, img)
            plan = M.Plan(g)
            M.run!(plan)
            KernelAbstractions.synchronize(M.backend(dev))
            px = reshape(Array(M.storage(raw)), 4, N, N)
            (blue = Int(px[1, N ÷ 2, N ÷ 2]), red = Int(px[3, N ÷ 2, N ÷ 2]), plan = plan)
        end

        # The declaration reaches the usage sequence, and a cleared depth target
        # discards, so its barrier comes from UNDEFINED like any other transient.
        probe = shot((near, far), ZPIPE)
        @test any(last(u) === M.Depth{M.ReadWrite,M.NoAccess,true}
                  for u in probe.plan.graph.passes[1].usages)
        E = Mantle
        zt = only(t for t in probe.plan.graph.transients
                  if t isa E.TransientImage && eltype(t) === Float32)
        @test zt.format == Mantle.VK.FORMAT_D32_SFLOAT
        @test E.aspect(zt) == Mantle.VK.IMAGE_ASPECT_DEPTH_BIT

        for order in ((near, far), (far, near))
            got = shot(order, ZPIPE)
            @test got.blue > 200        # the near point
            @test got.red < 50
        end

        # The control: with no depth attachment the same two draws are decided by
        # order alone, so this is what the test above would have measured if the
        # attachment had done nothing.
        @test shot((near, far), FLAT; withdepth = false).red > 200
        @test shot((far, near), FLAT; withdepth = false).blue > 200

        # And the depth buffer itself, rather than its effect on colour: the
        # target is copied out like any other image, which needs the depth aspect
        # in the copy region and TRANSFER_SRC in the image's usage.
        g = M.Graph(dev)
        img = M.Transient.Image(g, BGRA{ColorTypes.FixedPointNumbers.N0f8}, (N, N))
        z = M.Transient.Image(g, Float32, (N, N))
        raw = M.Transient.Buffer(g, Float32, N * N)
        M.render!(g, "scene", img => M.Clear((0f0, 0f0, 0f0, 1f0)), z => M.Clear(1f0)) do p
            for s in (near, far)
                M.draw!(p, ZPIPE, bind(p, s, mvp), s.positions)
            end
        end
        M.copy!(g, "read z", raw, z)
        plan = M.Plan(g)
        M.run!(plan)
        KernelAbstractions.synchronize(M.backend(dev))
        zs = reshape(Array(M.storage(raw)), N, N)
        @test zs[N ÷ 2, N ÷ 2] ≈ 0.3f0 atol = 1e-5   # the near point's z, not the far one's
        @test zs[1, 1] ≈ 1f0                          # and the clear everywhere else
    end

    @testset "several colour targets, written by one fragment" begin
        # The fragment returns a tuple, and element i goes to attachment i. Two
        # targets that get different colours is the whole claim: one attachment
        # written twice would pass any test that only looked at one of them.
        two_frag(inputs) = (inputs.color, Vec4f(0, 1, 0, 1))
        MRT = Rasterizer(vertex = scatter_vertex, fragment = two_frag,
                         varyings = (color = Vec4f,), topology = PointList(),
                         blend = Opaque(), cull = NoCull(), depth = DepthOff())

        N = 64
        dev = M.Device(M.VulkanAPI())
        mvp = Ref(Mat4f(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1))
        blue = Scatter(M.Buffer(dev, [Vec3f(0, 0, 0.5)]),
                       M.Scalar(dev, Vec4f(0, 0, 1, 1)), M.Scalar(dev, 40f0))

        g = M.Graph(dev)
        T = BGRA{ColorTypes.FixedPointNumbers.N0f8}
        a = M.Transient.Image(g, T, (N, N))
        b = M.Transient.Image(g, T, (N, N))
        ra = M.Transient.Buffer(g, UInt8, N * N * 4)
        rb = M.Transient.Buffer(g, UInt8, N * N * 4)
        M.render!(g, "gbuffer", a => M.Clear((0f0, 0f0, 0f0, 1f0)),
                                b => M.Clear((0f0, 0f0, 0f0, 1f0))) do p
            M.draw!(p, MRT, bind(p, blue, mvp), blue.positions)
        end
        M.copy!(g, "read a", ra, a)
        M.copy!(g, "read b", rb, b)
        # `alias = false` because the reader is outside the graph. Nothing in the
        # graph reads `ra` or `rb`, so their intervals end where they are written,
        # they do not overlap, and the placer is right to give them the same bytes
        # — at which point both readbacks return whichever copy ran last. That is
        # aliasing working, not MRT failing, and it cost a debugging detour.
        plan = M.Plan(g; alias = false)

        # Both attachments are declared, so both get a derived barrier.
        @test count(u -> last(u) <: M.ColorAttachment, plan.graph.passes[1].usages) == 2

        # And each copy transitions its own source. A layout change is per image,
        # so the barrier the first copy emitted cannot have put the second target
        # into TRANSFER_SRC however wide its stage and access masks were — but the
        # coalescing tested it by masks alone and dropped the second transition.
        # Synchronization validation reports the second copy reading its source in
        # COLOR_ATTACHMENT_OPTIMAL; the pixels below come out right anyway, which
        # is why this is asserted on the barrier rather than on the picture.
        copies = [pp for pp in plan.passes if pp.pass.kind === :copy]
        @test length(copies) == 2
        for pp in copies
            @test any(b -> b.new == Mantle.VK.IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, pp.images)
        end

        M.run!(plan)
        KernelAbstractions.synchronize(M.backend(dev))
        pa = reshape(Array(M.storage(ra)), 4, N, N)
        pb = reshape(Array(M.storage(rb)), 4, N, N)
        c = N ÷ 2
        @test Int(pa[1, c, c]) > 200 && Int(pa[2, c, c]) < 50    # target 0: blue
        @test Int(pb[2, c, c]) > 200 && Int(pb[1, c, c]) < 50    # target 1: green

        # Attachments that disagree about their size share one render area, so it
        # is a compile error rather than a partly-covered second target.
        h = M.Graph(dev)
        big = M.Transient.Image(h, T, (N, N))
        small = M.Transient.Image(h, T, (N ÷ 2, N ÷ 2))
        M.render!(h, "mismatched", big => M.Clear((0f0, 0f0, 0f0, 1f0)),
                                   small => M.Clear((0f0, 0f0, 0f0, 1f0))) do p
            M.draw!(p, MRT, bind(p, blue, mvp), blue.positions)
        end
        @test_throws ArgumentError M.Plan(h)
    end

    # Three sources of new positions, built here rather than by including
    # `bench/three_updates.jl`: that file is an eval-block script now, and
    # including it would start its render loop.
    function three_plots(dev, win, n; profile = false)
        g = M.Graph(dev)
        screen = M.Surface(g, win)
        dt = Ref(1f0 / 60)
        mk(c) = Scatter(M.Buffer(dev, c), M.Buffer(dev, tint.(c)), M.Scalar(dev, 2f0))
        a, b, c = cloud(n), cloud(n), cloud(n)
        sim, gpu, cpu = mk(a), mk(b), mk(c)

        simvel = M.Buffer(dev, drift(n))
        M.compute!(g, "advect") do p
            x = M.use(p, sim.positions; read = true, write = true)
            v = M.use(p, simvel; read = true, write = true)
            M.dispatch!(p, advect!, (x, v, dt, 1.4f0), n)
        end
        from_gpu = M.Update(g, gpu.positions)
        from_cpu = M.Update(g, cpu.positions)

        gpupos, gpuvel = M.Buffer(dev, copy(b)), M.Buffer(dev, drift(n))
        cpupos, cpuvel = copy(c), drift(n)
        mvp = Ref(camera(0f0))
        M.render!(g, "three", screen => M.Clear((0.02f0, 0.02f0, 0.04f0, 1f0))) do p
            for s in (sim, gpu, cpu)
                M.draw!(p, SCATTER, bind(p, s, mvp), s.positions)
            end
        end
        (; plan = M.Plan(g; profile), dev, dt, n, sim, gpu, cpu,
           from_gpu, from_cpu, gpupos, gpuvel, cpupos, cpuvel)
    end

    function three_step!(s)
        advect!(M.backend(s.dev))(M.storage(s.gpupos), M.storage(s.gpuvel), s.dt[], 1.4f0;
                                  ndrange = s.n)
        s.from_gpu(M.storage(s.gpupos))         # device array: device to device
        advect!(CPU())(s.cpupos, s.cpuvel, s.dt[], 1.4f0; ndrange = s.n)
        KernelAbstractions.synchronize(CPU())
        s.from_cpu(s.cpupos)                    # host array: staged
    end

    @testset "three sources of new positions in one graph" begin
        # A compute pass in the graph, a device array produced outside it, and a
        # host array. The three differ only in where the bytes come from, and the
        # same handover takes whichever route the data calls for.
        dev = M.Device(M.VulkanAPI())
        win = M.Window(W, H; title = "three updates")
        s = Base.invokelatest(three_plots, dev, win, 5_000)
        @test M.npipelines(s.plan) == 1          # three plots, one shader

        plots = (s.sim, s.gpu, s.cpu)
        before = map(p -> copy(Array(p.positions)), plots)
        for _ in 1:20
            Base.invokelatest(three_step!, s)
            M.run!(s.plan)
            KernelAbstractions.synchronize(M.backend(dev))
        end
        after = map(p -> Array(p.positions), plots)

        for (b, a) in zip(before, after)
            @test count(<(1e-6), [norm(a[i] - b[i]) for i in eachindex(b)]) == 0
            @test maximum(norm, a) <= 1.4f0 + 1e-4      # each stayed in its shell
        end
        # The device route copies device to device: what the plot holds is what
        # the user's buffer holds, bit for bit, and the host never saw it.
        @test Array(s.gpu.positions) == Array(M.storage(s.gpupos))
        # And the three are genuinely separate simulations.
        @test after[1] != after[2] && after[2] != after[3]

        # A handover from another task reaches the next frame: it stores a
        # reference and touches no Vulkan, so where it is called from is free.
        fresh = cloud(5_000)
        fetch(Threads.@spawn s.from_cpu(fresh))
        M.run!(s.plan)
        KernelAbstractions.synchronize(M.backend(dev))
        @test Array(s.cpu.positions) == fresh
        close(win)
    end

    @testset "a profiled plan times every pass" begin
        # `profile = true` is the whole of it, and asking an unprofiled plan is an
        # error rather than a table of zeros.
        dev = M.Device(M.VulkanAPI())
        win = M.Window(W, H; title = "profiled")

        plain = Base.invokelatest(three_plots, dev, win, 5_000)
        @test_throws ArgumentError M.timings(plain.plan)

        s = Base.invokelatest(three_plots, dev, win, 5_000; profile = true)
        for _ in 1:30
            Base.invokelatest(three_step!, s)
            M.run!(s.plan)
            KernelAbstractions.synchronize(M.backend(dev))
        end
        ts = M.timings(s.plan)

        @test [t.name for t in ts] == [pp.pass.name for pp in s.plan.passes]
        @test [t.kind for t in ts] == [:update, :compute, :render]
        # Timestamps came back for every pass, and they are times rather than
        # zeros: a query pool that is never reset, or read with the wrong stride,
        # gives exactly zeros.
        for t in ts
            @test t.samples > 0
            @test 0 < t.gpu_ms < 100
            @test 0 < t.host_ms < 100
        end
        # Kept bounded: the ring is NSAMPLES long however long the plan runs.
        @test all(t -> t.samples <= M.NSAMPLES, ts)
        close(win)
    end

    @testset "a frame waits for what the last one left" begin
        # A plan is replayed, so frame N+1's first use of a resource follows frame
        # N's last use of it — and with frames in flight the two overlap on the
        # GPU. Seeding the tracker with `Undefined` each frame left the first
        # barrier with no source scope: nothing to wait for. Synchronization
        # validation calls it a write-after-write against the previous frame's
        # store op, and it is a race the moment the loop stops flushing.
        zfrag(inputs) = inputs.color
        zpipe = Rasterizer(vertex = scatter_vertex, fragment = zfrag,
                           varyings = (color = Vec4f,), topology = PointList(),
                           blend = Opaque(), cull = NoCull(), depth = DepthLess())
        dev = M.Device(M.VulkanAPI())
        g = M.Graph(dev)
        T = BGRA{ColorTypes.FixedPointNumbers.N0f8}
        img = M.Transient.Image(g, T, (64, 64))
        z = M.Transient.Image(g, Float32, (64, 64))
        pts = cloud(500)
        sc = Scatter(M.Buffer(dev, pts), M.Buffer(dev, tint.(pts)), M.Scalar(dev, 2f0))
        mvp = Ref(camera(0f0))
        M.render!(g, "scene", img => M.Clear((0f0, 0f0, 0f0, 1f0)), z => M.Clear(1f0)) do p
            M.draw!(p, zpipe, bind(p, sc, mvp), sc.positions)
        end
        plan = M.Plan(g)

        E = Mantle
        depth = only(b for b in plan.passes[1].images
                     if E.aspect(b.resource) == Mantle.VK.IMAGE_ASPECT_DEPTH_BIT)
        # The layout still comes from UNDEFINED — the clear discards — but the
        # barrier has to wait for the previous frame's depth write all the same.
        @test depth.old == Mantle.VK.IMAGE_LAYOUT_UNDEFINED
        @test depth.src_access != Mantle.VK.AccessFlag2(0)
        @test depth.src_stage != Mantle.VK.PipelineStageFlag2(0)
    end

    @testset "a scalar attribute is written in place" begin
        # A scalar is a one-element buffer, so a new value is a one-element write:
        # inline in the command buffer, four bytes riding along with the frame.
        # Renaming would be absurd for that, and it must not happen — a scalar and
        # a per-element attribute are one pipeline precisely because the binding
        # is the same shape, and swapping the store would be a needless allocation.
        dev = M.Device(M.VulkanAPI())
        win = M.Window(W, H; title = "scalar update")
        pts = cloud(1_000)
        sc = Scatter(M.Buffer(dev, pts), M.Buffer(dev, tint.(pts)), M.Scalar(dev, 2f0))
        mvp = Ref(camera(0f0))
        g = M.Graph(dev)
        screen = M.Surface(g, win)
        marker = M.Update(g, sc.markersize)
        M.render!(g, "plot", screen => M.Clear((0.02f0, 0.02f0, 0.04f0, 1f0))) do p
            M.draw!(p, SCATTER, bind(p, sc, mvp), sc.positions)
        end
        plan = M.Plan(g)
        store = M.storage(sc.markersize)

        M.run!(plan)                                  # nothing set
        KernelAbstractions.synchronize(M.backend(dev))
        @test Array(store)[1] == 2f0

        marker(9f0)
        M.run!(plan)
        KernelAbstractions.synchronize(M.backend(dev))
        @test Array(store)[1] == 9f0                  # the new value arrived
        @test M.storage(sc.markersize) === store      # in the same buffer
        close(win)
    end

    @testset "Update writes at the reserved position, by two routes" begin
        # The call decides the route, not a flag: a whole-buffer replacement
        # renames (nothing reads the fresh store, so there is no hazard), and a
        # partial write goes in place inline in the command buffer.
        dev = M.Device(M.VulkanAPI())
        win = M.Window(W, H; title = "update")

        function scene(n; range = nothing)
            pts = cloud(n)
            sc = Scatter(M.Buffer(dev, pts), M.Buffer(dev, tint.(pts)), M.Scalar(dev, 2f0))
            mvp = Ref(camera(0f0))
            g = M.Graph(dev)
            screen = M.Surface(g, win)
            ref = M.Update(g, sc.positions; range)
            M.render!(g, "plot", screen => M.Clear((0.02f0, 0.02f0, 0.04f0, 1f0))) do p
                M.draw!(p, SCATTER, bind(p, sc, mvp), sc.positions)
            end
            (; sc, ref, g, plan = M.Plan(g), before = copy(pts))
        end

        # The reserved position is a pass, ahead of the render that reads it.
        w = Base.invokelatest(scene, 5_000)
        @test [pp.pass.name for pp in w.plan.passes] == ["updates", "plot"]

        # A clean frame renames nothing: the barrier is derived as if it fired,
        # but nothing is emitted when the source access never happened.
        store0 = M.storage(w.sc.positions)
        M.run!(w.plan); KernelAbstractions.synchronize(M.backend(dev))
        @test M.storage(w.sc.positions) === store0

        fresh = cloud(5_000)
        w.ref(fresh)
        M.run!(w.plan); KernelAbstractions.synchronize(M.backend(dev))
        @test M.storage(w.sc.positions) !== store0        # renamed
        @test Array(w.sc.positions) == fresh

        # Renaming recycles rather than allocating: the store count is bounded
        # and does not grow with the number of updates. The exact count is two to
        # four depending on whether the timeline has passed the retiring store by
        # the time the next take asks for it, which varies run to run — so what is
        # pinned is the bound, not equality between the two loops. A leak shows up
        # as a count that tracks the loop: 8 and 24.
        rename_stores(k) = begin
            seen = Any[]
            for _ in 1:k
                w.ref(cloud(5_000))
                M.run!(w.plan); KernelAbstractions.synchronize(M.backend(dev))
                push!(seen, objectid(M.storage(w.sc.positions)))
            end
            length(unique(seen))
        end
        few, many = rename_stores(8), rename_stores(24)
        @test few <= 4           # reuse is happening at all
        @test many <= 4          # and the count does not track the loop

        # Partial: in place, and only the declared range moves.
        v = Base.invokelatest(scene, 5_000; range = 1:100)
        s0 = M.storage(v.sc.positions)
        v.ref([Vec3f(9, 9, 9) for _ in 1:100])
        M.run!(v.plan); KernelAbstractions.synchronize(M.backend(dev))
        got = Array(v.sc.positions)
        @test M.storage(v.sc.positions) === s0            # no rename
        @test all(got[1:100] .== Ref(Vec3f(9, 9, 9)))
        @test got[101:end] == v.before[101:end]
        close(win)
    end

    @testset "a transient nothing uses says so" begin
        # Liveness derives the interval from use, so one that nothing touches
        # keeps `typemax(Int)` as its first pass and reaches the placer as an
        # inverted span. That reported the allocator's internals for a mistake in
        # the graph, and named neither the transient nor what to do about it.
        include(joinpath(@__DIR__, "..", "bench", "chain.jl"))
        dev = M.Device(M.VulkanAPI())
        g = M.Graph(dev)
        used = M.Transient.Buffer(g, Float32, 1 << 10)
        M.Transient.Buffer(g, Float32, 1 << 9)          # declared, never used
        src = M.Buffer(dev, rand(Float32, 1 << 10))
        M.compute!(g, "c") do p
            M.dispatch!(p, stir!, (M.use(p, used; write = true),
                                   M.use(p, src; read = true), 2.0f0), 1 << 10)
        end
        @test_throws "is never used by any pass" M.Plan(g)
    end

    @testset "a custom pass declares what it touches, not how" begin
        # `dispatch!` takes one kernel, its arguments and an ndrange. Work that is
        # one unit to the caller and several launches underneath — an ATen
        # operator, a fused block, a library call — has no way to say so, and
        # splitting it into a pass per launch would declare a resource sequence
        # the caller does not have. `custom!` is the declaration without the how.
        n = 1 << 12
        dev = M.Device(M.VulkanAPI())
        g = M.Graph(dev)
        src = M.Buffer(dev, fill(1f0, n))
        mid = M.Transient.Buffer(g, Float32, n)
        out = M.Buffer(dev, zeros(Float32, n))
        M.custom!(g, "two launches") do p
            a = M.use(p, mid; read = true, write = true)
            b = M.use(p, src; read = true)
            c = M.use(p, out; write = true)
            return function ()
                bump!(Mantle.LavaBackend())(M.storage(a), M.storage(b), 1f0; ndrange = n)
                bump!(Mantle.LavaBackend())(M.storage(c), M.storage(a), 10f0; ndrange = n)
            end
        end
        plan = M.Plan(g)
        @test length(plan.passes) == 1
        M.run!(plan)
        KernelAbstractions.synchronize(M.backend(dev))
        @test all(==(12f0), Array(out))      # 1 + 1, then + 10
    end

    @testset "the launches inside a custom pass are ordered against each other" begin
        # The graph derives hazards between *passes*. Inside one it derives
        # nothing, because the body did not say — and a body's second launch
        # reading what its first wrote is the ordinary case, not an exotic one: a
        # two-pass reduction, a split-K matmul, an operator with scratch.
        #
        # `run!(; barriers = :derived)` records inside a `concurrent_dispatch_group`
        # so that Lava's automatic per-dispatch barrier does not double up with the
        # derived ones. Left switched on through a custom body that suppressed
        # exactly the barrier nobody else was going to emit, which is why SAM 2's
        # image encoder came back as NaN through this path and matched to the bit
        # under `:backend`.
        #
        # Two assertions, because the numeric one alone passes on a lucky day: the
        # group has to be *lifted* for the body, and the result has to be right.
        #
        # `n` is SMALL on purpose. Two unordered dispatches only produce a wrong
        # answer where the second one's workgroups can start before the first has
        # finished, and a big grid saturates the device so thoroughly that they
        # cannot. Measured here, wrong elements over ten runs of the same pair
        # inside a `concurrent_dispatch_group`: 1<<14 fails 10/10 and mostly
        # everywhere, 1<<16 fails 3/10, 1<<18 and up 0/10.
        n = 1 << 14
        dev = M.Device(M.VulkanAPI())
        g = M.Graph(dev)
        src = M.Buffer(dev, fill(1f0, n))
        mid = M.Transient.Buffer(g, Float32, n)
        out = M.Buffer(dev, zeros(Float32, n))
        active = Bool[]
        M.custom!(g, "dependent launches") do p
            a = M.use(p, mid; read = true, write = true)
            b = M.use(p, src; read = true)
            c = M.use(p, out; write = true)
            return function ()
                push!(active, Mantle.CONCURRENT_GROUP_ACTIVE[])
                bump!(Mantle.LavaBackend())(M.storage(a), M.storage(b), 1f0; ndrange = n)
                bump!(Mantle.LavaBackend())(M.storage(c), M.storage(a), 10f0; ndrange = n)
            end
        end
        plan = M.Plan(g)
        for mode in (:derived, :backend, :both), _ in 1:5
            M.run!(plan; barriers = mode)
            KernelAbstractions.synchronize(M.backend(dev))
            @test all(==(12f0), Array(out))
        end
        @test all(!, active)                 # the group is lifted in every mode
    end

    @testset "a dispatch takes a workgroup size" begin
        # The default partitions an ndrange along its first axis, so for anything
        # image-shaped a workgroup is one long row and one of the two accesses is
        # uncoalesced. A square group costs 0.061 ms here against 0.127 ms, and
        # 0.12 against 1.06 for the four-component write a real g-buffer pass
        # does. The assertion is on the result rather than the time, because how
        # much it is worth is the driver's business and that it is correct is not.
        w, h = 256, 128
        dev = M.Device(M.VulkanAPI())
        src = rand(Float32, w * h)
        transposed(group) = begin
            g = M.Graph(dev)
            a = M.Buffer(dev, src)
            out = M.Buffer(dev, zeros(Float32, w * h))
            M.compute!(g, "transpose") do p
                M.dispatch!(p, transpose_scale!,
                            (M.use(p, out; write = true), M.use(p, a; read = true),
                             Int32(w), Int32(h), 2.0f0), (w, h); group)
            end
            M.run!(M.Plan(g))
            KernelAbstractions.synchronize(M.backend(dev))
            Array(out)
        end
        want = [2.0f0 * src[(y - 1) * w + x] for x in 1:w for y in 1:h]
        @test Base.invokelatest(transposed, nothing) == want
        @test Base.invokelatest(transposed, (16, 16)) == want
    end

    @testset "a fragment shader reads a buffer per pixel" begin
        # What deferred shading is: a fullscreen pass whose vertex stage makes a
        # triangle out of `vertex_index()` and takes nothing, and whose fragment
        # stage reads the g-buffer. Without it the lighting has to go image ->
        # buffer -> compute -> blit, which is three full-screen copies a frame.
        N = 64
        dev = M.Device(M.VulkanAPI())
        g = M.Graph(dev)
        # A horizontal ramp, so a transposed read or an off-by-one shows up as a
        # gradient in the wrong direction rather than as "something was drawn".
        vals = Float32[(x - 1) / N for y in 1:N for x in 1:N]
        src = M.Buffer(dev, vals)
        img = M.Transient.Image(g, RGBA{N0f8}, (N, N))
        out = M.Transient.Buffer(g, UInt32, N * N)
        M.render!(g, "light", img => M.Discard) do p
            M.draw!(p, RAMP, (), 3;
                    frag_args = (M.use(p, src; read = true), Int32(N), Int32(N)))
        end
        M.copy!(g, "read", out, img)
        plan = M.Plan(g)
        M.run!(plan)
        KernelAbstractions.synchronize(M.backend(dev))

        px = Array(M.storage(out))
        red(i) = Float32(px[i] & 0xff) / 255
        @test red(1) < 0.02
        @test abs(red(N ÷ 2) - 0.5) < 0.03
        @test red(N) > 0.97
        @test red(N * N) > 0.97          # the last row ramps the same way

        # One push constant range, so the arguments belong to one stage. Caught
        # where the draw is written, not as one stage failing to compile against
        # an argument it never declared.
        h = M.Graph(dev)
        other = M.Transient.Image(h, RGBA{N0f8}, (N, N))
        @test_throws "both stages" M.render!(h, "both", other => M.Discard) do p
            M.draw!(p, RAMP, (M.Attribute(p, src),), 3;
                    frag_args = (M.use(p, src; read = true), Int32(N), Int32(N)))
        end
    end

    @testset "a draw count the GPU wrote" begin
        # What GPU culling needs to be worth anything: a compute pass decides how
        # much there is to draw and the draw reads that number from device
        # memory. Without it the count has to come from the host, so culling can
        # compact a list but never shorten the draw, and every culled instance
        # still costs a vertex invocation that collapses outside the clip volume.
        #
        # The element type is what makes the draw indirect, so this also pins
        # that a buffer of `DrawIndirectCommand` is allocated with
        # INDIRECT_BUFFER_BIT — without it the draw is a validation error a long
        # way from the allocation.
        N, NB = 64, 8
        dev = M.Device(M.VulkanAPI())
        kref = Ref(UInt32(3))
        g = M.Graph(dev)
        cmds = M.Buffer(dev, Mantle.DrawIndirectCommand, 1)
        img = M.Transient.Image(g, RGBA{N0f8}, (N, N))
        out = M.Transient.Buffer(g, UInt32, N * N)
        M.compute!(g, "count") do p
            M.dispatch!(p, set_draw!, (M.use(p, cmds; write = true), kref), 1)
        end
        M.render!(g, "bands", img => M.Clear((0f0, 0f0, 0f0, 1f0))) do p
            M.draw!(p, BANDS, (Int32(NB),), cmds)
        end
        M.copy!(g, "read", out, img)
        plan = M.Plan(g)

        # The draw declares `Indirect` on it, so the compute write is ordered
        # against DRAW_INDIRECT and not against the vertex stage.
        E = Mantle
        bands = only(p for p in plan.graph.passes if p.name == "bands")
        @test any(u -> last(u) === Mantle.Indirect, bands.usages)

        litbands() = begin
            M.run!(plan)
            KernelAbstractions.synchronize(M.backend(dev))
            px = reshape(Array(M.storage(out)), N, N)      # row major, so [x, y]
            count(y -> px[N ÷ 2, y] != 0xff000000, 1:N) ÷ (N ÷ NB)
        end
        @test litbands() == 3
        kref[] = UInt32(6)
        @test litbands() == 6
        kref[] = UInt32(1)
        @test litbands() == 1          # and it shrinks, so nothing is stale
    end

    @testset "an aliased region waits for what the old tenant last did" begin
        # The handover barrier is the one transition no per-resource sequence can
        # produce: the hazard is between two transients that never mention each
        # other. It used to be assumed — "wait for every stage, both directions"
        # — which is correct and unfalsifiable. It is now assembled from what the
        # placer says vacated the bytes and what the state walk says that
        # transient was last doing.
        #
        # So the discriminating question is whether the masks *follow* the old
        # tenant. Two graphs, identical except for what last touched the memory:
        # if the barrier is derived they differ, and if it is assumed they cannot.
        N = 256
        dev = M.Device(M.VulkanAPI())
        every = Mantle.VK.PipelineStageFlag2(Mantle.VK.PIPELINE_STAGE_2_ALL_COMMANDS_BIT)
        copybit = Mantle.VK.PipelineStageFlag2(Mantle.VK.PIPELINE_STAGE_2_COPY_BIT)

        # (a) the vacating transient was last *read by a shader*
        function shaderlast(dev)
            g = M.Graph(dev)
            img = M.Transient.Image(g, RGBA{N0f8}, (N, N))
            raw = M.Transient.Buffer(g, UInt32, N * N)
            later = M.Transient.Buffer(g, Float32, N * N)
            mid = M.Buffer(dev, zeros(Float32, N * N)); out = M.Buffer(dev, zeros(Float32, N * N))
            M.render!(g, "clear", img => M.Clear((0.25f0, 0f0, 0f0, 1f0))) do p
            end
            M.copy!(g, "read", raw, img)
            M.compute!(g, "consume") do p          # the last thing raw sees
                M.dispatch!(p, frombytes!, (M.use(p, mid; write = true),
                                            M.use(p, raw; read = true)), N * N)
            end
            M.compute!(g, "fill") do p             # later takes raw's bytes here
                M.dispatch!(p, scaleby!, (M.use(p, later; write = true),
                                          M.use(p, mid; read = true), 2f0), N * N)
            end
            M.compute!(g, "out") do p
                M.dispatch!(p, scaleby!, (M.use(p, out; write = true),
                                          M.use(p, later; read = true), 1f0), N * N)
            end
            (; plan = M.Plan(g), out)
        end

        # (b) the vacating transient was last *written by a copy* and never read
        function copylast(dev)
            g = M.Graph(dev)
            img = M.Transient.Image(g, RGBA{N0f8}, (N, N))
            raw = M.Transient.Buffer(g, UInt32, N * N)
            later = M.Transient.Buffer(g, Float32, N * N)
            seed = M.Buffer(dev, fill(3f0, N * N)); out = M.Buffer(dev, zeros(Float32, N * N))
            M.render!(g, "clear", img => M.Clear((0.25f0, 0f0, 0f0, 1f0))) do p
            end
            M.copy!(g, "read", raw, img)           # the last thing raw sees
            M.compute!(g, "fill") do p             # later takes raw's bytes here
                M.dispatch!(p, scaleby!, (M.use(p, later; write = true),
                                          M.use(p, seed; read = true), 2f0), N * N)
            end
            M.compute!(g, "out") do p
                M.dispatch!(p, scaleby!, (M.use(p, out; write = true),
                                          M.use(p, later; read = true), 1f0), N * N)
            end
            (; plan = M.Plan(g), out)
        end

        E = Mantle
        handover(plan) = first(passmasks(pp) for pp in plan.passes if pp.pass.name == "fill")

        a = Base.invokelatest(shaderlast, dev)
        b = Base.invokelatest(copylast, dev)
        @test M.peakbytes(a.plan) < M.naivebytes(a.plan)   # they really do share bytes
        @test M.peakbytes(b.plan) < M.naivebytes(b.plan)

        # The barrier tracks the old tenant: a copy in one, not in the other.
        @test (first(handover(b.plan)) & copybit) != zero(copybit)
        @test (first(handover(a.plan)) & copybit) == zero(copybit)

        # And neither of them, nor any other barrier in either frame, says
        # "wait for everything" — which is what both used to say.
        for plan in (a.plan, b.plan)
            masks = filter(!isnothing, [passmasks(pp) for pp in plan.passes])
            @test !isempty(masks)
            @test all(m -> (first(m) & every) == zero(every), masks)
            @test all(m -> (last(m) & every) == zero(every), masks)
        end

        # And they compute what they should, derived barriers against Lava's own.
        for s in (a, b)
            M.run!(s.plan; barriers = :backend)
            KernelAbstractions.synchronize(M.backend(dev))
            reference = Array(s.out)
            M.run!(s.plan; barriers = :derived)
            KernelAbstractions.synchronize(M.backend(dev))
            @test Array(s.out) == reference
            @test all(>(0f0), reference)
        end
    end

    @testset "a timestamp pair that cannot be from one frame is dropped" begin
        # The queries are UInt64, so an end before its start wraps rather than
        # going negative: 1.8e19 ns, which `timings` reports as 1.8e14 ms. Seen
        # in the wild while rebuilding a profiled plan with frames still in
        # flight — the pool is reset at the head of each frame's recording, and a
        # read that races it can take one word from each frame with both
        # availability bits set.
        E = Mantle
        @test E.elapsed(UInt64(100), UInt64(350), 1.0) == 250.0
        @test E.elapsed(UInt64(100), UInt64(100), 2.0) == 0.0
        @test E.elapsed(UInt64(350), UInt64(100), 1.0) === nothing
        @test E.elapsed(typemax(UInt64), UInt64(0), 1.0) === nothing
    end

    @testset "a render pass with only a depth target" begin
        # A shadow map is a pass that writes depth and nothing else, which used to
        # be "a render pass needs a colour target". The render area then has to
        # come from the depth attachment, because there is nothing else to take it
        # from, and the fragment shader has to be allowed to write no attachment.
        N = 64
        dev = M.Device(M.VulkanAPI())
        g = M.Graph(dev)
        z = M.Transient.Image(g, Float32, (N, N))
        out = M.Transient.Buffer(g, Float32, N * N)
        M.render!(g, "depth only", z => M.Clear(1f0)) do p
            M.draw!(p, DEPTHONLY, (0.35f0,), 6)
        end
        M.copy!(g, "read z", out, z)
        plan = M.Plan(g)

        pass = only(pp for pp in plan.passes if pp.pass.name == "depth only")
        @test isempty(pass.pass.targets)          # no colour attachment at all
        @test pass.pass.depth === z

        M.run!(plan)
        KernelAbstractions.synchronize(M.backend(dev))
        zs = reshape(Array(M.storage(out)), N, N)   # row major, so [x, y]
        @test zs[N ÷ 4, N ÷ 2] ≈ 0.35f0 atol = 1e-5   # the quad's depth, written
        @test zs[3N ÷ 4, N ÷ 2] ≈ 1f0                 # and the clear beside it
    end
end
