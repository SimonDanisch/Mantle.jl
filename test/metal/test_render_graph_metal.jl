"""
A render GRAPH on Metal, drawn by Julia shaders.

`test_graphics_metal.jl` proves the portable `draw!` — one pipeline, one target
the caller made, recorded immediately. This is the other half and the one the
deferred-shading demo needs: attachments the COMPILER placed, draws baked by the
`Pipelines` phase, and a frame executed by Mantle's own render-pass sequencing.

Three things had to become portable for it, and each is asserted below:

  * `TransientImage` lived in `src/vulkan/graph.jl` with `VK.Format` and
    `VK.Image` fields, so a render target was a Vulkan concept. It is Mantle's
    now, with the five driver-held fields as type parameters.
  * The image arena had to become a `MTLHeapTypePlacement` heap: a texture over
    buffer memory is LINEAR, and an Apple GPU cannot render into one.
  * A graph draw's arguments arrive as device arrays, so they compile and bind
    through the same ABI a kernel launch uses.
"""

using Test, Mantle, Metal, ColorTypes, KernelAbstractions
using Metal: vertex_index
using GeometryBasics: Vec4f, Vec3f
const M = Mantle
const N0f8 = ColorTypes.FixedPointNumbers.N0f8
const MEXTr = Base.get_extension(Mantle, :MantleMetalExt)

# Indexed, NOT `unsafe_load`: the point is that a graph draw's argument is a
# device ARRAY with its length, the same thing a kernel gets, so a shader
# written once compiles on either backend. That indexing carries a bounds check,
# which is what forced a graphics stage to compile at debug level 0.
rg_vertex(verts) = (position = verts[Int(vertex_index())], tint = (0f0, 1f0, 0f0, 1f0))
rg_fragment(inputs) = inputs.tint

const RG_DEV = M.Device(M.MetalAPI())
const RG_PIPE = M.GraphicsPipeline(; vertex = rg_vertex, fragment = rg_fragment,
                                     cull = M.NoCull(),
                                     varyings = (tint = NTuple{4,Float32},))
const RG_TRI = M.Buffer(RG_DEV, NTuple{4,Float32}[(-0.9f0, -0.9f0, 0f0, 1f0),
                                                  ( 0.9f0, -0.9f0, 0f0, 1f0),
                                                  ( 0f0,    0.9f0, 0f0, 1f0)])

@testset "a transient render target is portable" begin
    g = M.Graph(RG_DEV)
    color = M.Transient.Image(g, RGBA{N0f8}, (64, 64))
    z     = M.Transient.Image(g, Float32, (64, 64))

    # The type is core's, and only the element type is spelled by the caller.
    @test color isa M.TransientImage{RGBA{N0f8}}
    @test z isa M.TransientImage{Float32}
    @test eltype(color) === RGBA{N0f8}
    @test size(color) == (64, 64)
    @test M.target_extent(color) == (64, 64)
    @test M.arena(color) === M.Images()
    @test M.describe(color) == "Transient.Image(RGBA{N0f8}, (64, 64))"

    # `Float32` is depth on every backend, and Metal's format table has to agree
    # with the portable one — it answered `R32Float`, a COLOUR format, while the
    # one caller hardcoded `Depth32Float` beside it and so never noticed.
    @test M.isdepth(z)
    @test !M.isdepth(color)
    @test z.format == Metal.MTL.MTLPixelFormatDepth32Float
    @test color.format == Metal.MTL.MTLPixelFormatRGBA8Unorm
    @test MEXTr.mtlformat(RGBA{N0f8}; srgb = true) == Metal.MTL.MTLPixelFormatRGBA8Unorm_sRGB

    # What the placer reads, asked of the driver before any memory exists.
    @test M.nbytes(color) > 64 * 64 * 4 - 1
    @test M.alignment(RG_DEV, color) > 0
    @test ispow2(M.alignment(RG_DEV, color))
    # Unbound until the plan places it: the whole reason the object exists this
    # early is to be ASKED how much room it wants.
    @test color.image === nothing
    @test color.view === nothing
end

@testset "the graph places targets in a heap and draws Julia shaders into them" begin
    g = M.Graph(RG_DEV)
    color = M.Transient.Image(g, RGBA{N0f8}, (64, 64))
    M.render!(g, "tri", color => M.Clear((0f0, 0f0, 0f0, 1f0))) do p
        M.draw!(p, RG_PIPE, (RG_TRI,), 3)
    end
    plan = M.Plan(g)

    # The `Pipelines` phase compiled the draw. It used to push an empty
    # `CompiledDraw[]` on this path, so a render pass in a graph did nothing at
    # all on any backend without a batch queue.
    @test length(plan.passes) == 1
    @test length(plan.passes[end].draws) == 1

    # Placed: a real texture, in device memory, out of a placement heap rather
    # than over a buffer (`buffer === nothing` is what says it is not linear).
    @test color.image isa Metal.MTL.MTLTexture
    @test color.view === color.image
    @test color.image.storageMode == Metal.MTL.MTLStorageModePrivate
    @test color.image.buffer === nothing
    @test color.memory isa Metal.MTL.MTLHeap

    M.run!(plan)
    img = M.readback_target(color)
    @test size(img) == (64, 64)
    @test img isa Matrix{RGBA{N0f8}}

    # The same 1682 pixels the hand-written MSL reference covers in
    # `test_graphics_metal.jl` — the geometry is identical, so the count is too.
    green = count(p -> ColorTypes.green(p) > 0.5, img)
    @test green == 1682
    @test img[32, 32] == RGBA{N0f8}(0, 1, 0, 1)
    # …and the clear survived outside it, so the draw was bounded by geometry
    # rather than covering the target.
    @test count(p -> ColorTypes.green(p) <= 0.5, img) == 4096 - 1682

    # Running the same plan again must give the same frame: the arena is
    # reclaimed and re-placed between runs, and a target that came back at a
    # different offset would render into someone else's bytes.
    M.run!(plan)
    @test M.readback_target(color) == img
end

@testset "two targets that never overlap share the arena" begin
    # The whole point of placing render targets: `a` is finished before `b`
    # starts, so the compiler is free to give them the same bytes. Without a
    # heap this could not be expressed at all on Metal — `Automatic` picks its
    # own offsets, so the placement would have been advisory.
    g = M.Graph(RG_DEV)
    a = M.Transient.Image(g, RGBA{N0f8}, (64, 64))
    b = M.Transient.Image(g, RGBA{N0f8}, (64, 64))
    M.render!(g, "first",  a => M.Clear((1f0, 0f0, 0f0, 1f0))) do p
        M.draw!(p, RG_PIPE, (RG_TRI,), 3)
    end
    M.render!(g, "second", b => M.Clear((0f0, 0f0, 1f0, 1f0))) do p
        M.draw!(p, RG_PIPE, (RG_TRI,), 3)
    end
    plan = M.Plan(g)
    @test a.memory === b.memory          # one heap
    M.run!(plan)
    # Both were written by their own pass, so whatever the aliasing decided, the
    # second pass's clear is what `b` holds outside the triangle.
    ib = M.readback_target(b)
    @test count(p -> ColorTypes.green(p) > 0.5, ib) == 1682
end

@testset "a tracking target refits without naming a driver" begin
    # `refit!` was Vulkan's, and recreated a `VkImage`. It is core's now and the
    # backend supplies only `remakeimage!`.
    g = M.Graph(RG_DEV)
    src = Ref((32, 32))
    # `target_extent` of anything is what a tracking target follows; a `Ref` of a
    # size is the smallest thing that is one.
    Mantle.target_extent(r::Base.RefValue{Tuple{Int,Int}}) = r[]
    t = M.Transient.Image(g, RGBA{N0f8}, src)
    @test size(t) == (32, 32)
    @test !M.refit!(t)                   # nothing moved
    src[] = (48, 48)
    @test M.refit!(t)
    @test size(t) == (48, 48)
    @test t.req.size >= 48 * 48 * 4
    @test t.image === nothing            # unbound again; the placement is void
end

# ── the g-buffer shape ───────────────────────────────────────────────────────
#
# `bench/showcase.jl`'s deferred pass, reduced to what makes it different from
# the triangle above: three colour attachments written in one pass, and varyings
# spelled `Vec4f`/`Vec3f` rather than `NTuple`.

mrt_vertex(verts) = (position = verts[Int(vertex_index())],
                     albedo = Vec4f(1f0, 0f0, 0f0, 1f0),
                     normal = Vec3f(0f0, 1f0, 0f0))
# One value per attachment. A single colour is ALSO a four-tuple, so telling the
# two apart is on the element type — `r isa Tuple` read one `NTuple{4,Float32}`
# as four render targets, threw inside the `NamedTuple` constructor, and left a
# fragment stage that was `noreturn` and drew nothing.
mrt_fragment(inputs) = (inputs.albedo,
                        Vec4f(0f0, 0f0, 1f0, 1f0),
                        Vec4f(inputs.normal[1], inputs.normal[2], inputs.normal[3], 1f0))

const RG_MRT = M.GraphicsPipeline(; vertex = mrt_vertex, fragment = mrt_fragment,
                                    cull = M.NoCull(),
                                    varyings = (albedo = Vec4f, normal = Vec3f))

@testset "a Vec4f varying manges the same as its tuple" begin
    # The two stages link by this string and nothing else, so `Vec4f` and
    # `NTuple{4,Float32}` producing DIFFERENT strings would be a vertex output
    # no fragment could read — silently.
    @test Metal.mangle_varying("tint", Vec4f) == Metal.mangle_varying("tint", NTuple{4,Float32})
    @test Metal.air_type_mangling(Vec3f) == "Dv3_f"
    @test Metal.air_stage_name(Vec4f) == "float4"
    # …and a struct that is merely small is still refused rather than guessed at.
    @test_throws ErrorException Metal.air_type_mangling(typeof((a = 1f0, b = 2f0)))
end

@testset "three attachments in one pass" begin
    g = M.Graph(RG_DEV)
    albedo = M.Transient.Image(g, RGBA{N0f8}, (64, 64))
    matter = M.Transient.Image(g, RGBA{N0f8}, (64, 64))
    normal = M.Transient.Image(g, RGBA{N0f8}, (64, 64))
    M.render!(g, "gbuffer", albedo => M.Clear((0f0, 0f0, 0f0, 1f0)),
                            matter => M.Clear((0f0, 0f0, 0f0, 1f0)),
                            normal => M.Clear((0f0, 0f0, 0f0, 1f0))) do p
        M.draw!(p, RG_MRT, (RG_TRI,), 3)
    end
    plan = M.Plan(g)

    # Every render target needs its offset aligned to what the driver asked for,
    # and so does the REGION they sit in. It was reserved with `reserve!`'s
    # default 256 while a Metal texture wants 2048, which put the second target
    # 128 bytes short of its boundary — and `reserve!`'s fast path handed back a
    # kept region without re-checking either.
    for t in (albedo, matter, normal)
        @test t.image isa Metal.MTL.MTLTexture
    end

    M.run!(plan)
    ia, im, inm = M.readback_target(albedo), M.readback_target(matter),
                  M.readback_target(normal)
    # Each attachment got ITS OWN colour: a pipeline that wrote target 0 three
    # times, or dropped two of the three, fails here rather than looking right.
    @test ia[32, 32] == RGBA{N0f8}(1, 0, 0, 1)
    @test im[32, 32] == RGBA{N0f8}(0, 0, 1, 1)
    @test inm[32, 32] == RGBA{N0f8}(0, 1, 0, 1)
    @test count(p -> ColorTypes.red(p)   > 0.5, ia)  == 1682
    @test count(p -> ColorTypes.blue(p)  > 0.5, im)  == 1682
    @test count(p -> ColorTypes.green(p) > 0.5, inm) == 1682
end

# ── the three things the deferred demo needed beyond a triangle ──────────────

rg_depth_only(verts) = (position = verts[Int(vertex_index())],)
rg_no_colour(inputs) = nothing

const RG_SHADOW = M.GraphicsPipeline(; vertex = rg_depth_only, fragment = rg_no_colour,
                                       varyings = NamedTuple(), cull = M.NoCull(),
                                       depth = M.DepthLess())

@testset "a depth-only pass writes depth and a copy pass reads it back" begin
    # A shadow pass has no colour attachment at all, and Metal spells that as a
    # pipeline with NO fragment function — compiling a stage that returns an
    # empty struct is not something AIR has.
    #
    # `copy!` is the other half: five of showcase's twenty passes are copies,
    # and the shared graph did not execute them at all. `run!` walked dispatches
    # and draws, so on any backend without a batch queue the deferred half of
    # that frame read an empty g-buffer.
    g = M.Graph(RG_DEV)
    z   = M.Transient.Image(g, Float32, (64, 64))
    zpx = M.Transient.Buffer(g, Float32, 64 * 64)
    tri = M.Buffer(RG_DEV, NTuple{4,Float32}[(-0.9f0, -0.9f0, 0.25f0, 1f0),
                                             ( 0.9f0, -0.9f0, 0.25f0, 1f0),
                                             ( 0f0,    0.9f0, 0.25f0, 1f0)])
    M.render!(g, "shadow", z => M.Clear(1f0)) do p
        M.draw!(p, RG_SHADOW, (tri,), 3)
    end
    M.copy!(g, "read shadow", zpx, z)
    plan = M.Plan(g)
    M.run!(plan)

    d = Array(M.storage(zpx))
    @test length(d) == 64 * 64
    # 0.25 inside the triangle, the clear outside, and the same 1682 pixels.
    @test minimum(d) ≈ 0.25f0
    @test maximum(d) ≈ 1.0f0
    @test count(x -> x < 0.9f0, d) == 1682
end

@kernel function rg_setcount!(cmd, n::Int32)
    cmd[1] = M.DrawIndirectCommand(UInt32(n), UInt32(1), UInt32(0), UInt32(0))
end

@testset "an indirect draw reads a count a compute pass wrote" begin
    # The form where the host never learns the count. Two things this pins:
    #
    #   * the indirect buffer must be made RESIDENT for the render encoder.
    #     Nothing in the shader names it — the command processor reads it — so
    #     it is easy to miss, and missing it does not fail: the GPU reads
    #     whatever is mapped, takes the vertex count from it, and asks for
    #     however many that is. Four billion vertices is a command buffer that
    #     never completes.
    #   * the compute pass that writes it has to have run first, which is the
    #     graph's ordering and the reason the render path shares ONE queue with
    #     compute rather than opening a second.
    g = M.Graph(RG_DEV)
    color = M.Transient.Image(g, RGBA{N0f8}, (64, 64))
    cmd = M.Buffer(RG_DEV, M.DrawIndirectCommand, 1)
    M.compute!(g, "count") do p
        M.dispatch!(p, rg_setcount!, (M.use(p, cmd; write = true), Int32(3)), 1)
    end
    M.render!(g, "tri", color => M.Clear((0f0, 0f0, 0f0, 1f0))) do p
        M.draw!(p, RG_PIPE, (RG_TRI,), cmd)
    end
    plan = M.Plan(g)
    M.run!(plan)

    @test Array(cmd)[1].vertices == 3
    @test count(p -> ColorTypes.green(p) > 0.5, M.readback_target(color)) == 1682
end

@testset "the FIRST run of a plan is already correct" begin
    # Not a tautology: it was not. The image arena's heap and its textures were
    # created `Untracked`, on the reasoning that Mantle's graph had emitted the
    # dependency — and the KernelAbstractions path this backend runs on does not
    # consume the barrier phase at all. The driver was free to overlap a
    # readback with the render pass feeding it, and did, exactly once per
    # process: the first frame came back as the untouched texture, alpha and
    # all, and every frame after it was right.
    #
    # Committing to one queue orders when command buffers START. Tracking is
    # what keeps them from overlapping.
    g = M.Graph(RG_DEV)
    color = M.Transient.Image(g, RGBA{N0f8}, (64, 64))
    M.render!(g, "tri", color => M.Clear((0f0, 0f0, 0f0, 1f0))) do p
        M.draw!(p, RG_PIPE, (RG_TRI,), 3)
    end
    plan = M.Plan(g)
    M.run!(plan)
    first_frame = M.readback_target(color)
    # The clear alone would fail this: an untouched target reads as zeros, and
    # the clear colour has alpha 1.
    @test all(p -> ColorTypes.alpha(p) == 1, first_frame)
    @test count(p -> ColorTypes.green(p) > 0.5, first_frame) == 1682
    M.run!(plan)
    @test M.readback_target(color) == first_frame
end

@testset "an arena serves tenants that need more alignment than it defaults to" begin
    # Two 2048-aligned targets in one arena. `reserve!` defaults to 256 and its
    # fast path handed back a kept region without re-checking, so the second
    # target landed 128 bytes short of its boundary and Metal refused to place
    # it. Sizes that are NOT a multiple of the alignment are what expose it —
    # 64×64 RGBA8 is 16512 bytes and 2048 does not divide it.
    g = M.Graph(RG_DEV)
    a = M.Transient.Image(g, RGBA{N0f8}, (64, 64))
    b = M.Transient.Image(g, RGBA{N0f8}, (64, 64))
    M.render!(g, "a", a => M.Clear((0f0, 0f0, 0f0, 1f0))) do p
        M.draw!(p, RG_PIPE, (RG_TRI,), 3)
    end
    M.render!(g, "b", b => M.Clear((0f0, 0f0, 0f0, 1f0))) do p
        M.draw!(p, RG_PIPE, (RG_TRI,), 3)
    end
    plan = M.Plan(g)
    for t in (a, b)
        @test t.image isa Metal.MTL.MTLTexture
        # The offset the placer chose has to satisfy what the driver asked for.
        @test M.alignment(RG_DEV, t) == t.req.alignment
    end
    M.run!(plan)
    @test count(p -> ColorTypes.green(p) > 0.5, M.readback_target(b)) == 1682
end

@testset "a plan can be profiled on a backend without timestamp queries" begin
    # `makeprofiler` answered `nothing` for anything but Vulkan, so
    # `Plan(g; profile = true)` quietly produced a plan with no profiler and
    # `timings` did not exist to be called. The host side is something every
    # backend can measure, and it is what says where a frame goes when there
    # are no GPU timestamps: it is the time `run!` spends in a pass, including
    # whatever the backend waits for inside it.
    g = M.Graph(RG_DEV)
    color = M.Transient.Image(g, RGBA{N0f8}, (64, 64))
    M.render!(g, "tri", color => M.Clear((0f0, 0f0, 0f0, 1f0))) do p
        M.draw!(p, RG_PIPE, (RG_TRI,), 3)
    end
    M.copy!(g, "read", M.Transient.Buffer(g, RGBA{N0f8}, 64 * 64), color)

    plain = M.Plan(g)
    @test plain.profiler === nothing
    # Asking a plan that was not built for it says so, rather than returning
    # zeros that look like a fast frame.
    @test_throws ArgumentError M.timings(plain)

    plan = M.Plan(g; profile = true)
    @test plan.profiler !== nothing
    for _ in 1:3
        M.run!(plan)
    end
    ts = M.timings(plan)
    @test length(ts) == length(plan.passes)
    @test [t.name for t in ts] == ["tri", "read"]
    @test [t.kind for t in ts] == [:render, :copy]
    @test all(t -> t.samples == 3, ts)
    @test all(t -> t.host_ms >= 0, ts)
    # This backend answers the GPU question too, without a single timestamp
    # query: a `MTLCommandBuffer` reports `GPUStartTime` and `GPUEndTime` once
    # it has run, so `gpupasstime!` submits what the pass recorded and waits.
    @test all(t -> !isnan(t.gpu_ms) && t.gpu_ms >= 0, ts)
    # And the host number is the larger of the two, because the wait that
    # produced the GPU number happens inside it. That ordering is the whole
    # reason a profiled frame is slower than a real one.
    @test all(t -> t.host_ms >= t.gpu_ms, ts)
    # …and the frame still renders while being measured.
    @test count(p -> ColorTypes.green(p) > 0.5, M.readback_target(color)) == 1682
end

# ── the presentation surface ─────────────────────────────────────────────────
#
# A `CAMetalLayer` hands out drawables with or without a window on screen, which
# is what makes this testable at all: the layer path below needs no window
# server, and attaching that same layer to a view is one call that changes
# nothing about how frames are drawn.

using ColorTypes: BGRA

@testset "a graph renders into a presentation surface" begin
    win = MEXTr.MetalWindow(BGRA{N0f8}, 64, 64; vsync = false)
    @test win isa M.Window
    @test size(win) == (64, 64)
    @test eltype(win) == BGRA{N0f8}
    @test isopen(win)
    # No drawable between frames, and asking says so rather than handing back a
    # stale texture from the frame before.
    @test_throws ErrorException M.target_view(win)

    g = M.Graph(RG_DEV)
    screen = M.Surface(g, win)
    # A surface answers the same four questions an image does — that is what
    # lets `runrenderpass!` bind it without knowing which it has. They used to
    # be answered in CORE by reading `s.win.views[s.win.current_image_idx + 1]`,
    # a Vulkan swapchain's anatomy, which a layer has no counterpart for.
    @test M.target_extent(screen) == (64, 64)
    @test eltype(screen) == BGRA{N0f8}

    M.render!(g, "tri", screen => M.Clear((0f0, 0f0, 0f0, 1f0))) do p
        M.draw!(p, RG_PIPE, (RG_TRI,), 3)
    end
    plan = M.Plan(g)

    # Hand-bracketed, because the readback has to happen BEFORE the present:
    # after presenting, the drawable belongs to the compositor.
    M.beginframe!(win)
    M.acquire_next_image!(win)
    @test M.target_view(win) isa Metal.MTL.MTLTexture
    for pp in plan.passes
        M.runpass!(plan, pp)
    end
    img = M.readback_window(win)
    @test size(img) == (64, 64)
    @test count(p -> ColorTypes.green(p) > 0.5, img) == 1682
    M.present_frame!(RG_DEV, win)
    # …and the drawable is gone again once presented.
    @test_throws ErrorException M.target_view(win)

    # `run!` brackets all three itself, which is the point: a graph with a
    # surface is a frame, and the shared loop is what makes it one. It walked
    # passes and nothing else before, so a `Surface` was an attachment nobody
    # acquired.
    for _ in 1:3
        M.run!(plan)
    end
    @test isopen(win)
    close(win)
    @test !isopen(win)
    @test_throws ErrorException M.beginframe!(win)
end

# Top level, not inside the testset: a shader defined in a local scope closes
# over what it reads, and a closure's captured field is a dynamic `getproperty`
# the GPU compiler refuses.
const RG_TINT = Vec4f(0.8f0, 0.3f0, 0.1f0, 1f0)      # distinct in all three
tinted_vertex(pos) = (position = pos[Int(vertex_index())], tint = RG_TINT)
tinted_fragment(inputs) = inputs.tint

@testset "a BGRA surface and an RGBA target show the same colours" begin
    # The pixel type is the format, and the two differ in BYTE ORDER: a display
    # wants `BGRA{N0f8}` and an offscreen target is usually `RGBA{N0f8}`. The
    # shader writes `(r, g, b, a)` either way and the attachment's format is
    # what reorders them — so a pipeline compiled for the wrong one swaps red
    # and blue, which looks like a plausible image and is why this compares
    # CHANNELS rather than eyeballing.
    pipe = M.GraphicsPipeline(; vertex = tinted_vertex, fragment = tinted_fragment,
                                cull = M.NoCull(), varyings = (tint = Vec4f,))

    function draw_into(target_of)
        g = M.Graph(RG_DEV)
        t = target_of(g)
        M.render!(g, "tri", t => M.Clear((0f0, 0f0, 0f0, 1f0))) do p
            M.draw!(p, pipe, (RG_TRI,), 3)
        end
        return M.Plan(g), t
    end

    plan_off, off = draw_into(g -> M.Transient.Image(g, RGBA{N0f8}, (64, 64)))
    M.run!(plan_off)
    a = M.readback_target(off)

    win = MEXTr.MetalWindow(BGRA{N0f8}, 64, 64; vsync = false)
    plan_win, _ = draw_into(g -> M.Surface(g, win))
    M.beginframe!(win); M.acquire_next_image!(win)
    for pp in plan_win.passes
        M.runpass!(plan_win, pp)
    end
    b = M.readback_window(win)
    M.present_frame!(RG_DEV, win)

    @test size(a) == size(b)
    for ch in (ColorTypes.red, ColorTypes.green, ColorTypes.blue, ColorTypes.alpha)
        @test all(p -> ch(p[1]) == ch(p[2]), zip(a, b))
    end
    # …and the tint really is asymmetric, so the test above could have failed.
    @test ColorTypes.red(a[32, 32]) != ColorTypes.blue(a[32, 32])
end

@testset "a surface resizes and the plan follows it" begin
    # The case a fixed-size demo never reaches: an attachment declared to FOLLOW
    # the surface. `Transient.Image(g, Float32, screen)` is a depth buffer for a
    # window, and a window that grows without it is a depth test against a
    # buffer that no longer covers the render area.
    #
    # `refit!(::Plan)` — refit the transients, recompile, adopt the new
    # placement, re-register as a tenant — was `refit!(pl::Plan{LavaDevice})`.
    # Every line of it is the graph's; the one driver-shaped line, rebuilding
    # the argument memory, already had a hook.
    win = MEXTr.MetalWindow(BGRA{N0f8}, 64, 64; vsync = false)
    g = M.Graph(RG_DEV)
    screen = M.Surface(g, win)
    z = M.Transient.Image(g, Float32, screen)
    @test size(z) == (64, 64)
    M.render!(g, "tri", screen => M.Clear((0f0, 0f0, 0f0, 1f0)), z => M.Clear(1f0)) do p
        M.draw!(p, RG_PIPE, (RG_TRI,), 3)
    end
    plan = M.Plan(g)

    function frame(w)
        M.beginframe!(w); M.refit!(plan); M.acquire_next_image!(w)
        for pp in plan.passes
            M.runpass!(plan, pp)
        end
        img = M.readback_window(w)
        M.present_frame!(RG_DEV, w)
        return img
    end

    small = frame(win)
    @test size(small) == (64, 64)
    @test count(p -> ColorTypes.green(p) > 0.5, small) == 1682

    # `resize!` reports whether anything moved, so a frame loop can tell a
    # resize from a redraw.
    @test resize!(win, 128, 96)
    @test !resize!(win, 128, 96)
    big = frame(win)
    @test size(big) == (128, 96)
    @test size(z) == (128, 96)          # the depth buffer came along
    # The same triangle covers the same FRACTION of a bigger frame; the exact
    # count cannot match, and a depth buffer left at the old size would have
    # rejected most of it rather than shrinking the fraction slightly.
    frac(img) = count(p -> ColorTypes.green(p) > 0.5, img) / length(img)
    @test isapprox(frac(big), frac(small); atol = 0.02)

    # …and `run!` does all of it: poll, refit, acquire, passes, present.
    @test resize!(win, 96, 96)
    M.run!(plan)
    @test size(z) == (96, 96)
    @test size(win) == (96, 96)
end

# ── how the frame is submitted ───────────────────────────────────────────────

@kernel function rg_touch!(x)
    i = @index(Global)
    @inbounds x[i] += 1f0
end

@testset "consecutive render and copy passes share one command buffer" begin
    # A command buffer per pass is nine of them in the deferred demo, and a
    # submission costs far more than the drawing in it — the whole frame's GPU
    # work is 5 ms and the frame took ten times that. Encoders are cheap and a
    # command buffer takes as many as you like in sequence, so only a DISPATCH
    # forces one out: buffers on a queue run in COMMIT order, and drawing still
    # open when a dispatch is committed would run after it.
    g = M.Graph(RG_DEV)
    a = M.Transient.Image(g, RGBA{N0f8}, (64, 64))
    px = M.Transient.Buffer(g, RGBA{N0f8}, 64 * 64)
    b = M.Transient.Image(g, RGBA{N0f8}, (64, 64))
    M.render!(g, "one", a => M.Clear((0f0, 0f0, 0f0, 1f0))) do p
        M.draw!(p, RG_PIPE, (RG_TRI,), 3)
    end
    M.copy!(g, "read", px, a)
    M.render!(g, "two", b => M.Clear((0f0, 0f0, 0f0, 1f0))) do p
        M.draw!(p, RG_PIPE, (RG_TRI,), 3)
    end
    plan = M.Plan(g)

    M.submit!(RG_DEV)                       # start from nothing open
    @test MEXTr.OPEN_CB[] === nothing
    seen = Any[]
    for pp in plan.passes
        M.runpass!(plan, pp)
        push!(seen, MEXTr.OPEN_CB[])
    end
    # All three passes wrote into the SAME buffer, and it is still open.
    @test all(cb -> cb === seen[1], seen)
    @test seen[1] !== nothing
    # `submit!` is what closes it, and it is idempotent.
    M.submit!(RG_DEV)
    @test MEXTr.OPEN_CB[] === nothing
    M.submit!(RG_DEV)
    @test MEXTr.OPEN_CB[] === nothing
    # …and the frame is still correct.
    @test count(p -> ColorTypes.green(p) > 0.5, M.readback_target(b)) == 1682
end

@testset "a dispatch forces the open command buffer out" begin
    # The ordering that makes `submit!` necessary rather than an optimisation.
    g = M.Graph(RG_DEV)
    a = M.Transient.Image(g, RGBA{N0f8}, (64, 64))
    scratch = M.Buffer(RG_DEV, zeros(Float32, 16))
    M.render!(g, "before", a => M.Clear((0f0, 0f0, 0f0, 1f0))) do p
        M.draw!(p, RG_PIPE, (RG_TRI,), 3)
    end
    M.compute!(g, "between") do p
        M.dispatch!(p, rg_touch!, (M.use(p, scratch; write = true),), 16)
    end
    M.render!(g, "after", a => M.Keep) do p
        M.draw!(p, RG_PIPE, (RG_TRI,), 3)
    end
    plan = M.Plan(g)

    M.submit!(RG_DEV)
    M.runpass!(plan, plan.passes[1])
    first_cb = MEXTr.OPEN_CB[]
    @test first_cb !== nothing
    M.runpass!(plan, plan.passes[2])        # the dispatch
    # Closed by `run!`'s call to `submit!` before the dispatch — the drawing
    # cannot still be open, or it would be committed after the work that
    # follows it.
    @test MEXTr.OPEN_CB[] === nothing
    M.runpass!(plan, plan.passes[3])
    @test MEXTr.OPEN_CB[] !== nothing
    @test MEXTr.OPEN_CB[] !== first_cb
    M.submit!(RG_DEV)
    @test Array(scratch)[1] == 1f0
end

@testset "an unprofiled frame holds no command buffers" begin
    # `gpupasstime!` reads the buffers a pass committed, and it is the only
    # thing that drains them — so collecting unconditionally was a leak that
    # nothing bounded: ~8 per frame, held alive, stopping Metal recycling them.
    # The frame time climbed with the frame count, 532 ms over fifty frames and
    # 1040 over the next fifty, for a frame whose GPU work is 5 ms.
    g = M.Graph(RG_DEV)
    color = M.Transient.Image(g, RGBA{N0f8}, (64, 64))
    M.render!(g, "tri", color => M.Clear((0f0, 0f0, 0f0, 1f0))) do p
        M.draw!(p, RG_PIPE, (RG_TRI,), 3)
    end
    plan = M.Plan(g)                        # NOT profiled
    @test plan.profiler === nothing
    empty!(MEXTr.COMMITTED)
    for _ in 1:20
        M.run!(plan)
    end
    # Either collection is off, or it is bounded — never unbounded.
    @test length(MEXTr.COMMITTED) <= MEXTr.COMMITTED_MAX
    # And with a profiled plan, every pass drains what it committed.
    prof = M.Plan(g; profile = true)
    for _ in 1:5
        M.run!(prof)
    end
    @test isempty(MEXTr.COMMITTED)
end

@testset "clip space is Mantle's, not the backend's" begin
    # The bug a person found by looking at a window: the scene was rendered
    # upside down. **Mantle's clip space is Vulkan's — +y is DOWN the screen** —
    # and Metal's is the other one, so the same `position` came out mirrored.
    #
    # Nothing automated caught it. The pixel counts were identical, the
    # window/offscreen comparison agreed perfectly (both were equally wrong),
    # and a mirrored scene still looks like a scene. This asserts the ORIENTATION
    # rather than the count.
    g = M.Graph(RG_DEV)
    color = M.Transient.Image(g, RGBA{N0f8}, (64, 64))
    # Apex at NDC +0.9, base at −0.9. Narrow end and wide end are the marker.
    M.render!(g, "tri", color => M.Clear((0f0, 0f0, 0f0, 1f0))) do p
        M.draw!(p, RG_PIPE, (RG_TRI,), 3)
    end
    plan = M.Plan(g)
    M.run!(plan)
    img = M.readback_target(color)
    width(y) = count(x -> ColorTypes.green(img[x, y]) > 0.5, 1:64)

    # Row 1 of a readback is the TOP of the image. Under Mantle's convention the
    # apex (+y) belongs at the BOTTOM, so the triangle must be narrow at large y
    # and wide at small y.
    @test width(6) > 40          # base, near the top
    @test width(58) < 10         # apex, near the bottom
    @test width(6) > width(32) > width(58)

    # `clip_y` is the same transform, exposed so a shader that does the viewport
    # step BY HAND — a shadow lookup turning `sunvp * world` into a texel row —
    # applies the one the rasteriser applied. It is its own inverse.
    @test M.clip_y(M.clip_y(0.37f0)) == 0.37f0
    @test abs(M.clip_y(0.37f0)) == 0.37f0
end

@testset "a window reports the extent its drawables actually have" begin
    # The other half of the bug a person found by looking at a window: the frame
    # was distorted as well as upside down. A `CAMetalLayer` derives its
    # `drawableSize` from bounds times `contentsScale` unless one is set
    # EXPLICITLY, and it has no bounds until it is put on a view — so the size
    # it was constructed with was replaced by the view's, in points times the
    # Retina factor. `target_extent` kept answering the old one.
    #
    # Nothing here needs a view: a detached layer already shows the invariant,
    # which is that the extent a window reports is the extent of the texture it
    # hands out. Every pass that strides a buffer by hand depends on it.
    win = MEXTr.MetalWindow(BGRA{N0f8}, 96, 48; vsync = false)
    @test M.target_extent(win) == (96, 48)

    M.beginframe!(win)
    M.acquire_next_image!(win)
    tex = M.target_view(win)
    @test (Int(tex.width), Int(tex.height)) == M.target_extent(win)
    M.present_frame!(RG_DEV, win)

    # And after a resize, still.
    resize!(win, 64, 112)
    @test M.target_extent(win) == (64, 112)
    M.beginframe!(win)
    M.acquire_next_image!(win)
    tex2 = M.target_view(win)
    @test (Int(tex2.width), Int(tex2.height)) == (64, 112)
    M.present_frame!(RG_DEV, win)
end
