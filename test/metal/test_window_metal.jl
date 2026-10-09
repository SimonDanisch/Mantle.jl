"""
A presentation surface on Metal: a graph drawn into a window's drawable, a
drawable that follows a resize, and a window that closes.

Metal-only, and only because of what these need: a `CAMetalLayer` hands out
drawables with or without a window on screen, so on a Mac they run headless,
while the portable `Window(backend, w, h)` opens a GLFW window and needs a
display everywhere else. The portable window questions are
`test/test_window_portable.jl`, which runs in its own process with a deadline;
these belong there once they are written against `Window(backend, …)`. Everything
headless that sat beside them in `test_render_graph_metal.jl` and
`test_graphics_metal.jl` is now `test/test_render_graph.jl`,
`test/test_graphics_pipeline.jl`, `test/test_mesh_geometry_draw.jl` and
`test/test_texture_sampling.jl`, on every backend.
"""

using Test, Mantle, Metal, ColorTypes, KernelAbstractions
# Named here rather than borrowed from the `Main` that `runtests.jl` includes
# every one of these files into.
using ColorTypes.FixedPointNumbers: N0f8
using Metal: vertex_index
using GeometryBasics: Vec4f
const M = Mantle

rg_vertex(verts) = (position = verts[Int(vertex_index())], tint = (0f0, 1f0, 0f0, 1f0))
rg_fragment(inputs) = inputs.tint

const RG_DEV = M.Device(M.MetalAPI())
const RG_PIPE = M.GraphicsPipeline(; vertex = M.VertexShader(rg_vertex; outputs = (tint = NTuple{4,Float32},)),
                                     fragment = M.FragmentShader(rg_fragment),
                                     cull = M.NoCull())
const RG_TRI = M.Buffer(RG_DEV, NTuple{4,Float32}[(-0.9f0, -0.9f0, 0f0, 1f0),
                                                  ( 0.9f0, -0.9f0, 0f0, 1f0),
                                                  ( 0f0,    0.9f0, 0f0, 1f0)])

# ── the presentation surface ─────────────────────────────────────────────────
#
# A `CAMetalLayer` hands out drawables with or without a window on screen, which
# is what makes this testable at all: the layer path below needs no window
# server, and attaching that same layer to a view is one call that changes
# nothing about how frames are drawn.

using ColorTypes: BGRA

@testset "a graph renders into a presentation surface" begin
    win = Mantle.MetalWindow(BGRA{N0f8}, 64, 64; vsync = false)
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
    # lets `runrenderpass!` bind it without knowing which it has. Answered in
    # CORE by reading `s.win.views[s.win.current_image_idx + 1]`, they would be
    # a Vulkan swapchain's anatomy, which a layer has no counterpart for.
    @test M.target_extent(screen) == (64, 64)
    @test eltype(screen) == BGRA{N0f8}

    M.render!(g, "tri", screen => M.Clear((0f0, 0f0, 0f0, 1f0))) do p
        M.draw!(p, RG_PIPE, (RG_TRI,), 3)
    end
    plan = M.Plan(g)

    # Hand-bracketed, because the readback has to happen BEFORE the present,
    # and `closerun!` is what presents: after that the drawable belongs to the
    # compositor. This is `run!`'s own sequence with the readback spliced in —
    # `beforeframe!` acquires, `openrun`/`emitplan!` walk the passes, `closerun!`
    # closes the run and presents — so it stays a frame and not a special path.
    M.beforeframe!(RG_DEV, plan)
    @test M.target_view(win) isa Metal.MTL.MTLTexture
    e = M.openrun(RG_DEV, plan)
    M.emitplan!(e, plan)
    img = M.readback_window(win)
    @test size(img) == (64, 64)
    @test count(p -> ColorTypes.green(p) > 0.5, img) == 1682
    M.closerun!(RG_DEV, plan, e)
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
    pipe = M.GraphicsPipeline(; vertex = M.VertexShader(tinted_vertex; outputs = (tint = Vec4f,)),
                                fragment = M.FragmentShader(tinted_fragment),
                                cull = M.NoCull())

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

    win = Mantle.MetalWindow(BGRA{N0f8}, 64, 64; vsync = false)
    plan_win, _ = draw_into(g -> M.Surface(g, win))
    M.beforeframe!(RG_DEV, plan_win)
    e = M.openrun(RG_DEV, plan_win)
    M.emitplan!(e, plan_win)
    b = M.readback_window(win)
    M.closerun!(RG_DEV, plan_win, e)

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
    win = Mantle.MetalWindow(BGRA{N0f8}, 64, 64; vsync = false)
    g = M.Graph(RG_DEV)
    screen = M.Surface(g, win)
    z = M.Transient.Image(g, Float32, screen)
    @test size(z) == (64, 64)
    M.render!(g, "tri", screen => M.Clear((0f0, 0f0, 0f0, 1f0)), z => M.Clear(1f0)) do p
        M.draw!(p, RG_PIPE, (RG_TRI,), 3)
    end
    plan = M.Plan(g)

    # `refit!` after the acquire and not before it: the acquire is the last
    # place a resize is noticed, and a frame refit ahead of it is recorded for
    # a surface the acquire then rebuilds. `run!` takes the same two steps in
    # the same order; this splices the readback in before the present.
    function frame(w)
        M.beforeframe!(RG_DEV, plan)
        M.refit!(plan)
        e = M.openrun(RG_DEV, plan)
        M.emitplan!(e, plan)
        img = M.readback_window(w)
        M.closerun!(RG_DEV, plan, e)
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
    # count cannot match, and a depth buffer left at the previous size would
    # reject most of it rather than shrinking the fraction slightly.
    frac(img) = count(p -> ColorTypes.green(p) > 0.5, img) / length(img)
    @test isapprox(frac(big), frac(small); atol = 0.02)

    # …and `run!` does all of it: poll, refit, acquire, passes, present.
    @test resize!(win, 96, 96)
    M.run!(plan)
    @test size(z) == (96, 96)
    @test size(win) == (96, 96)
end


@testset "a window reports the extent its drawables actually have" begin
    # The other half of the bug a person found by looking at a window: the frame
    # was distorted as well as upside down. A `CAMetalLayer` derives its
    # `drawableSize` from bounds times `contentsScale` unless one is set
    # EXPLICITLY, and it has no bounds until it is put on a view — so the size it
    # was constructed with is replaced by the view's, in points times the Retina
    # factor, while `target_extent` keeps answering the constructed one.
    #
    # Nothing here needs a view: a detached layer already shows the invariant,
    # which is that the extent a window reports is the extent of the texture it
    # hands out. Every pass that strides a buffer by hand depends on it.
    win = Mantle.MetalWindow(BGRA{N0f8}, 96, 48; vsync = false)
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


# ── Closing a window takes the platform window down ─────────────────────────
#
# `close` only marked a MetalWindow closed, so the OS window stayed on screen,
# frozen on its last frame, after the render loop drawing into it had ended:
# every RayMakie window survived its close button. The Vulkan window has always
# destroyed its GLFW window here; `test_window.jl` pins that side.
@testset "closing a window destroys the platform window" begin
    w = Mantle.Window(MetalBackend(), 64, 64; title = "close test")
    @test isopen(w)
    # The close button. `isopen` says so before anything is torn down, which is
    # what a render loop polls.
    Mantle.GLFW.SetWindowShouldClose(w.handle, true)
    @test !isopen(w)
    close(w)
    @test w.handle === nothing
    close(w)                                        # idempotent
    @test !isopen(w)
end
