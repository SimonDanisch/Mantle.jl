# Graphics pipelines on the device: compile, draw, read back.
#
# Immediate draws into an offscreen `Framebuffer` — `Mantle.draw!(dev, pipeline,
# OffscreenTarget(fb), count)` — so no window is needed and lavapipe runs it. The
# same assertions on every backend that rasterises; `supports_graphics` is the
# gate. The render GRAPH's draws are `test_render_graph.jl`, mesh and geometry
# pipelines `test_mesh_geometry_draw.jl`, sampling `test_texture_sampling.jl`.

using Test
using Mantle
using GeometryBasics
using ColorTypes: RGBA, BGRA
using ColorTypes.FixedPointNumbers: N0f8
include(joinpath(@__DIR__, "testbackend.jl"))

# `readback_framebuffer` answers the attachment's COLORANT, so a channel is named
# (`q.g`) rather than indexed. `n8` keeps the 8-bit thresholds the exact byte
# values they have always been.
n8(b::UInt8) = reinterpret(N0f8, b)

# Draw once into a fresh offscreen framebuffer and read it back, as a
# `(width, height)` matrix of the attachment's element type.
function draw_and_readback(dev, pipeline, vertex_count;
        width = 16, height = 16, args = (), frag_args = (),
        clear_color = (0f0, 0f0, 0f0, 1f0),
        color_format = RGBA{Float32},
        depth = false, instances = 1)
    fb = Mantle.Framebuffer(TESTBACKEND, width, height; depth, color_format)
    Mantle.draw!(dev, pipeline, OffscreenTarget(fb), vertex_count;
                 args, frag_args, instances, clear_color)
    Mantle.flush!(dev)
    return Mantle.readback_framebuffer(fb)
end

# Positions straight out of a device array, so the vertex path adds no
# arithmetic. Top level: a shader defined in a local scope closes over what it
# reads.
posbuf_vertex(verts) =
    (position = verts[Int(Mantle.vertex_index())], tint = (0f0, 1f0, 0f0, 1f0))
# Reads the varying rather than repeating the constant, so a vertex-to-fragment
# link that silently failed shows as a black triangle.
posbuf_fragment(inputs) = inputs.tint

@testset "a capability is answered the same for a device and its backend" begin
    # Each question has an untyped `false` default and a `::Device` forwarder in
    # core; a backend that answered for its backend object and not for its device
    # gave every portable caller the degrade path on a device that could do it.
    dev = Mantle.Device(TESTBACKEND)
    for q in (Mantle.supports_graphics, Mantle.supports_geometry_stage,
              Mantle.supports_tessellation, Mantle.supports_mesh_pipeline,
              Mantle.supports_batch_queue)
        @test q(dev) == q(TESTBACKEND)
    end
    # The default is `false`, so a backend that answers nothing answers no —
    # which is the safe direction for a capability.
    @test !Mantle.supports_geometry_stage(:something_that_is_not_a_backend)
    # A backend with no batch queue says so when asked for one, rather than
    # handing back a stub that records nothing. The two questions came apart
    # when Metal learned to draw: it rasterises and has no command pool.
    Mantle.supports_batch_queue(TESTBACKEND) ||
        @test_throws ArgumentError Mantle.allocate_batch_queue!(TESTBACKEND)
end

@testset "Graphics Pipeline" begin
    dev = Mantle.Device(TESTBACKEND)
    if !Mantle.supports_graphics(dev)
        @info "no graphics pipeline on this backend; skipping"
        @test_skip Mantle.supports_graphics(dev)
        return
    end

    # ── Basic vertex + fragment ──

    @testset "solid color triangle" begin
        function solid_vert()
            vid = Mantle.vertex_index() - Int32(1)
            x = Float32(Int32(vid & Int32(1)) * 4 - 1)
            y = Float32(Int32((vid >> Int32(1)) & Int32(1)) * 4 - 1)
            return (position = Vec4f(x, y, 0.5f0, 1.0f0),)
        end
        solid_frag(_inputs) = Vec4f(1.0f0, 1.0f0, 0.0f0, 1.0f0)
        pip = GraphicsPipeline(; vertex = VertexShader(solid_vert),
                                 fragment = FragmentShader(solid_frag),
                                 blend = Opaque(),
                                 cull = NoCull(),
                                 depth = DepthOff())
        pixels = draw_and_readback(dev, pip, 3)
        @test all(p -> p.g ≈ 1f0, pixels)  # all green
        @test all(p -> p.alpha ≈ 1f0, pixels)  # alpha = 1
    end

    @testset "clear color works" begin
        # Draw zero vertices — only clear should be visible
        noop_vert() = (position = Vec4f(0f0, 0f0, 0f0, 1f0),)
        noop_frag(_inputs) = Vec4f(0f0, 0f0, 0f0, 0f0)
        pip = GraphicsPipeline(; vertex = VertexShader(noop_vert),
                                 fragment = FragmentShader(noop_frag),
                                 blend = Opaque(),
                                 cull = NoCull(),
                                 depth = DepthOff())
        pixels = draw_and_readback(dev, pip, 0; clear_color=(0.25f0, 0.5f0, 0.75f0, 1f0))
        @test all(p -> p.r ≈ 0.25f0 && p.g ≈ 0.5f0 && p.b ≈ 0.75f0, pixels)
    end

    # ── Device-array arguments ──

    @testset "a vertex stage reads a device array argument" begin
        function bda_vert(positions)
            vid = Mantle.vertex_index()
            @inbounds p = positions[vid]
            return (position = Vec4f(p[1], p[2], p[3], 1.0f0),
                    color = Vec4f(1.0f0, 0.0f0, 0.0f0, 1.0f0))
        end
        bda_frag(inputs) = inputs.color
        pip = GraphicsPipeline(; vertex = VertexShader(bda_vert; outputs = (color = Vec4f,)),
                                 fragment = FragmentShader(bda_frag),
                                 blend = Opaque(),
                                 cull = NoCull(),
                                 depth = DepthOff())
        # Fullscreen triangle
        positions = Mantle.devicearray(TESTBACKEND, Vec3f[Vec3f(-1,-1,0), Vec3f(3,-1,0), Vec3f(-1,3,0)])
        pixels = draw_and_readback(dev, pip, 3; args=(positions,))
        @test all(p -> p.r ≈ 1f0, pixels)  # all red
    end

    @testset "vertex-to-fragment varying" begin
        function vary_vert()
            vid = Mantle.vertex_index() - Int32(1)
            x = Float32(Int32(vid & Int32(1)) * 4 - 1)
            y = Float32(Int32((vid >> Int32(1)) & Int32(1)) * 4 - 1)
            u = (x + 1f0) * 0.5f0
            v = (y + 1f0) * 0.5f0
            return (position = Vec4f(x, y, 0.5f0, 1.0f0), uv = Vec2f(u, v))
        end
        vary_frag(inputs) = Vec4f(inputs.uv[1], inputs.uv[2], 0f0, 1f0)
        pip = GraphicsPipeline(; vertex = VertexShader(vary_vert; outputs = (uv = Vec2f,)),
                                 fragment = FragmentShader(vary_frag),
                                 blend = Opaque(),
                                 cull = NoCull(),
                                 depth = DepthOff())
        pixels = draw_and_readback(dev, pip, 3; width=8, height=8)
        # Center pixel should have UV ≈ (0.5, 0.5)
        center = pixels[4, 4]
        @test center.r ≈ 0.5f0 atol=0.15  # u
        @test center.g ≈ 0.5f0 atol=0.15  # v
        # Corner (0,0) should have UV near (0, 0)
        @test pixels[1,1].r < 0.2f0  # u near 0
        @test pixels[1,1].g < 0.2f0  # v near 0
    end

    @testset "fragment uses frag_coord" begin
        function fc_vert()
            vid = Mantle.vertex_index() - Int32(1)
            x = Float32(Int32(vid & Int32(1)) * 4 - 1)
            y = Float32(Int32((vid >> Int32(1)) & Int32(1)) * 4 - 1)
            return (position = Vec4f(x, y, 0.5f0, 1.0f0),)
        end
        function fc_frag(_inputs)
            fx = Mantle.frag_coord_x()
            fy = Mantle.frag_coord_y()
            # Normalize to [0,1] range using 16x16 framebuffer
            return Vec4f(fx / 16f0, fy / 16f0, 0f0, 1f0)
        end
        pip = GraphicsPipeline(; vertex = VertexShader(fc_vert),
                                 fragment = FragmentShader(fc_frag),
                                 blend = Opaque(),
                                 cull = NoCull(),
                                 depth = DepthOff())
        pixels = draw_and_readback(dev, pip, 3)
        # Pixel at (8,8) should have frag_coord ≈ (8.5, 8.5) → normalized ≈ (0.53, 0.53)
        p = pixels[8, 8]
        @test 0.4f0 < p.r < 0.7f0
        @test 0.4f0 < p.g < 0.7f0
    end

    # ── Blend modes ──

    @testset "alpha blend" begin
        function ab_vert()
            vid = Mantle.vertex_index() - Int32(1)
            x = Float32(Int32(vid & Int32(1)) * 4 - 1)
            y = Float32(Int32((vid >> Int32(1)) & Int32(1)) * 4 - 1)
            return (position = Vec4f(x, y, 0.5f0, 1.0f0),)
        end
        # Semi-transparent red
        ab_frag(_inputs) = Vec4f(1f0, 0f0, 0f0, 0.5f0)
        pip = GraphicsPipeline(; vertex = VertexShader(ab_vert),
                                 fragment = FragmentShader(ab_frag),
                                 blend = AlphaBlend(),
                                 cull = NoCull(),
                                 depth = DepthOff())
        # Clear to white, draw semi-transparent red
        pixels = draw_and_readback(dev, pip, 3; clear_color=(1f0, 1f0, 1f0, 1f0))
        # Result should be blended: red = 0.5*1 + 0.5*1 = 1, green = 0.5*0 + 0.5*1 = 0.5
        p = pixels[8, 8]
        @test p.r ≈ 1f0 atol=0.05  # red
        @test p.g ≈ 0.5f0 atol=0.05  # green (blended)
    end

    # ── Depth test ──

    @testset "depth test" begin
        # Draw two fullscreen triangles: first at z=0.7 (red), second at z=0.3 (blue).
        # With depth < test, closer (z=0.3) blue should win.
        #
        # `depth_clear = nothing` on the second draw is what makes this a test of
        # the depth test. Clearing depth per draw — which is what the attachment
        # did unconditionally — leaves every fragment passing against 1.0, so the
        # later draw always won and the assertion held whatever the z values were.
        function depth_vert(color::Vec4f, z_val::Float32)
            vid = Mantle.vertex_index() - Int32(1)
            x = Float32(Int32(vid & Int32(1)) * 4 - 1)
            y = Float32(Int32((vid >> Int32(1)) & Int32(1)) * 4 - 1)
            return (position = Vec4f(x, y, z_val, 1.0f0), color = color)
        end
        depth_frag(inputs) = inputs.color
        pip = GraphicsPipeline(; vertex = VertexShader(depth_vert; outputs = (color = Vec4f,)),
                                 fragment = FragmentShader(depth_frag),
                                 blend = Opaque(),
                                 cull = NoCull(),
                                 depth = DepthLess())

        fb = Mantle.Framebuffer(TESTBACKEND, 8, 8; depth=true, color_format=RGBA{Float32})
        target = OffscreenTarget(fb)

        # Draw far red at z=0.7
        Mantle.draw!(dev, pip, target, 3;
            args=(Vec4f(1f0, 0f0, 0f0, 1f0), 0.7f0),
            clear_color=(0f0, 0f0, 0f0, 1f0))
        # Draw close blue at z=0.3 (no clear — load previous colour and depth)
        Mantle.draw!(dev, pip, target, 3;
            args=(Vec4f(0f0, 0f0, 1f0, 1f0), 0.3f0),
            clear_color=nothing, depth_clear=nothing)
        Mantle.flush!(dev)

        pixels = Mantle.readback_framebuffer(fb)
        # Blue should win (closer)
        # `.b`/`.r`, not `p[3]`/`p[1]`: readback returns the element type
        # `eltypeof` names — `RGBA{Float32}` here — so the channels have names.
        # A second format table would return `NTuple{4, Float32}` here and
        # disagree with `eltypeof` about the same format.
        p = pixels[4, 4]
        @test p.b ≈ 1f0 atol=0.05
        @test p.r ≈ 0f0 atol=0.05

        # The other order is the half that a per-draw depth clear could not fail:
        # near first, far second, and the far one must be rejected.
        fb2 = Mantle.Framebuffer(TESTBACKEND, 8, 8; depth=true, color_format=RGBA{Float32})
        t2 = OffscreenTarget(fb2)
        Mantle.draw!(dev, pip, t2, 3;
            args=(Vec4f(0f0, 0f0, 1f0, 1f0), 0.3f0),
            clear_color=(0f0, 0f0, 0f0, 1f0))
        Mantle.draw!(dev, pip, t2, 3;
            args=(Vec4f(1f0, 0f0, 0f0, 1f0), 0.7f0),
            clear_color=nothing, depth_clear=nothing)
        Mantle.flush!(dev)

        q = Mantle.readback_framebuffer(fb2)[4, 4]
        @test q.b ≈ 1f0 atol=0.05   # still blue: the far draw failed the test
        @test q.r ≈ 0f0 atol=0.05
    end

    # ── Pipeline state vs. the compiled-pipeline cache ──

    # One shader pair, two blend modes. The pipeline cache keyed only on the
    # shaders, their argument types and the colour format, so the second pipeline
    # got whatever state the first was compiled with: an Additive draw rendered
    # opaque, or the reverse, depending on which ran first. Every other testset
    # here defines its own shader functions, which is why nothing caught it.
    function state_vert()
        vid = Mantle.vertex_index() - Int32(1)
        x = Float32(Int32(vid & Int32(1)) * 4 - 1)
        y = Float32(Int32((vid >> Int32(1)) & Int32(1)) * 4 - 1)
        return (position = Vec4f(x, y, 0.5f0, 1.0f0),)
    end
    state_frag(_inputs) = Vec4f(0.25f0, 0f0, 0f0, 1f0)

    @testset "pipeline state is part of the cache key" begin
        opaque = GraphicsPipeline(; vertex = VertexShader(state_vert),
                                    fragment = FragmentShader(state_frag),
                                    blend = Opaque(),
                                    cull = NoCull(),
                                    depth = DepthOff())
        additive = GraphicsPipeline(; vertex = VertexShader(state_vert),
                                      fragment = FragmentShader(state_frag),
                                      blend = Additive(),
                                      cull = NoCull(),
                                      depth = DepthOff())

        # Opaque first, so a shared cache entry would hand the additive draws
        # opaque blending.
        @test draw_and_readback(dev, opaque, 3)[4, 4].r ≈ 0.25f0 atol=0.01

        fb = Mantle.Framebuffer(TESTBACKEND, 8, 8; depth=false, color_format=RGBA{Float32})
        target = OffscreenTarget(fb)
        Mantle.draw!(dev, additive, target, 3; clear_color=(0f0, 0f0, 0f0, 1f0))
        Mantle.draw!(dev, additive, target, 3; clear_color=nothing)
        Mantle.flush!(dev)
        @test Mantle.readback_framebuffer(fb)[4, 4].r ≈ 0.5f0 atol=0.01
    end

    # ── Depth attachment vs. depth mode ──

    # The pipeline's depth attachment format is a property of the render target:
    # dynamic rendering requires it to equal the format of the bound depth view,
    # and UNDEFINED when none is bound. Deriving it from the depth mode instead
    # made both mismatched combinations produce an invalid pipeline. A depth test
    # against no depth attachment is refused by name rather than drawn untested.
    @testset "depth testing needs a depth attachment" begin
        pip = GraphicsPipeline(; vertex = VertexShader(state_vert),
                                 fragment = FragmentShader(state_frag),
                                 blend = Opaque(),
                                 cull = NoCull(),
                                 depth = DepthLess())
        @test_throws ArgumentError draw_and_readback(dev, pip, 3; depth=false)
    end

    @testset "DepthOff into a target that has depth" begin
        # Legal, and not the same as having no attachment: the pass still binds
        # depth, so the pipeline has to declare its format. Nothing is tested or
        # written, so the later draw wins whatever its z is.
        function zvert(color::Vec4f, z_val::Float32)
            vid = Mantle.vertex_index() - Int32(1)
            x = Float32(Int32(vid & Int32(1)) * 4 - 1)
            y = Float32(Int32((vid >> Int32(1)) & Int32(1)) * 4 - 1)
            return (position = Vec4f(x, y, z_val, 1.0f0), color = color)
        end
        zfrag(inputs) = inputs.color
        pip = GraphicsPipeline(; vertex = VertexShader(zvert; outputs = (color = Vec4f,)),
                                 fragment = FragmentShader(zfrag),
                                 blend = Opaque(),
                                 cull = NoCull(),
                                 depth = DepthOff())

        fb = Mantle.Framebuffer(TESTBACKEND, 8, 8; depth=true, color_format=RGBA{Float32})
        target = OffscreenTarget(fb)
        Mantle.draw!(dev, pip, target, 3; args=(Vec4f(0f0, 0f0, 1f0, 1f0), 0.3f0),
            clear_color=(0f0, 0f0, 0f0, 1f0))          # near blue
        Mantle.draw!(dev, pip, target, 3; args=(Vec4f(1f0, 0f0, 0f0, 1f0), 0.7f0),
            clear_color=nothing)                        # far red, drawn later
        Mantle.flush!(dev)
        p = Mantle.readback_framebuffer(fb)[4, 4]
        @test p.r ≈ 1f0 atol=0.05                       # red: no depth test ran
        @test p.b ≈ 0f0 atol=0.05
    end

    # ── Multiple outputs / instances ──

    @testset "instanced drawing" begin
        function inst_vert()
            vid = Mantle.vertex_index() - Int32(1)
            iid = Mantle.instance_index() - Int32(1)
            # Shift each instance right by 0.5 NDC
            base_x = Float32(Int32(vid & Int32(1)) * 4 - 1)
            base_y = Float32(Int32((vid >> Int32(1)) & Int32(1)) * 4 - 1)
            x = base_x + Float32(iid) * 0.5f0
            # Color by instance: 0=red, 1=green
            r = iid == Int32(0) ? 1f0 : 0f0
            g = iid == Int32(1) ? 1f0 : 0f0
            return (position = Vec4f(x, base_y, 0.5f0, 1.0f0), color = Vec4f(r, g, 0f0, 1f0))
        end
        inst_frag(inputs) = inputs.color
        pip = GraphicsPipeline(; vertex = VertexShader(inst_vert; outputs = (color = Vec4f,)),
                                 fragment = FragmentShader(inst_frag),
                                 blend = Opaque(),
                                 cull = NoCull(),
                                 depth = DepthOff())
        pixels = draw_and_readback(dev, pip, 3; instances=2, width=32, height=32)
        # Should have some non-black pixels from both instances
        has_red = any(p -> p.r > 0.5f0, pixels)
        has_green = any(p -> p.g > 0.5f0, pixels)
        @test has_red
        @test has_green
    end

    @testset "a fragment shader writing several attachments draws to each" begin
        # A fragment shader that returns a tuple gets its varyings unpacked by
        # FragmentWrapper, and inlining that leaves an
        # llvm.experimental.noalias.scope.decl behind — a metadata declaration
        # that emits no code and that a SPIR-V emitter can reject as an
        # unsupported intrinsic. Found by a deferred renderer whose g-buffer pass
        # writes albedo and normals from one fragment.
        #
        # Asserted on the pixels rather than on the compiled object's type: each
        # attachment has to come out holding its own value, read from the
        # varyings the fragment unpacked.
        function gbuf_vertex()
            vid = Mantle.vertex_index() - Int32(1)
            x = Float32(Int32(vid & Int32(1)) * 4 - 1)
            y = Float32(Int32((vid >> Int32(1)) & Int32(1)) * 4 - 1)
            return (position = Vec4f(x, y, 0.5f0, 1f0),
                    albedo = Vec4f(1, 0, 0, 1), normal = Vec3f(0, 1, 0))
        end
        gbuf_fragment(inputs) = (inputs.albedo,
                                 Vec4f(0.5f0 * inputs.normal[1] + 0.5f0,
                                       0.5f0 * inputs.normal[2] + 0.5f0,
                                       0.5f0 * inputs.normal[3] + 0.5f0, 1f0))
        pipe = Rasterizer(; vertex = VertexShader(gbuf_vertex; outputs = (albedo=Vec4f, normal=Vec3f)),
                            fragment = FragmentShader(gbuf_fragment),
                            topology = TriangleList(),
                            blend = Opaque(),
                            cull = NoCull(),
                            depth = DepthOff())
        g = Mantle.Graph(dev)
        a = Mantle.Transient.Image(g, RGBA{Float32}, (8, 8))
        n = Mantle.Transient.Image(g, RGBA{Float32}, (8, 8))
        Mantle.render!(g, "gbuffer", a => Mantle.Clear((0f0, 0f0, 0f0, 1f0)),
                                     n => Mantle.Clear((0f0, 0f0, 0f0, 1f0))) do p
            Mantle.draw!(p, pipe, (), 3)
        end
        # Persistent, not transient: nothing in the graph reads either copy, so
        # two transients may share memory and the second copy lands in both.
        oa = Mantle.Buffer(dev, RGBA{Float32}, 64)
        on = Mantle.Buffer(dev, RGBA{Float32}, 64)
        Mantle.copy!(g, "read albedo", oa, a)
        Mantle.copy!(g, "read normal", on, n)
        Mantle.run!(Mantle.record!(Mantle.Plan(g)))
        @test all(==(RGBA{Float32}(1, 0, 0, 1)), Array(Mantle.storage(oa)))
        @test all(p -> p.r ≈ 0.5f0 && p.g ≈ 1f0 && p.b ≈ 0.5f0 && p.alpha ≈ 1f0,
                  Array(Mantle.storage(on)))
        Mantle.free!(oa)
        Mantle.free!(on)
    end

    @testset "a blit source is a (height, width) matrix" begin
        # `copy_image_to_buffer!` packs rows and the blit shader reads a matrix
        # whose row index varies fastest, so the two are transposes of each
        # other. Reading an image out and blitting it back with the same index
        # shears the picture rather than breaking it, which is how it survived in
        # two benches; a matrix source now says so instead.
        w, h = 32, 16
        fb = Mantle.Framebuffer(TESTBACKEND, w, h; depth=false, color_format=RGBA{Float32})
        right = Mantle.devicearray(TESTBACKEND, reshape([Vec4f(0, 1, 0, 1) for _ in 1:(w * h)], h, w))
        wrong = Mantle.devicearray(TESTBACKEND, reshape([Vec4f(0, 1, 0, 1) for _ in 1:(w * h)], w, h))
        @test_throws DimensionMismatch blit!(dev, OffscreenTarget(fb), wrong)
        @test_throws DimensionMismatch blit!(dev, OffscreenTarget(fb),
                                            Mantle.devicearray(TESTBACKEND, [Vec4f(0, 0, 0, 1) for _ in 1:(w * h - 1)]))

        blit!(dev, OffscreenTarget(fb), right)
        Mantle.flush!(dev)
        px = Mantle.readback_framebuffer(fb)
        @test all(p -> p.g > 0.9f0, px)
    end

    # ── What the Metal backend's graphics tests pinned, on every backend ──

    @testset "a buffer of positions bounds what is drawn" begin
        # The portable spelling `bench/showcase.jl` uses: the vertex stage
        # RETURNS `(position = …, varyings…)` and the fragment stage takes the
        # varyings as its first argument. A portable `Buffer` and not a driver
        # handle: core's `draw!` reads the shader argument types off the arrays
        # it is given, which is why it needs no `vert_tt`.
        fb = Mantle.Framebuffer(TESTBACKEND, 64, 64; depth = false, color_format = BGRA{N0f8})
        pipe = GraphicsPipeline(; vertex = VertexShader(posbuf_vertex; outputs = (tint = NTuple{4,Float32},)),
                                  fragment = FragmentShader(posbuf_fragment),
                                  cull = NoCull(), depth = DepthOff())
        verts = NTuple{4,Float32}[(-0.9f0, -0.9f0, 0f0, 1f0), (0.9f0, -0.9f0, 0f0, 1f0),
                                  (0f0, 0.9f0, 0f0, 1f0)]
        vbuf = Mantle.Buffer(dev, verts)
        Mantle.draw!(dev, pipe, OffscreenTarget(fb), 3; args = (vbuf,))

        # `width` x `height`, one COLORANT per pixel, which is the shape AND the
        # element type `readback_framebuffer` promises. Both halves were wrong on
        # one backend once: it returned a `4 x width x height` byte array, so
        # the one portable caller — RayMakie's compositor — hit a `BoundsError`
        # there; and then a `Matrix{NTuple{4,UInt8}}`, which has the right shape
        # and still makes the caller guess where red is.
        px = Mantle.readback_framebuffer(fb)
        @test px isa Matrix{BGRA{N0f8}}
        @test size(px) == (64, 64)
        green = count(q -> q.g > n8(0x80), px)
        # About 40 % of the frame, which is this triangle's area in NDC. A blank or
        # a fully-covered target both fail here.
        @test 1000 < green < 3000
        # …and the pixels outside it kept the clear colour, so the draw was bounded
        # by the geometry rather than covering everything.
        @test count(q -> q.g <= n8(0x80), px) > 1000

        # Compiling the same pipeline again must hit the cache: a pipeline rebuilt
        # per draw is the difference between a frame and a stall.
        second = Mantle.Framebuffer(TESTBACKEND, 64, 64; depth = false, color_format = BGRA{N0f8})
        Mantle.draw!(dev, pipe, OffscreenTarget(second), 3; args = (vbuf,))
        @test Mantle.readback_framebuffer(second) == px
        Mantle.free!(vbuf)
    end

    @testset "a readback waits for the pass that filled the target" begin
        # `readback_framebuffer` promises to synchronise. On Metal it is a HOST
        # copy out of the texture that neither joins the queue nor blocks on it,
        # and with no wait a readback taken straight after a pass read whatever
        # the texture held — zeros, for a fresh one.
        #
        # A race, so it hid for a long time: RayMakie's compositor synchronises
        # elsewhere in the frame and the window is narrow. Back to back it is lost
        # every time, and the whole composited image comes back transparent black.
        # No `flush!` here on purpose — that is the thing being tested.
        fb = Mantle.Framebuffer(TESTBACKEND, 8, 8; depth = false, color_format = BGRA{N0f8})
        orange = Mantle.devicearray(TESTBACKEND, fill(RGBA{Float32}(1f0, 0.5f0, 0f0, 1f0), 64))
        blit!(dev, OffscreenTarget(fb), orange; clear = true)
        px = Mantle.readback_framebuffer(fb)
        # Orange, named by channel: the colorant carries where blue sits in a
        # `BGRA` attachment, so the test states the COLOUR.
        @test all(==(BGRA{N0f8}(1, n8(0x80), 0, 1)), px)
    end
end
