# Tier 3: Graphics Pipeline GPU Execution Tests
#
# Tests the full graphics pipeline on GPU: compilation, draw, readback.
# Uses offscreen framebuffers (no window needed — works on lavapipe/CI).

using Test
using Lava, Mantle
using Vulkan
using GeometryBasics

# Helper: create offscreen framebuffer, draw, flush, readback
function draw_and_readback(pipeline, vertex_count;
        width=16, height=16, args=(), frag_args=(),
        clear_color=(0f0, 0f0, 0f0, 1f0),
        color_format=Vulkan.FORMAT_R32G32B32A32_SFLOAT,
        depth=false, instances=1)
    fb = VulkanFramebuffer(width, height; depth, color_format)
    target = OffscreenTarget(fb)
    ctx = MVE.vk_context()
    bq = ctx.default_bq
    draw!(bq, pipeline, target, vertex_count;
        args, frag_args, instances, clear_color)
    MVE.vk_flush!(MVE.vk_context())
    return readback_framebuffer(fb)
end

@testset "Graphics Pipeline" begin

    # ── Basic vertex + fragment ──

    @testset "solid color triangle" begin
        function solid_vert()
            vid = Lava.vertex_index() - Int32(1)
            x = Float32(Int32(vid & Int32(1)) * 4 - 1)
            y = Float32(Int32((vid >> Int32(1)) & Int32(1)) * 4 - 1)
            Lava.set_position!(Vec4f(x, y, 0.5f0, 1.0f0))
            return nothing
        end
        function solid_frag()
            Lava.gfx_output(0, Vec4f(0.0f0, 1.0f0, 0.0f0, 1.0f0))
            return nothing
        end
        pip = GraphicsPipeline(; vertex=solid_vert, fragment=solid_frag,
            blend=Opaque(), cull=NoCull(), depth=DepthOff())
        pixels = draw_and_readback(pip, 3)
        @test all(p -> p[2] ≈ 1f0, pixels)  # all green
        @test all(p -> p[4] ≈ 1f0, pixels)  # alpha = 1
    end

    @testset "clear color works" begin
        # Draw zero vertices — only clear should be visible
        function noop_vert()
            Lava.set_position!(Vec4f(0f0, 0f0, 0f0, 1f0))
            return nothing
        end
        function noop_frag()
            Lava.gfx_output(0, Vec4f(0f0, 0f0, 0f0, 0f0))
            return nothing
        end
        pip = GraphicsPipeline(; vertex=noop_vert, fragment=noop_frag,
            blend=Opaque(), cull=NoCull(), depth=DepthOff())
        pixels = draw_and_readback(pip, 0; clear_color=(0.25f0, 0.5f0, 0.75f0, 1f0))
        @test all(p -> p[1] ≈ 0.25f0 && p[2] ≈ 0.5f0 && p[3] ≈ 0.75f0, pixels)
    end

    # ── BDA arguments ──

    @testset "vertex BDA args" begin
        function bda_vert(positions::Lava.LavaDeviceArray{Vec3f, 1})
            vid = Lava.vertex_index()
            @inbounds p = positions[vid]
            Lava.set_position!(Vec4f(p[1], p[2], p[3], 1.0f0))
            Lava.gfx_output(0, Vec4f(1.0f0, 0.0f0, 0.0f0, 1.0f0))
            return nothing
        end
        function bda_frag()
            c = Lava.gfx_input(Vec4f, 0)
            Lava.gfx_output(0, c)
            return nothing
        end
        pip = GraphicsPipeline(; vertex=bda_vert, fragment=bda_frag,
            blend=Opaque(), cull=NoCull(), depth=DepthOff())
        # Fullscreen triangle
        positions = LavaArray(Vec3f[Vec3f(-1,-1,0), Vec3f(3,-1,0), Vec3f(-1,3,0)])
        pixels = draw_and_readback(pip, 3; args=(positions,))
        @test all(p -> p[1] ≈ 1f0, pixels)  # all red
    end

    @testset "vertex-to-fragment varying" begin
        function vary_vert()
            vid = Lava.vertex_index() - Int32(1)
            x = Float32(Int32(vid & Int32(1)) * 4 - 1)
            y = Float32(Int32((vid >> Int32(1)) & Int32(1)) * 4 - 1)
            Lava.set_position!(Vec4f(x, y, 0.5f0, 1.0f0))
            # Pass UV as varying
            u = (x + 1f0) * 0.5f0
            v = (y + 1f0) * 0.5f0
            Lava.gfx_output(0, Vec2f(u, v))
            return nothing
        end
        function vary_frag()
            uv = Lava.gfx_input(Vec2f, 0)
            Lava.gfx_output(0, Vec4f(uv[1], uv[2], 0f0, 1f0))
            return nothing
        end
        pip = GraphicsPipeline(; vertex=vary_vert, fragment=vary_frag,
            blend=Opaque(), cull=NoCull(), depth=DepthOff())
        pixels = draw_and_readback(pip, 3; width=8, height=8)
        # Center pixel should have UV ≈ (0.5, 0.5)
        center = pixels[4, 4]
        @test center[1] ≈ 0.5f0 atol=0.15  # u
        @test center[2] ≈ 0.5f0 atol=0.15  # v
        # Corner (0,0) should have UV near (0, 0)
        @test pixels[1,1][1] < 0.2f0  # u near 0
        @test pixels[1,1][2] < 0.2f0  # v near 0
    end

    @testset "fragment uses frag_coord" begin
        function fc_vert()
            vid = Lava.vertex_index() - Int32(1)
            x = Float32(Int32(vid & Int32(1)) * 4 - 1)
            y = Float32(Int32((vid >> Int32(1)) & Int32(1)) * 4 - 1)
            Lava.set_position!(Vec4f(x, y, 0.5f0, 1.0f0))
            return nothing
        end
        function fc_frag()
            fx = Lava.frag_coord_x()
            fy = Lava.frag_coord_y()
            # Normalize to [0,1] range using 16x16 framebuffer
            Lava.gfx_output(0, Vec4f(fx / 16f0, fy / 16f0, 0f0, 1f0))
            return nothing
        end
        pip = GraphicsPipeline(; vertex=fc_vert, fragment=fc_frag,
            blend=Opaque(), cull=NoCull(), depth=DepthOff())
        pixels = draw_and_readback(pip, 3)
        # Pixel at (8,8) should have frag_coord ≈ (8.5, 8.5) → normalized ≈ (0.53, 0.53)
        p = pixels[8, 8]
        @test 0.4f0 < p[1] < 0.7f0
        @test 0.4f0 < p[2] < 0.7f0
    end

    # ── Blend modes ──

    @testset "alpha blend" begin
        function ab_vert()
            vid = Lava.vertex_index() - Int32(1)
            x = Float32(Int32(vid & Int32(1)) * 4 - 1)
            y = Float32(Int32((vid >> Int32(1)) & Int32(1)) * 4 - 1)
            Lava.set_position!(Vec4f(x, y, 0.5f0, 1.0f0))
            return nothing
        end
        function ab_frag()
            # Semi-transparent red
            Lava.gfx_output(0, Vec4f(1f0, 0f0, 0f0, 0.5f0))
            return nothing
        end
        pip = GraphicsPipeline(; vertex=ab_vert, fragment=ab_frag,
            blend=AlphaBlend(), cull=NoCull(), depth=DepthOff())
        # Clear to white, draw semi-transparent red
        pixels = draw_and_readback(pip, 3; clear_color=(1f0, 1f0, 1f0, 1f0))
        # Result should be blended: red = 0.5*1 + 0.5*1 = 1, green = 0.5*0 + 0.5*1 = 0.5
        p = pixels[8, 8]
        @test p[1] ≈ 1f0 atol=0.05  # red
        @test p[2] ≈ 0.5f0 atol=0.05  # green (blended)
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
            vid = Lava.vertex_index() - Int32(1)
            x = Float32(Int32(vid & Int32(1)) * 4 - 1)
            y = Float32(Int32((vid >> Int32(1)) & Int32(1)) * 4 - 1)
            Lava.set_position!(Vec4f(x, y, z_val, 1.0f0))
            Lava.gfx_output(0, color)
            return nothing
        end
        function depth_frag()
            c = Lava.gfx_input(Vec4f, 0)
            Lava.gfx_output(0, c)
            return nothing
        end
        pip = GraphicsPipeline(; vertex=depth_vert, fragment=depth_frag,
            blend=Opaque(), cull=NoCull(), depth=DepthLess())

        fb = VulkanFramebuffer(8, 8; depth=true, color_format=Vulkan.FORMAT_R32G32B32A32_SFLOAT)
        target = OffscreenTarget(fb)
        ctx = MVE.vk_context()
        bq = ctx.default_bq

        # Draw far red at z=0.7
        draw!(bq, pip, target, 3;
            args=(Vec4f(1f0, 0f0, 0f0, 1f0), 0.7f0),
            clear_color=(0f0, 0f0, 0f0, 1f0))
        # Draw close blue at z=0.3 (no clear — load previous colour and depth)
        draw!(bq, pip, target, 3;
            args=(Vec4f(0f0, 0f0, 1f0, 1f0), 0.3f0),
            clear_color=nothing, depth_clear=nothing)
        MVE.vk_flush!(MVE.vk_context())

        pixels = readback_framebuffer(fb)
        # Blue should win (closer)
        p = pixels[4, 4]
        @test p[3] ≈ 1f0 atol=0.05  # blue channel
        @test p[1] ≈ 0f0 atol=0.05  # red channel

        # The other order is the half that a per-draw depth clear could not fail:
        # near first, far second, and the far one must be rejected.
        fb2 = VulkanFramebuffer(8, 8; depth=true,
            color_format=Vulkan.FORMAT_R32G32B32A32_SFLOAT)
        t2 = OffscreenTarget(fb2)
        draw!(bq, pip, t2, 3;
            args=(Vec4f(0f0, 0f0, 1f0, 1f0), 0.3f0),
            clear_color=(0f0, 0f0, 0f0, 1f0))
        draw!(bq, pip, t2, 3;
            args=(Vec4f(1f0, 0f0, 0f0, 1f0), 0.7f0),
            clear_color=nothing, depth_clear=nothing)
        MVE.vk_flush!(MVE.vk_context())

        q = readback_framebuffer(fb2)[4, 4]
        @test q[3] ≈ 1f0 atol=0.05  # still blue: the far draw failed the test
        @test q[1] ≈ 0f0 atol=0.05
    end

    # ── Pipeline state vs. the compiled-pipeline cache ──

    # One shader pair, two blend modes. The pipeline cache keyed only on the
    # shaders, their argument types and the colour format, so the second pipeline
    # got whatever state the first was compiled with: an Additive draw rendered
    # opaque, or the reverse, depending on which ran first. Every other testset
    # here defines its own shader functions, which is why nothing caught it.
    function state_vert()
        vid = Lava.vertex_index() - Int32(1)
        x = Float32(Int32(vid & Int32(1)) * 4 - 1)
        y = Float32(Int32((vid >> Int32(1)) & Int32(1)) * 4 - 1)
        Lava.set_position!(Vec4f(x, y, 0.5f0, 1.0f0))
        return nothing
    end
    function state_frag()
        Lava.gfx_output(0, Vec4f(0.25f0, 0f0, 0f0, 1f0))
        return nothing
    end

    @testset "pipeline state is part of the cache key" begin
        opaque = GraphicsPipeline(; vertex=state_vert, fragment=state_frag,
            blend=Opaque(), cull=NoCull(), depth=DepthOff())
        additive = GraphicsPipeline(; vertex=state_vert, fragment=state_frag,
            blend=Additive(), cull=NoCull(), depth=DepthOff())

        ctx = MVE.vk_context()
        bq = ctx.default_bq
        # Opaque first, so a shared cache entry would hand the additive draws
        # opaque blending.
        @test draw_and_readback(opaque, 3)[4, 4][1] ≈ 0.25f0 atol=0.01

        fb = VulkanFramebuffer(8, 8; depth=false,
            color_format=Vulkan.FORMAT_R32G32B32A32_SFLOAT)
        target = OffscreenTarget(fb)
        draw!(bq, additive, target, 3; clear_color=(0f0, 0f0, 0f0, 1f0))
        draw!(bq, additive, target, 3; clear_color=nothing)
        MVE.vk_flush!(ctx)
        @test readback_framebuffer(fb)[4, 4][1] ≈ 0.5f0 atol=0.01
    end

    # ── Depth attachment vs. depth mode ──

    # The pipeline's depth attachment format is a property of the render target:
    # dynamic rendering requires it to equal the format of the bound depth view,
    # and UNDEFINED when none is bound. Deriving it from the depth mode instead
    # made both mismatched combinations produce an invalid pipeline.
    @testset "depth testing needs a depth attachment" begin
        pip = GraphicsPipeline(; vertex=state_vert, fragment=state_frag,
            blend=Opaque(), cull=NoCull(), depth=DepthLess())
        @test_throws ArgumentError draw_and_readback(pip, 3; depth=false)
    end

    @testset "DepthOff into a target that has depth" begin
        # Legal, and not the same as having no attachment: the pass still binds
        # depth, so the pipeline has to declare its format. Nothing is tested or
        # written, so the later draw wins whatever its z is.
        function zvert(color::Vec4f, z_val::Float32)
            vid = Lava.vertex_index() - Int32(1)
            x = Float32(Int32(vid & Int32(1)) * 4 - 1)
            y = Float32(Int32((vid >> Int32(1)) & Int32(1)) * 4 - 1)
            Lava.set_position!(Vec4f(x, y, z_val, 1.0f0))
            Lava.gfx_output(0, color)
            return nothing
        end
        function zfrag()
            Lava.gfx_output(0, Lava.gfx_input(Vec4f, 0))
            return nothing
        end
        pip = GraphicsPipeline(; vertex=zvert, fragment=zfrag,
            blend=Opaque(), cull=NoCull(), depth=DepthOff())

        fb = VulkanFramebuffer(8, 8; depth=true,
            color_format=Vulkan.FORMAT_R32G32B32A32_SFLOAT)
        target = OffscreenTarget(fb)
        ctx = MVE.vk_context()
        bq = ctx.default_bq
        draw!(bq, pip, target, 3; args=(Vec4f(0f0, 0f0, 1f0, 1f0), 0.3f0),
            clear_color=(0f0, 0f0, 0f0, 1f0))          # near blue
        draw!(bq, pip, target, 3; args=(Vec4f(1f0, 0f0, 0f0, 1f0), 0.7f0),
            clear_color=nothing)                        # far red, drawn later
        MVE.vk_flush!(ctx)
        p = readback_framebuffer(fb)[4, 4]
        @test p[1] ≈ 1f0 atol=0.05                      # red: no depth test ran
        @test p[3] ≈ 0f0 atol=0.05
    end

    # ── Multiple outputs / instances ──

    @testset "instanced drawing" begin
        function inst_vert()
            vid = Lava.vertex_index() - Int32(1)
            iid = Lava.instance_index() - Int32(1)
            # Shift each instance right by 0.5 NDC
            base_x = Float32(Int32(vid & Int32(1)) * 4 - 1)
            base_y = Float32(Int32((vid >> Int32(1)) & Int32(1)) * 4 - 1)
            x = base_x + Float32(iid) * 0.5f0
            Lava.set_position!(Vec4f(x, base_y, 0.5f0, 1.0f0))
            # Color by instance: 0=red, 1=green
            r = iid == Int32(0) ? 1f0 : 0f0
            g = iid == Int32(1) ? 1f0 : 0f0
            Lava.gfx_output(0, Vec4f(r, g, 0f0, 1f0))
            return nothing
        end
        function inst_frag()
            c = Lava.gfx_input(Vec4f, 0)
            Lava.gfx_output(0, c)
            return nothing
        end
        pip = GraphicsPipeline(; vertex=inst_vert, fragment=inst_frag,
            blend=Opaque(), cull=NoCull(), depth=DepthOff())
        pixels = draw_and_readback(pip, 3; instances=2, width=32, height=32)
        # Should have some non-black pixels from both instances
        has_red = any(p -> p[1] > 0.5f0, pixels)
        has_green = any(p -> p[2] > 0.5f0, pixels)
        @test has_red
        @test has_green
    end

    @testset "a fragment shader writing several attachments compiles" begin
        # A fragment shader that returns a tuple gets its varyings unpacked by
        # FragmentWrapper, and inlining that leaves an
        # llvm.experimental.noalias.scope.decl behind — a metadata declaration
        # that emits no code and that the SPIR-V emitter used to reject as an
        # unsupported intrinsic. Found by a deferred renderer whose g-buffer pass
        # writes albedo and normals from one fragment.
        gbuf_vertex() = (position = Vec4f(0, 0, 0.5, 1),
                         albedo = Vec4f(1, 0, 0, 1), normal = Vec3f(0, 1, 0))
        gbuf_fragment(inputs) = (inputs.albedo,
                                 Vec4f(0.5f0 * inputs.normal[1] + 0.5f0,
                                       0.5f0 * inputs.normal[2] + 0.5f0,
                                       0.5f0 * inputs.normal[3] + 0.5f0, 1f0))
        pipe = Rasterizer(vertex=gbuf_vertex, fragment=gbuf_fragment,
            varyings=(albedo=Vec4f, normal=Vec3f), topology=TriangleList(),
            blend=Opaque(), cull=NoCull(), depth=DepthOff())
        vfn, vtt, ffn, ftt = MVE.resolve_shader_pair(pipe, Tuple{}, Tuple{})
        _, compiled = MVE.ensure_compiled_with_shader!(pipe, vfn, ffn, vtt, ftt;
            color_format=Vulkan.Format[Vulkan.FORMAT_R8G8B8A8_UNORM,
                                       Vulkan.FORMAT_R8G8B8A8_UNORM],
            depth_format=Vulkan.FORMAT_UNDEFINED)
        @test compiled isa VulkanCompiledGraphicsPipeline
    end

    @testset "a blit source is a (height, width) matrix" begin
        # `copy_image_to_buffer!` packs rows and the blit shader reads a matrix
        # whose row index varies fastest, so the two are transposes of each
        # other. Reading an image out and blitting it back with the same index
        # shears the picture rather than breaking it, which is how it survived in
        # two benches; a matrix source now says so instead.
        w, h = 32, 16
        fb = VulkanFramebuffer(w, h; depth=false,
            color_format=Vulkan.FORMAT_R32G32B32A32_SFLOAT)
        bq = MVE.vk_context().default_bq
        right = LavaArray(reshape([Vec4f(0, 1, 0, 1) for _ in 1:(w * h)], h, w))
        wrong = LavaArray(reshape([Vec4f(0, 1, 0, 1) for _ in 1:(w * h)], w, h))
        @test_throws DimensionMismatch blit!(bq, OffscreenTarget(fb), wrong)
        @test_throws DimensionMismatch blit!(bq, OffscreenTarget(fb),
                                            LavaArray([Vec4f(0, 0, 0, 1) for _ in 1:(w * h - 1)]))

        blit!(bq, OffscreenTarget(fb), right)
        MVE.vk_flush!(MVE.vk_context())
        px = readback_framebuffer(fb)
        @test all(p -> p[2] > 0.9f0, px)
    end
end
