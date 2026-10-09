# Sampling a bound texture, and the layout it is bound in.
#
# `KernelInterface.sample_texture_2d` names a SLOT, and how a texture reaches the
# intrinsic is each backend's business (a descriptor set on Vulkan, a lowering
# onto an argument on Metal). What is asserted here is what a caller can see:
# which texel a coordinate lands on, that the FILTERING is the sampler's rather
# than the shader's, and that drawing through the graph samples what drawing by
# hand does. `test_texture_upload.jl` covers where the texels come from.
#
# Gated on `supports_graphics`, the same assertions everywhere.

using Test, Mantle
using Mantle: Vec4f
using ColorTypes: RGBA, BGRA
using ColorTypes.FixedPointNumbers: N0f8
include(joinpath(@__DIR__, "testbackend.jl"))

# A fullscreen triangle whose `uv` sweeps 0..1 across the target.
function tex_vertex()
    vid = Mantle.vertex_index() - Int32(1)
    x = Float32(Int32(vid & Int32(1)) * 4 - 1)
    y = Float32(Int32((vid >> Int32(1)) & Int32(1)) * 4 - 1)
    return (position = Vec4f(x, Mantle.clip_y(y), 0f0, 1f0),
            uv = (0.5f0 * (x + 1f0), 0.5f0 * (y + 1f0)))
end

tex_fragment(inputs) =
    (Mantle.sample_texture_2d(UInt32(0), inputs.uv[1], inputs.uv[2], UInt32(0)),
     0f0, 0f0, 1f0)

tex_pipeline() = Mantle.GraphicsPipeline(;
    vertex = Mantle.VertexShader(tex_vertex; outputs = (uv = NTuple{2,Float32},)),
    fragment = Mantle.FragmentShader(tex_fragment; textures = 1),
    cull = Mantle.NoCull(), depth = Mantle.DepthOff())

function tex_bindings(data, filter)
    tex = Mantle.Texture2D(TESTBACKEND, data)
    sampler = Mantle.Sampler(TESTBACKEND; filter, wrap = :clamp)
    return Mantle.bind_textures([Mantle.SampledTexture(tex, sampler)])
end

"""Draw the sampling quad over `data` into an n x n target by hand and read it back."""
function tex_draw(dev, data; n = 4, filter = :nearest)
    bindings = tex_bindings(data, filter)
    fb = Mantle.Framebuffer(TESTBACKEND, n, n; depth = false, color_format = BGRA{N0f8})
    target = Mantle.OffscreenTarget(fb)
    compiled = Mantle.compile_draw(dev, tex_pipeline(), (Mantle.blittarget(target),),
                                   nothing, (), (); bindings)
    Mantle.pass!(dev, target; clear = (0f0, 0f0, 0f0, 1f0)) do pr
        Mantle.viewport!(pr, 0, 0, n, n)
        Mantle.bindings!(pr, compiled, bindings)
        Mantle.draw!(pr, compiled, (), 3)
    end
    px = Mantle.readback_framebuffer(fb)
    # The red channel carries component 0. Its second index counts from the top
    # while `uv.y = 0` is the bottom of the target, so the row is flipped.
    #
    # `.r`, and no division: the colorant says where red sits in a BGRA
    # attachment, and `Float32(::N0f8)` is already 0..1.
    return [Float32(px[x, n + 1 - y].r) for x in 1:n, y in 1:n]
end

"""The same sampling quad, drawn through the render GRAPH rather than by hand."""
function tex_graph(dev, data; n = 4, filter = :nearest)
    bindings = tex_bindings(data, filter)
    g = Mantle.Graph(dev)
    img = Mantle.Transient.Image(g, RGBA{N0f8}, (n, n))
    out = Mantle.Transient.Buffer(g, UInt32, n * n)
    # No `compile_draw` and no `bindings!` here: the draw carries what it samples
    # and the plan does both, which is the whole difference from `tex_draw`.
    Mantle.render!(g, "sample", img => Mantle.Clear((0f0, 0f0, 0f0, 1f0));
                   viewport = (0, 0, n, n)) do p
        Mantle.draw!(p, tex_pipeline(), (), 3; bindings)
    end
    Mantle.copy!(g, "read", out, img)
    Mantle.run!(Mantle.record!(Mantle.Plan(g)))
    px = reshape(Array(Mantle.storage(out)), n, n)
    # RGBA8 packed little-endian, so component 0 is the low byte; second index
    # counts from the top and `uv.y = 0` is the bottom, as in `tex_draw`.
    return [Float32(px[x, n + 1 - y] & 0xff) / 255 for x in 1:n, y in 1:n]
end

@testset "sampling a bound texture" begin
    dev = Mantle.Device(TESTBACKEND)
    if !Mantle.supports_graphics(dev)
        @info "no graphics pipeline on this backend; skipping"
        @test_skip Mantle.supports_graphics(dev)
        return
    end

    @testset "a texture is bound as data[x, y]" begin
        # NON-SQUARE on purpose: 4 wide and 2 tall is the case a square atlas
        # cannot tell apart, and it is the one an upload gets wrong by reading the
        # dimensions one way round and copying the bytes the other, so only a
        # square texture or a single row came out whole.
        data = Float32[(16 * (x - 1) + 2 * (y - 1)) / 100 for x in 1:4, y in 1:2]
        got = tex_draw(dev, data; n = 4)

        # Four columns of the target over four texel columns; the two texel rows
        # each cover half the height. `data[x, y]`: x is the FIRST index.
        for x in 1:4, y in 1:2
            rows = y == 1 ? (1:2) : (3:4)     # y = 1 is v ≈ 0, the bottom half
            for r in rows
                @test isapprox(got[x, r], data[x, y]; atol = 0.01)
            end
        end
    end

    @testset "the filtering is the sampler's" begin
        # Two texels, black and white. A LINEAR sampler mixes them across the
        # middle of the target and a NEAREST one cannot: no intermediate value
        # exists in the texture, so anything between the two came from the
        # texture unit.
        data = Float32[0f0 1f0]'          # 2 wide, 1 tall — data[x, y]
        near = tex_draw(dev, data; n = 16, filter = :nearest)
        lin  = tex_draw(dev, data; n = 16, filter = :linear)
        @test all(v -> v < 0.01 || v > 0.99, near)
        @test any(v -> 0.2 < v < 0.8, lin)
        # …and the ends still read as the texels themselves.
        @test lin[1, 8] < 0.1 && lin[end, 8] > 0.9
    end

    @testset "a graph draw samples what it was given" begin
        # Bindings reachable only from `pass!` would mean a renderer that samples
        # anything records its passes by hand and cannot be a graph at all. They
        # travel with the COMPILE as well as with the bind — Vulkan builds the
        # pipeline layout around the descriptor set layout — so the plan has to
        # hand them to `compile_draw` too, and a test that only checked the bind
        # would pass on one backend and fail on the other.
        #
        # Asserted against the hand-recorded path rather than against numbers: the
        # claim is that routing a draw through the graph changes nothing, and two
        # independent expectations could drift apart while both stayed green.
        #
        # `Float32.` only because the two readbacks divide in different precisions;
        # both come from the same byte, so the comparison is still exact.
        data = Float32[(16 * (x - 1) + 2 * (y - 1)) / 100 for x in 1:4, y in 1:2]
        @test tex_graph(dev, data) == Float32.(tex_draw(dev, data))
        # And with filtering on, where the texture unit rather than the shader
        # decides the value, so a wrong sampler would show.
        two = Float32[0f0 1f0]'
        @test tex_graph(dev, two; n = 16, filter = :linear) ==
              Float32.(tex_draw(dev, two; n = 16, filter = :linear))
    end
end
