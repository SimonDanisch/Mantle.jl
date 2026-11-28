# A varying spelled as a tuple.
#
# `NTuple{N,Float32}` and `Vec{N,Float32}` are the same N floats, and Metal's
# stages take either. Lava's varying intrinsics had methods for the vector
# spellings only, so a stage writing or reading a tuple inferred to `Union{}`,
# and Mantle's access walk refused the draw ("a signature it cannot infer").
# `test_many_stage_args.jl` surfaced it through a mesh stage and a geometry
# stage writing `Flat{NTuple{4,Float32}}`. A flat `Vec3f` fragment input had the
# same gap: the emitter knew `_lava_gfx_input_flat_vec3`, but nothing called it.
#
# A fragment returning its colour as a tuple was read as one target per element.
#
# Every spelling, smooth and flat, through every stage that writes or reads a
# varying. Each value is constant across the primitive, so interpolation hands
# it back unchanged and the fragment can compare it.

using Test, Mantle
using Mantle: Vec4f, Vec3f, Vec2f
using ColorTypes: BGRA
using ColorTypes.FixedPointNumbers: N0f8

const TV_OUTPUTS = (a = NTuple{2,Float32}, b = NTuple{3,Float32}, c = NTuple{4,Float32},
                    fa = Mantle.Flat{NTuple{2,Float32}}, fb = Mantle.Flat{NTuple{3,Float32}},
                    fc = Mantle.Flat{NTuple{4,Float32}}, fv = Mantle.Flat{Vec3f})

@inline tv_values() = (a = (0.25f0, 0.5f0), b = (0.125f0, 0.375f0, 0.625f0),
                       c = (0.5f0, 0.25f0, 0.75f0, 1f0),
                       fa = (1f0, 2f0), fb = (3f0, 4f0, 5f0), fc = (6f0, 7f0, 8f0, 9f0),
                       fv = Vec3f(10, 11, 12))

@inline tv_close(x, y) = all(ntuple(i -> abs(x[i] - y[i]) < 1f-3, Val(length(y))))

@inline function tv_fragment(inputs, colours)
    w = tv_values()
    ok = tv_close(inputs.a, w.a) & tv_close(inputs.b, w.b) & tv_close(inputs.c, w.c) &
         (inputs.fa == w.fa) & (inputs.fb == w.fb) & (inputs.fc == w.fc) &
         (inputs.fv == w.fv) & (inputs.fv isa Vec3f) & (inputs.c isa NTuple{4,Float32})
    return ok ? (@inbounds colours[1]) : Vec4f(1, 0, 0, 1)
end

# The colour as the varying carried it, a tuple, when every input arrived.
@inline function tv_tuple_fragment(inputs, colours)
    c = tv_fragment(inputs, colours)
    return (c[1], c[2], c[3], c[4])
end

tv_two_targets(inputs, colours) = (Vec4f(0, 1, 0, 1), Vec4f(1, 0, 0, 1))

function tv_vertex(colours)
    v = vertex_index()
    x = v == Int32(2) ? 3f0 : -1f0
    y = v == Int32(3) ? 3f0 : -1f0
    return (; position = Vec4f(x, y, 0f0, 1f0), tv_values()...)
end

function tv_point_vertex(vid::Mantle.VertexIndex, colours)
    return (; position = Vec4f(0f0, 0f0, 0f0, 1f0), tv_values()...)
end
tv_point_vertex(colours) = tv_point_vertex(Mantle.VertexIndex(vertex_index()), colours)

# Reads what the vertex stage wrote, as the tuples it was declared as, and
# passes it on: a wrong read reaches the fragment as a wrong value.
function tv_geometry(gs, prim, colours)
    v = (a = prim.a[1], b = prim.b[1], c = prim.c[1],
         fa = prim.fa[1], fb = prim.fb[1], fc = prim.fc[1], fv = prim.fv[1])
    Mantle.emit!(gs, (; position = Vec4f(-1f0, -1f0, 0f0, 1f0), v...))
    Mantle.emit!(gs, (; position = Vec4f( 3f0, -1f0, 0f0, 1f0), v...))
    Mantle.emit!(gs, (; position = Vec4f(-1f0,  3f0, 0f0, 1f0), v...))
    Mantle.endprimitive!(gs)
    return nothing
end

function tv_mesh(out, colours)
    Mantle.set_mesh_outputs!(out, 3, 1)
    Mantle.set_mesh_vertex!(out, 1, (; position = (-1f0, -1f0, 0f0, 1f0), tv_values()...))
    Mantle.set_mesh_vertex!(out, 2, (; position = ( 3f0, -1f0, 0f0, 1f0), tv_values()...))
    Mantle.set_mesh_vertex!(out, 3, (; position = (-1f0,  3f0, 0f0, 1f0), tv_values()...))
    Mantle.set_mesh_triangle!(out, 1, 1, 2, 3)
    return nothing
end

function tv_draw(dev, pipe, args, count)
    W, H = 8, 8
    g = Mantle.Graph(dev)
    img = Mantle.Transient.Image(g, BGRA{N0f8}, (W, H))
    Mantle.render!(g, "tuples", img => Mantle.Clear((0f0, 0f0, 1f0, 1f0))) do p
        Mantle.draw!(p, pipe, args, count; frag_args = args)
    end
    out = Mantle.Transient.Buffer(g, BGRA{N0f8}, W * H)
    Mantle.copy!(g, "read", out, img)
    Mantle.run!(Mantle.record!(Mantle.Plan(g)))
    return Array(Mantle.storage(out))
end

@testset "a varying spelled as a tuple" begin
    be = Mantle.defaultbackend()
    if !Mantle.supports_graphics(be)
        @info "no graphics pipeline on this backend; skipping"
        return
    end
    dev = Mantle.todevice(be)
    green = BGRA{N0f8}(0, 1, 0, 1)
    colours = Mantle.Buffer(dev, [Vec4f(0, 1, 0, 1)])
    args = (colours,)

    @testset "vertex and fragment" begin
        pipe = Mantle.Rasterizer(; vertex = Mantle.VertexShader(tv_vertex; outputs = TV_OUTPUTS),
                                   fragment = Mantle.FragmentShader(tv_fragment),
                                   topology = Mantle.TriangleList(), blend = Mantle.Opaque(),
                                   cull = Mantle.NoCull(), depth = Mantle.DepthOff())
        @test all(==(green), tv_draw(dev, pipe, args, 3))
    end
    # A colour IS a four-tuple, and Lava's fragment output read any tuple as
    # one target per element: the attachment got four locations of one float
    # each, and a mesh pipeline whose fragment returned its `Flat{NTuple{4,
    # Float32}}` varying drew nothing (`test_mesh_pipeline_graph.jl`).
    @testset "a colour returned as a tuple is one target" begin
        pipe = Mantle.Rasterizer(; vertex = Mantle.VertexShader(tv_vertex; outputs = TV_OUTPUTS),
                                   fragment = Mantle.FragmentShader(tv_tuple_fragment),
                                   topology = Mantle.TriangleList(), blend = Mantle.Opaque(),
                                   cull = Mantle.NoCull(), depth = Mantle.DepthOff())
        @test all(==(green), tv_draw(dev, pipe, args, 3))
    end
    # And the other side of that rule: a tuple of colours is still one per target.
    @testset "a tuple of colours is one per target" begin
        pipe = Mantle.Rasterizer(; vertex = Mantle.VertexShader(tv_vertex; outputs = TV_OUTPUTS),
                                   fragment = Mantle.FragmentShader(tv_two_targets),
                                   topology = Mantle.TriangleList(), blend = Mantle.Opaque(),
                                   cull = Mantle.NoCull(), depth = Mantle.DepthOff())
        g = Mantle.Graph(dev)
        a = Mantle.Transient.Image(g, BGRA{N0f8}, (8, 8))
        b = Mantle.Transient.Image(g, BGRA{N0f8}, (8, 8))
        Mantle.render!(g, "two", a => Mantle.Clear((0f0, 0f0, 1f0, 1f0)),
                                 b => Mantle.Clear((0f0, 0f0, 1f0, 1f0))) do p
            Mantle.draw!(p, pipe, args, 3; frag_args = args)
        end
        # Persistent, not transient: nothing in the graph reads either copy, so
        # two transients may share memory and the second copy lands in both.
        oa = Mantle.Buffer(dev, BGRA{N0f8}, 64)
        ob = Mantle.Buffer(dev, BGRA{N0f8}, 64)
        Mantle.copy!(g, "read a", oa, a)
        Mantle.copy!(g, "read b", ob, b)
        Mantle.run!(Mantle.record!(Mantle.Plan(g)))
        @test all(==(green), Array(Mantle.storage(oa)))
        @test all(==(BGRA{N0f8}(1, 0, 0, 1)), Array(Mantle.storage(ob)))
        Mantle.free!(oa)
        Mantle.free!(ob)
    end
    @testset "vertex, geometry and fragment" begin
        if Mantle.supports_geometry_stage(be) || Mantle.supports_mesh_pipeline(be)
            pipe = Mantle.GraphicsPipeline(;
                vertex = Mantle.VertexShader(tv_point_vertex; outputs = TV_OUTPUTS),
                geometry = Mantle.GeometryShader(tv_geometry; outputs = TV_OUTPUTS,
                                                 input = Mantle.PointList(),
                                                 output = Mantle.TriangleStrip(), max_vertices = 3),
                fragment = Mantle.FragmentShader(tv_fragment),
                topology = Mantle.PointList(), cull = Mantle.NoCull(), depth = Mantle.DepthOff())
            @test all(==(green), tv_draw(dev, pipe, args, 1))
        else
            @info "no geometry stage and nothing to lower it onto; skipping"
        end
    end
    @testset "mesh and fragment" begin
        if Mantle.supports_mesh_pipeline(be)
            pipe = Mantle.MeshPipeline(;
                mesh = Mantle.MeshShader(tv_mesh; outputs = TV_OUTPUTS,
                                         max_vertices = 3, max_primitives = 1,
                                         topology = Mantle.TriangleList(), threads = 1),
                fragment = Mantle.FragmentShader(tv_fragment),
                cull = Mantle.NoCull(), depth = Mantle.DepthOff())
            @test all(==(green), tv_draw(dev, pipe, args, 1))
        else
            @info "no mesh pipeline on this backend; skipping"
        end
    end
    Mantle.free!(colours)
end
