# A stage with more arguments than a buffer table has entries.
#
# Metal's table has 31, and a stage whose arguments were bound one per entry
# failed at pipeline creation with "indirect argument buffer resources exceeded
# (37/31)". RayMakie's port of GLMakie's mesh shader has 51 arguments, and the
# isubd overlay 37. Lava never had the limit because it packs every argument
# into one block, which is what the Metal backend does now too.
#
# The packing then broke the geometry lowering: it asked whether the vertex body
# takes a leading `VertexIndex` for the PACKED argument types, which a typed body
# never matches, so every Makie `scatter!` and `lines!` was refused on Metal.
# Past that, the lowering forwarded the arguments by splatting, and inference
# leaves a splat of more than 32 elements dynamic. Lava's four stage wrappers
# splatted the same way, so on Vulkan this file pins those too.
#
# Every argument has to land in its own slot, so the check is a weighted sum:
# argument i carries the value i and is weighted by i, and any two slots swapped,
# dropped or shifted change the total.

using Test, Mantle
import KernelInterface as KI
using Mantle: Vec4f
using ColorTypes: BGRA
using ColorTypes.FixedPointNumbers: N0f8

const MS_N = 39
const MS_EXPECTED = Float32(sum(i * i for i in 1:MS_N))

@inline weighted(a::NTuple{MS_N,Float32}) = sum(ntuple(i -> Float32(i) * a[i], Val(MS_N)))

# Green when every argument arrived in its slot, red otherwise.
@inline verdict(colours, a) = weighted(a) == MS_EXPECTED ? (@inbounds colours[1]) : Vec4f(1, 0, 0, 1)

function ms_vertex(colours, a::Vararg{Float32,MS_N})
    v = vertex_index()
    x = v == Int32(2) ? 3f0 : -1f0
    y = v == Int32(3) ? 3f0 : -1f0
    return (position = Vec4f(x, y, 0f0, 1f0), tint = verdict(colours, a))
end

ms_fragment(inputs, colours, a::Vararg{Float32,MS_N}) = inputs.tint * verdict(colours, a)

function ms_mesh(out, colours, a::Vararg{Float32,MS_N})
    KI.set_mesh_outputs!(out, 3, 1)
    c = verdict(colours, a)
    col = (c[1], c[2], c[3], c[4])
    KI.set_mesh_vertex!(out, 1, (position = (-1f0, -1f0, 0f0, 1f0), tint = col))
    KI.set_mesh_vertex!(out, 2, (position = ( 3f0, -1f0, 0f0, 1f0), tint = col))
    KI.set_mesh_vertex!(out, 3, (position = (-1f0,  3f0, 0f0, 1f0), tint = col))
    KI.set_mesh_triangle!(out, 1, 1, 2, 3)
    return nothing
end

ms_mesh_fragment(inputs, colours, a::Vararg{Float32,MS_N}) =
    Vec4f(inputs.tint...) * verdict(colours, a)

# A geometry stage, which a backend without one lowers onto a mesh stage. The
# vertex body's arguments are TYPED: the lowering asks the method table whether
# the body takes a leading `VertexIndex`, and an untyped body answered yes to the
# packed block where RayMakie's `scatter_vertex` answered no.
function ms_geom_vertex(vid::Mantle.VertexIndex, colours::AbstractVector{Vec4f}, a::Vararg{Float32,MS_N})
    return (position = Vec4f(0f0, 0f0, 0f0, 1f0), tint = verdict(colours, a))
end

# Written out, not splatted: inference resolves a splat of at most 32 elements.
@generated function ms_geom_vertex(args::Vararg{Any,MS_N + 1})
    return :(ms_geom_vertex(Mantle.VertexIndex(vertex_index()), $((:(args[$i]) for i in 1:(MS_N + 1))...)))
end

function ms_geometry(gs, prim, colours::AbstractVector{Vec4f}, a::Vararg{Float32,MS_N})
    t = prim.tint[1] * verdict(colours, a)
    col = (t[1], t[2], t[3], t[4])
    Mantle.emit!(gs, (position = Vec4f(-1f0, -1f0, 0f0, 1f0), tint = col))
    Mantle.emit!(gs, (position = Vec4f( 3f0, -1f0, 0f0, 1f0), tint = col))
    Mantle.emit!(gs, (position = Vec4f(-1f0,  3f0, 0f0, 1f0), tint = col))
    Mantle.endprimitive!(gs)
    return nothing
end

const MS_GEOMETRY = Mantle.GraphicsPipeline(;
    vertex = Mantle.VertexShader(ms_geom_vertex; outputs = (tint = Vec4f,)),
    geometry = Mantle.GeometryShader(ms_geometry;
                                     outputs = (tint = Mantle.Flat{NTuple{4,Float32}},),
                                     input = Mantle.PointList(),
                                     output = Mantle.TriangleStrip(), max_vertices = 3),
    fragment = Mantle.FragmentShader(ms_mesh_fragment),
    topology = Mantle.PointList(), cull = Mantle.NoCull(), depth = Mantle.DepthOff())

const MS_RASTER = Mantle.Rasterizer(; vertex = Mantle.VertexShader(ms_vertex; outputs = (tint = Vec4f,)),
                                      fragment = Mantle.FragmentShader(ms_fragment),
                                      topology = Mantle.TriangleList(),
                                      blend = Mantle.Opaque(),
                                      cull = Mantle.NoCull(),
                                      depth = Mantle.DepthOff())

const MS_MESH = Mantle.MeshPipeline(;
    mesh = Mantle.MeshShader(ms_mesh;
        outputs = (tint = Mantle.Flat{NTuple{4,Float32}},),
        max_vertices = 3, max_primitives = 1,
        topology = Mantle.TriangleList(), threads = 1),
    fragment = Mantle.FragmentShader(ms_mesh_fragment),
    cull = Mantle.NoCull(), depth = Mantle.DepthOff())

function ms_draw(dev, pipe, args, count)
    W, H = 16, 16
    g = Mantle.Graph(dev)
    img = Mantle.Transient.Image(g, BGRA{N0f8}, (W, H))
    Mantle.render!(g, "many", img => Mantle.Clear((0f0, 0f0, 1f0, 1f0))) do p
        Mantle.draw!(p, pipe, args, count; frag_args = args)
    end
    out = Mantle.Transient.Buffer(g, BGRA{N0f8}, W * H)
    Mantle.copy!(g, "read", out, img)
    Mantle.run!(Mantle.Plan(g))
    return Array(Mantle.storage(out))
end

@testset "a stage with more arguments than a buffer table" begin
    be = Mantle.defaultbackend()
    green = BGRA{N0f8}(0, 1, 0, 1)
    if !Mantle.supports_graphics(be)
        @info "no graphics pipeline on this backend; skipping"
        return
    end
    dev = Mantle.todevice(be)
    colours = Mantle.Buffer(dev, [Vec4f(0, 1, 0, 1)])
    args = (colours, ntuple(Float32, MS_N)...)
    @test length(args) > 31

    @testset "vertex and fragment" begin
        @test all(==(green), ms_draw(dev, MS_RASTER, args, 3))
    end
    @testset "vertex, geometry and fragment" begin
        if Mantle.supports_geometry_stage(be) || Mantle.supports_mesh_pipeline(be)
            @test all(==(green), ms_draw(dev, MS_GEOMETRY, args, 1))
        else
            @info "no geometry stage and nothing to lower it onto; skipping"
        end
    end
    @testset "mesh and fragment" begin
        if Mantle.supports_mesh_pipeline(be)
            @test all(==(green), ms_draw(dev, MS_MESH, args, 1))
        else
            @info "no mesh pipeline on this backend; skipping"
        end
    end
end
