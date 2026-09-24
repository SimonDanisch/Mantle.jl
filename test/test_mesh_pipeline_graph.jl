# A MeshPipeline drawn through the GRAPH, not by hand.
#
# Every mesh-pipeline test before this one called the immediate `draw!` or the
# example's own renderer, and none went through `draw!(::PassHandle, …)`. That
# path asks each stage what it does to its arguments BEFORE compiling
# (`vertextouches`/`fragmenttouches`), and on Metal the question was put to
# `shader.vertex` — which a mesh pipeline does not have. So the first mesh draw
# any real renderer recorded through a graph died with
# `FieldError(MeshPipeline, :vertex)`: RayMakie's RASTER mode, the first time
# the isubd demo was toggled out of path tracing. It then stopped twice more on
# the way to a frame: the walk reached the mesh-output intrinsics nothing had
# declared, and the fragment walk had no varying type for a mesh pipeline.
#
# Portable, gated on the capability and nothing else: the same assertions run on
# every backend that has a mesh pipeline.

using Test, Mantle
import KernelInterface as KI
using Mantle: Vec4f
using ColorTypes: BGRA
using ColorTypes.FixedPointNumbers: N0f8

# The colour comes from a BUFFER the mesh stage reads, so the draw is only right
# if the argument arrives, and the walk has something to classify.
# `set_mesh_outputs!` first: Vulkan requires it before any output is written.
function gm_mesh(out, colours)
    KI.set_mesh_outputs!(out, 3, 1)
    @inbounds c = colours[1]
    col = (c[1], c[2], c[3], c[4])
    KI.set_mesh_vertex!(out, 1, (position = (-1f0, -1f0, 0f0, 1f0), colour = col))
    KI.set_mesh_vertex!(out, 2, (position = ( 3f0, -1f0, 0f0, 1f0), colour = col))
    KI.set_mesh_vertex!(out, 3, (position = (-1f0,  3f0, 0f0, 1f0), colour = col))
    KI.set_mesh_triangle!(out, 1, 1, 2, 3)
    return nothing
end

# The same argument list on both stages, as every Makie render object has it —
# which is the case that reaches `fragmenttouches` for a mesh pipeline.
gm_frag(inputs, colours) = inputs.colour

const GM_PIPE = Mantle.MeshPipeline(;
    mesh = Mantle.MeshShader(gm_mesh;
        outputs = (colour = Mantle.Flat{NTuple{4,Float32}},),
        max_vertices = 3, max_primitives = 1,
        topology = Mantle.TriangleList(), threads = 1),
    fragment = Mantle.FragmentShader(gm_frag),
    cull = Mantle.NoCull(), depth = Mantle.DepthOff())

@testset "a MeshPipeline draws through the graph" begin
    be = Mantle.defaultbackend()
    if !Mantle.supports_mesh_pipeline(be)
        @info "no mesh pipeline on this backend; skipping"
        return
    end
    dev = Mantle.todevice(be)
    W, H = 32, 32
    colours = Mantle.Buffer(dev, [Vec4f(0, 1, 0, 1)])
    args = (colours,)

    # What the graph asks before it compiles anything. The buffer is READ and
    # nothing else: the mesh-output intrinsics are declared as writes, and that
    # must stay on the mesh object rather than leak onto a buffer whose loaded
    # value happens to be what gets stored into it.
    vt = Mantle.vertextouches(dev, GM_PIPE, args)
    @test length(vt) == 1
    @test only(vt).read
    @test !only(vt).write
    ft = Mantle.fragmenttouches(dev, GM_PIPE, args)
    @test length(ft) == 1
    @test !only(ft).write

    g = Mantle.Graph(dev)
    img = Mantle.Transient.Image(g, BGRA{N0f8}, (W, H))
    Mantle.render!(g, "mesh", img => Mantle.Clear((1f0, 0f0, 0f0, 1f0))) do p
        Mantle.draw!(p, GM_PIPE, args, 1; frag_args = args)
    end
    out = Mantle.Transient.Buffer(g, BGRA{N0f8}, W * H)
    Mantle.copy!(g, "read", out, img)
    plan = Mantle.Plan(g)
    Mantle.run!(plan)
    px = Array(Mantle.storage(out))

    # The triangle covers the whole target, so every pixel is the buffer's green
    # and none is the red it was cleared to.
    @test length(px) == W * H
    @test all(==(BGRA{N0f8}(0, 1, 0, 1)), px)

    # And again: a plan is replayed, and the second run is the recorded one.
    Mantle.run!(plan)
    @test all(==(BGRA{N0f8}(0, 1, 0, 1)), Array(Mantle.storage(out)))
end
