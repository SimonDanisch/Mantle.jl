# Turning a geometry stage into a mesh pipeline.
#
# No GPU. That is not a compromise here: the whole lowering except one builtin is
# portable code over an output object, so `runprimitive` with a primitive index
# and `KernelInterface.HostMeshOutput` runs exactly what a mesh stage runs and
# says what came out. `test/metal/test_graphics_metal.jl` draws the same bodies on
# a device and asserts the pixels; this asserts the vertices, the indices and the
# per-primitive values, which a frame can only show indirectly.

const LOW = Mantle
using GeometryBasics: Vec2f, Vec4f
const LOW_KI = parentmodule(Mantle.MeshEmitter)

# ── the bodies ───────────────────────────────────────────────────────────────
#
# A point becomes a quad, which is `scatter` in miniature: the vertex stage hands
# the geometry stage one vertex's attributes and the geometry stage expands it.

low_point_vertex(vid::LOW.VertexIndex, points) =
    (position = Vec4f(points[vid.value][1], points[vid.value][2], 0f0, 1f0),
     tint = Float32(vid.value))
low_point_vertex(points) = low_point_vertex(LOW.VertexIndex(LOW.vertex_index()), points)

function low_point_geometry(gs, prim, points)
    c = prim.position[1]
    t = prim.tint[1]
    for k in Int32(1):Int32(4)
        dx = (k == Int32(1) || k == Int32(2)) ? -0.25f0 : 0.25f0
        dy = (k == Int32(1) || k == Int32(3)) ? -0.25f0 : 0.25f0
        LOW.emit!(gs, (position = Vec4f(c[1] + dx, c[2] + dy, c[3], c[4]),
                       uv = Vec2f(dx, dy), tint = t))
    end
    LOW.endprimitive!(gs)
    return nothing
end

low_point_fragment(inputs, points) = Vec4f(inputs.tint, 0f0, 0f0, 1f0)

low_point_pipeline() = LOW.GraphicsPipeline(;
    vertex = LOW.VertexShader(low_point_vertex; outputs = (tint = Float32,)),
    geometry = LOW.GeometryShader(low_point_geometry;
                                  outputs = (uv = Vec2f, tint = LOW.Flat{Float32}),
                                  input = LOW.PointList(), output = LOW.TriangleStrip(),
                                  max_vertices = 4),
    fragment = LOW.FragmentShader(low_point_fragment),
    topology = LOW.PointList(), cull = LOW.NoCull(), depth = LOW.DepthOff())

# The `lines` shape: four input vertices per primitive out of an index buffer,
# and the emitted quad sits between the middle two.
low_seg_vertex(vid::LOW.VertexIndex, points) =
    (position = Vec4f(points[vid.value][1], points[vid.value][2], 0f0, 1f0),
     tint = Float32(vid.value))

function low_seg_geometry(gs, prim, points)
    a = prim.position[2]
    b = prim.position[3]
    t = prim.tint[1] + prim.tint[4]
    LOW.emit!(gs, (position = a, uv = Vec2f(0f0, 0f0), tint = t))
    LOW.emit!(gs, (position = b, uv = Vec2f(1f0, 1f0), tint = t))
    LOW.endprimitive!(gs)
    return nothing
end

low_seg_pipeline() = LOW.GraphicsPipeline(;
    vertex = LOW.VertexShader(low_seg_vertex; outputs = (tint = Float32,)),
    geometry = LOW.GeometryShader(low_seg_geometry;
                                  outputs = (uv = Vec2f, tint = LOW.Flat{Float32}),
                                  input = LOW.LineStripAdjacency(),
                                  output = LOW.LineStrip(), max_vertices = 2),
    fragment = LOW.FragmentShader((inputs, points) -> Vec4f(0f0, 0f0, 0f0, 1f0)),
    topology = LOW.LineStripAdjacency(), cull = LOW.NoCull(), depth = LOW.DepthOff())

@testset "the topology arithmetic a lowering walks with" begin
    # A LIST advances by its arity, a STRIP by one. That single difference is what
    # separates `LineListAdjacency` from `LineStripAdjacency`, which hand a
    # geometry stage the same four vertices.
    @test LOW.primitivestride(LOW.PointList()) == 1
    @test LOW.primitivestride(LOW.LineList()) == 2
    @test LOW.primitivestride(LOW.LineStrip()) == 1
    @test LOW.primitivestride(LOW.TriangleList()) == 3
    @test LOW.primitivestride(LOW.TriangleStrip()) == 1
    @test LOW.primitivestride(LOW.LineListAdjacency()) == 4
    @test LOW.primitivestride(LOW.LineStripAdjacency()) == 1

    @test LOW.primitivecount(LOW.PointList(), 5) == 5
    @test LOW.primitivecount(LOW.LineList(), 6) == 3
    @test LOW.primitivecount(LOW.LineStrip(), 6) == 5
    @test LOW.primitivecount(LOW.TriangleList(), 9) == 3
    @test LOW.primitivecount(LOW.TriangleStrip(), 5) == 3
    @test LOW.primitivecount(LOW.LineListAdjacency(), 8) == 2
    # `0 0 1 2 3 3`, which is what `lines_generate_indices` builds for four
    # points: a 4-wide window sliding one index at a time gives three segments.
    @test LOW.primitivecount(LOW.LineStripAdjacency(), 6) == 3

    # Fewer vertices than one primitive needs is a draw of nothing, not an error:
    # a `lines` plot with one point assembles no segment.
    @test LOW.primitivecount(LOW.TriangleList(), 2) == 0
    @test LOW.primitivecount(LOW.LineStripAdjacency(), 3) == 0

    # The inverse, and it stays `Int32` so a mesh stage does no 64-bit arithmetic.
    @test LOW.firstinputvertex(LOW.LineStripAdjacency(), 2) == 2
    @test LOW.firstinputvertex(LOW.LineList(), 3) == 5
    @test LOW.firstinputvertex(LOW.TriangleList(), 2) == 4
    @test LOW.firstinputvertex(LOW.PointList(), Int32(7)) === Int32(7)
end

@testset "a geometry pipeline is a type a backend can dispatch on" begin
    p = low_point_pipeline()
    @test typeof(p) <: LOW.GeometryPipeline
    @test LOW.GeometryPipeline <: LOW.GraphicsPipeline
    # And one without a geometry stage is NOT, which is the whole point — the two
    # `compile_draw` methods are selected by this and nothing else.
    novertex = LOW.GraphicsPipeline(;
        vertex = LOW.VertexShader(low_point_vertex),
        fragment = LOW.FragmentShader(low_point_fragment))
    @test !(typeof(novertex) <: LOW.GeometryPipeline)
end

@testset "lower_geometry_to_mesh reads one pipeline and builds the other" begin
    p = low_point_pipeline()
    mp = LOW.lower_geometry_to_mesh(p)
    @test mp isa LOW.MeshPipeline
    # No object stage: the host dispatches the threadgroups, one per input
    # primitive, so there is nothing to amplify.
    @test mp.object === nothing
    # The fragment stage is the SAME one, compiled against the same interface —
    # which is why nothing downstream of the lowering has to know it happened.
    @test mp.fragment === p.fragment
    @test LOW.fragmentinputs(mp) == LOW.fragmentinputs(p)
    @test LOW.fragmentinputtype(mp) == LOW.fragmentinputtype(p)
    # The state carries over unchanged.
    @test mp.blend === p.blend && mp.cull === p.cull && mp.depth === p.depth

    cfg = LOW.meshconfig(mp)
    @test cfg.max_vertices == 4
    # Four strip vertices are two triangles, and that is an upper bound: a body
    # that restarts the strip emits fewer from the same vertices, never more.
    @test cfg.max_primitives == 2
    @test cfg.topology === LOW.TriangleStrip()
    # One invocation per threadgroup, which is why the lowering never pads.
    @test cfg.threads == 1

    f = LOW.stagefunction(mp.mesh)
    @test f isa LOW.GeometryAsMesh
    # Named after the body an author wrote, so a shader dump or a compiler error
    # says which of the two lowerings produced the program.
    @test nameof(f) === :low_point_geometry_as_mesh

    # An indexed draw is a DIFFERENT shader — it reads the index buffer itself —
    # so a backend asks for the one its draw needs.
    @test LOW.stagefunction(LOW.lower_geometry_to_mesh(p; indexed = true).mesh) !== f
end

@testset "the lowered stage runs on the host, over HostMeshOutput" begin
    points = [Vec2f(0.0, 0.0), Vec2f(0.5, 0.5), Vec2f(-0.25, 0.75)]
    m = LOW.stagefunction(LOW.lower_geometry_to_mesh(low_point_pipeline()).mesh)

    out = LOW_KI.HostMeshOutput()
    # The SECOND point, so a lowering that ignored the primitive index would put
    # the quad in the wrong place and carry the wrong tint.
    LOW.runprimitive(m, out, 2, points)

    vs = LOW_KI.vertices(out)
    @test length(vs) == 4
    # `position` plus the SMOOTH outputs only: `tint` is `Flat` and belongs to the
    # primitive, so it is not on the vertex plane.
    @test keys(vs[1]) == (:position, :uv)
    @test [v.position for v in vs] == [Vec4f(0.25, 0.25, 0, 1), Vec4f(0.25, 0.75, 0, 1),
                                       Vec4f(0.75, 0.25, 0, 1), Vec4f(0.75, 0.75, 0, 1)]
    @test [v.uv for v in vs] == [Vec2f(-0.25, -0.25), Vec2f(-0.25, 0.25),
                                 Vec2f(0.25, -0.25), Vec2f(0.25, 0.25)]

    # Two triangles out of four strip vertices, and the winding alternates so both
    # face the same way — `(1,2,3)` then `(3,2,4)` and not `(2,3,4)`.
    @test LOW_KI.primitives(out) == [(Int32(1), Int32(2), Int32(3)),
                                     (Int32(3), Int32(2), Int32(4))]

    # The flat value once per primitive, and it is the SECOND vertex's: the vertex
    # body was evaluated at index 2 because that is the primitive being run.
    @test LOW_KI.primitivedata(out) == [(tint = 2f0,), (tint = 2f0,)]

    # Declared exactly, with no padding: one invocation per threadgroup means its
    # count IS the threadgroup's.
    @test out.declared[] == (Int32(4), Int32(2))
end

@testset "an indexed lowering fetches the vertex index itself" begin
    points = [Vec2f(0, 0), Vec2f(1, 0), Vec2f(2, 0), Vec2f(3, 0), Vec2f(4, 0)]
    # A `LineStripAdjacency` list as `lines_generate_indices` builds one: ZERO
    # based, because that is what a driver's index buffer holds.
    indices = UInt32[0, 0, 1, 2, 3, 4, 4]
    m = LOW.stagefunction(LOW.lower_geometry_to_mesh(low_seg_pipeline(); indexed = true).mesh)

    # Primitive 2 reads indices 2..5 — `0 1 2 3` zero-based — so the vertex body
    # runs at one-based 1, 2, 3, 4 and the segment is points[2] to points[3].
    out = LOW_KI.HostMeshOutput()
    LOW.runprimitive(m, out, 2, points, indices)
    vs = LOW_KI.vertices(out)
    @test [v.position for v in vs] == [Vec4f(1, 0, 0, 1), Vec4f(2, 0, 0, 1)]
    # `tint` is vertex 1 plus vertex 4 of the primitive: 1 + 4.
    @test LOW_KI.primitivedata(out) == [(tint = 5f0,)]
    @test LOW_KI.primitives(out) == [(Int32(1), Int32(2))]

    # …and the next primitive slides ONE index along, not four. A stride that read
    # the list form would draw every fourth segment and skip the rest.
    out2 = LOW_KI.HostMeshOutput()
    LOW.runprimitive(m, out2, 3, points, indices)
    @test [v.position for v in LOW_KI.vertices(out2)] == [Vec4f(2, 0, 0, 1), Vec4f(3, 0, 0, 1)]
    # Indices 3..6 are `1 2 3 4`, so one-based vertices 2..5 and `tint` is 2 + 5.
    @test LOW_KI.primitivedata(out2) == [(tint = 7f0,)]
end

@testset "a body that culls its primitive declares nothing" begin
    # `set_mesh_outputs!(out, _, 0)` is what a geometry stage returning early
    # becomes, and a threadgroup that declares zero primitives is dropped whole.
    # That is the right answer, and it is the reason a mesh stage cannot be probed
    # by writing to a buffer without also emitting something.
    culling(gs, prim, points) = nothing
    p = LOW.GraphicsPipeline(;
        vertex = LOW.VertexShader(low_point_vertex; outputs = (tint = Float32,)),
        geometry = LOW.GeometryShader(culling;
                                      outputs = (uv = Vec2f, tint = LOW.Flat{Float32}),
                                      input = LOW.PointList(),
                                      output = LOW.TriangleStrip(), max_vertices = 4),
        fragment = LOW.FragmentShader(low_point_fragment),
        topology = LOW.PointList())
    out = LOW_KI.HostMeshOutput()
    LOW.runprimitive(LOW.stagefunction(LOW.lower_geometry_to_mesh(p).mesh), out, 1,
                     [Vec2f(0, 0)])
    @test isempty(LOW_KI.primitives(out))
    @test out.declared[] == (Int32(0), Int32(0))
end

@testset "the lowering refuses what it cannot translate" begin
    # Nothing to lower.
    plain = LOW.GraphicsPipeline(; vertex = LOW.VertexShader(low_point_vertex),
                                  fragment = LOW.FragmentShader(low_point_fragment))
    @test_throws ArgumentError LOW.lower_geometry_to_mesh(plain)

    # The assembler's topology and the stage's declared input have to describe the
    # same primitive. They disagree silently otherwise: the stage would read a
    # different number of vertices than the assembler grouped.
    mismatched = LOW.GraphicsPipeline(;
        vertex = LOW.VertexShader(low_point_vertex; outputs = (tint = Float32,)),
        geometry = LOW.GeometryShader(low_point_geometry;
                                      outputs = (uv = Vec2f, tint = LOW.Flat{Float32}),
                                      input = LOW.TriangleList(),
                                      output = LOW.TriangleStrip(), max_vertices = 4),
        fragment = LOW.FragmentShader(low_point_fragment),
        topology = LOW.PointList())
    @test_throws ArgumentError LOW.lower_geometry_to_mesh(mismatched)

    # More than one invocation per primitive needs a builtin saying WHICH, and
    # there is no portable name for that yet.
    amplifying = LOW.GraphicsPipeline(;
        vertex = LOW.VertexShader(low_point_vertex; outputs = (tint = Float32,)),
        geometry = LOW.GeometryShader(low_point_geometry;
                                      outputs = (uv = Vec2f, tint = LOW.Flat{Float32}),
                                      input = LOW.PointList(),
                                      output = LOW.TriangleStrip(), max_vertices = 4,
                                      invocations = 2),
        fragment = LOW.FragmentShader(low_point_fragment),
        topology = LOW.PointList())
    @test_throws ArgumentError LOW.lower_geometry_to_mesh(amplifying)
end

@testset "a vertex body without the index parameter is refused by name" begin
    # The question is asked of the METHOD TABLE — "can this be called with a
    # leading `VertexIndex`" — so the answer comes from the shader's own signature
    # and cannot disagree with a flag recorded somewhere else. An untyped body of
    # the right arity therefore answers YES, which is the honest answer: such a
    # call works and the body reads the marker as its first argument.
    @test LOW.wantsvertexindex(low_point_vertex, (Vector{Vec2f},))
    @test !LOW.wantsvertexindex(low_point_geometry, (Vector{Vec2f},))

    @test LOW.requirevertexindex(low_point_vertex, Tuple{Vector{Vec2f}}) === nothing
    # And the error names the body and shows both spellings, because a
    # `MethodError` from inside a shader compilation says neither.
    err = try
        LOW.requirevertexindex(low_point_geometry, Tuple{Vector{Vec2f}})
    catch e
        e
    end
    @test err isa ErrorException
    @test occursin("VertexIndex", err.msg)
    @test occursin("low_point_geometry", err.msg)
end
