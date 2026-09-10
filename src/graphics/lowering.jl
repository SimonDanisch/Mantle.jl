# Turning a geometry stage into a mesh pipeline.
#
# ── Why it is needed ─────────────────────────────────────────────────────────
#
# A `GraphicsPipeline` with a `GeometryShader` runs natively on Vulkan and cannot
# run at all on Metal: Apple removed the geometry stage and replaced it with the
# mesh pipeline (object + mesh stages). Everything a geometry shader does is
# expressible on a mesh stage — one invocation reads a primitive's vertices and
# emits new ones — so the translation is mechanical, and doing it once here means
# a shader author never writes it twice.
#
# What is affected today, in tree: RayMakie's `scatter` and `lines` overlays, the
# only two `GeometryShader` construction sites. Scatter is also the sprite path,
# so text goes through it. Without this, a Metal figure has no axis grid, no
# ticks, no labels, and no scatter or line plot.
#
# ── One mesh threadgroup per input primitive, one thread ─────────────────────
#
# That single thread evaluates the vertex body once per input vertex of its
# primitive — `inputvertices(topology)` times, in a plain loop — gathers the
# results into the `prim` NamedTuple the geometry body already reads
# (`prim.world_pos[1]` and so on: one field per declared output, each an NTuple
# over the primitive's vertices), builds a `MeshEmitter` over the output object,
# and calls the geometry body unchanged.
#
# An earlier reading of this had one thread PER INPUT VERTEX, with the vertex
# outputs meeting in threadgroup memory behind a barrier. That was forced by a
# constraint that does not exist: it is the only arrangement in which
# `vertex_index()` — a zero-argument builtin — can answer correctly for each
# vertex, because the answer has to come from the thread's own identity. Pass the
# index to the vertex body INSTEAD and one thread can evaluate it at as many
# indices as it likes. Then there is no threadgroup memory, no barrier, and no
# special handling for the overlapping primitives of a strip or adjacency
# topology — `MeshEmitter` already turns strip emission into explicit indices.
#
# So the contract is `KernelInterface.VertexIndex`: a vertex body may declare a
# leading `::VertexIndex` parameter, and one that does is passed the index rather
# than calling `vertex_index()`. Opt-in per shader; a body that does not declare
# it is untouched on either backend, and a body that declares BOTH spellings runs
# natively where the geometry stage exists and lowers where it does not.
#
# ── Why one thread is also why nothing has to be padded ──────────────────────
#
# `MeshEmitter` reserves a fixed slot range per invocation because a mesh
# threadgroup declares ONE output count for all of them, so an invocation that
# emits fewer primitives than it reserved leaves slots inside the count — which
# `finish!` fills with primitives that draw nothing. With one invocation per
# threadgroup there is no such gap: what it wrote IS the threadgroup's count, so
# the lowering declares exactly `nprimitives(e)` and never pads. A body that
# culled everything declares zero, and the driver eliminates the threadgroup.
#
# The cost of one thread is that a threadgroup uses one lane of a SIMD group. It
# is the right trade for the overlays this exists for — an axis's ticks and
# labels are a handful of primitives against a fragment-bound sprite pass — and
# widening it is not free: `threads > 1` needs `set_mesh_outputs!` called once
# with the group's total, which is a barrier and a reduction, or `finish!`'s
# padding and the wasted slots that come with it. Measure before paying either.

"""
    GeometryPipeline

A [`GraphicsPipeline`](@ref) that HAS a geometry stage, as a type a backend can
dispatch on — which is what lets one whose geometry stage does not exist route the
whole draw through the lowering instead of branching on a field.

The two leading bounds are not decoration. `GraphicsPipeline` declares
`V<:VertexShader` and `F<:FragmentShader`, and spelling those positions `<:Any`
builds a type that is not a subtype of `GraphicsPipeline` at all: the method is
then simply never selected, silently, and the general one runs instead.
"""
const GeometryPipeline =
    GraphicsPipeline{<:VertexShader, <:FragmentShader, <:GeometryShader}

"""
    GeometryAsMesh{IT,OT,NP,FN,IX}(vertexbody, geometrybody)

A mesh stage that runs a vertex body and a geometry body, in one invocation.

`IT` is the DRAW's input topology and `OT` what the geometry body emits, `NP` the
output primitive bound, `FN` the `Flat`-declared output names, and `IX` whether
the draw is indexed — in which case the last argument is the index buffer and
everything before it is what the two bodies take.

Built by [`lower_geometry_to_mesh`](@ref), which is where the shape of the
argument list is decided; nothing else should construct one.
"""
struct GeometryAsMesh{IT<:Topology, OT<:Topology, NP, FN, IX, VF, GF}
    vertex::VF
    geometry::GF
end

# The two bodies' types are inferred and the five that describe the lowering are
# given, which the default constructor cannot do — it takes all seven or none.
GeometryAsMesh{IT,OT,NP,FN,IX}(vertex::VF, geometry::GF) where {IT,OT,NP,FN,IX,VF,GF} =
    GeometryAsMesh{IT,OT,NP,FN,IX,VF,GF}(vertex, geometry)

# A backend names the compiled stage after the function it compiled, and a
# callable struct has no name of its own. The geometry body's is the one worth
# seeing: it is what the author wrote, and `_as_mesh` says which of the two
# lowerings produced the program in a shader dump or a compiler error.
Base.nameof(m::GeometryAsMesh) = Symbol(nameof(m.geometry), :_as_mesh)

# Which input primitive this threadgroup is for is the ONLY thing the mesh stage
# reads from a builtin, so it is the only thing that stops the whole lowering from
# running on the host. `runprimitive` takes it as an argument for exactly that
# reason: over `KernelInterface.HostMeshOutput` the emitted vertices, indices and
# per-primitive values can be read back and diffed against what the GPU produced,
# which is what makes this worth more than a per-backend rewrite.
function (m::GeometryAsMesh)(out, args::Vararg{Any,N}) where {N}
    return runprimitive(m, out, mesh_group_index(), args...)
end

"""
    runprimitive(m::GeometryAsMesh, out, primitive::Integer, args...)

Run one input primitive of a lowered geometry stage into the output object `out`.

`primitive` counts from one. On a mesh stage it is `mesh_group_index()`; on the
host it is whichever primitive is being checked.
"""
function runprimitive(m::GeometryAsMesh{IT,OT,NP,FN,false}, out, primitive::Integer,
                      args::Vararg{Any,N}) where {IT,OT,NP,FN,N}
    # A NON-INDEXED draw: the vertex index IS the position in the stream.
    base = firstinputvertex(IT(), Int32(primitive))
    vs = ntuple(k -> m.vertex(VertexIndex(base + Int32(k) - Int32(1)), args...),
                Val(inputvertices(IT())))
    return emitprimitive(m, out, vs, args...)
end

# An INDEXED draw. `vertex_index()` on a real vertex stage is the value FETCHED
# from the index buffer, not the position in the draw, so the lowered stage has
# to do the fetch itself — which is why the index buffer becomes one of the mesh
# stage's own arguments and why `lines` is not simply `scatter` again.
#
# The buffer holds what a driver's index buffer holds, counting vertices from
# ZERO, and `VertexIndex` carries what `vertex_index()` would answer, counting
# from one. The `+ 1` is that conversion and nothing else.
function runprimitive(m::GeometryAsMesh{IT,OT,NP,FN,true}, out, primitive::Integer,
                      args::Vararg{Any,N}) where {IT,OT,NP,FN,N}
    indices = args[N]
    rest = Base.front(args)
    base = firstinputvertex(IT(), Int32(primitive))
    vs = ntuple(Val(inputvertices(IT()))) do k
        i = Int32(indices[base + Int32(k) - Int32(1)]) + Int32(1)
        return m.vertex(VertexIndex(i), rest...)
    end
    return emitprimitive(m, out, vs, rest...)
end

"""
    emitprimitive(m::GeometryAsMesh, out, vs::Tuple, args...)

Run the geometry body over the gathered vertices and declare what it wrote.

Separate from the two call methods above because it is everything they share: the
only difference between an indexed draw and a direct one is where the vertex
index comes from.
"""
@inline function emitprimitive(m::GeometryAsMesh{IT,OT,NP,FN}, out, vs::Tuple,
                               args::Vararg{Any,N}) where {IT,OT,NP,FN,N}
    e = MeshEmitter{OT,FN}(out, 1, 1, NP)
    # `@inline` at the CALL, and it is load-bearing. `MeshEmitter` is mutable —
    # `emit!` advances its cursors — so an emitter handed to a function that is
    # not inlined ESCAPES, and an escaping mutable is a heap allocation, which no
    # GPU has: the shader fails to compile with `gc_pool_alloc` in the trace and
    # nothing pointing at the emitter. Inlined, it is a handful of registers.
    @inline m.geometry(e, transposeprimitive(vs), args...)
    # Once, and with the emitter's own counts: one invocation per threadgroup
    # means its count is the threadgroup's, so there is nothing to pad and
    # nothing to reduce over. Zero primitives is a real answer — the body culled
    # the primitive — and the threadgroup is then eliminated whole.
    set_mesh_outputs!(out, nvertices(e), nprimitives(e))
    return nothing
end

"""
    transposeprimitive(vs::Tuple) -> NamedTuple

The primitive's vertex outputs, from one NamedTuple per vertex to one tuple per
field.

`prim.position[2]` is the second vertex's clip position — which is how a geometry
body already reads its input, so this is the shape the lowering has to produce
and not a choice. `@generated` because it is pure type arithmetic and sits in the
one loop a mesh stage runs.
"""
@generated function transposeprimitive(vs::Tuple)
    n = fieldcount(vs)
    n > 0 || error("transposeprimitive: a primitive has no vertices, which cannot " *
                   "happen — `inputvertices` is at least one for every topology.")
    V = fieldtype(vs, 1)
    V <: NamedTuple || error(
        "transposeprimitive: a vertex body must return a NamedTuple of `position` " *
        "plus its declared outputs; this one returned a $V.")
    for k in 2:n
        fieldtype(vs, k) === V || error(
            "transposeprimitive: the vertex body returned a $V for one vertex of " *
            "the primitive and a $(fieldtype(vs, k)) for another. Every vertex of " *
            "a primitive runs the same body, so the two can only differ if the " *
            "body is not type stable.")
    end
    K = fieldnames(V)
    fields = Expr(:tuple, (Expr(:tuple, (:(Base.getfield(vs[$k], $(QuoteNode(name))))
                                         for k in 1:n)...) for name in K)...)
    return Expr(:block, Expr(:meta, :inline), :(NamedTuple{$K}($fields)))
end

"""
    lower_geometry_to_mesh(p::GraphicsPipeline; indexed = false) -> MeshPipeline

Rewrite a pipeline's vertex + geometry stages as a mesh stage.

`indexed` is whether the DRAW supplies an index buffer, which is a property of
the draw and not of the pipeline: an indexed draw's lowered mesh stage reads the
index buffer as its last argument, so the two are different shaders and a backend
asks for the one its draw needs.

The mesh stage's `outputs` are the geometry stage's verbatim, so the fragment
stage is compiled against exactly what it was before — [`fragmentinputs`](@ref)
answers the same thing for either pipeline.

A backend with no geometry stage runs a geometry pipeline by lowering it, which
is why `supports_geometry_stage` is not a question a renderer has to branch on.
The vertex body must accept a leading `KernelInterface.VertexIndex`; see the top
of `src/graphics/lowering.jl` for why the index cannot come from a builtin here.
"""
function lower_geometry_to_mesh(p::GraphicsPipeline; indexed::Bool = false)
    g = p.geometry
    g === nothing && throw(ArgumentError(
        "lower_geometry_to_mesh: this pipeline has no geometry stage, so there is " *
        "nothing to lower. Draw it as it is."))
    (p.tess_control === nothing && p.tess_eval === nothing) && return _lower(p, g, indexed)
    throw(ArgumentError(
        "lower_geometry_to_mesh: this pipeline has a tessellation stage as well as " *
        "a geometry stage, and only the geometry stage can be lowered onto a mesh " *
        "one. A mesh pipeline has no tessellator: what a tessellation control and " *
        "evaluation pair produces would have to be evaluated by the mesh stage " *
        "itself, which is a different translation and not this one."))
end

function _lower(p::GraphicsPipeline, g::GeometryShader, indexed::Bool)
    cfg = stageconfig(g)
    # The assembler's topology decides the WALK — how many vertices a primitive
    # is and how far the next one starts along — and the geometry stage declares
    # what it expects to see. Vulkan requires the two to agree; here a
    # disagreement would silently gather the wrong vertices, so it is checked.
    inputvertices(p.topology) == inputvertices(cfg.input_topology) || throw(ArgumentError(
        "lower_geometry_to_mesh: the draw assembles $(p.topology), which is " *
        "$(inputvertices(p.topology)) vertices per primitive, and the geometry " *
        "stage declares $(cfg.input_topology), which is " *
        "$(inputvertices(cfg.input_topology)). The stage would read a different " *
        "primitive than the assembler built."))
    cfg.invocations == 1 || throw(ArgumentError(
        "lower_geometry_to_mesh: this geometry stage asks for $(cfg.invocations) " *
        "invocations per primitive, and the lowering runs one. Which invocation a " *
        "body is would have to reach it as a builtin, and there is no portable " *
        "name for that yet — nothing in tree amplifies, so the name comes with " *
        "the first thing that does rather than being guessed at now."))
    # An upper bound, and it has to be one: a body that restarts a strip emits
    # FEWER primitives from the same vertices than a contiguous run would, never
    # more, and a list topology's count does not depend on restarts at all.
    np = primitivecount(cfg.output_topology, cfg.max_vertices)
    np > 0 || throw(ArgumentError(
        "lower_geometry_to_mesh: `max_vertices = $(cfg.max_vertices)` is not " *
        "enough for one $(cfg.output_topology) primitive, so this stage cannot " *
        "emit anything."))
    f = GeometryAsMesh{typeof(p.topology), typeof(cfg.output_topology), np,
                       flatoutputs(g), indexed}(stagefunction(p.vertex),
                                                stagefunction(g))
    mesh = MeshShader(f; outputs = stageoutputs(g),
                        max_vertices = cfg.max_vertices, max_primitives = np,
                        topology = cfg.output_topology, threads = 1)
    return MeshPipeline(; mesh, fragment = p.fragment,
                          blend = p.blend, cull = p.cull, depth = p.depth)
end

"""
    requirevertexindex(f, argtypes::Type)

Throw unless the vertex body `f` accepts a leading `VertexIndex` for `argtypes`.

Asked by a backend before it compiles a lowered mesh stage, because the
alternative is a `MethodError` raised from inside a shader compilation — which
names the body, but not what is missing from it or why a mesh stage needs it.
"""
function requirevertexindex(@nospecialize(f), @nospecialize(argtypes::Type))
    tt = Tuple(argtypes.parameters)
    wantsvertexindex(f, tt) && return nothing
    error("""
        $f cannot be lowered onto a mesh stage: it does not accept a leading
        `KernelInterface.VertexIndex`.

        A mesh stage has no `vertex_index()`. One invocation evaluates the vertex
        body once per vertex of its input primitive, at indices IT computes, and a
        zero-argument builtin cannot answer differently on each of those calls.

        Declare the parameter:

            function $(nameof(f))(vid::VertexIndex, buffers...)
                i = vid.value
                …
            end

        Keep the old spelling beside it if the shader also runs where the geometry
        stage exists natively — the two are the same number:

            $(nameof(f))(buffers...) = $(nameof(f))(VertexIndex(vertex_index()), buffers...)

        Argument types this was asked for: $tt""")
end
