# Lowers a geometry stage onto a mesh pipeline, for backends without one (Metal).
# One mesh threadgroup of one thread per input primitive: the thread evaluates the
# vertex body once per input vertex, passing the index as a leading
# `KernelInterface.VertexIndex` since a zero-argument `vertex_index()` cannot
# answer per call. The emitter's count is then the threadgroup's, so nothing is
# padded; `threads > 1` would need a reduction before `set_mesh_outputs!`.

"""
    GeometryPipeline

A [`GraphicsPipeline`](@ref) with a geometry stage, as a type a backend without
that stage dispatches on to route the draw through the lowering. The leading
bounds must repeat `GraphicsPipeline`'s `<:VertexShader`/`<:FragmentShader`:
`<:Any` there builds a type that is not a subtype, and the method silently never
matches.
"""
const GeometryPipeline =
    GraphicsPipeline{<:VertexShader, <:FragmentShader, <:GeometryShader}

"""
    GeometryAsMesh{IT,OT,NP,FN,IX}(vertexbody, geometrybody)

A mesh stage that runs a vertex body and a geometry body in one invocation. `IT`
is the draw's input topology, `OT` what the geometry body emits, `NP` the output
primitive bound, `FN` the `Flat` output names, and `IX` whether the draw is
indexed, in which case the last argument is the index buffer. Only
[`lower_geometry_to_mesh`](@ref) constructs one.
"""
struct GeometryAsMesh{IT<:Topology, OT<:Topology, NP, FN, IX, VF, GF}
    vertex::VF
    geometry::GF
end

# Infers the two body types; the default constructor takes all seven or none.
GeometryAsMesh{IT,OT,NP,FN,IX}(vertex::VF, geometry::GF) where {IT,OT,NP,FN,IX,VF,GF} =
    GeometryAsMesh{IT,OT,NP,FN,IX,VF,GF}(vertex, geometry)

# Backends name a compiled stage after its function; this names the program in
# shader dumps and compiler errors after the geometry body the author wrote.
Base.nameof(m::GeometryAsMesh) = Symbol(nameof(m.geometry), :_as_mesh)

"""
    splatcall(f, lead::Tuple, args::Tuple)

`f(lead..., args...)` with every element written out, and the call inlined.
Inference resolves a splat of at most 32 elements and a stage can take more
(RayMakie's mesh shader takes 51); past that `args...` is a dynamic
`_apply_iterate` no GPU compiler accepts. Inlined because the geometry body
receives the mutable `MeshEmitter`; see `emitprimitive`.
"""
@generated function splatcall(f, lead::Tuple, args::Tuple)
    call = Expr(:call, :f, (:(lead[$i]) for i in 1:fieldcount(lead))...,
                (:(args[$i]) for i in 1:fieldcount(args))...)
    return quote
        Base.@_inline_meta
        @inline $call
    end
end

# The primitive index is the only builtin the mesh stage reads. `runprimitive`
# takes it as an argument instead, so the lowering also runs on the host over
# `KernelInterface.HostMeshOutput` for comparison against the GPU's output.
function (m::GeometryAsMesh)(out, args::Vararg{Any,N}) where {N}
    return primitivewith(m, out, mesh_group_index(), args)
end

"""
    runprimitive(m::GeometryAsMesh, out, primitive::Integer, args...)

Run one input primitive of a lowered geometry stage into the output object `out`.
`primitive` counts from one: `mesh_group_index()` on a mesh stage, the checked
primitive on the host.
"""
runprimitive(m::GeometryAsMesh, out, primitive::Integer, args::Vararg{Any,N}) where {N} =
    primitivewith(m, out, primitive, args)

# `args` as one tuple from here to the bodies; see `splatcall`.
function primitivewith(m::GeometryAsMesh{IT,OT,NP,FN,false}, out, primitive::Integer,
                       args::Tuple) where {IT,OT,NP,FN}
    # Non-indexed: the vertex index is the position in the stream.
    base = firstinputvertex(IT(), Int32(primitive))
    vs = ntuple(k -> splatcall(m.vertex, (VertexIndex(base + Int32(k) - Int32(1)),), args),
                Val(inputvertices(IT())))
    return emitprimitive(m, out, vs, args)
end

# Indexed: `vertex_index()` on a vertex stage is the value fetched from the index
# buffer, so the lowered stage fetches it from its last argument. The buffer
# counts from zero, `VertexIndex` from one, hence the `+ 1`. `ntuple` rather than
# `Base.front`, which recurses by splatting; see `splatcall`.
function primitivewith(m::GeometryAsMesh{IT,OT,NP,FN,true}, out, primitive::Integer,
                       args::NTuple{N,Any}) where {IT,OT,NP,FN,N}
    indices = args[N]
    rest = ntuple(i -> args[i], Val(N - 1))
    base = firstinputvertex(IT(), Int32(primitive))
    vs = ntuple(Val(inputvertices(IT()))) do k
        i = Int32(indices[base + Int32(k) - Int32(1)]) + Int32(1)
        return splatcall(m.vertex, (VertexIndex(i),), rest)
    end
    return emitprimitive(m, out, vs, rest)
end

"""
    emitprimitive(m::GeometryAsMesh, out, vs::Tuple, args::Tuple)

Run the geometry body over the gathered vertices and declare what it wrote; the
part both `primitivewith` methods share.
"""
@inline function emitprimitive(m::GeometryAsMesh{IT,OT,NP,FN}, out, vs::Tuple,
                               args::Tuple) where {IT,OT,NP,FN}
    e = MeshEmitter{OT,FN}(out, 1, 1, NP)
    # Must inline: `MeshEmitter` is mutable, so passing it to a call that is not
    # inlined makes it escape into a heap allocation and the shader fails to
    # compile with `gc_pool_alloc` in the trace. `splatcall` inlines the body.
    splatcall(m.geometry, (e, transposeprimitive(vs)), args)
    # One invocation per threadgroup, so the emitter's counts are the
    # threadgroup's. Zero primitives (the body culled) eliminates the threadgroup.
    set_mesh_outputs!(out, nvertices(e), nprimitives(e))
    return nothing
end

"""
    transposeprimitive(vs::Tuple) -> NamedTuple

The primitive's vertex outputs, from one NamedTuple per vertex to one tuple per
field, the shape a geometry body reads: `prim.position[2]` is the second vertex's
clip position. `@generated` because it is type arithmetic inside the mesh stage's
loop.
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

Rewrite a pipeline's vertex + geometry stages as a mesh stage. `indexed` is
whether the draw supplies an index buffer; the indexed stage reads it as its last
argument, so the two are different shaders. The mesh stage's `outputs` are the
geometry stage's, so the fragment stage and [`fragmentinputs`](@ref) match for
both pipelines. The vertex body must accept a leading
`KernelInterface.VertexIndex` (see the top of this file).
"""
function lower_geometry_to_mesh(p::GraphicsPipeline; indexed::Bool = false)
    g = p.geometry
    g === nothing && throw(ArgumentError(
        "lower_geometry_to_mesh: this pipeline has no geometry stage, so there is " *
        "nothing to lower. Draw it as it is."))
    (p.tess_control === nothing && p.tess_eval === nothing) && return lower_geometry_to_mesh(p, g, indexed)
    throw(ArgumentError(
        "lower_geometry_to_mesh: this pipeline has a tessellation stage as well as " *
        "a geometry stage, and only the geometry stage can be lowered onto a mesh " *
        "one. A mesh pipeline has no tessellator: what a tessellation control and " *
        "evaluation pair produces would have to be evaluated by the mesh stage " *
        "itself, which is a different translation and not this one."))
end

function lower_geometry_to_mesh(p::GraphicsPipeline, g::GeometryShader, indexed::Bool)
    cfg = stageconfig(g)
    # The assembler's topology decides how vertices are walked, the geometry
    # stage what it expects; a mismatch would silently gather the wrong vertices.
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
    # An upper bound: a restarted strip emits fewer primitives from the same
    # vertices, never more, and a list's count ignores restarts.
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
A backend asks before compiling a lowered mesh stage, so the error says what is
missing instead of a `MethodError` from inside shader compilation.
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

        Keep the index-free spelling beside it if the shader also runs where the
        geometry stage exists natively — the two are the same number:

            $(nameof(f))(buffers...) = $(nameof(f))(VertexIndex(vertex_index()), buffers...)

        Argument types this was asked for: $tt""")
end
