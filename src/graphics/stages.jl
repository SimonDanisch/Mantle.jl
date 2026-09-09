# The stages a pipeline is made of.
#
# A pipeline used to be a flat list of keywords — `vertex`, `fragment`,
# `geometry` as a `(func, GeometryConfig)` tuple, `tess_control`, `tess_eval`,
# `varyings`, plus the fixed-function state — so one stage's function, its
# configuration and its interface were three separate arguments that only
# convention held together. A stage is one thing; this makes it one type.
#
# ── Only outputs are declared ────────────────────────────────────────────────
#
# A stage declares what it PRODUCES and nothing about what it consumes, because
# its inputs are the previous stage's outputs. Declaring both would put every
# interface in the description twice and turn a wiring mistake into something
# that has to be validated; declared once, there is nothing to disagree with.
# [`stageinputs`](@ref) reads a consumer's inputs off its producer.
#
# ── The word `varyings` is gone ──────────────────────────────────────────────
#
# It named one interface, which was only ever enough because no pipeline in the
# tree combined it with a geometry stage — RayMakie's two still use numbered
# locations, precisely because the declarative form could not express the second
# interface. A pipeline with a geometry stage has two: vertex → geometry and
# geometry → fragment. One list per stage covers any number of them, and
# `outputs` says what it is without knowing what OpenGL called it.
#
# ── Flat ─────────────────────────────────────────────────────────────────────
#
# An entry may be `Flat{T}`, which says the value belongs to the primitive
# rather than the vertex. See `KernelInterface.Flat`: nothing wraps the value,
# only the declaration carries it.

"""
    ShaderStage

One stage of a pipeline: its function, its configuration and what it produces.
"""
abstract type ShaderStage end

"""
    stagefunction(s::ShaderStage)

The Julia function this stage runs.
"""
stagefunction(s::ShaderStage) = s.f

"""
    stageoutputs(s::ShaderStage) -> NamedTuple

What this stage produces, as a NamedTuple of types. Entries may be `Flat{T}`.

`position` is never in it: every stage that has one has one, and a list that had
to include it would eventually be written without it.
"""
stageoutputs(s::ShaderStage) = s.outputs

"""
    VertexShader(f; outputs = (;))

The per-vertex stage.

`f` takes the pass's buffer arguments and returns `(position = …, outputs…)`.
Which buffers those are is the DRAW's business, not the pipeline's, so they are
not declared here.

    VertexShader(project; outputs = (uv = Vec2f, tint = Flat{Vec4f}))
"""
struct VertexShader{F, O<:NamedTuple} <: ShaderStage
    f::F
    outputs::O
end
VertexShader(f; outputs::NamedTuple = (;)) = VertexShader(f, outputs)

"""
    FragmentShader(f)

The per-fragment stage.

Nothing to declare: its inputs are the previous stage's outputs, and its outputs
are the pass's colour attachments — a shader compiled for one attachment when
the pass has three would silently drop two, so the count comes from the pass.
"""
struct FragmentShader{F} <: ShaderStage
    f::F
end
stageoutputs(::FragmentShader) = (;)

"""
    GeometryShader(f; outputs, input, output, max_vertices, invocations = 1)

The per-primitive stage, which sees a whole input primitive and emits vertices.

`input` and `output` are topologies: `input` fixes how many vertices one
invocation sees, `output` is what the emitted vertices are grouped into. `f`
takes `(emitter, prim)` — see `KernelInterface.emit!`.

    GeometryShader(expand; outputs = (uv = Vec2f, width = Flat{Float32}),
                           input = LineStripAdjacency(), output = TriangleStrip(),
                           max_vertices = 4)

Ask [`supports_geometry_stage`](@ref) before building a pipeline with one. A
backend without the stage may still be able to run the body through a
[`MeshShader`](@ref), which is the more general of the two.
"""
struct GeometryShader{F, O<:NamedTuple, C<:GeometryConfig} <: ShaderStage
    f::F
    outputs::O
    config::C
end

function GeometryShader(f; outputs::NamedTuple = (;),
                          input::Topology = TriangleList(),
                          output::Topology = TriangleStrip(),
                          max_vertices::Integer, invocations::Integer = 1)
    GeometryShader(f, outputs,
                   GeometryConfig(; input, output, max_vertices, invocations))
end

"""
    MeshShader(f; outputs, max_vertices, max_primitives,
                 topology = TriangleList(), threads = 1)

The stage that writes vertices and primitives into a mesh output object.

`f` takes `(out, …)` and cooperates with the rest of its threadgroup on one
object; `max_vertices` and `max_primitives` bound it and are compile-time
constants a backend needs to emit the entry point at all.

Ask [`supports_mesh_pipeline`](@ref) first.
"""
struct MeshShader{F, O<:NamedTuple, C<:MeshConfig} <: ShaderStage
    f::F
    outputs::O
    config::C
end

function MeshShader(f; outputs::NamedTuple = (;), max_vertices::Integer,
                      max_primitives::Integer, topology::Topology = TriangleList(),
                      threads::Integer = 1)
    MeshShader(f, outputs,
               MeshConfig(; max_vertices, max_primitives, topology, threads))
end

"""
    ObjectShader(f; threads = 1)

The stage that decides how many mesh threadgroups to dispatch.

Optional on both APIs, and it produces no interface of its own — what it hands
down is a payload, which has no ground truth yet; see
`KernelInterface.ObjectConfig`.
"""
struct ObjectShader{F, C<:ObjectConfig} <: ShaderStage
    f::F
    config::C
end
ObjectShader(f; threads::Integer = 1) = ObjectShader(f, ObjectConfig(; threads))
stageoutputs(::ObjectShader) = (;)

"""
    stageconfig(s::ShaderStage)

The compile-time configuration a backend reads: a `GeometryConfig`, `MeshConfig`
or `ObjectConfig`. `nothing` for the stages that have none.
"""
stageconfig(::ShaderStage) = nothing
stageconfig(s::GeometryShader) = s.config
stageconfig(s::MeshShader) = s.config
stageconfig(s::ObjectShader) = s.config

"""
    stageinputs(producer) -> NamedTuple

What the stage after `producer` receives, which is what `producer` produces.

A separate name from [`stageoutputs`](@ref) even though it returns the same
thing, because the two read differently at the call site and the identity is the
point being made.
"""
stageinputs(producer::ShaderStage) = stageoutputs(producer)

"""
    flatoutputs(s::ShaderStage) -> Tuple{Vararg{Symbol}}
    smoothoutputs(s::ShaderStage) -> Tuple{Vararg{Symbol}}

The names on each side of the `Flat` split, in declaration order. A geometry
body emits the smooth ones per vertex and the flat ones per primitive.
"""
flatoutputs(s::ShaderStage) = flatnames(stageoutputs(s))
@doc (@doc flatoutputs) smoothoutputs(s::ShaderStage) = smoothnames(stageoutputs(s))

"""
    outputtype(s::ShaderStage) -> Type

The NamedTuple type a backend builds this stage's output struct from: the
declared list with every `Flat` removed, and `position` in front.

Flatness decides WHERE a value is delivered, never what it is, so it does not
appear here.
"""
function outputtype(s::ShaderStage)
    v = valuetypes(stageoutputs(s))
    # `Vec4f`, because a clip position IS one. The Metal backend used to declare
    # the field as `NTuple{4,Float32}` and absorb the difference in a `getfield`,
    # which cost a conversion and a paragraph explaining why one thing had two
    # spellings. Nothing needed the tuple: `mangle_varying` gives both the same
    # string, which is all the two stages link by.
    return NamedTuple{(:position, keys(v)...), Tuple{Vec4f, values(v)...}}
end
