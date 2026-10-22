# The stages a pipeline is made of, one type per stage holding its function,
# configuration and interface. A stage declares only its outputs; its inputs are
# the previous stage's outputs (`stageinputs`). An entry may be `Flat{T}`: the
# value belongs to the primitive rather than the vertex, and only the declaration
# carries it (see `KernelInterface.Flat`).

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

What this stage produces, as a NamedTuple of types; entries may be `Flat{T}`.
`position` is never in it; [`outputtype`](@ref) adds it.
"""
stageoutputs(s::ShaderStage) = s.outputs

"""
    VertexShader(f; outputs = (;))

The per-vertex stage. `f` takes the draw's buffer arguments, which the draw and
not the pipeline declares, and returns `(position = …, outputs…)`.

    VertexShader(project; outputs = (uv = Vec2f, tint = Flat{Vec4f}))
"""
struct VertexShader{F, O<:NamedTuple} <: ShaderStage
    f::F
    outputs::O
end
VertexShader(f; outputs::NamedTuple = (;)) = VertexShader(f, outputs)

"""
    FragmentShader(f; textures = 0)

The per-fragment stage. Its inputs are the previous stage's outputs; its outputs
are the pass's colour attachments, whose count comes from the pass. `textures` is
how many bound textures the body samples through `sample_texture_2d`; see
[`ntextures`](@ref).
"""
struct FragmentShader{F} <: ShaderStage
    f::F
    textures::Int
end
FragmentShader(f; textures::Integer = 0) = FragmentShader(f, Int(textures))
stageoutputs(::FragmentShader) = (;)

"""
    ntextures(s::ShaderStage) -> Int

How many bound textures this stage samples: the highest binding it names plus
one, zero for most stages. Declared rather than derived because both backends
need it before seeing the body: it sizes the descriptor set layout on Vulkan and
the entry-function signature on Metal.
"""
ntextures(::ShaderStage) = 0
ntextures(s::FragmentShader) = s.textures

"""
    GeometryShader(f; outputs, input, output, max_vertices, invocations = 1)

The per-primitive stage: sees a whole input primitive and emits vertices.
`input` fixes how many vertices one invocation sees, `output` what the emitted
vertices are grouped into. `f` takes `(emitter, prim)`; see
`KernelInterface.emit!`. Ask [`supports_geometry_stage`](@ref) before building a
pipeline with one; a backend without the stage can run the body through a
[`MeshShader`](@ref).
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

The stage that writes vertices and primitives into a mesh output object. `f`
takes `(out, …)` and cooperates with its threadgroup on one object;
`max_vertices` and `max_primitives` are compile-time bounds a backend needs to
emit the entry point. Ask [`supports_mesh_pipeline`](@ref) first.
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

The optional stage that decides how many mesh threadgroups to dispatch. It
declares no outputs; what it hands down is a payload, see
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
"""
function outputtype(s::ShaderStage)
    v = valuetypes(stageoutputs(s))
    # `Vec4f`, as both backends declare a clip position; an `NTuple{4,Float32}`
    # would cost a conversion in `getfield`.
    return NamedTuple{(:position, keys(v)...), Tuple{Vec4f, values(v)...}}
end
