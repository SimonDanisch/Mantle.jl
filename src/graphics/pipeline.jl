# The description of a graphics pipeline and the draw command it can be fed:
# Julia functions and fixed-function state, nothing a driver owns. A backend
# compiles it into a `CompiledGraphicsPipeline`.
"""
    GraphicsPipeline(; vertex, fragment, geometry = nothing,
                       tess_control = nothing, tess_eval = nothing,
                       blend = Opaque(), cull = CullBack(),
                       topology = TriangleList(), depth = DepthLess())

The stages a rasterised draw runs and the fixed-function state around them.
`vertex` and `fragment` are a [`VertexShader`](@ref) and a
[`FragmentShader`](@ref), `geometry` a [`GeometryShader`](@ref) or `nothing`.
Each stage declares its own `outputs`; [`fragmentinputs`](@ref) is what the
fragment stage reads. `topology` describes the input vertex stream; a geometry
stage states its own input and output topologies.
"""
struct GraphicsPipeline{V<:VertexShader, F<:FragmentShader, G, TC, TE,
                        B<:BlendMode, C<:CullFace, T<:Topology, D<:DepthMode}
    vertex::V
    fragment::F
    geometry::G       # Nothing or GeometryShader
    tess_control::TC  # Nothing or (func, TessConfig)
    tess_eval::TE     # Nothing or func
    blend::B
    cull::C
    topology::T
    depth::D
end

const Rasterizer = GraphicsPipeline

function GraphicsPipeline(;
        vertex::VertexShader, fragment::FragmentShader,
        geometry::Union{Nothing,GeometryShader} = nothing,
        tess_control = nothing, tess_eval = nothing,
        blend::BlendMode = Opaque(), cull::CullFace = CullBack(),
        topology::Topology = TriangleList(), depth::DepthMode = DepthLess())
    GraphicsPipeline(vertex, fragment, geometry, tess_control, tess_eval,
                     blend, cull, topology, depth)
end

TrianglePipeline(; kw...) = GraphicsPipeline(; kw...)
LinePipeline(; kw...) = GraphicsPipeline(; topology = LineList(), kw...)

"""Fixed native pipeline state; shader identities/configuration are keyed separately."""
pipeline_state_key(p::GraphicsPipeline) = (p.blend, p.cull, p.topology, p.depth)

"""
    lastgeometrystage(p::GraphicsPipeline) -> ShaderStage

The stage whose outputs reach the rasteriser: the geometry stage when there is
one, the vertex stage otherwise.
"""
lastgeometrystage(p::GraphicsPipeline) =
    p.geometry === nothing ? p.vertex : p.geometry

"""
    fragmentinputs(p::GraphicsPipeline) -> NamedTuple

What the fragment stage receives, as a NamedTuple of types; entries may be
`Flat{T}`. `position` is not among them: the fragment stage reads it through
`frag_coord`.
"""
fragmentinputs(p::GraphicsPipeline) = stageoutputs(lastgeometrystage(p))

"""
    fragmentinputtype(p::GraphicsPipeline) -> Type
    fragmentinputtype(p::MeshPipeline) -> Type

[`fragmentinputs`](@ref) as the NamedTuple type a backend types its fragment
stage against, with every `Flat` removed: flatness decides how a value is
delivered, not its type.
"""
function fragmentinputtype(p)
    v = valuetypes(fragmentinputs(p))
    return NamedTuple{keys(v), Tuple{values(v)...}}
end

"""
    DrawIndirectCommand(vertices, instances, firstvertex, firstinstance)

A draw the GPU wrote, laid out as `VkDrawIndirectCommand` (the same four
`UInt32` fields in the same order). A nonzero `firstinstance` needs the
`drawIndirectFirstInstance` feature.
"""
struct DrawIndirectCommand
    vertices::UInt32
    instances::UInt32
    firstvertex::UInt32
    firstinstance::UInt32
end
