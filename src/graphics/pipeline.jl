# The description of a graphics pipeline, and the draw command it can be fed.
#
# Moved out of `src/vulkan/graphics/api.jl`, where the rest of the file is
# compilation and drawing and genuinely the backend's. This half is not: it
# holds Julia functions and the fixed-function state from `graphics/state.jl`,
# and names nothing a driver owns. `CompiledGraphicsPipeline` is what a backend
# turns it into.
"""
    GraphicsPipeline(; vertex, fragment, geometry = nothing,
                       blend = Opaque(), cull = CullBack(),
                       topology = TriangleList(), depth = DepthLess())

The stages a rasterised draw runs, and the fixed-function state around them.

`vertex` and `fragment` are a [`VertexShader`](@ref) and a
[`FragmentShader`](@ref); `geometry` is a [`GeometryShader`](@ref) or nothing.
Each stage carries its own function, its own configuration and its own
`outputs` — see `graphics/stages.jl` for why only outputs are declared.

`topology` describes the INPUT vertex stream the assembler groups into
primitives. A geometry stage states its own input and output topologies, which
are a different thing: what one invocation sees, and what it emits.

There is no `varyings` field. It named exactly one interface, which was only
ever enough because no pipeline combined it with a geometry stage — that one has
two, vertex → geometry and geometry → fragment, and one list per stage covers
any number of them. [`fragmentinputs`](@ref) is what the fragment stage reads.
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

"""
    lastgeometrystage(p::GraphicsPipeline) -> ShaderStage

The stage whose outputs reach the rasteriser: the geometry stage when there is
one, the vertex stage otherwise.

Named for the question it answers rather than for a field, because "what does
the fragment stage read" is the only reason anyone asks and the answer depends
on which stages a pipeline has.
"""
lastgeometrystage(p::GraphicsPipeline) =
    p.geometry === nothing ? p.vertex : p.geometry

"""
    fragmentinputs(p::GraphicsPipeline) -> NamedTuple

What the fragment stage receives, as a NamedTuple of types. Entries may be
`Flat{T}`.

`position` is not among them: a fragment stage reads its position through
`frag_coord`, which the rasteriser supplies, not as a value a previous stage
passed along.
"""
fragmentinputs(p::GraphicsPipeline) = stageoutputs(lastgeometrystage(p))

"""
    fragmentinputtype(p::GraphicsPipeline) -> Type

[`fragmentinputs`](@ref) as the NamedTuple TYPE a backend types its fragment
stage against, with every `Flat` removed — flatness decides how a value is
delivered, never what it is.
"""
function fragmentinputtype(p::GraphicsPipeline)
    v = valuetypes(fragmentinputs(p))
    return NamedTuple{keys(v), Tuple{values(v)...}}
end

"""
    DrawIndirectCommand(vertices, instances, firstvertex, firstinstance)

A draw the GPU wrote, laid out as `VkDrawIndirectCommand`: same four `UInt32`
fields in the same order, so a buffer of these is one.

    @kernel function finish!(cmds, counter)
        @inbounds cmds[1] = DrawIndirectCommand(counter[1] * UInt32(36),
                                                UInt32(1), UInt32(0), UInt32(0))
    end

`firstinstance` other than zero needs the `drawIndirectFirstInstance` feature; it
is here because the struct is the Vulkan one, not because everything can use it.
"""
struct DrawIndirectCommand
    vertices::UInt32
    instances::UInt32
    firstvertex::UInt32
    firstinstance::UInt32
end

