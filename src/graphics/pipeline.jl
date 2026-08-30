# The description of a graphics pipeline, and the draw command it can be fed.
#
# Moved out of `src/vulkan/graphics/api.jl`, where the rest of the file is
# compilation and drawing and genuinely the backend's. This half is not: it
# holds Julia functions and the fixed-function state from `graphics/state.jl`,
# and names nothing a driver owns. `CompiledGraphicsPipeline` is what a backend
# turns it into.
"""
    GraphicsPipeline

High-level graphics pipeline wrapping Julia shader functions.
Compiles lazily on first use and caches the result.

# Fields
- `vertex`, `fragment`: Required Julia shader functions
- `geometry`: Optional (func, GeometryConfig) tuple
- `tess_control`, `tess_eval`: Optional tessellation stages
- `blend`, `cull`, `topology`, `depth`: Pipeline state types
"""
struct GraphicsPipeline{V, F, G, TC, TE, B<:BlendMode, C<:CullFace, T<:Topology, D<:DepthMode, VY}
    vertex::V
    fragment::F
    geometry::G       # Nothing or (func, GeometryConfig)
    tess_control::TC  # Nothing or (func, TessConfig)
    tess_eval::TE     # Nothing or func
    blend::B
    cull::C
    topology::T
    depth::D
    varyings::VY      # Nothing or NamedTuple of types, e.g. (normal=Vec3f, uv=Vec2f)
end

const Rasterizer = GraphicsPipeline

function GraphicsPipeline(;
        vertex, fragment,
        geometry=nothing, tess_control=nothing, tess_eval=nothing,
        blend::BlendMode=Opaque(), cull::CullFace=CullBack(),
        topology::Topology=TriangleList(), depth::DepthMode=DepthLess(),
        varyings=nothing)
    GraphicsPipeline(
        vertex, fragment, geometry, tess_control, tess_eval,
        blend, cull, topology, depth,
        varyings,
    )
end

TrianglePipeline(; vertex, fragment, kw...) = GraphicsPipeline(; vertex, fragment, kw...)
LinePipeline(; vertex, fragment, kw...) = GraphicsPipeline(; vertex, fragment, topology=LineList(), kw...)

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

