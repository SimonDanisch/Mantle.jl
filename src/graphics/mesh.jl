# The description of a mesh pipeline.
#
# Beside `pipeline.jl` and for the same reason that file gives: this holds Julia
# functions and the fixed-function state from `graphics/state.jl`, and names
# nothing a driver owns. A backend turns it into a compiled pipeline.
#
# The stage vocabulary it describes is `KernelInterface`'s — `MeshConfig`,
# `ObjectConfig`, `MeshEmitter`, `set_mesh_vertex!` and the rest — because a
# COMPILER reads it: `max_vertices` and `max_primitives` are part of the output
# object's type in MSL and of the execution mode in SPIR-V, so they have to be
# reachable from below Mantle. See the header of `KernelInterface/src/mesh.jl`
# for why a mesh pipeline is portable vocabulary rather than a Metal workaround.

"""
    MeshPipeline(; mesh, fragment, object = nothing,
                   blend = Opaque(), cull = CullBack(), depth = DepthLess())

A pipeline whose primitives are written by a mesh stage rather than assembled
from a vertex stream.

`mesh` is a [`MeshShader`](@ref) and `fragment` a [`FragmentShader`](@ref);
`object` is an [`ObjectShader`](@ref) or nothing, in which case the host
dispatches the mesh threadgroups.

There is no `topology` field. In a `GraphicsPipeline` the topology describes the
INPUT vertex stream that the fixed-function assembler groups into primitives; a
mesh stage has no input stream and states what it EMITS, which its
`MeshShader`'s does. A second one here would be a field with nothing to mean.

Ask [`supports_mesh_pipeline`](@ref) before building one. A backend that answers
`false` cannot run it, and finding that out from a shader compile is the wrong
place and the wrong time.
"""
struct MeshPipeline{O, M<:MeshShader, F<:FragmentShader,
                    B<:BlendMode, C<:CullFace, D<:DepthMode}
    object::O     # Nothing or ObjectShader
    mesh::M
    fragment::F
    blend::B
    cull::C
    depth::D
end

function MeshPipeline(;
        mesh::MeshShader, fragment::FragmentShader,
        object::Union{Nothing,ObjectShader} = nothing,
        blend::BlendMode = Opaque(), cull::CullFace = CullBack(),
        depth::DepthMode = DepthLess())
    MeshPipeline(object, mesh, fragment, blend, cull, depth)
end

"""
    meshconfig(p::MeshPipeline) -> MeshConfig

What the mesh stage of `p` may produce.
"""
meshconfig(p::MeshPipeline) = stageconfig(p.mesh)

"""
    objectconfig(p::MeshPipeline) -> ObjectConfig or nothing

The object stage's configuration, or `nothing` when `p` has no object stage and
its mesh threadgroups are dispatched by the host.
"""
objectconfig(p::MeshPipeline) = p.object === nothing ? nothing : stageconfig(p.object)

"""
    fragmentinputs(p::MeshPipeline) -> NamedTuple
    fragmentinputtype(p::MeshPipeline) -> Type

What the fragment stage receives: the mesh stage's outputs. Entries may be
`Flat{T}`, and the type form has them removed.
"""
fragmentinputs(p::MeshPipeline) = stageoutputs(p.mesh)

@doc (@doc fragmentinputs) function fragmentinputtype(p::MeshPipeline)
    v = valuetypes(fragmentinputs(p))
    return NamedTuple{keys(v), Tuple{values(v)...}}
end
