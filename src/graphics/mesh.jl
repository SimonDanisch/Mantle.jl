# The description of a mesh pipeline: Julia functions and fixed-function state,
# nothing a driver owns. The stage vocabulary (`MeshConfig`, `MeshEmitter`, ...)
# is `KernelInterface`'s because the compiler reads `max_vertices` and
# `max_primitives`: they are part of the output type in MSL and of the execution
# mode in SPIR-V.

"""
    MeshPipeline(; mesh, fragment, object = nothing,
                   blend = Opaque(), cull = CullBack(), depth = DepthLess())

A pipeline whose primitives are written by a mesh stage rather than assembled
from a vertex stream. `mesh` is a [`MeshShader`](@ref), `fragment` a
[`FragmentShader`](@ref), `object` an [`ObjectShader`](@ref) or `nothing`, in
which case the host dispatches the mesh threadgroups. There is no `topology`:
the `MeshShader` states what it emits. Ask [`supports_mesh_pipeline`](@ref)
before building one.
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

pipeline_state_key(p::MeshPipeline) = (p.blend, p.cull, p.depth)

# No backend emits an object stage: a mesh pipeline's threadgroups are dispatched by
# the host, with the group count `draw!` takes.
function checkstages(dev, p::MeshPipeline)
    p.object === nothing || throw(ArgumentError(
        "a mesh pipeline's object stage is not emitted by any backend yet; build it " *
        "without one and pass the mesh group count to `draw!`"))
    return nothing
end

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
