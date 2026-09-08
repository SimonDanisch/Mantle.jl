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
    MeshPipeline(; mesh, fragment, object = nothing, varyings = nothing,
                   blend = Opaque(), cull = CullBack(), depth = DepthLess())

A pipeline whose primitives are written by a mesh stage rather than assembled
from a vertex stream.

# Fields
- `mesh`: a `(func, MeshConfig)` tuple. Required: the stage that writes vertices
  and primitives into the output object.
- `object`: `nothing`, or a `(func, ObjectConfig)` tuple. The stage that decides
  how many mesh threadgroups to dispatch.
- `fragment`: the fragment stage, exactly as [`GraphicsPipeline`](@ref) takes it.
- `varyings`: a NamedTuple of types, as `GraphicsPipeline` takes it. These are
  the fields of the vertex the mesh stage writes, beside `position`.

There is deliberately no `topology` field. In a `GraphicsPipeline` the topology
describes the INPUT vertex stream that the fixed-function assembler groups into
primitives; a mesh stage has no input stream and states what it EMITS, which is
`MeshConfig`'s `topology`. Carrying a second one here would be a field with
nothing to mean.

Ask [`supports_mesh_pipeline`](@ref) before building one. A backend that answers
`false` cannot run it, and finding that out from a shader compile is the wrong
place and the wrong time.
"""
struct MeshPipeline{O, M, F, B<:BlendMode, C<:CullFace, D<:DepthMode, VY}
    object::O     # Nothing or (func, ObjectConfig)
    mesh::M       # (func, MeshConfig)
    fragment::F
    blend::B
    cull::C
    depth::D
    varyings::VY  # Nothing or NamedTuple of types
end

function MeshPipeline(;
        mesh, fragment, object = nothing,
        blend::BlendMode = Opaque(), cull::CullFace = CullBack(),
        depth::DepthMode = DepthLess(), varyings = nothing)
    mesh isa Tuple{Any,MeshConfig} || throw(ArgumentError(
        "MeshPipeline: `mesh` must be a (function, MeshConfig) tuple, got $(typeof(mesh))"))
    object === nothing || object isa Tuple{Any,ObjectConfig} || throw(ArgumentError(
        "MeshPipeline: `object` must be nothing or a (function, ObjectConfig) " *
        "tuple, got $(typeof(object))"))
    MeshPipeline(object, mesh, fragment, blend, cull, depth, varyings)
end

"""
    meshconfig(p::MeshPipeline) -> MeshConfig

What the mesh stage of `p` may produce.
"""
meshconfig(p::MeshPipeline) = p.mesh[2]

"""
    objectconfig(p::MeshPipeline) -> ObjectConfig or nothing

The object stage's configuration, or `nothing` when `p` has no object stage and
its mesh threadgroups are dispatched by the host.
"""
objectconfig(p::MeshPipeline) = p.object === nothing ? nothing : p.object[2]
