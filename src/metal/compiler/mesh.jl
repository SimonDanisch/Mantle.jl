# Julia → AIR mesh programs.
#
# ── The mesh stage, read out of a shipping metallib ──────────────────────────
#
# Obtained the same way as the vertex and fragment specification in `stages.jl`: a
# scan of the 296 metallibs under /System/Library found exactly one mesh function,
# `particle_gaussian_mesh` in `VFX.framework/Versions/A/Resources/default.metallib`.
# Reproduce with `read(io, Metal.MetalLib)` and `parse(LLVM.Module, fn.air_module)`.
#
# PROGRAM TYPE is 7, which Metal.jl's metallib writer derives from the `air.mesh`
# named metadata.
#
# NAMED METADATA is `air.mesh`, and its node has the same three operands as
# `air.kernel` ({ptr @entry, outputs, inputs}) with the OUTPUTS EMPTY. That is the
# structural difference from the vertex and fragment stages: they return a packed
# struct whose fields match the output list, and a mesh stage returns nothing at
# all. Everything it produces goes through the object it is handed.
#
# ENTRY SIGNATURE is `void`, and its FIRST parameter is the output object:
#
#   void (ptr addrspace(7), ptr addrspace(2), …, i32, i32, i32, i16)
#
# Address space 7 is the mesh object's. It is a parameter and not a global, which is
# why an intrinsic that took only a slot could never be lowered here.
#
# THE OBJECT'S ARGUMENT ENTRY carries the bounds as compile-time constants:
#
#   {i32 0, air.mesh,
#    !{"air.mesh_type_info", <vertex type>, <primitive type>,
#      i32 96, i32 32, "air.triangle"},
#    air.arg_type_name, "mesh<particle_vertex_io, particle_primitive_io, 96, 32, triangle>",
#    air.arg_name, "output"}
#
# The two integers are max_vertices and max_primitives, and the string is the
# topology. They are part of the TYPE, which is why `Lava.MeshConfig` holds them
# rather than a runtime: a backend cannot emit this entry without them.
#
# `<vertex info>` is one node per field of the per-vertex struct, the first of which
# is the clip position:
#
#   {air.position, air.arg_type_name, "float4", air.arg_name, "position"}
#   {air.mesh_vertex_data, i32 <n>, "generated(<mangled>)",
#    air.arg_type_name, "float2", air.arg_name, "uv0"}
#
# `<primitive info>` is one node per field of the per-primitive struct, all of the
# second shape with `air.mesh_primitive_data`.
#
# The `generated(...)` string is what `mangle_varying` already produces for the
# vertex and fragment stages, verified against the shipping metallib, which spells
# `crworld_position::float3` as `generated(16crworld_positionDv3_f)`. So a fragment
# stage reading a mesh stage's outputs links by exactly the string it already used.
#
# The intrinsics that write through the object are in `intrinsics/mesh.jl`.
#
# NO OBJECT STAGE was found. No function in any of the 296 libraries had a program
# type of 8 or above, and this mesh shader has none: it reads
# `air.thread_position_in_grid` and is dispatched by the host. So the object stage's
# payload has no ground truth here, which is why `ObjectConfig` declares none.
#
# ── A mesh stage is a compute stage that rasterises ──────────────────────────
#
# It has everything a kernel has (`thread_index_in_threadgroup`,
# `threadgroup_position_in_grid`, threadgroup memory, `threadgroup_barrier`), and
# its own buffers are numbered from zero exactly like a vertex stage's. The object
# is a parameter, not a binding, and consumes no slot.
#
# That last sentence cost a day, because two things in `retag_stage!` were wrong and
# both failed silently:
#
#   * GPUCompiler's `air.buffer` description of the object was left in place at
#     location 0, so the object and the stage's first buffer claimed one slot.
#     Binding that buffer replaced the object the driver handed the entry, every
#     `set_*_mesh` call wrote through a pointer to something else, and no primitive
#     was assembled. No validation error, no warning, an empty frame.
#   * The `air.mesh` node was written at an offset computed from the END of the
#     argument list, which overwrote whichever builtin landed there. With the
#     compute builtins threaded in that was `air.thread_index_in_threadgroup`, so
#     every thread read index 0, did the first thread's work, and the stage behaved
#     like a single-threaded one.
#
# Both are fixed by putting the object at position 1, where it is, and moving the
# buffers behind it down one slot. Measured after, against a reference compiled
# from MSL at runtime by the system compiler:
#
#   MSL mesh   + MSL fragment                     4096/4096 covered
#   Julia mesh + MSL fragment                     4096/4096
#   Julia mesh + Julia fragment                   4096/4096, whole buffer equal
#   Julia mesh, own buffer at Metal slot 0        4096/4096, value read back
#   3 groups x 4 threads                          12 distinct (group, thread) pairs
#   threadgroup memory + barrier                  4096/4096
#
# A mesh threadgroup that emits NOTHING is dropped whole (side effects and all) once
# the fragment stage reads its planes. A probe store from such a stage does not
# land, and `set_primitive_count_mesh(out, 0)` drops it just as surely. That is the
# right behaviour for a geometry body that culls everything, but it means a mesh
# stage cannot be tested by writing to a buffer and emitting no primitive: the test
# has to draw.
#
# Setting `maxTotalThreadsPerObjectThreadgroup` on a pipeline with no object stage
# is refused outright ("mismatch between shader and descriptor"), so the driver does
# compare descriptor values against shader-declared ones. And the shipping mesh
# function carries 110069 bytes of `reflection_data` where ours carries none; it
# runs without it, so reflection is for tooling.

"""What AIR calls a mesh stage's output topology."""
air_mesh_topology(t::Symbol) =
    t === :triangle ? "air.triangle" :
    t === :line     ? "air.line" :
    t === :point    ? "air.point" :
    error("no AIR mesh topology for :$t; expected :triangle, :line or :point")

"""
    air_mesh_vertex_info(V) -> Vector{Metadata}

One node per field of the per-vertex struct. `position` is `air.position` and
carries no index; every other field is numbered from zero, in declaration order,
which is the contract the fragment stage's inputs are matched against.
"""
function air_mesh_vertex_info(@nospecialize(V::Type))
    names = fieldnames(V)
    isempty(names) && error("a mesh stage's vertex struct needs at least a position")
    first(names) === :position ||
        error("the first field of a mesh stage's vertex struct must be `position`, got `$(first(names))`")
    out = Metadata[]
    for (i, name) in enumerate(names)
        T = fieldtype(V, name)
        if i == 1
            push!(out, MDNode(Metadata[MDString("air.position"),
                                       MDString("air.arg_type_name"),
                                       MDString(air_stage_name(T)),
                                       MDString("air.arg_name"),
                                       MDString("position")]))
        else
            push!(out, MDNode(Metadata[MDString("air.mesh_vertex_data"),
                                       Metadata(ConstantInt(Int32(i - 2))),
                                       MDString(mangle_varying(string(name), T)),
                                       MDString("air.arg_type_name"),
                                       MDString(air_stage_name(T)),
                                       MDString("air.arg_name"),
                                       MDString(string(name))]))
        end
    end
    return out
end

"""
    air_mesh_primitive_info(P) -> Vector{Metadata}

One node per field of the per-primitive struct: the `Flat`-declared outputs. An
empty struct is a real answer: a stage whose every output varies across the
primitive has no per-primitive plane.
"""
function air_mesh_primitive_info(@nospecialize(P::Type))
    out = Metadata[]
    for (i, name) in enumerate(fieldnames(P))
        T = fieldtype(P, name)
        push!(out, MDNode(Metadata[MDString("air.mesh_primitive_data"),
                                   Metadata(ConstantInt(Int32(i - 1))),
                                   MDString(mangle_varying(string(name), T)),
                                   MDString("air.arg_type_name"),
                                   MDString(air_stage_name(T)),
                                   MDString("air.arg_name"),
                                   MDString(string(name))]))
    end
    return out
end

"""
    air_mesh_type_name(NV, NP, Topo) -> String

How AIR spells the object's type, which the shipping metallib gives as
`mesh<particle_vertex_io, particle_primitive_io, 96, 32, triangle>`.

The two struct names are ours to choose (nothing matches on them; the
`generated(...)` strings inside carry the linkage), so they say what they are.
"""
air_mesh_type_name(NV::Integer, NP::Integer, Topo::Symbol) =
    "mesh<mesh_vertex_io, mesh_primitive_io, $NV, $NP, $Topo>"

"""
    air_mesh_argument(T) -> Metadata

The argument entry for a mesh stage's output object, which replaces the
`air.buffer` description GPUCompiler emitted for parameter one.
"""
function air_mesh_argument(@nospecialize(T::Type))
    T <: MeshObject ||
        error("the first argument of a mesh stage must point at a MeshObject, got $T")
    V, P, NV, NP, Topo = T.parameters
    info = MDNode(Metadata[MDString("air.mesh_type_info"),
                           MDNode(air_mesh_vertex_info(V)),
                           MDNode(air_mesh_primitive_info(P)),
                           Metadata(ConstantInt(Int32(NV))),
                           Metadata(ConstantInt(Int32(NP))),
                           MDString(air_mesh_topology(Topo))])
    return MDNode(Metadata[Metadata(ConstantInt(Int32(0))),
                           MDString("air.mesh"),
                           info,
                           MDString("air.arg_type_name"),
                           MDString(air_mesh_type_name(NV, NP, Topo)),
                           MDString("air.arg_name"),
                           MDString("output")])
end

"""
    mesh_object_type(job) -> Type

The `MeshObject` this mesh stage writes through, read off the first argument.

Derived rather than carried in `MetalStageParams` for the reason `stage_output_type`
is: the argument the body writes through IS the declaration of what the stage
produces, and a second source of truth is a second thing to get out of sync.
"""
function mesh_object_type(@nospecialize(job::CompilerJob))
    args = job.source.specTypes.parameters[2:end]
    isempty(args) &&
        error("a mesh stage needs its output object as its first argument; the signature has none")
    P = first(args)
    P <: Core.LLVMPtr ||
        error("the first argument of a mesh stage must be a `MeshPtr`, got $P")
    return P.parameters[1]
end
