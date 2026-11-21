# The mesh stage's output object and the `air.set_*_mesh` intrinsics that write
# through it. The metadata that declares the object to AIR is `compiler/mesh.jl`.
#
# A mesh stage's entry stays `void`, and everything it produces goes through an
# OBJECT it is handed as its first parameter, in address space 7. All five
# intrinsics take that object as their first argument, which is why an intrinsic
# that took only a slot could never be lowered: in AIR the object is a PARAMETER of
# the entry, so there is nothing global to reach.
#
#   air.set_position_mesh          (ptr addrspace(7), i32 slot, <4 x float>)
#   air.set_vertex_data_mesh.<T>   (ptr addrspace(7), i32 field, i32 slot, <T>)
#   air.set_primitive_data_mesh.<T>(ptr addrspace(7), i32 field, i32 slot, <T>)
#   air.set_index_mesh             (ptr addrspace(7), i32 slot, i8)
#   air.set_primitive_count_mesh   (ptr addrspace(7), i32)
#
# Slots are ZERO-based here: this is the AIR boundary, and one-based is the
# convention above it. `metal/mesh.jl`'s overlays subtract.

"""
    MeshObject{V, P, NV, NP, Topo}

The output object a mesh stage writes through, as a type.

`V` is the per-vertex NamedTuple (its first field must be `position`), `P` the
per-primitive one, `NV`/`NP` the bounds, and `Topo` a `Symbol` naming what the
emitted primitives are. Phantom: the value is a pointer to it, and every field here
is something a compiler must know before it can emit the entry at all.
"""
struct MeshObject{V, P, NV, NP, Topo} end

"""
    MeshPtr{V, P, NV, NP, Topo}

The first parameter of a Julia mesh stage: a pointer to its output object in
address space 7, which is the one AIR gives a mesh object.
"""
const MeshPtr{V, P, NV, NP, Topo} = Core.LLVMPtr{MeshObject{V, P, NV, NP, Topo}, 7}

"""An LLVM vector, which is what AIR's `<N x T>` operands are."""
@inline air_vec(v::NTuple{N,T}) where {N,T} = ntuple(i -> VecElement(v[i]), Val(N))
@inline air_vec(v) = air_vec(air_unwrap_vector(v))

"""The tuple inside a one-field vector wrapper such as `Vec4f`."""
@inline air_unwrap_vector(v::NTuple{N,T}) where {N,T} = v
@inline air_unwrap_vector(v) = Base.getfield(v, 1)

# set_position_mesh(out, slot, position)
#
# The clip position of the vertex at `slot`, counting from ZERO. Its own intrinsic
# rather than field zero of the vertex data, which is what the shipping metallib
# shows.
Metal.@device_function @inline function set_position_mesh(out::Core.LLVMPtr{T,7}, slot::Int32,
                                                          position) where {T}
    @typed_ccall("air.set_position_mesh", llvmcall, Nothing,
                 (Core.LLVMPtr{T,7}, Int32, NTuple{4,VecElement{Float32}}),
                 out, slot, air_vec(position))
end

# set_index_mesh(out, slot, vertexindex)
#
# The vertex index at index-slot `slot`, both counting from ZERO. `vertexindex` is
# written as an i8: a mesh output object addresses at most 256 vertices, which is a
# bound the caller has to respect and `MeshConfig` refuses to exceed.
Metal.@device_function @inline function set_index_mesh(out::Core.LLVMPtr{T,7}, slot::Int32,
                                                       vertexindex::UInt8) where {T}
    @typed_ccall("air.set_index_mesh", llvmcall, Nothing,
                 (Core.LLVMPtr{T,7}, Int32, UInt8), out, slot, vertexindex)
end

# set_primitive_count_mesh(out, n)
#
# How many primitives this threadgroup filled. Called once per threadgroup.
Metal.@device_function @inline function set_primitive_count_mesh(out::Core.LLVMPtr{T,7},
                                                                n::Int32) where {T}
    @typed_ccall("air.set_primitive_count_mesh", llvmcall, Nothing,
                 (Core.LLVMPtr{T,7}, Int32), out, n)
end

# ── The two data planes ──────────────────────────────────────────────────────
#
#   air.set_vertex_data_mesh.<T>   (ptr addrspace(7), i32 field, i32 slot, <T>)
#   air.set_primitive_data_mesh.<T>(ptr addrspace(7), i32 field, i32 slot, <T>)
#
# One call per FIELD, and the suffix is the field's value type. The shipping
# metallib's module carried .f16 .i16 .i32 .v2f32 .v2i16 .v3f32 .v3f16 .v4f16, so
# the set is open and driven by what a shader writes rather than fixed.
#
# These take exactly the types AIR can represent: a scalar, or an `NTuple` that
# becomes an LLVM vector. Unwrapping a `Vec4f` is the CALLER's, because the caller
# has the declaration and knows which field it is writing; a fallback here that
# reached for `getfield(v, 1)` would silently write a `Vec4f`'s first component as a
# scalar for any type it did not recognise.
#
# ── The FIELD comes before the SLOT ─────────────────────────────────────────
#
# They were declared the other way round once, matching `set_position_mesh`, and
# every test agreed with the mistake because of how the two orders overlap: a stage
# that writes field 0 of vertices 0..n calls `(0,0) (1,0) (2,0) …`, which read as
# `(field, slot)` is field 0 of vertex 0, then field 1 of vertex 0, then field 2,
# so the FIRST vertex gets the right value and the rest are written to fields that
# do not exist and vanish. Measured both ways, with the same shader and one operand
# pair swapped:
#
#   (slot, field)   uv correct at vertex 0, zero at vertices 1..3
#   (field, slot)   uv correct at all four, and correct with a DYNAMIC slot too
#
# and again on the primitive plane, where two triangles given different colours
# came out as "first one right, second one black" under the old order. So these
# mirror AIR's order rather than hiding it behind a nicer one.

for (fn, airname) in ((:set_vertex_data_mesh, "air.set_vertex_data_mesh"),
                      (:set_primitive_data_mesh, "air.set_primitive_data_mesh"))
    # Scalars.
    for (T, sfx) in ((Float32, "f32"), (Float16, "f16"), (Int32, "i32"),
                     (UInt32, "i32"), (Int16, "i16"), (UInt16, "i16"))
        intr = airname * "." * sfx
        @eval Metal.@device_function @inline function $fn(out::Core.LLVMPtr{O,7}, field::Int32,
                                                          slot::Int32, v::$T) where {O}
            @typed_ccall($intr, llvmcall, Nothing,
                         (Core.LLVMPtr{O,7}, Int32, Int32, $T), out, field, slot, v)
        end
    end
    # Vectors, as LLVM `<N x T>`.
    for N in (2, 3, 4), (T, sc) in ((Float32, "f32"), (Float16, "f16"),
                                    (Int32, "i32"), (Int16, "i16"))
        intr = airname * ".v$(N)$(sc)"
        @eval Metal.@device_function @inline function $fn(out::Core.LLVMPtr{O,7}, field::Int32,
                                                          slot::Int32, v::NTuple{$N,$T}) where {O}
            @typed_ccall($intr, llvmcall, Nothing,
                         (Core.LLVMPtr{O,7}, Int32, Int32, NTuple{$N,VecElement{$T}}),
                         out, field, slot, air_vec(v))
        end
    end
end
