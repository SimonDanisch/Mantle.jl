# Render stage builtins: the vertex and instance index, the interpolated fragment
# position, the point size a vertex stage writes, screen-space derivatives, fragment
# discard and sampling a bound 2D texture; and the marker types a stage signature
# declares its inputs with. `compiler/` turns each of them into what AIR expects.
#
# Same names and the same shape as Lava's (`device/gfx_intrinsics.jl`), so a shader
# written once compiles on either backend. Lava reads an external global in
# `addrspace(7)` that its SPIR-V emitter turns into a BuiltIn Input variable; these
# read an external global too, and `compiler/stages.jl`'s `stage_builtins!` turns
# each one that is actually used into an entry parameter carrying the matching
# `air.*` tag.
#
# The indirection is what lets a builtin be a plain zero-argument call. AIR has no
# such thing as an ambient vertex id: it is a function parameter and nothing else,
# so something has to put it there, and a global the caller cannot see is how the
# shader asks for it without spelling it in its signature.
#
# `vertex_index()` and `instance_index()` return ONE-BASED indices, matching Lava,
# because Julia code indexes from one and a shader that has to remember which
# convention a builtin follows will eventually get it wrong.
#
# Device-only functions are `Metal.@device_function`s: a host method that throws,
# and the implementation as an overlay in Metal.jl's method table, which every
# stage job stacks under Mantle's own. Comments, not docstrings, on them: the
# macro expands to a host stub plus an overlay method.

using LLVM.Interop: @typed_ccall

# ── Implicit stage inputs and outputs ───────────────────────────────────────

# The external globals the builtins below load from (and `set_point_size!` stores
# to). `compiler/stages.jl` finds them by these names, so both sides use the
# constants.
const VERTEX_ID_GLOBAL   = "__air_stage_vertex_id"
const INSTANCE_ID_GLOBAL = "__air_stage_instance_id"
const FRAG_COORD_GLOBAL  = "__air_stage_frag_coord"
const POINT_SIZE_GLOBAL  = "__air_stage_point_size"

"""The `llvmcall` IR that loads the `i32` external global `name`."""
stage_index_ir(name::String) = """
    @$name = external global i32
    define i32 @entry() #0 {
        %val = load i32, ptr @$name, align 4
        ret i32 %val
    }
    attributes #0 = { alwaysinline }
    """

# vertex_index() / instance_index()
#
# `raw % Int32`, not `Int32(raw)`: the checked conversion can throw, and a stage has
# no kernel state to report a throw through. The value is identical for every
# vertex id a GPU can produce; only the unreachable error path differs.
for (fname, name) in ((:vertex_index, VERTEX_ID_GLOBAL), (:instance_index, INSTANCE_ID_GLOBAL))
    @eval Metal.@device_function @inline function $fname()
        raw = Base.llvmcall(($(stage_index_ir(name)), "entry"), UInt32, Tuple{})
        return (raw % Int32) + Int32(1)
    end
end

# frag_coord(dim) -> Float32
#
# One component of the interpolated fragment position, `[[position]]` in MSL.
# `dim` is one-based: 1 is x, 4 is w.
@eval Metal.@device_function @inline function frag_coord(dim::Integer)
    Base.llvmcall(($("""
        @$FRAG_COORD_GLOBAL = external global <4 x float>
        define float @entry(i32 %dim) #0 {
            %vec = load <4 x float>, ptr @$FRAG_COORD_GLOBAL, align 16
            %val = extractelement <4 x float> %vec, i32 %dim
            ret float %val
        }
        attributes #0 = { alwaysinline }
        """), "entry"), Float32, Tuple{UInt32}, (dim - 1) % UInt32)
end

@inline frag_coord_x() = frag_coord(1)
@inline frag_coord_y() = frag_coord(2)
@inline frag_coord_z() = frag_coord(3)
@inline frag_coord_w() = frag_coord(4)
@inline frag_coord_xy() = (frag_coord(1), frag_coord(2))

# set_point_size!(s::Float32)
#
# The size in pixels of the point this vertex is drawn as: `[[point_size]]` in MSL.
# An OUTPUT, so the other way round from the builtins above: a store to an external
# global, which `stage_return!` turns into a trailing `float` field of what the
# vertex stage returns, tagged `air.point_size`. A vertex stage that never calls
# this returns no such field.
@eval Metal.@device_function @inline function set_point_size!(s::Float32)
    Base.llvmcall(($("""
        @$POINT_SIZE_GLOBAL = external global float
        define void @entry(float %s) #0 {
            store float %s, ptr @$POINT_SIZE_GLOBAL, align 4
            ret void
        }
        attributes #0 = { alwaysinline }
        """), "entry"), Cvoid, Tuple{Float32}, s)
    return nothing
end

# ── Screen-space derivatives and discard ────────────────────────────────────
#
# `dfdx(v)` / `dfdy(v)`: how `v` changes across the fragment quad in x and in y,
# MSL's `dfdx`/`dfdy`. Fragment stage only: the value is a difference between
# neighbouring invocations, so it exists only where invocations run in a quad. AIR
# gives no diagnostic for calling one elsewhere; the quad simply does not exist and
# the result is undefined rather than an error.
#
# A CALL, unlike the builtins above: those are entry parameters that
# `stage_builtins!` appends, but a derivative takes an argument and reads the
# neighbouring invocations, so there is nothing to pass in.
#
# The names are not guessed. They come from scanning the metallibs Apple ships:
# `air.dfdx.f32` and `air.dfdy.f32` both appear in, among others, IconRendering's
# and RenderBox's `default.metallib`, beside `air.discard_fragment`.

Metal.@device_function dfdx(v::Float32) =
    @typed_ccall("air.dfdx.f32", llvmcall, Cfloat, (Cfloat,), v)

Metal.@device_function dfdy(v::Float32) =
    @typed_ccall("air.dfdy.f32", llvmcall, Cfloat, (Cfloat,), v)

# `discard_fragment()`: MSL's `discard_fragment()`, the fragment-only instruction
# that throws this invocation's colour AND depth away.
#
# `air.discard_fragment`: no type suffix and no operands, unlike the derivatives
# above. Found the same way, and it is the most abundant graphics intrinsic in the
# system metallibs: dozens of Apple's shipped fragment shaders carry it, repeatedly
# in the company of `air.dfdx.f32`.
#
# Execution CONTINUES past this call: it is not a terminator, and the shader still
# returns a colour that is then not used. Nothing here marks the call `readnone` or
# `willreturn`, so LLVM treats an unknown external void call as side-effecting and
# neither sinks nor deletes it, which is exactly right: the whole point of the call
# IS its side effect.
Metal.@device_function discard_fragment() =
    @typed_ccall("air.discard_fragment", llvmcall, Cvoid, ())

# ── Stage input markers ─────────────────────────────────────────────────────
#
# A buffer argument needs no help: the kernel ABI already passes it as
# `ptr addrspace(1)` with the `air.buffer` metadata AIR wants. A STAGE input does
# (the vertex id, the interpolated fragment position, a varying), because those
# arrive as VALUES in a register, and the kernel ABI passes every scalar through a
# buffer instead. Declaring `vid::UInt32` gets you `ptr addrspace(1)`, which is a
# buffer holding a number, not `[[vertex_id]]`.
#
# The marker types below say which is which. They are ordinary immutable structs so
# a shader reads `vid.value`, and `stage_inputs!` turns the parameter into a bare
# scalar.

"""The vertex index, `[[vertex_id]]` in MSL. Read `.value`."""
struct VertexID
    value::UInt32
end

"""The instance index, `[[instance_id]]` in MSL. Read `.value`."""
struct InstanceID
    value::UInt32
end

"""The interpolated clip position a fragment shader receives, `[[position]]`."""
struct FragCoord
    value::NTuple{4,Float32}
end

"""
    Varying{name, T}

One interpolated value a fragment stage reads from the vertex (or mesh) stage.
Read `.value`.

A varying is not a builtin (there is a different one per pipeline), so it is
parameterised by the NAME the vertex stage gave it and by its type, and those two
are exactly what `mangle_varying` needs. The stages link by that mangled string and
by nothing else, so declaring `Varying{:tint, NTuple{4,Float32}}` here and
`tint = NTuple{4,Float32}` in the vertex output is what connects them; a mismatch
in either half silently produces a fragment stage that reads nothing.

One marker per varying rather than one struct holding all of them, because AIR
delivers each as its OWN entry parameter with its own tag.
"""
struct Varying{name, T}
    value::T
end

# ── Sampling a bound 2D texture ─────────────────────────────────────────────
#
# `Lava.sample_texture_2d(binding, u, v, component)` names a SLOT, not a handle,
# because that is the shape a descriptor set gives on Vulkan: the texture is a
# module-level resource there and a shader reaches it by number. AIR has no such
# thing: a texture is a PARAMETER of the entry function, and a shader body several
# calls deep cannot name one. So the body emits a placeholder standing for "the
# thing bound at binding N", and `lower_texture_bindings!` (`compiler/texture.jl`)
# replaces it with the entry's parameter after the shader is inlined into the entry.

"""
    AIRTexture2D{T}

A 2D texture a stage samples, as the pointee of a `Texture2DPtr{T}`. `T` is what a
sample RETURNS (`Float32` for `texture2d<float>`), not the texture's storage format,
which the hardware converts from.

Zero-sized and never loaded: it exists so the parameter type says what the pointer
is, which is what `retag_stage!` reads to emit `air.texture` instead of the
`air.buffer` GPUCompiler described it as.
"""
struct AIRTexture2D{T} end

"""
    AIRSamplerState

A sampler, as the pointee of a [`SamplerPtr`](@ref). Address space 2 is AIR's
`constant`, which is where a bound sampler lives.
"""
struct AIRSamplerState end

"""A 2D texture parameter: `ptr addrspace(1)`, `texture2d<T, sample>` in MSL."""
const Texture2DPtr{T} = Core.LLVMPtr{AIRTexture2D{T}, 1}

"""A sampler parameter: `ptr addrspace(2)`, `sampler` in MSL."""
const SamplerPtr = Core.LLVMPtr{AIRSamplerState, 2}

"""
    AIRSample4f

What `air.sample_texture_2d.v4f32` returns: the four components, and the residency
bit a sparse texture reports.

A struct rather than a tuple because the LLVM type has to be `{ <4 x float>, i8 }`
exactly: `NTuple{4,VecElement{Float32}}` is the `<4 x float>`, and a plain
`NTuple{4,Float32}` would be `[4 x float]`, which is a different type.
"""
struct AIRSample4f
    value::NTuple{4,VecElement{Float32}}
    residency::UInt8
end

# air_sample_texture_2d(texture, sampler, u, v)
#
# One filtered sample at normalised coordinates, with no offset, no explicit LOD and
# the default access hint: the operands the reference call in the header passes for
# `tex.sample(s, uv)`.
Metal.@device_function @inline function air_sample_texture_2d(tex::Texture2DPtr{Float32},
                                                             samp::SamplerPtr,
                                                             u::Float32, v::Float32)
    @typed_ccall("air.sample_texture_2d.v4f32", llvmcall, AIRSample4f,
                 (Texture2DPtr{Float32}, SamplerPtr, NTuple{2,VecElement{Float32}},
                  Bool, NTuple{2,VecElement{Int32}}, Bool, Float32, Float32, Int32),
                 tex, samp, (VecElement(u), VecElement(v)),
                 true, (VecElement(Int32(0)), VecElement(Int32(0))), false,
                 0f0, 0f0, Int32(0))
end

# The placeholders a shader body emits for "the texture/sampler bound at binding N".
# `air.jl.` and not `air.`, because nothing in AIR answers these: they exist only
# between the shader body and `lower_texture_bindings!`, and a module still holding
# one when it reaches the driver is a bug that pass catches.
const TEXTURE_PLACEHOLDER = "air.jl.texture_2d_at_binding"
const SAMPLER_PLACEHOLDER = "air.jl.sampler_at_binding"

# texture_2d_at_binding(binding) / sampler_at_binding(binding)
#
# "The texture bound at `binding`", as a value the shader can hand the intrinsic.
# `binding` counts from ZERO, matching `Lava.sample_texture_2d`, and has to be a
# literal: `lower_texture_bindings!` reads it off the call. Through `@eval` so the
# placeholder names have ONE spelling: `@typed_ccall` needs a literal, which the
# interpolation gives it.
for (fname, placeholder, RT) in
        ((:texture_2d_at_binding, TEXTURE_PLACEHOLDER, :(Texture2DPtr{Float32})),
         (:sampler_at_binding, SAMPLER_PLACEHOLDER, :SamplerPtr))
    @eval Metal.@device_function @inline $fname(binding::Int32) =
        @typed_ccall($placeholder, llvmcall, $RT, (Int32,), binding)
end

"""
    sample_texture_2d(binding, u, v, component) -> Float32

One component of a filtered sample from the 2D texture bound at `binding`.

`binding` and `component` both count from zero, which is
`Lava.sample_texture_2d`'s convention: a component is a vector lane and a binding
is an encoder slot, and neither is an index into anything Julia owns.
"""
@inline function sample_texture_2d(binding::Integer, u::Float32, v::Float32,
                                   component::Integer)
    s = air_sample_texture_2d(texture_2d_at_binding(Int32(binding)),
                              sampler_at_binding(Int32(binding)), u, v)
    return @inbounds s.value[Int(component) + 1].value
end
