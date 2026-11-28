# Pixel formats are element types, not an enum: `RGBA{Float16}` names
# `R16G16B16A16_SFLOAT` in a type kernels, `fill!` and readback can use.
# ColorTypes' layouts match the backends' byte for byte (`RGBA{N0f8}`/`BGRA{N0f8}`
# 8-bit UNORM, `RGBA{Float16}`/`RGBA{Float32}` float, `Float32` as 32-bit depth,
# `UInt32` 32-bit unsigned, `SVector{2,UInt32}` two of those). sRGB changes
# conversion, not memory, so it is an image keyword; depth is the `Depth` usage.
# Core owns which element types are pixels and the rules on them; a backend only
# names its value for each (`nativeformat`).

"""Bytes one pixel occupies: `sizeof` of its element type, which is its layout."""
pixelbytes(::Type{T}) where {T} = sizeof(T)

"""
The pixel layout an element type names. The tuple spellings portable shaders
write are the colour types of the same layout.
"""
pixeltype(::Type{T}) where {T} = T
pixeltype(::Type{NTuple{4,UInt8}}) = RGBA{N0f8}
pixeltype(::Type{NTuple{4,Float16}}) = RGBA{Float16}
pixeltype(::Type{NTuple{4,Float32}}) = RGBA{Float32}
pixeltype(::Type{<:StaticVector{2,UInt32}}) = SVector{2,UInt32}

"""
What `Float32` names as a render target: a 32-bit float depth attachment, since no
backend renders into a single-channel float colour target the way it tests depth.
"""
struct Depth32 end

"""Whether `P` has an sRGB form: sRGB encodes 8-bit colour only."""
hassrgb(::Type{P}) where {P} = P <: Union{RGBA{N0f8},BGRA{N0f8}}

"""
    pixelformat(api, T; srgb = false) -> native format

The backend's format for render targets and images of element type `T`; `Float32`
is depth. `srgb` picks the encoding form, which only 8-bit colour has.
"""
function pixelformat(api::Backend, ::Type{T}; srgb::Bool = false) where {T}
    P = T === Float32 ? Depth32 : pixeltype(T)
    srgb && !hassrgb(P) && throw(ArgumentError(
        "there is no sRGB format for $T: sRGB encodes 8-bit colour only"))
    return nativeformat(api, P, srgb)
end

"""
    sampledformat(api, T) -> native format

The backend's format for a sampled texture of `T`: as [`pixelformat`](@ref), except
that a scalar float is a colour channel, since a sampler rejects depth.
"""
sampledformat(api::Backend, ::Type{T}) where {T} = nativeformat(api, pixeltype(T), false)

"""
    nativeformat(api, P, srgb) -> native format

A backend's value for the pixel layout `P` ([`pixeltype`](@ref), or [`Depth32`](@ref)),
in its sRGB form if `srgb`, which core has checked `P` has.
"""
nativeformat(api::Backend, ::Type{P}, srgb::Bool) where {P} = throw(ArgumentError(
    "$(nameof(typeof(api))) has no pixel format for $P"))

"""
    eltypefor(api, f) -> Type

The element type whose [`pixelformat`](@ref) is the native format `f`, in its plain
or sRGB form: sRGB and UNORM share a memory layout, so both read back as the same
colorant.
"""
function eltypefor(api::Backend, f)
    for T in (BGRA{N0f8}, RGBA{N0f8}, RGBA{Float16}, RGBA{Float32}, Float32, UInt32, SVector{2,UInt32})
        (pixelformat(api, T) == f || hassrgb(T) && pixelformat(api, T; srgb = true) == f) && return T
    end
    throw(ArgumentError("no Julia element type is registered for the format $f"))
end
