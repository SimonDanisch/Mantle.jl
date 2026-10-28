# The resources a graphics or ray-tracing pipeline is built from. The abstract
# type is the API; each backend names its concrete subtype (`VulkanTexture2D <:
# Texture2D`), and nothing downstream names a concrete type. `SampledTexture`,
# `WindowTarget` and `OffscreenTarget` only pair abstract types, so they are
# defined here in full.

"""
    Texture{T,N}

An `N`-dimensional image of element type `T`, owned by a device. Backends
subtype [`Texture2D`](@ref), not this.
"""
abstract type Texture{T, N} end

"""
    Texture2D{T}

A two-dimensional texture; backends provide the concrete type (`maketexture`).
"""
abstract type Texture2D{T} <: Texture{T, 2} end

"""
    mipmapped(tex::Texture2D) -> Bool

Whether `tex` has mip levels above level 0 (it was made with `mipmap = true`).
"""
function mipmapped end

"""
    Sampler

How a texture is filtered and addressed when a shader reads it. The driver
object is the backend's.
"""
abstract type Sampler end

"""
    SampledTexture(texture, sampler)

A texture paired with the sampler a shader reads it through. Parameterised on the
concrete texture and sampler types so a backend can dispatch
[`bind_textures`](@ref), which takes no device argument, on its own texture.
"""
struct SampledTexture{T, N, TEX<:Texture{T, N}, SAM<:Sampler}
    texture::TEX
    sampler::SAM
end

# No outer constructor: the default one solves `T` and `N` from `TEX <: Texture{T,N}`,
# and a hand-written one with that signature overwrites it and breaks precompilation.

"""
    TextureBindings

A bound table of [`SampledTexture`](@ref)s a pipeline can index: a descriptor set
on Vulkan, an argument buffer on Metal.
"""
abstract type TextureBindings end

"""
    bind_textures(textures) -> TextureBindings

Bind `textures` as one indexable table for a pipeline to sample from.
"""
function bind_textures end

"""
    upload_texture_data!(texture, data) -> texture

Write new pixels into an existing texture. A recorded draw bakes the descriptor
set naming the image, so replacing the texture would leave the recording on the
old one. `data` must match the texture's `size` and `eltype`: the copy covers the
texture's extent and would read past the end of smaller data. Backends implement
[`copytexels!`](@ref) and `Base.size` for their texture type.
"""
function upload_texture_data!(t::Texture2D, data::AbstractMatrix)
    size(data) == size(t) || throw(DimensionMismatch(
        "upload_texture_data!: $(size(data)) texels for a $(size(t)) texture"))
    copytexels!(t, data)
    return t
end

"""
    copytexels!(texture, data)

The backend's copy of a host or device matrix of the texture's size and element
type into its first mip level, rebuilding the rest of the chain.
"""
function copytexels! end

# From the type parameter, for every backend. Without it `eltype` is `Any`, no
# upload fits, and every update rebuilds the texture, its bindings and the plan.
Base.eltype(::Texture{T}) where {T} = T

"""
    Framebuffer(device, width, height; depth = true, color_format = BGRA{N0f8}, srgb = false)

Colour and depth attachments a pass renders into, off-screen: a colour target of
element type `color_format` (`srgb` picks the encoding form) and, with `depth`, a
`Float32` depth target. The backend answers with its own type
(`makeframebuffer`).
"""
abstract type Framebuffer end

function Framebuffer(dev::Device, width::Integer, height::Integer; depth::Bool = true,
                     color_format::Type = BGRA{N0f8}, srgb::Bool = false)
    api = syncbackend(dev)
    return makeframebuffer(dev, Int(width), Int(height), pixelformat(api, color_format; srgb),
                           depth ? pixelformat(api, Float32) : nothing)
end

"""
    makeframebuffer(device, width, height, color, depth) -> Framebuffer

The backend's offscreen target with a colour attachment of native format `color`
and a depth attachment of native format `depth`, or none for `nothing`.
"""
function makeframebuffer end

"""
    Texture2D(device, data::AbstractMatrix; mipmap = false)

A sampled 2D texture holding `data` (host or device), with the element type naming
the format (`sampledformat`). `data[x, y]`: the first index is horizontal, as in
GLMakie; the upload copies the column-major bytes as they are, so a Julia column
is one image row. `mipmap = true` makes the full mip chain, regenerated from level
0 at every upload.
"""
function Texture2D(dev::Device, data::AbstractMatrix{T}; mipmap::Bool = false) where {T}
    w, h = size(data)
    levels = mipmap ? floor(Int, log2(max(w, h))) + 1 : 1
    t = maketexture(dev, T, sampledformat(syncbackend(dev), T), w, h, levels)
    upload_texture_data!(t, data)
    return t
end

"""
    maketexture(device, T, format, width, height, levels) -> Texture2D{T}

The backend's empty sampled texture of native `format` with `levels` mip levels,
which `upload_texture_data!` fills.
"""
function maketexture end

"""
    Sampler(device; filter = :linear, wrap = :repeat, anisotropy = 0)

How a texture is sampled: `filter` `:nearest` or `:linear`, `wrap` `:repeat`,
`:clamp`, `:mirror` or `:clamp_border` (transparent black outside), and up to
`anisotropy` anisotropic samples, 0 for none. Mip levels are blended linearly.
"""
function Sampler(dev::Device; filter::Symbol = :linear, wrap::Symbol = :repeat, anisotropy::Real = 0)
    filter in (:nearest, :linear) || throw(ArgumentError(
        "unknown filter :$filter; expected :nearest or :linear"))
    wrap in (:repeat, :clamp, :mirror, :clamp_border) || throw(ArgumentError(
        "unknown wrap :$wrap; expected :repeat, :clamp, :mirror or :clamp_border"))
    anisotropy >= 0 || throw(ArgumentError("anisotropy is a sample count, got $anisotropy"))
    return makesampler(dev, filter, wrap, Float32(anisotropy))
end

"""
    makesampler(device, filter, wrap, anisotropy) -> Sampler

The backend's sampler for keywords [`Sampler`](@ref) has checked.
"""
function makesampler end

"""
    WindowTarget(window)

Render into a [`Window`](@ref)'s swapchain.
"""
struct WindowTarget <: RenderTarget
    window::Window
end

"""
    OffscreenTarget(framebuffer)

Render into a [`Framebuffer`](@ref) rather than to a window.
"""
struct OffscreenTarget <: RenderTarget
    fb::Framebuffer
end

# Core's, because the wrappers are: a backend answers for its own framebuffer and
# window. `pass!` and `blit!` ask a target for its extent.
target_extent(t::OffscreenTarget) = target_extent(t.fb)
target_extent(t::WindowTarget) = target_extent(t.window)

"""
    eltypeof(format) -> Type

The Julia element type of an attachment format: a backend's answer for its own
format type, `VK.Format` or `MTLPixelFormat`, through [`eltypefor`](@ref).
"""
function eltypeof end

# A graph names a framebuffer as an attachment directly; its pixel type is what a
# readback of it holds.
Base.eltype(fb::Framebuffer) = eltypeof(target_format(fb))
target_format(t::OffscreenTarget) = target_format(t.fb)
target_format(t::WindowTarget) = target_format(t.window)

"""
    CompiledGraphicsPipeline

A [`GraphicsPipeline`](@ref) after a backend has compiled it for a target; the
description is portable, the compiled object is not.
"""
abstract type CompiledGraphicsPipeline end

"""
    HWTLAS{Tri}
    HWTLAS{Tri}(backend) -> HWTLAS{Tri}

A top-level acceleration structure: the instances a ray is traced against,
holding triangles of type `Tri`, which the caller picks (Hikari uses
`Raycore.Triangle{TriangleMeta}` for per-face data). Constructed through the
abstract type with a KernelAbstractions backend; each backend implements the
constructor (`VulkanTLAS`, `MetalHWTLAS`).
"""
abstract type HWTLAS{Tri} <: Raycore.AbstractAccel end

# `AccelBuildContext` (`raytracing/accel.jl`) and `SubmitChannel`
# (`graph/lifetime.jl`) are concrete shared structs, so they have no abstract type.

"""
    ExternalImage

An image whose memory is shared with another process or API. The handle type is
the backend's: an fd or Windows handle on Vulkan, an `IOSurface` on Metal.
"""
abstract type ExternalImage end
