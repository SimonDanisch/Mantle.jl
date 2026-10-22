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

A two-dimensional texture; backends provide the concrete type (`VulkanTexture2D`).
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
    upload_texture_data!(texture, data)

Write new pixels into an existing texture. A recorded draw bakes the descriptor
set naming the image, so replacing the texture would leave the recording on the
old one. `data` must match the texture's `size` and `eltype`. Backends implement
it and `Base.size` for their texture type; `Base.eltype` is answered here.
"""
function upload_texture_data! end

# From the type parameter, for every backend. Without it `eltype` is `Any`, no
# upload fits, and every update rebuilds the texture, its bindings and the plan.
Base.eltype(::Texture{T}) where {T} = T

"""
    Framebuffer(backend, width, height; depth = true, color_format = …)

Colour and depth attachments a pass renders into, off-screen. Constructed through
the abstract type with a backend as the first argument, which dispatch needs once
two backends are loaded; each answers with its own (`VulkanFramebuffer`,
`MetalFramebuffer`).
"""
abstract type Framebuffer end

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

"""
    eltypefor(format, f) -> Type

The attachment element type that the backend's `format(T; srgb)` maps to `f`, in
its plain or sRGB form. sRGB and UNORM share a memory layout, so both read back as
the same colorant.
"""
function eltypefor(format, f)
    for T in (BGRA{N0f8}, RGBA{N0f8}, RGBA{Float16}, RGBA{Float32}, Float32, UInt32, SVector{2,UInt32})
        srgb = T <: Union{BGRA{N0f8},RGBA{N0f8}}
        (format(T) == f || srgb && format(T; srgb) == f) && return T
    end
    throw(ArgumentError("no Julia element type is registered for the format $f"))
end

# The element type a pipeline is compiled against: the target's format read back
# as the Julia type it was lowered from. In core, since two loaded backends cannot
# both define it.
blittarget(t::OffscreenTarget) = eltypeof(target_format(t.fb))

# A graph names a framebuffer as an attachment directly and `attachment_format`
# asks its `eltype`; without this the pipeline is compiled for `Any`.
Base.eltype(fb::Framebuffer) = eltypeof(target_format(fb))
blittarget(t::WindowTarget) = eltypeof(target_format(t.window))
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
