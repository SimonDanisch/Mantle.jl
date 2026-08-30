# The resources a graphics or ray-tracing pipeline is built from.
#
# Every type here is Mantle's, and every one of them was the Vulkan backend's
# until now — `LavaTexture2D`, `LavaSampler`, `LavaFramebuffer`, `HWTLAS`. The
# names said which backend happened to implement them first, which is not a
# thing a caller should have to know: Hikari asks for a 2D texture, and on a Mac
# it should get a Metal one.
#
# So each splits in two. The ABSTRACT type is the API and lives here; the
# CONCRETE type is one backend's and is named for it —
# `VulkanTexture2D <: Texture2D`, and `MetalTexture2D` beside it later. Nothing
# downstream names either concrete.
#
# Three of these turn out to be portable outright, because all they do is pair
# things that are now abstract: `SampledTexture`, `WindowTarget` and
# `OffscreenTarget` are defined here in full, not declared.

# ── Textures and samplers ─────────────────────────────────────────────────────

"""
    Texture{T,N}

An `N`-dimensional image of element type `T`, owned by a device.

Backends subtype [`Texture1D`](@ref) / [`Texture2D`](@ref) rather than this
directly; this is the type to write when the rank does not matter.
"""
abstract type Texture{T, N} end

"""
    Texture1D{T}

A one-dimensional texture. Backends provide the concrete type — `VulkanTexture1D`.
"""
abstract type Texture1D{T} <: Texture{T, 1} end

"""
    Texture2D{T}

A two-dimensional texture. Backends provide the concrete type — `VulkanTexture2D`.
"""
abstract type Texture2D{T} <: Texture{T, 2} end

"""
    Sampler

How a texture is filtered and addressed when a shader reads it.

Filtering, wrap mode and anisotropy are the portable vocabulary; the object a
driver needs for them is the backend's.
"""
abstract type Sampler end

"""
    SampledTexture(texture, sampler)

A texture paired with the sampler a shader will read it through.

Defined here rather than declared, and that is the whole payoff of the split
above: this type does nothing but hold one [`Texture`](@ref) and one
[`Sampler`](@ref), so once those are the API the pairing is portable and no
backend needs its own.
"""
struct SampledTexture{T, N}
    texture::Texture{T, N}
    sampler::Sampler
end

"""
    TextureBindings

A bound table of [`SampledTexture`](@ref)s a pipeline can index.

Concrete on every backend, because it IS the driver object — a descriptor set on
Vulkan, an argument buffer on Metal.
"""
abstract type TextureBindings end

"""
    bind_textures(textures) -> TextureBindings

Bind `textures` as one indexable table for a pipeline to sample from.
"""
function bind_textures end

# ── Render targets ────────────────────────────────────────────────────────────

"""
    Framebuffer(backend, width, height; depth = true, color_format = …)

Colour and depth attachments a pass renders into, off-screen.

Constructed through the abstract type with a backend as the first argument, so
a caller names no concrete type: the Vulkan backend answers with a
`VulkanFramebuffer` and a Metal one will answer with its own. That first
argument is the reason it is here rather than a bare `Framebuffer(w, h)` —
without it there is nothing to dispatch on once two backends are loaded.

[`RenderTarget`](@ref) is what a pipeline is pointed at; this is one of the two
things that can be behind it.
"""
abstract type Framebuffer end

"""
    WindowTarget(window)

Render into a window's swapchain.

Portable for the same reason [`SampledTexture`](@ref) is: [`Window`](@ref) is
already the API type, so this wrapper has nothing backend-specific left in it.
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

"""
    CompiledGraphicsPipeline

A [`GraphicsPipeline`](@ref) after a backend has compiled it for a target.

The description is portable and the compiled object is not, which is why they
are two types rather than one with a cache field.
"""
abstract type CompiledGraphicsPipeline end

# ── Acceleration structures ───────────────────────────────────────────────────

"""
    HWTLAS{Tri}

A top-level acceleration structure: the instances a ray is traced against,
holding triangles of type `Tri`.

Was `HWTLAS`. Hardware ray tracing is not a Vulkan feature — Metal has
`MTLAccelerationStructure` and DXR has its own — so the name carries no vendor.

Parameterised on the triangle type because a caller picks it: Hikari pushes
`Raycore.Triangle{TriangleMeta}` for its per-face data, where the default
`Triangle{UInt32}` would reject those with a convert error.

    HWTLAS{Tri}(backend) -> HWTLAS{Tri}

Constructed through the abstract type with a KernelAbstractions backend, so a
caller names no concrete type — the Vulkan backend answers with a `VulkanTLAS`
and Metal with a `MetalTLAS`. Backends implement this constructor.
"""
abstract type HWTLAS{Tri} <: Raycore.AbstractAccel end

# `AccelBuildContext` is a concrete shared struct in `raytracing/accel.jl`:
# once `BatchQueue` was shared, both of its fields were portable.

# ── Queues and interop ────────────────────────────────────────────────────────

# `BatchQueue` is a concrete, shared struct in `graph/queue.jl`. It was abstract
# here with one backend-specific implementation, and thirty-one of its
# forty-one fields turned out to be backend-independent scheduling state.

"""
    ExternalImage

An image whose memory is shared with another process or API.

The handle type is the backend's — an fd or a Windows handle on Vulkan, an
`IOSurface` on Metal — so only the concept is declared here.
"""
abstract type ExternalImage end
