# Fixed-function rasterizer state: blending, culling, depth. Pipeline creation
# state, so Mantle's and not the compiler's; `Topology`, which the compiler
# dispatches on, is `KernelInterface`'s. Carried in type parameters, so a
# pipeline's configuration is part of its type.

"""
    BlendMode

How a fragment's colour combines with what is already in the target. Backends
map these to `VkBlendFactor` or `MTLBlendFactor`.
"""
abstract type BlendMode end

"No blending: the fragment replaces the target."
struct Opaque <: BlendMode end

"`src.a * src + (1 - src.a) * dst`; alpha composites as `src.a + (1 - src.a) * dst.a`."
struct AlphaBlend <: BlendMode end

"`src + dst`."
struct Additive <: BlendMode end

"Alpha already multiplied into the colour: `src + (1 - src.a) * dst`."
struct Premultiplied <: BlendMode end

"""
The fragment writes nothing to this attachment, such as the id attachment of a
[`PerAttachment`](@ref) blend whose colour should land. Depth is still tested
and written as the pipeline's depth mode says.
"""
struct NoWrite <: BlendMode end

"""
    PerAttachment(modes...)

One blend mode per colour attachment, in pass order. A plain mode applies to
every floating-point attachment and writes integer (id) attachments unblended;
here an integer attachment takes `Opaque()` or `NoWrite()`.
"""
struct PerAttachment{M<:Tuple{Vararg{BlendMode}}} <: BlendMode
    modes::M
end
PerAttachment(modes::BlendMode...) = PerAttachment(modes)

"""
    blendfactors(mode) -> (src, dst, srcalpha, dstalpha) or nothing

The factors of `mode`'s additive blend equation, as `:one`, `:srcalpha` or
`:oneminussrcalpha`; `nothing` when the fragment is not blended. Backends map the
symbols to their native factors.
"""
blendfactors(::Union{Opaque,NoWrite}) = nothing
blendfactors(::AlphaBlend) = (:srcalpha, :oneminussrcalpha, :one, :oneminussrcalpha)
blendfactors(::Premultiplied) = (:one, :oneminussrcalpha, :one, :oneminussrcalpha)
blendfactors(::Additive) = (:one, :one, :one, :one)

"""Whether `mode` writes the colour attachment at all."""
writescolor(::BlendMode) = true
writescolor(::NoWrite) = false

"""
    attachmentblend(blend, i, integer) -> BlendMode

The mode colour attachment `i` is written with: `integer` is whether it holds
integers, which are never blended.
"""
attachmentblend(blend::BlendMode, i::Integer, integer::Bool) = integer ? Opaque() : blend
function attachmentblend(blend::PerAttachment, i::Integer, integer::Bool)
    i <= length(blend.modes) || throw(ArgumentError(
        "PerAttachment has $(length(blend.modes)) modes and the pass has an attachment $i"))
    mode = blend.modes[i]
    integer && !(mode isa Union{Opaque, NoWrite}) && throw(ArgumentError(
        "attachment $i holds integers, which are written (Opaque) or left alone (NoWrite), not $(nameof(typeof(mode)))"))
    return mode
end

"""
    CullFace

Which triangle winding is discarded before rasterization.
"""
abstract type CullFace end

"Draw both faces."
struct NoCull <: CullFace end

"Discard back faces."
struct CullBack <: CullFace end

"Discard front faces."
struct CullFront <: CullFace end

"""
    DepthMode

The depth comparison a fragment must pass, or [`DepthOff`](@ref) for no depth
test and no depth write.
"""
abstract type DepthMode end

"Closer wins. The default."
struct DepthLess <: DepthMode end

"Closer or equal wins."
struct DepthLessEq <: DepthMode end

"""
Closer or equal wins, without writing depth: for see-through draws drawn after
the opaque ones, hidden by what is solid but hiding nothing.
"""
struct DepthLessEqReadOnly <: DepthMode end

"Farther wins (reversed-Z)."
struct DepthGreater <: DepthMode end

"Every fragment passes, and still writes depth."
struct DepthAlways <: DepthMode end

"No depth test and no depth write."
struct DepthOff <: DepthMode end

"""
    depthtest(mode) -> Symbol or nothing

The comparison `mode` tests with (`:less`, `:lessequal`, `:greater`, `:always`),
or `nothing` for no depth test. Backends map it to their native compare op.
"""
depthtest(::DepthLess) = :less
depthtest(::Union{DepthLessEq,DepthLessEqReadOnly}) = :lessequal
depthtest(::DepthGreater) = :greater
depthtest(::DepthAlways) = :always
depthtest(::DepthOff) = nothing

"""Whether `mode` writes depth."""
depthwrites(::DepthMode) = true
depthwrites(::Union{DepthLessEqReadOnly,DepthOff}) = false

"""
    RenderTarget

What a graphics pipeline renders into. Concrete targets belong to a backend,
since each wraps a driver-owned image (a swapchain image, an offscreen
framebuffer).
"""
abstract type RenderTarget end
