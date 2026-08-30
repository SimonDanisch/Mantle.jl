# Fixed-function rasterizer state: blending, culling, depth.
#
# These were Lava's, and nothing about them is a compiler's business — the
# search for uses outside its own `graphics/types.jl` found exactly one hit
# each, the `export` line. Blending and culling are not shader code; they are
# state a pipeline is created with, which is Mantle's job on every backend.
#
# `Topology` did NOT come here, and the difference is the point: the compiler
# genuinely dispatches on it, to pick a geometry stage's input execution mode.
# Two owners means it belongs to neither, so it is `KernelInterface`'s, beside
# `MatrixShape` and `DeviceCaps`.
#
# The state is carried in type parameters rather than fields so a pipeline's
# configuration is part of its type and dispatch is free — `GraphicsPipeline`
# is parameterised on all three.

"""
    BlendMode

How a fragment's colour combines with what is already in the target.

Backends map these to their own enumerations —
`VK_BLEND_FACTOR_SRC_ALPHA` and friends on Vulkan, `MTLBlendFactor` on Metal.
"""
abstract type BlendMode end

"No blending: the fragment replaces the target."
struct Opaque <: BlendMode end

"`src.a * src + (1 - src.a) * dst`."
struct AlphaBlend <: BlendMode end

"`src + dst`."
struct Additive <: BlendMode end

"Alpha already multiplied into the colour: `src + (1 - src.a) * dst`."
struct Premultiplied <: BlendMode end

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

"Farther wins — reversed-Z."
struct DepthGreater <: DepthMode end

"Every fragment passes, and still writes depth."
struct DepthAlways <: DepthMode end

"No depth test and no depth write."
struct DepthOff <: DepthMode end

"""
    RenderTarget

What a graphics pipeline renders into.

Also Lava's until now, and also used nowhere in it: what a pipeline draws to is
the runtime's concern, not the compiler's. The concrete targets belong to a
backend, since each wraps something a driver owns — a swapchain image, an
offscreen framebuffer — so they subtype this rather than replacing it.
"""
abstract type RenderTarget end
