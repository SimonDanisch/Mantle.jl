# The Vulkan lowering: the only place in Mantle that names a pipeline stage, an
# access mask or an image layout. Constants come from VK.jl, so a typo fails at
# load instead of producing a wrong barrier.
import Vulkan as VK
using ColorTypes: RGBA, BGRA
using ColorTypes.FixedPointNumbers: N0f8


# VK.jl types a `_2_` constant that fits in 32 bits as the sync1 flag type, and
# BitMasks refuses to combine the two, so every constant is normalised here.
stage(x) = VK.PipelineStageFlag2(x)
accessflags(x) = VK.AccessFlag2(x)

const NO_STAGE = stage(0)
"""Where a shader-addressable resource is touched from. Every usage that means "a
shader reads or writes this" lowers to these three and nothing wider."""
const SHADER_STAGES = VK.PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                      VK.PIPELINE_STAGE_2_VERTEX_SHADER_BIT |
                      VK.PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT
const NO_ACCESS = accessflags(0)

# Vertices are pulled, not bound: pipelines have an empty vertex input state and
# the vertex shader loads attributes through a device address, so the read is a
# storage load in VERTEX_SHADER. `VERTEX_ATTRIBUTE_READ` would not make a
# compute write visible to it, silently, on hardware with separate caches.
stages(::VulkanAPI, ::Type{Vertices}, _) = stage(VK.PIPELINE_STAGE_2_VERTEX_SHADER_BIT)
stages(::VulkanAPI, ::Type{Indices}, _) = stage(VK.PIPELINE_STAGE_2_INDEX_INPUT_BIT)
stages(::VulkanAPI, ::Type{Indirect}, _) = stage(VK.PIPELINE_STAGE_2_DRAW_INDIRECT_BIT)
# Read by the conditional-rendering unit and by the fused prepare's compute
# kernel; both stages, or the gate's write is visible to only one.
stages(::VulkanAPI, ::Type{Predicated}, _) =
    stage(VK.PIPELINE_STAGE_2_CONDITIONAL_RENDERING_BIT_EXT) | stage(VK.PIPELINE_STAGE_2_COMPUTE_SHADER_BIT)
# Shader reads, so the shader stages; no usage here lowers to ALL_COMMANDS.
stages(::VulkanAPI, ::Type{Uniform}, _) = stage(SHADER_STAGES)
stages(::VulkanAPI, ::Type{Sampled}, _) = stage(SHADER_STAGES)
# The shader stages rather than ALL_COMMANDS: for precision, not speed (frame
# time was the same). The aliasing barrier derives its own mask from the
# vacating transient.
stages(::VulkanAPI, ::Type{<:Storage}, _) = stage(SHADER_STAGES)
stages(::VulkanAPI, ::Type{CopySrc}, _) = stage(VK.PIPELINE_STAGE_2_COPY_BIT)
stages(::VulkanAPI, ::Type{CopyDst}, _) = stage(VK.PIPELINE_STAGE_2_COPY_BIT)
stages(::VulkanAPI, ::Type{TraceRead}, _) = stage(VK.PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR)
stages(::VulkanAPI, ::Type{TraceBuild}, _) = stage(VK.PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR)
stages(::VulkanAPI, ::Type{<:ColorAttachment}, _) = stage(VK.PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT)

# Nothing produced the contents, so nothing has to complete first. NONE is the
# sync2 spelling of that and is legal on either side, unlike TOP_OF_PIPE.
stages(::VulkanAPI, ::Type{Undefined}, _) = stage(VK.PIPELINE_STAGE_2_NONE)

# Present as a source is the acquire, which signals at colour attachment output.
# TOP_OF_PIPE creates no dependency on it and corrupted frames.
stages(::VulkanAPI, ::Type{Present}, ::Src) = stage(VK.PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT)
# Nothing in the submission waits on the transition into PRESENT_SRC; the
# render-finished semaphore orders presentation.
stages(::VulkanAPI, ::Type{Present}, ::Dst) = stage(VK.PIPELINE_STAGE_2_NONE)

# Depth writes complete late and reads begin early, so source and destination differ.
stages(::VulkanAPI, ::Type{<:Depth}, ::Src) =
    stage(VK.PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT)
stages(::VulkanAPI, ::Type{<:Depth}, ::Dst) =
    stage(VK.PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT)

# Must match the VERTEX_SHADER stage above: `VERTEX_ATTRIBUTE_READ` there is
# VUID-VkMemoryBarrier2-srcAccessMask-03902.
access(::VulkanAPI, ::Type{Vertices}, _) = accessflags(VK.ACCESS_2_SHADER_STORAGE_READ_BIT)
access(::VulkanAPI, ::Type{Indices}, _) = accessflags(VK.ACCESS_2_INDEX_READ_BIT)
access(::VulkanAPI, ::Type{Indirect}, _) = accessflags(VK.ACCESS_2_INDIRECT_COMMAND_READ_BIT)
access(::VulkanAPI, ::Type{Predicated}, _) =
    accessflags(VK.ACCESS_2_CONDITIONAL_RENDERING_READ_BIT_EXT) | accessflags(VK.ACCESS_2_SHADER_STORAGE_READ_BIT)
access(::VulkanAPI, ::Type{Uniform}, _) = accessflags(VK.ACCESS_2_UNIFORM_READ_BIT)
access(::VulkanAPI, ::Type{Sampled}, _) = accessflags(VK.ACCESS_2_SHADER_SAMPLED_READ_BIT)
access(::VulkanAPI, ::Type{CopySrc}, _) = accessflags(VK.ACCESS_2_TRANSFER_READ_BIT)
access(::VulkanAPI, ::Type{CopyDst}, _) = accessflags(VK.ACCESS_2_TRANSFER_WRITE_BIT)
access(::VulkanAPI, ::Type{TraceRead}, _) = accessflags(VK.ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR)
access(::VulkanAPI, ::Type{TraceBuild}, _) = accessflags(VK.ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR)
access(::VulkanAPI, ::Type{Present}, _) = NO_ACCESS
access(::VulkanAPI, ::Type{Undefined}, _) = NO_ACCESS

function access(::VulkanAPI, ::Type{Storage{K,A}}, _) where {K,A}
    a = NO_ACCESS
    reads(A) && (a |= accessflags(VK.ACCESS_2_SHADER_STORAGE_READ_BIT))
    writes(A) && (a |= accessflags(VK.ACCESS_2_SHADER_STORAGE_WRITE_BIT))
    a
end

# `Unordered` only drops transitions between unordered accesses; a barrier
# against an ordinary access (the clear before, the read after) is the inner
# usage's.
stages(be::VulkanAPI, ::Type{Unordered{U}}, d) where {U} = stages(be, U, d)
access(be::VulkanAPI, ::Type{Unordered{U}}, d) where {U} = access(be, U, d)
layout(be::VulkanAPI, ::Type{Unordered{U}}, d) where {U} = layout(be, U, d)

# `Traced` changes only the stage, to RAY_TRACING_SHADER; access and layout are
# the inner usage's.
stages(::VulkanAPI, ::Type{Traced{U}}, _) where {U} =
    stage(VK.PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR)
access(be::VulkanAPI, ::Type{Traced{U}}, d) where {U} = access(be, U, d)
layout(be::VulkanAPI, ::Type{Traced{U}}, d) where {U} = layout(be, U, d)

# A discarding load op reads nothing back, so the read bit is dropped.
access(::VulkanAPI, ::Type{ColorAttachment{Discard}}, _) where {Discard} =
    Discard ? accessflags(VK.ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT) :
              accessflags(VK.ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT) | accessflags(VK.ACCESS_2_COLOR_ATTACHMENT_READ_BIT)

function access(::VulkanAPI, ::Type{Depth{DA,SA,D}}, _) where {DA,SA,D}
    a = NO_ACCESS
    # `D`, a discarding load op, drops the read bit as for colour attachments.
    !D && (reads(DA) || reads(SA)) && (a |= accessflags(VK.ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT))
    (writes(DA) || writes(SA)) && (a |= accessflags(VK.ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT))
    a
end

layout(::VulkanAPI, ::Type{U}, _) where {U<:Usage} = VK.IMAGE_LAYOUT_UNDEFINED
layout(::VulkanAPI, ::Type{Sampled}, _) = VK.IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
layout(::VulkanAPI, ::Type{CopySrc}, _) = VK.IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL
layout(::VulkanAPI, ::Type{CopyDst}, _) = VK.IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL
layout(::VulkanAPI, ::Type{<:ColorAttachment}, _) = VK.IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
layout(::VulkanAPI, ::Type{<:Storage{ImageKind}}, _) = VK.IMAGE_LAYOUT_GENERAL

# Nothing is preserved out of a presented image, so its source layout is UNDEFINED.
layout(::VulkanAPI, ::Type{Present}, ::Src) = VK.IMAGE_LAYOUT_UNDEFINED
layout(::VulkanAPI, ::Type{Present}, ::Dst) = VK.IMAGE_LAYOUT_PRESENT_SRC_KHR

# The depth and stencil write bits select one of four layouts.
layout(::VulkanAPI, ::Type{Depth{DA,SA,D}}, _) where {DA,SA,D} =
    writes(DA) && writes(SA) ? VK.IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL :
    writes(DA) && !writes(SA) ? VK.IMAGE_LAYOUT_DEPTH_ATTACHMENT_STENCIL_READ_ONLY_OPTIMAL :
    !writes(DA) && writes(SA) ? VK.IMAGE_LAYOUT_DEPTH_READ_ONLY_STENCIL_ATTACHMENT_OPTIMAL :
                                VK.IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL

# No stencil aspect: the combined layouts, legal for depth-only images on every
# device; the separate-aspect ones need `separateDepthStencilLayouts`.
layout(::VulkanAPI, ::Type{Depth{DA,NoAccess,D}}, _) where {DA,D} =
    writes(DA) ? VK.IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL :
                 VK.IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL

"""
Vulkan needs a transition when the portable rule says so, and for images also
when the layout changes, which can happen between two reads (sampling, then
copying from the same image).
"""
needs_transition(be::VulkanAPI, kind::BufferKind, before::Type, after::Type) =
    invoke(needs_transition, Tuple{Backend,ResourceKind,Type,Type}, be, kind, before, after)

function needs_transition(be::VulkanAPI, kind::ImageKind, before::Type, after::Type)
    invoke(needs_transition, Tuple{Backend,ResourceKind,Type,Type},
           be, kind, before, after) && return true
    layout(be, inner(before), Src()) != layout(be, inner(after), Dst())
end


"""
    vkformat(api, T; srgb = false) -> VK.Format

The Vulkan format a Julia element type is stored in. `srgb` is a flag, not a
type, because sRGB and UNORM share a memory layout. Backend-local, like Metal's
`mtlformat`.
"""
function vkformat end

vkformat(::VulkanAPI, ::Type{RGBA{N0f8}}; srgb::Bool = false) =
    srgb ? VK.FORMAT_R8G8B8A8_SRGB : VK.FORMAT_R8G8B8A8_UNORM
vkformat(::VulkanAPI, ::Type{BGRA{N0f8}}; srgb::Bool = false) =
    srgb ? VK.FORMAT_B8G8R8A8_SRGB : VK.FORMAT_B8G8R8A8_UNORM
vkformat(::VulkanAPI, ::Type{RGBA{Float16}}; srgb::Bool = false) =
    srgb ? error("there is no sRGB half-float format") : VK.FORMAT_R16G16B16A16_SFLOAT
vkformat(::VulkanAPI, ::Type{RGBA{Float32}}; srgb::Bool = false) =
    srgb ? error("there is no sRGB float format") : VK.FORMAT_R32G32B32A32_SFLOAT
# Depth-ness is the usage, not the element type: this is D32_SFLOAT because
# nothing else in Vulkan is a single-component 32-bit float attachment.
vkformat(::VulkanAPI, ::Type{Float32}; srgb::Bool = false) = VK.FORMAT_D32_SFLOAT
vkformat(::VulkanAPI, ::Type{UInt32}; srgb::Bool = false) =
    srgb ? error("there is no sRGB integer format") : VK.FORMAT_R32_UINT
vkformat(::VulkanAPI, ::Type{<:StaticVector{2,UInt32}}; srgb::Bool = false) =
    srgb ? error("there is no sRGB integer format") : VK.FORMAT_R32G32_UINT
