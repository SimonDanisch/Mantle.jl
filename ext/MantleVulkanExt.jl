"""
The Vulkan lowering: the only place in Mantle that names a pipeline stage, an
access mask or an image layout.

Every constant comes from VK.jl rather than being transcribed, so a typo is a
`MethodError` at load rather than a wrong barrier at run time.
"""
module MantleVulkanExt

using Mantle
using Mantle: Vulkan as MantleVulkan
using Mantle: Usage, Storage, ColorAttachment, Depth, Vertices, Indices, Indirect, Uniform,
              Sampled, Present, Undefined, CopySrc, CopyDst, TraceRead, TraceBuild,
              Access, ReadOnly, WriteOnly, ReadWrite, NoAccess, Src, Dst,
              BufferKind, ImageKind, AccelKind, reads, writes, kindof, inner,
              Unordered
import Vulkan as VK
using ColorTypes: RGBA, BGRA
using ColorTypes.FixedPointNumbers: N0f8
import Mantle: stages, access, layout, needs_transition, vkformat

# Vulkan.jl types the `_2_` constants inconsistently: a stage or access whose
# value still fits in 32 bits comes back as the *sync1* flag type, and only the
# ones that overflow are typed sync2. BitMasks refuses to combine the two, so
# every constant is normalised once here rather than at each use. Lava hit the
# same thing for stages (command.jl:114) but not for access.
stage(x) = VK.PipelineStageFlag2(x)
acc(x) = VK.AccessFlag2(x)

const NO_STAGE = stage(0)
"""Where a shader-addressable resource is touched from. Every usage that means "a
shader reads or writes this" lowers to these three and nothing wider."""
const SHADER_STAGES = VK.PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                      VK.PIPELINE_STAGE_2_VERTEX_SHADER_BIT |
                      VK.PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT
const NO_ACCESS = acc(0)

# ── stages ────────────────────────────────────────────────────────────────────
# Pulled, not bound. This backend's pipelines declare an empty
# `VkPipelineVertexInputStateCreateInfo` and the vertex shader loads its
# attributes through a buffer device address, so the read happens in
# VERTEX_SHADER as a storage load — `vkCmdBindVertexBuffers` is never called.
#
# The stage was survivable either way: VERTEX_ATTRIBUTE_INPUT precedes
# VERTEX_SHADER, so an execution dependency ending there also precedes the
# shader. The access mask was not. `VERTEX_ATTRIBUTE_READ` makes a write visible
# to attribute fetch and says nothing about shader storage reads, which is the
# access that actually happens — so on hardware where those are different cache
# domains the vertex shader may read what was there before the compute pass that
# filled it. Nothing reports it: the picture is right until it is not.
stages(::MantleVulkan, ::Type{Vertices}, _) = stage(VK.PIPELINE_STAGE_2_VERTEX_SHADER_BIT)
stages(::MantleVulkan, ::Type{Indices}, _) = stage(VK.PIPELINE_STAGE_2_INDEX_INPUT_BIT)
stages(::MantleVulkan, ::Type{Indirect}, _) = stage(VK.PIPELINE_STAGE_2_DRAW_INDIRECT_BIT)
# Shader reads, both of them, so they name the shader stages like `Storage` does.
# No Mantle API produces either yet — there is no uniform binding and no
# sampled-image binding — but a vocabulary entry that lowers to "everything" is
# the thing this file exists to not have.
stages(::MantleVulkan, ::Type{Uniform}, _) = stage(SHADER_STAGES)
stages(::MantleVulkan, ::Type{Sampled}, _) = stage(SHADER_STAGES)
# The stages a shader-addressable resource is actually touched from, rather than
# every stage there is. This is precision, not speed: measured against
# `ALL_COMMANDS` on the twenty-pass renderer in bench, five runs each in
# separate sessions, the medians were 0.947 ms and 0.954 ms a frame — the same.
# That frame is not barrier bound, and a `timings` column that appears to say
# otherwise is mostly each pass waiting for the one before it to drain.
#
# It is kept because a sync layer whose every buffer barrier says "all commands"
# has stopped deriving anything — and because narrowing it is what exposed the
# aliasing barrier, which had been borrowing this mask for a meaning it does not
# have. That barrier is now derived from what the vacating transient was actually
# doing, so no usage in this file lowers to `ALL_COMMANDS` any more.
stages(::MantleVulkan, ::Type{<:Storage}, _) = stage(SHADER_STAGES)
stages(::MantleVulkan, ::Type{CopySrc}, _) = stage(VK.PIPELINE_STAGE_2_COPY_BIT)
stages(::MantleVulkan, ::Type{CopyDst}, _) = stage(VK.PIPELINE_STAGE_2_COPY_BIT)
stages(::MantleVulkan, ::Type{TraceRead}, _) = stage(VK.PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR)
stages(::MantleVulkan, ::Type{TraceBuild}, _) = stage(VK.PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR)
stages(::MantleVulkan, ::Type{<:ColorAttachment}, _) = stage(VK.PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT)

# Nothing produced the contents, so nothing has to complete first. NONE is the
# sync2 spelling of that and is legal on either side, unlike TOP_OF_PIPE.
stages(::MantleVulkan, ::Type{Undefined}, _) = stage(VK.PIPELINE_STAGE_2_NONE)

# Present as a source is the acquire, which signals at colour attachment output.
# TOP_OF_PIPE would create no execution dependency with it, which is the bug that
# corrupted frames in Lava's window path.
stages(::MantleVulkan, ::Type{Present}, ::Src) = stage(VK.PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT)
# NONE, not BOTTOM_OF_PIPE. Nothing in this submission waits on the transition
# into PRESENT_SRC: the presentation engine is ordered by the render-finished
# semaphore. BOTTOM_OF_PIPE as a destination is the sync1 spelling of the same
# thing and is exactly as meaningless as TOP_OF_PIPE as a source.
stages(::MantleVulkan, ::Type{Present}, ::Dst) = stage(VK.PIPELINE_STAGE_2_NONE)

# Depth writes complete late and reads begin early: one value cannot serve both
# (rps_vk_runtime_backend.cpp:168).
stages(::MantleVulkan, ::Type{<:Depth}, ::Src) =
    stage(VK.PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT)
stages(::MantleVulkan, ::Type{<:Depth}, ::Dst) =
    stage(VK.PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT)

# ── access ────────────────────────────────────────────────────────────────────
# The pair of the stage above, and it has to move with it: `VERTEX_ATTRIBUTE_READ`
# is only legal alongside a vertex-input stage, so leaving it here while the stage
# became VERTEX_SHADER is VUID-VkMemoryBarrier2-srcAccessMask-03902. It is also
# what the read actually is — a storage load through a buffer device address.
access(::MantleVulkan, ::Type{Vertices}, _) = acc(VK.ACCESS_2_SHADER_STORAGE_READ_BIT)
access(::MantleVulkan, ::Type{Indices}, _) = acc(VK.ACCESS_2_INDEX_READ_BIT)
access(::MantleVulkan, ::Type{Indirect}, _) = acc(VK.ACCESS_2_INDIRECT_COMMAND_READ_BIT)
access(::MantleVulkan, ::Type{Uniform}, _) = acc(VK.ACCESS_2_UNIFORM_READ_BIT)
access(::MantleVulkan, ::Type{Sampled}, _) = acc(VK.ACCESS_2_SHADER_SAMPLED_READ_BIT)
access(::MantleVulkan, ::Type{CopySrc}, _) = acc(VK.ACCESS_2_TRANSFER_READ_BIT)
access(::MantleVulkan, ::Type{CopyDst}, _) = acc(VK.ACCESS_2_TRANSFER_WRITE_BIT)
access(::MantleVulkan, ::Type{TraceRead}, _) = acc(VK.ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR)
access(::MantleVulkan, ::Type{TraceBuild}, _) = acc(VK.ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR)
access(::MantleVulkan, ::Type{Present}, _) = NO_ACCESS
access(::MantleVulkan, ::Type{Undefined}, _) = NO_ACCESS

function access(::MantleVulkan, ::Type{Storage{K,A}}, _) where {K,A}
    a = NO_ACCESS
    reads(A) && (a |= acc(VK.ACCESS_2_SHADER_STORAGE_READ_BIT))
    writes(A) && (a |= acc(VK.ACCESS_2_SHADER_STORAGE_WRITE_BIT))
    a
end

# ── unordered ─────────────────────────────────────────────────────────────────
# `Unordered` says two accesses may overlap, which `needs_transition` answers by
# emitting no transition BETWEEN them. It says nothing about what a barrier
# looks like when one is emitted anyway — against an ordinary access, which is
# every clear before the accumulation and every read after it. That barrier is
# the inner usage's: an atomic add is a storage read-modify-write in the compute
# stage whether or not its neighbours had to wait for it.
#
# Without these the vocabulary existed in core and no backend could lower it, so
# the first graph to declare an unordered access died in `build_pass_barrier`
# with a MethodError rather than running faster.
stages(be::MantleVulkan, ::Type{Unordered{U}}, d) where {U} = stages(be, U, d)
access(be::MantleVulkan, ::Type{Unordered{U}}, d) where {U} = access(be, U, d)
layout(be::MantleVulkan, ::Type{Unordered{U}}, d) where {U} = layout(be, U, d)

# A discarding load op means nothing is read back, so the read bit is dropped.
# RPS derives the same from DISCARD_DATA_BEFORE (rps_vk_runtime_backend.cpp:143);
# our design instead scanned recorded draws for blending, which is later and more
# fragile.
access(::MantleVulkan, ::Type{ColorAttachment{Discard}}, _) where {Discard} =
    Discard ? acc(VK.ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT) :
              acc(VK.ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT) | acc(VK.ACCESS_2_COLOR_ATTACHMENT_READ_BIT)

function access(::MantleVulkan, ::Type{Depth{DA,SA,D}}, _) where {DA,SA,D}
    a = NO_ACCESS
    # `D` is the discarding load op, and it drops the read bit for the same reason
    # it does on a colour attachment: nothing before the pass is read back.
    !D && (reads(DA) || reads(SA)) && (a |= acc(VK.ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT))
    (writes(DA) || writes(SA)) && (a |= acc(VK.ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT))
    a
end

# ── layout ────────────────────────────────────────────────────────────────────
layout(::MantleVulkan, ::Type{U}, _) where {U<:Usage} = VK.IMAGE_LAYOUT_UNDEFINED
layout(::MantleVulkan, ::Type{Sampled}, _) = VK.IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
layout(::MantleVulkan, ::Type{CopySrc}, _) = VK.IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL
layout(::MantleVulkan, ::Type{CopyDst}, _) = VK.IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL
layout(::MantleVulkan, ::Type{<:ColorAttachment}, _) = VK.IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
layout(::MantleVulkan, ::Type{<:Storage{ImageKind}}, _) = VK.IMAGE_LAYOUT_GENERAL

# Nothing is preserved out of a presented image, so its source layout is
# UNDEFINED (rps_vk_runtime_backend.cpp:29).
layout(::MantleVulkan, ::Type{Present}, ::Src) = VK.IMAGE_LAYOUT_UNDEFINED
layout(::MantleVulkan, ::Type{Present}, ::Dst) = VK.IMAGE_LAYOUT_PRESENT_SRC_KHR

# Four layouts from four bits. A single write flag reaches only two of them.
layout(::MantleVulkan, ::Type{Depth{DA,SA,D}}, _) where {DA,SA,D} =
    writes(DA) && writes(SA) ? VK.IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL :
    writes(DA) && !writes(SA) ? VK.IMAGE_LAYOUT_DEPTH_ATTACHMENT_STENCIL_READ_ONLY_OPTIMAL :
    !writes(DA) && writes(SA) ? VK.IMAGE_LAYOUT_DEPTH_READ_ONLY_STENCIL_ATTACHMENT_OPTIMAL :
                                VK.IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL

# An image with no stencil aspect is the fifth case, not the depth half of one of
# the four: the separate-aspect layouts above are the `separateDepthStencilLayouts`
# ones, and naming a stencil state for an image that has no stencil is what makes
# them wrong here. The combined layout is legal for a depth-only image on every
# device, feature or no feature.
layout(::MantleVulkan, ::Type{Depth{DA,NoAccess,D}}, _) where {DA,D} =
    writes(DA) ? VK.IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL :
                 VK.IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL

# ── the backend's own hazard rule ─────────────────────────────────────────────
"""
Vulkan needs a transition whenever the portable rule says so, and additionally
whenever the image layout changes, which happens between two *reads* that want
different layouts (sampling then copying from the same image).

RPS delegates exactly this question to its runtime device for copy, clear,
render-pass and depth-stencil accesses (`rps_access_dag_build.hpp:477-487`).
"""
needs_transition(be::MantleVulkan, kind::Mantle.BufferKind, before::Type, after::Type) =
    invoke(needs_transition, Tuple{Mantle.Backend,Mantle.ResourceKind,Type,Type}, be, kind, before, after)

function needs_transition(be::MantleVulkan, kind::ImageKind, before::Type, after::Type)
    invoke(needs_transition, Tuple{Mantle.Backend,Mantle.ResourceKind,Type,Type},
           be, kind, before, after) && return true
    layout(be, inner(before), Src()) != layout(be, inner(after), Dst())
end


# ── formats ───────────────────────────────────────────────────────────────────
# sRGB and UNORM have identical memory and differ only in the hardware's
# conversion on read and write, so it is a flag rather than a different type.
vkformat(::MantleVulkan, ::Type{RGBA{N0f8}}; srgb::Bool = false) =
    srgb ? VK.FORMAT_R8G8B8A8_SRGB : VK.FORMAT_R8G8B8A8_UNORM
vkformat(::MantleVulkan, ::Type{BGRA{N0f8}}; srgb::Bool = false) =
    srgb ? VK.FORMAT_B8G8R8A8_SRGB : VK.FORMAT_B8G8R8A8_UNORM
vkformat(::MantleVulkan, ::Type{RGBA{Float16}}; srgb::Bool = false) =
    srgb ? error("there is no sRGB half-float format") : VK.FORMAT_R16G16B16A16_SFLOAT
vkformat(::MantleVulkan, ::Type{RGBA{Float32}}; srgb::Bool = false) =
    srgb ? error("there is no sRGB float format") : VK.FORMAT_R32G32B32A32_SFLOAT
# Depth-ness is the usage, not the element type: this is D32_SFLOAT because
# nothing else in Vulkan is a single-component 32-bit float attachment.
vkformat(::MantleVulkan, ::Type{Float32}; srgb::Bool = false) = VK.FORMAT_D32_SFLOAT

end