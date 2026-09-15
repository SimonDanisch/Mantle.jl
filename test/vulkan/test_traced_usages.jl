# A trace pass's shader accesses lower to the ray-tracing stage.
#
# Hikari's hardware trace is a `trace!` pass, and its queues are declared from
# what the raygen and the per-material closest-hit shaders do to them. Lowered as
# plain `Storage`, its barriers named
# COMPUTE|VERTEX|FRAGMENT on both sides, and on a device where ray tracing is a
# stage of its own that is no execution dependency at all: the trace could start
# before the pass that filled its queue finished, and the pass after it could
# read what it was still writing. AMD runs ray tracing as compute, so RADV never
# showed it; NVIDIA gave bit-identical images almost every time and one
# `VK_ERROR_DEVICE_LOST` on the black-hole scene when it did not. `compute!`
# rewrites such a pass's shader usages to `Traced`, whose stage is
# RAY_TRACING_SHADER and whose access and layout are the inner usage's. `trace!`
# rewrites every usage of the pass it makes.
using Test, Mantle, Lava, KernelAbstractions
import Vulkan as VK
import Atomix

@kernel function tu_bump!(x)
    i = @index(Global)
    @inbounds x[i] = x[i] + 1f0
end

const RT_STAGE = VK.PipelineStageFlag2(VK.PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR)

@testset "traced: which usages a trace pass changes" begin
    @test Mantle.traced(Storage{BufferKind, ReadWrite}) === Traced{Storage{BufferKind, ReadWrite}}
    @test Mantle.traced(Mantle.Uniform) === Traced{Mantle.Uniform}
    @test Mantle.traced(Mantle.Sampled) === Traced{Mantle.Sampled}
    # Unordered stays outermost, so the hazard rule still sees it.
    @test Mantle.traced(Unordered{Storage{BufferKind, ReadWrite}}) ===
          Unordered{Traced{Storage{BufferKind, ReadWrite}}}
    @test Mantle.unordered(Unordered{Traced{Storage{BufferKind, ReadWrite}}})
    # What the command processor or a copy engine reads is not a shader access.
    for U in (Mantle.Indirect, Mantle.Predicated, TraceRead, TraceBuild, CopySrc, CopyDst)
        @test Mantle.traced(U) === U
    end
    # The portable properties are the inner usage's.
    T = Traced{Storage{BufferKind, ReadOnly}}
    @test Mantle.reads(T) && !Mantle.writes(T)
    @test Mantle.kindof(T) === BufferKind()
end

@testset "traced: the Vulkan lowering" begin
    be = Mantle.VulkanAPI()
    T = Traced{Storage{BufferKind, ReadWrite}}
    S = Storage{BufferKind, ReadWrite}
    for d in (Mantle.Src(), Mantle.Dst())
        @test (Mantle.stages(be, T, d) & RT_STAGE) == RT_STAGE
        @test (Mantle.stages(be, S, d) & RT_STAGE) == VK.PipelineStageFlag2(0)
        @test Mantle.access(be, T, d) == Mantle.access(be, S, d)
    end
end

# The shaders below are never compiled. `trace!` infers what a trace touches from
# the same typed IR the emitter would see, and inference needs a function it can
# follow — not a valid SPIR-V entry point. Keeping them trivial is what makes
# this a unit test of the DECLARATION rather than of the ray-tracing stack.
tu_raygen(src, dst) = (@inbounds dst[1] = src[1]; nothing)
tu_chit(src, dst) = (@inbounds Atomix.@atomic dst[1] += src[1]; nothing)
tu_miss(src, dst) = nothing

@testset "traced: trace! applies it to the pass it makes" begin
    dev = Mantle.Device(Mantle.VulkanAPI())
    a = Mantle.Buffer(dev, zeros(Float32, 16))
    b = Mantle.Buffer(dev, zeros(Float32, 16))
    g = Mantle.Graph(dev)
    plain = Mantle.dispatch!(g, tu_bump!, (a,), 16; name = "plain")
    pipe = Mantle.RayTracingPipeline(; raygen = tu_raygen, closest_hit = tu_chit,
                                       miss = tu_miss, chit_miss_take_args = true)
    tracing = Mantle.trace!(g, pipe, nothing, (a, b), 16)

    @test all(U -> !(U <: Traced), last.(plain.usages))
    byid = Dict(id => U for (id, U) in tracing.usages)
    # The raygen reads `a` and writes `b`; the closest-hit adds into `b`
    # atomically, and an ordinary write beside an atomic one is still ordered.
    @test byid[Mantle.resourceid(g, a)] === Traced{Storage{BufferKind, ReadOnly}}
    @test byid[Mantle.resourceid(g, b)] === Traced{Storage{BufferKind, ReadWrite}}
end

@testset "traced: a trace target only ever added into is unordered" begin
    dev = Mantle.Device(Mantle.VulkanAPI())
    a = Mantle.Buffer(dev, zeros(Float32, 16))
    b = Mantle.Buffer(dev, zeros(Float32, 16))
    g = Mantle.Graph(dev)
    pipe = Mantle.RayTracingPipeline(; raygen = tu_chit, closest_hit = tu_chit,
                                       miss = tu_miss, chit_miss_take_args = true)
    tracing = Mantle.trace!(g, pipe, nothing, (a, b), 16)
    byid = Dict(id => U for (id, U) in tracing.usages)
    @test byid[Mantle.resourceid(g, b)] ===
          Unordered{Traced{Storage{BufferKind, ReadWrite}}}
end
