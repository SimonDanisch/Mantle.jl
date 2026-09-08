# A trace pass's shader accesses lower to the ray-tracing stage.
#
# Hikari's hardware trace is a `compute!` pass that holds a `Trace` dispatch and
# declares its queues with `use`. Lowered as plain `Storage`, its barriers named
# COMPUTE|VERTEX|FRAGMENT on both sides, and on a device where ray tracing is a
# stage of its own that is no execution dependency at all: the trace could start
# before the pass that filled its queue finished, and the pass after it could
# read what it was still writing. AMD runs ray tracing as compute, so RADV never
# showed it; NVIDIA gave bit-identical images almost every time and one
# `VK_ERROR_DEVICE_LOST` on the black-hole scene when it did not. `compute!`
# rewrites such a pass's shader usages to `Traced`, whose stage is
# RAY_TRACING_SHADER and whose access and layout are the inner usage's.
using Test, Mantle, Lava
import Vulkan as VK

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

@testset "traced: compute! applies it to a pass that traces" begin
    dev = Mantle.Device(Mantle.VulkanAPI())
    a = Mantle.Buffer(dev, zeros(Float32, 16))
    b = Mantle.Buffer(dev, zeros(Float32, 16))
    g = Mantle.Graph(dev)
    plain = Mantle.compute!(g, "plain") do p
        Mantle.use(p, a; read = true, write = true)
    end
    tracing = Mantle.compute!(g, "tracing") do p
        Mantle.use(p, a; read = true)
        Mantle.use(p, b; write = true, unordered = true)
        # The dispatch alone decides; no shader is compiled here.
        push!(Mantle.dispatches(Mantle.passof(p)), Mantle.Trace(nothing, nothing, (), 1))
    end
    @test all(U -> !(U <: Traced), last.(plain.usages))
    @test last(tracing.usages[1]) === Traced{Storage{BufferKind, ReadOnly}}
    @test last(tracing.usages[2]) === Unordered{Traced{Storage{BufferKind, WriteOnly}}}
end
