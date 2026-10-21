using Test, Mantle
using ColorTypes: RGBA

function publication_vertex()
    i = Int32(Mantle.vertex_index()) - Int32(1)
    x = Float32((i & Int32(1)) * Int32(4) - Int32(1))
    y = Float32(((i >> Int32(1)) & Int32(1)) * Int32(4) - Int32(1))
    return (position=(x, y, 0f0, 1f0), tint=(0f0, 1f0, 0f0, 1f0))
end
publication_fragment(inputs) = inputs.tint

function check_pipeline_publication(ctx, other=nothing)
    @testset "concurrent native pipeline publication" begin
        dev = Mantle.lavadevice(ctx)
        description = Mantle.GraphicsPipeline(;
            vertex=Mantle.VertexShader(publication_vertex; outputs=(tint=NTuple{4,Float32},)),
            fragment=Mantle.FragmentShader(publication_fragment),
            cull=Mantle.NoCull(), depth=Mantle.DepthOff())
        compile(d) = Mantle.compile_draw(d, description, (RGBA{Float32},), nothing, (), ()).pipeline
        # Compile Julia stages before concurrent native creation. Queue submission
        # itself is task-owned and is deliberately performed on the caller only.
        warm = compile(dev)
        empty!(ctx.caches.gfx_pipelines)
        jobs = [Threads.@spawn compile(dev) for _ in 1:16]
        pipelines = fetch.(jobs)
        @test all(p -> p.pipeline === pipelines[1].pipeline, pipelines)
        @test pipelines[1].pipeline !== warm.pipeline
        @test compile(dev).pipeline === pipelines[1].pipeline
        if other !== nothing
            @test compile(Mantle.lavadevice(other)).pipeline !== pipelines[1].pipeline
        end

        fb = Mantle.Framebuffer(Mantle.backend(dev), 8, 8; depth=false, color_format=RGBA{Float32})
        Mantle.draw!(dev, description, Mantle.OffscreenTarget(fb), 3)
        Mantle.flush!(dev)
        @test all(p -> p.g > 0.99f0 && p.r < 0.01f0, Mantle.readback_framebuffer(fb))
        @test isempty(Mantle.validationmessages!(dev))
    end
end
