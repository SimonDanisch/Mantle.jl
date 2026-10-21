using Test, Mantle

function cache_clear_probe!(a)
    @inbounds a[1] += 1f0
    return nothing
end

function cache_clear_entry(ctx)
    Mantle.get_compiled_kernel_and_pipeline(ctx, cache_clear_probe!,
        Tuple{Mantle.LavaDeviceArray{Float32,1}}, (1, 1, 1))
end

Base.@noinline function submit_cache_probe(ctx, a)
    code, pipeline = cache_clear_entry(ctx)
    # LavaComputePipeline is immutable and may be reboxed. Observe the actual
    # native owner, whose destructor must wait for the accepted submission.
    weak = WeakRef(pipeline.pipeline)
    Mantle.lava_launch!(ctx.default_bq, cache_clear_probe!, a;
                       ndrange=1, workgroup_size=(1, 1, 1))
    return code, weak
end

function check_cache_invalidation(ctx, other=nothing)
    @testset "one compute-cache invalidation preserves code and accepted work" begin
        code, pipeline = cache_clear_entry(ctx)
        @test cache_clear_entry(ctx)[2] === pipeline
        other_pipeline = other === nothing ? nothing : cache_clear_entry(other)[2]
        other === nothing || @test other_pipeline !== pipeline

        a = Mantle.LavaArray(Float32[0]; bq=ctx.default_bq)
        code, weak = submit_cache_probe(ctx, a)
        pipeline = nothing
        dev = Mantle.lavadevice(ctx)
        Mantle.resetkernelcompiles!(dev)
        Mantle.clear_kernel_cache!(ctx)
        GC.gc(true)
        # The submission, not this fixture, must keep the old pipeline alive.
        @test weak.value !== nothing
        rebuilt_code, rebuilt_pipeline = cache_clear_entry(ctx)
        @test rebuilt_code === code
        @test rebuilt_pipeline.pipeline !== weak.value
        @test cache_clear_entry(ctx)[2] === rebuilt_pipeline
        other === nothing || @test cache_clear_entry(other)[2] === other_pipeline
        @test Mantle.kernelcompiles(dev).misses == 0
        Mantle.flush!(ctx.default_bq)
        @test Array(a) == Float32[1]
        Mantle.lava_launch!(ctx.default_bq, cache_clear_probe!, a;
                           ndrange=1, workgroup_size=(1, 1, 1))
        Mantle.flush!(ctx.default_bq)
        @test Array(a) == Float32[2]
        Mantle.unsafe_free!(a)
    end
end
