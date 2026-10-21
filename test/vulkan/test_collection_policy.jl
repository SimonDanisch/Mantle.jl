using Test, Mantle

function check_pressure_collection(ctx)
    @testset "Vulkan heap pressure uses the pool collection budget" begin
        pool = Mantle.spans(ctx)
        policy, native = pool.policy, Mantle.mempolicy(ctx)
        stats = native.stats
        saved = [getfield(policy, k) for k in fieldnames(Mantle.PoolPolicy)]
        size, updated, enabled = (@atomic stats.size), (@atomic stats.last_updated), native.eager_gc
        buf = nothing
        try
            # Inject a heap-pressure observation, not an invalid allocation or
            # an assertion of actual physical exhaustion. The native buffer is legal.
            buf = Mantle.try_vk_alloc(ctx.default_bq, 4096)
            @test buf isa Mantle.VkManagedBuffer
            native.eager_gc = true
            policy.trim_threshold = typemax(Int)
            policy.gc_mingap = 3600.0
            policy.gc_full_mingap = Inf
            policy.gc_last = 0.0
            policy.soft_cap = 0
            @atomic stats.size = 1
            @atomic stats.last_updated = time()
            before = policy.gc_count
            Mantle.maybe_collect(ctx)
            @test policy.gc_count == before + 1
            @test !Mantle.autocollect!(() -> error("second controller ran"), pool)
            @test policy.gc_count == before + 1
            Mantle.drain_validation_messages!(ctx)
            @test isempty(ctx.validation.messages)
        finally
            if buf isa Mantle.VkManagedBuffer
                Mantle.vk_free!(buf)
                Mantle.drain!(ctx.default_bq)
            end
            @atomic stats.size = size
            @atomic stats.last_updated = updated
            native.eager_gc = enabled
            for (k, value) in zip(fieldnames(Mantle.PoolPolicy), saved)
                setfield!(policy, k, value)
            end
        end
    end
end
