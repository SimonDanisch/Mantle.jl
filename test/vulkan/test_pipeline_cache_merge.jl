using Test, Mantle
include(joinpath(@__DIR__, "..", "testbackend.jl"))

function persisted_pipeline_probe!(out, ::Val{K}) where {K}
    @inbounds out[1] = Float32(K)
    return nothing
end

if TESTBACKEND isa Mantle.LavaDevice
    @testset "an older pipeline-cache writer preserves newer kernels" begin
        dev = Mantle.Device(TESTBACKEND)
        VK = Mantle.VK
        original = dev.pipeline_cache
        ci = VK.PipelineCacheCreateInfo(Ptr{Cvoid}(C_NULL); initial_data_size = UInt64(0))
        older, newer = VK.PipelineCache(dev.device, ci), VK.PipelineCache(dev.device, ci)
        restored = VK.PipelineCache[]
        a = Mantle.devicearray(dev, zeros(Float32, 1))
        value = rand(10_000:99_999)
        try
            dev.pipeline_cache = newer
            Mantle.clear_kernel_cache!(dev)
            Mantle.lava_launch!(dev.default_bq, persisted_pipeline_probe!, a, Val(value);
                               ndrange = 1, workgroup_size = (1, 1, 1))
            @test Array(a) == Float32[value]
            mktempdir() do dir
                path = joinpath(dir, "pipelines.bin")
                snapshot(cache) = (; device_lost = false, pipeline_cache = cache,
                    device = dev.device, physical_device = dev.physical_device)
                Mantle.save_pipeline_cache!(snapshot(newer), path)
                @test Mantle.pipeline_cache_compatible(read(path), dev.physical_device)
                function refused_from_disk()
                    cache = Mantle.create_lava_pipeline_cache(dev.device, path, dev.physical_device)
                    push!(restored, cache)
                    dev.pipeline_cache = cache
                    tt = Mantle.devicetype(dev, (a, Val(value)))
                    Mantle.no_pipeline_compilation(; ctx = dev) do
                        Mantle.get_compiled_kernel_and_pipeline(dev, persisted_pipeline_probe!, tt, (1, 1, 1))
                    end
                    return dev.diag.pipeline_compiles_refused
                end
                # A driver may persist shader IR without persisting executable
                # code. Compare with its own round-trip, rather than requiring
                # a native-binary cache on every Vulkan implementation.
                baseline_refusals = refused_from_disk()
                # Both caches were opened before either save. The older process
                # learned nothing, but must not erase the newer one's compile.
                Mantle.save_pipeline_cache!(snapshot(older), path)
                @test refused_from_disk() <= baseline_refusals
                @test !ispath(path * ".pid")
            end
        finally
            Mantle.waitidle(dev)
            Mantle.clear_kernel_cache!(dev)
            dev.pipeline_cache = original
            Mantle.unsafe_free!(a)
            foreach(finalize, restored)
            finalize(newer)
            finalize(older)
        end
    end
end
