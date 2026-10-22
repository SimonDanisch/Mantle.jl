using Test, Mantle, KernelAbstractions

@kernel function novel_scale!(o, ::Val{K}) where {K}
    i = @index(Global)
    o[i] = o[i] * Float32(K)
end

# A constant no cache has seen: the driver keeps its own shader cache across processes.
novelconstant() = Int(time_ns() % 1_000_000) + 2

function novelscale(ctx)
    backend = Mantle.backend(Mantle.lavadevice(ctx))
    o = Mantle.LavaArray{Float32}(undef, (64,); bq = ctx.default_bq)
    fill!(o, 1f0)
    K = novelconstant()
    novel_scale!(backend)(o, Val(K); ndrange = 64)
    KernelAbstractions.synchronize(backend)
    return K, Array(o)
end

function check_pipeline_creation(ctx)
    @testset "no_pipeline_compilation records refusals on its device and still compiles" begin
        K, out = Mantle.no_pipeline_compilation(() -> novelscale(ctx); ctx)
        @test out == fill(Float32(K), 64)
        @test ctx.diag.pipeline_compiles_refused >= 1
        @test any(m -> occursin("spirv=", m), ctx.diag.pipeline_compile_misses)
        @test !ctx.diag.no_pipeline_compile
    end
    others = filter(i -> i.name != ctx.device_name, Mantle.devices())
    isempty(others) && return
    other = Mantle.Device(Mantle.VulkanAPI(); select = first(others).index).ctx
    @testset "validation errors are checked on the calling device" begin
        push!(other.validation.messages, "synthetic validation error")
        @test_throws Mantle.LavaError novelscale(other)
        @test isempty(other.validation.messages)
        push!(ctx.validation.messages, "synthetic validation error")
        K, out = novelscale(other)
        @test out == fill(Float32(K), 64)
        @test_throws Mantle.LavaError Mantle.check_validation_errors!("probe", ctx)
    end
end
