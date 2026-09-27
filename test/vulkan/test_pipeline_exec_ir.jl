"""
The driver's view of a pipeline (`pipeline_exec_stats`, `pipeline_exec_ir`)
answers for the context it is asked about.

Both queries, and the capture flags on pipeline creation, used to check
`PIPELINE_EXEC_PROPERTIES_REQUESTED[]`. That flag only says the NEXT device will
be created with `VK_KHR_pipeline_executable_properties`. Set after a context
existed, it sent that context's pipelines out with capture flags for an
extension the device never enabled, and a query into a function pointer the
device never resolved. Now each context records what it was created with.

`pipeline_exec_ir` is how RADV's assembly for the implicit-GEMM convolution was
read next to ROCm's `code_gcn` on 2026-09-27; what it returns is checked only
for being there, since every driver names and fills its representations its
own way.
"""

using Test, Mantle, Lava

exec_ir_probe!(out) = (i = Lava.lava_global_invocation_id_x(); @inbounds out[i] = Float32(i); nothing)

@testset "pipeline executable properties belong to a context" begin
    spv = Lava.lava_compile_gpu(exec_ir_probe!, Tuple{Lava.LavaDeviceArray{Float32,1}};
                                workgroup_size = (64, 1, 1))
    old = Mantle.PIPELINE_EXEC_PROPERTIES_REQUESTED[]
    ctx0 = Mantle.vk_context()
    pipe0 = Mantle.get_compute_pipeline(ctx0, spv.spirv_bytes, spv.entry_name)

    # A request made after `ctx0` exists does not reach it.
    if !ctx0.pipeline_exec_props_available
        Mantle.enable_pipeline_executable_properties!()
        try
            @test Mantle.pipeline_exec_stats(ctx0, pipe0) === nothing
            @test Mantle.pipeline_exec_ir(ctx0, pipe0) === nothing
        finally
            Mantle.enable_pipeline_executable_properties!(old)
        end
    end

    # A context created after the request has it, where the device offers it.
    Mantle.enable_pipeline_executable_properties!()
    ctx1 = try
        Mantle.VkContext()
    finally
        Mantle.enable_pipeline_executable_properties!(old)
    end
    try
        offered = Mantle.has_extension(ctx1.physical_device, "VK_KHR_pipeline_executable_properties")
        @test ctx1.pipeline_exec_props_available == offered
        pipe1 = Mantle.get_compute_pipeline(ctx1, spv.spirv_bytes, spv.entry_name)
        if offered
            ir = Mantle.pipeline_exec_ir(ctx1, pipe1)
            @test !isempty(ir)
            # The second call of the two-call idiom filled the text, which
            # Vulkan.jl's own wrapper never makes.
            @test any(r -> !isempty(r.text), ir)
            @test !isempty(Mantle.pipeline_exec_stats(ctx1, pipe1).raw_stats)
        else
            @info "$(ctx1.device_name) does not offer VK_KHR_pipeline_executable_properties"
            @test Mantle.pipeline_exec_ir(ctx1, pipe1) === nothing
        end
    finally
        # Nothing else retires a context built directly, and its handles'
        # finalizers would otherwise run against a torn-down device at exit.
        Mantle.mark_device_lost!(ctx1)
    end
end
