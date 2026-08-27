# test_handwritten_spirv.jl
#
# End-to-end test: Generate a compute shader with Lava's SPIR-V module builder,
# validate it, run it on the GPU via Vulkan.jl, and verify the output.
#
# Shader: output[global_id] = input[global_id] * 2.0
# Uses descriptor sets (StorageBuffer bindings) for buffer access.

using Lava, Mantle
using Vulkan
using Test

# =====================================================================
# SPIR-V Shader Generation
# =====================================================================

"""
    build_times2_shader(; local_size=64) -> Vector{UInt8}

Generate SPIR-V binary for a compute shader that doubles every element.
Layout:
  - Binding 0: input StorageBuffer (readonly, runtime array of f32)
  - Binding 1: output StorageBuffer (runtime array of f32)
  - GlobalInvocationId.x used for indexing
"""
function build_times2_shader(; local_size=64)
    mod = Lava.SPIRVModule()

    # Capabilities & memory model
    Lava.require_capability!(mod, Lava.Cap.Shader)
    Lava.setup_memory_model!(mod; physical_storage_buffer=false)

    # Types
    void_t  = Lava.emit_type_void!(mod)
    f32_t   = Lava.emit_type_float!(mod, UInt32(32))
    u32_t   = Lava.emit_type_int!(mod, UInt32(32), UInt32(0))
    uvec3_t = Lava.emit_type_vector!(mod, u32_t, UInt32(3))

    # OpTypeRuntimeArray (opcode 29) — not yet in builder API
    rta_f32_t = Lava.fresh_id!(mod)
    Lava.encode_instruction!(mod.types_constants, UInt16(29), rta_f32_t, f32_t)
    Lava.emit_decorate!(mod, rta_f32_t, Lava.Dec.ArrayStride, UInt32(4))

    # Input struct { float data[]; } — Block decorated
    input_struct_t = Lava.emit_type_struct!(mod, UInt32[rta_f32_t])
    Lava.emit_decorate!(mod, input_struct_t, Lava.Dec.Block)
    Lava.emit_member_decorate!(mod, input_struct_t, UInt32(0), Lava.Dec.Offset, UInt32(0))

    # Output struct — manually emitted to avoid type deduplication with input struct
    output_struct_t = Lava.fresh_id!(mod)
    push!(mod.types_constants, (UInt32(3) << 16) | UInt32(Lava.Op.OpTypeStruct))
    push!(mod.types_constants, output_struct_t)
    push!(mod.types_constants, rta_f32_t)
    Lava.emit_decorate!(mod, output_struct_t, Lava.Dec.Block)
    Lava.emit_member_decorate!(mod, output_struct_t, UInt32(0), Lava.Dec.Offset, UInt32(0))

    # Pointer types
    ptr_input_t    = Lava.emit_type_pointer!(mod, Lava.SC.StorageBuffer, input_struct_t)
    ptr_output_t   = Lava.emit_type_pointer!(mod, Lava.SC.StorageBuffer, output_struct_t)
    ptr_f32_sb_t   = Lava.emit_type_pointer!(mod, Lava.SC.StorageBuffer, f32_t)
    ptr_uvec3_in_t = Lava.emit_type_pointer!(mod, Lava.SC.Input, uvec3_t)
    func_t         = Lava.emit_type_function!(mod, void_t)

    # Constants
    const_0_u32 = Lava.emit_constant_u32!(mod, UInt32(0))
    const_2_f32 = Lava.emit_constant_f32!(mod, Float32(2.0))

    # Global variables
    input_var = Lava.emit_global_variable!(mod, ptr_input_t, Lava.SC.StorageBuffer)
    Lava.emit_decorate!(mod, input_var, Lava.Dec.DescriptorSet, UInt32(0))
    Lava.emit_decorate!(mod, input_var, Lava.Dec.Binding, UInt32(0))
    Lava.emit_decorate!(mod, input_var, Lava.Dec.NonWritable)

    output_var = Lava.emit_global_variable!(mod, ptr_output_t, Lava.SC.StorageBuffer)
    Lava.emit_decorate!(mod, output_var, Lava.Dec.DescriptorSet, UInt32(0))
    Lava.emit_decorate!(mod, output_var, Lava.Dec.Binding, UInt32(1))

    gl_global_id = Lava.emit_global_variable!(mod, ptr_uvec3_in_t, Lava.SC.Input)
    Lava.emit_decorate!(mod, gl_global_id, Lava.Dec.BuiltIn, Lava.BuiltIn.GlobalInvocationId)

    # Entry point & execution mode
    func_id = Lava.fresh_id!(mod)
    Lava.emit_entry_point!(mod, Lava.ExecModel.GLCompute, func_id, "main",
        UInt32[gl_global_id, input_var, output_var])
    Lava.emit_execution_mode!(mod, func_id, Lava.ExecMode.LocalSize,
        UInt32(local_size), UInt32(1), UInt32(1))

    # Function body
    label_id = Lava.fresh_id!(mod)
    Lava.encode_instruction!(mod.functions, Lava.Op.OpFunction,
        void_t, func_id, Lava.FuncControl.None, func_t)
    Lava.encode_instruction!(mod.functions, Lava.Op.OpLabel, label_id)

    # %gid_vec = OpLoad %v3uint %gl_GlobalInvocationID
    gid_vec = Lava.fresh_id!(mod)
    Lava.encode_instruction!(mod.functions, Lava.Op.OpLoad, uvec3_t, gid_vec, gl_global_id)

    # %gid_x = OpCompositeExtract %uint %gid_vec 0
    gid_x = Lava.fresh_id!(mod)
    Lava.encode_instruction!(mod.functions, UInt16(81), u32_t, gid_x, gid_vec, UInt32(0))

    # %input_ptr = OpAccessChain %ptr_f32_sb %input_var %0 %gid_x
    input_elem_ptr = Lava.fresh_id!(mod)
    Lava.encode_instruction!(mod.functions, Lava.Op.OpAccessChain,
        ptr_f32_sb_t, input_elem_ptr, input_var, const_0_u32, gid_x)

    # %val = OpLoad %float %input_ptr
    input_val = Lava.fresh_id!(mod)
    Lava.encode_instruction!(mod.functions, Lava.Op.OpLoad, f32_t, input_val, input_elem_ptr)

    # %result = OpFMul %float %val %const_2
    result_val = Lava.fresh_id!(mod)
    Lava.encode_instruction!(mod.functions, Lava.Op.OpFMul, f32_t, result_val, input_val, const_2_f32)

    # %output_ptr = OpAccessChain %ptr_f32_sb %output_var %0 %gid_x
    output_elem_ptr = Lava.fresh_id!(mod)
    Lava.encode_instruction!(mod.functions, Lava.Op.OpAccessChain,
        ptr_f32_sb_t, output_elem_ptr, output_var, const_0_u32, gid_x)

    # OpStore %output_ptr %result
    Lava.encode_instruction!(mod.functions, Lava.Op.OpStore, output_elem_ptr, result_val)

    Lava.encode_instruction!(mod.functions, Lava.Op.OpReturn)
    Lava.encode_instruction!(mod.functions, Lava.Op.OpFunctionEnd)

    # Debug names
    Lava.emit_name!(mod, func_id, "main")
    Lava.emit_name!(mod, input_var, "input_buf")
    Lava.emit_name!(mod, output_var, "output_buf")
    Lava.emit_name!(mod, gl_global_id, "gl_GlobalInvocationID")

    return Lava.serialize(mod)
end

# =====================================================================
# Vulkan Runtime Helpers
# =====================================================================

"""Find first memory type index matching required property flags."""
function find_memory_type(phys_device, type_bits::UInt32, required_flags)
    mem_props = get_physical_device_memory_properties(phys_device)
    # `length(memory_types)`, not `memory_type_count`: Vulkan.jl's high-level
    # struct returns the vector already truncated to the valid count and no
    # longer carries the count field. Same read `Mantle.find_memory_type` does.
    for i in 0:(length(mem_props.memory_types) - 1)
        if (type_bits >> i) & 1 == 1
            mt = mem_props.memory_types[i + 1]
            if required_flags in mt.property_flags
                return i
            end
        end
    end
    error("No suitable memory type found for type_bits=$type_bits, flags=$required_flags")
end

"""Create a buffer, allocate memory for it, and bind them together."""
function create_bound_buffer(device, phys_device, size, usage, mem_flags)
    buf = unwrap(create_buffer(device, size, usage, SHARING_MODE_EXCLUSIVE, UInt32[]))
    req = get_buffer_memory_requirements(device, buf)
    mem_type = find_memory_type(phys_device, req.memory_type_bits, mem_flags)
    mem = unwrap(allocate_memory(device, req.size, mem_type))
    unwrap(bind_buffer_memory(device, buf, mem, 0))
    return buf, mem
end

# =====================================================================
# Tests
# =====================================================================

@testset "Handwritten SPIR-V Compute Shader" begin

    # ---- 1. Generate and validate SPIR-V ----
    @testset "SPIR-V Generation & Validation" begin
        spirv_binary = build_times2_shader()
        @test length(spirv_binary) > 0
        @test length(spirv_binary) % 4 == 0  # must be UInt32 aligned

        # Check magic number
        magic = reinterpret(UInt32, spirv_binary[1:4])[1]
        @test magic == 0x07230203

        # Validate with spirv-val
        @test_nowarn Lava.validate_spirv(spirv_binary)

        # Disassemble and check key instructions
        disasm = Lava.disassemble_spirv(spirv_binary)
        @test occursin("OpEntryPoint GLCompute", disasm)
        @test occursin("OpExecutionMode %main LocalSize 64 1 1", disasm)
        @test occursin("OpFMul", disasm)
        @test occursin("BuiltIn GlobalInvocationId", disasm)
        @test occursin("StorageBuffer", disasm)
    end

    # ---- 2. Run on GPU and verify results ----
    @testset "GPU Execution (times 2)" begin
        local_size = 64
        N = 1024
        buf_size = N * sizeof(Float32)

        spirv_binary = build_times2_shader(; local_size)

        # Create Vulkan instance
        instance = unwrap(create_instance(String[], String[];
            application_info = ApplicationInfo(v"0.1.0", v"0.1.0", v"1.3";
                application_name="Lava Test", engine_name="Lava.jl")))

        # Select first physical device
        phys_devices = unwrap(enumerate_physical_devices(instance))
        @test length(phys_devices) >= 1
        phys_device = phys_devices[1]

        # Find compute queue family
        qf_props = get_physical_device_queue_family_properties(phys_device)
        compute_qf = UInt32(0)
        for (i, qf) in enumerate(qf_props)
            if QUEUE_COMPUTE_BIT in qf.queue_flags
                compute_qf = UInt32(i - 1)
                break
            end
        end

        # Create logical device
        device = unwrap(create_device(phys_device,
            [DeviceQueueCreateInfo(compute_qf, [1.0f0])],
            String[], String[]))
        queue = get_device_queue(device, compute_qf, 0)

        # Required memory flags: host-visible + host-coherent for easy CPU access
        required_mem_flags = MEMORY_PROPERTY_HOST_VISIBLE_BIT | MEMORY_PROPERTY_HOST_COHERENT_BIT

        # Create buffers
        input_buf, input_mem = create_bound_buffer(device, phys_device, buf_size,
            BUFFER_USAGE_STORAGE_BUFFER_BIT, required_mem_flags)
        output_buf, output_mem = create_bound_buffer(device, phys_device, buf_size,
            BUFFER_USAGE_STORAGE_BUFFER_BIT, required_mem_flags)

        # Write input data: [1.0, 2.0, ..., N]
        input_data = Float32.(1:N)
        ptr = unwrap(map_memory(device, input_mem, 0, buf_size))
        unsafe_copyto!(Ptr{Float32}(ptr), pointer(input_data), N)
        unmap_memory(device, input_mem)

        # Zero output buffer
        ptr = unwrap(map_memory(device, output_mem, 0, buf_size))
        for i in 1:buf_size
            unsafe_store!(Ptr{UInt8}(ptr), 0x00, i)
        end
        unmap_memory(device, output_mem)

        # Create shader module
        spirv_words = collect(reinterpret(UInt32, spirv_binary))
        shader_module = unwrap(create_shader_module(device, length(spirv_binary), spirv_words))

        # Descriptor set layout: 2 storage buffers
        ds_layout = unwrap(create_descriptor_set_layout(device, [
            DescriptorSetLayoutBinding(UInt32(0), DESCRIPTOR_TYPE_STORAGE_BUFFER,
                SHADER_STAGE_COMPUTE_BIT; descriptor_count=1),
            DescriptorSetLayoutBinding(UInt32(1), DESCRIPTOR_TYPE_STORAGE_BUFFER,
                SHADER_STAGE_COMPUTE_BIT; descriptor_count=1),
        ]))

        # Pipeline layout (no push constants)
        pipeline_layout = unwrap(create_pipeline_layout(device, [ds_layout], PushConstantRange[]))

        # Compute pipeline
        pipelines = unwrap(create_compute_pipelines(device, [
            ComputePipelineCreateInfo(
                PipelineShaderStageCreateInfo(SHADER_STAGE_COMPUTE_BIT, shader_module, "main"),
                pipeline_layout, -1)
        ]))
        compute_pipeline = pipelines[1][1]

        # Descriptor pool + set
        desc_pool = unwrap(create_descriptor_pool(device, UInt32(1),
            [DescriptorPoolSize(DESCRIPTOR_TYPE_STORAGE_BUFFER, UInt32(2))]))
        desc_set = unwrap(allocate_descriptor_sets(device,
            DescriptorSetAllocateInfo(desc_pool, [ds_layout])))[1]

        # Bind buffers to descriptors
        update_descriptor_sets(device, [
            WriteDescriptorSet(desc_set, UInt32(0), UInt32(0), DESCRIPTOR_TYPE_STORAGE_BUFFER,
                DescriptorImageInfo[], [DescriptorBufferInfo(input_buf, 0, buf_size)], BufferView[]),
            WriteDescriptorSet(desc_set, UInt32(1), UInt32(0), DESCRIPTOR_TYPE_STORAGE_BUFFER,
                DescriptorImageInfo[], [DescriptorBufferInfo(output_buf, 0, buf_size)], BufferView[]),
        ], [])

        # Command buffer
        cmd_pool = unwrap(create_command_pool(device, compute_qf;
            flags=COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT))
        cmd_buf = unwrap(allocate_command_buffers(device,
            CommandBufferAllocateInfo(cmd_pool, COMMAND_BUFFER_LEVEL_PRIMARY, 1)))[1]

        # Record commands
        unwrap(begin_command_buffer(cmd_buf, CommandBufferBeginInfo()))
        cmd_bind_pipeline(cmd_buf, PIPELINE_BIND_POINT_COMPUTE, compute_pipeline)
        cmd_bind_descriptor_sets(cmd_buf, PIPELINE_BIND_POINT_COMPUTE,
            pipeline_layout, 0, [desc_set], UInt32[])
        cmd_dispatch(cmd_buf, UInt32(cld(N, local_size)), UInt32(1), UInt32(1))
        unwrap(end_command_buffer(cmd_buf))

        # Submit and wait
        fence = unwrap(create_fence(device, FenceCreateInfo()))
        unwrap(queue_submit(queue, [SubmitInfo([], [], [cmd_buf], [])]; fence))
        unwrap(wait_for_fences(device, [fence], true, typemax(UInt64)))

        # Read back results
        ptr = unwrap(map_memory(device, output_mem, 0, buf_size))
        output_data = Vector{Float32}(undef, N)
        unsafe_copyto!(pointer(output_data), Ptr{Float32}(ptr), N)
        unmap_memory(device, output_mem)

        # Verify correctness
        expected = input_data .* 2.0f0
        @test output_data == expected

        # Cleanup — use finalize() so GC finalizers don't double-free handles.
        # Order: wait for GPU idle, then children before parents.
        device_wait_idle(device)
        finalize(fence)
        finalize(cmd_pool)
        finalize(desc_pool)
        finalize(compute_pipeline)
        finalize(pipeline_layout)
        finalize(ds_layout)
        finalize(shader_module)
        finalize(input_buf)
        finalize(output_buf)
        finalize(input_mem)
        finalize(output_mem)
        finalize(device)
        finalize(instance)
    end
end
