# test_handwritten_rt.jl
#
# End-to-end test: Handwritten SPIR-V RT shaders (raygen, closest-hit, miss),
# built using Lava's SPIR-V module builder, validated with spirv-val, and
# dispatched against a single triangle via VK_KHR_ray_tracing_pipeline.
#
# Triangle: (0,0,0), (1,0,0), (0,1,0) in the XY plane at z=0.
# Rays: 8x8 grid from (-0.5..1.5, -0.5..1.5) at z=-1, direction (0,0,1).
# Expected: rays hitting the triangle get hit_t=1.0, others get -1.0 (miss).

using Lava, Mantle
using Vulkan
using Test

# The hand-written pipeline's dispatch, as the unmodelled path spells it today:
# one one-shot holding the trace, the acceleration structures it reads pinned
# by core's `pintrace!`, submitted when it closes. `rt_dispatch!` was the name
# of the open-batch version.
function rt_dispatch!(bq, pipeline, tlas, push_bda, W, H)
    MVE.oneshot!(bq; tag = :trace) do e
        Mantle.pintrace!(e, tlas)
        MVE.emit_trace!(e, pipeline, tlas, push_bda, W, H, 1)
    end
    return nothing
end

# =====================================================================
# SPIR-V Shader Builders
# =====================================================================

"""Build a raygen shader that traces rays and writes hit_t to an output buffer.

Layout:
  - Descriptor set 0, binding 0: AccelerationStructure (HWTLAS)
  - Push constant: u64 BDA pointer to output buffer (float32 array)
  - LaunchIdKHR.xy determines ray origin
  - LaunchSizeKHR.xy determines grid size
  - Payload location 0: single float32 (hit_t or -1.0 for miss)
"""
function build_raygen_shader()
    mod = Lava.SPIRVModule()

    Lava.require_capability!(mod, Lava.Cap.Shader)
    Lava.require_capability!(mod, Lava.Cap.RayTracingKHR)
    Lava.require_capability!(mod, Lava.Cap.Int64)
    Lava.require_extension!(mod, "SPV_KHR_ray_tracing")
    Lava.setup_memory_model!(mod; physical_storage_buffer=true)

    # Types
    void_t  = Lava.emit_type_void!(mod)
    f32_t   = Lava.emit_type_float!(mod, UInt32(32))
    u32_t   = Lava.emit_type_int!(mod, UInt32(32), UInt32(0))
    i32_t   = Lava.emit_type_int!(mod, UInt32(32), UInt32(1))
    u64_t   = Lava.emit_type_int!(mod, UInt32(64), UInt32(0))
    vec3_t  = Lava.emit_type_vector!(mod, f32_t, UInt32(3))
    uvec3_t = Lava.emit_type_vector!(mod, u32_t, UInt32(3))
    func_t  = Lava.emit_type_function!(mod, void_t)

    # Acceleration structure type
    as_t = Lava.emit_type_acceleration_structure!(mod)

    # Pointer types
    ptr_as_uc_t     = Lava.emit_type_pointer!(mod, Lava.SC.UniformConstant, as_t)
    ptr_uvec3_in_t  = Lava.emit_type_pointer!(mod, Lava.SC.Input, uvec3_t)
    ptr_f32_pay_t   = Lava.emit_type_pointer!(mod, Lava.SC.RayPayloadKHR, f32_t)
    ptr_f32_psb_t   = Lava.emit_type_pointer!(mod, Lava.SC.PhysicalStorageBuffer, f32_t)
    ptr_u64_pc_t    = Lava.emit_type_pointer!(mod, Lava.SC.PushConstant, u64_t)

    # Push constant struct: { u64 output_bda }
    pc_struct_t = Lava.emit_type_struct!(mod, UInt32[u64_t])
    Lava.emit_decorate!(mod, pc_struct_t, Lava.Dec.Block)
    Lava.emit_member_decorate!(mod, pc_struct_t, UInt32(0), Lava.Dec.Offset, UInt32(0))
    ptr_pc_t = Lava.emit_type_pointer!(mod, Lava.SC.PushConstant, pc_struct_t)

    # Constants
    c_0u = Lava.emit_constant_u32!(mod, UInt32(0))
    c_1u = Lava.emit_constant_u32!(mod, UInt32(1))
    c_0f = Lava.emit_constant_f32!(mod, 0f0)
    c_m1f = Lava.emit_constant_f32!(mod, -1f0)
    c_neg_half = Lava.emit_constant_f32!(mod, -0.5f0)
    c_2f = Lava.emit_constant_f32!(mod, 2f0)
    c_0_0_1 = Lava.fresh_id!(mod)  # vec3(0,0,1) direction
    Lava.encode_instruction!(mod.types_constants, Lava.Op.OpConstantComposite,
        vec3_t, c_0_0_1, c_0f, c_0f, Lava.emit_constant_f32!(mod, 1f0))

    c_tmin = Lava.emit_constant_f32!(mod, 0.001f0)
    c_tmax = Lava.emit_constant_f32!(mod, 100f0)
    c_ff = Lava.emit_constant_u32!(mod, UInt32(0xff))  # cull mask
    c_ray_flags = Lava.emit_constant_u32!(mod, UInt32(0))  # no flags
    c_4u = Lava.emit_constant_u32!(mod, UInt32(4))  # sizeof(f32)
    c_0u64 = Lava.emit_constant_u64!(mod, UInt64(0))

    # i32 constant for sbt offset (needs signed)
    c_0i = Lava.fresh_id!(mod)
    Lava.encode_instruction!(mod.types_constants, Lava.Op.OpConstant, i32_t, c_0i, UInt32(0))

    # Global variables
    as_var = Lava.emit_global_variable!(mod, ptr_as_uc_t, Lava.SC.UniformConstant)
    Lava.emit_decorate!(mod, as_var, Lava.Dec.DescriptorSet, UInt32(0))
    Lava.emit_decorate!(mod, as_var, Lava.Dec.Binding, UInt32(0))

    launch_id_var = Lava.emit_global_variable!(mod, ptr_uvec3_in_t, Lava.SC.Input)
    Lava.emit_decorate!(mod, launch_id_var, Lava.Dec.BuiltIn, Lava.BuiltIn.LaunchIdKHR)

    launch_size_var = Lava.emit_global_variable!(mod, ptr_uvec3_in_t, Lava.SC.Input)
    Lava.emit_decorate!(mod, launch_size_var, Lava.Dec.BuiltIn, Lava.BuiltIn.LaunchSizeKHR)

    # Payload variable (RayPayloadKHR storage class, single f32).
    # Intentionally NO `Location` decoration: post-revision 16 of
    # SPV_KHR_ray_tracing, ray payloads are matched by *pointer* (the
    # Payload operand of OpTraceRayKHR is a pointer to the RayPayloadKHR
    # OpVariable), not by integer location. Adding `Location` here is legal
    # per spirv-val but triggers a warning in Mesa's spirv_to_nir validator
    # (vtn_variables.c) whose allow-list was never updated for the ray-
    # tracing storage classes.
    payload_var = Lava.emit_global_variable!(mod, ptr_f32_pay_t, Lava.SC.RayPayloadKHR)

    # Push constant variable
    pc_var = Lava.emit_global_variable!(mod, ptr_pc_t, Lava.SC.PushConstant)

    # Entry point
    func_id = Lava.fresh_id!(mod)
    Lava.emit_entry_point!(mod, Lava.ExecModel.RayGenerationKHR, func_id, "main",
        UInt32[as_var, launch_id_var, launch_size_var, payload_var, pc_var])

    # Function body
    label_id = Lava.fresh_id!(mod)
    Lava.encode_instruction!(mod.functions, Lava.Op.OpFunction,
        void_t, func_id, Lava.FuncControl.None, func_t)
    Lava.encode_instruction!(mod.functions, Lava.Op.OpLabel, label_id)

    # Load launch ID (uvec3)
    launch_id = Lava.fresh_id!(mod)
    Lava.encode_instruction!(mod.functions, Lava.Op.OpLoad, uvec3_t, launch_id, launch_id_var)

    # Extract x, y
    lid_x = Lava.fresh_id!(mod)
    Lava.encode_instruction!(mod.functions, UInt16(81), u32_t, lid_x, launch_id, UInt32(0))  # OpCompositeExtract
    lid_y = Lava.fresh_id!(mod)
    Lava.encode_instruction!(mod.functions, UInt16(81), u32_t, lid_y, launch_id, UInt32(1))

    # Load launch size
    launch_size = Lava.fresh_id!(mod)
    Lava.encode_instruction!(mod.functions, Lava.Op.OpLoad, uvec3_t, launch_size, launch_size_var)
    lsz_x = Lava.fresh_id!(mod)
    Lava.encode_instruction!(mod.functions, UInt16(81), u32_t, lsz_x, launch_size, UInt32(0))

    # Convert to float: ray_x = (float(lid_x) / float(lsz_x)) * 2.0 - 0.5
    # This maps [0, lsz_x) to [-0.5, 1.5) to cover the triangle and some misses
    fid_x = Lava.fresh_id!(mod)
    Lava.encode_instruction!(mod.functions, Lava.Op.OpConvertUToF, f32_t, fid_x, lid_x)
    fid_y = Lava.fresh_id!(mod)
    Lava.encode_instruction!(mod.functions, Lava.Op.OpConvertUToF, f32_t, fid_y, lid_y)
    fsz_x = Lava.fresh_id!(mod)
    Lava.encode_instruction!(mod.functions, Lava.Op.OpConvertUToF, f32_t, fsz_x, lsz_x)

    # ray_x = lid_x / lsz_x * 2.0 - 0.5
    norm_x = Lava.fresh_id!(mod)
    Lava.encode_instruction!(mod.functions, Lava.Op.OpFDiv, f32_t, norm_x, fid_x, fsz_x)
    scaled_x = Lava.fresh_id!(mod)
    Lava.encode_instruction!(mod.functions, Lava.Op.OpFMul, f32_t, scaled_x, norm_x, c_2f)
    ray_x = Lava.fresh_id!(mod)
    Lava.encode_instruction!(mod.functions, Lava.Op.OpFAdd, f32_t, ray_x, scaled_x, c_neg_half)

    # ray_y = lid_y / lsz_x * 2.0 - 0.5 (same scale for both axes)
    norm_y = Lava.fresh_id!(mod)
    Lava.encode_instruction!(mod.functions, Lava.Op.OpFDiv, f32_t, norm_y, fid_y, fsz_x)
    scaled_y = Lava.fresh_id!(mod)
    Lava.encode_instruction!(mod.functions, Lava.Op.OpFMul, f32_t, scaled_y, norm_y, c_2f)
    ray_y = Lava.fresh_id!(mod)
    Lava.encode_instruction!(mod.functions, Lava.Op.OpFAdd, f32_t, ray_y, scaled_y, c_neg_half)

    # origin = vec3(ray_x, ray_y, -1.0)
    origin = Lava.fresh_id!(mod)
    Lava.encode_instruction!(mod.functions, UInt16(80), vec3_t, origin,
        ray_x, ray_y, c_m1f)  # OpCompositeConstruct

    # Load acceleration structure
    as_val = Lava.fresh_id!(mod)
    Lava.encode_instruction!(mod.functions, Lava.Op.OpLoad, as_t, as_val, as_var)

    # Initialize payload to -1.0 (miss marker)
    Lava.encode_instruction!(mod.functions, Lava.Op.OpStore, payload_var, c_m1f)

    # OpTraceRayKHR %as %ray_flags %cull_mask %sbt_offset %sbt_stride %miss_index
    #               %origin %tmin %direction %tmax %payload
    # OpTraceRayKHR has 11 operands, no result
    Lava.encode_instruction!(mod.functions, Lava.Op.OpTraceRayKHR,
        as_val,       # acceleration structure
        c_ray_flags,  # ray flags
        c_ff,         # cull mask (0xFF)
        c_0u,         # SBT record offset
        c_0u,         # SBT record stride
        c_0u,         # miss index
        origin,       # ray origin (vec3)
        c_tmin,       # tmin
        c_0_0_1,      # ray direction (vec3)
        c_tmax,       # tmax
        payload_var)  # payload (location 0)

    # Load payload result (hit_t or -1.0)
    payload_val = Lava.fresh_id!(mod)
    Lava.encode_instruction!(mod.functions, Lava.Op.OpLoad, f32_t, payload_val, payload_var)

    # Compute output buffer index: lid_y * lsz_x + lid_x
    idx = Lava.fresh_id!(mod)
    Lava.encode_instruction!(mod.functions, Lava.Op.OpIMul, u32_t, idx, lid_y, lsz_x)
    idx2 = Lava.fresh_id!(mod)
    Lava.encode_instruction!(mod.functions, Lava.Op.OpIAdd, u32_t, idx2, idx, lid_x)

    # Convert index to byte offset: idx * 4
    byte_off = Lava.fresh_id!(mod)
    Lava.encode_instruction!(mod.functions, Lava.Op.OpIMul, u32_t, byte_off, idx2, c_4u)
    byte_off64 = Lava.fresh_id!(mod)
    Lava.encode_instruction!(mod.functions, Lava.Op.OpUConvert, u64_t, byte_off64, byte_off)

    # Load BDA from push constant
    pc_ptr = Lava.fresh_id!(mod)
    Lava.encode_instruction!(mod.functions, Lava.Op.OpAccessChain,
        ptr_u64_pc_t, pc_ptr, pc_var, c_0u)
    bda = Lava.fresh_id!(mod)
    Lava.encode_instruction!(mod.functions, Lava.Op.OpLoad, u64_t, bda, pc_ptr)

    # Compute store address: bda + byte_off
    store_addr = Lava.fresh_id!(mod)
    Lava.encode_instruction!(mod.functions, Lava.Op.OpIAdd, u64_t, store_addr, bda, byte_off64)

    # Convert to pointer and store
    store_ptr = Lava.fresh_id!(mod)
    Lava.encode_instruction!(mod.functions, Lava.Op.OpConvertUToPtr, ptr_f32_psb_t, store_ptr, store_addr)
    # OpStore with Aligned memory access (required for PSB)
    Lava.encode_instruction!(mod.functions, Lava.Op.OpStore, store_ptr, payload_val,
        UInt32(0x02), UInt32(4))  # Aligned 4

    Lava.encode_instruction!(mod.functions, Lava.Op.OpReturn)
    Lava.encode_instruction!(mod.functions, Lava.Op.OpFunctionEnd)

    # Debug names
    Lava.emit_name!(mod, func_id, "main")

    return Lava.serialize(mod)
end

"""Build a closest-hit shader that writes hit_t to the incoming ray payload."""
function build_closesthit_shader()
    mod = Lava.SPIRVModule()

    Lava.require_capability!(mod, Lava.Cap.Shader)
    Lava.require_capability!(mod, Lava.Cap.RayTracingKHR)
    Lava.require_extension!(mod, "SPV_KHR_ray_tracing")
    Lava.setup_memory_model!(mod; physical_storage_buffer=false)

    void_t = Lava.emit_type_void!(mod)
    f32_t  = Lava.emit_type_float!(mod, UInt32(32))
    func_t = Lava.emit_type_function!(mod, void_t)

    # Pointer types
    ptr_f32_in_t   = Lava.emit_type_pointer!(mod, Lava.SC.Input, f32_t)
    ptr_f32_pay_t  = Lava.emit_type_pointer!(mod, Lava.SC.IncomingRayPayloadKHR, f32_t)

    # Built-in: RayTmaxKHR gives the t value at the closest hit
    hit_t_var = Lava.emit_global_variable!(mod, ptr_f32_in_t, Lava.SC.Input)
    Lava.emit_decorate!(mod, hit_t_var, Lava.Dec.BuiltIn, Lava.BuiltIn.RayTmaxKHR)

    # Incoming payload — no `Location` decoration (see raygen for rationale).
    payload_var = Lava.emit_global_variable!(mod, ptr_f32_pay_t, Lava.SC.IncomingRayPayloadKHR)

    # Entry point
    func_id = Lava.fresh_id!(mod)
    Lava.emit_entry_point!(mod, Lava.ExecModel.ClosestHitKHR, func_id, "main",
        UInt32[hit_t_var, payload_var])

    # Function body: payload = hit_t
    label_id = Lava.fresh_id!(mod)
    Lava.encode_instruction!(mod.functions, Lava.Op.OpFunction,
        void_t, func_id, Lava.FuncControl.None, func_t)
    Lava.encode_instruction!(mod.functions, Lava.Op.OpLabel, label_id)

    hit_t_val = Lava.fresh_id!(mod)
    Lava.encode_instruction!(mod.functions, Lava.Op.OpLoad, f32_t, hit_t_val, hit_t_var)
    Lava.encode_instruction!(mod.functions, Lava.Op.OpStore, payload_var, hit_t_val)
    Lava.encode_instruction!(mod.functions, Lava.Op.OpReturn)
    Lava.encode_instruction!(mod.functions, Lava.Op.OpFunctionEnd)

    Lava.emit_name!(mod, func_id, "main")

    return Lava.serialize(mod)
end

"""Build a miss shader that writes -1.0 to the incoming ray payload."""
function build_miss_shader()
    mod = Lava.SPIRVModule()

    Lava.require_capability!(mod, Lava.Cap.Shader)
    Lava.require_capability!(mod, Lava.Cap.RayTracingKHR)
    Lava.require_extension!(mod, "SPV_KHR_ray_tracing")
    Lava.setup_memory_model!(mod; physical_storage_buffer=false)

    void_t = Lava.emit_type_void!(mod)
    f32_t  = Lava.emit_type_float!(mod, UInt32(32))
    func_t = Lava.emit_type_function!(mod, void_t)

    ptr_f32_pay_t = Lava.emit_type_pointer!(mod, Lava.SC.IncomingRayPayloadKHR, f32_t)

    c_m1f = Lava.emit_constant_f32!(mod, -1f0)

    # No `Location` decoration — payload matching is via pointer.
    payload_var = Lava.emit_global_variable!(mod, ptr_f32_pay_t, Lava.SC.IncomingRayPayloadKHR)

    func_id = Lava.fresh_id!(mod)
    Lava.emit_entry_point!(mod, Lava.ExecModel.MissKHR, func_id, "main",
        UInt32[payload_var])

    label_id = Lava.fresh_id!(mod)
    Lava.encode_instruction!(mod.functions, Lava.Op.OpFunction,
        void_t, func_id, Lava.FuncControl.None, func_t)
    Lava.encode_instruction!(mod.functions, Lava.Op.OpLabel, label_id)
    Lava.encode_instruction!(mod.functions, Lava.Op.OpStore, payload_var, c_m1f)
    Lava.encode_instruction!(mod.functions, Lava.Op.OpReturn)
    Lava.encode_instruction!(mod.functions, Lava.Op.OpFunctionEnd)

    Lava.emit_name!(mod, func_id, "main")

    return Lava.serialize(mod)
end

# =====================================================================
# Test
# =====================================================================

@testset "Handwritten RT SPIR-V" begin

    @testset "SPIR-V Generation & Validation" begin
        raygen_spirv = build_raygen_shader()
        chit_spirv = build_closesthit_shader()
        miss_spirv = build_miss_shader()

        @test length(raygen_spirv) > 0
        @test length(chit_spirv) > 0
        @test length(miss_spirv) > 0

        # Validate all three
        @test_nowarn Lava.validate_spirv(raygen_spirv)
        @test_nowarn Lava.validate_spirv(chit_spirv)
        @test_nowarn Lava.validate_spirv(miss_spirv)

        # Check key instructions in raygen
        raygen_dis = Lava.disassemble_spirv(raygen_spirv)
        @test occursin("OpEntryPoint RayGenerationKHR", raygen_dis)
        @test occursin("OpTraceRayKHR", raygen_dis)
        @test occursin("RayTracingKHR", raygen_dis)

        # Check closest-hit
        chit_dis = Lava.disassemble_spirv(chit_spirv)
        @test occursin("OpEntryPoint ClosestHitKHR", chit_dis)

        # Check miss
        miss_dis = Lava.disassemble_spirv(miss_spirv)
        @test occursin("OpEntryPoint MissKHR", miss_dis)
    end

    @testset "RT Dispatch Against Triangle" begin
        ctx = MVE.vk_context()
        rt_props = ctx.rt_pipeline_properties
        if rt_props === nothing
            @warn "Skipping RT test: no ray tracing support"
            return
        end

        W, H = 16, 16

        # Build triangle: (0,0,0), (1,0,0), (0,1,0)
        vertices = [(0f0, 0f0, 0f0), (1f0, 0f0, 0f0), (0f0, 1f0, 0f0)]
        indices = UInt32[0, 1, 2]
        blas, tlas = Mantle.build_accel!() do ctx
            b = MVE.build_blas(ctx, vertices, indices)
            t = MVE.build_tlas(ctx, [b])
            (b, t)
        end

        # Build shaders
        raygen_spirv = build_raygen_shader()
        chit_spirv = build_closesthit_shader()
        miss_spirv = build_miss_shader()

        # Create output buffer (W*H float32 values)
        output_buf = MVE.vk_alloc(ctx.default_bq, W * H * sizeof(Float32))

        # Create RT pipeline (argument order: ctx, raygen, miss, chit)
        pipeline = MVE.create_rt_pipeline(ctx, raygen_spirv, miss_spirv, chit_spirv;
            push_constant_size=8)

        # Push constant: BDA of output buffer
        push_bda = output_buf.address

        # Dispatch
        rt_dispatch!(MVE.vk_context().default_bq, pipeline, tlas, push_bda, W, H)

        # Read back results
        result_bytes = Vector{UInt8}(undef, W * H * sizeof(Float32))
        MVE.download!(result_bytes, output_buf)
        result = reinterpret(Float32, result_bytes)

        # Verify: rays clearly inside the triangle should hit (t=1.0),
        # rays clearly outside should miss (t=-1.0).
        # Rays on edges may go either way (hardware edge-exclusion rules).
        eps = 0.15f0  # generous margin for hardware edge-exclusion rules
        n_hits = 0
        n_misses = 0
        n_edge = 0
        for iy in 0:H-1, ix in 0:W-1
            ray_x = Float32(ix) / Float32(W) * 2f0 - 0.5f0
            ray_y = Float32(iy) / Float32(W) * 2f0 - 0.5f0
            idx = iy * W + ix + 1
            # Triangle: (0,0,0), (1,0,0), (0,1,0)
            # Interior: ray_x > eps, ray_y > eps, ray_x + ray_y < 1 - eps
            clearly_inside = ray_x > eps && ray_y > eps && (ray_x + ray_y) < 1f0 - eps
            # Exterior: ray_x < -eps || ray_y < -eps || ray_x + ray_y > 1 + eps
            clearly_outside = ray_x < -eps || ray_y < -eps || (ray_x + ray_y) > 1f0 + eps
            if clearly_inside
                @test result[idx] ≈ 1.0f0 atol=0.01f0
                n_hits += 1
            elseif clearly_outside
                @test result[idx] == -1.0f0
                n_misses += 1
            else
                # Edge case: either hit or miss is acceptable
                @test isapprox(result[idx], 1.0f0; atol=0.01f0) || result[idx] == -1.0f0
                n_edge += 1
            end
        end

        @test n_hits > 0
        @test n_misses > 0
        println("RT test: $n_hits interior hits, $n_misses exterior misses, $n_edge edge cases out of $(W*H) rays")

        # Explicit cleanup: idle the device and finalize Vulkan handles in the
        # correct order (children before parents). After the AS-storage refactor,
        # LavaBLAS/LavaTLAS hold their backing memory via LavaArray fields
        # (`storage`, `preserves`) whose own finalizers handle vk_free! with
        # timeline-gated deferred destruction — so we only need to finalize the
        # AccelerationStructureKHR handles here.
        Vulkan.device_wait_idle(ctx.device)
        finalize(pipeline.pipeline)
        finalize(pipeline.pipeline_layout)
        finalize(pipeline.descriptor_set_layout)
        for sm in pipeline.shader_modules
            finalize(sm)
        end
        finalize(tlas.accel)
        finalize(blas.accel)
    end
end
