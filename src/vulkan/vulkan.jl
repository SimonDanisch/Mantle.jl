# Mantle's Vulkan backend: the runtime that used to be Lava's.
#
# 26,000 lines moved here on 2026-08-27, and the reason is the goal the split was
# for: **Lava is a Julia→SPIR-V compiler and nothing else.** Memory, queues,
# state, pipelines and the graph are Mantle's, so a second backend implements one
# interface instead of copying a runtime.
#
# The boundary was made real before anything moved. Lava's half names no Vulkan
# type at all — `Lava/test/test_compiler_runtime_split.jl` asserts it from the
# source — which is why this is a move rather than a rewrite. Four things had to
# be untangled first: `DeviceCaps` (written twice and copied positionally, now
# `KernelInterface`'s), the emitter's `VK_CONTEXT_REF` reads (now
# `Lava.TargetFeatures`, pushed from here by `bind_context!`),
# `spirv_content_hash`, and the frozen cache, whose SPIR-V half is the compiler's
# and whose `VkPipelineCache` half is this one's.
#
# These files are included INTO `Mantle` rather than a submodule, because the
# methods in them are methods on Mantle's own functions — `caps`, `upload!`,
# `draw!`, `release!` — and a submodule would mean re-exporting every one. The
# four names that exist on both sides all took distinct signatures and merged
# into one function each, which is what they should have been.

# The compiler. Lava exports the user-facing half; everything below is reached by
# name because it is internal to the compiler and this is its one consumer.
#
# The list is long and it is exact — every name was found by asking which
# identifiers this code uses that only Lava defines, not by guessing. Forty-three
# of them are the DEVICE INTRINSICS (`lava_local_invocation_id_x` and friends),
# which a syntactic scan misses entirely because they are generated in an `@eval`
# loop; those only surfaced when a kernel refused to compile.
using Lava
using Lava: @lava_device_override, AcceleratedMatrix, Accumulator, Additive,
            AlphaBlend, BlendMode, Cap, CoopMatrix, CullBack, CullFace,
            CullFront, DepthAlways, DepthGreater, DepthLess, DepthLessEq,
            DepthMode, DepthOff, FROZEN_HITS, FROZEN_LOG_MISSES,
            FROZEN_MISSES, FROZEN_PRUNED, FROZEN_RECORDING, FROZEN_RT_MEM,
            FROZEN_STORES, FROZEN_VERSION, FragmentWrapper, GeometryConfig,
            KI_SHFL_TYPES, LavaCompilationError, LavaCompilerParams,
            LavaDeviceArray, LavaError, LavaGPUKernel, LavaGfxShader,
            LavaRTShader, LavaSharedArray, LineList, LineListAdjacency,
            LineStrip, LineStripAdjacency, MatrixA, MatrixB, NoCull, Op,
            Opaque, PatchList, PointList, Premultiplied, PushConstantInfo,
            RenderTarget, Scope, SourceMap, TENSOR_CLAMP_CONSTANT,
            TENSOR_CLAMP_UNDEFINED, TargetFeatures, TessConfig, Topology,
            TriangleList, TriangleStrip, VertexWrapper, WorkgroupMatrix,
            cache_io_error, coopmat_convert, coopmat_getcomp, coopmat_length,
            coopmat_load, coopmat_muladd, coopmat_setcomp, coopmat_store,
            coopmat_undef, coopmat_zero, disassemble_spirv,
            dump_spirv_to_disk, frag_coord_x, frag_coord_y, frozen_binpath,
            frozen_cache_dir, frozen_eligible, frozen_key, frozen_logging,
            frozen_max_bytes, frozen_path, frozen_rt_load, frozen_rt_store,
            gfx_input, gfx_output, invoke_frozen, kernel_dump_wanted,
            kernel_source_name,
            lava_alloc_shared, lava_compile_gfx_shader,
            lava_compile_gpu_from_job, lava_compile_rt_shader,
            lava_compiler_config, lava_local_invocation_id_x,
            lava_local_invocation_index, lava_num_workgroups,
            lava_num_workgroups_x, lava_num_workgroups_y,
            lava_num_workgroups_z, lava_ray_query_get_barycentrics,
            lava_ray_query_get_instance_custom_index,
            lava_ray_query_get_instance_id,
            lava_ray_query_get_primitive_index, lava_ray_query_get_t,
            lava_ray_query_get_type, lava_ray_query_init,
            lava_ray_query_proceed, lava_rt_hit_bary_u, lava_rt_hit_bary_v,
            lava_rt_instance_custom_index, lava_rt_instance_id,
            lava_rt_launch_id_x, lava_rt_payload_load_f32_at,
            lava_rt_payload_store_f32_at, lava_rt_primitive_id,
            lava_rt_ray_tmax, lava_rt_trace_ray, lava_workgroup_barrier,
            lava_workgroup_id_x, lava_workgroup_id_y, lava_workgroup_id_z,
            run_spirv_opt, set_position!, spirv_content_hash, subgroup_add,
            subgroup_elect, subgroup_shuffle, subgroup_size, targetfeatures,
            targetfeatures!, tensor_layout, tensor_load, tensor_setdim,
            tensor_setstride, tensor_slice, tensor_store, typestring,
            unroll_loops!, validate_spirv, vertex_index,
            wg_compute_type_alignment, wg_compute_type_size


# Include order is Lava's, unchanged: it encodes real type dependencies.
include("runtime/vkerrors.jl")
include("runtime/coretypes.jl")
include("runtime/device.jl")
include("runtime/matrixshapes.jl")
include("runtime/memory.jl")
include("runtime/pipeline.jl")
include("runtime/command.jl")
include("array/lavaarray.jl")
include("array/pin_leaves.jl")
include("runtime/launch.jl")
include("runtime/frozen_pipeline.jl")
include("runtime/workload.jl")
include("runtime/pipeline_cache.jl")
include("array/ka_backend.jl")
include("array/kernelinterface_host.jl")
include("array/gpuarrays.jl")
include("array/gemm.jl")
include("array/gemm_cm2.jl")
include("array/mapreduce.jl")
include("runtime/debug.jl")
include("runtime/diagnostics.jl")   # gpu_memory_usage, dump_state
include("runtime/external.jl")
include("runtime/video.jl")
include("runtime/videoapi.jl")   # decode_h264_gpu and friends
include("runtime/profiling.jl")
include("graphics/window.jl")
include("graphics/framebuffer.jl")
include("graphics/pipeline.jl")
include("graphics/textures.jl")
include("graphics/api.jl")
include("raytracing/geometry_types.jl")
include("raytracing/instance_record.jl")
include("raytracing/acceleration.jl")
include("raytracing/pipeline.jl")
include("raytracing/shaders.jl")
include("raytracing/raycore_compat.jl")
include("raytracing/hwtlas.jl")
include("raytracing/convex_shape.jl")
include("raytracing/gjk.jl")
include("raytracing/epa.jl")
include("kernels/instance_writer.jl")
include("kernels/narrow_phase.jl")
# `runtime/video_h265.jl` is not listed: `runtime/video.jl` includes it itself,
# beside the `decodeprofile` hook it implements.

# The Vulkan lowering — stages, access masks, image layouts — and the graph, plan
# and pass machinery over it. These were `MantleVulkanExt` and `MantleLavaExt`;
# both stopped being extensions when Vulkan and Lava became hard dependencies.
# Last, because they are written against everything above.
include("lowering.jl")
include("graph.jl")

# ── The exported surface, restored ────────────────────────────────────────────
#
# These 79 names moved here with the runtime and lost their export on the way:
# they were `export`ed from `Lava.jl`, the definitions came to Mantle, and the
# export lines stayed behind. Nothing failed at load — an unexported name is
# perfectly legal — so it surfaced one `UndefVarError` at a time, at whatever
# line first said `Rasterizer` or `LavaArray` or `trim_gpu_pool!`, in 43 test
# files and anything downstream that had said `using Lava` rather than
# qualifying.
#
# Derived rather than guessed: the pre-split `export` list from Lava, filtered to
# what Mantle defines and neither package exports. Names Lava still owns — the
# shader intrinsics, `LavaDeviceArray`, the graphics enums — are not here,
# because Lava still exports them and a name exported by both would be ambiguous
# to anything that loaded the two.

export AABB, AABBsGeometry, ASBuildContext, BatchQueue, CompiledGraphicsPipeline,
       DebugConfig, allocate_batch_queue!, build_4x3, build_4x3_pervec,
       dump_state, gpu_memory_usage, quat_to_rot3x3, readback_framebuffer,
       readback_window, release_batch_queue!,
       ContactRecord, ConvexShape, DrawIndirectCommand, EPAResult, ExternalImage,
       GJKResult, GeometryType, GraphicsPipeline, HWAdaptedAccel, HWTLAS, HardwareAccel,
       LavaFramebuffer, LavaInstanceRecord, LavaSampler, LavaTexture, LavaTexture1D,
       LavaTexture2D, LinePipeline, Mat3x4f, NO_CONTACT, OffscreenTarget, RTHitResult,
       RTRay, Rasterizer, RayTracingPipeline, RenderWindow, SampledTexture,
       TextureBindings, TrianglePipeline, TrianglesGeometry, UnitCube, WindowTarget,
       acquire_next_image!, alloc_index_buffer, as_build, bind_textures, blit!,
       concurrent_dispatch_group, concurrent_indirect_group, copy_framebuffer!,
       ensure_compiled!, epa, exclusive_dispatch_group, get_dispatch_log, gjk,
       identity_transform, indirect_buffer, memoryfd, narrow_phase_contacts_kernel,
       narrow_phase_kernel, pack_gfx_args, present_frame!, refit_tlas!,
       set_anyhit_pipeline!, set_dispatch_logging!, support, sync_swapchain!,
       trace_closest_hits!, trace_closest_hits_anyhit!,
       trace_closest_hits_anyhit_indirect!, trace_closest_hits_indirect!, trace_rays!,
       trace_rays_indirect!, transition_image!, trim_gpu_pool!, verify_gpu_av,
       vk_begin_pass!, vk_draw_in_pass!, vk_draw_indexed_in_pass!,
       vk_draw_indirect_in_pass!, vk_end_pass!, vk_reset_device!, vk_set_viewport!,
       write_grain_instances_kernel

# `@compile_workload` separately, because a macro cannot go in a list of plain
# names. The version-taking one freezes kernels into the on-disk cache, which
# needs a device to compile them for — so it came here with `runtime/workload.jl`
# while `@setup_workload` (PrecompileTools') stayed re-exported from Lava.
export @compile_workload
