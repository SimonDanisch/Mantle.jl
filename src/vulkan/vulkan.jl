# Mantle's Vulkan backend. Lava is only the Julia→SPIR-V compiler and names no
# Vulkan type (`Lava/test/test_compiler_runtime_split.jl`); the emitter reads
# the device's `Lava.TargetFeatures` (`ctx.features`), and SPIR-V lives in Lava while
# pipelines live here. Included into `Mantle`, not a submodule, because these
# files add methods to Mantle's own functions.

# Compiler internals this backend reaches by name, including device intrinsics
# Lava generates in an `@eval` loop. Shader builtins (`vertex_index`,
# `frag_coord`, `rt_*`) are KernelInterface's, through `using KernelInterface`
# in `Mantle.jl`.
using Lava
using Lava: @lava_device_override, AcceleratedMatrix, Accumulator, Cap,
            CoopMatrix, FragmentWrapper, GeometryConfig,
            KI_REDUCE_ADD_TYPES, KI_SHFL_TYPES, LavaCompilationError,
            LavaCompilerParams, LavaDeviceArray, LavaError, LavaGPUKernel,
            LavaGfxShader, LavaSharedArray, MatrixA, MatrixB,
            Op, PushConstantInfo, Scope, SourceMap, TargetFeatures, TessConfig,
            GeometryWrapper, MeshWrapper, VertexWrapper, WorkgroupMatrix,
            cached_gfx_shader, compile_or_lookup, compile_stats,
            coopmat_convert, coopmat_getcomp, coopmat_length, coopmat_load,
            coopmat_muladd, coopmat_setcomp, coopmat_store, coopmat_undef,
            coopmat_zero, disassemble_spirv, dump_spirv_to_disk,
            gfx_input,
            gfx_output, kernel_dump_wanted,
            kernel_source_name, lava_alloc_shared, lava_kernel_job,
            lava_compiler_config, lava_local_invocation_id_x,
            lava_local_invocation_index, lava_num_workgroups,
            lava_num_workgroups_x, lava_num_workgroups_y,
            lava_num_workgroups_z, lava_ray_query_get_barycentrics,
            lava_ray_query_get_instance_custom_index,
            lava_ray_query_get_instance_id,
            lava_ray_query_get_primitive_index, lava_ray_query_get_t,
            lava_ray_query_get_type, lava_ray_query_init,
            lava_ray_query_proceed,
            # Candidates: accept a triangle, or commit a box's own hit
            # (`raytracing/hwtlas.jl`).
            lava_ray_query_confirm, lava_ray_query_generate_intersection,
            lava_ray_query_get_object_ray_origin,
            lava_ray_query_get_object_ray_direction,
            lava_workgroup_barrier,
            lava_workgroup_id_x, lava_workgroup_id_y, lava_workgroup_id_z,
            run_spirv_opt, set_position!, spirv_content_hash, subgroup_add,
            subgroup_elect, subgroup_shuffle, subgroup_size,
            reset_compile_stats!, unroll_loops!, validate_spirv,
            wg_compute_type_alignment, wg_compute_type_size


# Include order encodes type dependencies.
include("runtime/vkerrors.jl")
include("runtime/coretypes.jl")
include("runtime/device.jl")
include("runtime/matrixshapes.jl")
include("runtime/memory.jl")
include("runtime/pipeline.jl")
include("runtime/command.jl")
include("array/lavaarray.jl")
include("runtime/launch.jl")
include("runtime/pipeline_cache.jl")
include("array/ka_backend.jl")
include("array/kernelinterface_host.jl")
include("array/gpuarrays.jl")
include("array/gemm.jl")
include("array/mapreduce.jl")
include("runtime/debug.jl")
include("runtime/diagnostics.jl")   # gpu_memory_usage
include("runtime/external.jl")
include("runtime/video.jl")
include("runtime/videoapi.jl")   # decode_h264_gpu and friends
include("runtime/profiling.jl")
include("graphics/window.jl")
include("graphics/framebuffer.jl")
include("graphics/pipeline.jl")
include("graphics/textures.jl")
include("graphics/api.jl")
include("raytracing/instance_record.jl")
include("raytracing/acceleration.jl")
include("raytracing/hwtlas.jl")
# `runtime/video_h265.jl` is included by `runtime/video.jl`.

# The Vulkan lowering (stages, access masks, image layouts) and the graph, plan
# and pass machinery; last because they use everything above.
include("lowering.jl")
include("graph.jl")
# Both after `graph.jl`, which declares `LavaDevice` and `adaptor`.
include("access.jl")
include("readonly.jl")
include("graphics/record.jl")

# No `export`s: public names (`Texture2D`, `HWTLAS`, `begin_pass!`,
# `trace_rays!`, ...) are declared and exported by Mantle; these files add the
# Vulkan methods.

# Load-time setup; `initbackend!` is declared in `graph/backend.jl`.
function initbackend!(::VulkanAPI)
    # Never the default on a Mac: Metal is the GPU there, Vulkan a CPU driver
    # chosen by name.
    @static Sys.isapple() || register_backend!(; name = :vulkan, priority = 100) do
        vulkan_available() ? Device(VulkanAPI()) : nothing
    end
    init_pipeline_thread!()
end
