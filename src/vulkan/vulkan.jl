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
#
# What is NOT here any more: the shader builtins a body writes. `vertex_index`,
# the `frag_coord` family and the `rt_*` intrinsics are KernelInterface's names
# (phase 2.1), which this package reaches through `using KernelInterface` in
# `Mantle.jl` — Lava OVERRIDES them for this target rather than defining names
# of its own. `gfx_input`/`gfx_output`/`set_position!` and the two stage
# wrappers stay, because they are the lowering the wrappers compile a stage's
# NamedTuple into and nobody outside writes them.
using Lava
using Lava: @lava_device_override, AcceleratedMatrix, Accumulator, Cap,
            CoopMatrix, FROZEN_HITS, FROZEN_LOG_MISSES, FROZEN_MISSES,
            FROZEN_PRUNED, FROZEN_RECORDING, FROZEN_RT_MEM, FROZEN_STORES,
            FROZEN_VERSION, FragmentWrapper, GeometryConfig,
            KI_REDUCE_ADD_TYPES, KI_SHFL_TYPES, LavaCompilationError,
            LavaCompilerParams, LavaDeviceArray, LavaError, LavaGPUKernel,
            LavaGfxShader, LavaRTShader, LavaSharedArray, MatrixA, MatrixB,
            Op, PushConstantInfo, Scope, SourceMap, TENSOR_CLAMP_CONSTANT,
            TENSOR_CLAMP_UNDEFINED, TargetFeatures, TessConfig,
            GeometryWrapper, VertexWrapper, WorkgroupMatrix, cache_io_error,
            coopmat_convert, coopmat_getcomp, coopmat_length, coopmat_load,
            coopmat_muladd, coopmat_setcomp, coopmat_store, coopmat_undef,
            coopmat_zero, disassemble_spirv, dump_spirv_to_disk,
            frozen_binpath, frozen_cache_dir,
            frozen_eligible, frozen_key, frozen_logging, frozen_max_bytes,
            frozen_path, frozen_rt_load, frozen_rt_store, gfx_input,
            gfx_output, invoke_frozen, kernel_dump_wanted,
            kernel_source_name, lava_alloc_shared, lava_compile_gfx_shader,
            lava_compile_gpu_from_job, lava_compile_rt_shader,
            lava_compiler_config, lava_local_invocation_id_x,
            lava_local_invocation_index, lava_num_workgroups,
            lava_num_workgroups_x, lava_num_workgroups_y,
            lava_num_workgroups_z, lava_ray_query_get_barycentrics,
            lava_ray_query_get_instance_custom_index,
            lava_ray_query_get_instance_id,
            lava_ray_query_get_primitive_index, lava_ray_query_get_t,
            lava_ray_query_get_type, lava_ray_query_init,
            lava_ray_query_proceed,
            # The four RT intrinsics KernelInterface does not declare — the set
            # there is what CALLERS use, and these are called only by this
            # backend's own raygen and closest-hit shaders
            # (`raytracing/raycore_compat.jl`).
            lava_rt_payload_load_f32_at, lava_rt_payload_store_f32_at,
            lava_rt_terminate_ray, lava_rt_trace_ray,
            lava_workgroup_barrier,
            lava_workgroup_id_x, lava_workgroup_id_y, lava_workgroup_id_z,
            run_spirv_opt, set_position!, spirv_content_hash, subgroup_add,
            subgroup_elect, subgroup_shuffle, subgroup_size,
            tensor_layout, tensor_load,
            tensor_setdim, tensor_setstride, tensor_slice, tensor_store,
            typestring, unroll_loops!, validate_spirv,
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
include("raytracing/instance_record.jl")
include("raytracing/acceleration.jl")
include("raytracing/pipeline.jl")
include("raytracing/shaders.jl")
include("raytracing/raycore_compat.jl")
include("raytracing/hwtlas.jl")
include("kernels/instance_writer.jl")
# `runtime/video_h265.jl` is not listed: `runtime/video.jl` includes it itself,
# beside the `decodeprofile` hook it implements.

# The Vulkan lowering — stages, access masks, image layouts — and the graph, plan
# and pass machinery over it. These were `MantleVulkanExt` and `MantleLavaExt`;
# both stopped being extensions when Vulkan and Lava became hard dependencies.
# Last, because they are written against everything above.
include("lowering.jl")
include("graph.jl")
# After `graph.jl`: the hand-recorded pass opens a one-shot on a `LavaDevice`,
# which is declared there.
include("graphics/record.jl")

# ── No exports here ───────────────────────────────────────────────────────────
#
# This file used to end with 90 `export` lines, restored after the runtime moved
# from Lava and lost them. They are gone, and not because the problem came back:
# these files are an EXTENSION now, and an extension cannot export into its
# parent no matter what it writes.
#
# That is not a loss of surface, it is where the surface belongs. Every name a
# caller needs — `Texture2D`, `HWTLAS`, `begin_pass!`, `trace_rays!` — is declared
# and exported by Mantle, and what is defined below are the Vulkan METHODS on
# them. `test_no_stale_exports.jl` covers the parent; nothing needs to cover an
# export list that cannot exist.

# `@compile_workload` was exported here too, and cannot be either — same reason.
# It is reachable as `MantleVulkanExt.@compile_workload`; giving it a home in
# Mantle means deciding what it does with no device, which is a question for
# whoever needs it from a second backend.

# ── What this backend does once, at load ──────────────────────────────────────
#
# Declared by core in `graph/backend.jl`; the body is here because every name in
# it is this backend's.
function initbackend!()
    register_backend!(; name = :vulkan, priority = 100) do
        vulkan_available() ? LavaBackend() : nothing
    end
    register_kernel_recorder!(with_frozen_recording; name = :vulkan)
    init_pipeline_thread!()
    atexit() do
        mark_all_devices_lost!()
        bind_context!(nothing)
    end
end
