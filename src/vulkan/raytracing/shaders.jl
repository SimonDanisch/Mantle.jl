# `RayTracingPipeline` is Mantle's, in `src/raytracing/pipeline.jl`. It is a
# pure description now — the compiled pipelines it used to cache in a field live
# in `DeviceCaches.rt_pipelines`, beside the graphics ones, because a compiled
# pipeline is a device object.

# No-op: cache is tied to the current VkContext's lifetime.  On reset_device!,
# the whole module should re-initialize its pipelines anyway.


# High-level Ray Tracing Pipeline API for Lava.jl
#
# Provides a Julia-function-based API:
#   rt = RayTracingPipeline(raygen=f, closest_hit=g, miss=h)
#   trace_rays!(rt, tlas, args...; width=W, height=H)
#
# Compilation is lazy — shaders are compiled on first use and cached.



"""
Cache key for a compiled RT pipeline: what it is, plus what it is called with.

The per-object cache keyed on argument types alone, because the object it hung
off supplied the rest of the identity. Now that the cache is the device's — as
the graphics one always was — the description has to be in the key.
"""
rt_cache_key(p::RayTracingPipeline, tt_key) =
    hash((p.raygen_func, p.closesthit_funcs, p.miss_func, p.anyhit_func,
          p.payload_type, p.chit_miss_take_args, tt_key))

# `_normalise_chit` moved to core with the pipeline it normalises for.

invalidate_stale_rt_cache!(::RayTracingPipeline) = nothing

"""
    trace_rays!(pipeline, tlas, args...; width, height, depth=1)

Dispatch a ray tracing pipeline. The `args` are passed to the raygen shader
via the BDA argument buffer (same as compute kernel arguments).

- `tlas`: A `LavaTLAS` acceleration structure
- `args`: Arguments passed to the raygen function (buffers, scalars, structs)
- `width`, `height`, `depth`: Dispatch dimensions (number of rays per dimension)
"""
function trace_rays!(bq::VulkanBatchQueue, pipeline::RayTracingPipeline, tlas::LavaTLAS,
                     args...;
                     width::Integer, height::Integer, depth::Integer=1)
    # Resolve or compile the RT pipeline FIRST.  A cold compile builds the
    # SBT via `upload_typed!`, which calls `flush!(bq)` and invalidates any
    # active batch — so we must run it before `ensure_active_batch!` below.
    # On warm calls this is a Dict lookup and does not flush.
    invalidate_stale_rt_cache!(pipeline)
    tt_key = Tuple{map(arg_sigtype, args)...}   # pre-adapt types (same as post-adapt for non-LavaArray)
    cache_key = rt_cache_key(pipeline, tt_key)
    cached = get((bq.ctx::VkContext).caches.rt_pipelines, cache_key, nothing)
    if cached === nothing
        # Compile with post-adapt signature: LavaArray args are seen as
        # LavaDeviceArray in the kernel, matching what pack_args_direct!
        # writes.  We drive the adapt through a throwaway batch since the
        # real batch isn't opened yet; adaptor-side pinning on the throwaway
        # is irrelevant (we re-adapt against the real batch below).
        dummy_batch = ensure_active_batch!(bq)
        tt = Tuple{map(a -> arg_sigtype(Adapt.adapt(LavaAdaptor(dummy_batch), a)), args)...}
        cached = compile_rt_pipeline(bq.ctx::VkContext, pipeline, tt)
        (bq.ctx::VkContext).caches.rt_pipelines[cache_key] = cached
    end
    vk_pipeline, raygen_compiled, offsets, byval_sizes = cached

    # Now open (or re-open) the real active batch and adapt args into it.
    batch = ensure_active_batch!(bq)
    pin_leaves!(batch, pipeline.raygen_func)
    for chit in pipeline.closesthit_funcs
        pin_leaves!(batch, chit)
    end
    pin_leaves!(batch, pipeline.miss_func)
    pin_leaves!(batch, pipeline.anyhit_func)   # pin_leaves!(::Nothing) is a no-op
    pin_leaves!(batch, args)
    adaptor = LavaAdaptor(batch)
    converted_raygen = Adapt.adapt(adaptor, pipeline.raygen_func)
    converted_args = map(a -> Adapt.adapt(adaptor, a), args)

    all_args = (converted_raygen, converted_args...)
    inline_extra = compute_inline_extra_from_byval(byval_sizes)
    total_size = raygen_compiled.push_info.arg_buffer_size + inline_extra

    arg_buf = get_arg_buffer(bq, total_size)

    pack_args_direct!(bq, arg_buf.mapped_ptr, arg_buf.address, offsets,
                       raygen_compiled.push_info.arg_buffer_size, byval_sizes, all_args)
    # HWTLAS/BLAS handles are bound via descriptor set, not the arg tuple — pin explicitly.
    pin!(batch, tlas.accel)
    pin!(batch, tlas.storage)
    for blas in tlas.blases
        pin!(batch, blas.accel)
        pin!(batch, blas.storage)
    end

    rt_dispatch!(bq, vk_pipeline, tlas, arg_buf.address, width, height; depth=depth)
end

"""
    trace_rays_indirect!(pipeline, tlas, args...; n_rays::LavaArray{Int32})

Dispatch a ray tracing pipeline with the ray count read from a GPU buffer.
No CPU readback — a prepare kernel writes the indirect command, then
`cmd_trace_rays_indirect_khr` reads it from GPU memory.
"""
function trace_rays_indirect!(bq::VulkanBatchQueue, pipeline::RayTracingPipeline,
                              tlas::LavaTLAS, args...;
                              n_rays::LavaArray{Int32})
    # See comment in trace_rays!: a cold compile's SBT upload flushes the
    # active batch, so compile FIRST, then open the real batch below.
    invalidate_stale_rt_cache!(pipeline)
    tt_key = Tuple{map(arg_sigtype, args)...}
    cache_key = rt_cache_key(pipeline, tt_key)
    cached = get((bq.ctx::VkContext).caches.rt_pipelines, cache_key, nothing)
    if cached === nothing
        dummy_batch = ensure_active_batch!(bq)
        tt = Tuple{map(a -> arg_sigtype(Adapt.adapt(LavaAdaptor(dummy_batch), a)), args)...}
        cached = compile_rt_pipeline(bq.ctx::VkContext, pipeline, tt)
        (bq.ctx::VkContext).caches.rt_pipelines[cache_key] = cached
    end
    vk_pipeline, raygen_compiled, offsets, byval_sizes = cached

    # Prepare the indirect buffer before the RT arg buffer: prepare_indirect
    # dispatches its own kernel which may flush-and-reset slab pools, so if
    # we allocated the RT arg buffer first it could be invalidated.
    indirect_view = get_indirect_buffer(bq)
    prepare_indirect_rt_dispatch!(bq, indirect_view, n_rays)

    batch = ensure_active_batch!(bq)
    pin_leaves!(batch, pipeline.raygen_func)
    for chit in pipeline.closesthit_funcs
        pin_leaves!(batch, chit)
    end
    pin_leaves!(batch, pipeline.miss_func)
    pin_leaves!(batch, pipeline.anyhit_func)   # pin_leaves!(::Nothing) is a no-op
    pin_leaves!(batch, args)
    adaptor = LavaAdaptor(batch)
    converted_raygen = Adapt.adapt(adaptor, pipeline.raygen_func)
    converted_args = map(a -> Adapt.adapt(adaptor, a), args)

    all_args = (converted_raygen, converted_args...)
    inline_extra = compute_inline_extra_from_byval(byval_sizes)
    total_size = raygen_compiled.push_info.arg_buffer_size + inline_extra

    arg_buf = get_arg_buffer(bq, total_size)

    pack_args_direct!(bq, arg_buf.mapped_ptr, arg_buf.address, offsets,
                       raygen_compiled.push_info.arg_buffer_size, byval_sizes, all_args)
    pin!(batch, tlas.accel)
    pin!(batch, tlas.storage)
    for blas in tlas.blases
        pin!(batch, blas.accel)
        pin!(batch, blas.storage)
    end

    rt_dispatch_indirect!(bq, vk_pipeline, tlas, arg_buf.address, indirect_view)
end


"""Prepare indirect RT dispatch buffer: writes (n_rays, 1, 1) from a GPU-resident count."""
function prepare_indirect_rt_dispatch!(bq::VulkanBatchQueue,
                                       indirect::LavaArray{UInt32,1},
                                       n_rays::LavaArray{Int32})
    lava_launch!(bq, prepare_indirect_rt_kernel, indirect, n_rays;
                 ndrange=1, workgroup_size=(1, 1, 1))
end

function prepare_indirect_rt_kernel(indirect::LavaDeviceArray{UInt32,1},
                                     n_rays_buf::LavaDeviceArray{Int32,1})
    n = UInt32(n_rays_buf[1])
    indirect[1] = n            # width = n_rays
    indirect[2] = UInt32(1)    # height = 1
    indirect[3] = UInt32(1)    # depth = 1
    return nothing
end

# ── Internal: Compile RT pipeline ──

function compile_rt_pipeline(ctx::VkContext, pipeline::RayTracingPipeline, raygen_tt)
    pt = pipeline.payload_type

    # Compile raygen
    raygen_compiled = lava_compile_rt_shader(pipeline.raygen_func, raygen_tt;
        stage=:raygen, push_constant_size=8, payload_type=pt, validate=true)

    # Compile closesthits & miss.  When `chit_miss_take_args` is set the chit
    # and miss receive the raygen's BDA arg signature (same push-constant
    # pointer the raygen sees), enabling the pbrt-v4 OptiX pattern where
    # shading happens in closesthit.  Otherwise they're compiled with no
    # args (legacy `hw_closesthit` / `hw_miss` contract).
    chit_tt, chit_push = pipeline.chit_miss_take_args ? (raygen_tt, 8) : (Tuple{}, 0)
    chit_spirvs = Vector{UInt8}[]
    for chit in pipeline.closesthit_funcs
        c = lava_compile_rt_shader(chit, chit_tt;
            stage=:closesthit, push_constant_size=chit_push, payload_type=pt, validate=true)
        push!(chit_spirvs, c.spirv_bytes)
    end

    miss_tt, miss_push = pipeline.chit_miss_take_args ? (raygen_tt, 8) : (Tuple{}, 0)
    miss_compiled = lava_compile_rt_shader(pipeline.miss_func, miss_tt;
        stage=:miss, push_constant_size=miss_push, payload_type=pt, validate=true)

    # Compile any-hit (optional)
    anyhit_spirv = nothing
    if pipeline.anyhit_func !== nothing
        # Any-hit gets same args as raygen (shares BDA arg buffer via push constant)
        anyhit_compiled = lava_compile_rt_shader(pipeline.anyhit_func, raygen_tt;
            stage=:anyhit, push_constant_size=8, payload_type=pt, validate=true)
        anyhit_spirv = anyhit_compiled.spirv_bytes
    end

    # Create Vulkan RT pipeline from compiled SPIR-V
    vk_pipeline = create_rt_pipeline(
        ctx,
        raygen_compiled.spirv_bytes,
        miss_compiled.spirv_bytes,
        chit_spirvs;
        anyhit_spirv=anyhit_spirv,
        push_constant_size=8)

    # Cache arg layout offsets and byval sizes for zero-alloc packing
    offsets = raygen_compiled.push_info.arg_offsets
    byval_sizes = raygen_compiled.push_info.byval_llvm_sizes

    return (vk_pipeline, raygen_compiled, offsets, byval_sizes)
end

# RT args go through the same LavaAdaptor contract as lava_launch! / KA —
# no parallel type mapping here.  See `trace_rays!` / `trace_rays_indirect!`.
