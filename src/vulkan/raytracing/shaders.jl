# Native RT compilation and eager submission; descriptions and shader ownership are core types.

"""
Cache key for a compiled RT pipeline: its description plus the argument types,
since the cache is the device's and shared by every pipeline.
"""
rt_cache_key(p::RayTracingPipeline, tt_key) =
    hash((p.raygen_func, p.closesthit_funcs, p.miss_func, p.anyhit_func,
          p.payload_type, p.chit_miss_take_args, tt_key))

"""Retain shader/argument leaves and pack into caller-owned memory; works for eager and recorded traces."""
function packtrace!(owner, pipeline::RayTracingPipeline, args::Tuple, ptr, address,
                    offsets, argbytes, byval)
    holdleaves!(owner, pipeline)
    holdleaves!(owner, args)
    ad = LavaAdaptor(owner)
    all_args = (Adapt.adapt(ad, pipeline.raygen_func), devargs(ad, args)...)
    pack_args_direct!(owner, ptr, address, offsets, argbytes, byval, all_args)
    return address
end

function traceargbuffer!(owner, pipeline, args, raygen, offsets, byval)
    argbytes = raygen.push_info.arg_buffer_size
    buffer = get_arg_buffer(owner, argbytes + compute_inline_extra_from_byval(byval))
    packtrace!(owner, pipeline, args, buffer.mapped_ptr, buffer.address, offsets, argbytes, byval)
    return buffer
end

"""
    trace_rays!(bq, pipeline, tlas::LavaTLAS, args...; width, height, depth=1)

Trace `width * height * depth` rays through `pipeline` in a one-shot on `bq`.
`args` reach the raygen shader through the argument buffer, as for a compute kernel.
"""
function trace_rays!(bq::SubmitChannel{<:VulkanQueue}, pipeline::RayTracingPipeline, tlas::LavaTLAS,
                     args...;
                     width::Integer, height::Integer, depth::Integer=1)
    # Before the one-shot; see `rt_compiled_for`.
    vk_pipeline, raygen_compiled, offsets, byval_sizes = rt_compiled_for(bq, pipeline, args)

    oneshot!(bq; tag = :trace) do e
        owner = e.owner
        arg_buf = traceargbuffer!(owner, pipeline, args, raygen_compiled, offsets, byval_sizes)
        driver(bq).last_dispatch_info = "rt_trace w=$width h=$height"
        emit_trace!(e, vk_pipeline, tlas, arg_buf.address, (width,height,depth),
                    driver(bq).last_dispatch_info)
    end
    return nothing
end

"""The device's pipeline for `pipeline` in the current world, compiled if needed.
Call before opening a one-shot: a cold SBT upload submits its own work."""
function rt_compiled_for(bq::SubmitChannel{<:VulkanQueue}, pipeline::RayTracingPipeline, args)
    ctx = ctxof(bq)
    tt_key = Tuple{map(arg_sigtype, args)...}
    key = rt_cache_key(pipeline, tt_key)
    world = Base.tls_world_age()
    current = get(ctx.caches.rt_current, key, nothing)
    current !== nothing && current[1] == world && return ctx.caches.rt_pipelines[current[2]]
    # A new world looks the stages up again: free for unchanged code, a recompile
    # for edited code. Keyed on the description alone, edits would never apply.
    tt = Tuple{map(a -> arg_sigtype(Adapt.adapt(LavaAdaptor(nothing), a)), args)...}
    stages = rt_stages(ctx, pipeline, tt)
    full = hash(map(objectid, stages.all), key)
    cached = get!(() -> build_rt_pipeline(ctx, stages), ctx.caches.rt_pipelines, full)
    ctx.caches.rt_current[key] = (world, full)
    return cached
end

"""
    trace_rays_indirect!(bq, pipeline, tlas::LavaTLAS, args...; n_rays::LavaArray{Int32})

[`trace_rays!`](@ref) with the ray count read from `n_rays` on the device: a
prepare kernel writes the indirect command the trace reads. Arguments go into its
one-shot's scratch; inside a graph use [`trace!`](@ref).
"""
function trace_rays_indirect!(bq::SubmitChannel{<:VulkanQueue}, pipeline::RayTracingPipeline,
                              tlas::LavaTLAS, args...;
                              n_rays::LavaArray{Int32})
    vk_pipeline, raygen_compiled, offsets, byval_sizes = rt_compiled_for(bq, pipeline, args)

    oneshot!(bq; tag = :trace) do e
        owner = e.owner
        # Prepare, barrier for the command processor, trace: one command buffer.
        indirect_view = indirect_command!(owner)
        prepare_indirect_rt_dispatch!(e, indirect_view, n_rays)
        indirectbarrier!(e, VK.PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR | VK.PIPELINE_STAGE_DRAW_INDIRECT_BIT,
                         VK.ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR | VK.ACCESS_INDIRECT_COMMAND_READ_BIT)

        arg_buf = traceargbuffer!(owner, pipeline, args, raygen_compiled, offsets, byval_sizes)
        driver(bq).last_dispatch_info = "rt_indirect"
        emit_trace!(e, vk_pipeline, tlas, arg_buf.address, indirect_view,
                             driver(bq).last_dispatch_info)
    end
    return nothing
end


"""Prepare indirect RT dispatch buffer: writes (n_rays, 1, 1) from a GPU-resident count."""
function prepare_indirect_rt_dispatch!(e::Emitter,
                                       indirect::LavaArray{UInt32,1},
                                       n_rays::LavaArray{Int32})
    emitkernel!(e, prepare_indirect_rt_kernel, indirect, n_rays;
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

"""
    get_or_compile_rt(f, tt; stage, push_constant_size, payload_type, features) -> LavaRTShader

Stage `stage` of `f`, compiled for `f`'s current code (`Lava.cached_rt_shader`).
"""
get_or_compile_rt(@nospecialize(f), @nospecialize(tt); stage::Symbol, push_constant_size::Integer,
                  payload_type::Symbol, features::TargetFeatures) =
    cached_rt_shader(f, tt; stage, push_constant_size, payload_type, features)

"""
    rt_stages(ctx, pipeline, raygen_tt) -> NamedTuple

Every stage of `pipeline`, compiled for its current code: `raygen`, `chits`,
`miss`, `anyhit` (or `nothing`), and `all` of them in one tuple.
"""
function rt_stages(ctx::VkContext, pipeline::RayTracingPipeline, raygen_tt)
    pt = pipeline.payload_type
    features = ctx.features
    raygen = get_or_compile_rt(pipeline.raygen_func, raygen_tt; stage = :raygen,
                               push_constant_size = 8, payload_type = pt, features)
    # With `chit_miss_take_args` closest-hit and miss get the raygen's argument
    # signature; otherwise none (`hw_closesthit` / `hw_miss`).
    takes = pipeline.chit_miss_take_args
    chit_tt, chit_push = takes ? (raygen_tt, 8) : (Tuple{}, 0)
    chits = LavaRTShader[get_or_compile_rt(chit, chit_tt; stage = :closesthit,
                                           push_constant_size = chit_push, payload_type = pt, features)
                         for chit in pipeline.closesthit_funcs]
    miss = get_or_compile_rt(pipeline.miss_func, chit_tt; stage = :miss,
                             push_constant_size = chit_push, payload_type = pt, features)
    # Any-hit gets the raygen's args (shares the BDA arg buffer via push constant)
    anyhit = pipeline.anyhit_func === nothing ? nothing :
             get_or_compile_rt(pipeline.anyhit_func, raygen_tt; stage = :anyhit,
                               push_constant_size = 8, payload_type = pt, features)
    return (; raygen, chits, miss, anyhit, all = (raygen, chits..., miss, anyhit))
end

"""The Vulkan pipeline of compiled `stages` (`rt_stages`), with the raygen's argument layout."""
function build_rt_pipeline(ctx::VkContext, stages)
    vk_pipeline = create_rt_pipeline(
        ctx,
        stages.raygen.spirv_bytes,
        stages.miss.spirv_bytes,
        Vector{UInt8}[c.spirv_bytes for c in stages.chits];
        anyhit_spirv = stages.anyhit === nothing ? nothing : stages.anyhit.spirv_bytes,
        push_constant_size = 8)
    # Cache arg layout offsets and byval sizes for zero-alloc packing
    raygen = stages.raygen
    return (vk_pipeline, raygen, raygen.push_info.arg_offsets, raygen.push_info.byval_llvm_sizes)
end

# Trace arguments are adapted by `LavaAdaptor`, as for `lava_launch!` and KA.
