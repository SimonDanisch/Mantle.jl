# Native RT compilation and eager submission; descriptions and shader ownership are core types.

"""
Cache key for a compiled RT pipeline: its description plus the argument types,
since the cache is the device's and shared by every pipeline.
"""
rt_cache_key(p::RayTracingPipeline, tt_key) =
    hash((p.raygen_func, p.closesthit_funcs, p.miss_func, p.anyhit_func,
          p.payload_type, tt_key))

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

"""The device's pipeline for `pipeline` in the current world, compiled if needed.
Call before opening a one-shot: a cold SBT upload submits its own work."""
function rt_compiled_for(bq::SubmitChannel{<:VulkanQueue}, pipeline::RayTracingPipeline, args)
    ctx = deviceof(bq)
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
    rt_stages(ctx, pipeline, raygen_tt) -> NamedTuple

Every stage of `pipeline`, compiled for its current code: `raygen`, `chits`,
`miss`, `anyhit` (or `nothing`), and `all` of them in one tuple.
"""
function rt_stages(ctx::LavaDevice, pipeline::RayTracingPipeline, raygen_tt)
    pt = pipeline.payload_type
    features = ctx.features
    raygen = cached_rt_shader(pipeline.raygen_func, raygen_tt; stage = :raygen,
                               push_constant_size = 8, payload_type = pt, features)
    # Every stage gets the raygen's arguments: the push constant is the address
    # of the one argument buffer.
    chits = LavaRTShader[cached_rt_shader(chit, raygen_tt; stage = :closesthit,
                                           push_constant_size = 8, payload_type = pt, features)
                         for chit in pipeline.closesthit_funcs]
    miss = cached_rt_shader(pipeline.miss_func, raygen_tt; stage = :miss,
                             push_constant_size = 8, payload_type = pt, features)
    anyhit = pipeline.anyhit_func === nothing ? nothing :
             cached_rt_shader(pipeline.anyhit_func, raygen_tt; stage = :anyhit,
                               push_constant_size = 8, payload_type = pt, features)
    return (; raygen, chits, miss, anyhit, all = (raygen, chits..., miss, anyhit))
end

"""The Vulkan pipeline of compiled `stages` (`rt_stages`), with the raygen's argument layout."""
function build_rt_pipeline(ctx::LavaDevice, stages)
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
