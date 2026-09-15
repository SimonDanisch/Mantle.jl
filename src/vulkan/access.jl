# The signature a dispatch is inferred at, on this backend.
#
# Core walks the IR (`graph/access.jl`); what is here is the three things only a
# Vulkan/Lava dispatch knows: that arguments are `Adapt`ed on their way to the
# shader, that a `KernelAbstractions` kernel takes its iteration context as a
# leading argument, and that the method table the kernel will be compiled
# against is Lava's.

argtype(dev::LavaDevice, @nospecialize(y)) = typeof(Adapt.adapt(adaptor(dev.bq), y))

accesscache(dev::LavaDevice) = dev.ctx.caches.accesses::AccessCache

devicebuffertype(::LavaDevice, ::Type{T}) where {T} = LavaDeviceArray{T,1}

"""
What the kernel does to each dispatch argument, inferred through Lava's method
table.

Two leading entries are dropped on the way out: `accessof` reports the function
itself at 1, and a KA kernel is compiled at `(ctx, args...)` — the iteration
context is the kernel's, not the caller's, so neither is an argument the caller
declared. A compute kernel may not close over a device array at all (`dispatch!`
refuses one), so nothing is lost with the first.

`group` is threaded through because the iteration context's TYPE depends on it:
a two-dimensional ndrange and a one-dimensional one give different
`CompilerMetadata` parameters, and inferring the kernel at the wrong one infers
different code.
"""
function kerneltouches(dev::LavaDevice, kernel, args::Tuple, ndrange, group)
    obj  = kernelfor(kernel, group, backend(dev))
    iter = get_or_build_iter_plan(obj, dispatchrange(ndrange), nothing, dev.ctx)
    tt   = (typeof(iter.ka_ctx), map(a -> devicetype(dev, a), args)...)
    interp = Lava.kernelinterpreter(obj.f, Tuple{tt...}; workgroup_size = iter.ws_3d)
    return accessof(interp, obj.f, tt)[3:end]
end

isdevicearray(::LavaArray) = true

"""
The union over a trace's shaders.

`chit_miss_take_args` decides who sees the arguments at all: with it set, the
closest-hit and miss shaders are compiled against the raygen's own signature and
read the same queues, which is the pbrt-v4 pattern Hikari is on; without it they
take none and only the raygen's accesses are the pass's. The any-hit always
takes them — `compile_rt_pipeline` compiles it at `raygen_tt` either way.
"""
function kerneltouches(dev::LavaDevice, pipe::RayTracingPipeline, args::Tuple, ndrange, group)
    argT = map(a -> devicetype(dev, a), args)
    merged = fill(NOTOUCH, length(args))
    for (sh, witharg) in shaderargs(pipe)
        witharg || continue
        ts = accessof(Lava.kernelinterpreter(sh, Tuple{argT...}), sh, argT;
                      cache = accesscache(dev))
        for i in eachindex(merged)
            merged[i] = merged[i] | ts[i + 1]
        end
    end
    return merged
end

"""What each shader does to what it closed over, in `shaders` order."""
function shadertouches(dev::LavaDevice, pipe::RayTracingPipeline, args::Tuple)
    argT = map(a -> devicetype(dev, a), args)
    return [begin
                tt = witharg ? argT : ()
                accessof(Lava.kernelinterpreter(sh, Tuple{tt...}), sh, tt; cache = accesscache(dev))[1]
            end
            for (sh, witharg) in shaderargs(pipe)]
end

"""Each shader beside whether it is compiled with the raygen's arguments."""
shaderargs(p::RayTracingPipeline) =
    ((p.raygen_func, true),
     ((c, p.chit_miss_take_args) for c in p.closesthit_funcs)...,
     (p.miss_func, p.chit_miss_take_args),
     (p.anyhit_func === nothing ? () : ((p.anyhit_func, true),))...)

# ── draws ─────────────────────────────────────────────────────────────────────
#
# `resolve_shader_pair` is what the compile calls, and it is called here for the
# same reason `kernelfor` is above: the thing compiled is a WRAPPER around the
# user's function — it supplies the varyings a fragment stage reads and the
# output plumbing a vertex stage writes — so the wrapper is what the arguments
# line up against, and the raw function's arity does not even match.

function vertextouches(dev::LavaDevice, shader, args::Tuple)
    argT = map(a -> devicetype(dev, a), args)
    tt = Tuple{argT...}
    vfn, _, _, _ = resolve_shader_pair(shader, tt, Tuple{})
    return accessof(Lava.kernelinterpreter(vfn, tt), vfn, argT; cache = accesscache(dev))[2:end]
end

function fragmenttouches(dev::LavaDevice, shader, args::Tuple)
    argT = map(a -> devicetype(dev, a), args)
    tt = Tuple{argT...}
    _, _, ffn, _ = resolve_shader_pair(shader, Tuple{}, tt)
    return accessof(Lava.kernelinterpreter(ffn, tt), ffn, argT; cache = accesscache(dev))[2:end]
end
