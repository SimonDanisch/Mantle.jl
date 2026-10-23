# The signature a dispatch's accesses are inferred at, on this backend. Core
# walks the IR (`graph/access.jl`); this adds what only a Lava dispatch knows:
# arguments are `Adapt`ed, a KA kernel takes its iteration context first, and
# the method table is Lava's.

# `Core.Typeof`: a type-valued argument is `Type{Float32}` in the compiled
# signature but `DataType` under `typeof`.
argtype(dev::LavaDevice, @nospecialize(y)) =
    Core.Typeof(Adapt.adapt(adaptor(dev.default_bq), y))

accesscache(dev::LavaDevice) = dev.caches.accesses::AccessCache

devicebuffertype(::LavaDevice, ::Type{T}, N::Int) where {T} = LavaDeviceArray{T,N}

"""Lava's interpreter, the one a macro-free kernel is compiled through."""
kernelinterpreter(::LavaDevice, kernel, tt) = Lava.kernelinterpreter(kernel, tt)

# The iteration context comes from the cached `IterPlan`. Its type depends on
# `group` (a 1-D and a 2-D ndrange give different `CompilerMetadata`), and the
# kernel is inferred at the plan's workgroup size.
function kakernelaccesssignature(dev::LavaDevice, kernel, argT::Tuple, ndrange, group)
    obj  = kernelfor(kernel, group, backend(dev))
    iter = get_or_build_iter_plan(obj, dispatchrange(ndrange), callgroup(obj, group), dev)
    tt   = (typeof(iter.ka_ctx), argT...)
    interp = Lava.kernelinterpreter(obj.f, Tuple{tt...}; workgroup_size = iter.ws_3d)
    return interp, obj.f, tt
end

isdevicearray(::LavaArray) = true

"""
The union of what a trace's shaders do to `args`; every shader takes them
(`rt_stages`).
"""
function kerneltouches(dev::LavaDevice, pipe::RayTracingPipeline, args::Tuple, ndrange, group)
    argT = map(a -> devicetype(dev, a), args)
    merged = fill(NOTOUCH, length(args))
    for sh in shaders(pipe)
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
    return [accessof(Lava.kernelinterpreter(sh, Tuple{argT...}), sh, argT; cache = accesscache(dev))[1]
            for sh in shaders(pipe)]
end

# Draw stages are compiled as wrappers (`resolve_shader_pair`) that add the
# varyings and outputs, so accesses are inferred on the wrapper; the user
# function's arity does not match the arguments.
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

# Lava's `_lava_coopmat_<op>_...` / `_lava_tensor_<op>_...` intrinsics are
# external calls the IR walk cannot classify, and unclassified means read+write
# (needless barriers on `@Const` weights). Declared here because Lava cannot
# extend `Mantle.intrinsic_usage`. `load*` ops read, `store*` ops write, other
# ops take no pointer; any other name is `nothing`, a refusal, not a guess.
function intrinsic_usage(::VulkanAPI, name::Symbol)
    s = String(name)
    op = if startswith(s, "_lava_coopmat_")
        SubString(s, ncodeunits("_lava_coopmat_") + 1)
    elseif startswith(s, "_lava_tensor_")
        SubString(s, ncodeunits("_lava_tensor_") + 1)
    else
        return nothing
    end
    startswith(op, "load") && return READ
    startswith(op, "store") && return WRITE
    return nothing
end
