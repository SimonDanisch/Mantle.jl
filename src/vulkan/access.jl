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
    obj  = kernelfor(kernel, group, dev)
    iter = get_or_build_iter_plan(obj, dispatchrange(ndrange), callgroup(obj, group), dev)
    return kaaccesssignature(obj, iter, argT)
end

"""The signature a built KA kernel's accesses are inferred at, for iteration plan `iter`."""
function kaaccesssignature(obj, iter::IterPlan, argT::Tuple)
    tt     = (typeof(iter.ka_ctx), argT...)
    interp = Lava.kernelinterpreter(obj.f, Tuple{tt...}; workgroup_size = iter.ws_3d)
    return interp, obj.f, tt
end

# A KA kernel is inferred at its launch's workgroup size, which Lava compiles in;
# the launch is flat (`get_or_build_iter_plan`).
kernelinterpreter(::LavaDevice, kernel, tt, threads::Integer) =
    Lava.kernelinterpreter(kernel, tt; workgroup_size = (Int(threads), 1, 1))

"""
    cachedtouches(dev, sig, skip) -> view or nothing

What the walk already found for `sig`, past its first `skip` entries (the kernel
function, and a KA kernel's iteration context). An eager launch asks on every call,
and building the interpreter and walk again only to reach this entry allocated more
than the rest of the launch (`test_dispatch_allocation.jl`).
"""
function cachedtouches(dev::LavaDevice, @nospecialize(sig::Type), skip::Int)
    summary = get(summaries!(accesscache(dev)), sig, nothing)
    summary === nothing && return nothing
    holds(summary, Base.get_world_counter()) || return nothing
    touches = summary.touches
    return view(touches, (skip + 1):length(touches))
end

"""
What an eagerly launched KA kernel does to each of `args` at iteration plan `iter`.
The graph path asks `argument_usage` with the kernel's constructor; an eager launch
only has the built kernel.
"""
function katouches(obj::KA.Kernel{LavaDevice}, args::Tuple, iter::IterPlan)
    dev = obj.backend
    argT = map(a -> devicetype(dev, a), args)
    cached = cachedtouches(dev, Tuple{typeof(obj.f), typeof(iter.ka_ctx), argT...}, 2)
    cached === nothing || return cached
    interp, body, tt = kaaccesssignature(obj, iter, argT)
    t = accessof(interp, body, tt; cache = accesscache(dev))
    return view(t, 3:length(t))
end

"""
What an eagerly launched KernelInterface kernel does to each of `args`. A view either
way: the launch's closure captures it, and a union-typed capture is boxed.
"""
function kitouches(dev::LavaDevice, kern::K, args::Tuple, ndrange) where {K}
    cached = cachedtouches(dev, Tuple{K, map(a -> devicetype(dev, a), args)...}, 1)
    cached === nothing || return cached
    t = argument_usage(dev, kern, args, ndrange, nothing)
    return view(t, 1:length(t))
end

isdevicearray(::LavaArray) = true
resource_identity(a::LavaArray)=a.buf.rc

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
