# The signature a dispatch is inferred at, on this backend.
#
# Core walks the IR (`graph/access.jl`); what is here is the three things only a
# Vulkan/Lava dispatch knows: that arguments are `Adapt`ed on their way to the
# shader, that a `KernelAbstractions` kernel takes its iteration context as a
# leading argument, and that the method table the kernel will be compiled
# against is Lava's.

# `Core.Typeof`, for the reason core's default gives: a type-valued argument is
# `Type{Float32}` in the signature the kernel is compiled at and `DataType` under
# `typeof`, and the latter is not a dispatch tuple element.
argtype(dev::LavaDevice, @nospecialize(y)) =
    Core.Typeof(Adapt.adapt(adaptor(dev.bq), y))

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
    argT = map(a -> devicetype(dev, a), args)
    # The same split `compile_dispatch` makes, and for the same reason: what a
    # dispatch MEANS is core's question (`buildskernel`), and a macro-free kernel
    # is compiled at its arguments alone — no constructor, no iteration context.
    if !buildskernel(kernel, backend(dev))
        interp = Lava.kernelinterpreter(kernel, Tuple{argT...})
        return accessof(interp, kernel, argT; cache = accesscache(dev))[2:end]
    end
    obj  = kernelfor(kernel, group, backend(dev))
    iter = get_or_build_iter_plan(obj, dispatchrange(ndrange), callgroup(obj, group), dev.ctx)
    tt   = (typeof(iter.ka_ctx), argT...)
    interp = Lava.kernelinterpreter(obj.f, Tuple{tt...}; workgroup_size = iter.ws_3d)
    return accessof(interp, obj.f, tt; cache = accesscache(dev))[3:end]
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

# ── What Lava's device intrinsics do, declared ────────────────────────────────
#
# A cooperative-matrix load is a read of the buffer its pointer names, and the
# only thing that knows so is the generator that emitted
# `declare i32 @_lava_coopmat_load_f16_16x16_a(i64, i32)`. There is nothing for
# the walk to read: the `@generated` wrapper is inlined by the time it sees the
# IR, so the Julia callee is gone, and the module holds a `call` to an external
# symbol and no memory instruction.
#
# It used to be classified by looking for `"load "` in the LLVM text, which that
# name misses by a space, so it fell through to read+write. Measured on
# `gemm_cm2!`: A and B came out `Touch(true, true, false)` with `@Const` on both
# in the source, so every pass sharing a weight matrix with another got a
# barrier it did not need.
#
# These belong in Lava, next to the generators, and cannot go there: Mantle
# depends on Lava and not the other way round, so `Mantle.intrinsic_usage` is not
# a name Lava can extend. This file is where the Vulkan backend already states
# what only it knows about a Lava dispatch.
#
# Keyed on the OP, which is what both naming schemes put first and what Lava's
# own emitter parses them back out as (`spirv/coopmat.jl`):
#
#     _lava_coopmat_<op>_<dtype>_<MxN>_<use><scope>[_row]
#     _lava_tensor_<op>_<dim>_<clamp>[_<rest>]
#
# `load`, `loadw`, `loadw2`, `loadw4`, `loadv` are the loads and `store`,
# `storew`, `storev` the stores, which a prefix covers exactly. Every other op --
# `create`, `setdim`, `setstride`, `setclampvalue`, `slice`, `view`, `muladd` --
# takes no pointer, so the walk short-circuits before asking. One that took a
# pointer and was not named for what it does would answer `nothing` here, and
# `nothing` is a refusal rather than a guess.

function intrinsic_usage(name::Symbol)
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
