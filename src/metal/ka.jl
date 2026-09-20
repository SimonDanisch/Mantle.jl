# Running a Mantle graph on Metal.
#
# Two methods. That is the whole compute path, and it is the measurement this
# refactor was for: the Vulkan backend's equivalent is `array/ka_backend.jl`,
# 1,122 lines, because it implements a KernelAbstractions backend from scratch —
# `LavaBackend <: KA.GPU`, `launch_config`, `mkcontext`, ndrange padding, the
# kernel object, indirect dispatch.
#
# None of that is needed here twice over. Metal.jl already IS a KA backend, so
# the launch machinery is its; and `graph/kalaunch.jl` already IS the plan
# execution, shared with the host backend, so the baking is Mantle's. What is
# genuinely this backend's is how a graph resource becomes a kernel argument,
# and whether barriers have to be emitted.

# ── Why an interpreted run needs no barriers, and a recorded one does ────────
#
# Not because every pooled allocation is `Shared` (images are
# `PrivateStorage`), and not because the driver's hazard tracking stands in for
# something unfinished.
#
# It is COMMIT ORDER. Every render and copy pass opens its own
# `MTLCommandBuffer` and commits it before the call returns, all on one queue,
# and Metal runs command buffers on a queue in the order they were committed.
# For a plan walked per frame, the order the graph scheduled IS the order they
# execute, with nothing between them to arrange — which is why
# `emitbarriers!(::Immediate, …)`, core's no-op, is still the right answer for
# that path.
#
# A RECORDED plan has no commit order to lean on. Its commands live in one
# indirect command buffer and run CONCURRENTLY unless a command carries a
# barrier (`Metal/test/indirect_command_buffer.jl` measures it: 32 serialised
# read-modify-writes sum to 32, the same 32 concurrent sum short). So the graph
# has to derive the hazards after all, and this backend takes the portable rule
# in `sync/transition.jl` — read after read needs nothing, anything with a write
# in it does — rather than answering `false` and deriving none. `passbarriers`
# in `record.jl` is where they come back out.
#
# Nothing about the heap changes. Measured 2026-09-08, a chained graph (render
# A, copy A, render B, copy B) at 2048x2048, median of five runs of forty
# frames:
#
#     Tracked      0.364 ms/frame     4194304 pixels correct
#     Untracked    0.332 ms/frame     4194304 pixels correct
#
# Hazard tracking costs about 9% and buys nothing for the interpreted path, and
# the heap stays `Tracked` anyway: untracked is only safe BECAUSE of the one
# command buffer per pass, and object reuse batching two passes into one would
# turn the 9% into a race.


syncbackend(::MetalDevice) = MetalAPI()

# Metal's simdgroup matrix load/store are opaque AIR intrinsics after inlining:
# the IR contains no ordinary LLVM load or store for the access walker to read.
# Their globally unique symbol names are therefore the authoritative direction.
function intrinsic_usage(name::Symbol)
    s = String(name)
    startswith(s, "air.simdgroup_matrix_8x8_load.") && return READ
    startswith(s, "air.simdgroup_matrix_8x8_store.") && return WRITE
    # Inline tensors are descriptors over the bound device arrays.  The
    # initializer preserves that resource dependency through an opaque AIR
    # call, and matmul2d consumes both operands while writing its destination.
    # `intrinsic_usage` applies to every traced resource argument, so ATOMIC is
    # the conservative combined read/write classification for the latter.
    startswith(s, "air.init_strided_private_tensor.i32.global") && return READ
    startswith(s, "__tensorops_impl_matmul2d_op_run") && return ATOMIC
    if startswith(s, "air.atomic.global.") || startswith(s, "air.atomic.local.")
        occursin(".load.", s) && return READ
        occursin(".store.", s) && return WRITE
        return ATOMIC
    end
    s == "__metal_linked_closest" && return READ
    s == "__metal_linked_any" && return READ
    return nothing
end

# `resolve` is NOT overridden here. Core's default is `storage(x)`, and that is
# already right: a `Buffer`'s storage is `deviceview(dev, store)` — the borrowed
# `MtlArray` over its pool region — and a transient's is what `materialize!`
# below put in `block`. The hook exists; this backend has nothing to say to it.

"""
Give a transient its slice of the pool block.

`block` holds the borrowed `MtlArray` a launch will pass, built once here
rather than per launch — core's `storage(::TransientBuffer)` hands it straight
back.

The slab is an `MTLBuffer`, where the host backend's is a `Vector{UInt8}` and
the Vulkan backend's is a `BufferBlock`. Same hook, three storages.
"""
function deviceslice(::MetalDevice, ::Type{T}, dims::Dims{N},
                     buf::MTL.MTLBuffer, off::Int) where {T,N}
    ref = GPUArrays.DataRef(_ -> nothing, buf)
    MtlArray{T,N}(ref, dims; maxsize = Int(buf.length) - off, offset = off)
end

"""
16 bytes, not Vulkan's 256 and not the host's cache line.

Metal has no `minStorageBufferOffsetAlignment`: a buffer binding needs only its
element type's natural alignment, and 16 covers every vector type MSL has.
Over-aligning would waste arena space the placer could have packed.
"""
alignment(::MetalDevice, ::TransientBuffer) = 16


# ── What a pass touches, read off its kernels ────────────────────────────────
#
# Core walks the IR (`graph/access.jl`); what is here is the three things only a
# Metal dispatch knows: that arguments are `mtlconvert`ed on their way to the
# shader, that a `KernelAbstractions` kernel takes its iteration context as a
# leading argument, and that the method table the kernel will be compiled
# against is Metal.jl's, overlays and all.

Mantle.argtype(::MetalDevice, @nospecialize(y)) = Core.Typeof(Metal.mtlconvert(y))

# What `Adapt.adapt_storage(::Metal.Adaptor, ::MtlArray)` produces, which is what
# a kernel handed a materialised transient receives.
Mantle.devicebuffertype(::MetalDevice, ::Type{T}, N::Int) where {T} =
    Metal.MtlDeviceArray{T,N,Metal.AS.Device}

Mantle.isdevicearray(::Metal.MtlArray) = true

"""Whether Metal's native SIMD-group GEMM covers these element types."""
Mantle.native_gemm_available(::Union{MetalDevice,Metal.MetalBackend},
                             ::Type{A}, ::Type{B}, ::Type{C}) where {A,B,C} =
    Metal.gemm_simd_eltype(A, B, C)

"""Declare Metal.jl's fastest recordable GEMM so a Mantle plan can bake it."""
function Mantle.native_gemm_dispatch!(dev::MetalDevice, g, out, A, B;
                                      bias = nothing, epilogue = identity, name)
    config = Metal.gemm_kernel_config(out, A, B; bias, epilogue)
    config === nothing && return nothing
    dispatch!(g, config.kernel, config.args, config.ndrange;
              group = config.group, name)
    return config.fused
end

"""Declare Metal's tensor-ops strided-batched GEMM into a Mantle graph."""
function Mantle.native_batched_gemm_dispatch!(dev::MetalDevice, g, out, A, B;
                                               transpose_a = false,
                                               transpose_b = false,
                                               alpha = 1,
                                               name)
    config = Metal.batched_gemm_kernel_config(
        out, A, B; transpose_a, transpose_b, alpha)
    config === nothing && return false
    dispatch!(g, config.kernel, config.args, config.ndrange;
              group = config.group, name)
    return true
end

Mantle.accesscache(d::MetalDevice) = d.accesses

"""
The interpreter whose method table this backend's kernels are compiled against.

Metal.jl's, not Julia's: `@device_override` puts `vertex_index`, `clip_y`,
`sample_texture_2d` and the rest in an overlay table, and a walk through the
host's would see the host bodies — which for those names are `error` calls, so
every access they reach would be missed.
"""
function metalinterpreter(@nospecialize(f), @nospecialize(tt))
    config = Metal.compiler_config(Metal.device())
    source = GPUCompiler.methodinstance(Core.Typeof(f), tt)
    return GPUCompiler.get_interpreter(GPUCompiler.CompilerJob(source, config))
end

"""
What the kernel does to each dispatch argument, inferred through Metal's method
table.

The leading entries `accessof` reports are dropped the same way the Vulkan side
drops them: index 1 is the function itself, and a KA kernel is compiled at
`(ctx, args...)` where the iteration context is the kernel's own rather than
anything the caller declared. A macro-free kernel has neither.

`group` is threaded through because the context's TYPE depends on it — the same
`launch_config`/`mkcontext` pair `compile_dispatch` uses, so the signature walked
here is the signature compiled there.
"""
function Mantle.kerneltouches(dev::MetalDevice, kernel, args::Tuple, ndrange, group)
    argT = map(a -> Mantle.devicetype(dev, a), args)
    if !Mantle.buildskernel(kernel, Mantle.backend(dev))
        interp = metalinterpreter(kernel, Tuple{argT...})
        return Mantle.accessof(interp, kernel, argT; cache = Mantle.accesscache(dev))[2:end]
    end
    obj = Mantle.kernelfor(kernel, group, Mantle.backend(dev))
    nd = Mantle.recordedrange(ndrange)
    ndr, _ws, iterspace, _ = KA.launch_config(obj, nd, Mantle.callgroup(obj, group))
    ctx = KA.mkcontext(obj, ndr, iterspace)
    tt = (Core.Typeof(Metal.mtlconvert(ctx)), argT...)
    interp = metalinterpreter(obj.f, Base.to_tuple_type(tt))
    return Mantle.accessof(interp, obj.f, tt; cache = Mantle.accesscache(dev))[3:end]
end


"""
What a vertex stage does to each of its arguments.

The BODY, not the stage wrapper: `MetalVertexStage` exists to write the output
struct, and what it stores through is the pointer the driver hands it, not
anything the caller declared.

A body that declares the leading `VertexIndex` is walked with one, because that
is the signature it is compiled at — asked of the method table exactly as
`stage_signatures` asks it, so the two cannot disagree. Its entry is dropped
with the function's own.
"""
function Mantle.vertextouches(dev::MetalDevice, shader, args::Tuple)
    f = Mantle.stagefunction(shader.vertex)
    argT = map(a -> Mantle.devicetype(dev, a), args)
    lead = KI.wantsvertexindex(f, argT) ? (KI.VertexIndex,) : ()
    return stagetouches(dev, f, args; lead)
end

"""
What a fragment stage does to each of its arguments.

A fragment body's FIRST parameter is the varyings — `MetalFragmentStage` builds
that NamedTuple from the interpolated values and the body reads it — so the
signature walked here leads with `varying_type`, exactly as `stage_signatures`
compiles it. Its entry is dropped with the function's own; a varying is a value
the rasteriser produced, not a resource anyone declared.
"""
function Mantle.fragmenttouches(dev::MetalDevice, shader, args::Tuple)
    f = Mantle.stagefunction(shader.fragment)
    return stagetouches(dev, f, args; lead = (varying_type(shader),))
end

function stagetouches(dev::MetalDevice, f, args::Tuple; lead::Tuple = ())
    argT = map(a -> Mantle.devicetype(dev, a), args)
    tt = (lead..., argT...)
    interp = metalinterpreter(f, Base.to_tuple_type(tt))
    touches = Mantle.accessof(interp, f, tt; cache = Mantle.accesscache(dev))
    # Index 1 is `f` itself, and everything in `lead` is the STAGE's rather than
    # the caller's, so all of it goes together.
    return touches[(2 + length(lead)):end]
end
