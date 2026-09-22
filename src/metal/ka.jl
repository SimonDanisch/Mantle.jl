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

"""
Declare this backend's fastest product for these operands.

Apple's MPSGraph product where it covers the operands and the activation, this
backend's `matmul2d` kernel otherwise, with the parts the chosen one could not fold
reported back so the caller declares them as passes.

`librarygemm` is asked first by the declaring caller and answers for exactly the same
cases, so what reaches HERE is either a product it never asked about — the
convolution's im2col GEMM — or one the library cannot express. Both routes are tried
in the same order either way, which is why the library is offered again here rather
than left to the one call site that happens to ask for it.
"""
function Mantle.native_gemm_dispatch!(dev::MetalDevice, g, out, A, B;
                                      bias = nothing, epilogue = identity, name)
    denseoperands(out, A, B, bias) || return nothing
    lib = Mantle.librarygemm(dev, out, A, B, bias, epilogue)
    if lib !== nothing
        dispatch!(g, lib, (out, A, B, bias); name)
        return (; bias = bias !== nothing, epilogue = true)
    end
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
                                               coldiv = nothing,
                                               name)
    denseoperands(out, A, B) || return false
    config = Metal.batched_gemm_kernel_config(
        out, A, B; transpose_a, transpose_b, alpha, coldiv)
    config === nothing && return false
    dispatch!(g, config.kernel, config.args, config.ndrange;
              group = config.group, name)
    return true
end

# ── Apple's libraries, as pass members ───────────────────────────────────────
#
# MPSGraph builds a pipeline and encodes it; there is no Julia kernel to record, so
# these are CALLS (`dispatch!` with no ndrange) and a recording keeps a hole for each.
# See `runscalls` and `replaywithcalls!` in `device.jl`/`record.jl` for how a hole is
# replayed, and `Mantle.Call` for why it is core's type rather than one of ours.
#
# A type per operation rather than a closure, because `argument_usage` has to be
# declarable: a call is opaque to the access walk — the body is on the other side of
# the boundary — and core's answer to an undeclared argument is read-AND-write, which
# would order every product after every other one and serialise the frame.

"""
Apple's MPSGraph matrix product, `out = act.(A * B .+ bias)`, as a pass member.

The activation rides on the callable — a `Symbol` naming which one, from
`Mantle.activationkind` — rather than being a fourth argument, because it is decided
at declare time and is not a resource.
"""
struct MPSGraphGemm
    act::Symbol
end

Mantle.argument_usage(::Type{MPSGraphGemm}, ::Type{<:Tuple{Any,Any,Any,Any}}) =
    (Mantle.WRITE, Mantle.READ, Mantle.READ, Mantle.READ)

(c::MPSGraphGemm)(out, A, B, bias) =
    (Metal.MPSGraphs.gemm_batched!(out, A, B, bias, c.act); nothing)

"""
Apple's fused scaled dot-product attention as a pass member.

The three layout keys ride on the callable rather than in the argument list. An
operand is often a strided window on a shared projection buffer — q, k and v are
three windows on one array — and what a call passes are RESOURCES, so the window is
described here, once, at declare time. `Metal.MPSGraphs.packoperand` says which
windows qualify and the graph does the narrowing.
"""
struct MPSGraphAttention{Q,K,V}
    scale::Float32
    q::Q
    k::K
    v::V
end

Mantle.argument_usage(::Type{<:MPSGraphAttention}, ::Type{<:Tuple{Any,Any,Any,Any}}) =
    (Mantle.WRITE, Mantle.READ, Mantle.READ, Mantle.READ)

(c::MPSGraphAttention)(out, q, k, v) =
    (Metal.MPSGraphs.sdpa_batched!(out, q, k, v, c.q, c.k, c.v, c.scale); nothing)

"""
Apple's own product where it covers the operands, which is measurably faster than
this backend's best recordable kernel.

The ACTIVATION goes into the graph too, where MPSGraph has a node for it: an
activation applied to a product already rounded to half loses accuracy twice, so the
graph widens to Float32 after the product and rounds once, which is what the kernel
this replaces does with its accumulator. `Mantle.activationkind` is how the
activation is named without this backend knowing whose function it is.

Measured on an M5 over SAM 2.1's encoder, fp16, the 195 `addmm` products of one
frame: 213 ms through `gemm_tensor!` against 130 through MPSGraph. What is left for
`native_gemm_dispatch!` — asked next by the caller — is a product MPSGraph does not
cover, or one whose activation it has no node for.
"""
function Mantle.librarygemm(d::MetalDevice, out, A, B, bias, epilogue)
    # The question is about this device's RUN path, and it is core's to answer.
    Mantle.runscalls(d) || return nothing
    denseoperands(out, A, B, bias) || return nothing
    act = Mantle.activationkind(epilogue)
    Metal.MPSGraphs.gemm_shape_supported(out, A, B, bias, act) || return nothing
    return MPSGraphGemm(act)
end

"""
    denseoperands(xs...) -> Bool

Whether every operand is one this backend can hand a dense kernel or Apple's
library: a graph resource it will place, or an array it already holds.

`gemm_shape_supported` and its siblings are deliberately duck-typed — they are asked
of graph RESOURCES, before anything is placed, so nothing in them is more specific
than `eltype`, `ndims` and `size`. A caller's own wrapper answers all three like a
matrix without being one. Qwen-Image 2.1's text encoder projects through a packed
int8 weight whose `size` and `eltype` are its LOGICAL ones and whose storage is
`UInt32`; it passed every shape test and then found no `gemm_batched!` method at
replay, thirteen gigabytes into the run.

Asked by every hook here, not just the library ones: `gemm_tensor_kernel!` is typed
`MtlDeviceArray` too, so declining to the library and then handing the same operand
to this backend's own kernel only moves the `MethodError`. A caller that packs its
own weights has its own product for them, and what it needs from these hooks is a
`nothing`.

Here rather than in `Metal.MPSGraphs`, because this is the one place that knows both
vocabularies: what a graph resource is, and what an `MtlArray` is. The library knows
only the second and the caller only the first.
"""
denseoperands(xs...) = all(xs) do x
    x === nothing && return true
    x isa Mantle.Buffer || x isa Mantle.TransientBuffer ||
        x isa Mantle.ResourceView || x isa Metal.MtlArray
end

"""Apple's MPSGraph 2-D convolution, `out = act.(conv(x, w) .+ bias)`, as a pass
member. The geometry rides on the callable, like the activation: it is fixed when the
op is declared and none of it is a resource."""
struct MPSGraphConv2D
    stride::NTuple{2,Int}
    pad::NTuple{2,Int}
    dilation::NTuple{2,Int}
    groups::Int
    act::Symbol
    # A transposed convolution is the SAME MPSGraph node's data gradient, so it is a
    # field here rather than a second callable: the operands, the descriptor and the
    # epilogue are identical and only which node is built differs.
    transposed::Bool
end

Mantle.argument_usage(::Type{MPSGraphConv2D}, ::Type{<:Tuple{Any,Any,Any,Any}}) =
    (Mantle.WRITE, Mantle.READ, Mantle.READ, Mantle.READ)

(c::MPSGraphConv2D)(out, x, w, bias) =
    (Metal.MPSGraphs.conv2d_batched!(out, x, w, bias, c.stride, c.pad, c.dilation,
                                     c.groups, c.act; c.transposed); nothing)

"""
Apple's direct convolution where it covers the operands.

Nothing recordable on this backend competes with it. The alternative is an im2col
matrix materialised into a transient and a product over a padded reduction axis —
20.0 MiB written and read back before SAM 2.1's `7x7x3` stem starts, whose product
alone was the single dearest pass in the frame at 6.35 ms.

`nothing` when the graph cannot express it, and then the caller's own lowering runs:
this declines rather than falling back, because a convolution has four of those and
choosing between them is the caller's arithmetic, not a backend's.
"""
function Mantle.native_conv2d_dispatch!(dev::MetalDevice, g, out, x, w;
                                        bias = nothing, stride, pad, dilation,
                                        groups, epilogue = identity,
                                        transposed = false, name)
    Mantle.runscalls(dev) || return nothing
    denseoperands(out, x, w, bias) || return nothing
    act = Mantle.activationkind(epilogue)
    ok(b) = Metal.MPSGraphs.conv2d_shape_supported(out, x, w, b, act;
                                                   transposed, groups = Int(groups))
    # A bias MPSGraph cannot BIND is not a reason to hand back the whole
    # convolution. One value per output channel is a single axis, so there is no flat
    # shape to fall back to and a channel count whose bytes are not a multiple of
    # sixteen cannot be bound at an offset -- RIFE's decoder has 52 channels, which is
    # 104 bytes. The `(; bias, epilogue)` answer exists precisely so a caller can be
    # told which post-operations were folded, and adding a bias afterwards is one
    # elementwise pass over the result. Refusing instead cost seven transposed
    # convolutions their library node and 438 ms of a 581 ms frame.
    fold = bias
    if !ok(fold)
        (fold === nothing || !ok(nothing)) && return nothing
        fold = nothing
    end
    dispatch!(g, MPSGraphConv2D(Tuple(Int.(stride)), Tuple(Int.(pad)),
                                Tuple(Int.(dilation)), Int(groups), act,
                                Bool(transposed)),
              (out, x, w, fold); name)
    return (; bias = fold !== nothing, epilogue = true)
end

"""
Declare attention: Apple's fused op where it fits, this backend's fused kernel
otherwise.

Measured on an M5 over the same frame, the 42 attention ops: 78 ms through the
`matmul2d` kernel against 24 ms through `scaledDotProductAttentionWithQueryTensor`.

A STRIDED operand needs no copy on either route. SAM 2.1's projection leaves q, k
and v as three windows on one buffer, and both the library and the kernel read them
where they lie — the library because `sdpa_operands` hands the dense array to the
graph and narrows it there, the kernel because a tensor descriptor carries a leading
dimension. Materialising three operands per attention op would have been most of
what the library saves.
"""
function Mantle.native_attention_dispatch!(dev::MetalDevice, g, out, q, k, v;
                                          scale, name)
    if Mantle.runscalls(dev) && denseoperands(out, q, k, v)
        ops = Metal.MPSGraphs.sdpa_operands(out, q, k, v)
        if ops !== nothing
            dispatch!(g, MPSGraphAttention(Float32(scale), ops.keys...),
                      (out, ops.res...); name)
            return true
        end
    end
    config = Metal.attention_kernel_config(out, q, k, v; scale)
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
