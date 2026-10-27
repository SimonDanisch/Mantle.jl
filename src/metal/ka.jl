# Immediate passes submit in graph order on one queue with tracked resources.
# Recorded ICB commands can execute concurrently, so their read/write hazards are
# derived and their barriers emitted. Retain tracked heaps when changing
# submission boundaries.

syncbackend(::MetalDevice) = MetalAPI()

# Access directions of opaque AIR intrinsics, which have no LLVM load or store
# for the access walker to read; their symbol names are unique.
function intrinsic_usage(::MetalAPI, name::Symbol)
    s = String(name)
    startswith(s, "air.simdgroup_matrix_8x8_load.") && return READ
    startswith(s, "air.simdgroup_matrix_8x8_store.") && return WRITE
    # Tensor descriptors retain operand dependencies; matmul consumes and writes them.
    startswith(s, "air.init_strided_private_tensor.i32.global") && return READ
    startswith(s, "__tensorops_impl_matmul2d_op_run") && return ATOMIC
    if startswith(s, "air.atomic.global.") || startswith(s, "air.atomic.local.")
        occursin(".load.", s) && return READ
        occursin(".store.", s) && return WRITE
        return ATOMIC
    end
    # Mesh-stage outputs write through the stage-owned leading object argument.
    (s == "air.set_position_mesh" || s == "air.set_index_mesh" ||
     s == "air.set_primitive_count_mesh" ||
     startswith(s, "air.set_vertex_data_mesh.") ||
     startswith(s, "air.set_primitive_data_mesh.")) && return WRITE
    return nothing
end

"""Borrow a typed slice of the pool's MTLBuffer; the DataRef never frees it."""
function deviceslice(d::MetalDevice, ::Type{T}, dims::Dims{N},
                     buf::MTL.MTLBuffer, off::Int) where {T,N}
    checkspan(Int(buf.length),off,checkedbytes(T,dims))
    ref = GPUArrays.DataRef(_ -> nothing, managedbuffer(d, buf))
    MtlArray{T,N}(ref, dims; maxsize = Int(buf.length) - off, offset = off)
end

# Metal buffer bindings need natural element alignment; 16 covers MSL vectors.
alignment(::MetalDevice, ::TransientBuffer) = 16

# Infer through the same converted argument ABI the shader receives.
Mantle.argtype(::MetalDevice, @nospecialize(y)) = Core.Typeof(Metal.mtlconvert(y))

# What `Adapt.adapt_storage(::Metal.Adaptor, ::MtlArray)` produces: the type a
# kernel receives for a materialised transient.
Mantle.devicebuffertype(::MetalDevice, ::Type{T}, N::Int) where {T} =
    Metal.MtlDeviceArray{T,N,Metal.AS.Device}

Mantle.isdevicearray(::Metal.MtlArray) = true

# Include derived-array offsets when choosing aligned wide loads.
Mantle.basealignment(a::Metal.MtlArray) =
    Mantle.powalign(UInt64(a.data[].buffer.gpuAddress) + UInt64(a.offset))

"""Whether Metal's native SIMD-group GEMM covers these element types."""
Mantle.native_gemm_available(::MetalDevice,
                             ::Type{A}, ::Type{B}, ::Type{C}) where {A,B,C} =
    Metal.gemm_simd_eltype(A, B, C)

"""Declare MPSGraph GEMM when eligible, otherwise a native kernel. Report which
post-operations were fused so callers can declare the remainder. Im2col callers
may enter here without asking librarygemm first, so both routes are considered."""
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

# Library calls are opaque graph members with explicit resource access directions.
# Operation configuration lives on the callable; resources remain its arguments.
"""MPSGraph matrix product with a declaration-time activation kind."""
struct MPSGraphGemm
    act::Symbol
end

Mantle.argument_usage(::Type{MPSGraphGemm}, ::Type{<:Tuple{Any,Any,Any,Any}}) =
    (Mantle.WRITE, Mantle.READ, Mantle.READ, Mantle.READ)

(c::MPSGraphGemm)(out, A, B, bias) =
    (Metal.MPSGraphs.gemm_batched!(out, A, B, bias, c.act); nothing)

"""MPSGraph attention. Layout keys describe strided projection windows;
packoperand determines eligibility before the graph narrows their storage."""
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

"""Return a supported MPSGraph GEMM call or nothing. The library fuses activation
before the final output rounding; the epilogue is named through activationkind."""
function Mantle.librarygemm(d::MetalDevice, out, A, B, bias, epilogue)
    Mantle.runscalls(d) || return nothing
    denseoperands(out, A, B, bias) || return nothing
    act = Mantle.activationkind(epilogue)
    Metal.MPSGraphs.gemm_shape_supported(out, A, B, bias, act) || return nothing
    return MPSGraphGemm(act)
end

"""MPSGraph 2-D convolution, `out = act.(conv(x, w) .+ bias)`, as a pass member.
Geometry and activation are fixed at declaration and live on the callable."""
struct MPSGraphConv2D
    stride::NTuple{2,Int}
    pad::NTuple{2,Int}
    dilation::NTuple{2,Int}
    groups::Int
    act::Symbol
    # MPSGraph uses the same descriptor's data-gradient node for transposed convolution.
    transposed::Bool
end

Mantle.argument_usage(::Type{MPSGraphConv2D}, ::Type{<:Tuple{Any,Any,Any,Any}}) =
    (Mantle.WRITE, Mantle.READ, Mantle.READ, Mantle.READ)

(c::MPSGraphConv2D)(out, x, w, bias) =
    (Metal.MPSGraphs.conv2d_batched!(out, x, w, bias, c.stride, c.pad, c.dilation,
                                     c.groups, c.act; c.transposed); nothing)

"""Declare a supported MPSGraph convolution; return nothing for caller lowering.
Report fused bias/activation separately when a bias cannot bind at its offset."""
function Mantle.native_conv2d_dispatch!(dev::MetalDevice, g, out, x, w;
                                        bias = nothing, stride, pad, dilation,
                                        groups, epilogue = identity,
                                        transposed = false, name)
    Mantle.runscalls(dev) || return nothing
    denseoperands(out, x, w, bias) || return nothing
    act = Mantle.activationkind(epilogue)
    ok(b) = Metal.MPSGraphs.conv2d_shape_supported(out, x, w, b, act;
                                                   transposed, groups = Int(groups))
    # Keep an eligible library convolution when only bias needs a separate pass.
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

"""Declare supported MPSGraph attention, otherwise a native fused kernel.
Strided windows are read in place. Permuted descriptors may be declined so the
caller can materialize and retry the library path."""
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
    # The library cannot bind permuted descriptors; the caller owns materialize/retry.
    # Ordinary projection windows already took the no-copy library route above.
    Mantle.runscalls(dev) && any(x -> x isa NamedTuple, (q, k, v)) && return false
    config = Metal.attention_kernel_config(out, q, k, v; scale)
    config === nothing && return false
    dispatch!(g, config.kernel, config.args, config.ndrange;
              group = config.group, name)
    return true
end

Mantle.accesscache(d::MetalDevice) = d.accesses

"""Infer against Metal's compiler overlay, where device intrinsics have their
shader bodies rather than host error stubs."""
function metalinterpreter(@nospecialize(f), @nospecialize(tt))
    config = Metal.compiler_config(Metal.device())
    source = GPUCompiler.methodinstance(Core.Typeof(f), tt)
    return GPUCompiler.get_interpreter(GPUCompiler.CompilerJob(source, config))
end

"""The interpreter a macro-free kernel is compiled through on Metal."""
Mantle.kernelinterpreter(::MetalDevice, kernel, tt) = metalinterpreter(kernel, tt)

"""
    vertextouches(dev, shader, args)

Access of the vertex body (not `MetalVertexStage`, whose output pointer is the
driver's) to each argument. A body taking a leading `VertexIndex` is walked with
one, decided as `stage_signatures` decides it; that entry is dropped.
"""
function Mantle.vertextouches(dev::MetalDevice, shader, args::Tuple)
    f = Mantle.stagefunction(shader.vertex)
    argT = map(a -> Mantle.devicetype(dev, a), args)
    lead = KI.wantsvertexindex(f, argT) ? (KI.VertexIndex,) : ()
    return stagetouches(dev, f, args; lead)
end

# Mesh arguments lead with the mesh-stage object, matching native compilation.
Mantle.vertextouches(dev::MetalDevice, shader::Mantle.MeshPipeline, args::Tuple) =
    stagetouches(dev, Mantle.stagefunction(shader.mesh), args;
                 lead = mesh_stage_lead(shader))

"""
    fragmenttouches(dev, shader, args)

Access of the fragment body to each argument. The body's first parameter is the
varyings NamedTuple, so the walk leads with `varying_type` as `stage_signatures`
compiles it; that entry is dropped.
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
    # Drop `f` itself and the stage-owned `lead` arguments.
    return touches[(2 + length(lead)):end]
end
