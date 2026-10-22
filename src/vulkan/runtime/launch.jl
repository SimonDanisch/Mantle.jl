# Compute kernel preparation, argument packing and owner-held scratch. SPIR-V
# lives with its CodeInstance, pipelines in the device cache.
# Type-valued arguments specialize as Type{T}, not DataType.
@inline arg_sigtype(@nospecialize(a)) = a isa Type ? Type{a} : typeof(a)

"""
    validate_launch_args(ctx, args)

Throw a `LavaError` for a `LavaArray` argument whose `DataRef` was released or
whose buffer is `BUF_STATE_DEAD`. On by default; disable with
`ctx.diag.launch_arg_validation = false`.
"""
@generated function validate_launch_args(ctx::VkContext, args::T) where T <: Tuple
    exprs = Expr[]
    # The toggle is read at run time, not baked into the generated body.
    push!(exprs, :((ctx.diag.launch_arg_validation || return)))
    for i in 1:fieldcount(T)
        Ti = fieldtype(T, i)
        if Ti <: LavaArray
            push!(exprs, quote
                let arg = args[$i]
                    local buf
                    try
                        buf = arg.buf[]
                    catch
                        throw(LavaError("kernel launch",
                            "Argument $($i): LavaArray has been freed (DataRef released)",
                            "Don't pass freed arrays to GPU kernels. Check array lifetime."))
                    end
                    # Lifecycle state gates host access; poison is a GPU diagnostic only.
                    if (@atomic :acquire buf.state) == BUF_STATE_DEAD
                        throw(LavaError("kernel launch",
                            "Argument $($i): LavaArray backing buffer is dead (state=BUF_STATE_DEAD)",
                            "This array was freed. Reallocate before use."))
                    end
                end
            end)
        end
    end
    push!(exprs, :(return nothing))
    return Expr(:block, exprs...)
end

"""lava_launch!(bq, f, args...; ndrange, workgroup_size=(64,1,1), tlas=nothing)

Submit a plain kernel in a one-shot on `bq` (see `emitkernel!`)."""
function lava_launch!(bq::SubmitChannel{<:VulkanQueue}, @nospecialize(f), args...;
                       ndrange::Union{Integer, NTuple{3,<:Integer}},
                       workgroup_size::NTuple{3,Int} = (64, 1, 1),
                       tlas=nothing)  # Union{Nothing, VulkanTLAS}, declared later in raytracing/hwtlas.jl
    oneshot!(bq; tag = :launch) do e
        emitkernel!(e, f, args...; ndrange, workgroup_size, tlas)
    end
    return nothing
end

"""
    emitkernel!(emitter, f, args...; ndrange, workgroup_size, tlas = nothing)

Prepare a kernel (`preparekernel`) and emit its dispatch into `emitter`'s
owner. Plans supply derived dependencies; one-shots rely on their head barrier.
"""
function emitkernel!(e::Emitter, @nospecialize(f), args...;
                     ndrange::Union{Integer, NTuple{3,<:Integer}},
                     workgroup_size::NTuple{3,Int} = (64, 1, 1),
                     tlas = nothing)
    pipeline, argaddr, groups, name =
        preparekernel(e.owner, f, args, ndrange, workgroup_size, tlas)
    if e.ctx.diag.dispatch_logging
        driver(queueof(e)).last_dispatch_info = name
    end
    emit_dispatch!(e, pipeline, argaddr, groups, tlas, name)
    return nothing
end

"""
Compile `f` for `args`, hold their array leaves in `owner`, adapt and pack them
into owner-held memory, and return the pipeline, argument address, workgroup
counts and log name (`""` unless logging or timing is on).
"""
function preparekernel(owner::O, @nospecialize(f), args::Tuple,
                       ndrange::Union{Integer, NTuple{3,<:Integer}},
                       workgroup_size::NTuple{3,Int}, tlas) where {O<:Closed}
    bq = queueof(owner)
    ctx = ctxof(bq)
    validate_launch_args(ctx, args)
    if ndrange isa Integer
        ndrange_3d = (Int(ndrange), 1, 1)
    else
        ndrange_3d = (Int(ndrange[1]), Int(ndrange[2]), Int(ndrange[3]))
    end
    groups = (
        cld(ndrange_3d[1], workgroup_size[1]),
        cld(ndrange_3d[2], workgroup_size[2]),
        cld(ndrange_3d[3], workgroup_size[3]),
    )

    # Hold every array leaf of `f` and `args` so its buffer outlives the
    # submissions naming it; then adapt, which is pure, to device types.
    holdleaves!(owner, f)
    holdleaves!(owner, args)
    adaptor = LavaAdaptor(owner)
    converted_f = Adapt.adapt(adaptor, f)
    converted_args = argmap(a -> Adapt.adapt(adaptor, a), args)

    # The kernel is compiled at the adapted types (LavaDeviceArray, not Ptr{T}).
    tt = Tuple{map(arg_sigtype, converted_args)...}

    enable_ray_query = tlas !== nothing
    compiled, pipeline, offsets, byval_sizes =
        get_compiled_kernel_and_pipeline(ctx, converted_f, tt, workgroup_size;
                                         enable_ray_query)

    if pipeline.needs_tlas_descriptor && tlas === nothing
        error("kernel was compiled with enable_ray_query=true but launch_kernel was " *
              "called without a tlas keyword. Pass tlas=<VulkanTLAS> to bind the " *
              "acceleration structure to descriptor set 0, binding 0.")
    end

    # The LLVM entry takes `f` as its first parameter; it gets a slot unless
    # it is a ghost type.
    all_args = (converted_f, converted_args...)
    inline_extra = compute_inline_extra_from_byval(byval_sizes)
    total_size = compiled.push_info.arg_buffer_size + inline_extra
    arg_buf = get_arg_buffer(owner, total_size)

    pack_args_direct!(owner, arg_buf.mapped_ptr, arg_buf.address, offsets,
                       compiled.push_info.arg_buffer_size, byval_sizes, all_args)

    name = ctx.diag.dispatch_logging || ctx.diag.dispatch_timing ?
           "compute f=$(nameof(typeof(converted_f))) groups=$groups" : ""
    return pipeline, arg_buf.address, groups, name
end

# Argument buffer: one slot per passed argument (`offsets`), then the inlined
# by-value aggregates. An aggregate's slot holds the address of its inlined
# copy; an allocation's slot holds its BDA.

"""
    compute_inline_extra_from_byval(byval_sizes::Vector{Int})

Bytes after the slot layout for the inlined aggregates, each 8-byte aligned.
Uses LLVM byval sizes, which can exceed Julia's `sizeof` (zero-sized fields).
"""
function compute_inline_extra_from_byval(byval_sizes::Vector{Int})
    extra = 0
    for sz in byval_sizes
        sz > 0 || continue
        extra = (extra + 7) & ~7
        extra += sz
    end
    return extra
end

# Per-type argument packers, each returning the next inline offset:
#   * default: isbits struct inlined, slot gets its address; primitive stored
#   * Ptr: the pointer value is the BDA
#   * VkManagedBuffer: held by the owner, slot gets its BDA
@inline function pack_arg!(x::T,
                           mapped_ptr::Ptr{UInt8}, arg_buf_bda::UInt64,
                           offset::Int, byval_size::Int, inline_offset::Int,
                           batch::O) where {T,O<:Closed}
    if isbitstype(T) && !isprimitivetype(T)
        inline_offset = (inline_offset + 7) & ~7
        ccall(:memset, Ptr{Cvoid}, (Ptr{Cvoid}, Cint, Csize_t),
              mapped_ptr + inline_offset, 0, byval_size)
        unsafe_store!(Ptr{T}(mapped_ptr + inline_offset), x)
        unsafe_store!(Ptr{UInt64}(mapped_ptr + offset),
                      arg_buf_bda + UInt64(inline_offset))
        recpatchfields!(batch, T, mapped_ptr + inline_offset)
        return inline_offset + byval_size
    else
        # Native allocation handles need a packer that retains them and writes a BDA.
        if passedas(T) isa DeviceAllocation
            throw(LavaError("pack_arg!",
                "no method for $T, which `passedas` declares a DeviceAllocation",
                "add a `pack_arg!(::$T, ...)` that holds the allocation and writes " *
                "its device address, as the `VkManagedBuffer` method does"))
        end
        unsafe_store!(Ptr{T}(mapped_ptr + offset), x)
        return inline_offset
    end
end

@inline function pack_arg!(p::Ptr,
                           mapped_ptr::Ptr{UInt8}, arg_buf_bda::UInt64,
                           offset::Int, byval_size::Int, inline_offset::Int,
                           batch::O) where {O<:Closed}
    unsafe_store!(Ptr{UInt64}(mapped_ptr + offset), UInt64(p))
    return inline_offset
end

# The one argument on this backend whose slot binds an allocation instead of a
# copy of the value's bytes; the method below is what binding means here.
passedas(::Type{<:VkManagedBuffer}) = DeviceAllocation()

@inline function pack_arg!(buf::VkManagedBuffer,
                           mapped_ptr::Ptr{UInt8}, arg_buf_bda::UInt64,
                           offset::Int, byval_size::Int, inline_offset::Int,
                           batch::O) where {O<:Closed}
    # The kernel reads through this address, so the submission must outlive the
    # buffer and be ordered against whoever wrote it last.
    hold!(batch, buf)
    if (buf.ctx::VkContext).diag.pack_arg_assert_live
        st = @atomic :acquire buf.state
        if st != BUF_STATE_ALIVE
            throw(LavaError("pack_arg!",
                "packing buffer with state=$st (not ALIVE) at addr=0x$(string(buf.address, base=16, pad=16))",
                "buffer was freed but its BDA is being packed into an arg slab — use-after-free"))
        end
    end
    unsafe_store!(Ptr{UInt64}(mapped_ptr + offset), buf.address)
    # A single store at the layout slot, so a move can rewrite exactly there.
    recpatch!(batch, buf.address, mapped_ptr + offset)
    return inline_offset
end

"""
    pack_args_direct!(owner, mapped_ptr, arg_buf_bda, offsets, base_size, byval_sizes, all_args)

Write kernel arguments into mapped GPU memory through `pack_arg!`, without heap
allocation. Slots follow core's `passedslots`. The owner holds allocation
arguments: a one-shot until completion, a recording until release.
"""
@generated function pack_args_direct!(batch::Closed,
                                        mapped_ptr::Ptr{UInt8}, arg_buf_bda::UInt64,
                                        offsets::Vector{Int}, base_size::Int,
                                        byval_sizes::Vector{Int},
                                        all_args::T) where {T <: Tuple}
    exprs = Expr[]
    for (layout_i, arg_i) in enumerate(passedslots(T.parameters))
        push!(exprs, :(inline_offset = pack_arg!(
            all_args[$arg_i], mapped_ptr, arg_buf_bda,
            @inbounds(offsets[$layout_i]),
            @inbounds(byval_sizes[$layout_i]),
            inline_offset, batch)))
    end
    quote
        inline_offset = base_size
        $(exprs...)
        return nothing
    end
end

"""
    clear_kernel_cache!(ctx = vk_context())

Drop `ctx`'s pipelines, linked kernels and launch plans so each kernel's next
dispatch builds them again; the SPIR-V stays with its `CodeInstance`. All three
go together, since each can return a pipeline the others would rebuild.
Submissions and plans keep their own references, so accepted work stays valid.
"""
function clear_kernel_cache!(ctx::VkContext = vk_context())
    empty!(ctx.caches.pipelines)
    empty!(ctx.caches.pipeline_order)
    empty!(ctx.caches.linked)
    empty!(ctx.caches.launchplans)
    return nothing
end

"""
    link_kernel(ctx, compiled::LavaGPUKernel) -> LavaLinkedKernel

`ctx`'s pipeline and argument layout for a compiled kernel.
"""
function link_kernel(ctx::VkContext, compiled::LavaGPUKernel)
    pipeline = get_compute_pipeline(ctx, compiled.spirv_bytes, compiled.entry_name;
                                    push_constant_size=compiled.push_info.push_size,
                                    needs_tlas_descriptor=compiled.enable_ray_query)
    offsets = compiled.push_info.arg_offsets
    byval_sizes = compiled.push_info.byval_llvm_sizes
    return LavaLinkedKernel(compiled, pipeline, offsets, byval_sizes)
end

"""
    get_compiled_kernel_and_pipeline(ctx, f, tt, workgroup_size; enable_ray_query=false)
      -> (compiled, pipeline, offsets, byval_sizes)

The SPIR-V for `(f, tt, workgroup_size)` (`compile_or_lookup`) and this device's
pipeline for it. `@noinline` and `@nospecialize` keep compiler inference out of
specialised launch callers.
"""
@noinline function get_compiled_kernel_and_pipeline(ctx::VkContext, @nospecialize(f), @nospecialize(tt),
                                          workgroup_size;
                                          enable_ray_query::Bool=false)
    job = lava_kernel_job(f, tt; workgroup_size, enable_ray_query, features = ctx.features)
    # `invokelatest` keeps compiler invalidations away from the warm launch path.
    compiled = Base.invokelatest(compile_or_lookup, job)::LavaGPUKernel
    linked = get!(() -> link_kernel(ctx, compiled), ctx.caches.linked, compiled)

    dd = ctx.diag.spirv_dump_dir
    if dd !== nothing && !isempty(dd)
        ctx.diag.spirv_dump_counter += 1
        fname = string(nameof(typeof(f)))
        path = joinpath(dd, "$(lpad(ctx.diag.spirv_dump_counter, 3, '0'))_$(fname).spv")
        write(path, linked.compiled.spirv_bytes)
    end

    return linked.compiled, linked.pipeline, linked.offsets, linked.byval_sizes
end

# Scratch belongs to the closed command owner, not the queue. Plans allocate
# their argument and indirect slots at compile time.

const ARG_ALIGN = 256                     # BDA alignment for sub-allocations
"""A block of the [`Unified`](@ref) arena. Small, because BAR memory is scarce on
a device without resizable BAR and a recording's arguments are kilobytes."""
const UNIFIED_BLOCK_SIZE = 4 * 1024 * 1024
"""A block of the [`Readback`](@ref) arena. `acquire!` sizes a block to the larger
of the request and this, so it only sets how many small transfers share one."""
const READBACK_BLOCK_SIZE = 16 * 1024 * 1024

"""A slice of the unified arena a recording writes its arguments into."""
struct ArgBufferAlloc
    address::UInt64           # BDA of this sub-allocation
    mapped_ptr::Ptr{UInt8}    # CPU-writable pointer
    size::Int
end

"""
    scratch!(owner, nbytes) -> Region

Borrow Unified memory for this owner's arguments. A one-shot returns it after
completion, a recording at release, since it can be replayed.
"""
@inline scratch!(owner::O, nbytes::Integer) where {O<:Closed} = scratch!(owner, nbytes, Unified())

"""
    scratch!(owner, nbytes, kind) -> Region

Borrow owner-held memory: `Unified` for addressed arguments, `Readback` for host
staging that a GPU copy reads. The owner returns it at its release point.
"""
@inline function scratch!(owner::O, nbytes::Integer, kind) where {O<:Closed}
    bq = queueof(owner)
    ownthread(bq)
    dev = lavadevice(ctxof(bq))
    # `maywait = false`: this can run inside an open recording, and a waiting
    # reclaim sweeps the channel and pops that recording's hold frame.
    r = acquire!(pool(dev), dev, kind, nothing, max(Int(nbytes), 16);
                 align = ARG_ALIGN, blocksize = scratchblock(kind), maywait = false)
    push!(owner.regions, r)
    return r
end
scratchblock(::Unified) = UNIFIED_BLOCK_SIZE
scratchblock(::Readback) = READBACK_BLOCK_SIZE

"""Bytes for one launch's arguments, as the address the push constant carries and
the pointer the packer writes through."""
@inline function get_arg_buffer(owner::O, nbytes::Integer) where {O<:Closed}
    r = scratch!(owner, nbytes)
    blk = memoryof(r)::BufferBlock
    off = offset(r)
    return ArgBufferAlloc(blk.address + UInt64(off),
                          (blk.ref[]::VkManagedBuffer).mapped_ptr + off,
                          length(r))
end

"""
    indirect_command!(owner) -> LavaArray{UInt32,1}

One owner-held indirect dispatch triple, written by a prepare kernel and read
by the command processor. Plans use their compiled argument slots instead.
"""
function indirect_command!(owner::O) where {O<:Closed}
    r = scratch!(owner, INDIRECT_STRIDE)
    blk = memoryof(r)::BufferBlock
    return LavaArray{UInt32,1}(copy(blk.ref), (3,); offset = offset(r))
end
