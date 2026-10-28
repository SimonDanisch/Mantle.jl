# KernelAbstractions backends on Vulkan: the host side of KA launches.

import KernelAbstractions as KA
import Adapt

# Kernel name for dispatch logging. A `workqueue_map_kernel!` launch (Hikari)
# carries its inner function in `all_args[3]`, so that name is appended.
# `@noinline` keeps the string handling out of the launch path's inference.
@noinline function dispatch_name(@nospecialize(f), @nospecialize(all_args))
    fname = nameof(typeof(f))
    if length(all_args) >= 3 && occursin("workqueue_map_kernel", string(fname))
        inner = all_args[3]
        return "$(fname)[$(nameof(typeof(inner)))]"
    end
    return string(fname)
end

# A `LavaDevice` is the KernelAbstractions backend on Vulkan and launches on its
# default channel. It never follows a later default device; after
# `reset_device!` it is dead, like the arrays it allocated.

"""
    LavaDeviceBackend <: KA.Backend

A `LavaDevice` as a kernel sees it: an empty type to dispatch on. Kernels may
capture their backend to choose a code path by type, and a device holds driver
objects, which are not bits.
"""
struct LavaDeviceBackend <: KA.Backend end

Adapt.adapt_structure(::LavaAdaptor, ::LavaDevice) = LavaDeviceBackend()

# A device holds no array a kernel reads. Walking it reaches the cycle through
# `Vulkan.Instance` (see `graph/lifetime.jl`) and overflows the stack.
holdleaves!(holder, ::LavaDevice) = nothing

"""
    vk_context(a::LavaArray) -> LavaDevice

The device an array belongs to, read from its buffer. Use it instead of the
global `vk_context()` whenever an array is at hand.
"""
vk_context(a::LavaArray) = (a.buf[].ctx)::LavaDevice
# `caps` is `KernelInterface.caps` (imported in `Mantle.jl`), so libraries query
# a device through its array.
caps(a::LavaArray) = caps(vk_context(a))
# Core's query (`runtime/backendhooks.jl`), fixed at device creation.
videodecodes(ctx::LavaDevice) = ctx.video_decode_available
# Queried from the physical device, as device creation does before enabling it.
supports_int64_atomics(ctx::LavaDevice) =
    VK.get_physical_device_features_2(ctx.physical_device,
        VK.PhysicalDeviceVulkan12Features).next.shader_buffer_int_64_atomics
# A view is on its parent's device, for every `AnyLavaArray` wrapper.
vk_context(a::Union{SubArray, Base.ReshapedArray, PermutedDimsArray,
                     LinearAlgebra.Transpose, LinearAlgebra.Adjoint}) = vk_context(parent(a))

# The array's own device, not the global default.
KA.get_backend(a::LavaArray) = (a.buf[].ctx)::LavaDevice
# Wait for all work submitted on the device's default channel, so the host can
# read results. Dispatches are ordered on the GPU without it.
KA.synchronize(b::LavaDevice) = (flush!(batchqueue(b)); nothing)
KA.supports_unified(::LavaDevice) = true
# `unified = nothing` is KernelInterface's generic spelling of "no preference".
function KA.allocate(b::LavaDevice, ::Type{T}, dims::Tuple; unified::Union{Nothing,Bool}=nothing) where T
    bq = batchqueue(b)
    dims = Int.(dims)
    nbytes = checkedbytes(T,dims)
    # Allocations up to 64 bytes (counters) are host-mapped, so the host reads
    # them without staging.
    want_unified = something(unified, false) || nbytes <= 64
    return LavaArray{T,length(dims)}(undef, dims; bq, unified=want_unified)
end
KA.unsafe_free!(x::LavaArray) = unsafe_free!(x)

function KA.copyto!(::LavaDevice, A, B)
    GC.@preserve A B begin
        copyto!(A, 1, B, 1, length(A))
    end
    return
end

# `Adapt.adapt(device, x)` uploads to the device; `Adapt.adapt(LavaArray, x)`
# uses the process default device.
Adapt.adapt_storage(d::LavaDevice, a::Array) = LavaArray(a; bq = d.default_bq)
Adapt.adapt_storage(::LavaDevice, x) = x

"""
    alloc_index_buffer(bq, data) -> LavaArray{UInt32,1}

Upload `data` into a new index buffer (`INDEX_BUFFER_BIT`) on `bq`'s device.
Takes a channel because its caller, `Mantle.indexbuffer(::LavaDevice, ...)`,
holds one.
"""
function alloc_index_buffer(bq::SubmitChannel{<:VulkanQueue}, data::AbstractVector{UInt32})
    arr = LavaArray{UInt32,1}(undef, (length(data),); bq,
        extra_usage=UInt32(VK.BUFFER_USAGE_INDEX_BUFFER_BIT))
    upload!(arr, data)
    return arr
end
Adapt.adapt_storage(::KA.CPU, a::LavaArray) = Array(a)

# `@Const` wraps a device array in `Lava.Const`, whose loads Lava declares
# NonWritable; `readonlyargs` adapts the arguments a dispatch only reads the same way.
# KA 0.10 rebuilds reshaped and permuted wrappers around the adapted parent
# without re-validating, so no throwing `reshape` constructor reaches the device.
Adapt.adapt_storage(::KA.ConstAdaptor, a::LavaDeviceArray) = Lava.Const(a)

# `adapt(LavaArray, x)` names a type, not a device, so it allocates on the
# process default device. ambient-allocation-ok
Adapt.adapt_storage(::Type{<:LavaArray}, a::Array) = LavaArray(a)
Adapt.adapt_storage(::Type{<:LavaArray}, a::LavaArray) = a

"""
    workgroup_limit(ctx) -> Int

The device's `maxComputeWorkGroupInvocations`, from [`DeviceCaps`](@ref);
requesting more throws. Modules for two workgroup sizes differ in one
`LocalSize` byte, so the pipeline cache key must hash every byte
([`spirv_content_hash`](@ref)), not a sample as `Base.hash` does.
"""
workgroup_limit(ctx::LavaDevice) = caps(ctx).workgrouplimit

"The extents behind a KA size parameter, or `nothing` when they are dynamic."
@inline statictuple(::Type{<:KA.NDIteration.StaticSize{S}}) where {S} = S
@inline statictuple(@nospecialize(_)) = nothing

"""
    check_workgroup(ctx, wg; static = false) -> wg

Throw if `wg` exceeds [`workgroup_limit`](@ref). Not clamped: a kernel sizes its
`@localmem` tile to the workgroup it asked for.
"""
@inline check_workgroup(ctx::LavaDevice, ::Nothing; static::Bool = false) = nothing
@inline function check_workgroup(ctx::LavaDevice, wg; static::Bool = false)
    n = prod(wg)
    lim = workgroup_limit(ctx)
    n > lim || return wg
    throw(ArgumentError("""
        workgroupsize $wg has $n threads, above this device's limit of $lim,
        which is its `maxComputeWorkGroupInvocations`. Asking for more is
        a validation error, not a slow launch. See `DeviceCaps`."""))
end

function KA.launch_config(kernel::KA.Kernel{LavaDevice}, ndrange, workgroupsize)
    if ndrange isa Integer
        ndrange = (ndrange,)
    end
    if workgroupsize isa Integer
        workgroupsize = (workgroupsize,)
    end

    if KA.ndrange(kernel) <: KA.StaticSize
        ndrange = nothing
    end

    iterspace, dynamic = if KA.workgroupsize(kernel) <: KA.DynamicSize && workgroupsize === nothing
        # Core's default, the same on every backend.
        workgroupsize = Mantle.defaultgroup(ndrange, workgroup_limit(kernel.backend))
        KA.partition(kernel, ndrange, workgroupsize)
    else
        # `kernel(backend, wg)` puts the size in the kernel's type and passes the
        # keyword as `nothing`, so check whichever holds it.
        static = workgroupsize === nothing
        check_workgroup(kernel.backend,
                        static ? statictuple(KA.workgroupsize(kernel)) : workgroupsize;
                        static = static)
        KA.partition(kernel, ndrange, workgroupsize)
    end

    return ndrange, workgroupsize, iterspace, dynamic
end

function KA.mkcontext(kernel::KA.Kernel{LavaDevice}, _ndrange, iterspace)
    KA.CompilerMetadata{KA.ndrange(kernel), KA.DynamicCheck}(_ndrange, iterspace)
end

function KA.mkcontext(kernel::KA.Kernel{LavaDevice}, I, _ndrange, iterspace, ::Dynamic) where Dynamic
    KA.CompilerMetadata{KA.ndrange(kernel), Dynamic}(I, _ndrange, iterspace)
end

# Pure conversion for type reflection; launch paths hold the originals with
# `holdleaves!`.
KA.argconvert(::KA.Kernel{LavaDevice}, a::LavaArray{T,N}) where {T,N} =
    LavaDeviceArray{T,N}(Ptr{T}(bda_address(a)), a.dims)
KA.argconvert(::KA.Kernel{LavaDevice}, x) = x

# `find_tlas_in_args` returns the hardware TLAS of any pre-adapt argument with
# an `hwtlas` field (such as `AdaptedAccel`), so passing an accel compiles with
# ray query and binds the TLAS descriptor. Resolved from the argument types at
# compile time, so kernels without one pay nothing. Later arguments win.
@generated function find_tlas_in_args(args::Tuple)
    ts = fieldtypes(args)
    all(isconcretetype, ts) || return :(find_tlas(args, nothing))
    expr = :(nothing)
    for (i, T) in enumerate(ts)
        if T <: Tuple
            # A tuple argument is a grouped argument list: walk into it, as
            # `storage` and `holdleaves!` do.
            expr = :(let h = find_tlas_in_args(args[$i])
                         h === nothing ? $expr : h
                     end)
        elseif hasfield(T, :hwtlas)
            expr = :(let h = getfield(args[$i], :hwtlas)
                         h === nothing ? $expr : h
                     end)
        end
    end
    expr
end
@inline find_tlas(::Tuple{}, acc) = acc
@inline find_tlas(args::Tuple, acc) = find_tlas(Base.tail(args), maybe_tlas(first(args), acc))
# Fallback for non-concrete argument tuples: a field check, so any value with
# an `hwtlas` field counts.
@inline function maybe_tlas(x, acc)
    if hasfield(typeof(x), :hwtlas)
        h = getfield(x, :hwtlas)
        h !== nothing && return h
    end
    return acc
end

# The `IterPlan` for `obj` at this shape, cached in `vkctx.caches.iterplans` by
# `typeof(obj)` (function and KA's static sizes) and `(ndrange, workgroupsize)`.
# Per device, because `pad_to_3d` reads the device's `max_wg_dims`.
function get_or_build_iter_plan(obj::KA.Kernel{LavaDevice}, ndrange, workgroupsize,
                                vkctx::LavaDevice)
    # `K` is concrete here, so the scan tests `IterEntry{K}` and compares an
    # isbits key with `===` (see `IterEntry`).
    k = (ndrange, workgroupsize)
    K = typeof(k)
    cache = vkctx.caches.iterplans
    v = get(cache, typeof(obj), nothing)
    if v !== nothing
        for q in v
            if q isa IterEntry{K} && q.key === k
                return q.plan
            end
        end
    end

    ndr_canon, _, iterspace, _ = KA.launch_config(obj, ndrange, workgroupsize)
    ka_ctx  = KA.mkcontext(obj, ndr_canon, iterspace)
    blocks  = KA.blocks(iterspace)
    nblocks = length(blocks)
    if nblocks == 0
        new_plan = IterPlan(ka_ctx, (0, 0, 0), (0, 0, 0), 0)
    else
        block_dims = pad_to_3d(vkctx, size(blocks))
        nthreads   = length(KA.workitems(iterspace))
        ws_3d      = (nthreads, 1, 1)
        new_plan   = IterPlan(ka_ctx, block_dims, ws_3d, nblocks)
    end
    push!(get!(() -> Any[], cache, typeof(obj)), IterEntry{K}(k, new_plan))
    return new_plan
end

"""
    launchindirect!(bq, obj, args, count, workgroupsize, tlas)

Launch `obj` over the element count in the device array `count` as one
one-shot: the prepare that turns it into group counts, a barrier, the dispatch.
`count` is held like an argument, because only the prepare names it.
"""
function launchindirect!(bq, obj, args, count::LavaArray, workgroupsize, tlas)
    oneshot!(bq; tag = :launch) do e
        owner = e.owner
        holdleaves!(owner, obj.f)
        holdleaves!(owner, args)
        holdleaves!(owner, count)
        adaptor = LavaAdaptor(owner)
        converted_args = argmap(a -> Adapt.adapt(adaptor, a), args)
        ka_launch_indirect!(e, obj, converted_args, count, workgroupsize, args, adaptor, tlas)
    end
    return nothing
end

function (obj::KA.Kernel{LavaDevice})(args...; ndrange=nothing, workgroupsize=nothing)
    # The kernel's own device, not the global context.
    validate_launch_args(obj.backend, args)
    # Always an immediate launch; graphs declare passes with `dispatch!`.
    bq = batchqueue(obj.backend)

    # Before Adapt, which sets `hwtlas` to `nothing`.
    tlas = find_tlas_in_args(args)

    # A device-resident count (`DeviceRange` or `LavaArray`) launches indirectly,
    # without a readback; the `DeviceRange` ceiling is unused here.
    ndrange isa DeviceRange &&
        return launchindirect!(bq, obj, args, storage(ndrange.count), workgroupsize, tlas)
    ndrange isa LavaArray &&
        return launchindirect!(bq, obj, args, ndrange, workgroupsize, tlas)

    # KA's launch configuration plus `pad_to_3d` cost about half the host time of
    # a launch and depend only on kernel type and shape, so the plan is cached.
    plan = get_or_build_iter_plan(obj, ndrange, workgroupsize, deviceof(bq))
    # Function barrier: the cache returns an abstract `IterPlan`, and reading its
    # fields here (including `nblocks`) would box them. One dynamic dispatch makes
    # every use inside `launch_planned!` concrete.
    return launch_planned!(bq, obj, args, plan, tlas)
end

@noinline function launch_planned!(bq::SubmitChannel{<:VulkanQueue}, obj::KA.Kernel{LavaDevice},
                                   args::Tuple, plan::IterPlan, tlas)
    plan.nblocks == 0 && return nothing
    ka_ctx     = plan.ka_ctx
    block_dims = plan.block_dims
    ws_3d      = plan.ws_3d

    # Before the one-shot, so a kernel the access walk refuses opens no command buffer.
    touches = katouches(obj, args, plan)
    # One command buffer per launch, never joining an open batch, which would
    # depend on task-global state; batching is what a declared graph is for.
    oneshot!(bq; tag = :launch) do e
        emitlaunch!(e, obj, args, touches, ka_ctx, block_dims, ws_3d, tlas)
    end
    return nothing
end

# Hold every leaf of the closure and arguments in the one-shot, then adapt (which
# holds nothing) and launch.
@inline function emitlaunch!(e, obj, args, touches, ka_ctx, block_dims, ws_3d, tlas)
    owner = e.owner
    holdleaves!(owner, obj.f)
    holdleaves!(owner, args)
    adaptor = LavaAdaptor(owner)
    converted_f = Adapt.adapt(adaptor, obj.f)
    converted_args = argmap(a -> Adapt.adapt(adaptor, a), args)
    # Which arguments `readonlyargs` wraps is decided per call (two may share a
    # buffer), so its result has no inferable type: one dynamic call into the launch,
    # taken only when there is something to wrap.
    any(isreadonly, touches) &&
        return launchwith!(e, converted_f, ka_ctx,
                           readonlyargs(nothing, touches, args, converted_args),
                           block_dims, ws_3d, tlas)
    launchwith!(e, converted_f, ka_ctx, converted_args, block_dims, ws_3d, tlas)
end

# `::F`: a function argument only passed on is not specialised on, and an abstract
# `f` boxes the argument tuple.
@noinline launchwith!(e, f::F, ka_ctx, args::Tuple, block_dims, ws_3d, tlas) where {F} =
    ka_launch!(e, f, (f, ka_ctx, args...), block_dims, ws_3d, tlas)

# Vulkan dispatches are at most 3-D: `pad_to_3d` maps the block grid onto
# `ctx.max_wg_dims`, splitting large dimensions across Y and Z. Extra workgroups
# write out of bounds in kernels that index auxiliary arrays by workgroup id
# without checks (AK's `_accumulate_block!`), so splits use exact factors where
# they exist.

# A divisor of `n` within 3 of `max_dim`, else the smallest divisor in
# `cld(n, max_dim):max_dim`, else `max_dim`, which over-dispatches.
function find_split_factor(n::Int, max_dim::Int)
    for d in (max_dim, max_dim-1, max_dim-2, max_dim-3)
        d > 0 && n % d == 0 && return d
    end
    lo = cld(n, max_dim)
    for x in lo:min(n, max_dim)
        n % x == 0 && return x
    end
    return max_dim
end

function pad_to_3d(ctx::LavaDevice, t::NTuple{1,<:Integer})
    n = Int(t[1])
    max_x, max_y, _ = ctx.max_wg_dims
    n <= max_x && return (n, 1, 1)
    x = find_split_factor(n, max_x)
    y = min(cld(n, x), max_y)
    z = cld(n, x * y)
    # Unsafe for kernels that index by workgroup id without bounds checks.
    x * y * z == n || @warn "pad_to_3d: over-dispatching $n as ($x, $y, $z) = $(x*y*z) workgroups" maxlog=1
    return (x, y, z)
end
function pad_to_3d(ctx::LavaDevice, t::NTuple{2,<:Integer})
    x, y = Int(t[1]), Int(t[2])
    max_x, max_y, _ = ctx.max_wg_dims
    if x <= max_x && y <= max_y
        return (x, y, 1)
    end
    return pad_to_3d(ctx, (x * y,))
end
function pad_to_3d(::LavaDevice, t::NTuple{3,<:Integer})
    (Int(t[1]), Int(t[2]), Int(t[3]))
end
# N > 3: drop trailing unit axes and flatten only if a non-unit axis remains past
# the third. This keeps the common `(W, H, C, 1)` grid 1:1 with the dispatch,
# which `directdispatch` needs to index without dividing.
function pad_to_3d(ctx::LavaDevice, t::NTuple{N,<:Integer}) where N
    n = N
    while n > 3 && Int(t[n]) == 1
        n -= 1
    end
    n > 3 && return pad_to_3d(ctx, (Int(prod(t)),))
    return pad_to_3d(ctx, ntuple(i -> Int(t[i]), n))
end

# Per device, because a `LaunchPlan` holds a `VkPipeline`. The `DataType` key
# makes the lookup a pointer compare.
@inline launch_plan_cache(ctx) = ctx.caches.launchplans

@inline function launch_plan(bq::SubmitChannel{<:VulkanQueue}, @nospecialize(f), all_args::Tuple,
                             wg::NTuple{3,Int}, ray_query::Bool)
    # The key includes the kernel type: a KI kernel has `nothing` in
    # `all_args[1]`, so two with the same argument types would share a pipeline.
    key = Tuple{typeof(f), typeof(all_args)}
    world = Base.get_world_counter()
    # `deviceof(bq)` asserts `LavaDevice`; the driver's `ctx` field is untyped, and an
    # untyped cache makes the lookup below allocate on every dispatch.
    cache = launch_plan_cache(deviceof(bq))
    v = get(cache, key, nothing)
    if v !== nothing
        for q in v
            # A typeassert on a `Vector{Any}` element does not copy; a
            # `Vector{LaunchPlan}` would box on every read.
            p = q::LaunchPlan
            (p.world === world && p.wg === wg && p.ray_query === ray_query) && return p
        end
    end
    build_launch_plan!(bq, f, all_args, wg, ray_query, key, world)
end

# `all_args` has a different concrete type per kernel, so any method taking it is
# inferred per kernel. This method only computes `tt`; the kernel-independent
# rest is `build_launch_plan_tt!` behind `Ref{Any}` boxes, so it is inferred once.
# `@nospecialize` alone does not prevent the per-kernel specialisation.
@noinline function build_launch_plan!(bq::SubmitChannel{<:VulkanQueue}, @nospecialize(f), all_args::Tuple,
                                      wg::NTuple{3,Int}, ray_query::Bool,
                                      key::DataType, world::UInt64)
    # Drops `all_args[1]`, the function slot GPUCompiler prepends itself.
    # `arg_sigtype` recovers `Type{X}` for type-valued arguments, which
    # `typeof(all_args)` erases to `DataType`.
    tt = Tuple{map(arg_sigtype, Base.tail(all_args))...}
    # `key` is boxed too: a type-valued argument specialises like `tt`.
    build_launch_plan_tt!(bq, Ref{Any}(f), Ref{Any}(tt), Ref{Any}(key), wg, ray_query, world)
end

"""
    build_launch_plan_tt!(bq, fbox, ttbox, keybox, wg, ray_query, world) -> LaunchPlan

The kernel-independent half of `build_launch_plan!`, inferred once. The
`Ref{Any}` boxes erase `f`, `tt` and `key`, for which Julia ignores
`@nospecialize`. Runs only on a launch-plan cache miss, before a compile.
"""
@noinline function build_launch_plan_tt!(bq::SubmitChannel{<:VulkanQueue}, fbox::Base.RefValue{Any},
                                         ttbox::Base.RefValue{Any},
                                         keybox::Base.RefValue{Any},
                                         wg::NTuple{3,Int}, ray_query::Bool,
                                         world::UInt64)
    f = fbox[]
    tt = ttbox[]
    key = keybox[]::DataType
    compiled, pipeline, offsets, byval_sizes = get_compiled_kernel_and_pipeline(
        deviceof(bq), f, tt, wg; enable_ray_query=ray_query)
    base = compiled.push_info.arg_buffer_size
    p = LaunchPlan(compiled, pipeline, offsets, byval_sizes, base,
                   base + compute_inline_extra_from_byval(byval_sizes, base),
                   world, wg, ray_query)
    v = get!(() -> Any[], launch_plan_cache(deviceof(bq)), key)
    filter!(q -> (q::LaunchPlan).world === world, v)  # superseded worlds are dead
    push!(v, p)
    p
end

"""
    ka_launch!(e, f, all_args, block_dims, workgroup_size, tlas = nothing)

Look up or compile `f` for the post-adapt `all_args`, pack the arguments into
`e`'s one-shot and record the dispatch over `block_dims`. `all_args[1]` is the
function slot and is not part of the signature.
"""
function ka_launch!(e::Emitter, @nospecialize(f), all_args::Tuple,
                    block_dims::NTuple{3,Int}, workgroup_size::NTuple{3,Int},
                    tlas=nothing)  # positional: no keyword NamedTuple on the hot path
    bq = queueof(e)
    # A TLAS argument compiles with ray query and binds the TLAS descriptor.
    plan = launch_plan(bq, f, all_args, workgroup_size, tlas !== nothing)

    # The one-shot owns the argument memory. Taken once from the emitter:
    # reaching it through the queue allocates when that field infers abstract.
    owner = e.owner
    arg_buf = get_arg_buffer(owner, plan.total_size)

    # `all_args` is post-adapt; the caller already held the original leaves.
    pack_args_direct!(owner, arg_buf.mapped_ptr, arg_buf.address, plan.offsets,
                       plan.arg_buffer_size, plan.byval_sizes, all_args)

    name = ""
    if (deviceof(bq)).diag.dispatch_logging
        name = Base.invokelatest(dispatch_log_string, "ka f=",
                                 dispatch_name(f, all_args), " groups=", block_dims)::String
        driver(bq).last_dispatch_info = name
    end
    emit_dispatch!(e, plan.pipeline, arg_buf.address, block_dims, tlas, name)
    return nothing
end

# Indirect dispatch. Kernels take `LavaDeviceArray`s, never raw pointers.

function prepare_indirect_kernel(indirect::LavaDeviceArray{UInt32,1},
                                  ndrange_buf::LavaDeviceArray{Int32,1},
                                  ws::UInt32)
    n = UInt32(ndrange_buf[1])
    groups = (n + ws - UInt32(1)) ÷ ws
    indirect[1] = groups         # groupCountX
    indirect[2] = UInt32(1)      # groupCountY
    indirect[3] = UInt32(1)      # groupCountZ
    return nothing
end

function prepare_indirect_dispatch!(e::Emitter,
                                    indirect::LavaArray{UInt32,1},
                                    ndrange_buf::LavaArray{<:Integer},
                                    workgroup_size::Integer)
    emitkernel!(e, prepare_indirect_kernel,
                indirect, ndrange_buf, UInt32(workgroup_size);
                ndrange=1, workgroup_size=(1, 1, 1))
end

"""
    ka_launch_indirect!(e, obj, args, ndrange_buf, workgroupsize, original_args, adaptor, tlas)

Record an indirect launch of `obj` into `e`: a prepare kernel turns the item
count in `ndrange_buf` (1-element `Int32`) into group counts, a barrier makes
them visible, and the dispatch reads them.
"""
function ka_launch_indirect!(e::Emitter, obj, args, ndrange_buf::LavaArray, workgroupsize,
                             original_args, adaptor::LavaAdaptor,
                             tlas=nothing)  # positional, as in ka_launch!
    bq = queueof(e)
    # Keyword, then the `@kernel` static size, then 256.
    ws = if workgroupsize !== nothing
        workgroupsize isa Integer ? (workgroupsize,) : workgroupsize
    elseif KA.workgroupsize(obj) <: KA.StaticSize
        KA.get(KA.workgroupsize(obj))
    else
        (256,)
    end
    ws_3d = pad_to_3d(deviceof(bq), ws)
    ws_prod = prod(ws)

    # The group count comes from the indirect buffer. A large ndrange makes
    # `__validindex` pass for every thread; the kernel must bound itself against
    # the real count (e.g. `i <= queue.size[1]`).
    big_ndrange = (1024 * 1024 * ws_prod,)
    ndrange_tuple = length(ws) == 1 ? big_ndrange :
                    length(ws) == 2 ? (big_ndrange[1], 1) :
                    (big_ndrange[1], 1, 1)
    iterspace, dynamic = KA.partition(obj, ndrange_tuple, ws)
    ctx = KA.mkcontext(obj, ndrange_tuple, iterspace)

    # The caller held the arguments; hold the closure's captures as well.
    batch = adaptor.batch
    holdleaves!(batch, obj.f)
    converted_f = Adapt.adapt(adaptor, obj.f)
    all_args = (converted_f, ctx, args...)

    # Signature of the post-adapt arguments, function slot dropped.
    tt = Tuple{map(arg_sigtype, Base.tail(all_args))...}
    enable_ray_query = tlas !== nothing
    compiled, pipeline, offsets, byval_sizes = get_compiled_kernel_and_pipeline(
        deviceof(bq), converted_f, tt, ws_3d; enable_ray_query)

    inline_extra = compute_inline_extra_from_byval(byval_sizes, compiled.push_info.arg_buffer_size)
    total_size = compiled.push_info.arg_buffer_size + inline_extra

    # Keep the pre-adapt arrays reachable while their addresses are packed.
    GC.@preserve original_args begin

    arg_buf = get_arg_buffer(batch, total_size)
    pack_args_direct!(batch, arg_buf.mapped_ptr, arg_buf.address, offsets,
                       compiled.push_info.arg_buffer_size, byval_sizes, all_args)

    indirect_view = indirect_command!(batch)

    name = ""
    if (deviceof(bq)).diag.dispatch_logging
        name = Base.invokelatest(dispatch_log_string, "indirect f=",
                                 dispatch_name(obj.f, all_args))::String
        driver(bq).last_dispatch_info = name
    end
    # Prepare, barrier for the indirect read, dispatch. Graph passes fuse their
    # prepares instead (`emitprepares!`).
    prepare_indirect_dispatch!(e, indirect_view, ndrange_buf, ws_prod)
    indirectbarrier!(e, VK.PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK.PIPELINE_STAGE_DRAW_INDIRECT_BIT,
                     VK.ACCESS_INDIRECT_COMMAND_READ_BIT)
    emit_dispatch!(e, pipeline, arg_buf.address, indirect_view, tlas, name)

    end # GC.@preserve

    return nothing
end

# Device-side code (`LavaDeviceArray` indexing and the KA index, synchronize and
# scratchpad overrides) is in Lava's `device/devicearray.jl` and
# `device/ndrange.jl`, so Lava compiles kernels without Mantle loaded.
