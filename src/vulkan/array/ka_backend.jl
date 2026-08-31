# KernelAbstractions.jl backend for Lava.jl
#
# Implements LavaBackend <: KA.GPU for Vulkan compute dispatch.
# Pattern follows AMDGPU.jl's ROCKernels implementation.

import KernelAbstractions as KA
import Adapt

export LavaBackend

# ── Dispatch name helper ──
# Extract a descriptive kernel name for dispatch logging.
# For workqueue foreach dispatches, the KA kernel is always `_workqueue_map_kernel!`
# but the actual inner function (e.g. `vp_trace_shadow_rays_kernel!`) is in args.
# all_args layout: (kernel_func, ctx, inner_func, queue, extra_args...)
# `@noinline`: callers reach this only through `invokelatest`, so the string
# machinery it touches stays out of the launch path's inference chain.
@noinline function dispatch_name(@nospecialize(f), @nospecialize(all_args))
    fname = nameof(typeof(f))
    # Check if this is a workqueue_map_kernel dispatch — inner func is arg 3
    if length(all_args) >= 3 && occursin("workqueue_map_kernel", string(fname))
        inner = all_args[3]
        return "$(fname)[$(nameof(typeof(inner)))]"
    end
    return string(fname)
end

# ── Backend struct ──

"""
    LavaBackend <: KA.GPU

Lava's GPU compute backend. Carries the Vulkan context and batch queues explicitly.

  * `dispatch_bq`: where KA kernel dispatches are recorded.
  * `upload_bq`:   where CPU→GPU transfers (upload!, download!, staging)
                   record their copies. May be the same queue as dispatch_bq
                   (single-queue mode) or a separate async queue for true
                   upload/compute overlap.

    LavaBackend()                      # default: dispatch + upload both on default_bq
    LavaBackend(bq)                    # single queue for both
    LavaBackend(dispatch_bq, upload_bq)  # split — enables pipelining

`LavaBackend()` with no arguments resolves `dispatch_bq` / `upload_bq`
lazily via `vk_context().default_bq` at every property access.  Pinning
would break after `reset_device!()`: a `const BACKEND = LavaBackend()`
created at module-load would keep a stale `VulkanBatchQueue` tied to the old
`VkDevice`, and every subsequent buffer created via that backend would end
up allocated on the dead device — later triggering
`VUID-vkCmdCopyBuffer-commonparent` and a page-aligned GPUVM fault.
Explicit queues passed to `LavaBackend(bq)` / `LavaBackend(d, u)` are
pinned on purpose (the caller wants those exact queues, e.g. an async
upload queue).
"""
struct LavaBackend <: KA.GPU
    # nothing = "use vk_context().default_bq at each access" (survives resets)
    # non-nothing = caller pinned this specific queue
    dispatch_bq::Union{VulkanBatchQueue, Nothing}
    upload_bq::Union{VulkanBatchQueue, Nothing}
end

LavaBackend() = LavaBackend(nothing, nothing)
LavaBackend(ctx::VkContext) = (let bq = ctx.default_bq; LavaBackend(bq, bq); end)
LavaBackend(bq::VulkanBatchQueue) = LavaBackend(bq, bq)

# The three Mantle verbs that dispatch on the BACKEND rather than on the context.
#
# Here and not in `runtime/device.jl` beside their `VkContext` methods, which is
# where they were written: that file is included seven files before this one, so
# each signature named a type that did not exist yet. A method's argument types
# are resolved when it is DEFINED — unlike a function body, which is why the
# `vk_context(b)` call below is fine and the `::LavaBackend` above it was not.
#
# `supports_batch_queue` vs `supports_graphics`: having a queue to batch onto is
# a separate question from being able to rasterise. See `graph/queue.jl` and
# `graphics/commands.jl` for the declarations.
supports_batch_queue(::LavaBackend) = true
supports_graphics(::LavaBackend) = true
waitidle(b::LavaBackend) = waitidle(vk_context(b))

"""
    vk_context(backend) -> VkContext
    vk_context(a::LavaArray) -> VkContext

The device a backend or an array belongs to.

The accessor the rest of the stack should reach for instead of calling
`vk_context()` and hoping. `vk_context()` stays as the convenience default for
the single-device case; what has to stop is code *depending* on it, because a
global cannot answer "which device" once there are two.

**No new state.** Both are derived from what these objects already carried:
`VulkanBatchQueue.ctx` for a backend and `Buffer.ctx` for an array. That is worth
saying because it was briefly got wrong in the other direction — a `ctx` field
was added to `LavaBackend` on the belief that no path existed, which came from
reading the first half of `VulkanBatchQueue`'s field list, where `ctx::Any` sits
sixty-odd lines down. A second copy of a fact the queue already holds can only
ever disagree with it, so this derives instead.

`b.dispatch_bq` resolves through `vk_context()` when the backend is unpinned, so
an unpinned backend answers "whichever device is current" — which is the correct
answer for it, and the reason the queue-only constructors need nothing extra.
"""
vk_context(b::LavaBackend) = (b.dispatch_bq.ctx)::VkContext
vk_context(a::LavaArray) = (a.buf[].ctx)::VkContext
# What a kernel library asks: `caps(backend)`, so a downstream package never has
# to hold a `VkContext` to find out what the device it is launching on offers.
#
# This IS `KernelInterface.caps` — imported at the top of `Lava.jl` — so these two
# methods are also Lava's whole implementation of KI's one required capability
# query. `matrix_shapes`, `supports`, `bestshape` and `wggranularity` derive from
# it in KI and are not written on this side at all.
caps(b::LavaBackend) = caps(vk_context(b))
caps(a::LavaArray) = caps(vk_context(a))
# A view of a device array is still on that device. `probe_broadcast!` is handed
# `dest`, which the broadcast machinery may hand it as a `SubArray` or a
# `ReshapedArray` — walking to the parent is the whole answer, and it is the same
# `AnyLavaArray` nest `_copyto!` already understands.
#
# One method per member of `AnyLavaArray` (gpuarrays.jl:91), and they have to stay
# in step: that Union lists six wrappers and this list had three, so
# `probe_broadcast!` — which does `vk_context(dest).diag.broadcast_probe` — threw
# a MethodError for `Adjoint`/`Transpose`. GPUArrays' own broadcasting suite
# caught it, 12 errors in "Adjoint and Transpose", because a transposed device
# array is an ordinary thing to broadcast over.
vk_context(a::SubArray) = vk_context(parent(a))
vk_context(a::Base.ReshapedArray) = vk_context(parent(a))
vk_context(a::Base.PermutedDimsArray) = vk_context(parent(a))
vk_context(a::LinearAlgebra.Transpose) = vk_context(parent(a))
vk_context(a::LinearAlgebra.Adjoint) = vk_context(parent(a))

# Property access resolves a `nothing`-pinned queue through the live
# `vk_context()` so a module-level `const BACKEND = LavaBackend()` keeps
# working across `reset_device!()`. `:bq` stays a back-compat alias for
# `:dispatch_bq`.
function Base.getproperty(b::LavaBackend, s::Symbol)
    if s === :dispatch_bq
        f = getfield(b, :dispatch_bq)
        return f === nothing ? vk_context().default_bq : f
    elseif s === :upload_bq
        f = getfield(b, :upload_bq)
        return f === nothing ? vk_context().default_bq : f
    elseif s === :bq
        return getproperty(b, :dispatch_bq)
    end
    return getfield(b, s)
end

# ── Backend queries ──

# Derive the backend from the array's own context so cross-context arrays
# dispatch against the correct queue instead of the global default.
KA.get_backend(a::LavaArray) = LavaBackend((a.buf[].ctx)::VkContext)
# KA.synchronize submits all recorded dispatches and waits for GPU completion.
# This matches CUDA/AMDGPU semantics: after synchronize(), the CPU can safely
# read GPU results. GPU-side ordering between dispatches is handled by pipeline
# barriers in record_dispatch!, so synchronize() is only needed when the CPU
# must observe GPU results (or at natural batch boundaries like end-of-sample).
function KA.synchronize(backend::LavaBackend)
    flush!(backend.dispatch_bq, backend.dispatch_bq.device)
    # If split, the upload queue may have pending transfers worth flushing too.
    if backend.upload_bq !== backend.dispatch_bq
        flush!(backend.upload_bq, backend.upload_bq.device)
    end
    return
end
KA.supports_unified(::LavaBackend) = true
function KA.allocate(backend::LavaBackend, ::Type{T}, dims::Tuple; unified::Bool=false) where T
    bq = backend.dispatch_bq
    nbytes = prod(dims) * sizeof(T)
    # Auto-unified for tiny allocations (≤ 64 bytes, e.g. WorkQueue.size
    # counters) — BAR memory enables direct CPU readback without staging.
    want_unified = unified || nbytes <= 64
    return LavaArray{T,length(dims)}(undef, Int.(dims); bq, unified=want_unified)
end
KA.unsafe_free!(x::LavaArray) = unsafe_free!(x)

function KA.copyto!(::LavaBackend, A, B)
    GC.@preserve A B begin
        copyto!(A, 1, B, 1, length(A))
    end
    return
end

# Adapt: convert Array ↔ LavaArray
# Use Adapt.adapt(LavaArray, a) for recursive element adaptation (like AMDGPU does).
# This handles non-isbits element types by recursively adapting struct fields.
Adapt.adapt_storage(::LavaBackend, a::Array) = Adapt.adapt(LavaArray, a)
Adapt.adapt_storage(::LavaBackend, a::LavaArray) = a
Adapt.adapt_storage(::KA.CPU, a::LavaArray) = Array(a)

# @Const support: stop Adapt recursion at the device array level.
# Without this, constify on SubArray{...,ReshapedArray{...,LavaDeviceArray}} recurses into
# ReshapedArray → calls Base.reshape → SignedMultiplicativeInverse constructor on GPU,
# which has a throw(ArgumentError("...$d")) that generates ijl_get_nth_field_checked.
# AMDGPU handles this by casting the pointer to constant address space; we just return
# the device array as-is since Vulkan BDA pointers are already readonly-capable.
Adapt.adapt_storage(::KA.ConstAdaptor, a::LavaDeviceArray) = a
# Prevent ReshapedArray reconstruction during constify — @Const marks readonly,
# it should not reconstruct wrapper arrays (which triggers SignedMultiplicativeInverse
# constructor with its throwing string interpolation path on GPU).
Adapt.adapt_structure(::KA.ConstAdaptor, A::Base.ReshapedArray) = A

# Type-based adapt_storage for Adapt.adapt(LavaArray, x) dispatch
Adapt.adapt_storage(::Type{<:LavaArray}, a::Array) = LavaArray(a)
Adapt.adapt_storage(::Type{<:LavaArray}, a::LavaArray) = a

# ── Launch configuration ──

"""
    caps(ctx).workgrouplimit :: Int

Largest workgroup Lava will ask the device for — its own
`maxComputeWorkGroupInvocations`. Requesting more throws.

This was a module-level `Ref(1024)` whose docstring said exactly the sentence
above while being a hardcoded constant, and before that it sat at **256** on a
diagnosis that was wrong in an instructive way and is worth keeping written down.

The recorded claim was that "above 256 this driver silently runs fewer
invocations than the shader declares" — a workgroup of 512 wrote half its output,
1024 wrote a quarter, no error anywhere. Everything about it pointed at hardware:
`spirv-val` passed, the SPIR-V declared `LocalSize 512 1 1`, the driver reported
identical Register Count for a body that failed and one that did not, and it was
body-dependent (64 simultaneously live values failed, 32 and 128 did not).

**None of it was the device.** A kernel reading `lava_local_invocation_index()`
with an unconditional store reports every lane running and writing at 256, 384,
512, 768 and 1024. The real cause was one line in `get_compute_pipeline`:

    cache_key = hash((spirv_bytes, ...))

`Base.hash` on a large `Vector` *samples* elements rather than reading all of
them. The 256- and 512-wide modules of one kernel differ at **exactly one byte**
— the `LocalSize` operand — and collide. So the 512 launch looked up the 256
pipeline, dispatched a 256-thread shader over a grid computed for 512, and wrote
`256/wg` of its output. Whichever size compiled first won, which is the whole of
the "order dependence"; whether a given body's two modules happened to collide is
the whole of the "body dependence"; and adding an unrelated store "fixed" it by
changing enough bytes to miss the collision.

See [`spirv_content_hash`](@ref), which reads every byte. `test/test_workgroup_limit.jl`
pins the coverage at every size on both launch spellings, and asserts directly
that two one-byte-different modules no longer share a cache key.

The lesson generalises past workgroups: **any** two SPIR-V modules differing only
in bytes the sampling hash skips shared a pipeline. That is a silent
wrong-results bug for any pair of kernel instantiations that differ in a literal.

See [`DeviceCaps`](@ref), which is where it lives now — the limit differs between
devices, so a process-wide `Ref` answered one device's question for all of them.
"""
workgroup_limit(ctx::VkContext = vk_context()) = caps(ctx).workgrouplimit

"The extents behind a KA size parameter, or `nothing` when they are dynamic."
@inline statictuple(::Type{<:KA.NDIteration.StaticSize{S}}) where {S} = S
@inline statictuple(@nospecialize(_)) = nothing

"""
Throw if `wg` exceeds [`workgroup_limit`](@ref) for `ctx`.

Loud rather than clamped: a kernel that stages into `@localmem` sizes its tile to
the workgroup it asked for, so quietly handing it a smaller one trades a wrong
answer for a different wrong answer.
"""
@inline check_workgroup(ctx::VkContext, ::Nothing; static::Bool = false) = nothing
@inline function check_workgroup(ctx::VkContext, wg; static::Bool = false)
    n = prod(wg)
    lim = workgroup_limit(ctx)
    n > lim || return wg
    throw(ArgumentError("""
        workgroupsize $wg has $n threads, above this device's limit of $lim,
        which is its `maxComputeWorkGroupInvocations`. Asking for more is
        a validation error, not a slow launch. See `DeviceCaps`."""))
end

function KA.launch_config(kernel::KA.Kernel{LavaBackend}, ndrange, workgroupsize)
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
        # Default workgroup size: 64 for 1D, capped to ndrange.
        #
        # `Val(length(ndrange))`, not a plain `Int`: the dynamic form goes through
        # `Base._ntuple`, whose body is `1:n` with `n` unknown, so it records an
        # edge on `Tuple{Colon, Int64, Any}`. Any package that adds a `(:)` method
        # — `Unitful` adds `Colon(::Any, ::Quantity)` — then makes that edge stale,
        # and every precompiled CodeInstance that inferred through it is thrown
        # away when the image loads. That single edge accounts for 25 025 of
        # Lava's 36 245 rejected CodeInstances when SAM 2's image loads after
        # VideoEditor. `Val` makes the tuple length static and the range vanishes.
        workgroupsize = ntuple(
            i -> i == 1 ? min(prod(ndrange), 64) : 1,
            Val(length(ndrange)))
        KA.partition(kernel, ndrange, workgroupsize)
    else
        # Both spellings land here, and only one of them is in `workgroupsize`:
        # `kernel(backend, wg)` puts the size in the kernel's TYPE and passes the
        # keyword as `nothing`, so checking the keyword alone would miss it. Which
        # of the two it is also decides the cap — see `check_workgroup`.
        static = workgroupsize === nothing
        # `vk_context(kernel.backend)`, not the global: the limit is the device's
        # and a KA kernel carries the backend it was built for.
        check_workgroup(vk_context(kernel.backend),
                        static ? statictuple(KA.workgroupsize(kernel)) : workgroupsize;
                        static = static)
        KA.partition(kernel, ndrange, workgroupsize)
    end

    return ndrange, workgroupsize, iterspace, dynamic
end

# ── Context creation ──

function KA.mkcontext(kernel::KA.Kernel{LavaBackend}, _ndrange, iterspace)
    KA.CompilerMetadata{KA.ndrange(kernel), KA.DynamicCheck}(_ndrange, iterspace)
end

function KA.mkcontext(kernel::KA.Kernel{LavaBackend}, I, _ndrange, iterspace, ::Dynamic) where Dynamic
    KA.CompilerMetadata{KA.ndrange(kernel), Dynamic}(I, _ndrange, iterspace)
end

# ── Argument conversion ──

# Convert kernel arguments for GPU compilation via Adapt.jl.
# LavaArray → LavaDeviceArray (Ptr-wrapping). The `LavaAdaptor(batch)` in this
# conversion path is the single place a LavaArray gets stripped to a pointer,
# and it pins the original LavaArray into `batch` at the same point.  There is
# no separate pinning walker — Adapt.jl's own recursion over wrapper structs /
# Broadcasted / NamedTuple lands every LavaArray at `adapt_storage`, and that
# method does both the strip and the pin.
#
# `obj.f` is the kernel closure — it goes through the same adaptor so any
# LavaArray captured in closure-over fields is pinned too.
#
# `KA.argconvert(kernel, arg)` is a pure strip used by callers (Raycore's
# MultiTypeSet, etc.) to cache a device-side view of a LavaArray into their
# own CPU-side structs *outside* any kernel dispatch.  No pin here — the
# caller is responsible for keeping the original LavaArray alive until the
# cached device form is used, and the subsequent kernel dispatch pins that
# original LavaArray via `LavaAdaptor`.
KA.argconvert(::KA.Kernel{LavaBackend}, a::LavaArray{T,N}) where {T,N} =
    LavaDeviceArray{T,N}(Ptr{T}(bda_address(a)), a.dims)
KA.argconvert(::KA.Kernel{LavaBackend}, x) = x

# ── Kernel call (main entry point) ──
#
# `find_tlas_in_args` scans the kernel's pre-adapt arg tuple for any value
# that holds a live VulkanTLAS reference, so callers can write
#   `kernel(accel, args...; ndrange=...)`
# with `accel::AdaptedAccel` (or any value carrying one) and have
# `enable_ray_query=true` + the HWTLAS descriptor binding wired automatically
# at compile + dispatch.  This keeps the SW/HW swap a single-arg change at
# the call site — Hikari's `Raycore.closest_hit(accel, ray)` polymorphism
# falls through unchanged.
#
# Whether an argument *can* carry a HWTLAS is a property of its type, so the scan
# is resolved at compile time: a compute kernel — every kernel in a DNN or
# broadcast workload — gets `nothing` and no code at all. The recursive
# `Base.tail` walk it replaces did not fold away on the long argument tuples KA
# produces (kernel args plus a `CompilerMetadata` plus a `Val` per static
# parameter), and showed up as the second-largest host cost in the MatAnyone
# inference loop, on kernels that have no HWTLAS and never could.
#
# Later arguments win, matching the accumulate-forward order of the walk.
@generated function find_tlas_in_args(args::Tuple)
    ts = fieldtypes(args)
    all(isconcretetype, ts) || return :(_find_tlas(args, nothing))
    expr = :(nothing)
    for (i, T) in enumerate(ts)
        hasfield(T, :hwtlas) || continue
        expr = :(let h = getfield(args[$i], :hwtlas)
                     h === nothing ? $expr : h
                 end)
    end
    expr
end
@inline _find_tlas(::Tuple{}, acc) = acc
@inline _find_tlas(args::Tuple, acc) = _find_tlas(Base.tail(args), _maybe_tlas(first(args), acc))
# `AdaptedAccel` is declared in raytracing/hwtlas.jl (loaded after this file);
# loose-typed dispatch + a runtime field check keeps the include order intact.
@inline function _maybe_tlas(x, acc)
    if hasfield(typeof(x), :hwtlas)
        h = getfield(x, :hwtlas)
        h !== nothing && return h
    end
    return acc
end

# Cached per-kernel iteration plan, on `vkctx.caches` — see `IterPlan`. Key =
# (typeof(obj), ndrange, workgroupsize); `typeof(obj)` carries the F type plus
# KA's static NDRange / WG-size type parameters, so two Kernel instances with the
# same shape share the plan. Built lazily on first launch.
#
# The device is not in the key and does not need to be, now that the cache
# belongs to one. As a process-wide dict it was a wrong-answer path: `block_dims`
# is `pad_to_3d(vkctx, …)` over the device's `max_wg_dims`, so the first device
# to launch a given kernel shape decided the block grid for every device after.
function get_or_build_iter_plan(obj::KA.Kernel{LavaBackend}, ndrange, workgroupsize,
                                vkctx::VkContext)
    # `K` is known at THIS call site — the caller has `ndrange` in hand — so the
    # scan below tests a concrete `IterEntry{K}` and compares an isbits tuple with
    # `===`. No hash of a heterogeneous key, no dynamic `isequal`. See `IterEntry`.
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

    ndr_canon, _ws, iterspace, _dyn = KA.launch_config(obj, ndrange, workgroupsize)
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
    bq.barrier_elision :: Bool

Drop the barrier in front of a dispatch whose buffers are disjoint from every
buffer touched since the last barrier.

Sound without read/write annotations: disjoint memory cannot alias, so it does
not matter which side reads and which writes. Unsound only if a kernel reaches
device memory that is not in its argument tree — Lava kernels address memory
through `LavaDeviceArray` BDAs derived from exactly that tree.

Off by default because the check is O(ranges since last barrier) *per dispatch*,
which is real host cost. It is meant to be turned on around `capture`: the
analysis is then paid once and every `replay!` gets the shorter command buffer
for free. On the MatAnyone step barriers are 3.35 ms of an 11.07 ms replay.
"""

# `bq.touched_ranges` is a flat [lo₁,hi₁,lo₂,hi₂,…] of everything touched since
# the last barrier on that queue, and `bq.dispatch_ranges` is scratch for the
# dispatch being recorded. Both live on the queue: the ranges describe one
# command buffer's contents, so a second queue recording concurrently would
# otherwise merge its writes into the first queue's set and let it elide a
# barrier it needed.

"""Reset the elision state — call whenever the queue is known to be drained."""
@inline reset_barrier_elision!(bq::VulkanBatchQueue) = (empty!(bq.touched_ranges); nothing)

"""
    poison_barrier_elision!(bq)

Record that something touched memory we cannot enumerate, so nothing may be
elided until a barrier clears it.

Only the KA launch path knows a dispatch's buffers. Everything else that records
into the same command buffer — `cmd_copy_buffer!`, `lava_launch!` used directly
by Lava's own internals, indirect prepares — writes memory this tracker never
sees. Left alone that is not conservative but *wrong*: a later dispatch reading
what a copy just wrote finds no overlap in the tracker and drops the barrier it
needed. It cost 0.024 of alpha on the MatAnyone step, which is the sort of small
plausible error a race gives you.

Modelled as one range covering the whole address space: every subsequent
dispatch overlaps it, takes its barrier, and clears it — no special cases.
"""
@inline function poison_barrier_elision!(bq::VulkanBatchQueue)
    touched = bq.touched_ranges
    empty!(touched)
    push!(touched, UInt64(0))
    push!(touched, typemax(UInt64))
    nothing
end

"""
True when `new` overlaps anything in `bq.touched_ranges`. On overlap the caller
emits a barrier and the set restarts from `new`; otherwise `new` is merged in and
the barrier is skipped.
"""
function barrier_needed!(bq::VulkanBatchQueue, new::Vector{UInt64})
    touched = bq.touched_ranges
    hit = false
    @inbounds for i in 1:2:length(new)
        lo, hi = new[i], new[i+1]
        for j in 1:2:length(touched)
            if lo < touched[j+1] && touched[j] < hi
                hit = true
                break
            end
        end
        hit && break
    end
    # The set only clears when a barrier fires, so a long run of elided
    # dispatches would make this scan quadratic. Past the cap, force a barrier
    # and start over — conservative, so it can only cost performance.
    if hit || length(touched) > 512
        empty!(touched)
        hit = true
    end
    append!(touched, new)
    hit
end

"""
    interior_unit_workgroup(W) -> Bool

Whether the static workgroup `W` has an extent of 1 in an *interior* dimension —
anything but the first or the last.

That is the exact trigger for a Lava codegen fault: with such a workgroup baked
into the kernel's TYPE, the block index decode takes the wrong divisor for
dimension 2 and only `min(1, blocks[3] / blocks[2])` of the output is written,
silently. A *trailing* unit extent is harmless, which is why ranks 1-3 look
clean — `(32, 4, 1)`'s unit extent is the last one.

Established as Lava's and not KernelAbstractions': the identical kernel,
workgroup and ndrange are correct on `KA.CPU()`, which runs the same `expand`
and `NDRange` machinery with no SPIR-V emitter anywhere. `test_static_workgroup.jl`
pins the trigger, the law, and the interior-versus-trailing rule.
"""
const WORKGROUP_FALLBACK = Ref(true)

@inline function interior_unit_workgroup(W::NTuple{N,Int}) where {N}
    N >= 3 || return false
    @inbounds for d in 2:(N - 1)
        W[d] == 1 && return true
    end
    false
end
interior_unit_workgroup(::Any) = false

"""
A second, independent trigger for the same silent fault: a workgroup in the
kernel's TYPE, launched on an ndrange whose **last extent is 1**, at rank ≥ 5.
The kernel writes part of its output and reports nothing — measured 2026-07-29:

    N=6  (576,16,16,4,4,1)  wg (16,2,2,2,2,1)   half written
    N=6  (288,4,4,32,32,1)  wg (16,2,2,2,2,1)   one sixteenth written
    N=5  (64,4,4,4,1)       wg (16,2,2,2,1)     half written
    N=6  (64,4,4,4,4,2)     wg (16,2,2,2,2,1)   correct — last extent is 2
    N=4  (256,256,144,1)    (via `permutedims!`) correct — rank 4 is unaffected

`interior_unit_workgroup` cannot see this one: `(16,2,2,2,2,1)` has no interior
unit extent, and its unit extent is *trailing*, which for the earlier fault was
harmless. The written fraction is not a clean function of the block counts (0.5
and 0.062 on shapes differing only in extents), so this pins the trigger and
makes no claim about a law.

Found through `permutedims!`, which returned wrong data for SAM 2's rank-6
window-partition shapes while the ordinary broadcast of the same permutation was
correct — the broadcast uses a linear ndrange and never reaches here.
"""
@inline function trailing_unit_ndrange(nd::NTuple{N,Int}) where {N}
    N >= 5 && @inbounds nd[N] == 1
end
trailing_unit_ndrange(::Any) = false

"""
Re-launch a kernel whose workgroup lives in its TYPE as one that takes the size
at the call, which is the path that computes correct indices.

Set `WORKGROUP_FALLBACK[] = false` to take the raw path anyway — only
`test_static_workgroup.jl` does that, to check the underlying fault is still
present and this guard is still load-bearing.

Only for shapes `interior_unit_workgroup` rejects, so this cannot change any
launch that is currently right — it turns a silently wrong answer into a correct
one. It is slower (the index arithmetic stops being compile-time constant, ~2x on
the kernels measured), and that is the trade until the codegen fault is fixed.
"""
@inline function relaunch_dynamic(obj::KA.Kernel{LavaBackend, KA.NDIteration.StaticSize{W}, ND, F},
                                  args, ndrange) where {W, ND, F}
    dyn = KA.Kernel{LavaBackend, KA.NDIteration.DynamicSize, ND, F}(obj.backend, obj.f)
    dyn(args...; ndrange = ndrange, workgroupsize = W)
end

function (obj::KA.Kernel{LavaBackend})(args...; ndrange=nothing, workgroupsize=nothing)
    # `vk_context(obj.backend)`, not the global: a KA kernel carries the backend
    # it was built for, and that backend knows its device. This is the accessor
    # `test_backend_context.jl` exists to pin.
    validate_launch_args(vk_context(obj.backend), args)
    # A static workgroup with an interior unit extent miscompiles; take the
    # dynamic path instead. Checked on the TYPE, so it folds away for every
    # kernel that is not affected.
    if WORKGROUP_FALLBACK[] && workgroupsize === nothing && ndrange isa Tuple
        WGT = typeof(obj).parameters[2]
        if WGT <: KA.NDIteration.StaticSize &&
           (interior_unit_workgroup(WGT.parameters[1]) || trailing_unit_ndrange(ndrange))
            return relaunch_dynamic(obj, args, ndrange)
        end
    end
    bq = obj.backend.bq

    # Auto-discover HWTLAS for ray-query kernels — extract BEFORE Adapt strips
    # hwtlas (kernel form has hwtlas=nothing).
    tlas = find_tlas_in_args(args)

    # GPU-resident ndrange → indirect dispatch (no CPU readback)
    if ndrange isa LavaArray
        batch = ensure_active_batch!(bq)
        pin_leaves!(batch, obj.f)
        pin_leaves!(batch, args)
        adaptor = LavaAdaptor(batch)
        converted_args = map(a -> Adapt.adapt(adaptor, a), args)
        ka_launch_indirect!(obj, converted_args, ndrange, workgroupsize, args, adaptor, bq, tlas)
        return nothing
    end

    # KA's launch_config / partition / mkcontext / blocks / workitems plus our
    # pad_to_3d add up to ~50% of per-record cost in tight loops, yet they only
    # depend on (typeof(obj), ndrange, workgroupsize) — typeof(obj) carries the
    # static NDRange/WG-size and F type parameters. Cache the whole plan so the
    # second-and-later launch with the same shape is one Dict lookup.
    plan = get_or_build_iter_plan(obj, ndrange, workgroupsize, bq.ctx::VkContext)
    # NOTE the `nblocks == 0` check lives inside the barrier, not here. Reading
    # any field off `plan` at this point is a dynamic `getfield` on an abstract
    # value, which boxes — the very cost the barrier exists to remove.
    # FUNCTION BARRIER, and it is worth 176 bytes on every dispatch.
    #
    # `IterPlan{Ctx}` is parametric, so the `::IterPlan` the cache lookup can
    # promise is a UnionAll, not a concrete type. Reading `plan.ka_ctx` /
    # `.block_dims` / `.ws_3d` inline here is therefore a dynamic `getfield` that
    # boxes its result — 32 bytes each, measured — and because `ka_ctx` then has
    # no type, the `all_args` tuple downstream cannot be inferred either, for
    # another 80.
    #
    # Handing `plan` to a separate function costs ONE dynamic dispatch and makes
    # every use of it inside concrete. This is the standard cure for an
    # unavoidably-abstract value on a hot path, and the alternative (threading
    # `Ctx` through the caller) would infect the whole entry point with a type
    # parameter that exists only to satisfy inference.
    return launch_planned!(bq, obj, args, plan, tlas)
end

"""
    compiled(kernel, ndrange; workgroupsize = nothing) -> LavaKernel

A kernel with its iteration plan already resolved and **concretely typed**, so a
launch is a static call with no cache lookup and no boxing.

The plain path cannot get there. `IterPlan{Ctx}` is parametric, the per-device
cache can only promise the `IterPlan` UnionAll, and every route out of an
abstract container costs at least one dynamic dispatch — measured, on an
otherwise-warm dispatch that builds nothing:

    abstract cache, fields read inline   96 bytes
    abstract cache + function barrier    16 bytes     <- the plain path today
    plan held in a concrete field         0 bytes     <- this

`Ctx` is a pure function of the kernel TYPE (verified: identical for ndrange 16,
1000, 1024 and 4096) and is `isbits`, so resolving it once at construction is
sound — the plan's *values* depend on `ndrange`, its *type* never does. That is
what makes holding it legitimate rather than a cache that can go stale on shape.

`P` is a type parameter so a caller that stores a `LavaKernel` in a concretely
typed field gets `lk.plan` for free. Storing one in a `::LavaKernel` field (the
UnionAll) throws that away and lands back on the dynamic path — parameterise the
holder, e.g. `struct MyOp{KK}; kern::KK; end`.
"""
struct LavaKernel{K,P,Q}
    inner::K
    plan::P
    # The queue, RESOLVED. `LavaBackend` stores `dispatch_bq::Union{VulkanBatchQueue,
    # Nothing}` and has no `bq` field, so `backend.bq` goes through an accessor
    # that does not infer concretely — and leaving it abstract made this whole
    # idea BACKFIRE: the call below stayed dynamic, and a dynamic call has to box
    # its arguments, so a concrete `IterPlan{Ctx}` (a large isbits struct) got
    # copied into a fresh box on every launch. Measured 144 B/dispatch, i.e.
    # THREE TIMES the abstract-plan barrier it was meant to beat. Typing the plan
    # is worthless unless the call it feeds is static.
    bq::Q
end

function compiled(obj::KA.Kernel{LavaBackend}, ndrange; workgroupsize = nothing)
    bq = obj.backend.bq
    plan = get_or_build_iter_plan(obj, ndrange, workgroupsize, bq.ctx::VkContext)
    # `P` and `Q` come from the runtime types here, once, off the hot path.
    return LavaKernel(obj, plan, bq)
end

"""Launch a [`compiled`](@ref) kernel: no lookup, no barrier, static dispatch."""
function (lk::LavaKernel)(args...)
    tlas = find_tlas_in_args(args)
    return launch_planned!(lk.bq, lk.inner, args, lk.plan, tlas)
end

@noinline function launch_planned!(bq::VulkanBatchQueue, obj::KA.Kernel{LavaBackend},
                                   args::Tuple, plan::IterPlan, tlas)
    plan.nblocks == 0 && return nothing
    ka_ctx     = plan.ka_ctx
    block_dims = plan.block_dims
    ws_3d      = plan.ws_3d

    batch = ensure_active_batch!(bq)
    # Side-effect pass: pin every LavaArray leaf in the closure + args into
    # `batch.pinned`, once, via @generated walker (zero alloc, straight-line
    # code).  `Adapt.adapt` below is now pure — it only strips.
    pin_leaves!(batch, obj.f)
    pin_leaves!(batch, args)
    # Same tree the pins just walked, so every buffer this dispatch can reach is
    # covered. Decided here rather than in `record_dispatch!` because this is the
    # last point that still has the pre-adapt arguments.
    if bq.barrier_elision
        ranges = bq.dispatch_ranges
        empty!(ranges)
        range_leaves!(ranges, obj.f)
        range_leaves!(ranges, args)
        bq.next_skip_barrier = !barrier_needed!(bq, ranges)
        bq.ranges_declared = true
    end
    adaptor = LavaAdaptor(batch)
    converted_f = Adapt.adapt(adaptor, obj.f)
    converted_args = map(a -> Adapt.adapt(adaptor, a), args)
    all_args = (converted_f, ka_ctx, converted_args...)

    ka_launch!(bq, converted_f, all_args, block_dims, ws_3d, tlas)

    return nothing
end

# Vulkan dispatch is max 3D. pad_to_3d maps the N-D block grid to a 3D dispatch,
# splitting large dimensions across Y/Z when needed to stay within device limits.
# Uses actual device maxComputeWorkGroupCount (queried once, cached).
#
# CRITICAL: pad_to_3d must NEVER over-dispatch (produce more workgroups than requested).
# Kernels like AK._accumulate_block! write to auxiliary arrays indexed by workgroup ID
# without bounds checks — phantom workgroups cause out-of-bounds GPU memory writes.

# Per-dimension workgroup count limits live on `VkContext.max_wg_dims` — queried
# once at device creation. `pad_to_3d` and friends reach them via `bq.ctx`.

# Find exact factor of n that is ≤ max_dim, for splitting workgroup counts.
# Returns the largest factor ≤ max_dim, or max_dim if none found (over-dispatch).
function find_split_factor(n::Int, max_dim::Int)
    # Try exact division first (fast path for powers of 2, multiples of common factors)
    for d in (max_dim, max_dim-1, max_dim-2, max_dim-3)
        d > 0 && n % d == 0 && return d
    end
    # Try small factors of n that would make the other dimension ≤ max_dim
    # We need X such that X ≤ max_dim and n/X ≤ max_dim (for 2D split)
    # So X ≥ cld(n, max_dim)
    lo = cld(n, max_dim)
    for x in lo:min(n, max_dim)
        n % x == 0 && return x
    end
    # No exact factor found — find tightest over-approximation
    # This should be extremely rare (requires n > max_dim^2 with no factors in range)
    return max_dim
end

function pad_to_3d(ctx::VkContext, t::NTuple{1,<:Integer})
    n = Int(t[1])
    max_x, max_y, _ = ctx.max_wg_dims
    n <= max_x && return (n, 1, 1)
    # Need to split into X * Y (or X * Y * Z)
    # Find X such that X divides n exactly and X ≤ max_x
    x = find_split_factor(n, max_x)
    y = cld(n, x)
    if x * y == n && y <= max_y
        return (x, y, 1)
    end
    # 2D split didn't work exactly, try 3D
    if y > max_y
        # Flatten into 3D
        xy = x * min(y, max_y)
        z = cld(n, xy)
        return (x, min(y, max_y), z)
    end
    # Over-dispatch (y > exact). This is dangerous for kernels without bounds checks.
    @warn "pad_to_3d: over-dispatching $n as ($x, $y, 1) = $(x*y) workgroups" maxlog=1
    return (x, y, 1)
end
function pad_to_3d(ctx::VkContext, t::NTuple{2,<:Integer})
    x, y = Int(t[1]), Int(t[2])
    max_x, max_y, _ = ctx.max_wg_dims
    if x <= max_x && y <= max_y
        return (x, y, 1)
    end
    # Flatten and re-split
    return pad_to_3d(ctx, (x * y,))
end
function pad_to_3d(::VkContext, t::NTuple{3,<:Integer})
    (Int(t[1]), Int(t[2]), Int(t[3]))
end
# For N>3: drop trailing unit axes first, and only flatten if something non-unit
# is left past the third.
#
# A trailing `1` carries no information — `(8, 1152, 4, 1)` is the same grid as
# `(8, 1152, 4)` — but flattening it to 1D and re-splitting loses the 1:1
# correspondence between the dispatch and the block grid, which is exactly what
# `directdispatch` needs. Rank-4 ndranges ending in a batch of 1 are the common
# shape in this codebase (`(W, H, C, 1)`), so this is what lets them index
# without dividing.
function pad_to_3d(ctx::VkContext, t::NTuple{N,<:Integer}) where N
    n = N
    while n > 3 && Int(t[n]) == 1
        n -= 1
    end
    n > 3 && return pad_to_3d(ctx, (Int(prod(t)),))
    return pad_to_3d(ctx, ntuple(i -> Int(t[i]), n))
end

"""
Internal launch function for KA kernels. Compiles and dispatches the GPU function.
"""

# Per device: a `LaunchPlan` holds a `VkPipeline` (GUARDRAILS §8). Same shape as
# `LINKED_KERNEL_CACHE` — an outer dict keyed by `ctx.id`, so the inner one keeps
# the `DataType` key that makes the lookup a pointer compare.

@inline launch_plan_cache(ctx) = ctx.caches.launchplans

@inline function launch_plan(bq::VulkanBatchQueue, @nospecialize(f), all_args::Tuple,
                             wg::NTuple{3,Int}, ray_query::Bool)
    key = typeof(all_args)
    world = Base.get_world_counter()
    # `bq.ctx::VkContext`, and the assert is the whole point: `VulkanBatchQueue.ctx` is
    # declared `::Any` (it must be — `VkContext` owns the queue, so one direction
    # of the cycle is untyped). Without the assert `ctx.caches.launchplans` infers
    # as `Any`, which makes the `get` below a dynamic dispatch and the loop over
    # `v` a fully dynamic iteration — measured at **464 bytes per dispatch**, on a
    # warm cache that builds nothing. Every other `bq.ctx` in this file already
    # carries the assert; this one did not.
    cache = launch_plan_cache(bq.ctx::VkContext)
    v = get(cache, key, nothing)
    if v !== nothing
        for q in v
            # `q::LaunchPlan` on an element of a `Vector{Any}` is a pointer
            # check, not a copy. Iterating a `Vector{LaunchPlan}` instead boxes
            # 64 bytes here on every dispatch — see `DeviceCaches.launchplans`.
            p = q::LaunchPlan
            (p.world === world && p.wg === wg && p.ray_query === ray_query) && return p
        end
    end
    build_launch_plan!(bq, f, all_args, wg, ray_query, key, world)
end

# ── The function barrier, and why this is two functions instead of one.
#
# `all_args` is a fully concrete tuple type — a different one for every kernel in
# the program — so ANY method taking it is inferred once per kernel. That is
# correct and unavoidable for the code that packs arguments, which genuinely
# needs each type. It is pure waste for everything downstream of it, which needs
# only `tt`.
#
# Measured on SAM 2's encoder, first call in a fresh process: 16 981
# MethodInstances inferred, 11 729 of them owned by Lava, ~26 s of the ~36 s of
# first-call latency. The launch path was ~1 000 of each of `build_launch_plan!`,
# `get_compiled_kernel_and_pipeline`, `frozen_load` and `frozen_store` — one per
# kernel, for four functions whose bodies do not depend on the kernel at all.
# `@nospecialize` was already on `f` and `tt` and did not help: inference still
# specialises a callee to inline it into a specialised caller, so the widening
# has to be paired with a barrier the compiler will not cross.
#
# This is CUDA.jl's shape (`cudacall` keeps the argument-dependent part to a thin
# shell over a type-erased worker). The split point is `tt`: everything above it
# is per-kernel and tiny, everything below is per-kernel-invariant and large.
@noinline function build_launch_plan!(bq::VulkanBatchQueue, @nospecialize(f), all_args::Tuple,
                                      wg::NTuple{3,Int}, ray_query::Bool,
                                      key::DataType, world::UInt64)
    # Excludes f — GPUCompiler prepends typeof(f). all_args are already
    # post-adapt (LavaDeviceArray, not Ptr{T}).
    #
    # Computed here, and only here, because `typeof(all_args)` cannot stand in for
    # it: a type-valued argument erases to `DataType` in the tuple type, and
    # `arg_sigtype` recovers the `Type{X}` GPUCompiler needs.
    tt = Tuple{map(arg_sigtype, Base.tail(all_args))...}
    # `key` is boxed for the same reason `tt` is, and it is worth naming because
    # it is easy to miss: `key::DataType` looks like an ordinary value parameter,
    # but a *type-valued* argument enters the lattice as `Type{Tuple{…}}`, so
    # Julia specialises on it exactly as it would on `tt`. Leaving it unboxed
    # while boxing the other two took this from 1 238 specialisations to 909, not
    # to 1.
    build_launch_plan_tt!(bq, Ref{Any}(f), Ref{Any}(tt), Ref{Any}(key), wg, ray_query, world)
end

"""The half of `build_launch_plan!` that does not depend on the kernel's argument
types — inferred once for the whole program rather than once per kernel.

**The `Ref{Any}` boxes are the erasure, and `@nospecialize` is not a substitute.**
Julia ignores `@nospecialize` for exactly the two kinds of argument this takes:
specialising on a singleton function type is free, and a `Type{T}` argument is
how dispatch is spelled, so both are always specialised on. Measured: with
`@nospecialize` on both and `@noinline` on the method, this still had 1 238
specialisations, one per kernel — the annotation had no effect at all. Boxed, the
signature is `RefValue{Any}` twice and there is one.

The cost is that `f` and `tt` come out as `Any` and every call below here is a
dynamic dispatch. That is the right trade on this path and only on this path:
`launch_plan` reaches it solely on a cache miss, i.e. once per kernel per world,
and what follows is a SPIR-V compile. On the hit path nothing here runs."""
@noinline function build_launch_plan_tt!(bq::VulkanBatchQueue, fbox::Base.RefValue{Any},
                                         ttbox::Base.RefValue{Any},
                                         keybox::Base.RefValue{Any},
                                         wg::NTuple{3,Int}, ray_query::Bool,
                                         world::UInt64)
    f = fbox[]
    tt = ttbox[]
    key = keybox[]::DataType
    compiled, pipeline, offsets, byval_sizes = get_compiled_kernel_and_pipeline(
        bq.ctx::VkContext, f, tt, wg; enable_ray_query=ray_query)
    base = compiled.push_info.arg_buffer_size
    p = LaunchPlan(compiled, pipeline, offsets, byval_sizes, base,
                   base + compute_inline_extra_from_byval(byval_sizes),
                   world, wg, ray_query)
    v = get!(() -> Any[], launch_plan_cache(bq.ctx), key)
    filter!(q -> (q::LaunchPlan).world === world, v)  # superseded worlds are dead
    push!(v, p)
    p
end

function ka_launch!(bq::VulkanBatchQueue, @nospecialize(f), all_args::Tuple,
                    block_dims::NTuple{3,Int}, workgroup_size::NTuple{3,Int},
                    tlas=nothing)  # positional, Nothing default — hot path
    # When `tlas` was auto-discovered from kernel args (e.g. an AdaptedAccel
    # was passed), enable ray_query so the SPIR-V emitter binds the HWTLAS
    # descriptor and accepts OpRayQueryInitializeKHR / Proceed / Get*KHR.
    plan = launch_plan(bq, f, all_args, workgroup_size, tlas !== nothing)

    # Get host-visible mapped arg buffer (per-BQ slab pool)
    arg_buf = get_arg_buffer(bq, plan.total_size)

    # The KA.Kernel entry point already ran `Adapt.adapt(LavaAdaptor(batch), ..)`
    # on each original arg, which both pinned every LavaArray (and nested
    # LavaArrays in wrapper structs) AND stripped them to LavaDeviceArray.
    # Here `all_args` is post-adapt, so pack sees no further pinnable leaves.
    ensure_active_batch!(bq)
    pack_args_direct!(bq, arg_buf.mapped_ptr, arg_buf.address, plan.offsets,
                       plan.arg_buffer_size, plan.byval_sizes, all_args)

    # Dispatch with N-D block grid (preserves KA's block dimensions)
    if (bq.ctx::VkContext).diag.dispatch_logging
        bq.last_dispatch_info = Base.invokelatest(dispatch_log_string, "ka f=",
                                   dispatch_name(f, all_args), " groups=", block_dims)::String
    end
    vk_dispatch!(bq, plan.pipeline, arg_buf.address, block_dims, tlas)

    return nothing
end

# ── Indirect dispatch support ──
#
# Internal Lava kernels only ever see `LavaDeviceArray` / `LavaDeviceArray`
# — never `Ptr{T}` / `unsafe_load`.  Callers wrap raw GPU memory in
# `LavaArray`/`LavaArray view` before handing it to the kernel.

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

# ── Fast prepare-indirect path ──
# Bypasses full lava_launch! to avoid per-dispatch ceremony overhead.
# The prepare-indirect kernel is always the same function with the same types,
# so we compile once and cache everything. Saves ~30K lava_launch! calls per render
# (hash lookups, validation, auto-flush, keep_alive, etc.).


# Register cleanup callback for reset_device!

function init_prepare_indirect_pipeline!(ctx::VkContext)
    ctx.caches.prepare_indirect === nothing || return
    # Kernel signature (post-adapt): indirect::LavaDeviceArray{UInt32,1},
    # ndrange_buf::LavaDeviceArray{Int32,1}, ws::UInt32.
    tt = Tuple{LavaDeviceArray{UInt32,1}, LavaDeviceArray{Int32,1}, UInt32}
    ws = (1, 1, 1)
    compiled, pipeline, offsets, byval_sizes = get_compiled_kernel_and_pipeline(
        ctx, prepare_indirect_kernel, tt, ws)
    ctx.caches.prepare_indirect = PrepareIndirect(pipeline, offsets, byval_sizes,
                                               compiled.push_info.arg_buffer_size)
end

"""
    fast_prepare_indirect!(bq, indirect::LavaArray{UInt32,1}, ndrange_buf::LavaArray{<:Integer}, workgroup_size)

Fast path for prepare-indirect dispatch.  Bypasses lava_launch!'s validation/
logging overhead: manually adapts the two LavaArrays (pin + strip) and
packs directly.
"""
function fast_prepare_indirect!(bq::VulkanBatchQueue,
                                indirect::LavaArray{UInt32,1},
                                ndrange_buf::LavaArray{<:Integer},
                                workgroup_size::Integer)
    ctx = bq.ctx::VkContext
    init_prepare_indirect_pipeline!(ctx)

    pi = ctx.caches.prepare_indirect::PrepareIndirect
    pipeline, offsets = pi.pipeline, pi.offsets
    byval_sizes, arg_size = pi.byval_sizes, pi.arg_buffer_size

    batch = ensure_active_batch!(bq)
    adaptor = LavaAdaptor(batch)
    dev_indirect = Adapt.adapt(adaptor, indirect)::LavaDeviceArray{UInt32,1}
    dev_ndrange  = Adapt.adapt(adaptor, ndrange_buf)

    # f is a ghost singleton (prepare_indirect_kernel), pack_args_direct! skips it.
    all_args = (prepare_indirect_kernel, dev_indirect, dev_ndrange, UInt32(workgroup_size))
    arg_buf = get_arg_buffer(bq, arg_size)
    pack_args_direct!(bq, arg_buf.mapped_ptr, arg_buf.address, offsets, arg_size, byval_sizes, all_args)

    vk_dispatch_base!(bq, pipeline, arg_buf.address, 0, 0, 0, 1, 1, 1)
end

function prepare_indirect_dispatch!(bq::VulkanBatchQueue,
                                    indirect::LavaArray{UInt32,1},
                                    ndrange_buf::LavaArray{<:Integer},
                                    workgroup_size::Integer)
    lava_launch!(bq, prepare_indirect_kernel,
                 indirect, ndrange_buf, UInt32(workgroup_size);
                 ndrange=1, workgroup_size=(1, 1, 1))
end

"""
    ka_launch_indirect!(obj, args, ndrange_buf, workgroupsize)

Launch a KA kernel using indirect dispatch. `ndrange_buf` is a GPU array containing
the work item count (1-element Int32 array). The prepare-indirect kernel writes
group counts to an indirect buffer, then vk_dispatch_indirect! dispatches the main kernel.
"""
function ka_launch_indirect!(obj, args, ndrange_buf::LavaArray, workgroupsize, original_args,
                             adaptor::LavaAdaptor,
                             bq::VulkanBatchQueue=obj.backend.bq,
                             tlas=nothing)  # positional — same NamedTuple-avoidance as ka_launch!
    # Respect static workgroup size from @kernel definition
    ws = if workgroupsize !== nothing
        workgroupsize isa Integer ? (workgroupsize,) : workgroupsize
    elseif KA.workgroupsize(obj) <: KA.StaticSize
        KA.get(KA.workgroupsize(obj))
    else
        (256,)
    end
    ws_3d = pad_to_3d(bq.ctx::VkContext, ws)
    ws_prod = prod(ws)

    # We need to compile the kernel with a static ndrange for __validindex.
    # Use DynamicCheck so the kernel checks bounds at runtime via the iterspace.
    # We compile with a large ndrange; the actual dispatch count comes from indirect buffer.
    # Key insight: __validindex checks `I in __ndrange(ctx)`, and __ndrange(ctx) returns
    # the ndrange from CompilerMetadata. We set this to a large value so threads always pass.
    # The kernel's own bounds check (e.g. `if i <= queue.size[1]`) handles the real bound.
    big_ndrange = (1024 * 1024 * ws_prod,)
    ndrange_tuple = length(ws) == 1 ? big_ndrange :
                    length(ws) == 2 ? (big_ndrange[1], 1) :
                    (big_ndrange[1], 1, 1)
    iterspace, dynamic = KA.partition(obj, ndrange_tuple, ws)
    ctx = KA.mkcontext(obj, ndrange_tuple, iterspace)

    # Caller already pinned the original args via `pin_leaves!`; pin the
    # closure's captures here. `Adapt.adapt` is pure now (strip only).
    batch = adaptor.batch
    pin_leaves!(batch, obj.f)
    converted_f = Adapt.adapt(adaptor, obj.f)
    all_args = (converted_f, ctx, args...)

    # Kernel ABI is post-adapt: LavaDeviceArray etc.
    tt = Tuple{map(arg_sigtype, Base.tail(all_args))...}
    enable_ray_query = tlas !== nothing
    compiled, pipeline, offsets, byval_sizes = get_compiled_kernel_and_pipeline(
        bq.ctx::VkContext, converted_f, tt, ws_3d; enable_ray_query)

    inline_extra = compute_inline_extra_from_byval(byval_sizes)
    total_size = compiled.push_info.arg_buffer_size + inline_extra

    # GC.@preserve on original_args keeps the pre-strip LavaArrays reachable
    # through this function.
    GC.@preserve original_args begin

    arg_buf = get_arg_buffer(bq, total_size)
    @assert batch === bq.active_batch  "adaptor batch diverged from active bq batch"
    pack_args_direct!(bq, arg_buf.mapped_ptr, arg_buf.address, offsets,
                       compiled.push_info.arg_buffer_size, byval_sizes, all_args)

    indirect_view = get_indirect_buffer(bq)

    if (bq.ctx::VkContext).diag.dispatch_logging
        bq.last_dispatch_info = Base.invokelatest(dispatch_log_string, "indirect f=",
                                   dispatch_name(obj.f, all_args))::String
    end
    deferred = bq.deferred_indirect
    if deferred !== nothing
        # Inside `concurrent_indirect_group`: record NOTHING here — both the
        # prepare (fused into one multi-prepare dispatch) and the indirect
        # dispatch happen at the group's flush, so all pairs in the group
        # share two barriers total and the dispatches overlap on the GPU.
        # The packed args + indirect slot stay valid across the gap: both
        # live in this batch's slabs and the pool-reset guard keeps the
        # cursors from rewinding while the batch holds recorded dispatches.
        push!(deferred, (bq, pipeline, arg_buf.address, indirect_view, tlas,
                         ndrange_buf, ws_prod))
    else
        fast_prepare_indirect!(bq, indirect_view, ndrange_buf, ws_prod)
        vk_dispatch_indirect!(bq, pipeline, arg_buf.address, indirect_view, tlas)
    end

    end # GC.@preserve

    return nothing
end

# ── Device code is not here ──
#
# Everything a kernel BODY calls left this file for Lava when the runtime moved:
# `getindex`/`setindex!`/`linear_index` on a `LavaDeviceArray` are in
# `Lava/src/device/devicearray.jl`, and the `KA.__index_*`, `KA.__validindex`,
# `KA.__synchronize` and `KA.Scratchpad` overrides are in
# `Lava/src/device/ndrange.jl`.
#
# They were still here after the split, which meant Lava could not compile a
# kernel that indexed an array without Mantle loaded — silently, because the
# result is a valid empty entry point rather than an error. What stays in this
# file is the HOST half: the backend struct, launch configuration, argument
# packing and dispatch.
