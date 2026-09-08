#
# Lava's implementation of `KernelInterface` (KI), HOST side.
#
# The split is where the dependency is. Everything in `device/kernelinterface.jl`
# is code a KERNEL runs — index queries, barriers, shuffles, printing — and needs
# only the compiler. Everything here needs a device: a queue to submit to, a pool
# to allocate from, a batch to pin into. That is the runtime, so this is the half
# that travels with it.
#
# `KI.caps(::LavaBackend)` is in neither: `caps(b::LavaBackend)` in
# `array/ka_backend.jl` already IS that method, because `caps` is imported from
# KI at the top of `Lava.jl` rather than being a Lava function of the same name.
# `matrix_shapes`, `supports`, `bestshape` and `wggranularity` then derive from it
# inside KI, with nothing to implement on this side at all.
#

import KernelInterface as KI

KI.synchronize(backend::LavaBackend) = KA.synchronize(backend)

KI.allocate(backend::LavaBackend, ::Type{T}, dims::Tuple; unified::Union{Nothing,Bool} = nothing) where {T} =
    KA.allocate(backend, T, dims; unified = something(unified, false))

KI.get_backend(a::LavaArray) = KA.get_backend(a)

KI.supports_unified(::LavaBackend) = true
KI.supports_atomics(::LavaBackend) = true

# ── The KA 0.9 gap ──────────────────────────────────────────────────────────
#
# `LavaBackend <: KA.GPU`, and on KernelAbstractions 0.9 that is KA's OWN
# `Backend` hierarchy — a different abstract type from `KI.Backend`, because 0.9
# predates KernelInterface. Julia has single inheritance, so `LavaBackend` cannot
# be both, and KI's derived methods (`matrix_shapes(::Backend)` and the
# `supports`/`bestshape` forwards beside it) do not reach it.
#
# So the forwards are written here. Three lines, and each is the SAME body KI
# has — they are not a second implementation, they are the dispatch KI would have
# done if the type hierarchies had been one. Everything they forward to,
# including the `coopmat` gate, is still KI's and written once.
#
# This block is deleted by the KernelAbstractions 0.10 upgrade, where
# `KA.Backend` IS `KI.Backend` and the derivations apply directly. It is the
# clearest cost of staying on 0.9, and it is three lines.
KI.matrix_shapes(b::LavaBackend) = KI.matrix_shapes(caps(b))
KI.supports(b::LavaBackend, s::MatrixShape) = KI.supports(caps(b), s)
KI.bestshape(b::LavaBackend, ab, acc; scope::MatrixScope = SubgroupScope()) =
    KI.bestshape(caps(b), ab, acc; scope)

# Which element types `shfl_down` covers. It dispatches on a backend — a host
# handle — so it is here, while the methods themselves are generated on the
# device side. `KI_SHFL_TYPES` is the one list both read, so KI's own suite
# cannot end up exercising a type Lava never generated.
KI.shfl_down_types(::LavaBackend) = collect(KI_SHFL_TYPES)

# …and the same for the reduce-add, off its own list for the reason given beside
# `KI_REDUCE_ADD_TYPES`: the two families are separate capabilities and only
# happen to cover the same six types on this backend.
KI.sub_group_reduce_add_types(::LavaBackend) = collect(KI_REDUCE_ADD_TYPES)

"""
The width the shader will actually run at, from the device rather than a guess:
32 on NVIDIA, 32 *or* 64 on RDNA3 depending on how the driver compiled the
shader, 8 on lavapipe. Nothing may hard-code it.
"""
KI.sub_group_size(backend::LavaBackend) = caps(backend).subgroup

KI.max_work_group_size(backend::LavaBackend) = caps(backend).workgrouplimit

KI.multiprocessor_count(backend::LavaBackend) = caps(backend).cores

# `KI.caps(::LavaBackend)` is not written here — `caps(b::LavaBackend)` in
# `array/ka_backend.jl` IS it, because `caps` is imported from KI at the top of
# `Lava.jl` rather than being a Lava function of the same name. So the one method
# Lava already had satisfies the interface, and `matrix_shapes`, `supports`,
# `bestshape` and `wggranularity` all derive from it with nothing further to
# implement. That is what moving `DeviceCaps` into KI bought.

# The device limit, until the driver gives a per-kernel one. A kernel whose
# register pressure lowers it would report less, which is why KI asks per kernel
# rather than once — VK_KHR_pipeline_executable_properties is where that number
# would come from.
KI.kernel_max_work_group_size(backend::LavaBackend, kernel; max_work_items::Int = typemax(Int)) =
    min(KI.max_work_group_size(backend), max_work_items)

# ── Launch ──────────────────────────────────────────────────────────────────
#
# KI's `@kernel` macro is sugar over three calls a backend supplies:
#
#     kernel_f = argconvert(backend, f)
#     tt       = Tuple{map(Core.Typeof, map(x -> argconvert(backend, x), args))...}
#     kernel   = kernel_function(backend, kernel_f, tt)
#     kernel(args...; numworkgroups = …, workgroupsize = …)
#
# Note the kernel is invoked with the ORIGINAL args: `argconvert` exists to
# derive the compiled signature, and the adaptation that pins device memory
# happens at launch. That is Lava's split already, which is why this is an
# adapter over `ka_launch!` and not a second launch path.

"""
A pure strip to the device-side form, matching `KA.argconvert`: no pin, because
KI uses this only to derive the type tuple to compile against. The pinning
adaptation happens in the launch below, against the original arguments.
"""
KI.argconvert(::LavaBackend, a::LavaArray{T,N}) where {T,N} =
    LavaDeviceArray{T,N}(Ptr{T}(bda_address(a)), a.dims)
KI.argconvert(::LavaBackend, x) = x

# The signature is compiled from the actual arguments at launch, so `tt` is not
# held here; `KI.Kernel` carries the function and the backend, which is all the
# launch needs. `tt` still defaults, because KI documents the two-argument form.
#
# `name` is accepted and ignored: Lava's kernels are named after the function
# they were compiled from, and there is nowhere to put an override that the
# SPIR-V module or the pipeline cache would read back.
KI.kernel_function(backend::LavaBackend, @nospecialize(f), @nospecialize(tt) = Tuple{};
                   name = nothing, kwargs...) =
    KI.Kernel(backend, f)

"""Normalise KI's scalar / tuple / empty launch extents to a 3-tuple."""
@inline function ki_extent(x, default::Int)
    x isa Integer && return (Int(x), default, default)
    n = length(x)
    n == 0 && return (default, default, default)
    return (Int(x[1]),
            n >= 2 ? Int(x[2]) : default,
            n >= 3 ? Int(x[3]) : default)
end

function (k::KI.Kernel{LavaBackend})(args...;
                                     numworkgroups = (), workgroupsize = (),
                                     ndrange = (), max_work_group_size::Int = typemax(Int))
    KI.check_launch_args(numworkgroups, workgroupsize, ndrange)

    # "An ndrange with a zero-sized dimension … is not an error: the call must be
    # a no-op and return `nothing` instead of launching." Same for an explicit
    # zero workgroup count, which `cld` below would otherwise turn into one.
    (length(ndrange) > 0 && any(==(0), ki_extent(ndrange, 1))) && return nothing
    (length(numworkgroups) > 0 && any(==(0), ki_extent(numworkgroups, 1))) && return nothing

    limit = min(max_work_group_size, KI.max_work_group_size(k.backend))

    if length(ndrange) > 0
        nd = ki_extent(ndrange, 1)
        wg = length(workgroupsize) > 0 ? ki_extent(workgroupsize, 1) :
             ki_extent(KI.threads_to_workgroupsize(limit, nd), 1)
        blocks = ntuple(i -> cld(nd[i], wg[i]), 3)
    else
        # KI's default is one workgroup of one workitem.
        wg     = ki_extent(workgroupsize, 1)
        blocks = ki_extent(numworkgroups, 1)
    end

    prod(wg) <= limit ||
        throw(ArgumentError("workgroupsize $wg exceeds the device limit of $limit workitems"))

    bq = k.backend.bq
    # `find_tlas_in_args` BEFORE Adapt, which strips the hwtlas — same ordering
    # the KA entry point depends on.
    tlas    = find_tlas_in_args(args)
    oneshot!(bq; tag = :launch) do e
        owner = e.owner
        pin_leaves!(owner, k.kern)
        pin_leaves!(owner, args)
        adaptor = LavaAdaptor(owner)
        # `ka_launch!` drops `all_args[1]` from the compiled signature
        # (`Base.tail`, ka_backend.jl) and `pack_args_direct!` skips it as a
        # ghost, because the KA path puts its `CompilerMetadata` there. A KI
        # kernel is a plain function with no such leading value, so it passes
        # `nothing` — zero-sized, hence dropped by both — to satisfy the same
        # contract.
        all_args = (nothing, map(a -> Adapt.adapt(adaptor, a), args)...)
        ka_launch!(e, k.kern, all_args, blocks, wg, tlas)
    end
    return nothing
end
