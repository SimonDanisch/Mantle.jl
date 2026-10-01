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

# `synchronize`, `allocate`, `get_backend` and `supports_unified` are not
# restated here: on KernelAbstractions 0.10 they ARE KernelInterface's functions,
# and `array/ka_backend.jl` implements them once.
KI.supports_atomics(::LavaBackend) = true
# `KernelInterface`'s docstring says a backend implements this only if it does
# NOT support Float64, and Vulkan's `shaderFloat64` is core. It still has to be
# stated: the `= true` default is declared on `KI.Backend`, and `LavaBackend` is
# a `KA.GPU`, so nothing dispatched and a Float64 literal reaching
# `DNNKernels.kernelnumber` threw `MethodError` rather than being kept. The
# Metal backend states the other side of the same trait in
# `src/metal/kernelinterface.jl`.
KI.supports_float64(::LavaBackend) = true

# Which element types `shfl_down` covers. It dispatches on a backend — a host
# handle — so it is here, while the methods themselves are generated on the
# device side. `KI_SHFL_TYPES` is the one list both read, so KI's own suite
# cannot end up exercising a type Lava never generated.
KI.shfl_down_types(::LavaBackend) = collect(KI_SHFL_TYPES)
KI.shfl_types(::LavaBackend) = collect(KI_SHFL_TYPES)

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

function (k::KI.Kernel{LavaBackend})(args...;
                                     numworkgroups = (), workgroupsize = (),
                                     ndrange = (), max_work_group_size::Int = typemax(Int))
    KI.check_launch_args(numworkgroups, workgroupsize, ndrange)

    # "An ndrange with a zero-sized dimension … is not an error: the call must be
    # a no-op and return `nothing` instead of launching." Same for an explicit
    # zero workgroup count, which `cld` below would otherwise turn into one.
    (length(ndrange) > 0 && any(==(0), ki_extent(ndrange, 1))) && return nothing
    (length(numworkgroups) > 0 && any(==(0), ki_extent(numworkgroups, 1))) && return nothing

    wg, blocks = ki_launch_extents(k.backend, ndrange, workgroupsize, numworkgroups;
                                   max_work_group_size)

    # A `Buffer` or `GPURef` resolves to the array over its region, exactly as
    # `dispatch!` resolves one when it packs a declared pass. Without this a
    # resource is a legal argument to the declared path and an
    # `InvalidIRError` on this one — "passing non-bitstype argument … `.store`
    # is of type `DeviceArray`" — and the same operand would have to be spelled
    # two ways depending on how it is run. `storage` is the identity on
    # anything that is already an array, so nothing else changes. Top level
    # only: a kernel that genuinely takes a tuple keeps it.
    args = map(storage, args)

    bq = k.backend.dispatch_bq
    # `find_tlas_in_args` BEFORE Adapt, which strips the hwtlas — same ordering
    # the KA entry point depends on.
    tlas    = find_tlas_in_args(args)
    oneshot!(bq; tag = :launch) do e
        owner = e.owner
        holdleaves!(owner, k.kern)
        holdleaves!(owner, args)
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
