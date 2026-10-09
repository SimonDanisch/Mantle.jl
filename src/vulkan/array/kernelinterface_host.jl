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
# a `KA.Backend`, so nothing dispatched and a Float64 literal reaching
# `DNNKernels.kernelnumber` threw `MethodError` rather than being kept. The
# Metal backend states the other side of the same trait in
# `src/metal/kernelinterface.jl`.
KI.supports_float64(::LavaBackend) = true

# Which element types `shfl_down` and `shfl` cover. They dispatch on a backend — a
# host handle — so they are here, while the methods themselves are generated on
# the device side. `KI_SHFL_TYPES` is the one list both read, so KI's own suite
# cannot end up exercising a type Lava never generated.
KI.supports_subgroups(::LavaBackend) = true
KI.supports_shuffle(::LavaBackend, ::Type{T}) where {T} = T in KI_SHFL_TYPES
KI.shfl_types(::LavaBackend) = collect(KI_SHFL_TYPES)

# …and the same for the reduce-add, off its own list for the reason given beside
# `KI_REDUCE_ADD_TYPES`: the two families are separate capabilities and only
# happen to cover the same six types on this backend.
KI.sub_group_reduce_add_types(::LavaBackend) = collect(KI_REDUCE_ADD_TYPES)

# The cooperative-matrix extensions and tensor addressing, all four from
# `VK_NV_cooperative_matrix2` as the device reported it at creation.
KI.supports_tensor_addressing(b::LavaBackend) = vk_context(b).coopmat2.tensor_addressing
KI.supports_coopmat_perelement(b::LavaBackend) = vk_context(b).coopmat2.per_element_operations
KI.supports_coopmat_reduce(b::LavaBackend) = vk_context(b).coopmat2.reductions
KI.supports_flexible_coopmat_shapes(b::LavaBackend) = vk_context(b).coopmat2.flexible_dimensions

"""
The width the shader will actually run at, from the device rather than a guess:
32 on NVIDIA, 32 *or* 64 on RDNA3 depending on how the driver compiled the
shader, 8 on lavapipe. Nothing may hard-code it.
"""
KI.sub_group_size(backend::LavaBackend) = caps(backend).subgroup

KI.max_work_group_size(backend::LavaBackend) = caps(backend).workgrouplimit

KI.max_work_group_dims(backend::LavaBackend) = max_workgroup_dims(vk_context(backend))

KI.max_num_groups(backend::LavaBackend) = vk_context(backend).max_wg_dims

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
KI.max_work_group_size(k::KI.Kernel{LavaBackend}) = KI.max_work_group_size(k.backend)

# ── Launch ──────────────────────────────────────────────────────────────────
#
# Calling a `KI.Kernel` validates the launch keywords and resolves the geometry
# (`KI.launch_geometry`), then calls `KI.launch` with three-dimensional workgroup
# counts and sizes. The kernel is launched with the ORIGINAL args: `argconvert`
# exists to derive the compiled signature, and the adaptation that pins device
# memory happens here. That is Lava's split already, which is why this is an
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
# launch needs.
#
# `name` is accepted and ignored: Lava's kernels are named after the function
# they were compiled from, and there is nowhere to put an override that the
# SPIR-V module or the pipeline cache would read back.
KI.kernel_function(backend::LavaBackend, @nospecialize(f), @nospecialize(tt) = Tuple{};
                   name = nothing, kwargs...) =
    KI.Kernel(backend, f)

function KI.launch(k::KI.Kernel{LavaBackend}, groups::Dims{3}, items::Dims{3}, args::Tuple;
                   kwargs...)
    isempty(kwargs) ||
        throw(ArgumentError("Lava kernels take no launch keywords besides KernelInterface's, got $(keys(kwargs))"))
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
        ka_launch!(e, k.kern, all_args, groups, items, tlas)
    end
    return nothing
end
