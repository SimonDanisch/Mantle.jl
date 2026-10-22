# Host side of `KernelInterface` (KI) for `LavaBackend`: the methods that need
# a device. Device-side intrinsics are in Lava's `device/kernelinterface.jl`.
# `caps(b::LavaBackend)` in `ka_backend.jl` is `KI.caps` (imported in
# `Mantle.jl`); KI derives `matrix_shapes`, `supports`, `bestshape` and
# `wggranularity` from it.

import KernelInterface as KI

# `synchronize`, `allocate`, `get_backend` and `supports_unified` are shared
# with KernelAbstractions 0.10 and implemented in `ka_backend.jl`.
KI.supports_atomics(::LavaBackend) = true
# KI's fallback is `false`; every Lava device enables `shaderFloat64`.
KI.supports_float64(::LavaBackend) = true

# Element types for `shfl` and `shfl_down`, read from `KI_SHFL_TYPES`, the list
# the device-side methods are generated from.
KI.supports_subgroups(::LavaBackend) = true
KI.supports_shuffle(::LavaBackend, ::Type{T}) where {T} = T in KI_SHFL_TYPES
KI.shfl_types(::LavaBackend) = collect(KI_SHFL_TYPES)

# Reduce-add has its own list: it is a separate capability from shuffles.
KI.sub_group_reduce_add_types(::LavaBackend) = collect(KI_REDUCE_ADD_TYPES)

# `VK_NV_cooperative_matrix2` features as reported at device creation.
KI.supports_tensor_addressing(b::LavaBackend) = vk_context(b).coopmat2.tensor_addressing
KI.supports_coopmat_perelement(b::LavaBackend) = vk_context(b).coopmat2.per_element_operations
KI.supports_coopmat_reduce(b::LavaBackend) = vk_context(b).coopmat2.reductions
KI.supports_flexible_coopmat_shapes(b::LavaBackend) = vk_context(b).coopmat2.flexible_dimensions

"""
    KI.sub_group_size(backend) -> Int

The device's reported subgroup width: 32 on NVIDIA, 32 or 64 on RDNA3, 8 on
lavapipe. Kernels must not hard-code it.
"""
KI.sub_group_size(backend::LavaBackend) = caps(backend).subgroup

KI.max_work_group_size(backend::LavaBackend) = caps(backend).workgrouplimit

KI.max_work_group_dims(backend::LavaBackend) = max_workgroup_dims(vk_context(backend))

KI.max_num_groups(backend::LavaBackend) = vk_context(backend).max_wg_dims

KI.multiprocessor_count(backend::LavaBackend) = caps(backend).cores

# The device limit; the per-kernel limit under register pressure can be lower
# but is not queried.
KI.max_work_group_size(k::KI.Kernel{LavaBackend}) = KI.max_work_group_size(k.backend)

"""
    KI.argconvert(::LavaBackend, a)

Device-side value for KI's type derivation; holds no resource. `KI.launch`
receives the original arguments and holds their resources itself.
"""
KI.argconvert(::LavaBackend, a::LavaArray{T,N}) where {T,N} =
    LavaDeviceArray{T,N}(Ptr{T}(bda_address(a)), a.dims)
KI.argconvert(::LavaBackend, x) = x

# `tt` is ignored: the signature is compiled from the actual arguments at
# launch. `name` is ignored: kernels are named after their function.
KI.kernel_function(backend::LavaBackend, @nospecialize(f), @nospecialize(tt) = Tuple{};
                   name = nothing, kwargs...) =
    KI.Kernel(backend, f)

function KI.launch(k::KI.Kernel{LavaBackend}, groups::Dims{3}, items::Dims{3}, args::Tuple;
                   kwargs...)
    isempty(kwargs) ||
        throw(ArgumentError("Lava kernels take no launch keywords besides KernelInterface's, got $(keys(kwargs))"))
    # Resolve top-level `Buffer`/`GPURef` arguments to their arrays as
    # `dispatch!` does; otherwise they fail compilation as non-bitstype.
    args = argmap(storage, args)

    bq = k.backend.dispatch_bq
    # Before Adapt, which strips the hwtlas.
    tlas    = find_tlas_in_args(args)
    oneshot!(bq; tag = :launch) do e
        owner = e.owner
        holdleaves!(owner, k.kern)
        holdleaves!(owner, args)
        adaptor = LavaAdaptor(owner)
        # `ka_launch!` drops `all_args[1]`, where the KA path passes the kernel
        # function; a KI kernel passes a zero-sized `nothing` there.
        all_args = (nothing, argmap(a -> Adapt.adapt(adaptor, a), args)...)
        ka_launch!(e, k.kern, all_args, groups, items, tlas)
    end
    return nothing
end
