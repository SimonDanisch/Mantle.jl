# A `MetalDevice` is a KernelInterface backend, as every `Device` is. Metal.jl
# implements the vocabulary for its `MetalBackend`; the device answers each verb
# as `metalbackend(d)`, Metal.jl's backend bound to the device's batch, so there
# is one implementation and a launch goes to the device's queue. Metal.jl defines
# the device intrinsics itself; redefining them here is a method overwrite, which
# fails precompilation.

KI.allocate(d::MetalDevice, ::Type{T}, dims::Tuple; unified::Union{Nothing,Bool} = nothing) where {T} =
    KI.allocate(metalbackend(d), T, dims; unified = something(unified, false))
# As `KA.synchronize(::LavaDevice)`: the device's default channel.
KI.synchronize(d::MetalDevice) = (flush!(batchqueue(d)); nothing)
KI.copyto!(d::MetalDevice, dest, src) = KI.copyto!(metalbackend(d), dest, src)
KI.versioninfo(io::IO, d::MetalDevice) = KI.versioninfo(io, metalbackend(d))
KI.supports_shuffle(d::MetalDevice, ::Type{T}) where {T} = KI.supports_shuffle(metalbackend(d), T)
KI.argconvert(d::MetalDevice, x) = KI.argconvert(metalbackend(d), x)
KI.kernel_function(d::MetalDevice, f::F, tt::TT = Tuple{}; kwargs...) where {F,TT} =
    KI.kernel_function(metalbackend(d), f, tt; kwargs...)
KI.record_event(d::MetalDevice) = KI.record_event(metalbackend(d))
KI.wait_event(d::MetalDevice, ev::Tuple{MTL.MTLSharedEvent,UInt64}) =
    KI.wait_event(metalbackend(d), ev)
for f in (:functional, :supports_float64, :supports_atomics, :supports_unified,
          :supports_subgroups, :supports_linear_subgroups, :supports_independent_subgroups,
          :supports_partitionable_subgroups, :shfl_types, :sub_group_reduce_add_types,
          :max_work_group_size, :max_work_group_dims, :max_num_groups, :sub_group_size,
          :multiprocessor_count)
    @eval KI.$f(d::MetalDevice) = KI.$f(metalbackend(d))
end

# `Adapt.adapt(dev, x)` uploads to the device.
Adapt.adapt_storage(d::MetalDevice, x) = Adapt.adapt_storage(metalbackend(d), x)

# AcceleratedKernels tunes per backend, and its Metal extension tunes Metal.jl's.
for f in (:scan_tuning, :reduce_tuning, :predicate_tuning, :findall_tuning, :sort_tuning)
    @eval AcceleratedKernels.$f(d::MetalDevice, ::Type{T}) where {T} =
        AcceleratedKernels.$f(metalbackend(d), T)
end

# `KI.Kernel(backend, f)` around a plain function, on the device or on Metal.jl's
# backend (`KA.get_backend` of an `MtlArray`): its argument types are known only
# at launch, so it compiles for the launch arguments and answers its workgroup
# limit from the device until then. Per backend type, not over their `Union`: Metal.jl
# answers `Kernel{MetalBackend}` for its compiled kernels, and only these methods, on a
# plain function, are more specific.
for K in (:(KI.Kernel{Metal.MetalBackend, <:Function}), :(KI.Kernel{<:MetalDevice, <:Function}))
    @eval begin
        KI.max_work_group_size(k::$K) = KI.max_work_group_size(k.backend)

        function KI.launch(k::$K, groups::Dims{3}, items::Dims{3}, args::Tuple; kwargs...)
            # A top-level `Buffer` or `GPURef` resolves to the array over its region, as in
            # `dispatch!`; Metal.jl's compiler rejects it whole as a non-bitstype argument.
            args = argmap(storage, args)
            tt = Tuple{map(a -> Core.Typeof(KI.argconvert(k.backend, a)), args)...}
            KI.launch(KI.kernel_function(k.backend, k.kern, tt), groups, items, args; kwargs...)
        end
    end
end

# Mantle's entry points meet Metal.jl's backend as often as the device, since
# `KA.get_backend` of an `MtlArray` returns it, and answer it as its device.
caps(b::Metal.MetalBackend) = caps(Device(b))
waitidle(b::Metal.MetalBackend) = waitidle(Device(b))
Mantle.devicearray(b::Metal.MetalBackend, data::AbstractArray) = Mantle.devicearray(Device(b), data)
for f in (:supports_batch_queue, :supports_graphics, :supports_geometry_stage,
          :supports_tessellation, :supports_mesh_pipeline, :supports_hwtlas,
          :supports_procedural_traversal)
    @eval Mantle.$f(b::Metal.MetalBackend) = Mantle.$f(Device(b))
end
