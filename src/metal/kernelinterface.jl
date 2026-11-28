# A `MetalDevice` is a KernelInterface backend, as every `Device` is. Metal.jl
# implements the vocabulary for its `MetalBackend`; the device answers each verb
# as `metalbackend(d)`, so there is one implementation. Metal.jl's backend carries
# no queue, so the verbs that submit work pass the device's batch (`queue = ...`)
# or run under `Metal.global_queue!`. Metal.jl defines the device intrinsics
# itself; redefining them here is a method overwrite, which fails precompilation.

"""
    KI.allocate(dev::MetalDevice, T, dims; unified = false) -> MtlArray

A Metal.jl array in a buffer of the device's own, in its [`ledger`](@ref) until the
array is freed: in Metal.jl's default storage mode (its `default_storage` preference),
or shared with the host when `unified`.
"""
function KI.allocate(d::MetalDevice, ::Type{T}, dims::Tuple; unified::Union{Nothing,Bool} = nothing) where {T}
    # Any integer extents, as `KA.allocate` takes them (Hikari's NanoVDB sizes are `Int32`).
    dims = Dims(dims)
    bytes = checkedbytes(T, dims)
    # Metal asserts that a buffer fits `maxBufferLength`; past it, the device is out
    # of memory for this request, which callers catch as such.
    bytes > maxalloc(d) && throw(OutOfMemoryError())
    storage = something(unified, false) ? Metal.SharedStorage : Metal.DefaultStorageMode
    return ownedarray(d, newbuffer(d, bytes; storage), T, dims)
end
# As `KA.synchronize(::LavaDevice)`: the device's default channel.
KI.synchronize(d::MetalDevice) = (flush!(batchqueue(d)); nothing)
KI.copyto!(d::MetalDevice, dest, src) =
    Metal.global_queue!(() -> KI.copyto!(metalbackend(d), dest, src), batchqueue(d))
KI.versioninfo(io::IO, d::MetalDevice) = KI.versioninfo(io, metalbackend(d))
KI.supports_shuffle(d::MetalDevice, ::Type{T}) where {T} = KI.supports_shuffle(metalbackend(d), T)
"""
    KI.argconvert(dev::MetalDevice, x)

`x` as a kernel takes it, with every Metal.jl array in it made resident first. A
converted array is a device address, and what keeps one outside a launch (Raycore's
type slots and texture tables, a light list) reaches the buffer by that address
alone, which Metal maps only for a buffer in a residency set. Arrays Mantle fills
from host data are in the device's set already (`residentarray`); one Metal.jl
allocated (`resize!`, `similar`) is not, and read zeros.
"""
function KI.argconvert(d::MetalDevice, x)
    Mantle.holdleaves!(ResidentLeaves(residency(d)), x)
    return KI.argconvert(metalbackend(d), x)
end

"""
A holder that `holdleaves!` hands the Metal.jl arrays of an argument. Metal declares
buffers, not objects, so an array that only wraps others (a view, Raycore's set of
typed slots) is walked for them instead of held whole, and a host `Array`, which no
kernel takes, is skipped. A pool region is a leaf: its block is the pool's.
"""
abstract type MetalLeaves end
Mantle.holdleaves!(c::MetalLeaves, a::AbstractArray) = Mantle.holdfields!(c, a)
Mantle.holdleaves!(c::MetalLeaves, a::Metal.MtlArray) = (Mantle.hold!(c, a); nothing)
Mantle.holdleaves!(c::MetalLeaves, a::DeviceArray) = (Mantle.hold!(c, a); nothing)
Mantle.holdleaves!(::MetalLeaves, ::Array) = nothing

# Makes each array of a converted argument resident (`KI.argconvert`).
struct ResidentLeaves <: MetalLeaves
    residency::MetalBindings.Residency
end
function Mantle.hold!(c::ResidentLeaves, a::Metal.MtlArray)
    m = a.data[]
    MetalBindings.resident!(c.residency, m.buffer, m)
    return a
end
Mantle.hold!(::ResidentLeaves, a) = a
"""
    KI.kernel_function(dev::MetalDevice, f, tt) -> KI.Kernel

Metal.jl's kernel for `f`, as a kernel of `dev`, which a launch submits to `dev`'s
queue. Metal.jl's own kernel launches on the queue Metal.jl's task-local
`global_queue` holds for the calling task: from a task the device never ran on,
Metal.jl's queue, which none of the device's residency reaches. A kernel then read
zeros through a stored handle, and nothing ordered it against the device's work.
"""
KI.kernel_function(d::MetalDevice, f::F, tt::TT = Tuple{}; kwargs...) where {F,TT} =
    KI.Kernel(d, KI.kernel_function(metalbackend(d), f, tt; kwargs...).kern)
KI.max_work_group_size(k::KI.Kernel{<:MetalDevice, <:Metal.HostKernel}) = k.kern.maxthreads
KI.launch(k::KI.Kernel{<:MetalDevice, <:Metal.HostKernel}, groups::Dims{3}, items::Dims{3},
          args::Tuple; queue = batchqueue(k.backend), kwargs...) =
    KI.launch(KI.Kernel(metalbackend(k.backend), k.kern), groups, items, args; queue, kwargs...)
KI.record_event(d::MetalDevice) =
    Metal.global_queue!(() -> KI.record_event(metalbackend(d)), batchqueue(d))
KI.wait_event(d::MetalDevice, ev::Tuple{MTL.MTLSharedEvent,UInt64}) =
    Metal.global_queue!(() -> KI.wait_event(metalbackend(d), ev), batchqueue(d))
for f in (:functional, :supports_float64, :supports_atomics, :supports_unified,
          :supports_subgroups, :supports_linear_subgroups, :supports_independent_subgroups,
          :supports_partitionable_subgroups,
          :max_work_group_size, :max_work_group_dims, :max_num_groups, :sub_group_size,
          :multiprocessor_count)
    @eval KI.$f(d::MetalDevice) = KI.$f(metalbackend(d))
end

# `Adapt.adapt(dev, x)` uploads host arrays to the device, into an array `residentarray`
# made; device arrays and views of them convert as Metal.jl converts them.
Adapt.adapt_storage(d::MetalDevice, x) = Adapt.adapt_storage(metalbackend(d), x)
Adapt.adapt_storage(d::MetalDevice, x::Array) = buffercopy(d, x)

# AcceleratedKernels tunes per backend, and its Metal extension tunes Metal.jl's.
for f in (:scan_tuning, :reduce_tuning, :predicate_tuning, :findall_tuning, :sort_tuning)
    @eval AcceleratedKernels.$f(d::MetalDevice, ::Type{T}) where {T} =
        AcceleratedKernels.$f(metalbackend(d), T)
end

# `KI.Kernel(dev, f)` around a plain function: its argument types are known only at
# launch, so it compiles for the launch arguments and answers its workgroup limit
# from the device until then. It launches on the device's batch unless the call
# names a queue. Code holding Metal.jl's backend (`KA.get_backend` of an `MtlArray`)
# builds the kernel on `Device(backend)`.
KI.max_work_group_size(k::KI.Kernel{<:MetalDevice, <:Function}) = KI.max_work_group_size(k.backend)

function KI.launch(k::KI.Kernel{<:MetalDevice, <:Function}, groups::Dims{3}, items::Dims{3},
                   args::Tuple; queue = batchqueue(k.backend), kwargs...)
    # A top-level `Buffer` or `GPURef` resolves to the array over its region, as in
    # `dispatch!`; Metal.jl's compiler rejects it whole as a non-bitstype argument.
    args = argmap(storage, args)
    # Typed through Metal.jl's conversion: the launch declares its arguments itself.
    tt = Tuple{map(a -> Core.Typeof(KI.argconvert(metalbackend(k.backend), a)), args)...}
    KI.launch(KI.kernel_function(k.backend, k.kern, tt), groups, items, args; queue, kwargs...)
end

# Mantle's entry points meet Metal.jl's backend as often as the device, since
# `KA.get_backend` of an `MtlArray` returns it, and answer it as its device.
waitidle(b::Metal.MetalBackend) = waitidle(Device(b))
Mantle.devicearray(b::Metal.MetalBackend, data::AbstractArray) = Mantle.devicearray(Device(b), data)
for f in (:supports_batch_queue, :supports_graphics, :supports_geometry_stage,
          :supports_tessellation, :supports_mesh_pipeline, :supports_hwtlas,
          :supports_procedural_traversal)
    @eval Mantle.$f(b::Metal.MetalBackend) = Mantle.$f(Device(b))
end
