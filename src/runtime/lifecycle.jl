# Portable validation, submission accounting, retirement and allocation refusal.

"""DebugConfig(; validation=false, gpu_av=false, gpu_av_safe=true,
                  gpu_av_shaders=String[], sync_val=false, best_practices=false,
                  printf=false, pool_disabled=false)

Device-construction validation settings. Instrumentation implies validation;
gpu_av and printf together throw. verify_gpu_av checks that GPU-AV reports
faults. GPU-AV shader selection, sync/best-practices checks, printf and
pool_disabled are Vulkan-only; unsupported settings are refused. Metal
validation requires debugenv in the process environment at startup."""
struct DebugConfig
    validation::Bool
    gpu_av::Bool
    gpu_av_safe::Bool
    gpu_av_shaders::Vector{String}
    sync_val::Bool
    best_practices::Bool
    printf::Bool
    pool_disabled::Bool

    function DebugConfig(; validation::Bool = false,
                           gpu_av::Bool = false,
                           gpu_av_safe::Bool = true,
                           gpu_av_shaders::AbstractVector{<:AbstractString} = String[],
                           sync_val::Bool = false,
                           best_practices::Bool = false,
                           printf::Bool = false,
                           pool_disabled::Bool = false)
        if gpu_av && printf
            throw(ArgumentError("""
                DebugConfig: `gpu_av` and `printf` cannot both be on — the validation
                layer instruments shaders for each and does not do both at once.
                Pick one: `DebugConfig(gpu_av = true)` to hunt out-of-bounds accesses,
                or `DebugConfig(printf = true)` to read `@lava_printf` output."""))
        end
        # Instrumentation requires the validation layer.
        validation |= gpu_av || sync_val || best_practices || printf
        new(validation, gpu_av, gpu_av_safe, collect(String, gpu_av_shaders),
            sync_val, best_practices, printf, pool_disabled)
    end
end

"""Copy c with named settings replaced, rechecking constructor constraints."""
DebugConfig(c::DebugConfig;
            validation = c.validation, gpu_av = c.gpu_av,
            gpu_av_safe = c.gpu_av_safe, gpu_av_shaders = c.gpu_av_shaders,
            sync_val = c.sync_val, best_practices = c.best_practices,
            printf = c.printf, pool_disabled = c.pool_disabled) =
    DebugConfig(; validation, gpu_av, gpu_av_safe, gpu_av_shaders,
                  sync_val, best_practices, printf, pool_disabled)

Base.:(==)(a::DebugConfig, b::DebugConfig) =
    all(f -> getfield(a, f) == getfield(b, f), fieldnames(DebugConfig))
Base.hash(c::DebugConfig, h::UInt) =
    foldl((h, f) -> hash(getfield(c, f), h), fieldnames(DebugConfig); init = hash(DebugConfig, h))

"""debugenv(backend, cfg::DebugConfig) -> Vector{Pair{String,String}}

Environment for a new process using cfg. Metal validation requires startup
variables; APIs that configure it during device creation return an empty vector."""
debugenv(x, ::DebugConfig) = Pair{String,String}[]

"""validating(device) -> Bool

Whether API validation is active. A requested layer may be unavailable, so an
empty validationmessages! result alone does not establish a validated run."""
validating(dev) = false

"""validationmessages!(device) -> Vector{String}

Drain validation warnings/errors in arrival order. Setup notices are excluded;
tolerated trydevicearray refusals consume their own diagnostics. Empty without
active validation. Wait for relevant GPU work before checking a complete run."""
validationmessages!(dev) = String[]

"""submissions(device) -> Int

Monotone count of native submissions over every device queue. Vulkan counts
queue submits; Metal counts committed command buffers. Direct host writes submit
nothing. For deferred launch APIs, include the wait/commit when comparing counts."""
function submissions end

"""retire!(device)

Idempotently prohibit future driver calls and allocations on this device.
Later resource releases skip native destruction. This marks shutdown/reset state;
it does not tear down every live native object. Query retired for that state."""
retire!(::Device)

"""retired(device) -> Bool

Whether the device was retired explicitly, by reset/exit, or through device loss.
The default is false for backends without retirement support."""
retired(dev) = false

"""Reject allocation on a retired device from core allocation paths."""
refuseretired(dev) = retired(dev) && throw(ArgumentError(
    "Mantle: this device is retired (`retire!`, `reset_device!`, or the process is " *
    "exiting), so nothing is allocated on it. Allocate on the device `reset_device!` " *
    "returned."))

# ── The devices of a process ───────────────────────────────────────────────

"""
The devices of one API in this process: the default `Device(api)` hands out, made
on first use under `lock`, since two made at once would put two pools over one
GPU, and every device made, weakly, so [`retireall!`](@ref) retires each at exit
before the finalizers run into a torn-down driver.
"""
mutable struct DeviceSlot
    default::Union{Nothing,Device}
    live::Vector{WeakRef}
    lock::ReentrantLock
end

"""
Every API's [`DeviceSlot`](@ref), by API type: filled once by `__init__`
([`initdevices!`](@ref)) and only read after, so the default device is found
without a lock.
"""
const DEVICES = Dict{DataType,DeviceSlot}()

initdevices!(apis) = foreach(api -> DEVICES[api] = DeviceSlot(nothing, WeakRef[], ReentrantLock()), apis)

deviceslot(api::Backend) = DEVICES[typeof(api)]

"""
    makedevice(api, select, debug) -> Device

The backend's new device on the hardware `select` names ([`selectdevice`](@ref);
`nothing` for its own choice), with the validation settings `debug`. Core
registers it and decides whether it becomes the default.
"""
function makedevice end

"""
    remakedevice(device, debug) -> Device

A new device on the same hardware as `device`, which is retired next, with the
validation settings `debug`.
"""
function remakedevice end

"""
    adopt!(device) -> device

Make `device` the one launches outside Mantle go to: Metal.jl's queue, AMDGPU.jl's
stream. Called when it becomes the default, so an upload before its first run is
ordered against the runs.
"""
adopt!(d::Device) = d

"""A new device, registered for retirement at exit."""
function newdevice(api::Backend, select, debug::DebugConfig)
    d = makedevice(api, select, debug)::Device
    slot = deviceslot(api)
    @lock slot.lock push!(slot.live, WeakRef(d))
    return d
end

function Device(api::Backend; select = nothing, debug::Union{Nothing,DebugConfig} = nothing)
    (select === nothing && debug === nothing) || return newdevice(api, select, something(debug, DebugConfig()))
    slot = deviceslot(api)
    d = slot.default
    d === nothing || return d
    return @lock slot.lock begin
        slot.default === nothing && defaultdevice!(newdevice(api, nothing, DebugConfig()))
        slot.default
    end
end

"""The default device of `api` if one was made, else `nothing`; makes none."""
existingdefault(api::Backend) = deviceslot(api).default

function defaultdevice!(d::Device)
    slot = deviceslot(syncbackend(d))
    @lock slot.lock slot.default = d
    return adopt!(d)
end

"""Every device of `api` this process made and still references."""
livedevices(api::Backend) =
    (slot = deviceslot(api); @lock slot.lock Device[r.value for r in slot.live if r.value isa Device])

"""
    retireall!(api)
    retireall!()

Retire every device of `api`, or of every API, and drop the defaults. The exit
hook: Julia runs `atexit` before the final finalizer sweep, and a finalizer
reaching a driver that is already torn down crashes.
"""
function retireall!(api::Backend)
    slot = deviceslot(api)
    @lock slot.lock begin
        foreach(r -> r.value isa Device && retire!(r.value), slot.live)
        slot.default = nothing
    end
    return nothing
end
retireall!() = foreach(api -> retireall!(api()), keys(DEVICES))

"""reset_device!(device; debug=<current config>) -> Device

Retire the old device and construct a replacement on the same hardware.
Replace its API default if needed. Old allocations become unusable; allocate
again on the returned device. Debug settings carry over unless overridden."""
function reset_device!(d::Device; debug::DebugConfig = d.debug)
    api = syncbackend(d)
    new = remakedevice(d, debug)::Device
    slot = deviceslot(api)
    @lock slot.lock push!(slot.live, WeakRef(new))
    retire!(d)
    slot.default === d && defaultdevice!(new)
    return new
end

"""trydevicearray(x, T, dims...) -> array or nothing

Allocate without reclaim/retry. Return nothing for size/overflow or native OOM
refusal, leaving no partial allocation or validation message. Retired devices
and unrelated failures throw. x is a Device or KernelAbstractions backend."""
function trydevicearray(x, ::Type{T}, dims::Dims) where {T}
    dev = todevice(x)
    refuseretired(dev)
    bytes = try
        checkedbytes(T,dims)
    catch err
        err isa OverflowError || rethrow()
        return nothing
    end
    bytes > maxalloc(dev) && return nothing
    return tryallocate(dev, T, dims)
end
trydevicearray(x, ::Type{T}, dims::Integer...) where {T} = trydevicearray(x, T, Dims(dims))

"""tryallocate(device, T, dims) -> array or nothing

Native half of trydevicearray, called within maxalloc. An OOM refusal returns
nothing without retry or residual diagnostics; unrelated failures propagate."""
function tryallocate end
