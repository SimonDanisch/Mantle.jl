# Portable validation, submission accounting, retirement and allocation refusal.

"""DebugConfig(; validation=false, gpu_av=false, gpu_av_safe=true,
                  gpu_av_shaders=String[], sync_val=false, best_practices=false,
                  printf=false, pool_disabled=false)

Device-construction validation settings. Instrumentation implies validation;
gpu_av and printf together throw. GPU-AV safe mode defaults on; verify_gpu_av
checks that instrumentation actually reports faults. Vulkan-only settings include
GPU-AV shader selection, sync/best-practices checks, printf and pool disabling.
Metal API/shader validation requires debugenv in the process environment at startup.
Unsupported settings are refused. Query validating to check what was enabled."""
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

"""reset_device!(device; debug=<current config>) -> Device

Retire the old device and construct a replacement on the same hardware.
Replace its API default if needed. Old allocations become unusable; allocate
again on the returned device. Debug settings carry over unless overridden."""
function reset_device! end

"""trydevicearray(x, T, dims...) -> array or nothing

Allocate without reclaim/retry. Return nothing for size/overflow or native OOM
refusal, leaving no partial allocation or validation message. Retired devices
and unrelated failures throw. x is a Device or KernelAbstractions backend."""
function trydevicearray(x, ::Type{T}, dims::Dims) where {T}
    dev = todevice(x)
    refuseretired(dev)
    bytes, overflow = Base.mul_with_overflow(prod(dims), sizeof(T))
    (overflow || bytes > maxalloc(dev)) && return nothing
    return tryallocate(dev, T, dims)
end
trydevicearray(x, ::Type{T}, dims::Integer...) where {T} = trydevicearray(x, T, Dims(dims))

"""tryallocate(device, T, dims) -> array or nothing

Native half of trydevicearray, called within maxalloc. An OOM refusal returns
nothing without retry or residual diagnostics; unrelated failures propagate."""
function tryallocate end
