# A device's life, asked of the device: what it was built to check and what that
# reported, how much work it has handed to the driver, and how it ends.
#
# Every verb here is declared in core and answered by each backend, because both
# backends have each concept: a validation layer and its messages, a count of
# submissions, a device that is retired and one that replaces it. The decisions
# are core's — a token nothing submitted is an error, a refused allocation is
# `nothing` — and a backend answers the driver half.

"""
    DebugConfig(; validation, gpu_av, gpu_av_safe, gpu_av_shaders,
                  sync_val, best_practices, printf, pool_disabled)

Every validation and instrumentation setting, chosen **at device construction**.

    dev = reset_device!(Device(backend); debug = DebugConfig(validation = true))

There is no way to switch any of this on for a device that already exists. On
Vulkan every setting is a property of the `VkInstance`, fixed by
`vkCreateInstance`. On Metal the API validation layer is loaded when the process
starts, so a process that is to validate has to be started with
[`debugenv`](@ref)'s environment; asking for validation in one that was not is
an error rather than a device that silently does not check.

What each setting is on each API:

  * `validation`: the API's validation layer. Vulkan: `VK_LAYER_KHRONOS_validation`
    and a debug messenger. Metal: the API validation layer (`MTLDebugDevice`).
    Either way, what it reports is drained with [`validationmessages!`](@ref), and
    [`validating`](@ref) says whether it is running.
  * `gpu_av`: instrumented shaders that report out-of-bounds accesses. Vulkan:
    GPU-assisted validation. Metal: shader validation, also chosen at process
    start.
  * `gpu_av_safe`, `gpu_av_shaders`: GPU-AV's Safe Mode and the kernels it
    instruments (empty: all of them). Vulkan only; both need
    `VK_EXT_layer_settings`.
  * `sync_val`, `best_practices`: the Vulkan layer's synchronization validation
    and best-practices checks. Metal has neither.
  * `printf`: `NonSemantic.DebugPrintf` output from `@lava_printf`. Vulkan only.
  * `pool_disabled`: every Vulkan device array gets its own `VkBuffer`, so GPU-AV,
    which tracks bounds per `VkBuffer`, sees an overrun inside a pool block.

A backend refuses a setting it has no counterpart for, at construction, rather
than building a device that does not do what was asked.

After `gpu_av` on Vulkan, call `verify_gpu_av`: "GPU-AV is enabled" and "GPU-AV
is catching errors" are not the same thing on every driver, and a clean run
under an instrument that never fired reads exactly like a clean run.

## The two rules, enforced in the constructor rather than warned about

`validation` is implied by everything else. GPU-AV, sync validation, best
practices and debug printf are all features **of** the validation layer, so
asking for one turns the layer on. Passing `validation = true` alone means core
API checks with no shader instrumentation, which is the cheap mode.

`gpu_av` and `printf` are mutually exclusive and **throw** together: the Vulkan
layer instruments shaders for each and cannot do both. Preferring one and
warning gives a clean run out of a disabled instrument.

`gpu_av_safe` defaults to **true**, unlike everything else here. Khronos' own
documentation: "Safe Mode will have GPU-AV try and prevent crashes, but will be
much slower to validate". Lava shipped GPU-AV without it for a long time, which
is most of why reaching for it produced a SIGSEGV rather than a report.
"""
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
        # Implied, not required: every feature below is a feature OF the layer,
        # so asking for one without it is a half-configuration that reports a
        # clean run from a disabled instrument.
        validation |= gpu_av || sync_val || best_practices || printf
        new(validation, gpu_av, gpu_av_safe, collect(String, gpu_av_shaders),
            sync_val, best_practices, printf, pool_disabled)
    end
end

"""
    DebugConfig(c::DebugConfig; kw...) -> DebugConfig

`c` with named settings replaced. The two rules above are re-checked, so a copy
cannot reach a state the constructor refuses.
"""
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

"""
    debugenv(backend, cfg::DebugConfig) -> Vector{Pair{String,String}}

The environment a NEW process has to be started with so that a device of
`backend`'s API can be built with `cfg` in it.

Empty where the API configures validation when the device is created, which is
Vulkan's case. Metal's validation layers are loaded when the process starts, so
there this names the variables that load them:

    cmd = addenv(`\$(Base.julia_cmd()) script.jl`, Mantle.debugenv(backend, cfg)...)

`backend` is a device or the KernelAbstractions backend under one.
"""
debugenv(x, ::DebugConfig) = Pair{String,String}[]

"""
    validating(device) -> Bool

Whether the API's validation layer is checking this device's calls.

What was ACHIEVED, which a [`DebugConfig`](@ref) only requests: on Vulkan the
layer may not be installed, and the device is built anyway with a warning. A
check that "this run produced no validation messages" means something only on a
device that answers `true` here. `false` for a backend with no validation layer.
"""
validating(dev) = false

"""
    validationmessages!(device) -> Vector{String}

Every message the validation layer reported for `device` since the last call,
oldest first, and forgets them.

What "this run produced no validation messages" is asked of, on every backend:

    validationmessages!(dev)                # start clean
    run!(plan); waitidle(dev)
    @test isempty(validationmessages!(dev))

Only what the layer flagged as an error or a warning; its own setup notices are
not messages. A failure the caller tolerated — an allocation refused through
[`trydevicearray`](@ref) — leaves nothing here: the refusal owns what the layer
said about it. Empty for a backend with no validation layer, and on a device
that is not [`validating`](@ref).
"""
validationmessages!(dev) = String[]

"""
    submissions(device) -> Int

How many submissions `device` has handed to its driver since it was built: each
one a `vkQueueSubmit2` on Vulkan, a committed command buffer on Metal. Counted
over every queue of the device, monotone, and never reset.

For asserting how much reaches the driver, not for timing: a run of a recorded
plan is one submission, and waiting for it is none.

    n = submissions(dev)
    run!(plan); waitfor!(plan)
    @test submissions(dev) - n == 1

What a call submits differs where the APIs do. A Vulkan launch submits when it is
called; a Metal launch joins an open command buffer that the next wait commits,
so "a launch and the wait for it" is the unit both answer the same. An upload to
memory the host addresses directly (Metal's shared buffers) is a copy and
submits nothing.
"""
function submissions end

"""
    retire!(device)

Promise that nothing calls into `device`'s driver again.

Every array, buffer and pool block of `device` that is released afterwards — by
a finalizer, by `free!`, by a trim — skips the driver call that would release
it, and a new allocation on it is refused. What Julia's exit does for every
device a process built, before its last finalizer pass, and what
[`reset_device!`](@ref) does to the device it replaces.

Not a teardown. The driver's objects are not destroyed here; whatever still
holds them keeps them until it is collected, and its finalizer then knows not to
ask a driver that may already be gone. That is the property the shutdown
needs: at exit the driver can be torn down before the arrays that name it are
finalized, in no order Julia defines.

Idempotent. [`retired`](@ref) is the question.
"""
retire!(::Device)

"""
    retired(device) -> Bool

Whether [`retire!`](@ref) has been called on `device`, by the caller, by
[`reset_device!`](@ref) or by the exit hook, or the driver has lost it. `false`
for a device that cannot be retired.
"""
retired(dev) = false

"""
    refuseretired(device)

The refusal every allocation on a retired device gets, from core's allocation
paths ([`acquire!`](@ref), [`trydevicearray`](@ref)) rather than from each
backend's: its memory would belong to a device nothing calls into.
"""
refuseretired(dev) = retired(dev) && throw(ArgumentError(
    "Mantle: this device is retired (`retire!`, `reset_device!`, or the process is " *
    "exiting), so nothing is allocated on it. Allocate on the device `reset_device!` " *
    "returned."))

"""
    reset_device!(device; debug = <device's config>) -> Device

Retire `device` and build a new one on the same hardware, with the debugging
configuration `debug`. When `device` was its API's default (`Device(api)`), the
new one is the default from now on.

Two reasons to call it.

1. **Recovery**, after the driver lost the device: `reset_device!(dev)`.
   `debug` carries across, so a reset in the middle of a session does not
   silently turn the instruments off.

2. **Switching validation on or off**, which can only be decided when a device
   is built: `reset_device!(dev; debug = DebugConfig(validation = true))`.

Everything allocated on `device` stays allocated on it and becomes unusable:
its memory belongs to a device nothing calls into any more. Allocate again on
the device this returns.
"""
function reset_device! end

"""
    trydevicearray(x, T, dims...) -> array or nothing

An uninitialised device array, as [`devicearray`](@ref)`(x, T, dims...)` makes,
or `nothing` when the device cannot hold one that large.

For a caller that expects a refusal and has an answer to it — a smaller batch, a
host fallback — rather than an error. A refusal is an outcome here and not a
fault: nothing is thrown, nothing is logged, and nothing is left for whoever
calls the device next: no validation message (see
[`validationmessages!`](@ref)) and no half-built allocation.

Refused without asking the driver when the array is larger than
[`maxalloc`](@ref); otherwise the device is asked, and its out-of-memory answer
is the `nothing`. Any other failure still throws.

`x` is a device or the KernelAbstractions backend under one.
"""
function trydevicearray(x, ::Type{T}, dims::Dims) where {T}
    dev = todevice(x)
    refuseretired(dev)
    bytes, overflow = Base.mul_with_overflow(prod(dims), sizeof(T))
    (overflow || bytes > maxalloc(dev)) && return nothing
    return tryallocate(dev, T, dims)
end
trydevicearray(x, ::Type{T}, dims::Integer...) where {T} = trydevicearray(x, T, Dims(dims))

"""
    tryallocate(device, T, dims) -> array or nothing

The backend half of [`trydevicearray`](@ref): the device array, or `nothing` if
the driver refused it for lack of memory. Called only for a size within
[`maxalloc`](@ref). A refusal leaves no validation message behind and no
reclaim or retry is attempted: the caller asked to be told, not rescued.
"""
function tryallocate end
