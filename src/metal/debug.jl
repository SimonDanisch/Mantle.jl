# What `DebugConfig` is on Metal, and how the validation layer's reports reach
# `validationmessages!`.
#
# Metal decides validation per PROCESS. `MTL_DEBUG_LAYER=1` in the environment a
# process starts with wraps every device in `MTLDebugDevice` (measured on macOS 27,
# an M5: the device, its queues and their command buffers all report `MTLDebug…`
# classes, and the framework logs "Metal API Validation Enabled" before Julia has
# run a line); set later, it does nothing. `MTL_SHADER_VALIDATION=1` is the same
# for instrumented shaders. So a device cannot be built "with validation" in a
# process that did not start with it, and `checkdebug` refuses rather than build
# one that silently does not check.
#
# A failure the layer finds goes to `MTLReportFailure`, which by default asserts
# and aborts the process. `MTLSetReportFailureBlock` replaces that with a block,
# and the block here records the message instead: the process lives, and the
# message is what `validationmessages!` drains. Measured: an `offset(128) must be
# < [buffer length](64)` on `setBuffer:offset:atIndex:` reaches the block on the
# thread that made the call, with the formatted message, and the process goes on.
# `MTLSetReportFailureBlock` is exported by Metal.framework and is not in its
# public headers; it is the one way to receive the layer's messages in-process.

"""
    debugenv(::MetalDevice / ::MetalBackend, cfg) -> environment pairs

`MTL_DEBUG_LAYER=1` for `validation`, `MTL_SHADER_VALIDATION=1` for `gpu_av`.
"""
debugenv(::Union{MetalDevice,Metal.MetalBackend}, cfg::DebugConfig) =
    Pair{String,String}[(cfg.validation ? ["MTL_DEBUG_LAYER" => "1"] : Pair{String,String}[])...,
                        (cfg.gpu_av ? ["MTL_SHADER_VALIDATION" => "1"] : Pair{String,String}[])...]

"""The Objective-C class of `dev`: `MTLDebugDevice` when the process loaded the
API validation layer, the driver's own (`AGXG17GDevice`) when it did not."""
classname(dev::MTL.MTLDevice) =
    unsafe_string(ccall(:object_getClassName, Cstring, (id{MTL.MTLDevice},), dev))

validating(d::MetalDevice) = startswith(classname(d.dev), "MTLDebug")

"""
    checkdebug(mtldev, cfg)

Refuse a configuration this process cannot honour, before a device is built
with it: a setting Metal has no counterpart for, or a layer the process did not
load at start. Installs the report block when `validation` is asked for.
"""
function checkdebug(mtldev::MTL.MTLDevice, cfg::DebugConfig)
    unsupported = String[]
    cfg.sync_val && push!(unsupported, "sync_val")
    cfg.best_practices && push!(unsupported, "best_practices")
    cfg.printf && push!(unsupported, "printf")
    isempty(cfg.gpu_av_shaders) || push!(unsupported, "gpu_av_shaders")
    cfg.pool_disabled && push!(unsupported, "pool_disabled")
    isempty(unsupported) || throw(ArgumentError(
        "DebugConfig: Metal has no counterpart for $(join(unsupported, ", ")); these are " *
        "settings of the Vulkan validation layer and its pool."))
    if cfg.validation && !startswith(classname(mtldev), "MTLDebug")
        throw(ArgumentError(
            "DebugConfig(validation = true): Metal loads its API validation layer " *
            "when the process starts, and this one started without it. Start the " *
            "process with `addenv(cmd, Mantle.debugenv(backend, cfg)...)` " *
            "(MTL_DEBUG_LAYER=1)."))
    end
    if cfg.gpu_av && get(ENV, "MTL_SHADER_VALIDATION", "0") != "1"
        throw(ArgumentError(
            "DebugConfig(gpu_av = true): Metal's shader validation is chosen when the " *
            "process starts, and this one started without it. Start the process with " *
            "`addenv(cmd, Mantle.debugenv(backend, cfg)...)` (MTL_SHADER_VALIDATION=1)."))
    end
    cfg.validation && capturereports!()
    return nothing
end

"""
Install the block that records what the validation layer reports, once per
process. The block runs on whichever thread made the failing call, which is a
Julia thread for every call Mantle and Metal.jl make, and only appends.
"""
function capturereports!()
    METAL.reporter === nothing || return nothing
    report = function (type::UInt32, func::Cstring, line::Cint, msg::id{NSString})
        text = string(func == C_NULL ? "" : unsafe_string(func), ": ",
                      String(NSString(msg)))
        lock(METAL.reportlock) do
            push!(METAL.reports, text)
        end
        return nothing
    end
    blk = @objcblock(report, Nothing, (UInt32, Cstring, Cint, id{NSString}))
    setblock = Libdl.dlsym(Libdl.dlopen("/System/Library/Frameworks/Metal.framework/Metal"),
                           :MTLSetReportFailureBlock)
    ccall(setblock, Cvoid, (id{NSBlock},), blk)
    METAL.reporter = blk
    return nothing
end

"""The reports the layer made since the last call — the process's, since Metal
reports to one block: on Metal, validation is a property of the process."""
function validationmessages!(::MetalDevice)
    lock(METAL.reportlock) do
        msgs = copy(METAL.reports)
        empty!(METAL.reports)
        msgs
    end
end
