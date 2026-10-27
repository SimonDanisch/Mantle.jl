# Metal validation is per process: `MTL_DEBUG_LAYER=1` (wraps every device in
# `MTLDebugDevice`) and `MTL_SHADER_VALIDATION=1` act only in the environment the
# process starts with, so `checkdebug` refuses rather than build a device that
# silently does not check.

"""
    debugenv(::MetalDevice, cfg) -> environment pairs

`MTL_DEBUG_LAYER=1` for `validation`, `MTL_SHADER_VALIDATION=1` for `gpu_av`.
"""
debugenv(::MetalDevice, cfg::DebugConfig) =
    Pair{String,String}[(cfg.validation ? ["MTL_DEBUG_LAYER" => "1"] : Pair{String,String}[])...,
                        (cfg.gpu_av ? ["MTL_SHADER_VALIDATION" => "1"] : Pair{String,String}[])...]

"""The Objective-C class of `dev`: `MTLDebugDevice` under the API validation
layer, otherwise the driver's own (`AGXG17GDevice`)."""
classname(dev::MTL.MTLDevice) =
    unsafe_string(ccall(:object_getClassName, Cstring, (id{MTL.MTLDevice},), dev))

validating(d::MetalDevice) = startswith(classname(d.dev), "MTLDebug")

"""
    checkdebug(mtldev, cfg)

Throw `ArgumentError` for a setting Metal has no counterpart for, or a layer the
process did not load at start. Installs the report block when `validation` is set.
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
    capturereports!()

Once per process, replace `MTLReportFailure`'s default abort with a block that
appends the message to `METAL.reports`. `MTLSetReportFailureBlock` is exported by
Metal.framework but not in its public headers. The block runs on the thread that
made the failing call (a Julia thread for Mantle's and Metal.jl's calls).
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

"""Drain the validation reports since the last call. They are the process's, not
this device's: Metal reports to one block."""
function validationmessages!(::MetalDevice)
    lock(METAL.reportlock) do
        msgs = copy(METAL.reports)
        empty!(METAL.reports)
        msgs
    end
end
