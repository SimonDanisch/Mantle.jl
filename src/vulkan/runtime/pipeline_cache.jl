# VkPipelineCache persistence.
#
# A single VkPipelineCache per device, seeded from disk and saved back so
# AMDVLK / RADV / etc. don't recompile SPIR-V → ISA every session. Lives in the
# same scratchspace as the SPIR-V cache (`lava_disk_cache_dir()`) for cohesion.
#
# Safety: the header is validated HERE, before the driver ever sees the bytes.
#
# The spec says an implementation must detect incompatible `pInitialData` and
# behave as if the cache were empty, and it is tempting to lean on that — the
# blob carries vendor/device IDs and a cache UUID precisely so it can be
# checked. Drivers are not reliably defensive about it in practice: a blob from
# another device, or one truncated by a half-written file, is a documented way
# to crash inside `vkCreatePipelineCache`, and a crash there takes the process
# with it before any Julia-level `try` can see it. `catch` does not help
# against a segfault.
#
# So the 32-byte `VkPipelineCacheHeaderVersionOne` is parsed and compared
# against this physical device before the data is passed on, and anything that
# does not match exactly is discarded. That is cheap, it is the one check that
# makes the "load whatever is on disk" path safe, and it is what lets the
# frozen kernel cache keep a driver-specific level at all.

"""
Disk path for this device's pipeline-cache blob. Co-located with the SPIR-V
disk cache; one file per (device, driver version) pair.
"""
function lava_pipeline_cache_path(device_name::AbstractString, driver_version::AbstractString)
    sanitized_dev = replace(device_name, r"[^a-zA-Z0-9]+" => "_")
    sanitized_drv = replace(driver_version, r"[^a-zA-Z0-9]+" => "_")
    return joinpath(lava_disk_cache_dir(),
                    "vk_pipeline_cache_$(sanitized_dev)_drv$(sanitized_drv).bin")
end

"""
    PIPELINE_CACHE_HEADER_BYTES

Size of `VkPipelineCacheHeaderVersionOne`: four `uint32` then a 16-byte UUID.
"""
const PIPELINE_CACHE_HEADER_BYTES = 32
const PIPELINE_CACHE_HEADER_VERSION_ONE = UInt32(1)

"""
    pipeline_cache_compatible(bytes, phys_device) -> Bool

Whether `bytes` is a pipeline-cache blob this device will accept.

Parses `VkPipelineCacheHeaderVersionOne` — `headerSize`, `headerVersion`,
`vendorID`, `deviceID`, `pipelineCacheUUID[16]`, all little-endian — and
requires every field to match the device. The UUID is the authoritative one: a
driver update changes it even when vendor and device do not, which is exactly
the case a filename keyed on the driver *version string* can miss (two builds
can report the same version).

Checked here rather than delegated to `vkCreatePipelineCache` because a
mismatch there is not reliably a clean rejection. See the note at the top of
this file.
"""
function pipeline_cache_compatible(bytes::Vector{UInt8}, phys_device)
    length(bytes) >= PIPELINE_CACHE_HEADER_BYTES || return false
    u32(off) = UInt32(bytes[off + 1]) | (UInt32(bytes[off + 2]) << 8) |
               (UInt32(bytes[off + 3]) << 16) | (UInt32(bytes[off + 4]) << 24)
    u32(0) == UInt32(PIPELINE_CACHE_HEADER_BYTES) || return false
    u32(4) == PIPELINE_CACHE_HEADER_VERSION_ONE || return false
    props = VK.get_physical_device_properties(phys_device)
    u32(8) == props.vendor_id || return false
    u32(12) == props.device_id || return false
    return all(i -> bytes[16 + i] == props.pipeline_cache_uuid[i], 1:16)
end

"""
Read the saved pipeline-cache blob from disk, or empty bytes when there is none,
it is unreadable, or its header does not match `phys_device`.
"""
function load_pipeline_cache_data(path::String, phys_device)
    isfile(path) || return UInt8[]
    bytes = try
        read(path)
    catch ex
        ex isa Union{SystemError, Base.IOError, EOFError} || rethrow()
        @warn "Lava: pipeline cache read failed; the driver will recompile from SPIR-V" path exception=ex
        return UInt8[]
    end
    if !pipeline_cache_compatible(bytes, phys_device)
        @debug "Lava: pipeline cache header does not match this device; ignoring" path
        return UInt8[]
    end
    return bytes
end

"""
Create a `VkPipelineCache` for `device`, seeded from `path` if it exists.
VK.jl's `PipelineCacheCreateInfo(initial_data::Ptr{Cvoid}; initial_data_size)`
takes a raw pointer — the bytes vector is GC-pinned for the duration of the call.

The header is checked against `phys_device` first, so the driver only ever sees
data it declared compatible. Creation is still wrapped: a blob can pass the
header check and be damaged past it, and that case should cost a recompile
rather than the session.
"""
function create_lava_pipeline_cache(device::VK.Device, path::String, phys_device)
    initial = load_pipeline_cache_data(path, phys_device)
    function _create_empty()
        ci = VK.PipelineCacheCreateInfo(Ptr{Cvoid}(C_NULL); initial_data_size=UInt64(0))
        return VK.PipelineCache(device, ci)
    end
    isempty(initial) && return _create_empty()
    try
        return GC.@preserve initial begin
            ptr = Ptr{Cvoid}(pointer(initial))
            ci = VK.PipelineCacheCreateInfo(ptr; initial_data_size=UInt64(length(initial)))
            VK.PipelineCache(device, ci)
        end
    catch ex
        @warn "Lava: pipeline cache create from disk failed; starting empty" path exception=ex
        # `force=true` already tolerates a missing file, so the only failures
        # left are real (permissions, a directory in the way) and worth saying.
        try
            rm(path; force=true)
        catch rmex
            rmex isa Union{SystemError, Base.IOError} || rethrow()
            @warn "Lava: could not delete the unusable pipeline cache; it will fail again next session" path exception=rmex
        end
        return _create_empty()
    end
end

"""
Snapshot `ctx.pipeline_cache` to its on-disk path. Atomic write via mktemp+rename
so concurrent processes can't observe a partial blob. No-op if the device was
lost (handles are invalid).
"""
function save_pipeline_cache!(ctx)
    ctx.device_lost && return nothing
    isdefined(ctx, :pipeline_cache) || return nothing
    try
        # VK.jl returns (size::UInt, ptr::Ptr{Cvoid}) where ptr was Libc.malloc'd
        # on our behalf; we must `Libc.free` after copying out.
        size, ptr = unwrap(
            VK.get_pipeline_cache_data(ctx.device, ctx.pipeline_cache))
        try
            size == 0 && return nothing
            bytes = unsafe_wrap(Array, Ptr{UInt8}(ptr), Int(size); own=false)
            path = lava_pipeline_cache_path(ctx.device_name, ctx.driver_version)
            mkpath(dirname(path))
            tmppath, io = mktemp(dirname(path); cleanup=false)
            try
                write(io, bytes)
            finally
                close(io)
            end
            mv(tmppath, path; force=true)
        finally
            Libc.free(ptr)
        end
    catch ex
        ex isa Union{SystemError, Base.IOError} || rethrow()
        @warn "Lava: pipeline cache save failed; next session recompiles ISA from SPIR-V" exception=ex
    end
    return nothing
end

# atexit hook: persist on Julia shutdown. Single-shot (registered once).
# Fully guarded — at shutdown the Vulkan device may already be torn down,
# the loader DLL may be unloaded, the process may be in an unrecoverable
# state. We never want a crash here to abort the host process or break
# parent supervisors (MCP servers, IDE harnesses, etc.).
const _PIPELINE_CACHE_ATEXIT_REGISTERED = Ref(false)
function _register_pipeline_cache_atexit!()
    _PIPELINE_CACHE_ATEXIT_REGISTERED[] && return
    _PIPELINE_CACHE_ATEXIT_REGISTERED[] = true
    atexit() do
        try
            ctx = VK_CONTEXT_REF[]
            ctx === nothing && return
            save_pipeline_cache!(ctx)
        catch ex
            # Never propagate from atexit — Julia is shutting down and a throw
            # here replaces the exit code. But SAY so: a pipeline cache that has
            # silently failed to save every session looks exactly like one that
            # is working.
            safe_fin_log("Lava: pipeline cache save at exit failed: " *
                         sprint(showerror, ex) * "\n")
        end
    end
end
