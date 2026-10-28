# One VkPipelineCache per device, seeded from disk and saved back so the driver
# does not recompile SPIR-V to ISA every session. The blob's header is checked
# against the device before the driver sees it: despite the spec, drivers can
# crash inside `vkCreatePipelineCache` on a blob from another device or a
# truncated file, and a segfault is not catchable.

"""
Disk path of the pipeline-cache blob for a (device, driver version) pair, in
the `lava_spirv_cache` scratch space.
"""
function lava_pipeline_cache_path(device_name::AbstractString, driver_version::AbstractString)
    sanitized_dev = replace(device_name, r"[^a-zA-Z0-9]+" => "_")
    sanitized_drv = replace(driver_version, r"[^a-zA-Z0-9]+" => "_")
    return joinpath(first(Base.DEPOT_PATH), "scratchspaces", "lava_spirv_cache",
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

Whether `bytes` starts with a `VkPipelineCacheHeaderVersionOne` (little-endian)
whose size, version, vendor, device and `pipelineCacheUUID` all match the
device. The UUID catches driver builds that report the same version string.
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
Create a `VkPipelineCache` for `device`, seeded from `path` when its header
matches `phys_device`. A blob damaged past the header that makes creation throw
is deleted and an empty cache returned.
"""
function create_lava_pipeline_cache(device::VK.Device, path::String, phys_device)
    initial = load_pipeline_cache_data(path, phys_device)
    function create_empty()
        ci = VK.PipelineCacheCreateInfo(Ptr{Cvoid}(C_NULL); initial_data_size=UInt64(0))
        return VK.PipelineCache(device, ci)
    end
    isempty(initial) && return create_empty()
    try
        return GC.@preserve initial begin
            ptr = Ptr{Cvoid}(pointer(initial))
            ci = VK.PipelineCacheCreateInfo(ptr; initial_data_size=UInt64(length(initial)))
            VK.PipelineCache(device, ci)
        end
    catch ex
        @warn "Lava: pipeline cache create from disk failed; starting empty" path exception=ex
        # `force=true` tolerates a missing file, so what fails here is real.
        try
            rm(path; force=true)
        catch rmex
            rmex isa Union{SystemError, Base.IOError} || rethrow()
            @warn "Lava: could not delete the unusable pipeline cache; it will fail again next session" path exception=rmex
        end
        return create_empty()
    end
end

"""
Write `ctx.pipeline_cache` to its disk path through a temp file and rename, so
no process reads a partial blob. No-op on a lost device.
"""
function save_pipeline_cache!(ctx)
    ctx.device_lost && return nothing
    isdefined(ctx, :pipeline_cache) || return nothing
    try
        # `ptr` is Libc.malloc'd by VK.jl; ours to `Libc.free`.
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

# Save the current context's cache at exit; registered once. Guarded, since the
# device or loader may already be gone at shutdown.
const _PIPELINE_CACHE_ATEXIT_REGISTERED = Ref(false)
function register_pipeline_cache_atexit!()
    _PIPELINE_CACHE_ATEXIT_REGISTERED[] && return
    _PIPELINE_CACHE_ATEXIT_REGISTERED[] = true
    atexit() do
        try
            ctx = existingdefault(VulkanAPI())
            ctx === nothing && return
            save_pipeline_cache!(ctx)
        catch ex
            # A throw here would replace the exit code; log instead, or a cache
            # that never saves looks like one that works.
            safe_fin_log("Lava: pipeline cache save at exit failed: " *
                         sprint(showerror, ex) * "\n")
        end
    end
end
