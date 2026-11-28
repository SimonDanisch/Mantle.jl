# Native qualification of the existing host-import implementation; no Mantle load.
module DirectHostProbe
import Vulkan as VK
using Mmap, Test, SHA, Dates, Statistics

const ROOT = normpath(joinpath(@__DIR__, "../.."))
const ERRORS = String[]
function debugcallback(severity, typ, data, user)
    if severity == VK.DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT
        push!(ERRORS, unsafe_string(unsafe_load(data).pMessage))
    end
    if severity == VK.DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT ||
       severity == VK.DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT
        VK.default_debug_callback(severity, typ, data, user)
    end
    UInt32(0)
end
const CALLBACK = @cfunction(debugcallback, UInt32,
    (VK.DebugUtilsMessageSeverityFlagEXT, VK.DebugUtilsMessageTypeFlagEXT,
     Ptr{VK.VkCore.VkDebugUtilsMessengerCallbackDataEXT}, Ptr{Cvoid}))

mutable struct Native
    instance
    messenger
    physical
    device
    memory_properties
    queue
    commands
    done
    host_import_alignment::Int
    next::UInt64
    name::String
    period::Float64
end
const VkContext = Native
struct VulkanQueue end
struct SubmitChannel{Q}
    ctx::Native
end
ctxof(ch::SubmitChannel) = ch.ctx
unwrap(x) = VK.unwrap(x)
LavaError(args...) = ErrorException(join(args, "\n"))
struct Stamp{T}
    value::T
end
Stamp{T}() where T = Stamp(zero(T))
const BUF_STATE_ALIVE = 1
struct VkManagedBuffer
    buffer
    memory
    address
    mapped_ptr
    bytes
    block
    last_write
    state
    ctx
end

# Evaluate exact existing function definitions with small structural scaffolding.
# No rewritten import or allocation algorithm is substituted for them.
function definition(file, marker)
    source = read(joinpath(ROOT, file), String)
    span = findfirst(marker, source)
    span === nothing && error("Missing source definition: $marker")
    ex, _ = Meta.parse(source, first(span); raise=true)
    ex.head == :function || error("Not a function: $marker")
    ex
end
Core.eval(@__MODULE__, definition("src/vulkan/runtime/memory.jl", "function find_memory_type(ctx::"))
Core.eval(@__MODULE__, definition("src/vulkan/runtime/external.jl", "function importhost(bq::"))
Core.eval(@__MODULE__, definition("src/runtime/api.jl", "function importablehost("))

function Native(match)
    inst = unwrap(VK.create_instance(["VK_LAYER_KHRONOS_validation"],
        ["VK_EXT_debug_utils", "VK_EXT_validation_features"];
        next=VK.ValidationFeaturesEXT([VK.VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT],
                                     VK.ValidationFeatureDisableEXT[]),
        application_info=VK.ApplicationInfo(v"0.1", v"0.1", v"1.3")))
    msg = VK.DebugUtilsMessengerEXT(inst, CALLBACK)
    pd = only(filter(p -> occursin(match, VK.get_physical_device_properties(p).device_name),
                     unwrap(VK.enumerate_physical_devices(inst))))
    props = VK.get_physical_device_properties(pd)
    queues = VK.get_physical_device_queue_family_properties(pd)
    family = UInt32(findfirst(q -> (q.queue_flags & VK.QUEUE_TRANSFER_BIT) != VK.QueueFlag(0), queues)-1)
    @assert queues[family+1].timestamp_valid_bits > 0
    alignment = Int(VK.get_physical_device_properties_2(pd,
        VK.PhysicalDeviceExternalMemoryHostPropertiesEXT).next.min_imported_host_pointer_alignment)
    features = VK.PhysicalDeviceFeatures2(VK.PhysicalDeviceFeatures();
        next=VK.PhysicalDeviceVulkan12Features(:buffer_device_address, :timeline_semaphore;
            next=VK.PhysicalDeviceVulkan13Features(:synchronization2)))
    info = VK.DeviceCreateInfo([VK.DeviceQueueCreateInfo(family, [1f0])], String[],
                              ["VK_EXT_external_memory_host"]; next=features)
    # Existing Vulkan.jl empty-layer-pointer normalization, as in recorded-slots.
    low = VK._DeviceCreateInfo(info)
    raw = low.vks
    fixed = typeof(raw)((name === :ppEnabledLayerNames ? Ptr{Cstring}(C_NULL) : getfield(raw, name)
                        for name in fieldnames(typeof(raw)))...)
    dev = unwrap(VK._create_device(pd, VK._DeviceCreateInfo(fixed, low.deps)))
    commands = unwrap(VK.create_command_pool(dev, family;
        flags=VK.COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT))
    done = VK.Semaphore(dev; next=VK.SemaphoreTypeCreateInfo(VK.SEMAPHORE_TYPE_TIMELINE, 0))
    Native(inst, msg, pd, dev, VK.get_physical_device_memory_properties(pd),
           VK.get_device_queue(dev, family, 0), commands, done, alignment, 0,
           props.device_name, Float64(props.limits.timestamp_period))
end

struct Region
    buffer
    memory
    pointer::Ptr{UInt8}
    bytes::Int
end
function region(d, bytes; host=false, required_flags=nothing)
    buf = VK.Buffer(d.device, bytes, VK.BUFFER_USAGE_TRANSFER_SRC_BIT |
        VK.BUFFER_USAGE_TRANSFER_DST_BIT, VK.SHARING_MODE_EXCLUSIVE, UInt32[])
    req = VK.get_buffer_memory_requirements(d.device, buf)
    flags = host ? VK.MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK.MEMORY_PROPERTY_HOST_COHERENT_BIT :
                   VK.MEMORY_PROPERTY_DEVICE_LOCAL_BIT
    required_flags === nothing || (flags = required_flags)
    index = find_memory_type(d, req.memory_type_bits, flags)
    mem = VK.DeviceMemory(d.device, req.size, index)
    unwrap(VK.bind_buffer_memory(d.device, buf, mem, 0))
    ptr = host ? Ptr{UInt8}(unwrap(VK.map_memory(d.device, mem, 0, bytes))) : Ptr{UInt8}(C_NULL)
    Region(buf, mem, ptr, bytes)
end
function release(d, r::Region)
    r.pointer == C_NULL || VK.unmap_memory(d.device, r.memory)
    finalize(r.buffer); finalize(r.memory)
end
function command(f, d)
    cb = only(unwrap(VK.allocate_command_buffers(d.device,
        VK.CommandBufferAllocateInfo(d.commands, VK.COMMAND_BUFFER_LEVEL_PRIMARY, 1))))
    unwrap(VK.begin_command_buffer(cb, VK.CommandBufferBeginInfo()))
    f(cb)
    unwrap(VK.end_command_buffer(cb))
    cb
end
function barrier(cb, srcstage, srcaccess, dststage, dstaccess)
    VK.cmd_pipeline_barrier_2(cb, VK.DependencyInfo([VK.MemoryBarrier2(;
        src_stage_mask=srcstage, src_access_mask=srcaccess,
        dst_stage_mask=dststage, dst_access_mask=dstaccess)], [], []))
end
passed(d, value) = unwrap(VK.get_semaphore_counter_value(d.device, d.done)) >= value
waitdone(d, value=d.next) = unwrap(VK.wait_semaphores(d.device,
    VK.SemaphoreWaitInfo([d.done], [value]), UInt64(10_000_000_000)))
function submit(d, cb; gate=nothing)
    waits = gate === nothing ? VK.SemaphoreSubmitInfo[] :
        [VK.SemaphoreSubmitInfo(gate, 1, 0; stage_mask=VK.PIPELINE_STAGE_2_ALL_COMMANDS_BIT)]
    value = d.next + 1
    packet = [VK.SubmitInfo2(waits, [VK.CommandBufferSubmitInfo(cb, 0)],
        [VK.SemaphoreSubmitInfo(d.done, value, 0; stage_mask=VK.PIPELINE_STAGE_2_ALL_COMMANDS_BIT)])]
    unwrap(VK.queue_submit_2(d.queue, packet))
    d.next = value
    value
end

mutable struct HostSource
    bytes::Vector{UInt8}
    writable::Bool
    kind::Symbol
end
mutable struct Lease
    source::Union{Nothing,HostSource}
    native::Union{Nothing,VkManagedBuffer}
    last::UInt64
end
@noinline isalive(w::WeakRef) = w.value !== nothing
@noinline function update_source!(lease, expected, seed)
    if lease.source.writable
        pattern!(lease.source.bytes, seed)
        pattern!(expected, seed)
    end
    nothing
end
@noinline source_matches(lease, expected) = lease.source.bytes == expected
@noinline function imported_flags(d, lease)
    props = unwrap(VK.get_memory_host_pointer_properties_ext(d.device,
        VK.EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT, Ptr{Cvoid}(pointer(lease.source.bytes))))
    req = VK.get_buffer_memory_requirements(d.device, lease.native.buffer)
    index = find_memory_type(d, req.memory_type_bits & props.memory_type_bits, VK.MemoryPropertyFlag(0))
    d.memory_properties.memory_types[index+1].property_flags
end
function retire!(d, lease)
    passed(d, lease.last) || return false
    native = lease.native
    if native !== nothing
        finalize(native.buffer); finalize(native.memory)
        lease.native = nothing
        lease.source = nothing
    end
    true
end
function pattern!(a, seed)
    @inbounds @simd for i in eachindex(a)
        x = UInt64(i) ⊻ (UInt64(seed) << 32)
        x = (x ⊻ (x >> 30)) * 0xbf58476d1ce4e5b9
        x = (x ⊻ (x >> 27)) * 0x94d049bb133111eb
        a[i] = UInt8((x ⊻ (x >> 31)) & 255)
    end
    a
end
function makesource(kind, bytes, path)
    expected = pattern!(Vector{UInt8}(undef, bytes), 17)
    if kind in (:private_anon, :cache_read)
        a = importablehost(UInt8, bytes)
        if kind === :cache_read
            write(path, expected)
            open(path, "r") do io
                GC.@preserve a unsafe_read(io, pointer(a), bytes)
            end
        else
            copyto!(a, expected)
        end
        return HostSource(a, true, kind), expected
    end
    write(path, expected)
    writable = kind in (:file_private_rw, :file_shared_rw)
    shared = kind in (:file_shared_rw, :file_shared_ro)
    a = open(path, writable ? "r+" : "r") do io
        Mmap.mmap(io, Vector{UInt8}, bytes; shared)
    end
    HostSource(a, writable, kind), expected
end
function makelease(d, kind, bytes, path)
    source, expected = makesource(kind, bytes, path)
    @assert UInt(pointer(source.bytes)) % d.host_import_alignment == 0
    @assert bytes % d.host_import_alignment == 0
    started = time_ns()
    native = GC.@preserve source importhost(SubmitChannel{VulkanQueue}(d), pointer(source.bytes), bytes)
    import_ms = (time_ns()-started)/1e6
    Lease(source, native, 0), WeakRef(source), expected, import_ms
end

function verify_case(d, kind, bytes, path)
    lease = nothing; dst = nothing; readback = nothing; cb = nothing; pool = nothing; gate = nothing
    gateopened = false
    errors_before = length(ERRORS)
    payload = bytes - 12 # logical extent need not be a whole import-alignment page
    try
        lease, weak, expected, import_ms = makelease(d, kind, bytes, path)
        native = lease.native
        flags = imported_flags(d, lease)
        # Existing importhost maps the same bytes without a flush; check its assumption.
        @test (flags & VK.MEMORY_PROPERTY_HOST_COHERENT_BIT) != VK.MemoryPropertyFlag(0)
        dst = region(d, payload); readback = region(d, payload; host=true)
        pool = unwrap(VK.create_query_pool(d.device, VK.QUERY_TYPE_TIMESTAMP, 2))
        gate = VK.Semaphore(d.device; next=VK.SemaphoreTypeCreateInfo(VK.SEMAPHORE_TYPE_TIMELINE, 0))
        cb = command(d) do cmd
            VK.cmd_reset_query_pool(cmd, pool, 0, 2)
            barrier(cmd, VK.PIPELINE_STAGE_2_HOST_BIT, VK.ACCESS_2_HOST_WRITE_BIT,
                    VK.PIPELINE_STAGE_2_COPY_BIT, VK.ACCESS_2_TRANSFER_READ_BIT)
            VK.cmd_write_timestamp_2(cmd, pool, 0; stage=VK.PIPELINE_STAGE_2_ALL_COMMANDS_BIT)
            VK.cmd_copy_buffer(cmd, native.buffer, dst.buffer, [VK.BufferCopy(0, 0, payload)])
            VK.cmd_write_timestamp_2(cmd, pool, 1; stage=VK.PIPELINE_STAGE_2_ALL_COMMANDS_BIT)
            barrier(cmd, VK.PIPELINE_STAGE_2_COPY_BIT, VK.ACCESS_2_TRANSFER_WRITE_BIT,
                    VK.PIPELINE_STAGE_2_COPY_BIT, VK.ACCESS_2_TRANSFER_READ_BIT)
            VK.cmd_copy_buffer(cmd, dst.buffer, readback.buffer, [VK.BufferCopy(0, 0, payload)])
            barrier(cmd, VK.PIPELINE_STAGE_2_COPY_BIT, VK.ACCESS_2_TRANSFER_WRITE_BIT,
                    VK.PIPELINE_STAGE_2_HOST_BIT, VK.ACCESS_2_HOST_READ_BIT)
        end
        # Release creator scope, GC, and replay one recording with changed data.
        GC.gc()
        @test isalive(weak)
        times = Float64[]; wall = Float64[]
        for iteration in 1:5
            update_source!(lease, expected, 16+iteration)
            started = time_ns()
            lease.last = submit(d, cb)
            waitdone(d, lease.last)
            push!(wall, (time_ns()-started)/1e6)
            actual = unsafe_wrap(Vector{UInt8}, readback.pointer, payload; own=false)
            @test actual == view(expected, 1:payload)
            @test source_matches(lease, expected)
            ticks = zeros(UInt64, 2)
            GC.@preserve ticks unwrap(VK.get_query_pool_results(d.device, pool, 0, 2,
                sizeof(ticks), Ptr{Cvoid}(pointer(ticks)), 8;
                flags=VK.QUERY_RESULT_64_BIT | VK.QUERY_RESULT_WAIT_BIT))
            push!(times, (ticks[2]-ticks[1])*d.period/1e6)
        end
        # Deterministically keep an accepted submission pending to test ownership.
        lease.last = submit(d, cb; gate)
        @test !passed(d, lease.last)
        @test !retire!(d, lease)
        GC.gc()
        @test isalive(weak)
        @test lease.native !== nothing
        unwrap(VK.signal_semaphore(d.device, VK.SemaphoreSignalInfo(gate, 1)))
        gateopened = true
        waitdone(d, lease.last)
        @test unsafe_wrap(Vector{UInt8}, readback.pointer, payload; own=false) == view(expected, 1:payload)
        @test retire!(d, lease)
        @test lease.source === nothing && lease.native === nothing
        GC.gc()
        @test !isalive(weak)
        @test length(ERRORS) == errors_before
        println("COPIED kind=",kind," import_bytes=",bytes," payload_bytes=",payload," imported_flags=",flags,
            " import_ms=",round(import_ms;digits=3),
            " upload_gpu_ms_median=",round(median(times[2:end]);digits=4),
            " upload_plus_readback_wall_ms_median=",round(median(wall[2:end]);digits=4),
            " explicit_upload_staging_bytes=0 validation_errors=",length(ERRORS)-errors_before)
        return (; device=d.name, kind, bytes, ok=true, gpu_ms=median(times[2:end]),
                 wall_ms=median(wall[2:end]), import_ms, error="")
    catch err
        message = sprint(showerror, err)
        println("FAILED kind=",kind," bytes=",bytes," error=",message)
        return (; device=d.name, kind, bytes, ok=false, gpu_ms=NaN, wall_ms=NaN,
                 import_ms=NaN, error=message)
    finally
        if gate !== nothing && !gateopened
            unwrap(VK.signal_semaphore(d.device, VK.SemaphoreSignalInfo(gate, 1)))
        end
        waitdone(d)
        lease === nothing || retire!(d, lease)
        cb === nothing || VK.free_command_buffers(d.device, d.commands, [cb])
        pool === nothing || finalize(pool)
        gate === nothing || finalize(gate)
        dst === nothing || release(d, dst)
        readback === nothing || release(d, readback)
        GC.gc() # Also release objects from a failed exact import function.
    end
end
function cleanup(d)
    unwrap(VK.device_wait_idle(d.device))
    finalize(d.done); finalize(d.commands); finalize(d.device)
    finalize(d.messenger); finalize(d.instance)
end
function probe(match; bytes=4<<20, kinds=(:private_anon, :cache_read, :file_private_rw,
                                      :file_shared_rw, :file_private_ro, :file_shared_ro))
    d = Native(match)
    results = NamedTuple[]
    println("DEVICE ", d.name, " host_alignment=",d.host_import_alignment)
    try
        mktempdir() do dir
            @testset "direct host transfer: $(d.name)" begin
                for kind in kinds
                    push!(results, verify_case(d, kind, bytes, joinpath(dir, string(kind)*".bin")))
                end
            end
        end
    finally
        cleanup(d)
    end
    results
end
end
