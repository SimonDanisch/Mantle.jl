# Kernel profiling, all opt-in:
#   1. `kernel_stats` / `spirv_op_stats`: SPIR-V size and opcode histogram.
#   2. `with_dispatch_timing(f)`: timestamp queries around every dispatch,
#      aggregated per kernel name; two `vkCmdWriteTimestamp` per dispatch when on.
#   3. `pipeline_exec_stats` / `pipeline_exec_ir`: driver statistics and IR via
#      `VK_KHR_pipeline_executable_properties`, which must be requested with
#      `enable_pipeline_executable_properties!()` before device creation.

import Vulkan as VK

"""
    SPIRVOpStats

Per-Op statistics extracted from a SPIR-V binary.
"""
struct SPIRVOpStats
    bytes::Int
    n_instructions::Int
    n_functions::Int
    op_histogram::Dict{UInt16,Int}   # opcode → count
end

"""
    spirv_op_stats(bytes::AbstractVector{UInt8}) -> SPIRVOpStats

Instruction count, `OpFunction` count and per-opcode histogram of a SPIR-V
binary, parsed directly (5-word header, then instructions whose first word is
`(WordCount << 16) | Opcode`; SPIR-V spec §2.3).
"""
function spirv_op_stats(bytes::AbstractVector{UInt8})
    n_bytes = length(bytes)
    n_bytes >= 20 && n_bytes % 4 == 0 && reinterpret(UInt32, bytes)[1] == 0x07230203 ||
        error("not a SPIR-V module in this byte order: $(n_bytes) bytes")
    n_instructions = n_functions = 0
    hist = Dict{UInt16,Int}()
    for (op, _) in spirvinstructions(bytes)
        n_instructions += 1
        hist[op] = get(hist, op, 0) + 1
        op == Op.OpFunction && (n_functions += 1)
    end
    return SPIRVOpStats(n_bytes, n_instructions, n_functions, hist)
end

"""
    KernelStats

Per-kernel SPIR-V statistics, plus driver-reported registers and scratch bytes
when `VK_KHR_pipeline_executable_properties` is enabled.
"""
struct KernelStats
    name::String                     # SPIR-V entry point, always "main"; see `source`
    source::String                   # the Julia kernel this came from, or "" if unknown
    workgroup_size::NTuple{3,Int}
    spirv::SPIRVOpStats
    registers::Union{Int,Nothing}    # filled from VK_KHR_pipeline_executable_properties when available
    scratch_bytes::Union{Int,Nothing}
end

function Base.show(io::IO, ::MIME"text/plain", s::KernelStats)
    print(io, "KernelStats(")
    print(io, isempty(s.source) ? s.name : s.source, ", ")
    print(io, "wg=", s.workgroup_size, ", ")
    print(io, "spirv=", s.spirv.bytes, "B (", s.spirv.n_instructions, " ops, ", s.spirv.n_functions, " fns)")
    s.registers !== nothing && print(io, ", regs=", s.registers)
    s.scratch_bytes !== nothing && print(io, ", scratch=", s.scratch_bytes, "B")
    print(io, ")")
end

"""
    kernel_stats(ctx, linked::LavaLinkedKernel) -> KernelStats

Stats for one compiled and linked kernel on `ctx`.
"""
function kernel_stats(ctx::LavaDevice, linked::LavaLinkedKernel)
    c = linked.compiled
    spirv = spirv_op_stats(c.spirv_bytes)
    exec = pipeline_exec_stats(ctx, linked)
    regs = exec === nothing ? nothing : get(exec, :registers, nothing)
    scratch = exec === nothing ? nothing : get(exec, :scratch_bytes, nothing)
    return KernelStats(c.entry_name, kernel_source_name(c), c.workgroup_size, spirv, regs, scratch)
end

# Per-dispatch GPU timing: with timing on, each dispatch is bracketed by two
# timestamps in the device's query pool (created on first use,
# `TIMESTAMP_POOL_SIZE[]` slots); `dispatch_timing_report` converts them with
# `timestampPeriod` and aggregates by name. Dispatches past the pool's capacity
# are not timed.

"""Per-dispatch timing record, one per timed dispatch."""
struct DispatchTiming
    kernel_name::String        # the dispatch's log name
    slot::Int                  # query pool slot of the START timestamp
    gpu_ns::Float64            # filled in on read-back; 0.0 during recording
end


# Two slots per dispatch. A device's pool is sized from it once, so change it
# only before that pool exists.
const TIMESTAMP_POOL_SIZE = Ref(MAX_DISPATCH_LOG * 2)

"""
    reset_dispatch_timing!(ctx = Device(VulkanAPI()))

Clear recorded timings and reset the slot counter; `with_dispatch_timing` calls it.
"""
function reset_dispatch_timing!(ctx::LavaDevice = Device(VulkanAPI()))
    empty!(ctx.caches.recorded_dispatches)
    ctx.caches.timestamp_next_slot = 0
    return nothing
end

"""
    KernelTimingReport

Per-kernel-name aggregation produced by `dispatch_timing_report()`.
"""
struct KernelTimingReport
    name::String
    n_dispatches::Int
    total_ns::Float64
    mean_ns::Float64
    median_ns::Float64
    p95_ns::Float64
end

function Base.show(io::IO, r::KernelTimingReport)
    print(io, "KernelTimingReport(", r.name, ": ", r.n_dispatches, "×, total=",
          round(r.total_ns / 1e6, digits=3), "ms, mean=",
          round(r.mean_ns / 1e3, digits=2), "µs)")
end

"""
    dispatch_timing_report(ctx = Device(VulkanAPI()); flush_first=true) -> Vector{KernelTimingReport}

Read the recorded timestamps back, convert to ns and aggregate by kernel name,
sorted by total time, largest first. `flush_first` waits for submitted work
first; pass `false` only after synchronizing yourself.
"""
function dispatch_timing_report(ctx::LavaDevice = Device(VulkanAPI()); flush_first::Bool=true)
    pool = ctx.caches.timestamp_pool
    pool === nothing && return KernelTimingReport[]
    isempty(ctx.caches.recorded_dispatches) && return KernelTimingReport[]
    if flush_first
        flush!(ctx.default_bq)
    end
    n_slots = ctx.caches.timestamp_next_slot
    n_slots == 0 && return KernelTimingReport[]
    raw = Vector{UInt64}(undef, n_slots)
    flags = VK.QUERY_RESULT_64_BIT | VK.QUERY_RESULT_WAIT_BIT
    GC.@preserve raw begin
        VK.get_query_pool_results(ctx.device, pool, UInt32(0), UInt32(n_slots),
                                  sizeof(raw), Ptr{Nothing}(pointer(raw)),
                                  UInt64(sizeof(UInt64)); flags=flags)
    end
    period = ctx.caches.timestamp_period_ns
    sized = DispatchTiming[]
    for d in ctx.caches.recorded_dispatches::Vector{Any}
        s_start = d.slot
        s_end = d.slot + 1
        s_end < n_slots || continue   # slot wrap-around safety
        ticks = raw[s_end + 1] - raw[s_start + 1]   # 1-based julia index
        push!(sized, DispatchTiming(d.kernel_name, d.slot, Float64(ticks) * period))
    end
    grouped = Dict{String,Vector{Float64}}()
    for d in sized
        push!(get!(grouped, d.kernel_name, Float64[]), d.gpu_ns)
    end
    reports = KernelTimingReport[]
    for (name, vals) in grouped
        sort!(vals)
        n = length(vals)
        total = sum(vals)
        median = vals[(n + 1) ÷ 2]
        p95 = vals[min(n, ceil(Int, 0.95 * n))]
        push!(reports, KernelTimingReport(name, n, total, total / n, median, p95))
    end
    sort!(reports; by=r -> -r.total_ns)
    return reports
end

"""
    with_dispatch_timing(f, ctx = Device(VulkanAPI())) -> Vector{KernelTimingReport}

Run `f()` with per-dispatch timing on and return the per-kernel report, after
waiting for submitted work. Restores the previous settings.

```julia
report = with_dispatch_timing() do
    Makie.colorbuffer(scene; backend=RayMakie, samples=4)
end
for r in report
    println(r)
end
```
"""
function with_dispatch_timing(f, ctx::LavaDevice = Device(VulkanAPI()))
    d = ctx.diag
    prev_timing = d.dispatch_timing
    # `ka_launch!` names a dispatch only with logging on, and records need the name.
    prev_logging = d.dispatch_logging
    reset_dispatch_timing!(ctx)
    d.dispatch_timing = true
    d.dispatch_logging = true
    try
        f()
        return dispatch_timing_report(ctx; flush_first=true)
    finally
        d.dispatch_timing = prev_timing
        d.dispatch_logging = prev_logging
    end
end

# `ctx`'s timestamp query pool, created on first use. Takes the context, never
# `Device(VulkanAPI())`: another device's pool on this command buffer is invalid.
function ensure_timestamp_pool!(ctx::LavaDevice)
    ctx.caches.timestamp_pool === nothing || return ctx.caches.timestamp_pool
    # ns per tick
    props = VK.get_physical_device_properties(ctx.physical_device)
    ctx.caches.timestamp_period_ns = Float64(props.limits.timestamp_period)
    info = VK.QueryPoolCreateInfo(VK.QUERY_TYPE_TIMESTAMP, UInt32(TIMESTAMP_POOL_SIZE[]))
    pool = VK.unwrap(VK.create_query_pool(ctx.device, info))
    ctx.caches.timestamp_pool = pool
    return pool
end

# Reset the whole pool on `cb`; done before the first timed dispatch (slot 0).
function reset_timestamp_pool_on_cb!(ctx::LavaDevice, cb::VK.CommandBuffer)
    pool = ctx.caches.timestamp_pool
    pool === nothing && return
    VK.cmd_reset_query_pool(cb, pool, UInt32(0), UInt32(TIMESTAMP_POOL_SIZE[]))
    return nothing
end

# Write the START timestamp before a dispatch. Returns the start slot, or -1
# if timing is off or the pool is full.
function maybe_write_dispatch_start_timestamp!(ctx::LavaDevice, cb::VK.CommandBuffer,
                                               kernel_name::AbstractString;
                                               stage = VK.PIPELINE_STAGE_COMPUTE_SHADER_BIT)
    ctx.diag.dispatch_timing || return -1
    pool = ensure_timestamp_pool!(ctx)
    slot = ctx.caches.timestamp_next_slot
    if slot == 0
        reset_timestamp_pool_on_cb!(ctx, cb)
    end
    if slot + 2 > TIMESTAMP_POOL_SIZE[]
        return -1  # pool full; the END writer also checks this
    end
    # Check totals against the frame's wall time: a missing share means some
    # command type is not timestamped.
    VK.cmd_write_timestamp(cb, stage, pool, UInt32(slot))
    ctx.caches.timestamp_next_slot = slot + 2
    push!(ctx.caches.recorded_dispatches, DispatchTiming(String(kernel_name), slot, 0.0))
    return slot
end

# Write the END timestamp after a dispatch, behind an execution-only barrier
# (`stage_mask` to `stage_mask`) so it fires after the dispatch's workgroups
# retire. Without it consecutive dispatches overlap and summed intervals exceed
# wall time, and on NVIDIA an indirect dispatch reads 2-12 µs whatever its work.
# This serialises dispatches while timing is on, so timed runs are slower.
# `barrier_fptr` is the device's own (`LavaDevice.cmd_pipeline_barrier_fptr`).
function maybe_write_dispatch_end_timestamp!(ctx::LavaDevice, cb::VK.CommandBuffer,
                                             start_slot::Int,
                                             barrier_fptr::Ptr{Nothing} = C_NULL;
                                             stage = VK.PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                             stage_mask::UInt32 = UInt32(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT))
    start_slot < 0 && return
    pool = ctx.caches.timestamp_pool
    pool === nothing && return
    if barrier_fptr != C_NULL
        ccall(barrier_fptr, Cvoid,
              (Ptr{Nothing}, VkPipelineStageFlags, VkPipelineStageFlags, VkDependencyFlags,
               UInt32, Ptr{VkMemoryBarrier}, UInt32, Ptr{Nothing}, UInt32, Ptr{Nothing}),
              cb.vks,
              VkPipelineStageFlags(stage_mask),
              VkPipelineStageFlags(stage_mask),
              VkDependencyFlags(0),
              UInt32(0), C_NULL, UInt32(0), C_NULL, UInt32(0), C_NULL)
    end
    VK.cmd_write_timestamp(cb, stage, pool, UInt32(start_slot + 1))
    return nothing
end

# VK_KHR_pipeline_executable_properties: per-pipeline driver statistics and IR.
# The extension can only be enabled at device creation.

const PIPELINE_EXEC_PROPERTIES_REQUESTED = Ref(false)

"""
    enable_pipeline_executable_properties!(enable::Bool=true)

Request `VK_KHR_pipeline_executable_properties` for the next `LavaDevice`
created; no effect on existing devices. Without it `pipeline_exec_stats` and
`pipeline_exec_ir` return `nothing`.
"""
function enable_pipeline_executable_properties!(enable::Bool=true)
    PIPELINE_EXEC_PROPERTIES_REQUESTED[] = enable
    return enable
end

"""
    pipeline_exec_stats(ctx, linked::LavaLinkedKernel) -> NamedTuple or Nothing

Driver statistics for a compute pipeline, as `(registers, scratch_bytes,
raw_stats)`: `registers` and `scratch_bytes` are `Int` or `nothing` when no
matching statistic exists; `raw_stats` holds every statistic as
`(name, description, value)`. `nothing` without the extension or when the
driver reports no executable.
"""
pipeline_exec_stats(ctx::LavaDevice, linked::LavaLinkedKernel) = pipeline_exec_stats(ctx, linked.pipeline)

# The KA launch path keeps only `LaunchPlan.pipeline`, hence this method.
function pipeline_exec_stats(ctx::LavaDevice, pipeline::LavaComputePipeline)
    ctx.pipeline_exec_props_available || return nothing
    pipe = pipeline.pipeline
    # No try/catch: a device without the extension returned above, so a failure
    # here is a bug. Swallowing one made a VK.jl construction error read as
    # "this driver reports no statistics".
    exec_info = VK.PipelineInfoKHR(pipe)
    execs = VK.unwrap(VK.get_pipeline_executable_properties_khr(ctx.device, exec_info))
    isempty(execs) && return nothing
    # A compute pipeline has exactly one executable.
    stats_info = VK.PipelineExecutableInfoKHR(pipe, UInt32(0))
    stats = VK.unwrap(VK.get_pipeline_executable_statistics_khr(ctx.device, stats_info))
    raw = NamedTuple[]
    registers = nothing
    scratch = nothing
    for s in stats
        name = String(s.name)
        # NVIDIA 595.80 returns 64-bit values in the upper 32 bits with
        # 0xFFFFFFFF in the lower half, likely a VK.jl union layout mismatch.
        # Take the high half unless it is 0 and the low half is not 0xFFFFFFFF.
        v = if s.format == VK.PIPELINE_EXECUTABLE_STATISTIC_FORMAT_BOOL32_KHR
            s.value.vks.b32 != 0
        elseif s.format == VK.PIPELINE_EXECUTABLE_STATISTIC_FORMAT_INT64_KHR
            i64 = s.value.vks.i64
            hi = Int(UInt64(reinterpret(UInt64, i64)) >> 32)
            lo = Int(reinterpret(UInt64, i64) & 0xFFFFFFFF)
            lo == 0xFFFFFFFF ? hi : (hi == 0 ? lo : hi)
        elseif s.format == VK.PIPELINE_EXECUTABLE_STATISTIC_FORMAT_UINT64_KHR
            u64 = s.value.vks.u64
            hi = Int((u64 >> 32) & 0xFFFFFFFF)
            lo = Int(u64 & 0xFFFFFFFF)
            lo == 0xFFFFFFFF ? hi : (hi == 0 ? lo : hi)
        elseif s.format == VK.PIPELINE_EXECUTABLE_STATISTIC_FORMAT_FLOAT64_KHR
            Float64(s.value.vks.f64)
        else
            nothing
        end
        push!(raw, (; name, description=String(s.description), value=v))
        lname = lowercase(name)
        # Statistic names are driver-defined: NVIDIA "Register Count", RADV
        # "VGPRs". Every spelling is tried on every driver; VGPRs win where both
        # appear, since they bound occupancy on AMD.
        if occursin("vgpr", lname) && !occursin("spill", lname) && !occursin("pre-sched", lname) && v isa Int
            registers = v                                  # authoritative, overwrite
        elseif registers === nothing && occursin("register", lname) && v isa Int
            registers = v
        end
        if scratch === nothing && (occursin("scratch", lname) || occursin("spill", lname)) && v isa Int
            scratch = v
        end
    end
    return (; registers, scratch_bytes=scratch, raw_stats=raw)
end

"""
    pipeline_exec_ir(ctx, pipeline::LavaComputePipeline) -> Vector{NamedTuple} or Nothing

The driver's text representations of a compute pipeline as
`(name, description, text)`, on RADV the NIR, ACO IR and GPU assembly.
`nothing` unless `enable_pipeline_executable_properties!()` preceded device
creation.
"""
pipeline_exec_ir(ctx::LavaDevice, linked::LavaLinkedKernel) = pipeline_exec_ir(ctx, linked.pipeline)

# Raw `ccall`: Vulkan.jl's wrapper stops after the count call, returning sizes
# with `pData = NULL`; the second call with a buffer per entry fills the text.
function pipeline_exec_ir(ctx::LavaDevice, pipeline::LavaComputePipeline)
    ctx.pipeline_exec_props_available || return nothing
    C = VK.VkCore
    fptr = VK.function_pointer(ctx.device, "vkGetPipelineExecutableInternalRepresentationsKHR")
    info = Ref(C.VkPipelineExecutableInfoKHR(C.VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_INFO_KHR,
                                             C_NULL, pipeline.pipeline.vks, UInt32(0)))
    count = Ref{UInt32}(0)
    getreps(reps) = GC.@preserve info count reps begin
        r = ccall(fptr, VK.VkCore.VkResult,
                  (VK.VkCore.VkDevice, Ptr{VK.VkCore.VkPipelineExecutableInfoKHR}, Ptr{UInt32},
                   Ptr{VK.VkCore.VkPipelineExecutableInternalRepresentationKHR}),
                  ctx.device.vks, info, count,
                  reps === nothing ? C_NULL : pointer(reps))
        r == C.VK_SUCCESS ||
            error("vkGetPipelineExecutableInternalRepresentationsKHR returned $r")
    end
    empty = C.VkPipelineExecutableInternalRepresentationKHR(
        C.VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_INTERNAL_REPRESENTATION_KHR, C_NULL,
        ntuple(_ -> Cchar(0), 256), ntuple(_ -> Cchar(0), 256), C.VkBool32(0), Csize_t(0), C_NULL)
    getreps(nothing)
    reps = fill(empty, count[])
    getreps(reps)                                   # sizes
    bufs = [Vector{UInt8}(undef, r.dataSize) for r in reps]
    for (i, r) in enumerate(reps)
        reps[i] = C.VkPipelineExecutableInternalRepresentationKHR(
            r.sType, C_NULL, r.name, r.description, r.isText, r.dataSize, pointer(bufs[i]))
    end
    GC.@preserve bufs getreps(reps)                 # text
    cstr(t) = String(UInt8.(collect(Iterators.takewhile(!iszero, t))))
    return [(; name = cstr(r.name), description = cstr(r.description),
               text = String(bufs[i][1:something(findfirst(iszero, bufs[i]), length(bufs[i]) + 1) - 1]))
            for (i, r) in enumerate(reps) if r.isText != 0]
end
