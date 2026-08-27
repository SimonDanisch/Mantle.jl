# Lava Profiling — kernel stats + per-dispatch GPU timing + driver pipeline introspection.
#
# Three orthogonal capabilities, all opt-in / off by default:
#
#   1. `kernel_stats(compiled)` — pure SPIR-V analysis (size, instruction count,
#      Op histogram). Zero runtime cost; works without a render in flight.
#   2. `with_dispatch_timing(f)` — wrap a render with Vulkan timestamp queries
#      around every dispatch. Returns a per-kernel-name aggregated report.
#      Adds two `vkCmdWriteTimestamp` calls per dispatch when enabled; the
#      hot path is untouched when disabled.
#   3. `pipeline_exec_stats(linked)` — driver-side stats via
#      `VK_KHR_pipeline_executable_properties` (register count, scratch,
#      vendor-specific ISA). Only available when the extension is enabled
#      via `enable_pipeline_executable_properties!()` BEFORE device
#      creation. Returns `nothing` otherwise.
#
# Designed to stay in the codebase. No `# TODO: remove` markers anywhere.

import Vulkan as VK
# `const VK = Vulkan` used to be here. Inside Mantle the bare name `Vulkan`
# is the backend MARKER — a singleton — so that line aliased VK to a
# DataType and every `VK.CommandBuffer` became a field access on it. The
# import above is the alias.

# ============================================================================
# 1. SPIR-V instruction-level analysis
# ============================================================================

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

Parse a SPIR-V binary and return total instruction count, the count of
`OpFunction` definitions, and the per-opcode histogram. Does NOT execute
spirv-dis — it parses the binary directly, ~µs per kernel.

The SPIR-V binary format is a header of five 32-bit words followed by an
instruction stream. Each instruction's first word is
`(WordCount << 16) | Opcode`; the next `WordCount - 1` words are operands.
See SPIR-V spec §2.3.
"""
function spirv_op_stats(bytes::AbstractVector{UInt8})
    n_bytes = length(bytes)
    n_bytes >= 20 || error("SPIR-V too short: $(n_bytes) bytes (header is 20)")
    n_bytes % 4 == 0 || error("SPIR-V byte length not word-aligned: $(n_bytes)")
    words = reinterpret(UInt32, bytes)
    # Header: magic (0), version (1), generator (2), bound (3), schema (4)
    words[1] == 0x07230203 || error("Bad SPIR-V magic: $(string(words[1]; base=16))")
    n_instructions = 0
    n_functions = 0
    hist = Dict{UInt16,Int}()
    i = 6                              # 1-based, skip 5-word header
    while i <= length(words)
        word0 = words[i]
        wc = UInt16(word0 >> 16)
        op = UInt16(word0 & 0xFFFF)
        wc >= 1 || error("Zero word-count instruction at word index $(i-1)")
        n_instructions += 1
        hist[op] = get(hist, op, 0) + 1
        # OpFunction = 54 in SPIR-V core spec
        op == 0x36 && (n_functions += 1)
        i += Int(wc)
    end
    return SPIRVOpStats(n_bytes, n_instructions, n_functions, hist)
end

"""
    KernelStats

Per-kernel statistics: SPIR-V size + instruction profile, plus optional
driver-reported register/scratch numbers when pipeline-executable-properties
are available.
"""
struct KernelStats
    name::String                     # SPIR-V entry point — always "main"; see `source`
    source::String                   # the Julia kernel this came from, or "" if unknown
    workgroup_size::NTuple{3,Int}
    spirv::SPIRVOpStats
    registers::Union{Int,Nothing}    # filled from VK_KHR_pipeline_executable_properties when available
    scratch_bytes::Union{Int,Nothing}
end

# `kernel_source_name` is in Lava, at the end of `compiler/compilation.jl`. It
# parses a mangled entry symbol out of a compiled kernel's `source_name` or its
# IR — string work on something the compiler produced, with no device in it, and
# `Lava/test/test_compile_overhead.jl` needs it without one.

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
    kernel_stats(linked::LavaLinkedKernel) -> KernelStats

Stats for a single compiled+linked kernel. Use this from a debugger or
test when you already have a `LavaLinkedKernel` in hand.
"""
function kernel_stats(linked::LavaLinkedKernel; source::AbstractString = "")
    c = linked.compiled
    spirv = spirv_op_stats(c.spirv_bytes)
    exec = pipeline_exec_stats(linked)
    regs = exec === nothing ? nothing : get(exec, :registers, nothing)
    scratch = exec === nothing ? nothing : get(exec, :scratch_bytes, nothing)
    # `source` given by the caller wins: a FROZEN entry has no IR to recover the
    # name from — `frozen_store` writes `ir = ""` on purpose, the string being
    # "session-specific and large" — but its cache KEY is
    # `(typeof(f), tt, workgroup_size)`, which names the kernel outright.
    name = isempty(source) ? kernel_source_name(c) : source
    return KernelStats(c.entry_name, name, c.workgroup_size, spirv, regs, scratch)
end

"""Kernel name from a frozen-cache key, whose first element is `typeof(f)`."""
function frozen_key_source_name(key)
    key isa Tuple && !isempty(key) || return ""
    K = key[1]
    K isa DataType || return ""
    return replace(string(nameof(K)), r"^#" => "", r"#\d+$" => "")
end

"""
    list_compiled_kernels() -> Vector{KernelStats}

Stats for every kernel this device has compiled. Useful right after a render —
you see every kernel involved.

```julia
render!(vp, scene, film, camera)
sort(list_compiled_kernels(); by=k -> -k.spirv.bytes)  # biggest first
```

Takes a context, because there is no longer a global registry of them to walk —
which is the point. This briefly read the OUTER level of a two-level dict and
handed `kernel_stats` another `Dict`; a cache that belongs to a device cannot be
iterated at the wrong depth.

**Both of the context's caches, because the shipped path only populates one.**
`get_compiled_kernel_and_pipeline` consults the frozen cache first and *returns*
on a hit, so a kernel that came off disk never reaches
`GPUCompiler.cached_compilation` and never enters `caches.linked`. Every runner
calls `use_frozen_kernels` in its `__init__`, so for the configuration that
actually ships `caches.linked` is empty and this returned an empty vector —
measured on a Depth Anything forward: **0 kernels reported against 45 live
dispatch names**, taking `kernel_stats`, `pipeline_exec_stats` and every
register/scratch number with it. A profiler blind exactly where it is needed.
"""
function list_compiled_kernels(ctx::VkContext = vk_context())
    stats = KernelStats[]
    seen = Set{UInt64}()
    for (_, linked) in ctx.caches.linked
        push!(seen, objectid(linked))
        push!(stats, kernel_stats(linked))
    end
    for (key, linked) in ctx.caches.frozen_mem
        # `Any`-valued, and one kernel can land in both caches across a session.
        linked isa LavaLinkedKernel && !(objectid(linked) in seen) || continue
        push!(seen, objectid(linked))
        # The frozen entry has no IR to recover a name from — `frozen_store`
        # writes `ir = ""` on purpose — but its KEY is `(typeof(f), tt, wg)`.
        push!(stats, kernel_stats(linked; source = frozen_key_source_name(key)))
    end
    return stats
end

# ============================================================================
# 2. Per-dispatch GPU timing via Vulkan timestamp queries
# ============================================================================
#
# Strategy: when `with_dispatch_timing(f)` is active, every `vk_dispatch!` writes
# a TOP_OF_PIPE timestamp before the dispatch and a BOTTOM_OF_PIPE timestamp
# after.  Pairs of timestamps are read back after `f` returns (after the
# implicit `KA.synchronize`), converted to ns via the device's
# `timestampPeriod`, and aggregated by dispatch name.
#
# The query pool is allocated lazily on first use, sized to MAX_DISPATCH_LOG*2.
# If a render exceeds that budget we ring back to slot 0 and the late records
# overwrite the early ones — caller is warned.

"""Per-dispatch timing record. One entry per `vk_dispatch!` call while timing is on."""
struct DispatchTiming
    kernel_name::String        # from `bq.last_dispatch_info` at dispatch time
    slot::Int                  # query pool slot of the START timestamp
    gpu_ns::Float64            # filled in on read-back; 0.0 during recording
end


# Two query slots per dispatch (start + end).  Default size is conservative;
# auto-grows on first activation if `MAX_DISPATCH_LOG` was bumped.
const TIMESTAMP_POOL_SIZE = Ref(MAX_DISPATCH_LOG * 2)
# `timestamp_pool`, `timestamp_next_slot` and `timestamp_period_ns` are
# `VkContext` fields — a query pool is a device-owned handle, and holding one
# in a module `Ref` is the defect that corrupted the memory pool next door.

"""
    enable_dispatch_timing!(enable::Bool=true)

Toggle the recording of Vulkan timestamp queries around every dispatch. When
on, each `vk_dispatch!` writes two timestamps; when off, the dispatch hot
path is unchanged. Equivalent to (and used by) `with_dispatch_timing(f)`.

If timing is being turned ON for the first time in this session, the
timestamp query pool is created lazily inside `vk_dispatch!`.
"""
function enable_dispatch_timing!(enable::Bool=true, ctx::VkContext = vk_context())
    ctx.diag.dispatch_timing = enable
    return enable
end

"""
    reset_dispatch_timing!()

Clear recorded timings and reset the slot counter. Called automatically by
`with_dispatch_timing(f)`.
"""
function reset_dispatch_timing!(ctx::VkContext = vk_context())
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
    dispatch_timing_report(; flush_first=true) -> Vector{KernelTimingReport}

Read back the recorded timestamps from the query pool, convert to ns, and
aggregate by kernel name. Returns a vector sorted by total time descending.

If `flush_first=true` (default), forces a `vk_flush!` first so all submitted
dispatches' timestamps are written to the pool before read-back. Pass
`false` only if you've already synchronized and want to inspect a partial
record.
"""
function dispatch_timing_report(ctx::VkContext = vk_context(); flush_first::Bool=true)
    pool = ctx.caches.timestamp_pool
    pool === nothing && return KernelTimingReport[]
    isempty(ctx.caches.recorded_dispatches) && return KernelTimingReport[]
    if flush_first
        vk_flush!(ctx.default_bq)
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
    # Apply the actual ns values back into the records.
    sized = DispatchTiming[]
    for d in ctx.caches.recorded_dispatches::Vector{Any}
        s_start = d.slot
        s_end = d.slot + 1
        s_end < n_slots || continue   # slot wrap-around safety
        ticks = raw[s_end + 1] - raw[s_start + 1]   # 1-based julia index
        push!(sized, DispatchTiming(d.kernel_name, d.slot, Float64(ticks) * period))
    end
    # Aggregate by name.
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
    with_dispatch_timing(f) -> Vector{KernelTimingReport}

Run `f()` with per-dispatch GPU timing enabled. Returns the aggregated
per-kernel report after `f` completes (waiting for all submitted work).

```julia
report = with_dispatch_timing() do
    Makie.colorbuffer(scene; backend=RayMakie, integrator=vp)
end
for r in report
    println(r)
end
```
"""
function with_dispatch_timing(f, ctx::VkContext = vk_context())
    d = ctx.diag
    prev_timing = d.dispatch_timing
    # `ka_launch!` only writes `bq.last_dispatch_info` when dispatch logging is on
    # (it's behind a flag for the production hot path).  We need that name to
    # tag timing records, so flip dispatch logging on for the duration too.
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

# Internal: ensure the timestamp pool exists. Called from `vk_dispatch!`.
#
# Takes the context rather than asking `vk_context()` for it. A query pool is a
# device-owned handle, and this runs per dispatch: reaching for the global here
# would put the wrong device's pool on the recording command buffer as surely as
# the caches did before they became fields.
function ensure_timestamp_pool!(ctx::VkContext)
    ctx.caches.timestamp_pool === nothing || return ctx.caches.timestamp_pool
    # Capture the device's timestamp period (ns per tick) once.
    props = VK.get_physical_device_properties(ctx.physical_device)
    ctx.caches.timestamp_period_ns = Float64(props.limits.timestamp_period)
    info = VK.QueryPoolCreateInfo(VK.QUERY_TYPE_TIMESTAMP, UInt32(TIMESTAMP_POOL_SIZE[]))
    pool = VK.unwrap(VK.create_query_pool(ctx.device, info))
    ctx.caches.timestamp_pool = pool
    return pool
end

# Internal: reset the pool between captures (must happen on a command buffer).
# Called from `vk_dispatch!` when slot index hits 0.
function reset_timestamp_pool_on_cb!(ctx::VkContext, cb::VK.CommandBuffer)
    pool = ctx.caches.timestamp_pool
    pool === nothing && return
    VK.cmd_reset_query_pool(cb, pool, UInt32(0), UInt32(TIMESTAMP_POOL_SIZE[]))
    return nothing
end

# Internal: write the START timestamp before a dispatch.  Returns the start
# slot, or -1 if timing is off or the pool is full.
function maybe_write_dispatch_start_timestamp!(ctx::VkContext, cb::VK.CommandBuffer,
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
    # COMPUTE_SHADER_BIT for both endpoints.  Spec says the timestamp fires
    # when all prior commands have completed at this stage; for a pure-
    # compute queue that's the natural boundary.
    #
    # Note: indirect dispatches in our Hikari volpath path-tracing kernels
    # (vp_trace_and_shade, vp_shade_surface_hits, vp_handle_escaped_rays)
    # report 2–12 µs / dispatch which looks too low for 1.4 Mpx of work.
    # Initially suspected NVIDIA driver bug, but: 1.4 Mpx / 6000-warp
    # occupancy × 32 threads/warp = ~7 rays/thread × ~50 ns/ray (RT-core
    # accelerated BVH + simple BSDF + light + atomic queue push) =
    # ~350 ns wave-time = ~12 µs wall-time for the dispatch.  That matches.
    # The timing is right; the RT-core path is just THAT cheap.
    #
    # An earlier version of this note concluded from that "all the GPU time in
    # surface-only scenes lives in vp_generate_camera_rays (Sobol-bound)".  That
    # was an artefact of a blind spot: only `cmd_dispatch` and
    # `cmd_dispatch_indirect` were timestamped, so `cmd_trace_rays_khr` — the
    # entire hw_accel=true ray-tracing workload — was missing from the report,
    # which accounted for ~155 ms of a ~3.3 s frame.  With the RT paths
    # instrumented (see raytracing/pipeline.jl), killeroo_gold on hw_accel=true
    # measures 97.5 % of GPU time in rt_indirect (~19 ms/dispatch) and 1.7 % in
    # vp_generate_camera_rays.  Read totals against the frame's wall time before
    # trusting a breakdown; if they do not roughly add up, something is not
    # being timestamped.
    VK.cmd_write_timestamp(cb, stage, pool, UInt32(slot))
    ctx.caches.timestamp_next_slot = slot + 2
    push!(ctx.caches.recorded_dispatches, DispatchTiming(String(kernel_name), slot, 0.0))
    return slot
end

# Internal: write the END timestamp after a dispatch.
#
# Inserts an execution barrier (compute → BOTTOM_OF_PIPE, no memory barrier)
# before the timestamp so it really fires after this dispatch's workgroups
# complete.  Without the barrier, NVIDIA's driver was firing the
# BOTTOM_OF_PIPE timestamp when the cmd_dispatch_indirect *command* was
# processed — not when its workgroups completed — which made every indirect
# dispatch appear to take 2–12 µs no matter how much work it did.  Direct
# dispatches were unaffected; the symptom only showed on indirect because
# indirect-dispatch parameters are read at dispatch time and workgroup
# launches are visibly deferred.
# `barrier_fptr` is passed in rather than read from a global: `vkCmdPipelineBarrier`
# is resolved per device, and a global one records the wrong driver's barrier
# into the other device's command buffer. See `VkContext.cmd_pipeline_barrier_fptr`.
function maybe_write_dispatch_end_timestamp!(ctx::VkContext, cb::VK.CommandBuffer,
                                             start_slot::Int,
                                             barrier_fptr::Ptr{Nothing} = C_NULL;
                                             stage = VK.PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                             stage_mask::UInt32 = UInt32(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT))
    start_slot < 0 && return
    pool = ctx.caches.timestamp_pool
    pool === nothing && return
    # The barrier this comment has always described was missing from the code.
    # Without it, consecutive dispatches overlap and each measured interval spans
    # this dispatch *plus* whatever is still in flight ahead of it, so summing the
    # records double-counts: the totals came out larger than the wall-clock time of
    # the very block being measured (73.7 ms of "GPU time" for a 45 ms block), and
    # varied 26 -> 74 ms between runs as barrier patterns shifted.
    #
    # An execution-only barrier (no memory barrier) makes the end timestamp fire
    # after this dispatch's workgroups retire. That serialises dispatches while
    # timing is on, which is the point: per-kernel attribution needs them
    # serialised, and the overlap you give up is not attributable to any one
    # kernel anyway. Timing runs are therefore slower than production runs — use
    # them for the breakdown, not for the step rate.
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

# The pool, the slot counter and the enable flag are all `VkContext` fields now,
# so a reset makes fresh ones and there is nothing to clear. Only the record list
# is genuinely module-level — it is a host-side diagnostic buffer, not device
# state — and dropping it is the whole callback.

# ============================================================================
# 3. VK_KHR_pipeline_executable_properties — driver-side stats
# ============================================================================
#
# The extension reports per-pipeline statistics like "Number of registers"
# and "Scratch bytes" and (driver permitting) the actual ISA.  It must be
# enabled at device creation; once a device is up the only thing we can do
# is query.  We expose `enable_pipeline_executable_properties!()` to set the
# flag for the NEXT device creation, plus `pipeline_exec_stats(linked)` to
# read on-demand.

const PIPELINE_EXEC_PROPERTIES_REQUESTED = Ref(false)

"""
    enable_pipeline_executable_properties!(enable::Bool=true)

Request that the next `VkContext` enable `VK_KHR_pipeline_executable_properties`.
Has no effect on a device that's already up — must be called before the
session creates its Vulkan device (typically before `using RayMakie`).

When enabled, `pipeline_exec_stats(linked)` returns driver-reported
statistics (register count, scratch space, etc.); otherwise that function
returns `nothing`.
"""
function enable_pipeline_executable_properties!(enable::Bool=true)
    PIPELINE_EXEC_PROPERTIES_REQUESTED[] = enable
    return enable
end

"""
    pipeline_exec_stats(linked::LavaLinkedKernel) -> NamedTuple or Nothing

Query `VK_KHR_pipeline_executable_properties` for a linked compute pipeline.
Returns a NamedTuple `(registers, scratch_bytes, raw_stats)` where:

- `registers::Union{Int,Nothing}` — driver's "registers per thread" or
  similar; nothing if no matching statistic was returned.
- `scratch_bytes::Union{Int,Nothing}` — driver's scratch/spill bytes if
  reported.
- `raw_stats::Vector{NamedTuple}` — every statistic the driver returned,
  with `name`, `description`, `value_kind`, `value` fields.

Returns `nothing` if the extension wasn't enabled at device creation, or
if the driver doesn't expose anything for this pipeline.
"""
pipeline_exec_stats(linked::LavaLinkedKernel) = pipeline_exec_stats(linked.pipeline)

# Takes the pipeline, not the linked kernel: the KA launch path keeps
# `LaunchPlan.pipeline` and drops the `LavaLinkedKernel`, so the linked-kernel
# method was unreachable from the only place that launches anything.
function pipeline_exec_stats(pipeline::LavaComputePipeline)
    PIPELINE_EXEC_PROPERTIES_REQUESTED[] || return nothing
    ctx = vk_context()
    pipe = pipeline.pipeline
    # Discover the pipeline's executables.
    # NO try/catch around either query, and that is the point.
    #
    # Both were wrapped in `catch ex; @debug; return nothing`, and it cost this
    # project a day. `get_pipeline_executable_statistics_khr` was throwing an
    # outright `ConstructionBase` error — VK.jl could not even build the
    # result struct, for every driver — and the swallow turned that into
    # `registers = nothing`, which reads as "this driver declines to report
    # statistics". It was written up as an AMD driver limitation. RADV in fact
    # returns twenty statistics per pipeline, more than NVIDIA does.
    #
    # A profiler that hides its own failure is worse than one that has none: the
    # absent numbers look like a fact about the hardware. `PIPELINE_EXEC_PROPERTIES_REQUESTED[]`
    # above already covers "the caller did not ask for this", and the extension is
    # only enabled when the device advertises it, so anything reaching here and
    # failing is a bug that must be seen.
    exec_info = VK.PipelineInfoKHR(pipe)
    execs = VK.unwrap(VK.get_pipeline_executable_properties_khr(ctx.device, exec_info))
    isempty(execs) && return nothing
    # For compute pipelines there is exactly one executable; query its stats.
    stats_info = VK.PipelineExecutableInfoKHR(pipe, UInt32(0))
    stats = VK.unwrap(VK.get_pipeline_executable_statistics_khr(ctx.device, stats_info))
    raw = NamedTuple[]
    registers = nothing
    scratch = nothing
    for s in stats
        name = String(s.name)
        # VK.jl wraps the C union VkPipelineExecutableStatisticValueKHR as
        # a `vks` field with one `data::NTuple{8,UInt8}` member — accessing
        # that member calls back into a wrapper whose lifetime is transient,
        # so we copy the 8 bytes up-front into a local before doing any
        # interpretation.  NVIDIA driver 595.80 lays the actual value in bytes
        # 4–7 of the union for our UINT64 stats (verified empirically:
        # Register Count and Binary Size are sensible there; lower 4 bytes are
        # garbage or a header tag).  Pick the bytes that decode to a
        # non-ASCII-looking number under 2^28.
        # VulkanCore.LibVulkan exposes the C union via overloaded getproperty
        # on `:b32`/`:i64`/`:u64`/`:f64`.  On NVIDIA driver 595.80 the actual
        # data lives in the upper 32 bits of the u64 union value with
        # `0xFFFFFFFF` sentinel in the lower half — likely a VK.jl
        # padding mismatch around the format field, but the pattern is
        # consistent (72 registers, 32 KB binary, 6.4 KB shared all decode
        # correctly when shifted).  Take the high half for INT64/UINT64.
        v = if s.format == VK.PIPELINE_EXECUTABLE_STATISTIC_FORMAT_BOOL32_KHR
            s.value.vks.b32 != 0
        elseif s.format == VK.PIPELINE_EXECUTABLE_STATISTIC_FORMAT_INT64_KHR
            i64 = s.value.vks.i64
            # Take whichever half is non-sentinel.  If high is 0xFFFFFFFF the
            # value is in low half (e.g. genuine 0); otherwise high.
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
        # The statistic names are driver-defined and NOT portable. Matching only
        # `"register"` found NVIDIA's "Register Count" and nothing on RADV, which
        # names the same quantity `VGPRs` — so `registers` came back `nothing` for
        # every kernel and the occupancy denominator looked unavailable on AMD
        # when the driver was in fact reporting it.
        #
        # This is a NAME table, not a vendor branch: every spelling is tried on
        # every driver, and a driver that uses two of them gets the first. VGPRs
        # are the ones that bound occupancy on AMD (SGPRs are scalar and rarely
        # the limit), so they are preferred where both appear.
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
