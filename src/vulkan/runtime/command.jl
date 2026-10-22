# Command buffers and their submission. No command buffer is open when a Mantle
# call returns: a plan's `Recording` is written once and submitted every run;
# everything else is a `OneShot` written and sealed inside one call. Both reach
# the driver only through `submit!`/`submitlist!`, which submit immediately.
# Everything a command names goes through `hold!` and stays alive until the
# submission's token has passed. Cross-queue order: core's `crosswaits!` maps the
# named buffers' stamps to (channel, token) waits, lowered here to timeline
# semaphore waits, and `stamp!` records the new owner after a successful submit.

# Capacity of the per-context dispatch log ring.
const MAX_DISPATCH_LOG = 2000

"""
Build a dispatch's log string. Called through `invokelatest` so the launch path
does not infer `print`/`show`: those gain methods from every plotting package,
and loading GLMakie would invalidate the precompiled launch code. Runs only when
dispatch logging is on.
"""
@noinline dispatch_log_string(args...) = string(args...)

function log_dispatch!(bq::SubmitChannel{<:VulkanQueue}, info::String)
    d = (ctxof(bq)).diag
    d.dispatch_logging || return
    log = d.dispatch_log
    if length(log) >= MAX_DISPATCH_LOG
        popfirst!(log)
    end
    push!(log, info)
    # `dispatch_log_file` mirrors each entry to disk, flushed per line: a hung
    # dispatch blocks the process in a fence wait where the in-memory ring cannot
    # be printed, so the file's last line names the kernel that hung.
    f = d.dispatch_log_file
    f === nothing || open(io -> (println(io, info); flush(io)), f, "a")
    return
end

# Raw `VkMemoryBarrier` through a direct ccall: VK.jl's `MemoryBarrier` wrapper
# allocates about 1.2 KB per `cmd_pipeline_barrier`.
import Vulkan.VkCore: VkMemoryBarrier, VK_STRUCTURE_TYPE_MEMORY_BARRIER,
    VkAccessFlags, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
    VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
    VK_ACCESS_MEMORY_READ_BIT, VK_ACCESS_MEMORY_WRITE_BIT,
    VkPipelineStageFlags, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
    VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
    VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
    VK_PIPELINE_STAGE_TRANSFER_BIT, VkDependencyFlags
# `vkCmdPipelineBarrier` is a per-device pointer, resolved in the `VkContext`
# constructor.

# VK.jl types `PIPELINE_STAGE_2_ALL_COMMANDS_BIT` as the sync1
# `PipelineStageFlag`; sync2 fields need a `PipelineStageFlag2` (same value, 0x10000).
const STAGE2_ALL_COMMANDS = VK.PipelineStageFlag2(VK.PIPELINE_STAGE_2_ALL_COMMANDS_BIT)

"""
    recpatch!(owner, address, at)

Record that the eight bytes at host pointer `at`, in memory the recording owns,
hold device address `address`. `pack_arg!` calls it for every pointer it writes.
A [`Recording`](@ref) collects them for the plan's patch table, which a resource
move consults (`closerecording!`); other owners ignore them.
"""
recpatch!(owner, address::UInt64, at::Ptr{UInt8}) = nothing
recpatch!(rec::Recording, address::UInt64, at::Ptr{UInt8}) =
    (push!(rec.patches, (address, at)); nothing)

"""
    recpatchfields!(owner, T, at)

[`recpatch!`](@ref) every nonzero device pointer inside the packed `T` at `at`.
Which fields hold pointers is core's `devicepointeroffsets`; collecting host
pointers and resolving them to `(region, offset)` at seal (`patchtarget`) is
this backend's. Not a method of core's `notepacked!`: core never calls it, and
`test/test_backend_vocabulary.jl` rejects backend methods on core functions
outside the vocabulary.
"""
@inline function recpatchfields!(owner, ::Type{T}, at::Ptr{UInt8}) where {T}
    fields = devicepointeroffsets(T)
    isempty(fields) && return nothing
    for f in fields
        addr = unsafe_load(Ptr{UInt64}(at + f))
        addr == 0 && continue
        recpatch!(owner, addr, at + f)
    end
    return nothing
end

"""Allocate one primary command buffer from the queue's pool. It stays with its
`OneShot`, which core pools; a `Recording` borrows it until `release!`."""
function allocate_cmd(bq::SubmitChannel{<:VulkanQueue})
    alloc_info = VK.CommandBufferAllocateInfo(driver(bq).cmd_pool, VK.COMMAND_BUFFER_LEVEL_PRIMARY, 1)
    return throw_if_error(bq, "vkAllocateCommandBuffers",
        VK.allocate_command_buffers(vkdevice(bq), alloc_info))[1]
end

# Begin infos built once: the low-level struct is boxed (it carries a GC deps
# vector), so one per begin allocated about 100 bytes in an allocation-free
# `run!`. They hold only flags, so one instance serves every command buffer.
const BEGIN_INFO_ONE_TIME =
    VK._CommandBufferBeginInfo(flags = VK.COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT)
const BEGIN_INFO_SIMULTANEOUS =
    VK._CommandBufferBeginInfo(flags = VK.COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT)

"""
    recording!(bq) -> Recording

Open and begin a plan's command buffer with `SIMULTANEOUS_USE`, which is
required: there is one recording per plan, and `run!` may submit it again while
the previous run is still pending.
"""
function recording!(bq::SubmitChannel{<:VulkanQueue})
    ownthread(bq)
    device_lost(ctxof(bq)) && throw(LavaError(
        "recording!", "Vulkan device is lost — cannot record",
        "Call reset_device!() to reinitialize, or restart Julia."))
    # Same pool as a one-shot: core's `acquire!` sweeps, returns a reset buffer
    # and opens the hold frame this recording takes at `seal!`. The `OneShot`
    # shell is dropped here and rebuilt by `release!`.
    o = acquire!(bq)
    throw_if_error(bq, "vkBeginCommandBuffer",
        VK._begin_command_buffer(o.cmd, BEGIN_INFO_SIMULTANEOUS))
    return Recording(bq, o.cmd, Region[], Any[], UInt64(0), true,
                     Tuple{UInt64,Ptr{UInt8}}[], nothing, VkManagedBuffer[])
end

"""Close a command buffer: nothing more may be emitted, and it can be submitted.
Idempotent."""
function seal!(c::Closed)
    c.open || return c
    throw_if_error(queueof(c), "vkEndCommandBuffer", VK.end_command_buffer(c.cmd))
    c.open = false
    sealed!(c)
    return c
end

# A one-shot is submitted once, so its hold frame goes to that submission
# (`submissionholds!`). A recording is submitted every run and must outlive all
# of them, so it takes the frame at seal and keeps it until `release!`.
sealed!(::OneShot) = nothing
sealed!(rec::Recording) = (rec.holds = takeholds!(queueof(rec)); nothing)

# A recording took its frame at seal; handing it over must not pop another.
submissionholds!(::SubmitChannel, ::Recording) = nothing

"""
    release!(rec::Recording)

Give back the recording's regions, holds, descriptor sets and command buffer;
running it afterwards is an error. `Mantle.free!` on a plan calls it. Waits for
the recording's last submission: the regions are only retired, but the command
buffer goes on to serve one-shots, and re-beginning one the device still reads
is undefined behaviour.
"""
function release!(rec::Recording)
    bq = queueof(rec)
    dev = lavadevice(ctxof(bq))
    let tok = rec.token
        tok == 0 || waitfor!(bq, tok)
    end
    rec.token = 0
    rec.open && seal!(rec)          # takes the hold frame if it is still open
    # The hold list goes back to the channel's spares; `sealed!` takes a fresh
    # frame if this is recorded again.
    unhold!(bq, rec.holds)
    rec.holds = nothing
    empty!(rec.sync)
    let p = pool(dev)
        for r in rec.regions
            retire!(p, dev, r)
        end
    end
    empty!(rec.regions)
    empty!(rec.sets)
    # The command buffer returns to core's free list as a one-shot.
    finishrecording!(bq, OneShot(bq, rec.cmd, Region[], VkManagedBuffer[], false))
    return nothing
end

# The recording primitives core calls (`graph/lifetime.jl`); core owns the free
# list and decides when each runs.

makerecording(bq::SubmitChannel{<:VulkanQueue}) =
    OneShot(bq, allocate_cmd(bq), Region[], VkManagedBuffer[], false)

function resetrecording!(bq::SubmitChannel{<:VulkanQueue}, o::OneShot)
    throw_if_error(bq, "vkResetCommandBuffer",
        VK.reset_command_buffer(o.cmd; flags = VK.CommandBufferResetFlag(0)))
    o.open = false
    return o
end

destroyrecording!(bq::SubmitChannel{<:VulkanQueue}, o::OneShot) =
    VK.free_command_buffers(vkdevice(bq), driver(bq).cmd_pool, [o.cmd])

"""
Give a one-shot back to core's free list and its regions back to the pool.
Called once its submission has passed or when it never reached the device, so
nothing reads the regions and they are released, not retired. `sync` is cleared
here rather than in `resetrecording!` so a pooled one-shot keeps no
`VkManagedBuffer` alive.
"""
function finishrecording!(bq::SubmitChannel{<:VulkanQueue}, o::OneShot)
    # A build that threw returns a begun, never-ended buffer; seal it so no
    # pooled one-shot reads as open.
    o.open && seal!(o)
    if !isempty(o.regions)
        for r in o.regions
            release!(r)
        end
        empty!(o.regions)
    end
    empty!(o.sync)
    push!(bq.free, o)
    return nothing
end

"""Emit an allocation-free global memory barrier with the given stage and access
masks."""
@inline function memorybarrier!(cmd::VK.CommandBuffer, ctx::VkContext,
                               srcstage, dststage, srcaccess, dstaccess)
    barrier_ref = Ref(VkMemoryBarrier(VK_STRUCTURE_TYPE_MEMORY_BARRIER, C_NULL,
                                      VkAccessFlags(srcaccess), VkAccessFlags(dstaccess)))
    GC.@preserve barrier_ref begin
        ccall(ctx.cmd_pipeline_barrier_fptr, Cvoid,
              (Ptr{Nothing}, VkPipelineStageFlags, VkPipelineStageFlags, VkDependencyFlags,
               UInt32, Ptr{VkMemoryBarrier}, UInt32, Ptr{Nothing}, UInt32, Ptr{Nothing}),
              cmd.vks, VkPipelineStageFlags(srcstage), VkPipelineStageFlags(dststage),
              VkDependencyFlags(0), UInt32(1), barrier_ref,
              UInt32(0), C_NULL, UInt32(0), C_NULL)
    end
    return nothing
end

@inline function headbarrier!(cmd::VK.CommandBuffer, ctx::VkContext)
    both = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT
    memorybarrier!(cmd,ctx,VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,both,both)
    return nothing
end

"""
    oneshot(f, bq) -> OneShot
    oneshot!(f, bq; tag = nothing) -> token      (core, `graph/lifetime.jl`)

Write a command buffer inside one call: `f` gets an `Emitter` over a pooled
`OneShot`. `oneshot!` also submits it; `oneshot` seals it and returns it
unsubmitted, for a caller that submits it together with other work (a frame
with its swapchain semaphores). Every one-shot opens with
[`headbarrier!`](@ref), since nothing declares what its commands touch.
"""
function oneshot(f, bq::SubmitChannel{<:VulkanQueue})
    o = openoneshot(bq)
    try
        f(Emitter(o, nothing))
    catch
        # Back to core with its holds; `finishrecording!` seals it.
        release!(bq, o)
        rethrow()
    end
    seal!(o)
    return o
end

"""Open a one-shot and return it open, for a caller that emits into it over
several calls (a run's front one-shot, a windowed frame) and seals it itself."""
function openoneshot(bq::SubmitChannel{<:VulkanQueue})
    ownthread(bq)
    ctx = ctxof(bq)
    device_lost(ctx) && throw(LavaError(
        "oneshot", "Vulkan device is lost — cannot record",
        "Call reset_device!() to reinitialize, or restart Julia. " *
        "All existing LavaArrays are invalid after reset and must be re-allocated."))
    # `acquire!` drains, then returns a reset buffer with a hold frame open.
    return begin!(acquire!(bq), BEGIN_INFO_ONE_TIME)
end

# Begin a pooled command buffer and write the head barrier. Shared by core's
# `recorder` and `openoneshot` so both start a one-shot the same way.
function begin!(o::OneShot, info)
    bq = queueof(o)
    throw_if_error(bq, "vkBeginCommandBuffer", VK._begin_command_buffer(o.cmd, info))
    o.open = true
    headbarrier!(o.cmd, ctxof(bq))
    return o
end

# The emitter core's `oneshot!` records through.
recorder(bq::SubmitChannel{<:VulkanQueue}, o::OneShot) = Emitter(begin!(o, BEGIN_INFO_ONE_TIME), nothing)

"""
    hold!(owner::Closed, obj) -> obj
    hold!(e::Emitter, obj) -> obj

Every object a command names goes through here. The channel keeps it alive until
the submission carrying these commands has passed (core's [`hold!`](@ref)), and
a buffer is also recorded in `owner.sync` for cross-channel ordering at submit.
"""
@inline function hold!(owner::Closed, obj)
    hold!(queueof(owner), obj)
    syncbuf!(owner, obj)
    return obj
end

"""
    syncbuf!(owner, obj)

Record that these commands name `obj`'s buffer, for `crosswaits!` and `stamp!`
at submit; a no-op for objects without device-visible bytes. Deduplicated in
O(1) through `buf.syncpos`, the buffer's index in the list it last joined; an
identity scan made a frame naming thousands of buffers quadratic. A buffer
alternating between owners can appear twice in one list, which is harmless.
"""
@inline syncbuf!(::Closed, @nospecialize(obj)) = nothing
@inline function syncbuf!(owner::Closed, buf::VkManagedBuffer)
    list = owner.sync
    p = buf.syncpos
    (1 <= p <= length(list) && @inbounds(list[p]) === buf) && return nothing
    push!(list, buf)
    buf.syncpos = length(list)
    return nothing
end

"""
    submit!(bq, closed...; waits = (), signals = (), fence = nothing) -> token

Submit closed command buffers immediately, in order, in one `vkQueueSubmit2`,
and return the timeline value the submission signals. Buffers share a
submission only by being passed together. `waits` and `signals` are extra
`(semaphore, value, stage)` triples for presentation, beside the derived
cross-channel waits and the timeline signal; `fence` is signalled on completion.

Does not record the submission: the caller hands it to core (`handover!`),
since only the caller knows whether it gives away a one-shot or will resubmit
a recording.
"""
submit!(bq::SubmitChannel{<:VulkanQueue}, closed::Closed...;
        waits = (), signals = (), fence::Union{Nothing,VK.Fence} = nothing) =
    submitlist!(bq, nothing, closed; waits, signals, fence)

"""
    submitlist!(bq, front, closed; waits, signals, fence) -> token

[`submit!`](@ref) of `front` (a one-shot or `nothing`) followed by every element
of `closed`, a tuple or vector that may name one recording several times (a loop
body's pieces). A caller-kept vector lets a run submit without allocating.
"""
function submitlist!(bq::SubmitChannel{<:VulkanQueue}, front::Union{Nothing,Closed}, closed;
                     waits = (), signals = (), fence::Union{Nothing,VK.Fence} = nothing)
    ownthread(bq)
    ctx = ctxof(bq)
    q = driver(bq)
    device_lost(ctx) && throw(LavaError(
        "submit!", "Vulkan device is lost — cannot submit",
        "Call reset_device!()"))
    # Submitting is when nothing more can be added; `seal!` is idempotent.
    front === nothing || seal!(front)
    for c in closed
        seal!(c)
    end
    Threads.atomic_add!(ctx.diag.flush_counter, 1)
    # Reclaim finished work once per submission: per launch it allocates
    # (`test_dispatch_allocation.jl`).
    drain!(bq)

    nf = front === nothing ? 0 : 1
    ncb = nf + length(closed)
    raw_cbs = q.raw_cb_infos
    length(raw_cbs) == ncb || resize!(raw_cbs, ncb)
    front === nothing || (raw_cbs[1] = VK.vk.VkCommandBufferSubmitInfo(
        VK.vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO, C_NULL, front.cmd.vks, UInt32(0)))
    for (i, c) in enumerate(closed)
        raw_cbs[nf + i] = VK.vk.VkCommandBufferSubmitInfo(
            VK.vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO, C_NULL, c.cmd.vks, UInt32(0))
    end

    # Core derives the waits from the named buffers' stamps; lowered below to
    # timeline semaphore waits.
    cross = q.waits
    empty!(cross)
    front === nothing || crosswaits!(bq, front.sync, cross)
    for c in closed
        crosswaits!(bq, c.sync, cross)
    end

    # Not rolled back if the submit fails: nothing is stamped with it, and
    # reusing a value could give it to two submissions.
    q.next_timeline += 1
    token = q.next_timeline

    raw_waits = q.raw_wait_infos
    nw = length(cross) + length(waits)
    length(raw_waits) == nw || resize!(raw_waits, nw)
    for i in eachindex(cross)
        (other, v) = cross[i]
        raw_waits[i] = VK.vk.VkSemaphoreSubmitInfo(
            VK.vk.VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO, C_NULL,
            crosssemaphore(bq, other).vks, v, UInt64(STAGE2_ALL_COMMANDS), UInt32(0))
    end
    for (j, (sem, v, stage)) in enumerate(waits)
        raw_waits[length(cross) + j] = VK.vk.VkSemaphoreSubmitInfo(
            VK.vk.VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO, C_NULL,
            sem.vks, UInt64(v), UInt64(stage), UInt32(0))
    end
    raw_sigs = q.raw_signal_infos
    ns = 1 + length(signals)
    length(raw_sigs) == ns || resize!(raw_sigs, ns)
    raw_sigs[1] = VK.vk.VkSemaphoreSubmitInfo(
        VK.vk.VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO, C_NULL,
        timelineof(bq).vks, token,
        UInt64(VK.PIPELINE_STAGE_2_ALL_COMMANDS_BIT), UInt32(0))
    for (j, (sem, v, stage)) in enumerate(signals)
        raw_sigs[1 + j] = VK.vk.VkSemaphoreSubmitInfo(
            VK.vk.VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO, C_NULL,
            sem.vks, UInt64(v), UInt64(stage), UInt32(0))
    end

    raw_subs = q.raw_submits
    # This device's pointer from the dispatch table; the lookup allocates nothing.
    fptr = VK.function_pointer(VK.global_dispatcher[], vkdevice(bq), :vkQueueSubmit2)
    vkfence = fence === nothing ? VK.vk.VkFence(C_NULL) : fence.vks
    submit_result = GC.@preserve raw_cbs raw_waits raw_sigs raw_subs begin
        raw_subs[1] = VK.vk.VkSubmitInfo2(
            VK.vk.VK_STRUCTURE_TYPE_SUBMIT_INFO_2, C_NULL, 0,
            UInt32(nw), pointer(raw_waits), UInt32(ncb), pointer(raw_cbs),
            UInt32(ns), pointer(raw_sigs))
        VK.vk.vkQueueSubmit2(vkqueue(bq).vks, UInt32(1), pointer(raw_subs), vkfence, fptr)
    end
    if submit_result != VK.vk.VK_SUCCESS
        # The raw call has no result wrapper, so mark device loss here. Nothing
        # was stamped, so nothing waits for this token.
        submit_result == VK.vk.VK_ERROR_DEVICE_LOST && (ctx.device_lost = true)
        throw(LavaError("vkQueueSubmit2",
            "queue submission failed with $(submit_result) " *
            "($(ncb) command buffer(s), last dispatch: $(driver(bq).last_dispatch_info))",
            "If the device is lost, call reset_device!() or restart Julia."))
    end

    # Submitted: this token is now the last work naming these buffers.
    front === nothing || stamp!(bq, token, front.sync)
    for c in closed
        stamp!(bq, token, c.sync)
    end

    q.prev_dispatch_info = q.last_dispatch_info
    return token
end

"""
The timeline semaphore of another channel this submission waits on. Throws if
that channel belongs to another `VkContext`: a semaphore of device A in a device
B submit segfaults inside `vkQueueSubmit2`. A released channel was flushed first
(`release_batch_queue!`), so `crosswaits!` finds its tokens passed and never
asks for it.
"""
@inline function crosssemaphore(bq::SubmitChannel{<:VulkanQueue}, other)
    o = other::SubmitChannel{VulkanQueue{VkContext},OneShot,UInt64,Submission{Union{Nothing,OneShot}}}
    ctxof(o) === ctxof(bq) || throw(LavaError(
        "submit!",
        "a buffer was last used on a channel from a DIFFERENT VkContext",
        "Two Vulkan devices are live and one buffer has been used on both. " *
        "Allocate and use the buffer under a single context."))
    return timelineof(o)
end

"""
    Emitter(owner, args)

Where commands are written. Every `emit_*` takes an emitter and never a queue,
so no queue state can change what a command buffer contains. `owner` is a
[`Recording`](@ref) for a plan recorded once or a [`OneShot`](@ref) for work
belonging to this call; both answer `hold!` and `scratch!`. `args` is the plan's
`ArgMemory`, or `nothing` off the plan path, where arguments go to scratch.
"""
struct Emitter{O,C,A}
    cmd::VK.CommandBuffer
    owner::O
    ctx::C
    args::A
end

Emitter(c::Closed, am) = Emitter(c.cmd, c, ctxof(queueof(c)), am)

@inline hold!(e::Emitter, obj) = hold!(e.owner, obj)

"""The queue an emitter's commands will be submitted on."""
queueof(e::Emitter{O}) where {O<:Closed} = queueof(e.owner)

"""
    emit_dispatch!(emitter, pipeline, argaddr, groups, tlas = nothing, name = "")
    emit_dispatch_indirect!(emitter, pipeline, argaddr, indirect, tlas = nothing, name = "")

Write one compute dispatch: bind, push the argument address, dispatch. No
barrier: a plan emits its derived hazards (`emitpass!`), and a one-shot opens
with [`headbarrier!`](@ref). The acceleration structure goes through
[`bindtlas!`](@ref), a no-op for `nothing`; its descriptor set belongs to the
emitter's owner ([`tlasset!`](@ref)).
"""
@inline function emit_dispatch!(e::Emitter, pipeline::LavaComputePipeline, argaddr::UInt64,
                        groups::NTuple{3,<:Integer}, tlas = nothing,
                        name::AbstractString = "")
    cmd = e.cmd
    VK.cmd_bind_pipeline(cmd, VK.PIPELINE_BIND_POINT_COMPUTE, pipeline.pipeline)
    hold!(e, pipeline)
    bindtlas!(e, pipeline, tlas)
    pushargs!(e, pipeline, argaddr)
    ts = maybe_write_dispatch_start_timestamp!(e.ctx, cmd, name)
    VK.cmd_dispatch(cmd, UInt32(groups[1]), UInt32(groups[2]), UInt32(groups[3]))
    maybe_write_dispatch_end_timestamp!(e.ctx, cmd, ts, e.ctx.cmd_pipeline_barrier_fptr)
    emitted!(e, name)
    return nothing
end

@inline function emit_dispatch_indirect!(e::Emitter, pipeline::LavaComputePipeline,
                                 argaddr::UInt64, indirect, tlas = nothing,
                                 name::AbstractString = "")
    cmd = e.cmd
    VK.cmd_bind_pipeline(cmd, VK.PIPELINE_BIND_POINT_COMPUTE, pipeline.pipeline)
    hold!(e, pipeline)
    bindtlas!(e, pipeline, tlas)
    pushargs!(e, pipeline, argaddr)
    mb = indirect.buf[]::VkManagedBuffer
    ts = maybe_write_dispatch_start_timestamp!(e.ctx, cmd, name)
    VK.cmd_dispatch_indirect(cmd, mb.buffer, UInt64(indirect.offset))
    maybe_write_dispatch_end_timestamp!(e.ctx, cmd, ts, e.ctx.cmd_pipeline_barrier_fptr)
    # The device reads the workgroup counts from this buffer.
    hold!(e, indirect)
    emitted!(e, name)
    return nothing
end

"""
    bindtlas!(emitter, pipeline, tlas)

Bind an acceleration structure at set 0, binding 0, and hold everything a ray
query reads. A no-op for `nothing`; the `VulkanTLAS` method is in
`raytracing/hwtlas.jl`.
"""
@inline bindtlas!(::Emitter, ::LavaComputePipeline, ::Nothing) = nothing

"""
Log a written dispatch for the DEVICE_LOST report. Runs at emit time, so once
per recording rather than per run: the log names the kernels in the command
buffer that failed.
"""
@inline function emitted!(e::Emitter, name::AbstractString)
    d = e.ctx.diag
    (d.dispatch_logging && !isempty(name)) || return nothing
    Threads.atomic_add!(d.total_dispatches, 1)
    log_dispatch!(queueof(e), Base.invokelatest(dispatch_log_string,
                                                d.total_dispatches[], " ", name)::String)
    return nothing
end


"""
Poll interval of `flush!` within its budget `driver(bq).flush_timeout_ns`
(nanoseconds, `0` waits forever). Polling turns a TDR or a hung dispatch into an
error, where one unbounded `vkWaitSemaphores` leaves a process only a kill ends.
"""
const FLUSH_WAIT_QUANTUM_NS = UInt64(2) * 1_000_000_000



"""
    pushargs!(emitter, pipeline, argaddr)

A compute dispatch's push constants: the address of its argument block, and for a
kernel that can throw (`push_constant_size == 16`, see Lava's
`wrap_entry_for_vulkan!`) the address of the device's exception flag after it.
"""
@inline function pushargs!(e, pipeline::LavaComputePipeline, argaddr::UInt64)
    if pipeline.push_constant_size == 16
        ref = Ref((argaddr, e.ctx.exception_flag.address))
        GC.@preserve ref begin
            VK.cmd_push_constants(e.cmd, pipeline.pipeline_layout, VK.SHADER_STAGE_COMPUTE_BIT,
                UInt32(0), UInt32(16),
                Ptr{Nothing}(Base.unsafe_convert(Ptr{NTuple{2,UInt64}}, ref)))
        end
    else
        push_constants_bda!(e.cmd, pipeline.pipeline_layout, VK.SHADER_STAGE_COMPUTE_BIT, argaddr)
    end
    return nothing
end

"""
    checkexceptions!(ctx)

Throw if a kernel on `ctx` threw since the last check, and clear the flag. Call
after a wait. The invocation that threw stopped there while the others finished,
so the kernel's output is incomplete: an error, not a warning.
"""
function checkexceptions!(ctx::VkContext)
    p = Ptr{UInt32}(ctx.exception_flag.mapped_ptr)
    unsafe_load(p) == 0 && return nothing
    unsafe_store!(p, UInt32(0))
    throw(LavaError("kernel exception",
        "a kernel threw an exception on $(ctx.device_name): an invocation reached an " *
        "`error`, a `throw` or a failed bounds check, and stopped there. What that kernel " *
        "wrote is missing that invocation's results.",
        "Run the kernel on the host, or bisect it with `@lava_printf`, to find which " *
        "invocation and which check."))
end

"""
    push_constants_bda!(cmd, layout, stage_flags, bda)

Record an 8-byte push constant holding device address `bda`. The `Ref` does not
escape, so this allocates nothing.
"""
@inline function push_constants_bda!(cmd::VK.CommandBuffer, layout::VK.PipelineLayout,
                                      stage_flags, bda::UInt64)
    ref = Ref(bda)
    GC.@preserve ref begin
        VK.cmd_push_constants(cmd, layout, stage_flags,
            UInt32(0), UInt32(8),
            Ptr{Nothing}(Base.unsafe_convert(Ptr{UInt64}, ref)))
    end
end

"""
Barrier between an unmodelled prepare and the indirect launch that reads what it
wrote, inside one one-shot: makes the prepare's shader writes visible to the
indirect read and the launch's own reads. Plans get this from `emitprepares!`.
`dst_stage` and `extra_dst_access` name the consumer (compute or ray tracing).
"""
@inline function indirectbarrier!(e::Emitter, dst_stage::VK.PipelineStageFlag,
                                  extra_dst_access::VK.AccessFlag)
    dst_access = VkAccessFlags(VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT) |
                 VkAccessFlags(extra_dst_access)
    memorybarrier!(e.cmd,e.ctx,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,dst_stage,
                   VK_ACCESS_SHADER_WRITE_BIT,dst_access)
    return nothing
end

"""
    query_timeline(bq::SubmitChannel{<:VulkanQueue}) -> UInt64

Current counter of `timelineof(bq)`. Raw and allocation-free because every sweep
calls it; on error it repeats the query through the checked wrapper, marks
device loss and throws `LavaVulkanError`. Callers gate on `device_lost` first,
so the throw only fires if the device died in between.
"""
@inline function query_timeline(bq::SubmitChannel{<:VulkanQueue})::UInt64
    # The shared counter cell belongs to the owning thread; another thread gets a
    # fresh `Ref`. Two threads writing one cell could return a wrong counter, and
    # a token reported passed too early is a use-after-free.
    ref = Threads.threadid() == bq.thread ?
          (driver(bq).slots).counter : Ref(UInt64(0))
    fptr = VK.function_pointer(VK.global_dispatcher[], vkdevice(bq),
                               :vkGetSemaphoreCounterValue)
    res = VK.vk.vkGetSemaphoreCounterValue(vkdevice(bq).vks, timelineof(bq).vks,
                                        ref, fptr)
    res == VK.vk.VK_SUCCESS && return ref[]
    result = VK.get_semaphore_counter_value(vkdevice(bq), timelineof(bq))
    iserror(result) || return unwrap(result)
    mark_if_lost!(bq, result)
    e = unwrap_error(result)::VK.VulkanError
    throw(LavaVulkanError("get_semaphore_counter_value", Int32(e.code), e.msg,
        e.code == VK.ERROR_DEVICE_LOST ?
            "Device lost. Call reset_device!() to reinitialize, or restart Julia." :
            "Timeline semaphore query failed with an unexpected VkResult. " *
            "This is a driver bug, stack corruption, or an invalid semaphore handle."))
end

"""
    wait_timeline!(bq, val)

Block until this queue's timeline reaches `val`. Raw, through the queue's slots,
so a wait allocates nothing; errors fall through to the checked wrapper, which
throws with context.
"""
function wait_timeline!(bq::SubmitChannel{<:VulkanQueue}, val::UInt64)
    # The slots belong to the owning thread; other threads take the checked wrapper.
    if Threads.threadid() != bq.thread
        wait_semaphores!(bq, VK.SemaphoreWaitInfo([timelineof(bq)], [val]))
        return nothing
    end
    slots = driver(bq).slots
    slots.wait_sems[1] = timelineof(bq).vks
    slots.wait_values[1] = val
    fptr = VK.function_pointer(VK.global_dispatcher[], vkdevice(bq), :vkWaitSemaphores)
    res = GC.@preserve slots begin
        slots.wait_info[] = VK.vk.VkSemaphoreWaitInfo(
            VK.vk.VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO, C_NULL, 0,
            UInt32(1), pointer(slots.wait_sems), pointer(slots.wait_values))
        VK.vk.vkWaitSemaphores(vkdevice(bq).vks, slots.wait_info, typemax(UInt64), fptr)
    end
    res == VK.vk.VK_SUCCESS && return nothing
    wait_semaphores!(bq, VK.SemaphoreWaitInfo([timelineof(bq)], [val]))  # throws
    return nothing
end

"""
    waitfor!(bq, token)

Block until this queue's timeline reaches `token`. A token belongs to one queue's
timeline; the device-level `waitfor!(dev, tok)` waits on the primary queue and is
wrong for work submitted on another.
"""
function waitfor!(bq::SubmitChannel{<:VulkanQueue}, tok::UInt64)
    passed(bq, tok) && return nothing
    wait_timeline!(bq, tok)
    return nothing
end

# Fallible Vulkan calls go through `throw_if_error`, or `mark_if_lost!` for
# callers with their own recovery, so every DEVICE_LOST sets the flag the entry
# gates (`oneshot`, `submit!`, `flush!`) check.

"""
    mark_if_lost!(bq::SubmitChannel{<:VulkanQueue}, result)
    mark_if_lost!(ctx::VkContext, result)

Mark the device lost if `result` is `ERROR_DEVICE_LOST`; never unwraps or
throws. For callers that recover before throwing; otherwise use `throw_if_error`.
"""
@inline function mark_if_lost!(ctx::VkContext, result)
    iserror(result) || return
    e = unwrap_error(result)::VK.VulkanError
    e.code == VK.ERROR_DEVICE_LOST && mark_device_lost!(ctx)
    return
end
@inline mark_if_lost!(bq::SubmitChannel{<:VulkanQueue}, result) = mark_if_lost!(ctxof(bq), result)

device_lost_hint(call) =
    "Vulkan device is lost during $(call). " *
    "Call reset_device!() to reinitialize, or restart Julia."

"""
    throw_if_error(bq::SubmitChannel{<:VulkanQueue}, result)
    throw_if_error(ctx::VkContext, result)
    throw_if_error(ctx_or_bq, call::String, result)       # adds a call name

Return the unwrapped value on success. On error, mark device loss for
`ERROR_DEVICE_LOST` and throw `LavaVulkanError` naming `call`. Use instead of
`unwrap(VK.foo(...))`.
"""
@inline function throw_if_error(ctx::VkContext, call::String, result)
    iserror(result) || return unwrap(result)
    mark_if_lost!(ctx, result)
    e = unwrap_error(result)::VK.VulkanError
    suggestion = e.code == VK.ERROR_DEVICE_LOST ? device_lost_hint(call) : ""
    throw(LavaVulkanError(call, Int32(e.code), e.msg, suggestion))
end
@inline throw_if_error(ctx::VkContext, result) = throw_if_error(ctx, "Vulkan call", result)
@inline throw_if_error(bq::SubmitChannel{<:VulkanQueue}, args...) = throw_if_error(ctxof(bq), args...)

"""
    wait_for_fences!(bq, fences; wait_all=true, timeout=typemax(UInt64))

`vkWaitForFences` through `throw_if_error`.
"""
@inline wait_for_fences!(bq::SubmitChannel{<:VulkanQueue}, fences;
                         wait_all::Bool=true, timeout::UInt64=typemax(UInt64)) =
    throw_if_error(bq, "vkWaitForFences", VK.wait_for_fences(vkdevice(bq), fences, wait_all, timeout))

"""
    wait_semaphores!(bq, info; timeout=typemax(UInt64))

`vkWaitSemaphores` through `throw_if_error`.
"""
@inline wait_semaphores!(bq::SubmitChannel{<:VulkanQueue}, info::VK.SemaphoreWaitInfo;
                         timeout::UInt64=typemax(UInt64)) =
    throw_if_error(bq, "vkWaitSemaphores", VK.wait_semaphores(vkdevice(bq), info, timeout))

"""
    flush_stall_report(bq, target) -> String

The queue's state when `flush!` timed out, as text for the error. Separates a
slow or wedged kernel (counter one below `target`, its submission in flight)
from a wait on a value nothing will signal (every in-flight submission already
signalled): the queue runs in order, so the second is a deadlock, not slowness.
"""
function flush_stall_report(bq::SubmitChannel{<:VulkanQueue}, target::UInt64)
    io = IOBuffer()
    ctx = ctxof(bq)
    # Runs after a stall on a possibly lost device: a failed counter read (a
    # Vulkan error only) goes into the report instead of replacing it.
    cur = try
        unwrap(VK.get_semaphore_counter_value(ctx.device, timelineof(bq)))
    catch err
        err isa VK.VulkanError || rethrow()
        err
    end
    # `cur` is the exception when the read failed, never `nothing`.
    readable = cur isa Integer
    println(io, "  timeline counter = ", readable ? cur : "unreadable ($cur)",
                ", next_timeline = ", driver(bq).next_timeline,
                ", outstanding = ", length(bq.outstanding))
    for (i, o) in enumerate(bq.outstanding)
        println(io, "  submission $i (", something(o.tag, "untagged"), "): signals ", o.token,
                    readable && o.token <= cur ? "  [already signalled]" : "")
    end
    if readable && all(o -> o.token <= cur, bq.outstanding) && target > cur
        println(io, "  >> every in-flight submission is already signalled and the target is not: ",
                    "the wait is on a value nothing will signal.")
    end
    return String(take!(io))
end

# `::Device` because `LavaDevice` is declared later, in `graph.jl`.
"""
    flush!(bq::SubmitChannel{<:VulkanQueue}, dev::Device)

Block until the newest submission on `bq` has signalled the queue's timeline,
then drain and check kernel exceptions and this context's validation messages.
Submits nothing. Waits in quanta of `FLUSH_WAIT_QUANTUM_NS`, probing for device
loss between them; throws on device loss or after `driver(bq).flush_timeout_ns`.
"""
function flush!(bq::SubmitChannel{<:VulkanQueue}, ::Device)
    ownthread(bq)
    device_lost(ctxof(bq)) && throw(LavaError("command flush", "Vulkan device lost",
        "Call reset_device!() to reinitialize, or restart Julia session."))
    target = something(newest(bq), UInt64(0))
    # Nothing in flight still drains: regions dropped since the last submission
    # are given back only at a drain.
    target == UInt64(0) && (drain!(bq); checkexceptions!(ctxof(bq)); return)
    budget = driver(bq).flush_timeout_ns
    quantum = budget == 0 ? typemax(UInt64) : min(budget, FLUSH_WAIT_QUANTUM_NS)
    waited = UInt64(0)
    while true
        wait_result = VK.wait_semaphores(vkdevice(bq),
            VK.SemaphoreWaitInfo([timelineof(bq)], [target]), quantum)
        if iserror(wait_result)
            # Adds validation messages and the dispatch log to the error.
            mark_if_lost!(bq, wait_result)
            throw_with_validation_context("vkWaitSemaphores", wait_result, bq)
        end
        unwrap(wait_result) == VK.SUCCESS && break
        waited += quantum
        # Probe for device loss: the flag is only set from another call's
        # `VkResult`, and on a lost device `vkWaitSemaphores` keeps returning
        # TIMEOUT and `vkGetSemaphoreCounterValue` a stale value. Measured on a
        # lost device, only `vkQueueWaitIdle` reported ERROR_DEVICE_LOST (fence
        # status, empty submit and zero-timeout wait did not). It runs only after
        # a quantum without progress, where blocking until idle is what `flush!`
        # wants anyway.
        let st = VK.queue_wait_idle(vkqueue(bq))
            iserror(st) && mark_if_lost!(bq, st)
        end
        if device_lost(ctxof(bq))
            throw(LavaError("vkWaitSemaphores",
                            "device was lost while waiting for timeline $target " *
                            "(counter stuck at $(query_timeline(bq)), " *
                            "$(length(bq.outstanding)) submission(s) outstanding)",
                            "The GPU faulted — a dispatch wrote out of bounds or hung. " *
                            "Raising `driver(bq).flush_timeout_ns` cannot help; call " *
                            "`reset_device!()` to reinitialize. To find the " *
                            "dispatch: `journalctl -k` for an NVIDIA Xid names the fault " *
                            "class, and " *
                            "`reset_device!(debug = DebugConfig(gpu_av = true, " *
                            "gpu_av_shaders = [\"my_kernel\"], pool_disabled = true))` " *
                            "instruments the shaders — that rebuilds the device, so every " *
                            "LavaArray alive is invalid afterwards, and `verify_gpu_av()` " *
                            "is what proves the layer actually fires. Narrow it with " *
                            "`gpu_av_shaders` first: GPU-AV can itself crash on a workload " *
                            "that kills the device (measured: it does on MatAnyone's step, " *
                            "Safe Mode included)."))
        end
        if budget != UInt64(0) && waited >= budget
            throw(LavaError("vkWaitSemaphores",
                            "timed out after $(round(waited / 1e9, digits = 1)) s waiting for " *
                            "timeline value $target on $(length(bq.outstanding)) outstanding submission(s)\n" *
                            flush_stall_report(bq, target),
                            "A dispatch is not completing. Set `ctx.diag.dispatch_logging = true` " *
                            "(and `ctx.diag.dispatch_log_file` to keep it across a restart) to see which " *
                            "kernel, or raise `driver(bq).flush_timeout_ns` if the work is genuinely this long."))
        end
    end
    drain!(bq)
    checkexceptions!(ctxof(bq))
    # This channel's context, not the bound one, so a second device does not
    # report the default device's validation messages.
    check_validation_errors!("flush!", ctxof(bq))
    return
end

"""
    stampof(buf) -> Stamp

Where core records which channel last submitted work naming `buf`, under which
token, and how many recordings hold it ([`Mantle.Stamp`](@ref)). A new resource
kind needs `stampof` only if it has device-visible bytes of its own, and a
`syncbuf!` method only if commands naming it must be ordered against other
channels; `hold!` needs neither.
"""
@inline stampof(buf::VkManagedBuffer) = buf.stamp

"""
    cmd_copy_buffer!(e, src, dst, nbytes; src_off=0, dst_off=0)

Write a device-to-device buffer copy, with no barrier of its own (a plan derives
them, a one-shot opens with the head barrier). `src` and `dst` are
`VkManagedBuffer`s, held and named for ordering, or raw `VK.Buffer` handles the
caller keeps alive.
"""
function cmd_copy_buffer!(e::Emitter, src, dst, nbytes::Integer;
                          src_off::Integer=0, dst_off::Integer=0)
    src_vkbuf = src isa VkManagedBuffer ? src.buffer : src
    dst_vkbuf = dst isa VkManagedBuffer ? dst.buffer : dst
    region = VK.BufferCopy(UInt64(src_off), UInt64(dst_off), UInt64(nbytes))
    VK.cmd_copy_buffer(e.cmd, src_vkbuf, dst_vkbuf, [region])
    # Holding also orders the copy after another channel's last use of them.
    src isa VkManagedBuffer && hold!(e, src)
    dst isa VkManagedBuffer && hold!(e, dst)
    return nothing
end

"""Throw a `LavaError` with recent validation messages and the dispatch log."""
function throw_with_validation_context(call_name::String, err_result,
        bq::Union{Nothing,SubmitChannel{<:VulkanQueue}}=nothing)
    # Re-enable dispatch logging so the next run records names: on `bq`'s
    # context, or the bound one without `bq`.
    let c = bq === nothing ? VK_CONTEXT_REF[] : ctxof(bq)
        c === nothing || (c.diag.dispatch_logging = true)
    end
    vk_err = unwrap_error(err_result)
    msgs = get_validation_messages()
    validation_detail = if isempty(msgs)
        "No validation messages captured. Install vulkan-validationlayers for GPU error diagnostics."
    else
        n = min(length(msgs), 10)
        "Last $n validation message(s):\n" * join(["  [$i] $(msgs[end-n+i])" for i in 1:n], "\n")
    end

    dlog = bq === nothing ? String[] : (ctxof(bq)).diag.dispatch_log
    dispatch_detail = if isempty(dlog)
        "No dispatches logged."
    else
        "Recent dispatch log (last $(length(dlog))):\n" *
        join(["  $d" for d in dlog], "\n")
    end

    total = bq === nothing ? 0 : (ctxof(bq)).diag.total_dispatches[]
    # Per queue, so the failure names the failing queue's kernel.
    prev_info = bq === nothing ? "" : driver(bq).prev_dispatch_info
    curr_info = bq === nothing ? "" : driver(bq).last_dispatch_info
    throw(LavaError(
        call_name,
        """$vk_err ($total dispatches total)
Crashed submission's dispatch: $prev_info
Triggered by recording: $curr_info
$validation_detail
$dispatch_detail""",
        "DEVICE_LOST usually means invalid SPIR-V, out-of-bounds BDA access, or GPU timeout (Xid 109). Check dispatch log above for the crashing kernel. Call reset_device!() to reinitialize."
    ))
end

