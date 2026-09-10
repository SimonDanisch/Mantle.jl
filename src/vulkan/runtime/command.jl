# Command buffers, closed, and their submission.
#
# A command buffer is never open when a Mantle call returns. A plan's
# `Recording` is written once at `record!` and submitted every run; everything
# else — an ad hoc launch, an upload, a readback, a run's host stores, a frame
# — is a `OneShot`, written and sealed inside one call (`oneshot`). Both reach
# the driver through `submit!(bq, closed...)`, which submits the moment it is
# called, and the `Submission` that carried them is what the queue keeps until
# the timeline says the device is done: it owns the one-shots and pins the
# recordings, and the sweep (`drain!`, over core `sweep!`) gives the first back
# and drops the second.
#
# GC safety: every launch `pin!`s its arguments, pipelines and every other GPU
# object it names through `hold!`, and core keeps the reference until the
# submission's token has passed (`graph/lifetime.jl`). A Vulkan destructor fired
# by Julia's GC records the destroy with `retire!` rather than running it, and
# core runs it at the first drain after the work that named the buffer is done.
#
# Cross-queue sync: at `submit!`, core's `crosswaits!` reads the stamp of every
# buffer these command buffers name and answers with the (channel, token) pairs
# to wait on; this file lowers a pair to a timeline semaphore and a value, and
# `stamp!` records the new owner once the submission has gone.

# Flush counter for benchmarking (atomic for thread safety)

# Global dispatch counter for debugging (total dispatches across all flushes)

# Dispatch info for debugging DEVICE_LOST

# Ring buffer of last N dispatch names for crash debugging
const MAX_DISPATCH_LOG = 2000

# Toggle dispatch logging (disabled by default for zero-alloc dispatch path).
# Enable with `ctx.diag.dispatch_logging = true` for debugging.
# On DEVICE_LOST, the error handler re-enables logging automatically.
# TEMP DEBUG: if true, submit! waits for the batch it just submitted to complete
# and records wall-clock GPU time in BATCH_WAIT_TIMES. This SERIALIZES the pipeline
# — only for measurement, not production. Used to identify batches whose actual
# GPU execution time is near the amdgpu TDR threshold (~10 s).

"""
Path to mirror the dispatch log to, or `nothing`.

The in-memory ring buffer is worthless for the failure it exists to diagnose: a
dispatch that never completes leaves the process blocked inside
`vkWaitForFences`, where nothing can print it and SIGINT does not land, so the
log dies with the session. Set this and each dispatch name is appended and
flushed as it is recorded, which makes the last line in the file the kernel that
hung.

Off by default — it is a write and a flush per dispatch, on a path whose whole
point is to allocate nothing.
"""

"""
Build a dispatch's log string, behind an `invokelatest`.

String interpolation reaches `print`/`show`, and `show` is a function every
plotting package adds methods to — so a launch path that *infers* this depends
on all of them, and loading GLMakie throws the precompiled launch code away.
Measured: `Base.println` is the root of 2 121 of the rejected verification
groups when SAM 2's image loads after VideoEditor.

It only runs when dispatch logging is on, so the dynamic call is free.
"""
@noinline dispatch_log_string(args...) = string(args...)

function log_dispatch!(bq::VulkanBatchQueue, info::String)
    d = (ctxof(bq)).diag
    d.dispatch_logging || return
    log = d.dispatch_log
    if length(log) >= MAX_DISPATCH_LOG
        popfirst!(log)
    end
    push!(log, info)
    f = d.dispatch_log_file
    f === nothing || open(io -> (println(io, info); flush(io)), f, "a")
    return
end

# Register cleanup callback for reset_device!

# Pre-allocated barrier buffer using raw VkMemoryBarrier (isbits).
# VK.jl's MemoryBarrier wrapper allocates ~1.2KB per cmd_pipeline_barrier call
# due to high-level → low-level struct conversion. Using direct ccall with the raw
# VkMemoryBarrier struct is zero-alloc. Saves ~16MB/render for 13k dispatches.
import Vulkan.VkCore: VkMemoryBarrier, VK_STRUCTURE_TYPE_MEMORY_BARRIER,
    VkAccessFlags, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
    VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
    VK_ACCESS_MEMORY_READ_BIT, VK_ACCESS_MEMORY_WRITE_BIT,
    VkPipelineStageFlags, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
    VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
    VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
    VK_PIPELINE_STAGE_TRANSFER_BIT, VkDependencyFlags
# Function pointer for vkCmdPipelineBarrier — resolved in the VkContext constructor
# Was `const CMD_PIPELINE_BARRIER_FPTR = Ref{Ptr{Nothing}}(C_NULL)`. A device
# function pointer is per device, so it lives on `VkContext` now — see the field
# there for what a global one did the first time two contexts existed.
@inline barrier_fptr(bq::VulkanBatchQueue) = (ctxof(bq)).cmd_pipeline_barrier_fptr

# Despite the `_2_` in its name, VK.jl types PIPELINE_STAGE_2_ALL_COMMANDS_BIT
# as the *sync1* `PipelineStageFlag`, while `Submission.wait_semaphores` is
# sync2-typed — so pushing the constant straight in threw a MethodError on every
# cross-queue wait. Same numeric value (0x10000); only the wrapper type differs.
const STAGE2_ALL_COMMANDS = VK.PipelineStageFlag2(VK.PIPELINE_STAGE_2_ALL_COMMANDS_BIT)

# ── Closed command buffers ───────────────────────────────────────────────────
#
# `Recording` and `OneShot` are declared in `device.jl`, ahead of the queue
# that holds them. What is here is how each is opened, sealed, submitted and
# given back — and `submit!`, the one way either reaches the driver.

"""
    recpatch!(owner, address, at)

Note that the eight bytes at `at` (a host pointer into memory the recording
owns) hold the device address `address`. Called by `pack_arg!` for every
pointer it writes. A [`Recording`](@ref) collects these so `record!` can build
the plan's patch table — the map a resource move consults; anything else emits
once and forgets, so the default is a no-op.
"""
recpatch!(owner, address::UInt64, at::Ptr{UInt8}) = nothing
recpatch!(rec::Recording, address::UInt64, at::Ptr{UInt8}) =
    (push!(rec.patches, (address, at)); nothing)

"""One primary command buffer from the queue's pool. A recording keeps its for
as long as it lives; a one-shot's stays with the one-shot, which the queue
pools for ever."""
function allocate_cmd(bq::VulkanBatchQueue)
    alloc_info = VK.CommandBufferAllocateInfo(driver(bq).cmd_pool, VK.COMMAND_BUFFER_LEVEL_PRIMARY, 1)
    return throw_if_error(bq, "vkAllocateCommandBuffers",
        VK.allocate_command_buffers(vkdevice(bq), alloc_info))[1]
end

# The two infos this file opens command buffers with, built ONCE. The
# low-level `_` struct is a boxed value (it carries a deps vector for GC
# rooting), so constructing one per open cost ~100 bytes of a `run!` that
# allocates nothing. They name no handle — only flags — so one instance per
# flag set is correct for every device and every command buffer.
const BEGIN_INFO_ONE_TIME =
    VK._CommandBufferBeginInfo(flags = VK.COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT)
const BEGIN_INFO_SIMULTANEOUS =
    VK._CommandBufferBeginInfo(flags = VK.COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT)

"""
    recording!(bq) -> Recording

Open a plan's command buffer, and begin it.

**`SIMULTANEOUS_USE`, and it is required rather than chosen.** A command buffer
without it must not be pending execution when it is submitted again, and there is
one recording per plan: nothing stops `run!` being called twice before the device
has finished the first, so two runs routinely overlap on the queue.

That was not true while a plan had an argument RING. There was a recording per
slot, and the run that would have reused a slot waited for the token covering the
run that last used it — so a given command buffer was provably idle before it was
submitted again, and no flag was correct and cheaper. Deleting the ring (see
[`GPURef`](@ref)) deleted that guarantee with it, so the flag comes back.
`ONE_TIME_SUBMIT` was never a candidate: it means "submitted once, then reset or
freed".
"""
function recording!(bq::VulkanBatchQueue)
    ownthread(bq)
    device_lost(ctxof(bq)) && throw(LavaError(
        "recording!", "Vulkan device is lost — cannot record",
        "Call reset_device!() to reinitialize, or restart Julia."))
    # The command buffer comes from the same pool a one-shot's does — core's
    # `acquire!`, which sweeps first, hands back a reset buffer and opens the
    # hold frame this recording will take at `seal!`. The `OneShot` shell around
    # it is dropped here and rebuilt by `release!`: what is pooled is the
    # command buffer, and a recording is what it is being used AS.
    o = acquire!(bq)
    throw_if_error(bq, "vkBeginCommandBuffer",
        VK._begin_command_buffer(o.cmd, BEGIN_INFO_SIMULTANEOUS))
    return Recording(bq, o.cmd, Region[], Any[], UInt64(0), true,
                     Tuple{UInt64,Ptr{UInt8}}[], nothing, VkManagedBuffer[])
end

"""Close a command buffer. Nothing more may be emitted into it, and it can be
submitted."""
function seal!(c::Closed)
    c.open || return c
    throw_if_error(queueof(c), "vkEndCommandBuffer", VK.end_command_buffer(c.cmd))
    c.open = false
    sealed!(c)
    return c
end

# What a closed buffer does with the channel's hold frame, which is the one
# place the two lifetimes differ. A one-shot is submitted once, so its holds
# belong to that submission and `submitted!` takes them. A recording is
# submitted every run and must outlive all of them, so it takes the frame
# itself and keeps it until `release!`; each submission of it then holds only
# the recording.
sealed!(::OneShot) = nothing
sealed!(rec::Recording) = (rec.holds = takeholds!(queueof(rec)); nothing)

"""
    release!(rec::Recording)

Give back everything the recording holds: its regions, its holds, its descriptor
sets and its command buffer. Running it afterwards is an error.

Explicit because the caller knows when a recording is dead and the GC does not
know it is holding device memory; `Mantle.free!` on a plan calls it.

It WAITS if the last submission of this recording has not completed, and that is
not a precondition moved onto the caller — the regions are retired rather than
released, so they cost nothing either way, but the command buffer goes on to
serve one-shots from here, and re-beginning one the device is still reading is
undefined behaviour. `CapturedSequence` dropped its buffers on the floor instead,
which left VK.jl's finalizer to free them at whatever moment the GC chose.
"""
function release!(rec::Recording)
    bq = queueof(rec)
    dev = lavadevice(ctxof(bq))
    let tok = rec.token
        tok == 0 || waitfor!(bq, tok)
    end
    rec.token = 0
    rec.open && seal!(rec)          # takes the hold frame if it is still open
    # The list itself goes back to the channel's spares, and the recording is
    # left with none — `sealed!` takes a fresh frame if it is recorded again.
    unhold!(bq, rec.holds)
    rec.holds = nothing
    empty!(rec.sync)
    let p = pool(dev)
        for r in rec.regions
            retire!(p, r)
        end
    end
    empty!(rec.regions)
    empty!(rec.sets)
    # The command buffer outlives the recording as a one-shot's: core's free
    # list is where every command buffer this channel has allocated ends up.
    finishrecording!(bq, OneShot(bq, rec.cmd, Region[], VkManagedBuffer[], false))
    return nothing
end

# ── The four recording primitives core asks of a backend — 2.3 ──────────────
#
# Core owns the free list, the reuse discipline and the moment each of these is
# called (`graph/lifetime.jl`). What is here is a `VkCommandBuffer` being
# allocated, reset and freed, and nothing else.

makerecording(bq::VulkanBatchQueue) =
    OneShot(bq, allocate_cmd(bq), Region[], VkManagedBuffer[], false)

function resetrecording!(bq::VulkanBatchQueue, o::OneShot)
    throw_if_error(bq, "vkResetCommandBuffer",
        VK.reset_command_buffer(o.cmd; flags = VK.CommandBufferResetFlag(0)))
    o.open = false
    return o
end

destroyrecording!(bq::VulkanBatchQueue, o::OneShot) =
    VK.free_command_buffers(vkdevice(bq), driver(bq).cmd_pool, [o.cmd])

"""
Give a one-shot back to core's free list, and what it borrowed back to the pool.

Called once the submission that carried it has passed, or when it never reached
the device (`release!`) — which is why the regions are RELEASED outright rather
than retired: both paths establish that nothing can be reading them, and
retiring would hold the span for one more submission boundary on top of a wait
that has already happened.

Clearing `sync` here and not in `resetrecording!` is the difference between a
pooled one-shot that holds a buffer alive until it is next used and one that
holds nothing: the list is a strong reference to every `VkManagedBuffer` the
commands named.
"""
function finishrecording!(bq::VulkanBatchQueue, o::OneShot)
    # A build that threw hands back a command buffer that was BEGUN and never
    # ended. `vkResetCommandBuffer` would take it either way, but a pooled
    # one-shot that reads as open is indistinguishable from one that is being
    # written, and every assertion about the pool has to special-case it.
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

"""
The one barrier that is never derived: everything before, against everything
after, over all memory. At the head of every one-shot and every recording,
because a closed command buffer knows nothing about what ran before it on the
queue, and the queue no longer remembers either — the arena handover that was
decided at record time from who ran last went with the open batch, since a
cross-plan hazard cannot be derived and that barrier was wrong the moment two
plans alternated. This is the honest answer, and it is one
barrier per closed buffer rather than one per launch.

Raw `vkCmdPipelineBarrier` through the context's function pointer, with an
isbits `VkMemoryBarrier`: the VK.jl wrapper allocates ~1.2 KB per call.
Counted on the context, so a test can say a launch began with it.
"""
@inline function headbarrier!(cmd::VK.CommandBuffer, ctx::VkContext)
    both = VkAccessFlags(VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT)
    barrier_ref = Ref(VkMemoryBarrier(VK_STRUCTURE_TYPE_MEMORY_BARRIER, C_NULL, both, both))
    GC.@preserve barrier_ref begin
        ccall(ctx.cmd_pipeline_barrier_fptr, Cvoid,
              (Ptr{Nothing}, VkPipelineStageFlags, VkPipelineStageFlags, VkDependencyFlags,
               UInt32, Ptr{VkMemoryBarrier}, UInt32, Ptr{Nothing}, UInt32, Ptr{Nothing}),
              cmd.vks,
              VkPipelineStageFlags(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT),
              VkPipelineStageFlags(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT), VkDependencyFlags(0),
              UInt32(1), barrier_ref,
              UInt32(0), C_NULL,
              UInt32(0), C_NULL)
    end
    Threads.atomic_add!(ctx.diag.head_barriers, 1)
    return nothing
end

"""
    oneshot(f, bq) -> OneShot
    oneshot!(f, bq; tag = :oneshot) -> token      (core, `graph/lifetime.jl`)

A command buffer written, sealed and — with the bang — submitted inside one
call:

    oneshot!(bq) do e            # e is an Emitter over a pooled OneShot
        emit_barrier!(e, …)
        emit_dispatch!(e, …)
    end                          # sealed and submitted on the way out

`f` gets the emitter and writes what it needs. Nothing it writes is visible to
any other call and nothing another call wrote is visible to it: the one-shot
opens with the global barrier [`headbarrier!`](@ref) writes, because nothing
has declared what these commands touch — that is what an ad hoc launch, an
upload or a readback IS — and everything against everything is the only sound
ordering against whatever ran before.

The form without the bang seals and hands the one-shot back unsubmitted, for
the caller who has to put it in ONE submission with something else: `run!`
puts the run's host stores and pointer patches in front of the plan's
recording that way, and `submit!(bq, closed...)` is the only way two closed
buffers ever share a submission.

Every path that used to reach for the open batch reaches for this. What it
costs is a `vkBeginCommandBuffer`/`vkEndCommandBuffer` pair and a
`vkQueueSubmit2` per call where the open buffer amortised them, measured in
`docs/submission-refactor.md` (step 7) and accepted: how many ad hoc launches
are batched efficiently is a question for a graph built to hold them, not for
the queue to answer with a buffer and a threshold.
"""
function oneshot(f, bq::VulkanBatchQueue)
    o = openoneshot(bq)
    try
        f(Emitter(o, nothing))
    catch
        # What was written goes with the buffer: given back to core with the
        # holds it took, and sealed on the way (`finishrecording!`) so it can be
        # begun again.
        release!(bq, o)
        rethrow()
    end
    seal!(o)
    return o
end

"""Open a one-shot and hand it back open, for a caller that emits into it over
several calls — a run's front, a windowed frame — and seals it itself."""
function openoneshot(bq::VulkanBatchQueue)
    ownthread(bq)
    ctx = ctxof(bq)
    device_lost(ctx) && throw(LavaError(
        "oneshot", "Vulkan device is lost — cannot record",
        "Call reset_device!() to reinitialize, or restart Julia. " *
        "All existing LavaArrays are invalid after reset and must be re-allocated."))
    # Core sweeps, drains the destroys that were waiting on those submissions,
    # and hands back a reset buffer with a hold frame open on it.
    return begin!(acquire!(bq), BEGIN_INFO_ONE_TIME)
end

# Begin a pooled command buffer and write the head barrier into it. The one
# place a `OneShot` becomes writable, so core's `recorder` and this backend's
# `openoneshot` cannot disagree about what a started recording is.
function begin!(o::OneShot, info)
    bq = queueof(o)
    throw_if_error(bq, "vkBeginCommandBuffer", VK._begin_command_buffer(o.cmd, info))
    o.open = true
    headbarrier!(o.cmd, ctxof(bq))
    return o
end

# What core's `oneshot!` records through — see `graph/lifetime.jl`.
recorder(bq::VulkanBatchQueue, o::OneShot) = Emitter(begin!(o, BEGIN_INFO_ONE_TIME), nothing)

"""
    hold!(owner::Closed, obj) -> obj
    hold!(e::Emitter, obj) -> obj

Everything a command names goes through here: the channel keeps a reference
until the submission that carries these commands has passed (core's
[`hold!`](@ref)), and a buffer is additionally recorded in `owner.sync`, which
is what the ordering between channels is derived from at submit.

This is what `pin!` was, at the same 35 sites, with the decision on the other
side of the line: what has to stay alive and until when is core's, and what is
here is the driver fact "these commands name this buffer".
"""
@inline function hold!(owner::Closed, obj)
    hold!(queueof(owner), obj)
    syncbuf!(owner, obj)
    return obj
end

"""
    syncbuf!(owner, obj)

Record that these commands name `obj`'s buffer, for `crosswaits!` and `stamp!`
at submit. A no-op for anything with no device-visible bytes of its own.

Deduplicated by an identity SCAN and not by a set, for the reason the pin list
was: the lists are short (a one-shot names a launch's worth, a plan's recording
everything it names, once), and at that size a scan beats a hash — measured, at
|list| = 4: 0.04 µs against 0.221 µs, with the crossover at 256. **If a
recording ever names thousands of buffers this should become a set.**
"""
@inline syncbuf!(::Closed, @nospecialize(obj)) = nothing
@inline function syncbuf!(owner::Closed, buf::VkManagedBuffer)
    for b in owner.sync
        b === buf && return nothing
    end
    push!(owner.sync, buf)
    return nothing
end

"""
    submit!(bq, closed...; waits = (), signals = (), fence = nothing) -> token

Hand closed command buffers to the device in one `vkQueueSubmit2`, in the
order given, and return the timeline value that submission signals.

The moment it is called. There is no list on the queue for them to join and
no threshold that decides when they go: what a caller closed is what the
driver gets, and several closed buffers go together only by being passed
together here. `run!` puts a run's stores and patches in front of the plan's
recording that way; a frame's one-shot goes with the swapchain's semaphores
and the frame's fence.

`waits` and `signals` are extra `(semaphore, value, stage)` triples beside the
cross-channel waits core derives and the timeline signal every submission
makes; `fence` is signalled when the submission completes. All three exist for
the present path and nothing else.

**It does not record the submission.** What was handed over and what has to
outlive it is the caller's to say — `submitted!(bq, token, Submission(oneshot,
takeholds!(bq)))` — because only the caller knows whether the recording it
submitted is a pooled one-shot it is giving away or a plan's, submitted again
next run. Core's `oneshot!` is the common case written once.
"""
function submit!(bq::VulkanBatchQueue, closed::Closed...;
                 waits = (), signals = (), fence::Union{Nothing,VK.Fence} = nothing)
    ownthread(bq)
    ctx = ctxof(bq)
    q = driver(bq)
    device_lost(ctx) && throw(LavaError(
        "submit!", "Vulkan device is lost — cannot submit",
        "Call reset_device!()"))
    # Sealing here rather than refusing an open buffer: submitting IS the moment
    # nothing more can be added, and `seal!` is idempotent, so the caller that
    # closed its own buffer (`oneshot`, a frame) pays nothing.
    for c in closed
        seal!(c)
    end
    Threads.atomic_add!(ctx.diag.flush_counter, 1)
    # Give back what the device has finished — the command buffers this call
    # takes from the free list, and the destroys that were waiting on them.
    # Once per SUBMISSION: per launch this was a `Dict` lookup for the device
    # plus a dynamic `passed` on an `Any` token — 171 bytes, measured by
    # `test_dispatch_allocation.jl`.
    drain!(bq)

    ncb = length(closed)
    raw_cbs = q.raw_cb_infos
    length(raw_cbs) == ncb || resize!(raw_cbs, ncb)
    for (i, c) in enumerate(closed)
        raw_cbs[i] = VK.vk.VkCommandBufferSubmitInfo(
            VK.vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO, C_NULL, c.cmd.vks, UInt32(0))
    end

    # What this submission has to wait for, derived by core from the buffers
    # these command buffers name and the stamps core wrote on them. The
    # decision is `crosswaits!`'s; what is here is the lowering of a (channel,
    # token) pair to a timeline semaphore and a value.
    cross = q.waits
    empty!(cross)
    for c in closed
        crosswaits!(bq, c.sync, cross)
    end

    # The value this submission will signal. Deliberately NOT rolled back if the
    # submit fails: nothing is stamped with it either, and giving a value back
    # would risk handing it to two submissions.
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

    # Pre-submit safety scan: catch stale-BDA-in-argument-memory corruption
    # BEFORE the GPU sees it.  Off by default (`ctx.diag.presubmit_scan = true`
    # to turn on for debugging).  Cost ~hundreds-of-µs per submit; never on by
    # default.
    if ctx.diag.presubmit_scan
        unknowns = scan_slabs_for_unknown_bdas(bq)
        if !isempty(unknowns)
            @warn "Pre-submit found $(length(unknowns)) unknown BDA(s) in argument memory"
            for u in unknowns
                @warn "  STALE: slab=$(u.slab) idx=$(u.idx) offset=$(u.offset) val=0x$(string(u.val, base=16, pad=16))"
            end
            if ctx.diag.presubmit_scan_throws
                throw(LavaError("submit!", "stale BDA in argument memory", "see warnings"))
            end
        end
    end

    # SLAB DUMP for cascade investigation: if `ctx.diag.slab_dump_target` is
    # non-zero, search the unified arena for any UInt64 == target and log offsets.
    if ctx.diag.slab_dump_target != UInt64(0)
        target = ctx.diag.slab_dump_target
        hits = Int[]
        for blk in unifiedblocks(ctx)
            mb = (blk.memory::BufferBlock).ref[]::VkManagedBuffer
            mb.mapped_ptr == Ptr{UInt8}(0) && continue
            p = Ptr{UInt64}(mb.mapped_ptr)
            for k in 0:(Int(mb.size) ÷ 8 - 1)
                unsafe_load(p, k+1) == target && push!(hits, k*8)
            end
        end
        if !isempty(hits)
            push!(ctx.diag.slab_dump_log,
                  (sub=Int(token), target=target, offsets=hits))
        end
    end

    raw_subs = q.raw_submits
    # The fptr, not the symbol: the dispatch table's answer for this device,
    # looked up per submit (a Dict{Symbol, Ptr} read — it allocates nothing).
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
        # The raw fast path gets no ResultTypes wrapper, so what `mark_if_lost!`
        # does happens directly: flip device_lost on DEVICE_LOST and throw.
        # Nothing was stamped, so nothing waits for a token that will never be
        # signalled.
        submit_result == VK.vk.VK_ERROR_DEVICE_LOST && (ctx.device_lost = true)
        throw(LavaError("vkQueueSubmit2",
            "queue submission failed with $(submit_result) " *
            "($(ncb) command buffer(s), last dispatch: $(driver(bq).last_dispatch_info))",
            "If the device is lost, call reset_device!() or restart Julia."))
    end

    # It ran, so it is now the work that last named these buffers: core writes
    # the stamp the next `crosswaits!`, the next explicit destroy and every
    # host readback read.
    for c in closed
        stamp!(bq, token, c.sync)
    end

    q.prev_dispatch_info = q.last_dispatch_info
    dg = ctx.diag
    # DEBUG: synchronous per-submission wall-clock timing. Serializes the
    # pipeline but lets us see GPU execution time per submission. Opt-in.
    if dg.batch_timing
        t0 = time()
        wr = VK.wait_semaphores(vkdevice(bq),
            VK.SemaphoreWaitInfo([timelineof(bq)], [token]),
            typemax(UInt64))
        push!(dg.batch_wait_times, time() - t0)
        push!(dg.batch_wait_info, driver(bq).last_dispatch_info)
        push!(dg.batch_wait_dispatches, ncb)
        # Don't throw from here — let the next call surface it normally — but
        # do mark device_lost so the next gate fires.
        mark_if_lost!(bq, wr)
    end
    return token
end

"""
The timeline semaphore of another channel this submission has to wait on.

The lowering half of `crosswaits!`, and the two checks that belong to it are
driver facts rather than decisions:

  * a timeline semaphore belongs to the device that created it, so waiting on
    one from a different `VkContext` hands device B a handle from device A,
    which the driver takes as a segfault inside `vkQueueSubmit2` — no Julia
    frame, nothing to grep for. That state means two contexts exist and buffers
    have been mixed between them; say so here.
  * a RELEASED channel was idle when it went, so its work has landed and its
    semaphore is gone with it. Waiting on it is both unnecessary and unsafe.
"""
@inline function crosssemaphore(bq::VulkanBatchQueue, other)
    o = other::VulkanBatchQueue{VkContext}
    ctxof(o) === ctxof(bq) || throw(LavaError(
        "submit!",
        "a buffer was last used on a channel from a DIFFERENT VkContext",
        "Two Vulkan devices are live and one buffer has been used on both. " *
        "Allocate and use the buffer under a single context."))
    return timelineof(o)
end

# ── The emitter ─────────────────────────────────────────────────────────────
#
# Where commands go, and who keeps alive what they name. Every `emit_*` takes one
# of these and never a queue, which is the whole of step 2: a queue is what
# the split and submit thresholds, the elision tracker,
# `next_skip_barrier` and `ranges_declared` were reached through, and none of them
# can fire on something that has no queue to consult. Step 5 then deleted all
# five outright, so what a queue can still decide is when to submit.

"""
    Emitter(owner, args)

`owner` is a [`Recording`](@ref) when the plan was recorded once and a
[`OneShot`](@ref) when the work belongs to this call alone (a run's stores, a
frame, the unmodelled launch path). Both answer `hold!` and `scratch!`, which
is all an emitter asks of one.

`args` is the plan's `ArgMemory`, or `nothing` off the plan path, where
arguments come from a scratch region instead.

A `base::Int` was carried here too — the byte offset of the argument SLOT the
emitter was writing, added to every address it produced. There is one copy of a
plan's arguments now, so the base is zero everywhere and the field is gone with
the ring.
"""
struct Emitter{O,C,A}
    cmd::VK.CommandBuffer
    owner::O
    ctx::C
    args::A
end

Emitter(c::Closed, am) = Emitter(c.cmd, c, ctxof(queueof(c)), am)

# The emit sites hold through the emitter — see `hold!(::Closed, obj)` above.
# Here rather than beside it because `Emitter` is declared after the closed
# types it wraps.
@inline hold!(e::Emitter, obj) = hold!(e.owner, obj)

"""The queue an emitter's commands will be submitted on."""
queueof(e::Emitter{O}) where {O<:Closed} = queueof(e.owner)

"""
    emit_dispatch!(emitter, pipeline, argaddr, groups, tlas, name = "")
    emit_dispatch_indirect!(emitter, pipeline, argaddr, indirect, tlas, name = "")

Write one compute dispatch. Bind, push the argument address, dispatch — and
nothing else.

**No barrier.** A plan already knows: `use(p, x; read/write)` says what each
pass touches, the compile derives the hazards, and `emitpass!` emits exactly
those. Between independent passes it emits nothing and the GPU pipelines them.
The unmodelled path gets the one barrier a closed buffer opens with
([`headbarrier!`](@ref)), because nothing has told it what its kernels touch.

The acceleration structure goes through [`bindtlas!`](@ref), which is a no-op
for `nothing` — so there is one body here rather than one per HWTLAS-ness, and
no `pipeline.needs_tlas_descriptor` branch. The set it binds belongs to the
emitter's owner rather than being allocated per dispatch; see [`tlasset!`](@ref).
"""
@inline function emit_dispatch!(e::Emitter, pipeline::LavaComputePipeline, argaddr::UInt64,
                        groups::NTuple{3,<:Integer}, tlas = nothing,
                        name::AbstractString = "")
    cmd = e.cmd
    VK.cmd_bind_pipeline(cmd, VK.PIPELINE_BIND_POINT_COMPUTE, pipeline.pipeline)
    hold!(e, pipeline)
    bindtlas!(e, pipeline, tlas)
    push_constants_bda!(cmd, pipeline.pipeline_layout, VK.SHADER_STAGE_COMPUTE_BIT, argaddr)
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
    push_constants_bda!(cmd, pipeline.pipeline_layout, VK.SHADER_STAGE_COMPUTE_BIT, argaddr)
    mb = indirect.buf[]::VkManagedBuffer
    ts = maybe_write_dispatch_start_timestamp!(e.ctx, cmd, name)
    VK.cmd_dispatch_indirect(cmd, mb.buffer, UInt64(indirect.offset))
    maybe_write_dispatch_end_timestamp!(e.ctx, cmd, ts, e.ctx.cmd_pipeline_barrier_fptr)
    # The workgroup counts are READ BY THE DEVICE from this buffer, so it has to
    # outlive the dispatch exactly as an argument does.
    hold!(e, indirect)
    emitted!(e, name)
    return nothing
end

"""
    bindtlas!(emitter, pipeline, tlas)

Bind an acceleration structure at set 0, binding 0, and hold everything a ray
query walks into. A no-op without one — the `VulkanTLAS` method is in
`raytracing/hwtlas.jl`, where the type exists.
"""
@inline bindtlas!(::Emitter, ::LavaComputePipeline, ::Nothing) = nothing

"""
Note that a dispatch was written, for the log a DEVICE_LOST error prints.

At EMIT time, which for a recorded plan is once rather than once per run. That is
the honest place for it: the log answers "which kernel is in the command buffer
that died", and the command buffer is built here.
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
How long `flush!` waits for the queue to drain before it gives up, in
nanoseconds. `0` restores the old behaviour of waiting forever.

`vkWaitSemaphores` took `typemax(UInt64)` here, which is not a timeout — a
dispatch that never completes turned into a process that could only be killed,
taking the in-memory dispatch log with it. That is the failure this wait exists
to survive, so it gets a budget: long enough that no legitimate submission comes
near it (a whole VAE decode is ~30 s of device work), short enough that a hang
is a diagnosable error instead of a wedged session.
"""

"""Poll interval inside that budget, so a TDR is noticed without waiting it out."""
const FLUSH_WAIT_QUANTUM_NS = UInt64(2) * 1_000_000_000



"""
    push_constants!(cmd, layout, stage_flags, push_data)

Record push constant update. No-op if push_data is empty.
"""
function push_constants!(cmd::VK.CommandBuffer, layout::VK.PipelineLayout,
                          stage_flags, push_data::Vector{UInt8})
    isempty(push_data) && return
    GC.@preserve push_data begin
        VK.cmd_push_constants(cmd, layout, stage_flags,
            UInt32(0), UInt32(length(push_data)), Ptr{Nothing}(pointer(push_data)))
    end
end

"""
    push_constants_bda!(cmd, layout, stage_flags, bda)

Record an 8-byte BDA push constant update. Uses a stack-allocated Ref
(concurrency-safe and still zero-heap-alloc since Refs of primitives are
stack-promoted by the Julia compiler).
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
What to log and to label a timestamp with, or `""` when nobody is looking.

`suffix...` and not one built string: the caller's pieces have to stay UNBUILT
past the gate, or a dispatch pays for a name nothing reads. That is 835 bytes per
dispatch when the pieces are `" g="` and a tuple — measured, and
`test_dispatch_allocation.jl` is the test that measures it.
"""
@inline function dispatchinfo(bq::VulkanBatchQueue, suffix...)
    d = (ctxof(bq)).diag
    (d.dispatch_logging || d.dispatch_timing) || return ""
    return Base.invokelatest(dispatch_log_string, driver(bq).last_dispatch_info, suffix...)::String
end

"""
Between an unmodelled prepare and the indirect launch that reads what it wrote,
inside one one-shot: the prepare's shader write, made visible to the command
processor's indirect read and to the launch's own reads. A plan's fused prepare
gets this from `emitprepares!`; the ad hoc paths (`ka_launch_indirect!`,
`trace_rays_indirect!`) write both commands into one closed buffer and need it
between them. `dst_stage` and `extra_dst_access` name the consumer — a compute
dispatch or a ray-tracing launch.
"""
@inline function indirectbarrier!(e::Emitter, dst_stage::VK.PipelineStageFlag,
                                  extra_dst_access::VK.AccessFlag)
    dst_access = VkAccessFlags(VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT) |
                 VkAccessFlags(extra_dst_access)
    barrier_ref = Ref(VkMemoryBarrier(VK_STRUCTURE_TYPE_MEMORY_BARRIER, C_NULL,
                                      VkAccessFlags(VK_ACCESS_SHADER_WRITE_BIT), dst_access))
    GC.@preserve barrier_ref begin
        ccall(e.ctx.cmd_pipeline_barrier_fptr, Cvoid,
              (Ptr{Nothing}, VkPipelineStageFlags, VkPipelineStageFlags, VkDependencyFlags,
               UInt32, Ptr{VkMemoryBarrier}, UInt32, Ptr{Nothing}, UInt32, Ptr{Nothing}),
              e.cmd.vks,
              VkPipelineStageFlags(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT),
              VkPipelineStageFlags(dst_stage), VkDependencyFlags(0),
              UInt32(1), barrier_ref,
              UInt32(0), C_NULL,
              UInt32(0), C_NULL)
    end
    return nothing
end

# ── Flush ──

"""
    query_timeline(bq::VulkanBatchQueue) -> UInt64

Return the current counter of `timelineof(bq)`.  Replaces the old
`try ... catch; typemax(UInt64); end` sentinel:

  * On success, return the real counter value.
  * On any `VulkanError`, set `ctxof(bq).device_lost` (if the code is
    `ERROR_DEVICE_LOST`) and throw `LavaVulkanError`.  No silent fallback —
    every caller must either have cleared the device-lost flag upstream
    (so the happy path runs) or be prepared to propagate the throw.

Callers are expected to gate on `device_lost(ctxof(bq))` themselves before
calling this.  The only way the throw path fires is the race window where
the device died between that upstream check and this query — and in that
case, loud is correct: we want the dispatcher / finalizer log to show it.
"""
# Raw and refilled, because this runs in every sweep: the checked wrapper
# allocates a result box and a counter cell per call (~80 bytes), and a `run!`
# allocates nothing. Errors take the cold path — the same query through the
# checked wrapper, which fails the same way and throws with full context.
@inline function query_timeline(bq::VulkanBatchQueue)::UInt64
    # The shared cell is the owning thread's. A finalizer asks the same question
    # from the GC thread (see `deferred_as_frees`), and two threads writing one
    # RefValue would hand each other torn or wrong counter values — a monotone
    # counter read early is a premature "passed", which is a use-after-free.
    # Off the owning thread the answer costs one fresh Ref, which is exactly
    # the right price for how rare that call is.
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

Block until this queue's timeline reaches `val`. The one-semaphore case of
`wait_semaphores!`, raw and refilled out of the queue's slots: a fresh
`SemaphoreWaitInfo` and its two vectors were ~300 bytes of every `waitfor!`.
Errors fall through to the checked wrapper, which throws with context.
"""
function wait_timeline!(bq::VulkanBatchQueue, val::UInt64)
    # Same discipline as `query_timeline`: the slots are the owning thread's.
    if Threads.threadid() != bq.thread
        # Not a path with a fast lane: a caller off the owning thread gets the
        # checked wrapper, fresh info and all.
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

Block until THIS queue's timeline reaches `token`. The device-level
`waitfor!(dev, tok)` asks the device's primary queue, which is right for a
plan's run and wrong for a one-shot submitted on any other queue — a download
of a buffer last written on a second queue, an acceleration-structure build
on a queue the caller chose. A token is a value of one queue's timeline, and
this waits on that queue.
"""
function waitfor!(bq::VulkanBatchQueue, tok::UInt64)
    passed(bq, tok) && return nothing
    wait_timeline!(bq, tok)
    return nothing
end

# ── Vulkan call helpers: single source of truth for device-lost handling ──
#
# Every Vulkan call that can fail should funnel through `throw_if_error` (or
# its non-throwing sibling `mark_if_lost!` for callers with custom recovery).
# This is the SOLE place the "VkResult → device_lost flag" rule lives, so
# the gate at `oneshot` reliably fires after any
# DEVICE_LOST regardless of which low-level call surfaced the error.

"""
    mark_if_lost!(bq::VulkanBatchQueue, result)
    mark_if_lost!(ctx::VkContext, result)

Mark `ctx.device_lost = true` iff `result` is a Vulkan `ERROR_DEVICE_LOST`.
Does not unwrap, does not throw.  Use this when a caller needs to do its
own recovery before throwing (`submit!`, `flush!`).  Otherwise prefer
`throw_if_error` which handles the whole pattern.
"""
@inline function mark_if_lost!(ctx::VkContext, result)
    iserror(result) || return
    e = unwrap_error(result)::VK.VulkanError
    e.code == VK.ERROR_DEVICE_LOST && mark_device_lost!(ctx)
    return
end
@inline mark_if_lost!(bq::VulkanBatchQueue, result) = mark_if_lost!(ctxof(bq), result)

device_lost_hint(call) =
    "Vulkan device is lost during $(call). " *
    "Call reset_device!() to reinitialize, or restart Julia."

"""
    throw_if_error(bq::VulkanBatchQueue, result)
    throw_if_error(ctx::VkContext, result)
    throw_if_error(ctx_or_bq, call::String, result)       # adds a call name

Canonical "do everything" Vulkan-call wrapper.
On success → returns the unwrapped value.
On error   → marks `ctx.device_lost` (if the error is `DEVICE_LOST`) and
             throws `LavaVulkanError` with a hint.

Replace any `unwrap(VK.foo(...))` with `throw_if_error(ctx, VK.foo(...))`
to get the device-lost handling for free.  The wrappers below
(`queue_submit!`, `wait_for_fences!`, ...) are thin spellings of this.
"""
@inline function throw_if_error(ctx::VkContext, call::String, result)
    iserror(result) || return unwrap(result)
    mark_if_lost!(ctx, result)
    e = unwrap_error(result)::VK.VulkanError
    suggestion = e.code == VK.ERROR_DEVICE_LOST ? device_lost_hint(call) : ""
    throw(LavaVulkanError(call, Int32(e.code), e.msg, suggestion))
end
@inline throw_if_error(ctx::VkContext, result) = throw_if_error(ctx, "Vulkan call", result)
@inline throw_if_error(bq::VulkanBatchQueue, args...) = throw_if_error(ctxof(bq), args...)

"""
    queue_submit!(bq, submits; fence=VK.Fence(C_NULL))

`vkQueueSubmit` wrapper.  Auto-marks device_lost on failure and throws
`LavaVulkanError`.
"""
@inline queue_submit!(bq::VulkanBatchQueue, submits::AbstractVector{VK.SubmitInfo};
                      fence=VK.Fence(C_NULL)) =
    throw_if_error(bq, "vkQueueSubmit", VK.queue_submit(vkqueue(bq), submits; fence=fence))

"""
    queue_submit_2!(bq, submits; fence=VK.Fence(C_NULL))

`vkQueueSubmit2` wrapper.  Auto-marks device_lost on failure.
"""
@inline queue_submit_2!(bq::VulkanBatchQueue, submits::AbstractVector{VK.SubmitInfo2};
                        fence=VK.Fence(C_NULL)) =
    throw_if_error(bq, "vkQueueSubmit2", VK.queue_submit_2(vkqueue(bq), submits; fence=fence))

"""
    wait_for_fences!(bq, fences; wait_all=true, timeout=typemax(UInt64))

`vkWaitForFences` wrapper.  Auto-marks device_lost on failure.
"""
@inline wait_for_fences!(bq::VulkanBatchQueue, fences;
                         wait_all::Bool=true, timeout::UInt64=typemax(UInt64)) =
    throw_if_error(bq, "vkWaitForFences", VK.wait_for_fences(vkdevice(bq), fences, wait_all, timeout))

"""
    wait_semaphores!(bq, info; timeout=typemax(UInt64))

`vkWaitSemaphores` wrapper.  Auto-marks device_lost on failure.
"""
@inline wait_semaphores!(bq::VulkanBatchQueue, info::VK.SemaphoreWaitInfo;
                         timeout::UInt64=typemax(UInt64)) =
    throw_if_error(bq, "vkWaitSemaphores", VK.wait_semaphores(vkdevice(bq), info, timeout))

"""
    flush!(bq::VulkanBatchQueue, dev::Device)

Block until every submission on `bq` has been signalled on the queue's
timeline semaphore.  Uses a single `wait_semaphores` call on the HIGHEST
signal value — the timeline is monotonic, so once that value is reached every
lower value is too. Nothing is submitted here: nothing recorded is ever
unsubmitted.
"""
# What the queue looked like when `flush!` gave up, as text for the error.
#
# A bare "timed out waiting for timeline value N" does not distinguish the two
# things that produce it, and they want opposite responses:
#
#   * a kernel that is genuinely slow or wedged — the counter sits one below
#     `target`, the batch that signals it is in flight, and the dispatch log
#     names the shader;
#   * a wait on a value nothing will signal — every in-flight batch has already
#     been signalled (`signal_value <= counter`) yet `target` is higher, or a
#     batch waits on a value above the counter that no queued batch signals. A
#     queue runs its submissions in order, so a batch blocked on such a wait
#     also blocks the batch that would have signalled past it: a deadlock, not
#     slowness.
#
# The second is what the intermittent decode-loop hang looks like from outside
# (4 in-flight batches, a target that never arrives), and it could not be told
# from the first without this. Cheap: three queries, only on the failure path.
#
# A comment, not a docstring: `flush!`'s own docstring follows immediately, and
# two adjacent strings make `@doc` document the second one.
function flush_stall_report(bq::VulkanBatchQueue, target::UInt64)
    io = IOBuffer()
    ctx = ctxof(bq)
    # The one place a swallow is right, and it is narrowed to say why: this
    # builds the diagnostic printed when a flush has ALREADY stalled, and the
    # device may be lost. Failing to read the counter must not replace the report
    # the caller is waiting for — but only a Vulkan error is tolerated, and the
    # reason is printed rather than left blank.
    cur = try
        unwrap(VK.get_semaphore_counter_value(ctx.device, timelineof(bq)))
    catch err
        err isa VK.VulkanError || rethrow()
        err
    end
    println(io, "  timeline counter = ", cur isa Exception ? "unreadable ($cur)" : cur,
                ", next_timeline = ", driver(bq).next_timeline,
                ", outstanding = ", length(bq.outstanding))
    for (i, o) in enumerate(bq.outstanding)
        b = o.payload
        waits = [v for (_, v, _) in b.wait_semaphores]
        done = cur !== nothing && b.signal_value <= cur
        println(io, "  submission $i: signals ", b.signal_value,
                    ", waits on ", isempty(waits) ? "nothing" : string(waits),
                    done ? "  [already signalled]" : "")
    end
    if cur !== nothing && all(o -> o.token <= cur, bq.outstanding) && target > cur
        println(io, "  >> every in-flight submission is already signalled and the target is not: ",
                    "the wait is on a value nothing will signal.")
    end
    return String(take!(io))
end

# `::Device`, core's abstract type, and not `LavaDevice`: this file is included
# before `graph.jl` declares that one. What matters is the ARGUMENT — the
# portable `flush!(channel, device)` takes the Mantle device, which is what
# `deviceof` answers and what core's one-argument form fills in. It used to take
# a raw `VK.Device`, so the portable spelling could not reach it.
function flush!(bq::VulkanBatchQueue, ::Device)
    @assert Threads.threadid() == bq.thread  "VulkanBatchQueue is single-writer; cross-thread flush forbidden"
    # One question, one list. This used to seed `target` from `replay_watermark`
    # and then fold a maximum over `in_flight`, because the two submission paths
    # kept separate records; a caller that forgot either returned early.
    target = something(newest(bq), UInt64(0))
    target == UInt64(0) && return
    budget = driver(bq).flush_timeout_ns
    quantum = budget == 0 ? typemax(UInt64) : min(budget, FLUSH_WAIT_QUANTUM_NS)
    waited = UInt64(0)
    while true
        wait_result = VK.wait_semaphores(vkdevice(bq),
            VK.SemaphoreWaitInfo([timelineof(bq)], [target]), quantum)
        if iserror(wait_result)
            # Rich rethrow: flush! gets validation context + dispatcher hints the
            # generic wrapper can't.  The mark_if_dl! call is the single source of
            # truth for the device_lost flag.
            mark_if_lost!(bq, wait_result)
            throw_with_validation_context("vkWaitSemaphores", wait_result, bq)
        end
        unwrap(wait_result) == VK.SUCCESS && break
        waited += quantum
        # A TDR marks the device lost while this call is still waiting on a value
        # that will now never be signalled, so ask between quanta rather than
        # sitting in one unbounded wait.
        #
        # ASK THE DRIVER, don't just read the flag. `device_lost(ctx)` is only
        # ever set by `mark_if_lost!` from some OTHER call's `VkResult`, and
        # during this wait there is no other call — so on a dead device the flag
        # stays false forever. Worse, the two things we could ask disagree:
        # `vkWaitSemaphores` returns TIMEOUT (not an error) and
        # `vkGetSemaphoreCounterValue` returns a perfectly good stale counter,
        # while `vkQueueWaitIdle` and `vkGetFenceStatus` report DEVICE_LOST.
        # Lava polled the two that lie, so a GPU fault presented as a 120 s stall
        # and the error told the user to RAISE the timeout — the one action that
        # cannot help. Measured on MatAnyone: queue dead, counter happily
        # answering 355, `vkQueueWaitIdle` -> ERROR_DEVICE_LOST.
        #
        # `vkQueueWaitIdle` is the probe, and it has to be: MEASURED on this
        # driver with the device already dead, only one of five reports it —
        #
        #     get_fence_status(the AS fence)  silent
        #     get_semaphore_counter_value     silent (returns a stale 355!)
        #     empty queue_submit_2            silent
        #     wait_semaphores(0 timeout)      silent
        #     queue_wait_idle                 ERROR_DEVICE_LOST, 2.5 s
        #
        # Which calls surface a lost device is a per-driver fact, not a spec
        # guarantee, so it is measured rather than reasoned about.
        #
        # It blocks, but only in a situation that is already abnormal — the first
        # quantum has passed without the target being reached — and on a HEALTHY
        # device blocking until the queue drains is exactly what `flush!` wants.
        # So this costs nothing in the common case, where the wait succeeds on
        # the first quantum and this line is never reached.
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
    check_validation_errors!("vk_flush!")
    return
end

# ── What a command names, and what core does with it ─────────────────────────
#
# One verb at the emit site — `hold!(owner, obj)`, declared above beside the
# one-shot it writes into. It says the same thing `pin!` said and decides
# nothing: core keeps the reference until the submission has passed
# (`graph/lifetime.jl`), and `syncbuf!` records the buffer so `crosswaits!` and
# `stamp!` can order this submission against the other channels at `submit!`.
#
# `sync_access!` was the other half and is gone: it read `buf.last_write_bq`,
# decided whether a wait was needed, pushed the semaphore itself and wrote the
# new stamp — four decisions about submissions, in a backend, from two fields on
# a buffer. Core makes all four now; what is left here is `stampof`, which says
# where the answer is stored, and `crosssemaphore`, which lowers it.
#
# A new resource kind needs `stampof(::MyRes)` only if it has device-visible
# bytes of its own, and `syncbuf!(owner, ::MyRes)` only if a command that names
# it has to be ordered against another channel. Neither is needed to be held.

"""
    stampof(buf) -> Stamp

Where core writes what it knows about this buffer: which channel last submitted
work naming it, under which token, and how many recordings that can still submit
hold it. See [`Mantle.Stamp`](@ref).

The three fields it replaces were `last_write_bq`, `last_write_val` and
`@atomic pins`, and each was read by backend code that decided something with
it.
"""
@inline stampof(buf::VkManagedBuffer) = buf.stamp

"""
    vk_flush!(bq::VulkanBatchQueue)
    vk_flush!(ctx::VkContext)       # flushes ctx.default_bq

Flush a specific batch queue.  Always spell the queue (or its ctx) explicitly —
zero-arg convenience forms have been removed per the "Explicit arguments over
implicit state" rule.
"""
function vk_flush!(bq::VulkanBatchQueue)
    ctx = ctxof(bq)
    device_lost(ctx) && throw(LavaError("command flush", "Vulkan device lost",
        "Call reset_device!() to reinitialize, or restart Julia session."))
    flush!(bq)
    return
end
vk_flush!(ctx::VkContext) = vk_flush!(ctx.default_bq)

# ── Buffer copy ──────────────────────────────────────────────────────────
#
# One `cmd_copy_buffer` into whatever the emitter is writing, with both sides
# pinned. No barrier of its own: inside a plan the barriers around a copy pass
# are derived, and a one-shot opens with the global barrier that orders it
# against everything before it — the shader-write → transfer-read barrier that
# used to guess here from the batch's dispatch count, and the transfer-write →
# everything barrier after the copy, were both the open batch's problem.

"""
    cmd_copy_buffer!(e, src, dst, nbytes; src_off=0, dst_off=0)

Write a GPU→GPU buffer copy.  `src` and `dst` may be either `VkManagedBuffer`
(held, and named for ordering) or raw `VK.Buffer` handles, which the caller
must keep alive itself.
"""
function cmd_copy_buffer!(e::Emitter, src, dst, nbytes::Integer;
                          src_off::Integer=0, dst_off::Integer=0)
    src_vkbuf = src isa VkManagedBuffer ? src.buffer : src
    dst_vkbuf = dst isa VkManagedBuffer ? dst.buffer : dst
    region = VK.BufferCopy(UInt64(src_off), UInt64(dst_off), UInt64(nbytes))
    VK.cmd_copy_buffer(e.cmd, src_vkbuf, dst_vkbuf, [region])
    # Hold both sides: the submission must outlive them, and naming them is what
    # makes the transfer wait for whatever other channel wrote them last.
    src isa VkManagedBuffer && hold!(e, src)
    dst isa VkManagedBuffer && hold!(e, dst)
    return nothing
end

# `signalof(e)` is gone with the recycler. It answered "the timeline value
# covering what this emitter is writing", which only a batch could answer, and
# its one caller was the rename route handing an outgoing store back against it.

# ── Error Reporting ──

"""Throw a LavaError enriched with recent validation layer messages and dispatch log."""
function throw_with_validation_context(call_name::String, err_result,
        bq::Union{Nothing,VulkanBatchQueue}=nothing)
    # Re-enable dispatch logging so the next run captures debug info. `bq` when
    # the caller has one — all three currently do — and the current context
    # otherwise, since this is also reachable from a raw `VkResult` check.
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
    # Which kernel, on the queue that failed. Read process-wide these named
    # whatever dispatched last anywhere, so a two-queue session could attribute
    # one queue's DEVICE_LOST to another queue's kernel.
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

# ── Debugging API ──

"""
    set_dispatch_logging!(enabled::Bool)

Enable or disable dispatch name logging. When enabled, each dispatch records
its kernel name and parameters for crash debugging. Disabled by default for
zero-alloc performance. Auto-enabled on DEVICE_LOST.
"""
set_dispatch_logging!(enabled::Bool, ctx::VkContext = vk_context()) =
    (ctx.diag.dispatch_logging = enabled)

"""
    get_dispatch_log() -> Vector{String}

Return a copy of the recent dispatch log (up to $MAX_DISPATCH_LOG entries).
"""
get_dispatch_log(ctx::VkContext = vk_context()) = copy(ctx.diag.dispatch_log)
