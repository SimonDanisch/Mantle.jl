# Command buffer recording and dispatch for Lava.jl
#
# Batch-based command buffer management: each CommandBatch owns its own
# command buffer and strong refs (via `batch.pinned`) to every GPU object
# the dispatch references.  Retirement is synchronous: `sweep_retired_batches!`
# polls each queue's timeline semaphore and recycles batches whose signal
# value has been reached.
#
# GC safety: every dispatch `pin!`s its arg buffers + pipelines + other
# GPU-referenced objects into `batch.pinned`.  Objects remain reachable until
# the timeline semaphore reaches `batch.signal_value`, at which point
# `reclaim_batch!` empties `batch.pinned` and returns the batch to the free
# pool.  Vulkan destructors fired by Julia's GC (via DataRef refcount) hit
# the deferred-free path in `memory.jl`, which itself waits on the same
# timeline before destroying the underlying `VkBuffer`.
#
# Cross-queue sync: at `submit!` time, `sync_access!(batch, buf)` walks every
# pinned object and, for `VkManagedBuffer`s, inserts a timeline-semaphore wait
# on the prior writing queue (if any) and updates `buf.last_write` to this
# batch's (bq, signal_value).

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

function log_dispatch!(bq::BatchQueue, info::String)
    d = (bq.ctx::VkContext).diag
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

# Register cleanup callback for vk_reset_device!

# Pre-allocated barrier buffer using raw VkMemoryBarrier (isbits).
# VK.jl's MemoryBarrier wrapper allocates ~1.2KB per cmd_pipeline_barrier call
# due to high-level → low-level struct conversion. Using direct ccall with the raw
# VkMemoryBarrier struct is zero-alloc. Saves ~16MB/render for 13k dispatches.
import Vulkan.VkCore: VkMemoryBarrier, VK_STRUCTURE_TYPE_MEMORY_BARRIER,
    VkAccessFlags, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
    VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
    VkPipelineStageFlags, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
    VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
    VK_PIPELINE_STAGE_TRANSFER_BIT, VkDependencyFlags
# Function pointer for vkCmdPipelineBarrier — resolved in the VkContext constructor
# Was `const CMD_PIPELINE_BARRIER_FPTR = Ref{Ptr{Nothing}}(C_NULL)`. A device
# function pointer is per device, so it lives on `VkContext` now — see the field
# there for what a global one did the first time two contexts existed.
@inline barrier_fptr(bq::BatchQueue) = (bq.ctx::VkContext).cmd_pipeline_barrier_fptr

# Despite the `_2_` in its name, VK.jl types PIPELINE_STAGE_2_ALL_COMMANDS_BIT
# as the *sync1* `PipelineStageFlag`, while `CommandBatch.wait_semaphores` is
# sync2-typed — so pushing the constant straight in threw a MethodError on every
# cross-queue wait. Same numeric value (0x10000); only the wrapper type differs.
const STAGE2_ALL_COMMANDS = VK.PipelineStageFlag2(VK.PIPELINE_STAGE_2_ALL_COMMANDS_BIT)

# CB split threshold: seal the current command buffer and start a new one after
# this many dispatches per segment. All segments are submitted in a single
# vkQueueSubmit. This avoids NVIDIA driver crashes from enormous command buffers
# (30k+ dispatches) while maintaining single-submit efficiency.
# Default 3000 ≈ 1 Hikari volpath sample (50 bounces × 60 dispatches/bounce).
# Set to 0 to disable splitting.

# Auto-submit threshold: submit the current batch (starting a new one on the
# next dispatch) when its dispatch count reaches this value. Measured per-batch
# GPU execution time is capped at ~51 ms on dolphin HQ (well under amdgpu's
# ~10 s TDR) so the original TDR hypothesis this was meant to address turned
# out to be wrong.
#
# It is worth a great deal for a different reason. With this disabled, nothing
# reaches the queue until `flush!`, so recording the host side of a workload and
# executing it on the GPU are strictly serial: the card sits idle for the whole
# recording pass and the host sits idle for the whole execution pass. On the
# MatAnyone inference step (~2050 dispatches, 16.3 ms to record, ~12 ms to
# execute) that is 28.1 ms a step; submitting every 64 dispatches lets the two
# overlap and the same step takes 19.3 ms — 35.6 to 51.9 steps/s for a one-line
# change, with the GPU now starting its first work ~64 dispatches in instead of
# 2050.
#
# 64 is a floor, not a peak: the curve is flat from 16 to 96 (19.3-20.8 ms) and
# degrades outside it — below 16 the per-submit cost (fence, command buffer,
# retire sweep) starts to show, above 128 the overlap window shrinks back.
# Workloads with fewer than 64 dispatches per flush never reach it and are
# unaffected.


# ── Capture / replay ──
#
# A workload whose launch sequence is identical every iteration pays to rebuild
# that sequence every iteration. On the MatAnyone inference step that is 12.1 ms
# of host time against 11.6 ms of GPU time — recording the step costs slightly
# more than running it. Capturing the command buffers once and re-submitting
# them removes the recording entirely.
#
# The precondition is that every device address the recorded commands refer to
# is the same next time: a statically planned slab (DNNKernels' `planslab`), fixed
# weights, and an input buffer written in place rather than reallocated. Command
# buffers recorded under `capture` therefore use SIMULTANEOUS_USE rather than
# ONE_TIME_SUBMIT, and ownership of them moves to the `CapturedSequence` so the
# batch pool can never re-record over one.
#
# Arg buffers are the subtle part. `pack_args_direct!` writes each dispatch's
# arguments into a bump-allocated slab and bakes that slab's address into the
# command buffer as a push constant, so a replay reads whatever those bytes hold
# *now*. Nothing rewrites them as long as no other recording happens on this
# queue between replays — which is why `capture` reserves the slab range it used
# instead of letting `reset_arg_buffer_pool!` hand it out again.

mutable struct CapturedSequence
    bq::BatchQueue
    cmd_bufs::Vector{VK.CommandBuffer}
    pinned::Vector{Any}          # keeps every referenced GPU object alive
    submissions::Int             # submit! boundaries folded into one replay
    # The argument memory this capture's dispatches point at, OWNED here.
    #
    # It used to come from the queue's shared bump allocator, with
    # `reserve_arg_slabs!` pushing a high-water mark past it so a later recording
    # could not overwrite the bytes a replay reads. That mark only ever went up:
    # the slabs were never handed back, by `release!`, by dropping the sequence,
    # or by a full GC. Capturing was a permanent ~4 MB, which is invisible when a
    # model is baked once for the life of a process and a leak when a renderer
    # rebuilds its plans on every scene edit.
    #
    # Owning them instead makes the lifetime ordinary: nothing else can allocate
    # from these, so nothing can overwrite them, and when the sequence goes they
    # go — promptly via [`release!`](@ref), or through the array finalizers if it
    # is simply dropped.
    slabs::Vector{Any}
    slab_idx::Int
    slab_offset::Int
end
CapturedSequence(bq, cmd_bufs, pinned, submissions) =
    CapturedSequence(bq, cmd_bufs, pinned, submissions, Any[], 1, 0)

"""
    release!(seq::CapturedSequence)

Give back everything a capture holds: its argument slabs, its pinned set and its
command buffers. The sequence is empty afterwards and replaying it does nothing.

Explicit because the caller knows when the recording is dead and the GC does not
know it is holding device memory. Dropping the sequence works too — the slabs are
ordinary arrays with ordinary finalizers — this just makes it prompt, and is what
`Mantle.free!` on a baked plan calls.

The device has to be past the last replay, which is the same precondition every
other `free!` here carries.
"""
function release!(seq::CapturedSequence)
    empty!(seq.cmd_bufs)
    empty!(seq.pinned)
    for s in seq.slabs
        finalize(s)
    end
    empty!(seq.slabs)
    seq.slab_idx = 1
    seq.slab_offset = 0
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
One-shot request to drop the barrier in front of the very next dispatch.

Set by the launch path when it has proved the next dispatch touches no memory
that anything since the last barrier touched (`ka_backend.jl`), and consumed by
`record_dispatch!`. A plain `Ref` is the right shape here for the same reason
`CONCURRENT_GROUP_ACTIVE` is: a `BatchQueue` is single-writer by construction.
"""

"""Begin-flags for a command buffer: reusable while capturing, one-shot otherwise."""
@inline cb_begin_flags(bq::BatchQueue) = bq.capturing === nothing ?
    VK.COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT :
    VK.COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT

"""
    capture(f, bq) -> CapturedSequence

Run `f` once, recording and executing it normally, and keep its command buffers
for `replay!`. `f` must not allocate device memory whose address it then
dispatches against, or the replay will point at freed storage.
"""
function capture(f, bq::BatchQueue)
    bq.capturing === nothing || throw(LavaError("capture", "already capturing", "nested capture is not supported"))
    flush!(bq, bq.device)                       # start from a drained queue
    seq = CapturedSequence(bq, VK.CommandBuffer[], Any[], 0)
    bq.capturing = seq
    try
        f()
        submit!(bq)                             # close and collect the trailing batch
    finally
        bq.capturing = nothing
    end
    flush!(bq, bq.device)
    seq
end

"""
    replay!(seq)

Re-submit a captured sequence: one `vkQueueSubmit2` for the whole thing, no
recording. Serialised against the previous replay, since an inference step reads
what the last one wrote.
"""
function replay!(seq::CapturedSequence)
    bq = seq.bq
    @assert Threads.threadid() == bq.owning_thread  "BatchQueue is single-writer; cross-thread replay forbidden"
    isempty(seq.cmd_bufs) && return
    device_lost(bq.ctx::VkContext) && throw(LavaError(
        "replay!", "Vulkan device is lost — cannot replay", "Call vk_reset_device!()"))
    # Close any batch still being recorded, FIRST. `ensure_active_batch!` hands a
    # new batch `bq.next_timeline + 1` as its signal value and `submit!` asserts
    # that reservation still holds — so bumping the shared counter here while a
    # batch is open makes it stale and the next `submit!` dies with
    # "batch signal desync: N vs N+1". It costs nothing in the intended usage
    # (a replay after a drained queue) and it is the only thing that makes
    # replaying *interleaved* with ordinary recording legal, which is exactly
    # what a decoder replaying per click inside a live editor does.
    bq.active_batch !== nothing && bq.active_batch.recording && submit!(bq)
    bq.next_timeline += 1
    v = bq.next_timeline
    prev = bq.replay_watermark
    cb_infos = [VK.CommandBufferSubmitInfo(cb, UInt32(0)) for cb in seq.cmd_bufs]
    waits = prev == UInt64(0) ? VK.SemaphoreSubmitInfo[] :
        [VK.SemaphoreSubmitInfo(bq.timeline_sem, prev, UInt32(0);
                                    stage_mask=VK.PIPELINE_STAGE_2_ALL_COMMANDS_BIT)]
    signal = VK.SemaphoreSubmitInfo(bq.timeline_sem, v, UInt32(0);
                                        stage_mask=VK.PIPELINE_STAGE_2_ALL_COMMANDS_BIT)
    # Called the same way `submit!` does: the `queue_submit_2!` helper's default
    # `fence=VK.Fence(C_NULL)` evaluates `create_fence` on a null device.
    res = VK.queue_submit_2(bq.queue, [VK.SubmitInfo2(waits, cb_infos, [signal])])
    if iserror(res)
        mark_if_lost!(bq, res)
        bq.next_timeline -= 1
        throw_with_validation_context("vkQueueSubmit2 (replay)", res, length(seq.cmd_bufs), false, bq)
    end
    bq.replay_watermark = v
    return
end

# ── Batch lifecycle ──

"""
    ensure_active_batch!(bq::BatchQueue) -> CommandBatch

Get the active batch, allocating one from the free pool if needed.
Begins command buffer recording if not already started.
"""
function ensure_active_batch!(bq::BatchQueue)
    @assert Threads.threadid() == bq.owning_thread  "BatchQueue is single-writer; cross-thread dispatch forbidden"
    # Refuse to start new work on a lost device.  Without this gate, kernel
    # dispatches keep recording into command buffers that will never run,
    # leaking arg slabs/cmd bufs until vk_reset_device! and (worse) hiding
    # the original error behind a flood of follow-on failures.
    device_lost(bq.ctx::VkContext) && throw(LavaError(
        "ensure_active_batch!",
        "Vulkan device is lost — cannot record new dispatches",
        "Call vk_reset_device!() to reinitialize, or restart Julia. " *
        "All existing LavaArrays are invalid after reset and must be re-allocated."))
    # Sweep any reaper-retired batches first so their resources recycle and
    # their errors surface before we start new work.
    sweep_retired_batches!(bq)

    batch = bq.active_batch
    if batch !== nothing
        if !batch.recording
            throw_if_error(bq, "vkBeginCommandBuffer",
                VK.begin_command_buffer(batch.cmd_buf, VK.CommandBufferBeginInfo(
                    flags=cb_begin_flags(bq)
                )))
            batch.recording = true
            # Fresh open on reused batch — assign the timeline value it will
            # signal, so record_buffer_access! can write it into buf.last_write.
            batch.signal_value = bq.next_timeline + 1
            # `vkBeginCommandBuffer` has just reset this command buffer, so the
            # segment starts empty and its dispatch count starts with it.
            # (Hygiene, not a fix: zeroing it here does not change the periodic
            # dropped frame — measured, bursts unchanged at every 600 frames for
            # a 20-dispatch graph. Whatever `cb_split_threshold` is doing to that
            # is not this counter surviving a reopen.)
            batch.segment_dispatches = 0
        end
        return batch
    end

    if !isempty(bq.free_batches)
        batch = pop!(bq.free_batches)
    else
        batch = allocate_batch(bq)
    end

    bq.active_batch = batch

    throw_if_error(bq, "vkBeginCommandBuffer",
        VK.begin_command_buffer(batch.cmd_buf, VK.CommandBufferBeginInfo(
            flags=cb_begin_flags(bq)
        )))
    batch.recording = true
    batch.signal_value = bq.next_timeline + 1
    return batch
end


# VkContext convenience: use default batch queue
function ensure_active_batch!(ctx::VkContext)
    return ensure_active_batch!(ctx.default_bq)
end

"""Allocate a new CommandBatch (new command buffer from the pool)."""
function allocate_batch(bq::BatchQueue)
    cmd_buf = alloc_cmd_buf(bq)
    batch = init_batch(cmd_buf)
    batch.bq = bq
    return batch
end

"""Allocate a command buffer from the free pool, or create a new one."""
function alloc_cmd_buf(bq::BatchQueue)
    if !isempty(bq.free_cmd_bufs)
        return pop!(bq.free_cmd_bufs)
    end
    alloc_info = VK.CommandBufferAllocateInfo(
        bq.cmd_pool, VK.COMMAND_BUFFER_LEVEL_PRIMARY, 1
    )
    return throw_if_error(bq, "vkAllocateCommandBuffers",
        VK.allocate_command_buffers(bq.device, alloc_info))[1]
end

# Drop the DataRefs retained by `pin!`.  Called once the batch can no longer be
# submitted — it either completed (`reclaim_batch!`) or its submit failed.  This
# is what lets a buffer whose owning LavaArray was `unsafe_free!`d mid-batch
# finally reach refcount zero and release its VkManagedBuffer.
function release_pinned_refs!(batch::CommandBatch)
    for ref in batch.pinned_refs
        # Drop the buffer pin first: this is the point where a free that was
        # requested mid-batch actually happens, and it must happen while the
        # DataRef is still valid so `ref[]` can name the buffer.
        unpin_buffer!(ref[])
        GPUArrays.unsafe_free!(ref)
    end
    empty!(batch.pinned_refs)
    return nothing
end

"""Reclaim a completed batch: clear pinned set, return to free pool."""
function reclaim_batch!(bq::BatchQueue, batch::CommandBatch)
    batch.recording = false
    batch.dispatch_count = 0
    batch.segment_dispatches = 0
    batch.last_was_rt = false
    empty!(batch.pinned)
    release_pinned_refs!(batch)
    empty!(batch.wait_semaphores)
    empty!(batch.dispatch_log)
    append!(bq.free_cmd_bufs, batch.sealed_cmd_bufs)
    empty!(batch.sealed_cmd_bufs)
    # Segments `present_frame!` already submitted: the fence has passed by the
    # time a batch is reclaimed, so the GPU is done reading them and they can go
    # back to the pool with the rest.
    append!(bq.free_cmd_bufs, batch.submitted_cmd_bufs)
    empty!(batch.submitted_cmd_bufs)
    push!(bq.free_batches, batch)
    # Note: pool reset + deferred-free drain are done in `sweep_retired_batches!`
    # AFTER the batch is actually removed from `bq.in_flight` (reclaim_batch! is
    # called with the batch still present in `in_flight`, so `isempty` would
    # always read `false` here and the reset would never fire — that bug leaked
    # ~70 MiB/frame via arg/indirect slabs on long per-frame dispatch streams).
end


"""
    maybe_split_cb!(batch, ctx)

If the current CB segment has reached `bq.cb_split_threshold` dispatches, seal it
and start a fresh CB. The sealed CB is stored in `batch.sealed_cmd_bufs` and will
be submitted alongside the active CB in `vk_flush!`.

Barriers work across CB boundaries per Vulkan spec — submission order defines
the scope of pipeline barriers, not command buffer boundaries.
"""
function maybe_split_cb!(batch::CommandBatch, bq::BatchQueue)
    threshold = bq.cb_split_threshold
    threshold <= 0 && return
    batch.segment_dispatches < threshold && return

    # Seal current CB
    throw_if_error(bq, "vkEndCommandBuffer", VK.end_command_buffer(batch.cmd_buf))
    push!(batch.sealed_cmd_bufs, batch.cmd_buf)

    # Start fresh CB segment
    batch.cmd_buf = alloc_cmd_buf(bq)
    throw_if_error(bq, "vkBeginCommandBuffer",
        VK.begin_command_buffer(batch.cmd_buf, VK.CommandBufferBeginInfo(
            flags=cb_begin_flags(bq)
        )))
    batch.segment_dispatches = 0
    # recording stays true; dispatch_count stays (for barrier logic + total tracking)
end

# ── Recording API ──
#
# All dispatch functions (compute, indirect, RT) use `record_dispatch!` to
# handle the shared boilerplate: get batch, insert barrier, run user code,
# update bookkeeping. The `do` block contains only the dispatch-specific
# Vulkan commands.

"""
    record_dispatch!(f, bq; dst_stage, extra_dst_access=0, is_rt=false, info="")

Record a dispatch into the active command batch. Handles:
1. Getting/creating the active batch
2. Inserting a pipeline barrier if this isn't the first dispatch
3. Calling `f(batch)` with the CommandBatch
4. Updating dispatch count, counter, and debug log

The do-block receives the `CommandBatch`, not a raw command buffer. This lets
dispatch functions `pin!` resources (pipelines, indirect buffers, closures)
into `batch.pinned` alongside recording commands — every pinned object is
kept alive until the timeline signals and also visited by `sync_access!` at
submit.

Example:
    record_dispatch!(bq; dst_stage=PIPELINE_STAGE_COMPUTE_SHADER_BIT) do batch
        VK.cmd_bind_pipeline(batch.cmd_buf, ...)
        pin!(batch, pipeline)
        VK.cmd_dispatch(batch.cmd_buf, ...)
    end
"""
# Inter-dispatch barrier overrides for the concurrent-dispatch-group flow.
#
# By default every `record_dispatch!` after the first one in a batch inserts
# a global SHADER_WRITE → SHADER_READ|WRITE memory barrier. That guarantees
# safety but forces strict serialisation on the GPU. For a group of dispatches
# that are mutually independent (different output buffers, or shared output
# via atomically-claimed slots) the barriers are unnecessary and prevent
# overlap on idle SMs — proven 3.06× → 1.13× on a 3-dispatch group of
# 1024-thread kernels.
#
# Within a `concurrent_dispatch_group(...) do ... end`:
#   * The very first dispatch keeps its barrier (it must still sync against
#     the producer that came before the group).
#   * Subsequent dispatches in the group skip the barrier.
#   * The barrier between the last dispatch in the group and the next
#     dispatch outside the group is re-established automatically by that
#     next dispatch's normal pre-barrier.
#
# The two flags are task-local-style (plain `Threads.Atomic{Bool}`); they
# could be `task_local_storage` instead if multi-task command recording
# becomes a thing, but Lava's BatchQueue is single-threaded today.
const CONCURRENT_GROUP_ACTIVE  = Threads.Atomic{Bool}(false)
const CONCURRENT_GROUP_STARTED = Threads.Atomic{Bool}(false)

"""
    bq.barrier_mode :: Symbol

How `record_dispatch!` orders one dispatch against the previous one.

  * `:memory` (default) — a `VkMemoryBarrier` making shader writes available to
    shader reads and writes. Correct everywhere.
  * `:execution` — the same stage dependency with *no* memory barrier. On a
    device whose L2 is shared and coherent across shader cores (all NVIDIA and
    AMD parts Lava targets) the availability operation is redundant for
    buffer traffic that never leaves L2, so this is a diagnostic for how much of
    the per-dispatch cost is cache maintenance versus the pipeline drain. It is
    not portable and not the default.

Measured on an RTX 4000 Ada: a dependent 64-element dispatch costs ~12 µs with a
barrier and ~3 µs without one, so on a 2500-dispatch inference step the barriers
alone are the majority of the wall time. Knowing which half of the barrier that
cost sits in is what this knob is for.
"""

"""
    concurrent_dispatch_group(f)

Run `f()` with inter-dispatch barriers suppressed *between* dispatches
inside `f`. The first dispatch inside the group still pre-barriers
against whatever ran before (producer→consumer), and the next dispatch
*outside* the group will pre-barrier against the group's writes
(group→consumer). Inside the group the dispatches may overlap on the GPU.
"""
function concurrent_dispatch_group(f::F) where F
    prev_active  = CONCURRENT_GROUP_ACTIVE[]
    prev_started = CONCURRENT_GROUP_STARTED[]
    CONCURRENT_GROUP_ACTIVE[]  = true
    CONCURRENT_GROUP_STARTED[] = false
    try
        f()
    finally
        CONCURRENT_GROUP_ACTIVE[]  = prev_active
        CONCURRENT_GROUP_STARTED[] = prev_started
    end
end

"""
    exclusive_dispatch_group(f)

Run `f()` with the automatic inter-dispatch barrier back on, whatever group is
open around it.

For a caller that has derived the dependencies between *units of work* but not
inside them. A render graph knows what each pass reads and writes and emits the
barrier between passes itself, which is the whole reason it opens a
`concurrent_dispatch_group`; it knows nothing about the launches within one pass
when that pass is an opaque body. Suppressing barriers there is not an
optimisation, it is a race — an operator whose second kernel reads what its first
wrote is ordinary, and a two-pass reduction or a split-K matmul is exactly that.

So the group is lifted for the body, and the caller sets `bq.next_skip_barrier`
if the *first* launch in it is the one it already emitted a barrier for.
"""
function exclusive_dispatch_group(f::F) where F
    prev_active  = CONCURRENT_GROUP_ACTIVE[]
    prev_started = CONCURRENT_GROUP_STARTED[]
    CONCURRENT_GROUP_ACTIVE[] = false
    try
        f()
    finally
        CONCURRENT_GROUP_ACTIVE[]  = prev_active
        CONCURRENT_GROUP_STARTED[] = prev_started
    end
end

# ── Deferred indirect dispatch group ──
#
# An indirect dispatch's args read depends on its own prepare-indirect
# kernel, so it can never skip its pre-barrier (`force_pre_barrier` above).
# Inside a plain `concurrent_dispatch_group` that forced barrier serialises
# every prepare+dispatch pair — N pairs degrade to fully sequential
# execution, which costs real wall-clock when the dispatches are small
# (Hikari's 12 per-material shading kernels per bounce).
#
# `concurrent_indirect_group` restores the overlap CORRECTLY by splitting
# the group into three phases:
#   1. While `f()` runs, `ka_launch_indirect!` packs args + allocates the
#      indirect slot for each dispatch but records NOTHING — it defers
#      (pipeline, args, indirect slot, queue-size buffer, workgroup size)
#      into `bq.deferred_indirect`.
#   2. On exit, ONE fused multi-prepare dispatch writes every deferred
#      indirect command (a single thread looping over the slots — same
#      pattern as `empty_all!`). Its normal pre-barrier orders it after
#      the producers that filled the queues.
#   3. The dispatches record back-to-back: the FIRST carries one barrier
#      (orders ALL subsequent commands after the multi-prepare's writes,
#      with INDIRECT_COMMAND_READ access), the rest skip — they only share
#      atomically-claimed queue slots, same contract as a plain concurrent
#      group.
# Net for N pairs: 2 barriers + N+1 commands, instead of N barriers + 2N
# commands; all N dispatches free to overlap on the GPU.
#
# CONTRACT: everything dispatched inside `f` must be mutually independent
# (it's a CONCURRENT group), and because the indirect dispatches flush at
# group exit, a DIRECT dispatch recorded inside `f` lands BEFORE them in
# the command stream regardless of lexical order.


# One thread writes every deferred VkDispatchIndirectCommand. Statically
# unrolled over the tuples; compiled per arity via the normal kernel cache.
@inline _multi_prepare!(::Tuple{}, ::Tuple{}, ::Tuple{}) = nothing
@inline function _multi_prepare!(inds::Tuple, sizes::Tuple, wss::Tuple)
    n = UInt32(@inbounds sizes[1][1])
    ws = wss[1]
    groups = (n + ws - UInt32(1)) ÷ ws
    @inbounds begin
        inds[1][1] = groups      # groupCountX
        inds[1][2] = UInt32(1)   # groupCountY
        inds[1][3] = UInt32(1)   # groupCountZ
    end
    _multi_prepare!(Base.tail(inds), Base.tail(sizes), Base.tail(wss))
    return nothing
end

function multi_prepare_indirect_kernel(inds, sizes, wss)
    _multi_prepare!(inds, sizes, wss)
    return nothing
end

function concurrent_indirect_group(f::F, bq::BatchQueue = vk_context().default_bq) where F
    prev = bq.deferred_indirect
    list = Any[]
    bq.deferred_indirect = list
    try
        concurrent_dispatch_group(f)
    finally
        bq.deferred_indirect = prev
    end
    isempty(list) && return nothing

    # Phase 2: one fused prepare for every deferred indirect slot.
    bq0 = list[1][1]::BatchQueue
    # `Val`: see the note in `ka_backend.jl` about `Base._ntuple` and the
    # `Tuple{Colon, Int64, Any}` edge — this is on the dispatch path too.
    nl = length(list)
    inds  = ntuple(i -> list[i][4], Val(nl))
    sizes = ntuple(i -> list[i][6], Val(nl))
    wss   = ntuple(i -> UInt32(list[i][7]), Val(nl))
    lava_launch!(bq0, multi_prepare_indirect_kernel, inds, sizes, wss;
                 ndrange=1, workgroup_size=(1, 1, 1))

    # Phase 3: the dispatches, overlapped behind one shared barrier.
    for (i, entry) in enumerate(list)
        bq, pipeline, push_bda, indirect, tlas = entry
        @assert bq === bq0  "concurrent_indirect_group: all dispatches must share one BatchQueue"
        vk_dispatch_indirect_base!(bq, pipeline, push_bda, indirect, tlas;
                                   first_in_group = i == 1)
    end
    return nothing
end

@inline function record_dispatch!(f, bq::BatchQueue;
                           dst_stage::VK.PipelineStageFlag,
                           extra_dst_access::VK.AccessFlag=VK.AccessFlag(0),
                           is_rt::Bool=false,
                           skip_pre_barrier::Bool=false,
                           force_pre_barrier::Bool=false,
                           info::String="")
    batch = ensure_active_batch!(bq)
    cmd = batch.cmd_buf

    # Memory barrier between dispatches (write→read synchronization).
    # Uses direct ccall to avoid VK.jl wrapper allocations (~1.2KB/call).
    #
    # `skip_pre_barrier=true` opts a dispatch out of the inter-dispatch
    # barrier — the caller is asserting that this dispatch has no execution
    # or memory dependency on the previous one in the same batch (they
    # write to disjoint memory, or share only atomically-claimed slots in
    # a work queue, etc). Vulkan permits the implementation to overlap
    # those dispatches on different SMs; with the barrier they're serialised.
    # Also honoured: the task-local concurrent_dispatch_group state — the
    # FIRST dispatch inside a group still barriers (producer→consumer),
    # but all subsequent dispatches inside the group skip.
    #
    # `force_pre_barrier=true` overrides BOTH skip mechanisms. Indirect
    # dispatches (vk_dispatch_indirect!/rt) must use it: the command
    # processor's read of the VkDispatchIndirectCommand depends on the
    # immediately preceding prepare-indirect kernel's write. Inside a
    # concurrent_dispatch_group the group skip used to drop exactly that
    # barrier, leaving a prepare→indirect-read race that the GPU usually —
    # but not always — won. It reliably LOST right after a mid-pipeline
    # flush (empty queue, commands execute as they arrive), which is how
    # Hikari's volpath bounce-loop early-exit surfaced it (2026-06-10):
    # `vp_shade_typed!`'s 12 in-group prepare+indirect pairs read garbage
    # group counts and silently dropped shading work (~15% energy loss on
    # shadow_bumpgold).
    # A dispatch that did not declare its buffers (Lava's own `lava_launch!`
    # callers, indirect prepares) is opaque to the elision tracker: it must take
    # its barrier, and nothing after it may elide until a barrier clears the
    # poison.
    if bq.barrier_elision && !bq.ranges_declared
        bq.next_skip_barrier = false
        poison_barrier_elision!(bq)
    end
    bq.ranges_declared = false

    effective_skip = (skip_pre_barrier || bq.next_skip_barrier ||
                      (CONCURRENT_GROUP_ACTIVE[] && CONCURRENT_GROUP_STARTED[])) &&
                     !force_pre_barrier
    bq.next_skip_barrier = false     # one-shot: consumed by exactly this dispatch
    if CONCURRENT_GROUP_ACTIVE[]
        CONCURRENT_GROUP_STARTED[] = true
    end
    # The first dispatch of a batch still needs a barrier when an earlier batch
    # is in flight. `dispatch_count > 0` alone assumes a batch boundary is also a
    # synchronisation point, which it is not: `submit!` adds no wait on the
    # previous submission, so without this the first dispatch of every new batch
    # could read what the last dispatch of the previous one was still writing.
    # Harmless while nothing submitted mid-stream; `bq.auto_submit_threshold` makes
    # it happen every 64 dispatches. After a `flush!` the queue is drained and
    # `in_flight` is empty, so the genuinely-first dispatch still skips.
    needs_boundary_barrier = batch.dispatch_count == 0 && !isempty(bq.in_flight)
    if (batch.dispatch_count > 0 || needs_boundary_barrier) && !effective_skip
        src_stage = batch.last_was_rt ?
            VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR :
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT
        dst_access = VkAccessFlags(VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT) | VkAccessFlags(extra_dst_access)
        nmem = bq.barrier_mode === :execution ? UInt32(0) : UInt32(1)
        barrier_ref = Ref(VkMemoryBarrier(
            VK_STRUCTURE_TYPE_MEMORY_BARRIER, C_NULL,
            VkAccessFlags(VK_ACCESS_SHADER_WRITE_BIT), dst_access))
        GC.@preserve barrier_ref begin
            ccall(barrier_fptr(bq), Cvoid,
                  (Ptr{Nothing}, VkPipelineStageFlags, VkPipelineStageFlags, VkDependencyFlags,
                   UInt32, Ptr{VkMemoryBarrier}, UInt32, Ptr{Nothing}, UInt32, Ptr{Nothing}),
                  cmd.vks,
                  VkPipelineStageFlags(src_stage), VkPipelineStageFlags(dst_stage), VkDependencyFlags(0),
                  # The pointer is ignored when the count is zero, so pass it
                  # unconditionally: a `count == 0 ? C_NULL : ptr` ternary mixes
                  # `Ptr{Nothing}` with `Ptr{VkMemoryBarrier}`, and the boxing
                  # that costs shows up as milliseconds over a 2500-dispatch step.
                  nmem, barrier_ref,
                  UInt32(0), C_NULL,
                  UInt32(0), C_NULL)
        end
    end

    # Record the actual dispatch commands. The do-block receives the batch
    # so it can pin! resources alongside recording commands.
    f(batch)

    # Bookkeeping
    batch.dispatch_count += 1
    batch.segment_dispatches += 1
    batch.last_was_rt = is_rt
    let d = (bq.ctx::VkContext).diag
        if d.dispatch_logging
            Threads.atomic_add!(d.total_dispatches, 1)
            log_dispatch!(bq, Base.invokelatest(dispatch_log_string,
                                            d.total_dispatches[], " ", info)::String)
        end
    end

    # Split to a new CB if this segment is full
    maybe_split_cb!(batch, bq)

    # Auto-submit to avoid TDR when a single submission's GPU execution time
    # approaches amdgpu's ~10 s lockup_timeout.  `submit!` queues the batch
    # onto the in-flight list without blocking — next `record_dispatch!` will
    # `ensure_active_batch!` a fresh batch.  Cross-batch buffer synchronisation
    # is already handled via `sync_access!` writing `buf.last_write` and
    # wait_semaphores picking it up on the next pin.
    threshold = bq.auto_submit_threshold
    if threshold > 0 && batch.dispatch_count >= threshold
        submit!(bq)
    end
end

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

# ── Compute Dispatch ──

"""
    vk_dispatch!(pipeline, push_bda, groups)

Record a compute dispatch.
`push_bda` is the BDA address of the argument buffer (passed as 8-byte push constant).
"""
function vk_dispatch!(bq::BatchQueue, pipeline::LavaComputePipeline, push_bda::UInt64,
                      groups::NTuple{3, Integer},
                      tlas=nothing)  # positional with Nothing default — hot-path callers pass `tlas` positionally to avoid per-dispatch NamedTuple construction (~3 µs/dispatch)
    vk_dispatch_base!(bq, pipeline, push_bda, 0, 0, 0,
                      Int(groups[1]), Int(groups[2]), Int(groups[3]), tlas)
end

# Two methods specialized on `tlas` type. The fast (no-TLAS) path is here; the
# `tlas::HWTLAS` overload lives in raytracing/hwtlas.jl (where HWTLAS exists). The
# kwarg call site above gets the right method by ordinary dispatch — there is no
# `pipeline.needs_tlas_descriptor` branch and no `extra_dst_access` ternary on the
# pure-compute hot path.
"""Record a single compute dispatch with optional base group offset (no-TLAS fast path)."""
@inline function vk_dispatch_base!(bq::BatchQueue, pipeline::LavaComputePipeline, push_bda::UInt64,
                            base_x::Int, base_y::Int, base_z::Int,
                            gx::Int, gy::Int, gz::Int, ::Nothing=nothing)
    dispatch_info = (bq.ctx::VkContext).diag.dispatch_logging ?
        Base.invokelatest(dispatch_log_string, bq.last_dispatch_info, " base=(",
                          base_x, ",", base_y, ",", base_z, ") g=(",
                          gx, ",", gy, ",", gz, ")")::String : ""
    record_dispatch!(bq;
        dst_stage=VK.PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        info=dispatch_info
    ) do batch
        cmd = batch.cmd_buf
        VK.cmd_bind_pipeline(cmd, VK.PIPELINE_BIND_POINT_COMPUTE, pipeline.pipeline)
        pin!(batch, pipeline)
        push_constants_bda!(cmd, pipeline.pipeline_layout, VK.SHADER_STAGE_COMPUTE_BIT, push_bda)
        # Profiling: optional GPU-side timestamp around the dispatch.  Returns
        # -1 (and does nothing) when `with_dispatch_timing` is not active, so
        # the hot path stays unperturbed.
        ts_slot = maybe_write_dispatch_start_timestamp!(bq.ctx::VkContext, cmd, bq.last_dispatch_info)
        if base_x == 0 && base_y == 0 && base_z == 0
            VK.cmd_dispatch(cmd, UInt32(gx), UInt32(gy), UInt32(gz))
        else
            VK.cmd_dispatch_base(cmd,
                UInt32(base_x), UInt32(base_y), UInt32(base_z),
                UInt32(gx), UInt32(gy), UInt32(gz))
        end
        maybe_write_dispatch_end_timestamp!(bq.ctx::VkContext, cmd, ts_slot, barrier_fptr(bq))
    end
end

# ── Indirect Dispatch ──

"""
    vk_dispatch_indirect!(bq, pipeline, push_bda, indirect::LavaArray{UInt32,1})

Record an indirect compute dispatch.  `indirect` is a LavaArray view of 3
UInt32s (groupCountX/Y/Z), typically obtained from `get_indirect_buffer(bq)`
and populated by a prepare-indirect kernel.
"""
function vk_dispatch_indirect!(bq::BatchQueue, pipeline::LavaComputePipeline,
                               push_bda::UInt64,
                               indirect,  # LavaArray{UInt32,1} — declared later in array/lavaarray.jl
                               tlas=nothing)  # positional with Nothing default — same NamedTuple-avoidance as vk_dispatch!
    vk_dispatch_indirect_base!(bq, pipeline, push_bda, indirect, tlas)
end

# Specialized on `tlas` type. The TLAS overload lives in raytracing/hwtlas.jl.
"""Record an indirect compute dispatch (no-TLAS fast path).

`first_in_group=false` is ONLY for `concurrent_indirect_group`'s flush
phase: the group's first dispatch already recorded the barrier covering
every prepare's writes, so the rest may skip (they share only
atomically-claimed queue slots)."""
@inline function vk_dispatch_indirect_base!(bq::BatchQueue, pipeline::LavaComputePipeline,
                                            push_bda::UInt64,
                                            indirect,  # LavaArray{UInt32,1}
                                            ::Nothing=nothing;
                                            first_in_group::Bool=true)
    dispatch_info = (bq.ctx::VkContext).diag.dispatch_logging ?
        Base.invokelatest(dispatch_log_string, bq.last_dispatch_info, " (indirect)")::String : ""
    record_dispatch!(bq;
        dst_stage=VK.PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK.PIPELINE_STAGE_DRAW_INDIRECT_BIT,
        extra_dst_access=VK.ACCESS_INDIRECT_COMMAND_READ_BIT,
        # The indirect-args read depends on the preceding prepare-indirect
        # write — never elide this barrier (see record_dispatch! docs),
        # except behind a deferred group's shared barrier.
        force_pre_barrier=first_in_group,
        skip_pre_barrier=!first_in_group,
        info=dispatch_info
    ) do batch
        cmd = batch.cmd_buf
        VK.cmd_bind_pipeline(cmd, VK.PIPELINE_BIND_POINT_COMPUTE, pipeline.pipeline)
        pin!(batch, pipeline)
        push_constants_bda!(cmd, pipeline.pipeline_layout, VK.SHADER_STAGE_COMPUTE_BIT, push_bda)
        mb = indirect.buf[]::VkManagedBuffer
        byte_offset = UInt64(indirect.offset)
        ts_slot = maybe_write_dispatch_start_timestamp!(bq.ctx::VkContext, cmd, bq.last_dispatch_info)
        VK.cmd_dispatch_indirect(cmd, mb.buffer, byte_offset)
        maybe_write_dispatch_end_timestamp!(bq.ctx::VkContext, cmd, ts_slot, barrier_fptr(bq))
        pin!(batch, indirect)
    end
end

# ── Flush ──

"""
    query_timeline(bq::BatchQueue) -> UInt64

Return the current counter of `bq.timeline_sem`.  Replaces the old
`try ... catch; typemax(UInt64); end` sentinel:

  * On success, return the real counter value.
  * On any `VulkanError`, set `bq.ctx.device_lost` (if the code is
    `ERROR_DEVICE_LOST`) and throw `LavaVulkanError`.  No silent fallback —
    every caller must either have cleared the device-lost flag upstream
    (so the happy path runs) or be prepared to propagate the throw.

Callers are expected to gate on `device_lost(bq.ctx)` themselves before
calling this.  The only way the throw path fires is the race window where
the device died between that upstream check and this query — and in that
case, loud is correct: we want the dispatcher / finalizer log to show it.
"""
@inline function query_timeline(bq::BatchQueue)::UInt64
    result = VK.get_semaphore_counter_value(bq.device, bq.timeline_sem)
    iserror(result) || return unwrap(result)
    mark_if_lost!(bq, result)
    e = unwrap_error(result)::VK.VulkanError
    throw(LavaVulkanError("get_semaphore_counter_value", Int32(e.code), e.msg,
        e.code == VK.ERROR_DEVICE_LOST ?
            "Device lost. Call vk_reset_device!() to reinitialize, or restart Julia." :
            "Timeline semaphore query failed with an unexpected VkResult. " *
            "This is a driver bug, stack corruption, or an invalid semaphore handle."))
end

# ── Vulkan call helpers: single source of truth for device-lost handling ──
#
# Every Vulkan call that can fail should funnel through `throw_if_error` (or
# its non-throwing sibling `mark_if_lost!` for callers with custom recovery).
# This is the SOLE place the "VkResult → device_lost flag" rule lives, so
# the dispatcher gate at `ensure_active_batch!` reliably fires after any
# DEVICE_LOST regardless of which low-level call surfaced the error.

"""
    mark_if_lost!(bq::BatchQueue, result)
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
@inline mark_if_lost!(bq::BatchQueue, result) = mark_if_lost!(bq.ctx::VkContext, result)

device_lost_hint(call) =
    "Vulkan device is lost during $(call). " *
    "Call vk_reset_device!() to reinitialize, or restart Julia."

"""
    throw_if_error(bq::BatchQueue, result)
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
@inline throw_if_error(bq::BatchQueue, args...) = throw_if_error(bq.ctx::VkContext, args...)

"""
    queue_submit!(bq, submits; fence=VK.Fence(C_NULL))

`vkQueueSubmit` wrapper.  Auto-marks device_lost on failure and throws
`LavaVulkanError`.
"""
@inline queue_submit!(bq::BatchQueue, submits::AbstractVector{VK.SubmitInfo};
                      fence=VK.Fence(C_NULL)) =
    throw_if_error(bq, "vkQueueSubmit", VK.queue_submit(bq.queue, submits; fence=fence))

"""
    queue_submit_2!(bq, submits; fence=VK.Fence(C_NULL))

`vkQueueSubmit2` wrapper.  Auto-marks device_lost on failure.
"""
@inline queue_submit_2!(bq::BatchQueue, submits::AbstractVector{VK.SubmitInfo2};
                        fence=VK.Fence(C_NULL)) =
    throw_if_error(bq, "vkQueueSubmit2", VK.queue_submit_2(bq.queue, submits; fence=fence))

"""
    wait_for_fences!(bq, fences; wait_all=true, timeout=typemax(UInt64))

`vkWaitForFences` wrapper.  Auto-marks device_lost on failure.
"""
@inline wait_for_fences!(bq::BatchQueue, fences;
                         wait_all::Bool=true, timeout::UInt64=typemax(UInt64)) =
    throw_if_error(bq, "vkWaitForFences", VK.wait_for_fences(bq.device, fences, wait_all, timeout))

"""
    wait_semaphores!(bq, info; timeout=typemax(UInt64))

`vkWaitSemaphores` wrapper.  Auto-marks device_lost on failure.
"""
@inline wait_semaphores!(bq::BatchQueue, info::VK.SemaphoreWaitInfo;
                         timeout::UInt64=typemax(UInt64)) =
    throw_if_error(bq, "vkWaitSemaphores", VK.wait_semaphores(bq.device, info, timeout))

"""
    submit!(bq::BatchQueue) -> CommandBatch or nothing

Close the active batch's command buffer, run `sync_access!` over every
pinned object to populate `wait_semaphores` + update `last_write`, and
submit to the queue.  Retirement is synchronous: `sweep_retired_batches!`
polls the timeline semaphore at natural sync points (flush, next
`ensure_active_batch!`) and reclaims batches whose signal value has been
reached.

Returns the submitted `CommandBatch` (or `nothing` if no batch was active).
Use `flush!(bq, device)` for the blocking wrapper that waits for GPU
completion before returning.
"""
function submit!(bq::BatchQueue)
    @assert Threads.threadid() == bq.owning_thread  "BatchQueue is single-writer; cross-thread submit forbidden"
    batch = bq.active_batch
    batch === nothing && return nothing
    !batch.recording && return nothing
    @assert batch.bq === bq "batch.bq desync: batch was not bound to this BatchQueue"
    Threads.atomic_add!((bq.ctx::VkContext).diag.flush_counter, 1)

    throw_if_error(bq, "vkEndCommandBuffer", VK.end_command_buffer(batch.cmd_buf))

    n_sealed = length(batch.sealed_cmd_bufs)
    all_cmd_bufs = Vector{VK.CommandBuffer}(undef, n_sealed + 1)
    for i in 1:n_sealed
        all_cmd_bufs[i] = batch.sealed_cmd_bufs[i]
    end
    all_cmd_bufs[n_sealed + 1] = batch.cmd_buf

    # Capture takes ownership of these command buffers. Clearing `sealed_cmd_bufs`
    # keeps `reclaim_batch!` from handing them back to the free pool, and swapping
    # in a fresh `cmd_buf` keeps the batch itself from re-recording over the one we
    # just captured — either would silently rewrite a sequence `replay!` still points at.
    let cap = bq.capturing
        if cap !== nothing
            append!(cap.cmd_bufs, all_cmd_bufs)
            for obj in batch.pinned
                push!(cap.pinned, obj)
            end
            empty!(batch.sealed_cmd_bufs)
            batch.cmd_buf = alloc_cmd_buf(bq)
            cap.submissions += 1
        end
    end

    saved_dispatch_count = batch.dispatch_count
    saved_last_was_rt = batch.last_was_rt
    bq.prev_dispatch_info = bq.last_dispatch_info

    # ensure_active_batch! pre-assigned batch.signal_value = next_timeline + 1;
    # bump the counter now and assert consistency.
    bq.next_timeline += 1
    @assert batch.signal_value == bq.next_timeline "batch signal desync: $(batch.signal_value) vs $(bq.next_timeline)"

    # Apply per-object access semantics over the full pinned set.  For
    # VkManagedBuffers this populates batch.wait_semaphores and writes
    # (bq, signal_value) into buf.last_write.  Runs ONCE per batch, not per
    # dispatch — the IdSet dedupes multi-dispatch reuse for free.
    # `vkEndCommandBuffer` has already run, so the command buffer is no longer
    # recording — but `batch.recording` still says it is and `bq.active_batch`
    # still points here. Anything that throws in between leaves the queue in a
    # state where the next `ensure_active_batch!` hands this batch straight back
    # and the caller records into an ENDED command buffer: undefined behaviour
    # that NVIDIA takes as a SIGSEGV inside the driver, with no Julia frame to
    # show for it. (Exactly how a sync1/sync2 type error in `sync_access!` used
    # to present: a segfault in vkCmdPipelineBarrier during SAM 2's weight
    # upload, three frames removed from the actual bug.) Drop the batch instead,
    # the same way the vkQueueSubmit2 failure path below does, and let the error
    # surface as an error.
    try
        for obj in batch.pinned
            # LavaArrays are handled via `pinned_refs` below.  Between `pin!` and
            # here, an explicit `unsafe_free!(a)` (HW-accel BLAS/TLAS teardown) can
            # have marked `a.buf` freed; `a.buf[]` would throw even though the
            # VkManagedBuffer is still alive via our retained ref.
            obj isa LavaArray && continue
            sync_access!(batch, obj)
        end
        for ref in batch.pinned_refs
            sync_access!(batch, ref[])
        end
    catch
        batch.recording = false
        empty!(batch.pinned)
        release_pinned_refs!(batch)
        empty!(batch.wait_semaphores)
        bq.active_batch = nothing
        # `bq.next_timeline` is deliberately NOT rolled back. `sync_access!` has
        # already stamped `buf.last_write` on the buffers it reached with this
        # batch's signal value, which nothing will now signal — but timeline
        # waits are `>=` (`sweep_retired_batches!`: `signal_value <= current`),
        # so the next batch's larger signal satisfies them. Giving the value back
        # would instead risk handing it to a batch while another still holds it.
        rethrow()
    end

    # Pre-submit safety scan: catch stale-BDA-in-arg-slab corruption BEFORE
    # the GPU sees it.  Off by default (`ctx.diag.presubmit_scan = true` to
    # turn on for debugging).  Cost ~hundreds-of-µs per submit; never on by
    # default.
    if (bq.ctx::VkContext).diag.presubmit_scan
        unknowns = scan_slabs_for_unknown_bdas(bq)
        if !isempty(unknowns)
            @warn "Pre-submit found $(length(unknowns)) unknown BDA(s) in arg slabs"
            for u in unknowns
                @warn "  STALE: slab=$(u.slab) idx=$(u.idx) offset=$(u.offset) val=0x$(string(u.val, base=16, pad=16))"
            end
            if (bq.ctx::VkContext).diag.presubmit_scan_throws
                throw(LavaError("submit!", "stale BDA in arg slab", "see warnings"))
            end
        end
    end

    # SLAB DUMP for cascade investigation: if `ctx.diag.slab_dump_target` is non-zero,
    # search the active arg slab for any UInt64 == target and log offsets.
    if (bq.ctx::VkContext).diag.slab_dump_target != UInt64(0) &&
       !isempty(bq.arg_slabs) && bq.arg_slab_idx <= length(bq.arg_slabs)
        target = (bq.ctx::VkContext).diag.slab_dump_target
        slab = bq.arg_slabs[bq.arg_slab_idx]
        mb = slab.buf[]
        mp = mb.mapped_ptr
        if mp != Ptr{UInt8}(0)
            n = bq.arg_slab_offset ÷ 8
            p = Ptr{UInt64}(mp)
            hits = Int[]
            for k in 0:(n-1)
                if unsafe_load(p, k+1) == target
                    push!(hits, k*8)
                end
            end
            if !isempty(hits)
                push!((bq.ctx::VkContext).diag.slab_dump_log, (sub=Int(bq.next_timeline)+1, target=target, offsets=hits))
            end
        end
    end

    cb_infos = [VK.CommandBufferSubmitInfo(cb, UInt32(0)) for cb in all_cmd_bufs]
    wait_infos = [VK.SemaphoreSubmitInfo(s, v, UInt32(0); stage_mask=stage)
                  for (s, v, stage) in batch.wait_semaphores]
    signal_info = VK.SemaphoreSubmitInfo(bq.timeline_sem, batch.signal_value, UInt32(0);
                                             stage_mask=VK.PIPELINE_STAGE_2_ALL_COMMANDS_BIT)
    submit_info = VK.SubmitInfo2(wait_infos, cb_infos, [signal_info])
    submit_result = VK.queue_submit_2(bq.queue, [submit_info])
    if iserror(submit_result)
        # `mark_if_lost!` is the canonical place that flips the
        # device_lost flag for any VkResult error.  We then run our own
        # recovery (drop the recording batch) and rethrow with rich
        # validation context, since submit! has dispatch-count / RT-state
        # detail that the generic wrapper can't produce.
        mark_if_lost!(bq, submit_result)
        batch.recording = false
        empty!(batch.pinned)
        release_pinned_refs!(batch)
        empty!(batch.wait_semaphores)
        bq.active_batch = nothing
        throw_with_validation_context("vkQueueSubmit2", submit_result,
            saved_dispatch_count, saved_last_was_rt, bq)
    end

    push!(bq.in_flight, batch)
    bq.active_batch = nothing
    # Everything handed out of the arg pool so far is read by this batch, and the
    # pool may rewind once the timeline passes it. Recorded here rather than only
    # in `present_frame!`, because a compute-only queue never presents and would
    # otherwise never be allowed to rewind at all.
    arg_pool_in_use!(bq, batch.signal_value)
    dg = (bq.ctx::VkContext).diag
    # GATED. This built a fresh `String` on EVERY submit — 7.9 bytes per dispatch
    # amortised — for a field only ever read by `maybe_write_dispatch_start_
    # timestamp!` (which returns -1 immediately unless `dispatch_timing`) and by
    # the hwtlas debug strings (gated on `dispatch_logging`). Same shape as the
    # slab-scan bug: a diagnostic doing real work while switched off.
    if dg.dispatch_logging || dg.dispatch_timing
        bq.last_dispatch_info = "$saved_dispatch_count dispatches ($(saved_last_was_rt ? "RT" : "compute"))"
    end
    Threads.atomic_add!(dg.total_dispatches, saved_dispatch_count)
    if dg.dispatch_logging
        append!(dg.dispatch_log, batch.dispatch_log)
    end
    # DEBUG: synchronous per-batch wall-clock timing. Serializes the pipeline
    # but lets us see GPU execution time per batch. Guarded by opt-in flag.
    if dg.batch_timing
        t0 = time()
        wr = VK.wait_semaphores(bq.device,
            VK.SemaphoreWaitInfo([bq.timeline_sem], [batch.signal_value]),
            typemax(UInt64))
        dt = time() - t0
        push!(dg.batch_wait_times, dt)
        push!(dg.batch_wait_info, bq.last_dispatch_info)
        push!(dg.batch_wait_dispatches, saved_dispatch_count)
        # Don't throw from here — let the next call surface it normally — but
        # do mark device_lost so the next dispatcher gate fires.
        mark_if_lost!(bq, wr)
    end
    return batch
end

"""
    sweep_retired_batches!(bq::BatchQueue)

Reclaim any in-flight batches whose signal_value has been reached.  Uses
the non-blocking `get_semaphore_counter_value` to poll the timeline.
Called from the main thread at natural quiescent points.
"""
function sweep_retired_batches!(bq::BatchQueue)
    @assert Threads.threadid() == bq.owning_thread  "BatchQueue is single-writer; cross-thread sweep forbidden"
    isempty(bq.in_flight) && return
    device_lost(bq.ctx::VkContext) && return
    current = query_timeline(bq)
    i = 1
    while i <= length(bq.in_flight)
        b = bq.in_flight[i]
        if b.signal_value <= current
            # Batch has been signalled — safe to reclaim (clears pinned +
            # wait_semaphores inside reclaim_batch!).
            reclaim_batch!(bq, b)
            deleteat!(bq.in_flight, i)
        else
            i += 1
        end
    end
    drain_deferred_frees!(bq)
    drain_deferred_as_frees!(bq)
    # Reset pools once the queue is idle.  Done HERE (not inside reclaim_batch!)
    # because reclaim is called with the batch still in `bq.in_flight` — the
    # isempty check had to run after `deleteat!` to ever return true.
    #
    # CRITICAL: "idle" must include the ACTIVE batch. sweep_retired_batches!
    # runs opportunistically from `ensure_active_batch!` and the allocation
    # paths — i.e. potentially MID-RECORDING. If the active batch already
    # holds recorded dispatches, their packed args / indirect-command slots
    # live in the slabs at offsets below the current cursors; resetting the
    # cursors here lets the very next dispatch overwrite them, so the earlier
    # dispatches read garbage arg buffers when the batch finally submits
    # (silently — typically as kernels seeing wrong buffer pointers).
    # Latent for a long time because `in_flight` only drains to empty
    # mid-recording when the GPU runs ahead of the host — exactly what
    # happens after a host-side mid-pipeline synchronize, e.g. volpath's
    # bounce-loop early-exit check (Hikari, 2026-06-10: every sample after
    # the first rendered black because round-0's trace dispatch read a
    # clobbered queue pointer).
    #
    # A capture widens "idle" further still: its command buffers outlive the
    # batches that recorded them, and every dispatch in them keeps reading its
    # arguments from the slab at replay time. So the bytes of batches that have
    # already been submitted AND signalled are still live — the one case this
    # test otherwise treats as the safest of all. Mid-capture the queue drains to
    # empty routinely (the GPU is running a step's worth of work while the host
    # records the next), so without this the second half of a capture overwrites
    # the argument records of the first, and the replay dispatches valid commands
    # against wrong pointers: no validation error, no device fault, just a kernel
    # that never returns. A capture's own slabs (see `CapturedSequence`) are
    # never rewound at all, which is what actually protects it — this test still
    # matters for the SHARED pool, which a capture's non-argument allocations and
    # every ordinary recording keep using.
    # "Recorded dispatches" is the wrong question, and asking it is a GPU crash.
    # An arg buffer is handed out *before* the dispatch that uses it is recorded,
    # and a caller may take several before recording any — every draw in a frame
    # packs its arguments and only then records the draw. `arg_alloc_count` is
    # the handouts since the last submit, which is what "someone is still holding
    # pool memory" actually means.
    capturing_here = bq.capturing !== nothing
    active = bq.active_batch
    active_has_commands = active !== nothing && active.dispatch_count > 0
    if isempty(bq.in_flight) && !active_has_commands && !capturing_here &&
       bq.arg_alloc_count == 0
        reset_arg_buffer_pool!(bq)
        reset_indirect_buffer_pool!(bq)
    end
    return nothing
end

"""
    flush!(bq::BatchQueue, device::VK.Device)

Submit the active batch (if any) and block until every in-flight batch on
`bq` has been signalled on the queue's timeline semaphore.  Uses a single
`wait_semaphores` call on the HIGHEST pending signal value — the timeline
is monotonic, so once that value is reached every lower value is too.
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
function flush_stall_report(bq::BatchQueue, target::UInt64)
    io = IOBuffer()
    ctx = bq.ctx::VkContext
    # The one place a swallow is right, and it is narrowed to say why: this
    # builds the diagnostic printed when a flush has ALREADY stalled, and the
    # device may be lost. Failing to read the counter must not replace the report
    # the caller is waiting for — but only a Vulkan error is tolerated, and the
    # reason is printed rather than left blank.
    cur = try
        unwrap(VK.get_semaphore_counter_value(ctx.device, bq.timeline_sem))
    catch err
        err isa VK.VulkanError || rethrow()
        err
    end
    println(io, "  timeline counter = ", cur isa Exception ? "unreadable ($cur)" : cur,
                ", next_timeline = ", bq.next_timeline,
                ", replay watermark = ", bq.replay_watermark)
    for (i, b) in enumerate(bq.in_flight)
        waits = [v for (_, v, _) in b.wait_semaphores]
        done = cur !== nothing && b.signal_value <= cur
        println(io, "  batch $i: signals ", b.signal_value,
                    ", waits on ", isempty(waits) ? "nothing" : string(waits),
                    done ? "  [already signalled]" : "")
    end
    if cur !== nothing && all(b -> b.signal_value <= cur, bq.in_flight) && target > cur
        println(io, "  >> every in-flight batch is already signalled and the target is not: ",
                    "the wait is on a value nothing will signal.")
    end
    return String(take!(io))
end

function flush!(bq::BatchQueue, device::VK.Device)
    @assert Threads.threadid() == bq.owning_thread  "BatchQueue is single-writer; cross-thread flush forbidden"
    submit!(bq)
    # A replay signals the timeline without putting a batch in `in_flight`, so
    # the in-flight scan alone would return before the GPU had run any of it.
    target = bq.replay_watermark
    for b in bq.in_flight
        target = max(target, b.signal_value)
    end
    target == UInt64(0) && return
    budget = bq.flush_timeout_ns
    quantum = budget == 0 ? typemax(UInt64) : min(budget, FLUSH_WAIT_QUANTUM_NS)
    waited = UInt64(0)
    while true
        wait_result = VK.wait_semaphores(device,
            VK.SemaphoreWaitInfo([bq.timeline_sem], [target]), quantum)
        if iserror(wait_result)
            # Rich rethrow: flush! gets validation context + dispatcher hints the
            # generic wrapper can't.  The mark_if_dl! call is the single source of
            # truth for the device_lost flag.
            mark_if_lost!(bq, wait_result)
            throw_with_validation_context("vkWaitSemaphores", wait_result, 0, false, bq)
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
        #     get_fence_status(as_fence)      silent
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
        let st = VK.queue_wait_idle(bq.queue)
            iserror(st) && mark_if_lost!(bq, st)
        end
        if device_lost(bq.ctx::VkContext)
            throw(LavaError("vkWaitSemaphores",
                            "device was lost while waiting for timeline $target " *
                            "(counter stuck at $(query_timeline(bq)), " *
                            "$(length(bq.in_flight)) batch(es) in flight)",
                            "The GPU faulted — a dispatch wrote out of bounds or hung. " *
                            "Raising `bq.flush_timeout_ns` cannot help; call " *
                            "`vk_reset_device!()` to reinitialize. To find the " *
                            "dispatch: `journalctl -k` for an NVIDIA Xid names the fault " *
                            "class, and " *
                            "`vk_reset_device!(debug = DebugConfig(gpu_av = true, " *
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
                            "timeline value $target on $(length(bq.in_flight)) in-flight batch(es)\n" *
                            flush_stall_report(bq, target),
                            "A dispatch is not completing. Set `ctx.diag.dispatch_logging = true` " *
                            "(and `ctx.diag.dispatch_log_file` to keep it across a restart) to see which " *
                            "kernel, or raise `bq.flush_timeout_ns` if the work is genuinely this long."))
        end
    end
    sweep_retired_batches!(bq)
    # The queue is drained here, so nothing recorded next can race anything
    # recorded before: the elision tracker starts empty again.
    reset_barrier_elision!(bq)
    check_validation_errors!("vk_flush!")
    return
end

# ── Unified pin + sync primitive ──────────────────────────────────────────
#
# Two phases, two functions.  Every object a dispatch touches goes through:
#
#   1. `pin!(batch, obj)` — called from the @generated arg-pack walker and
#      from dispatch entry points (rt_dispatch!, vk_draw!).  Idempotent
#      insert into `batch.pinned::IdSet`.  User never calls this.
#
#   2. `sync_access!(batch, obj)` — invoked in `submit!` once per pinned
#      object, before queue_submit_2.  Default is no-op.  `VkManagedBuffer`
#      specialization updates `last_write` and inserts a cross-queue timeline-
#      semaphore wait when the prior writer was on a different queue.
#
# Overloadable per type: a new resource kind adds `pin!(batch, ::MyRes)`
# (optional; generic method already handles it) and `sync_access!(batch,
# ::MyRes)` (optional; default no-op).  No edits to the pack walker or to
# submit! required.

"""
    pin!(batch::CommandBatch, obj) -> nothing

Keep `obj` alive until this batch's timeline signals.  Idempotent: repeated
calls with the same `obj` (within one batch) are no-ops.  `VkManagedBuffer`
specialization additionally asserts the ctx invariant so cross-context use
surfaces immediately.

Called from `pack_args_direct!` at each buffer-typed leaf, and directly at
dispatch entry points for objects not in the kernel arg tuple (pipelines,
closures, RT AS handles, indirect buffers).
"""
# `===` in an explicit loop rather than `obj in batch.pinned`, and the reason is
# NOT the one an earlier version of this comment gave. It claimed `pinned` was a
# `Vector{Any}` whose `in` fell back to a generic `==`. It is an `IdSet{Any}`,
# whose `in` is already identity-based and O(1) — verified directly:
# `big1 in IdSet[big2]` is `false` for equal-content distinct arrays. The swap
# fixed no correctness bug; there was none.
#
# What it did buy is 48.8 bytes per dispatch, measured — from boxing at this
# call site, where `obj` is not inferred, rather than from `in` itself (both
# forms allocate zero when the argument is concretely typed).
#
# THE TRADE IS SIZE-DEPENDENT, so it is written down. The scan is O(n) where the
# hash is O(1); measured on this device:
#
#     |pinned|     IdSet `in`     === scan
#            4       0.221 us      0.04 us
#           64       0.21          0.10
#          256       0.22          0.33     <- crossover
#         4096       0.11          2.37
#
# A Whisper encode holds **129** entries in the open batch, i.e. essentially at
# the crossover: this is break-even today, chosen for the bytes. It degrades
# quadratically if batches grow, and `auto_submit_threshold` (64) is what sets
# that. **Raise the threshold and this should go back to `in`.**
@inline function pin!(batch::CommandBatch, obj)
    for x in batch.pinned
        x === obj && return nothing
    end
    push!(batch.pinned, obj)
    return nothing
end

@inline function pin!(batch::CommandBatch, buf::VkManagedBuffer)
    bq = batch.bq::BatchQueue
    @assert buf.ctx === bq.ctx  "cross-ctx buffer use forbidden"
    # Same trade as the method above, for the same reason: `in` on this `IdSet`
    # is identity too, so this is about the boxing, not about correctness.
    for x in batch.pinned
        x === buf && return nothing
    end
    push!(batch.pinned, buf)
    return nothing
end

"""
    sync_access!(batch::CommandBatch, obj) -> nothing

Apply access semantics for `obj` at `submit!` time.  Default: no-op (plain
GC pinning was enough).  Overload for resource types that need GPU-side
synchronization.

The `VkManagedBuffer` specialization:
  • If `buf.last_write` was set by a different queue, pushes a timeline-
    semaphore wait onto `batch.wait_semaphores` so the submit blocks on the
    prior queue's signal.
  • Updates `buf.last_write = (bq, batch.signal_value)`.
"""
@inline sync_access!(::CommandBatch, _) = nothing

@inline function sync_access!(batch::CommandBatch, buf::VkManagedBuffer)
    # Any buffer reaching sync_access! must be ALIVE at submit time — if a
    # dead buffer's state transitioned to DEFERRED/DEAD before we got here,
    # the GPU is about to read freed memory.  Trip loudly.
    @assert (@atomic :acquire buf.state) == BUF_STATE_ALIVE  "sync_access!: buffer is not ALIVE (state=$(@atomic :acquire buf.state)) — use-after-free; size=$(buf.size) pool_offset=$(buf.pool_offset) pooled=$(buf.pool_block !== nothing)"
    bq = batch.bq::BatchQueue
    lw = @atomic :acquire buf.last_write
    if lw !== nothing
        prev_bq, prev_val = lw[1]::BatchQueue, lw[2]::UInt64
        if prev_bq !== bq
            # A timeline semaphore belongs to the device that created it. Waiting
            # on one from a *different* VkContext hands device B a handle from
            # device A, which the driver takes as a segfault inside
            # vkQueueSubmit2 — no Julia frame, nothing to grep for. That state
            # means two contexts exist and buffers have been mixed between them;
            # say so here rather than a hundred frames later in the driver.
            prev_bq.ctx === bq.ctx || throw(LavaError(
                "sync_access!",
                "buffer was last written on a BatchQueue from a DIFFERENT VkContext",
                "Two Vulkan devices are live and one buffer has been used on both. " *
                "Allocate and use the buffer under a single context."))
            push!(batch.wait_semaphores,
                (prev_bq.timeline_sem, prev_val, STAGE2_ALL_COMMANDS))
        end
    end
    @atomic :release buf.last_write = (bq, batch.signal_value)
    return nothing
end

# LavaArray forwarder lives co-located with its type def in
# array/lavaarray.jl — it unwraps to VkManagedBuffer so the leaf
# pin!/sync_access! do the work.

"""
    wait_for_write(buf::VkManagedBuffer)

Block the calling thread until the GPU has finished the latest write to
`buf` (per `buf.last_write`). Used by CPU-side readbacks. No-op if the
buffer was never written by any queue.
"""
function wait_for_write(buf::VkManagedBuffer)
    lw = @atomic :acquire buf.last_write
    lw === nothing && return nothing
    bq, val = lw[1]::BatchQueue, lw[2]::UInt64
    # If the target signal value hasn't been submitted yet (there's still an
    # active recording on that bq whose signal_value >= val), the semaphore
    # would never reach it.  Flush that bq first to force the submit.
    active = bq.active_batch
    if active !== nothing && active.signal_value <= val
        flush!(bq, bq.device)
    end
    # Any value <= current counter is already signalled; skip the wait to
    # avoid a pointless syscall.  query_timeline rethrows on a
    # healthy-device failure (no silent "pretend not yet" fallback).
    current = query_timeline(bq)
    if current >= val
        return nothing
    end
    wait_semaphores!(bq, VK.SemaphoreWaitInfo([bq.timeline_sem], [val]))
    return nothing
end

"""
    vk_flush!(bq::BatchQueue)
    vk_flush!(ctx::VkContext)       # flushes ctx.default_bq

Flush a specific batch queue.  Always spell the queue (or its ctx) explicitly —
zero-arg convenience forms have been removed per the "Explicit arguments over
implicit state" rule.
"""
function vk_flush!(bq::BatchQueue)
    ctx = bq.ctx::VkContext
    device_lost(ctx) && throw(LavaError("command flush", "Vulkan device lost",
        "Call vk_reset_device!() to reinitialize, or restart Julia session."))
    flush!(bq, bq.device)
    return
end
vk_flush!(ctx::VkContext) = vk_flush!(ctx.default_bq)

# ── Buffer copy inside the batch ────────────────────────────────────────
#
# Replaces the legacy `one_shot_copy` / `append_copy_and_flush!` pair.  All
# buffer-to-buffer transfers record a `cmd_copy_buffer` command into the
# active CommandBatch, pin both source and destination into `batch.pinned`,
# and insert a shader-write → transfer-read barrier when necessary.  The
# caller decides whether to flush afterwards.
#
# Pinning the VkManagedBuffer wrappers routes through sync_access! at submit,
# which inserts the cross-queue timeline-semaphore wait if src/dst were last
# written by a different queue — so the transfer automatically observes any
# prior producer regardless of which queue it ran on.

"""
    cmd_copy_buffer!(bq, src, dst, nbytes; src_off=0, dst_off=0)

Record a GPU→GPU buffer copy into `bq`'s active CommandBatch.  `src` and
`dst` may be either `VkManagedBuffer` (pinned + sync-tracked) or raw
`VK.Buffer` handles (no pinning — caller must keep them alive).
"""
function cmd_copy_buffer!(bq::BatchQueue, src, dst, nbytes::Integer;
                          src_off::Integer=0, dst_off::Integer=0)
    # Recording is single-writer for the same reason dispatch is, and this call
    # is where an upload first touches the command buffer — `copy_buffer!`'s
    # `flush!` on the next line already asserts it, but by then the driver has
    # dereferenced a command buffer it does not own and the process is gone with
    # SIGSEGV inside vkCmdPipelineBarrier instead of a Julia error naming the
    # thread. Check before the driver sees it, not after.
    @assert Threads.threadid() == bq.owning_thread  "BatchQueue is single-writer; cross-thread copy forbidden (owner=$(bq.owning_thread), caller=$(Threads.threadid()))"
    batch = ensure_active_batch!(bq)
    cmd = batch.cmd_buf
    # A copy writes memory the tracker never saw — but the post-copy barrier
    # below already orders that write against every later shader read, so the
    # tracker can simply start clean rather than poison. (Poisoning here would
    # force a redundant barrier on the next dispatch and, worse, on every
    # dispatch until one fired.)
    bq.barrier_elision && reset_barrier_elision!(bq)

    # Barrier: shader writes → transfer read. Only needed if we already
    # recorded dispatches into this batch (barrier across those writes).
    if batch.dispatch_count > 0
        src_stage = batch.last_was_rt ?
            VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR :
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT
        barrier_ref = Ref(VkMemoryBarrier(
            VK_STRUCTURE_TYPE_MEMORY_BARRIER, C_NULL,
            VkAccessFlags(VK_ACCESS_SHADER_WRITE_BIT),
            VkAccessFlags(VK_ACCESS_TRANSFER_READ_BIT)))
        GC.@preserve barrier_ref begin
            ccall(barrier_fptr(bq), Cvoid,
                  (Ptr{Nothing}, VkPipelineStageFlags, VkPipelineStageFlags, VkDependencyFlags,
                   UInt32, Ptr{VkMemoryBarrier}, UInt32, Ptr{Nothing}, UInt32, Ptr{Nothing}),
                  cmd.vks,
                  VkPipelineStageFlags(src_stage),
                  VkPipelineStageFlags(VK_PIPELINE_STAGE_TRANSFER_BIT),
                  VkDependencyFlags(0),
                  UInt32(1), barrier_ref,
                  UInt32(0), C_NULL,
                  UInt32(0), C_NULL)
        end
    end

    src_vkbuf = src isa VkManagedBuffer ? src.buffer : src
    dst_vkbuf = dst isa VkManagedBuffer ? dst.buffer : dst
    region = VK.BufferCopy(UInt64(src_off), UInt64(dst_off), UInt64(nbytes))
    VK.cmd_copy_buffer(cmd, src_vkbuf, dst_vkbuf, [region])

    # Barrier: transfer write → everything after. `record_dispatch!`'s
    # pre-dispatch barrier is COMPUTE→COMPUTE with src_access=SHADER_WRITE, so
    # it does NOT order this transfer's write against a later shader read —
    # and `batch.dispatch_count` doesn't advance here, so a copy recorded as
    # the first command in a batch wouldn't even get that barrier. Without
    # this, callers have to `flush!` (a blocking host-side GPU drain) purely to
    # get ordering; device→device `copyto!` used to do exactly that and it cost
    # 45% of a DNNKernels inference step.
    #
    # Guarded on the function pointer: unlike the pre-barrier above (which only
    # runs once a dispatch has been recorded, by which time the device is
    # certainly up) this one can be the very first Vulkan command a process
    # records, and calling through a null `vkCmdPipelineBarrier` is a segfault
    # inside the driver rather than an error.
    barrier_post = Ref(VkMemoryBarrier(
        VK_STRUCTURE_TYPE_MEMORY_BARRIER, C_NULL,
        VkAccessFlags(VK_ACCESS_TRANSFER_WRITE_BIT),
        VkAccessFlags(VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT |
                      VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT)))
    barrier_fptr(bq) == C_NULL || GC.@preserve barrier_post begin
        ccall(barrier_fptr(bq), Cvoid,
              (Ptr{Nothing}, VkPipelineStageFlags, VkPipelineStageFlags, VkDependencyFlags,
               UInt32, Ptr{VkMemoryBarrier}, UInt32, Ptr{Nothing}, UInt32, Ptr{Nothing}),
              cmd.vks,
              VkPipelineStageFlags(VK_PIPELINE_STAGE_TRANSFER_BIT),
              VkPipelineStageFlags(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                                   VK_PIPELINE_STAGE_TRANSFER_BIT),
              VkDependencyFlags(0),
              UInt32(1), barrier_post,
              UInt32(0), C_NULL,
              UInt32(0), C_NULL)
    end

    # Pin + sync-track the VkManagedBuffers so the transfer respects any
    # prior cross-queue writer via batch.wait_semaphores.
    src isa VkManagedBuffer && pin!(batch, src)
    dst isa VkManagedBuffer && pin!(batch, dst)
    return nothing
end

# ── Error Reporting ──

"""Throw a LavaError enriched with recent validation layer messages and dispatch log."""
function throw_with_validation_context(call_name::String, err_result,
        dispatch_count::Int=0, last_was_rt::Bool=false,
        bq::Union{Nothing,BatchQueue}=nothing)
    # Re-enable dispatch logging so the next run captures debug info. `bq` when
    # the caller has one — all three currently do — and the current context
    # otherwise, since this is also reachable from a raw `VkResult` check.
    let c = bq === nothing ? VK_CONTEXT_REF[] : bq.ctx::VkContext
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

    dlog = bq === nothing ? String[] : (bq.ctx::VkContext).diag.dispatch_log
    dispatch_detail = if isempty(dlog)
        "No dispatches logged."
    else
        "Recent dispatch log (last $(length(dlog))):\n" *
        join(["  $d" for d in dlog], "\n")
    end

    total = bq === nothing ? 0 : (bq.ctx::VkContext).diag.total_dispatches[]
    # Which kernel, on the queue that failed. Read process-wide these named
    # whatever dispatched last anywhere, so a two-queue session could attribute
    # one queue's DEVICE_LOST to another queue's kernel.
    prev_info = bq === nothing ? "" : bq.prev_dispatch_info
    curr_info = bq === nothing ? "" : bq.last_dispatch_info
    throw(LavaError(
        call_name,
        """$vk_err after $dispatch_count dispatches in batch ($total total, last_was_rt=$last_was_rt)
Crashed batch dispatch: $prev_info
Triggered by recording: $curr_info
$validation_detail
$dispatch_detail""",
        "DEVICE_LOST usually means invalid SPIR-V, out-of-bounds BDA access, or GPU timeout (Xid 109). Check dispatch log above for the crashing kernel. Call vk_reset_device!() to reinitialize."
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
