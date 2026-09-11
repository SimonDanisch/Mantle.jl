# The Metal device, and the eleven primitives `Mantle.Pool` asks of one.
#
# This is the whole of what the allocator needs. Everything above it — which
# block, what offset, when to grow, when to coalesce, when to release — is
# `Mantle.Pool`'s and is shared with every other backend; `test/test_pool.jl`
# drives 8,927 assertions of it against a device that allocates nothing at all.
# A backend that finds itself making placement decisions here has taken work
# that is not its own.

"""
    MetalDevice([mtldevice])

A Metal device: the `MTLDevice`, one submission queue, one pool.

Parametric in the queue because that is the only thing two Metal generations
disagree about. Allocation, storage modes, budgets, capabilities and the eleven
pool primitives below are the same whether work goes out through an
`MTLCommandQueue` or an `MTL4CommandQueue`, and none of them names either.

The parameter is not there for speed — Julia specialises a method on the
concrete argument type, so `openrun(d::MetalDevice)` compiles exactly as
concrete with the parameter as it did without one. It is there so both
generations can be exercised on ONE machine. Choosing at precompile time from
`supportsMTL4CommandQueue` would read as tidier and would mean this Mac can
only ever run one of the two paths, with the other going untested until someone
else's machine finds the regression. Same reasoning as the no-vendor-conditional
rule: one build has to be right everywhere, and the way you know is that both
halves run here.
"""
mutable struct MetalDevice{Q} <: Device
    dev::MTL.MTLDevice
    # How work reaches the GPU, and how the pool learns the GPU is done with it.
    queue::Q
    pool::Pool
    # Built on first ask and kept: reading `threadExecutionWidth` needs a
    # compiled pipeline, and `gemv` asks per launch.
    caps::Union{Nothing,DeviceCaps}
    # The hardware-traversal pipeline (`trace.jl`). Compiled from MSL source at
    # runtime, which runs the Metal frontend and costs tens of milliseconds, so
    # it is built on first trace and kept for the life of the device.
    trace_pipeline::Union{Nothing,MTL.MTLComputePipelineState}
end

"""
Submission the pre-MTL4 way: through Metal.jl's own batched command queue.

Mantle does not own the submissions on this path. Compute goes out through a
`Metal.BatchedCommandQueue`, so this backend never sees the command buffer that
reads a pooled region and has nothing to signal an `MTLSharedEvent` from.

The first version kept the Vulkan shape anyway — `next` beside a
`signaledValue` — and since nothing ever advanced `next`, `fence` returned 0 and
`passed(d, 0)` was `0 >= 0`: **every retired region looked finished the instant
it was retired**, so the pool recycled memory that in-flight kernels were still
reading. It only showed once a session allocated enough to reuse anything, which
is why one scene rendered correctly and the next came out NaN.

So this timeline counts *retirements*, not submissions, and the only thing that
advances `completed` is a real `Metal.synchronize()`. That is conservative —
reuse costs a device sync — but it is the honest answer to "has the GPU finished
with these bytes" from a queue this backend cannot observe. An `MTL4Queue` can
answer it properly, because there Mantle does own the submission.

DELETED with the split: `event::MTLSharedEvent` and `recording::Bool`. Nothing
read either. The event was kept "for the hardware ray tracing path", which
builds its own command buffer off `mtl` and waits on it directly; `recording`
was to have made `fence` conservative while that command buffer was open, and
never got a reader.
"""
mutable struct LegacyQueue
    # THE queue. One per device, Mantle's, and the only one anything on this
    # backend submits to: replays and kernel launches through the batch below,
    # acceleration-structure builds and the traversal pipeline through command
    # buffers of their own.
    mtl::MTL.MTLCommandQueue
    # Metal.jl's batching over it, and the reason the queue has to be ours. Its
    # `global_queue(dev)` is TASK-LOCAL — a task that has not made one gets a
    # fresh `MTLCommandQueue`, and Metal.jl keeps a residency set per queue. A
    # graph whose blocks were made resident on one task's set and replayed from
    # another's reads zeros and drops its writes, with nothing reported. So the
    # device makes one and `openrun` adopts it for whatever task is running.
    bq::Metal.BatchedCommandQueue
    # `next` is the fence value the next retirement gets; `completed` is the
    # highest value a `Metal.synchronize()` has covered. A region is reusable
    # only when its fence is <= `completed`.
    next::UInt64
    completed::UInt64
end

function LegacyQueue(dev::MTL.MTLDevice)
    mtl = MTL.MTLCommandQueue(dev)
    mtl.label = "Mantle"
    return LegacyQueue(mtl, Metal.BatchedCommandQueue(mtl), UInt64(0), UInt64(0))
end

"""
    metalqueue(mtldevice) -> queue

Which submission generation this device gets. The one place that decides, so
that a test can decide differently by handing `MetalDevice` a queue directly.
"""
#
# LEGACY, on a machine that has an MTL4 queue, and the reason is one open defect
# rather than a preference.
#
# `MTL4Queue` is complete and is exercised by the whole suite: gates, the
# timeline, residency, the ordering against the other queue, replays that do not
# overlap. It renders RayDemo's crown correctly with hardware traversal at every
# size tried up to 2000x2800 and at bounce depth 100 — with ONE sample per frame.
# At sixteen samples per frame it stops: after six or seven submissions the
# timeline stops advancing, the driver reports no failure through
# `MTL4CommitFeedback`, and nothing appears in the system log. The trigger is the
# sample count and not the depth or the resolution (depth 8 at sixteen samples
# hangs; depth 100 at one sample does not), which points at many replays of one
# recording issued back to back — but the ordering that was supposed to cover
# that is in place and a twelve-deep version of it passes in the suite, so the
# cause is not yet known and the default must not be a path that can hang a real
# scene.
#
# A caller asks for it by name — `MetalDevice(mtldev, MTL4Queue(mtldev))` — which
# is what the parameter is for, and how the suite runs both here.
#
# A replayed segment is otherwise IDENTICAL on the two: one
# `executeCommandsInBuffer:`, and the barrier bit a command carries is honoured on
# both. That was measured rather than assumed — 40 chained dispatches, each adding
# one, come out at exactly 40 on an MTL4 encoder from a single execute. A run that
# read 9 was not a barrier being ignored; it was a replay reaching memory that no
# residency set this queue had ever been given, which is what `adoptqueue!` is for.
metalqueue(dev::MTL.MTLDevice) = LegacyQueue(dev)

MetalDevice(mtldev::MTL.MTLDevice, q) = MetalDevice(mtldev, q, Pool(), nothing, nothing)
MetalDevice(mtldev::MTL.MTLDevice = Metal.device()) = MetalDevice(mtldev, metalqueue(mtldev))

# One device per process, cached: a second `MetalDevice` would mean a second
# `Pool` over the same `MTLDevice`, which is two allocators over one memory.
const METAL_DEVICE = Ref{Union{Nothing,MetalDevice}}(nothing)
function Device(::MetalAPI; select = nothing)
    if select === nothing
        d = METAL_DEVICE[]
        d === nothing || return d
        # `MANTLE_DEVICE` names the default the way it does on Vulkan; without
        # it Metal.jl's own choice stands.
        s = get(ENV, "MANTLE_DEVICE", nothing)
        mtl = s === nothing ? Metal.device() :
              Metal.devices()[selectdevice(s, devices(MetalAPI()))]
        return defaultdevice!(MetalDevice(mtl))
    end
    # A device asked for BY NAME is not the process device and does not adopt the
    # queue — see `defaultdevice!`.
    return MetalDevice(Metal.devices()[selectdevice(select, devices(MetalAPI()))])
end
# One entry per `MTLDevice`. Apple silicon reports unified memory, which is what
# `:integrated` means here; a discrete part (an Intel Mac with a Radeon, an
# eGPU) does not.
devices(::MetalAPI) = [DeviceInfo(i, String(d.name), d.hasUnifiedMemory ? :integrated : :discrete, "Metal")
                       for (i, d) in enumerate(Metal.devices())]
"""
Make `d` the device this process hands out — and the one Metal.jl launches on.

`adoptqueue!` HERE, and not in the constructor and not only in `openrun`. Not in
the constructor, because a second `MetalDevice` built for a test or a capability
query would take the queue away from the process device. Not only in `openrun`,
because an upload is a blit: `Buffer(dev, data)` before the first run would go to
whatever queue Metal.jl handed the running task, which is ordered against the
replay by nothing at all — it read as a first frame that produced nothing while
every frame after it was right.
"""
defaultdevice!(d::MetalDevice) = (METAL_DEVICE[] = d; adoptqueue!(d); d)

# `Device(backend)` — how a caller holding a KernelAbstractions backend gets the
# Mantle device for it, without naming an API marker.
Device(::Metal.MetalBackend) = Device(MetalAPI())
pool(d::MetalDevice) = d.pool
backend(::MetalDevice) = Metal.MetalBackend()

# ── What memory a request needs ───────────────────────────────────────────────
#
# On Vulkan the constraint is a bitmask: buffer usage flags, unioned across an
# arena's tenants, because a `VkBuffer` must be created with every use it will
# be put to. **Metal has no such thing.** A `MTLBuffer` is bytes; what a shader
# does with it is decided at bind time, not at creation.
#
# So the one property that does constrain sharing is the STORAGE MODE, and that
# is what a Metal constraint is. `Shared` is visible to CPU and GPU, `Private`
# is GPU-only and faster on discrete parts. On Apple silicon memory is unified
# and `Shared` costs nothing, which is why it is the default here — and why
# `deviceview` can hand back a real array over the same bytes.

# A storage mode is a TYPE in Metal.jl (`Metal.SharedStorage`), not an enum
# value — that is what `MTLBuffer`'s `storage` keyword takes, and it is what a
# constraint is here. `MTL.MTLStorageModeShared` is the *runtime* enum a buffer
# reports back; the two are not interchangeable.
"""The storage mode a set of transients needs."""
constraintof(::MetalDevice, ::Buffers, ts) = Metal.SharedStorage
constraintof(::MetalDevice, ::Images, ts) = Metal.PrivateStorage
constraintof(::MetalDevice, ::Persistent, ts) = Metal.SharedStorage

"""
How accessible a storage mode is, as an order.

`Shared` can serve anything: CPU and GPU both reach it. `Private` can serve only
requests that never touch the CPU. This ordering is what makes
[`compatible`](@ref) and [`mergeconstraints`](@ref) answerable — the Vulkan
backend gets the same two answers out of a bitmask, `&` and `|`, and the shape
of the question is identical even though the encoding is not.
"""
permissiveness(::Type{Metal.SharedStorage})  = 3
permissiveness(::Type{Metal.ManagedStorage}) = 2
permissiveness(::Type{Metal.PrivateStorage}) = 1
permissiveness(::Nothing) = 0

# A block may host a request when its memory is at least as accessible.
compatible(::MetalDevice, blk, req) = permissiveness(blk) >= permissiveness(req)

# An arena must serve every tenant, so it takes the most accessible mode any of
# them asked for. Unlike Vulkan's image case this can never fail: `Shared`
# satisfies everything, so there is always a mode that works.
mergeconstraints(::MetalDevice, kind, a, b) =
    permissiveness(a) >= permissiveness(b) ? a : b

# ── Allocation ────────────────────────────────────────────────────────────────

"""
    rawalloc(dev, kind, bytes, storage) -> MTLBuffer

One device allocation. The pool suballocates within it and never asks for
another until this one cannot serve a request.

`max(bytes, 1)` because Metal rejects a zero-length buffer and the pool is
entitled to ask for an empty block — the same guard the host and Vulkan
backends have.
"""
function rawalloc(d::MetalDevice, ::Union{Buffers,Persistent}, bytes::Int, storage)
    mode = storage === nothing ? Metal.SharedStorage : storage
    buf = MTL.MTLBuffer(d.dev, max(bytes, 1); storage = mode)
    buf === nothing && throw(OutOfMemoryError())
    return buf
end

"""
    rawalloc(dev, ::Images, bytes, storage) -> MTLHeap

The image arena, which is a heap rather than a buffer.

A texture over buffer memory is a LINEAR texture and an Apple GPU cannot render
into one, so the image arena cannot be the `MTLBuffer` every other arena is. A
`MTLHeapTypePlacement` heap is the same bargain in Metal's vocabulary: one
device allocation, the caller picks the offsets, and two resources placed at
overlapping offsets share the bytes — which is what the graph's aliasing pass
decided when it gave two targets the same offset.

`Automatic` would have been the easier heap type and is the wrong one: it
chooses its own offsets, so the placement Mantle computed would be advisory.
"""
function rawalloc(d::MetalDevice, ::Images, bytes::Int, storage)
    desc = MTL.MTLHeapDescriptor()
    desc.size = max(bytes, 1)
    desc.storageMode = storage === Metal.PrivateStorage ? MTL.MTLStorageModePrivate :
                                                          MTL.MTLStorageModeShared
    desc.type = MTL.MTLHeapTypePlacement
    # DELETED in phase 1.8: the reason given for `Tracked`. See `metal/ka.jl`.
    desc.hazardTrackingMode = MTL.MTLHazardTrackingModeTracked
    heap = MTL.MTLHeap(d.dev, desc)
    heap === nothing && throw(OutOfMemoryError())
    return heap
end

"""
Release a block's device memory.

Immediate, not left to the GC. `trim!` exists so a caller can say "I have
finished and want the memory back now", and an answer that depends on when
Julia next collects is not an answer to that — the same reasoning the Vulkan
backend's `rawfree` records.
"""
rawfree(::MetalDevice, buf::MTL.MTLBuffer) = (Metal.free(buf); nothing)
# A heap is freed by releasing it: `Metal.free` is the buffer path (it goes
# through `MTLBuffer`'s own deallocation), and a heap's textures die with it.
rawfree(::MetalDevice, heap::MTL.MTLHeap) = (Metal.ObjectiveC.release(heap); nothing)

# ── The timeline ──────────────────────────────────────────────────────────────
#
# The queue's, not the device's: what a fence MEANS is precisely what the two
# generations disagree about. A `LegacyQueue` counts retirements and can only
# answer `passed` by draining the device; an `MTL4Queue` signals a shared event
# per submission and answers it by reading `signaledValue`. The pool asks the
# same three questions either way.

fence(d::MetalDevice) = fence(d.queue)
passed(d::MetalDevice, f) = passed(d.queue, f)
waitfor(d::MetalDevice, f) = waitfor(d.queue, f)

# DELETED in phase 1.7: the docstring, which said "Read, never forced" of a
# function whose first line is `q.next += 1`. Whether a fence is a read or an
# allocation is phase 2.2's question, since it is the same question as what a
# submission holds.
function fence(q::LegacyQueue)
    q.next += UInt64(1)
    return q.next
end

passed(q::LegacyQueue, f) = UInt64(f) <= q.completed

"""
Wait for the timeline to reach `f`, unless nothing has been submitted that will.

`f > next` names a value belonging to a command buffer still being recorded.
Blocking on it would wait for a submission that has not happened, and the one
caller is an allocation — so it declines, and the pool grows instead. That costs
a block and never a deadlock.
"""
function waitfor(q::LegacyQueue, f)
    passed(q, f) && return true
    # Drain everything: `Metal.synchronize()` waits on this task's queue, which
    # is the one Metal.jl dispatches kernels on. There is no finer-grained wait
    # available to this generation, and a coarse-but-correct one is the point —
    # the alternative was handing back live memory.
    issued = q.next
    Metal.synchronize()
    q.completed = max(q.completed, issued)
    return true
end

# ── Submission ────────────────────────────────────────────────────────────────
#
# Two verbs, and with the timeline above they are the whole of what a Metal
# generation changes. `replay!` in `record.jl` is written against them and names
# no generation at all.
#
#     s   = opensubmit!(d, buffers)   # what it touches, and an encoder for it
#     ...encode...
#     tok = closesubmit!(d, s)        # end, commit, and the fence it will signal
#
# What a submission touches is an ARGUMENT to opening it rather than a verb of
# its own, because the two generations need it at different moments and only one
# of those moments is expressible after the fact: legacy declares on the encoder
# (`useResource`, which is hazard tracking as much as residency), MTL4 has to
# have the residency set complete before the command buffer declares it.

"""What one legacy submission is: the batched queue, and its compute encoder."""
struct LegacySubmission
    bq::Metal.BatchedCommandQueue
    enc::MTL.MTLComputeCommandEncoder
end

"""The compute encoder a submission records into."""
encoder(s::LegacySubmission) = s.enc

opensubmit!(d::MetalDevice, bufs) = opensubmit!(d.queue, d.dev, bufs)

# The task-local BATCHED queue rather than `q.mtl`: it is the one every ordinary
# kernel launch goes through, and a replay has to stay ordered against the
# launches around it. Asking for it also hands back an encoder that may already
# be open, which is what lets a walked pass and a replay share one command
# buffer.
#
# `useResource` is not only about residency: it is how the driver's hazard
# tracking learns that this work read and wrote those bytes, so it has to name
# the encoder and therefore happens here rather than before.
function opensubmit!(q::LegacyQueue, ::MTL.MTLDevice, bufs)
    bq = q.bq
    enc = Metal.compute_encoder(bq)
    MTL.use!(enc, bufs, MTL.ReadWriteUsage)
    return LegacySubmission(bq, enc)
end

function executesegment!(::LegacyQueue, s::LegacySubmission, rec, seg)
    if seg.slot < 0
        MTL.execute_commands!(s.enc, rec.icb, seg.first:seg.last)
    else
        MTL.execute_commands_indirect!(s.enc, rec.icb, rec.rangebuf,
                                       rec.rangeoff + 8 * seg.slot)
    end
    return nothing
end

closesubmit!(d::MetalDevice, s) = closesubmit!(d.queue, s)

function closesubmit!(q::LegacyQueue, s::LegacySubmission)
    # An indirect execution leaves the encoder's own pipeline binding undefined,
    # and `set_pipeline!` skips a set it believes is already in place. Forgetting
    # this sends the next ordinary launch through whatever the recording left.
    s.bq.last_pipeline = nothing
    Metal.note_recorded!(s.bq, 0, nothing)
    Metal.flush!(s.bq)
    # Nothing here signals anything, so the token is the retirement counter and
    # waiting on it is a device drain. See `LegacyQueue`.
    return fence(q)
end

# ── Metal 4 ───────────────────────────────────────────────────────────────────
#
# The same three verbs, on a queue Mantle owns outright.
#
# What it buys is not host time — measured 2026-09-10, an MTL4 submission is if
# anything slightly slower per call than a legacy one. It is the TIMELINE. A
# `LegacyQueue` cannot answer "has the GPU finished with these bytes" without
# `Metal.synchronize()`, because Metal.jl owns the command buffer that read them
# and there is nothing for this backend to signal from; every pooled region
# therefore costs a device drain to reuse, which is why the retiring list on this
# backend grows and stays. An MTL4 queue signals an `MTLSharedEvent` per submit,
# so `passed` is a load of `signaledValue` and reuse costs nothing.

"""How many frames may be recording or in flight at once."""
const MTL4_FRAMES_IN_FLIGHT = 3

"""
Submission the Metal 4 way: Mantle owns the queue, the command buffers and the
timeline.

**The allocator is a command pool.** `reset` hands back the storage a command
buffer recorded into, so resetting one the GPU has not finished with is a
use-after-free — and unlike the legacy path there is no driver bookkeeping to
save you. Hence the RING: one allocator and one command buffer per frame in
flight, each stamped with the token its last submission will signal, and
`opensubmit!` waits only when it comes back round to a slot still running. At a
steady frame rate three deep it never waits.

**Both queues, not one.** `mtl` is the ordinary `MTLCommandQueue` and it is not a
fallback: render passes, blits and acceleration-structure builds submit command
buffers of their own, Metal.jl's kernel launches go through its batch, and a
residency set belongs to an `MTLCommandQueue`. What changes on this path is where
a RECORDED plan's replay goes, and what a fence means.
"""
mutable struct MTL4Queue
    q::MTL.MTL4CommandQueue
    allocs::Vector{MTL.MTL4CommandAllocator}
    cbs::Vector{MTL.MTL4CommandBuffer}
    # The token each slot's last submission will signal; 0 for never used.
    at::Vector{UInt64}
    slot::Int
    # The one-element array `commit:count:` takes, kept so that a replayed frame
    # allocates nothing — the invariant `test_record_metal.jl` pins.
    batch::Vector{MTL.MTL4CommandBuffer}
    # The timeline. `signaledValue` is what the GPU has finished; `next` is the
    # value the last submission was given.
    event::MTL.MTLSharedEvent
    next::UInt64
    mtl::MTL.MTLCommandQueue
    bq::Metal.BatchedCommandQueue
    # How a failed submission is heard about. MTL4 has no error on a command
    # buffer and no `waitUntilCompleted`, so without this a fault is a value that
    # never arrives and a `waitfor` that never returns, with nothing in the system
    # log. See `MTL.MTL4Feedback`.
    feedback::MTL.MTL4Feedback
    # The residency set of `mtl`, and the one this queue was given. There is
    # exactly one because there is exactly one `MTLCommandQueue`: `adoptqueue!`
    # makes Metal.jl use it too, so a buffer Metal.jl makes resident and a buffer
    # the replay reaches are in the same set. Getting that wrong is silent —
    # unmapped reads return zero and writes are dropped, the frame takes just as
    # long, and `MTL_SHADER_VALIDATION=1` hides it by making everything resident.
    resset::Union{Nothing,MTL.MTLResidencySet}
end

function MTL4Queue(dev::MTL.MTLDevice)
    n = MTL4_FRAMES_IN_FLIGHT
    q = MTL.MTL4CommandQueue(dev)
    mtl = MTL.MTLCommandQueue(dev)
    mtl.label = "Mantle"
    # `BatchedCommandQueue` installs the residency set for its queue, so asking
    # for it afterwards returns that one rather than making a second.
    bq = Metal.BatchedCommandQueue(mtl)
    resset = Metal.can_use_residency_sets(dev) ?
        Metal.install_queue_residency!(mtl, dev) : nothing
    # Both: on the queue so every submission has it, and on each command buffer
    # in `opensubmit!` because that is what the API asks for.
    resset === nothing || MTL.add_residency_set!(q, resset)
    return MTL4Queue(q,
                     [MTL.MTL4CommandAllocator(dev) for _ in 1:n],
                     [MTL.MTL4CommandBuffer(dev) for _ in 1:n],
                     zeros(UInt64, n), 0,
                     Vector{MTL.MTL4CommandBuffer}(undef, 1),
                     MTL.MTLSharedEvent(dev), UInt64(0), mtl, bq,
                     MTL.MTL4Feedback(), resset)
end

batchqueue(q::MTL4Queue) = q.bq

cmdqueue(q::MTL4Queue) = q.mtl

# Metal.jl's batch is still where launches and blits go, so the flush heuristic
# still has to be taken out of the frame. Same reasoning as the legacy path.
ownflush!(q::MTL4Queue, ::MTL.MTLDevice, own::Bool) =
    (Metal.own_flushes!(q.bq, own); nothing)

"""What one MTL4 submission is: a ring slot, its command buffer and its encoder."""
struct MTL4Submission
    cb::MTL.MTL4CommandBuffer
    enc::MTL.MTL4ComputeCommandEncoder
    slot::Int
end

encoder(s::MTL4Submission) = s.enc

function executesegment!(::MTL4Queue, s::MTL4Submission, rec, seg)
    if seg.slot < 0
        MTL.execute_commands!(s.enc, rec.icb, seg.first:seg.last)
        return nothing
    end
    # Before, so the dispatch that wrote this range has landed; after, so the
    # next segment does not start on top of whatever the iteration wrote.
    MTL.barrier!(s.enc)
    MTL.execute_commands_indirect!(s.enc, rec.icb, rec.rangebuf,
                                   rec.rangeoff + 8 * seg.slot)
    MTL.barrier!(s.enc)
    return nothing
end

function opensubmit!(q::MTL4Queue, dev::MTL.MTLDevice, bufs)
    # BEFORE the command buffer names the set, not after. There is no
    # `useResource` on an MTL4 encoder, so residency is membership of a set, and
    # a set the command buffer has already declared does not pick up an
    # allocation added afterwards: the replay reads unmapped memory and writes
    # nothing, with no error anywhere. It cost a day. Under
    # `MTL_SHADER_VALIDATION=1` it does not reproduce, because GPU validation
    # makes everything resident — so the first frame was correct under the
    # debug layer and silently empty without it.
    #
    # `make_persistently_resident!` is Metal.jl's own dedup: a set never gives an
    # allocation back, so "already added" is permanent and a steady frame is one
    # pointer lookup per block rather than a driver call.
    foreach(Metal.make_persistently_resident!, bufs)
    q.slot = q.slot == length(q.allocs) ? 1 : q.slot + 1
    f = q.at[q.slot]
    # Zero means the slot has never been used. Otherwise this is the one place
    # the host can block, and only when the ring has come all the way round to a
    # frame the GPU is still running.
    iszero(f) || passed(q, f) || waitfor(q, f)
    alloc = q.allocs[q.slot]
    MTL.reset!(alloc)
    cb = q.cbs[q.slot]
    MTL.begin_command_buffer!(cb, alloc)
    enc = MTL.compute_encoder(cb)
    q.resset === nothing || MTL.use_residency_set!(cb, q.resset)
    return MTL4Submission(cb, enc, q.slot)
end

"""
Close whatever this frame put on the LEGACY queue, and give back the timeline
value it will signal (`0` if it put nothing there).

Both queues are live on this path and Metal orders command buffers only WITHIN a
queue. A plan's host updates are blits, every Metal.jl kernel launch is a
dispatch, and both go to `bq` — while the replay goes to the MTL4 queue with
nothing between them. Unordered, that is not subtle: a `Buffer(dev, data)`
uploaded on the first run landed AFTER the dispatch that read it, so every frame
showed the previous frame's answer, and at 4 M elements the 32 MB of blits
overlapped a 9 ms replay and 40 chained adds came out at 10.

So the legacy side signals the same timeline the MTL4 side does, and the caller
makes its submission wait for the value. One counter, both queues.
"""
function flushlegacy!(q::MTL4Queue)
    bq = q.bq
    # A signal cannot be encoded while an encoder is open.
    Metal.end_encoder!(bq)
    cb = bq.cmdbuf
    cb === nothing && return UInt64(0)
    f = (q.next += UInt64(1))
    MTL.encode_signal!(cb, q.event, f)
    Metal.flush!(bq)
    return f
end

function closesubmit!(q::MTL4Queue, s::MTL4Submission)
    MTL.endEncoding!(s.enc)
    MTL.end_command_buffer!(s.cb)
    # Wait for EVERYTHING earlier: the legacy work this frame put on the other
    # queue, and the previous submission on this one.
    #
    # The previous submission matters because a recording is not read-only. A
    # `repeat!` gate's execution range is `(location, length)` in the recording's
    # own buffer, written by a one-thread dispatch inside the replay and read by
    # the command processor later in the same replay — so two replays of one
    # recording in flight at once are two writers of those bytes. The legacy path
    # never has to say this: everything it submits is on ONE queue, and Metal runs
    # a queue's command buffers in commit order. MTL4 has no hazard tracking
    # between command buffers at all, so overlapping replays corrupt each other's
    # ranges and the command processor stops on one that no longer names a valid
    # range — no error, no fault in the log, just a submission that never signals.
    #
    # It took the crown to find, because it needs all three at once: a gated plan,
    # several submissions in flight (a 16-sample frame issues them back to back),
    # and frames long enough to actually overlap. Every one of those alone is
    # fine, which is why the suite and every smaller scene passed.
    #
    # The cost is no cross-frame overlap on the GPU — which is exactly what the
    # legacy queue already does, so it is not a regression against it.
    prev = q.next
    fl = flushlegacy!(q)
    w = iszero(fl) ? prev : fl
    iszero(w) || MTL.wait_for_event!(q.q, q.event, w)
    q.batch[1] = s.cb
    MTL.commit!(q.q, q.batch, q.feedback)
    f = (q.next += UInt64(1))
    MTL.signal_event!(q.q, q.event, f)
    q.at[s.slot] = f
    # And the bridge back. `flushlegacy!` above orders the legacy queue BEFORE
    # this submission; this orders everything that reaches the legacy queue after
    # it — the readback blit an `Array(x)` issues, the next frame's uploads —
    # AFTER it. An empty command buffer whose only content is a wait, which is
    # `vkQueueSubmit` with a wait semaphore and nothing else.
    #
    # Without it a device-written count read straight after `run!` came back
    # zero: `Metal.synchronize()` waits on the plain queue, and the work was on
    # the other one. Safe to commit directly because `flushlegacy!` just closed
    # the batch, so no command buffer of Metal.jl's is open to be reordered
    # against.
    bridge = MTL.MTLCommandBuffer(q.mtl)
    MTL.encode_wait!(bridge, q.event, f)
    MTL.commit!(bridge)
    return f
end

"""
End a walked frame on the MTL4 path.

The passes went through Metal.jl's batch and its own command buffers, which are
on `mtl` and not on the MTL4 queue. Signalling the timeline from that command
buffer is what keeps ONE counter across both: the next MTL4 submission waits for
this value before it runs.

A frame that encoded nothing has no command buffer to signal from and no work to
cover, so it keeps the token it already had.
"""
function closeframe!(q::MTL4Queue, ::MTL.MTLDevice)
    f = flushlegacy!(q)
    # Nothing was encoded, so there is no submission to cover and the token the
    # run already has still stands.
    return iszero(f) ? q.next : f
end

# ── The Metal 4 timeline ─────────────────────────────────────────────────────
#
# This is the contract `pool.jl` writes down and the Vulkan backend already
# meets: `fence` READS "everything submitted up to now", `passed` compares
# against what the GPU has signalled, and `waitfor` declines rather than block on
# a value nothing has submitted.

"""
The value the NEXT submission will signal, which is what a region retired now has
to wait for.

Not `next`, deliberately. A submission that does not signal — a render pass or an
acceleration-structure build, which commit command buffers of their own — sits
between two that do, and a token equal to the last signalled value would not
cover it. `next + 1` does, because both queues are ordered: everything already
committed runs before the submission that signals it.
"""
fence(q::MTL4Queue) = q.next + UInt64(1)

"""
Has the GPU finished everything `f` stands for?

`min(f, next)` and not `f`, which is the counterpart of `fence` handing out
`next + 1`. That token means "wait for the submission after the last one", and
when no such submission has been made there is nothing left to wait FOR: every
submission that exists is at or below `next`, so if the event has reached `next`
then everything that could be reading those bytes is done. Comparing against `f`
itself instead would leave a region retired on an idle device unreleasable until
something unrelated was submitted — which is exactly the backlog
`test_pool_metal.jl` measures.
"""
passed(q::MTL4Queue, f) = min(UInt64(f), q.next) <= q.event.signaledValue

"""
Wait for the timeline, and say whether it got there.

Never drains the device and never submits: it waits on the shared event for the
last value actually submitted, which is what `passed` compares against. A token
beyond that names a submission nobody has made, and `min` is what stops this from
being a wait nothing can satisfy.
"""
function waitfor(q::MTL4Queue, f)
    passed(q, f) && return true
    target = min(UInt64(f), q.next)
    # Bounded waits in a loop rather than one unbounded one, so that a submission
    # the driver has FAILED is reported instead of waited on forever. This is not
    # a timeout: healthy work is waited for as long as it takes, and the loop only
    # ends when the value arrives or the queue says a submission failed.
    while !MTL.waitUntilSignaledValue(q.event, target, UInt64(1000))
        why = MTL.failure(q.feedback)
        why === nothing && continue
        error("Mantle: a Metal 4 submission failed, so the timeline stopped at " *
              "$(q.event.signaledValue) and value $target will never arrive. " *
              "The driver says: $why")
    end
    return true
end

"""
    ownflush!(dev, own::Bool)

Take the decision of WHEN to submit away from Metal.jl for the duration of a run,
or give it back.

Metal.jl batches: kernel launches and blits accumulate in one open command buffer
and `maybe_autoflush!` commits it once the batch crosses 32 operations or 64 MB.
That is the right default for code that submits work without thinking about
command buffers, and it is the wrong one here, because a graph knows exactly
where its submission boundaries are — they are the frame. A plan with 140
dispatches was four command buffers rather than one, each an extra commit, and
each a submission Mantle did not make and therefore cannot count.

Cannot count is the part that matters beyond the commits. `fence` has to stand
for "everything submitted up to now"; a submission that happens between two of
Mantle's own is one no token covers, so a region retired afterwards would be
handed back while that command buffer was still reading it. One submission per
run is what makes the counter mean something.

Turning it off is a promise to flush, which `closeframe!` and `closesubmit!` keep
— every path out of `submitrun!` goes through one of them.
"""
ownflush!(d::MetalDevice, own::Bool) = ownflush!(d.queue, d.dev, own)

ownflush!(q::LegacyQueue, ::MTL.MTLDevice, own::Bool) =
    (Metal.own_flushes!(q.bq, own); nothing)

"""
    closeframe!(dev) -> UInt64

End a run that submitted through the batch rather than through a recording, and
give back the token that covers it.

A walked plan's passes each encode into the open batch (compute) or commit a
command buffer of their own (render, copy). Nothing in that path decides the
frame is over, which is what this is: with `ownflush!` holding the batch open for
the whole run, the flush here is the ONE submission that run made.
"""
closeframe!(d::MetalDevice) = closeframe!(d.queue, d.dev)

function closeframe!(q::LegacyQueue, ::MTL.MTLDevice)
    Metal.flush!(q.bq)
    return fence(q)
end

"""
    cmdqueue(dev) -> MTLCommandQueue

The plain queue, for the two paths that submit a command buffer of their own
rather than going through the batch: building an acceleration structure
(`raytracing.jl`) and compiling the traversal pipeline (`trace.jl`). Both are
setup, not frame work.
"""
cmdqueue(d::MetalDevice) = cmdqueue(d.queue)
cmdqueue(q::LegacyQueue) = q.mtl

"""
    executesegment!(dev, sub, rec, seg)

Run one segment of a recording — the only place a replay differs between the two
generations, and only for a GATED segment.

An ungated segment is one `executeCommandsInBuffer:` on both. The barrier bit a
command carries is honoured on both too, so the pass boundaries inside a segment
are the recorder's business either way; 40 chained dispatches replayed from one
execute come out at 40 on an MTL4 encoder at every size tried.

A gated segment is different because of WHERE the dependency is. Its range is
`(location, length)` in memory, written by a one-thread dispatch in the segment
before and read by the COMMAND PROCESSOR when it reaches the execute — and the
execute is a call on the encoder, not a command inside the buffer, so no bit can
order it. The legacy driver infers that hazard from `useResource`. MTL4 does no
hazard tracking at all, so it has to be said: a `barrier!` on either side, two
sends per gated segment.
"""
executesegment!(d::MetalDevice, sub, rec, seg) = executesegment!(d.queue, sub, rec, seg)

"""
    batchqueue(dev) -> Metal.BatchedCommandQueue

The device's batch: where a kernel launch, a blit and a walked pass go.

The backend never asks `Metal.global_queue` for one. That function answers with
the CURRENT TASK's queue, which is a different `MTLCommandQueue` — and therefore a
different residency set — for every task that ever runs a plan.
"""
batchqueue(d::MetalDevice) = batchqueue(d.queue)
batchqueue(q::LegacyQueue) = q.bq

"""
Make this device's queue the one Metal.jl launches on, for the running task.

Called at the head of every run. One dictionary store in task-local storage, and
it is what makes "Mantle owns the queue" true on whatever task the caller drives
the graph from.
"""
adoptqueue!(d::MetalDevice) = (Metal.adopt_queue!(d.dev, batchqueue(d)); nothing)

# ── Budgets ───────────────────────────────────────────────────────────────────

"""The largest single `MTLBuffer` this device will create."""
devicename(d::MetalDevice) = String(d.dev.name)
maxalloc(d::MetalDevice) = Int(d.dev.maxBufferLength)

"""
How much device memory a transient arena may use.

`recommendedMaxWorkingSetSize` is Apple's "stay under this and you will not be
paged", which is the same kind of number Vulkan's memory-budget extension
reports and is used the same way: an upper bound the placer checks against, not
a reservation.
"""
capacity(d::MetalDevice) = Int(d.dev.recommendedMaxWorkingSetSize)

"""
Block size for the suballocator.

64 MiB, matching the Vulkan backend. Unified memory makes a large block cheaper
here than on a discrete part — nothing is copied to reach it — but the reason
for the size is the same: one device allocation amortised over many transients.
"""
blocksize(::MetalDevice) = 64 << 20

# ── Device-wide synchronisation and the graphics answer ──────────────────────

"""
Wait for everything submitted to this device.

`Metal.synchronize()` waits on the current task's command queue, which is the
whole of what Metal exposes here — there is no `vkDeviceWaitIdle` taking a
device handle, because a `MTLDevice` does not own the submission order; its
queues do.
"""
waitidle(::Metal.MetalBackend) = Metal.synchronize()
waitidle(d::MetalDevice) = (Metal.synchronize(); nothing)

# `supports_graphics` is answered in `graphics.jl`, where the rasterisation half
# lives. It used to say `false` right here, because Metal.jl compiled Julia to
# compute kernels only — no vertex or fragment stage existed for a shader to
# become. It compiles both now.

# DELETED in phase 1.7: the body of `allocate_batch_queue!` and its message.
# It told the caller "Metal.jl has no graphics pipeline, check
# `supports_graphics` and take the compute path" on a backend where
# `supports_graphics` measurably answers `true`, fifteen lines below a comment
# saying so.
#
# A second channel is a driver fact — Vulkan takes another `VkQueue` from the
# family — and this backend has one queue, which `supports_batch_queue` is the
# question for. Throwing is the answer to asking anyway.
function allocate_batch_queue!(::Union{Metal.MetalBackend,MetalDevice})
    throw(ArgumentError(
        "Mantle: this backend has one submission channel. Ask " *
        "`supports_batch_queue(device)` before allocating a second one."))
end
