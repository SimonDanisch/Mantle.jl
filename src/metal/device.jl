# The Metal device, its submission queues, and the primitives `Mantle.Pool` asks
# of it. Placement (block, offset, growth, release) is the pool's, not this file's.

"""
    MetalDevice([mtldevice])

A Metal device: the `MTLDevice`, one submission queue, one pool. Parametric in
the queue (`LegacyQueue` or `MTL4Queue`), the only part the two Metal generations
disagree on, so both can run on one machine instead of being chosen at
precompile time.
"""
mutable struct MetalDevice{Q} <: Device
    dev::MTL.MTLDevice
    # Submission, and the timeline the pool's fences read.
    queue::Q
    pool::Pool
    # Built on first `caps` call (`caps.jl`).
    caps::Union{Nothing,DeviceCaps}
    # Per-argument access summaries keyed on the inferred signature, asked per
    # dispatch per build; a property of the code, not the graph.
    accesses::AccessCache
    # Compiled raster and mesh pipelines.
    pipelines::CompilationCache{Any,Any}
    # Metal.jl's ownership record per pool buffer, keyed by handle, shared by all
    # views of the buffer (`managedbuffer`).
    managed::Dict{UInt,Metal.Managed}
    # Validated against what the process loaded at construction (`debug.jl`).
    debug::DebugConfig
    # `retire!`d: no further allocation, and no block is freed through the driver.
    retired::Bool
    # Unreleased channels from `allocate_batch_queue!`, held so a channel outlives
    # every buffer launched on it.
    channels::Vector{Metal.ConcreteBatchedQueue}
    # Commits of released channels, so `submissions` stays monotone.
    releasedsubmits::Int
end

"""
    LegacyQueue(mtldevice)

Pre-MTL4 submission through Metal.jl's `BatchedCommandQueue`, whose command
buffers Mantle cannot signal from. The timeline counts retirements; `completed`
advances only on a drain or when `fence` finds the queue quiet.
"""
mutable struct LegacyQueue
    # The device's only queue: replays and launches go through `bq`; render
    # passes, acceleration-structure builds and traces commit their own buffers.
    mtl::MTL.MTLCommandQueue
    # Metal.jl's batch over `mtl`. Metal.jl's `global_queue` is task-local and
    # residency sets are per queue, so a replay from another task's queue reads
    # zeros and drops writes silently; `enterrun!` adopts this one for the task.
    bq::Metal.ConcreteBatchedQueue
    # `next`: the latest retirement's fence. `completed`: the highest fence known
    # done. A region is reusable when its fence is <= `completed`.
    next::UInt64
    completed::UInt64
    # Encode recorded plans every frame instead of replaying (`encodesrecordings`).
    encodes::Bool
end

function LegacyQueue(dev::MTL.MTLDevice)
    mtl = MTL.MTLCommandQueue(dev)
    mtl.label = "Mantle"
    bq = Metal.BatchedCommandQueue(mtl)
    pinpipeline!(bq)
    return LegacyQueue(mtl, bq, UInt64(0), UInt64(0), encodesrecordings(dev))
end

"""
    encodesrecordings(dev) -> Bool

Whether a legacy queue on `dev` encodes recorded plans instead of replaying their
indirect command buffer. On M1 (Apple7), `executeCommandsInBuffer:` while
WindowServer composites hangs the GPU until the watchdog resets it; encoding the
same dispatches does not. `MANTLE_METAL_ENCODE=1` or `0` overrides the default.
"""
function encodesrecordings(dev::MTL.MTLDevice)
    override = get(ENV, "MANTLE_METAL_ENCODE", "")
    isempty(override) || return override == "1"
    return MTL.is_m1(dev)
end

"""
    pinpipeline!(bq)

Keep up to 8 command buffers in flight (Metal.jl defaults to 3), pinned against
`JULIA_METAL_COMMAND_BATCHING_INFLIGHT`. 8 is the latency knee for many small
plans. Safe only because `opensubmit!` declares resources with `useResource`;
without it deeper pipelining overlaps frames and loses accumulations.
"""
pinpipeline!(bq) = Metal.inflight!(bq, 8)

"""
    metalqueue(mtldevice) -> queue

The submission generation for `mtldevice`: `LegacyQueue`. On MTL4 queues
`executeCommandsInBuffer:` stops completing after about a second of cumulative
replayed work (M5), so MTL4 encodes recorded dispatches instead of replaying
them and is opt-in via `MetalDevice(dev, MTL4Queue(dev))`.
"""
metalqueue(dev::MTL.MTLDevice) = LegacyQueue(dev)

function MetalDevice(mtldev::MTL.MTLDevice, q, debug::DebugConfig = DebugConfig())
    checkdebug(mtldev, debug)
    d = MetalDevice(mtldev, q, Pool(), nothing, AccessCache(), CompilationCache{Any,Any}(),
                    Dict{UInt,Metal.Managed}(), debug, false, Metal.ConcreteBatchedQueue[], 0)
    # Registered before anything allocates on it, so the exit hook reaches it.
    push!(METAL.live, WeakRef(d))
    return d
end
MetalDevice(mtldev::MTL.MTLDevice = Metal.device()) = MetalDevice(mtldev, metalqueue(mtldev))

"""
Process-wide Metal state. `default` is the one device `Device(MetalAPI())`
returns (a second would put a second `Pool` over the same memory). `live` holds
every device weakly so the exit hook can retire it. `reports`, `reportlock` and
`reporter` belong to the process-wide validation report block (`debug.jl`).
"""
mutable struct MetalProcess
    default::Union{Nothing,MetalDevice}
    live::Vector{WeakRef}
    reports::Vector{String}
    reportlock::ReentrantLock
    # The block given to `MTLSetReportFailureBlock`, rooted here because Metal
    # holds it only by pointer.
    reporter::Any
end
const METAL = MetalProcess(nothing, WeakRef[], String[], ReentrantLock(), nothing)

function Device(::MetalAPI; select = nothing, debug::Union{Nothing,DebugConfig} = nothing)
    if select === nothing && debug === nothing
        d = METAL.default
        d === nothing || return d
        # Metal.jl's own choice; `defaultdevice!(select)` is how a caller picks another.
        return defaultdevice!(MetalDevice(Metal.device()))
    end
    # A device selected by name or with a debug configuration is not the process
    # device and does not adopt the queue (see `defaultdevice!`).
    mtldev = select === nothing ? Metal.device() :
             Metal.devices()[selectdevice(select, devices(MetalAPI()))]
    return MetalDevice(mtldev, metalqueue(mtldev), something(debug, DebugConfig()))
end
# One entry per `MTLDevice`; `:integrated` means unified memory (Apple silicon).
devices(::MetalAPI) = [DeviceInfo(i, String(d.name), d.hasUnifiedMemory ? :integrated : :discrete, "Metal")
                       for (i, d) in enumerate(Metal.devices())]
"""
    defaultdevice!(d) -> d

Make `d` the device this process hands out and the one Metal.jl launches on.
Adopts the queue here, not in the constructor (a second `MetalDevice` would take
it from the process device) and not only in `enterrun!` (an upload before the
first run would go to the task's own queue, unordered against the replay).
"""
defaultdevice!(d::MetalDevice) = (METAL.default = d; adoptqueue!(d); d)

# The Mantle device of a KernelAbstractions backend; a backend bound to a channel
# belongs to the device that handed the channel out.
Device(b::Metal.MetalBackend) = b.queue === nothing ? Device(MetalAPI()) : deviceof(b.queue)
pool(d::MetalDevice) = d.pool
backend(::MetalDevice) = Metal.MetalBackend()

# A Metal constraint is a storage mode: an `MTLBuffer` has no usage flags, so
# nothing else limits sharing. It is a Metal.jl type (`Metal.SharedStorage`, what
# `MTLBuffer`'s `storage` keyword takes), not the `MTL.MTLStorageModeShared` enum
# a buffer reports.
"""
The storage mode a set of transients needs: `Private` for images, otherwise
`Shared`, which costs nothing on unified memory and lets `deviceview` return an
array over the bytes.
"""
constraintof(::MetalDevice, ::Buffers, ts) = Metal.SharedStorage
constraintof(::MetalDevice, ::Images, ts) = Metal.PrivateStorage
constraintof(::MetalDevice, ::Persistent, ts) = Metal.SharedStorage

"""
How accessible a storage mode is, as an order for [`compatible`](@ref) and
[`mergeconstraints`](@ref): `Shared` serves CPU and GPU requests, `Private` only
requests that never touch the CPU.
"""
permissiveness(::Type{Metal.SharedStorage})  = 3
permissiveness(::Type{Metal.ManagedStorage}) = 2
permissiveness(::Type{Metal.PrivateStorage}) = 1
permissiveness(::Nothing) = 0

# A block may host a request when its memory is at least as accessible.
compatible(::MetalDevice, blk, req) = permissiveness(blk) >= permissiveness(req)

# An arena takes the most accessible mode its tenants ask for; never fails,
# since `Shared` satisfies everything.
mergeconstraints(::MetalDevice, kind, a, b) =
    permissiveness(a) >= permissiveness(b) ? a : b

"""
    rawalloc(dev, kind, bytes, storage) -> MTLBuffer

One device allocation for the pool to suballocate. `max(bytes, 1)` because
Metal rejects a zero-length buffer and the pool may ask for an empty block.
Throws `OutOfMemoryError` when Metal returns no buffer.
"""
function rawalloc(d::MetalDevice, ::Union{Buffers,Persistent}, bytes::Int, storage)
    mode = storage === nothing ? Metal.SharedStorage : storage
    buf = MTL.MTLBuffer(d.dev, max(bytes, 1); storage = mode)
    buf === nothing && throw(OutOfMemoryError())
    return buf
end

"""
    rawalloc(dev, ::Images, bytes, storage) -> MTLHeap

The image arena: a heap, since a texture over buffer memory is linear and Apple
GPUs cannot render into one. `MTLHeapTypePlacement` so the caller's offsets hold
and resources placed at overlapping offsets alias, as the graph's aliasing pass
decided; an `Automatic` heap picks its own offsets.
"""
function rawalloc(d::MetalDevice, ::Images, bytes::Int, storage)
    desc = MTL.MTLHeapDescriptor()
    desc.size = max(bytes, 1)
    desc.storageMode = storage === Metal.PrivateStorage ? MTL.MTLStorageModePrivate :
                                                          MTL.MTLStorageModeShared
    desc.type = MTL.MTLHeapTypePlacement
    # `Tracked`, for the reason `metal/ka.jl` gives.
    desc.hazardTrackingMode = MTL.MTLHazardTrackingModeTracked
    heap = MTL.MTLHeap(d.dev, desc)
    heap === nothing && throw(OutOfMemoryError())
    return heap
end

"""
Release a block's device memory immediately rather than at the next GC, so
`trim!` returns memory when asked.
"""
function rawfree(d::MetalDevice, buf::MTL.MTLBuffer)
    # Not through the driver on a retired device: at exit Metal may be gone. The
    # buffer is released when whatever still holds it is collected.
    d.retired && return nothing
    forgetmanaged!(d, buf)
    Metal.free(buf)
    return nothing
end

"""The `managedbuffer` record of `buf`, dropped before its memory goes back, so a
later buffer at the same address does not inherit it."""
forgetmanaged!(d::MetalDevice, buf::MTL.MTLBuffer) =
    (delete!(d.managed, Base.bitcast(UInt, pointer(buf))); nothing)

"""
    managedbuffer(dev, buf) -> Metal.Managed

Metal.jl's record of which queue last used `buf`, which a host read through an
`MtlArray` waits for. One per pool buffer, so a read through one view sees a
launch's claim made through another.
"""
managedbuffer(d::MetalDevice, buf::MTL.MTLBuffer) =
    get!(() -> Metal.Managed(buf), d.managed, Base.bitcast(UInt, pointer(buf)))

"""
    claimbuffers!(bq, managed)

Record that `bq`'s next submission uses `managed`'s buffers, as Metal.jl's
launches do, so a host read through a view waits for it. A buffer another queue
may still be using is waited for first, outside `submission_lock`.
"""
function claimbuffers!(bq::Metal.BatchedCommandQueue, managed::Vector{Metal.Managed})
    while true
        conflict = Base.@lock Metal.submission_lock begin
            i = findfirst(m -> !Metal.can_take_ownership(m, bq), managed)
            i === nothing && foreach(m -> Metal.take_ownership!(m, bq), managed)
            i === nothing ? nothing : managed[i]
        end
        conflict === nothing && return
        Metal.synchronize(conflict)
    end
end
# A heap is released, not `Metal.free`d (the buffer path); its textures die with it.
rawfree(d::MetalDevice, heap::MTL.MTLHeap) =
    (d.retired || Metal.ObjectiveC.release(heap); nothing)

# The timeline is the queue's: a `LegacyQueue` counts retirements and answers
# `passed` by draining, an `MTL4Queue` reads a shared event's `signaledValue`.

fence(d::MetalDevice) = fence(d.queue)
passed(d::MetalDevice, f) = passed(d.queue, f)
waitfor(d::MetalDevice, f) = waitfor(d.queue, f)

# Asking a `LegacyQueue` for a fence advances it, except on a quiet queue: there
# every retirement is done, so the fence returned has already passed and a region
# dropped on an idle device needs no drain.
function fence(q::LegacyQueue)
    if quiet(q)
        q.completed = q.next
        return q.completed
    end
    q.next += UInt64(1)
    return q.next
end

"""
No open batch, and every command buffer committed to the queue's `MTLCommandQueue`
has finished (for an `MTL4Queue`, its legacy side). `MTL.all_completed` because
reading `.status` off `MTL.last_committed` is a dynamic call that allocates.
"""
function quiet(q)
    Metal.has_open_batch(batchqueue(q)) && return false
    return MTL.all_completed(cmdqueue(q))
end

passed(q::LegacyQueue, f) = UInt64(f) <= q.completed

"""
Wait for the timeline to reach `f` and return `true`, or return `false` when
`f > next`: that value belongs to work not yet submitted, and the caller (an
allocation) grows the pool instead of deadlocking.
"""
function waitfor(q::LegacyQueue, f)
    passed(q, f) && return true
    UInt64(f) > q.next && return false
    waitidle(q)
    return true
end

"""
Drain the queue and mark every fence handed out before the wait as passed.
`Metal.synchronize(q.bq)` commits the open batch, waits for the last command
buffer on `q.mtl` (where every submission of this backend goes) and reports a
failed command buffer or device-side exception. Not `Metal.synchronize()`, which
waits on the current task's queue, a different one for a device selected by name.
"""
function waitidle(q::LegacyQueue)
    issued = q.next
    Metal.synchronize(q.bq)
    q.completed = max(q.completed, issued)
    return nothing
end

# Submission: `s = opensubmit!(d, buffers)`, encode into `encoder(s)`, then
# `tok = closesubmit!(d, s)` commits and returns the fence it will signal.
# `replay!` uses only these. The touched buffers are an argument to opening
# because legacy declares them on the encoder, while MTL4 needs its residency set
# complete before the command buffer declares it.

"""A legacy submission: the batched queue and its compute encoder, borrowed from
the batch, which ends and releases it."""
struct LegacySubmission
    bq::Metal.ConcreteBatchedQueue
    enc::MTL.MTLComputeCommandEncoderRef
end

"""The compute encoder a submission records into."""
encoder(s::LegacySubmission) = s.enc

opensubmit!(d::MetalDevice, bufs) = opensubmit!(d.queue, d.dev, bufs)

# Through the batch every kernel launch uses, so a replay stays ordered against
# the launches around it and can share an already open encoder with a walked pass.
function opensubmit!(q::LegacyQueue, ::MTL.MTLDevice, bufs)
    enc = Metal.compute_encoder(q.bq)
    # `useResource` on `ensureresident!`'s list is hazard tracking, not only
    # residency: undeclared, the driver may overlap consecutive frames on shared
    # bytes, and with 8 in flight a 4-sample frame comes out as the 1-sample image.
    # Core derives no cross-frame ordering; Metal gets it from this plus the
    # in-order queue.
    MTL.use!(enc, bufs, MTL.ReadWriteUsage)
    return LegacySubmission(q.bq, enc)
end

function executesegment!(q::LegacyQueue, s::LegacySubmission, rec, seg)
    q.encodes && return encodecommands!(s.enc, rec, seg.first:seg.last)
    if seg.slot < 0
        MTL.execute_commands!(s.enc, rec.icb, seg.first:seg.last)
    else
        MTL.execute_commands_indirect!(s.enc, rec.icb, rec.rangebuf,
                                       rec.rangeoff + 8 * seg.slot)
    end
    return nothing
end

"""
    nextpiece!(dev, sub, ids)

The boundary between two pieces of one recording replayed into the same encoder.
Legacy declares the next piece's resources on the encoder (hazard tracking, see
`opensubmit!`); MTL4 has no hazard tracking, so the boundary is a dispatch barrier.
"""
nextpiece!(d::MetalDevice, s, ids) = nextpiece!(d.queue, s, ids)

nextpiece!(::LegacyQueue, s::LegacySubmission, ids) =
    (MTL.use!(s.enc, ids, MTL.ReadWriteUsage); nothing)

"""
    suspendsubmit!(dev, sub)

End the open encoder so a host call (`replaywithcalls!`) can encode its own into
the same command buffer; `opensubmit!` then opens a fresh one. Nothing is
committed, so the frame stays one submission, which `fence` depends on.
"""
suspendsubmit!(d::MetalDevice, s) = suspendsubmit!(d.queue, s)

# `end_encoder!` also drops Metal.jl's cached pipeline binding, which an indirect
# execution leaves undefined (as in `closesubmit!`).
suspendsubmit!(q::LegacyQueue, ::LegacySubmission) = (Metal.end_encoder!(q.bq); nothing)

closesubmit!(d::MetalDevice, s) = closesubmit!(d.queue, s)

function closesubmit!(q::LegacyQueue, s::LegacySubmission)
    # An indirect execution leaves the pipeline binding undefined, and
    # `set_pipeline!` skips a set it believes current: the next launch would run
    # whatever pipeline the recording left.
    s.bq.backend.last_pipeline = nothing
    Metal.note_recorded!(s.bq, 0, nothing)
    # Commit only: `replay!` holds `submission_lock` here and waits for a free
    # in-flight slot after releasing it.
    Metal.commit_batch!(s.bq)
    # Nothing signals, so the token is the retirement counter (see `LegacyQueue`).
    return fence(q)
end

# Metal 4: the same verbs on a queue Mantle owns. It buys no host time, only the
# timeline: an `MTLSharedEvent` signalled per submit makes `passed` a load where a
# `LegacyQueue` needs a device drain.

"""How many frames may be recording or in flight at once."""
const MTL4_FRAMES_IN_FLIGHT = 3

"""
    MTL4Queue(mtldevice)

Metal 4 submission of recorded plans; everything else still uses `mtl`.
Resetting an allocator the GPU still reads is a use-after-free, so allocators and
command buffers form a ring of `MTL4_FRAMES_IN_FLIGHT` slots, each stamped with
the token its last submission signals.
"""
mutable struct MTL4Queue
    q::MTL.MTL4CommandQueue
    allocs::Vector{MTL.MTL4CommandAllocator}
    cbs::Vector{MTL.MTL4CommandBuffer}
    # The token each slot's last submission will signal; 0 for never used.
    at::Vector{UInt64}
    slot::Int
    # The one-element array `commit:count:` takes, kept so a replayed frame
    # allocates nothing (`test_record_metal.jl`). Pointers, since an array of
    # wrappers is converted to a fresh pointer array on every commit.
    batch::Vector{id{MTL.MTL4CommandBuffer}}
    # The timeline: `signaledValue` is what the GPU finished, `next` the value
    # the last submission was given.
    event::MTL.MTLSharedEvent
    next::UInt64
    mtl::MTL.MTLCommandQueue
    bq::Metal.ConcreteBatchedQueue
    # How a failed submission is reported: MTL4 command buffers carry no error,
    # so without it a fault is a `waitfor` that never returns.
    feedback::MTL.MTL4Feedback
    # `mtl`'s residency set, also given to this queue; `adoptqueue!` makes Metal.jl
    # use it too. A replay reaching memory outside it reads zeros and drops writes
    # silently, and `MTL_SHADER_VALIDATION=1` hides that by making all resident.
    resset::Union{Nothing,MTL.MTLResidencySet}
    # For `submissions`: MTL4 command buffers committed, and the empty bridge
    # buffers on `mtl` that only carry a signal or wait between the queues.
    commits::Int
    bridges::Int
end

function MTL4Queue(dev::MTL.MTLDevice)
    n = MTL4_FRAMES_IN_FLIGHT
    # With a descriptor, so a refused queue reports through `error:` instead of nil.
    q = MTL.MTL4CommandQueue(dev, MTL.MTL4CommandQueueDescriptor())
    mtl = MTL.MTLCommandQueue(dev)
    mtl.label = "Mantle"
    # `BatchedCommandQueue` installs `mtl`'s residency set; asking afterwards
    # returns that one.
    bq = Metal.BatchedCommandQueue(mtl)
    pinpipeline!(bq)
    resset = Metal.can_use_residency_sets(dev) ?
        Metal.install_queue_residency!(mtl, dev) : nothing
    # On the queue here, and on each command buffer in `opensubmit!`, as the API asks.
    resset === nothing || MTL.add_residency_set!(q, resset)
    return MTL4Queue(q,
                     [MTL.MTL4CommandAllocator(dev) for _ in 1:n],
                     [MTL.MTL4CommandBuffer(dev) for _ in 1:n],
                     zeros(UInt64, n), 0,
                     Vector{id{MTL.MTL4CommandBuffer}}(undef, 1),
                     MTL.MTLSharedEvent(dev), UInt64(0), mtl, bq,
                     MTL.MTL4Feedback(), resset, 0, 0)
end

batchqueue(q::MTL4Queue) = q.bq

cmdqueue(q::MTL4Queue) = q.mtl

# Launches and blits still go through Metal.jl's batch (see `ownflush!`).
ownflush!(q::MTL4Queue, ::MTL.MTLDevice, own::Bool) =
    (Metal.own_flushes!(q.bq, own); nothing)

"""An MTL4 submission: a ring slot, its command buffer and its unmanaged encoder,
whose one reference `opensubmit!` takes and `closesubmit!` gives back."""
struct MTL4Submission
    cb::MTL.MTL4CommandBuffer
    enc::MTL.MTL4ComputeCommandEncoderRef
    slot::Int
end

encoder(s::MTL4Submission) = s.enc

"""
Whether this queue can replay an indirect command buffer: `false` on an MTL4
queue, where `executeCommandsInBuffer:` stops completing (see `metalqueue`).
"""
canreplay(q) = true
canreplay(::MTL4Queue) = false

"""
Whether a recording for this queue captures its dispatches so they are encoded
into the command buffer every frame instead of replayed: an MTL4 queue, or a
legacy one with `encodes` set.
"""
encodes(q) = false
encodes(::MTL4Queue) = true
encodes(q::LegacyQueue) = q.encodes

"""
One recorded dispatch's buffer bindings, built once at record time since a
recording's addresses do not move: an `MTL4ArgumentTable` on MTL4, a
`LegacyBindings` list on a legacy queue.
"""
argtable(d::MetalDevice, args, store::MTL.MTLBuffer) = argtable(d.queue, d, args, store)

function argtable(::MTL4Queue, d::MetalDevice, args, store::MTL.MTLBuffer)
    nslots = maximum(a -> a.index, args; init = 0)
    t = MTL.MTL4ArgumentTable(d.dev; buffers = max(nslots, 1), initialize = true)
    for a in args
        buf = a.buffer === nothing ? store : a.buffer
        off = a.buffer === nothing ? a.offset : a.bufoffset
        MTL.set_address!(t, UInt64(buf.gpuAddress) + UInt64(off), a.index)
    end
    return t
end

"""
    encodeplan!(sub, rec)

Run a recorded plan by encoding its dispatches: argument table, pipeline, grid,
and a dispatch barrier where the recorder marked a pass boundary. MTL4 does no
hazard tracking, so these barriers are the graph's only ordering.
"""
function encodeplan!(s::MTL4Submission, rec)
    enc = s.enc
    for c in rec.encoded
        c.barrier && MTL.barrierAfterEncoderStages!(enc, MTL.MTLStageDispatch, MTL.MTLStageDispatch)
        MTL.set_argument_table!(enc, c.table)
        MTL.set_function!(enc, c.pipeline)
        if c.slotoff >= 0
            # Device-sized: the grid is the three words core's prepare wrote into
            # the slot, before this command's barrier.
            MTL.dispatch_threadgroups_indirect!(enc, rec.argstore, c.slotoff, c.threads)
        elseif c.gridoff < 0
            MTL.dispatchThreadgroups!(enc, c.groups, c.threads)
        else
            # Gated: `gridoff` counts 4-byte words; a closed gate leaves three
            # zeros, which runs no threadgroups.
            MTL.dispatch_threadgroups_indirect!(enc, rec.gridbuf,
                                                rec.gridoff + 4 * c.gridoff,
                                                c.threads)
        end
    end
    return nothing
end

function argtable(::LegacyQueue, d::MetalDevice, args, store::MTL.MTLBuffer)
    return LegacyBindings([(a.buffer === nothing ? store : a.buffer,
                            a.buffer === nothing ? a.offset : a.bufoffset,
                            a.index) for a in args])
end

bufferbarrier!(enc::MTL.MTLComputeCommandEncoderRef) =
    @objc [enc::id{MTL.MTLComputeCommandEncoderRef} memoryBarrierWithScope:MTL.MTLBarrierScopeBuffers::MTL.MTLBarrierScope]::Nothing

function encodecommands!(enc::MTL.MTLComputeCommandEncoderRef, rec, commands::UnitRange{Int})
    for c in view(rec.encoded, commands)
        c.barrier && bufferbarrier!(enc)
        MTL.set_function!(enc, c.pipeline)
        for (buf, off, index) in c.table
            MTL.set_buffer!(enc, buf, off, index)
        end
        if c.gridoff < 0
            MTL.dispatchThreadgroups!(enc, c.groups, c.threads)
        else
            MTL.dispatchThreadgroupsIndirect!(enc, rec.gridbuf, rec.gridoff + 4 * c.gridoff,
                                              c.threads)
        end
    end
    return nothing
end

"""
Whether `rec` was recorded for this queue's route. An encode-shaped recording
carries dispatches and its gates write threadgroup counts; a replay-shaped one
carries none and its gates write execution ranges. Replaying an encode-shaped
recording would silently run no gated iteration, so both mismatches are refused.
"""
canrun(q, rec) = encodes(q) ? !isempty(rec.encoded) : isempty(rec.encoded)

"""Throw the `ArgumentError` for a recording this queue cannot run."""
refusereplay(::MTL4Queue) = throw(ArgumentError(
    "Mantle: this recording holds nothing an MTL4 queue can run. It is recorded " *
    "for a queue that replays an indirect command buffer, and an MTL4 queue cannot: " *
    "`executeCommandsInBuffer:` on an MTL4 compute encoder stops completing after " *
    "about a second of replayed GPU work, with no error from the commit feedback " *
    "and nothing in the system log (`Metal/research/mtl4_icb_stall.jl` is the " *
    "reproducer). Record the plan on the device that runs it, and its dispatches " *
    "are captured for encoding as well."))

refusereplay(::LegacyQueue) = throw(ArgumentError(
    "Mantle: this recording was made to be ENCODED, and this queue replays. Its " *
    "gates write threadgroup counts and its execution ranges were never written, " *
    "so replaying it would run every ungated pass and silently discard every " *
    "iteration of every `repeat!`. Record the plan on the device that runs it."))

# Unreached and untested while `canreplay(::MTL4Queue)` is `false`; kept for a
# driver that fixes the stall, where it is one send per segment against three per
# dispatch. Encode-shaped recordings leave execution ranges zero, so re-record.
function executesegment!(::MTL4Queue, s::MTL4Submission, rec, seg)
    if seg.slot < 0
        MTL.execute_commands!(s.enc, rec.icb, seg.first:seg.last)
        return nothing
    end
    # Before: the dispatch that wrote this range has landed. After: the next
    # segment sees what the iteration wrote.
    MTL.barrierAfterEncoderStages!(s.enc, MTL.MTLStageDispatch, MTL.MTLStageDispatch)
    MTL.execute_commands_indirect!(s.enc, rec.icb, rec.rangebuf,
                                   rec.rangeoff + 8 * seg.slot)
    MTL.barrierAfterEncoderStages!(s.enc, MTL.MTLStageDispatch, MTL.MTLStageDispatch)
    return nothing
end

function opensubmit!(q::MTL4Queue, dev::MTL.MTLDevice, bufs)
    # Residency must be complete before this call: MTL4 has no `useResource`, and
    # a set a command buffer already declared does not pick up later allocations
    # (the replay reads unmapped memory and writes nothing, silently).
    # `replay!` calls `ensureresident!` just before, which grants only when the
    # list is rebuilt.
    q.slot = q.slot == length(q.allocs) ? 1 : q.slot + 1
    f = q.at[q.slot]
    # Zero: slot never used. Otherwise the host blocks only when the ring has
    # come round to a frame the GPU is still running.
    iszero(f) || passed(q, f) || waitfor(q, f)
    alloc = q.allocs[q.slot]
    MTL.reset!(alloc)
    cb = q.cbs[q.slot]
    MTL.beginCommandBuffer!(cb, alloc)
    enc = MTL.MTL4ComputeCommandEncoderRef(cb)
    q.resset === nothing || MTL.use_residency_set!(cb, q.resset)
    return MTL4Submission(cb, enc, q.slot)
end

nextpiece!(::MTL4Queue, s::MTL4Submission, ids) = (MTL.barrierAfterEncoderStages!(s.enc, MTL.MTLStageDispatch, MTL.MTLStageDispatch); nothing)

"""
    flushlegacy!(q::MTL4Queue) -> UInt64

Close what is pending on the legacy queue and return the timeline value it will
signal (`0` if nothing is pending), for the next MTL4 submission to wait on: Metal
orders command buffers only within a queue. Work Metal.jl already committed
(`maybe_autoflush!`) gets an empty command buffer behind it carrying the signal.
"""
function flushlegacy!(q::MTL4Queue)
    bq = q.bq
    # A signal cannot be encoded while an encoder is open.
    Metal.end_encoder!(bq)
    cb = bq.backend.cmdbuf
    if cb === nothing
        quiet(q) && return UInt64(0)
        f = (q.next += UInt64(1))
        # Unmanaged and committed directly, as the bridge in `closesubmit!`.
        behind = MTL.MTLCommandBufferRef(q.mtl)
        MTL.encode_signal!(behind, q.event, f)
        MTL.commit_with_queue_key!(behind, pointer(q.mtl))
        Metal.ObjectiveC.release(behind)
        q.bridges += 1
        return f
    end
    f = (q.next += UInt64(1))
    MTL.encode_signal!(cb, q.event, f)
    Metal.flush!(bq)
    return f
end

function closesubmit!(q::MTL4Queue, s::MTL4Submission)
    close(s.enc)
    MTL.endCommandBuffer!(s.cb)
    # Wait for the legacy work this frame put on `mtl` and for the previous MTL4
    # submission. A recording is not read-only (gates write their execution
    # ranges into it during the replay), and MTL4 does no hazard tracking between
    # command buffers, so overlapping replays corrupt each other's ranges and a
    # submission never signals. This gives up cross-frame overlap, as legacy does.
    prev = q.next
    fl = flushlegacy!(q)
    w = iszero(fl) ? prev : fl
    iszero(w) || MTL.wait_event!(q.q, q.event, w)
    q.batch[1] = pointer(s.cb)
    MTL.commit!(q.q, q.batch, q.feedback)
    q.commits += 1
    f = (q.next += UInt64(1))
    MTL.signal_event!(q.q, q.event, f)
    q.at[s.slot] = f
    # The bridge back: an empty command buffer on `mtl` that waits for `f`, so
    # later legacy work (an `Array(x)` readback, the next uploads) runs after this
    # submission. Safe to commit directly since `flushlegacy!` just closed the
    # batch. `commit_with_queue_key!` skips `commit!`'s submission hook, which
    # would allocate a queue wrapper to flush a closed batch.
    bridge = MTL.MTLCommandBufferRef(q.mtl)
    MTL.encode_wait!(bridge, q.event, f)
    MTL.commit_with_queue_key!(bridge, pointer(q.mtl))
    Metal.ObjectiveC.release(bridge)
    q.bridges += 1
    return f
end

"""
End a walked frame on the MTL4 path: its passes went to `mtl`, so signal the
shared timeline from there for the next MTL4 submission to wait on. A frame with
nothing pending keeps the current token.
"""
function closeframe!(q::MTL4Queue, ::MTL.MTLDevice)
    f = flushlegacy!(q)
    return iszero(f) ? q.next : f
end

# The pool's timeline contract: `fence` covers everything submitted so far,
# `passed` compares against what the GPU signalled, and `waitfor` declines a
# value nothing has submitted.

"""
The value the next submission will signal: `next + 1`, so a non-signalling
command buffer committed since (a render pass, an acceleration-structure build)
is covered, because it runs before that submission.
"""
fence(q::MTL4Queue) = q.next + UInt64(1)

"""
Whether the GPU finished everything `f` covers. Compares `min(f, next)`, since a
`next + 1` token from `fence` has no submission behind it on an idle device. A
token past `next` also needs the legacy queue quiet: eager launches on `mtl` have
no signal until the next `flushlegacy!`.
"""
function passed(q::MTL4Queue, f)
    min(UInt64(f), q.next) <= q.event.signaledValue || return false
    return UInt64(f) <= q.next || quiet(q)
end

"""
Wait on the shared event for `min(f, next)`, or for a `flushlegacy!` value when
`f` is past `next` and legacy work is pending; return `false` for a token past
`next + 1`. Throws when the MTL4 feedback reports a failed submission.
"""
function waitfor(q::MTL4Queue, f)
    passed(q, f) && return true
    # Nothing hands out more than `next + 1`.
    UInt64(f) > q.next + UInt64(1) && return false
    target = UInt64(f) > q.next && !quiet(q) ? flushlegacy!(q) : min(UInt64(f), q.next)
    # Bounded waits in a loop so a failed submission is reported, not waited on
    # forever. Not a timeout: healthy work is waited for as long as it takes.
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

Disable Metal.jl's `maybe_autoflush!` for a run (`own = true`), or restore it.
An autoflush mid-run is a submission no fence covers, so a retired region could
be reused while it still reads. Owning the flush is a promise to flush, which
`closeframe!` and `closesubmit!` keep on every path out of `submitrun!`.
"""
ownflush!(d::MetalDevice, own::Bool) = ownflush!(d.queue, d.dev, own)

ownflush!(q::LegacyQueue, ::MTL.MTLDevice, own::Bool) =
    (Metal.own_flushes!(q.bq, own); nothing)

"""
    closeframe!(dev) -> UInt64

End a walked run (passes encoded into the open batch, or render and copy command
buffers of their own) by flushing the batch, and return the token covering it.
"""
closeframe!(d::MetalDevice) = closeframe!(d.queue, d.dev)

function closeframe!(q::LegacyQueue, ::MTL.MTLDevice)
    Metal.flush!(q.bq)
    return fence(q)
end

"""
Commit the open batch (`closeframe!`) and wait for it. Waiting without committing
would return before the open work ran, and a readback would show stale data.
"""
flush!(d::MetalDevice) = (waitfor(d, closeframe!(d)); nothing)

"""
`true` on a `LegacyQueue`: a frame is one open command buffer, so a host call
encodes into the same submission, ordered by encoder order (`replaywithcalls!`,
`suspendsubmit!`). An `MTL4Queue` keeps core's `false`: there is no open batch to
join and MPSGraph has no MTL4 entry point, so `bake` refuses such a plan.
"""
Mantle.runscalls(::MetalDevice{LegacyQueue}) = true

"""
    cmdqueue(dev) -> MTLCommandQueue

The plain queue, for command buffers committed outside the batch, such as
acceleration-structure builds (`raytracing.jl`) and traces (`trace.jl`).
"""
cmdqueue(d::MetalDevice) = cmdqueue(d.queue)
cmdqueue(q::LegacyQueue) = q.mtl

"""
    executesegment!(dev, sub, rec, seg)

Run one segment of a recording (encoded instead when the queue `encodes`). A
gated segment's range is written by a dispatch and read by the command processor
at the execute, which no barrier bit orders: legacy infers the hazard from
`useResource`, MTL4 needs a barrier on either side.
"""
executesegment!(d::MetalDevice, sub, rec, seg) = executesegment!(d.queue, sub, rec, seg)

"""
    batchqueue(dev) -> Metal.BatchedCommandQueue

The device's batch, where launches, blits and walked passes go. Never
`Metal.global_queue`, which returns a per-task queue with its own residency set.
"""
batchqueue(d::MetalDevice) = batchqueue(d.queue)
batchqueue(q::LegacyQueue) = q.bq

"""
Make this device's batch the one Metal.jl launches on for the running task (a
task-local store). Called at the head of every run.
"""
adoptqueue!(d::MetalDevice) = (Metal.adopt_queue!(d.dev, batchqueue(d)); nothing)

devicename(d::MetalDevice) = String(d.dev.name)
"""The largest single `MTLBuffer` this device will create."""
maxalloc(d::MetalDevice) = Int(d.dev.maxBufferLength)

"""
Device memory a transient arena may use: `recommendedMaxWorkingSetSize`, an upper
bound the placer checks against, not a reservation.
"""
capacity(d::MetalDevice) = Int(d.dev.recommendedMaxWorkingSetSize)

"""Wait for everything submitted to the device: its queue and every channel."""
waitidle(b::Metal.MetalBackend) = waitidle(Device(b))
waitidle(d::MetalDevice) = (waitidle(d.queue); foreach(Metal.synchronize, d.channels); nothing)

# Both queues, the MTL4 one through the timeline so the pool sees what passed.
waitidle(q::MTL4Queue) = (waitfor(q, q.next); Metal.synchronize(q.bq); nothing)

# `supports_graphics` is in `graphics.jl`.

# A channel is another `MTLCommandQueue` with Metal.jl's batch over it. Queues are
# unordered against each other; a buffer used on two is ordered by Metal.jl's
# ownership check, a host wait for the other queue before each launch and copy.

supports_batch_queue(::Union{Metal.MetalBackend,MetalDevice}) = true

function allocate_batch_queue!(d::MetalDevice)
    refuseretired(d)
    mtl = MTL.MTLCommandQueue(d.dev)
    mtl.label = "Mantle channel"
    bq = Metal.BatchedCommandQueue(mtl)
    pinpipeline!(bq)
    push!(d.channels, bq)
    return bq
end
allocate_batch_queue!(b::Metal.MetalBackend) = allocate_batch_queue!(Device(b))

"""
The live device that handed `bq` out or owns it as its queue. Throws when none does.
"""
function deviceof(bq::Metal.BatchedCommandQueue)
    for r in METAL.live
        d = r.value
        d isa MetalDevice || continue
        (batchqueue(d) === bq || any(c -> c === bq, d.channels)) && return d
    end
    throw(ArgumentError("Mantle: this Metal queue belongs to no device: it was " *
                        "released, or was never handed out by `allocate_batch_queue!`."))
end
todevice(bq::Metal.BatchedCommandQueue) = deviceof(bq)

backend(bq::Metal.BatchedCommandQueue) = Metal.MetalBackend(bq)

"""Wait for everything submitted on `bq`, committing what is open first."""
flush!(bq::Metal.BatchedCommandQueue) = (Metal.synchronize(bq); nothing)

"""Nothing is retired onto a Metal channel (arrays go back through Metal.jl,
regions through the pool), so there is nothing to destroy."""
reclaim!(::Metal.BatchedCommandQueue) = 0

"""
Drain `bq` and give it back. Releasing twice is a no-op; the device's own queue is
refused, as core's contract for `release_batch_queue!` says.
"""
function release_batch_queue!(bq::Metal.BatchedCommandQueue)
    for r in METAL.live
        d = r.value
        d isa MetalDevice || continue
        batchqueue(d) === bq && throw(ArgumentError(
            "Mantle: this is the device's own queue, which is not released; only a " *
            "channel from `allocate_batch_queue!` is."))
        i = findfirst(c -> c === bq, d.channels)
        i === nothing && continue
        Metal.synchronize(bq)
        d.releasedsubmits += MTL.committed(bq.queue)
        deleteat!(d.channels, i)
        return nothing
    end
    return nothing
end

# The lifecycle verbs of `runtime/lifecycle.jl`.

"""
Command buffers with work committed to the device's queue and channels (released
ones included); on MTL4, plus MTL4 commits, minus the empty bridge buffers.
"""
submissions(d::MetalDevice) = submissions(d.queue) + d.releasedsubmits +
    sum(c -> MTL.committed(c.queue), d.channels; init = 0)
submissions(q::LegacyQueue) = MTL.committed(q.mtl)
submissions(q::MTL4Queue) = MTL.committed(q.mtl) - q.bridges + q.commits

retire!(d::MetalDevice) = (d.retired = true; nothing)
retired(d::MetalDevice) = d.retired

"""
    reset_device!(d::MetalDevice; debug = d.debug) -> MetalDevice

Retire `d` and build a device on the same `MTLDevice`: a new queue of the same
generation, a new pool. The default device's replacement becomes the default and
adopts the queue. Nothing waits for `d`'s queue: a reset is also how a caller
recovers from a queue whose command buffers failed.
"""
function reset_device!(d::MetalDevice; debug::DebugConfig = d.debug)
    new = MetalDevice(d.dev, typeof(d.queue)(d.dev), debug)
    retire!(d)
    METAL.default === d && defaultdevice!(new)
    return new
end

"""
The array, or `nothing` when Metal has no memory for it. Allocates the `MTLBuffer`
directly because `MtlArray(undef, ...)` retries an out-of-memory with a full GC
and a device drain.
"""
function tryallocate(d::MetalDevice, ::Type{T}, dims::Dims{N}) where {T,N}
    buf = try
        MTL.MTLBuffer(d.dev, max(checkedbytes(T,dims),4); storage = Metal.SharedStorage)
    catch err
        err isa OutOfMemoryError || rethrow()
        return nothing
    end
    ref = GPUArrays.DataRef(m -> Metal.free(m.buffer), Metal.Managed(buf))
    a = MtlArray{T,N}(ref, dims)
    # The array took its own reference; this one was only for building it.
    GPUArrays.unsafe_free!(ref)
    return a
end
