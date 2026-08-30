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

A Metal device: the `MTLDevice`, one command queue, one timeline, one pool.

The timeline is what tells the pool when a retired region may be handed out
again, and on Metal it cannot be a submission counter the way Vulkan's is.

Mantle does not own the submissions here. Compute goes out through Metal.jl's
own `BatchedCommandQueue`, so this backend never sees the command buffer that
reads a region and has nothing to signal an `MTLSharedEvent` from. The first
version kept the Vulkan shape anyway — `next` beside a `signaledValue` — and
since nothing ever advanced `next`, `fence` returned 0 and `passed(d, 0)` was
`0 >= 0`: **every retired region looked finished the instant it was retired**,
so the pool recycled memory that in-flight kernels were still reading. It only
showed once a session allocated enough to reuse anything, which is why one
scene rendered correctly and the next came out NaN.

So the timeline counts *retirements*, not submissions, and the only thing that
advances `completed` is a real `Metal.synchronize()`. That is conservative —
reuse costs a device sync — but it is the honest answer to "has the GPU
finished with these bytes" when this backend cannot observe the queue that
holds them. `MTLSharedEvent` stays on the struct for the hardware ray tracing
path, which does submit its own command buffers.
"""
mutable struct MetalDevice <: Device
    dev::MTL.MTLDevice
    queue::MTL.MTLCommandQueue
    # The timeline. `signaledValue` is what the GPU has finished; `next` is what
    # the following submission will signal.
    event::MTL.MTLSharedEvent
    # `next` is the fence value the next retirement gets; `completed` is the
    # highest value a `Metal.synchronize()` has covered. A region is reusable
    # only when its fence is <= `completed`.
    next::UInt64
    completed::UInt64
    # Whether this backend has a command buffer of its own open (the ray-tracing
    # encoder). Retirements made while one is recording cannot come back until
    # it lands — see `fence`.
    recording::Bool
    pool::Pool
    # Built on first ask and kept: reading `threadExecutionWidth` needs a
    # compiled pipeline, and `gemv` asks per launch.
    caps::Union{Nothing,DeviceCaps}
    # The hardware-traversal pipeline (`trace.jl`). Compiled from MSL source at
    # runtime, which runs the Metal frontend and costs tens of milliseconds, so
    # it is built on first trace and kept for the life of the device.
    trace_pipeline::Union{Nothing,MTL.MTLComputePipelineState}
end

function MetalDevice(mtldev::MTL.MTLDevice = Metal.device())
    MetalDevice(mtldev, MTL.MTLCommandQueue(mtldev), MTL.MTLSharedEvent(mtldev),
                UInt64(0), UInt64(0), false, Pool(), nothing, nothing)
end

# One device per process, cached: a second `MetalDevice` would mean a second
# `Pool` over the same `MTLDevice`, which is two allocators over one memory.
const METAL_DEVICE = Ref{Union{Nothing,MetalDevice}}(nothing)
function Device(::MetalAPI)
    d = METAL_DEVICE[]
    d === nothing || return d
    return METAL_DEVICE[] = MetalDevice()
end

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
    # Tracked, and the heap has to agree with its resources — see
    # `image_descriptor` for why this is not untracked: nothing on this backend
    # turns the graph's transitions into fences yet, so the driver's own
    # tracking is the only thing keeping one pass behind the one that feeds it.
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

"""
What must complete before memory retired *now* may be handed out again.

With a command buffer open that is the value the next submission will signal,
because a command already encoded into it may name those bytes. With none open,
everything ever recorded has been submitted and `next` is the last of it —
waiting for one more would wait for a submission an idle application never
makes, pinning the region for ever.

Read, never forced: a reclaim must not go and submit a half-recorded command
buffer to collect its own memory.
"""
function fence(d::MetalDevice)
    d.next += UInt64(1)
    return d.next
end

passed(d::MetalDevice, f) = UInt64(f) <= d.completed

"""
Wait for the timeline to reach `f`, unless nothing has been submitted that will.

`f > next` names a value belonging to a command buffer still being recorded.
Blocking on it would wait for a submission that has not happened, and the one
caller is an allocation — so it declines, and the pool grows instead. That costs
a block and never a deadlock.
"""
function waitfor(d::MetalDevice, f)
    passed(d, f) && return true
    # Drain everything: `Metal.synchronize()` waits on this task's queue, which
    # is the one Metal.jl dispatches kernels on. There is no finer-grained wait
    # available to this backend, and a coarse-but-correct one is the point —
    # the alternative was handing back live memory.
    issued = d.next
    Metal.synchronize()
    d.completed = max(d.completed, issued)
    return true
end

# ── Budgets ───────────────────────────────────────────────────────────────────

"""The largest single `MTLBuffer` this device will create."""
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

# The queue lifecycle verbs are deliberately NOT implemented. `BatchQueue` is a
# command-pool/fence/timeline arrangement, and Metal has neither: compute goes
# through Metal.jl's own queue, `MetalDevice`'s `MTLSharedEvent` supplies the
# timeline, and drawing shares that same queue (see `framebuffer!` in
# `graphics.jl`). A caller reaching here wants Vulkan's shape, not a
# capability Metal lacks, so it says so rather than handing back a stub queue
# that silently records nothing.
function allocate_batch_queue!(b::Union{Metal.MetalBackend,MetalDevice})
    throw(ArgumentError(
        "Mantle: allocate_batch_queue! is not implemented on Metal, because a " *
        "BatchQueue exists to drive render passes and presentation and Metal.jl " *
        "has no graphics pipeline. Check `Mantle.supports_graphics(backend)` and " *
        "take the compute path; compute dispatch does not need a BatchQueue."))
end
