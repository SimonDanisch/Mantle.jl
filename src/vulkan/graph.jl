# The graph, plan and pass machinery for the Vulkan backend.
#
# This was `ext/MantleLavaExt.jl`, an extension, because Mantle only weak-depended
# on Lava. The runtime moved into `src/vulkan/` on 2026-08-27 and Lava became a
# hard dependency, so there is nothing optional left to gate and this is core.
#
# Every `Mantle.` qualifier is gone from it for the same reason: it IS Mantle now.
# What was `Mantle.caps(dev)` is `caps(dev)`, and the two `DeviceCaps` structs it
# used to copy between — positionally, with a comment in each warning that a field
# inserted in the middle would misalign silently — are one type in
# `KernelInterface`.
#
# It still reaches into `Lava` by name, and that is the boundary working: what it
# asks for there is the COMPILER — `lava_compile_gpu_from_job`, `LavaGPUKernel`,
# the SPIR-V module — and nothing else.

# ── device ────────────────────────────────────────────────────────────────────
"""
One allocation per arena, shared by every plan on the device.

A plan used to allocate its own arenas, so two plans in one process cost the
*sum* of their peaks even though only one of them is ever running. The device
owns the allocation instead and a plan **reserves** in it: the pool is sized to
the largest tenant, not to their total. On SAM 2's eight graphs that is the
difference between the largest peak and eight of them.

Sound because a transient never crosses a plan boundary — a value that outlives
a graph is a persistent `Buffer`, not a transient — so the only thing two tenants
share is bytes. The one hazard left is a plan's work still reading memory the
next plan is about to write, and that is what [`handover!`](@ref) emits.

`extra` and `typebits` accumulate across tenants because the allocation has to
satisfy all of them at once, and they accumulate in opposite directions: usage
bits are a **union** (the arena must permit everything any tenant does with it)
and image memory type bits are an **intersection** (the type has to be one every
image can bind to). A tenant that narrows the intersection to nothing is an
error, not a silent pick of a type some image cannot use.
"""

struct LavaDevice <: Device
    ctx::Any
    bq::Any
    pool::Pool          # the device owns it; nothing about it reaches the caller
end
LavaDevice(ctx, bq) = LavaDevice(ctx, bq, Pool())
pool(d::LavaDevice) = d.pool

"""
One `LavaDevice` per `VkContext`, cached.

Not a convenience: the device OWNS the pool, so a second `Device(Lava)` would be
a second `Pool` over one VkDevice — two allocators again, and every workload that
asked separately would get its own memory instead of sharing. Caching is what
makes "the device owns the pools" mean anything.

Measured before the cache: `reserved(pool(Device(VulkanAPI())))` came back 0 immediately
after a 120-frame render that had reserved 64 MiB, because the probe had made a
different device.
"""
const DEVICES = IdDict{Any,LavaDevice}()

# `Device(VulkanAPI())`, not `Device(Lava)`. Passing the Lava MODULE as the
# selector meant something while Lava was the backend package; the backend is
# this package now, and the marker is what names an API — the same spelling
# `Device(HostAPI())` uses.
Device(::VulkanAPI) = get!(DEVICES, vk_context()) do
    ctx = vk_context()
    LavaDevice(ctx, ctx.default_bq)
end

"""
    Device(LavaBackend())

The Mantle device for a KA backend.

The missing link for a KA workload that wants the pool: `DNNKernels` holds a
`LavaBackend`, not a module, and allocating through `KA.allocate` puts its slab
somewhere Mantle cannot see. This maps the backend it does have onto the cached
device — same VkContext, same pool, so a model's scratch and an editor's
transients land in one allocator.
"""
Device(::LavaBackend) = Device(VulkanAPI())

# ↑ moved to src/graph/build.jl


"""
    capacity(device) -> Int

Bytes a placement may use, which is the driver's answer and not the heap's size.

`VK_EXT_memory_budget` reports what is *available now* — the heap minus what
every process on the machine has already taken — and that is the number a
placement has to fit in. Falling back to the raw heap size when the extension is
missing is the honest degradation: it is an upper bound rather than a budget, so
the check still catches a plan that could never fit and stops catching one that
merely will not fit today.

Summed over device-local heaps, because that is where a transient arena lands.
"""
function capacity(dev::LavaDevice)
    heaps = probe_device_memory_budget(dev.ctx)
    total = 0
    for h in heaps
        h.device_local || continue
        total += h.budget > 0 ? h.budget : h.size
    end
    total
end

"""The KernelAbstractions backend, for kernels the graph does not own."""
backend(::LavaDevice) = LavaBackend()

"""
What this device can do.

One line, where this was a field-by-field copy between two `DeviceCaps` structs
that agreed on all ten fields. The copy was POSITIONAL, and both definitions
carried a comment warning that a field inserted anywhere but the end would
misalign it silently — a warning is what you write when the design cannot be
checked.

`DeviceCaps` is `KernelInterface`'s now, and `caps` and `caps` are
methods of KI's one function, so there is no second type to convert into and no
`caps(::LavaBackend)` to write: `caps(::LavaBackend)` already IS
that method.
"""
caps(dev::LavaDevice) = caps(dev.ctx)

# ── window ────────────────────────────────────────────────────────────────────
"""
A window, which is Mantle's because the frame loop is: `isopen` has to be a
predicate rather than a thing that also pumps events, and `run!` is the one call
per frame that can pump them.
"""
struct LavaWindow <: Window
    win::VulkanWindow
end

Window(width::Integer, height::Integer; title::AbstractString = "", vsync::Bool = false) =
    LavaWindow(VulkanWindow(width, height; title = String(title), vsync))

Base.isopen(w::LavaWindow) = isopen(w.win)
Base.close(w::LavaWindow) = close(w.win)
Base.size(w::LavaWindow) = size(w.win)
"""
What is on the window now, as a `(width, height)` matrix of BGRA byte tuples.

`(width, height)`, not `(height, width)` — the first index is x. Every caller here
counts or sums pixels, so it has never mattered, but a Julia image is indexed
`(row, column)`, and handing this straight to `save` writes the picture rotated a
quarter turn with no complaint. `permutedims` first if it is going to be looked at.
"""
screenshot(w::LavaWindow) = readback_window(w.win)

# ── persistent resources ──────────────────────────────────────────────────────
#
# `LavaBuffer`/`LavaScalar` are gone. `Buffer` and `Scalar` are core
# types over pool regions, so this supplies primitives and no type — and their
# memory now comes from the same pool as every transient, which is the point:
# one allocator sees both.

# ↑ moved to src/graph/build.jl
extrausage(::Type{DrawIndirectCommand}) = UInt32(VK.BUFFER_USAGE_INDIRECT_BUFFER_BIT)
# `repeat!`'s per-iteration flags, read by `vkCmdBeginConditionalRenderingEXT`.
extrausage(::Type{Predicate}) = UInt32(VK.BUFFER_USAGE_CONDITIONAL_RENDERING_BIT_EXT)

# What `repeat!` needs, and the reason it is a query rather than an assumption:
# the extension is optional, and a device without it must refuse the graph rather
# than record one that runs every iteration unconditionally.
supportspredicate(d::LavaDevice) = (d.ctx::VkContext).conditional_rendering_available
bufferusage(::LavaDevice, ::Type{T}) where {T} = extrausage(T)

rawalloc(dev::LavaDevice, ::Persistent, bytes::Int, usage) =
    rawalloc(dev, Buffers(), bytes, usage)
constraintof(::LavaDevice, ::Persistent, ts) = UInt32(0)

# What has to complete before a region retired NOW can go back.
#
# With a batch open, that is the value it will signal — `next_timeline + 1` —
# because a command already recorded into it may name those bytes. With no batch
# open, everything ever recorded has been submitted and `next_timeline` is the
# last of it, so waiting for one more would wait for a submission that an idle
# application never makes. A retired region would then be pinned for ever, which
# is the leak this whole path exists to remove, wearing a different hat.
#
# Read, never forced. `ensure_active_batch!` would assert the owning thread and
# allocate a command buffer, which is not something a reclaim should do.
fence(d::LavaDevice) =
    d.bq.next_timeline + (has_active_recording(d.bq) ? UInt64(1) : UInt64(0))
# One predicate for "has the device finished this", spelled once.
#
# It was written out as `query_timeline(bq) >= v` at each of the places that ask,
# which is how `flush!` came to fold over two lists and `nextslot!` came to
# compare against `next_timeline` — a raw counter, in code that had no business
# knowing there was one. The queue method is the primitive; the device forwards
# to it, and `Pool`, `Recycler` and the argument ring all go through the device.
# The portable spelling of "wait for this device to finish everything".
#
# `waitidle` had methods for `LavaBackend`, `VkContext` and `VK.Device` here and
# for `MetalDevice` on the other backend — everything except the one type core
# names in `graph/queue.jl`, which is the DEVICE. So `waitidle(device)` worked on
# Metal and was a `MethodError` on Vulkan, and any portable caller had to reach
# past the device to something backend-shaped.
#
# Hands the open batch over BEFORE waiting, which is the part `vkDeviceWaitIdle`
# does not do. The device is idle with respect to what it has been GIVEN, so a
# caller that had recorded a run into a batch nobody submitted got an immediate
# return and then read buffers the GPU had never written — and, worse, the "flush
# so one submission does not grow past the driver's timeout" callers got no
# flush at all. `waitidle` that ignores the work you are still holding is not a
# wait, and the caller cannot tell the difference from the outside.
#
# `flush!` covers this device's queue; `device_wait_idle` then covers everything
# else in the context — a split upload queue, async compute. It throws during a
# capture, which is correct: a capture has nothing to wait FOR, and returning
# quietly is how that was hidden before.
waitidle(d::LavaDevice) = (flush!(d.bq, d.bq.device); waitidle(d.ctx::VkContext))

passed(bq::VulkanBatchQueue, v) = query_timeline(bq) >= v

passed(d::LavaDevice, f) = passed(d.bq, f)

"""
Wait for the timeline to reach `f`, unless nothing has been submitted that will
signal it.

`f > next_timeline` means the value belongs to a batch still being recorded.
`vkWaitSemaphores` on it would block until something else submitted — and the
one caller here is an allocation, which must not go and flush a half-recorded
command buffer to collect its own memory. So it declines and the pool grows,
which costs a block and never a deadlock.
"""
function waitfor(d::LavaDevice, f)
    passed(d, f) && return true
    f > d.bq.next_timeline && return false        # not submitted; nothing to wait on
    wait_semaphores!(d.bq, VK.SemaphoreWaitInfo([d.bq.timeline_sem], [UInt64(f)]))
    return true
end

"""
A borrowed `LavaArray` over a region.

The `DataRef` releaser is a NO-OP, so `copy` bumps a refcount that frees nothing
and Mantle stays the owner — Lava's shape of `unsafe_wrap(…, own = false)`. What
a kernel gets is `todevice` of this; what a copy or a library gets is this.
"""
deviceview(::LavaDevice, a::DeviceArray{T,N}) where {T,N} =
    LavaArray{T,N}(copy(memoryof(a).ref), size(a); offset = offset(a))

upload!(d::LavaDevice, a::DeviceArray{T}, first::Integer,
               data::AbstractVector) where {T} =
    (copyto!(deviceview(d, a), Int(first), collect(T, data), 1, length(data)); a)
download(d::LavaDevice, a::DeviceArray) = Array(deviceview(d, a))
devicecopy!(d::LavaDevice, dst::DeviceArray, src::DeviceArray,
                   n::Integer) =
    (copyto!(deviceview(d, dst), 1, deviceview(d, src), 1, Int(n)); dst)

# ── the window ────────────────────────────────────────────────────────────────
# `WindowSurface` is Mantle's now — see `src/graph/types.jl`.

# ↑ moved to src/graph/build.jl
Surface(g, w::LavaWindow) = Surface(g, w.win)

# A render pass targets either the window or an offscreen framebuffer. These four
# are the only places that difference shows.
# ↑ moved to src/graph/build.jl
# ↑ moved to src/graph/build.jl
# ↑ moved to src/graph/build.jl
# ↑ moved to src/graph/build.jl

target_view(fb::VulkanFramebuffer) = fb.color_view
target_image(fb::VulkanFramebuffer) = fb.color_image
target_extent(fb::VulkanFramebuffer) = (Int(fb.width), Int(fb.height))
target_format(fb::VulkanFramebuffer) = fb.color_format

# What state the target is in when the pass starts. The window arrives from
# acquire; an offscreen target was last read by the copy that took it to the
# window. Both are discarded by a clearing pass, so this only matters when the
# pass loads. A placed target has no contents at all until something writes it,
# and after aliasing it is back in that state, so it starts from Undefined.
# ↑ moved to src/graph/build.jl
initial_usage(::VulkanFramebuffer) = CopySrc

# What state a resource is in when a replay begins. `nothing` means the first use
# establishes it and no transition into it is needed, which is the answer for
# every buffer: buffers have no layout, so there is nothing to transition from.
# ↑ moved to src/graph/build.jl
# ↑ moved to src/graph/build.jl
initial_state(fb::VulkanFramebuffer) = initial_usage(fb)
# Not `nothing`: an image whose first use is a colour attachment still needs the
# layout transition out of UNDEFINED, and `nothing` would emit none.

# ── graph ─────────────────────────────────────────────────────────────────────
# `DrawCall` is Mantle's now — see `src/graph/types.jl`.
# ↑ moved to src/graph/build.jl

# ↑ moved to src/graph/build.jl

"""The colour attachment a pass configures itself from: extent and viewport are
the same for all of them, so the first answers for the set."""
# The attachment a pass takes its render area from. A depth-only pass — which is
# what a shadow map is — has no colour target, and then the depth one is it.
# ↑ moved to src/graph/build.jl
loadop(::KeepOp) = VK.ATTACHMENT_LOAD_OP_LOAD
loadop(::DiscardOp) = VK.ATTACHMENT_LOAD_OP_DONT_CARE
loadop(::Clear) = VK.ATTACHMENT_LOAD_OP_CLEAR

# ↑ moved to src/graph/build.jl
# ↑ moved to src/graph/build.jl

# ↑ moved to src/graph/build.jl
# ↑ moved to src/graph/build.jl

"""
A transient. The compiler owns its interval and its offset; the handle carries no
storage until `Plan` places it. `first`/`last` are pass indices, filled in as the
graph is built, so liveness is not a separate declaration that could disagree.
"""
# `TransientResource`, not `Transient`: `Mantle.Transient` is a MODULE — the
# namespace `Transient.Buffer(g, T, n)` is reached through, and part of the
# public API. In the extension these were different modules and both names could
# be `Transient`; folded into Mantle they collide, and a type shadowing the
# module turns `Transient.Buffer` into a field access on a DataType.
# `TransientResource` is Mantle's now — see `src/graph/types.jl`.

# `TransientBuffer` is Mantle's now — see `src/graph/types.jl`.

Base.length(t::TransientBuffer) = t.n
Base.eltype(::TransientBuffer{T}) where {T} = T
# ↑ moved to src/graph/build.jl
# ↑ moved to src/graph/build.jl
# ↑ moved to src/graph/build.jl
# ↑ moved to src/graph/build.jl
# ↑ moved to src/graph/build.jl

# `TransientImage` is Mantle's now — see `src/graph/types.jl`. Five of its
# fields are this backend's objects and are type parameters; this is the
# spelling those five make, so `t.format` is a `VK.Format` here as it always
# was.
const VulkanTransientImage{T} = TransientImage{T,VK.Format,VK.ImageUsageFlag,VK.Image,
                                               NamedTuple{(:size, :alignment, :type_bits),
                                                          Tuple{Int,Int,UInt32}}}

# What Vulkan needs a storage-buffer binding aligned to, worst case.
alignment(::LavaDevice, ::TransientBuffer) = 256


# `Recycler` is Mantle's now — see `src/graph/types.jl`.

# ↑ moved to src/graph/build.jl

# ↑ moved to src/graph/build.jl

"""
A region for `n` elements of `T`, recycled if one of that size is idle.

From Mantle's pool, not `LavaArray{T,1}(undef, …)` — a rename is an allocation
like any other, and one that bypassed the pool would be invisible to it while
being exactly the kind of churn a pool exists to absorb.

The recycler stays in front of the pool rather than being replaced by it: it
holds a region until the queue timeline says the GPU is done reading, which is a
question about submitted work that the allocator has no way to answer.
"""
function take!(r::Recycler, dev::LavaDevice, ::Type{T}, n::Integer) where {T}
    bytes = Int(n) * sizeof(T)
    free = get(r.free, bytes, nothing)
    if free !== nothing && !isempty(free)
        return pop!(free)::DeviceArray{T,1}
    end
    allocate(pool(dev), dev, Persistent(), T, (Int(n),);
                    align = 256, blocksize = blocksize(dev))
end

# ↑ moved to src/graph/build.jl

# ↑ moved to src/graph/build.jl

# `Graph(dev)` and `Transient.Buffer` are Mantle's — both were generic already.

# ↑ moved to src/graph/build.jl
imageusage(::LavaDevice, ::Type) = COLOR_USAGE
# The same three as COLOR_USAGE, with the attachment bit that matches the
# aspect: a depth target is worth copying out (a test that asserts the depth
# buffer beats one that asserts its effect on colour) and worth sampling (depth
# of field, ambient occlusion, anything that reads the z it just wrote).
imageusage(::LavaDevice, ::Type{Float32}) = VK.IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT |
                                            VK.IMAGE_USAGE_TRANSFER_SRC_BIT |
                                            VK.IMAGE_USAGE_SAMPLED_BIT

# Vulkan's spelling of `Mantle.isdepth`, which is the portable question and now
# the primary one. It used to be the other way round — `isdepth` was defined as
# `aspect(x) == VK.IMAGE_ASPECT_DEPTH_BIT`, so the graph asked a portable
# question by comparing a driver enum.
aspect(x) = isdepth(x) ? VK.IMAGE_ASPECT_DEPTH_BIT : VK.IMAGE_ASPECT_COLOR_BIT

"""
Every usage bit a format is asked for, checked against what the device says the
format can do, before an image is created with it.

`vkCreateImage` fails with `ERROR_FORMAT_NOT_SUPPORTED` and no indication of
which bit was the problem, and the answer is per format and per driver — depth
formats are exactly where that bites, since `D32_SFLOAT` and `D24_UNORM_S8` are
not both available everywhere. This turns it into a message that names the format
and the bit.
"""
function checkusage(ctx, fmt::VK.Format, usage::VK.ImageUsageFlag)
    # `VK = Vulkan` stood on the next line, and Julia decides scope statically:
    # the assignment made `VK` a LOCAL for the whole body, so the call above ran
    # against an unassigned local and every `Transient.Image` threw
    # `UndefVarError: VK not defined in local scope`. Nothing caught it because
    # `test_window.jl`, which is where transient images are exercised, aborts in
    # its first testset on an unrelated shader compile failure.
    props = VK.get_physical_device_format_properties(ctx.physical_device, fmt)
    have = props.optimal_tiling_features
    for (bit, feature, name) in
            ((VK.IMAGE_USAGE_COLOR_ATTACHMENT_BIT, VK.FORMAT_FEATURE_COLOR_ATTACHMENT_BIT, "COLOR_ATTACHMENT"),
             (VK.IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, VK.FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT, "DEPTH_STENCIL_ATTACHMENT"),
             (VK.IMAGE_USAGE_SAMPLED_BIT, VK.FORMAT_FEATURE_SAMPLED_IMAGE_BIT, "SAMPLED"),
             (VK.IMAGE_USAGE_STORAGE_BIT, VK.FORMAT_FEATURE_STORAGE_IMAGE_BIT, "STORAGE"),
             (VK.IMAGE_USAGE_TRANSFER_SRC_BIT, VK.FORMAT_FEATURE_TRANSFER_SRC_BIT, "TRANSFER_SRC"),
             (VK.IMAGE_USAGE_TRANSFER_DST_BIT, VK.FORMAT_FEATURE_TRANSFER_DST_BIT, "TRANSFER_DST"))
        (usage & bit) == bit && (have & feature) != feature &&
            error("$fmt cannot be used as $name on this device")
    end
    usage
end

# `Transient.Image`, `refit!`, `arena` and `resourcekind` are Mantle's now — see
# `src/graph/build.jl`. What stayed is the driver half of both: making the
# `VkImage` and asking it what memory it wants.

function makeimage(dev::LavaDevice, ::Type{T}, width::Int, height::Int,
                   srgb::Bool, usage::VK.ImageUsageFlag, source) where {T}
    ctx = dev.ctx
    fmt = vkformat(VulkanAPI(), T; srgb)
    checkusage(ctx, fmt, usage)
    img = image_2d(ctx, width, height, fmt, usage)
    return VulkanTransientImage{T}(width, height, fmt, usage, img,
                                   image_requirements(ctx, img),
                                   typemax(Int), 0, nothing, nothing, source)
end

function remakeimage!(t::VulkanTransientImage)
    ctx = vk_context()
    t.image = image_2d(ctx, t.width, t.height, t.format, t.usage)
    t.req = image_requirements(ctx, t.image)
    return t
end
# ↑ moved to src/graph/build.jl
resourcekind(::VulkanFramebuffer) = ImageKind()

# ↑ moved to src/graph/build.jl

# ↑ moved to src/graph/build.jl
# ↑ moved to src/graph/build.jl
Base.length(a::Attr) = length(a.resource)

# ↑ moved to src/graph/build.jl

# ↑ moved to src/graph/build.jl
# ↑ moved to src/graph/build.jl
# ↑ moved to src/graph/build.jl

# ↑ moved to src/graph/build.jl

# ↑ moved to src/graph/build.jl

# ↑ moved to src/graph/build.jl
# ↑ moved to src/graph/build.jl

# ↑ moved to src/graph/build.jl

# ↑ moved to src/graph/build.jl

# ↑ moved to src/graph/build.jl

# ↑ moved to src/graph/build.jl
# ↑ moved to src/graph/build.jl
# ↑ moved to src/graph/build.jl
# ↑ moved to src/graph/build.jl
# ↑ moved to src/graph/build.jl
# ↑ moved to src/graph/build.jl
# ↑ moved to src/graph/build.jl


# ↑ moved to src/graph/build.jl

# ↑ moved to src/graph/build.jl

# ↑ moved to src/graph/build.jl
# ↑ moved to src/graph/build.jl

# ↑ moved to src/graph/build.jl

"""
Where the bytes already are, which is a third thing the call can mean.

Data that is already in device memory — produced by a kernel, a broadcast, or
anything else that never went through the host — needs no staging buffer and no
`memcpy`: the copy is device to device and the host never sees the array. This is
the same two routes as above, differing only in where the source is, so it is a
method rather than a branch.
"""
function rename!(g::Graph, e::Emitter, dst::Buffer{T,1}, data::LavaArray{T,1}) where {T}
    old = dst.store
    nbytes = length(data) * sizeof(T)
    fresh = take!(g.recycler, dst.dev, T, dst.capacity)
    fview = deviceview(dst.dev, fresh)

    smb, fmb = data.buf[], fview.buf[]
    cmd_copy_buffer!(e, smb, fmb, nbytes;
                          src_off = pool_offset(smb) + data.offset,
                          dst_off = pool_offset(fmb) + fview.offset)

    dst.store = fresh
    retire!(g.recycler, old, dst.capacity * sizeof(T), signalof(e))
    nothing
end

function inplace!(e::Emitter, dst, data::LavaArray{T,1}, from::Integer) where {T}
    store = storage(dst)
    dmb, smb = store.buf[], data.buf[]
    cmd_copy_buffer!(e, smb, dmb, length(data) * sizeof(T);
                          src_off = pool_offset(smb) + data.offset,
                          dst_off = pool_offset(dmb) + store.offset + (Int(from) - 1) * sizeof(T))
    nothing
end

# A Mantle buffer says the same thing as its store, so it takes the same route.
# ↑ moved to src/graph/build.jl
# ↑ moved to src/graph/build.jl

"""In-place, inline in the command buffer. Falls back to renaming when the
update is too big for `cmd_update_buffer` to carry."""
function inplace!(e::Emitter, dst, data::AbstractVector{T}, from::Integer) where {T}
    store = storage(dst)
    mb = store.buf[]
    off = pool_offset(mb) + store.offset + (Int(from) - 1) * sizeof(T)
    n = length(data) * sizeof(T)
    if n <= 65536 && n % 4 == 0 && off % 4 == 0
        src = data isa Vector{T} ? data : collect(data)
        GC.@preserve src VK.cmd_update_buffer(
            e.cmd, mb.buffer,
            UInt64(off), UInt64(n), Ptr{Cvoid}(pointer(src)))
    else
        upload!(store, data)      # partial and large: rare, and it stalls
    end
    nothing
end

# ↑ moved to src/graph/build.jl

# ↑ moved to src/graph/build.jl

# ↑ moved to src/graph/types.jl — how deep a plan pipelines is not a driver's.

"""
Lay out everything the plan's recordings read, and take the memory once — from
the pool, as one [`Unified`](@ref) region the plan owns.

A slot is the arguments of every draw and dispatch, and then one indirect command
per device-sized dispatch. Both are fixed by the plan: the draws and dispatches
are known, and so is each one's argument size, so nothing here is allocated per
frame and nothing has to work out when it may be reused. The slot is what the
device is still reading, and a slot comes round again only once it has passed the
run that used it.

The indirect commands are in the SAME slot rather than in a pool of their own —
which is what the queue held, rewound whenever the queue happened to drain. That
put a recording's workgroup counts on a free list while it still
named them, and two plans replaying in one frame wrote each other's counts. Here
the slot rule covers them for the same reason it covers the arguments.
"""
function ArgMemory(dev::LavaDevice, passes::AbstractVector)   # of PassPlan, defined below
    args = 0
    nind = 0
    for pp in passes
        for d in pp.draws
            args += argalign(d.argsize)
        end
        for d in pp.dispatches
            args += argalign(d.argsize)
            d.indirect == 0 || (nind += 1)
        end
    end
    indbase = argalign(args)
    stride = max(argalign(indbase + nind * INDIRECT_STRIDE), 256)
    region = acquire!(pool(dev), dev, Unified(), nothing, stride * ARG_SLOTS;
                      align = 256, blocksize = UNIFIED_BLOCK_SIZE)
    blk = memoryof(region)::BufferBlock
    base = offset(region)
    mb = blk.ref[]::VkManagedBuffer
    # One view per (slot, dispatch), built here because the offsets never move and
    # building one per record is an allocation on the recording path — which is
    # what `get_indirect_buffer` was, once per device-sized dispatch per frame.
    indirect = [Any[LavaArray{UInt32,1}(copy(blk.ref), (3,);
                                        offset = base + (s - 1) * stride + indbase +
                                                 (k - 1) * INDIRECT_STRIDE)
                    for k in 1:nind]
                for s in 1:ARG_SLOTS]
    ArgMemory(region, blk.address + UInt64(base), mb.mapped_ptr + base, stride,
              Vector{Any}(nothing, ARG_SLOTS), 0, indirect)
end

# ↑ moved to src/graph/backend.jl — how deep a plan pipelines is the graph's policy.

# ↑ moved to src/graph/build.jl
"""
An image barrier with everything resolved except the image.

The masks and layouts come from the usage types and are fixed once the plan is
compiled. The image cannot be: a window's target is a different swapchain image
every frame, so it is looked up at record time from the resource.
"""
struct ImageBarrier
    resource::Any
    old::VK.ImageLayout
    new::VK.ImageLayout
    src_stage::VK.PipelineStageFlag2
    src_access::VK.AccessFlag2
    dst_stage::VK.PipelineStageFlag2
    dst_access::VK.AccessFlag2
end

"""
Turn a transition into a barrier, with stages and access ORed over everything it
waits on and the layout taken from the single state it leaves.

This is the point of the exercise: no call site anywhere names a stage.
"""
function ImageBarrier(resource, t::Transition)
    be = VulkanAPI()
    ImageBarrier(resource,
        # A discarding destination does not care what was there, so the old
        # layout is UNDEFINED and no contents have to be preserved.
        discards(t.to) ? VK.IMAGE_LAYOUT_UNDEFINED :
                                layout(be, t.from, Src()),
        layout(be, t.to, Dst()),
        reduce(|, (stages(be, u, Src()) for u in t.waits)),
        reduce(|, (access(be, u, Src()) for u in t.waits)),
        stages(be, t.to, Dst()),
        access(be, t.to, Dst()))
end

"""
    emit_barrier!(emitter, what) -> Bool

Write one barrier where the emitter is writing. Three methods and no branch:
an image barrier looks its image up now (a window's target is a different
swapchain image every frame), a pass's `_DependencyInfo` was built at compile,
and `nothing` is a pass that waits for no one — which is the point, because that
absence is what lets the GPU overlap two passes.

`emit_pass_barrier!` was the second of these, taking a queue and reaching for its
active batch. One function, dispatching on what it was given.
"""
emit_barrier!(::Emitter, ::Nothing) = false

function emit_barrier!(e::Emitter, dep::VK._DependencyInfo)
    VK._cmd_pipeline_barrier_2(e.cmd, dep)
    return true
end

function emit_barrier!(e::Emitter, b::ImageBarrier)
    barrier = VK._ImageMemoryBarrier2(
        b.old, b.new,
        VK.QUEUE_FAMILY_IGNORED, VK.QUEUE_FAMILY_IGNORED,
        target_image(b.resource),
        VK._ImageSubresourceRange(aspect(b.resource),
                                           UInt32(0), UInt32(1), UInt32(0), UInt32(1));
        src_stage_mask = b.src_stage, src_access_mask = b.src_access,
        dst_stage_mask = b.dst_stage, dst_access_mask = b.dst_access)
    VK._cmd_pipeline_barrier_2(e.cmd, VK._DependencyInfo([], [], [barrier]))
    return true
end

# `PassPlan` is Mantle's now — see `src/graph/types.jl`.

# ↑ moved to src/graph/build.jl

function Profiler(ctx, passes)
    n = length(passes)
    nslots = 2n
    info = VK.QueryPoolCreateInfo(VK.QUERY_TYPE_TIMESTAMP, UInt32(nslots))
    pool = VK.unwrap(VK.create_query_pool(ctx.device, info))
    period = Float64(VK.get_physical_device_properties(ctx.physical_device).limits.timestamp_period)
    period > 0 || error("this device reports timestampPeriod = 0, so it cannot time passes")
    Profiler(pool, period, nslots, [pp.pass.name for pp in passes],
             [Float64[] for _ in 1:n], [Float64[] for _ in 1:n], false)
end

# ↑ moved to src/graph/build.jl

"""The backend object behind a resource: what actually gets bound or copied."""
# `storage(::Attr)` is Mantle's now — see `src/graph/build.jl`. An attribute
# forwarding its resource's storage is not this backend's rule.
# What a kernel receives: `LavaDeviceArray` is `(ptr, dims)` and owns nothing, so
# there is no ownership question to answer here — which is why Mantle holding the
# array type removes the problem instead of managing it.
# A borrowed view, not an owning array: the `DataRef` releaser is a no-op, so
# `copy` here bumps a refcount that frees nothing and Mantle stays the owner.
# `todevice` turns this into the `(ptr, dims)` a kernel receives; a host-side
# caller (copy_framebuffer!, a library) gets the handle it needs.
#
# On the BLOCK, which is this backend's `BufferBlock` — see `storage` in
# `src/graph/build.jl`. Written as `storage(t::TransientBuffer{T})` it was the
# same signature as core's and overwrote it, taking Metal and the host backend
# with it.
storage(t::TransientBuffer{T}, block::BufferBlock) where {T} =
    LavaArray{T,1}(copy(block.ref), (t.n,); offset = t.offset)

# What a draw or a dispatch hands the shader: the device-side form, not the host
# handle.
#
# `Adapt` rather than a method per leaf type, because an argument is not always a
# buffer. A wavefront tracer hands a kernel a work queue — a struct of a payload
# array and an atomic counter — and a scene: a BVH, a light set, a material set,
# each a struct of arrays several levels deep. Only the backend knows what a
# shader receives, Lava already states that as `Adapt` rules, and restating the
# leaf case here would mean two answers to one question the day one of them grows
# a case.
#
# The batch the adaptor carries is not used by the conversion — `adapt_storage`
# is a pure strip, and the pin it used to do is a separate pass now. Lifetime is
# the plan's: it holds every argument it names for as long as it lives, which is
# what a per-frame pin would have been bookkeeping for.
# ↑ moved to src/graph/build.jl
# ↑ moved to src/graph/build.jl
# ↑ moved to src/graph/build.jl

# ↑ moved to src/graph/build.jl
# ↑ moved to src/graph/build.jl
# ↑ moved to src/graph/build.jl

# ↑ moved to src/graph/build.jl

"""
The barrier a pass needs: one `VkMemoryBarrier2` per distinct hazard, and no
handles anywhere.

Which hazards exist is core's answer — [`barrierhazards`](@ref), derived per
resource and exact. This maps each to flags, which is the only part that is
Vulkan's.

What went away with the buffer barriers: a span list, a sort by `(handle,
offset)`, an adjacent-span merge, `barrierspan`, `barrierbuffer`, and the
`renameable` special case that already emitted a global barrier for exactly this
reason. That case was the general one all along — a recording cannot bake a
handle for anything that can move, and baking makes everything movable.

It also emits FEWER barriers than the span form on any pass touching several
buffers the same way: one per distinct mask tuple rather than one per buffer.

Built at compile time: a plan is compiled once and replayed, so constructing
VK.jl wrapper objects per frame would be ~2.3 kB of allocation per barrier for a
value that never changes. The low-level `_` form specifically — VK.jl's wrapper
converts to the C struct on every call and allocates 1840 bytes doing it, where
`_DependencyInfo` through `_cmd_pipeline_barrier_2` allocates nothing. Measured,
not assumed.

`nothing` when the pass waits for nothing, which is the point: two passes over
disjoint resources have no barrier between them at all, and that absence is what
lets the GPU overlap them.
"""
function build_pass_barrier(g, ts::Vector{Transition})
    isempty(ts) && return nothing
    be = VulkanAPI()
    mem = VK._MemoryBarrier2[]
    for (waits, to) in barrierhazards(ts)
        push!(mem, VK._MemoryBarrier2(;
            src_stage_mask  = reduce(|, (stages(be, u, Src()) for u in waits)),
            src_access_mask = reduce(|, (access(be, u, Src()) for u in waits)),
            dst_stage_mask  = stages(be, to, Dst()),
            dst_access_mask = access(be, to, Dst())))
    end
    VK._DependencyInfo(mem, VK._BufferMemoryBarrier2[], [])
end

"""
The barrier between two plans that share an arena pool.

Placement makes every tenant of a pool start at offset 0, so two plans on one
device alias each other by construction. Within a plan the `Aliasing` phase finds
that hazard and scopes a barrier to it; across plans there is nothing to scope
to, because the outgoing plan's transients are not this plan's resources and its
last use is not in this plan's schedule.

So it is one global barrier at the head of the recording, and only when the pool
actually has more than one live tenant — a single-plan process pays a dictionary
lookup per run and emits nothing. That is the correct price: the alternative is
tracking cross-plan lifetimes, and a plan boundary is already a point where the
graph knows nothing about what ran before it.

Not needed *between* runs of the same plan: that hazard is the plan's own, and
`nextslot!` plus the derived barriers already cover it.
"""
function handover!(pl::Plan, e::Emitter)
    # `p`, not `pool`. Written `pool = pool(pl.graph.dev)`, which makes `pool` a
    # local for the whole body — so the call on the right resolved to the local
    # that had not been assigned yet, and this threw
    #
    #     UndefVarError: `pool` not defined in local scope
    #
    # on every call. Not intermittently: Julia decides scope statically, so the
    # handover barrier has never been emitted, and two plans sharing an arena
    # have been relying on whatever ordering they happened to get. Found by
    # `test_arena_recording.jl` and `test_devicerange.jl` erroring together, which is
    # how a bug in a shared path presents.
    p = pool(pl.graph.dev)
    # `foldl`, not `any`: short-circuiting would skip recording this plan as the
    # runner of the arenas after the first hit, and the next run would then think
    # it was taking over from itself.
    handed = foldl(pl.arenas; init = Any[]) do acc, ar
        takeover!(p, ar, pl) && push!(acc, ar)
        acc
    end
    isempty(handed) && return false
    both = VK.AccessFlag2(VK.ACCESS_2_MEMORY_READ_BIT) |
           VK.AccessFlag2(VK.ACCESS_2_MEMORY_WRITE_BIT)
    # SCOPED to the arena's own range, not `ALL_COMMANDS` over all memory. The
    # bytes changing hands are exactly the region, and a buffer arena's region is
    # one VkBuffer slice — so this says what it means. The global version cost
    # 1.2 ms per composite frame at 1080p (measured: 1.95 -> 3.15 ms with two
    # plans alternating, while the single-plan path, which never hands over, went
    # 3.218 -> 0.999 ms across the same change). An image arena keeps the global
    # one: layouts are not a byte range.
    bufs = VK._BufferMemoryBarrier2[]
    mems = VK._MemoryBarrier2[]
    for ar in handed
        reg = arenaof(p, ar).region
        blk = reg === nothing ? nothing : memoryof(reg)
        if blk isa BufferBlock
            push!(bufs, VK._BufferMemoryBarrier2(
                VK.QUEUE_FAMILY_IGNORED, VK.QUEUE_FAMILY_IGNORED,
                blk.buffer, UInt64(offset(reg)), UInt64(length(reg));
                src_stage_mask = STAGE2_ALL_COMMANDS, src_access_mask = both,
                dst_stage_mask = STAGE2_ALL_COMMANDS, dst_access_mask = both))
        else
            push!(mems, VK._MemoryBarrier2(;
                src_stage_mask = STAGE2_ALL_COMMANDS, src_access_mask = both,
                dst_stage_mask = STAGE2_ALL_COMMANDS, dst_access_mask = both))
        end
    end
    dep = VK._DependencyInfo(mems, bufs, VK._ImageMemoryBarrier2[])
    emit_barrier!(e, dep)
end

# `peakbytes`/`naivebytes` are NOT defined here. `Plan <: Plan` and
# both read a field the same way in every backend, so core owns them.

# `Compile` is Mantle's now — see `src/graph/types.jl`.

# ↑ moved to src/graph/build.jl
# ↑ moved to src/graph/build.jl
# ↑ moved to src/graph/build.jl
# ↑ moved to src/graph/build.jl
# ↑ moved to src/graph/build.jl
# ↑ moved to src/graph/build.jl
# ↑ moved to src/graph/build.jl
# ↑ moved to src/graph/build.jl
# ↑ moved to src/graph/build.jl

# ↑ moved to src/graph/build.jl

"""
An edge from i to j when they share a resource and at least one writes it.

Read-after-read is deliberately not an edge: two passes that only read the same
thing may run in either order, which is the freedom the scheduler spends.
"""
# ── Place ─────────────────────────────────────────────────────────────────────
"""Back an arena. Buffers get a Lava array; images get raw device memory."""
# The arena is one allocation shared by transients of several element types, so
# its usage bits are the union of what they ask for.
# `BufferBlock` is Mantle's now — see `src/graph/types.jl`.

function rawalloc(dev::LavaDevice, ::Buffers, bytes::Int, usage)
    n = max(bytes, 1)
    # SHADER_DEVICE_ADDRESS is not optional: kernels reach a suballocated
    # transient by `address + offset`, which is the whole bridge.
    u = VK.BufferUsageFlag(usage) |
        VK.BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
        VK.BUFFER_USAGE_STORAGE_BUFFER_BIT |
        VK.BUFFER_USAGE_TRANSFER_SRC_BIT |
        VK.BUFFER_USAGE_TRANSFER_DST_BIT
    buf = unbound_buffer(dev.ctx, n, u)
    req = buffer_requirements(dev.ctx, buf)
    mem = device_memory(dev.ctx, req.size, req.type_bits)
    bind_buffer!(dev.ctx, buf, mem, 0)
    addr = VK.get_buffer_device_address(
        dev.ctx.device, VK.BufferDeviceAddressInfo(buf))
    managed = VkManagedBuffer(buf, mem, UInt64(addr), Ptr{UInt8}(C_NULL), Int(req.size),
                              nothing, nothing, BUF_STATE_ALIVE, 0, false, dev.ctx)
    ref = GPUArrays.DataRef(_ -> nothing, managed)
    return BufferBlock(buf, mem, UInt64(addr), Int(req.size), ref)
end

rawalloc(dev::LavaDevice, ::Images, bytes::Int, bits) =
    device_memory(dev.ctx, max(bytes, 1), bits)

"""
Back a [`Unified`](@ref) arena: one buffer in BAR memory, mapped for the whole
life of the block.

`INDIRECT_BUFFER` unconditionally, because the arena holds both halves of what a
recording reads — argument blocks the host writes and workgroup counts the
command processor does — and one usage mask is what makes them one allocation.
Splitting them would be two blocks to satisfy two bits.

Mapped once here rather than per region: `vkMapMemory` may be called only once
per allocation, and a `Region` is a slice of one. The pointer rides in the
`VkManagedBuffer` the block already carries, so a region's host address is
`ptr + offset` exactly as its device address is `address + offset`.
"""
function rawalloc(dev::LavaDevice, ::Unified, bytes::Int, usage)
    ctx = dev.ctx::VkContext
    n = max(bytes, 1)
    u = VK.BufferUsageFlag(usage) |
        VK.BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
        VK.BUFFER_USAGE_STORAGE_BUFFER_BIT |
        VK.BUFFER_USAGE_INDIRECT_BUFFER_BIT |
        VK.BUFFER_USAGE_TRANSFER_SRC_BIT |
        VK.BUFFER_USAGE_TRANSFER_DST_BIT
    buf = unbound_buffer(ctx, n, u)
    req = buffer_requirements(ctx, buf)
    # Device-local AND host-visible where the device has such a heap, plain
    # host-visible where it does not. `find_memory_type_optional` is the one that
    # may answer `nothing`, which is the whole reason for the pair: a card with no
    # resizable BAR still has to be able to back this arena.
    idx = find_memory_type_optional(ctx, req.type_bits,
              VK.MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
              VK.MEMORY_PROPERTY_HOST_VISIBLE_BIT |
              VK.MEMORY_PROPERTY_HOST_COHERENT_BIT)
    idx === nothing && (idx = find_memory_type(ctx, req.type_bits,
              VK.MEMORY_PROPERTY_HOST_VISIBLE_BIT |
              VK.MEMORY_PROPERTY_HOST_COHERENT_BIT))
    flags = VK.MemoryAllocateFlagsInfo(UInt32(0);
        flags = VK.MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT)
    mem = VK.DeviceMemory(ctx.device, UInt64(req.size), idx; next = flags)
    bind_buffer!(ctx, buf, mem, 0)
    ptr = Ptr{UInt8}(unwrap(VK.map_memory(ctx.device, mem, 0, UInt64(req.size))))
    addr = VK.get_buffer_device_address(ctx.device, VK.BufferDeviceAddressInfo(buf))
    managed = VkManagedBuffer(buf, mem, UInt64(addr), ptr, Int(req.size),
                              nothing, nothing, BUF_STATE_ALIVE, 0, false, ctx)
    ref = GPUArrays.DataRef(_ -> nothing, managed)
    return BufferBlock(buf, mem, UInt64(addr), Int(req.size), ref)
end

# Nothing a caller of this arena declares narrows it: the usage mask above is
# fixed, because what a recording reads is the same on every plan.
constraintof(::LavaDevice, ::Unified, ts) = UInt32(0)
mergeconstraints(::LavaDevice, ::Unified, a::Integer, b::Integer) = a | b

"""
Destroy a block's Vulkan objects.

This used to be a no-op — "Lava frees its own" — which was true while the only
blocks were graph arenas whose `VkBuffer` and `VkDeviceMemory` wrappers carried
Julia finalizers and got collected eventually. "Eventually" stopped being good
enough when this pool took over device allocation: `trim_gpu_pool!` exists so a
caller can say "I have finished and want the VRAM back NOW", and a figure that
depends on when the GC next runs is not an answer to that.

Destructor failures are logged rather than thrown. The two callers are `trim!`
and `destroy_pool!`, and the second runs while a device is being torn down or is
already lost, where the driver may have released the handles itself.
"""
function rawfree(::LavaDevice, blk::BufferBlock)
    try
        blk.buffer.destructor()
        blk.memory.destructor()
    catch ex
        safe_fin_log("Mantle rawfree: Vulkan destructor failed " *
                     "(expected while a device is being reset): " *
                     sprint(showerror, ex) * "\n")
    end
    return nothing
end

# Image arenas hand back raw `VkDeviceMemory` from `rawalloc(dev, Images(), …)`.
rawfree(::LavaDevice, mem::VK.DeviceMemory) = (mem.destructor(); nothing)

# What memory the given transients can legally share. For images that is the
# INTERSECTION of their type bits — `device_memory` errors on an empty one rather
# than picking a type some image cannot use.
constraintof(::LavaDevice, ::Images, ts) = reduce(&, (t.req.type_bits for t in ts))
constraintof(::LavaDevice, ::Buffers, ts) =
    reduce(|, (extrausage(eltype(t)) for t in ts))

# A block may host a request when its memory satisfies it: for images the block's
# type bits must still include a type the request allows; for buffers the block's
# usage flags must be a superset of what the request needs.
compatible(::LavaDevice, blk::Integer, req::Integer) = (blk & req) == req

# Union for buffers, intersection for images — and the two are not symmetrical,
# which is why core cannot guess. A buffer arena must PERMIT everything any
# tenant does with it; an image arena's type has to be one every image can bind
# to, and an empty intersection means no allocation can serve them all.
mergeconstraints(::LavaDevice, ::Buffers, a::Integer, b::Integer) = a | b
function mergeconstraints(::LavaDevice, ::Images, a::Integer, b::Integer)
    m = a & b
    m == 0 && throw(ArgumentError(
        "arena Images() has no memory type every image in it can bind to: this " *
        "plan's images and an earlier plan's on this device intersect to nothing, " *
        "so one allocation cannot serve both."))
    return m
end
compatible(::LavaDevice, blk, req) = blk == req

# ↑ moved to src/graph/build.jl

function materialize!(t::VulkanTransientImage{T}, slab, offset) where {T}
    t.memory = slab                       # the image outlives the call; the slab must too
    bind_image!(vk_context(), t.image, slab, offset)
    t.view = image_view(vk_context(), t.image, t.format, aspect(T))
end

"""`VK_KHR_maintenance3`'s `maxMemoryAllocationSize`. See `maxalloc`."""
function maxalloc(dev::LavaDevice)
    props = VK.get_physical_device_properties_2(
        dev.ctx.physical_device, VK.PhysicalDeviceMaintenance3Properties)
    m3 = props.next::VK.PhysicalDeviceMaintenance3Properties
    n = m3.max_memory_allocation_size
    # NVIDIA answers 0xffff_ffff_ffff_ffff — "no limit" — which does not fit an
    # `Int`, and `Int(n)` threw here on the first device that said it. That is
    # the same thing core's default means, so say it the same way.
    return n >= UInt64(typemax(Int)) ? typemax(Int) : Int(n)
end

"""
Assign offsets, one placement per arena.

Peak is summed over arenas because they are separate allocations; the alternative
would be to report the larger and pretend the other is free.
"""
# ── Aliasing ──────────────────────────────────────────────────────────────────
const EMPTY_HANDOVER = Tuple{Int,Int}[]

# ↑ moved to src/graph/build.jl

# ↑ moved to src/graph/build.jl

# ↑ moved to src/graph/build.jl
"""
Walk the passes in order carrying per-resource state. What a pass needs before it
runs is what `transition!` appends while its usages are replayed, so a pass
touching only resources nobody else touched needs nothing and can overlap
whatever ran before it.

Coverage is by position *and* mask width. An earlier barrier covers a later
dependency only if it is at least as wide: a write-only alias barrier does not
make a subsequent read visible, and treating it as coverage silently drops that
read's dependency. That was the second fuzzing bug.
"""
function run!(::Barriers, c::Compile)
    be = VulkanAPI()
    g = c.graph
    states = Dict{Int,ResourceState}()
    settled = Dict{Int,Int}()
    last_barrier = 0
    covered_src_stage = covered_dst_stage = VK.PipelineStageFlag2(0)
    covered_src_access = covered_dst_access = VK.AccessFlag2(0)
    # `ALL_COMMANDS` and `MEMORY_READ|MEMORY_WRITE` are Vulkan's "any" encodings,
    # and as bit patterns they are single distinct bits — so a bitwise test does
    # not see that they cover every stage and every access. Expanded here, on
    # both sides: a barrier that named everything covers a later one that names
    # something, and a later one that names everything is still only covered by
    # another that does.
    #
    # Nothing in the lowering produces either encoding any more, so this changes
    # no answer today. It stays because it is the semantics: coalescing was wrong
    # about them for as long as they were used, and it was invisible precisely
    # because *every* `Storage` barrier was `ALL_COMMANDS` and so both sides held
    # the same value and compared equal by accident. Reintroduce one without this
    # and the coalescing goes quietly back to dropping nothing.
    # Converted, not written bare: VK.jl types the `_2_` constants that still
    # fit in 32 bits as the sync1 flag type, and BitMasks refuses to mix the two.
    anystage = VK.PipelineStageFlag2(VK.PIPELINE_STAGE_2_ALL_COMMANDS_BIT)
    anyread = VK.AccessFlag2(VK.ACCESS_2_MEMORY_READ_BIT)
    anywrite = VK.AccessFlag2(VK.ACCESS_2_MEMORY_WRITE_BIT)
    every(x) = ~zero(x)
    widen(s::VK.PipelineStageFlag2) = (s & anystage) != zero(s) ? every(s) : s
    widen(a::VK.AccessFlag2) =
        (a & anyread) != zero(a) && (a & anywrite) != zero(a) ? every(a) : a
    subsumed(have, want) = (widen(want) & ~widen(have)) == zero(want)

    # A resource's state at the start of a replay is not "unknown" for anything
    # that outlives the frame: the window comes back from present, an offscreen
    # target from whatever last read it, and every other resource from whatever
    # this same plan did to it last time round.
    #
    # That last one has to be seeded from the *end* of the schedule, because a
    # plan is replayed: frame N+1's first use of a resource follows frame N's last
    # use of it, and with frames in flight the two overlap on the GPU. Seeding
    # `Undefined` instead left the first barrier of a frame with no source scope —
    # nothing to wait for — which synchronization validation reports as a
    # write-after-write against the previous frame's store op, and which is a race
    # the moment the frame loop stops flushing.
    #
    # The layout is unaffected: a discarding destination still transitions from
    # UNDEFINED (see `ImageBarrier`). This is only about what the barrier waits on.
    # Atomic segments, so a partial overlap is tracked rather than refused.
    #
    # This is what VVL's `AccessMap` does incrementally (`layers/sync/sync_access_map.h`:
    # `Split` at each range bound, then `InfillGaps`), done once instead: by the
    # time a plan compiles, every range that will ever be declared is known, so
    # the cuts can be taken up front and the walk left alone. Each usage then
    # stands for the segments its range covers, and a whole-resource usage stands
    # for all of them — after which two usages either name the same segment or do
    # not, which is the only question the per-resource walk knows how to answer.
    slices = sliceindex(g)
    segs = Dict{Int,Vector{Int}}()
    let byparent = Dict{Int,Vector{UnitRange{Int}}}()
        for ((pid, r), _) in g.views
            push!(get!(byparent, pid, UnitRange{Int}[]), r)
        end
        for (pid, ranges) in byparent
            parent = g.ids.by_id[pid]
            n = length(parent)
            cuts = sort!(unique!(vcat([1, n + 1], first.(ranges), last.(ranges) .+ 1)))
            spans = [cuts[k]:(cuts[k + 1] - 1) for k in 1:(length(cuts) - 1)]
            filter!(!isempty, spans)
            ids = map(spans) do s
                resourceid(g, get!(() -> BufferRange(parent, s), g.views, (pid, s)))
            end
            segs[pid] = ids                      # the whole buffer is every segment
            for r in ranges
                sid = resourceid(g, g.views[(pid, r)])
                segs[sid] = [ids[k] for (k, s) in enumerate(spans) if first(s) >= first(r) &&
                                                                      last(s) <= last(r)]
            end
        end
    end
    segments_of(id) = get(segs, id, (id,))

    final = Dict{Int,Type}()
    for p in ordered(c), (id, U) in p.usages, sid in segments_of(id)
        final[sid] = U
    end

    state(id) = get!(states, id) do
        r = g.ids.by_id[id]
        u = get(final, id, nothing)
        u === nothing && (u = initial_state(r))
        u === nothing ? ResourceState(resourcekind(r)) :
                        ResourceState(resourcekind(r), u)
    end

    for (i, p) in enumerate(ordered(c))
        pre = Transition[]
        for (id, U) in p.usages, sid in segments_of(id)
            transition!(pre, be, sid, state(sid), U)
        end

        # A layout change is per image, so no barrier on another resource can have
        # performed it however wide its masks were. Only the memory dependency is
        # coalescable; an image transition dropped here never becomes an
        # `ImageBarrier` below, and the copy after a second colour attachment then
        # reads it in COLOR_ATTACHMENT_OPTIMAL.
        # Coalescing across resources is what a global barrier bought, and it is
        # exactly what a scoped one cannot: a buffer barrier orders that buffer's
        # memory and nothing else, so a barrier emitted for resource A never
        # stands in for a hazard on B however wide its masks are. Dropping one on
        # that reasoning is a race — it is how the showcase's `sun cull` lost the
        # barrier between the clear of its counter and the atomic that reads it.
        #
        # Nothing is left to remove per resource either: `transition!` emits only
        # where a resource's own state actually changes, so the walk is already
        # minimal and `pre` *is* the local hazard set. Kept as a flag because
        # `bench/` still compiles both ways to measure what the old strategy did.
        needed = copy(pre)

        # The barrier where bytes change hands, derived like every other one.
        #
        # It is the one transition not produced by `transition!`, because the
        # hazard is between two resources that never mention each other: no
        # per-resource sequence can see that a different transient was living in
        # this memory. What the compiler *does* know is which ones vacated it —
        # the placer worked that out — and what each was last doing, which is the
        # state the walk above is carrying right now. So it waits for exactly
        # those usages, at exactly the new tenant's first one.
        #
        # It used to name every stage and both access directions instead. That is
        # correct and says nothing: a barrier that waits for everything cannot be
        # wrong, and cannot be checked either.
        handovers = get(analysis(c).alias_begins, i, EMPTY_HANDOVER)
        for newi in unique(first(h) for h in handovers)
            to = usage_of(p, resourceid(g, c.graph.transients[newi]), g)
            to === nothing && continue
            waits = Type[]
            for (a, o) in handovers
                a == newi || continue
                for u in lastuses(states, slices, resourceid(g, c.graph.transients[o]))
                    u in waits || push!(waits, u)
                end
            end
            isempty(waits) && continue
            push!(needed, Transition(0, first(waits), waits, to))
        end

        if !isempty(needed)
            last_barrier = i
            covered_src_stage = reduce(|, (stages(be, u, Src()) for t in needed for u in t.waits))
            covered_src_access = reduce(|, (access(be, u, Src()) for t in needed for u in t.waits))
            covered_dst_stage = reduce(|, (stages(be, t.to, Dst()) for t in needed))
            covered_dst_access = reduce(|, (access(be, t.to, Dst()) for t in needed))
        end
        c.prepass[p] = needed
        append!(c.transitions, pre)
        for (id, _) in p.usages
            settled[id] = i
        end
    end
    c
end

# ── Pipelines ─────────────────────────────────────────────────────────────────
"""
Resolve and build every draw's pipeline here rather than per frame, so `run!`
only packs push constants and submits.

Two draws whose binding tuples share a type resolve to the same compiled
pipeline. That is the whole reason `Attr{T}` erases a `Buffer` from a `Scalar`.
"""
# More specific than core's KA default in `graph/kalaunch.jl`: this backend
# records commands into a command buffer rather than closing over callables.
function run!(::Pipelines, c::Compile{LavaDevice})
    g = c.graph
    argcursor = 0        # every draw's argument block, laid out once
    # …and every device-sized dispatch's indirect command beside them. Counted
    # here rather than allocated at record because the plan knows how many it has,
    # which is the same reason `argcursor` is here.
    indcursor = 0
    # A layout change is per-image and cannot be folded into the pass's one memory
    # barrier, so the needed transitions split by resource kind: images become an
    # image barrier each, everything else is ORed into the memory barrier.
    isimage(t) = t.resource != 0 && resourcekind(g.ids.by_id[t.resource]) isa ImageKind
    for p in ordered(c)
        need = c.prepass[p]
        imgs = [ImageBarrier(g.ids.by_id[t.resource], t) for t in need if isimage(t)]
        rest = filter(!isimage, need)
        cds = CompiledDraw[]
        # One render area for the pass, so attachments that disagree about their
        # size would silently render into part of the larger one. Checked once at
        # compile rather than per frame.
        if p.kind === :render
            want = target_extent(first_target(p))
            for t in (p.targets..., p.depth)
                t === nothing && continue
                target_extent(t) == want ||
                    throw(ArgumentError("pass \"$(p.name)\": attachments are $(want) and " *
                                        "$(target_extent(t)); every attachment shares one render area"))
            end
        end
        ad = adaptor(g.dev.bq)
        for d in p.draws
            args = devargs(ad, rawargs(d.args))
            vtt = typeof(convert_args(args))
            ftt = typeof(convert_args(devargs(ad, rawargs(d.frag_args))))
            vfn, vtt, ffn, ftt = resolve_shader_pair(d.shader, vtt, ftt)
            # The pass says whether there is a depth attachment, and with dynamic
            # rendering the pipeline has to declare the same thing: a pipeline
            # built for one and drawn into a pass without it is invalid, and the
            # reverse is too.
            shader, compiled = ensure_compiled_with_shader!(d.shader, vfn, ffn, vtt, ftt;
                                                                color_format = VK.Format[target_format(t) for t in p.targets],
                                                                depth_format = p.depth === nothing ?
                                                                    VK.FORMAT_UNDEFINED :
                                                                    target_format(p.depth))
            push!(c.pipelines, compiled.pipeline)
            # The block is laid out to whichever stage's shader reads it.
            # `ensure_compiled_with_shader!` hands back the vertex shader because
            # that is the usual answer; a fullscreen pass whose vertex stage takes
            # nothing and whose fragment stage reads a g-buffer is the other one.
            isempty(d.frag_args) || (shader = get_or_compile_gfx(ffn, ftt, :fragment))
            packed = isempty(d.frag_args) ? d.args : d.frag_args
            info = shader.push_info
            nbytes = info.arg_buffer_size +
                     compute_inline_extra_from_byval(info.byval_llvm_sizes)
            push!(cds, CompiledDraw(compiled, shader, packed, d.count, argcursor, nbytes))
            argcursor += argalign(nbytes)
        end
        # `Any[]` and then narrowed, not `CompiledDispatch[]`: a pass declares
        # `Dispatch`es or `Trace`s, `compile_dispatch` answers both, and the two
        # compile to different types. `identity.` gives back a concrete vector
        # when the pass holds one kind — which is every pass in tree — so the
        # per-frame loop over `pp.dispatches` still specialises.
        compiled = Any[]
        for d in p.dispatches
            ind = d.ndrange isa DeviceRange ? (indcursor += 1) : 0
            cd = compile_dispatch(g.dev, d, argcursor, ind)
            # Not into `c.pipelines`: that set answers how many shaders the draws
            # resolved to — two draws sharing one is the thing worth counting —
            # and a dispatch's pipeline is held by its `CompiledDispatch` anyway.
            push!(compiled, cd)
            argcursor += argalign(cd.argsize)
        end
        cps = identity.(compiled)
        push!(c.passes, PassPlan(p, cds, cps, imgs, need, build_pass_barrier(g, rest)))
    end
    c
end

"""
Compile one dispatch: everything `KernelAbstractions` would work out per launch,
worked out once.

The pieces are the ones `ka_launch!` uses — an iteration plan for the ndrange and
a launch plan for the argument types — taken here so that recording is only
packing and `vk_dispatch!`. The kernel itself is compiled by the launch plan, and
this is the only place a Mantle frame can compile one.
"""
function compile_dispatch(dev::LavaDevice, d::Dispatch, argoff::Int, indirect::Int)
    # Three arguments. `kernelfor` took two when it lived here and takes the
    # backend now that it is core's (`graph/kalaunch.jl`), which is what makes it
    # answerable by a backend at all; this call site kept the old arity and threw
    # a `MethodError` on the first dispatch a plan compiled.
    obj = kernelfor(d.kernel, d.group, backend(dev))
    isempty(fieldnames(typeof(obj.f))) ||
        throw(ArgumentError("dispatch!: the kernel closes over $(fieldnames(typeof(obj.f))). " *
                            "A plan resolves its arguments once, so a captured device array " *
                            "would be the one it had when the plan was built — pass it as a " *
                            "dispatch argument instead, where a rename is followed."))
    raw = rawargs(d.args)
    args = devargs(adaptor(dev.bq), raw)
    nd = dispatchrange(d.ndrange)
    # `nothing` for the workgroup size, not `d.group`: a group given to
    # `dispatch!` is baked into the kernel's type by `kernelfor`, and KA reads it
    # from there. Passing it here as well made a second, differently keyed
    # iteration plan for the same launch.
    iter = get_or_build_iter_plan(obj, nd, nothing, dev.ctx)
    tlas = find_tlas_in_args(raw)
    all_args = (obj.f, iter.ka_ctx, args...)
    launch = launch_plan(dev.bq, obj.f, all_args, iter.ws_3d, tlas !== nothing)
    CompiledDispatch(launch, iter, nd, obj, obj.f, d.args, d.ndrange,
                     tlas !== nothing, argoff, launch.total_size, indirect)
end

"""
What this backend needs to record a trace, worked out once at compile.

The ray-tracing counterpart of a `LaunchPlan`: the `VkPipeline` with its shader
binding table, the raygen function whose adapted form is packed as argument one,
and the argument LAYOUT — offsets, by-value sizes, and the total the raygen
shader's push-constant buffer occupies.

That total is the whole reason this is resolved at compile rather than at
record: it is the `argsize` `ArgMemory` needs in order to give the trace a fixed
offset in the plan's slot, which is what makes the arguments rebindable.
"""
struct VulkanTracePipeline{P,D,O,B}
    pipeline::P              # LavaRTPipeline: VkPipeline + SBT + layout
    # The DESCRIPTION, kept beside the compiled pipeline rather than just the
    # raygen function: the raygen is packed as `all_args[1]`, and all four of
    # raygen/chit/miss/anyhit have to be pinned per record, because a per-material
    # closest-hit closes over the device arrays its material reads.
    desc::D
    offsets::O
    byval::B
    argbytes::Int            # raygen push_info.arg_buffer_size, before inline extra
end

"""The `LavaTLAS` a trace descriptor set names, from what the caller declared."""
tlasof(a::AdaptedAccel) = tlasof(a.hwtlas)
tlasof(t::VulkanTLAS) = t.hw_tlas
tlasof(t::LavaTLAS) = t
tlasof(::Nothing) = throw(ArgumentError(
    "trace!: the acceleration structure has no hardware TLAS. It was stripped " *
    "before the pass ran, or the accel was built for a ray-query traversal, " *
    "which `dispatch!` records rather than `trace!`."))

"""
Compile one trace: the pipeline and the argument layout, worked out once.

The `Trace` counterpart of [`compile_dispatch`](@ref) above, and reached by the
same name so the Pipelines phase needs no branch — a pass holds `Dispatch`es or
`Trace`s and multiple dispatch decides which of these runs.

`rt_compiled_for` has to happen HERE and not at record time, and not only for
speed: a cold compile flushes the batch queue to upload the shader binding table,
which is legal while a plan is being compiled and is not while a frame is being
recorded into an open batch.
"""
function compile_dispatch(dev::LavaDevice, t::Trace, argoff::Int, indirect::Int)
    raw = rawargs(t.args)
    vk_pipeline, raygen, offsets, byval = rt_compiled_for(dev.bq, t.pipeline, raw)
    argbytes = raygen.push_info.arg_buffer_size
    total = argbytes + compute_inline_extra_from_byval(byval)
    compiled = VulkanTracePipeline(vk_pipeline, t.pipeline, offsets, byval, argbytes)
    CompiledTrace(compiled, t.accel, t.args, t.ndrange, argoff, total, indirect)
end

"""
    argwrites(entry, argoff, offsets, byval, all_args, nprefix) -> Vector{ArgWrite}

The `Ref` arguments of one compiled entry, with the byte offset each occupies.

`all_args` is the tuple as `pack_args_direct!` sees it and `nprefix` is how many
of its leading entries are not the caller's — two for a dispatch (the kernel and
the KA context), one for a trace (the adapted raygen), none for a draw. The
layout rule is [`slotpositions`](@ref), the same one that packed them.

A `Ref` to a by-value aggregate — a camera, a block of filter parameters — is
handled, not skipped, and that is why this walks EVERY slot rather than only the
`Ref`s: such an argument goes into the inline area at an offset that depends on
every by-value argument before it, so the running `inline_offset` has to be
carried across the whole tuple.

It is carried by PACKING, not by predicting. This walk performs a real
`pack_arg!` for every argument and keeps the offset each one was handed, so the
positions recorded are the ones the packer actually used. A first version
predicted them instead, from a `standalone_slot(T)` restating the generic
`pack_arg!`'s branch condition — and `pack_arg!` has specialised methods for
`UInt64`, `Ptr` and `VkManagedBuffer` ahead of that generic one, so the prediction
diverged at the first argument reaching one of them and every by-value argument
after it landed wrong. `verifywrites` caught it, which is what it is for.

Packing here is free and safe: `record!` has just packed these same values, so this
writes the bytes that are already there.

Skipping an unsupported `Ref` would be invisible rather than merely incomplete —
`verifywrites` compares bytes after a pack where nothing has changed, so a
MISSING write leaves them identical and passes, and the argument silently never
updates. Hikari's camera is exactly that case.
"""
function argwrites(userargs::Tuple, argoff::Int, offsets, byval, base_size::Int,
                   all_args::Tuple, nprefix::Int, ptr::Ptr{UInt8}, bda::UInt64,
                   batch)
    ws = ArgWrite[]
    slots = slotpositions(map(typeof, all_args))
    inline = base_size
    for (layout_i, arg_i) in enumerate(slots)
        here = inline
        # The real pack, so the offset recorded is the one the packer used.
        inline = pack_arg!(all_args[arg_i], ptr, bda, offsets[layout_i],
                           byval[layout_i], inline, batch)
        j = arg_i - nprefix
        if 1 <= j <= length(userargs) && userargs[j] isa Base.RefValue
            push!(ws, ArgWrite(userargs[j], argoff, offsets[layout_i],
                               byval[layout_i], here))
        end
    end
    return ws
end

# Per entry kind, because only the entry knows what its packed tuple looks like
# in front of the caller's arguments. Each mirrors the `all_args` its `repack!`
# builds, and that is the coupling to keep an eye on: if one changes, so must the
# other. `bake!`'s verification is what catches it if they drift.
function argwrites(e::Emitter, d::CompiledDispatch)
    am = e.args
    all_args = (d.kernel, d.iter.ka_ctx, devargs(adaptor(e), rawargs(d.args))...)
    lp = d.launch
    off = e.base + d.argoff
    argwrites(d.args, d.argoff, lp.offsets, lp.byval_sizes, lp.arg_buffer_size,
              all_args, 2, am.ptr + off, am.address + off, e.owner)
end

function argwrites(e::Emitter, dr::CompiledDraw)
    am = e.args
    all_args = devargs(adaptor(e), rawargs(dr.args))
    info = dr.shader.push_info
    off = e.base + dr.argoff
    argwrites(dr.args, dr.argoff, info.arg_offsets, info.byval_llvm_sizes,
              info.arg_buffer_size, all_args, 0, am.ptr + off, am.address + off,
              e.owner)
end

function argwrites(e::Emitter, t::CompiledTrace)
    am = e.args
    c = t.compiled
    ad = adaptor(e)
    all_args = (Adapt.adapt(ad, c.desc.raygen_func), devargs(ad, rawargs(t.args))...)
    off = e.base + t.argoff
    argwrites(t.args, t.argoff, c.offsets, c.byval, c.argbytes, all_args, 1,
              am.ptr + off, am.address + off, e.owner)
end

"""
Perform one [`ArgWrite`](@ref): read the `Ref` as it is now and store it.

`pack_arg!` and nothing else — the same store the full pack does, at the offset
and with the inline position the full pack used, both recorded by
[`argwrites`](@ref). The pointer is the START of the entry's argument region, not
of the individual slot, because `pack_arg!` adds the offsets itself and a by-value
argument's inline position is relative to that same base.
"""
function writearg!(e::Emitter, w::ArgWrite)
    am = e.args
    entry = e.base + w.entryoff
    v = Adapt.adapt(adaptor(e), argvalue(w.ref))
    pack_arg!(v, am.ptr + entry, am.address + entry, w.slotoff, w.byval, w.inline,
              e.owner)
    return nothing
end

"""
Write the current arguments of one compiled thing into the plan's slot.

What `rebind!` does per entry, as a method rather than a branch: the loop had a
dispatch's iteration-plan lookup inlined into it, so there was nowhere for a
trace to go. Local to this backend — nothing in core calls it, and importing the
name would make a private recording helper into Mantle API.
"""
function repack!(e::Emitter, d::CompiledDispatch)
    nd = dispatchrange(d.ndrange)
    it = nd == d.nd0 ? d.iter :
         get_or_build_iter_plan(d.obj, nd, nothing, e.ctx)
    it.nblocks == 0 && return nothing
    packdispatch!(e, d, it)
    return nothing
end

repack!(e::Emitter, t::CompiledTrace) = (packtrace!(e, t); nothing)

# A draw packs the same way it does while being recorded, minus the recording.
# `record_draw!` was where these six lines lived and `rebind!` had a copy of
# them inlined into its loop, which is how the two came to disagree: the copy
# never learned about a trace, and adding one meant adding it twice.
function repack!(e::Emitter, d::CompiledDraw)
    am = e.args
    off = e.base + d.argoff
    info = d.shader.push_info
    pack_args_direct!(e.owner, am.ptr + off, am.address + off, info.arg_offsets,
                      info.arg_buffer_size, info.byval_llvm_sizes,
                      devargs(adaptor(e), rawargs(d.args)))
    return nothing
end

"""
Pack a trace's arguments into the plan's slot. The counterpart of `packdispatch!`.

Into `am` and not into `get_arg_buffer(bq, …)`, which is the whole difference
between this and the unmodelled path: the queue's scratch has its bump pointer
rewound every time the queue drains, so an address from it means nothing to a
replay. A slot in the plan's argument memory belongs to the plan for as long as
the plan lives, which is what recording needs and what `rebind!` rewrites.
"""
function packtrace!(e::Emitter, t::CompiledTrace)
    am = e.args
    off = e.base + t.argoff
    c = t.compiled
    owner = e.owner
    # Pinned as the unmodelled path pins: the shaders are closures, and a
    # per-material closest-hit holds the device arrays of the material it shades.
    pin_leaves!(owner, c.desc.raygen_func)
    for chit in c.desc.closesthit_funcs
        pin_leaves!(owner, chit)
    end
    pin_leaves!(owner, c.desc.miss_func)
    pin_leaves!(owner, c.desc.anyhit_func)     # a no-op on `nothing`
    ad = LavaAdaptor(owner)
    # `rawargs` first, so a `Ref` argument is re-read NOW rather than frozen at
    # compile — that is how a new sample index reaches a plan compiled once.
    raw = rawargs(t.args)
    pin_leaves!(owner, raw)
    all_args = (Adapt.adapt(ad, c.desc.raygen_func), devargs(ad, raw)...)
    pack_args_direct!(owner, am.ptr + off, am.address + off,
                      c.offsets, c.argbytes, c.byval, all_args)
    return am.address + off
end

"""The acceleration structures a trace reads, held for as long as it can run."""
function pintrace!(e::Emitter, tlas::LavaTLAS)
    pin!(e, tlas.accel)
    pin!(e, tlas.storage)
    for blas in tlas.blases
        pin!(e, blas.accel)
        pin!(e, blas.storage)
    end
    return nothing
end

"""
Emit the trace itself. A host-side ray count traces directly; a
[`DeviceRange`](@ref) prepares an indirect command from a count on the device.

Two methods rather than a branch, which is how `emitlaunch!` above already
distinguishes the same two cases.

The prepare is NOT fused with the compute prepares `emitprepares!` writes: it is
a different kernel writing a different command — `(n_rays, 1, 1)`, not workgroup
counts — and there is one trace per pass in every graph in tree. Its barrier is
the same one, and it is emitted here so the ordering is stated where the pair is
written rather than inferred from where they sit.
"""
function tracelaunch!(e::Emitter, r::DeviceRange, t::CompiledTrace, tlas,
                      name::AbstractString)
    indirect = indirectof(e.args, t.indirect)
    emitkernel!(e, prepare_indirect_rt_kernel, indirect, storage(r.count);
                ndrange = 1, workgroup_size = (1, 1, 1))
    emit_barrier!(e, preparebarrier())
    argaddr = packtrace!(e, t)
    pintrace!(e, tlas)
    emit_trace_indirect!(e, t.compiled.pipeline, tlas, argaddr, indirect, name)
    return nothing
end

function tracelaunch!(e::Emitter, n, t::CompiledTrace, tlas, name::AbstractString)
    argaddr = packtrace!(e, t)
    pintrace!(e, tlas)
    emit_trace!(e, t.compiled.pipeline, tlas, argaddr, Int(n), 1, 1, name)
    return nothing
end

"""Emit a trace: resolve the acceleration structure, then launch."""
emitdispatch!(e::Emitter, t::CompiledTrace, name::AbstractString) =
    tracelaunch!(e, t.ndrange, t, tlasof(argvalue(t.accel)), name)

# An ndrange fixed when the graph was built, or one that is read per frame — the
# same distinction `drawover` makes for a draw's vertex count.
# ↑ moved to src/graph/build.jl
# ↑ moved to src/graph/build.jl
# ↑ moved to src/graph/build.jl
# ↑ moved to src/graph/build.jl

# ↑ moved to src/graph/build.jl

# ↑ moved to src/graph/build.jl

# ↑ moved to src/graph/build.jl

"""
Read back whatever the last profiled frame left, without waiting for it.

`WAIT_BIT` would block until the frame completed, which is the one thing a
profiler must not do: it would report the frame time of a program that stalls
every frame, which is not the program being measured. `WITH_AVAILABILITY_BIT`
asks instead, and an unavailable frame is one skipped sample.
"""
function collect!(prof::Profiler, ctx)
    prof.pending || return prof
    # Two words per slot: the value and whether it is there yet.
    raw = Vector{UInt64}(undef, 2 * prof.nslots)
    ok = GC.@preserve raw begin
        VK.get_query_pool_results(
            ctx.device, prof.pool, UInt32(0), UInt32(prof.nslots),
            sizeof(raw), Ptr{Nothing}(pointer(raw)), UInt64(2 * sizeof(UInt64));
            flags = VK.QUERY_RESULT_64_BIT |
                    VK.QUERY_RESULT_WITH_AVAILABILITY_BIT)
    end
    prof.pending = false
    for i in 1:length(prof.gpu_ns)
        lo, hi = 2 * (2i - 2) + 1, 2 * (2i - 1) + 1     # value words of the pair
        raw[lo + 1] == 0 && continue                     # start not available
        raw[hi + 1] == 0 && continue                     # end not available
        ns = elapsed(raw[lo], raw[hi], prof.period_ns)
        ns === nothing || sample!(prof.gpu_ns[i], ns)
    end
    prof
end

# ↑ moved to src/graph/build.jl

# ↑ moved to src/graph/build.jl

"""
Every attachment has to cover the render area, and a window changes size under a
plan compiled for the old one: the swapchain follows a resize, a transient sized
when the graph was built does not. Vulkan calls the result undefined
(VUID-VkRenderingInfo-pNext-06079) and RADV draws it anyway, so without this it is
a wrong picture rather than a message.
"""
function checkextents(pl::Plan)
    for pp in pl.passes
        p = pp.pass
        p.kind === :render || continue
        want = target_extent(first_target(p))
        for t in (p.targets..., p.depth)
            t === nothing && continue
            e = target_extent(t)
            (e[1] < want[1] || e[2] < want[2]) &&
                error("pass \"$(p.name)\": the render area is $(want[1])x$(want[2]) " *
                      "but an attachment is $(e[1])x$(e[2]). A resize follows the " *
                      "swapchain and not a transient sized when the graph was built — " *
                      "rebuild the graph and the plan at the new size.")
        end
    end
    nothing
end

# ↑ moved to src/graph/build.jl

# ↑ moved to src/graph/build.jl

# ↑ moved to src/graph/build.jl

# ↑ moved to src/graph/build.jl

"""
Bracket one pass with timestamps and time its recording.

The query pool has to be reset on the device before it is written again, and the
reset is a command like any other, so it goes at the head of the frame's
recording rather than at the end of the last one — a frame that was never
recorded has nothing to reset.

The start timestamp is at `TOP_OF_PIPE` and the end at `BOTTOM_OF_PIPE`, which
brackets everything the pass does. Two adjacent passes therefore overlap in what
they report, because the GPU is free to overlap them; a sum of pass times is not
the frame time and is not meant to be.
"""
function profiled!(f, pl::Plan, e::Emitter, i::Integer)
    prof = pl.profiler
    prof === nothing && return f()
    VK.cmd_write_timestamp(e.cmd, VK.PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                                    prof.pool, UInt32(2i - 2))
    t0 = time_ns()
    r = f()
    sample!(prof.host_ns[i], Float64(time_ns() - t0))
    # One command buffer, so the second timestamp goes where the first did. It
    # used to re-open the batch here, with the comment "a pass may have split the
    # batch" — which is exactly what a plan's own command buffer cannot do.
    VK.cmd_write_timestamp(e.cmd, VK.PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                                    prof.pool, UInt32(2i - 1))
    prof.pending = true
    r
end

"""
    emit!(emitter, plan; updates = false)

Write the plan's passes, in the order the compile settled on, into the one
command buffer the emitter holds.

`updates = false` is the recorded case and is the default: an update writes
through a fresh store on the rename route, so the copy's destination handle is
not the one a recording could hold — see `renameable`. `run!` emits those alone,
fresh, in front of the recording that consumes them.

Three keyword arguments went from here. `derived` and `suppress` selected
between the barriers the graph computed and the ones Lava's recorder inserted
per dispatch, which was an A/B for a comparison that has been made: an emitter
writes what the plan says and there is no automatic barrier to suppress.
"""
function emit!(e::Emitter, pl::Plan; updates::Bool = false)
    g = pl.graph
    # Before anything this plan emits: the pool it is about to write may still be
    # being read by whichever plan ran last.
    handover!(pl, e)
    if pl.profiler !== nothing
        VK.cmd_reset_query_pool(e.cmd, pl.profiler.pool, UInt32(0),
                                UInt32(pl.profiler.nslots))
    end
    for (i, pp) in enumerate(pl.passes)
        (!updates && pp.pass.kind === :update) && continue
        profiled!(pl, e, i) do
            emitpass!(e, g, pp)
        end
    end
    nothing
end

# ↑ moved to src/graph/build.jl

"""One pass: its barriers, then whatever its kind does."""
function emitpass!(e::Emitter, g::Graph, pp::PassPlan)
    p = pp.pass
    # Mantle's own barriers, derived from the declared usage sequence. Layout
    # changes first: a pass may both need an image transitioned and wait on a
    # buffer, and the two are separate Vulkan barriers.
    #
    # An update pass whose refs are all clean writes nothing, so its barriers
    # order against a write that did not happen. Skipping a barrier whose
    # source access never occurred is safe; the compile-time coalescing is
    # what must not assume it fired.
    if p.kind === :update && !anypending(g.updates)
        return nothing
    end

    for b in pp.images
        emit_barrier!(e, b)
    end
    emit_barrier!(e, pp.barrier)

    # The predicate scope opens AFTER the barriers and closes before the next
    # pass's, so a discarded iteration still orders the ones around it. That is
    # not a nicety: `vkCmdPipelineBarrier` is not in the list of commands
    # conditional rendering affects, so putting the barriers inside would not
    # skip them either — it would only make the scope wider and the intent less
    # clear. Outside, the code says what the hardware does.
    p.predicate === nothing && return emitwork!(e, g, pp)
    withpredicate(e, p.predicate) do
        emitwork!(e, g, pp)
    end
end

"""Run `f` with this pass's work discarded unless its predicate is nonzero.

The scope's `begin` and `end` have to be in ONE command buffer. That used to need
`unsplittable!` around this whole block, because the work inside went through the
queue's recorder and a dispatch could take the batch past `cb_split_threshold` or
`auto_submit_threshold` — Hikari's fused sample ran at `max_depth` 8 (47
dispatches) and hung the GPU at 16 (~95), with the auto-submit threshold at 64
sitting exactly between, and the failure was a foreign call that never returned.

An emitter has one command buffer and nothing that ends it, so the scope cannot
be broken any more. That is the guard, and it is structural rather than a counter
saying "not now".
"""
function withpredicate(f, e::Emitter, (pred, i)::Tuple{Any,Int})
    arr = storage(pred)
    mb = arr.buf[]::VkManagedBuffer
    VK.cmd_begin_conditional_rendering_ext(e.cmd,
        VK.ConditionalRenderingBeginInfoEXT(mb.buffer,
            UInt64(arr.offset + i * sizeof(Predicate))))
    try
        f()
    finally
        # `finally`, because leaving a scope open makes every later command in
        # the buffer conditional on this iteration's flag — a failure that shows
        # up as unrelated passes silently not running.
        VK.cmd_end_conditional_rendering_ext(e.cmd)
    end
end

"""The pass's own commands, once its barriers and predicate are dealt with.

The `:custom` pass kind is gone from here. It ran an opaque body that recorded
through `Mantle.batchqueue(dev)` — the queue, reached from inside a pass — so
this function had to hand one over, and with it every heuristic reachable from a
queue: `exclusive_dispatch_group` to put the automatic barrier back for launches
the graph could not see, `next_skip_barrier` to take it away again for the first
of them, and a one-shot disarm afterwards in case the body recorded nothing. All
three lines were about a pass whose contents were not declared.

Its two consequences went with it: `rebindable(plan)`, which was false for a plan
containing one because a `custom!` body packs its own arguments while it runs and
a recorded plan never runs it again; and `bake(::Compile, body)` on the
KernelAbstractions path, whose whole content was "a body already IS a callable".
Hikari moved its hardware-RT pass off `custom!` to `trace!` for exactly the
rebinding reason; the graph verbs — `compute!`, `render!`, `copy!`, `trace!`,
`repeat!` — are what is left, and each of them says what it touches."""
function emitwork!(e::Emitter, g::Graph, pp::PassPlan)
    p, cds = pp.pass, pp.draws
    if p.kind === :compute
        emitprepares!(e, pp)
        for d in pp.dispatches
            emitdispatch!(e, d, p.name)
        end
        return nothing
    elseif p.kind === :update
        # Before the writes, not after: a rename takes a store from the free
        # list, so anything the GPU finished with has to be back on it first.
        # Recycling afterwards leaves last frame's store invisible to this
        # frame's take, and the resource ping-pongs across three buffers
        # instead of two.
        recycle!(g.recycler, lavadevice(e.ctx))
        applyupdates!(g.updates) do r, data
            write_update!(g, e, r, data)
        end
        return nothing
    elseif p.kind === :copy
        src, dst = first_target(p), p.dst
        copy_image_to_buffer!(e, storage(dst), target_image(src),
                                   size(src)..., target_format(src);
                                   aspect = aspect(src))
        return nothing
    end

    # One begin/end per pass. The clear is pass configuration: a fresh draw!
    # per item transitions the target from UNDEFINED every time and discards
    # everything drawn before it.
    # `transition = false` on both attachments: the layout each is in was
    # derived from the declared usages and emitted above, and Lava's own
    # transition would either double up with it or contradict it.
    begin_pass!(e,
                        VK.ImageView[target_view(t) for t in p.targets],
                        VK.Image[target_image(t) for t in p.targets],
                        VK.Extent2D(target_extent(first_target(p))...);
                        clear_color = map(clearvalue, p.loads),
                        load_op = map(loadop, p.loads),
                        depth_view = p.depth === nothing ? nothing : target_view(p.depth),
                        depth_clear = p.depth === nothing ? nothing : depthclear(p.depth_load),
                        depth_load_op = p.depth === nothing ? nothing : loadop(p.depth_load),
                        transition = false)
    # Viewport and scissor are dynamic pipeline state. vk_draw! sets them for
    # you; draw_in_pass! only does so when passed, and omitting them
    # rasterizes nothing without raising anything.
    ext = target_extent(first_target(p))
    set_viewport!(e,
        VK.Viewport(0f0, 0f0, Float32(ext[1]), Float32(ext[2]), 0f0, 1f0),
        VK.Rect2D(VK.Offset2D(0, 0), VK.Extent2D(ext...)))
    for d in cds
        emitdraw!(e, d)
    end
    end_pass!(e)
    nothing
end

"""
One fused prepare for every dispatch in this pass that sizes itself on the
device, and one barrier behind all of them.

A device-sized dispatch costs a prepare kernel and a barrier the command
processor's read of the count depends on, and that barrier cannot be derived
away — the prepare is the backend's own machinery rather than a pass, so nothing
declared it. Left per dispatch it serialises a pass's dispatches whatever the
schedule said they could do: twelve per-material shading kernels become twelve
barriers.

So: one prepare writing every count, then ONE barrier, then the dispatches
back-to-back with nothing between them. Net for N pairs, 2 barriers and N+1
commands instead of N barriers and 2N.

This is what `concurrent_indirect_group` did, and the machinery it needed to do
it was a list on the queue (`bq.deferred_indirect`), a launch path that checked
that list and pushed instead of recording, and two process-wide atomics saying
whether a concurrent group was open and whether it had started. The pass knows
which of its dispatches are device-sized — it is a field, `pp.indirect` — so the
deferral was carrying information from one part of a record to another part of
the same record.

The prepares are still ordered after this pass's derived barrier, which is what
puts them after whoever wrote the counts; within a pass the dispatches are
independent by construction, which is the same assumption that already lets them
run without barriers between them.
"""
function emitprepares!(e::Emitter, pp::PassPlan)
    pp.indirect || return nothing
    am = e.args
    inds = Any[]
    counts = Any[]
    wss = UInt32[]
    for d in pp.dispatches
        d isa CompiledDispatch || continue          # a trace prepares its own
        d.ndrange isa DeviceRange || continue
        push!(inds, indirectof(am, d.indirect))
        push!(counts, storage(d.ndrange.count))
        push!(wss, UInt32(prod(d.iter.ws_3d)))
    end
    n = length(inds)
    n == 0 && return nothing
    # Statically unrolled over the tuples, compiled per arity — the same kernel
    # `concurrent_indirect_group` flushed with.
    emitkernel!(e, multi_prepare_indirect_kernel,
                ntuple(i -> inds[i], n), ntuple(i -> counts[i], n),
                ntuple(i -> wss[i], n);
                ndrange = 1, workgroup_size = (1, 1, 1))
    emit_barrier!(e, preparebarrier())
    return nothing
end

"""
The one barrier a fused prepare is followed by: its shader writes, made visible
to the command processor's read of the workgroup counts and to the dispatches
behind it.

The same value for every pass of every plan — it names no handle, only masks —
so it is built once and kept. Memoised rather than `const` because constructing
it calls into Vulkan.jl's wrapper types, and this module precompiles on machines
with no driver.
"""
function preparebarrier()
    b = PREPARE_BARRIER[]
    b === nothing || return b
    b = VK._DependencyInfo(
        [VK._MemoryBarrier2(;
            src_stage_mask = VK.PipelineStageFlag2(VK.PIPELINE_STAGE_2_COMPUTE_SHADER_BIT),
            src_access_mask = VK.AccessFlag2(VK.ACCESS_2_SHADER_WRITE_BIT),
            dst_stage_mask = VK.PipelineStageFlag2(VK.PIPELINE_STAGE_2_COMPUTE_SHADER_BIT) |
                             VK.PipelineStageFlag2(VK.PIPELINE_STAGE_2_DRAW_INDIRECT_BIT),
            dst_access_mask = VK.AccessFlag2(VK.ACCESS_2_SHADER_READ_BIT) |
                              VK.AccessFlag2(VK.ACCESS_2_SHADER_WRITE_BIT) |
                              VK.AccessFlag2(VK.ACCESS_2_INDIRECT_COMMAND_READ_BIT))],
        VK._BufferMemoryBarrier2[], VK._ImageMemoryBarrier2[])
    PREPARE_BARRIER[] = b
    return b
end

const PREPARE_BARRIER = Ref{Any}(nothing)

# ── plan ─────────────────────────────────────────────────────────────────────
# ↑ moved to src/graph/build.jl
# ↑ moved to src/graph/build.jl
# ↑ moved to src/graph/build.jl
# ↑ moved to src/graph/build.jl

"""
One dispatch: the same two steps as a draw, into the same memory.

Nothing here can compile, allocate, or take a buffer from Lava's argument pool.
An ndrange that has not moved reuses the iteration plan the pipeline was built
with; one that has costs a lookup keyed on the shape, which is not keyed on the
world and so still cannot reach the compiler.
"""
function emitdispatch!(e::Emitter, d::CompiledDispatch{K,A,I},
                       name::AbstractString) where {K,A,I}
    nd = dispatchrange(d.ndrange)
    it = nd == d.nd0 ? d.iter :
         get_or_build_iter_plan(d.obj, nd, nothing, e.ctx)::I
    it.nblocks == 0 && return nothing
    am = e.args
    tlas = packdispatch!(e, d, it)
    emitlaunch!(e, d.ndrange, lp_of(d), am.address + e.base + d.argoff, it, tlas,
                indirectof(am, d.indirect), name)
    nothing
end

# ↑ moved to src/graph/build.jl

# ↑ moved to src/graph/build.jl

# ↑ moved to src/graph/build.jl

"""
    rebind!(plan) -> plan

Re-read the arguments a recorded plan's work was given and write the current
values into the recording's argument memory.

A recorded plan does not record, so without this the values packed at `record!`
are the ones it runs — for ever, and silently. A frozen sample index renders the
same sample every time and converges to a picture that looks plausible and is
wrong, which is the failure this exists to prevent. Anything given as a `Ref`
moves; this is what makes it move again.

It writes bytes and emits nothing: the command buffer holds the address of the
slot, and the slot is host-mapped, so this is a store per `Ref` and no command
buffer is touched.

It writes ARGUMENTS, not entries: one store per `Ref` the plan binds, at the byte
offset `record!` recorded for it. A plan with no `Ref` anywhere writes nothing,
so `run!` on it does no host work at all between the call and the queue.

That is the whole point and it was not always so. This used to re-run `repack!`
per entry — the record-time path, which re-adapts the argument tree because at
record time it must. It cannot have changed: `run!` throws if a transient moved,
so every device address and every adapted wrapper is provably what `record!`
wrote. Rebuilding them cost 1193 microseconds a run on Hikari's 72-pass chunk
plan, against 0.19 for the entire rest of the path, and it was rebuilding 266 KB
of argument tree to move one `Int32`.

The offsets come from [`argwrites`](@ref) and are checked by
[`verifywrites`](@ref) against the pack that just ran, which is what makes this
safe to do at all: two earlier attempts computed the offsets independently at run
time and both produced a wrong picture rather than an error.

`run!` calls this, and the slot it writes is the one [`nextslot!`](@ref) has just
claimed — so the device is known to be past the run that last read it. That used
to be the caller's problem and was documented as such, and the ring is what makes
it nobody's.

`rebindable(plan)` was checked here and is gone with the `:custom` pass kind it
was about: such a pass packed its own arguments while its body ran, so a plan
holding one could not be rebound at all. Every pass kind left declares what it
touches, so every argument a plan binds is reachable from the plan.
"""
function rebind!(pl::Plan)
    pl.recordings === nothing && return pl
    # Nothing to write: every argument was resolved at `record!` and is still
    # there. The emitter below would otherwise open a batch to hold pins nobody
    # takes, and `submit!` would send its empty command buffer alongside the
    # recording — a second command buffer per run for a plan whose whole point is
    # that it prepares nothing.
    isempty(pl.writes) && return pl
    # `pack_arg!` pins the buffers whose addresses it writes into the batch being
    # recorded, and this is the run's batch — `submit!` appends the recording to
    # it, so the pins and the work they belong to leave together.
    e = emitter(pl.graph.dev.bq, pl.args, slotbase(pl.args))
    for w in pl.writes
        writearg!(e, w)
    end
    return pl
end

# ↑ moved to src/graph/build.jl

# ↑ moved to src/graph/build.jl

# ↑ moved to src/graph/build.jl

# ↑ moved to src/graph/build.jl

Base.close(::Plan) = nothing
Base.isopen(s::WindowSurface) = isopen(s.win)


# The no-argument `Device()`. In the backend and not in core: choosing a default
# backend is not something the portable half can do, and on a machine with two
# loaded it would have to guess.
Device() = Device(VulkanAPI())


# `adaptor` for this backend. It names `LavaAdaptor`, which is exactly the line
# the move to core was drawn along.
#
# `kernelfor` was here too, as `kernelfor(k, ::Nothing, ::LavaBackend) =
# k(LavaBackend())` and its `group` twin. Core's `kernelfor(k, ::Nothing,
# backend) = k(backend)` already answers that — `backend(::LavaDevice)` IS
# `LavaBackend()` — so the pair was a second implementation of one rule, and the
# more specific one was the worse of the two: it discarded the backend it was
# handed and built a default, which drops the queue a `LavaBackend(bq)` pins.
adaptor(e::Emitter) = LavaAdaptor(e.owner)
# At COMPILE, where there is nothing to emit into yet and the adaptor is used for
# its pure half — `adapt_storage` is a strip, and the pinning is a separate walk.
adaptor(bq::VulkanBatchQueue) = LavaAdaptor(ensure_active_batch!(bq))


# The portable constructors. A caller writes `Framebuffer(backend, w, h)` and
# `Window(backend, w, h)` and never names a Vulkan type; these are where that
# resolves on this backend.
Framebuffer(::LavaBackend, w::Integer, h::Integer; kw...) = VulkanFramebuffer(w, h; kw...)
Window(::LavaBackend, w::Integer, h::Integer; kw...) = VulkanWindow(w, h; kw...)
Texture2D(::LavaBackend, data::AbstractArray; kw...) = VulkanTexture2D(data; kw...)
Sampler(::LavaBackend; kw...) = VulkanSampler(; kw...)

# `devicearray` on this backend is a `LavaArray`: pool-managed and capacity-aware
# on `resize!`, which the generic fallback in `memory/array.jl` cannot be. A
# caller that just wants "this data, on that device" gets the better one for
# free by asking Mantle instead of naming the array type.
devicearray(::LavaBackend, data::AbstractArray) = LavaArray(data)


# The two plan pieces that ARE this backend's: a timestamp query pool, and the
# argument memory a recorded launch reads from. Core's `Plan` asks for both and
# accepts `nothing`, which is what a KernelAbstractions backend answers.
makeprofiler(dev::LavaDevice, passes, profile::Bool) =
    profile ? Profiler(dev.ctx, passes) : nothing
makeargmemory(dev::LavaDevice, passes) = ArgMemory(dev, passes)

# `submit!` takes a DEVICE — that is how core calls it, in `graph/kalaunch.jl`
# and `graphics/commands.jl`, and how Metal answers it. This backend had only
# `submit!(::VulkanBatchQueue)`, so core's `submit!(::Any) = nothing` was what
# ran: the contract that a command buffer must not stay open across a dispatch
# was simply not enforced here, silently, because doing nothing is a legal
# implementation of the hook.
submit!(dev::LavaDevice) = submit!(dev.bq)

# What a `custom!` body records into — see `batchqueue` in `graph/queue.jl`.
batchqueue(d::LavaDevice) = d.bq


# ── Recording and submission ──────────────────────────────────────────────────
#
# Twenty definitions that went to `graph/build.jl` during the graph move and
# should not have. They name this backend's command queue (`dev.bq`), its
# context, GLFW, or `capture` — a classifier looking for `VK.`/`Vk` prefixes
# does not see `bq`, which is how they slipped through.
#
# Several are the backend half of hooks core declares in `graph/backend.jl`:
# `rename!`, `inplace!`, `refit!`, `record_pass!`. Those were always meant to be
# here; the portable halves of the same names stayed in core.

"""Take a host-visible buffer of `n` bytes, recycled if one is available.

Keyed negatively so host and device buffers of the same size never share a free
list — handing a device-local buffer to a memcpy would be a segfault, not a
wrong picture."""
function takehost!(r::Recycler, e::Emitter, n::Integer)
    pool = get(r.free, -Int(n), nothing)
    if pool !== nothing && !isempty(pool)
        return pop!(pool)
    end
    host_buffer(queueof(e), n)
end


# ↑ moved to src/graph/build.jl — one predicate, `passed`, and it is core's.

# `Graph` is Mantle's now — see `src/graph/types.jl`.


"""
Write one pending update, at the position the graph reserved for it.

Two routes, and the call decides which — not a flag:

  * a **partial** write goes in place with `cmd_update_buffer`, whose bytes ride
    inside the command buffer, so nothing is staged and nothing has to outlive
    the record. 64 KB and 4-byte alignment are the spec's limits.
  * a **whole-buffer replacement** renames instead: the contents land in a fresh
    store and the resource is pointed at it. Nothing reads the new store yet, so
    there is no hazard to schedule around, and the outgoing store goes back to
    the recycler once the GPU is past this frame.

Renaming a buffer to change one element would device-copy everything that did
not change, and writing a whole 2 MB array in place would need the hazard
handled. Each route is bad at the other's job, which is why both exist.
"""
write_update!(g::Graph, e::Emitter, r::UpdateRef, data::Buffer) =
    write_update!(g, e, r, storage(data))

# A scalar attribute is a one-element buffer, so a new value is a one-element
# write: in place, inline in the command buffer, four to sixteen bytes riding
# along with the frame. Renaming would be absurd for that, and the old
# `update!(scalar, x)` route flushes the queue to make its write safe, which is a
# stall per changed colour.


write_update!(g::Graph, e::Emitter, r::UpdateRef, x) =
    (inplace!(e, r.resource, [x], 1); nothing)


function write_update!(g::Graph, e::Emitter, r::UpdateRef, data::AbstractVector{T}) where {T}
    dst = r.resource
    n = length(data) * sizeof(T)
    n == 0 && return
    if r.range === nothing && length(data) == length(dst)
        rename!(g, e, dst, data)
    else
        inplace!(e, dst, data, r.range === nothing ? 1 : first(r.range))
    end
    nothing
end


rename!(g::Graph, e::Emitter, dst::Buffer{T,1}, data::Buffer{T,1}) where {T} =
    rename!(g, e, dst, storage(data))


inplace!(e::Emitter, dst, data::Buffer, from::Integer) =
    inplace!(e, dst, storage(data), from)


"""
Replace the contents by pointing the resource at a fresh store.

The copy is *recorded*, not executed. `upload!` would flush the queue to
make its write safe, but nothing reads the fresh store yet, so there is no hazard
to wait for — the frame's own submit carries the copy, and the barrier into the
first pass that reads it is derived from `CopyDst` like any other usage.

The staging buffer is recycled by size for the same reason the stores are: it is
the same size every frame, and a recorded copy reads it later, so it cannot be
handed out again until the GPU is past this frame.
"""
function rename!(g::Graph, e::Emitter, dst::Buffer{T,1}, data::AbstractVector{T}) where {T}
    old = dst.store
    nbytes = length(data) * sizeof(T)
    fresh = take!(g.recycler, dst.dev, T, dst.capacity)
    host = takehost!(g.recycler, e, nbytes)

    src = data isa Vector{T} ? data : collect(data)
    GC.@preserve src Base.unsafe_copyto!(host.mapped_ptr, Ptr{UInt8}(pointer(src)), nbytes)

    fview = deviceview(dst.dev, fresh)
    fmb = fview.buf[]
    cmd_copy_buffer!(e, host.buffer, fmb, nbytes;
                          dst_off = pool_offset(fmb) + fview.offset)

    dst.store = fresh
    signal = signalof(e)
    retire!(g.recycler, old, dst.capacity * sizeof(T), signal)
    retire!(g.recycler, host, -nbytes, signal)
    nothing
end


"""
Follow a resize: give every tracking transient its source's size and place them
again.

A recompile rather than a patch, because a different size is different offsets,
different aliasing and therefore different barriers — the placer answers all
three and there is nothing to salvage from the old answer. It costs what a
compile costs, which is a fraction of a millisecond, and it happens when a human
drags a window edge.

Safe to drop the old slabs here because `sync_swapchain!` waits for the device
before it rebuilds, so nothing is still reading them.
"""
# `refit!(::Plan)` is Mantle's now — see `src/graph/build.jl`. Every line of it
# was the graph's: refit the transients, recompile, adopt the new placement,
# re-register as a tenant. The one driver-shaped line, rebuilding the argument
# memory, already had a hook.


"""
    timings(plan)

Per pass, median host recording time and median GPU time over the samples kept.
"""
function timings(pl::Plan{LavaDevice})
    prof = pl.profiler
    prof === nothing &&
        throw(ArgumentError("this plan was not built to be profiled; use Plan(g; profile = true)"))
    collect!(prof, pl.graph.dev.ctx)
    med(x) = isempty(x) ? NaN : (q = sort(x); q[(length(q) + 1) ÷ 2])
    [PassTiming(prof.names[i], pl.passes[i].pass.kind,
                       med(prof.host_ns[i]) / 1e6, med(prof.gpu_ns[i]) / 1e6,
                       length(prof.gpu_ns[i])) for i in eachindex(prof.names)]
end


"""
    record!(plan) -> plan

Emit the plan's work into command buffers it owns, one per argument slot, and
seal them. `run!` then writes per-run values and submits; it never records.

Idempotent — a second call would strand the first set of recordings.

**One recording per slot**, and that is not an optimisation. `e.base` is folded
into every address a recording holds, so a recording cannot be pointed at a
different slot afterwards; recording once and pinning the slot for the plan's
life was the first shape of this, and it took the argument ring away from exactly
the plans that need it most, so `rebind!` wrote the bytes a submission still in
flight was reading. Measured as an accumulator reading 36 where an interpreted
run read 21.

Indexed BY THE SLOT, not by iteration order. `run!` looks a recording up as
`pl.recordings[pl.args.slot]`, so the two have to agree on what the index means —
and they do not agree for free: this was written as `map(1:ARG_SLOTS)`, which is
only right when the ring starts at 0. A plan that has already RUN is somewhere
else in the ring, so recording 1 named slot 2 and every submission afterwards
read a neighbouring slot's arguments. It shows up as an image that is wrong for
some sample counts and right for others — right exactly when the number of runs
brings the two indices back into phase.

This was `bake!`, and the name went with the idea it named: a plan that had been
"baked" was a plan whose commands had been frozen out of an ordinary interpreted
run, which is why `bake!` used to EXECUTE the plan as a side effect of recording
it, and why `bake!`/`run!` had to agree about barriers, slots and what a `Ref`
meant. Recording is what running a plan is.
"""
function record!(pl::Plan{LavaDevice})
    pl.recordings === nothing || return pl
    g = pl.graph
    isempty(g.surfaces) || throw(ArgumentError(
        "record!: this plan draws to a surface. A swapchain image is a different " *
        "image every frame and a recording names one, so a windowed plan needs a " *
        "recording per swapchain image — which is not built yet. Headless plans " *
        "record today."))
    renaming(g) === nothing || throw(ArgumentError(
        "record!: this plan has a whole-buffer `Update`, which lands by RENAMING — " *
        "the contents go into a fresh store and the resource is pointed at it, " *
        "because writing them in place would overwrite bytes the previous run may " *
        "still be reading. A recording holds the address it was written with, so " *
        "it would keep reading the store the update moved away from. Narrow the " *
        "update with `range =`, which always writes in place, or leave the plan " *
        "unrecorded."))
    bq = g.dev.bq
    refit!(pl)
    checkextents(pl)
    recordings = Vector{Any}(nothing, ARG_SLOTS)
    for _ in 1:ARG_SLOTS
        slot = nextslot!(pl.args, g.dev)
        rec = recording!(bq)
        emit!(Emitter(rec, pl.args, slotbase(pl.args)), pl)
        seal!(rec)
        recordings[slot] = rec
    end
    pl.recordings = recordings
    # Which arguments `run!` has to write again, and — by omission — which are
    # written here and never again. See `Plan.writes`.
    #
    # AFTER the recordings, not before: `argwrites` packs as it walks, so it
    # needs a slot to pack into, and `am.slot` is 0 until `nextslot!` has run.
    # Building it first made `slotbase` negative and the first `memset` a
    # segfault inside the allocator.
    #
    # The offsets it records are relative to a slot, so the list serves every
    # recording in the ring whichever one is current here.
    writes = ArgWrite[]
    let e = emitter(bq, pl.args, slotbase(pl.args))
        for pp in pl.passes
            for d in pp.dispatches
                append!(writes, argwrites(e, d))
            end
            for dr in pp.draws
                append!(writes, argwrites(e, dr))
            end
        end
    end
    pl.writes = writes
    # And each write stays inside its own bytes — checked here, naming the plan,
    # rather than surfacing as a rendered image that is subtly wrong three layers
    # away. That is how two earlier attempts at this failed.
    verifywrites(pl)
    pl
end

"""
Check that every [`ArgWrite`](@ref) touches only the bytes it claims.

This is the property worth checking, and the obvious one is FALSE. "Performing
the writes changes nothing, since everything was just packed" sounds right and is
not: a `Ref` may hold a value whose adapted form contains a device address, and
that address can legitimately differ between the recording and now. Hikari's
`filter_sampler` is exactly that, and the byte that gave it away was the high
half of a pointer — `0x00007f29…` against `0x00007f2b…`. Writing the current
address is the whole point of the write; a check that forbids it is checking the
wrong thing, and it cost two rounds of chasing a phantom offset bug.

What must hold is that a write stays inside its own footprint: the eight bytes at
its slot offset, and — for a by-value argument — the `byval` bytes at its inline
position. A write whose offsets are wrong reaches outside that and corrupts a
neighbour, which is the failure this exists to catch.
"""
function verifywrites(pl::Plan)
    am = pl.args
    n = am.stride
    e = emitter(pl.graph.dev.bq, am, slotbase(am))
    before = Vector{UInt8}(undef, n)
    after = Vector{UInt8}(undef, n)
    for w in pl.writes
        unsafe_copyto!(pointer(before), am.ptr + e.base, n)
        writearg!(e, w)
        unsafe_copyto!(pointer(after), am.ptr + e.base, n)
        # Slot-relative and 1-based, to match the indices being compared.
        slot = (w.entryoff + w.slotoff + 1):(w.entryoff + w.slotoff + 8)
        inl0 = w.entryoff + ((w.inline + 7) & ~7) + 1
        inl = inl0:(inl0 + w.byval - 1)
        for i in eachindex(before)
            before[i] == after[i] && continue
            (i in slot || i in inl) && continue
            throw(ArgumentError(
                "record!: an argument write reached outside its own bytes — wrote " *
                "byte $i, which is neither its slot " *
                "($(first(slot))..$(last(slot))) nor its inline region " *
                "($(isempty(inl) ? "none" : "$(first(inl))..$(last(inl))")). " *
                "Entry at $(w.entryoff), slot offset $(w.slotoff), byval " *
                "$(w.byval), inline $(w.inline), ref $(typeof(w.ref)). The " *
                "offsets in `plan.writes` disagree with what " *
                "`pack_args_direct!` used, so a run would corrupt a " *
                "neighbouring argument."))
        end
    end
    return pl
end

"""
The first `Update` on this graph that may rename its target, or `nothing`.

A whole-buffer write renames — see `write_update!` — and a ranged one never
does, so this is the property that decides whether the plan can be recorded. It
is asked of the GRAPH rather than of the values, so the answer is fixed for the
plan's life.

Conservative on purpose: a ref with no range that is written a SHORTER vector
goes in place, and this still says it may rename. The alternative is a plan that
records fine until the day a caller writes the whole buffer.
"""
function renaming(g::Graph)
    for u in g.updates
        u.range === nothing && return u
    end
    return nothing
end

"""Whether this plan's commands can be written once, or have to be emitted per
run. Two reasons they cannot, and both go away later in the refactor: a surface
(step 4) and an `Update` that renames (step 3, where a per-run value rides inline
in the command buffer and nothing has to move)."""
recordable(pl::Plan) = isempty(pl.graph.surfaces) && renaming(pl.graph) === nothing

"""
    run!(plan)

Write this run's values and hand the plan's recording to the device.

**It does not record.** The `barriers` keyword is gone with the branch it
selected: `:backend` meant "let Lava's per-dispatch recorder insert the
ordering", `:both` meant "emit the derived barriers as well", and both were A/B
scaffolding for a comparison that has been made. An emitter writes what the plan
derived, and there is no automatic barrier left to compare against.

A plan that cannot be recorded still emits per run, into the batch — see
[`recordable`](@ref) for the two reasons, both of which later steps remove.
"""
function run!(pl::Plan{LavaDevice})
    checklive(pl, pl.slabs, length(pl.graph.transients))
    # Anything dropped without a `free!` goes back here, one submission boundary
    # after it was dropped. Cheap and a no-op when nothing was — see
    # `reclaim!`. Here rather than in a user's frame loop because a
    # renderer that has to remember to call it is one that stops reclaiming the
    # day someone forgets, which is the failure this exists to remove.
    reclaim!(pool(pl.graph.dev), pl.graph.dev)
    g = pl.graph
    bq = g.dev.bq
    # Before this run overwrites them, and without waiting: see `collect!`.
    pl.profiler === nothing || collect!(pl.profiler, g.dev.ctx)
    # Once per frame, here rather than in `isopen`: a predicate that also pumps
    # the event queue is a surprise, and a frame loop that has to remember to
    # poll is a frame loop that stops responding the day someone forgets.
    isempty(g.surfaces) || GLFW.PollEvents()
    # Bring the swapchains up to date and check the plan still fits, both before
    # anything is acquired or recorded. Failing here leaves nothing behind; the
    # same check inside recording left a half-recorded batch and an acquired
    # image, which the next submit ran against destroyed swapchain images — a
    # GPUVM fault two testsets later, blamed on everything except the throw.
    for s in g.surfaces
        # `sync_swapchain!` refuses a closed window — a destroyed GLFW handle is a
        # segfault to ask anything of. Not checked with `isopen` here: a window
        # whose close button has been clicked is still fine to draw to, and the
        # loop condition is what decides to stop.
        sync_swapchain!(s.win)
    end
    # Ask the transients whether they moved, rather than asking the swapchain
    # whether it resized: `acquire_next_image!` syncs the swapchain too, so
    # whichever of the two got there first, the other reported "no change" and
    # the refit was skipped. A frame where nothing moved costs one size compare
    # per tracking transient.
    moved = refit!(pl)
    checkextents(pl)            # and anything with a fixed size has to still fit
    recordable(pl) && return submitrun!(pl, bq, moved)
    for s in g.surfaces
        acquire_next_image!(s.win)
    end
    # One slot of the plan's argument memory per frame, reused only once the GPU
    # has passed the frame that last used it.
    nextslot!(pl.args, g.dev)
    emit!(emitter(bq, pl.args, slotbase(pl.args)), pl; updates = true)
    # What this frame signals covers everything written into the slot.
    pl.args.slot_token[pl.args.slot] = ensure_active_batch!(bq).signal_value
    for s in g.surfaces
        present_frame!(bq, s.win)
    end
    nothing
end

"""One run of a recorded plan: its updates, this run's arguments, and a submit."""
function submitrun!(pl::Plan, bq, moved::Bool)
    pl.recordings === nothing && record!(pl)
    # A refit re-places every transient, so the recordings name storage that no
    # longer exists. Nothing tracking can move in a headless plan today, which is
    # why this is an assertion rather than a re-record.
    moved && throw(ArgumentError(
        "run!: a transient moved under a recorded plan, so its command buffer " *
        "names storage that has been replaced. Re-`Plan` and `record!` again."))
    # Updates first and fresh — `submit!` appends the recording behind whatever
    # is in the open batch, so the copies land ahead of it in the one submission,
    # which is the order the barriers inside the recording were built for.
    emitupdates!(pl, bq)
    # One slot per run, and one recording per slot. This is where a run waits,
    # and only when the host is `ARG_SLOTS` runs ahead of the device — which is
    # also what makes the `rebind!` below safe without a drain.
    nextslot!(pl.args, pl.graph.dev)
    # The arguments, every run, without the caller asking.
    #
    # `rebind!` was the caller's job, which made a recorded plan mean something
    # different from an interpreted one: interpreted re-recorded, so a `Ref`
    # argument was re-read by `argvalue` and repacked every frame; recorded
    # replayed the bytes from `bake!` unless somebody remembered. A renderer that
    # changes `sample_idx` per frame silently rendered sample 0 forever.
    #
    # That is the mistake `reclaim!` above is explicitly written to avoid — "a
    # renderer that has to remember to call it is one that stops reclaiming the
    # day someone forgets". Recording is a decision about WHEN commands are
    # built, not about what the arguments mean, and the graph holds every
    # argument, so it repacks them.
    rebind!(pl)
    # A run writes this arena's bytes like any other, so it has to claim them —
    # even though it cannot emit a barrier of its own, since the commands are
    # already written. Skipping the claim leaves the arena naming whoever
    # RECORDED last, and the next tenant then sees itself there and emits
    # nothing: a handover with no barrier at all. (This plan taking over from
    # someone else is still unbarriered — that is what recording into a shared
    # arena costs, and `remappable` already refuses the growth case.)
    let p = pool(pl.graph.dev)
        for ar in pl.arenas
            takeover!(p, ar, pl)
        end
    end
    # And what covers the slot this run wrote, so the next pass round the ring
    # knows what to wait for.
    pl.args.slot_token[pl.args.slot] = submit!(bq, pl.recordings[pl.args.slot]::Recording)
    return nothing
end

"""The update passes alone, emitted fresh in front of a recording."""
function emitupdates!(pl::Plan, bq)
    anypending(pl.graph.updates) || return nothing
    e = emitter(bq, pl.args, slotbase(pl.args))
    for pp in pl.passes
        pp.pass.kind === :update || continue
        emitpass!(e, pl.graph, pp)
    end
    nothing
end

"""
One draw: write its arguments into the plan's slot and emit it.

Its own function because a pass's draws are differently parameterised, so the
loop dispatches once per draw and everything inside is concrete — including
`devargs(ad, rawargs(d.args))`, which allocated a boxed tuple per draw per frame while it
was inlined into the loop.

Nothing is allocated and nothing is pinned: the memory belongs to the plan, and
every resource the arguments name is reachable from the plan for as long as it
lives.
"""
function emitdraw!(e::Emitter, d::CompiledDraw)
    # `repack!` is the packing, and it is the same packing `rebind!` does — which
    # is the point of it being a function: these six lines used to be written out
    # twice, here and inside `rebind!`'s loop, and the copies drifted.
    repack!(e, d)
    # No viewport, no scissor, no pin: the pass set the first two once, and the
    # plan holds the pipeline for longer than any frame.
    emit_draw!(e, d.compiled, d.count, e.args.address + e.base + d.argoff)
end

# Where the counts come from, decided once by type rather than per frame by a
# branch. The indirect one emits the same command whatever the numbers are,
# which is why nothing has to be read back to record a frame.

emit_draw!(e, pipe, n::Integer, addr::UInt64) =
    draw_in_pass!(e, pipe, n; push_bda = addr, pin = false)

emit_draw!(e, pipe, x, addr::UInt64) =
    draw_in_pass!(e, pipe, count(x); push_bda = addr, pin = false)

emit_draw!(e, pipe, c::Commands, addr::UInt64) =
    draw_indirect_in_pass!(e, pipe, storage(c.resource);
                           push_bda = addr, pin = false)

"""
Write one dispatch's arguments into the plan's slot, and answer with the
acceleration structure they name.

Separate from emitting because a recorded plan has to do exactly this and nothing
else: its command buffer holds the ADDRESS of the slot, so a value that moved is
a write to host-mapped memory rather than a new recording. See [`rebind!`](@ref).
"""
function packdispatch!(e::Emitter, d::CompiledDispatch, it)
    am = e.args
    off = e.base + d.argoff
    lp = d.launch
    raw = rawargs(d.args)
    args = devargs(adaptor(e), raw)
    pack_args_direct!(e.owner, am.ptr + off, am.address + off, lp.offsets,
                           lp.arg_buffer_size, lp.byval_sizes,
                           (d.kernel, it.ka_ctx, args...))
    return d.tlas ? find_tlas_in_args(raw) : nothing
end

"""
Emit the launch itself. A host-side ndrange dispatches directly; a
[`DeviceRange`](@ref) reads workgroup counts the pass's fused prepare already
wrote, and dispatches indirectly off them.

The conversion is a kernel, so it cannot live in Mantle core — but the CONTRACT
does: core hands over an element count and knows nothing about workgroups, and
the graph orders whatever wrote the count before this dispatch because
`indirectcount!` registered it as an `Indirect` read.

The prepare is NOT here any more. It was, and it consulted `bq.deferred_indirect`
to decide whether to write it now or hand it to a group flush that would fuse it
with the pass's others — a per-dispatch decision made from queue state about
something the pass already knows. `emitprepares!` writes them all, once, before
the first dispatch of the pass.
"""
emitlaunch!(e::Emitter, ::Any, lp, argaddr, it, tlas, ::Nothing, name) =
    emit_dispatch!(e, lp.pipeline, argaddr, it.block_dims, tlas, name)

emitlaunch!(e::Emitter, ::DeviceRange, lp, argaddr, it, tlas, indirect, name) =
    emit_dispatch_indirect!(e, lp.pipeline, argaddr, indirect, tlas, name)
