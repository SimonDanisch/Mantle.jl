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
next plan is about to write, and that is what the global barrier at the head
of every recording covers ([`headbarrier!`](@ref)).

`extra` and `typebits` accumulate across tenants because the allocation has to
satisfy all of them at once, and they accumulate in opposite directions: usage
bits are a **union** (the arena must permit everything any tenant does with it)
and image memory type bits are an **intersection** (the type has to be one every
image can bind to). A tenant that narrows the intersection to nothing is an
error, not a silent pick of a type some image cannot use.
"""

# `ctx`/`bq` typed, not Any: every read of them is on a hot path
# (`run!` → `waitfor!` → `driver(d.bq).next_timeline`), and an Any field makes each a
# dynamic getproperty that boxes the UInt64 it lands on.
struct LavaDevice <: Device
    ctx::VkContext
    bq::VulkanBatchQueue{VkContext}
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
function Device(::VulkanAPI; select = nothing, debug::Union{Nothing,DebugConfig} = nothing)
    # No selector and no debug configuration: the process default, cached per
    # context. Anything else is a device of the caller's own, built now and
    # never installed; `defaultdevice!` is how one becomes the default.
    select === nothing && debug === nothing && return lavadevice(vk_context())
    return lavadevice(VkContext(; select, debug = something(debug, DebugConfig())))
end

"""Make `d` the process default: what `Device(VulkanAPI())`, `LavaBackend()`
and `defaultbackend()` answer from now on. The previous default stays alive;
its arrays and backends are still its own."""
defaultdevice!(d::LavaDevice) = (bind_context!(d.ctx); d)

"""
    Device(backend::LavaBackend)

The Mantle device for a KA backend: the one on the backend's OWN context.

The missing link for a KA workload that wants the pool: `DNNKernels` holds a
`LavaBackend`, not a module, and allocating through `KA.allocate` puts its slab
somewhere Mantle cannot see. This maps the backend it does have onto the cached
device of its context, same pool, so a model's scratch and an editor's
transients land in one allocator. It used to ignore the backend and answer the
process default, which handed Hikari on a second device the first device's pool.
"""
Device(b::LavaBackend) = lavadevice(vk_context(b))

# The queue and build verbs a caller holding a device or a backend spells
# portably. The context methods are the implementation; these name the device
# the caller HAS instead of the process default.
allocate_batch_queue!(d::LavaDevice) = allocate_batch_queue!(d.ctx)
allocate_batch_queue!(b::LavaBackend) = allocate_batch_queue!(vk_context(b))
build_accel!(f, d::LavaDevice) = build_accel!(f, d.bq)
build_accel!(f, b::LavaBackend) = build_accel!(f, b.dispatch_bq)

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

"""The KernelAbstractions backend, for kernels the graph does not own. Pinned
to this device's queue: a graph on a second device hands out a backend that
dispatches there, not on whichever device is the default."""
backend(d::LavaDevice) = LavaBackend(d.bq, d.bq)

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

# The no-backend spelling opens on the process default device; it is the
# window-shaped member of the same convenience family as `LavaBackend()`.
Window(width::Integer, height::Integer; title::AbstractString = "", vsync::Bool = false) =
    LavaWindow(VulkanWindow(width, height; ctx = vk_context(), title = String(title), vsync))

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
# `LavaBuffer`/`LavaScalar` are gone. `Buffer` and `GPURef` are core
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

# Vulkan refuses an index buffer that was not allocated as one, so this backend
# has to say so at allocation; Metal's `drawIndexedPrimitives` takes any buffer,
# which is why the core default is a plain `Buffer`.
#
# Delegating to `alloc_index_buffer` rather than reimplementing it: that function
# already sets `BUFFER_USAGE_INDEX_BUFFER_BIT` through `LavaArray`'s
# `extra_usage`, and threading the same bit through `Mantle.Buffer` would mean
# giving `persistentarray` a per-allocation usage argument it does not have. What
# this replaces is the REACH: RayMakie called `alloc_index_buffer` through
# `Base.get_extension` at five sites, which is a backend name in a package that
# must not have one.
Mantle.indexbuffer(dev::LavaDevice, indices::AbstractVector{UInt32}) =
    alloc_index_buffer(dev.bq, indices)

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
# Nothing is ever recorded and unsubmitted, so the newest submission is the
# last thing that can be reading a region retired now.
fence(d::LavaDevice) = driver(d.bq).next_timeline
# One predicate for "has the device finished this", spelled once.
#
# It was written out as `query_timeline(bq) >= v` at each of the places that ask,
# which is how `flush!` came to fold over two lists and the argument ring came
# to compare against `next_timeline` — a raw counter, in code that had no
# business knowing there was one. The queue method is the primitive; the device forwards
# to it, and `Pool` and the plan's argument memory both go through the device.
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
waitidle(d::LavaDevice) = (flush!(d.bq); waitidle(d.ctx::VkContext))

passed(bq::VulkanBatchQueue, v) = query_timeline(bq) >= v

passed(d::LavaDevice, f) = passed(d.bq, f)

"""
Wait for the timeline to reach `f`.

`f > next_timeline` used to mean the value belonged to a batch still being
recorded, and this declined rather than block on a submission nobody had made.
Nothing recorded is unsubmitted now, so such a value is a token nothing will
ever signal, and waiting on it would hang in a foreign call: it is an error.
"""
function waitfor(d::LavaDevice, f)
    passed(d, f) && return true
    f > driver(d.bq).next_timeline && throw(LavaError("waitfor",
        "asked to wait for timeline value $f, but the queue has only submitted up to $(driver(d.bq).next_timeline)",
        "A token beyond the timeline covers work nothing submitted. Every closed command buffer is submitted when it is closed; a token comes from `submit!`."))
    wait_timeline!(d.bq, UInt64(f))
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

# ── Moves ────────────────────────────────────────────────────────────────────
#
# The backend half of `resource_moved!` / `arena_moved!`: the addresses are this
# backend's buffer device addresses, so the range math lives here. What a
# recorded plan packed for a resource is `bda_address` of the view the kernel
# was given, and a move shifts every address in the old range by the same delta.

resource_moved!(dev::LavaDevice, old::DeviceArray, fresh::DeviceArray) =
    notify_move!(pool(dev), bda_address(deviceview(dev, old)),
                 bda_address(deviceview(dev, fresh)), length(old.region))

arena_moved!(dev::LavaDevice, kind, old::Region, fresh::Region) =
    notify_move!(pool(dev), region_bda(old), region_bda(fresh), length(old))

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



# ↑ moved to src/graph/build.jl

# ↑ moved to src/graph/build.jl

# `take!(::Recycler, …)` is gone with renaming — it handed out a fresh region for
# a whole-buffer store to land in, recycled by byte size because the sizes repeat exactly.

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

function remakeimage!(dev::LavaDevice, t::VulkanTransientImage)
    ctx = dev.ctx
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
Bytes into a resource's store: inside the command buffer when
`vkCmdUpdateBuffer` can carry them — 64 KB, offset and size multiples of four —
and a staged upload past that, which stalls. Rare: every store the graph makes
is a whole struct or a range of them, and those are four-byte multiples.
"""
function storebytes!(e::Emitter, a::DeviceArray, off::Int, p::Ptr{Cvoid}, n::Int)
    if n <= 65536 && n % 4 == 0 && (offset(a) + off) % 4 == 0
        emitinline!(e, a, off, p, n)
    else
        bytes = DeviceArray{UInt8}(a.region, (length(a.region),))
        upload!(lavadevice(e.ctx), bytes, off + 1,
                unsafe_wrap(Vector{UInt8}, Ptr{UInt8}(p), n))
    end
    return nothing
end

# ↑ moved to src/graph/build.jl

# ↑ moved to src/graph/build.jl

# ↑ moved to src/graph/types.jl — how deep a plan pipelines is not a driver's.

"""
A plan's argument bytes: a `Unified` region — BAR memory, device-local and
host-visible — that the plan retires, with its device address and mapped host
pointer at the region's offset in its block.
"""
function argbytes(dev::LavaDevice, nbytes::Int)
    region = acquire!(pool(dev), dev, Unified(), nothing, nbytes;
                      align = 256, blocksize = UNIFIED_BLOCK_SIZE)
    blk = memoryof(region)::BufferBlock
    base = offset(region)
    mb = blk.ref[]::VkManagedBuffer
    return region, blk.address + UInt64(base), mb.mapped_ptr + base
end

"""The indirect command `off` bytes into a plan's argument region, as the
`LavaArray` a prepare kernel writes and `emit_dispatch_indirect!` reads."""
indirectslot(::LavaDevice, region::Region, off::Int) =
    LavaArray{UInt32,1}(copy((memoryof(region)::BufferBlock).ref), (3,);
                        offset = offset(region) + off)

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
renameable-resource special case that already emitted a global barrier for this
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
                              nothing, Stamp{UInt64}(), BUF_STATE_ALIVE, dev.ctx)
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
                              nothing, Stamp{UInt64}(), BUF_STATE_ALIVE, ctx)
    ref = GPUArrays.DataRef(_ -> nothing, managed)
    return BufferBlock(buf, mem, UInt64(addr), Int(req.size), ref)
end

# Nothing a caller of this arena declares narrows it: the usage mask above is
# fixed, because what a recording reads is the same on every plan.
constraintof(::LavaDevice, ::Unified, ts) = UInt32(0)
mergeconstraints(::LavaDevice, ::Unified, a::Integer, b::Integer) = a | b

"""
Back a [`Readback`](@ref) arena: one buffer in HOST-CACHED memory, mapped for
the whole life of the block.

Not the `Unified` arena, though both are host-visible: `Unified` is BAR memory,
which the host reads at write-combined speed (see `Readback`). `TRANSFER_DST`
is what a download copies into, `TRANSFER_SRC` what the framebuffer readback
and an upload out of the same kind would need; no device address, because no
kernel ever reads it.

`find_memory_type_optional` for the cached type, because a device may have no
host-cached heap at all (lavapipe reports every type cached; some mobile parts
none), and the plain host-visible fallback is still correct, just slow.
"""
function rawalloc(dev::LavaDevice, ::Readback, bytes::Int, usage)
    ctx = dev.ctx::VkContext
    n = max(bytes, 1)
    u = VK.BufferUsageFlag(usage) |
        VK.BUFFER_USAGE_TRANSFER_SRC_BIT |
        VK.BUFFER_USAGE_TRANSFER_DST_BIT
    buf = unbound_buffer(ctx, n, u)
    req = buffer_requirements(ctx, buf)
    idx = find_memory_type_optional(ctx, req.type_bits,
              VK.MEMORY_PROPERTY_HOST_VISIBLE_BIT |
              VK.MEMORY_PROPERTY_HOST_COHERENT_BIT |
              VK.MEMORY_PROPERTY_HOST_CACHED_BIT)
    idx === nothing && (idx = find_memory_type(ctx, req.type_bits,
              VK.MEMORY_PROPERTY_HOST_VISIBLE_BIT |
              VK.MEMORY_PROPERTY_HOST_COHERENT_BIT))
    mem = VK.DeviceMemory(ctx.device, UInt64(req.size), idx)
    bind_buffer!(ctx, buf, mem, 0)
    ptr = Ptr{UInt8}(unwrap(VK.map_memory(ctx.device, mem, 0, UInt64(req.size))))
    managed = VkManagedBuffer(buf, mem, UInt64(0), ptr, Int(req.size),
                              nothing, Stamp{UInt64}(), BUF_STATE_ALIVE, ctx)
    ref = GPUArrays.DataRef(_ -> nothing, managed)
    return BufferBlock(buf, mem, UInt64(0), Int(req.size), ref)
end

constraintof(::LavaDevice, ::Readback, ts) = UInt32(0)
mergeconstraints(::LavaDevice, ::Readback, a::Integer, b::Integer) = a | b

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

# The destructor core calls for a buffer whose destroy was REQUESTED
# (`vk_free!`) once the submission that named it has passed — see `retire!` /
# `reclaim!` in `graph/lifetime.jl`. `destroy_buffer!` is the whole of what a
# backend still decides here: which Vulkan objects to destroy, and giving a
# suballocated span back to the pool.
rawfree(::LavaDevice, buf::VkManagedBuffer) = destroy_buffer!(buf)

# The same for an acceleration structure, whose handle lives in the bytes of its
# storage buffer — see `unsafe_free!(::Union{LavaBLAS,LavaTLAS})`. Here rather
# than beside it because `LavaDevice` is declared in this file, after
# `raytracing/acceleration.jl` is included.
rawfree(::LavaDevice, as::Union{LavaBLAS, LavaTLAS}) = destroy_now!(as)

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

function materialize!(dev::LavaDevice, t::VulkanTransientImage{T}, slab, offset) where {T}
    # A VkImage binds memory exactly once, so re-materialising — the arena moved
    # out from under this image — means a NEW image, never a rebind. The old
    # handle is what a dropped recording named; the RAII finalizer destroys it.
    t.memory === nothing || remakeimage!(dev, t)
    t.memory = slab                       # the image outlives the call; the slab must too
    # `dev.ctx`, never the process default: the image was made on the graph's
    # device, and binding it through another device's handle is undefined.
    bind_image!(dev.ctx, t.image, slab, offset)
    t.view = image_view(dev.ctx, t.image, t.format, aspect(T))
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

# ↑ moved to src/graph/build.jl

# ↑ moved to src/graph/build.jl

# ↑ moved to src/graph/build.jl
syncbackend(::LavaDevice) = VulkanAPI()

# ── Pipelines ─────────────────────────────────────────────────────────────────
"""
A draw, compiled: the pipeline for its shader pair against the pass's
attachments, and the argument block it packs — laid out to whichever stage's
shader reads it. The layout is core's (`argoff`); what is compiled is this
backend's.
"""
function compiledraw(c::Compile{LavaDevice}, p::Pass, d, argoff::Int)
    ad = adaptor(c.graph.dev.bq)
    args = devargs(ad, rawargs(d.args))
    vtt = typeof(convert_args(args))
    ftt = typeof(convert_args(devargs(ad, rawargs(d.frag_args))))
    vfn, vtt, ffn, ftt = resolve_shader_pair(d.shader, vtt, ftt)
    # The pass says whether there is a depth attachment, and with dynamic
    # rendering the pipeline has to declare the same thing: a pipeline built for
    # one and drawn into a pass without it is invalid, and the reverse is too.
    shader, compiled = ensure_compiled_with_shader!(d.shader, vfn, ffn, vtt, ftt;
                                                    ctx = c.graph.dev.ctx,
                                                    color_format = VK.Format[target_format(t) for t in p.targets],
                                                    depth_format = p.depth === nothing ?
                                                        VK.FORMAT_UNDEFINED :
                                                        target_format(p.depth))
    # The block is laid out to whichever stage's shader reads it.
    # `ensure_compiled_with_shader!` hands back the vertex shader because that
    # is the usual answer; a fullscreen pass whose vertex stage takes nothing
    # and whose fragment stage reads a g-buffer is the other one.
    isempty(d.frag_args) || (shader = get_or_compile_gfx(ffn, ftt, :fragment; ctx = c.graph.dev.ctx))
    packed = isempty(d.frag_args) ? d.args : d.frag_args
    info = shader.push_info
    nbytes = info.arg_buffer_size + compute_inline_extra_from_byval(info.byval_llvm_sizes)
    return CompiledDraw(compiled, shader, packed, d.count, argoff, nbytes)
end

"""
The pass's barriers in the form this backend emits. A layout change is
per-image and cannot be folded into the pass's one memory barrier, so the
needed transitions split by resource kind: images become an image barrier
each, everything else is ORed into the memory barrier.
"""
function passbarriers(c::Compile{LavaDevice}, p::Pass)
    g = c.graph
    need = c.prepass[p]
    isimage(t) = t.resource != 0 && resourcekind(g.ids.by_id[t.resource]) isa ImageKind
    imgs = [ImageBarrier(g.ids.by_id[t.resource], t) for t in need if isimage(t)]
    return imgs, need, build_pass_barrier(g, filter(!isimage, need))
end

"""
Compile one dispatch: everything `KernelAbstractions` would work out per launch,
worked out once.

The pieces are the ones `ka_launch!` uses — an iteration plan for the ndrange and
a launch plan for the argument types — taken here so that recording is only
packing and `emit_dispatch!`. The kernel itself is compiled by the launch plan, and
this is the only place a Mantle frame can compile one.
"""
function compile_dispatch(c::Compile{LavaDevice}, d::Dispatch, argoff::Int, indirect::Int)
    dev = c.graph.dev
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
offset in the plan's argument memory, which is what a recording holds.
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
function compile_dispatch(c::Compile{LavaDevice}, t::Trace, argoff::Int, indirect::Int)
    dev = c.graph.dev
    raw = rawargs(t.args)
    vk_pipeline, raygen, offsets, byval = rt_compiled_for(dev.bq, t.pipeline, raw)
    argbytes = raygen.push_info.arg_buffer_size
    total = argbytes + compute_inline_extra_from_byval(byval)
    compiled = VulkanTracePipeline(vk_pipeline, t.pipeline, offsets, byval, argbytes)
    CompiledTrace(compiled, t.accel, t.args, t.ndrange, argoff, total, indirect)
end

# `argwrites`, `writearg!` and the `ArgWrite` they produced are gone.
#
# They were the write plan: at `record!`, walk every entry's argument tuple,
# perform the real `pack_arg!` for each slot so the recorded offset was the one
# the packer used, and keep an `ArgWrite` for every `Base.RefValue` — so a run
# could rewrite exactly the bytes that could differ. 602 of them on Hikari's
# fused sample, to move 268 bytes, because a dispatch carries its own COPY of
# every argument.
#
# A per-run value is a [`GPURef`](@ref) now. What the dispatches were packed with
# is its ADDRESS, which does not change, and the value behind it is written by
# one store (`ref[] = x`) however many dispatches read it — so there is nothing to rewrite,
# no offsets to record, and no host store into memory a submission may still be
# reading. `verifywrites` went with them: it existed to prove those offsets
# agreed with the packer's.

"""
Pack a draw's arguments into the plan's slot: the counterpart of `packdispatch!`
and `packtrace!`, and like them written once, at `record!`.
"""
function packdraw!(e::Emitter, d::CompiledDraw)
    am = e.args
    off = d.argoff
    info = d.shader.push_info
    pack_args_direct!(e.owner, am.ptr + off, am.address + off, info.arg_offsets,
                      info.arg_buffer_size, info.byval_llvm_sizes,
                      devargs(adaptor(e), rawargs(d.args)))
    return nothing
end

"""
Pack a trace's arguments into the plan's slot. The counterpart of `packdispatch!`.

Into `am` and not into `get_arg_buffer(bq, …)`, which is the whole difference
between this and the unmodelled path: a scratch region belongs to the batch that
took it and goes back when that batch completes, so an address from it means
nothing to a second run. The plan's argument memory belongs to the plan for as
long as the plan lives, which is what recording needs.
"""
function packtrace!(e::Emitter, t::CompiledTrace)
    am = e.args
    off = t.argoff
    c = t.compiled
    owner = e.owner
    # Held as the unmodelled path holds: the shaders are closures, and a
    # per-material closest-hit holds the device arrays of the material it shades.
    holdleaves!(owner, c.desc.raygen_func)
    for chit in c.desc.closesthit_funcs
        holdleaves!(owner, chit)
    end
    holdleaves!(owner, c.desc.miss_func)
    holdleaves!(owner, c.desc.anyhit_func)     # a no-op on `nothing`
    ad = LavaAdaptor(owner)
    # `rawargs` first, so a `Ref` argument is re-read NOW rather than frozen at
    # compile — that is how a new sample index reaches a plan compiled once.
    raw = rawargs(t.args)
    holdleaves!(owner, raw)
    all_args = (Adapt.adapt(ad, c.desc.raygen_func), devargs(ad, raw)...)
    pack_args_direct!(owner, am.ptr + off, am.address + off,
                      c.offsets, c.argbytes, c.byval, all_args)
    return am.address + off
end

"""
Emit the trace itself. A host-side ray count traces directly; a
[`DeviceRange`](@ref) prepares an indirect command from a count on the device.

Two methods rather than a branch, which is how `emitlaunch!` above already
distinguishes the same two cases.

The ray count of a device-sized trace is written by the pass's fused prepare
(`emitprepares!`, core's), the same kernel that writes the workgroup counts of
its dispatches: a trace's indirect command is `(n_rays, 1, 1)`, which is the
same three words with a "workgroup" of one. So nothing is prepared here; the
command is read from the slot the plan laid out for it.
"""
function tracelaunch!(e::Emitter, r::DeviceRange, t::CompiledTrace, tlas,
                      name::AbstractString)
    # The ray count was written by the pass's fused prepare (`emitprepares!`,
    # core's), with the trace's gate folded in; nothing to prepare here.
    indirect = indirectof(e.args, t.indirect)
    argaddr = packtrace!(e, t)
    emit_trace_indirect!(e, t.compiled.pipeline, tlas, argaddr, indirect, name)
    return nothing
end

function tracelaunch!(e::Emitter, n, t::CompiledTrace, tlas, name::AbstractString)
    argaddr = packtrace!(e, t)
    emit_trace!(e, t.compiled.pipeline, tlas, argaddr, Int(n), 1, 1, name)
    return nothing
end

"""Emit a trace: resolve the acceleration structure, then launch."""
emitdispatch!(e::Emitter, t::CompiledTrace, name::AbstractString) =
    tracelaunch!(e, t.ndrange, t, tlasof(storage(t.accel)), name)

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
function collect!(prof::Profiler, dev::LavaDevice)
    prof.pending || return prof
    ctx = dev.ctx
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
    r
end

"""
    emithead!(e::Emitter, plan)

Before anything this plan emits: whatever ran last on this queue may still be
reading the pool this plan is about to write, and a cross-plan hazard cannot be
derived — so one global barrier, always, and the arena no longer remembers who
ran. And the profiler's query pool reset, once per command buffer.
"""
function emithead!(e::Emitter, pl::Plan)
    headbarrier!(e.cmd, e.ctx)
    if pl.profiler !== nothing
        VK.cmd_reset_query_pool(e.cmd, pl.profiler.pool, UInt32(0),
                                UInt32(pl.profiler.nslots))
    end
    return nothing
end

# ↑ moved to src/graph/build.jl

"""Mantle's own barriers, derived from the declared usage sequence. Layout
changes first: a pass may both need an image transitioned and wait on a
buffer, and the two are separate Vulkan barriers."""
function emitbarriers!(e::Emitter, pp::PassPlan)
    for b in pp.images
        emit_barrier!(e, b)
    end
    emit_barrier!(e, pp.barrier)
    return nothing
end

"""Run `f` with this pass's work discarded unless its predicate is nonzero.

The scope's `begin` and `end` have to be in ONE command buffer. That used to need
`unsplittable!` around this whole block, because the work inside went through the
queue's recorder and a dispatch could take the batch past its split or submit
threshold — Hikari's fused sample ran at `max_depth` 8 (47
dispatches) and hung the GPU at 16 (~95), with the auto-submit threshold at 64
sitting exactly between, and the failure was a foreign call that never returned.

An emitter has one command buffer and nothing that ends it, so the scope cannot
be broken any more. That is the guard, and it is structural rather than a counter
saying "not now".
"""
function withpredicate(f, e::Emitter, (pred, i)::Tuple{Any,Int})
    # Without the extension the prepare's zero groups are the whole answer, and
    # `repeat!` refused any gated pass with fixed-size work at build.
    e.ctx.conditional_rendering_available || return f()
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

# An update pass in a RECORDING is nothing: a host store carries the bytes it
# writes inside the command buffer, so a recording holding one would replay this
# run's values for ever. The run writes them fresh, in front of the recording
# (`emitupdates!`). In a one-shot — a windowed frame, the front of a run — the
# stores land where the graph put the pass, before every reader.
emitupdate!(::Emitter{<:Recording}, ::Plan, ::PassPlan) = nothing
function emitupdate!(e::Emitter{<:OneShot}, pl::Plan, pp::PassPlan)
    anydirty(pl.hostwritten) || return nothing
    emitbarriers!(e, pp)
    landstores!(StoreEmitter(e), pl.hostwritten)
    return nothing
end

function emitcopy!(e::Emitter, ::Plan, pp::PassPlan)
    p = pp.pass
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
# derived from the declared usages and emitted by `emitbarriers!`, and Lava's
# own transition would either double up with it or contradict it.
function beginrender!(e::Emitter, ::Plan, pp::PassPlan)
    p = pp.pass
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
    return nothing
end
endrender!(e::Emitter, ::Nothing) = (end_pass!(e); nothing)


emitpreparebarrier!(e::Emitter) = (emit_barrier!(e, preparebarrier()); nothing)

"""Invocations per workgroup, from the iteration plan the dispatch was compiled
with — what core's prepare divides a device-written count by."""
workgroupsize(d::CompiledDispatch) = UInt32(prod(d.iter.ws_3d))

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
    emitlaunch!(e, d.ndrange, lp_of(d), am.address + d.argoff, it, tlas,
                indirectof(am, d.indirect), name)
    nothing
end

# ↑ moved to src/graph/build.jl

# ↑ moved to src/graph/build.jl

# ↑ moved to src/graph/build.jl

# `rebind!` is gone.
#
# It re-read every `Ref` a recorded plan was given and wrote the current value
# into the plan's argument memory — one host store per (entry, `Ref`) pair, on
# the slot `nextslot!` had just claimed, which is the only reason the argument
# ring existed. It was the answer to "a recorded plan does not record, so how
# does a sample index move", and the answer is now that the sample index is a
# [`GPURef`](@ref): the dispatches hold its ADDRESS, and one store writes the
# value as a command in the run's own submission.
#
# What that deletes is not just the loop. It is the write plan that fed it, the
# verification that proved the write plan's offsets, the ring that made the
# writes safe, and the wait at the top of every run that the ring needed.

# ↑ moved to src/graph/build.jl

# ↑ moved to src/graph/build.jl

# ↑ moved to src/graph/build.jl

# ↑ moved to src/graph/build.jl



# `Device()` with no argument is core's — see `Mantle.jl`.



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
# its pure half — `adapt_storage` is a strip, and the pinning is a separate walk
# — so it has no owner.
adaptor(::VulkanBatchQueue) = LavaAdaptor(nothing)


# The portable constructors. A caller writes `Framebuffer(backend, w, h)` and
# `Window(backend, w, h)` and never names a Vulkan type; these are where that
# resolves on this backend.
# On the backend's device: these used to drop the backend and fall through to a
# `ctx = vk_context()` default, so a framebuffer asked of a second device's
# backend was created on the first.
Framebuffer(b::LavaBackend, w::Integer, h::Integer; kw...) = VulkanFramebuffer(w, h; ctx = vk_context(b), kw...)
Window(b::LavaBackend, w::Integer, h::Integer; kw...) = VulkanWindow(w, h; ctx = vk_context(b), kw...)
Texture2D(b::LavaBackend, data::AbstractArray; kw...) = VulkanTexture2D(data; ctx = vk_context(b), kw...)
Sampler(b::LavaBackend; kw...) = VulkanSampler(; ctx = vk_context(b), kw...)

# `devicearray` on this backend is a `LavaArray`: pool-managed and capacity-aware
# on `resize!`, which the generic fallback in `memory/array.jl` cannot be. A
# caller that just wants "this data, on that device" gets the better one for
# free by asking Mantle instead of naming the array type.
devicearray(b::LavaBackend, data::AbstractArray) = LavaArray(data; bq = b.dispatch_bq)


# The two plan pieces that ARE this backend's: a timestamp query pool, and the
# argument memory a recorded launch reads from. Core's `Plan` asks for both and
# accepts `nothing`, which is what a KernelAbstractions backend answers.
makeprofiler(dev::LavaDevice, passes, profile::Bool) =
    profile ? Profiler(dev.ctx, passes) : nothing

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
# `inplace!`, `refit!`. Those were always meant to be
# here; the portable halves of the same names stayed in core.

# `takehost!` is gone with renaming — it was the staging buffer a fresh store was
# filled from, recycled by NEGATIVE size so a host and a device region of the
# same length never shared a free list.

# ↑ moved to src/graph/build.jl — one predicate, `passed`, and it is core's.

# `Graph` is Mantle's now — see `src/graph/types.jl`.



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


"""
    openrecording(dev::LavaDevice, plan) -> Emitter

A command buffer of the plan's own, opened for the walk; the plan's argument
memory is what the pack writes into, which is the whole difference between a
recording and the unmodelled path — see `packtrace!`.
"""
openrecording(dev::LavaDevice, pl::Plan) = Emitter(recording!(dev.bq), pl.args)

"""
    closerecording!(e::Emitter, plan) -> Recording

Seal the command buffer and fill the plan's patch table: where every device
address the pack wrote landed, keyed by the address — so a resource that MOVES
later (`resize!`, an arena growing) is a patch in a later run's submission
rather than a new recording.

`seal!` is also where the recording takes the channel's hold frame: it is
submitted again every run and must outlive all of them, so the holds are the
RECORDING's and go back at `release!`. The buffers it names were collected into
`rec.sync` as the commands were written (`syncbuf!`), so there is no separate
snapshot pass — `collectsync!` walked the pin list to build one.
"""
function closerecording!(e::Emitter{<:Recording}, pl::Plan)
    rec = e.owner
    seal!(rec)
    for (addr, at) in rec.patches
        push!(get!(Vector{Tuple{Region,Int}}, pl.patchtab, addr),
              patchtarget(pl, rec, at))
    end
    empty!(rec.patches)
    return rec
end

"""
Which eight bytes a packed pointer landed in, as the `(region, offset)` a later
move patches. Usually the plan's argument memory; a prepare kernel emitted into
the recording packs into a scratch region the RECORDING owns (`get_arg_buffer`),
which has exactly the recording's lifetime — both go back in `droprecording!`.
Anything else is a pack into memory the batch rewinds, which a recording must
never hold: fail loudly rather than bake a patch entry that points at bytes
someone else owns tomorrow.
"""
function patchtarget(pl::Plan, rec::Recording, at::Ptr{UInt8})
    u = UInt64(at)
    am = pl.args
    base = UInt64(am.ptr)
    base <= u < base + length(am.store) && return (am.store, Int(u - base))
    for r in rec.regions
        blk = memoryof(r)::BufferBlock
        rb = UInt64((blk.ref[]::VkManagedBuffer).mapped_ptr) + UInt64(offset(r))
        rb <= u < rb + length(r) && return (r, Int(u - rb))
    end
    throw(LavaError("record!",
        "a packed pointer landed at $(at) — outside the plan's argument memory and every scratch region the recording owns",
        "An unmodelled launch packed into batch scratch while a plan was being recorded; model it as a dispatch so its arguments live in the plan."))
end


"""Once per frame: pump the event queue and bring the swapchain up to date.
`sync_swapchain!` refuses a closed window — a destroyed GLFW handle is a
segfault to ask anything of. Not `isopen` here: a window whose close button has
been clicked is still fine to draw to, and the loop condition decides to stop."""
function beginframe!(win::VulkanWindow)
    GLFW.PollEvents()
    sync_swapchain!(win)
    return nothing
end

recordsplans(::LavaDevice) = true

"""
A run's own command buffer: a one-shot. Opened by core only when the run has
something to put in front of its recording, or for a windowed plan, which
cannot be recorded until step 9 and is walked into it per frame. The plan's one
surface already holds this frame's image — `beforeframe!` acquired it before
the plan was refit — and nothing here may rebuild the swapchain.
"""
function openrun(dev::LavaDevice, pl::Plan)
    surfaces = pl.graph.surfaces
    if !isempty(surfaces)
        length(surfaces) == 1 || throw(ArgumentError(
            "run!: a plan draws to at most one surface; this one has $(length(surfaces))"))
        only(surfaces).win.acquired || throw(ArgumentError(
            "run!: the frame's image was not acquired; `beforeframe!` acquires it before " *
            "the plan is refit, and `execute!` records against that image"))
    end
    return Emitter(openoneshot(dev.bq), pl.args)
end

"""
Hand the run to the device in one submission: the front one-shot, if there is
one, then the recording; a windowed frame's one-shot with the swapchain's
semaphores and the frame's fence, then the present.
"""
function closerun!(dev::LavaDevice, pl::Plan, e::Union{Nothing,Emitter})
    bq = dev.bq
    surfaces = pl.graph.surfaces
    if !isempty(surfaces)
        win = only(surfaces).win
        presentready!(e, win)
        seal!(e.owner)
        return present_frame!(bq, win, e.owner)
    end
    rec = pl.recording::Recording
    if e === nothing
        # Nothing to store this run: the recording alone, and nothing of it is
        # given back — it is submitted again next run, and the plan owns it.
        tok = submit!(bq, rec)
        rec.token = tok
        handover!(bq, tok, nothing; tag = :run)
        return tok
    end
    front = e.owner::OneShot
    seal!(front)
    # The submission holds the recording it executes; the recording holds
    # everything the commands name (taken at `seal!`, kept until `release!`), so
    # a run's own hold list is the front one-shot's stores and this.
    hold!(bq, rec)
    tok = submit!(bq, front, rec)
    rec.token = tok
    handover!(bq, tok, front; tag = :run)
    return tok
end

"""What was written goes with the buffer: sealed so it can be begun again, and
given back to core with the holds it took."""
function abandonrun!(dev::LavaDevice, e::Emitter)
    o = e.owner::OneShot
    seal!(o)
    release!(queueof(o), o)
    return nothing
end

"""A frame that failed after its image was acquired: the image goes back to the
presentation engine untouched, through the same submit-and-present as a drawn
frame so the slot's fence and semaphores stay paired. Nothing when the frame
was presented or never acquired, and nothing on a lost device, where no
present can be made."""
function abandonframe!(dev::LavaDevice, pl::Plan)
    for s in pl.graph.surfaces
        win = s.win
        (win.acquired && !device_lost(dev.ctx)) || continue
        o = oneshot(dev.bq) do e
            presentuntouched!(e, win)
        end
        present_frame!(dev.bq, win, o)
    end
    return nothing
end

# A frame's timestamps are pending from the SUBMISSION that writes them, once
# per run. `profiled!(f, pl, e, i)` used to set this where it emitted the
# timestamp commands, which for a recording is once: the first `collect!`
# cleared it and every later frame of the plan went unread — `timings` said
# `NaN` for a recorded plan's GPU time, which is every Hikari plan.

"""Bytes inside the command buffer, into a region or the array over one: a
pointer patch, a store `storebytes!` can carry."""
function emitinline!(e::Emitter, r::Union{Region,DeviceArray}, off::Int, p::Ptr{Cvoid}, n::Int)
    blk = memoryof(r)::BufferBlock
    VK.cmd_update_buffer(e.cmd, blk.buffer, UInt64(offset(r)) + UInt64(off), UInt64(n), p)
    return nothing
end


"""
One draw: write its arguments into the plan's slot and emit it.

Its own function because a pass's draws are differently parameterised, so the
loop dispatches once per draw and everything inside is concrete — including
`devargs(ad, rawargs(d.args))`, which allocated a boxed tuple per draw per frame while it
was inlined into the loop.

Nothing is allocated and nothing is held here: the memory belongs to the plan,
and every resource the arguments name was held when the recording was written,
for as long as the recording lives.
"""
function emitdraw!(e::Emitter, ::Nothing, d::CompiledDraw)
    packdraw!(e, d)
    # No viewport, no scissor, no pin: the pass set the first two once, and the
    # plan holds the pipeline for longer than any frame.
    emit_draw!(e, d.compiled, d.count, e.args.address + d.argoff)
end

# Where the counts come from, decided once by type rather than per frame by a
# branch. The indirect one emits the same command whatever the numbers are,
# which is why nothing has to be read back to record a frame.

emit_draw!(e, pipe, n::Integer, addr::UInt64) =
    draw_in_pass!(e, pipe, n; push_bda = addr)

emit_draw!(e, pipe, x, addr::UInt64) =
    draw_in_pass!(e, pipe, count(x); push_bda = addr)

emit_draw!(e, pipe, c::Commands, addr::UInt64) =
    draw_indirect_in_pass!(e, pipe, storage(c.resource);
                           push_bda = addr)

"""
Write one dispatch's arguments into the plan's argument memory, and answer with
the acceleration structure they name.

Separate from emitting because it happens once, at `record!`: the command buffer
holds the ADDRESS of these bytes, and every value in them is fixed for the plan's
life. A value that is not is a [`GPURef`](@ref), and what lands here is its
address.
"""
function packdispatch!(e::Emitter, d::CompiledDispatch, it)
    am = e.args
    off = d.argoff
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
